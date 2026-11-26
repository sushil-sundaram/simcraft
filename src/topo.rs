//! Topology optimization (SIMP): find where material is needed for a stiff part.
//!
//! Minimum compliance subject to a volume fraction, element densities ρ ∈ [ρmin, 1] with
//! E(ρ) = Emin + ρᵖ (E − Emin), a distance-weighted sensitivity filter, and the classic
//! optimality-criteria update. Elements under supports and loads are kept solid. The result
//! (ρ ≥ 0.5) is exported as a closed STL made of the kept voxels' outer faces.

use crate::fem::{self, Assembled};
use crate::mesh::{CORNER, Mesh};
use crate::model::Model;
use crate::results::Field;
use crate::sparse::{NodeSparse, Solver};
use anyhow::{Result, bail};
use rayon::prelude::*;
use std::io::Write;

pub struct TopoSettings {
    pub volume_fraction: f64,
    pub iterations: usize,
    pub filter_radius_mm: Option<f64>,
    pub penal: f64,
}

pub struct TopoResult {
    pub density: Vec<f64>,
    pub history: Vec<(usize, f64, f64)>,
    pub kept_elements: usize,
    pub results: crate::results::Results,
}

pub fn run(model: &Model, mesh: &Mesh, s: &TopoSettings, prog: fem::Prog) -> Result<TopoResult> {
    if model.analysis.kind != crate::model::AnalysisType::StaticStructural {
        bail!("topology optimization works on a Static Structural setup (supports + loads)");
    }
    if !(s.volume_fraction > 0.02 && s.volume_fraction < 0.98) {
        bail!("volume fraction must be between 0.02 and 0.98");
    }
    let h = mesh.h * 1e-3;
    let ne = mesh.elems.len();
    let nn = mesh.nodes.len();
    let (asm, units) = fem::assemble_structural(model, mesh, fem::Progress::cancel_only(prog.cancel))?;
    let Assembled { fixed, support_nodes, .. } = asm;
    if support_nodes.is_empty() {
        bail!("add a support before optimizing");
    }
    let f_ext = fem::external_forces(model, mesh, None)?;
    if f_ext.iter().all(|v| *v == 0.0) {
        bail!("add a load (force, pressure or gravity) before optimizing");
    }

    // passive solid: elements owning supported or loaded faces
    let mut passive = vec![false; ne];
    for it in &model.loads {
        if let Ok(fs) = mesh.resolve_faces(it.load.faces()) {
            for f in fs {
                passive[mesh.faces[f as usize].elem as usize] = true;
            }
        }
    }
    let design: Vec<usize> = (0..ne).filter(|&e| !passive[e]).collect();
    if design.is_empty() {
        bail!("every element is under a load or support; nothing left to optimize");
    }

    // filter neighbourhoods on the voxel grid
    let r = s.filter_radius_mm.unwrap_or(1.6 * mesh.h).max(mesh.h * 1.01);
    let reach = (r / mesh.h).floor() as i64;
    let [nx, ny, _nz] = mesh.dims;
    let cid = |c: [i64; 3]| -> usize { (c[0] as usize) + nx * ((c[1] as usize) + ny * (c[2] as usize)) };
    let mut elem_of = vec![u32::MAX; mesh.dims.iter().product()];
    for (e, c) in mesh.elem_cell.iter().enumerate() {
        elem_of[cid([c[0] as i64, c[1] as i64, c[2] as i64])] = e as u32;
    }
    let neigh: Vec<Vec<(u32, f64)>> = (0..ne)
        .into_par_iter()
        .map(|e| {
            let c = mesh.elem_cell[e];
            let mut v = Vec::new();
            for dz in -reach..=reach {
                for dy in -reach..=reach {
                    for dx in -reach..=reach {
                        let q = [c[0] as i64 + dx, c[1] as i64 + dy, c[2] as i64 + dz];
                        if (0..3).any(|k| q[k] < 0 || q[k] >= mesh.dims[k] as i64) {
                            continue;
                        }
                        let o = elem_of[cid(q)];
                        if o == u32::MAX {
                            continue;
                        }
                        let w = r - mesh.h * ((dx * dx + dy * dy + dz * dz) as f64).sqrt();
                        if w > 0.0 {
                            v.push((o, w));
                        }
                    }
                }
            }
            v
        })
        .collect();

    let e0: Vec<f64> = mesh.elem_mat.iter().map(|&m| mesh.materials[m as usize].youngs_modulus_gpa * 1e9).collect();
    let emin_ratio = 1e-9;
    let rho_min = 1e-3;
    let mut rho: Vec<f64> = (0..ne).map(|e| if passive[e] { 1.0 } else { s.volume_fraction }).collect();
    let mut history = Vec::new();
    let mut last_u = vec![0.0; nn * 3];
    let iters = s.iterations.clamp(5, 300);
    for it in 1..=iters {
        // stiffness for current densities
        let mut k = NodeSparse::new(nn, &mesh.elems, 3);
        for (e, en) in mesh.elems.iter().enumerate() {
            let ee = e0[e] * (emin_ratio + rho[e].powf(s.penal) * (1.0 - emin_ratio));
            k.add_element(en, &units[mesh.elem_mat[e] as usize].k, ee * h);
        }
        let mut f = f_ext.clone();
        k.apply_dirichlet(&fixed, &mut f);
        let mut u: Vec<f64> = last_u.clone();
        for (i, g) in fixed.iter().enumerate() {
            if let Some(v) = g {
                u[i] = *v;
            }
        }
        let solver = Solver::new(&k)?;
        solver.solve(&k, &f, &mut u, 1e-8, &prog.check_scalar())?;
        // element strain energy for the unit modulus
        let ce: Vec<f64> = mesh
            .elems
            .par_iter()
            .enumerate()
            .map(|(e, en)| {
                let ku = &units[mesh.elem_mat[e] as usize].k;
                let ue: Vec<f64> = (0..24).map(|i| u[en[i / 3] as usize * 3 + i % 3]).collect();
                let mut s2 = 0.0;
                for i in 0..24 {
                    let mut row = 0.0;
                    for j in 0..24 {
                        row += ku[i * 24 + j] * ue[j];
                    }
                    s2 += ue[i] * row;
                }
                s2 * h
            })
            .collect();
        let compliance: f64 = (0..ne).map(|e| e0[e] * (emin_ratio + rho[e].powf(s.penal) * (1.0 - emin_ratio)) * ce[e]).sum();
        let dc: Vec<f64> = (0..ne).map(|e| -s.penal * rho[e].powf(s.penal - 1.0) * (1.0 - emin_ratio) * e0[e] * ce[e]).collect();
        // sensitivity filter
        let dcf: Vec<f64> = (0..ne)
            .into_par_iter()
            .map(|e| {
                let (mut num, mut den) = (0.0, 0.0);
                for &(o, w) in &neigh[e] {
                    num += w * rho[o as usize] * dc[o as usize];
                    den += w;
                }
                num / (rho[e].max(rho_min) * den.max(1e-30))
            })
            .collect();
        // optimality criteria with bisection on the volume multiplier
        let target = s.volume_fraction * design.len() as f64;
        let (mut l1, mut l2) = (1e-30f64, 1e30f64);
        let mv = 0.2;
        let mut new = rho.clone();
        while (l2 - l1) / (l1 + l2) > 1e-6 {
            let lm = (l1 * l2).sqrt();
            for &e in &design {
                let b = (-dcf[e] / lm).max(0.0).sqrt();
                new[e] = (rho[e] * b).clamp((rho[e] - mv).max(rho_min), (rho[e] + mv).min(1.0));
            }
            let vol: f64 = design.iter().map(|&e| new[e]).sum();
            if vol > target {
                l1 = lm;
            } else {
                l2 = lm;
            }
        }
        let change = design.iter().map(|&e| (new[e] - rho[e]).abs()).fold(0.0, f64::max);
        rho = new;
        last_u = u;
        let vf = design.iter().map(|&e| rho[e]).sum::<f64>() / design.len() as f64;
        history.push((it, compliance, vf));
        prog.report(it as f64 / iters as f64, &format!("topology iteration {it}/{iters}: compliance {compliance:.4e} J, change {change:.3}"))?;
        if it > 10 && change < 0.01 {
            break;
        }
    }

    // nodal density field + deformation of the final design for display
    let mut acc = vec![0.0; nn];
    let mut cnt = vec![0u32; nn];
    for (e, en) in mesh.elems.iter().enumerate() {
        for &n in en {
            acc[n as usize] += rho[e];
            cnt[n as usize] += 1;
        }
    }
    let mut results = crate::results::Results::new(model.analysis.kind);
    results.extra.insert(Field::Density, acc.iter().zip(&cnt).map(|(a, &c)| a / c.max(1) as f64).collect());
    results.disp = Some((0..nn).map(|q| [last_u[q * 3] * 1e3, last_u[q * 3 + 1] * 1e3, last_u[q * 3 + 2] * 1e3]).collect());
    let kept = rho.iter().filter(|r| **r >= 0.5).count();
    let (c0, cf) = (history.first().map(|h| h.1).unwrap_or(0.0), history.last().map(|h| h.1).unwrap_or(0.0));
    results.quantity("Kept material (ρ ≥ 0.5)", kept as f64 / ne as f64 * 100.0, "% of elements");
    results.quantity("Compliance, first iteration", c0, "J");
    results.quantity("Compliance, final", cf, "J");
    results.info.nodes = nn;
    results.info.elements = ne;
    results.info.element_size_mm = mesh.h;
    results.info.iterations = history.len();
    results.info.solver = "SIMP topology optimization".into();
    Ok(TopoResult { density: rho, history, kept_elements: kept, results })
}

/// Write the kept voxels (ρ ≥ threshold) as a closed binary STL (mm).
pub fn write_stl(mesh: &Mesh, density: &[f64], threshold: f64, path: &str) -> Result<usize> {
    let [nx, ny, nz] = mesh.dims;
    let cid = |i: usize, j: usize, k: usize| i + nx * (j + ny * k);
    let mut keep = vec![false; nx * ny * nz];
    for (e, c) in mesh.elem_cell.iter().enumerate() {
        if density[e] >= threshold {
            keep[cid(c[0] as usize, c[1] as usize, c[2] as usize)] = true;
        }
    }
    let mut tris: Vec<[[f32; 3]; 4]> = Vec::new(); // normal + 3 vertices
    let h = mesh.h;
    for (e, c) in mesh.elem_cell.iter().enumerate() {
        if density[e] < threshold {
            continue;
        }
        let (i, j, k) = (c[0] as usize, c[1] as usize, c[2] as usize);
        for dir in 0..6usize {
            let a = dir / 2;
            let plus = dir % 2 == 1;
            let mut q = [i as i64, j as i64, k as i64];
            q[a] += if plus { 1 } else { -1 };
            let inside = q[a] >= 0 && (q[a] as usize) < mesh.dims[a];
            if inside && keep[cid(q[0] as usize, q[1] as usize, q[2] as usize)] {
                continue;
            }
            let (b, cax) = ((a + 1) % 3, (a + 2) % 3);
            let corner = |ob: usize, oc: usize| -> [f32; 3] {
                let mut d = [0usize; 3];
                d[a] = plus as usize;
                d[b] = ob;
                d[cax] = oc;
                let _ = CORNER;
                [(mesh.origin[0] + (i + d[0]) as f64 * h) as f32, (mesh.origin[1] + (j + d[1]) as f64 * h) as f32, (mesh.origin[2] + (k + d[2]) as f64 * h) as f32]
            };
            let mut quad = [corner(0, 0), corner(1, 0), corner(1, 1), corner(0, 1)];
            if !plus {
                quad.swap(1, 3);
            }
            let mut n = [0f32; 3];
            n[a] = if plus { 1.0 } else { -1.0 };
            tris.push([n, quad[0], quad[1], quad[2]]);
            tris.push([n, quad[0], quad[2], quad[3]]);
        }
    }
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    let mut header = [0u8; 80];
    let tag = b"SimCraft topology optimization";
    header[..tag.len()].copy_from_slice(tag);
    f.write_all(&header)?;
    f.write_all(&(tris.len() as u32).to_le_bytes())?;
    for t in &tris {
        for v in t {
            for c in v {
                f.write_all(&c.to_le_bytes())?;
            }
        }
        f.write_all(&[0, 0])?;
    }
    Ok(tris.len())
}
