//! Electromagnetic field solvers on the voxel hex mesh.
//!
//! * Electrostatic:  ∇·(ε∇V) = 0 with voltage electrodes → V, |E|, electrode charge, capacitance.
//! * DC current flow: ∇·(σ∇V) = 0 with voltage / current terminals → V, |J|, Joule heat,
//!   resistance, power (the Joule heat feeds the thermal-electric analysis).
//! * Magnetostatic: vector potential A with the Coulomb-gauged (curl-curl + div-div) nodal
//!   formulation, coil current densities and permanent magnets, A = 0 on the enclosure walls
//!   → |B|, |H|, stored magnetic energy.
//!
//! All internal quantities are SI.

use crate::fem::{Prog, center_grads, dn_dxi, face_nodes, gauss_points, hex_conduction_unit};
use crate::geometry::*;
use crate::mesh::{Mesh, REGION_SOLID};
use crate::model::*;
use crate::results::*;
use crate::sparse::*;
use anyhow::{Result, bail};

pub const EPS0: f64 = 8.854_187_812_8e-12;
pub const MU0: f64 = 1.256_637_062_12e-6;

/// Average element values onto nodes.
fn to_nodes(mesh: &Mesh, elem: &[f64], only_region: Option<u8>) -> Vec<f64> {
    let nn = mesh.nodes.len();
    let mut acc = vec![0.0; nn];
    let mut cnt = vec![0u32; nn];
    for (e, en) in mesh.elems.iter().enumerate() {
        if only_region.is_some_and(|r| mesh.elem_region[e] != r) {
            continue;
        }
        for &n in en {
            acc[n as usize] += elem[e];
            cnt[n as usize] += 1;
        }
    }
    acc.iter().zip(&cnt).map(|(a, &c)| if c > 0 { a / c as f64 } else { 0.0 }).collect()
}

/// Gradient of a nodal scalar at each element centre [unit/m].
fn elem_gradients(mesh: &Mesh, phi: &[f64]) -> Vec<V3> {
    let h = mesh.h * 1e-3;
    let cg = center_grads();
    mesh.elems
        .iter()
        .map(|en| {
            let mut g = [0.0; 3];
            for a in 0..8 {
                for c in 0..3 {
                    g[c] += cg[a][c] * phi[en[a] as usize] / h;
                }
            }
            g
        })
        .collect()
}

struct ScalarBc {
    fixed: Vec<Option<f64>>,
    sets: Vec<(u32, String, Vec<u32>, f64)>,
}

fn voltage_bcs(model: &Model, mesh: &Mesh) -> Result<ScalarBc> {
    let mut fixed = vec![None; mesh.nodes.len()];
    let mut sets = Vec::new();
    for it in &model.loads {
        if let Load::Voltage { faces, value } = &it.load {
            let nodes = face_nodes(mesh, &mesh.resolve_faces(faces)?);
            for &n in &nodes {
                fixed[n as usize] = Some(*value);
            }
            sets.push((it.id, it.load.label().to_string(), nodes, *value));
        }
    }
    Ok(ScalarBc { fixed, sets })
}

/// Solve the scalar Poisson problem K φ = f with Dirichlet values; returns (φ, K before BCs).
fn solve_scalar(mesh: &Mesh, coef: &[f64], fixed: &[Option<f64>], f: &[f64], prog: Prog, res: &mut Results) -> Result<(Vec<f64>, NodeSparse)> {
    let h = mesh.h * 1e-3;
    let ku = hex_conduction_unit();
    let ties = mesh.interface_ties();
    let mut k = NodeSparse::new_with_edges(mesh.nodes.len(), &mesh.elems, &ties, 1);
    for (e, en) in mesh.elems.iter().enumerate() {
        k.add_element(en, &ku, coef[e] * h);
    }
    if !ties.is_empty() {
        let avg = coef.iter().sum::<f64>() / coef.len().max(1) as f64;
        crate::fem::add_scalar_ties(mesh, &mut k, avg.max(1e-18) * h * 1e4);
    }
    let k0 = k.clone();
    let mut rhs = f.to_vec();
    k.apply_dirichlet(fixed, &mut rhs);
    let mut phi: Vec<f64> = fixed.iter().map(|g| g.unwrap_or(0.0)).collect();
    prog.report(0.1, "factorising")?;
    let solver = Solver::new(&k)?;
    let its = solver.solve(&k, &rhs, &mut phi, 1e-10, &prog.scalar(0.1, 0.8, "solving (PCG)"))?;
    res.info.solver = solver.name().to_string();
    res.info.iterations += its;
    res.info.dofs += mesh.nodes.len();
    Ok((phi, k0))
}

fn set_reactions(k0: &NodeSparse, phi: &[f64], f: &[f64], sets: &[(u32, String, Vec<u32>, f64)], res: &mut Results) -> Vec<f64> {
    let mut kp = vec![0.0; phi.len()];
    k0.matvec(phi, &mut kp);
    let mut claimed = vec![false; phi.len()];
    let mut out = Vec::new();
    for (id, label, nodes, _) in sets {
        let mut q = 0.0;
        for &n in nodes {
            let n = n as usize;
            if !claimed[n] {
                claimed[n] = true;
                q += kp[n] - f[n];
            }
        }
        res.reactions.push(Reaction { load_id: *id, label: label.clone(), value: [float(q), 0.0, 0.0] });
        out.push(q);
    }
    out
}

pub fn electrostatic(model: &Model, mesh: &Mesh, prog: Prog, res: &mut Results) -> Result<()> {
    let bc = voltage_bcs(model, mesh)?;
    if bc.sets.is_empty() {
        bail!("electrostatic analysis needs Voltage boundary conditions (e.g. 10 V on one electrode, 0 V on another)");
    }
    // metals (σ > 1 S/m) behave as ideal conductors: equipotential, modelled with a huge εr
    let eps: Vec<f64> = mesh
        .elem_mat
        .iter()
        .map(|&m| {
            let mat = &mesh.materials[m as usize];
            EPS0 * if mat.electrical_conductivity > 1.0 { 1e6 } else { mat.relative_permittivity }
        })
        .collect();
    let f = vec![0.0; mesh.nodes.len()];
    let (v, k0) = solve_scalar(mesh, &eps, &bc.fixed, &f, prog, res)?;
    let grads = elem_gradients(mesh, &v);
    let emag: Vec<f64> = grads.iter().map(mag).collect();
    let charges = set_reactions(&k0, &v, &f, &bc.sets, res);
    // stored energy W = ½ Vᵀ K V
    let mut kv = vec![0.0; v.len()];
    k0.matvec(&v, &mut kv);
    let energy = 0.5 * v.iter().zip(&kv).map(|(a, b)| a * b).sum::<f64>();
    res.quantity("Stored electrostatic energy", energy, "J");
    let vals: Vec<f64> = bc.sets.iter().map(|s| s.3).collect();
    let dv = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max) - vals.iter().cloned().fold(f64::INFINITY, f64::min);
    if dv > 0.0 {
        let (ihi, _) = vals.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap();
        res.quantity("Capacitance (high electrode)", charges[ihi].abs() / dv, "F");
        res.quantity("Capacitance (from energy)", 2.0 * energy / (dv * dv), "F");
    } else {
        res.warnings.push("all electrodes are at the same voltage: the field is zero".into());
    }
    res.extra.insert(Field::ElectricPotential, v);
    res.extra.insert(Field::ElectricField, to_nodes(mesh, &emag, None));
    prog.report(1.0, "done")?;
    Ok(())
}

/// DC current flow in the solid conductors. Returns Joule heat per element [W/m³].
pub fn current_flow(model: &Model, mesh: &Mesh, prog: Prog, res: &mut Results) -> Result<Vec<f64>> {
    let h = mesh.h * 1e-3;
    let bc = voltage_bcs(model, mesh)?;
    if bc.sets.is_empty() {
        bail!("current flow needs at least one Voltage boundary condition (e.g. 0 V ground) — add Current or a second Voltage to drive it");
    }
    let smax = mesh.elem_mat.iter().map(|&m| mesh.materials[m as usize].electrical_conductivity).fold(0.0, f64::max);
    if smax <= 0.0 {
        bail!("no electrically conductive material: assign a metal (e.g. Copper) to the bodies");
    }
    let sigma: Vec<f64> = mesh.elem_mat.iter().map(|&m| mesh.materials[m as usize].electrical_conductivity.max(smax * 1e-12)).collect();
    let mut f = vec![0.0; mesh.nodes.len()];
    let mut injected = Vec::new();
    for it in &model.loads {
        if let Load::Current { faces, value } = &it.load {
            let fs = mesh.resolve_faces(faces)?;
            let share = value / (fs.len() as f64 * 4.0);
            for &fi in &fs {
                for &n in &mesh.faces[fi as usize].nodes {
                    f[n as usize] += share;
                }
            }
            injected.push((face_nodes(mesh, &fs), *value));
        }
    }
    let (v, k0) = solve_scalar(mesh, &sigma, &bc.fixed, &f, prog, res)?;
    let grads = elem_gradients(mesh, &v);
    let jmag: Vec<f64> = grads.iter().zip(&sigma).map(|(g, s)| s * mag(g)).collect();
    let q: Vec<f64> = grads.iter().zip(&sigma).map(|(g, s)| s * dot(*g, *g)).collect();
    let power: f64 = q.iter().sum::<f64>() * h * h * h;
    let currents = set_reactions(&k0, &v, &f, &bc.sets, res);
    res.quantity("Total Joule heating power", power, "W");
    if bc.sets.len() >= 2 {
        let dv = (bc.sets[0].3 - bc.sets[1].3).abs();
        let i = currents[0].abs();
        if i > 0.0 && dv > 0.0 {
            res.quantity("Resistance (between first two voltage terminals)", dv / i, "Ω");
        }
    } else if let Some((nodes, i)) = injected.first() {
        let vin = nodes.iter().map(|&n| v[n as usize]).sum::<f64>() / nodes.len().max(1) as f64;
        if i.abs() > 0.0 {
            res.quantity("Resistance (current terminal to ground)", (vin - bc.sets[0].3).abs() / i.abs(), "Ω");
        }
    }
    res.extra.insert(Field::ElectricPotential, v);
    res.extra.insert(Field::CurrentDensity, to_nodes(mesh, &jmag.iter().map(|j| j * 1e-6).collect::<Vec<_>>(), None));
    res.extra.insert(Field::JouleHeat, to_nodes(mesh, &q, None));
    prog.report(1.0, "done")?;
    Ok(q)
}

/// Unit-cube (h = 1, ν = 1) curl-curl + div-div matrix, row-major 24x24, dof = node*3 + comp.
fn curl_div_unit() -> Vec<f64> {
    let mut k = vec![0.0; 24 * 24];
    for x in gauss_points() {
        let g: Vec<[f64; 3]> = (0..8).map(|i| dn_dxi(i, x).map(|v| 2.0 * v)).collect();
        for a in 0..8 {
            for b in 0..8 {
                let gab = dot(g[a], g[b]);
                for i in 0..3 {
                    for j in 0..3 {
                        let curl = if i == j { gab } else { 0.0 } - g[a][j] * g[b][i];
                        let div = g[a][i] * g[b][j];
                        k[(a * 3 + i) * 24 + b * 3 + j] += (curl + div) * 0.125;
                    }
                }
            }
        }
    }
    k
}

/// ∫∇N_a dV for the unit cube.
fn grad_integrals() -> Vec<V3> {
    let mut out = vec![[0.0; 3]; 8];
    for x in gauss_points() {
        for (a, o) in out.iter_mut().enumerate() {
            let g = dn_dxi(a, x);
            for c in 0..3 {
                o[c] += 2.0 * g[c] * 0.125;
            }
        }
    }
    out
}

pub fn magnetostatic(model: &Model, mesh: &Mesh, prog: Prog, res: &mut Results) -> Result<()> {
    let h = mesh.h * 1e-3;
    let nn = mesh.nodes.len();
    prog.report(0.0, "assembling magnetic system")?;
    let nu: Vec<f64> = mesh.elem_mat.iter().map(|&m| 1.0 / (MU0 * mesh.materials[m as usize].relative_permeability.max(1e-6))).collect();
    let ku = curl_div_unit();
    let ties = mesh.interface_ties();
    let mut k = NodeSparse::new_with_edges(nn, &mesh.elems, &ties, 3);
    for (e, en) in mesh.elems.iter().enumerate() {
        k.add_element(en, &ku, nu[e] * h);
    }
    if !ties.is_empty() {
        let avg = nu.iter().sum::<f64>() / nu.len().max(1) as f64;
        crate::fem::add_vec3_ties(mesh, &mut k, avg.max(1.0) * h * 1e4);
    }
    let k0 = k.clone();
    let gi = grad_integrals();
    let mut f = vec![0.0; nn * 3];
    let mut mag_m = vec![[0.0; 3]; mesh.elems.len()];
    let mut has_source = false;
    for it in &model.loads {
        match &it.load {
            Load::Coil { body, current_density, axis, mode, center } => {
                let els = mesh.body_elems(body)?;
                let ax = axis.idx();
                let c0 = center.unwrap_or_else(|| {
                    let mut s = [0.0; 3];
                    for &e in &els {
                        s = add(s, mesh.elem_center(e));
                    }
                    scale(s, 1.0 / els.len() as f64)
                });
                let mut e_ax = [0.0; 3];
                e_ax[ax] = 1.0;
                let jmag = current_density * 1e6;
                let mut total_area_current = 0.0;
                for &e in &els {
                    let dir = match mode {
                        CoilMode::Axial => e_ax,
                        CoilMode::Circumferential => {
                            let mut r = sub(mesh.elem_center(e), c0);
                            r[ax] = 0.0;
                            if norm(r) < 1e-9 {
                                continue;
                            }
                            normalize(cross(e_ax, r))
                        }
                    };
                    let j = scale(dir, jmag);
                    total_area_current += jmag * h * h * h;
                    for &n in &mesh.elems[e] {
                        for c in 0..3 {
                            f[n as usize * 3 + c] += j[c] * h * h * h / 8.0;
                        }
                    }
                }
                let _ = total_area_current;
                has_source = true;
            }
            Load::Magnet { body, remanence } => {
                let els = mesh.body_elems(body)?;
                let m = scale(*remanence, 1.0 / MU0);
                for &e in &els {
                    mag_m[e] = m;
                    for (a, &n) in mesh.elems[e].iter().enumerate() {
                        let fm = cross(m, scale(gi[a], h * h));
                        for c in 0..3 {
                            f[n as usize * 3 + c] += fm[c];
                        }
                    }
                }
                has_source = true;
            }
            _ => {}
        }
    }
    if !has_source {
        bail!("magnetostatic analysis needs a source: a Coil current or a Permanent Magnet on a body");
    }
    // A = 0 on the enclosure walls (flux parallel to the boundary)
    let mut fixed = vec![None; nn * 3];
    for face in mesh.faces.iter().filter(|f| f.outer) {
        for &n in &face.nodes {
            for c in 0..3 {
                fixed[n as usize * 3 + c] = Some(0.0);
            }
        }
    }
    let mut rhs = f.clone();
    k.apply_dirichlet(&fixed, &mut rhs);
    let mut a = vec![0.0; nn * 3];
    prog.report(0.1, "factorising")?;
    let solver = Solver::new(&k)?;
    let its = solver.solve(&k, &rhs, &mut a, 1e-9, &prog.scalar(0.1, 0.8, "solving (PCG)"))?;
    res.info.solver = solver.name().to_string();
    res.info.iterations += its;
    res.info.dofs = nn * 3;

    // B = curl A, H = νB − M
    let cg = center_grads();
    let mut bmag = vec![0.0; mesh.elems.len()];
    let mut hmag = vec![0.0; mesh.elems.len()];
    let mut energy = 0.0;
    for (e, en) in mesh.elems.iter().enumerate() {
        let mut b = [0.0; 3];
        for (q, &n) in en.iter().enumerate() {
            let av = [a[n as usize * 3], a[n as usize * 3 + 1], a[n as usize * 3 + 2]];
            b = add(b, scale(cross(cg[q], av), 1.0 / h));
        }
        let hv = sub(scale(b, nu[e]), mag_m[e]);
        bmag[e] = norm(b);
        hmag[e] = norm(hv);
        energy += 0.5 * dot(b, hv) * h * h * h;
    }
    let _ = k0;
    res.quantity("Stored magnetic energy", energy, "J");
    res.quantity("Peak |B| in solids", (0..mesh.elems.len()).filter(|&e| mesh.elem_region[e] == REGION_SOLID).map(|e| bmag[e]).fold(0.0, f64::max), "T");
    res.extra.insert(Field::MagneticFluxDensity, to_nodes(mesh, &bmag, None));
    res.extra.insert(Field::MagneticFieldIntensity, to_nodes(mesh, &hmag, None));
    prog.report(1.0, "done")?;
    Ok(())
}
