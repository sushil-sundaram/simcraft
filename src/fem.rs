//! Finite element solvers on the voxel hex mesh.
//!
//! * Structural: 8-node hexahedra with Wilson-Taylor incompatible modes (no shear locking in
//!   bending), statically condensed to 24x24.
//! * Thermal: 8-node conduction hexahedra, convection / flux / generation loads.
//! * Modal: lumped mass, shifted subspace iteration with Rayleigh-Ritz.
//!
//! Internally everything is SI (m, N, Pa, K).

use crate::geometry::*;
use crate::mesh::{CORNER, Mesh, dir_normal};
use crate::model::*;
use crate::results::*;
use crate::sparse::*;
use anyhow::{Result, bail};
use rayon::prelude::*;
use std::time::Instant;

pub type Prog<'a> = &'a dyn Fn(f64, &str);

pub const GP: f64 = 0.577_350_269_189_625_8;

pub fn sgn(i: usize) -> [f64; 3] {
    CORNER[i].map(|c| 2.0 * c as f64 - 1.0)
}

pub fn dn_dxi(i: usize, x: [f64; 3]) -> [f64; 3] {
    let s = sgn(i);
    let f = |k: usize| 1.0 + s[k] * x[k];
    [s[0] * f(1) * f(2) / 8.0, f(0) * s[1] * f(2) / 8.0, f(0) * f(1) * s[2] / 8.0]
}

pub fn gauss_points() -> Vec<[f64; 3]> {
    let mut v = Vec::new();
    for a in [-GP, GP] {
        for b in [-GP, GP] {
            for c in [-GP, GP] {
                v.push([a, b, c]);
            }
        }
    }
    v
}

/// Isotropic elasticity matrix for E = 1.
pub fn dmat(nu: f64) -> [[f64; 6]; 6] {
    let lam = nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    let mu = 1.0 / (2.0 * (1.0 + nu));
    let mut d = [[0.0; 6]; 6];
    for i in 0..3 {
        for j in 0..3 {
            d[i][j] = lam;
        }
        d[i][i] = lam + 2.0 * mu;
        d[i + 3][i + 3] = mu;
    }
    d
}

/// Strain-displacement matrix (6 x 3n) from shape-function gradients.
pub fn bmat(grads: &[[f64; 3]]) -> Vec<[f64; 6]> {
    // returned column-major: one [f64;6] per column
    let mut cols = Vec::with_capacity(grads.len() * 3);
    for g in grads {
        cols.push([g[0], 0.0, 0.0, g[1], 0.0, g[2]]);
        cols.push([0.0, g[1], 0.0, g[0], g[2], 0.0]);
        cols.push([0.0, 0.0, g[2], 0.0, g[1], g[0]]);
    }
    cols
}

pub fn btdb(a: &[[f64; 6]], d: &[[f64; 6]; 6], b: &[[f64; 6]], w: f64, out: &mut [f64], ncols: usize) {
    let db: Vec<[f64; 6]> = b
        .iter()
        .map(|c| {
            let mut r = [0.0; 6];
            for i in 0..6 {
                for j in 0..6 {
                    r[i] += d[i][j] * c[j];
                }
            }
            r
        })
        .collect();
    for (i, ai) in a.iter().enumerate() {
        for (j, dbj) in db.iter().enumerate() {
            let mut s = 0.0;
            for k in 0..6 {
                s += ai[k] * dbj[k];
            }
            out[i * ncols + j] += s * w;
        }
    }
}

pub struct ElemUnit {
    /// condensed 24x24 stiffness (E = 1, h = 1)
    pub k: Vec<f64>,
    /// recovery of incompatible-mode amplitudes: alpha = r * u  (9x24)
    pub r: Vec<f64>,
}

/// Unit-cube (h = 1, E = 1) stiffness with incompatible modes, condensed. Row-major 24x24.
pub fn hex_elastic_unit(nu: f64) -> ElemUnit {
    let d = dmat(nu);
    let mut kuu = vec![0.0; 24 * 24];
    let mut kua = vec![0.0; 24 * 9];
    let mut kaa = vec![0.0; 9 * 9];
    let w = 0.125; // detJ for unit cube, gauss weights = 1
    for x in gauss_points() {
        let gu: Vec<[f64; 3]> = (0..8).map(|i| dn_dxi(i, x).map(|v| 2.0 * v)).collect();
        let ga: Vec<[f64; 3]> = (0..3)
            .map(|k| {
                let mut g = [0.0; 3];
                g[k] = 2.0 * (-2.0 * x[k]);
                g
            })
            .collect();
        let bu = bmat(&gu);
        let ba = bmat(&ga);
        btdb(&bu, &d, &bu, w, &mut kuu, 24);
        btdb(&bu, &d, &ba, w, &mut kua, 9);
        btdb(&ba, &d, &ba, w, &mut kaa, 9);
    }
    let kinv = dense_inverse(&kaa, 9);
    // K = Kuu - Kua Kaa^-1 Kau
    let mut t = vec![0.0; 24 * 9]; // Kua * Kaa^-1
    for i in 0..24 {
        for j in 0..9 {
            t[i * 9 + j] = (0..9).map(|k| kua[i * 9 + k] * kinv[k * 9 + j]).sum();
        }
    }
    let mut k = kuu;
    for i in 0..24 {
        for j in 0..24 {
            let s: f64 = (0..9).map(|m| t[i * 9 + m] * kua[j * 9 + m]).sum();
            k[i * 24 + j] -= s;
        }
    }
    // symmetrise round-off
    for i in 0..24 {
        for j in 0..i {
            let a = 0.5 * (k[i * 24 + j] + k[j * 24 + i]);
            k[i * 24 + j] = a;
            k[j * 24 + i] = a;
        }
    }
    let mut r = vec![0.0; 9 * 24];
    for i in 0..9 {
        for j in 0..24 {
            r[i * 24 + j] = -(0..9).map(|q| kinv[i * 9 + q] * kua[j * 9 + q]).sum::<f64>();
        }
    }
    ElemUnit { k, r }
}

fn inv3(j: &[[f64; 3]; 3]) -> ([[f64; 3]; 3], f64) {
    let det = j[0][0] * (j[1][1] * j[2][2] - j[1][2] * j[2][1]) - j[0][1] * (j[1][0] * j[2][2] - j[1][2] * j[2][0]) + j[0][2] * (j[1][0] * j[2][1] - j[1][1] * j[2][0]);
    let id = 1.0 / det;
    let m = [
        [(j[1][1] * j[2][2] - j[1][2] * j[2][1]) * id, (j[0][2] * j[2][1] - j[0][1] * j[2][2]) * id, (j[0][1] * j[1][2] - j[0][2] * j[1][1]) * id],
        [(j[1][2] * j[2][0] - j[1][0] * j[2][2]) * id, (j[0][0] * j[2][2] - j[0][2] * j[2][0]) * id, (j[0][2] * j[1][0] - j[0][0] * j[1][2]) * id],
        [(j[1][0] * j[2][1] - j[1][1] * j[2][0]) * id, (j[0][1] * j[2][0] - j[0][0] * j[2][1]) * id, (j[0][0] * j[1][1] - j[0][1] * j[1][0]) * id],
    ];
    (m, det)
}

/// Jacobian J[a][b] = ∂x_a/∂ξ_b of a general hex (corners in CORNER order) at ξ.
fn jacobian(x: &[[f64; 3]; 8], xi: [f64; 3]) -> [[f64; 3]; 3] {
    let mut j = [[0.0; 3]; 3];
    for i in 0..8 {
        let g = dn_dxi(i, xi);
        for a in 0..3 {
            for b in 0..3 {
                j[a][b] += x[i][a] * g[b];
            }
        }
    }
    j
}

/// grad_x = J⁻ᵀ grad_ξ
fn map_grad(jinv: &[[f64; 3]; 3], g: [f64; 3]) -> [f64; 3] {
    [0, 1, 2].map(|a| (0..3).map(|b| jinv[b][a] * g[b]).sum())
}

/// Strain matrices at ξ for a general hex: (Bu 24 columns, Ba 9 columns, det J). Incompatible
/// modes use the centre Jacobian scaled by det J₀ / det J (Taylor, Beresford & Wilson), which
/// keeps the element passing the patch test when distorted.
fn general_b(x: &[[f64; 3]; 8], xi: [f64; 3], j0inv: &[[f64; 3]; 3], det0: f64) -> (Vec<[f64; 6]>, Vec<[f64; 6]>, f64) {
    let (jinv, det) = inv3(&jacobian(x, xi));
    let gu: Vec<[f64; 3]> = (0..8).map(|i| map_grad(&jinv, dn_dxi(i, xi))).collect();
    let ga: Vec<[f64; 3]> = (0..3)
        .map(|k| {
            let mut g = [0.0; 3];
            g[k] = -2.0 * xi[k];
            map_grad(j0inv, g).map(|v| v * det0 / det)
        })
        .collect();
    (bmat(&gu), bmat(&ga), det)
}

/// Condensed stiffness (E = 1, SI coordinates), recovery matrix and unit thermal load of a general
/// (boundary-fitted) hexahedron.
pub struct ElemGeneral {
    pub k: Vec<f64>,
    pub r: Vec<f64>,
    /// nodal forces for E = 1, α·ΔT = 1 (24)
    pub fth: Vec<f64>,
}

pub fn hex_elastic_general(x: &[[f64; 3]; 8], nu: f64) -> ElemGeneral {
    let d = dmat(nu);
    let (j0inv, det0) = inv3(&jacobian(x, [0.0; 3]));
    let mut kuu = vec![0.0; 24 * 24];
    let mut kua = vec![0.0; 24 * 9];
    let mut kaa = vec![0.0; 9 * 9];
    let mut fth = vec![0.0; 24];
    let mut sig = [0.0; 6];
    for i in 0..6 {
        for j in 0..3 {
            sig[i] += d[i][j];
        }
    }
    for xi in gauss_points() {
        let (bu, ba, det) = general_b(x, xi, &j0inv, det0);
        btdb(&bu, &d, &bu, det, &mut kuu, 24);
        btdb(&bu, &d, &ba, det, &mut kua, 9);
        btdb(&ba, &d, &ba, det, &mut kaa, 9);
        for (c, col) in bu.iter().enumerate() {
            fth[c] += (0..6).map(|k| col[k] * sig[k]).sum::<f64>() * det;
        }
    }
    let kinv = dense_inverse(&kaa, 9);
    let mut t = vec![0.0; 24 * 9];
    for i in 0..24 {
        for j in 0..9 {
            t[i * 9 + j] = (0..9).map(|k| kua[i * 9 + k] * kinv[k * 9 + j]).sum();
        }
    }
    let mut k = kuu;
    for i in 0..24 {
        for j in 0..24 {
            let s: f64 = (0..9).map(|m| t[i * 9 + m] * kua[j * 9 + m]).sum();
            k[i * 24 + j] -= s;
        }
    }
    for i in 0..24 {
        for j in 0..i {
            let a = 0.5 * (k[i * 24 + j] + k[j * 24 + i]);
            k[i * 24 + j] = a;
            k[j * 24 + i] = a;
        }
    }
    let mut r = vec![0.0; 9 * 24];
    for i in 0..9 {
        for j in 0..24 {
            r[i * 24 + j] = -(0..9).map(|q| kinv[i * 9 + q] * kua[j * 9 + q]).sum::<f64>();
        }
    }
    ElemGeneral { k, r, fth }
}

/// Element corner coordinates in metres.
pub fn elem_xyz_m(mesh: &Mesh, e: usize) -> [[f64; 3]; 8] {
    mesh.elem_nodes(e).map(|p| p.map(|v| v * 1e-3))
}

/// Strain matrices (unit cube) at each corner node: (Bu columns 24, Ba columns 9).
pub fn corner_bmats() -> Vec<(Vec<[f64; 6]>, Vec<[f64; 6]>)> {
    (0..8)
        .map(|c| {
            let x = sgn(c);
            let gu: Vec<[f64; 3]> = (0..8).map(|i| dn_dxi(i, x).map(|v| 2.0 * v)).collect();
            let ga: Vec<[f64; 3]> = (0..3)
                .map(|k| {
                    let mut g = [0.0; 3];
                    g[k] = 2.0 * (-2.0 * x[k]);
                    g
                })
                .collect();
            (bmat(&gu), bmat(&ga))
        })
        .collect()
}

/// Thermal-strain nodal load for unit cube, E = 1, alpha*dT = 1.
pub fn thermal_load_unit(nu: f64) -> Vec<f64> {
    let d = dmat(nu);
    let eth = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0];
    let mut sig = [0.0; 6];
    for i in 0..6 {
        for j in 0..6 {
            sig[i] += d[i][j] * eth[j];
        }
    }
    let mut f = vec![0.0; 24];
    for x in gauss_points() {
        let gu: Vec<[f64; 3]> = (0..8).map(|i| dn_dxi(i, x).map(|v| 2.0 * v)).collect();
        for (c, col) in bmat(&gu).iter().enumerate() {
            f[c] += (0..6).map(|k| col[k] * sig[k]).sum::<f64>() * 0.125;
        }
    }
    f
}

/// Unit-cube conduction matrix (k = 1, h = 1), row-major 8x8.
pub fn hex_conduction_unit() -> Vec<f64> {
    let mut k = vec![0.0; 64];
    for x in gauss_points() {
        let g: Vec<[f64; 3]> = (0..8).map(|i| dn_dxi(i, x).map(|v| 2.0 * v)).collect();
        for a in 0..8 {
            for b in 0..8 {
                k[a * 8 + b] += dot(g[a], g[b]) * 0.125;
            }
        }
    }
    k
}

/// Conduction unit matrices for each axis separately (∫ ∂Nᵢ/∂x_a ∂Nⱼ/∂x_a), for orthotropic k.
pub fn hex_conduction_unit_axes() -> [Vec<f64>; 3] {
    let mut k = [vec![0.0; 64], vec![0.0; 64], vec![0.0; 64]];
    for x in gauss_points() {
        let g: Vec<[f64; 3]> = (0..8).map(|i| dn_dxi(i, x).map(|v| 2.0 * v)).collect();
        for ax in 0..3 {
            for a in 0..8 {
                for b in 0..8 {
                    k[ax][a * 8 + b] += g[a][ax] * g[b][ax] * 0.125;
                }
            }
        }
    }
    k
}

/// Shape-function gradients at element centre for unit cube.
pub fn center_grads() -> Vec<[f64; 3]> {
    (0..8).map(|i| dn_dxi(i, [0.0; 3]).map(|v| 2.0 * v)).collect()
}

pub fn face_nodes(mesh: &Mesh, faces: &[u32]) -> Vec<u32> {
    let mut v: Vec<u32> = faces.iter().flat_map(|&f| mesh.faces[f as usize].nodes).collect();
    v.sort_unstable();
    v.dedup();
    v
}

pub fn solve(model: &Model, mesh: &Mesh, prog: Prog) -> Result<Results> {
    let t0 = Instant::now();
    let kind = model.analysis.kind;
    crate::gpu::set_mode(model.analysis.compute);
    crate::gpu::take_fallback();
    crate::sparse::SOLVE_NANOS.store(0, std::sync::atomic::Ordering::Relaxed);
    let mut res = Results::new(kind);
    res.warnings = mesh.warnings.clone();
    res.info.nodes = mesh.nodes.len();
    res.info.elements = mesh.elems.len();
    res.info.element_size_mm = mesh.h;
    res.info.mass_kg = (0..mesh.elems.len())
        .filter(|&e| mesh.elem_region[e] == crate::mesh::REGION_SOLID)
        .map(|e| mesh.materials[mesh.elem_mat[e] as usize].density * mesh.elem_volume(e) * 1e-9)
        .sum();

    if !mesh.fitted.is_empty() {
        res.quantity("Boundary-fitted elements", mesh.fitted.iter().filter(|f| **f).count() as f64, "");
    }
    for it in &model.loads {
        if !it.load.applies_to(kind) {
            res.warnings.push(format!("load #{} ({}) is ignored in a {} analysis", it.id, it.load.label(), kind.label()));
        }
    }
    if kind.requires_enclosure() && !mesh.with_enclosure {
        bail!("{} needs an enclosure (the surrounding air or fluid region): use add_enclosure / GEOMETRY ▸ Enclosure", kind.label());
    }

    use AnalysisType::*;
    match kind {
        StaticStructural => structural(model, mesh, None, prog, &mut res)?,
        SteadyThermal => thermal(model, mesh, None, prog, &mut res)?,
        ThermalStress => {
            thermal(model, mesh, None, &|f, m| prog(f * 0.4, m), &mut res)?;
            let t = res.temp.clone().unwrap();
            res.reactions.clear();
            structural(model, mesh, Some(&t), &|f, m| prog(0.4 + 0.6 * f, m), &mut res)?;
        }
        Modal => modal(model, mesh, prog, &mut res)?,
        HarmonicResponse => crate::vibration::harmonic(model, mesh, prog, &mut res)?,
        ResponseSpectrum => crate::vibration::spectrum(model, mesh, prog, &mut res)?,
        RandomVibration => crate::vibration::random(model, mesh, prog, &mut res)?,
        LinearBuckling => crate::dynamics::buckling(model, mesh, prog, &mut res)?,
        TransientStructural => crate::dynamics::transient_structural(model, mesh, prog, &mut res)?,
        NonlinearStructural => crate::dynamics::plasticity(model, mesh, prog, &mut res)?,
        TransientThermal => crate::dynamics::transient_thermal(model, mesh, prog, &mut res)?,
        FullWaveEm => {
            crate::fdtd::full_wave(model, mesh, prog, &mut res)?;
        }
        MicrowaveHeating => crate::fdtd::microwave_heating(model, mesh, prog, &mut res)?,
        Electrostatic => crate::fields::electrostatic(model, mesh, prog, &mut res)?,
        CurrentFlow => {
            crate::fields::current_flow(model, mesh, prog, &mut res)?;
        }
        ThermalElectric => {
            let q = crate::fields::current_flow(model, mesh, &|f, m| prog(f * 0.4, m), &mut res)?;
            let currents = std::mem::take(&mut res.reactions);
            thermal(model, mesh, Some(&q), &|f, m| prog(0.4 + 0.6 * f, m), &mut res)?;
            for r in std::mem::replace(&mut res.reactions, currents) {
                res.quantity(&format!("Heat flow through #{} ({})", r.load_id, r.label), r.value[0], "W");
            }
        }
        Magnetostatic => crate::fields::magnetostatic(model, mesh, prog, &mut res)?,
        Fluid | ConjugateHeatTransfer => crate::cfd::solve(model, mesh, prog, &mut res)?,
    }
    // a non-finite field means a degenerate model (zero stiffness, extreme properties, unstable
    // explicit run): report it instead of showing garbage
    let bad = |v: &[f64]| v.iter().any(|x| !x.is_finite());
    let nonfinite = res.disp.as_ref().is_some_and(|d| d.iter().any(|p| !(p[0].is_finite() && p[1].is_finite() && p[2].is_finite())))
        || res.temp.as_ref().is_some_and(|t| bad(t))
        || res.vm.as_ref().is_some_and(|t| bad(t))
        || res.extra.values().any(|v| bad(v));
    if nonfinite {
        bail!("the solution is not finite (NaN or infinity): check for parts without material stiffness or conductivity, extreme property values, or loads on a part that is not connected to any support");
    }
    if let Some(w) = crate::gpu::take_fallback() {
        res.warnings.push(w);
    }
    res.info.linear_solve_seconds = (crate::sparse::SOLVE_NANOS.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e6).round() / 1000.0;
    res.info.seconds = (t0.elapsed().as_secs_f64() * 1000.0).round() / 1000.0;
    Ok(res)
}

pub struct Assembled {
    pub k: NodeSparse,
    pub fixed: Vec<Option<f64>>,
    pub support_nodes: Vec<(u32, String, Vec<u32>)>,
    /// Compression-only (contact) dofs with the outward normal sign of the face.
    pub contact: Vec<(usize, f64)>,
    /// Load id owning each constrained dof (0 = free) — used for per-support reactions.
    pub owner: Vec<u32>,
}

pub fn assemble_structural(model: &Model, mesh: &Mesh, prog: Prog) -> Result<(Assembled, Vec<ElemUnit>)> {
    let h = mesh.h * 1e-3;
    let nn = mesh.nodes.len();
    prog(0.0, "assembling stiffness");
    let units: Vec<ElemUnit> = mesh.materials.iter().map(|m| hex_elastic_unit(m.poisson_ratio)).collect();
    let mut k = NodeSparse::new(nn, &mesh.elems, 3);
    let general: Vec<Option<ElemGeneral>> = (0..mesh.elems.len())
        .into_par_iter()
        .map(|e| mesh.is_fitted(e).then(|| hex_elastic_general(&elem_xyz_m(mesh, e), mesh.materials[mesh.elem_mat[e] as usize].poisson_ratio)))
        .collect();
    for (e, en) in mesh.elems.iter().enumerate() {
        let m = &mesh.materials[mesh.elem_mat[e] as usize];
        match &general[e] {
            Some(g) => k.add_element(en, &g.k, m.youngs_modulus_gpa * 1e9),
            None => k.add_element(en, &units[mesh.elem_mat[e] as usize].k, m.youngs_modulus_gpa * 1e9 * h),
        }
    }
    let mut fixed = vec![None; nn * 3];
    let mut support_nodes = Vec::new();
    let mut contact = Vec::new();
    let mut owner = vec![0u32; nn * 3];
    for it in &model.loads {
        match &it.load {
            Load::FixedSupport { faces } => {
                let nodes = face_nodes(mesh, &mesh.resolve_faces(faces)?);
                for &n in &nodes {
                    for c in 0..3 {
                        fixed[n as usize * 3 + c] = Some(0.0);
                        owner[n as usize * 3 + c] = it.id;
                    }
                }
                support_nodes.push((it.id, it.load.label().to_string(), nodes));
            }
            Load::Displacement { faces, x, y, z } => {
                let nodes = face_nodes(mesh, &mesh.resolve_faces(faces)?);
                for &n in &nodes {
                    for (c, v) in [x, y, z].iter().enumerate() {
                        if let Some(v) = v {
                            fixed[n as usize * 3 + c] = Some(v * 1e-3);
                            owner[n as usize * 3 + c] = it.id;
                        }
                    }
                }
                support_nodes.push((it.id, it.load.label().to_string(), nodes));
            }
            Load::CompressionOnly { faces } | Load::Symmetry { faces } => {
                let fs = mesh.resolve_faces(faces)?;
                let is_contact = matches!(it.load, Load::CompressionOnly { .. });
                for &fi in &fs {
                    let face = &mesh.faces[fi as usize];
                    let a = (face.dir / 2) as usize;
                    let sign = if face.dir % 2 == 0 { -1.0 } else { 1.0 };
                    for &n in &face.nodes {
                        let dof = n as usize * 3 + a;
                        if fixed[dof].is_none() {
                            fixed[dof] = Some(0.0);
                            owner[dof] = it.id;
                            if is_contact {
                                contact.push((dof, sign));
                            }
                        }
                    }
                }
                support_nodes.push((it.id, it.load.label().to_string(), face_nodes(mesh, &fs)));
            }
            _ => {}
        }
    }
    contact.sort_unstable_by_key(|c| c.0);
    contact.dedup_by_key(|c| c.0);
    Ok((Assembled { k, fixed, support_nodes, contact, owner }, units))
}

/// External nodal forces [N]: forces, pressure, gravity and thermal-expansion loads.
pub fn external_forces(model: &Model, mesh: &Mesh, temps: Option<&[f64]>) -> Result<Vec<f64>> {
    let h = mesh.h * 1e-3;
    let nn = mesh.nodes.len();
    let tunits: Vec<Vec<f64>> = mesh.materials.iter().map(|m| thermal_load_unit(m.poisson_ratio)).collect();
    let tref = model.analysis.reference_temperature;
    let elem_dt = |e: usize| -> f64 {
        match temps {
            Some(t) => mesh.elems[e].iter().map(|&n| t[n as usize]).sum::<f64>() / 8.0 - tref,
            None => 0.0,
        }
    };
    let mut f = vec![0.0; nn * 3];
    for it in &model.loads {
        match &it.load {
            Load::Force { faces, vector } => {
                let fs = mesh.resolve_faces(faces)?;
                let share = 1.0 / (fs.len() as f64 * 4.0);
                for &fi in &fs {
                    for &n in &mesh.faces[fi as usize].nodes {
                        for c in 0..3 {
                            f[n as usize * 3 + c] += vector[c] * share;
                        }
                    }
                }
            }
            Load::Pressure { faces, value_mpa } => {
                for &fi in &mesh.resolve_faces(faces)? {
                    let face = &mesh.faces[fi as usize];
                    // consistent nodal loads ∫ Nᵢ p n dA (exact for any bilinear, possibly
                    // boundary-fitted quad; equals pA/4 per node on a square face)
                    let shares: [V3; 4] = if mesh.fitted.is_empty() {
                        [scale(dir_normal(face.dir), h * h / 4.0); 4]
                    } else {
                        let q = face.nodes.map(|n| mesh.nodes[n as usize].map(|v| v * 1e-3));
                        let mut sh = [[0.0; 3]; 4];
                        for (a, b) in [(-GP, -GP), (GP, -GP), (GP, GP), (-GP, GP)] {
                            let nn = [(1.0 - a) * (1.0 - b) / 4.0, (1.0 + a) * (1.0 - b) / 4.0, (1.0 + a) * (1.0 + b) / 4.0, (1.0 - a) * (1.0 + b) / 4.0];
                            let da = [-(1.0 - b) / 4.0, (1.0 - b) / 4.0, (1.0 + b) / 4.0, -(1.0 + b) / 4.0];
                            let db = [-(1.0 - a) / 4.0, -(1.0 + a) / 4.0, (1.0 + a) / 4.0, (1.0 - a) / 4.0];
                            let mut ta = [0.0; 3];
                            let mut tb = [0.0; 3];
                            for k in 0..4 {
                                ta = add(ta, scale(q[k], da[k]));
                                tb = add(tb, scale(q[k], db[k]));
                            }
                            let dav = cross(ta, tb); // outward area density (nodes CCW from outside)
                            for k in 0..4 {
                                sh[k] = add(sh[k], scale(dav, nn[k]));
                            }
                        }
                        sh
                    };
                    let p = value_mpa * 1e6;
                    for (k, &n) in face.nodes.iter().enumerate() {
                        for c in 0..3 {
                            f[n as usize * 3 + c] -= shares[k][c] * p;
                        }
                    }
                }
            }
            Load::Gravity { acceleration } => {
                for (e, en) in mesh.elems.iter().enumerate() {
                    let m = mesh.materials[mesh.elem_mat[e] as usize].density * mesh.elem_volume(e) * 1e-9 / 8.0;
                    for &n in en {
                        for c in 0..3 {
                            f[n as usize * 3 + c] += acceleration[c] * m;
                        }
                    }
                }
            }
            _ => {}
        }
    }
    if temps.is_some() {
        for (e, en) in mesh.elems.iter().enumerate() {
            let mi = mesh.elem_mat[e] as usize;
            let m = &mesh.materials[mi];
            let eth = m.youngs_modulus_gpa * 1e9 * m.thermal_expansion * elem_dt(e);
            let (s, ft) = if mesh.is_fitted(e) { (eth, hex_elastic_general(&elem_xyz_m(mesh, e), m.poisson_ratio).fth) } else { (eth * h * h, tunits[mi].clone()) };
            for a in 0..8 {
                for c in 0..3 {
                    f[en[a] as usize * 3 + c] += s * ft[a * 3 + c];
                }
            }
        }
    }
    Ok(f)
}

pub fn structural(model: &Model, mesh: &Mesh, temps: Option<&[f64]>, prog: Prog, res: &mut Results) -> Result<()> {
    let nn = mesh.nodes.len();
    let (asm, units) = assemble_structural(model, mesh, prog)?;
    let Assembled { k: k0, mut fixed, support_nodes, contact, owner } = asm;
    if support_nodes.is_empty() {
        bail!("no supports defined: add a Fixed Support (or Displacement) so the part cannot fly away");
    }
    let f_ext = external_forces(model, mesh, temps)?;
    let fscale = f_ext.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-30);
    let mut u: Vec<f64>;
    let mut pass = 0;
    loop {
        let mut k = k0.clone();
        let mut f = f_ext.clone();
        k.apply_dirichlet(&fixed, &mut f);
        u = fixed.iter().map(|g| g.unwrap_or(0.0)).collect();
        prog(0.05, "factorising");
        let solver = Solver::new(&k).map_err(|e| if contact.is_empty() { e } else { anyhow::anyhow!("{e} (the part may be lifting off its compression-only supports: add a support that holds it)") })?;
        let its = solver.solve(&k, &f, &mut u, 1e-8, &|p| prog(0.05 + 0.85 * p, "solving (PCG)"))?;
        res.info.solver = solver.name().to_string();
        res.info.iterations += its;
        if contact.is_empty() {
            break;
        }
        // contact active set: release supports in tension, close gaps that penetrate
        let mut ku = vec![0.0; nn * 3];
        k0.matvec(&u, &mut ku);
        let mut changed = 0;
        for &(dof, sgn) in &contact {
            if fixed[dof].is_some() {
                if (ku[dof] - f_ext[dof]) * sgn > 1e-7 * fscale {
                    fixed[dof] = None;
                    changed += 1;
                }
            } else if u[dof] * sgn > 1e-12 {
                fixed[dof] = Some(0.0);
                changed += 1;
            }
        }
        pass += 1;
        prog(0.5, &format!("contact iteration {pass}: {changed} change(s)"));
        if changed == 0 {
            break;
        }
        if pass >= 60 {
            res.warnings.push("contact status did not fully settle after 60 iterations".into());
            break;
        }
    }
    if !contact.is_empty() {
        let open = contact.iter().filter(|(d, _)| fixed[*d].is_none()).count();
        res.quantity("Contact nodes in contact", (contact.len() - open) as f64, "");
        res.quantity("Contact nodes lifted off", open as f64, "");
    }
    res.info.dofs = nn * 3;
    prog(0.92, "computing stresses");
    structural_post(model, mesh, &u, temps, &units, &fixed, &f_ext, &support_nodes, &owner, res);
    prog(1.0, "done");
    Ok(())
}

/// Stresses, safety factor and reactions from a displacement field `u` [m].
#[allow(clippy::too_many_arguments)]
pub fn structural_post(model: &Model, mesh: &Mesh, u: &[f64], temps: Option<&[f64]>, units: &[ElemUnit], fixed: &[Option<f64>], f_ext: &[f64], support_nodes: &[(u32, String, Vec<u32>)], owner: &[u32], res: &mut Results) {
    let h = mesh.h * 1e-3;
    let nn = mesh.nodes.len();
    let tunits: Vec<Vec<f64>> = mesh.materials.iter().map(|m| thermal_load_unit(m.poisson_ratio)).collect();
    let tref = model.analysis.reference_temperature;
    let elem_dt = |e: usize| -> f64 {
        match temps {
            Some(t) => mesh.elems[e].iter().map(|&n| t[n as usize]).sum::<f64>() / 8.0 - tref,
            None => 0.0,
        }
    };
    // stresses recovered at element corners (incl. incompatible modes) and internal forces
    let cb = corner_bmats();
    let per_elem: Vec<([[f64; 6]; 8], [f64; 24])> = mesh
        .elems
        .par_iter()
        .enumerate()
        .map(|(e, en)| {
            let mi = mesh.elem_mat[e] as usize;
            let m = &mesh.materials[mi];
            let ue: Vec<f64> = (0..24).map(|i| u[en[i / 3] as usize * 3 + i % 3]).collect();
            if mesh.is_fitted(e) {
                return general_post(mesh, e, m, &ue, m.thermal_expansion * elem_dt(e));
            }
            let eu = &units[mi];
            let alpha: Vec<f64> = (0..9).map(|i| (0..24).map(|j| eu.r[i * 24 + j] * ue[j]).sum()).collect();
            let ath = m.thermal_expansion * elem_dt(e);
            let d = dmat(m.poisson_ratio);
            let e_pa = m.youngs_modulus_gpa * 1e9;
            let mut sigs = [[0.0; 6]; 8];
            for (corner, (bu, ba)) in cb.iter().enumerate() {
                let mut eps = [0.0; 6];
                for (c, col) in bu.iter().enumerate() {
                    for r in 0..6 {
                        eps[r] += col[r] * ue[c] / h;
                    }
                }
                for (c, col) in ba.iter().enumerate() {
                    for r in 0..6 {
                        eps[r] += col[r] * alpha[c] / h;
                    }
                }
                for r in 0..3 {
                    eps[r] -= ath;
                }
                for r in 0..6 {
                    for c in 0..6 {
                        sigs[corner][r] += d[r][c] * eps[c] * e_pa * 1e-6;
                    }
                }
            }
            let mut fi = [0.0; 24];
            let ks = e_pa * h;
            let ts = e_pa * m.thermal_expansion * elem_dt(e) * h * h;
            for i in 0..24 {
                let mut s = 0.0;
                for j in 0..24 {
                    s += eu.k[i * 24 + j] * ue[j];
                }
                fi[i] = s * ks - ts * tunits[mi][i];
            }
            (sigs, fi)
        })
        .collect();

    let mut fint = vec![0.0; nn * 3];
    let mut acc = vec![[0.0f64; 6]; nn];
    let mut cnt = vec![0u32; nn];
    let mut yld = vec![f64::INFINITY; nn];
    for (e, en) in mesh.elems.iter().enumerate() {
        let (sigs, fi) = &per_elem[e];
        let ys = mesh.materials[mesh.elem_mat[e] as usize].yield_strength_mpa;
        for (a, &n) in en.iter().enumerate() {
            let n = n as usize;
            for c in 0..3 {
                fint[n * 3 + c] += fi[a * 3 + c];
            }
            for r in 0..6 {
                acc[n][r] += sigs[a][r];
            }
            cnt[n] += 1;
            yld[n] = yld[n].min(ys);
        }
    }
    let stress: Vec<[f64; 6]> = acc.iter().zip(&cnt).map(|(s, &c)| s.map(|v| v / c.max(1) as f64)).collect();
    let vm: Vec<f64> = stress
        .iter()
        .map(|s| {
            (0.5 * ((s[0] - s[1]).powi(2) + (s[1] - s[2]).powi(2) + (s[2] - s[0]).powi(2))
                + 3.0 * (s[3] * s[3] + s[4] * s[4] + s[5] * s[5]))
                .sqrt()
        })
        .collect();
    let princ: Vec<(f64, f64)> = stress
        .par_iter()
        .map(|s| {
            let m = [s[0], s[3], s[5], s[3], s[1], s[4], s[5], s[4], s[2]];
            let (ev, _) = sym_eigen(&m, 3);
            let mx = ev.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let mn = ev.iter().cloned().fold(f64::INFINITY, f64::min);
            (mx, mn)
        })
        .collect();
    let safety: Vec<f64> = vm.iter().zip(&yld).map(|(v, y)| if *v > 1e-12 { (y / v).min(15.0) } else { 15.0 }).collect();

    // reactions per support: each constrained dof belongs to the support that fixed it
    for (id, label, _) in support_nodes {
        let mut r = [0.0; 3];
        for dof in 0..nn * 3 {
            if owner[dof] == *id && fixed[dof].is_some() {
                r[dof % 3] += fint[dof] - f_ext[dof];
            }
        }
        res.reactions.push(Reaction { load_id: *id, label: label.clone(), value: r.map(float) });
    }

    res.disp = Some((0..nn).map(|n| [u[n * 3] * 1e3, u[n * 3 + 1] * 1e3, u[n * 3 + 2] * 1e3]).collect());
    res.vm = Some(vm);
    res.s1 = Some(princ.iter().map(|p| p.0).collect());
    res.s3 = Some(princ.iter().map(|p| p.1).collect());
    res.safety = Some(safety);
    res.stress = Some(stress);
}

/// Corner stresses [MPa] and internal forces [N] of a boundary-fitted element.
fn general_post(mesh: &Mesh, e: usize, m: &crate::material::Material, ue: &[f64], ath: f64) -> ([[f64; 6]; 8], [f64; 24]) {
    let x = elem_xyz_m(mesh, e);
    let g = hex_elastic_general(&x, m.poisson_ratio);
    let (j0inv, det0) = inv3(&jacobian(&x, [0.0; 3]));
    let alpha: Vec<f64> = (0..9).map(|i| (0..24).map(|j| g.r[i * 24 + j] * ue[j]).sum()).collect();
    let d = dmat(m.poisson_ratio);
    let e_pa = m.youngs_modulus_gpa * 1e9;
    // stresses at the 2×2×2 Gauss points (well-behaved in distorted elements), then trilinear
    // extrapolation to the corners
    let gps = gauss_points();
    let mut sig_g = [[0.0; 6]; 8];
    for (gi, sig) in sig_g.iter_mut().enumerate() {
        let (bu, ba, _) = general_b(&x, gps[gi], &j0inv, det0);
        let mut eps = [0.0; 6];
        for (c, col) in bu.iter().enumerate() {
            for r in 0..6 {
                eps[r] += col[r] * ue[c];
            }
        }
        for (c, col) in ba.iter().enumerate() {
            for r in 0..6 {
                eps[r] += col[r] * alpha[c];
            }
        }
        for v in eps.iter_mut().take(3) {
            *v -= ath;
        }
        for r in 0..6 {
            for c in 0..6 {
                sig[r] += d[r][c] * eps[c] * e_pa * 1e-6;
            }
        }
    }
    let r3 = 3f64.sqrt();
    let mut sigs = [[0.0; 6]; 8];
    for (corner, sc) in sigs.iter_mut().enumerate() {
        let eta = sgn(corner).map(|v| v * r3);
        for (gi, sg) in sig_g.iter().enumerate() {
            let s_g = gps[gi].map(|v| v.signum());
            let w: f64 = (0..3).map(|k| (1.0 + s_g[k] * eta[k]) / 2.0).product();
            for r in 0..6 {
                sc[r] += w * sg[r];
            }
        }
    }
    let mut fi = [0.0; 24];
    for i in 0..24 {
        let s: f64 = (0..24).map(|j| g.k[i * 24 + j] * ue[j]).sum();
        fi[i] = s * e_pa - e_pa * ath * g.fth[i];
    }
    (sigs, fi)
}

/// Cells and power density [W/m³] of a heat-generation load. A total `power` is spread over the
/// body's cells; a W/m³ value on a primitive body (box, cylinder, sphere not cut by another body) is
/// rescaled so the generated power matches the exact body volume, not the voxel approximation.
pub fn heat_source(model: &Model, mesh: &Mesh, load: &Load) -> Result<(Vec<usize>, f64)> {
    let Load::HeatGeneration { value, body, power } = load else { bail!("not a heat-generation load") };
    let els: Vec<usize> = match body {
        Some(b) => mesh.body_elems(b)?,
        None => (0..mesh.elems.len()).filter(|&e| mesh.elem_region[e] == crate::mesh::REGION_SOLID).collect(),
    };
    if els.is_empty() {
        bail!("heat generation: the body has no cells in the mesh (too small for the element size?)");
    }
    let vox = els.len() as f64 * (mesh.h * 1e-3).powi(3);
    if let Some(p) = power {
        return Ok((els, p / vox));
    }
    let mut q = *value;
    if let Some(name) = body {
        if let Some(b) = model.bodies.iter().find(|b| b.name.eq_ignore_ascii_case(name)) {
            let (lo, hi) = b.shape.bounds();
            let cut = model.bodies.iter().any(|o| {
                o.op == crate::geometry::Op::Subtract && {
                    let (a, c) = o.shape.bounds();
                    (0..3).all(|k| a[k] < hi[k] && c[k] > lo[k])
                }
            });
            if let (Some(v), false) = (b.shape.volume(), cut) {
                q *= v * 1e-9 / vox;
            }
        }
    }
    Ok((els, q))
}

/// Conduction system with convection folded into K and loads in f (before Dirichlet).
#[derive(Clone)]
pub struct ThermalSystem {
    pub k: NodeSparse,
    pub f: Vec<f64>,
    pub fixed: Vec<Option<f64>>,
    pub conv_diag: Vec<f64>,
    pub temp_sets: Vec<(u32, String, Vec<u32>)>,
    pub has_sink: bool,
}

/// Assemble conduction + thermal loads. `elem_q` adds volumetric heat [W/m³] per element.
pub fn assemble_thermal(model: &Model, mesh: &Mesh, elem_q: Option<&[f64]>) -> Result<ThermalSystem> {
    assemble_thermal_sel(model, mesh, elem_q, None)
}

/// Like `assemble_thermal`, restricted to the elements where `sel` is true (e.g. the solids inside
/// an air cavity). Nodes not touched by a selected element are held at `rest_temperature`.
pub fn assemble_thermal_sel(model: &Model, mesh: &Mesh, elem_q: Option<&[f64]>, sel: Option<(&[bool], f64)>) -> Result<ThermalSystem> {
    let h = mesh.h * 1e-3;
    let nn = mesh.nodes.len();
    let ku = hex_conduction_unit();
    let on = |e: usize| sel.is_none_or(|(s, _)| s[e]);
    let elems: Vec<[u32; 8]> = match sel {
        Some((s, _)) => mesh.elems.iter().zip(s).filter(|(_, b)| **b).map(|(e, _)| *e).collect(),
        None => mesh.elems.clone(),
    };
    let mut k = NodeSparse::new(nn, &elems, 1);
    let kax = hex_conduction_unit_axes();
    for (e, en) in mesh.elems.iter().enumerate() {
        if on(e) {
            let m = &mesh.materials[mesh.elem_mat[e] as usize];
            match m.conductivity_xyz {
                None => k.add_element(en, &ku, m.thermal_conductivity * h),
                Some(kk) => {
                    for a in 0..3 {
                        k.add_element(en, &kax[a], kk[a] * h);
                    }
                }
            }
        }
    }
    let mut f = vec![0.0; nn];
    let mut fixed = vec![None; nn];
    if let Some((_, rest)) = sel {
        let mut touched = vec![false; nn];
        for en in &elems {
            for &n in en {
                touched[n as usize] = true;
            }
        }
        for n in 0..nn {
            if !touched[n] {
                fixed[n] = Some(rest);
                k.add_diag(n, 1.0);
            }
        }
    }
    let mut conv_diag = vec![0.0; nn];
    let mut temp_sets = Vec::new();
    let mut has_sink = false;
    for it in &model.loads {
        match &it.load {
            Load::Temperature { faces, value } => {
                let nodes = face_nodes(mesh, &mesh.resolve_faces(faces)?);
                for &n in &nodes {
                    fixed[n as usize] = Some(*value);
                }
                temp_sets.push((it.id, it.load.label().to_string(), nodes));
                has_sink = true;
            }
            Load::HeatFlux { faces, value } => {
                for &fi in &mesh.resolve_faces(faces)? {
                    for &n in &mesh.faces[fi as usize].nodes {
                        f[n as usize] += value * h * h / 4.0;
                    }
                }
            }
            Load::Convection { faces, coefficient, ambient } => {
                for &fi in &mesh.resolve_faces(faces)? {
                    for &n in &mesh.faces[fi as usize].nodes {
                        let c = coefficient * h * h / 4.0;
                        conv_diag[n as usize] += c;
                        f[n as usize] += c * ambient;
                    }
                }
                has_sink = true;
            }
            Load::HeatGeneration { .. } => {
                let (els, q) = heat_source(model, mesh, &it.load)?;
                for e in els {
                    for &n in &mesh.elems[e] {
                        f[n as usize] += q * h * h * h / 8.0;
                    }
                }
            }
            _ => {}
        }
    }
    if let Some(q) = elem_q {
        for (e, en) in mesh.elems.iter().enumerate().filter(|(e, _)| on(*e)) {
            for &n in en {
                f[n as usize] += q[e] * h * h * h / 8.0;
            }
        }
    }
    for (n, c) in conv_diag.iter().enumerate() {
        if *c != 0.0 {
            k.add_diag(n, *c);
        }
    }
    Ok(ThermalSystem { k, f, fixed, conv_diag, temp_sets, has_sink })
}

pub fn thermal(model: &Model, mesh: &Mesh, elem_q: Option<&[f64]>, prog: Prog, res: &mut Results) -> Result<()> {
    let nn = mesh.nodes.len();
    prog(0.0, "assembling conduction");
    let sys = assemble_thermal(model, mesh, elem_q)?;
    if !sys.has_sink {
        bail!("thermal model needs at least one Temperature or Convection boundary condition");
    }
    let mut k = sys.k.clone();
    let mut f = sys.f.clone();
    k.apply_dirichlet(&sys.fixed, &mut f);
    let mut t: Vec<f64> = sys.fixed.iter().map(|g| g.unwrap_or(model.analysis.reference_temperature)).collect();
    let solver = Solver::new(&k)?;
    let its = solver.solve(&k, &f, &mut t, 1e-10, &|p| prog(0.05 + 0.85 * p, "solving thermal (PCG)"))?;
    res.info.solver = solver.name().to_string();
    res.info.dofs += nn;
    res.info.iterations += its;
    thermal_post(mesh, &t, Some(&sys), res);
    res.temp = Some(t);
    prog(1.0, "done");
    Ok(())
}

/// Heat flux field and (optionally) heat flow through fixed-temperature faces.
pub fn thermal_post(mesh: &Mesh, t: &[f64], sys: Option<&ThermalSystem>, res: &mut Results) {
    let h = mesh.h * 1e-3;
    let nn = mesh.nodes.len();
    let ku = hex_conduction_unit();
    let kax = hex_conduction_unit_axes();
    let cg = center_grads();
    let mut fint = vec![0.0; nn];
    let mut acc = vec![[0.0; 3]; nn];
    let mut cnt = vec![0u32; nn];
    for (e, en) in mesh.elems.iter().enumerate() {
        let mat = &mesh.materials[mesh.elem_mat[e] as usize];
        let kc = mat.thermal_conductivity;
        let kv = [mat.k_axis(0), mat.k_axis(1), mat.k_axis(2)];
        let ortho = mat.conductivity_xyz.is_some();
        let mut grad = [0.0; 3];
        for a in 0..8 {
            for c in 0..3 {
                grad[c] += cg[a][c] * t[en[a] as usize] / h;
            }
        }
        for a in 0..8 {
            let n = en[a] as usize;
            let s: f64 = if ortho {
                (0..3).map(|ax| kv[ax] * (0..8).map(|b| kax[ax][a * 8 + b] * t[en[b] as usize]).sum::<f64>()).sum()
            } else {
                kc * (0..8).map(|b| ku[a * 8 + b] * t[en[b] as usize]).sum::<f64>()
            };
            fint[n] += s * h;
            for c in 0..3 {
                acc[n][c] += -kv[c] * grad[c];
            }
            cnt[n] += 1;
        }
    }
    if let Some(sys) = sys {
        let mut claimed = vec![false; nn];
        for (id, label, nodes) in &sys.temp_sets {
            let mut q = 0.0;
            for &n in nodes {
                let n = n as usize;
                if !claimed[n] {
                    claimed[n] = true;
                    q += fint[n] + sys.conv_diag[n] * t[n] - sys.f[n];
                }
            }
            // positive = heat flowing INTO the body through this face (negative = heat leaving)
            res.reactions.push(Reaction { load_id: *id, label: label.clone(), value: [float(q), 0.0, 0.0] });
        }
    }
    res.flux = Some(acc.iter().zip(&cnt).map(|(a, &c)| a.map(|v| v / c.max(1) as f64)).collect());
}

/// Mass-normalised modal basis: φᵢᵀ M φⱼ = δᵢⱼ (SI units, metres), lumped mass, constraints.
pub struct ModalBasis {
    /// Natural circular frequencies [rad/s], ascending.
    pub omega: Vec<f64>,
    /// Mode shapes, one dof vector per mode (zero on constrained dofs).
    pub phi: Vec<Vec<f64>>,
    /// Lumped mass per dof [kg] (zero on constrained dofs).
    pub mass: Vec<f64>,
    pub fixed: Vec<Option<f64>>,
}

pub fn modal(model: &Model, mesh: &Mesh, prog: Prog, res: &mut Results) -> Result<()> {
    let nn = mesh.nodes.len();
    let b = modal_basis(model, mesh, model.analysis.modes.clamp(1, 20), prog, res)?;
    res.modes = b
        .omega
        .iter()
        .zip(&b.phi)
        .map(|(w, v)| {
            let mx = (0..nn).map(|a| (v[a * 3].powi(2) + v[a * 3 + 1].powi(2) + v[a * 3 + 2].powi(2)).sqrt()).fold(0.0, f64::max).max(1e-300);
            Mode { load_multiplier: None, frequency_hz: w / (2.0 * std::f64::consts::PI), shape: (0..nn).map(|a| [v[a * 3] / mx, v[a * 3 + 1] / mx, v[a * 3 + 2] / mx]).collect() }
        })
        .collect();
    if res.modes.iter().take(6).filter(|m| m.frequency_hz < 1e-3 * res.modes.last().map(|l| l.frequency_hz).unwrap_or(1.0)).count() > 0 {
        res.warnings.push("near-zero frequencies found: the model has rigid-body modes (it is not fully supported)".into());
    }
    prog(1.0, "done");
    Ok(())
}

/// Lowest `nmodes` modes by shifted subspace iteration with Rayleigh–Ritz.
pub fn modal_basis(model: &Model, mesh: &Mesh, nmodes: usize, prog: Prog, res: &mut Results) -> Result<ModalBasis> {
    let h = mesh.h * 1e-3;
    let nn = mesh.nodes.len();
    let n = nn * 3;
    let (asm, _) = assemble_structural(model, mesh, prog)?;
    let Assembled { mut k, mut fixed, .. } = asm;
    for g in fixed.iter_mut() {
        if g.is_some() {
            *g = Some(0.0);
        }
    }
    let mut mdiag = vec![0.0; n];
    for (e, en) in mesh.elems.iter().enumerate() {
        let m = mesh.materials[mesh.elem_mat[e] as usize].density * h * h * h / 8.0;
        for &a in en {
            for c in 0..3 {
                mdiag[a as usize * 3 + c] += m;
            }
        }
    }
    let kd = k.diag();
    let ratio = kd.iter().sum::<f64>() / mdiag.iter().sum::<f64>();
    let sigma = -1e-4 * ratio; // shift so free-free models stay positive definite
    for i in 0..n {
        k.add_diag(i, -sigma * mdiag[i]);
    }
    let mut dummy = vec![0.0; n];
    k.apply_dirichlet(&fixed, &mut dummy);
    for i in 0..n {
        if fixed[i].is_some() {
            mdiag[i] = 0.0;
        }
    }
    prog(0.02, "factorising");
    let solver = Solver::new(&k)?;
    res.info.solver = solver.name().to_string();
    let free = fixed.iter().filter(|f| f.is_none()).count();
    let m = nmodes.clamp(1, 60).min(free.max(1));
    let p = (2 * m).min(m + 8).min(free);
    if p == 0 {
        bail!("all degrees of freedom are fixed");
    }

    // pseudo-random start vectors
    let mut seed = 0x2545F4914F6CDD1Du64;
    let mut rnd = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };
    let mut x: Vec<Vec<f64>> = (0..p).map(|_| (0..n).map(|i| if mdiag[i] > 0.0 { rnd() } else { 0.0 }).collect()).collect();
    let mut mu = vec![1.0f64; p];
    let mut old = vec![f64::INFINITY; p];
    let mut iters = 0;
    let max_it = 60;
    let mut first = true;
    for it in 0..max_it {
        prog(it as f64 / max_it as f64 * 0.95, &format!("subspace iteration {}", it + 1));
        let mut y = Vec::with_capacity(p);
        for j in 0..p {
            let rhs: Vec<f64> = x[j].iter().zip(&mdiag).map(|(a, b)| a * b).collect();
            let mut yj: Vec<f64> = if first { vec![0.0; n] } else { x[j].iter().map(|v| v / mu[j]).collect() };
            iters += solver.solve(&k, &rhs, &mut yj, 1e-8, &|_| {})?;
            y.push(yj);
        }
        first = false;
        // Rayleigh-Ritz
        let ay: Vec<Vec<f64>> = y
            .iter()
            .map(|v| {
                let mut o = vec![0.0; n];
                k.matvec(v, &mut o);
                o
            })
            .collect();
        let mut ka = vec![0.0; p * p];
        let mut ma = vec![0.0; p * p];
        for i in 0..p {
            for j in i..p {
                let a = dotp(&y[i], &ay[j]);
                let b: f64 = y[i].iter().zip(&y[j]).zip(&mdiag).map(|((a, b), m)| a * b * m).sum();
                ka[i * p + j] = a;
                ka[j * p + i] = a;
                ma[i * p + j] = b;
                ma[j * p + i] = b;
            }
        }
        let l = cholesky(&ma, p)?;
        // C = L^-1 Ka L^-T
        let linv = dense_inverse(&l, p);
        let mut tmp = vec![0.0; p * p];
        for i in 0..p {
            for j in 0..p {
                tmp[i * p + j] = (0..p).map(|q| linv[i * p + q] * ka[q * p + j]).sum();
            }
        }
        let mut c = vec![0.0; p * p];
        for i in 0..p {
            for j in 0..p {
                c[i * p + j] = (0..p).map(|q| tmp[i * p + q] * linv[j * p + q]).sum();
            }
        }
        let (ev, vecs) = sym_eigen(&c, p);
        let mut order: Vec<usize> = (0..p).collect();
        order.sort_by(|&a, &b| ev[a].total_cmp(&ev[b]));
        // Q = L^-T V
        let mut q = vec![0.0; p * p];
        for (col, &o) in order.iter().enumerate() {
            for i in 0..p {
                q[i * p + col] = (0..p).map(|r| linv[r * p + i] * vecs[r * p + o]).sum();
            }
        }
        for col in 0..p {
            mu[col] = ev[order[col]];
        }
        x = (0..p)
            .map(|col| {
                let mut v = vec![0.0; n];
                for (i, yi) in y.iter().enumerate() {
                    let s = q[i * p + col];
                    zip_apply(&mut v, yi, |a, b| *a += s * b);
                }
                v
            })
            .collect();
        let conv = (0..m).all(|i| ((mu[i] - old[i]) / mu[i]).abs() < 1e-6);
        old.copy_from_slice(&mu);
        if std::env::var("SIMCRAFT_DEBUG").is_ok() {
            eprintln!("subspace it {it}: inner cg iters so far {iters}, mu0 {}", mu[0]);
        }
        if conv && it > 1 {
            break;
        }
    }
    res.info.dofs = n;
    res.info.iterations = iters;
    let omega: Vec<f64> = (0..m).map(|i| (mu[i] + sigma).max(0.0).sqrt()).collect();
    x.truncate(m);
    Ok(ModalBasis { omega, phi: x, mass: mdiag, fixed })
}
