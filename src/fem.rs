//! Finite element solvers on the voxel hex mesh.
//!
//! * Structural: 8-node hexahedra with Wilson-Taylor incompatible modes (no shear locking in
//!   bending), statically condensed to 24x24.
//! * Thermal: 8-node conduction hexahedra (general hexes on boundary-fitted surfaces), convection /
//!   flux / generation loads with true face areas and consistent nodal volumes.
//! * Modal: lumped mass (consistent nodal volumes), shifted subspace iteration with Rayleigh-Ritz.
//!
//! Internally everything is SI (m, N, Pa, K).

use crate::geometry::*;
use crate::mesh::{CORNER, Mesh, dir_normal};
use crate::model::*;
use crate::results::*;
use crate::sparse::*;
use anyhow::{Result, bail};
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

/// Error returned when a long-running job is stopped cooperatively via an [`AtomicBool`].
#[derive(Debug, Clone, Copy)]
pub struct Cancelled;

impl std::fmt::Display for Cancelled {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("cancelled")
    }
}

impl std::error::Error for Cancelled {}

pub static NEVER_CANCEL: AtomicBool = AtomicBool::new(false);

#[inline]
pub fn check_cancel(cancel: &AtomicBool) -> Result<()> {
    if cancel.load(Ordering::Relaxed) {
        Err(anyhow::Error::new(Cancelled))
    } else {
        Ok(())
    }
}

/// Progress reporter paired with a shared cooperative cancellation flag (`Sync` across `rayon` workers).
#[derive(Clone, Copy)]
pub struct Progress<'a> {
    pub cancel: &'a AtomicBool,
    cb: &'a (dyn Fn(f64, &str) + Sync),
    offset: f64,
    scale: f64,
}

impl<'a> Progress<'a> {
    pub fn new(cancel: &'a AtomicBool, cb: &'a (dyn Fn(f64, &str) + Sync)) -> Self {
        Self { cancel, cb, offset: 0.0, scale: 1.0 }
    }
    pub fn cancel_only(cancel: &'a AtomicBool) -> Self {
        Self { cancel, cb: &|_, _| {}, offset: 0.0, scale: 1.0 }
    }
    pub fn noop() -> Self {
        Self::cancel_only(&NEVER_CANCEL)
    }
    pub fn slice(self, offset: f64, scale: f64) -> Self {
        Self { cancel: self.cancel, cb: self.cb, offset: self.offset + self.scale * offset, scale: self.scale * scale }
    }
    #[inline]
    pub fn check(self) -> Result<()> {
        check_cancel(self.cancel)
    }
    #[inline]
    pub fn report(self, frac: f64, msg: &str) -> Result<()> {
        self.check()?;
        (self.cb)(self.offset + self.scale * frac, msg);
        Ok(())
    }
    pub fn scalar(self, offset: f64, scale: f64, msg: &'a str) -> impl Fn(f64) -> Result<()> + Sync + 'a {
        let sub = self.slice(offset, scale);
        move |p| sub.report(p, msg)
    }
    pub fn check_scalar(self) -> impl Fn(f64) -> Result<()> + Sync + 'a {
        let cancel = self.cancel;
        move |_| check_cancel(cancel)
    }
}

pub type Prog<'a> = Progress<'a>;

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

/// Conduction matrices of a general (boundary-fitted) hex in SI coordinates: isotropic
/// ∫ ∇Nᵢ·∇Nⱼ dV and the per-axis parts ∫ ∂Nᵢ/∂x_a ∂Nⱼ/∂x_a dV (multiply by k in W/m·K).
pub fn hex_conduction_general(x: &[[f64; 3]; 8]) -> [Vec<f64>; 4] {
    let mut k = [vec![0.0; 64], vec![0.0; 64], vec![0.0; 64], vec![0.0; 64]];
    for xi in gauss_points() {
        let (jinv, det) = inv3(&jacobian(x, xi));
        let g: Vec<[f64; 3]> = (0..8).map(|i| map_grad(&jinv, dn_dxi(i, xi))).collect();
        for a in 0..8 {
            for b in 0..8 {
                for ax in 0..3 {
                    let v = g[a][ax] * g[b][ax] * det;
                    k[0][a * 8 + b] += v;
                    k[ax + 1][a * 8 + b] += v;
                }
            }
        }
    }
    k
}

/// Shape-function gradients [1/m] at the centre of a general hex (SI coordinates).
pub fn center_grads_general(x: &[[f64; 3]; 8]) -> Vec<[f64; 3]> {
    let (jinv, _) = inv3(&jacobian(x, [0.0; 3]));
    (0..8).map(|i| map_grad(&jinv, dn_dxi(i, [0.0; 3]))).collect()
}

/// Consistent nodal volumes ∫ Nᵢ dV [m³] of an element (h³/8 each for a cube); used for lumped
/// mass, heat capacity and volumetric loads so boundary-fitted elements carry their true volume.
pub fn node_volumes(mesh: &Mesh, e: usize) -> [f64; 8] {
    if !mesh.is_fitted(e) {
        return [mesh.elem_h_m(e).powi(3) / 8.0; 8];
    }
    let x = elem_xyz_m(mesh, e);
    let mut v = [0.0; 8];
    for xi in gauss_points() {
        let det = inv3(&jacobian(&x, xi)).1;
        for (i, vi) in v.iter_mut().enumerate() {
            let s = sgn(i);
            *vi += (1.0 + s[0] * xi[0]) * (1.0 + s[1] * xi[1]) * (1.0 + s[2] * xi[2]) / 8.0 * det;
        }
    }
    v
}

/// Consistent nodal shares of a surface face: vector ∫ Nᵢ n dA [m²] (outward) and scalar
/// ∫ Nᵢ dA [m²]. Exact for any bilinear (possibly boundary-fitted) quad; h²/4 on a voxel face.
pub fn face_shares(mesh: &Mesh, fi: usize) -> ([V3; 4], [f64; 4]) {
    let face = &mesh.faces[fi];
    let h = mesh.elem_h_m(face.elem as usize);
    if mesh.fitted.is_empty() {
        return ([scale(dir_normal(face.dir), h * h / 4.0); 4], [h * h / 4.0; 4]);
    }
    let q = face.nodes.map(|n| mesh.nodes[n as usize].map(|v| v * 1e-3));
    crate::mesh::quad_shares(&q)
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
            thermal(model, mesh, None, prog.slice(0.0, 0.4), &mut res)?;
            let t = res.temp.clone().unwrap();
            res.reactions.clear();
            structural(model, mesh, Some(&t), prog.slice(0.4, 0.6), &mut res)?;
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
        BroadbandEm => crate::broadband::solve(model, mesh, prog, &mut res)?,
        Electrostatic => crate::fields::electrostatic(model, mesh, prog, &mut res)?,
        CurrentFlow => {
            crate::fields::current_flow(model, mesh, prog, &mut res)?;
        }
        ThermalElectric => {
            let q = crate::fields::current_flow(model, mesh, prog.slice(0.0, 0.4), &mut res)?;
            let currents = std::mem::take(&mut res.reactions);
            thermal(model, mesh, Some(&q), prog.slice(0.4, 0.6), &mut res)?;
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

pub struct SurfaceSpec {
    pub nodes: Vec<u32>,
    pub weights: Vec<f64>,
    pub centroid_m: V3,
    pub normal: V3,
    pub r_m: Vec<V3>,
    pub j_inv: [f64; 9],
    pub j_trace: f64,
}

fn mat3_mul(a: &[f64; 9], b: &[f64; 9]) -> [f64; 9] {
    let mut c = [0.0; 9];
    for i in 0..3 {
        for j in 0..3 {
            c[i * 3 + j] = (0..3).map(|k| a[i * 3 + k] * b[k * 3 + j]).sum();
        }
    }
    c
}

fn mat3_t_mul(a: &[f64; 9], b: &[f64; 9]) -> [f64; 9] {
    let mut c = [0.0; 9];
    for i in 0..3 {
        for j in 0..3 {
            c[i * 3 + j] = (0..3).map(|k| a[k * 3 + i] * b[k * 3 + j]).sum();
        }
    }
    c
}

fn mat3_vec(a: &[f64; 9], v: V3) -> V3 {
    [a[0] * v[0] + a[1] * v[1] + a[2] * v[2], a[3] * v[0] + a[4] * v[1] + a[5] * v[2], a[6] * v[0] + a[7] * v[1] + a[8] * v[2]]
}

fn mat3_t_vec(a: &[f64; 9], v: V3) -> V3 {
    [a[0] * v[0] + a[3] * v[1] + a[6] * v[2], a[1] * v[0] + a[4] * v[1] + a[7] * v[2], a[2] * v[0] + a[5] * v[1] + a[8] * v[2]]
}

fn skew(r: V3) -> [f64; 9] {
    [0.0, -r[2], r[1], r[2], 0.0, -r[0], -r[1], r[0], 0.0]
}

fn outer3(a: V3, s: f64) -> [f64; 9] {
    [s * a[0] * a[0], s * a[0] * a[1], s * a[0] * a[2], s * a[1] * a[0], s * a[1] * a[1], s * a[1] * a[2], s * a[2] * a[0], s * a[2] * a[1], s * a[2] * a[2]]
}

pub fn surface_spec(mesh: &Mesh, faces: &[String]) -> Result<SurfaceSpec> {
    let fs = mesh.resolve_faces(faces)?;
    let mut map: std::collections::BTreeMap<u32, (f64, V3)> = std::collections::BTreeMap::new();
    let mut tot_a = 0.0;
    let mut nsum = [0.0; 3];
    for &fi in &fs {
        let (sh, sa) = face_shares(mesh, fi as usize);
        for (k, &nd) in mesh.faces[fi as usize].nodes.iter().enumerate() {
            let e = map.entry(nd).or_insert((0.0, [0.0; 3]));
            e.0 += sa[k];
            e.1 = add(e.1, sh[k]);
            tot_a += sa[k];
            nsum = add(nsum, sh[k]);
        }
    }
    if map.is_empty() || tot_a <= 0.0 {
        bail!("face selector {faces:?} has zero area");
    }
    let mut nodes = Vec::with_capacity(map.len());
    let mut weights = Vec::with_capacity(map.len());
    let mut centroid_m = [0.0; 3];
    for (nd, (a, _)) in map {
        let w = a / tot_a;
        let x = mesh.nodes[nd as usize].map(|v| v * 1e-3);
        centroid_m = add(centroid_m, scale(x, w));
        nodes.push(nd);
        weights.push(w);
    }
    let normal = if norm(nsum) > 1e-24 { scale(nsum, 1.0 / norm(nsum)) } else { [0.0, 0.0, 1.0] };
    Ok(finalize_surface_spec(mesh, nodes, weights, centroid_m, normal))
}

fn finalize_surface_spec(mesh: &Mesh, nodes: Vec<u32>, weights: Vec<f64>, centroid_m: V3, normal: V3) -> SurfaceSpec {
    let mut r_m = Vec::with_capacity(nodes.len());
    let mut j = [0.0; 9];
    for (&nd, &w) in nodes.iter().zip(&weights) {
        let r = sub(mesh.nodes[nd as usize].map(|v| v * 1e-3), centroid_m);
        let r2 = dot(r, r);
        for a in 0..3 {
            for b in 0..3 {
                j[a * 3 + b] += w * (if a == b { r2 } else { 0.0 } - r[a] * r[b]);
            }
        }
        r_m.push(r);
    }
    let (ev, vecs) = sym_eigen(&j, 3);
    let tr = (ev[0] + ev[1] + ev[2]).max(0.0);
    let mut j_inv = [0.0; 9];
    for k in 0..3 {
        if ev[k] > 1e-12 * tr {
            let v = [vecs[k], vecs[3 + k], vecs[6 + k]];
            let inv = 1.0 / ev[k];
            for a in 0..3 {
                for b in 0..3 {
                    j_inv[a * 3 + b] += inv * v[a] * v[b];
                }
            }
        }
    }
    SurfaceSpec { nodes, weights, centroid_m, normal, r_m, j_inv, j_trace: tr }
}

fn downsample_spec(mesh: &Mesh, spec: &SurfaceSpec, cap: usize) -> SurfaceSpec {
    if spec.nodes.len() <= cap {
        return SurfaceSpec {
            nodes: spec.nodes.clone(),
            weights: spec.weights.clone(),
            centroid_m: spec.centroid_m,
            normal: spec.normal,
            r_m: spec.r_m.clone(),
            j_inv: spec.j_inv,
            j_trace: spec.j_trace,
        };
    }
    let step = spec.nodes.len() as f64 / cap as f64;
    let mut nodes = Vec::with_capacity(cap);
    let mut raw_w = Vec::with_capacity(cap);
    let mut wsum = 0.0;
    for i in 0..cap {
        let idx = ((i as f64 + 0.5) * step) as usize;
        let idx = idx.min(spec.nodes.len() - 1);
        nodes.push(spec.nodes[idx]);
        raw_w.push(spec.weights[idx]);
        wsum += spec.weights[idx];
    }
    let weights: Vec<f64> = raw_w.iter().map(|w| w / wsum).collect();
    let mut centroid_m = [0.0; 3];
    for (&nd, &w) in nodes.iter().zip(&weights) {
        centroid_m = add(centroid_m, scale(mesh.nodes[nd as usize].map(|v| v * 1e-3), w));
    }
    finalize_surface_spec(mesh, nodes, weights, centroid_m, spec.normal)
}

/// RBE3 statically equivalent force + moment distribution onto `faces`.
pub fn distribute_force_moment(mesh: &Mesh, faces: &[String], point_mm: Option<V3>, force: V3, moment: V3, f: &mut [f64]) -> Result<()> {
    let spec = surface_spec(mesh, faces)?;
    let pt_m = point_mm.map(|p| p.map(|v| v * 1e-3)).unwrap_or(spec.centroid_m);
    let m_c = add(moment, cross(sub(pt_m, spec.centroid_m), force));
    let omega = mat3_vec(&spec.j_inv, m_c);
    for ((&nd, &w), &r) in spec.nodes.iter().zip(&spec.weights).zip(&spec.r_m) {
        let fi = scale(add(force, cross(omega, r)), w);
        for c in 0..3 {
            f[nd as usize * 3 + c] += fi[c];
        }
    }
    Ok(())
}

#[derive(Clone)]
struct SpringCoupling {
    links: Vec<(u32, u32, f64)>, // (a, b, w) when a==b -> ground spring
    k_mat: [f64; 9],
}

#[derive(Clone)]
enum RemoteCoupling {
    Rigid {
        pu: u32,
        pr: u32,
        nodes: Vec<(u32, V3, f64)>, // (node, r_i, k_i)
    },
    Dist {
        pu: u32,
        pr: u32,
        nodes: Vec<(u32, [f64; 9], [f64; 9])>, // (node, A_i = w_i I + s_pc B_i, B_i = -J_inv w_i skew(r_i))
        k_u: f64,
        k_theta: f64,
    },
    MassOffset {
        pm: u32,
        mass_kg: f64,
        terms: Vec<(u32, [f64; 9])>, // (node, C_i)
        k_m: f64,
    },
}

#[derive(Clone)]
struct CentroidMass {
    nodes: Vec<(u32, f64)>,
}

/// Additional structural couplings (contacts, bolt pretensions, springs, remote points, point masses).
#[derive(Clone)]
pub struct StructuralCouplings {
    pub total_nodes: usize,
    pub avg_eh: f64,
    pub extra_edges: Vec<(u32, u32)>,
    springs: Vec<SpringCoupling>,
    remotes: Vec<RemoteCoupling>,
    centroid_masses: Vec<CentroidMass>,
}

impl StructuralCouplings {
    pub fn build(model: &Model, mesh: &Mesh) -> Result<(Self, Vec<Option<f64>>, Vec<(u32, String, Vec<u32>)>, Vec<u32>)> {
        let nn = mesh.nodes.len();
        let h_m = mesh.h * 1e-3;
        let solid_es: Vec<f64> = mesh
            .elem_region
            .iter()
            .zip(&mesh.elem_mat)
            .filter(|(r, _)| **r == crate::mesh::REGION_SOLID)
            .map(|(_, &m)| mesh.materials[m as usize].youngs_modulus_gpa * 1e9)
            .filter(|e| *e > 0.0)
            .collect();
        let avg_e = if solid_es.is_empty() { 1e9 } else { solid_es.iter().sum::<f64>() / solid_es.len() as f64 };
        let avg_eh = avg_e * h_m;

        let mut extra_edges: Vec<(u32, u32)> = Vec::new();
        for c in &mesh.contacts {
            let nb = (c.n_b as usize).min(4);
            for i in 0..nb {
                extra_edges.push((c.node_a, c.nodes_b[i]));
                for j in (i + 1)..nb {
                    extra_edges.push((c.nodes_b[i], c.nodes_b[j]));
                }
            }
        }
        for ps in &mesh.pretensions {
            for pr in &ps.pairs {
                extra_edges.push((pr.minus, pr.plus));
            }
        }

        let mut springs = Vec::new();
        let mut remotes = Vec::new();
        let mut centroid_masses = Vec::new();
        let mut vnodes = 0usize;
        let mut remote_bcs: Vec<(u32, String, Vec<(usize, f64)>, Vec<u32>)> = Vec::new();
        let mut ground_springs: Vec<(u32, String, Vec<u32>)> = Vec::new();

        for it in &model.loads {
            match &it.load {
                Load::Spring { faces, to_faces, stiffness_n_per_mm, anchor, preload_n } => {
                    if !stiffness_n_per_mm.is_finite() || *stiffness_n_per_mm <= 0.0 {
                        bail!("spring #{}: stiffness must be finite and > 0", it.id);
                    }
                    if !preload_n.is_finite() {
                        bail!("spring #{}: preload must be finite", it.id);
                    }
                    let sa = surface_spec(mesh, faces)?;
                    let k_si = stiffness_n_per_mm * 1e3;
                    match to_faces {
                        None => {
                            let axis = if let Some(p) = anchor {
                                let d = sub(p.map(|v| v * 1e-3), sa.centroid_m);
                                if norm(d) > 1e-9 * h_m { scale(d, 1.0 / norm(d)) } else { sa.normal }
                            } else {
                                sa.normal
                            };
                            ground_springs.push((it.id, it.load.label().to_string(), sa.nodes.clone()));
                            springs.push(SpringCoupling {
                                links: sa.nodes.iter().zip(&sa.weights).map(|(&n, &w)| (n, n, w)).collect(),
                                k_mat: outer3(axis, k_si),
                            });
                        }
                        Some(tb) => {
                            let sb = surface_spec(mesh, tb)?;
                            let d = sub(sb.centroid_m, sa.centroid_m);
                            let axis = if norm(d) > 1e-9 * h_m { scale(d, 1.0 / norm(d)) } else { sa.normal };
                            let nearest = |nd: u32, targets: &[u32]| -> u32 {
                                let p = mesh.nodes[nd as usize];
                                *targets.iter().min_by(|&&x, &&y| {
                                    let dx = sub(mesh.nodes[x as usize], p);
                                    let dy = sub(mesh.nodes[y as usize], p);
                                    dot(dx, dx).total_cmp(&dot(dy, dy))
                                }).unwrap()
                            };
                            let mut v = Vec::with_capacity(sa.nodes.len() + sb.nodes.len());
                            for (&na, &wa) in sa.nodes.iter().zip(&sa.weights) {
                                let nb = nearest(na, &sb.nodes);
                                extra_edges.push((na, nb));
                                v.push((na, nb, 0.5 * wa));
                            }
                            for (&nb, &wb) in sb.nodes.iter().zip(&sb.weights) {
                                let na = nearest(nb, &sa.nodes);
                                extra_edges.push((na, nb));
                                v.push((na, nb, 0.5 * wb));
                            }
                            springs.push(SpringCoupling { links: v, k_mat: outer3(axis, k_si) });
                        }
                    }
                }
                Load::RemoteDisplacement { faces, point, x, y, z, rx, ry, rz, rigid } => {
                    if !point.iter().all(|v| v.is_finite())
                        || [x, y, z, rx, ry, rz].iter().any(|c| c.is_some_and(|v| !v.is_finite()))
                    {
                        bail!("remote displacement #{}: coordinates and displacements must be finite", it.id);
                    }
                    let spec_full = surface_spec(mesh, faces)?;
                    let pu = (nn + vnodes) as u32;
                    let pr = (nn + vnodes + 1) as u32;
                    vnodes += 2;
                    extra_edges.push((pu, pr));
                    let point_m = point.map(|v| v * 1e-3);
                    if *rigid {
                        let k_tot = 200.0 * avg_eh;
                        let mut nodes = Vec::with_capacity(spec_full.nodes.len());
                        for (&nd, &w) in spec_full.nodes.iter().zip(&spec_full.weights) {
                            extra_edges.push((nd, pu));
                            extra_edges.push((nd, pr));
                            nodes.push((nd, sub(mesh.nodes[nd as usize].map(|v| v * 1e-3), point_m), k_tot * w));
                        }
                        remotes.push(RemoteCoupling::Rigid { pu, pr, nodes });
                    } else {
                        let sp = downsample_spec(mesh, &spec_full, 128);
                        // For moment transfer to the centroid `(c - point) × F`, note `(u_0 + θ_0 × (p - c))` vs pilot at `point`:
                        // rewrite r_p = x_i - point_m -> wait: J_p about `centroid` gives θ_0 directly, and
                        // u(point) = u_c + θ_0 × (point - centroid) = sum w_i u_i - skew(p - c) * sum J_inv w_i skew(r_i) u_i!
                        let s_pc = skew(sub(point_m, sp.centroid_m));
                        let k_u = 200.0 * avg_eh;
                        let k_theta = 200.0 * avg_eh * sp.j_trace.max(h_m * h_m);
                        let m = sp.nodes.len();
                        let mut entries = Vec::with_capacity(m);
                        for i in 0..m {
                            let ni = sp.nodes[i];
                            let wi = sp.weights[i];
                            extra_edges.push((ni, pu));
                            extra_edges.push((ni, pr));
                            for j in (i + 1)..m {
                                extra_edges.push((ni, sp.nodes[j]));
                            }
                            let bi = mat3_mul(&sp.j_inv, &skew(sp.r_m[i])).map(|v| -wi * v);
                            let sc_bi = mat3_mul(&s_pc, &bi);
                            let mut ai = [0.0; 9];
                            for idx in 0..9 {
                                let eye_v = if idx % 4 == 0 { wi } else { 0.0 };
                                ai[idx] = eye_v + sc_bi[idx];
                            }
                            entries.push((ni, ai, bi));
                        }
                        remotes.push(RemoteCoupling::Dist { pu, pr, nodes: entries, k_u, k_theta });
                    }
                    let mut bcs = Vec::new();
                    for (c, opt) in [x, y, z].iter().enumerate() {
                        if let Some(v) = opt {
                            bcs.push((pu as usize * 3 + c, *v * 1e-3));
                        }
                    }
                    for (c, opt) in [rx, ry, rz].iter().enumerate() {
                        if let Some(v) = opt {
                            bcs.push((pr as usize * 3 + c, *v));
                        }
                    }
                    if !bcs.is_empty() {
                        remote_bcs.push((it.id, it.load.label().to_string(), bcs, spec_full.nodes.clone()));
                    }
                }
                Load::PointMass { faces, mass_kg, point } => {
                    if !mass_kg.is_finite() || *mass_kg <= 0.0 {
                        bail!("point mass #{}: mass_kg must be finite and > 0", it.id);
                    }
                    let spec_full = surface_spec(mesh, faces)?;
                    let r_off = point.map(|p| sub(p.map(|v| v * 1e-3), spec_full.centroid_m)).unwrap_or([0.0; 3]);
                    if norm(r_off) <= 1e-4 * h_m.max(1e-3) {
                        centroid_masses.push(CentroidMass {
                            nodes: spec_full.nodes.iter().zip(&spec_full.weights).map(|(&n, &w)| (n, w * *mass_kg)).collect(),
                        });
                    } else {
                        let sp = downsample_spec(mesh, &spec_full, 128);
                        let pm = (nn + vnodes) as u32;
                        vnodes += 1;
                        let k_m = 200.0 * avg_eh;
                        let sr = skew(r_off);
                        let mut terms = Vec::with_capacity(sp.nodes.len());
                        for i in 0..sp.nodes.len() {
                            let ni = sp.nodes[i];
                            let wi = sp.weights[i];
                            extra_edges.push((ni, pm));
                            for j in (i + 1)..sp.nodes.len() {
                                extra_edges.push((ni, sp.nodes[j]));
                            }
                            let prod = mat3_mul(&sr, &mat3_mul(&sp.j_inv, &skew(sp.r_m[i])));
                            let mut ci = [0.0; 9];
                            for k in 0..9 {
                                let eye = if k % 4 == 0 { 1.0 } else { 0.0 };
                                ci[k] = -wi * (eye - prod[k]);
                            }
                            terms.push((ni, ci));
                        }
                        remotes.push(RemoteCoupling::MassOffset { pm, mass_kg: *mass_kg, terms, k_m });
                    }
                }
                _ => {}
            }
        }

        let total_nodes = nn + vnodes;
        let (mut fixed, mut support_nodes, _contact, mut owner) = structural_bcs_len(model, mesh, total_nodes)?;
        for (id, label, bcs, nodes) in remote_bcs {
            for (dof, val) in bcs {
                fixed[dof] = Some(val);
                owner.0[dof] = id;
            }
            support_nodes.push((id, label, nodes));
        }
        support_nodes.extend(ground_springs);

        let plan = StructuralCouplings { total_nodes, avg_eh, extra_edges, springs, remotes, centroid_masses };
        Ok((plan, fixed, support_nodes, owner.0))
    }

    pub fn has_nonlocal(&self, mesh: &Mesh) -> bool {
        !mesh.contacts.is_empty()
            || !mesh.pretensions.is_empty()
            || !self.springs.is_empty()
            || !self.remotes.is_empty()
            || !self.centroid_masses.is_empty()
    }

    pub fn apply_elastic_couplings(&self, k: &mut NodeSparse) {
        let eye = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        for sp in &self.springs {
            for &(a, b, w) in &sp.links {
                let km = sp.k_mat.map(|v| v * w);
                k.add_symmetric_pair(a as usize, b as usize, &km);
            }
        }
        for r in &self.remotes {
            match r {
                RemoteCoupling::Rigid { pu, pr, nodes, .. } => {
                    let (pu, pr) = (*pu as usize, *pr as usize);
                    for &(ni, r_i, k_i) in nodes {
                        let ni = ni as usize;
                        let sk = skew(r_i);
                        let ks = eye.map(|v| v * k_i);
                        let ksk = sk.map(|v| v * k_i);
                        let neg_ksk = sk.map(|v| -v * k_i);
                        let kskt_sk = mat3_t_mul(&sk, &sk).map(|v| v * k_i);
                        k.add_symmetric_pair(ni, pu, &ks);
                        k.add_block(ni, pr, &ksk);
                        k.add_block(pr, ni, &neg_ksk);
                        k.add_block(pu, pr, &neg_ksk);
                        k.add_block(pr, pu, &ksk);
                        k.add_block(pr, pr, &kskt_sk);
                    }
                }
                RemoteCoupling::Dist { pu, pr, nodes, k_u, k_theta } => {
                    let (pu, pr) = (*pu as usize, *pr as usize);
                    // Translation: 0.5 * k_u * ||u_0 - sum A_i u_i||^2
                    k.add_block(pu, pu, &eye.map(|v| v * *k_u));
                    for &(ni, ai, _) in nodes {
                        let neg_ai = ai.map(|v| -v * *k_u);
                        let neg_ait = mat3_t_mul(&ai, &eye).map(|v| -v * *k_u);
                        k.add_block(pu, ni as usize, &neg_ai);
                        k.add_block(ni as usize, pu, &neg_ait);
                        for &(nj, aj, _) in nodes {
                            let blk = mat3_t_mul(&ai, &aj).map(|v| v * *k_u);
                            k.add_block(ni as usize, nj as usize, &blk);
                        }
                    }
                    // Rotation: 0.5 * k_theta * ||theta_0 + sum B_i u_i||^2
                    k.add_block(pr, pr, &eye.map(|v| v * *k_theta));
                    for &(ni, _, bi) in nodes {
                        k.add_block(pr, ni as usize, &bi.map(|v| v * *k_theta));
                        k.add_block(ni as usize, pr, &mat3_t_mul(&bi, &eye).map(|v| v * *k_theta));
                        for &(nj, _, bj) in nodes {
                            let blk = mat3_t_mul(&bi, &bj).map(|v| v * *k_theta);
                            k.add_block(ni as usize, nj as usize, &blk);
                        }
                    }
                }
                RemoteCoupling::MassOffset { pm, terms, k_m, .. } => {
                    let pm = *pm as usize;
                    k.add_block(pm, pm, &eye.map(|v| v * *k_m));
                    for &(ni, ci) in terms {
                        k.add_block(pm, ni as usize, &ci.map(|v| v * *k_m));
                        k.add_block(ni as usize, pm, &mat3_t_mul(&ci, &eye).map(|v| v * *k_m));
                        for &(nj, cj) in terms {
                            k.add_block(ni as usize, nj as usize, &mat3_t_mul(&ci, &cj).map(|v| v * *k_m));
                        }
                    }
                }
            }
        }
    }

    pub fn apply_linear_interface_ties(&self, mesh: &Mesh, k: &mut NodeSparse) {
        let h_m = mesh.h * 1e-3;
        let k_tie = 100.0 * self.avg_eh;
        let eye = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        for ps in &mesh.pretensions {
            for pr in &ps.pairs {
                let km = eye.map(|v| v * k_tie * (pr.area_m2 / (h_m * h_m)).max(0.25));
                k.add_symmetric_pair(pr.minus as usize, pr.plus as usize, &km);
            }
        }
        let tol = 1e-4 * h_m + 1e-9;
        for c in &mesh.contacts {
            if c.gap0_m > tol {
                continue;
            }
            let nb = (c.n_b as usize).min(4);
            let kn = 50.0 * self.avg_eh * (c.area_m2 / (h_m * h_m)).max(0.25);
            let m = match c.kind {
                ContactKind::Bonded | ContactKind::Frictional => eye.map(|v| v * kn),
                ContactKind::NoSeparation | ContactKind::Frictionless => outer3(c.normal, kn),
            };
            add_contact_matrix(k, c.node_a, &c.nodes_b[..nb], &c.weights_b[..nb], &m);
        }
    }

    pub fn apply_mass_diag(&self, mdiag: &mut [f64]) {
        for cm in &self.centroid_masses {
            for &(nd, dm) in &cm.nodes {
                for c in 0..3 {
                    mdiag[nd as usize * 3 + c] += dm;
                }
            }
        }
        for r in &self.remotes {
            if let RemoteCoupling::MassOffset { pm, mass_kg, .. } = r {
                for c in 0..3 {
                    mdiag[*pm as usize * 3 + c] += *mass_kg;
                }
            }
        }
    }

    pub fn add_internal_forces(&self, u: &[f64], fint: &mut [f64]) {
        for sp in &self.springs {
            for &(a, b, w) in &sp.links {
                let du = if a == b {
                    [u[a as usize * 3], u[a as usize * 3 + 1], u[a as usize * 3 + 2]]
                } else {
                    [u[a as usize * 3] - u[b as usize * 3], u[a as usize * 3 + 1] - u[b as usize * 3 + 1], u[a as usize * 3 + 2] - u[b as usize * 3 + 2]]
                };
                let f = scale(mat3_vec(&sp.k_mat, du), w);
                for c in 0..3 {
                    fint[a as usize * 3 + c] += f[c];
                    if a != b {
                        fint[b as usize * 3 + c] -= f[c];
                    }
                }
            }
        }
        for r in &self.remotes {
            match r {
                RemoteCoupling::Rigid { pu, pr, nodes, .. } => {
                    let u0 = [u[*pu as usize * 3], u[*pu as usize * 3 + 1], u[*pu as usize * 3 + 2]];
                    let t0 = [u[*pr as usize * 3], u[*pr as usize * 3 + 1], u[*pr as usize * 3 + 2]];
                    for &(ni, r_i, k_i) in nodes {
                        let ui = [u[ni as usize * 3], u[ni as usize * 3 + 1], u[ni as usize * 3 + 2]];
                        let res_i = add(sub(ui, u0), cross(r_i, t0));
                        let f_i = scale(res_i, k_i);
                        let m_pr = mat3_t_vec(&skew(r_i), f_i);
                        for c in 0..3 {
                            fint[ni as usize * 3 + c] += f_i[c];
                            fint[*pu as usize * 3 + c] -= f_i[c];
                            fint[*pr as usize * 3 + c] += m_pr[c];
                        }
                    }
                }
                RemoteCoupling::Dist { pu, pr, nodes, k_u, k_theta } => {
                    let mut cu = [u[*pu as usize * 3], u[*pu as usize * 3 + 1], u[*pu as usize * 3 + 2]];
                    let mut cr = [u[*pr as usize * 3], u[*pr as usize * 3 + 1], u[*pr as usize * 3 + 2]];
                    for &(ni, ai, bi) in nodes {
                        let ui = [u[ni as usize * 3], u[ni as usize * 3 + 1], u[ni as usize * 3 + 2]];
                        cu = sub(cu, mat3_vec(&ai, ui));
                        cr = add(cr, mat3_vec(&bi, ui));
                    }
                    let fu = scale(cu, *k_u);
                    let ft = scale(cr, *k_theta);
                    for c in 0..3 {
                        fint[*pu as usize * 3 + c] += fu[c];
                        fint[*pr as usize * 3 + c] += ft[c];
                    }
                    for &(ni, ai, bi) in nodes {
                        let fn_i = sub(mat3_t_vec(&bi, ft), mat3_t_vec(&ai, fu));
                        for c in 0..3 {
                            fint[ni as usize * 3 + c] += fn_i[c];
                        }
                    }
                }
                RemoteCoupling::MassOffset { pm, terms, k_m, .. } => {
                    let mut cm = [u[*pm as usize * 3], u[*pm as usize * 3 + 1], u[*pm as usize * 3 + 2]];
                    for &(ni, ci) in terms {
                        let ui = [u[ni as usize * 3], u[ni as usize * 3 + 1], u[ni as usize * 3 + 2]];
                        cm = add(cm, mat3_vec(&ci, ui));
                    }
                    let fm = scale(cm, *k_m);
                    for c in 0..3 {
                        fint[*pm as usize * 3 + c] += fm[c];
                    }
                    for &(ni, ci) in terms {
                        let fi = mat3_t_vec(&ci, fm);
                        for c in 0..3 {
                            fint[ni as usize * 3 + c] += fi[c];
                        }
                    }
                }
            }
        }
    }
}

fn add_contact_matrix(k: &mut NodeSparse, node_a: u32, nodes_b: &[u32], weights_b: &[f64], m: &[f64; 9]) {
    let mut terms = [(node_a as usize, 1.0f64); 5];
    let n = 1 + nodes_b.len();
    for (idx, (&b, &w)) in nodes_b.iter().zip(weights_b).enumerate() {
        terms[1 + idx] = (b as usize, -w);
    }
    for i in 0..n {
        for j in 0..n {
            let s = terms[i].1 * terms[j].1;
            if s.abs() > 1e-24 {
                k.add_block(terms[i].0, terms[j].0, &m.map(|v| v * s));
            }
        }
    }
}

/// Add strong penalty ties for zero-gap split nodes in scalar (thermal/electrostatic/current) systems.
pub fn add_scalar_ties(mesh: &Mesh, k: &mut NodeSparse, scale_val: f64) {
    let blk = [scale_val];
    for (a, b) in mesh.interface_ties() {
        k.add_symmetric_pair(a as usize, b as usize, &blk);
    }
}

/// Add strong isotropic 3×3 penalty ties for zero-gap split nodes in 3-component vector systems.
pub fn add_vec3_ties(mesh: &Mesh, k: &mut NodeSparse, scale_val: f64) {
    let blk = [scale_val, 0.0, 0.0, 0.0, scale_val, 0.0, 0.0, 0.0, scale_val];
    for (a, b) in mesh.interface_ties() {
        k.add_symmetric_pair(a as usize, b as usize, &blk);
    }
}

pub struct Assembled {
    pub k: NodeSparse,
    pub fixed: Vec<Option<f64>>,
    pub support_nodes: Vec<(u32, String, Vec<u32>)>,
    /// Compression-only (contact) dofs with the outward normal sign of the face.
    pub contact: Vec<(usize, f64)>,
    /// Load id owning each constrained dof (0 = free) — used for per-support reactions.
    pub owner: Vec<u32>,
    pub couplings: StructuralCouplings,
}

pub fn assemble_structural(model: &Model, mesh: &Mesh, prog: Prog) -> Result<(Assembled, Vec<ElemUnit>)> {
    prog.report(0.0, "assembling stiffness")?;
    let units: Vec<ElemUnit> = mesh.materials.iter().map(|m| hex_elastic_unit(m.poisson_ratio)).collect();
    let (couplings, mut fixed, support_nodes, owner) = StructuralCouplings::build(model, mesh)?;
    mesh.release_hanging(&mut fixed, 3);
    let (_, _, contact, _) = structural_bcs(model, mesh)?;
    let mut k = NodeSparse::new_with_edges(couplings.total_nodes, &mesh.elems, &couplings.extra_edges, 3);
    let general: Vec<Option<ElemGeneral>> = (0..mesh.elems.len())
        .into_par_iter()
        .map(|e| -> Result<_> {
            if e % 1024 == 0 {
                prog.check()?;
            }
            Ok(mesh.is_fitted(e).then(|| hex_elastic_general(&elem_xyz_m(mesh, e), mesh.materials[mesh.elem_mat[e] as usize].poisson_ratio)))
        })
        .collect::<Result<Vec<_>>>()?;
    for (e, en) in mesh.elems.iter().enumerate() {
        let m = &mesh.materials[mesh.elem_mat[e] as usize];
        match &general[e] {
            Some(g) => k.add_element(en, &g.k, m.youngs_modulus_gpa * 1e9),
            None => k.add_element(en, &units[mesh.elem_mat[e] as usize].k, m.youngs_modulus_gpa * 1e9 * mesh.elem_h_m(e)),
        }
    }
    couplings.apply_elastic_couplings(&mut k);
    k.hang = mesh.hang_ctx();
    Ok((Assembled { k, fixed, support_nodes, contact, owner, couplings }, units))
}

/// Supports of a structural model: prescribed dofs, support node sets, compression-only dofs and
/// the owning load of every constrained dof.
#[allow(clippy::type_complexity)]
pub fn structural_bcs(model: &Model, mesh: &Mesh) -> Result<(Vec<Option<f64>>, Vec<(u32, String, Vec<u32>)>, Vec<(usize, f64)>, Vec<u32>)> {
    let (fixed, support_nodes, contact, (owner, _)) = structural_bcs_len(model, mesh, mesh.nodes.len())?;
    Ok((fixed, support_nodes, contact, owner))
}

#[allow(clippy::type_complexity)]
fn structural_bcs_len(model: &Model, mesh: &Mesh, total_nodes: usize) -> Result<(Vec<Option<f64>>, Vec<(u32, String, Vec<u32>)>, Vec<(usize, f64)>, (Vec<u32>, usize))> {
    let mut fixed = vec![None; total_nodes * 3];
    let mut support_nodes = Vec::new();
    let mut contact = Vec::new();
    let mut owner = vec![0u32; total_nodes * 3];
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
                    let sign = if face.dir.is_multiple_of(2) { -1.0 } else { 1.0 };
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
    Ok((fixed, support_nodes, contact, (owner, total_nodes)))
}

/// Smallest structural model (elements) solved with matrix-free multigrid CG instead of the
/// assembled direct / Jacobi-PCG solvers (`SIMCRAFT_MG=0/1` overrides).
pub const MG_MIN_ELEMENTS: usize = 60_000;

fn use_multigrid(model: &Model, mesh: &Mesh) -> bool {
    if mesh.octree || !mesh.contacts.is_empty()
        || !mesh.pretensions.is_empty()
        || model.loads.iter().any(|l| {
            matches!(
                l.load,
                Load::Spring { .. } | Load::RemoteDisplacement { .. } | Load::RemoteForce { .. } | Load::PointMass { .. }
            )
        })
    {
        return false;
    }
    match std::env::var("SIMCRAFT_MG").ok().as_deref() {
        Some("1") => return true,
        Some("0") => return false,
        _ => {}
    }
    mesh.elems.len() >= MG_MIN_ELEMENTS && model.analysis.compute != crate::gpu::ComputeMode::Gpu
}

/// Static solve with matrix-free multigrid CG (never assembles K).
fn structural_mg(model: &Model, mesh: &Mesh, temps: Option<&[f64]>, prog: Prog, res: &mut Results) -> Result<()> {
    let nn = mesh.nodes.len();
    let (mut fixed, support_nodes, contact, owner) = structural_bcs(model, mesh)?;
    if support_nodes.is_empty() {
        bail!("no supports defined: add a Fixed Support (or Displacement) so the part cannot fly away");
    }
    let units: Vec<ElemUnit> = mesh.materials.iter().map(|m| hex_elastic_unit(m.poisson_ratio)).collect();
    let f_ext = external_forces(model, mesh, temps)?;
    let fscale = f_ext.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-30);
    let mut u = vec![0.0; nn * 3];
    let mut pass = 0;
    loop {
        prog.report(0.03, "building multigrid")?;
        let t0 = Instant::now();
        let mg = crate::elastic_mg::ElasticMg::new(mesh, &units, fixed.iter().map(|g| g.is_some()).collect())?;
        let ug: Vec<f64> = fixed.iter().map(|g| g.unwrap_or(0.0)).collect();
        let t1 = Instant::now();
        let its = mg.solve(&f_ext, &ug, &mut u, 1e-8, 500, &prog.scalar(0.08, 0.82, "solving (multigrid CG)"))?;
        if std::env::var_os("SIMCRAFT_PROFILE").is_some() {
            eprintln!("[mg] setup {:.2} s, {its} CG iterations {:.2} s", (t1 - t0).as_secs_f64(), t1.elapsed().as_secs_f64());
        }
        crate::sparse::SOLVE_NANOS.fetch_add(t0.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        res.info.solver = format!("matrix-free multigrid CG ({} levels, CPU)", mg.levels());
        res.info.iterations += its;
        if contact.is_empty() {
            break;
        }
        let mut ku = vec![0.0; nn * 3];
        mg.fine.apply_k(&u, &mut ku);
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
        prog.report(0.5, &format!("contact iteration {pass}: {changed} change(s)"))?;
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
    prog.report(0.92, "computing stresses")?;
    structural_post(model, mesh, &u, temps, &units, &fixed, &f_ext, &support_nodes, &owner, res);
    prog.report(1.0, "done")?;
    Ok(())
}

/// External nodal forces [N]: forces, pressure, gravity, remote forces, spring preloads, point masses and thermal expansion.
pub fn external_forces(model: &Model, mesh: &Mesh, temps: Option<&[f64]>) -> Result<Vec<f64>> {
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
    let mut gravity_acc: Option<V3> = None;
    for it in &model.loads {
        match &it.load {
            Load::Force { faces, vector } => {
                let fs = mesh.resolve_faces(faces)?;
                if mesh.octree {
                    // mixed face sizes: spread as a uniform traction (area-weighted nodal shares)
                    let sh: Vec<[f64; 4]> = fs.iter().map(|&fi| face_shares(mesh, fi as usize).1).collect();
                    let area: f64 = sh.iter().flatten().sum::<f64>().max(1e-300);
                    for (&fi, w) in fs.iter().zip(&sh) {
                        for (k, &n) in mesh.faces[fi as usize].nodes.iter().enumerate() {
                            for c in 0..3 {
                                f[n as usize * 3 + c] += vector[c] * w[k] / area;
                            }
                        }
                    }
                    continue;
                }
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
                    // consistent nodal loads ∫ Nᵢ p n dA
                    let shares = face_shares(mesh, fi as usize).0;
                    let p = value_mpa * 1e6;
                    for (k, &n) in face.nodes.iter().enumerate() {
                        for c in 0..3 {
                            f[n as usize * 3 + c] -= shares[k][c] * p;
                        }
                    }
                }
            }
            Load::Gravity { acceleration } => {
                gravity_acc = Some(add(gravity_acc.unwrap_or([0.0; 3]), *acceleration));
                for (e, en) in mesh.elems.iter().enumerate() {
                    let rho = mesh.materials[mesh.elem_mat[e] as usize].density;
                    let nv = node_volumes(mesh, e);
                    for (a, &n) in en.iter().enumerate() {
                        for c in 0..3 {
                            f[n as usize * 3 + c] += acceleration[c] * rho * nv[a];
                        }
                    }
                }
            }
            Load::RemoteForce { faces, point, force, moment } => {
                if !point.iter().chain(force).chain(moment).all(|v| v.is_finite()) {
                    bail!("remote force #{}: vectors must be finite", it.id);
                }
                distribute_force_moment(mesh, faces, Some(*point), *force, *moment, &mut f)?;
            }
            Load::Spring { faces, to_faces, preload_n, anchor, .. } if *preload_n != 0.0 => {
                if !preload_n.is_finite() {
                    bail!("spring #{}: preload must be finite", it.id);
                }
                let sa = surface_spec(mesh, faces)?;
                let h_m = mesh.h * 1e-3;
                match to_faces {
                    None => {
                        let axis = if let Some(p) = anchor {
                            let d = sub(p.map(|v| v * 1e-3), sa.centroid_m);
                            if norm(d) > 1e-9 * h_m { scale(d, 1.0 / norm(d)) } else { sa.normal }
                        } else {
                            sa.normal
                        };
                        for (&nd, &w) in sa.nodes.iter().zip(&sa.weights) {
                            for c in 0..3 {
                                f[nd as usize * 3 + c] += w * preload_n * axis[c];
                            }
                        }
                    }
                    Some(tb) => {
                        let sb = surface_spec(mesh, tb)?;
                        let d = sub(sb.centroid_m, sa.centroid_m);
                        let axis = if norm(d) > 1e-9 * h_m { scale(d, 1.0 / norm(d)) } else { sa.normal };
                        for (&nd, &w) in sa.nodes.iter().zip(&sa.weights) {
                            for c in 0..3 {
                                f[nd as usize * 3 + c] += w * preload_n * axis[c];
                            }
                        }
                        for (&nd, &w) in sb.nodes.iter().zip(&sb.weights) {
                            for c in 0..3 {
                                f[nd as usize * 3 + c] -= w * preload_n * axis[c];
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }
    if let Some(acc) = gravity_acc {
        for it in &model.loads {
            if let Load::PointMass { faces, mass_kg, point } = &it.load {
                if !mass_kg.is_finite() || *mass_kg <= 0.0 {
                    bail!("point mass #{}: mass_kg must be finite and > 0", it.id);
                }
                distribute_force_moment(mesh, faces, *point, scale(acc, *mass_kg), [0.0; 3], &mut f)?;
            }
        }
    }
    if temps.is_some() {
        for (e, en) in mesh.elems.iter().enumerate() {
            let mi = mesh.elem_mat[e] as usize;
            let m = &mesh.materials[mi];
            let eth = m.youngs_modulus_gpa * 1e9 * m.thermal_expansion * elem_dt(e);
            let (s, ft) = if mesh.is_fitted(e) { (eth, hex_elastic_general(&elem_xyz_m(mesh, e), m.poisson_ratio).fth) } else { (eth * mesh.elem_h_m(e).powi(2), tunits[mi].clone()) };
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
    if use_multigrid(model, mesh) {
        return structural_mg(model, mesh, temps, prog, res);
    }
    let nn = mesh.nodes.len();
    let (asm, units) = assemble_structural(model, mesh, prog)?;
    let Assembled { k: k0, fixed: full_fixed, support_nodes, contact: comp_contact, owner, couplings } = asm;
    let n = k0.n();
    if support_nodes.is_empty() && mesh.contacts.is_empty() {
        bail!("no supports defined: add a Fixed Support (or Displacement) so the part cannot fly away");
    }
    let mut f_ext = external_forces(model, mesh, temps)?;
    f_ext.resize(n, 0.0);
    mesh.condense_rhs(&mut f_ext, 3);

    // Fast path: no assembly contact pairs and no bolt pretensions -> existing active-set compression-only / linear solve
    if mesh.contacts.is_empty() && mesh.pretensions.is_empty() {
        let mut fixed = full_fixed;
        let fscale = f_ext.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-30);
        let mut u: Vec<f64>;
        let mut pass = 0;
        loop {
            let mut k = k0.clone();
            let mut f = f_ext.clone();
            k.apply_dirichlet(&fixed, &mut f);
            u = fixed.iter().map(|g| g.unwrap_or(0.0)).collect();
            prog.report(0.05, "factorising")?;
            let solver = Solver::new(&k).map_err(|e| if comp_contact.is_empty() { e } else { anyhow::anyhow!("{e} (the part may be lifting off its compression-only supports: add a support that holds it)") })?;
            let its = solver.solve(&k, &f, &mut u, 1e-8, &prog.scalar(0.05, 0.85, "solving (PCG)"))?;
            mesh.expand_solution(&mut u, 3);
            res.info.solver = solver.name().to_string();
            res.info.iterations += its;
            if comp_contact.is_empty() {
                break;
            }
            let mut ku = vec![0.0; n];
            k0.matvec(&u, &mut ku);
            mesh.condense_rhs(&mut ku, 3);
            let mut changed = 0;
            for &(dof, sgn) in &comp_contact {
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
            prog.report(0.5, &format!("contact iteration {pass}: {changed} change(s)"))?;
            if changed == 0 || pass >= 60 {
                if pass >= 60 {
                    res.warnings.push("contact status did not fully settle after 60 iterations".into());
                }
                break;
            }
        }
        if !comp_contact.is_empty() {
            let open = comp_contact.iter().filter(|(d, _)| fixed[*d].is_none()).count();
            res.quantity("Contact nodes in contact", (comp_contact.len() - open) as f64, "");
            res.quantity("Contact nodes lifted off", open as f64, "");
        }
        res.info.dofs = n;
        prog.report(0.92, "computing stresses")?;
        structural_post_ex(model, mesh, &u, temps, &units, &fixed, &f_ext, &support_nodes, &owner, Some(&couplings), res);
        prog.report(1.0, "done")?;
        return Ok(());
    }

    // Augmented-Lagrange solver for contact interfaces and two-step bolt pretension
    let h_m = mesh.h * 1e-3;
    let mut node_e = vec![0.0f64; nn];
    let mut node_ecnt = vec![0u32; nn];
    for (e, en) in mesh.elems.iter().enumerate() {
        let ep = mesh.materials[mesh.elem_mat[e] as usize].youngs_modulus_gpa * 1e9;
        for &nd in en {
            node_e[nd as usize] += ep;
            node_ecnt[nd as usize] += 1;
        }
    }
    for (v, &c) in node_e.iter_mut().zip(&node_ecnt) {
        *v = if c > 0 { *v / c as f64 } else { 1e9 };
    }

    let nc = mesh.contacts.len();
    let mut kn = Vec::with_capacity(nc);
    let mut kt = Vec::with_capacity(nc);
    let mut active_n = Vec::with_capacity(nc);
    let mut stick = Vec::with_capacity(nc);
    for c in &mesh.contacts {
        let ea = node_e[c.node_a as usize];
        let eb = node_e[c.nodes_b[0] as usize];
        let ec = (2.0 * ea * eb / (ea + eb).max(1.0)).max(1.0);
        kn.push(15.0 * ec * (c.area_m2 / h_m));
        kt.push(5.0 * ec * (c.area_m2 / h_m));
        let act = matches!(c.kind, ContactKind::NoSeparation) || c.gap0_m <= 1e-4 * h_m + 1e-9;
        active_n.push(act);
        stick.push(act && c.kind == ContactKind::Frictional && c.friction > 0.0);
    }
    struct PretState {
        force_n: Option<f64>,
        delta: f64,
        kc: Vec<f64>,
        l_cut: Vec<[f64; 3]>,
        step1_force: f64,
    }
    struct CachedAl {
        active_n: Vec<bool>,
        stick: Vec<bool>,
        fixed: Vec<Option<f64>>,
        k_factored: NodeSparse,
        corr: Vec<f64>,
        solver: Solver,
    }
    struct AlCtx {
        active_n: Vec<bool>,
        stick: Vec<bool>,
        ln: Vec<f64>,
        lt: Vec<[f64; 3]>,
        kn: Vec<f64>,
        kt: Vec<f64>,
        prets: Vec<PretState>,
        u: Vec<f64>,
        fixed: Vec<Option<f64>>,
        cache: Option<CachedAl>,
    }

    let mut prets = Vec::with_capacity(mesh.pretensions.len());
    for ps in &mesh.pretensions {
        let ld = model.loads.iter().find(|it| it.id == ps.load_id).map(|it| &it.load);
        let (force_n, adj_mm) = match ld {
            Some(Load::BoltPretension { force_n, adjustment_mm, .. }) => (*force_n, *adjustment_mm),
            _ => (None, None),
        };
        if let Some(fn_) = force_n
            && !fn_.is_finite() {
                bail!("bolt pretension #{}: force_n must be finite", ps.load_id);
            }
        if let Some(a) = adj_mm
            && !a.is_finite() {
                bail!("bolt pretension #{}: adjustment_mm must be finite", ps.load_id);
            }
        let eb = node_e[ps.pairs.first().map(|p| p.minus as usize).unwrap_or(0)];
        let area_tot: f64 = ps.pairs.iter().map(|p| p.area_m2).sum::<f64>().max(h_m * h_m);
        let kc: Vec<f64> = ps.pairs.iter().map(|p| 40.0 * eb * (p.area_m2 / h_m)).collect();
        let delta = match (force_n, adj_mm) {
            (_, Some(a)) => a * 1e-3,
            (Some(fn_), None) => (fn_ * (4.0 * h_m) / (eb * area_tot)).clamp(-0.05, 0.05),
            (None, None) => 0.0,
        };
        let l_cut = vec![[0.0; 3]; ps.pairs.len()];
        prets.push(PretState { force_n, delta, kc, l_cut, step1_force: 0.0 });
    }

    let mut ctx = AlCtx {
        active_n,
        stick,
        ln: vec![0.0f64; nc],
        lt: vec![[0.0f64; 3]; nc],
        kn,
        kt,
        prets,
        u: vec![0.0; n],
        fixed: full_fixed.clone(),
        cache: None,
    };

    #[allow(clippy::too_many_arguments)]
    fn run_al_stage(
        stage_name: &str,
        mesh: &Mesh,
        k0: &NodeSparse,
        comp_contact: &[(usize, f64)],
        f_stage: &[f64],
        stage_fixed: &[Option<f64>],
        adjust_bolt_force: bool,
        base_p: f64,
        span_p: f64,
        prog: Prog,
        res: &mut Results,
        ctx: &mut AlCtx,
    ) -> Result<()> {
        let n = k0.n();
        let h_m = mesh.h * 1e-3;
        let eye = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let mut comp_fixed = stage_fixed.to_vec();
        let fscale = f_stage.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-12);
        let bolt_target_scale: f64 = ctx.prets.iter().filter_map(|p| p.force_n).map(|v| v.abs()).sum::<f64>().max(fscale);
        for pass in 0..80usize {
            let need_rebuild = match &ctx.cache {
                None => true,
                Some(c) => c.active_n != ctx.active_n || c.stick != ctx.stick || c.fixed != comp_fixed,
            };
            if need_rebuild {
                let mut k_iter = k0.clone();
                for (i, c) in mesh.contacts.iter().enumerate() {
                    if !ctx.active_n[i] {
                        continue;
                    }
                    let nb = (c.n_b as usize).min(4);
                    let mut m = outer3(c.normal, ctx.kn[i]);
                    if ctx.stick[i] {
                        let pn = outer3(c.normal, 1.0);
                        for idx in 0..9 {
                            m[idx] += ctx.kt[i] * (eye[idx] - pn[idx]);
                        }
                    }
                    add_contact_matrix(&mut k_iter, c.node_a, &c.nodes_b[..nb], &c.weights_b[..nb], &m);
                }
                for (s, ps) in mesh.pretensions.iter().enumerate() {
                    for (j, pr) in ps.pairs.iter().enumerate() {
                        k_iter.add_symmetric_pair(pr.minus as usize, pr.plus as usize, &eye.map(|v| v * ctx.prets[s].kc[j]));
                    }
                }
                let g_vec: Vec<f64> = comp_fixed.iter().map(|g| g.unwrap_or(0.0)).collect();
                let mut corr = vec![0.0; n];
                k_iter.matvec(&g_vec, &mut corr);
                let mut dummy = vec![0.0; n];
                k_iter.apply_dirichlet(&comp_fixed, &mut dummy);
                let solver = Solver::new(&k_iter)?;
                res.info.solver = format!("{} (AL)", solver.name());
                ctx.cache = Some(CachedAl {
                    active_n: ctx.active_n.clone(),
                    stick: ctx.stick.clone(),
                    fixed: comp_fixed.clone(),
                    k_factored: k_iter,
                    corr,
                    solver,
                });
            }
            let c_ref = ctx.cache.as_ref().unwrap();
            let mut rhs = f_stage.to_vec();
            for (i, c) in mesh.contacts.iter().enumerate() {
                if !ctx.active_n[i] {
                    continue;
                }
                let nb = (c.n_b as usize).min(4);
                let fc = add(scale(c.normal, ctx.ln[i] - ctx.kn[i] * c.gap0_m), ctx.lt[i]);
                for d in 0..3 {
                    rhs[c.node_a as usize * 3 + d] += fc[d];
                }
                for (&bk, &wk) in c.nodes_b[..nb].iter().zip(&c.weights_b[..nb]) {
                    for d in 0..3 {
                        rhs[bk as usize * 3 + d] -= wk * fc[d];
                    }
                }
            }
            for (s, ps) in mesh.pretensions.iter().enumerate() {
                let ncut = ps.normal;
                let del = ctx.prets[s].delta;
                for (j, pr) in ps.pairs.iter().enumerate() {
                    let fc = add(ctx.prets[s].l_cut[j], scale(ncut, ctx.prets[s].kc[j] * del));
                    for d in 0..3 {
                        rhs[pr.minus as usize * 3 + d] += fc[d];
                        rhs[pr.plus as usize * 3 + d] -= fc[d];
                    }
                }
            }
            for (r, g) in comp_fixed.iter().enumerate() {
                if let Some(gv) = g {
                    rhs[r] = *gv;
                    ctx.u[r] = *gv;
                } else {
                    rhs[r] -= c_ref.corr[r];
                }
            }
            let its = c_ref.solver.solve(&c_ref.k_factored, &rhs, &mut ctx.u, 1e-9, &prog.scalar(base_p, base_p + span_p * 0.9, stage_name))?;
            res.info.iterations += its;

            let mut status_changed = 0usize;
            if !comp_contact.is_empty() {
                let mut ku = vec![0.0; n];
                k0.matvec(&ctx.u, &mut ku);
                for &(dof, sgn) in comp_contact {
                    if comp_fixed[dof].is_some() {
                        if (ku[dof] - f_stage[dof]) * sgn > 1e-7 * fscale {
                            comp_fixed[dof] = None;
                            status_changed += 1;
                        }
                    } else if ctx.u[dof] * sgn > 1e-12 {
                        comp_fixed[dof] = Some(0.0);
                        status_changed += 1;
                    }
                }
            }

            let mut max_gap_err = 0.0f64;
            let mut dl2 = 0.0f64;
            let mut l2 = 0.0f64;
            for (i, c) in mesh.contacts.iter().enumerate() {
                let nb = (c.n_b as usize).min(4);
                let ua = [ctx.u[c.node_a as usize * 3], ctx.u[c.node_a as usize * 3 + 1], ctx.u[c.node_a as usize * 3 + 2]];
                let mut ub = [0.0; 3];
                for (&bk, &wk) in c.nodes_b[..nb].iter().zip(&c.weights_b[..nb]) {
                    for d in 0..3 {
                        ub[d] += wk * ctx.u[bk as usize * 3 + d];
                    }
                }
                let du = sub(ua, ub);
                let dn = dot(du, c.normal);
                let g_n = c.gap0_m + dn;
                let dt = sub(du, scale(c.normal, dn));

                let prev_act = ctx.active_n[i];
                let prev_stk = ctx.stick[i];
                if !prev_act {
                    if g_n < -1e-9 * h_m {
                        ctx.active_n[i] = true;
                        ctx.stick[i] = false;
                        ctx.ln[i] = 0.0;
                        ctx.lt[i] = [0.0; 3];
                        status_changed += 1;
                    }
                } else {
                    let p_trial = ctx.ln[i] - ctx.kn[i] * g_n;
                    if c.kind != ContactKind::NoSeparation && p_trial <= 0.0 {
                        ctx.active_n[i] = false;
                        ctx.stick[i] = false;
                        dl2 += ctx.ln[i] * ctx.ln[i] + dot(ctx.lt[i], ctx.lt[i]);
                        ctx.ln[i] = 0.0;
                        ctx.lt[i] = [0.0; 3];
                        status_changed += 1;
                    } else {
                        max_gap_err = max_gap_err.max(g_n.abs());
                        let old_ln = ctx.ln[i];
                        let old_lt = ctx.lt[i];
                        ctx.ln[i] = p_trial;
                        if c.kind == ContactKind::Frictional && c.friction > 0.0 && ctx.ln[i] > 0.0 {
                            let kt_upd = if prev_stk { ctx.kt[i] } else { 0.05 * ctx.kt[i] };
                            let t_trial = sub(ctx.lt[i], scale(dt, kt_upd));
                            let lim = c.friction * ctx.ln[i];
                            let tn = norm(t_trial);
                            if tn <= lim {
                                ctx.lt[i] = t_trial;
                                ctx.stick[i] = true;
                            } else {
                                ctx.lt[i] = scale(t_trial, lim / tn);
                                ctx.stick[i] = false;
                            }
                        } else {
                            ctx.lt[i] = [0.0; 3];
                            ctx.stick[i] = false;
                        }
                        if ctx.stick[i] != prev_stk {
                            status_changed += 1;
                        }
                        dl2 += (ctx.ln[i] - old_ln).powi(2) + dot(sub(ctx.lt[i], old_lt), sub(ctx.lt[i], old_lt));
                        l2 += ctx.ln[i].powi(2) + dot(ctx.lt[i], ctx.lt[i]);
                    }
                }
            }

            for (s, ps) in mesh.pretensions.iter().enumerate() {
                let ncut = ps.normal;
                let del = ctx.prets[s].delta;
                for (j, pr) in ps.pairs.iter().enumerate() {
                    let um = [ctx.u[pr.minus as usize * 3], ctx.u[pr.minus as usize * 3 + 1], ctx.u[pr.minus as usize * 3 + 2]];
                    let up = [ctx.u[pr.plus as usize * 3], ctx.u[pr.plus as usize * 3 + 1], ctx.u[pr.plus as usize * 3 + 2]];
                    let jump = add(sub(up, um), scale(ncut, del));
                    max_gap_err = max_gap_err.max(norm(jump));
                    let old_lc = ctx.prets[s].l_cut[j];
                    let new_lc = add(old_lc, scale(jump, ctx.prets[s].kc[j]));
                    ctx.prets[s].l_cut[j] = new_lc;
                    dl2 += dot(sub(new_lc, old_lc), sub(new_lc, old_lc));
                    l2 += dot(new_lc, new_lc);
                }
            }

            let rel_dl = dl2.sqrt() / l2.sqrt().max(bolt_target_scale).max(1.0);
            let mut bolt_err = 0.0f64;
            if adjust_bolt_force && status_changed == 0 && (rel_dl < 5e-2 || pass >= 4) {
                for (s, ps) in mesh.pretensions.iter().enumerate() {
                    if let Some(f_target) = ctx.prets[s].force_n {
                        let f_cur: f64 = ctx.prets[s].l_cut.iter().map(|lc| dot(*lc, ps.normal)).sum();
                        let rel = (f_cur - f_target).abs() / f_target.abs().max(1.0);
                        bolt_err = bolt_err.max(rel);
                        if rel > 2e-4 && f_cur.abs() > 1e-9 {
                            let ratio = (f_target / f_cur).clamp(0.2, 5.0);
                            ctx.prets[s].delta *= ratio;
                            for lc in &mut ctx.prets[s].l_cut {
                                *lc = scale(*lc, ratio);
                            }
                        }
                    }
                }
            }

            if status_changed == 0 && max_gap_err <= 2e-5 * h_m.max(1e-4) && rel_dl <= 1e-4 && (!adjust_bolt_force || bolt_err <= 2e-4) && pass >= 1 {
                ctx.fixed = comp_fixed;
                return Ok(());
            }
            if pass == 79 {
                ctx.fixed = comp_fixed.clone();
                res.warnings.push(format!("{stage_name} did not fully tighten AL tolerance (gap err {max_gap_err:.2e} m, dL {rel_dl:.2e})"));
            }
        }
        Ok(())
    }

    if !ctx.prets.is_empty() {
        let fixed_step1: Vec<Option<f64>> = full_fixed.iter().map(|g| g.map(|_| 0.0)).collect();
        let f_step1 = vec![0.0; n];
        run_al_stage("Step 1: bolt pretension", mesh, &k0, &comp_contact, &f_step1, &fixed_step1, true, 0.05, 0.4, prog, res, &mut ctx)?;
        for (s, ps) in mesh.pretensions.iter().enumerate() {
            ctx.prets[s].step1_force = ctx.prets[s].l_cut.iter().map(|lc| dot(*lc, ps.normal)).sum();
        }
        run_al_stage("Step 2: external loads", mesh, &k0, &comp_contact, &f_ext, &full_fixed, false, 0.45, 0.45, prog, res, &mut ctx)?;
        for (s, ps) in mesh.pretensions.iter().enumerate() {
            let f2: f64 = ctx.prets[s].l_cut.iter().map(|lc| dot(*lc, ps.normal)).sum();
            let f1 = ctx.prets[s].step1_force;
            res.quantity(&format!("Bolt #{} force (Step 1)", ps.load_id), f1, "N");
            res.quantity(&format!("Bolt #{} force (Step 2)", ps.load_id), f2, "N");
            res.quantity(&format!("Bolt #{} adjustment", ps.load_id), ctx.prets[s].delta * 1e3, "mm");
            res.quantity(&format!("Bolt #{} preload loss", ps.load_id), f1 - f2, "N");
        }
    } else {
        run_al_stage("contact AL solve", mesh, &k0, &comp_contact, &f_ext, &full_fixed, false, 0.05, 0.85, prog, res, &mut ctx)?;
    }

    if !mesh.contacts.is_empty() {
        let mut p_node_f = vec![0.0f64; nn];
        let mut p_node_a = vec![0.0f64; nn];
        let mut peak_p = 0.0f64;
        let mut sum_fn = 0.0f64;
        let mut sum_ft = [0.0f64; 3];
        let mut closed = 0usize;
        for (i, c) in mesh.contacts.iter().enumerate() {
            if ctx.active_n[i] {
                closed += 1;
            }
            let f_n = ctx.ln[i];
            sum_fn += f_n;
            sum_ft = add(sum_ft, ctx.lt[i]);
            let p_mpa = (f_n / c.area_m2.max(1e-18)) * 1e-6;
            if ctx.active_n[i] {
                peak_p = peak_p.max(p_mpa);
            }
            p_node_f[c.node_a as usize] += f_n;
            p_node_a[c.node_a as usize] += c.area_m2;
            let nb = (c.n_b as usize).min(4);
            for (&bk, &wk) in c.nodes_b[..nb].iter().zip(&c.weights_b[..nb]) {
                p_node_f[bk as usize] += wk * f_n;
                p_node_a[bk as usize] += wk * c.area_m2;
            }
        }
        let cp_field: Vec<f64> = p_node_f.iter().zip(&p_node_a).map(|(f, a)| if *a > 0.0 { (*f / *a) * 1e-6 } else { 0.0 }).collect();
        res.extra.insert(Field::ContactPressure, cp_field);
        res.quantity("Peak contact pressure", peak_p, "MPa");
        res.quantity("Contact normal force", sum_fn, "N");
        res.quantity("Contact tangential force", norm(sum_ft), "N");
        res.quantity("Contact nodes closed", closed as f64, "");
        res.quantity("Contact nodes open", (mesh.contacts.len() - closed) as f64, "");
    }
    if !comp_contact.is_empty() {
        let open = comp_contact.iter().filter(|(d, _)| ctx.fixed[*d].is_none()).count();
        res.quantity("Contact nodes in contact", (comp_contact.len() - open) as f64, "");
        res.quantity("Contact nodes lifted off", open as f64, "");
    }

    let mut f_with_contact = f_ext.clone();
    for (i, c) in mesh.contacts.iter().enumerate() {
        if !ctx.active_n[i] {
            continue;
        }
        let fc = add(scale(c.normal, ctx.ln[i]), ctx.lt[i]);
        for d in 0..3 {
            f_with_contact[c.node_a as usize * 3 + d] += fc[d];
        }
        let nb = (c.n_b as usize).min(4);
        for (&bk, &wk) in c.nodes_b[..nb].iter().zip(&c.weights_b[..nb]) {
            for d in 0..3 {
                f_with_contact[bk as usize * 3 + d] -= wk * fc[d];
            }
        }
    }
    for (s, ps) in mesh.pretensions.iter().enumerate() {
        for (j, pr) in ps.pairs.iter().enumerate() {
            let fc = ctx.prets[s].l_cut[j];
            for d in 0..3 {
                f_with_contact[pr.minus as usize * 3 + d] += fc[d];
                f_with_contact[pr.plus as usize * 3 + d] -= fc[d];
            }
        }
    }

    res.info.dofs = n;
    prog.report(0.92, "computing stresses")?;
    structural_post_ex(model, mesh, &ctx.u, temps, &units, &ctx.fixed, &f_with_contact, &support_nodes, &owner, Some(&couplings), res);
    prog.report(1.0, "done")?;
    Ok(())
}

/// Stresses, safety factor and reactions from a displacement field `u` [m].
#[allow(clippy::too_many_arguments)]
pub fn structural_post(model: &Model, mesh: &Mesh, u: &[f64], temps: Option<&[f64]>, units: &[ElemUnit], fixed: &[Option<f64>], f_ext: &[f64], support_nodes: &[(u32, String, Vec<u32>)], owner: &[u32], res: &mut Results) {
    structural_post_ex(model, mesh, u, temps, units, fixed, f_ext, support_nodes, owner, None, res);
}

#[allow(clippy::too_many_arguments)]
pub fn structural_post_ex(model: &Model, mesh: &Mesh, u: &[f64], temps: Option<&[f64]>, units: &[ElemUnit], fixed: &[Option<f64>], f_ext: &[f64], support_nodes: &[(u32, String, Vec<u32>)], owner: &[u32], couplings: Option<&StructuralCouplings>, res: &mut Results) {
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
            let h = mesh.elem_h_m(e);
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

    let total_dofs = owner.len();
    let mut fint = vec![0.0; total_dofs.max(nn * 3)];
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
    if let Some(cp) = couplings {
        cp.add_internal_forces(u, &mut fint);
    }
    // reactions live on the independent dofs: hanging-node forces go to their parents
    mesh.condense_rhs(&mut fint, 3);
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
        for dof in 0..fixed.len().min(owner.len()) {
            if owner[dof] == *id && fixed[dof].is_some() {
                r[dof % 3] += fint[dof] - f_ext.get(dof).copied().unwrap_or(0.0);
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
    let vox: f64 = els.iter().map(|&e| mesh.elem_volume(e) * 1e-9).sum();
    if let Some(p) = power {
        return Ok((els, p / vox));
    }
    let mut q = *value;
    if let Some(name) = body
        && let Some(b) = model.bodies.iter().find(|b| b.name.eq_ignore_ascii_case(name)) {
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
    assemble_thermal_k(model, mesh, elem_q, sel, None)
}

/// Per-element conductivity factor k(T)/k at the element mean temperature (None when no material
/// has a conductivity table).
pub fn elem_kscale(mesh: &Mesh, t: &[f64]) -> Option<Vec<f64>> {
    if !mesh.materials.iter().any(|m| !m.conductivity_table.is_empty()) {
        return None;
    }
    Some(
        mesh.elems
            .par_iter()
            .enumerate()
            .map(|(e, en)| {
                let m = &mesh.materials[mesh.elem_mat[e] as usize];
                m.k_factor(en.iter().map(|&n| t[n as usize]).sum::<f64>() / 8.0)
            })
            .collect(),
    )
}

/// `assemble_thermal_sel` with per-element conductivity factors (temperature-dependent k).
pub fn assemble_thermal_k(model: &Model, mesh: &Mesh, elem_q: Option<&[f64]>, sel: Option<(&[bool], f64)>, kscale: Option<&[f64]>) -> Result<ThermalSystem> {
    let h = mesh.h * 1e-3;
    let nn = mesh.nodes.len();
    let ku = hex_conduction_unit();
    let on = |e: usize| sel.is_none_or(|(s, _)| s[e]);
    let elems: Vec<[u32; 8]> = match sel {
        Some((s, _)) => mesh.elems.iter().zip(s).filter(|(_, b)| **b).map(|(e, _)| *e).collect(),
        None => mesh.elems.clone(),
    };
    let ties = mesh.interface_ties();
    let mut k = NodeSparse::new_with_edges(nn, &elems, &ties, 1);
    k.hang = mesh.hang_ctx();
    let kax = hex_conduction_unit_axes();
    let general: Vec<Option<[Vec<f64>; 4]>> = (0..mesh.elems.len()).into_par_iter().map(|e| (on(e) && mesh.is_fitted(e)).then(|| hex_conduction_general(&elem_xyz_m(mesh, e)))).collect();
    for (e, en) in mesh.elems.iter().enumerate() {
        if on(e) {
            let m = &mesh.materials[mesh.elem_mat[e] as usize];
            let ks = kscale.map(|v| v[e]).unwrap_or(1.0);
            match (&general[e], m.conductivity_xyz) {
                (Some(g), None) => k.add_element(en, &g[0], m.thermal_conductivity * ks),
                (Some(g), Some(kk)) => {
                    for a in 0..3 {
                        k.add_element(en, &g[a + 1], kk[a] * ks);
                    }
                }
                (None, None) => k.add_element(en, &ku, m.thermal_conductivity * mesh.elem_h_m(e) * ks),
                (None, Some(kk)) => {
                    for a in 0..3 {
                        k.add_element(en, &kax[a], kk[a] * mesh.elem_h_m(e) * ks);
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
                    let a = face_shares(mesh, fi as usize).1;
                    for (j, &n) in mesh.faces[fi as usize].nodes.iter().enumerate() {
                        f[n as usize] += value * a[j];
                    }
                }
            }
            Load::Convection { faces, coefficient, ambient } => {
                for &fi in &mesh.resolve_faces(faces)? {
                    let a = face_shares(mesh, fi as usize).1;
                    for (j, &n) in mesh.faces[fi as usize].nodes.iter().enumerate() {
                        let c = coefficient * a[j];
                        conv_diag[n as usize] += c;
                        f[n as usize] += c * ambient;
                    }
                }
                has_sink = true;
            }
            Load::Radiation { .. } => has_sink = true,
            Load::HeatGeneration { .. } => {
                let (els, q) = heat_source(model, mesh, &it.load)?;
                for e in els {
                    let nv = node_volumes(mesh, e);
                    for (a, &n) in mesh.elems[e].iter().enumerate() {
                        f[n as usize] += q * nv[a];
                    }
                }
            }
            _ => {}
        }
    }
    if let Some(q) = elem_q {
        for (e, en) in mesh.elems.iter().enumerate().filter(|(e, _)| on(*e)) {
            let nv = node_volumes(mesh, e);
            for (a, &n) in en.iter().enumerate() {
                f[n as usize] += q[e] * nv[a];
            }
        }
    }
    for (n, c) in conv_diag.iter().enumerate() {
        if *c != 0.0 {
            k.add_diag(n, *c);
        }
    }
    if !ties.is_empty() {
        let avg_k = mesh.materials.iter().map(|m| m.thermal_conductivity).sum::<f64>() / mesh.materials.len().max(1) as f64;
        add_scalar_ties(mesh, &mut k, avg_k.max(1e-3) * h * 1e4);
    }
    Ok(ThermalSystem { k, f, fixed, conv_diag, temp_sets, has_sink })
}

/// Solve one linear conduction system (Dirichlet already applied) into `t`; returns iterations.
fn solve_thermal_linear(model: &Model, mesh: &Mesh, k: &NodeSparse, f: &[f64], t: &mut [f64], prog: Prog, res: &mut Results) -> Result<usize> {
    if use_multigrid(model, mesh) {
        // aggregation multigrid on the voxel lattice (nodes placed by position)
        prog.report(0.03, "building multigrid")?;
        let t0 = Instant::now();
        let coords: Vec<[u32; 3]> = mesh.nodes.iter().map(|p| [0, 1, 2].map(|k| ((p[k] - mesh.origin[k]) / mesh.h).round().max(0.0) as u32)).collect();
        let mg = crate::multigrid::Multigrid::new(k, &coords)?;
        let its = mg.solve_cancellable(f, t, 1e-10, 2000, prog.cancel)?;
        crate::sparse::SOLVE_NANOS.fetch_add(t0.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        res.info.solver = format!("multigrid CG ({} levels, CPU)", mg.levels());
        Ok(its)
    } else {
        let solver = Solver::new(k)?;
        let its = solver.solve(k, f, t, 1e-10, &prog.scalar(0.05, 0.85, "solving thermal (PCG)"))?;
        res.info.solver = solver.name().to_string();
        Ok(its)
    }
}

pub fn thermal(model: &Model, mesh: &Mesh, elem_q: Option<&[f64]>, prog: Prog, res: &mut Results) -> Result<()> {
    let nn = mesh.nodes.len();
    prog.report(0.0, "assembling conduction")?;
    let rad = crate::radiation::Radiation::new(model, mesh, prog.slice(0.0, 0.05))?;
    let kdep = mesh.materials.iter().any(|m| !m.conductivity_table.is_empty());
    let nonlinear = rad.is_some() || kdep;
    let base = assemble_thermal(model, mesh, elem_q)?;
    if !base.has_sink {
        bail!("thermal model needs at least one Temperature, Convection or Radiation boundary condition");
    }
    let mut t: Vec<f64> = base.fixed.iter().map(|g| g.unwrap_or(model.analysis.reference_temperature)).collect();
    // radiation and k(T) are non-linear: Picard iterations, each re-linearised at the current
    // temperature (radiation in secant form, so a converged state satisfies q = εσ(T⁴ − T∞⁴) exactly)
    let max_it = if nonlinear { 80 } else { 1 };
    let (mut omega, mut prev_change, mut change) = (1.0f64, f64::INFINITY, 0.0f64);
    let mut converged = !nonlinear;
    let mut used = base.clone();
    let mut iters = 0usize;
    for it in 0..max_it {
        let ks = elem_kscale(mesh, &t);
        let mut s = match &ks {
            Some(v) => assemble_thermal_k(model, mesh, elem_q, None, Some(v))?,
            None => base.clone(),
        };
        if let Some(r) = &rad {
            let lin = r.linearize(&t);
            for n in 0..nn {
                if lin.diag[n] != 0.0 {
                    s.k.add_diag(n, lin.diag[n]);
                    s.conv_diag[n] += lin.diag[n];
                }
                s.f[n] += lin.rhs[n];
            }
        }
        let mut k = s.k.clone();
        let mut f = s.f.clone();
        k.apply_dirichlet(&s.fixed, &mut f);
        let mut tn = t.clone();
        res.info.iterations += solve_thermal_linear(model, mesh, &k, &f, &mut tn, prog, res)?;
        used = s;
        iters = it + 1;
        if !nonlinear {
            t = tn;
            break;
        }
        change = tn.iter().zip(&t).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        if change > prev_change {
            omega = (omega * 0.6).max(0.25);
        }
        prev_change = change;
        for (a, b) in t.iter_mut().zip(&tn) {
            *a += omega * (b - *a);
        }
        prog.report(0.05 + 0.9 * (it as f64 / max_it as f64), &format!("non-linear iteration {} (ΔT {change:.2e} K)", it + 1))?;
        if change < 1e-4 {
            converged = true;
            break;
        }
    }
    if nonlinear {
        res.quantity("Non-linear iterations (radiation / k(T))", iters as f64, "");
        if !converged {
            res.warnings.push(format!("radiation / temperature-dependent conduction did not fully converge in {iters} iterations (last change {change:.2e} K)"));
        }
    }
    res.info.dofs += nn;
    thermal_post(mesh, &t, Some(&used), res);
    if let Some(r) = &rad {
        radiation_quantities(r, &t, res);
    }
    res.temp = Some(t);
    prog.report(1.0, "done")?;
    Ok(())
}

/// Heat radiated through each radiation load (positive = leaving the part).
pub fn radiation_quantities(r: &crate::radiation::Radiation, t: &[f64], res: &mut Results) {
    for (id, label, w) in r.heat_flows(t) {
        res.quantity(&format!("Heat radiated through #{id} ({label})"), w, "W");
    }
    if r.patches > 0 {
        res.quantity("Radiation patches (enclosure)", r.patches as f64, "");
        res.quantity("View-factor rays", r.rays as f64, "");
    }
}

/// Heat flux field and (optionally) heat flow through fixed-temperature faces.
pub fn thermal_post(mesh: &Mesh, t: &[f64], sys: Option<&ThermalSystem>, res: &mut Results) {
    let nn = mesh.nodes.len();
    let ku = hex_conduction_unit();
    let kax = hex_conduction_unit_axes();
    let cg = center_grads();
    let mut fint = vec![0.0; nn];
    let mut acc = vec![[0.0; 3]; nn];
    let mut cnt = vec![0u32; nn];
    let kscale = elem_kscale(mesh, t);
    for (e, en) in mesh.elems.iter().enumerate() {
        let mat = &mesh.materials[mesh.elem_mat[e] as usize];
        let ks = kscale.as_ref().map(|v| v[e]).unwrap_or(1.0);
        let kc = mat.thermal_conductivity * ks;
        let h = mesh.elem_h_m(e);
        let kv = [mat.k_axis(0) * ks, mat.k_axis(1) * ks, mat.k_axis(2) * ks];
        let ortho = mat.conductivity_xyz.is_some();
        let fitted = mesh.is_fitted(e);
        let xm = fitted.then(|| elem_xyz_m(mesh, e));
        let gk = xm.as_ref().map(hex_conduction_general);
        let gc = xm.as_ref().map(center_grads_general);
        let mut grad = [0.0; 3];
        for a in 0..8 {
            for c in 0..3 {
                grad[c] += match &gc {
                    Some(g) => g[a][c] * t[en[a] as usize],
                    None => cg[a][c] * t[en[a] as usize] / h,
                };
            }
        }
        // element matrices: general (SI, no h factor) or the scaled unit cube
        let (iso, axes, scale_h): (&[f64], [&[f64]; 3], f64) = match &gk {
            Some(g) => (&g[0], [&g[1], &g[2], &g[3]], 1.0),
            None => (&ku, [&kax[0], &kax[1], &kax[2]], h),
        };
        for a in 0..8 {
            let n = en[a] as usize;
            let s: f64 = if ortho {
                (0..3).map(|ax| kv[ax] * (0..8).map(|b| axes[ax][a * 8 + b] * t[en[b] as usize]).sum::<f64>()).sum()
            } else {
                kc * (0..8).map(|b| iso[a * 8 + b] * t[en[b] as usize]).sum::<f64>()
            };
            fint[n] += s * scale_h;
            for c in 0..3 {
                acc[n][c] += -kv[c] * grad[c];
            }
            cnt[n] += 1;
        }
    }
    if let Some(sys) = sys {
        let mut claimed = vec![false; nn];
        // nodal heat residual; on octree meshes hanging-node shares go to their parents
        let mut resid: Vec<f64> = (0..nn).map(|n| fint[n] + sys.conv_diag[n] * t[n] - sys.f[n]).collect();
        mesh.condense_rhs(&mut resid, 1);
        for (id, label, nodes) in &sys.temp_sets {
            let mut q = 0.0;
            for &n in nodes {
                let n = n as usize;
                if !claimed[n] {
                    claimed[n] = true;
                    q += resid[n];
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
    prog.report(1.0, "done")?;
    Ok(())
}

/// Lowest `nmodes` modes by shifted subspace iteration with Rayleigh–Ritz.
pub fn modal_basis(model: &Model, mesh: &Mesh, nmodes: usize, prog: Prog, res: &mut Results) -> Result<ModalBasis> {
    let (asm, _) = assemble_structural(model, mesh, prog)?;
    let Assembled { mut k, mut fixed, couplings, .. } = asm;
    couplings.apply_linear_interface_ties(mesh, &mut k);
    let n = k.n();
    for g in fixed.iter_mut() {
        if g.is_some() {
            *g = Some(0.0);
        }
    }
    let mut mdiag = vec![0.0; n];
    for (e, en) in mesh.elems.iter().enumerate() {
        let rho = mesh.materials[mesh.elem_mat[e] as usize].density;
        let nv = node_volumes(mesh, e);
        for (j, &a) in en.iter().enumerate() {
            for c in 0..3 {
                mdiag[a as usize * 3 + c] += rho * nv[j];
            }
        }
    }
    couplings.apply_mass_diag(&mut mdiag);
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
    prog.report(0.02, "factorising")?;
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
    let check_fn = prog.check_scalar();
    for it in 0..max_it {
        prog.report(it as f64 / max_it as f64 * 0.95, &format!("subspace iteration {}", it + 1))?;
        let mut y = Vec::with_capacity(p);
        for j in 0..p {
            let rhs: Vec<f64> = x[j].iter().zip(&mdiag).map(|(a, b)| a * b).collect();
            let mut yj: Vec<f64> = if first { vec![0.0; n] } else { x[j].iter().map(|v| v / mu[j]).collect() };
            iters += solver.solve(&k, &rhs, &mut yj, 1e-8, &check_fn)?;
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
