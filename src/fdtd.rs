//! Full-wave electromagnetics by FDTD (Yee scheme) on the voxel grid, with microwave heating.
//!
//! * Yee grid: E on cell edges, H on cell faces; the mesh's voxel cells give εr, σ and metal
//!   (PEC) per cell. Edge properties average the four cells around the edge; an edge touching a
//!   metal cell is PEC.
//! * Losses: σ_eff = σ + ωε₀εr·tanδ (loss tangent from the material) at the excitation frequency.
//! * Boundary: "pec" = the enclosure is a closed metal cavity (microwave oven); "open" = a
//!   10-cell graded absorbing layer (matched σ/ε = σ*/μ) around the enclosure, with materials
//!   extruded into it so waveguides and boards run into the absorber.
//! * Source: `Load::Port`, a current sheet on the port faces driving E along the polarization axis
//!   (half-sine "te10" profile across the face, or uniform), ramped over 8 periods.
//! * Steady state: running DFT over blocks of whole periods (the time step is chosen so a period
//!   is an integer number of steps) until the delivered and absorbed powers change < 0.5 %.
//! * Normalisation: the fields are scaled so the ports deliver their stated power
//!   (P = −½ Re ∫ E·J* dV), so absorbed power and fields are in W/m³ and V/m.
//! * Runs on the CPU (rayon) or the GPU (f32 WGSL, 8 storage buffers); same algorithm.
//! * Microwave heating: absorbed power per solid element → transient thermal on the solids.

use crate::fem::Prog;
use crate::mesh::{Mesh, REGION_SOLID};
use crate::model::{Load, Model};
use crate::results::*;
use anyhow::{Result, anyhow, bail};
use rayon::prelude::*;
use std::collections::BTreeMap;

const C0: f64 = 299_792_458.0;
const EPS0: f64 = 8.854_187_817e-12;
const MU0: f64 = 1.256_637_062_12e-6;
const NPML: usize = 10;
const RAMP_PERIODS: usize = 8;
const BLOCK_PERIODS: usize = 4;
const TOL: f64 = 0.005;

/// Discretised problem: grid, update coefficients and sources.
pub struct Setup {
    /// Cells per axis including the absorbing padding.
    pub n: [usize; 3],
    pub pad: usize,
    /// Cell size [m].
    pub h: f64,
    pub dt: f64,
    pub freq: f64,
    /// Time steps per RF period (exact).
    pub period: usize,
    /// Node-grid size (n+1)³ product: length of each field component array.
    pub nn: usize,
    /// ca (3·nn) then cb (3·nn) per E edge.
    pub ecoef: Vec<f32>,
    /// da (3·nn) then db (3·nn) per H face.
    pub hcoef: Vec<f32>,
    /// Per cell: εr, σ_eff [S/m], metal, inside the real (unpadded) domain.
    pub cell_eps: Vec<f64>,
    pub cell_sig: Vec<f64>,
    pub cell_pec: Vec<bool>,
    /// Flat field index (component·nn + node) of each source edge, its E decrement amplitude and
    /// current density amplitude [A/m²].
    pub src_idx: Vec<u32>,
    pub src_amp: Vec<f32>,
    pub src_j: Vec<f64>,
    pub power: f64,
    pub cells_per_wavelength: f64,
    pub warnings: Vec<String>,
}

impl Setup {
    fn n1(&self) -> [usize; 3] {
        [self.n[0] + 1, self.n[1] + 1, self.n[2] + 1]
    }
    fn node(&self, i: usize, j: usize, k: usize) -> usize {
        let n1 = self.n1();
        i + n1[0] * (j + n1[1] * k)
    }
    fn cell(&self, i: usize, j: usize, k: usize) -> usize {
        i + self.n[0] * (j + self.n[1] * k)
    }
    pub fn cells(&self) -> usize {
        self.n[0] * self.n[1] * self.n[2]
    }
}

fn axis_of(d: u8) -> (usize, bool) {
    ((d / 2) as usize, d % 2 == 1)
}

/// Build the FDTD grid from the mesh (which includes the enclosure).
pub fn setup(model: &Model, mesh: &Mesh) -> Result<Setup> {
    let st = &model.analysis;
    let freq = st.frequency_hz;
    if !(freq > 0.0) {
        bail!("frequency must be > 0");
    }
    let omega = std::f64::consts::TAU * freq;
    let open = match st.em_boundary.as_str() {
        "pec" => false,
        "open" => true,
        b => bail!("em_boundary must be pec or open, not '{b}'"),
    };
    let h = mesh.h * 1e-3;
    let pad = if open { NPML } else { 0 };
    let md = mesh.dims;
    let n = [md[0] + 2 * pad, md[1] + 2 * pad, md[2] + 2 * pad];
    let ncell = n[0] * n[1] * n[2];
    let n1 = [n[0] + 1, n[1] + 1, n[2] + 1];
    let nn = n1[0] * n1[1] * n1[2];
    if nn as u64 * 6 >= u32::MAX as u64 {
        bail!("grid too large ({} cells); use a larger element size", ncell);
    }
    let mut warnings = Vec::new();

    // ---- cell properties on the mesh grid, then extruded into the padding
    let mats = &mesh.materials;
    let mat_sig: Vec<f64> = mats.iter().map(|m| m.sigma_at(freq)).collect();
    let mat_pec: Vec<bool> = mats
        .iter()
        .zip(&mat_sig)
        .map(|(m, &s)| {
            let good = s > 10.0 * omega * EPS0 * m.relative_permittivity;
            let skin = (2.0 / (omega * MU0 * m.relative_permeability.max(1.0) * s.max(1e-30))).sqrt();
            good && skin < 0.5 * h
        })
        .collect();
    let mcells = md[0] * md[1] * md[2];
    let mut m_eps = vec![1.0f64; mcells];
    let mut m_sig = vec![0.0f64; mcells];
    let mut m_pec = vec![false; mcells];
    let mut lossy_conductor = Vec::new();
    for (e, c) in mesh.elem_cell.iter().enumerate() {
        let mi = mesh.elem_mat[e] as usize;
        let id = c[0] as usize + md[0] * (c[1] as usize + md[1] * c[2] as usize);
        m_eps[id] = mats[mi].relative_permittivity.max(1.0);
        m_sig[id] = mat_sig[mi];
        m_pec[id] = mat_pec[mi];
        if !mat_pec[mi] && mat_sig[mi] > 1e3 && !lossy_conductor.contains(&mats[mi].name) {
            lossy_conductor.push(mats[mi].name.clone());
        }
    }
    if !lossy_conductor.is_empty() {
        warnings.push(format!("{}: conductive but thicker than the cells' skin-depth resolution allows — treated as a lossy medium; refine the mesh or expect approximate losses.", lossy_conductor.join(", ")));
    }
    let mut cell_eps = vec![1.0; ncell];
    let mut cell_sig = vec![0.0; ncell];
    let mut cell_pec = vec![false; ncell];
    for k in 0..n[2] {
        for j in 0..n[1] {
            for i in 0..n[0] {
                let cl = |v: usize, a: usize| v.saturating_sub(pad).min(md[a] - 1);
                let src = cl(i, 0) + md[0] * (cl(j, 1) + md[1] * cl(k, 2));
                let c = i + n[0] * (j + n[1] * k);
                cell_eps[c] = m_eps[src];
                cell_sig[c] = m_sig[src];
                cell_pec[c] = m_pec[src];
            }
        }
    }

    // ---- resolution check
    let max_eps = (0..mcells).filter(|&c| !m_pec[c]).map(|c| m_eps[c]).fold(1.0, f64::max);
    let lambda_min = C0 / (freq * max_eps.sqrt());
    let cpw = lambda_min / h;
    if cpw < 3.0 {
        bail!(
            "the mesh is far too coarse for {:.3} GHz: {:.1} cells per wavelength in the densest material (εr = {:.0}); use an element size ≤ {:.2} mm",
            freq / 1e9,
            cpw,
            max_eps,
            lambda_min * 1e3 / 10.0
        );
    }
    if cpw < 10.0 {
        warnings.push(format!("only {cpw:.1} cells per wavelength in the densest material (εr = {max_eps:.0}): use an element size ≤ {:.2} mm for accurate fields (≥ 10 cells per wavelength).", lambda_min * 1e3 / 10.0));
    }

    // ---- time step: an integer number of steps per period, below the vacuum Courant limit
    let dt_max = 0.99 * h / (C0 * 3f64.sqrt());
    let period = (1.0 / freq / dt_max).ceil() as usize;
    let dt = 1.0 / freq / period as f64;

    // ---- absorbing-layer profile κ [1/s] (σ_e = ε κ, σ_m = μ0 κ: matched in any medium)
    let kmax = 3.2 * C0 / h;
    let depth = |pos: f64, a: usize| -> f64 {
        if pad == 0 {
            return 0.0;
        }
        let lo = pad as f64;
        let hi = (n[a] - pad) as f64;
        let d = if pos < lo { (lo - pos) / pad as f64 } else if pos > hi { (pos - hi) / pad as f64 } else { 0.0 };
        d.min(1.0).powi(3)
    };
    let kappa = |p: [f64; 3]| kmax * (depth(p[0], 0) + depth(p[1], 1) + depth(p[2], 2));

    // ---- E coefficients per edge
    let mut ecoef = vec![0.0f32; 6 * nn];
    let mut hcoef = vec![0.0f32; 6 * nn];
    let (ca, cb) = ecoef.split_at_mut(3 * nn);
    ca.par_chunks_mut(nn).zip(cb.par_chunks_mut(nn)).enumerate().for_each(|(a, (ca, cb))| {
        let (b, c) = ((a + 1) % 3, (a + 2) % 3);
        for idx in 0..nn {
            let p = [idx % n1[0], (idx / n1[0]) % n1[1], idx / (n1[0] * n1[1])];
            if p[a] >= n[a] || p[b] == 0 || p[b] >= n[b] || p[c] == 0 || p[c] >= n[c] {
                continue; // not an edge, or tangential on the outer wall (PEC)
            }
            let (mut eps, mut sig, mut pec) = (0.0, 0.0, false);
            for (db, dc) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                let mut q = p;
                q[b] -= db;
                q[c] -= dc;
                let cid = q[0] + n[0] * (q[1] + n[1] * q[2]);
                eps += cell_eps[cid] / 4.0;
                sig += cell_sig[cid] / 4.0;
                pec |= cell_pec[cid];
            }
            if pec {
                continue;
            }
            let mut pos = [p[0] as f64, p[1] as f64, p[2] as f64];
            pos[a] += 0.5;
            let e = EPS0 * eps;
            let s = sig + e * kappa(pos);
            let x = s * dt / (2.0 * e);
            ca[idx] = ((1.0 - x) / (1.0 + x)) as f32;
            cb[idx] = (dt / (e * h) / (1.0 + x)) as f32;
        }
    });
    let (da, db) = hcoef.split_at_mut(3 * nn);
    da.par_chunks_mut(nn).zip(db.par_chunks_mut(nn)).enumerate().for_each(|(a, (da, db))| {
        let (b, c) = ((a + 1) % 3, (a + 2) % 3);
        for idx in 0..nn {
            let p = [idx % n1[0], (idx / n1[0]) % n1[1], idx / (n1[0] * n1[1])];
            if p[b] >= n[b] || p[c] >= n[c] {
                continue;
            }
            let mut pos = [p[0] as f64, p[1] as f64, p[2] as f64];
            pos[b] += 0.5;
            pos[c] += 0.5;
            let x = kappa(pos) * dt / 2.0;
            da[idx] = ((1.0 - x) / (1.0 + x)) as f32;
            db[idx] = (dt / (MU0 * h) / (1.0 + x)) as f32;
        }
    });

    let mut s = Setup {
        n,
        pad,
        h,
        dt,
        freq,
        period,
        nn,
        ecoef,
        hcoef,
        cell_eps,
        cell_sig,
        cell_pec,
        src_idx: vec![],
        src_amp: vec![],
        src_j: vec![],
        power: 0.0,
        cells_per_wavelength: cpw,
        warnings,
    };

    // ---- ports
    let ports: Vec<(&Vec<String>, f64, usize, &String)> = model
        .loads
        .iter()
        .filter_map(|l| match &l.load {
            Load::Port { faces, power, polarization, profile } => Some((faces, *power, polarization.idx(), profile)),
            _ => None,
        })
        .collect();
    if ports.is_empty() {
        bail!("add an RF port (add_port): the faces where the microwave power enters, e.g. a waveguide opening on the cavity wall");
    }
    let total: f64 = ports.iter().map(|p| p.1.max(0.0)).sum();
    if !(total > 0.0) {
        bail!("port power must be > 0 W");
    }
    s.power = total;
    let mut src: BTreeMap<u32, f64> = BTreeMap::new();
    let mut skipped = 0usize;
    for (faces, power, pol, profile) in ports {
        let fids = mesh.resolve_faces(faces)?;
        let jp = (power.max(0.0) / total).sqrt();
        // tangential axes of the faces; u = across-face axis perpendicular to the polarization
        let mut u_rng = (usize::MAX, 0usize);
        let mut info = Vec::new();
        for &fi in &fids {
            let f = &mesh.faces[fi as usize];
            let (a, plus) = axis_of(f.dir);
            if a == pol {
                bail!("port polarization {} is normal to port face '{}': choose an axis lying in the face", ["x", "y", "z"][pol], faces.join(", "));
            }
            let q = 3 - a - pol;
            let c = mesh.elem_cell[f.elem as usize];
            let cell = [c[0] as usize + pad, c[1] as usize + pad, c[2] as usize + pad];
            u_rng = (u_rng.0.min(cell[q]), u_rng.1.max(cell[q] + 1));
            info.push((a, plus, q, cell));
        }
        for (a, plus, q, cell) in info {
            for dq in 0..2 {
                let mut p = cell;
                p[a] = cell[a] + plus as usize;
                p[q] = cell[q] + dq;
                let w = if profile == "uniform" {
                    1.0
                } else {
                    let span = (u_rng.1 - u_rng.0) as f64;
                    (std::f64::consts::PI * (p[q] - u_rng.0) as f64 / span).sin()
                };
                if w <= 1e-9 {
                    continue;
                }
                // a source on a metal/boundary edge does nothing: step off it, outward first
                let mut placed = None;
                for shift in [0i64, if plus { 1 } else { -1 }, if plus { -1 } else { 1 }] {
                    let pa = p[a] as i64 + shift;
                    if pa < 0 || pa as usize > s.n[a] {
                        continue;
                    }
                    let mut pp = p;
                    pp[a] = pa as usize;
                    let idx = pol * nn + s.node(pp[0], pp[1], pp[2]);
                    if s.ecoef[3 * nn + idx] != 0.0 {
                        placed = Some(idx as u32);
                        break;
                    }
                }
                match placed {
                    Some(idx) => *src.entry(idx).or_insert(0.0) = jp * w,
                    None => skipped += 1,
                }
            }
        }
    }
    if src.is_empty() {
        bail!("the port faces lie entirely on metal: put the port on a face that borders air or a dielectric");
    }
    if skipped > 0 {
        s.warnings.push(format!("{skipped} port edges touch metal and were left out"));
    }
    for (idx, j) in src {
        s.src_idx.push(idx);
        s.src_amp.push((s.ecoef[3 * nn + idx as usize] as f64 * h * j) as f32);
        s.src_j.push(j);
    }
    Ok(s)
}

// ------------------------------------------------------------------ engines

/// Time-stepping backend: CPU or GPU.
trait Engine {
    /// Advance `steps` time steps; accumulate the DFT when `acc`.
    fn run(&mut self, s: &Setup, steps: usize, acc: bool) -> Result<()>;
    fn reset_dft(&mut self) -> Result<()>;
    /// (Σ E cos ωt, Σ E sin ωt) per edge (3·nn each).
    fn dft(&mut self, s: &Setup) -> Result<(Vec<f32>, Vec<f32>)>;
    fn name(&self) -> String;
}

struct Cpu {
    e: Vec<f32>,
    h: Vec<f32>,
    re: Vec<f32>,
    im: Vec<f32>,
    step: u64,
}

impl Cpu {
    fn new(s: &Setup) -> Cpu {
        let z = || vec![0.0f32; 3 * s.nn];
        Cpu { e: z(), h: z(), re: z(), im: z(), step: 0 }
    }
}

fn ramp(step: f64, s: &Setup) -> f64 {
    let x = (step / (RAMP_PERIODS * s.period) as f64).min(1.0);
    0.5 - 0.5 * (std::f64::consts::PI * x).cos()
}

impl Engine for Cpu {
    fn run(&mut self, s: &Setup, steps: usize, acc: bool) -> Result<()> {
        let nn = s.nn;
        let n1 = s.n1();
        let (sy, sz) = (n1[0], n1[0] * n1[1]);
        let coord = |idx: usize| (idx % n1[0], (idx / n1[0]) % n1[1], idx / sz);
        for _ in 0..steps {
            // H^{n+1/2}
            {
                let e = &self.e;
                let (da, db) = s.hcoef.split_at(3 * nn);
                let (hx, rest) = self.h.split_at_mut(nn);
                let (hy, hz) = rest.split_at_mut(nn);
                let (ex, ey, ez) = (&e[..nn], &e[nn..2 * nn], &e[2 * nn..]);
                hx.par_iter_mut().enumerate().for_each(|(idx, v)| {
                    let (_, j, k) = coord(idx);
                    if j + 1 < n1[1] && k + 1 < n1[2] {
                        *v = da[idx] * *v - db[idx] * ((ez[idx + sy] - ez[idx]) - (ey[idx + sz] - ey[idx]));
                    }
                });
                hy.par_iter_mut().enumerate().for_each(|(idx, v)| {
                    let (i, _, k) = coord(idx);
                    if i + 1 < n1[0] && k + 1 < n1[2] {
                        *v = da[nn + idx] * *v - db[nn + idx] * ((ex[idx + sz] - ex[idx]) - (ez[idx + 1] - ez[idx]));
                    }
                });
                hz.par_iter_mut().enumerate().for_each(|(idx, v)| {
                    let (i, j, _) = coord(idx);
                    if i + 1 < n1[0] && j + 1 < n1[1] {
                        *v = da[2 * nn + idx] * *v - db[2 * nn + idx] * ((ey[idx + 1] - ey[idx]) - (ex[idx + sy] - ex[idx]));
                    }
                });
            }
            // E^{n+1} (DFT of E^n)
            {
                let ph = std::f64::consts::TAU * (self.step % s.period as u64) as f64 / s.period as f64;
                let (c, sn) = (ph.cos() as f32, ph.sin() as f32);
                let h = &self.h;
                let (ca, cb) = s.ecoef.split_at(3 * nn);
                let (hx, hy, hz) = (&h[..nn], &h[nn..2 * nn], &h[2 * nn..]);
                self.e.par_chunks_mut(nn).zip(self.re.par_chunks_mut(nn)).zip(self.im.par_chunks_mut(nn)).enumerate().for_each(|(a, ((ea, ra), ia))| {
                    ea.par_iter_mut().zip(ra.par_iter_mut()).zip(ia.par_iter_mut()).enumerate().for_each(|(idx, ((v, r), im))| {
                        let k = a * nn + idx;
                        if cb[k] == 0.0 {
                            return;
                        }
                        let old = *v;
                        if acc {
                            *r += old * c;
                            *im += old * sn;
                        }
                        let curl = match a {
                            0 => (hz[idx] - hz[idx - sy]) - (hy[idx] - hy[idx - sz]),
                            1 => (hx[idx] - hx[idx - sz]) - (hz[idx] - hz[idx - 1]),
                            _ => (hy[idx] - hy[idx - 1]) - (hx[idx] - hx[idx - sy]),
                        };
                        *v = ca[k] * old + cb[k] * curl;
                    });
                });
            }
            // source J at t^{n+1/2}
            let t = self.step as f64 + 0.5;
            let ph = std::f64::consts::TAU * ((self.step % s.period as u64) as f64 + 0.5) / s.period as f64;
            let amp = (ramp(t, s) * ph.sin()) as f32;
            for (q, &idx) in s.src_idx.iter().enumerate() {
                self.e[idx as usize] -= s.src_amp[q] * amp;
            }
            self.step += 1;
        }
        Ok(())
    }
    fn reset_dft(&mut self) -> Result<()> {
        self.re.iter_mut().for_each(|v| *v = 0.0);
        self.im.iter_mut().for_each(|v| *v = 0.0);
        Ok(())
    }
    fn dft(&mut self, _s: &Setup) -> Result<(Vec<f32>, Vec<f32>)> {
        Ok((self.re.clone(), self.im.clone()))
    }
    fn name(&self) -> String {
        format!("FDTD on CPU ({} threads)", rayon::current_num_threads())
    }
}

const SHADER: &str = r#"
struct P { nx1: u32, ny1: u32, nz1: u32, n: u32, period: u32, ramp: u32, nsrc: u32, acc: u32, row: u32, p1: u32, p2: u32, p3: u32 };
@group(0) @binding(0) var<storage, read_write> E: array<f32>;
@group(0) @binding(1) var<storage, read_write> H: array<f32>;
@group(0) @binding(2) var<storage, read> EC: array<f32>;
@group(0) @binding(3) var<storage, read> HC: array<f32>;
@group(0) @binding(4) var<storage, read_write> D: array<f32>;
@group(0) @binding(5) var<storage, read> SI: array<u32>;
@group(0) @binding(6) var<storage, read> SA: array<f32>;
@group(0) @binding(7) var<storage, read_write> ST: array<u32>;
@group(0) @binding(8) var<uniform> p: P;
const TAU: f32 = 6.2831853;

@compute @workgroup_size(256)
fn update_h(@builtin(global_invocation_id) g: vec3<u32>) {
    let idx = g.x + g.y * p.row;
    if (idx >= p.n) { return; }
    let n = p.n;
    let sy = p.nx1;
    let sz = p.nx1 * p.ny1;
    let i = idx % p.nx1;
    let j = (idx / p.nx1) % p.ny1;
    let k = idx / sz;
    if (j + 1u < p.ny1 && k + 1u < p.nz1) {
        H[idx] = HC[idx] * H[idx] - HC[3u * n + idx] * ((E[2u * n + idx + sy] - E[2u * n + idx]) - (E[n + idx + sz] - E[n + idx]));
    }
    if (i + 1u < p.nx1 && k + 1u < p.nz1) {
        H[n + idx] = HC[n + idx] * H[n + idx] - HC[4u * n + idx] * ((E[idx + sz] - E[idx]) - (E[2u * n + idx + 1u] - E[2u * n + idx]));
    }
    if (i + 1u < p.nx1 && j + 1u < p.ny1) {
        H[2u * n + idx] = HC[2u * n + idx] * H[2u * n + idx] - HC[5u * n + idx] * ((E[n + idx + 1u] - E[n + idx]) - (E[idx + sy] - E[idx]));
    }
}

@compute @workgroup_size(256)
fn update_e(@builtin(global_invocation_id) g: vec3<u32>) {
    let idx = g.x + g.y * p.row;
    if (idx >= p.n) { return; }
    let n = p.n;
    let sy = p.nx1;
    let sz = p.nx1 * p.ny1;
    let i = idx % p.nx1;
    let j = (idx / p.nx1) % p.ny1;
    let k = idx / sz;
    let ph = TAU * f32(ST[0] % p.period) / f32(p.period);
    let c = cos(ph);
    let s = sin(ph);
    let acc = p.acc != 0u;
    if (i + 1u < p.nx1 && j >= 1u && j + 1u < p.ny1 && k >= 1u && k + 1u < p.nz1) {
        let cb = EC[3u * n + idx];
        if (cb != 0.0) {
            let e = E[idx];
            if (acc) { D[idx] += e * c; D[3u * n + idx] += e * s; }
            E[idx] = EC[idx] * e + cb * ((H[2u * n + idx] - H[2u * n + idx - sy]) - (H[n + idx] - H[n + idx - sz]));
        }
    }
    if (j + 1u < p.ny1 && i >= 1u && i + 1u < p.nx1 && k >= 1u && k + 1u < p.nz1) {
        let cb = EC[4u * n + idx];
        if (cb != 0.0) {
            let e = E[n + idx];
            if (acc) { D[n + idx] += e * c; D[4u * n + idx] += e * s; }
            E[n + idx] = EC[n + idx] * e + cb * ((H[idx] - H[idx - sz]) - (H[2u * n + idx] - H[2u * n + idx - 1u]));
        }
    }
    if (k + 1u < p.nz1 && i >= 1u && i + 1u < p.nx1 && j >= 1u && j + 1u < p.ny1) {
        let cb = EC[5u * n + idx];
        if (cb != 0.0) {
            let e = E[2u * n + idx];
            if (acc) { D[2u * n + idx] += e * c; D[5u * n + idx] += e * s; }
            E[2u * n + idx] = EC[2u * n + idx] * e + cb * ((H[n + idx] - H[n + idx - 1u]) - (H[idx] - H[idx - sy]));
        }
    }
}

@compute @workgroup_size(64)
fn inject(@builtin(global_invocation_id) g: vec3<u32>) {
    let t = g.x;
    if (t >= p.nsrc) { return; }
    let st = ST[0];
    let ph = TAU * (f32(st % p.period) + 0.5) / f32(p.period);
    let x = min(1.0, (f32(st) + 0.5) / f32(p.ramp));
    let r = 0.5 - 0.5 * cos(3.14159265 * x);
    E[SI[t]] -= SA[t] * r * sin(ph);
}

@compute @workgroup_size(1)
fn advance() {
    ST[0] = ST[0] + 1u;
}
"#;

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipes: [wgpu::ComputePipeline; 4],
    bind: wgpu::BindGroup,
    params: wgpu::Buffer,
    dbuf: wgpu::Buffer,
    staging: wgpu::Buffer,
    groups: (u32, u32),
    params_data: [u32; 12],
    name: String,
    fault: std::sync::Arc<std::sync::Mutex<Option<String>>>,
}

impl Gpu {
    fn new(s: &Setup) -> Result<Gpu> {
        let adapter = crate::gpu::pick_adapter()?;
        let info = adapter.get_info();
        let al = adapter.limits();
        let nn = s.nn;
        let big = (6 * nn * 4) as u64;
        if big > al.max_storage_buffer_binding_size as u64 || big > al.max_buffer_size {
            bail!("FDTD grid ({} MB per buffer) exceeds the GPU's buffer size limit", big / 1_000_000);
        }
        let mut limits = wgpu::Limits::default();
        limits.max_storage_buffers_per_shader_stage = 8;
        limits.max_storage_buffer_binding_size = al.max_storage_buffer_binding_size;
        limits.max_buffer_size = al.max_buffer_size;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("simcraft-fdtd"),
            required_features: wgpu::Features::empty(),
            required_limits: limits,
            experimental_features: Default::default(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: Default::default(),
        }))
        .map_err(|e| anyhow!("could not open GPU device: {e}"))?;
        let fault: std::sync::Arc<std::sync::Mutex<Option<String>>> = Default::default();
        let f2 = fault.clone();
        device.on_uncaptured_error(std::sync::Arc::new(move |e: wgpu::Error| {
            *f2.lock().unwrap() = Some(e.to_string());
        }));
        let oom = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let val = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("fdtd"), source: wgpu::ShaderSource::Wgsl(SHADER.into()) });
        let entry = |binding: u32, ty: wgpu::BufferBindingType| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let rw = wgpu::BufferBindingType::Storage { read_only: false };
        let ro = wgpu::BufferBindingType::Storage { read_only: true };
        let entries = [entry(0, rw), entry(1, rw), entry(2, ro), entry(3, ro), entry(4, rw), entry(5, ro), entry(6, ro), entry(7, rw), entry(8, wgpu::BufferBindingType::Uniform)];
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &entries });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(&bgl)], immediate_size: 0 });
        let mk = |e: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some(e), layout: Some(&pl), module: &module, entry_point: Some(e), compilation_options: Default::default(), cache: None })
        };
        let pipes = [mk("update_h"), mk("update_e"), mk("inject"), mk("advance")];
        let buf = |label: &str, size: u64, usage: wgpu::BufferUsages| device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: size.max(16).div_ceil(4) * 4, usage, mapped_at_creation: false });
        let st = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC;
        let f = (3 * nn * 4) as u64;
        let e = buf("E", f, st);
        let h = buf("H", f, st);
        let ec = buf("EC", 2 * f, st);
        let hc = buf("HC", 2 * f, st);
        let d = buf("D", 2 * f, st);
        let si = buf("SI", (s.src_idx.len() * 4) as u64, st);
        let sa = buf("SA", (s.src_amp.len() * 4) as u64, st);
        let stp = buf("ST", 16, st);
        let params = buf("params", 48, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);
        let staging = buf("staging", 2 * f, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST);
        queue.write_buffer(&ec, 0, bytemuck::cast_slice(&s.ecoef));
        queue.write_buffer(&hc, 0, bytemuck::cast_slice(&s.hcoef));
        queue.write_buffer(&si, 0, bytemuck::cast_slice(&s.src_idx));
        queue.write_buffer(&sa, 0, bytemuck::cast_slice(&s.src_amp));
        let nwg = (nn as u32).div_ceil(256);
        let gx = nwg.min(32768);
        let gy = nwg.div_ceil(gx);
        let n1 = s.n1();
        let params_data = [n1[0] as u32, n1[1] as u32, n1[2] as u32, nn as u32, s.period as u32, (RAMP_PERIODS * s.period) as u32, s.src_idx.len() as u32, 0, gx * 256, 0, 0, 0];
        queue.write_buffer(&params, 0, bytemuck::cast_slice(&params_data));
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[&e, &h, &ec, &hc, &d, &si, &sa, &stp, &params].iter().enumerate().map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() }).collect::<Vec<_>>(),
        });
        if let Some(err) = pollster::block_on(val.pop()) {
            bail!("GPU setup failed: {err}");
        }
        if let Some(err) = pollster::block_on(oom.pop()) {
            bail!("not enough free GPU memory for the FDTD grid ({} MB): {err}", (f * 9) / 1_000_000);
        }
        *crate::gpu::LAST_DEVICE.lock().unwrap() = Some(info.name.clone());
        Ok(Gpu { device, queue, pipes, bind, params, dbuf: d, staging, groups: (gx, gy), params_data, name: info.name, fault })
    }
    fn check(&self) -> Result<()> {
        if let Some(e) = self.fault.lock().unwrap().take() {
            bail!("GPU error during FDTD: {e}");
        }
        Ok(())
    }
}

impl Engine for Gpu {
    fn run(&mut self, s: &Setup, steps: usize, acc: bool) -> Result<()> {
        if self.params_data[7] != acc as u32 {
            self.params_data[7] = acc as u32;
            self.queue.write_buffer(&self.params, 0, bytemuck::cast_slice(&self.params_data));
        }
        let nsrc_groups = (s.src_idx.len() as u32).div_ceil(64).max(1);
        let mut left = steps;
        while left > 0 {
            let chunk = left.min(200);
            let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
                pass.set_bind_group(0, &self.bind, &[]);
                for _ in 0..chunk {
                    pass.set_pipeline(&self.pipes[0]);
                    pass.dispatch_workgroups(self.groups.0, self.groups.1, 1);
                    pass.set_pipeline(&self.pipes[1]);
                    pass.dispatch_workgroups(self.groups.0, self.groups.1, 1);
                    pass.set_pipeline(&self.pipes[2]);
                    pass.dispatch_workgroups(nsrc_groups, 1, 1);
                    pass.set_pipeline(&self.pipes[3]);
                    pass.dispatch_workgroups(1, 1, 1);
                }
            }
            self.queue.submit(Some(enc.finish()));
            self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).map_err(|e| anyhow!("GPU poll failed: {e:?}"))?;
            self.check()?;
            left -= chunk;
        }
        Ok(())
    }
    fn reset_dft(&mut self) -> Result<()> {
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.clear_buffer(&self.dbuf, 0, None);
        self.queue.submit(Some(enc.finish()));
        Ok(())
    }
    fn dft(&mut self, s: &Setup) -> Result<(Vec<f32>, Vec<f32>)> {
        let bytes = (6 * s.nn * 4) as u64;
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        enc.copy_buffer_to_buffer(&self.dbuf, 0, &self.staging, 0, bytes);
        self.queue.submit(Some(enc.finish()));
        let slice = self.staging.slice(..bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).map_err(|e| anyhow!("GPU poll failed: {e:?}"))?;
        rx.recv().map_err(|_| anyhow!("GPU map callback lost"))?.map_err(|e| anyhow!("GPU buffer map failed: {e:?}"))?;
        let all: Vec<f32> = {
            let data = slice.get_mapped_range().map_err(|e| anyhow!("GPU buffer view failed: {e:?}"))?;
            bytemuck::cast_slice::<u8, f32>(&data[..bytes as usize]).to_vec()
        };
        self.staging.unmap();
        self.check()?;
        let im = all[3 * s.nn..].to_vec();
        let mut re = all;
        re.truncate(3 * s.nn);
        Ok((re, im))
    }
    fn name(&self) -> String {
        format!("FDTD on GPU ({})", self.name)
    }
}

// ------------------------------------------------------------------ steady state + post-processing

/// Steady-state phasors of every E edge (unnormalised: source current amplitude ≈ 1 A/m²).
pub struct Solution {
    /// Ê = re − i·im per edge (3·nn each), already scaled by 2/M.
    pub re: Vec<f32>,
    pub im: Vec<f32>,
    pub periods: usize,
    pub change: f64,
    pub converged: bool,
    pub engine: String,
    pub fallback: Option<String>,
}

fn use_gpu(model: &Model, s: &Setup) -> bool {
    use crate::gpu::ComputeMode::*;
    match model.analysis.compute {
        Cpu => false,
        Gpu => true,
        Auto => s.cells() >= 150_000,
    }
}

/// Delivered and absorbed power (raw units) from DFT sums over `m` steps.
fn powers(s: &Setup, re: &[f32], im: &[f32], m: usize) -> (f64, f64) {
    let k = 2.0 / m as f64;
    let h3 = s.h.powi(3);
    let pin: f64 = s.src_idx.iter().zip(&s.src_j).map(|(&i, &j)| 0.5 * j * h3 * (-k * im[i as usize] as f64)).sum();
    let pabs: f64 = cell_power(s, re, im, k).iter().sum::<f64>() * h3;
    (pin, pabs)
}

/// Absorbed power density per cell (raw units / m³) in the real domain (0 in the padding).
fn cell_power(s: &Setup, re: &[f32], im: &[f32], k: f64) -> Vec<f64> {
    let nn = s.nn;
    let n = s.n;
    let pad = s.pad;
    (0..s.cells())
        .into_par_iter()
        .map(|c| {
            let sig = s.cell_sig[c];
            if sig == 0.0 || s.cell_pec[c] {
                return 0.0;
            }
            let (i, j, kk) = (c % n[0], (c / n[0]) % n[1], c / (n[0] * n[1]));
            if i < pad || j < pad || kk < pad || i >= n[0] - pad || j >= n[1] - pad || kk >= n[2] - pad {
                return 0.0;
            }
            let mut e2 = 0.0;
            for (a, (da, db, dc)) in [(0usize, (0, 1, 1)), (1, (1, 0, 1)), (2, (1, 1, 0))] {
                for x in 0..=da {
                    for y in 0..=db {
                        for z in 0..=dc {
                            let idx = a * nn + s.node(i + x, j + y, kk + z);
                            e2 += (re[idx] as f64).powi(2) + (im[idx] as f64).powi(2);
                        }
                    }
                }
            }
            // ½σ|E|², each edge shared by 4 cells
            0.5 * sig * e2 * k * k / 4.0
        })
        .collect()
}

/// Peak |E| at a cell centre (raw units).
fn cell_e(s: &Setup, re: &[f32], im: &[f32], k: f64, c: usize) -> f64 {
    let nn = s.nn;
    let n = s.n;
    let (i, j, kk) = (c % n[0], (c / n[0]) % n[1], c / (n[0] * n[1]));
    let mut tot = 0.0;
    for (a, (da, db, dc)) in [(0usize, (0, 1, 1)), (1, (1, 0, 1)), (2, (1, 1, 0))] {
        let (mut sr, mut si) = (0.0, 0.0);
        for x in 0..=da {
            for y in 0..=db {
                for z in 0..=dc {
                    let idx = a * nn + s.node(i + x, j + y, kk + z);
                    sr += re[idx] as f64;
                    si += im[idx] as f64;
                }
            }
        }
        tot += (sr / 4.0).powi(2) + (si / 4.0).powi(2);
    }
    tot.sqrt() * k
}

pub fn solve_steady(model: &Model, s: &Setup, prog: Prog) -> Result<Solution> {
    let mut fallback = None;
    let mut engine: Box<dyn Engine> = if use_gpu(model, s) {
        match Gpu::new(s) {
            Ok(g) => Box::new(g),
            Err(e) => {
                let msg = format!("GPU unavailable for FDTD ({e:#}); ran on the CPU");
                crate::gpu::note_fallback(msg.clone());
                fallback = Some(msg);
                Box::new(Cpu::new(s))
            }
        }
    } else {
        Box::new(Cpu::new(s))
    };
    let max_periods = model.analysis.max_periods.clamp(RAMP_PERIODS + 2 * BLOCK_PERIODS, 20_000);
    let block = BLOCK_PERIODS * s.period;
    prog(0.02, "ramping up the source");
    engine.run(s, RAMP_PERIODS * s.period, false)?;
    let mut periods = RAMP_PERIODS;
    let mut prev: Option<(f64, f64)> = None;
    let mut stable = 0;
    let mut change = 1.0;
    let (re, im) = loop {
        engine.reset_dft()?;
        engine.run(s, block, true)?;
        periods += BLOCK_PERIODS;
        let tb = std::time::Instant::now();
        let (re, im) = engine.dft(s)?;
        let (pin, pabs) = powers(s, &re, &im, block);
        if std::env::var_os("SIMCRAFT_PROFILE").is_some() {
            eprintln!("[fdtd] block at period {periods}: readback+powers {:.3} s", tb.elapsed().as_secs_f64());
        }
        if !pin.is_finite() || !pabs.is_finite() {
            bail!("the FDTD solution became unstable (non-finite fields)");
        }
        if let Some((p0, a0)) = prev {
            let scale = pin.abs().max(1e-300);
            change = ((pin - p0).abs() / scale).max((pabs - a0).abs() / pabs.abs().max(1e-3 * scale));
            stable = if change < TOL { stable + 1 } else { 0 };
        }
        prev = Some((pin, pabs));
        prog(0.02 + 0.93 * (periods as f64 / max_periods as f64).min(1.0), &format!("{} · period {periods}, change {:.2}%", engine.name(), change * 100.0));
        if stable >= 2 || periods + BLOCK_PERIODS > max_periods {
            break (re, im);
        }
    };
    let k = 2.0 / block as f64;
    let re: Vec<f32> = re.iter().map(|v| (*v as f64 * k) as f32).collect();
    let im: Vec<f32> = im.iter().map(|v| (*v as f64 * k) as f32).collect();
    Ok(Solution { re, im, periods, change, converged: stable >= 2, engine: engine.name(), fallback })
}

/// Full-wave solve; fills E, absorbed power and SAR fields and quantities. Returns the absorbed
/// power density per mesh element [W/m³].
pub fn full_wave(model: &Model, mesh: &Mesh, prog: Prog, res: &mut Results) -> Result<Vec<f64>> {
    let t0 = std::time::Instant::now();
    prog(0.0, "building the FDTD grid");
    let s = setup(model, mesh)?;
    if std::env::var_os("SIMCRAFT_PROFILE").is_some() {
        eprintln!("[fdtd] setup {:.2} s", t0.elapsed().as_secs_f64());
    }
    res.warnings.extend(s.warnings.iter().cloned());
    let sol = solve_steady(model, &s, prog)?;
    if std::env::var_os("SIMCRAFT_PROFILE").is_some() {
        eprintln!("[fdtd] time stepping done {:.2} s", t0.elapsed().as_secs_f64());
    }
    let (pin, pabs) = powers(&s, &sol.re, &sol.im, 2);
    if !(pin > 0.0) || pin < 1e-12 * pabs.abs() {
        bail!("the port delivers no power into this model (closed lossless cavity or a port facing metal): add a lossy material such as food or water, or use em_boundary = open");
    }
    let scale2 = s.power / pin;
    let scale = scale2.sqrt();
    let cp = cell_power(&s, &sol.re, &sol.im, 1.0);
    let ne = mesh.elems.len();
    let pad = s.pad;
    let cell_of = |e: usize| {
        let c = mesh.elem_cell[e];
        s.cell(c[0] as usize + pad, c[1] as usize + pad, c[2] as usize + pad)
    };
    let q: Vec<f64> = (0..ne).map(|e| cp[cell_of(e)] * scale2).collect();
    let emag: Vec<f64> = (0..ne).into_par_iter().map(|e| cell_e(&s, &sol.re, &sol.im, scale, cell_of(e))).collect();

    // nodal fields
    let nn = mesh.nodes.len();
    let (mut e_acc, mut e_cnt) = (vec![0.0; nn], vec![0u32; nn]);
    let (mut q_acc, mut sar_acc, mut q_cnt) = (vec![0.0; nn], vec![0.0; nn], vec![0u32; nn]);
    for (e, en) in mesh.elems.iter().enumerate() {
        let lossy = s.cell_sig[cell_of(e)] > 0.0 && !s.cell_pec[cell_of(e)];
        let rho = mesh.materials[mesh.elem_mat[e] as usize].density.max(1e-9);
        for &n in en {
            let n = n as usize;
            e_acc[n] += emag[e];
            e_cnt[n] += 1;
            if lossy {
                q_acc[n] += q[e];
                sar_acc[n] += q[e] / rho;
                q_cnt[n] += 1;
            }
        }
    }
    let avg = |a: Vec<f64>, c: &[u32]| a.into_iter().zip(c).map(|(v, &k)| if k > 0 { v / k as f64 } else { 0.0 }).collect::<Vec<f64>>();
    res.extra.insert(Field::ElectricField, avg(e_acc, &e_cnt));
    res.extra.insert(Field::AbsorbedPower, avg(q_acc, &q_cnt));
    res.extra.insert(Field::Sar, avg(sar_acc, &q_cnt));

    // powers per body
    let v = (mesh.h * 1e-3).powi(3);
    let mut per_body: BTreeMap<String, f64> = BTreeMap::new();
    for e in 0..ne {
        if q[e] > 0.0 {
            let name = mesh.body_names.get(mesh.elem_body[e] as usize).cloned().unwrap_or_else(|| "?".into());
            *per_body.entry(name).or_insert(0.0) += q[e] * v;
        }
    }
    let total_abs = pabs * scale2;
    let lambda0 = C0 / s.freq;
    res.quantity("Frequency", s.freq / 1e9, "GHz");
    res.quantity("Wavelength in air", lambda0 * 1e3, "mm");
    res.quantity("Cells per wavelength (densest material)", s.cells_per_wavelength, "");
    res.quantity("Input power", s.power, "W");
    res.quantity("Absorbed power", total_abs, "W");
    res.quantity("Absorbed fraction", 100.0 * total_abs / s.power, "%");
    for (b, p) in &per_body {
        res.quantity(&format!("Absorbed in {b}"), *p, "W");
    }
    res.quantity("Peak |E|", emag.iter().cloned().fold(0.0, f64::max), "V/m");
    res.quantity("RF periods simulated", sol.periods as f64, "");
    res.quantity("Time step", s.dt * 1e12, "ps");
    res.quantity("FDTD cells", s.cells() as f64, "");
    if !sol.converged {
        res.warnings.push(format!(
            "did not reach a steady state within {} RF periods (last change {:.1}%): the cavity is weakly damped; raise max_periods or check that a lossy load is present",
            sol.periods,
            sol.change * 100.0
        ));
    }
    if s.pad == 0 && total_abs < 0.97 * s.power {
        res.warnings.push(format!("power balance: only {:.1}% of the input is absorbed in a closed cavity (expected ~100%) — results may not be converged", 100.0 * total_abs / s.power));
    }
    if let Some(f) = &sol.fallback {
        res.warnings.push(f.clone());
    }
    res.note = format!("{:.3} GHz steady state, {} W delivered; |E| is the peak amplitude", s.freq / 1e9, crate::results::float(s.power));
    res.info.solver = sol.engine.clone();
    res.info.dofs = 6 * s.cells();
    res.info.iterations = sol.periods * s.period;
    res.info.linear_solve_seconds = t0.elapsed().as_secs_f64();
    Ok(q)
}

/// Microwave heating: full-wave EM, then transient conduction in the solids with the absorbed power.
pub fn microwave_heating(model: &Model, mesh: &Mesh, prog: Prog, res: &mut Results) -> Result<()> {
    let q = full_wave(model, mesh, &|f, m| prog(0.7 * f, m), res)?;
    let fdtd = res.info.solver.clone();
    let periods = res.info.iterations;
    let sel: Vec<bool> = mesh.elem_region.iter().map(|r| *r == REGION_SOLID).collect();
    let qs: Vec<f64> = q.iter().zip(&sel).map(|(v, s)| if *s { *v } else { 0.0 }).collect();
    let v = (mesh.h * 1e-3).powi(3);
    let heat: f64 = qs.iter().sum::<f64>() * v;
    if heat <= 0.0 {
        res.warnings.push("no power is absorbed in the solid bodies: check their materials (loss tangent) and the port".into());
    }
    crate::dynamics::transient_thermal_sel(model, mesh, Some(&qs), Some(&sel), &|f, m| prog(0.7 + 0.3 * f, m), res)?;
    res.info.solver = format!("{fdtd} + {}", res.info.solver);
    res.info.iterations = periods;
    res.quantity("Heating power in solids", heat, "W");
    res.quantity("Energy absorbed", heat * model.analysis.end_time / 1000.0, "kJ");
    // temperatures of heated bodies at the end
    if let Some(t) = res.temp.clone() {
        let mut per: BTreeMap<String, (f64, f64, f64, f64)> = BTreeMap::new();
        for (e, en) in mesh.elems.iter().enumerate().filter(|(e, _)| sel[*e]) {
            let name = mesh.body_names.get(mesh.elem_body[e] as usize).cloned().unwrap_or_default();
            let te = en.iter().map(|&n| t[n as usize]).sum::<f64>() / 8.0;
            let x = per.entry(name).or_insert((0.0, 0.0, f64::INFINITY, f64::NEG_INFINITY));
            x.0 += te;
            x.1 += 1.0;
            x.2 = x.2.min(te);
            x.3 = x.3.max(te);
        }
        let mut spread: f64 = 0.0;
        for (b, (sum, cnt, lo, hi)) in per {
            res.quantity(&format!("Mean temperature of {b}"), sum / cnt, "°C");
            res.quantity(&format!("Coldest / hottest spot in {b}"), lo, &format!("°C / {hi:.1} °C"));
            spread = spread.max(hi - lo);
        }
        res.quantity("Temperature spread", spread, "K");
        let watery = mesh.materials.iter().any(|m| m.relative_permittivity > 20.0 && m.loss_tangent > 0.0);
        let tmax = t.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        if watery && tmax > 100.0 {
            res.warnings.push(format!(
                "hot spots reach {tmax:.0} °C: water-rich material would boil there (evaporation and changing dielectric properties are not modelled), so real peaks stay near 100 °C — use the pattern to judge uneven heating, or shorten the heating time"
            ));
        }
    }
    res.note = format!("{} — temperatures after {} s of heating", res.note, crate::results::float(model.analysis.end_time));
    Ok(())
}
