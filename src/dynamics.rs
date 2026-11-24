//! Time-dependent and nonlinear structural / thermal solvers.
//!
//! * Transient thermal: implicit Euler with lumped heat capacity.
//! * Transient structural: Newmark average-acceleration with lumped mass and Rayleigh damping
//!   (tuned to the requested damping ratio at the first natural frequency); step loading.
//! * Linear buckling: geometric stiffness from the linear pre-stress, subspace iteration for the
//!   lowest positive load multipliers.
//! * Nonlinear structural: J2 (von Mises) plasticity with bilinear isotropic hardening, load
//!   stepping with initial-stiffness Newton iterations (one factorisation).

use crate::fem::*;
use crate::geometry::*;
use crate::model::*;
use crate::results::*;
use crate::sparse::*;
use anyhow::{Result, bail};
use rayon::prelude::*;

fn frame_stride(steps: usize) -> usize {
    (steps / 60).max(1)
}

/// Dirichlet values as a vector (0 for free dofs) and the column-correction K·g.
fn dirichlet_correction(k: &NodeSparse, fixed: &[Option<f64>]) -> Vec<f64> {
    let g: Vec<f64> = fixed.iter().map(|v| v.unwrap_or(0.0)).collect();
    let mut corr = vec![0.0; g.len()];
    if g.iter().any(|v| *v != 0.0) {
        k.matvec(&g, &mut corr);
    }
    corr
}

fn apply_rhs_bc(rhs: &mut [f64], corr: &[f64], fixed: &[Option<f64>]) {
    for i in 0..rhs.len() {
        match fixed[i] {
            Some(g) => rhs[i] = g,
            None => rhs[i] -= corr[i],
        }
    }
}

// ------------------------------------------------------------------ transient thermal

pub fn transient_thermal(model: &Model, mesh: &crate::mesh::Mesh, prog: Prog, res: &mut Results) -> Result<()> {
    transient_thermal_with(model, mesh, None, prog, res)
}

/// Transient thermal with an extra volumetric heat source per element [W/m³]
/// (used by coupled analyses such as electromagnetic heating).
pub fn transient_thermal_with(model: &Model, mesh: &crate::mesh::Mesh, elem_q: Option<&[f64]>, prog: Prog, res: &mut Results) -> Result<()> {
    transient_thermal_sel(model, mesh, elem_q, None, prog, res)
}

/// Transient thermal on the selected elements only (`sel`); other nodes stay at the initial temperature.
pub fn transient_thermal_sel(model: &Model, mesh: &crate::mesh::Mesh, elem_q: Option<&[f64]>, sel: Option<&[bool]>, prog: Prog, res: &mut Results) -> Result<()> {
    let nn = mesh.nodes.len();
    let st = &model.analysis;
    let steps = st.time_steps.clamp(1, 5000);
    let dt = st.end_time / steps as f64;
    if !(dt > 0.0) {
        bail!("end time must be positive");
    }
    prog(0.0, "assembling");
    let tprof = std::time::Instant::now();
    let profile = std::env::var_os("SIMCRAFT_PROFILE").is_some();
    let sys = crate::fem::assemble_thermal_sel(model, mesh, elem_q, sel.map(|s| (s, st.initial_temperature)))?;
    let mut cap = vec![0.0; nn];
    for (e, en) in mesh.elems.iter().enumerate() {
        if sel.is_some_and(|s| !s[e]) {
            continue;
        }
        let m = &mesh.materials[mesh.elem_mat[e] as usize];
        let nv = crate::fem::node_volumes(mesh, e);
        for (a, &n) in en.iter().enumerate() {
            cap[n as usize] += m.density * m.specific_heat * nv[a];
        }
    }
    let mut a = sys.k.clone();
    for (i, c) in cap.iter().enumerate() {
        a.add_diag(i, c / dt);
    }
    let corr = dirichlet_correction(&a, &sys.fixed);
    let mut dummy = vec![0.0; nn];
    a.apply_dirichlet(&sys.fixed, &mut dummy);
    if profile {
        eprintln!("[thermal] assembled {:.2} s", tprof.elapsed().as_secs_f64());
    }
    prog(0.02, "factorising");
    let solver = Solver::new(&a)?;
    if profile {
        eprintln!("[thermal] solver ready ({}) {:.2} s", solver.name(), tprof.elapsed().as_secs_f64());
    }
    res.info.solver = solver.name().to_string();
    res.info.dofs = nn;
    let mut t: Vec<f64> = sys.fixed.iter().map(|g| g.unwrap_or(st.initial_temperature)).collect();
    let stride = frame_stride(steps);
    let tmax = |t: &[f64]| t.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    res.frames.push(TimeFrame { time: 0.0, disp: None, temp: Some(t.clone()) });
    res.history.push((0.0, tmax(&t)));
    for s in 1..=steps {
        let mut rhs: Vec<f64> = (0..nn).map(|i| sys.f[i] + cap[i] / dt * t[i]).collect();
        apply_rhs_bc(&mut rhs, &corr, &sys.fixed);
        let mut tn = t.clone();
        res.info.iterations += solver.solve(&a, &rhs, &mut tn, 1e-10, &|_| {})?;
        t = tn;
        let time = s as f64 * dt;
        res.history.push((time, tmax(&t)));
        if s % stride == 0 || s == steps {
            res.frames.push(TimeFrame { time, disp: None, temp: Some(t.clone()) });
        }
        if s % 5 == 0 {
            prog(0.05 + 0.9 * s as f64 / steps as f64, &format!("time step {s}/{steps}"));
        }
    }
    res.history_label = "max temperature [°C]".into();
    if profile {
        eprintln!("[thermal] stepped {:.2} s", tprof.elapsed().as_secs_f64());
    }
    thermal_post(mesh, &t, None, res);
    if profile {
        eprintln!("[thermal] post {:.2} s", tprof.elapsed().as_secs_f64());
    }
    res.temp = Some(t);
    prog(1.0, "done");
    Ok(())
}

// ------------------------------------------------------------------ transient structural

fn lumped_mass(mesh: &crate::mesh::Mesh) -> Vec<f64> {
    let mut m = vec![0.0; mesh.nodes.len() * 3];
    for (e, en) in mesh.elems.iter().enumerate() {
        let rho = mesh.materials[mesh.elem_mat[e] as usize].density;
        let nv = crate::fem::node_volumes(mesh, e);
        for (a, &n) in en.iter().enumerate() {
            for c in 0..3 {
                m[n as usize * 3 + c] += rho * nv[a];
            }
        }
    }
    m
}

/// Estimate the first natural circular frequency by inverse iteration (K must have BCs applied).
fn first_omega(k: &NodeSparse, solver: &Solver, m: &[f64], fixed: &[Option<f64>]) -> Result<f64> {
    let n = m.len();
    let mut x: Vec<f64> = (0..n).map(|i| if fixed[i].is_some() { 0.0 } else { 1.0 + ((i * 7919) % 13) as f64 * 0.01 }).collect();
    let mut lam = 0.0;
    for _ in 0..25 {
        let rhs: Vec<f64> = x.iter().zip(m).map(|(a, b)| a * b).collect();
        let mut y = vec![0.0; n];
        solver.solve(k, &rhs, &mut y, 1e-8, &|_| {})?;
        let mut ky = vec![0.0; n];
        k.matvec(&y, &mut ky);
        let num: f64 = y.iter().zip(&ky).enumerate().filter(|(i, _)| fixed[*i].is_none()).map(|(_, (a, b))| a * b).sum();
        let den: f64 = y.iter().zip(m).map(|(a, b)| a * a * b).sum();
        lam = num / den.max(1e-300);
        let s = den.sqrt().max(1e-300);
        x = y.iter().map(|v| v / s).collect();
    }
    Ok(lam.max(0.0).sqrt())
}

pub fn transient_structural(model: &Model, mesh: &crate::mesh::Mesh, prog: Prog, res: &mut Results) -> Result<()> {
    let nn = mesh.nodes.len();
    let n = nn * 3;
    let st = &model.analysis;
    let steps = st.time_steps.clamp(1, 5000);
    let dt = st.end_time / steps as f64;
    if !(dt > 0.0) {
        bail!("end time must be positive");
    }
    let (asm, units) = assemble_structural(model, mesh, prog)?;
    let Assembled { k: k0, fixed, support_nodes, contact, owner } = asm;
    if support_nodes.is_empty() {
        bail!("no supports defined: add a Fixed Support so the part cannot fly away");
    }
    if !contact.is_empty() {
        res.warnings.push("compression-only supports are treated as fixed in transient analyses".into());
    }
    let f_ext = external_forces(model, mesh, None)?;
    let m = lumped_mass(mesh);

    // first frequency for Rayleigh damping
    prog(0.02, "estimating first natural frequency");
    let zero_fixed: Vec<Option<f64>> = fixed.iter().map(|g| g.map(|_| 0.0)).collect();
    let mut kz = k0.clone();
    let mut dummy = vec![0.0; n];
    kz.apply_dirichlet(&zero_fixed, &mut dummy);
    let sz = Solver::new(&kz)?;
    let w1 = first_omega(&kz, &sz, &m, &zero_fixed)?;
    drop(sz);
    let zeta = st.damping_ratio.max(0.0);
    let (alpha, beta) = if w1 > 0.0 { (zeta * w1, zeta / w1) } else { (0.0, 0.0) };
    let f1 = w1 / (2.0 * std::f64::consts::PI);
    res.quantity("First natural frequency (estimate)", f1, "Hz");
    if f1 > 0.0 && dt > 0.1 / f1 {
        res.warnings.push(format!("time step {dt:.3e} s is coarse compared with the first vibration period {:.3e} s; use more time steps for accurate dynamics", 1.0 / f1));
    }

    // Newmark (average acceleration)
    let (a0, a1, a2) = (4.0 / (dt * dt), 2.0 / dt, 4.0 / dt);
    let mut keff = k0.clone();
    for v in keff.vals.iter_mut() {
        *v *= 1.0 + a1 * beta;
    }
    for i in 0..n {
        keff.add_diag(i, (a0 + a1 * alpha) * m[i]);
    }
    let corr = dirichlet_correction(&keff, &fixed);
    keff.apply_dirichlet(&fixed, &mut dummy);
    prog(0.05, "factorising");
    let solver = Solver::new(&keff)?;
    res.info.solver = solver.name().to_string();
    res.info.dofs = n;

    let mut u = vec![0.0; n];
    let mut v = vec![0.0; n];
    let mut acc: Vec<f64> = (0..n).map(|i| if fixed[i].is_some() || m[i] <= 0.0 { 0.0 } else { f_ext[i] / m[i] }).collect();
    let stride = frame_stride(steps);
    let to_mm = |u: &[f64]| -> Vec<V3> { (0..nn).map(|q| [u[q * 3] * 1e3, u[q * 3 + 1] * 1e3, u[q * 3 + 2] * 1e3]).collect() };
    let umax = |u: &[f64]| (0..nn).map(|q| (u[q * 3].powi(2) + u[q * 3 + 1].powi(2) + u[q * 3 + 2].powi(2)).sqrt()).fold(0.0, f64::max) * 1e3;
    res.frames.push(TimeFrame { time: 0.0, disp: Some(to_mm(&u)), temp: None });
    res.history.push((0.0, 0.0));
    let mut peak = (0.0, 0.0);
    let mut tmp = vec![0.0; n];
    for s in 1..=steps {
        // F_eff = F + M(a0 u + a2 v + acc) + C(a1 u + v),  C = αM + βK
        let cvec: Vec<f64> = (0..n).map(|i| a1 * u[i] + v[i]).collect();
        k0.matvec(&cvec, &mut tmp);
        let mut rhs: Vec<f64> = (0..n).map(|i| f_ext[i] + m[i] * (a0 * u[i] + a2 * v[i] + acc[i]) + alpha * m[i] * cvec[i] + beta * tmp[i]).collect();
        apply_rhs_bc(&mut rhs, &corr, &fixed);
        let mut un = u.clone();
        res.info.iterations += solver.solve(&keff, &rhs, &mut un, 1e-9, &|_| {})?;
        for i in 0..n {
            if fixed[i].is_some() {
                acc[i] = 0.0;
                v[i] = 0.0;
                continue;
            }
            let an = a0 * (un[i] - u[i]) - a2 * v[i] - acc[i];
            v[i] += dt / 2.0 * (acc[i] + an);
            acc[i] = an;
        }
        u = un;
        let time = s as f64 * dt;
        let um = umax(&u);
        if um > peak.1 {
            peak = (time, um);
        }
        res.history.push((time, um));
        if s % stride == 0 || s == steps {
            res.frames.push(TimeFrame { time, disp: Some(to_mm(&u)), temp: None });
        }
        if s % 5 == 0 {
            prog(0.1 + 0.8 * s as f64 / steps as f64, &format!("time step {s}/{steps}"));
        }
    }
    res.history_label = "max displacement [mm]".into();
    res.quantity("Peak displacement", peak.1, "mm");
    res.quantity("Time of peak displacement", peak.0, "s");
    prog(0.92, "computing stresses");
    structural_post(model, mesh, &u, None, &units, &fixed, &f_ext, &support_nodes, &owner, res);
    res.disp = Some(to_mm(&u));
    prog(1.0, "done");
    Ok(())
}

// ------------------------------------------------------------------ shared: strain operator at Gauss points

/// B̃ = Bu + Ba·r at the 8 Gauss points for the unit cube (6×24, stored column-wise).
fn btilde(eu: &ElemUnit) -> Vec<Vec<[f64; 6]>> {
    gauss_points()
        .into_iter()
        .map(|x| {
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
            (0..24)
                .map(|j| {
                    let mut col = bu[j];
                    for (k, bk) in ba.iter().enumerate() {
                        let r = eu.r[k * 24 + j];
                        for q in 0..6 {
                            col[q] += bk[q] * r;
                        }
                    }
                    col
                })
                .collect()
        })
        .collect()
}

fn principal(s: &[f64; 6]) -> (f64, f64) {
    let m = [s[0], s[3], s[5], s[3], s[1], s[4], s[5], s[4], s[2]];
    let (ev, _) = sym_eigen(&m, 3);
    (ev.iter().cloned().fold(f64::NEG_INFINITY, f64::max), ev.iter().cloned().fold(f64::INFINITY, f64::min))
}

fn von_mises(s: &[f64; 6]) -> f64 {
    (0.5 * ((s[0] - s[1]).powi(2) + (s[1] - s[2]).powi(2) + (s[2] - s[0]).powi(2)) + 3.0 * (s[3] * s[3] + s[4] * s[4] + s[5] * s[5])).sqrt()
}

// ------------------------------------------------------------------ linear buckling

pub fn buckling(model: &Model, mesh: &crate::mesh::Mesh, prog: Prog, res: &mut Results) -> Result<()> {
    let h = mesh.h * 1e-3;
    let nn = mesh.nodes.len();
    let n = nn * 3;
    let (asm, units) = assemble_structural(model, mesh, prog)?;
    let Assembled { k: k0, fixed, support_nodes, owner, .. } = asm;
    if support_nodes.is_empty() {
        bail!("no supports defined: add a Fixed Support");
    }
    let f_ext = external_forces(model, mesh, None)?;
    if f_ext.iter().all(|v| *v == 0.0) && fixed.iter().all(|g| g.is_none_or(|v| v == 0.0)) {
        bail!("linear buckling needs a load (e.g. a compressive Force) to scale");
    }
    // 1. linear pre-stress
    let mut k = k0.clone();
    let mut f = f_ext.clone();
    k.apply_dirichlet(&fixed, &mut f);
    prog(0.05, "pre-stress: factorising");
    let solver = Solver::new(&k)?;
    res.info.solver = solver.name().to_string();
    let mut u: Vec<f64> = fixed.iter().map(|g| g.unwrap_or(0.0)).collect();
    solver.solve(&k, &f, &mut u, 1e-9, &|_| {})?;
    structural_post(model, mesh, &u, None, &units, &fixed, &f_ext, &support_nodes, &owner, res);
    res.disp = Some((0..nn).map(|q| [u[q * 3] * 1e3, u[q * 3 + 1] * 1e3, u[q * 3 + 2] * 1e3]).collect());

    // 2. geometric stiffness
    prog(0.2, "assembling geometric stiffness");
    let bts: Vec<Vec<Vec<[f64; 6]>>> = units.iter().map(btilde).collect();
    let gps = gauss_points();
    let kg_elems: Vec<Vec<f64>> = mesh
        .elems
        .par_iter()
        .enumerate()
        .map(|(e, en)| {
            let mi = mesh.elem_mat[e] as usize;
            let m = &mesh.materials[mi];
            let d = dmat(m.poisson_ratio);
            let epa = m.youngs_modulus_gpa * 1e9;
            let ue: Vec<f64> = (0..24).map(|i| u[en[i / 3] as usize * 3 + i % 3]).collect();
            let mut kg = vec![0.0; 576];
            for (g, x) in gps.iter().enumerate() {
                let mut eps = [0.0; 6];
                for (j, col) in bts[mi][g].iter().enumerate() {
                    for q in 0..6 {
                        eps[q] += col[q] * ue[j] / h;
                    }
                }
                let mut sig = [0.0; 6];
                for r in 0..6 {
                    for c in 0..6 {
                        sig[r] += d[r][c] * eps[c] * epa;
                    }
                }
                let s3 = [[sig[0], sig[3], sig[5]], [sig[3], sig[1], sig[4]], [sig[5], sig[4], sig[2]]];
                let gr: Vec<[f64; 3]> = (0..8).map(|i| dn_dxi(i, *x).map(|v| 2.0 * v)).collect();
                for a in 0..8 {
                    for b in 0..8 {
                        let mut v = 0.0;
                        for p in 0..3 {
                            for q in 0..3 {
                                v += gr[a][p] * s3[p][q] * gr[b][q];
                            }
                        }
                        v *= h * 0.125;
                        for c in 0..3 {
                            kg[(a * 3 + c) * 24 + b * 3 + c] += v;
                        }
                    }
                }
            }
            kg
        })
        .collect();
    let mut kg = NodeSparse::new(nn, &mesh.elems, 3);
    for (e, en) in mesh.elems.iter().enumerate() {
        kg.add_element(en, &kg_elems[e], -1.0); // store B = -Kg
    }
    let zero_fixed: Vec<Option<f64>> = fixed.iter().map(|g| g.map(|_| 0.0)).collect();
    let mut kz = k0.clone();
    let mut dummy = vec![0.0; n];
    kz.apply_dirichlet(&zero_fixed, &mut dummy);
    kg.apply_dirichlet(&zero_fixed, &mut dummy);
    for i in 0..n {
        if zero_fixed[i].is_some() {
            kg.add_diag(i, -1.0);
        }
    }
    prog(0.3, "factorising");
    let solver = Solver::new(&kz)?;

    // 3. subspace iteration: B φ = μ K φ, largest μ = 1/λ
    let m = model.analysis.modes.clamp(1, 10);
    let p = (2 * m).min(m + 8);
    let mut seed = 0x9E3779B97F4A7C15u64;
    let mut rnd = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };
    let mut x: Vec<Vec<f64>> = (0..p).map(|_| (0..n).map(|i| if zero_fixed[i].is_some() { 0.0 } else { rnd() }).collect()).collect();
    let mut mu = vec![0.0; p];
    let mut old = vec![f64::INFINITY; p];
    let mut tmp = vec![0.0; n];
    for it in 0..60 {
        prog(0.35 + 0.6 * it as f64 / 60.0, &format!("buckling iteration {}", it + 1));
        let mut y = Vec::with_capacity(p);
        for xj in &x {
            kg.matvec(xj, &mut tmp);
            let mut yj = vec![0.0; n];
            res.info.iterations += solver.solve(&kz, &tmp, &mut yj, 1e-9, &|_| {})?;
            y.push(yj);
        }
        let ky: Vec<Vec<f64>> = y
            .iter()
            .map(|v| {
                let mut o = vec![0.0; n];
                kz.matvec(v, &mut o);
                o
            })
            .collect();
        let by: Vec<Vec<f64>> = y
            .iter()
            .map(|v| {
                let mut o = vec![0.0; n];
                kg.matvec(v, &mut o);
                o
            })
            .collect();
        let mut ka = vec![0.0; p * p];
        let mut ba = vec![0.0; p * p];
        for i in 0..p {
            for j in i..p {
                let a = dotp(&y[i], &ky[j]);
                let b = dotp(&y[i], &by[j]);
                ka[i * p + j] = a;
                ka[j * p + i] = a;
                ba[i * p + j] = b;
                ba[j * p + i] = b;
            }
        }
        let l = cholesky(&ka, p)?;
        let linv = dense_inverse(&l, p);
        let mut c = vec![0.0; p * p];
        for i in 0..p {
            for j in 0..p {
                let mut s = 0.0;
                for a in 0..p {
                    for b in 0..p {
                        s += linv[i * p + a] * ba[a * p + b] * linv[j * p + b];
                    }
                }
                c[i * p + j] = s;
            }
        }
        let (ev, vecs) = sym_eigen(&c, p);
        let mut order: Vec<usize> = (0..p).collect();
        order.sort_by(|&a, &b| ev[b].total_cmp(&ev[a]));
        let mut q = vec![0.0; p * p];
        for (col, &o) in order.iter().enumerate() {
            mu[col] = ev[o];
            for i in 0..p {
                q[i * p + col] = (0..p).map(|r| linv[r * p + i] * vecs[r * p + o]).sum();
            }
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
        let conv = (0..m).all(|i| ((mu[i] - old[i]) / mu[i].abs().max(1e-300)).abs() < 1e-7);
        old.copy_from_slice(&mu);
        if conv && it > 2 {
            break;
        }
    }
    res.info.dofs = n;
    let mut modes = Vec::new();
    for i in 0..m {
        if mu[i] <= 1e-12 {
            break;
        }
        let v = &x[i];
        let mx = (0..nn).map(|a| (v[a * 3].powi(2) + v[a * 3 + 1].powi(2) + v[a * 3 + 2].powi(2)).sqrt()).fold(0.0, f64::max).max(1e-300);
        modes.push(Mode { frequency_hz: 0.0, load_multiplier: Some(1.0 / mu[i]), shape: (0..nn).map(|a| [v[a * 3] / mx, v[a * 3 + 1] / mx, v[a * 3 + 2] / mx]).collect() });
    }
    if modes.is_empty() {
        bail!("no buckling found: the applied loads do not compress the structure (all load multipliers are negative). Reverse the load direction to check the opposite case.");
    }
    res.quantity("Critical load multiplier (mode 1)", modes[0].load_multiplier.unwrap(), "×");
    res.modes = modes;
    prog(1.0, "done");
    Ok(())
}

// ------------------------------------------------------------------ elastic-plastic

#[derive(Clone, Copy, Default)]
struct GpState {
    ep: [f64; 6],
    alpha: f64,
}

/// Radial return for J2 plasticity with linear isotropic hardening (SI stresses, engineering
/// shear strains). Returns stress, updated state and the consistent tangent (6×6).
fn return_map(eps: &[f64; 6], st: &GpState, e: f64, nu: f64, sy: f64, hmod: f64) -> ([f64; 6], GpState, [[f64; 6]; 6]) {
    let gmod = e / (2.0 * (1.0 + nu));
    let kmod = e / (3.0 * (1.0 - 2.0 * nu));
    let mut ee = [0.0; 6];
    for i in 0..6 {
        ee[i] = eps[i] - st.ep[i];
    }
    let tr = ee[0] + ee[1] + ee[2];
    let lam = kmod - 2.0 * gmod / 3.0;
    let mut sig = [lam * tr + 2.0 * gmod * ee[0], lam * tr + 2.0 * gmod * ee[1], lam * tr + 2.0 * gmod * ee[2], gmod * ee[3], gmod * ee[4], gmod * ee[5]];
    let p = (sig[0] + sig[1] + sig[2]) / 3.0;
    let s = [sig[0] - p, sig[1] - p, sig[2] - p, sig[3], sig[4], sig[5]];
    let snorm = (s[0] * s[0] + s[1] * s[1] + s[2] * s[2] + 2.0 * (s[3] * s[3] + s[4] * s[4] + s[5] * s[5])).sqrt();
    let q = (1.5f64).sqrt() * snorm;
    let fy = q - (sy + hmod * st.alpha);
    // tangent pieces: C = K 1⊗1 + 2Gβ I_dev − 2G γ̄ n⊗n
    let tangent = |beta: f64, gbar: f64, n: &[f64; 6]| {
        let mut c = [[0.0; 6]; 6];
        for i in 0..6 {
            for j in 0..6 {
                let idev = if i < 3 && j < 3 { if i == j { 2.0 / 3.0 } else { -1.0 / 3.0 } } else if i == j { 0.5 } else { 0.0 };
                let vol = if i < 3 && j < 3 { kmod } else { 0.0 };
                c[i][j] = vol + 2.0 * gmod * beta * idev - 2.0 * gmod * gbar * n[i] * n[j];
            }
        }
        c
    };
    if fy <= 0.0 || sy <= 0.0 {
        return (sig, *st, tangent(1.0, 0.0, &[0.0; 6]));
    }
    let dg = fy / (3.0 * gmod + hmod);
    let k = 1.0 - 3.0 * gmod * dg / q;
    for i in 0..6 {
        sig[i] = if i < 3 { p + s[i] * k } else { s[i] * k };
    }
    let mut ns = *st;
    for i in 0..6 {
        // engineering shear strains carry a factor 2
        let f = if i < 3 { 1.0 } else { 2.0 };
        ns.ep[i] += f * 1.5 * dg * s[i] / q;
    }
    ns.alpha += dg;
    let n = s.map(|v| v / snorm);
    let gbar = 1.0 / (1.0 + hmod / (3.0 * gmod)) - (1.0 - k);
    (sig, ns, tangent(k, gbar, &n))
}

pub fn plasticity(model: &Model, mesh: &crate::mesh::Mesh, prog: Prog, res: &mut Results) -> Result<()> {
    let h = mesh.h * 1e-3;
    let nn = mesh.nodes.len();
    let n = nn * 3;
    let (asm, units) = assemble_structural(model, mesh, prog)?;
    let Assembled { fixed, support_nodes, contact, owner, .. } = asm;
    if support_nodes.is_empty() {
        bail!("no supports defined: add a Fixed Support");
    }
    if !contact.is_empty() {
        res.warnings.push("compression-only supports are treated as fixed in nonlinear analyses".into());
    }
    let f_ext = external_forces(model, mesh, None)?;
    let zero_fixed: Vec<Option<f64>> = fixed.iter().map(|g| g.map(|_| 0.0)).collect();
    res.info.dofs = n;

    let bts: Vec<Vec<Vec<[f64; 6]>>> = units.iter().map(btilde).collect();
    let mats: Vec<(f64, f64, f64, f64)> = mesh
        .materials
        .iter()
        .map(|m| {
            let e = m.youngs_modulus_gpa * 1e9;
            let et = (m.tangent_modulus() * 1e9).min(e * 0.999);
            (e, m.poisson_ratio, m.yield_strength_mpa * 1e6, e * et / (e - et))
        })
        .collect();
    let ne = mesh.elems.len();
    let mut committed = vec![[GpState::default(); 8]; ne];
    let mut u = vec![0.0; n];
    let nsteps = model.analysis.load_steps.clamp(1, 200);
    let fnorm = f_ext.iter().map(|v| v * v).sum::<f64>().sqrt();

    struct ElemOut {
        fe: [f64; 24],
        ke: Vec<f64>,
        states: [GpState; 8],
        savg: [f64; 6],
    }
    // internal forces, tangent stiffness and trial states for displacement u
    let evaluate = |u: &[f64], committed: &[[GpState; 8]], want_k: bool| -> Vec<ElemOut> {
        mesh.elems
            .par_iter()
            .enumerate()
            .map(|(e, en)| {
                let mi = mesh.elem_mat[e] as usize;
                let (em, nu, sy, hm) = mats[mi];
                let ue: Vec<f64> = (0..24).map(|i| u[en[i / 3] as usize * 3 + i % 3]).collect();
                let mut fe = [0.0; 24];
                let mut ke = if want_k { vec![0.0; 576] } else { Vec::new() };
                let mut states = committed[e];
                let mut savg = [0.0; 6];
                for g in 0..8 {
                    let bt = &bts[mi][g];
                    let mut eps = [0.0; 6];
                    for (j, col) in bt.iter().enumerate() {
                        for q in 0..6 {
                            eps[q] += col[q] * ue[j] / h;
                        }
                    }
                    let (sig, ns, c) = return_map(&eps, &committed[e][g], em, nu, sy, hm);
                    states[g] = ns;
                    for q in 0..6 {
                        savg[q] += sig[q] / 8.0;
                    }
                    for (j, col) in bt.iter().enumerate() {
                        fe[j] += (0..6).map(|q| col[q] * sig[q]).sum::<f64>() * h * h * 0.125;
                    }
                    if want_k {
                        let cb: Vec<[f64; 6]> = bt
                            .iter()
                            .map(|col| {
                                let mut r = [0.0; 6];
                                for a in 0..6 {
                                    for b in 0..6 {
                                        r[a] += c[a][b] * col[b];
                                    }
                                }
                                r
                            })
                            .collect();
                        for i in 0..24 {
                            for j in 0..24 {
                                ke[i * 24 + j] += (0..6).map(|q| bt[i][q] * cb[j][q]).sum::<f64>() * h * 0.125;
                            }
                        }
                    }
                }
                ElemOut { fe, ke, states, savg }
            })
            .collect()
    };
    let gather = |outs: &[ElemOut]| -> Vec<f64> {
        let mut fint = vec![0.0; n];
        for (e, o) in outs.iter().enumerate() {
            for (a, &nd) in mesh.elems[e].iter().enumerate() {
                for c in 0..3 {
                    fint[nd as usize * 3 + c] += o.fe[a * 3 + c];
                }
            }
        }
        fint
    };

    let mut last: Vec<ElemOut> = Vec::new();
    let mut reached = 0.0;
    let mut factorisations = 0;
    res.history.push((0.0, 0.0));
    'steps: for step in 1..=nsteps {
        let lam = step as f64 / nsteps as f64;
        let u_start = u.clone();
        // prescribed-displacement increment, applied through the first linear solve
        let delta: Vec<Option<f64>> = (0..n).map(|i| fixed[i].map(|g| lam * g - u[i])).collect();
        let has_delta = delta.iter().any(|d| d.is_some_and(|v| v != 0.0));
        let mut converged = false;
        for it in 0..40 {
            let outs = evaluate(&u, &committed, true);
            let fint = gather(&outs);
            let mut r: Vec<f64> = (0..n).map(|i| if fixed[i].is_some() { 0.0 } else { lam * f_ext[i] - fint[i] }).collect();
            let rn = r.iter().map(|v| v * v).sum::<f64>().sqrt();
            let react = fint.iter().zip(&fixed).filter(|(_, g)| g.is_some()).map(|(f, _)| f * f).sum::<f64>().sqrt();
            let scale = (lam * fnorm).max(react).max(1e-12);
            let first = it == 0 && has_delta;
            if !first && (rn <= 1e-7 * scale && it > 0 || rn <= 1e-12 * scale) {
                converged = true;
                res.info.iterations += it;
                last = outs;
                break;
            }
            // tangent stiffness
            let mut kt = NodeSparse::new(nn, &mesh.elems, 3);
            for (e, o) in outs.iter().enumerate() {
                kt.add_element(&mesh.elems[e], &o.ke, 1.0);
            }
            for (i, g) in zero_fixed.iter().enumerate() {
                if g.is_some() {
                    r[i] = 0.0;
                }
            }
            kt.apply_dirichlet(if first { &delta } else { &zero_fixed }, &mut r);
            let solver = match Solver::new(&kt) {
                Ok(s) => s,
                Err(_) => break,
            };
            factorisations += 1;
            res.info.solver = format!("{} (Newton–Raphson, consistent tangent)", solver.name());
            let mut du = vec![0.0; n];
            solver.solve(&kt, &r, &mut du, 1e-10, &|_| {})?;
            for i in 0..n {
                u[i] += du[i];
            }
            prog(0.05 + 0.9 * (lam - 1.0 / nsteps as f64 * (1.0 - (it as f64 + 1.0) / 40.0)), &format!("load step {step}/{nsteps}, Newton iteration {}", it + 1));
        }
        if !converged {
            u = u_start;
            res.warnings.push(format!("load step {step}/{nsteps} did not converge — the structure is collapsing (plastic hinge / limit load); results shown at load factor {reached:.3}"));
            break 'steps;
        }
        for (e, o) in last.iter().enumerate() {
            committed[e] = o.states;
        }
        reached = lam;
        let umax = (0..nn).map(|q| (u[q * 3].powi(2) + u[q * 3 + 1].powi(2) + u[q * 3 + 2].powi(2)).sqrt()).fold(0.0, f64::max) * 1e3;
        res.history.push((lam, umax));
    }
    if last.is_empty() {
        last = evaluate(&u, &committed, false);
    }
    res.history_label = "max displacement [mm] vs load factor".into();
    res.quantity("Load factor reached", reached, "×");
    res.quantity("Stiffness factorisations", factorisations as f64, "");

    // nodal results
    let mut acc = vec![[0.0f64; 6]; nn];
    let mut pacc = vec![0.0; nn];
    let mut cnt = vec![0u32; nn];
    let mut yld = vec![f64::INFINITY; nn];
    for (e, en) in mesh.elems.iter().enumerate() {
        let sa = &last[e].savg;
        let eq: f64 = committed[e].iter().map(|s| s.alpha).sum::<f64>() / 8.0;
        let ys = mesh.materials[mesh.elem_mat[e] as usize].yield_strength_mpa;
        for &nd in en {
            let nd = nd as usize;
            for q in 0..6 {
                acc[nd][q] += sa[q] * 1e-6;
            }
            pacc[nd] += eq;
            cnt[nd] += 1;
            yld[nd] = yld[nd].min(ys);
        }
    }
    let stress: Vec<[f64; 6]> = acc.iter().zip(&cnt).map(|(s, &c)| s.map(|v| v / c.max(1) as f64)).collect();
    let vm: Vec<f64> = stress.iter().map(von_mises).collect();
    let pr: Vec<(f64, f64)> = stress.par_iter().map(principal).collect();
    res.safety = Some(vm.iter().zip(&yld).map(|(v, y)| if *v > 1e-12 { (y / v).min(15.0) } else { 15.0 }).collect());
    res.s1 = Some(pr.iter().map(|p| p.0).collect());
    res.s3 = Some(pr.iter().map(|p| p.1).collect());
    res.vm = Some(vm);
    res.stress = Some(stress);
    let pe: Vec<f64> = pacc.iter().zip(&cnt).map(|(p, &c)| p / c.max(1) as f64).collect();
    res.quantity("Max equivalent plastic strain", pe.iter().cloned().fold(0.0, f64::max), "mm/mm");
    res.extra.insert(Field::PlasticStrain, pe);
    // reactions per support dof owner
    let fint = gather(&last);
    for (id, label, _) in &support_nodes {
        let mut r = [0.0; 3];
        for dof in 0..n {
            if owner[dof] == *id && fixed[dof].is_some() {
                r[dof % 3] += fint[dof] - reached * f_ext[dof];
            }
        }
        res.reactions.push(Reaction { load_id: *id, label: label.clone(), value: r.map(float) });
    }
    res.disp = Some((0..nn).map(|q| [u[q * 3] * 1e3, u[q * 3 + 1] * 1e3, u[q * 3 + 2] * 1e3]).collect());
    prog(1.0, "done");
    Ok(())
}
