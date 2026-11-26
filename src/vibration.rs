//! Vibration analyses by mode superposition on top of the modal solver:
//!
//! * Harmonic response — steady-state amplitude under harmonic forces over a frequency sweep.
//! * Response spectrum — peak response to base excitation from a shock/seismic spectrum
//!   (SRSS or CQC modal combination), with participation factors and effective mass.
//! * Random vibration — 1σ (RMS) response to a base-acceleration PSD via the modal covariance
//!   matrix; also the zero-crossing frequency used for fatigue (Steinberg 3-band).
//!
//! All use the mass-normalised modes from `fem::modal_basis` (φᵢᵀMφⱼ = δᵢⱼ) and modal damping ζ.

use crate::fem::{self, ModalBasis, Prog};
use crate::mesh::Mesh;
use crate::model::Model;
use crate::results::*;
use anyhow::{Result, bail};
use rayon::prelude::*;

const G: f64 = 9.80665;
const TWO_PI: f64 = std::f64::consts::TAU;

fn modes_for(model: &Model) -> usize {
    model.analysis.modes.max(12).min(40)
}

fn von_mises(s: &[f64; 6]) -> f64 {
    (0.5 * ((s[0] - s[1]).powi(2) + (s[1] - s[2]).powi(2) + (s[2] - s[0]).powi(2)) + 3.0 * (s[3] * s[3] + s[4] * s[4] + s[5] * s[5])).sqrt()
}

fn principal(s: &[f64; 6]) -> (f64, f64) {
    let m = [s[0], s[3], s[5], s[3], s[1], s[4], s[5], s[4], s[2]];
    let (ev, _) = crate::sparse::sym_eigen(&m, 3);
    (ev.iter().cloned().fold(f64::NEG_INFINITY, f64::max), ev.iter().cloned().fold(f64::INFINITY, f64::min))
}

/// Nodal stress tensors [MPa] for a displacement vector u [m] (no thermal strain).
fn nodal_stress(model: &Model, mesh: &Mesh, u: &[f64], fixed: &[Option<f64>]) -> Vec<[f64; 6]> {
    let units: Vec<fem::ElemUnit> = mesh.materials.iter().map(|m| fem::hex_elastic_unit(m.poisson_ratio)).collect();
    let zeros = vec![0.0; u.len()];
    let owner = vec![0u32; u.len()];
    let mut tmp = Results::new(model.analysis.kind);
    fem::structural_post(model, mesh, u, None, &units, fixed, &zeros, &[], &owner, &mut tmp);
    tmp.stress.unwrap_or_default()
}

/// Fill stress-derived fields (von Mises, principals, safety factor) from nodal tensors.
fn set_stress_fields(mesh: &Mesh, stress: Vec<[f64; 6]>, res: &mut Results) {
    let nn = mesh.nodes.len();
    let mut yld = vec![f64::INFINITY; nn];
    for (e, en) in mesh.elems.iter().enumerate() {
        let ys = mesh.materials[mesh.elem_mat[e] as usize].yield_strength_mpa;
        for &n in en {
            yld[n as usize] = yld[n as usize].min(ys);
        }
    }
    let vm: Vec<f64> = stress.iter().map(von_mises).collect();
    let pr: Vec<(f64, f64)> = stress.par_iter().map(principal).collect();
    res.safety = Some(vm.iter().zip(&yld).map(|(v, y)| if *v > 1e-12 { (y / v).min(15.0) } else { 15.0 }).collect());
    res.s1 = Some(pr.iter().map(|p| p.0).collect());
    res.s3 = Some(pr.iter().map(|p| p.1).collect());
    res.vm = Some(vm);
    res.stress = Some(stress);
}

fn to_mm(u: &[f64]) -> Vec<[f64; 3]> {
    (0..u.len() / 3).map(|q| [u[q * 3] * 1e3, u[q * 3 + 1] * 1e3, u[q * 3 + 2] * 1e3]).collect()
}

/// Log-log interpolation of a [frequency, value] table. Outside the table: clamp (spectrum) or 0 (PSD).
pub fn table_value(table: &[[f64; 2]], f: f64, clamp: bool) -> f64 {
    if table.is_empty() {
        return 0.0;
    }
    let mut t: Vec<[f64; 2]> = table.to_vec();
    t.sort_by(|a, b| a[0].partial_cmp(&b[0]).unwrap_or(std::cmp::Ordering::Equal));
    if f <= t[0][0] {
        return if clamp || f == t[0][0] { t[0][1] } else { 0.0 };
    }
    if f >= t[t.len() - 1][0] {
        return if clamp || f == t[t.len() - 1][0] { t[t.len() - 1][1] } else { 0.0 };
    }
    for w in t.windows(2) {
        let ([f0, v0], [f1, v1]) = (w[0], w[1]);
        if f >= f0 && f <= f1 {
            if v0 > 0.0 && v1 > 0.0 && f0 > 0.0 {
                let s = (v1 / v0).ln() / (f1 / f0).ln();
                return v0 * (f / f0).powf(s);
            }
            return v0 + (v1 - v0) * (f - f0) / (f1 - f0);
        }
    }
    0.0
}

/// Participation factors Γᵢ = φᵢᵀ M r for a unit base motion along `dir`, and the total mass
/// that can move in that direction.
fn participation(b: &ModalBasis, dir: usize) -> (Vec<f64>, f64) {
    let total: f64 = b.mass.iter().enumerate().filter(|(i, _)| i % 3 == dir).map(|(_, m)| m).sum();
    let gam = b.phi.iter().map(|p| p.iter().zip(&b.mass).enumerate().filter(|(i, _)| i % 3 == dir).map(|(_, (a, m))| a * m).sum()).collect();
    (gam, total)
}

fn basis(model: &Model, mesh: &Mesh, prog: Prog, res: &mut Results) -> Result<ModalBasis> {
    let b = fem::modal_basis(model, mesh, modes_for(model), prog.slice(0.0, 0.6), res)?;
    if b.omega.first().is_some_and(|w| *w < 1e-3 * b.omega.last().copied().unwrap_or(1.0)) {
        bail!("the model has rigid-body modes (it is not fully supported); add supports before a vibration analysis");
    }
    res.modes.clear();
    res.quantity("Modes used", b.omega.len() as f64, "");
    res.quantity("Highest mode included", b.omega.last().copied().unwrap_or(0.0) / TWO_PI, "Hz");
    Ok(b)
}

// ------------------------------------------------------------------ harmonic response

pub fn harmonic(model: &Model, mesh: &Mesh, prog: Prog, res: &mut Results) -> Result<()> {
    let st = &model.analysis;
    let zeta = st.damping_ratio.max(1e-4);
    let f_ext = fem::external_forces(model, mesh, None)?;
    if f_ext.iter().all(|v| *v == 0.0) {
        bail!("harmonic response needs a Force or Pressure: its value is the amplitude of the harmonic load");
    }
    let b = basis(model, mesh, prog, res)?;
    let fm: Vec<f64> = b.phi.iter().map(|p| p.iter().zip(&f_ext).map(|(a, f)| a * f).sum()).collect();
    let fmodes: Vec<f64> = b.omega.iter().map(|w| w / TWO_PI).collect();
    let fmin = st.freq_min.unwrap_or(fmodes[0] * 0.5).max(1e-6);
    let fmax = st.freq_max.unwrap_or(fmodes[fmodes.len().min(6) - 1] * 1.5).max(fmin * 1.0001);
    // log grid plus dense points around every resonance in range
    let nsteps = st.freq_steps.clamp(20, 5000);
    let mut freqs: Vec<f64> = (0..nsteps).map(|i| fmin * (fmax / fmin).powf(i as f64 / (nsteps - 1) as f64)).collect();
    for fr in &fmodes {
        for k in -12..=12 {
            let f = fr * (1.0 + zeta * k as f64 / 4.0);
            if f >= fmin && f <= fmax {
                freqs.push(f);
            }
        }
    }
    freqs.sort_by(|a, b| a.total_cmp(b));
    freqs.dedup_by(|a, b| (*a - *b).abs() < 1e-12 * a.abs());
    let nn = mesh.nodes.len();
    let nm = b.omega.len();
    // amplitude of the largest-moving node at each frequency
    let amp_at = |f: f64| -> (f64, Vec<(f64, f64)>) {
        let w = TWO_PI * f;
        let q: Vec<(f64, f64)> = (0..nm)
            .map(|i| {
                let (re, im) = (b.omega[i] * b.omega[i] - w * w, 2.0 * zeta * b.omega[i] * w);
                let d = re * re + im * im;
                (fm[i] * re / d, -fm[i] * im / d)
            })
            .collect();
        let maxamp = (0..nn)
            .into_par_iter()
            .map(|nd| {
                let mut s2 = 0.0;
                for c in 0..3 {
                    let dof = nd * 3 + c;
                    let (mut re, mut im) = (0.0, 0.0);
                    for i in 0..nm {
                        re += b.phi[i][dof] * q[i].0;
                        im += b.phi[i][dof] * q[i].1;
                    }
                    s2 += re * re + im * im;
                }
                s2.sqrt()
            })
            .reduce(|| 0.0, f64::max);
        (maxamp, q)
    };
    let mut peak = (fmin, 0.0);
    let total = freqs.len();
    for (k, f) in freqs.iter().enumerate() {
        let (a, _) = amp_at(*f);
        res.history.push((*f, a * 1e3));
        if a > peak.1 {
            peak = (*f, a);
        }
        if k % 20 == 0 {
            prog.report(0.6 + 0.35 * k as f64 / total as f64, &format!("frequency {}/{total}", k + 1))?;
        } else {
            prog.check()?;
        }
    }
    // displacement shape at the peak, at the phase where the largest dof peaks
    let (_, q) = amp_at(peak.0);
    let n = nn * 3;
    let ure: Vec<f64> = (0..n).map(|d| (0..nm).map(|i| b.phi[i][d] * q[i].0).sum()).collect();
    let uim: Vec<f64> = (0..n).map(|d| (0..nm).map(|i| b.phi[i][d] * q[i].1).sum()).collect();
    let dmax = (0..n).max_by(|&a, &c| (ure[a].hypot(uim[a])).total_cmp(&ure[c].hypot(uim[c]))).unwrap_or(0);
    let th = -uim[dmax].atan2(ure[dmax]);
    let u: Vec<f64> = (0..n).map(|d| ure[d] * th.cos() - uim[d] * th.sin()).collect();
    set_stress_fields(mesh, nodal_stress(model, mesh, &u, &b.fixed), res);
    res.disp = Some(to_mm(&u));
    res.history_label = "max displacement amplitude [mm]".into();
    res.history_x = "frequency [Hz]".into();
    res.note = format!("response at the peak frequency {:.2} Hz (mode superposition, ζ = {zeta})", peak.0);
    res.quantity("Peak response frequency", peak.0, "Hz");
    res.quantity("Peak displacement amplitude", peak.1 * 1e3, "mm");
    res.quantity("Damping ratio", zeta, "");
    res.info.dofs = n;
    prog.report(1.0, "done")?;
    Ok(())
}

// ------------------------------------------------------------------ response spectrum

pub fn spectrum(model: &Model, mesh: &Mesh, prog: Prog, res: &mut Results) -> Result<()> {
    let st = &model.analysis;
    if st.table.is_empty() {
        bail!("define the response spectrum: table of [frequency Hz, acceleration g] points (set_analysis table)");
    }
    let zeta = st.damping_ratio.max(1e-4);
    let dir = st.direction.idx();
    let b = basis(model, mesh, prog, res)?;
    let (gam, total_mass) = participation(&b, dir);
    let nm = b.omega.len();
    let n = mesh.nodes.len() * 3;
    // modal peak displacements d_i = Γ_i S_a(f_i) / ω_i²
    let d: Vec<f64> = (0..nm).map(|i| gam[i] * table_value(&st.table, b.omega[i] / TWO_PI, true) * G / (b.omega[i] * b.omega[i])).collect();
    let cqc = st.combination.eq_ignore_ascii_case("cqc");
    let rho = |i: usize, j: usize| -> f64 {
        if !cqc {
            return if i == j { 1.0 } else { 0.0 };
        }
        let r = b.omega[j] / b.omega[i];
        8.0 * zeta * zeta * (1.0 + r) * r.powf(1.5) / ((1.0 - r * r).powi(2) + 4.0 * zeta * zeta * r * (1.0 + r).powi(2))
    };
    let combine = |vals: &dyn Fn(usize) -> f64| -> f64 {
        let mut s = 0.0;
        for i in 0..nm {
            let vi = vals(i);
            if vi == 0.0 {
                continue;
            }
            for j in 0..nm {
                s += rho(i, j) * vi * vals(j);
            }
        }
        s.max(0.0).sqrt()
    };
    prog.report(0.7, "combining modes")?;
    let u: Vec<f64> = (0..n).into_par_iter().map(|dof| combine(&|i| b.phi[i][dof] * d[i])).collect();
    // modal stresses, combined component-wise
    let ms: Vec<Vec<[f64; 6]>> = (0..nm)
        .map(|i| -> Result<_> {
            prog.report(0.7 + 0.25 * i as f64 / nm as f64, &format!("mode {} stresses", i + 1))?;
            let ui: Vec<f64> = b.phi[i].iter().map(|v| v * d[i]).collect();
            Ok(nodal_stress(model, mesh, &ui, &b.fixed))
        })
        .collect::<Result<Vec<_>>>()?;
    let stress: Vec<[f64; 6]> = (0..mesh.nodes.len())
        .into_par_iter()
        .map(|nd| {
            let mut s = [0.0; 6];
            for (c, sc) in s.iter_mut().enumerate() {
                *sc = combine(&|i| ms[i][nd][c]);
            }
            s
        })
        .collect();
    set_stress_fields(mesh, stress, res);
    res.disp = Some(to_mm(&u));
    let mut cum = 0.0;
    for i in 0..nm {
        let frac = gam[i] * gam[i] / total_mass.max(1e-300);
        cum += frac;
        if i < 10 {
            res.quantity(&format!("Mode {} ({:.1} Hz) effective mass", i + 1, b.omega[i] / TWO_PI), frac * 100.0, "%");
        }
    }
    res.quantity("Cumulative effective mass", cum * 100.0, "%");
    if cum < 0.9 {
        res.warnings.push(format!("the modes capture only {:.0}% of the mass in the excitation direction; increase the number of modes for accuracy", cum * 100.0));
    }
    res.note = format!("{} combined peak response (unsigned), base excitation along {}", if cqc { "CQC" } else { "SRSS" }, ["X", "Y", "Z"][dir]);
    res.info.dofs = n;
    prog.report(1.0, "done")?;
    Ok(())
}

// ------------------------------------------------------------------ random vibration

pub fn random(model: &Model, mesh: &Mesh, prog: Prog, res: &mut Results) -> Result<()> {
    let st = &model.analysis;
    if st.table.is_empty() {
        bail!("define the input PSD: table of [frequency Hz, g²/Hz] points (set_analysis table)");
    }
    let zeta = st.damping_ratio.max(1e-4);
    let dir = st.direction.idx();
    let b = basis(model, mesh, prog, res)?;
    let (gam, _) = participation(&b, dir);
    let nm = b.omega.len();
    // integration grid: log-spaced over the PSD band, refined around each resonance
    let (flo, fhi) = st.table.iter().fold((f64::INFINITY, 0.0f64), |(lo, hi), p| (lo.min(p[0]), hi.max(p[0])));
    let mut fs: Vec<f64> = (0..4000).map(|i| flo * (fhi / flo).powf(i as f64 / 3999.0)).collect();
    for w in &b.omega {
        let f = w / TWO_PI;
        for k in -200..=200 {
            let x = f * (1.0 + zeta * k as f64 / 20.0);
            if x > flo && x < fhi {
                fs.push(x);
            }
        }
    }
    fs.sort_by(|a, b| a.total_cmp(b));
    fs.dedup_by(|a, b| (*a - *b).abs() < 1e-12 * a.abs());
    // modal covariance C_ij = ∫ Γ_i Γ_j Re(H_i H_j*) S_a df and C2_ij = ∫ … ω² … (for crossing rate)
    prog.report(0.65, "integrating PSD")?;
    let mut c = vec![0.0; nm * nm];
    let mut c2 = vec![0.0; nm * nm];
    let mut grms2 = 0.0;
    let h = |i: usize, w: f64| -> (f64, f64) {
        let (re, im) = (b.omega[i] * b.omega[i] - w * w, 2.0 * zeta * b.omega[i] * w);
        let d = re * re + im * im;
        (re / d, -im / d)
    };
    for k in 0..fs.len() - 1 {
        let (fa, fb) = (fs[k], fs[k + 1]);
        let df = fb - fa;
        for (f, wgt) in [(fa, 0.5 * df), (fb, 0.5 * df)] {
            let s = table_value(&st.table, f, false) * G * G;
            if s == 0.0 {
                continue;
            }
            grms2 += s * wgt / (G * G);
            let w = TWO_PI * f;
            let hs: Vec<(f64, f64)> = (0..nm).map(|i| h(i, w)).collect();
            for i in 0..nm {
                for j in i..nm {
                    let v = gam[i] * gam[j] * (hs[i].0 * hs[j].0 + hs[i].1 * hs[j].1) * s * wgt;
                    c[i * nm + j] += v;
                    c2[i * nm + j] += v * w * w;
                }
            }
        }
    }
    for i in 0..nm {
        for j in 0..i {
            c[i * nm + j] = c[j * nm + i];
            c2[i * nm + j] = c2[j * nm + i];
        }
    }
    let quad = |a: &dyn Fn(usize) -> f64, m: &[f64]| -> f64 {
        let v: Vec<f64> = (0..nm).map(a).collect();
        let mut s = 0.0;
        for i in 0..nm {
            if v[i] == 0.0 {
                continue;
            }
            for j in 0..nm {
                s += v[i] * v[j] * m[i * nm + j];
            }
        }
        s.max(0.0)
    };
    let n = mesh.nodes.len() * 3;
    let u: Vec<f64> = (0..n).into_par_iter().map(|dof| quad(&|i| b.phi[i][dof], &c).sqrt()).collect();
    // zero-crossing frequency of the most-responding dof
    let dmax = (0..n).max_by(|&a, &bb| u[a].total_cmp(&u[bb])).unwrap_or(0);
    let nu0 = (quad(&|i| b.phi[i][dmax], &c2) / quad(&|i| b.phi[i][dmax], &c).max(1e-300)).sqrt() / TWO_PI;
    prog.report(0.8, "modal stresses")?;
    let ms: Vec<Vec<[f64; 6]>> = (0..nm).map(|i| nodal_stress(model, mesh, &b.phi[i], &b.fixed)).collect();
    let stress: Vec<[f64; 6]> = (0..mesh.nodes.len())
        .into_par_iter()
        .map(|nd| {
            let mut s = [0.0; 6];
            for (cc, sc) in s.iter_mut().enumerate() {
                *sc = quad(&|i| ms[i][nd][cc], &c).sqrt();
            }
            s
        })
        .collect();
    set_stress_fields(mesh, stress, res);
    res.disp = Some(to_mm(&u));
    let umax = u.iter().cloned().fold(0.0, f64::max) * 1e3;
    let vmax = res.vm.as_ref().map(|v| v.iter().cloned().fold(0.0, f64::max)).unwrap_or(0.0);
    res.quantity("Input acceleration (RMS)", grms2.sqrt(), "g");
    res.quantity("1σ max displacement", umax, "mm");
    res.quantity("3σ max displacement", 3.0 * umax, "mm");
    res.quantity("1σ max von Mises", vmax, "MPa");
    res.quantity("3σ max von Mises", 3.0 * vmax, "MPa");
    res.quantity("Zero-crossing frequency", nu0, "Hz");
    res.note = format!("1σ (RMS) response to the base PSD along {}; multiply by 3 for 3σ", ["X", "Y", "Z"][dir]);
    res.info.dofs = n;
    prog.report(1.0, "done")?;
    Ok(())
}
