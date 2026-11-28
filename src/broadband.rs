//! Broadband full-wave EM: one FDTD run with a Gaussian-modulated pulse gives the response over
//! a whole frequency band.
//!
//! * Lumped ports (`Load::LumpedPort`): a line of Yee edges across a gap between two conductors,
//!   loaded by the port resistance (σ = N / (R h) on its N edges) and, when driven, a Norton
//!   current source I = V_s / R. Port voltage V = −∫E·dl along the line, current I = ∮H·dl around
//!   its middle edge; their DFTs give Z_in = V / I and the power waves a = (V + Z I) / 2√Z,
//!   b = (V − Z I) / 2√Z, so S_pj = b_p / a_j with port j driven (one run per driven port).
//! * Time signals are recorded at the ports only; field DFTs are kept on the far-field transform
//!   shell at a few frequencies (memory stays bounded) plus the full grid at the band centre.
//! * Emissions (open boundary): far field per frequency, normalised to the port's source
//!   voltage; free space (strongest direction) or over a metal ground plane (image theory) with a
//!   1–4 m receive-antenna height scan and a turntable, compared with FCC / CISPR limits.

use crate::fdtd::{self, CV, Cx, Setup};
use crate::fem::Prog;
use crate::mesh::{Mesh, REGION_SOLID};
use crate::model::{Load, Model};
use crate::results::*;
use anyhow::{Result, bail};
use rayon::prelude::*;

const C0: f64 = 299_792_458.0;
const EPS0: f64 = 8.854_187_817e-12;
const MU0: f64 = 1.256_637_062_12e-6;
/// Ring-down threshold (port amplitude relative to its peak).
const DECAY: f64 = 3e-4;

/// A lumped port on the grid.
struct LPort {
    id: u32,
    /// Flat E indices of its edges, all along `axis`.
    edges: Vec<usize>,
    /// +1 when `to` lies on the +axis side of `from`.
    dir: f64,
    z: f64,
    volts: f64,
    mid: usize,
}

/// Gaussian-modulated sine `exp(−((t−t0)/τ)²) sin(2π fc (t−t0))`.
#[derive(Clone, Copy)]
struct Pulse {
    t0: f64,
    tau: f64,
    fc: f64,
}

impl Pulse {
    fn v(&self, t: f64) -> f64 {
        let x = (t - self.t0) / self.tau;
        (-x * x).exp() * (std::f64::consts::TAU * self.fc * (t - self.t0)).sin()
    }
}

/// Snap the lumped ports to grid edges and load them with their resistance.
fn lumped_ports(model: &Model, mesh: &Mesh, s: &mut Setup, warnings: &mut Vec<String>) -> Result<Vec<LPort>> {
    let nn = s.nn;
    let pad = s.pad as i64;
    let mut out = Vec::new();
    for it in &model.loads {
        let Load::LumpedPort { from, to, impedance, voltage } = &it.load else { continue };
        if !(*impedance > 0.0) || !impedance.is_finite() || !voltage.is_finite() {
            bail!("lumped port #{}: impedance must be > 0 and the voltage finite", it.id);
        }
        let g = |p: &[f64; 3], k: usize| ((p[k] - mesh.origin[k]) / mesh.h).round() as i64 + pad;
        let a0 = [g(from, 0), g(from, 1), g(from, 2)];
        let a1 = [g(to, 0), g(to, 1), g(to, 2)];
        let diff: Vec<usize> = (0..3).filter(|&k| a0[k] != a1[k]).collect();
        if diff.len() != 1 {
            bail!(
                "lumped port #{}: 'from' and 'to' must differ along exactly one axis after snapping to the {:.3} mm grid (got {} differing axes) — put both points on a line parallel to x, y or z",
                it.id,
                mesh.h,
                diff.len()
            );
        }
        for (p, a) in [(from, a0), (to, a1)] {
            for k in 0..3 {
                let snapped = (a[k] - pad) as f64 * mesh.h + mesh.origin[k];
                if (snapped - p[k]).abs() > 0.26 * mesh.h {
                    warnings.push(format!("lumped port #{}: point ({:.3}, {:.3}, {:.3}) mm snapped to the grid by {:.3} mm", it.id, p[0], p[1], p[2], (snapped - p[k]).abs()));
                    break;
                }
            }
        }
        let ax = diff[0];
        let (lo, hi) = (a0[ax].min(a1[ax]), a0[ax].max(a1[ax]));
        for k in 0..3 {
            let (v0, v1) = if k == ax { (lo, hi) } else { (a0[k], a0[k]) };
            if v0 < pad + 1 || v1 > s.n[k] as i64 - pad - 1 {
                bail!("lumped port #{} lies outside the enclosure (or on its wall): move it inside", it.id);
            }
        }
        let mut edges = Vec::new();
        for t in lo..hi {
            let mut p = a0;
            p[ax] = t;
            let k = ax * nn + s.node(p[0] as usize, p[1] as usize, p[2] as usize);
            if s.ecoef[3 * nn + k] == 0.0 {
                bail!("lumped port #{} runs along metal: the line from 'from' to 'to' must cross the gap between two conductors (at least one cell long, through air or dielectric)", it.id);
            }
            edges.push(k);
        }
        // resistance R in series over N edges: σ = N / (R h)
        let sig_p = edges.len() as f64 / (impedance * s.h);
        for &k in &edges {
            let (ca, cb) = (s.ecoef[k] as f64, s.ecoef[3 * nn + k] as f64);
            let x = (1.0 - ca) / (1.0 + ca);
            let e = s.dt / (cb * s.h * (1.0 + x));
            let x2 = x + sig_p * s.dt / (2.0 * e);
            s.ecoef[k] = ((1.0 - x2) / (1.0 + x2)) as f32;
            s.ecoef[3 * nn + k] = (s.dt / (e * s.h) / (1.0 + x2)) as f32;
        }
        let mid = edges[edges.len() / 2];
        out.push(LPort { id: it.id, edges, dir: if a1[ax] > a0[ax] { 1.0 } else { -1.0 }, z: *impedance, volts: *voltage, mid });
    }
    Ok(out)
}

struct RunOut {
    /// Port voltages at t = (n+1)·dt and currents at (n+½)·dt.
    v: Vec<Vec<f64>>,
    i: Vec<Vec<f64>>,
    /// Source waveform at (n+½)·dt.
    vs: Vec<f64>,
    decayed: bool,
}

/// Running DFT sums Σ E(t) (cos ωt, sin ωt) at t = (n+1)·dt on a set of edges.
struct Dft {
    freqs: Vec<f64>,
    idx: Vec<u32>,
    re: Vec<Vec<f32>>,
    im: Vec<Vec<f32>>,
}

impl Dft {
    fn new(freqs: Vec<f64>, idx: Vec<u32>) -> Dft {
        let z = || vec![vec![0.0f32; idx.len()]; freqs.len()];
        Dft { re: z(), im: z(), freqs, idx }
    }
    fn add(&mut self, e: &[f32], t: f64) {
        let idx = &self.idx;
        self.freqs.par_iter().zip(self.re.par_iter_mut()).zip(self.im.par_iter_mut()).for_each(|((f, re), im)| {
            let ph = std::f64::consts::TAU * f * t;
            let (c, s) = (ph.cos() as f32, ph.sin() as f32);
            for (q, &k) in idx.iter().enumerate() {
                let v = e[k as usize];
                re[q] += v * c;
                im[q] += v * s;
            }
        });
    }
}

#[allow(clippy::too_many_arguments)]
fn run(s: &Setup, ports: &[LPort], drive: usize, pulse: Pulse, max_steps: usize, dfts: &mut [&mut Dft], prog: &Prog, frac: (f64, f64)) -> Result<RunOut> {
    let nn = s.nn;
    let mut e = vec![0.0f32; 3 * nn];
    let mut h = vec![0.0f32; 3 * nn];
    let np = ports.len();
    let (mut v, mut cur) = (vec![Vec::new(); np], vec![Vec::new(); np]);
    let mut vs = Vec::new();
    let min_steps = (2.0 * pulse.t0 / s.dt).ceil() as usize;
    let window = ((1.0 / (pulse.fc * s.dt)).ceil() as usize).max(16);
    let mut peak = vec![0.0f64; np];
    let mut decayed = false;
    let dp = &ports[drive];
    let mut n = 0usize;
    while n < max_steps {
        fdtd::step_h(s, &e, &mut h);
        for (p, lp) in ports.iter().enumerate() {
            cur[p].push(lp.dir * s.h * fdtd::loop_current(s, &h, lp.mid));
        }
        fdtd::step_e(s, &h, &mut e);
        let t_half = (n as f64 + 0.5) * s.dt;
        let src = pulse.v(t_half);
        vs.push(src);
        // Norton source I = Vs / R along the port direction: ∂E/∂t = … − J/ε
        let j = dp.dir * src / dp.z / (s.h * s.h);
        for &k in &dp.edges {
            e[k] -= (s.ecoef[3 * nn + k] as f64 * s.h * j) as f32;
        }
        for (p, lp) in ports.iter().enumerate() {
            let sum: f64 = lp.edges.iter().map(|&k| e[k] as f64).sum();
            v[p].push(-lp.dir * sum * s.h);
        }
        let t1 = (n as f64 + 1.0) * s.dt;
        for d in dfts.iter_mut() {
            d.add(&e, t1);
        }
        n += 1;
        for p in 0..np {
            peak[p] = peak[p].max(v[p][n - 1].abs()).max((cur[p][n - 1] * ports[p].z).abs());
        }
        if n.is_multiple_of(window) {
            if !e.iter().step_by(97).all(|x| x.is_finite()) {
                bail!("the FDTD solution became unstable (non-finite fields)");
            }
            prog.report(frac.0 + (frac.1 - frac.0) * (n as f64 / max_steps as f64).min(1.0), &format!("broadband FDTD: step {n}"))?;
            if n >= min_steps {
                let quiet = (0..np).all(|p| {
                    let w = &v[p][n - window..];
                    let wi = &cur[p][n - window..];
                    let m = w.iter().map(|x| x.abs()).fold(0.0, f64::max).max(wi.iter().map(|x| (x * ports[p].z).abs()).fold(0.0, f64::max));
                    m <= DECAY * peak[p].max(1e-300)
                });
                if quiet {
                    decayed = true;
                    break;
                }
            }
        }
    }
    Ok(RunOut { v, i: cur, vs, decayed })
}

/// Σ x_n e^{−jω(t0 + n·dt)} (same convention as the field DFTs: Ê = re − j·im).
fn dft(x: &[f64], t0: f64, dt: f64, f: f64) -> Cx {
    let w = std::f64::consts::TAU * f;
    let (mut re, mut im) = (0.0, 0.0);
    for (n, v) in x.iter().enumerate() {
        let ph = w * (t0 + n as f64 * dt);
        re += v * ph.cos();
        im -= v * ph.sin();
    }
    Cx::new(re, im)
}

fn cdiv(a: Cx, b: Cx) -> Cx {
    let d = b.norm2().max(1e-300);
    Cx::new((a.re * b.re + a.im * b.im) / d, (a.im * b.re - a.re * b.im) / d)
}

fn db(x: f64) -> f64 {
    20.0 * x.max(1e-15).log10()
}

/// Turn raw DFT sums (re, im) into the phasor Ê/V̂s and return (re', im') in the same convention.
fn normalise(re: f32, im: f32, c: Cx) -> (f32, f32) {
    // Ê = re − j·im ;  Ê·c = re' − j·im'
    let (r, i) = (re as f64, im as f64);
    ((r * c.re + i * c.im) as f32, (i * c.re - r * c.im) as f32)
}

pub fn solve(model: &Model, mesh: &Mesh, prog: Prog, res: &mut Results) -> Result<()> {
    let t_start = std::time::Instant::now();
    let st = &model.analysis;
    let (flo, fhi) = st.band();
    if !(fhi > flo) || !(fhi > 0.0) {
        bail!("set a frequency band with freq_max > freq_min ≥ 0 (Hz), e.g. set_analysis freq_min_ghz 1, freq_max_ghz 3");
    }
    let fc = 0.5 * (flo + fhi);
    let bw = (0.5 * (fhi - flo)).max(0.2 * fc);
    prog.report(0.0, "building the FDTD grid")?;
    let mut s = fdtd::setup_grid(model, mesh, fc, fhi)?;
    let mut warnings = std::mem::take(&mut s.warnings);
    let ports = lumped_ports(model, mesh, &mut s, &mut warnings)?;
    if ports.is_empty() {
        bail!("broadband EM needs a lumped port (add_lumped_port): a source across the gap between two conductors, e.g. the feed of an antenna or the end of a trace");
    }
    let driven: Vec<usize> = (0..ports.len()).filter(|&p| ports[p].volts != 0.0).collect();
    if driven.is_empty() {
        bail!("no lumped port is driven: give one a source voltage (voltage > 0); passive ports are matched terminations");
    }
    let tau = (10f64.ln()).sqrt() / (std::f64::consts::PI * bw);
    let pulse = Pulse { t0: 4.5 * tau, tau, fc };
    let steps_per_period = 1.0 / (fc * s.dt);
    let max_steps = ((st.max_periods.max(20) as f64 * steps_per_period) as usize).max((3.0 * pulse.t0 / s.dt) as usize);

    // frequency points
    let nf = st.freq_steps.clamp(11, 2001);
    let f0 = if flo > 0.0 { flo } else { fhi / nf as f64 };
    let freqs: Vec<f64> = (0..nf).map(|i| f0 + (fhi - f0) * i as f64 / (nf - 1) as f64).collect();
    // far-field frequencies (open boundary only)
    let open = s.pad > 0;
    let nff = if open { nf.min(21) } else { 0 };
    let ff_freqs: Vec<f64> = (0..nff).map(|i| f0 + (fhi - f0) * i as f64 / (nff.max(2) - 1) as f64).collect();
    let shell = if open { fdtd::ntff_shell(&s)? } else { Vec::new() };
    let mut ff = Dft::new(ff_freqs.clone(), shell);
    let all: Vec<u32> = (0..3 * s.nn as u32).collect();
    let mut centre = Dft::new(vec![fc], all);

    let np = ports.len();
    // S[p][j][f]
    let mut smat: Vec<Vec<Option<Vec<Cx>>>> = vec![vec![None; np]; np];
    let mut zin: Vec<Option<Vec<Cx>>> = vec![None; np];
    let mut vs_spec: Vec<Cx> = Vec::new();
    let mut ff_vs: Vec<Cx> = Vec::new();
    let mut fc_vs = Cx::default();
    // power delivered at the band centre by the (single) driven port, for the power balance
    let mut p_fc = 0.0;
    let mut all_decayed = true;
    let mut total_steps = 0usize;
    let nd = driven.len();
    // field DFTs superpose the driven runs, each weighted by its port's voltage
    let mut ff_acc: Option<(Vec<Vec<f32>>, Vec<Vec<f32>>)> = None;
    let mut c_acc: Option<(Vec<f32>, Vec<f32>)> = None;
    for (r, &j) in driven.iter().enumerate() {
        for d in [&mut ff, &mut centre] {
            d.re.iter_mut().chain(d.im.iter_mut()).for_each(|v| v.iter_mut().for_each(|x| *x = 0.0));
        }
        let frac = (0.02 + 0.85 * r as f64 / nd as f64, 0.02 + 0.85 * (r + 1) as f64 / nd as f64);
        let mut dfts: Vec<&mut Dft> = if open { vec![&mut ff, &mut centre] } else { vec![&mut centre] };
        let out = run(&s, &ports, j, pulse, max_steps, &mut dfts, &prog, frac)?;
        total_steps += out.vs.len();
        all_decayed &= out.decayed;
        prog.report(frac.1, "Fourier transforms")?;
        let spec = |x: &[f64], t0: f64| -> Vec<Cx> { freqs.par_iter().map(|&f| dft(x, t0, s.dt, f)).collect() };
        vs_spec = spec(&out.vs, 0.5 * s.dt);
        let vp: Vec<Vec<Cx>> = out.v.iter().map(|x| spec(x, s.dt)).collect();
        let ip: Vec<Vec<Cx>> = out.i.iter().map(|x| spec(x, 0.5 * s.dt)).collect();
        let zj = ports[j].z;
        let a_j: Vec<Cx> = (0..nf).map(|f| vp[j][f].add(ip[j][f].scale(zj)).scale(0.5 / zj.sqrt())).collect();
        for p in 0..np {
            let zp = ports[p].z;
            let sp: Vec<Cx> = (0..nf).map(|f| cdiv(vp[p][f].sub(ip[p][f].scale(zp)).scale(0.5 / zp.sqrt()), a_j[f])).collect();
            smat[p][j] = Some(sp);
        }
        zin[j] = Some((0..nf).map(|f| cdiv(vp[j][f], ip[j][f])).collect());
        ff_vs = ff_freqs.iter().map(|&f| dft(&out.vs, 0.5 * s.dt, s.dt, f)).collect();
        fc_vs = dft(&out.vs, 0.5 * s.dt, s.dt, fc);
        {
            let vv = cdiv(dft(&out.v[j], s.dt, s.dt, fc), fc_vs).scale(ports[j].volts);
            let ii = cdiv(dft(&out.i[j], 0.5 * s.dt, s.dt, fc), fc_vs).scale(ports[j].volts);
            p_fc = 0.5 * (vv.re * ii.re + vv.im * ii.im);
            if std::env::var_os("SIMCRAFT_DEBUG_PORT").is_some() {
                let is = Cx::new(ports[j].volts / ports[j].z, 0.0);
                let psrc = 0.5 * (vv.re * is.re + vv.im * is.im);
                let pr = 0.5 * vv.norm2() / ports[j].z;
                eprintln!("[port] V {vv:?} I {ii:?}  P(VI*) {p_fc:.4e}  Psrc−PR {:.4e} (src {psrc:.4e}, R {pr:.4e})", psrc - pr);
            }
        }
        let w = ports[j].volts as f32;
        match &mut ff_acc {
            None => ff_acc = Some((ff.re.iter().map(|v| v.iter().map(|x| x * w).collect()).collect(), ff.im.iter().map(|v| v.iter().map(|x| x * w).collect()).collect())),
            Some((ar, ai)) => {
                for k in 0..nff {
                    for q in 0..ar[k].len() {
                        ar[k][q] += w * ff.re[k][q];
                        ai[k][q] += w * ff.im[k][q];
                    }
                }
            }
        }
        match &mut c_acc {
            None => c_acc = Some((centre.re[0].iter().map(|x| x * w).collect(), centre.im[0].iter().map(|x| x * w).collect())),
            Some((ar, ai)) => {
                for q in 0..ar.len() {
                    ar[q] += w * centre.re[0][q];
                    ai[q] += w * centre.im[0][q];
                }
            }
        }
    }
    let _ = vs_spec;
    if !all_decayed {
        warnings.push(format!(
            "the pulse did not ring down within {} periods of the band centre: S-parameters near sharp resonances are approximate — raise max_periods or add loss",
            st.max_periods
        ));
    }

    // ---- port results
    let ghz: Vec<f64> = freqs.iter().map(|f| f / 1e9).collect();
    let mut first = true;
    let ic = nf / 2;
    for &j in &driven {
        let pj = &ports[j];
        let s11 = smat[j][j].as_ref().unwrap();
        let mag: Vec<f64> = s11.iter().map(|c| c.norm2().sqrt()).collect();
        let (imin, mmin) = mag.iter().enumerate().fold((0, f64::INFINITY), |b, (i, &m)| if m < b.1 { (i, m) } else { b });
        let z = zin[j].as_ref().unwrap();
        let tag = format!("Port #{}", pj.id);
        if first {
            res.quantity("Minimum |S11|", db(mmin), "dB");
            res.history = ghz.iter().zip(&mag).map(|(f, m)| (*f, db(*m))).collect();
            res.history_label = format!("|S11| [dB] at port #{}", pj.id);
            res.history_x = "frequency [GHz]".into();
            first = false;
        } else {
            res.series.push((format!("|S{0}{0}| [dB]", pj.id), ghz.iter().zip(&mag).map(|(f, m)| (*f, db(*m))).collect(), true));
        }
        res.quantity(&format!("{tag}: best match |S11|"), db(mmin), "dB");
        res.quantity(&format!("{tag}: best match at"), ghz[imin], "GHz");
        res.quantity(&format!("{tag}: Zin at best match (real)"), z[imin].re, "Ω");
        res.quantity(&format!("{tag}: Zin at best match (imaginary)"), z[imin].im, "Ω");
        let vswr = (1.0 + mmin) / (1.0 - mmin).max(1e-9);
        res.quantity(&format!("{tag}: VSWR at best match"), vswr, "");
        // −10 dB bandwidth around the best match
        if db(mmin) < -10.0 {
            let (mut a, mut b) = (imin, imin);
            while a > 0 && db(mag[a - 1]) < -10.0 {
                a -= 1;
            }
            while b + 1 < nf && db(mag[b + 1]) < -10.0 {
                b += 1;
            }
            res.quantity(&format!("{tag}: −10 dB bandwidth"), (freqs[b] - freqs[a]) / 1e6, "MHz");
        }
        // series resonance: Im Zin crossing zero (lowest)
        if let Some(k) = (1..nf).find(|&k| z[k - 1].im.signum() != z[k].im.signum() && z[k - 1].im < 0.0) {
            let w = z[k - 1].im / (z[k - 1].im - z[k].im);
            let fr = freqs[k - 1] + w * (freqs[k] - freqs[k - 1]);
            let rr = z[k - 1].re + w * (z[k].re - z[k - 1].re);
            res.quantity(&format!("{tag}: resonance (Im Zin = 0)"), fr / 1e9, "GHz");
            res.quantity(&format!("{tag}: Re Zin at resonance"), rr, "Ω");
        }
        res.quantity(&format!("{tag}: Zin at {:.3} GHz (real)", ghz[ic]), z[ic].re, "Ω");
        res.quantity(&format!("{tag}: Zin at {:.3} GHz (imaginary)", ghz[ic]), z[ic].im, "Ω");
        res.series.push((format!("Re Zin #{} [Ω]", pj.id), ghz.iter().zip(z).map(|(f, c)| (*f, c.re)).collect(), false));
        res.series.push((format!("Im Zin #{} [Ω]", pj.id), ghz.iter().zip(z).map(|(f, c)| (*f, c.im)).collect(), false));
        for (p, lp) in ports.iter().enumerate() {
            if p == j {
                continue;
            }
            let sp = smat[p][j].as_ref().unwrap();
            let m: Vec<f64> = sp.iter().map(|c| c.norm2().sqrt()).collect();
            let nm = format!("|S{}{}|", lp.id, pj.id);
            res.series.push((format!("{nm} [dB]"), ghz.iter().zip(&m).map(|(f, v)| (*f, db(*v))).collect(), true));
            res.quantity(&format!("{nm} at {:.3} GHz", ghz[ic]), db(m[ic]), "dB");
            let (imx, mx) = m.iter().enumerate().fold((0, 0.0f64), |b, (i, &v)| if v > b.1 { (i, v) } else { b });
            res.quantity(&format!("{nm} maximum"), db(mx), "dB");
            res.quantity(&format!("{nm} maximum at"), ghz[imx], "GHz");
        }
    }
    s_params_json(res, &ports, &smat, &zin, &freqs);

    // ---- field at the band centre (for the drive voltages)
    let (cre, cim) = c_acc.unwrap();
    let cnorm = cdiv(Cx::new(1.0, 0.0), fc_vs);
    let (nre, nim): (Vec<f32>, Vec<f32>) = cre.par_iter().zip(cim.par_iter()).map(|(r, i)| normalise(*r, *i, cnorm)).unzip();
    let pad = s.pad;
    let cell_of = |e: usize| {
        let c = mesh.elem_cell[e];
        s.cell(c[0] as usize + pad, c[1] as usize + pad, c[2] as usize + pad)
    };
    let emag: Vec<f64> = (0..mesh.elems.len()).into_par_iter().map(|e| fdtd::cell_e(&s, &nre, &nim, 1.0, cell_of(e))).collect();
    let nn = mesh.nodes.len();
    let (mut acc, mut cnt) = (vec![0.0; nn], vec![0u32; nn]);
    for (e, en) in mesh.elems.iter().enumerate() {
        for &n in en {
            acc[n as usize] += emag[e];
            cnt[n as usize] += 1;
        }
    }
    res.extra.insert(Field::ElectricField, acc.iter().zip(&cnt).map(|(a, &c)| if c > 0 { a / c as f64 } else { 0.0 }).collect());

    // ---- power balance at the band centre (one driven port, open boundary)
    if open && nd == 1 && p_fc > 0.0 {
        let fcen = fdtd::far_field_at(&s, &nre, &nim, 1.0, fc)?;
        if std::env::var_os("SIMCRAFT_DEBUG_PORT").is_some() {
            let nt = fdtd::Ntff::new(&s, &nre, &nim, 1.0, fc)?;
            eprintln!("[ntff] poynting through box {:.4e} W, far-field integral {:.4e} W, delivered {:.4e} W", nt.poynting(), fcen.prad, p_fc);
        }
        res.quantity(&format!("Power delivered at {:.4} GHz", fc / 1e9), p_fc * 1e3, "mW");
        res.quantity(&format!("Radiated power at {:.4} GHz", fc / 1e9), fcen.prad * 1e3, "mW");
        res.quantity("Radiated / delivered power at the band centre", 100.0 * fcen.prad / p_fc, "%");
        let d = 4.0 * std::f64::consts::PI * fcen.umax / fcen.prad.max(1e-300);
        res.quantity("Directivity at the band centre", 10.0 * d.max(1e-12).log10(), "dBi");
    }

    // ---- clearance between the model and the absorbing layer (far-field accuracy)
    if open {
        let n = s.n;
        let (mut lo, mut hi) = ([usize::MAX; 3], [0usize; 3]);
        for k in s.pad..n[2] - s.pad {
            for j in s.pad..n[1] - s.pad {
                for i in s.pad..n[0] - s.pad {
                    let c = s.cell(i, j, k);
                    if s.cell_pec[c] || s.cell_eps[c] > 1.01 || s.cell_sig[c] > 1e-6 {
                        for (a, v) in [i, j, k].into_iter().enumerate() {
                            lo[a] = lo[a].min(v);
                            hi[a] = hi[a].max(v + 1);
                        }
                    }
                }
            }
        }
        if lo[0] != usize::MAX {
            let gap = (0..3).map(|a| (lo[a] - s.pad).min(n[a] - s.pad - hi[a])).min().unwrap_or(0) as f64 * s.h;
            let lam = C0 / flo.max(0.5 * fc);
            if gap < 0.25 * lam {
                warnings.push(format!(
                    "the enclosure leaves only {:.0} mm around the model (λ/{:.0} at {:.3} GHz): the absorbing boundary sits in the near field, so radiated power, gain and emissions can be off by 10 % or more — leave at least λ/4 ({:.0} mm)",
                    gap * 1e3,
                    lam / gap.max(1e-9),
                    C0 / lam / 1e9,
                    0.25 * lam * 1e3
                ));
            }
        }
    }

    // ---- emissions
    if open {
        prog.report(0.9, "far field and emissions")?;
        let (ar, ai) = ff_acc.unwrap();
        emissions(model, mesh, &s, &ff, &ar, &ai, &ff_vs, res, &mut warnings)?;
    }

    res.quantity("Frequency band", flo / 1e9, &format!("– {:.4} GHz", fhi / 1e9));
    res.quantity("Cells per wavelength at the top of the band", s.cells_per_wavelength, "");
    res.quantity("Time steps", total_steps as f64, "");
    res.quantity("Time step", s.dt * 1e12, "ps");
    res.quantity("FDTD cells", s.cells() as f64, "");
    res.warnings.extend(warnings);
    res.note = format!("|E| at the band centre ({:.4} GHz) for the port source voltages (peak)", fc / 1e9);
    res.info.solver = format!("broadband FDTD on CPU ({} threads), {} run{}", rayon::current_num_threads(), nd, if nd == 1 { "" } else { "s" });
    res.info.dofs = 6 * s.cells();
    res.info.iterations = total_steps;
    res.info.linear_solve_seconds = t_start.elapsed().as_secs_f64();
    prog.report(1.0, "done")?;
    Ok(())
}

/// S-parameters and input impedances as JSON (for get_results / scripting), on `freqs`.
fn s_params_json(res: &mut Results, ports: &[LPort], smat: &[Vec<Option<Vec<Cx>>>], zin: &[Option<Vec<Cx>>], freqs: &[f64]) {
    let mut sp = serde_json::Map::new();
    for (p, lp) in ports.iter().enumerate() {
        for (j, lj) in ports.iter().enumerate() {
            if let Some(v) = &smat[p][j] {
                sp.insert(format!("S{}{}", lp.id, lj.id), serde_json::json!(v.iter().map(|c| [float(c.re), float(c.im)]).collect::<Vec<_>>()));
            }
        }
        if let Some(z) = &zin[p] {
            sp.insert(format!("Zin{}", lp.id), serde_json::json!(z.iter().map(|c| [float(c.re), float(c.im)]).collect::<Vec<_>>()));
        }
    }
    sp.insert("freq_hz".into(), serde_json::json!(freqs.iter().map(|f| float(*f)).collect::<Vec<_>>()));
    sp.insert("ports".into(), serde_json::json!(ports.iter().map(|p| serde_json::json!({"id": p.id, "impedance": p.z})).collect::<Vec<_>>()));
    res.data.insert("s_parameters".into(), serde_json::Value::Object(sp));
}

/// Complex vector helpers.
fn cdot_r(a: CV, r: [f64; 3]) -> Cx {
    a[0].scale(r[0]).add(a[1].scale(r[1])).add(a[2].scale(r[2]))
}
fn cross_rc(a: CV, r: [f64; 3]) -> CV {
    // a × r
    [a[1].scale(r[2]).sub(a[2].scale(r[1])), a[2].scale(r[0]).sub(a[0].scale(r[2])), a[0].scale(r[1]).sub(a[1].scale(r[0]))]
}
fn mirror(v: CV) -> CV {
    [v[0], v[1], v[2].scale(-1.0)]
}

/// F(r̂) = η N⊥ + L × r̂, so the far field is E = −jk e^{−jkr}/(4πr) F.
fn ffun(nv: CV, lv: CV, r: [f64; 3]) -> CV {
    let eta = (MU0 / EPS0).sqrt();
    let nr = cdot_r(nv, r);
    let lr = cross_rc(lv, r);
    [0, 1, 2].map(|c| nv[c].sub(nr.scale(r[c])).scale(eta).add(lr[c]))
}

/// Peak |E| [V/m] at 3 m and 10 m, by frequency: free space (strongest direction), or over a
/// ground plane (direct + image ray, receive height 1–4 m, turntable, H and V polarisation).
#[allow(clippy::too_many_arguments)]
fn emissions(model: &Model, mesh: &Mesh, s: &Setup, ff: &Dft, ar: &[Vec<f32>], ai: &[Vec<f32>], vs: &[Cx], res: &mut Results, warnings: &mut Vec<String>) -> Result<()> {
    let st = &model.analysis;
    let nn3 = 3 * s.nn;
    let eta = (MU0 / EPS0).sqrt();
    // ground plane geometry: height of the transform centre above the ground
    let zmin_mm = mesh
        .elems
        .iter()
        .enumerate()
        .filter(|(e, _)| mesh.elem_region[*e] == REGION_SOLID)
        .flat_map(|(_, en)| en.iter().map(|&n| mesh.nodes[n as usize][2]))
        .fold(f64::INFINITY, f64::min);
    let rows: Vec<(f64, f64, f64)> = ff
        .freqs
        .par_iter()
        .enumerate()
        .map(|(k, &f)| -> Result<(f64, f64, f64)> {
            let c = cdiv(Cx::new(1.0, 0.0), vs[k]);
            let mut re = vec![0.0f32; nn3];
            let mut im = vec![0.0f32; nn3];
            for (q, &idx) in ff.idx.iter().enumerate() {
                let (r, i) = normalise(ar[k][q], ai[k][q], c);
                re[idx as usize] = r;
                im[idx as usize] = i;
            }
            let nt = fdtd::Ntff::new(s, &re, &im, 1.0, f)?;
            let k0 = nt.k0;
            if !st.ground_plane {
                // strongest direction on a 5° grid: |E|peak = √(2ηU)/r
                let mut umax: f64 = 0.0;
                for it in 0..36 {
                    for ip in 0..72 {
                        let th = (it as f64 + 0.5) * std::f64::consts::PI / 36.0;
                        let ph = ip as f64 * std::f64::consts::TAU / 72.0;
                        umax = umax.max(nt.u(th, ph));
                    }
                }
                let e = |r: f64| (2.0 * eta * umax).sqrt() / r;
                return Ok((f, e(3.0), e(10.0)));
            }
            let centre_z_mm = (nt.centre[2] / s.h - s.pad as f64) * mesh.h + mesh.origin[2];
            let hc = st.eut_height_m.max(0.0) + ((centre_z_mm - zmin_mm) * 1e-3).max(0.0);
            let field_at = |d: f64| -> f64 {
                let mut best: f64 = 0.0;
                for ia in 0..24 {
                    let psi = ia as f64 * std::f64::consts::TAU / 24.0;
                    let (cp, sp) = (psi.cos(), psi.sin());
                    for ih in 0..=12 {
                        let hr = 1.0 + 0.25 * ih as f64;
                        let rx = [d * cp, d * sp, hr];
                        let mut tot = [Cx::default(); 3];
                        for (sz, image) in [(hc, false), (-hc, true)] {
                            let rv = [rx[0], rx[1], rx[2] - sz];
                            let rl = (rv[0] * rv[0] + rv[1] * rv[1] + rv[2] * rv[2]).sqrt();
                            let rh = [rv[0] / rl, rv[1] / rl, rv[2] / rl];
                            let fv = if image {
                                // image currents: N' = −M N(M r̂), L' = M L(M r̂)
                                let (nv, lv) = nt.nl([rh[0], rh[1], -rh[2]]);
                                let nv = mirror(nv).map(|x| x.scale(-1.0));
                                let lv = mirror(lv);
                                ffun(nv, lv, rh)
                            } else {
                                let (nv, lv) = nt.nl(rh);
                                ffun(nv, lv, rh)
                            };
                            // −jk e^{−jkr} / (4π r)
                            let ph = -k0 * rl;
                            let g = Cx::new(0.0, -k0 / (4.0 * std::f64::consts::PI * rl)).mul(Cx::new(ph.cos(), ph.sin()));
                            for cc in 0..3 {
                                tot[cc] = tot[cc].add(fv[cc].mul(g));
                            }
                        }
                        let eh = tot[0].scale(-sp).add(tot[1].scale(cp)).norm2().sqrt();
                        let ev = tot[2].norm2().sqrt();
                        best = best.max(eh).max(ev);
                    }
                }
                best
            };
            Ok((f, field_at(3.0), field_at(10.0)))
        })
        .collect::<Result<Vec<_>>>()?;
    let dbuv = |e_peak: f64| 20.0 * (e_peak / 2f64.sqrt() * 1e6).max(1e-12).log10();
    let mhz: Vec<f64> = rows.iter().map(|r| r.0 / 1e9).collect();
    let e3: Vec<f64> = rows.iter().map(|r| dbuv(r.1)).collect();
    let e10: Vec<f64> = rows.iter().map(|r| dbuv(r.2)).collect();
    let fcc: Vec<f64> = rows.iter().map(|r| fdtd::fcc_class_b_3m(r.0)).collect();
    let cis: Vec<Option<f64>> = rows.iter().map(|r| fdtd::cispr32_class_b_10m(r.0)).collect();
    let site = if st.ground_plane { "over ground plane, 1–4 m height scan" } else { "free space, strongest direction" };
    res.series.push((format!("E at 3 m [dBµV/m] ({site})"), mhz.iter().zip(&e3).map(|(f, v)| (*f, *v)).collect(), true));
    res.series.push(("FCC class B limit, 3 m [dBµV/m]".into(), mhz.iter().zip(&fcc).map(|(f, v)| (*f, *v)).collect(), true));
    res.series.push((format!("E at 10 m [dBµV/m] ({site})"), mhz.iter().zip(&e10).map(|(f, v)| (*f, *v)).collect(), false));
    let cis_pts: Vec<(f64, f64)> = mhz.iter().zip(&cis).filter_map(|(f, c)| c.map(|v| (*f, v))).collect();
    if !cis_pts.is_empty() {
        res.series.push(("CISPR 32 class B limit, 10 m [dBµV/m]".into(), cis_pts, false));
    }
    let (i3, p3) = e3.iter().enumerate().fold((0, f64::NEG_INFINITY), |b, (i, &v)| if v > b.1 { (i, v) } else { b });
    res.quantity("Peak emission at 3 m", p3, "dBµV/m");
    res.quantity("Peak emission at", mhz[i3] * 1e3, "MHz");
    let m_fcc = e3.iter().zip(&fcc).map(|(e, l)| l - e).fold(f64::INFINITY, f64::min);
    res.quantity("Margin to FCC class B, 3 m (negative = fails)", m_fcc, "dB");
    let mut worst = m_fcc;
    let m_cis = e10.iter().zip(&cis).filter_map(|(e, l)| l.map(|l| l - e)).fold(f64::INFINITY, f64::min);
    if m_cis.is_finite() {
        res.quantity("Margin to CISPR 32 class B, 10 m (negative = fails)", m_cis, "dB");
        worst = worst.min(m_cis);
    }
    res.quantity("Worst emission margin", worst, "dB");
    let nt = fdtd::Ntff::new(s, &vec![0.0; nn3], &vec![0.0; nn3], 1.0, ff.freqs[0])?;
    if nt.outside > 0 {
        warnings.push(format!("{} cells of material lie within 2 cells of the enclosure boundary, outside the far-field surface: enlarge the enclosure", nt.outside));
    }
    warnings.push(format!(
        "emissions: {site}, for the port source voltages as a sine at each frequency (peak voltage → rms field); a pre-compliance estimate, expect ±6 dB against a real test site"
    ));
    Ok(())
}
