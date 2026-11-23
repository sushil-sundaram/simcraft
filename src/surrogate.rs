//! Gaussian-process surrogates of study responses (response surfaces).
//!
//! * Inputs normalised to [0,1] per parameter, output standardised.
//! * ARD squared-exponential kernel k(x,x') = σf² exp(−½ Σ (xᵢ−x'ᵢ)²/ℓᵢ²) + σn² δ.
//! * Hyper-parameters (log ℓᵢ, log σf², log σn²) by maximising the log marginal likelihood with
//!   multi-start Nelder–Mead.
//! * Closed-form leave-one-out residuals rᵢ = [K⁻¹y]ᵢ / [K⁻¹]ᵢᵢ → LOO RMSE and Q².
//! * Expected improvement and probability of feasibility for surrogate-based optimisation.
//!
//! Predictions are estimates: every surrogate answer is labelled "surrogate, not verified".

use anyhow::{Result, bail};

#[derive(Clone, Debug)]
pub struct Gp {
    pub x: Vec<Vec<f64>>,
    y_mean: f64,
    y_std: f64,
    /// length scales (normalised units), signal and noise variance (standardised output units)
    pub ls: Vec<f64>,
    pub sf2: f64,
    pub sn2: f64,
    alpha: Vec<f64>,
    /// lower Cholesky factor of K (row-major n×n)
    l: Vec<f64>,
    pub loo_rmse: f64,
    pub q2: f64,
}

fn kern(a: &[f64], b: &[f64], ls: &[f64], sf2: f64) -> f64 {
    let mut s = 0.0;
    for k in 0..a.len() {
        let d = (a[k] - b[k]) / ls[k];
        s += d * d;
    }
    sf2 * (-0.5 * s).exp()
}

/// In-place Cholesky; returns false if not positive definite.
fn cholesky(a: &mut [f64], n: usize) -> bool {
    for j in 0..n {
        let mut d = a[j * n + j];
        for k in 0..j {
            d -= a[j * n + k] * a[j * n + k];
        }
        if d <= 0.0 || !d.is_finite() {
            return false;
        }
        let d = d.sqrt();
        a[j * n + j] = d;
        for i in j + 1..n {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= a[i * n + k] * a[j * n + k];
            }
            a[i * n + j] = s / d;
        }
        for i in 0..j {
            a[i * n + j] = 0.0;
        }
    }
    true
}

fn solve_l(l: &[f64], n: usize, b: &[f64]) -> Vec<f64> {
    let mut y = b.to_vec();
    for i in 0..n {
        let mut s = y[i];
        for k in 0..i {
            s -= l[i * n + k] * y[k];
        }
        y[i] = s / l[i * n + i];
    }
    y
}

fn solve_lt(l: &[f64], n: usize, b: &[f64]) -> Vec<f64> {
    let mut x = b.to_vec();
    for i in (0..n).rev() {
        let mut s = x[i];
        for k in i + 1..n {
            s -= l[k * n + i] * x[k];
        }
        x[i] = s / l[i * n + i];
    }
    x
}

/// Factorised kernel matrix for given hyper-parameters: (L, α = K⁻¹y, log marginal likelihood).
fn factor(x: &[Vec<f64>], y: &[f64], ls: &[f64], sf2: f64, sn2: f64) -> Option<(Vec<f64>, Vec<f64>, f64)> {
    let n = x.len();
    let mut k = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let v = kern(&x[i], &x[j], ls, sf2);
            k[i * n + j] = v;
            k[j * n + i] = v;
        }
        k[i * n + i] += sn2 + 1e-10 * sf2;
    }
    if !cholesky(&mut k, n) {
        return None;
    }
    let alpha = solve_lt(&k, n, &solve_l(&k, n, y));
    let logdet: f64 = (0..n).map(|i| k[i * n + i].ln()).sum::<f64>() * 2.0;
    let fit: f64 = y.iter().zip(&alpha).map(|(a, b)| a * b).sum();
    let lml = -0.5 * fit - 0.5 * logdet - 0.5 * n as f64 * std::f64::consts::TAU.ln();
    Some((k, alpha, lml))
}

/// Bounded Nelder–Mead minimisation in a box (used for hyper-parameters and candidates).
pub fn nelder_mead(f: &dyn Fn(&[f64]) -> f64, x0: &[f64], step: f64, lo: f64, hi: f64, iters: usize) -> (Vec<f64>, f64) {
    let n = x0.len();
    let clamp = |v: Vec<f64>| v.into_iter().map(|x| x.clamp(lo, hi)).collect::<Vec<_>>();
    let mut s: Vec<(Vec<f64>, f64)> = vec![(clamp(x0.to_vec()), 0.0)];
    for i in 0..n {
        let mut v = x0.to_vec();
        v[i] += step;
        s.push((clamp(v), 0.0));
    }
    for p in s.iter_mut() {
        p.1 = f(&p.0);
    }
    for _ in 0..iters {
        s.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        if (s[n].1 - s[0].1).abs() < 1e-10 * (1.0 + s[0].1.abs()) {
            break;
        }
        let c: Vec<f64> = (0..n).map(|k| s[..n].iter().map(|p| p.0[k]).sum::<f64>() / n as f64).collect();
        let at = |t: f64| clamp(c.iter().zip(&s[n].0).map(|(a, w)| a + t * (a - w)).collect());
        let xr = at(1.0);
        let fr = f(&xr);
        if fr < s[0].1 {
            let xe = at(2.0);
            let fe = f(&xe);
            s[n] = if fe < fr { (xe, fe) } else { (xr, fr) };
        } else if fr < s[n - 1].1 {
            s[n] = (xr, fr);
        } else {
            let xc = at(if fr < s[n].1 { 0.5 } else { -0.5 });
            let fc = f(&xc);
            if fc < s[n].1.min(fr) {
                s[n] = (xc, fc);
            } else {
                let b = s[0].0.clone();
                for p in s.iter_mut().skip(1) {
                    p.0 = p.0.iter().zip(&b).map(|(x, y)| y + 0.5 * (x - y)).collect();
                    p.1 = f(&p.0);
                }
            }
        }
    }
    s.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    s.swap_remove(0)
}

impl Gp {
    /// Fit to inputs `x` (already normalised to [0,1]) and outputs `y`.
    pub fn fit(x: Vec<Vec<f64>>, y: &[f64]) -> Result<Gp> {
        let n = x.len();
        if n < 3 {
            bail!("a surrogate needs at least 3 solved designs (have {n})");
        }
        let d = x[0].len();
        let y_mean = y.iter().sum::<f64>() / n as f64;
        let var = y.iter().map(|v| (v - y_mean).powi(2)).sum::<f64>() / n as f64;
        let y_std = if var > 0.0 { var.sqrt() } else { 1.0 };
        let ys: Vec<f64> = y.iter().map(|v| (v - y_mean) / y_std).collect();
        // θ = [log ℓ₁..ℓ_d, log σf², log σn²]
        let unpack = |t: &[f64]| -> (Vec<f64>, f64, f64) { (t[..d].iter().map(|v| v.exp()).collect(), t[d].exp(), t[d + 1].exp()) };
        let nlml = |t: &[f64]| -> f64 {
            let (ls, sf2, sn2) = unpack(t);
            match factor(&x, &ys, &ls, sf2, sn2) {
                Some((_, _, lml)) if lml.is_finite() => -lml,
                _ => 1e30,
            }
        };
        let mut best: Option<(Vec<f64>, f64)> = None;
        for &l0 in &[0.25f64, 0.6, 1.5] {
            for &n0 in &[1e-6f64, 1e-3, 0.05] {
                let mut t0 = vec![l0.ln(); d];
                t0.push(0.0);
                t0.push(n0.ln());
                let r = nelder_mead(&nlml, &t0, 0.7, (1e-8f64).ln(), (50.0f64).ln(), 250 * (d + 2));
                if best.as_ref().is_none_or(|b| r.1 < b.1) {
                    best = Some(r);
                }
            }
        }
        let (t, _) = best.unwrap();
        let (ls, sf2, mut sn2) = unpack(&t);
        let mut f = factor(&x, &ys, &ls, sf2, sn2);
        while f.is_none() && sn2 < 1.0 {
            sn2 = (sn2 * 10.0).max(1e-8);
            f = factor(&x, &ys, &ls, sf2, sn2);
        }
        let Some((l, alpha, _)) = f else { bail!("could not fit the surrogate (singular kernel)") };
        // closed-form leave-one-out: diag(K⁻¹) from L⁻¹
        let mut kinv_diag = vec![0.0; n];
        for i in 0..n {
            let mut e = vec![0.0; n];
            e[i] = 1.0;
            // column i of L⁻¹; diag(K⁻¹)ᵢ = ‖L⁻¹ eᵢ‖² since K⁻¹ = L⁻ᵀ L⁻¹
            let c = solve_l(&l, n, &e);
            kinv_diag[i] = c.iter().map(|v| v * v).sum();
        }
        let res: Vec<f64> = (0..n).map(|i| alpha[i] / kinv_diag[i] * y_std).collect();
        let loo_rmse = (res.iter().map(|r| r * r).sum::<f64>() / n as f64).sqrt();
        let ss = var * n as f64;
        let q2 = if ss > 0.0 { 1.0 - res.iter().map(|r| r * r).sum::<f64>() / ss } else { 1.0 };
        Ok(Gp { x, y_mean, y_std, ls, sf2, sn2, alpha, l, loo_rmse, q2 })
    }

    /// Predictive mean and standard deviation at normalised input `u`.
    pub fn predict(&self, u: &[f64]) -> (f64, f64) {
        let n = self.x.len();
        let ks: Vec<f64> = self.x.iter().map(|xi| kern(xi, u, &self.ls, self.sf2)).collect();
        let mean: f64 = ks.iter().zip(&self.alpha).map(|(a, b)| a * b).sum();
        let v = solve_l(&self.l, n, &ks);
        let var = (self.sf2 - v.iter().map(|x| x * x).sum::<f64>()).max(0.0);
        (self.y_mean + mean * self.y_std, var.sqrt() * self.y_std)
    }

    /// Relative importance of each input (1/ℓ, normalised to sum 1).
    pub fn relevance(&self) -> Vec<f64> {
        let inv: Vec<f64> = self.ls.iter().map(|l| 1.0 / l).collect();
        let s: f64 = inv.iter().sum();
        inv.into_iter().map(|v| v / s.max(1e-300)).collect()
    }
}

fn phi(z: f64) -> f64 {
    (-0.5 * z * z).exp() / (std::f64::consts::TAU).sqrt()
}

/// Standard normal CDF (Abramowitz–Stegun 7.1.26 via erf, |error| < 1.5e-7).
pub fn big_phi(z: f64) -> f64 {
    let x = z / std::f64::consts::SQRT_2;
    let t = 1.0 / (1.0 + 0.327_591_1 * x.abs());
    let y = 1.0 - (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t - 0.284_496_736) * t + 0.254_829_592) * t * (-x * x).exp();
    0.5 * (1.0 + if x >= 0.0 { y } else { -y })
}

/// Expected improvement below `best` (minimisation) for mean μ and std σ.
pub fn expected_improvement(mu: f64, sigma: f64, best: f64) -> f64 {
    if sigma <= 1e-12 {
        return (best - mu).max(0.0);
    }
    let z = (best - mu) / sigma;
    (best - mu) * big_phi(z) + sigma * phi(z)
}
