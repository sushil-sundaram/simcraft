//! Stress-life (S-N) fatigue post-processing of a solved structural or vibration result.
//!
//! * Basquin curve σa = σf'·(2N)^b per material (`Material::sn_curve`), with an endurance limit
//!   below which life is infinite (capped at 1e12 cycles).
//! * Constant-amplitude loading (static, harmonic, spectrum results): the equivalent stress is a
//!   signed von Mises (sign of the hydrostatic stress); the cycle is fully reversed, zero-based
//!   or a given stress ratio R, and a mean-stress correction (Goodman, Gerber, Soderberg) gives the
//!   equivalent fully reversed amplitude.
//! * Random vibration (1σ result): Steinberg's three-band method — 68.3 % of cycles at 1σ,
//!   27.1 % at 2σ and 4.33 % at 3σ, at the zero-crossing rate, over an exposure time. The
//!   endurance limit is ignored for random loading (conservative).
//!
//! Writes nodal fields Fatigue Life [cycles], Fatigue Damage [-] and Fatigue Safety Factor [-].

use crate::mesh::Mesh;
use crate::model::AnalysisType;
use crate::results::{Field, Quantity, Results};
use anyhow::{Result, bail};

pub const INFINITE: f64 = 1e12;
const MAX_SAFETY: f64 = 15.0;

#[derive(Clone, Debug)]
pub struct FatigueOptions {
    /// "fully_reversed", "zero_based" or "ratio"
    pub loading: String,
    /// Stress ratio σmin/σmax when loading = "ratio".
    pub ratio: f64,
    /// "goodman", "gerber", "soderberg" or "none"
    pub correction: String,
    /// Multiplier on the solved stresses (load scaling).
    pub scale: f64,
    /// Required life in cycles (constant amplitude).
    pub design_life: f64,
    /// Exposure time in seconds (random vibration).
    pub exposure_s: f64,
}

impl Default for FatigueOptions {
    fn default() -> Self {
        FatigueOptions { loading: "fully_reversed".into(), ratio: -1.0, correction: "goodman".into(), scale: 1.0, design_life: 1e6, exposure_s: 3600.0 }
    }
}

fn signed_vm(s: &[f64; 6]) -> f64 {
    let vm = (0.5 * ((s[0] - s[1]).powi(2) + (s[1] - s[2]).powi(2) + (s[2] - s[0]).powi(2)) + 3.0 * (s[3] * s[3] + s[4] * s[4] + s[5] * s[5])).sqrt();
    if s[0] + s[1] + s[2] < 0.0 { -vm } else { vm }
}

/// Basquin life for a fully reversed amplitude (no endurance cut-off, not capped: damage sums
/// need the true value).
fn basquin(sa: f64, sigma_f: f64, b: f64) -> f64 {
    if sa <= 0.0 {
        return f64::INFINITY;
    }
    0.5 * (sa / sigma_f).powf(1.0 / b)
}

pub fn apply(res: &Results, mesh: &Mesh, o: &FatigueOptions) -> Result<Results> {
    let Some(stress) = res.stress.as_ref() else {
        bail!("fatigue needs a stress result: solve a static structural, harmonic, response spectrum or random vibration analysis first");
    };
    let random = res.analysis == AnalysisType::RandomVibration;
    let amplitude_result = matches!(res.analysis, AnalysisType::HarmonicResponse | AnalysisType::ResponseSpectrum);
    if o.scale <= 0.0 || o.design_life < 1.0 || o.exposure_s <= 0.0 {
        bail!("scale and exposure time must be > 0 and design_life ≥ 1");
    }
    if !(-1e6..1.0).contains(&o.ratio) {
        bail!("stress ratio R must be < 1");
    }
    let (loading, correction) = (o.loading.as_str(), o.correction.as_str());
    if !matches!(loading, "fully_reversed" | "zero_based" | "ratio") {
        bail!("loading must be fully_reversed, zero_based or ratio");
    }
    if !matches!(correction, "goodman" | "gerber" | "soderberg" | "none") {
        bail!("correction must be goodman, gerber, soderberg or none");
    }
    let nu0 = if random {
        res.quantities.iter().find(|q| q.name.starts_with("Zero-crossing")).map(|q| q.value).filter(|v| *v > 0.0).ok_or_else(|| anyhow::anyhow!("random result has no zero-crossing frequency; solve again"))?
    } else {
        0.0
    };

    // materials touching each node (solid elements only)
    let nn = mesh.nodes.len();
    let mut node_mats: Vec<Vec<u16>> = vec![Vec::new(); nn];
    for (e, el) in mesh.elems.iter().enumerate() {
        if mesh.elem_region[e] != crate::mesh::REGION_SOLID {
            continue;
        }
        for &n in el {
            let v = &mut node_mats[n as usize];
            if !v.contains(&mesh.elem_mat[e]) {
                v.push(mesh.elem_mat[e]);
            }
        }
    }
    let curves: Vec<_> = mesh.materials.iter().map(|m| m.sn_curve()).collect();
    let mut estimated: Vec<String> = Vec::new();
    for (i, c) in curves.iter().enumerate() {
        if c.estimated && node_mats.iter().any(|v| v.contains(&(i as u16))) && !estimated.contains(&mesh.materials[i].name) {
            estimated.push(mesh.materials[i].name.clone());
        }
    }

    let mut life = vec![f64::NAN; nn];
    let mut damage = vec![f64::NAN; nn];
    let mut safety = vec![f64::NAN; nn];
    let mut static_fail = 0usize;
    for n in 0..nn {
        if node_mats[n].is_empty() {
            continue;
        }
        let (mut best_life, mut best_dmg, mut best_sf) = (f64::INFINITY, 0.0f64, f64::INFINITY);
        for &mi in &node_mats[n] {
            let m = &mesh.materials[mi as usize];
            let c = &curves[mi as usize];
            let (l, d, sf) = if random {
                // 1σ von Mises; Steinberg bands
                let s1 = signed_vm(&stress[n]).abs() * o.scale;
                let rate = nu0 * o.exposure_s;
                let d = rate * (0.683 / basquin(s1, c.sigma_f, c.b) + 0.271 / basquin(2.0 * s1, c.sigma_f, c.b) + 0.0433 / basquin(3.0 * s1, c.sigma_f, c.b));
                // damage ∝ σ^(-1/b): factor k with D(kσ) = 1 is D^b
                let sf = if d > 0.0 { d.powf(c.b) } else { f64::INFINITY };
                let l = if d > 0.0 { rate / d } else { f64::INFINITY };
                (l, d, sf)
            } else {
                let smax = signed_vm(&stress[n]) * o.scale;
                let (sa, sm) = if amplitude_result {
                    (smax.abs(), 0.0)
                } else {
                    match loading {
                        "zero_based" => (smax.abs() / 2.0, smax / 2.0),
                        "ratio" => (smax.abs() * (1.0 - o.ratio) / 2.0, smax * (1.0 + o.ratio) / 2.0),
                        _ => (smax.abs(), 0.0),
                    }
                };
                // compressive means do not reduce life (conventional, conservative)
                let sm = sm.max(0.0);
                let su = m.ultimate_strength_mpa.max(1e-9);
                let sy = m.yield_strength_mpa.max(1e-9);
                let denom = match correction {
                    "goodman" => 1.0 - sm / su,
                    "gerber" => 1.0 - (sm / su).powi(2),
                    "soderberg" => 1.0 - sm / sy,
                    _ => 1.0,
                };
                if denom <= 0.0 {
                    static_fail += 1;
                    (0.0, f64::INFINITY, 0.0)
                } else {
                    let sar = sa / denom;
                    let l = if sar <= c.endurance { f64::INFINITY } else { basquin(sar, c.sigma_f, c.b) };
                    let d = o.design_life / l;
                    let allow = (c.sigma_f * (2.0 * o.design_life).powf(c.b)).max(c.endurance);
                    let sf = if sar > 0.0 { allow / sar } else { f64::INFINITY };
                    (l, d, sf)
                }
            };
            if l < best_life {
                best_life = l;
            }
            best_dmg = best_dmg.max(d);
            best_sf = best_sf.min(sf);
        }
        life[n] = best_life.min(INFINITE);
        damage[n] = best_dmg.min(1e12);
        safety[n] = best_sf.min(MAX_SAFETY);
    }

    let fin = |v: &[f64], f: fn(f64, f64) -> f64, init: f64| v.iter().filter(|x| x.is_finite()).fold(init, |a, &b| f(a, b));
    let min_life = fin(&life, f64::min, INFINITE);
    let max_dmg = fin(&damage, f64::max, 0.0);
    let min_sf = fin(&safety, f64::min, MAX_SAFETY);

    let mut out = res.clone();
    for (f, v) in [(Field::FatigueLife, life), (Field::FatigueDamage, damage), (Field::FatigueSafety, safety)] {
        out.extra.insert(f, v.into_iter().map(|x| if x.is_finite() { x } else { 0.0 }).collect());
    }
    out.quantities.retain(|q| !q.name.starts_with("Fatigue"));
    let q = |name: &str, value: f64, unit: &str| Quantity { name: name.into(), value, unit: unit.into() };
    out.quantities.push(q("Fatigue minimum life", min_life, "cycles"));
    if random {
        out.quantities.push(q("Fatigue exposure time", o.exposure_s, "s"));
        out.quantities.push(q("Fatigue minimum life (time)", min_life / nu0 / 3600.0, "h"));
    } else {
        out.quantities.push(q("Fatigue design life", o.design_life, "cycles"));
    }
    out.quantities.push(q("Fatigue maximum damage", max_dmg, ""));
    out.quantities.push(q("Fatigue minimum safety factor", min_sf, ""));
    let how = if random {
        format!("Steinberg 3-band, ν0 = {nu0:.1} Hz, exposure {:.0} s", o.exposure_s)
    } else if amplitude_result {
        "fully reversed amplitude".to_string()
    } else if loading == "ratio" {
        format!("R = {}, {correction}", o.ratio)
    } else {
        format!("{}, {correction}", loading.replace('_', " "))
    };
    out.note = format!("{}{}Fatigue: {how}{}", out.note, if out.note.is_empty() { "" } else { "; " }, if o.scale != 1.0 { format!(", stresses × {}", o.scale) } else { String::new() });
    out.warnings.retain(|w| !w.starts_with("Fatigue"));
    if !estimated.is_empty() {
        out.warnings.push(format!("Fatigue: S-N data estimated from the ultimate strength for {} — enter measured σf', b and endurance limit in the material for design use.", estimated.join(", ")));
    }
    if static_fail > 0 {
        out.warnings.push(format!("Fatigue: {static_fail} nodes have a mean stress beyond the {correction} line (static failure; life 0)."));
    }
    if amplitude_result && loading != "fully_reversed" {
        out.warnings.push("Fatigue: harmonic/spectrum results are amplitudes, so they are treated as fully reversed.".into());
    }
    Ok(out)
}
