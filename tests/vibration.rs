//! Vibration analyses vs closed-form Euler-Bernoulli cantilever results.
use serde_json::{Value, json};
use simcraft::session::{Shared, new_core};

fn call(c: &Shared, t: &str, a: Value) -> Value {
    let o = c.call(t, &a).unwrap_or_else(|e| panic!("{t}: {e:#}"));
    serde_json::from_str(&o.text).unwrap_or(Value::String(o.text))
}
fn close(name: &str, got: f64, want: f64, tol: f64) {
    let err = (got - want).abs() / want.abs();
    println!("{name}: got {got:.6e}, theory {want:.6e}, error {:.2}%", err * 100.0);
    assert!(err < tol, "{name} off by {:.1}%", err * 100.0);
}
fn q(r: &Value, name: &str) -> f64 {
    r["quantities"].as_array().unwrap().iter().find(|x| x["name"].as_str().unwrap().starts_with(name)).unwrap_or_else(|| panic!("no {name}"))["value"].as_f64().unwrap()
}
/// 100 × 10 × 6 mm steel cantilever (thin in z so the first mode bends in z).
fn beam() -> Shared {
    let c = new_core();
    call(&c, "add_box", json!({"name": "Beam", "size": [100, 10, 6]}));
    call(&c, "set_mesh", json!({"element_size_mm": 1}));
    call(&c, "add_fixed_support", json!({"faces": ["Beam.xmin"]}));
    c
}
fn f1(c: &Shared) -> f64 {
    let kind = call(c, "get_model", json!({}))["analysis"]["type"].clone();
    call(c, "set_analysis", json!({"type": "modal", "modes": 2}));
    let r = call(c, "solve", json!({}));
    call(c, "set_analysis", json!({"type": kind}));
    r["modes"][0]["frequency_hz"].as_f64().unwrap()
}

#[test]
fn response_spectrum_flat_matches_gravity_ratio() {
    let c = beam();
    call(&c, "add_gravity", json!({"acceleration": [0, 0, -9.80665]}));
    let stat = call(&c, "solve", json!({}))["fields"]["deformation_z"]["min"].as_f64().unwrap().abs();
    call(&c, "clear_loads", json!({}));
    call(&c, "add_fixed_support", json!({"faces": ["Beam.xmin"]}));
    call(&c, "set_analysis", json!({"type": "response_spectrum", "table": [[1, 1.0], [100000, 1.0]], "direction": "z", "damping_ratio": 0.02}));
    let r = call(&c, "solve", json!({}));
    let tip = r["fields"]["total_deformation"]["max"].as_f64().unwrap();
    // mode-1 tip / static = (∫ψ/L · ψ(L)) / (β⁴/8) = 1.5660 / 1.5452 = 1.0134
    close("spectrum tip / static 1g", tip / stat, 1.0134, 0.02);
    println!("cumulative effective mass {:.1}%", q(&r, "Cumulative effective mass"));
}

#[test]
fn harmonic_static_limit_and_resonance() {
    let c = beam();
    call(&c, "add_force", json!({"faces": ["Beam.xmax"], "vector": [0, 0, -100]}));
    let stat = call(&c, "solve", json!({}))["fields"]["deformation_z"]["min"].as_f64().unwrap().abs();
    let f1 = f1(&c);
    call(&c, "set_analysis", json!({"type": "harmonic_response", "freq_min": 1, "freq_max": 1.0001, "freq_steps": 20, "damping_ratio": 0.02}));
    let low = q(&call(&c, "solve", json!({})), "Peak displacement amplitude");
    close("harmonic at 1 Hz = static", low, stat, 0.02);
    call(&c, "set_analysis", json!({"freq_min": 0.9 * f1, "freq_max": 1.1 * f1, "freq_steps": 200}));
    let r = call(&c, "solve", json!({}));
    close("resonance frequency", q(&r, "Peak response frequency"), f1, 0.005);
    // resonance amplification of the tip: (12/β⁴)/(2ζ) = 0.9707/(2·0.02)
    close("resonance amplification", q(&r, "Peak displacement amplitude") / stat, 0.9707 / 0.04, 0.04);
}

#[test]
fn random_vibration_matches_white_noise_closed_form() {
    let c = beam();
    let f1 = f1(&c);
    let s_g = 0.01; // g²/Hz
    call(&c, "set_analysis", json!({"type": "random_vibration", "table": [[1, s_g], [20000, s_g]], "direction": "z", "damping_ratio": 0.02}));
    let r = call(&c, "solve", json!({}));
    let w1 = std::f64::consts::TAU * f1;
    let s = s_g * 9.80665f64.powi(2);
    // Miles: σ_rel = √(π/2·f₁·Q·S)/ω₁² = √(S / (8ζω₁³)) for S one-sided per Hz;
    // tip participation Γ₁φ₁(L) = 2·0.7830 = 1.566 for a uniform cantilever
    let theory_mm = 1.566 * (s / (8.0 * 0.02 * w1.powi(3))).sqrt() * 1e3;
    close("1σ tip displacement", q(&r, "1σ max displacement"), theory_mm, 0.04);
    close("zero-crossing ≈ f1", q(&r, "Zero-crossing frequency"), f1, 0.05);
    close("input Grms", q(&r, "Input acceleration (RMS)"), (s_g * (20000.0 - 1.0)).sqrt(), 0.01);
}

/// Uniaxial bar 100 MPa: Basquin life, Goodman mean-stress correction, safety factor.
#[test]
fn fatigue_basquin_goodman_bar() {
    let c = new_core();
    call(&c, "define_material", json!({"name": "Fatigue steel", "ultimate_strength_mpa": 500, "yield_strength_mpa": 350,
        "fatigue_strength_coefficient_mpa": 900, "fatigue_exponent": -0.1, "endurance_limit_mpa": 50}));
    call(&c, "add_box", json!({"name": "Bar", "size": [40, 10, 10]}));
    call(&c, "set_material", json!({"body": "Bar", "material": "Fatigue steel"}));
    call(&c, "set_mesh", json!({"element_size_mm": 2}));
    call(&c, "add_symmetry", json!({"faces": ["Bar.xmin", "Bar.ymin", "Bar.zmin"]}));
    call(&c, "add_force", json!({"faces": ["Bar.xmax"], "vector": [10000, 0, 0]})); // 100 MPa
    call(&c, "solve", json!({}));
    let life = |r: &Value| q(r, "Fatigue minimum life");
    // fully reversed 100 MPa: N = ½(100/900)^(1/-0.1)
    let r = call(&c, "fatigue_life", json!({"loading": "fully_reversed", "correction": "none"}));
    close("fully reversed life", life(&r), 0.5 * (100.0f64 / 900.0).powf(-10.0), 0.03);
    // σa' at 1e6: 900·(2e6)^-0.1 = 211.5 MPa → SF 2.115
    close("safety factor at 1e6", q(&r, "Fatigue minimum safety factor"), 900.0 * 2e6f64.powf(-0.1) / 100.0, 0.02);
    // zero-based: σa = σm = 50, Goodman σar = 50/(1-50/500) = 55.6 > Se=50
    let r = call(&c, "fatigue_life", json!({"loading": "zero_based", "correction": "goodman"}));
    close("zero-based Goodman life", life(&r), 0.5 * ((50.0 / 0.9) / 900.0f64).powf(-10.0), 0.03);
    // without the correction σa = 50 ≤ Se → infinite
    let r = call(&c, "fatigue_life", json!({"loading": "zero_based", "correction": "none"}));
    assert!(life(&r) >= 1e12, "below endurance limit must be infinite");
}

/// Steinberg 3-band damage on the random result: compare against the hand sum at the peak node.
#[test]
fn fatigue_random_steinberg() {
    let c = beam();
    call(&c, "set_analysis", json!({"type": "random_vibration", "table": [[10, 0.04], [2000, 0.04]], "direction": "z"}));
    let r = call(&c, "solve", json!({}));
    let s1 = q(&r, "1σ max von Mises");
    let nu0 = q(&r, "Zero-crossing frequency");
    let f = call(&c, "fatigue_life", json!({"exposure_time_s": 3600}));
    // structural steel: estimated σf' = Su + 345, b = -0.085
    let m = call(&c, "list_materials", json!({}));
    let su = m.as_array().unwrap().iter().find(|x| x["name"] == "Structural Steel").unwrap()["ultimate_strength_mpa"].as_f64().unwrap();
    let (sf, b) = (su + 345.0, -0.085);
    let n = |s: f64| 0.5 * (s / sf).powf(1.0 / b);
    let d = nu0 * 3600.0 * (0.683 / n(s1) + 0.271 / n(2.0 * s1) + 0.0433 / n(3.0 * s1));
    close("Steinberg damage", q(&f, "Fatigue maximum damage"), d, 0.02);
    assert!(f["warnings"].as_array().unwrap().iter().any(|w| w.as_str().unwrap().contains("estimated")));
}
