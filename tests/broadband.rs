//! P2.6 broadband FDTD: lumped ports, S-parameters, input impedance and emissions.
use serde_json::{Value, json};
use simcraft::session::{Shared, new_core};

fn call(c: &Shared, tool: &str, args: Value) -> Value {
    let r = c.call(tool, &args).unwrap_or_else(|e| panic!("{tool}({args}) failed: {e:#}"));
    serde_json::from_str(&r.text).unwrap_or_else(|_| json!({ "text": r.text }))
}

fn solve(c: &Shared) -> Value {
    let _ = call(c, "solve", json!({}));
    while c.is_busy() {
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    let st = call(c, "get_status", json!({}));
    if let Some(e) = st["last_error"].as_str() {
        panic!("async job failed: {e}");
    }
    call(c, "get_results", json!({}))
}

fn qty(r: &Value, name: &str) -> f64 {
    r["quantities"].as_array().unwrap().iter().find(|q| q["name"].as_str().unwrap().starts_with(name)).unwrap_or_else(|| panic!("no '{name}' in {}", r["quantities"]))["value"].as_f64().unwrap()
}

/// Centre-fed dipole (145 mm, 5 mm square arms, L/a ≈ 50): first series resonance just below
/// λ/2 with Rin ≈ 65–75 Ω (King / Hallén), directivity 2.15 dBi, radiated = delivered power.
#[test]
fn dipole_input_impedance() {
    let c = new_core();
    call(&c, "new_model", json!({}));
    call(&c, "add_box", json!({"name": "ArmA", "min": [-2.5, -2.5, -75], "size": [5, 5, 70], "material": "Copper C101"}));
    call(&c, "add_box", json!({"name": "ArmB", "min": [-2.5, -2.5, 0], "size": [5, 5, 70], "material": "Copper C101"}));
    // λ/4 of clearance at the band centre keeps the absorber out of the near field
    call(&c, "add_enclosure", json!({"min": [-152.5, -152.5, -225], "size": [305, 305, 445]}));
    call(&c, "set_analysis", json!({"type": "broadband_em", "freq_min_ghz": 0.5, "freq_max_ghz": 1.5, "em_boundary": "open", "freq_steps": 201}));
    call(&c, "set_mesh", json!({"element_size_mm": 5}));
    call(&c, "add_lumped_port", json!({"from": [-2.5, -2.5, -5], "to": [-2.5, -2.5, 0], "impedance": 73}));
    let r = solve(&c);
    let fr = qty(&r, "Port #1: resonance");
    let rr = qty(&r, "Port #1: Re Zin at resonance");
    eprintln!("resonance {fr:.4} GHz, R {rr:.1} Ω, S11 min {:.1} dB; {}", qty(&r, "Minimum |S11|"), r["warnings"]);
    for q in r["quantities"].as_array().unwrap() {
        eprintln!("  {} = {} {}", q["name"], q["value"], q["unit"]);
    }
    let lam_half = 0.299_792_458 / (2.0 * 0.150);
    assert!(fr > 0.85 * lam_half && fr < 1.0 * lam_half, "resonance {fr} GHz vs λ/2 at {lam_half}");
    assert!(rr > 63.0 && rr < 78.0, "Rin {rr}");
    // energy conservation: a lossless dipole radiates what the port delivers
    let bal = qty(&r, "Radiated / delivered power");
    assert!((bal - 100.0).abs() < 4.0, "radiated / delivered {bal} %");
    let d = qty(&r, "Directivity at the band centre");
    assert!((d - 2.15).abs() < 0.4, "directivity {d} dBi (half-wave dipole 2.15 dBi)");
}
