//! Full-wave EM (FDTD) and microwave heating vs closed-form results.
use serde_json::{Value, json};
use simcraft::session::{Shared, new_core};

fn call(c: &Shared, t: &str, a: Value) -> Value {
    let o = c.call(t, &a).unwrap_or_else(|e| panic!("{t}: {e:#}"));
    serde_json::from_str(&o.text).unwrap_or(Value::String(o.text))
}
fn close(name: &str, got: f64, want: f64, tol: f64) {
    let err = (got - want).abs() / want.abs();
    println!("{name}: got {got:.5}, theory {want:.5}, error {:.2}%", err * 100.0);
    assert!(err < tol, "{name} off by {:.1}%", err * 100.0);
}
fn q(r: &Value, name: &str) -> f64 {
    r["quantities"].as_array().unwrap().iter().find(|x| x["name"].as_str().unwrap().starts_with(name)).unwrap_or_else(|| panic!("no {name}: {r}"))["value"].as_f64().unwrap()
}

/// WR-340 waveguide (86 × 44 mm), shorted at x = 252 mm, fed at x = 0 and open (absorbing) behind
/// the feed: the standing-wave nulls in front of the short are λg/2 apart,
/// λg = λ0 / √(1 − (λ0 / 2a)²).
#[test]
fn waveguide_wavelength_from_standing_wave() {
    let c = new_core();
    for (name, min, size) in [
        ("WallY0", [0, -4, -4], [260, 4, 52]),
        ("WallY1", [0, 86, -4], [260, 4, 52]),
        ("WallZ0", [0, 0, -4], [260, 86, 4]),
        ("WallZ1", [0, 0, 44], [260, 86, 4]),
        ("Short", [252, 0, 0], [8, 86, 44]),
    ] {
        call(&c, "add_box", json!({"name": name, "min": min, "size": size}));
        call(&c, "set_material", json!({"body": name, "material": "Copper C101"}));
    }
    call(&c, "add_enclosure", json!({"min": [0, -4, -4], "size": [260, 94, 52]}));
    call(&c, "set_mesh", json!({"element_size_mm": 2}));
    call(&c, "set_analysis", json!({"type": "full_wave_em", "frequency_ghz": 2.45, "em_boundary": "open"}));
    call(&c, "add_port", json!({"faces": ["Enclosure.xmin"], "power": 100, "polarization": "z"}));
    let r = call(&c, "solve", json!({}));
    println!("{} — {} periods, warnings {:?}", r["info"]["solver"], q(&r, "RF periods"), r["warnings"]);
    // |E| along the guide axis
    let xs: Vec<f64> = (10..=125).map(|i| i as f64 * 2.0).collect();
    let e: Vec<f64> = xs.iter().map(|&x| call(&c, "probe", json!({"point": [x, 43, 22]}))["values"]["electric_field"].as_f64().unwrap()).collect();
    // nulls: local minima of |E|², refined by a parabola
    let mut nulls = vec![];
    for i in 1..e.len() - 1 {
        if e[i] < e[i - 1] && e[i] <= e[i + 1] && e[i] < 0.3 * e.iter().cloned().fold(0.0, f64::max) {
            let (a, b, cc) = (e[i - 1].powi(2), e[i].powi(2), e[i + 1].powi(2));
            let off = 0.5 * (a - cc) / (a - 2.0 * b + cc);
            nulls.push(xs[i] + off * 2.0);
        }
    }
    println!("nulls at {nulls:?} mm");
    let l0 = 299.792458 / 2.45; // mm
    let lg = l0 / (1.0 - (l0 / 172.0f64).powi(2)).sqrt();
    assert!(nulls.len() >= 2, "expected two standing-wave nulls");
    let n = nulls.len();
    close("λg/2 from null spacing", nulls[n - 1] - nulls[n - 2], lg / 2.0, 0.02);
    close("short to first null = λg/2", 252.0 - nulls[n - 1], lg / 2.0, 0.02);
}

fn cavity(c: &Shared, compute: &str) {
    // εr 10, tanδ 0.3: λ in food ≈ 39 mm, ≥ 12 cells at 3 mm
    call(c, "define_material", json!({"name": "Test food", "based_on": "Food (water-rich, 2.45 GHz)", "density": 1000, "specific_heat": 4000, "relative_permittivity": 10, "loss_tangent": 0.3}));
    call(c, "add_box", json!({"name": "Food", "min": [45, 45, 15], "size": [60, 60, 30]}));
    call(c, "set_material", json!({"body": "Food", "material": "Test food"}));
    call(c, "add_enclosure", json!({"min": [0, 0, 0], "size": [150, 150, 120]}));
    call(c, "set_mesh", json!({"element_size_mm": 3}));
    call(c, "set_analysis", json!({"type": "full_wave_em", "compute": compute, "em_boundary": "pec"}));
    call(c, "add_port", json!({"faces": ["Enclosure.xmin"], "power": 100, "polarization": "z"}));
}

/// Closed metal cavity: everything the port delivers must be absorbed in the lossy load.
#[test]
fn closed_cavity_power_balance_and_heating_energy() {
    let c = new_core();
    cavity(&c, "cpu");
    let r = call(&c, "solve", json!({}));
    println!("{} — {} periods, warnings {:?}", r["info"]["solver"], q(&r, "RF periods"), r["warnings"]);
    close("absorbed / delivered in a closed cavity", q(&r, "Absorbed power"), 100.0, 0.03);
    // microwave heating: insulated food, energy must end up as heat
    call(&c, "set_analysis", json!({"type": "microwave_heating", "end_time": 10, "time_steps": 10, "initial_temperature": 5}));
    let r = call(&c, "solve", json!({}));
    let p = q(&r, "Heating power in solids");
    let mass = 1000.0 * 60e-3 * 60e-3 * 30e-3;
    close("mean temperature rise = P·t / (m·c)", q(&r, "Mean temperature of Food") - 5.0, p * 10.0 / (mass * 4000.0), 0.01);
    assert!(q(&r, "Temperature spread") > 0.0);
}

#[test]
fn gpu_fdtd_matches_cpu() {
    let c = new_core();
    cavity(&c, "gpu");
    let g = call(&c, "solve", json!({}));
    if !g["info"]["solver"].as_str().unwrap_or("").contains("GPU") {
        println!("SKIP: no usable GPU ({:?})", g["warnings"]);
        return;
    }
    call(&c, "set_analysis", json!({"compute": "cpu"}));
    let cpu = call(&c, "solve", json!({}));
    println!("{} vs {} ({} s vs {} s)", g["info"]["solver"], cpu["info"]["solver"], g["info"]["seconds"], cpu["info"]["seconds"]);
    close("GPU vs CPU absorbed power", q(&g, "Absorbed in Food"), q(&cpu, "Absorbed in Food"), 0.01);
    close("GPU vs CPU peak |E|", q(&g, "Peak |E|"), q(&cpu, "Peak |E|"), 0.02);
}
