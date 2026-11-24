//! Boundary-fitted meshes: patch test and stress concentrations on curved surfaces.
use serde_json::{Value, json};
use simcraft::session::{Shared, new_core};

fn call(c: &Shared, t: &str, a: Value) -> Value {
    let o = c.call(t, &a).unwrap_or_else(|e| panic!("{t}: {e:#}"));
    serde_json::from_str(&o.text).unwrap_or(Value::String(o.text))
}
fn fmax(r: &Value, f: &str) -> f64 {
    r["fields"][f]["max"].as_f64().unwrap()
}
fn fmin(r: &Value, f: &str) -> f64 {
    r["fields"][f]["min"].as_f64().unwrap()
}
fn err(got: f64, want: f64) -> f64 {
    (got - want).abs() / want.abs()
}

/// Off-grid box (10.3 × 5.7 × 4.1 mm at h = 1): fitting moves every surface node, so all surface
/// elements are distorted; uniform tension must still give a uniform stress (patch test).
#[test]
fn patch_test_distorted_box_uniform_tension() {
    let c = new_core();
    call(&c, "add_box", json!({"name": "Bar", "min": [0.2, 0.1, 0.35], "size": [10.3, 5.7, 4.1]}));
    call(&c, "set_mesh", json!({"element_size_mm": 1}));
    call(&c, "add_symmetry", json!({"faces": ["Bar.xmin", "Bar.ymin", "Bar.zmin"]}));
    call(&c, "add_pressure", json!({"faces": ["Bar.xmax"], "value_mpa": -10}));
    let r = call(&c, "solve", json!({}));
    let fitted = r["quantities"].as_array().unwrap().iter().find(|q| q["name"] == "Boundary-fitted elements").unwrap()["value"].as_f64().unwrap();
    println!("fitted elements {fitted}, von Mises {} … {}", fmin(&r, "von_mises"), fmax(&r, "von_mises"));
    assert!(fitted > 100.0);
    assert!(err(fmax(&r, "von_mises"), 10.0) < 1e-6 && err(fmin(&r, "von_mises"), 10.0) < 1e-6, "stress not uniform");
    // reaction balances the load on the true (fitted) area: 10 MPa × 5.7 × 4.1 mm²
    let rx = r["reactions"][0]["force_n"][0].as_f64().unwrap();
    assert!(err(rx.abs(), 10.0 * 5.7 * 4.1) < 1e-6, "reaction {rx}");
}

/// Quarter of a long strip with a hole (d/W = 0.1) in tension: Kt (gross) = 3.03 (Heywood /
/// Howland). The strip is 3 widths long so the loaded end does not disturb the hole.
fn kirsch(fit: bool) -> f64 {
    let c = new_core();
    call(&c, "add_box", json!({"name": "Plate", "min": [0, 0, 0], "size": [300, 100, 2]}));
    call(&c, "add_cylinder", json!({"name": "Hole", "base": [0, 0, 0], "axis": "z", "radius": 10, "height": 2, "operation": "subtract"}));
    call(&c, "set_mesh", json!({"element_size_mm": 1}));
    call(&c, "set_analysis", json!({"type": "static_structural", "boundary_fit": fit}));
    call(&c, "add_symmetry", json!({"faces": ["Plate.xmin", "Plate.ymin", "Plate.zmin"]}));
    call(&c, "add_pressure", json!({"faces": ["Plate.xmax"], "value_mpa": -10}));
    let r = call(&c, "solve", json!({}));
    fmax(&r, "max_principal") / 10.0
}

#[test]
fn kirsch_plate_with_hole() {
    let kt = 2.0 + (1.0 - 0.1f64).powi(3); // net-section Kt (Heywood), d/W = 0.1
    let want = kt / (1.0 - 0.1);
    let (vox, fit) = (kirsch(false), kirsch(true));
    println!("Kt: voxel {vox:.3} ({:.1}%), fitted {fit:.3} ({:.1}%), theory {want:.3}", 100.0 * err(vox, want), 100.0 * err(fit, want));
    assert!(err(fit, want) < 0.03, "fitted Kt {fit}");
    assert!(err(fit, want) < err(vox, want), "fitting must improve the peak stress");
}

/// Thick cylinder (Lamé), quarter model in plane strain: hoop stress at the bore
/// σθ = p (ro² + ri²) / (ro² − ri²).
fn lame(fit: bool) -> f64 {
    let c = new_core();
    call(&c, "add_cylinder", json!({"name": "Pipe", "base": [0, 0, 0], "axis": "z", "radius": 20, "height": 4}));
    call(&c, "add_cylinder", json!({"name": "Bore", "base": [0, 0, 0], "axis": "z", "radius": 10, "height": 4, "operation": "subtract"}));
    call(&c, "add_box", json!({"name": "CutX", "min": [-30, -30, 0], "size": [30, 60, 4], "operation": "subtract"}));
    call(&c, "add_box", json!({"name": "CutY", "min": [-30, -30, 0], "size": [60, 30, 4], "operation": "subtract"}));
    call(&c, "set_mesh", json!({"element_size_mm": 1}));
    call(&c, "set_analysis", json!({"type": "static_structural", "boundary_fit": fit}));
    call(&c, "add_symmetry", json!({"faces": ["CutX.xmax", "CutY.ymax", "Pipe.bottom", "Pipe.top"]}));
    call(&c, "add_pressure", json!({"faces": ["Bore.side"], "value_mpa": 10}));
    let r = call(&c, "solve", json!({}));
    fmax(&r, "max_principal")
}

#[test]
fn lame_thick_cylinder() {
    let want = 10.0 * (400.0 + 100.0) / (400.0 - 100.0);
    let (vox, fit) = (lame(false), lame(true));
    println!("bore hoop stress: voxel {vox:.3} ({:.1}%), fitted {fit:.3} ({:.1}%), theory {want:.3}", 100.0 * err(vox, want), 100.0 * err(fit, want));
    assert!(err(fit, want) < 0.03, "fitted hoop stress {fit}");
}

fn heat_flow(r: &Value, i: usize) -> f64 {
    r["reactions"][i]["heat_flow_w"].as_f64().unwrap()
}

/// Thermal patch test on the off-grid box: fixed temperature on one end, uniform flux into the
/// other → exactly linear temperature, T(L) = T0 + q L / k, on distorted fitted elements.
#[test]
fn thermal_patch_test_distorted_box() {
    let c = new_core();
    call(&c, "add_box", json!({"name": "Bar", "min": [0.2, 0.1, 0.35], "size": [10.3, 5.7, 4.1], "material": "Aluminum 6061-T6"}));
    call(&c, "set_mesh", json!({"element_size_mm": 1}));
    call(&c, "set_analysis", json!({"type": "steady_thermal"}));
    call(&c, "add_temperature", json!({"faces": ["Bar.xmin"], "value": 20}));
    call(&c, "add_heat_flux", json!({"faces": ["Bar.xmax"], "value": 1e5}));
    let r = call(&c, "solve", json!({}));
    let k = call(&c, "list_materials", json!({}));
    let k = k.as_array().and_then(|a| a.iter().find(|m| m["name"] == "Aluminum 6061-T6")).and_then(|m| m["thermal_conductivity"].as_f64()).unwrap_or(167.0);
    let want = 20.0 + 1e5 * 10.3e-3 / k;
    println!("T max {:.5} (theory {want:.5}), heat in {:.5} W (theory {:.5})", fmax(&r, "temperature"), -heat_flow(&r, 0), 1e5 * 5.7e-3 * 4.1e-3);
    assert!(err(fmax(&r, "temperature") - 20.0, want - 20.0) < 1e-4, "temperature rise not linear");
    // all flux on the true (fitted) area leaves through the cold face
    assert!(err(-heat_flow(&r, 0), 1e5 * 5.7e-3 * 4.1e-3) < 1e-6);
}

/// Radial conduction through a quarter pipe wall: Q = (π/2) k L ΔT / ln(ro/ri).
fn pipe_heat(fit: bool) -> (f64, f64) {
    let c = new_core();
    call(&c, "add_cylinder", json!({"name": "Pipe", "base": [0, 0, 0], "axis": "z", "radius": 20, "height": 4, "material": "Copper"}));
    call(&c, "add_cylinder", json!({"name": "Bore", "base": [0, 0, 0], "axis": "z", "radius": 10, "height": 4, "operation": "subtract"}));
    call(&c, "add_box", json!({"name": "CutX", "min": [-30, -30, 0], "size": [30, 60, 4], "operation": "subtract"}));
    call(&c, "add_box", json!({"name": "CutY", "min": [-30, -30, 0], "size": [60, 30, 4], "operation": "subtract"}));
    call(&c, "set_mesh", json!({"element_size_mm": 1}));
    call(&c, "set_analysis", json!({"type": "steady_thermal", "boundary_fit": fit}));
    call(&c, "add_temperature", json!({"faces": ["Bore.side"], "value": 100}));
    call(&c, "add_temperature", json!({"faces": ["Pipe.side"], "value": 0}));
    let r = call(&c, "solve", json!({}));
    let mats = call(&c, "list_materials", json!({}));
    let k = mats.as_array().and_then(|a| a.iter().find(|m| m["name"] == "Copper C101")).and_then(|m| m["thermal_conductivity"].as_f64()).unwrap();
    let want = std::f64::consts::FRAC_PI_2 * k * 4e-3 * 100.0 / 2f64.ln();
    (heat_flow(&r, 0), want)
}

#[test]
fn radial_conduction_through_pipe_wall() {
    let (vox, want) = pipe_heat(false);
    let (fit, _) = pipe_heat(true);
    println!("pipe wall heat flow: voxel {vox:.2} W ({:.1}%), fitted {fit:.2} W ({:.1}%), theory {want:.2} W", 100.0 * err(vox, want), 100.0 * err(fit, want));
    assert!(err(fit, want) < 0.02, "fitted heat flow {fit}");
    assert!(err(fit, want) < err(vox, want), "fitting must improve the heat flow");
}

/// Cantilever round rod, first bending frequency f₁ = (1.8751²/2π) √(EI/ρA) / L².
fn rod_frequency(fit: bool) -> (f64, f64) {
    let c = new_core();
    call(&c, "add_cylinder", json!({"name": "Rod", "base": [0, 0, 0], "axis": "x", "radius": 3.3, "height": 80}));
    call(&c, "set_mesh", json!({"element_size_mm": 1}));
    call(&c, "set_analysis", json!({"type": "modal", "modes": 2, "boundary_fit": fit}));
    call(&c, "add_fixed_support", json!({"faces": ["Rod.bottom"]}));
    let r = call(&c, "solve", json!({}));
    let mats = call(&c, "list_materials", json!({}));
    let name = call(&c, "get_model", json!({}))["bodies"][0]["material"].as_str().unwrap().to_string();
    let m = mats.as_array().unwrap().iter().find(|m| m["name"] == name.as_str()).unwrap().clone();
    let e = m["youngs_modulus_gpa"].as_f64().unwrap() * 1e9;
    let rho = m["density"].as_f64().unwrap();
    let rr = 3.3e-3;
    let want = 1.8751f64.powi(2) / std::f64::consts::TAU * (e * rr * rr / 4.0 / rho).sqrt() / 0.08f64.powi(2);
    (r["modes"][0]["frequency_hz"].as_f64().unwrap(), want)
}

#[test]
fn cantilever_rod_bending_frequency() {
    let (vox, want) = rod_frequency(false);
    let (fit, _) = rod_frequency(true);
    println!("rod f1: voxel {vox:.1} Hz ({:.1}%), fitted {fit:.1} Hz ({:.1}%), Euler-Bernoulli {want:.1} Hz", 100.0 * err(vox, want), 100.0 * err(fit, want));
    assert!(err(fit, want) < 0.03, "fitted f1 {fit}");
    assert!(err(fit, want) < err(vox, want), "fitting must improve the frequency");
}
