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
