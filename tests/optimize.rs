//! Parameter studies, design optimization and topology optimization against hand calculations.
use serde_json::{Value, json};
use simcraft::session::{Shared, new_core};

fn call(c: &Shared, t: &str, a: Value) -> Value {
    let o = c.call(t, &a).unwrap_or_else(|e| panic!("{t}: {e:#}"));
    serde_json::from_str(&o.text).unwrap_or(Value::String(o.text))
}

fn cantilever(h: f64) -> Shared {
    let c = new_core();
    call(&c, "add_box", json!({"name": "Beam", "size": [100, 10, 8]}));
    call(&c, "set_mesh", json!({"element_size_mm": h}));
    call(&c, "add_fixed_support", json!({"faces": ["Beam.xmin"]}));
    call(&c, "add_force", json!({"faces": ["Beam.xmax"], "vector": [0, 0, -100]}));
    c
}

#[test]
fn deflection_scales_with_inverse_cube_of_thickness() {
    let c = cantilever(1.0);
    let r = call(&c, "run_parameter_study", json!({"parameters": [{"target": "Beam.size.z", "min": 6, "max": 12, "steps": 2}], "responses": ["min:deformation_z", "mass"]}));
    let rows = r["runs"].as_array().unwrap();
    let d = |i: usize| rows[i]["min:deformation_z"].as_f64().unwrap().abs();
    let ratio = d(0) / d(1);
    println!("tip deflection t=6: {:.4} mm, t=12: {:.4} mm, ratio {ratio:.2} (theory 8)", d(0), d(1));
    assert!((ratio - 8.0).abs() / 8.0 < 0.08);
    let m = |i: usize| rows[i]["mass"].as_f64().unwrap();
    assert!((m(1) / m(0) - 2.0).abs() < 1e-6);
}

#[test]
fn minimum_mass_beam_meets_stress_limit() {
    let c = cantilever(0.5);
    let r = call(&c, "optimize_design", json!({
        "parameters": [{"target": "Beam.size.z", "min": 4, "max": 20}],
        "objective": {"response": "mass", "goal": "minimize"},
        "constraints": [{"response": "max:von_mises", "op": "<=", "value": 150}],
        "max_evaluations": 24
    }));
    println!("{}", r["message"]);
    let best = &r["runs"][r["best_run"].as_u64().unwrap() as usize - 1];
    let t = best["Beam.size.z"].as_f64().unwrap();
    let vm = best["max:von_mises"].as_f64().unwrap();
    // beam theory: σ = 6FL/(b t²) ≤ 150 MPa → t ≥ 6.32 mm; the 0.5 mm grid rounds thickness up
    println!("optimum thickness {t:.3} mm (theory ≥ 6.32), max von Mises {vm:.1} MPa");
    assert!(best["feasible"].as_bool().unwrap());
    assert!(vm <= 150.0);
    assert!(t > 5.9 && t < 7.2, "t = {t}");
    // apply_design changes the model
    call(&c, "apply_design", json!({}));
    let m = call(&c, "get_model", json!({}));
    let size = m["bodies"][0]["shape"]["size"][2].as_f64().unwrap();
    assert!((size - t).abs() < 1e-9);
}

#[test]
fn topology_reduces_material_and_exports_shape() {
    let c = new_core();
    call(&c, "add_box", json!({"name": "Plate", "size": [80, 10, 40]}));
    call(&c, "set_mesh", json!({"element_size_mm": 2}));
    call(&c, "add_fixed_support", json!({"faces": ["Plate.xmin"]}));
    call(&c, "add_force", json!({"faces": ["Plate.xmax"], "vector": [0, 0, -1000]}));
    let path = std::env::temp_dir().join(format!("simcraft-topo-{}.stl", std::process::id())).to_string_lossy().to_string();
    let r = call(&c, "topology_optimize", json!({"volume_fraction": 0.4, "iterations": 30, "export_path": path}));
    println!("{}", r["message"]);
    let h = r["history"].as_array().unwrap();
    let c0 = h[0]["compliance_j"].as_f64().unwrap();
    let cn = h.last().unwrap()["compliance_j"].as_f64().unwrap();
    let vf = h.last().unwrap()["volume_fraction"].as_f64().unwrap();
    println!("compliance {c0:.4e} -> {cn:.4e} J, volume fraction {vf:.3}");
    assert!(cn < c0, "compliance should drop as material is redistributed");
    assert!((vf - 0.4).abs() < 0.01);
    // the exported shape is a valid closed solid SimCraft can mesh again
    let c2 = new_core();
    call(&c2, "import_stl", json!({"path": path, "name": "Optimized"}));
    call(&c2, "set_analysis", json!({"boundary_fit": false})); // count plain voxels
    let m = call(&c2, "set_mesh", json!({"element_size_mm": 2}));
    let e = m["elements"].as_u64().unwrap() as f64;
    println!("re-imported optimized shape: {e} elements (full plate 4000)");
    assert!(e > 1000.0 && e < 2600.0);
    let _ = std::fs::remove_file(&path);
}
