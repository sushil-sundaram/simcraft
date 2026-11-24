//! Large models: matrix-free multigrid CG for structural and thermal solves.
use serde_json::{Value, json};
use simcraft::session::{Shared, new_core};

fn call(c: &Shared, t: &str, a: Value) -> Value {
    let o = c.call(t, &a).unwrap_or_else(|e| panic!("{t}: {e:#}"));
    serde_json::from_str(&o.text).unwrap_or(Value::String(o.text))
}

/// 115 000-element cantilever (above the multigrid threshold): tip deflection vs Timoshenko beam
/// theory δ = FL³/(3EI) + FL/(κGA).
#[test]
fn multigrid_cantilever_matches_beam_theory() {
    let c = new_core();
    call(&c, "add_box", json!({"name": "Beam", "size": [120, 12, 10]}));
    call(&c, "set_mesh", json!({"element_size_mm": 0.5}));
    call(&c, "add_fixed_support", json!({"faces": ["Beam.xmin"]}));
    call(&c, "add_force", json!({"faces": ["Beam.xmax"], "vector": [0, 0, -500]}));
    let t = std::time::Instant::now();
    let r = call(&c, "solve", json!({}));
    let name = call(&c, "get_model", json!({}))["bodies"][0]["material"].as_str().unwrap().to_string();
    let mats = call(&c, "list_materials", json!({}));
    let m = mats.as_array().unwrap().iter().find(|m| m["name"] == name.as_str()).unwrap();
    let (e, nu) = (m["youngs_modulus_gpa"].as_f64().unwrap() * 1e3, m["poisson_ratio"].as_f64().unwrap());
    let (l, b, h, f) = (120.0, 12.0, 10.0, 500.0);
    let i = b * h * h * h / 12.0;
    let g = e / (2.0 * (1.0 + nu));
    let want = f * l * l * l / (3.0 * e * i) + f * l / (5.0 / 6.0 * g * b * h);
    let got = -r["fields"]["deformation_z"]["min"].as_f64().unwrap();
    println!("{} elements in {:.1} s — {} ({} iterations): tip {got:.5} mm, theory {want:.5} mm", r["info"]["elements"], t.elapsed().as_secs_f64(), r["info"]["solver"], r["info"]["iterations"]);
    assert!(r["info"]["solver"].as_str().unwrap().contains("multigrid"));
    assert!(r["info"]["iterations"].as_u64().unwrap() < 80);
    assert!((got - want).abs() / want < 0.02, "tip {got} vs {want}");
}

/// ~1.8 M elements (5.5 M DOF): `cargo test --release --test large -- --ignored --nocapture`
#[test]
#[ignore]
fn two_million_elements() {
    for (kind, h) in [("structural", 1.15), ("thermal", 1.15)] {
        let c = new_core();
        call(&c, "add_box", json!({"name": "Plate", "size": [300, 300, 30]}));
        call(&c, "add_cylinder", json!({"name": "Hole", "base": [150, 150, 0], "radius": 30, "height": 30, "operation": "subtract"}));
        call(&c, "set_mesh", json!({"element_size_mm": h}));
        if kind == "thermal" {
            call(&c, "set_analysis", json!({"type": "steady_thermal"}));
            call(&c, "add_temperature", json!({"faces": ["Plate.xmin"], "value": 100}));
            call(&c, "add_convection", json!({"faces": ["all"], "coefficient": 20, "ambient": 20}));
        } else {
            call(&c, "add_fixed_support", json!({"faces": ["Plate.xmin"]}));
            call(&c, "add_pressure", json!({"faces": ["Plate.zmax"], "value_mpa": 0.1}));
        }
        let t = std::time::Instant::now();
        let r = call(&c, "solve", json!({}));
        println!("{kind}: {} elements, {} DOF, total {:.1} s, linear solve {} s, {} iterations — {}", r["info"]["elements"], r["info"]["dofs"], t.elapsed().as_secs_f64(), r["info"]["linear_solve_seconds"], r["info"]["iterations"], r["info"]["solver"]);
        let rss = std::fs::read_to_string("/proc/self/status").unwrap_or_default().lines().find(|l| l.starts_with("VmHWM")).unwrap_or("").to_string();
        println!("  peak memory {rss}");
    }
}
