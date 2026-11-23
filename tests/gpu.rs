//! GPU solver must reproduce the CPU results (skipped when no suitable GPU is present).
use serde_json::{Value, json};
use simcraft::session::new_core;

fn solve_with(compute: &str, calls: &[(&str, Value)]) -> Value {
    let c = new_core();
    for (t, a) in calls {
        c.call(t, a).unwrap_or_else(|e| panic!("{t}: {e:#}"));
    }
    c.call("set_analysis", &json!({"compute": compute})).unwrap();
    serde_json::from_str(&c.call("solve", &json!({})).unwrap().text).unwrap()
}

fn gpu_or_skip(r: &Value) -> bool {
    let used = r["info"]["solver"].as_str().unwrap_or("").contains("GPU");
    if !used {
        println!("SKIP: no usable GPU ({:?})", r["warnings"]);
    }
    used
}

#[test]
fn gpu_matches_cpu_static() {
    let calls = [
        ("add_box", json!({"name": "Beam", "size": [100, 10, 10]})),
        ("set_mesh", json!({"element_size_mm": 2.5})),
        ("add_fixed_support", json!({"faces": ["Beam.xmin"]})),
        ("add_force", json!({"faces": ["Beam.xmax"], "vector": [0, 0, -100]})),
    ];
    let cpu = solve_with("cpu", &calls);
    let gpu = solve_with("gpu", &calls);
    if !gpu_or_skip(&gpu) {
        return;
    }
    let tip = |r: &Value| r["fields"]["deformation_z"]["min"].as_f64().unwrap();
    let vm = |r: &Value| r["fields"]["von_mises"]["max"].as_f64().unwrap();
    println!("solver: {}", gpu["info"]["solver"]);
    println!("tip  cpu {} gpu {}", tip(&cpu), tip(&gpu));
    println!("vm   cpu {} gpu {}", vm(&cpu), vm(&gpu));
    assert!((tip(&cpu) - tip(&gpu)).abs() / tip(&cpu).abs() < 1e-5);
    assert!((vm(&cpu) - vm(&gpu)).abs() / vm(&cpu) < 1e-5);
}

#[test]
fn gpu_matches_cpu_modal_and_thermal() {
    let modal = [
        ("add_box", json!({"name": "Beam", "size": [100, 10, 10]})),
        ("set_mesh", json!({"element_size_mm": 2.5})),
        ("set_analysis", json!({"type": "modal", "modes": 4})),
        ("add_fixed_support", json!({"faces": ["Beam.xmin"]})),
    ];
    let cpu = solve_with("cpu", &modal);
    let gpu = solve_with("gpu", &modal);
    if !gpu_or_skip(&gpu) {
        return;
    }
    for i in 0..4 {
        let (a, b) = (cpu["modes"][i]["frequency_hz"].as_f64().unwrap(), gpu["modes"][i]["frequency_hz"].as_f64().unwrap());
        println!("mode {} cpu {a} gpu {b}", i + 1);
        assert!((a - b).abs() / a < 1e-5);
    }
    let thermal = [
        ("add_box", json!({"name": "Bar", "size": [100, 10, 10], "material": "Copper"})),
        ("set_mesh", json!({"element_size_mm": 2})),
        ("set_analysis", json!({"type": "steady_thermal"})),
        ("add_temperature", json!({"faces": ["Bar.xmin"], "value": 100})),
        ("add_convection", json!({"faces": ["all"], "coefficient": 50, "ambient": 20})),
    ];
    let cpu = solve_with("cpu", &thermal);
    let gpu = solve_with("gpu", &thermal);
    let t = |r: &Value| r["fields"]["temperature"]["min"].as_f64().unwrap();
    println!("thermal min T cpu {} gpu {}", t(&cpu), t(&gpu));
    assert!((t(&cpu) - t(&gpu)).abs() < 1e-4);
}
