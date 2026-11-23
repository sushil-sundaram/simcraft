//! Benchmark: cargo test --release --test gpu_bench -- --ignored --nocapture
use serde_json::{Value, json};
use simcraft::session::new_core;
use std::time::Instant;

fn run(compute: &str, h: f64, kind: &str) -> (f64, Value) {
    let c = new_core();
    c.call("add_box", &json!({"name": "Plate", "size": [300, 300, if kind == "thermal" { 9 } else { 30 }]})).unwrap();
    c.call("add_cylinder", &json!({"name": "Hole", "base": [150, 150, 0], "radius": 30, "height": 30, "operation": "subtract"})).unwrap();
    c.call("set_mesh", &json!({"element_size_mm": h})).unwrap();
    if kind == "thermal" {
        c.call("set_analysis", &json!({"type": "steady_thermal", "compute": compute})).unwrap();
        c.call("add_temperature", &json!({"faces": ["Plate.xmin"], "value": 100})).unwrap();
        c.call("add_convection", &json!({"faces": ["all"], "coefficient": 20, "ambient": 20})).unwrap();
    } else {
        c.call("set_analysis", &json!({"compute": compute})).unwrap();
        c.call("add_fixed_support", &json!({"faces": ["Plate.xmin"]})).unwrap();
        c.call("add_pressure", &json!({"faces": ["Plate.zmax"], "value_mpa": 0.1})).unwrap();
    }
    let t = Instant::now();
    let r: Value = serde_json::from_str(&c.call("solve", &json!({})).unwrap().text).unwrap();
    (t.elapsed().as_secs_f64(), r)
}

#[test]
#[ignore]
fn bench() {
    for (kind, h) in [("structural", 2.0), ("thermal", 1.5)] {
        let (tc, rc) = run("cpu", h, kind);
        let (tg, rg) = run("gpu", h, kind);
        let f = if kind == "thermal" { "temperature" } else { "total_deformation" };
        println!(
            "{kind}: {} DOF | total CPU {:.1}s GPU {:.1}s | linear solve CPU {}s GPU {}s ({} iterations) | {} | max {f}: cpu {} gpu {}",
            rc["info"]["dofs"], tc, tg, rc["info"]["linear_solve_seconds"], rg["info"]["linear_solve_seconds"], rg["info"]["iterations"], rg["info"]["solver"], rc["fields"][f]["max"], rg["fields"][f]["max"]
        );
    }
}
