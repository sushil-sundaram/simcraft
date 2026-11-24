//! Validation of the extended solvers against closed-form solutions, driven through the tool API.
use serde_json::{Value, json};
use simcraft::session::{Shared, new_core};

fn run(c: &Shared, calls: &[(&str, Value)]) -> Value {
    let mut last = Value::Null;
    for (t, a) in calls {
        let o = c.call(t, a).unwrap_or_else(|e| panic!("{t} failed: {e:#}"));
        last = serde_json::from_str(&o.text).unwrap_or(Value::String(o.text));
    }
    last
}
fn q(r: &Value, name: &str) -> f64 {
    r["quantities"].as_array().unwrap().iter().find(|x| x["name"].as_str().unwrap().starts_with(name)).unwrap_or_else(|| panic!("no quantity {name} in {r}"))["value"].as_f64().unwrap()
}
fn fmax(r: &Value, f: &str) -> f64 {
    r["fields"][f]["max"].as_f64().unwrap()
}
fn close(name: &str, got: f64, want: f64, tol: f64) {
    let err = (got - want).abs() / want.abs();
    println!("{name}: got {got:.6e}, theory {want:.6e}, error {:.2}%", err * 100.0);
    assert!(err < tol, "{name} off by {:.1}%", err * 100.0);
}

#[test]
fn capacitor() {
    let c = new_core();
    let r = run(&c, &[
        ("add_box", json!({"name": "Dielectric", "size": [40, 40, 2], "material": "PTFE"})),
        ("set_mesh", json!({"element_size_mm": 1})),
        ("set_analysis", json!({"type": "electrostatic"})),
        ("add_voltage", json!({"faces": ["Dielectric.zmax"], "value": 100})),
        ("add_voltage", json!({"faces": ["Dielectric.zmin"], "value": 0})),
        ("solve", json!({})),
    ]);
    close("capacitance", q(&r, "Capacitance (from energy)"), 8.8541878e-12 * 2.1 * 0.04 * 0.04 / 0.002, 0.02);
    close("E field", fmax(&r, "electric_field"), 100.0 / 0.002, 0.02);
}

#[test]
fn resistance_and_joule_heating() {
    let c = new_core();
    let r = run(&c, &[
        ("add_box", json!({"name": "Bar", "size": [100, 10, 10], "material": "Copper"})),
        ("set_mesh", json!({"element_size_mm": 2})),
        ("set_analysis", json!({"type": "thermal_electric"})),
        ("add_voltage", json!({"faces": ["Bar.xmin"], "value": 0.001})),
        ("add_voltage", json!({"faces": ["Bar.xmax"], "value": 0})),
        ("add_temperature", json!({"faces": ["Bar.xmin", "Bar.xmax"], "value": 22})),
        ("solve", json!({})),
    ]);
    let rth = 0.1 / (5.8e7 * 1e-4);
    close("resistance", q(&r, "Resistance"), rth, 0.01);
    close("joule power", q(&r, "Total Joule heating power"), 1e-6 / rth, 0.01);
    let qv = (1e-6 / rth) / 1e-5;
    close("temperature rise", fmax(&r, "temperature") - 22.0, qv * 0.1 * 0.1 / (8.0 * 391.0), 0.03);
}

#[test]
fn ring_coil_center_field() {
    let c = new_core();
    let r = run(&c, &[
        ("add_cylinder", json!({"name": "Coil", "base": [0, 0, -2], "radius": 22, "height": 4, "material": "Copper"})),
        ("add_cylinder", json!({"name": "Bore", "base": [0, 0, -2], "radius": 18, "height": 4, "operation": "subtract"})),
        ("add_enclosure", json!({"padding_mm": [50, 50, 50, 50, 40, 40]})),
        ("set_analysis", json!({"type": "magnetostatic"})),
        ("set_mesh", json!({"element_size_mm": 2})),
        ("add_coil", json!({"body": "Coil", "current_density": 10, "axis": "z", "center": [0, 0, 0]})),
        ("solve", json!({})),
        ("probe", json!({"point": [0, 0, 0]})),
    ]);
    // I = J·A = 10 A/mm² × 16 mm² ; B = μ0 I / (2 R)
    let b = r["values"]["magnetic_flux_density"].as_f64().unwrap();
    close("ring centre B", b, 1.25663706e-6 * 160.0 / (2.0 * 0.020), 0.12);
}

#[test]
fn euler_buckling() {
    let c = new_core();
    let r = run(&c, &[
        ("add_box", json!({"name": "Col", "min": [0, 0, 0], "size": [10, 10, 200]})),
        ("set_mesh", json!({"element_size_mm": 2})),
        ("set_analysis", json!({"type": "linear_buckling", "modes": 2})),
        ("add_fixed_support", json!({"faces": ["Col.zmin"]})),
        ("add_force", json!({"faces": ["Col.zmax"], "vector": [0, 0, -1000]})),
        ("solve", json!({})),
    ]);
    let pcr = std::f64::consts::PI.powi(2) * 200e3 * (10.0f64.powi(4) / 12.0) / (4.0 * 200.0f64.powi(2));
    close("buckling multiplier", r["modes"][0]["load_multiplier"].as_f64().unwrap(), pcr / 1000.0, 0.05);
}

#[test]
fn bar_plasticity() {
    let c = new_core();
    let r = run(&c, &[
        ("add_box", json!({"name": "Bar", "size": [100, 10, 10]})),
        ("set_mesh", json!({"element_size_mm": 2.5})),
        ("set_analysis", json!({"type": "nonlinear_structural", "load_steps": 10})),
        ("add_displacement", json!({"faces": ["Bar.xmin"], "x": 0})),
        ("add_displacement", json!({"faces": ["Bar.ymin"], "y": 0})),
        ("add_displacement", json!({"faces": ["Bar.zmin"], "z": 0})),
        ("add_displacement", json!({"faces": ["Bar.xmax"], "x": 0.5})),
        ("solve", json!({})),
    ]);
    let f = r["reactions"].as_array().unwrap().iter().find(|x| x["load_id"] == 4).unwrap()["force_n"][0].as_f64().unwrap();
    let sigma = 250.0 + 1450.0 * (0.005 - 250.0 / 200e3);
    close("plastic bar force", f, sigma * 100.0, 0.01);
    assert!(fmax(&r, "plastic_strain") > 0.003);
}

#[test]
fn dynamic_amplification() {
    let c = new_core();
    let r = run(&c, &[
        ("add_box", json!({"name": "Beam", "size": [100, 10, 10]})),
        ("set_mesh", json!({"element_size_mm": 2.5})),
        ("set_analysis", json!({"type": "transient_structural", "end_time": 0.002, "time_steps": 200, "damping_ratio": 0.0})),
        ("add_fixed_support", json!({"faces": ["Beam.xmin"]})),
        ("add_force", json!({"faces": ["Beam.xmax"], "vector": [0, 0, -100]})),
        ("solve", json!({})),
    ]);
    close("step-load peak = 2 × static", q(&r, "Peak displacement"), 2.0 * 0.1996, 0.04);
    close("first frequency", q(&r, "First natural frequency"), 813.8, 0.02);
}

#[test]
fn uniform_heating() {
    let c = new_core();
    let r = run(&c, &[
        ("add_box", json!({"name": "Block", "size": [20, 20, 20]})),
        ("set_mesh", json!({"element_size_mm": 2})),
        ("set_analysis", json!({"type": "transient_thermal", "end_time": 10, "time_steps": 20, "initial_temperature": 20})),
        ("add_heat_generation", json!({"value": 1e7})),
        ("solve", json!({})),
    ]);
    close("temperature rise", fmax(&r, "temperature") - 20.0, 1e7 * 10.0 / (7850.0 * 434.0), 0.005);
}

#[test]
fn channel_flow() {
    let c = new_core();
    let r = run(&c, &[
        ("add_enclosure", json!({"min": [0, 0, 0], "size": [60, 4, 10], "material": "Water"})),
        ("set_analysis", json!({"type": "fluid", "max_iterations": 6000})),
        ("set_mesh", json!({"element_size_mm": 1})),
        ("add_inlet", json!({"faces": ["Enclosure.xmin"], "velocity": 0.001})),
        ("add_outlet", json!({"faces": ["Enclosure.xmax"], "pressure": 0})),
        ("add_symmetry", json!({"faces": ["Enclosure.ymin", "Enclosure.ymax"]})),
        ("solve", json!({})),
    ]);
    close("centreline velocity = 1.5 U", fmax(&r, "velocity"), 0.0015, 0.08);
    let pin = r["reactions"].as_array().unwrap().iter().find(|x| x["label"] == "Velocity Inlet").unwrap()["mean_pressure_pa"].as_f64().unwrap();
    // fully developed Poiseuille drop over the length (entrance adds a little)
    close("pressure drop", pin, 12.0 * 1e-3 * 0.001 * 0.06 / 1e-4, 0.15);
    let qin = r["reactions"].as_array().unwrap().iter().find(|x| x["label"] == "Velocity Inlet").unwrap()["flow_rate_m3_s"].as_f64().unwrap();
    let qout = r["reactions"].as_array().unwrap().iter().find(|x| x["label"] == "Pressure Outlet").unwrap()["flow_rate_m3_s"].as_f64().unwrap();
    close("mass conservation", -qin, qout, 0.01);
}

#[test]
fn compression_only_contact() {
    let c = new_core();
    let base = [
        ("add_box", json!({"name": "Block", "size": [40, 40, 10]})),
        ("set_mesh", json!({"element_size_mm": 2.5})),
        ("add_compression_only_support", json!({"faces": ["Block.zmin"]})),
        ("add_displacement", json!({"faces": ["Block.xmin"], "x": 0})),
        ("add_displacement", json!({"faces": ["Block.ymin"], "y": 0})),
    ];
    let mut calls: Vec<(&str, Value)> = base.to_vec();
    calls.push(("add_force", json!({"faces": ["Block.zmax"], "vector": [0, 0, -1000]})));
    calls.push(("solve", json!({})));
    let r = run(&c, &calls);
    let rz = r["reactions"].as_array().unwrap()[0]["force_n"][2].as_f64().unwrap();
    close("contact reaction", rz, 1000.0, 0.001);
    // pulling up must lift off (no tension in a compression-only support)
    let c2 = new_core();
    for (t, a) in base.iter() {
        c2.call(t, a).unwrap();
    }
    c2.call("add_force", &json!({"faces": ["Block.zmax"], "vector": [0, 0, 1000]})).unwrap();
    let e = c2.call("solve", &json!({})).err().expect("lift-off should be detected");
    println!("pull-off correctly rejected: {e:#}");
}

#[test]
fn new_model_and_open_keep_theme() {
    let c = new_core();
    c.call("set_display", &json!({"dark_mode": true})).unwrap();
    c.call("add_box", &json!({"size": [10, 10, 10]})).unwrap();
    c.call("new_model", &json!({})).unwrap();
    assert!(c.session.lock().unwrap().display.dark, "new_model must keep dark mode");
    // a project saved in light mode must not switch the theme when opened
    let dir = std::env::temp_dir().join(format!("simcraft-theme-{}.simcraft", std::process::id()));
    let path = dir.to_string_lossy().to_string();
    c.call("set_display", &json!({"dark_mode": false})).unwrap();
    c.call("add_box", &json!({"size": [10, 10, 10]})).unwrap();
    c.call("save_project", &json!({"path": path})).unwrap();
    c.call("set_display", &json!({"dark_mode": true})).unwrap();
    c.call("open_project", &json!({"path": path})).unwrap();
    assert!(c.session.lock().unwrap().display.dark, "open_project must keep dark mode");
    let _ = std::fs::remove_file(&path);
}

#[test]
fn planner_gap_detection() {
    use simcraft::assistant::known_gaps;
    assert!(known_gaps("simulate EMI on my PCB").iter().any(|g| g.contains("S-parameters")));
    // radiation and temperature-dependent conduction are available now
    assert!(known_gaps("heat sink radiation cooling with emissivity 0.9").is_empty());
    // microwave heating itself is supported now; only phase change remains a gap for food
    let mw = known_gaps("food heating in a microwave");
    assert!(mw.iter().all(|g| !g.contains("Full-wave")), "{mw:?}");
    assert!(mw.iter().any(|g| g.contains("phase change")));
    assert!(known_gaps("stress in a steel bracket under 2 kN").is_empty());
}

#[test]
fn heat_generation_in_one_body() {
    // a heated chip on a passive spreader: all generated heat must leave through the cold face
    let c = new_core();
    let r = run(&c, &[
        ("add_box", json!({"name": "Spreader", "min": [0, 0, 0], "size": [40, 40, 4], "material": "Aluminum 6061-T6"})),
        ("add_box", json!({"name": "Chip", "min": [15, 15, 4], "size": [10, 10, 2], "material": "Copper"})),
        ("set_mesh", json!({"element_size_mm": 1})),
        ("set_analysis", json!({"type": "steady_thermal"})),
        ("add_heat_generation", json!({"value": 1e7, "body": "Chip"})),
        ("add_temperature", json!({"faces": ["Spreader.zmin"], "value": 25})),
        ("solve", json!({})),
    ]);
    let q = r["reactions"][0]["heat_flow_w"].as_f64().unwrap();
    // reaction sign convention: positive = heat into the part, so the outflow is negative
    close("heat out = chip power", -q, 1e7 * 10e-3 * 10e-3 * 2e-3, 1e-6);
    assert!(fmax(&r, "temperature") > 25.0);
}

#[test]
fn heat_generation_exact_on_coarse_mesh() {
    // a 15×15×2 mm chip on a 2 mm mesh voxelises to 16×16×2: the power must still be exact
    for (args, want) in [(json!({"value": 1e7, "body": "Chip"}), 1e7 * 15e-3 * 15e-3 * 2e-3), (json!({"power": 3.0, "body": "Chip"}), 3.0)] {
        let c = new_core();
        let r = run(&c, &[
            ("add_box", json!({"name": "Spreader", "min": [0, 0, 0], "size": [40, 40, 4], "material": "Aluminum 6061-T6"})),
            ("add_box", json!({"name": "Chip", "min": [12.5, 12.5, 4], "size": [15, 15, 2], "material": "Copper"})),
            ("set_mesh", json!({"element_size_mm": 2})),
            ("set_analysis", json!({"type": "steady_thermal"})),
            ("add_heat_generation", args),
            ("add_temperature", json!({"faces": ["Spreader.zmin"], "value": 25})),
            ("solve", json!({})),
        ]);
        close("heat out = specified power", -r["reactions"][0]["heat_flow_w"].as_f64().unwrap(), want, 1e-6);
    }
}
