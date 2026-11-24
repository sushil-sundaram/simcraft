//! CFD and conjugate heat transfer against published benchmarks.
use serde_json::{Value, json};
use simcraft::session::{Shared, new_core};

fn call(c: &Shared, t: &str, a: Value) -> Value {
    let o = c.call(t, &a).unwrap_or_else(|e| panic!("{t}: {e:#}"));
    serde_json::from_str(&o.text).unwrap_or(Value::String(o.text))
}
fn close(name: &str, got: f64, want: f64, tol: f64) {
    let err = (got - want).abs() / want.abs();
    println!("{name}: got {got:.5}, reference {want:.5}, error {:.2}%", err * 100.0);
    assert!(err < tol, "{name} off by {:.1}%", err * 100.0);
}
fn q(r: &Value, name: &str) -> f64 {
    r["quantities"].as_array().unwrap().iter().find(|x| x["name"].as_str().unwrap().starts_with(name)).unwrap_or_else(|| panic!("no {name}: {}", r["quantities"]))["value"].as_f64().unwrap()
}
fn probe(c: &Shared, p: [f64; 3], field: &str) -> f64 {
    call(c, "probe", json!({"point": p}))["values"][field].as_f64().unwrap()
}
/// Linear interpolation between the grid nodes bracketing `p` along `axis` (node spacing `h`).
fn probe_lerp(c: &Shared, p: [f64; 3], axis: usize, h: f64, field: &str) -> f64 {
    let lo = (p[axis] / h).floor() * h;
    let w = (p[axis] - lo) / h;
    let (mut a, mut b) = (p, p);
    a[axis] = lo;
    b[axis] = lo + h;
    (1.0 - w) * probe(c, a, field) + w * probe(c, b, field)
}
const NU_AIR: f64 = 1.81e-5 / 1.204;

/// Lid-driven cavity, Re = 100 (2-D: one cell thick with free-slip sides) vs Ghia, Ghia & Shin (1982).
#[test]
fn lid_driven_cavity_ghia_re100() {
    let c = new_core();
    let u = 100.0 * NU_AIR / 0.1;
    call(&c, "add_enclosure", json!({"min": [0, 0, 0], "size": [100, 2, 100]}));
    call(&c, "set_analysis", json!({"type": "fluid", "max_iterations": 30000}));
    call(&c, "set_mesh", json!({"element_size_mm": 2}));
    call(&c, "add_moving_wall", json!({"faces": ["Enclosure.zmax"], "velocity": [u, 0, 0]}));
    call(&c, "add_symmetry", json!({"faces": ["Enclosure.ymin", "Enclosure.ymax"]}));
    let r = call(&c, "solve", json!({}));
    println!("{} iterations, {:?}", r["info"]["iterations"], r["warnings"]);
    // u on the vertical centreline, v on the horizontal centreline (Ghia Table I/II, Re = 100)
    let gu = [(0.9531, 0.68717), (0.8516, 0.23151), (0.7344, 0.00332), (0.6172, -0.13641), (0.5, -0.20581), (0.4531, -0.21090), (0.2813, -0.15662), (0.1719, -0.10150), (0.1016, -0.06434)];
    let gv = [(0.0938, 0.12317), (0.1563, 0.16077), (0.2344, 0.17527), (0.5, 0.05454), (0.8047, -0.24533), (0.9063, -0.16914), (0.9453, -0.10313)];
    let mut worst: f64 = 0.0;
    for (y, ref_u) in gu {
        let got = probe_lerp(&c, [50.0, 1.0, y * 100.0], 2, 2.0, "velocity_x") / u;
        println!("  u({y:.4}) = {got:+.4}  Ghia {ref_u:+.4}");
        worst = worst.max((got - ref_u).abs());
    }
    for (x, ref_v) in gv {
        let got = probe_lerp(&c, [x * 100.0, 1.0, 50.0], 0, 2.0, "velocity_z") / u;
        println!("  v({x:.4}) = {got:+.4}  Ghia {ref_v:+.4}");
        worst = worst.max((got - ref_v).abs());
    }
    println!("largest deviation from Ghia: {worst:.4} U");
    assert!(worst < 0.02, "largest deviation {worst:.3} U");
}

/// Differentially heated square cavity (de Vahl Davis 1983): mean Nusselt number on the hot wall.
fn natural_convection(ra: f64, cells: usize) -> f64 {
    let c = new_core();
    let alpha = 0.026 / (1.204 * 1006.0);
    let (beta, dt, g) = (3.4e-3, 10.0, 9.80665);
    let l = (ra * NU_AIR * alpha / (g * beta * dt)).cbrt() * 1e3; // mm
    let h = l / cells as f64;
    call(&c, "add_enclosure", json!({"min": [0, 0, 0], "size": [l, h, l]}));
    call(&c, "set_analysis", json!({"type": "conjugate_heat_transfer", "max_iterations": 60000, "reference_temperature": 22}));
    call(&c, "set_mesh", json!({"element_size_mm": h}));
    call(&c, "add_symmetry", json!({"faces": ["Enclosure.ymin", "Enclosure.ymax"]}));
    let hot = call(&c, "add_temperature", json!({"faces": ["Enclosure.xmin"], "value": 27}));
    call(&c, "add_temperature", json!({"faces": ["Enclosure.xmax"], "value": 17}));
    call(&c, "add_gravity", json!({"acceleration": [0, 0, -g]}));
    let r = call(&c, "solve", json!({}));
    println!("Ra {ra:.0e}: {} iterations, {} — {:?}", r["info"]["iterations"], r["info"]["solver"], r["warnings"]);
    let id = hot["added"]["id"].as_u64().unwrap_or(2);
    let qh = r["quantities"].as_array().unwrap().iter().find(|x| x["name"].as_str().unwrap().starts_with(&format!("Heat flow through #{id}"))).unwrap()["value"].as_f64().unwrap();
    qh * (l * 1e-3) / (0.026 * dt * (l * 1e-3) * (h * 1e-3))
}

#[test]
fn natural_convection_de_vahl_davis_ra1e4() {
    close("Nu, Ra = 1e4", natural_convection(1e4, 40), 2.243, 0.03);
}

#[test]
fn natural_convection_de_vahl_davis_ra1e5() {
    close("Nu, Ra = 1e5", natural_convection(1e5, 64), 4.519, 0.04);
}

/// Heated aluminium block in a duct: everything generated must leave with the air.
#[test]
fn conjugate_heat_balance_heated_block() {
    let c = new_core();
    call(&c, "add_box", json!({"name": "Block", "min": [40, 0, 0], "size": [10, 4, 6], "material": "Aluminum 6061-T6"}));
    call(&c, "add_enclosure", json!({"min": [0, 0, 0], "size": [120, 4, 20]}));
    call(&c, "set_analysis", json!({"type": "conjugate_heat_transfer", "max_iterations": 40000}));
    call(&c, "set_mesh", json!({"element_size_mm": 1}));
    call(&c, "add_inlet", json!({"faces": ["Enclosure.xmin"], "velocity": 0.1, "temperature": 20}));
    call(&c, "add_outlet", json!({"faces": ["Enclosure.xmax"], "pressure": 0}));
    call(&c, "add_symmetry", json!({"faces": ["Enclosure.ymin", "Enclosure.ymax"]}));
    call(&c, "add_heat_generation", json!({"body": "Block", "value": 0.2 / (10e-3 * 4e-3 * 6e-3)})); // 0.2 W, block Re ≈ 40 (steady)
    let r = call(&c, "solve", json!({}));
    println!("{} iterations, {} — {:?}", r["info"]["iterations"], r["info"]["solver"], r["warnings"]);
        assert!(r["warnings"].as_array().unwrap().is_empty(), "{}", r["warnings"]);
    close("heat generated", q(&r, "Heat generated"), 0.2, 1e-6);
    close("heat carried away by the air", q(&r, "Heat carried away by the flow"), 0.2, 0.01);
    let tmax = q(&r, "Max temperature of Block");
    println!("block max temperature {tmax:.2} °C");
    // 0.2 W through ~88 mm² of exposed surface at h ≈ 10 W/m²K (laminar, 0.1 m/s) → ~230 K rise
    assert!(tmax > 120.0 && tmax < 400.0, "block max temperature {tmax}");
}

/// Turbulent plane channel (Re_m = 20 000) with LVEL: wall friction vs Dean's correlation
/// Cf = 0.073 Re_m^-1/4, from the developed pressure gradient dp/dx = τw / δ.
#[test]
fn lvel_turbulent_channel_friction() {
    let c = new_core();
    let (ub, two_delta) = (15.0, 0.02);
    call(&c, "add_enclosure", json!({"min": [0, 0, 0], "size": [1000, 1, 20]}));
    call(&c, "set_analysis", json!({"type": "fluid", "turbulence": "lvel", "max_iterations": 60000}));
    call(&c, "set_mesh", json!({"element_size_mm": 1}));
    call(&c, "add_inlet", json!({"faces": ["Enclosure.xmin"], "velocity": ub}));
    call(&c, "add_outlet", json!({"faces": ["Enclosure.xmax"], "pressure": 0}));
    call(&c, "add_symmetry", json!({"faces": ["Enclosure.ymin", "Enclosure.ymax"]}));
    let r = call(&c, "solve", json!({}));
    println!("{} iterations, {} — {:?}, ν ratio {}", r["info"]["iterations"], r["info"]["solver"], r["warnings"], q(&r, "Peak turbulent"));
    let dpdx = (probe(&c, [600.0, 0.5, 10.0], "fluid_pressure") - probe(&c, [900.0, 0.5, 10.0], "fluid_pressure")) / 0.3;
    let re = ub * two_delta / NU_AIR;
    let cf = 0.073 * re.powf(-0.25);
    let tau = cf * 0.5 * 1.204 * ub * ub;
    close("developed dp/dx (Dean)", dpdx, tau / (two_delta / 2.0), 0.15);
}

/// 3-D forced-convection electronics box (~90k cells, LVEL, two 5 W parts): SIMPLE vs the
/// projection method. `cargo test --release --test cfd scheme_benchmark -- --ignored --nocapture`
#[test]
#[ignore]
fn scheme_benchmark() {
    for scheme in ["projection", "simple"] {
        let c = new_core();
        call(&c, "add_enclosure", json!({"min": [0, 0, 0], "size": [200, 120, 60]}));
        call(&c, "add_box", json!({"name": "Board", "min": [20, 10, 10], "size": [160, 100, 2], "material": "FR-4 (PCB laminate)"}));
        call(&c, "add_box", json!({"name": "U1", "min": [60, 40, 12], "size": [20, 20, 6], "material": "Aluminum 6061-T6"}));
        call(&c, "add_box", json!({"name": "U2", "min": [120, 60, 12], "size": [20, 20, 10], "material": "Aluminum 6061-T6"}));
        call(&c, "set_analysis", json!({"type": "conjugate_heat_transfer", "turbulence": "lvel", "max_iterations": 30000, "cfd_scheme": scheme}));
        call(&c, "set_mesh", json!({"element_size_mm": 2.5}));
        call(&c, "add_inlet", json!({"faces": ["Enclosure.xmin"], "velocity": 1.5, "temperature": 25}));
        call(&c, "add_outlet", json!({"faces": ["Enclosure.xmax"], "pressure": 0}));
        call(&c, "add_heat_generation", json!({"body": "U1", "power": 5}));
        call(&c, "add_heat_generation", json!({"body": "U2", "power": 5}));
        let t = std::time::Instant::now();
        let r = call(&c, "solve", json!({}));
        println!("{scheme}: {:.1} s, {} iterations, {} cells — {}", t.elapsed().as_secs_f64(), r["info"]["iterations"], r["info"]["elements"], r["info"]["solver"]);
        println!("   U1 max {:.2} °C, U2 max {:.2} °C, carried {:.3} W, drag {:.4e} N, warnings {}", q(&r, "Max temperature of U1"), q(&r, "Max temperature of U2"), q(&r, "Heat carried away"), q(&r, "Force on solids X"), r["warnings"]);
    }
}

/// Cylinder in a channel at Re 200 (2-D): the wake sheds vortices, so a steady scheme cannot
/// settle — `auto` must still return a sensible (time-averaged) answer with a warning.
#[test]
#[ignore]
fn unsteady_wake_fallback() {
    let c = new_core();
    let u = 200.0 * NU_AIR / 0.02;
    call(&c, "add_enclosure", json!({"min": [0, 0, 0], "size": [300, 1, 100]}));
    call(&c, "add_cylinder", json!({"name": "Pin", "base": [80, 0, 50], "axis": "y", "radius": 10, "height": 1}));
    call(&c, "set_analysis", json!({"type": "fluid", "max_iterations": 20000}));
    call(&c, "set_mesh", json!({"element_size_mm": 1}));
    call(&c, "add_inlet", json!({"faces": ["Enclosure.xmin"], "velocity": u}));
    call(&c, "add_outlet", json!({"faces": ["Enclosure.xmax"], "pressure": 0}));
    call(&c, "add_symmetry", json!({"faces": ["Enclosure.ymin", "Enclosure.ymax", "Enclosure.zmin", "Enclosure.zmax"]}));
    let t = std::time::Instant::now();
    let r = call(&c, "solve", json!({}));
    let cd = q(&r, "Force on solids X") / (0.5 * 1.204 * u * u * 0.02 * 0.001);
    println!("{:.1} s, {} iterations — {}\n  Cd {cd:.3}, warnings {}", t.elapsed().as_secs_f64(), r["info"]["iterations"], r["info"]["solver"], r["warnings"]);
}
