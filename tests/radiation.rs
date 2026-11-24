//! Thermal radiation and temperature-dependent properties against closed forms.
use serde_json::{Value, json};
use simcraft::radiation::{SIGMA, parallel_rectangles_vf};
use simcraft::session::{Shared, new_core};

const K: f64 = 273.15;

fn call(c: &Shared, t: &str, a: Value) -> Value {
    let o = c.call(t, &a).unwrap_or_else(|e| panic!("{t}: {e:#}"));
    serde_json::from_str(&o.text).unwrap_or(Value::String(o.text))
}
fn close(name: &str, got: f64, want: f64, tol: f64) {
    let err = (got - want).abs() / want.abs();
    println!("{name}: got {got:.5}, theory {want:.5}, error {:.2}%", err * 100.0);
    assert!(err < tol, "{name} off by {:.2}%", err * 100.0);
}
fn fmax(r: &Value, f: &str) -> f64 {
    r["fields"][f]["max"].as_f64().unwrap()
}
fn q(r: &Value, name: &str) -> f64 {
    r["quantities"].as_array().unwrap().iter().find(|x| x["name"].as_str().unwrap().starts_with(name)).unwrap_or_else(|| panic!("no {name}: {}", r["quantities"]))["value"].as_f64().unwrap()
}

/// A copper cube dissipating 2 W only by radiation: P = εσA(T⁴ − T∞⁴).
#[test]
fn cube_radiating_to_ambient() {
    let c = new_core();
    call(&c, "add_box", json!({"name": "Cube", "size": [20, 20, 20], "material": "Copper C101"}));
    call(&c, "set_mesh", json!({"element_size_mm": 2}));
    call(&c, "set_analysis", json!({"type": "steady_thermal"}));
    call(&c, "add_heat_generation", json!({"body": "Cube", "power": 2.0}));
    call(&c, "add_radiation", json!({"faces": ["all"], "emissivity": 0.8, "ambient": 20}));
    let r = call(&c, "solve", json!({}));
    let ta = 20.0 + K;
    let t = (ta.powi(4) + 2.0 / (0.8 * SIGMA * 6.0 * 4e-4)).powf(0.25) - K;
    println!("{} iterations", q(&r, "Non-linear iterations"));
    close("cube temperature rise", fmax(&r, "temperature") - 20.0, t - 20.0, 0.005);
    close("radiated power", q(&r, "Heat radiated through"), 2.0, 0.002);
}

/// Two parallel plates exchanging radiation (s2s), ambient at 0 K, back faces held at T1 / T2.
fn plates(eps: f64) -> (Value, f64) {
    let c = new_core();
    call(&c, "add_box", json!({"name": "Hot", "min": [0, 0, 0], "size": [100, 100, 4], "material": "Copper C101"}));
    call(&c, "add_box", json!({"name": "Cold", "min": [0, 0, 54], "size": [100, 100, 4], "material": "Copper C101"}));
    call(&c, "set_mesh", json!({"element_size_mm": 2}));
    call(&c, "set_analysis", json!({"type": "steady_thermal"}));
    call(&c, "add_temperature", json!({"faces": ["Hot.zmin"], "value": 500}));
    call(&c, "add_temperature", json!({"faces": ["Cold.zmax"], "value": 100}));
    call(&c, "add_radiation", json!({"faces": ["Hot.zmax", "Cold.zmin"], "emissivity": eps, "ambient": -K, "enclosure": true}));
    let r = call(&c, "solve", json!({}));
    (r, parallel_rectangles_vf(100.0, 100.0, 50.0))
}

#[test]
fn view_factor_formula() {
    // Howell catalogue C-11 values: X = Y = 1 → 0.1998; X = Y = 2 → 0.4152
    close("F(1,1)", parallel_rectangles_vf(1.0, 1.0, 1.0), 0.1998, 0.001);
    close("F(2,2)", parallel_rectangles_vf(2.0, 2.0, 1.0), 0.4152, 0.001);
}

#[test]
fn black_parallel_plates() {
    let (r, f) = plates(1.0);
    let (t1, t2) = (500.0 + K, 100.0 + K);
    let a = 0.01;
    // black surfaces, ambient 0 K: the hot plate emits σT1⁴ and absorbs F σT2⁴
    let want = a * SIGMA * (t1.powi(4) - f * t2.powi(4));
    let hot_in = r["reactions"][0]["heat_flow_w"].as_f64().unwrap();
    println!("F12 = {f:.4}; {} patches, {} rays", q(&r, "Radiation patches"), q(&r, "View-factor rays"));
    close("heat into the hot plate", hot_in, want, 0.02);
}

#[test]
fn grey_parallel_plates() {
    let eps = 0.5;
    let (r, f) = plates(eps);
    let (t1, t2) = (500.0 + K, 100.0 + K);
    let (e1, e2) = (eps * SIGMA * t1.powi(4), eps * SIGMA * t2.powi(4));
    // radiosity network with the exact view factor: J1 = εσT1⁴ + (1−ε) F J2, J2 = εσT2⁴ + (1−ε) F J1
    let b = (1.0 - eps) * f;
    let j1 = (e1 + b * e2) / (1.0 - b * b);
    let j2 = e2 + b * j1;
    let want = 0.01 * (j1 - f * j2);
    close("heat into the hot plate (ε = 0.5)", r["reactions"][0]["heat_flow_w"].as_f64().unwrap(), want, 0.025);
}

/// Slab with k = 50 (1 + 0.005 T): q = (A/L) ∫ k dT (Kirchhoff).
#[test]
fn temperature_dependent_conductivity() {
    let c = new_core();
    call(&c, "define_material", json!({"name": "kT", "based_on": "Structural Steel", "thermal_conductivity": 50, "conductivity_table": [[0, 50], [200, 100]]}));
    call(&c, "add_box", json!({"name": "Slab", "size": [50, 10, 10], "material": "kT"}));
    call(&c, "set_mesh", json!({"element_size_mm": 2.5}));
    call(&c, "set_analysis", json!({"type": "steady_thermal"}));
    call(&c, "add_temperature", json!({"faces": ["Slab.xmin"], "value": 200}));
    call(&c, "add_temperature", json!({"faces": ["Slab.xmax"], "value": 0}));
    let r = call(&c, "solve", json!({}));
    let want = 1e-4 / 0.05 * 50.0 * (200.0 + 0.0025 * 200.0 * 200.0);
    close("heat flow with k(T)", r["reactions"][0]["heat_flow_w"].as_f64().unwrap(), want, 0.005);
    // the profile is no longer linear: hotter half conducts better, so the midpoint is above 100 °C
    let mid = call(&c, "probe", json!({"point": [25, 5, 5]}))["values"]["temperature"].as_f64().unwrap();
    let want_mid = (-1.0 + (1.0f64 + 0.01 * (0.5 * (200.0 + 0.0025 * 40000.0))).sqrt()) / 0.005;
    close("midpoint temperature", mid, want_mid, 0.01);
}

/// Adiabatic block heated uniformly with c = 400 (1 + 0.001 T): ρ ∫ c dT = q t.
#[test]
fn temperature_dependent_specific_heat() {
    let c = new_core();
    call(&c, "define_material", json!({"name": "cT", "based_on": "Structural Steel", "specific_heat": 400, "specific_heat_table": [[0, 400], [1000, 800]]}));
    call(&c, "add_box", json!({"name": "Block", "size": [10, 10, 10], "material": "cT"}));
    call(&c, "set_mesh", json!({"element_size_mm": 2.5}));
    call(&c, "set_analysis", json!({"type": "transient_thermal", "end_time": 100, "time_steps": 200, "initial_temperature": 20}));
    call(&c, "add_heat_generation", json!({"value": 1e7}));
    let r = call(&c, "solve", json!({}));
    let rho = call(&c, "list_materials", json!({})).as_array().unwrap().iter().find(|m| m["name"] == "cT").unwrap()["density"].as_f64().unwrap();
    // ρ·400 [(T − T0) + 0.0005 (T² − T0²)] = q t
    let (t0, e) = (20.0, 1e7 * 100.0 / (rho * 400.0));
    let (a, b, cc) = (0.0005, 1.0, -(e + t0 + 0.0005 * t0 * t0));
    let want = (-b + (b * b - 4.0 * a * cc).sqrt()) / (2.0 * a);
    close("temperature with c(T)", fmax(&r, "temperature") - t0, want - t0, 0.01);
}

/// Hot copper cube cooling by radiation (lumped): ρcV dT/dt = −εσA(T⁴ − T∞⁴).
#[test]
fn transient_radiative_cooling() {
    let c = new_core();
    call(&c, "add_box", json!({"name": "Cube", "size": [10, 10, 10], "material": "Copper C101"}));
    call(&c, "set_mesh", json!({"element_size_mm": 2.5}));
    call(&c, "set_analysis", json!({"type": "transient_thermal", "end_time": 600, "time_steps": 300, "initial_temperature": 500}));
    call(&c, "add_radiation", json!({"faces": ["all"], "emissivity": 0.9, "ambient": 20}));
    let r = call(&c, "solve", json!({}));
    let m = call(&c, "list_materials", json!({}));
    let cu = m.as_array().unwrap().iter().find(|x| x["name"] == "Copper C101").unwrap();
    let rcv = cu["density"].as_f64().unwrap() * cu["specific_heat"].as_f64().unwrap() * 1e-6;
    let (ta, mut t) = (20.0 + K, 500.0 + K);
    let dt = 0.01;
    for _ in 0..60000 {
        let f = |t: f64| -0.9 * SIGMA * 6e-4 * (t.powi(4) - ta.powi(4)) / rcv;
        let (k1, k2) = (f(t), f(t + 0.5 * dt * f(t)));
        let k3 = f(t + 0.5 * dt * k2);
        let k4 = f(t + dt * k3);
        t += dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
    }
    close("temperature after 600 s", fmax(&r, "temperature"), t - K, 0.01);
}

/// Conjugate heat transfer: a heated block in a duct whose bottom face (on the duct wall) also
/// radiates to the surroundings — the generated heat leaves with the air plus by radiation.
#[test]
fn cht_with_radiation_energy_balance() {
    let c = new_core();
    call(&c, "add_box", json!({"name": "Block", "min": [40, 0, 0], "size": [10, 4, 6], "material": "Aluminum 6061-T6"}));
    call(&c, "add_enclosure", json!({"min": [0, 0, 0], "size": [120, 4, 20]}));
    call(&c, "set_analysis", json!({"type": "conjugate_heat_transfer", "max_iterations": 40000}));
    call(&c, "set_mesh", json!({"element_size_mm": 1}));
    call(&c, "add_inlet", json!({"faces": ["Enclosure.xmin"], "velocity": 0.1, "temperature": 20}));
    call(&c, "add_outlet", json!({"faces": ["Enclosure.xmax"], "pressure": 0}));
    call(&c, "add_symmetry", json!({"faces": ["Enclosure.ymin", "Enclosure.ymax"]}));
    call(&c, "add_heat_generation", json!({"body": "Block", "power": 0.2}));
    call(&c, "add_radiation", json!({"faces": ["Block.zmin"], "emissivity": 1.0, "ambient": 20}));
    let r = call(&c, "solve", json!({}));
    let carried = q(&r, "Heat carried away by the flow");
    let radiated = -q(&r, "Heat flow through #");
    println!("carried {carried:.4} W, radiated {radiated:.4} W, block max {:.1} °C", q(&r, "Max temperature of Block"));
    assert!(radiated > 0.005, "the hot block bottom must radiate: {radiated}");
    close("carried + radiated = generated", carried + radiated, 0.2, 0.01);
    // and with radiation the block runs cooler than the convection-only case (~230 K rise)
    assert!(q(&r, "Max temperature of Block") < 240.0);
}
