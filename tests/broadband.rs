//! P2.6 broadband FDTD: lumped ports, S-parameters, input impedance and emissions.
use serde_json::{Value, json};
use simcraft::session::{Shared, new_core};

fn call(c: &Shared, tool: &str, args: Value) -> Value {
    let r = c.call(tool, &args).unwrap_or_else(|e| panic!("{tool}({args}) failed: {e:#}"));
    serde_json::from_str(&r.text).unwrap_or_else(|_| json!({ "text": r.text }))
}

fn solve(c: &Shared) -> Value {
    let _ = call(c, "solve", json!({}));
    while c.is_busy() {
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    let st = call(c, "get_status", json!({}));
    if let Some(e) = st["last_error"].as_str() {
        panic!("async job failed: {e}");
    }
    call(c, "get_results", json!({}))
}

fn qty(r: &Value, name: &str) -> f64 {
    r["quantities"].as_array().unwrap().iter().find(|q| q["name"].as_str().unwrap().starts_with(name)).unwrap_or_else(|| panic!("no '{name}' in {}", r["quantities"]))["value"].as_f64().unwrap()
}

/// Centre-fed dipole (145 mm, 5 mm square arms, L/a ≈ 50): first series resonance just below
/// λ/2 with Rin ≈ 65–75 Ω (King / Hallén), directivity 2.15 dBi, radiated = delivered power.
#[test]
fn dipole_input_impedance() {
    let c = new_core();
    call(&c, "new_model", json!({}));
    call(&c, "add_box", json!({"name": "ArmA", "min": [-2.5, -2.5, -75], "size": [5, 5, 70], "material": "Copper C101"}));
    call(&c, "add_box", json!({"name": "ArmB", "min": [-2.5, -2.5, 0], "size": [5, 5, 70], "material": "Copper C101"}));
    // λ/4 of clearance at the band centre keeps the absorber out of the near field
    call(&c, "add_enclosure", json!({"min": [-152.5, -152.5, -225], "size": [305, 305, 445]}));
    call(&c, "set_analysis", json!({"type": "broadband_em", "freq_min_ghz": 0.5, "freq_max_ghz": 1.5, "em_boundary": "open", "freq_steps": 201}));
    call(&c, "set_mesh", json!({"element_size_mm": 5}));
    call(&c, "add_lumped_port", json!({"from": [-2.5, -2.5, -5], "to": [-2.5, -2.5, 0], "impedance": 73}));
    let r = solve(&c);
    let fr = qty(&r, "Port #1: resonance");
    let rr = qty(&r, "Port #1: Re Zin at resonance");
    eprintln!("resonance {fr:.4} GHz, R {rr:.1} Ω, S11 min {:.1} dB; {}", qty(&r, "Minimum |S11|"), r["warnings"]);
    for q in r["quantities"].as_array().unwrap() {
        eprintln!("  {} = {} {}", q["name"], q["value"], q["unit"]);
    }
    let lam_half = 0.299_792_458 / (2.0 * 0.150);
    assert!(fr > 0.85 * lam_half && fr < 1.0 * lam_half, "resonance {fr} GHz vs λ/2 at {lam_half}");
    assert!(rr > 63.0 && rr < 78.0, "Rin {rr}");
    // energy conservation: a lossless dipole radiates what the port delivers
    let bal = qty(&r, "Radiated / delivered power");
    assert!((bal - 100.0).abs() < 4.0, "radiated / delivered {bal} %");
    let d = qty(&r, "Directivity at the band centre");
    assert!((d - 2.15).abs() < 0.4, "directivity {d} dBi (half-wave dipole 2.15 dBi)");
}

#[derive(Clone, Copy, Debug)]
struct C(f64, f64);
impl C {
    fn add(self, o: C) -> C { C(self.0 + o.0, self.1 + o.1) }
    fn sub(self, o: C) -> C { C(self.0 - o.0, self.1 - o.1) }
    fn mul(self, o: C) -> C { C(self.0 * o.0 - self.1 * o.1, self.0 * o.1 + self.1 * o.0) }
    fn div(self, o: C) -> C { let d = o.0 * o.0 + o.1 * o.1; C((self.0 * o.0 + self.1 * o.1) / d, (self.1 * o.0 - self.0 * o.1) / d) }
}

/// S11, S21 of a microstrip of length `len` mm between two 50 Ω lumped ports, by frequency.
fn microstrip(len: f64) -> (Vec<f64>, Vec<(C, C)>) {
    let c = new_core();
    let x0 = len / 2.0;
    call(&c, "new_model", json!({}));
    call(&c, "add_box", json!({"name": "Ground", "min": [-x0 - 5.0, -8, 0], "size": [len + 10.0, 16, 0.5], "material": "Copper C101"}));
    call(&c, "add_box", json!({"name": "Substrate", "min": [-x0 - 5.0, -8, 0.5], "size": [len + 10.0, 16, 2], "material": "FR-4 (PCB laminate)"}));
    call(&c, "add_box", json!({"name": "Strip", "min": [-x0, -2, 2.5], "size": [len, 4, 0.5], "material": "Copper C101"}));
    call(&c, "add_enclosure", json!({"min": [-x0 - 5.0, -8, 0], "size": [len + 10.0, 16, 9]}));
    call(&c, "set_analysis", json!({"type": "broadband_em", "freq_min_ghz": 0.5, "freq_max_ghz": 3, "em_boundary": "open", "freq_steps": 51}));
    call(&c, "set_mesh", json!({"element_size_mm": 0.5}));
    call(&c, "add_lumped_port", json!({"from": [-x0, 0, 0.5], "to": [-x0, 0, 2.5], "impedance": 50}));
    call(&c, "add_lumped_port", json!({"from": [x0, 0, 0.5], "to": [x0, 0, 2.5], "impedance": 50, "voltage": 0}));
    let r = solve(&c);
    let sp = &r["s_parameters"];
    let f: Vec<f64> = sp["freq_hz"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    let get = |k: &str, i: usize| C(sp[k][i][0].as_f64().unwrap(), sp[k][i][1].as_f64().unwrap());
    (f.clone(), (0..f.len()).map(|i| (get("S11", i), get("S21", i))).collect())
}

/// ABCD of a symmetric reciprocal two-port from (S11, S21), reference 50 Ω.
fn abcd(s11: C, s21: C) -> [C; 4] {
    let one = C(1.0, 0.0);
    let d = s21.add(s21);
    let a = one.add(s11).mul(one.sub(s11)).add(s21.mul(s21)).div(d);
    let b = one.add(s11).mul(one.add(s11)).sub(s21.mul(s21)).div(d).mul(C(50.0, 0.0));
    let c = one.sub(s11).mul(one.sub(s11)).sub(s21.mul(s21)).div(d).mul(C(1.0 / 50.0, 0.0));
    [a, b, c, a]
}

/// Microstrip (w = 4 mm, h = 2 mm, t = 0.5 mm on εr 4.4) between 50 Ω lumped ports, de-embedded
/// with two line lengths: εeff from trace(M₁M₂⁻¹) = 2 cosh(γΔL) (port fixtures cancel) and
/// Zc = sinh(γL)/C (immune to the series inductance of the vertical port feeds), against
/// Hammerstad–Jensen with Wheeler's thickness correction.
#[test]
fn microstrip_impedance() {
    let (f, m40) = microstrip(40.0);
    let (_, m20) = microstrip(20.0);
    let (w, h, t, er) = (4.0f64, 2.0f64, 0.5f64, 4.4f64);
    let we = w + t / std::f64::consts::PI * (1.0 + (2.0 * h / t).ln());
    let u = we / h;
    let eeff = (er + 1.0) / 2.0 + (er - 1.0) / 2.0 / (1.0 + 12.0 / u).sqrt();
    let z0 = 120.0 * std::f64::consts::PI / (eeff.sqrt() * (u + 1.393 + 0.667 * (u + 1.444).ln()));
    let mut checked = 0;
    for i in (2..f.len()).step_by(4) {
        let m1 = abcd(m40[i].0, m40[i].1);
        let m2 = abcd(m20[i].0, m20[i].1);
        // M1 · M2⁻¹ (det M2 = 1 for a reciprocal network): inv = [D, −B; −C, A]
        let tr = m1[0].mul(m2[3]).sub(m1[1].mul(m2[2])).add(m1[3].mul(m2[0]).sub(m1[2].mul(m2[1])));
        let bdl = (tr.0 / 2.0).clamp(-1.0, 1.0).acos(); // β·ΔL (lossless part)
        let beta = bdl / 0.020;
        let ee = (beta * 299_792_458.0 / (std::f64::consts::TAU * f[i])).powi(2);
        // Zc = sinh(γL)/C ≈ j sin(βL)/C for the 40 mm line
        let s = C(0.0, (beta * 0.040).sin());
        let zc = s.div(m1[2]);
        eprintln!("{:.2} GHz: Zc {:.1}{:+.1}j Ω (H–J {z0:.1}), εeff {ee:.2} (H–J {eeff:.2})", f[i] / 1e9, zc.0, zc.1);
        // εeff over the whole band; Zc below 1.25 GHz, where the port feeds (2 mm vertical
        // lumped ports, modelled here as a series impedance only) are electrically small
        assert!((ee - eeff).abs() < 0.06 * eeff, "εeff {ee} vs {eeff} at {} GHz", f[i] / 1e9);
        if f[i] <= 1.25e9 {
            assert!((zc.0 - z0).abs() < 0.05 * z0 && zc.1.abs() < 0.1 * z0, "Zc {zc:?} vs Hammerstad–Jensen {z0} at {} GHz", f[i] / 1e9);
            checked += 1;
        }
    }
    assert!(checked >= 4, "only {checked} frequencies checked");
}

/// Probe-fed rectangular patch (L 30 × W 38 mm on 2 mm, εr 4.4): S11 dips at the TM010
/// resonance; the transmission-line (cavity) model gives f = c / (2 (L + 2ΔL) √εeff).
#[test]
fn patch_antenna_resonance() {
    let c = new_core();
    call(&c, "new_model", json!({}));
    call(&c, "add_box", json!({"name": "Ground", "min": [-30, -30, 0], "size": [60, 60, 1], "material": "Copper C101"}));
    call(&c, "add_box", json!({"name": "Substrate", "min": [-30, -30, 1], "size": [60, 60, 2], "material": "FR-4 (PCB laminate)"}));
    call(&c, "add_box", json!({"name": "Patch", "min": [-15, -19, 3], "size": [30, 38, 1], "material": "Copper C101"}));
    call(&c, "add_enclosure", json!({"min": [-45, -45, -15], "size": [90, 90, 35]}));
    call(&c, "set_analysis", json!({"type": "broadband_em", "freq_min_ghz": 1.5, "freq_max_ghz": 3.5, "em_boundary": "open", "freq_steps": 201, "max_periods": 2000}));
    call(&c, "set_mesh", json!({"element_size_mm": 1}));
    // probe 6 mm in from the radiating edge along the resonant length (x)
    call(&c, "add_lumped_port", json!({"from": [-9, 0, 1], "to": [-9, 0, 3], "impedance": 50}));
    let r = solve(&c);
    let fmin = qty(&r, "Port #1: best match at");
    let s11 = qty(&r, "Minimum |S11|");
    let (l, w, h, er) = (30.0f64, 38.0f64, 2.0f64, 4.4f64);
    let eeff = (er + 1.0) / 2.0 + (er - 1.0) / 2.0 / (1.0 + 12.0 * h / w).sqrt();
    let dl = 0.412 * h * (eeff + 0.3) * (w / h + 0.264) / ((eeff - 0.258) * (w / h + 0.8));
    let f_cavity = 299.792_458 / (2.0 * (l + 2.0 * dl) * eeff.sqrt());
    eprintln!("S11 minimum {s11:.1} dB at {fmin:.3} GHz; cavity model {f_cavity:.3} GHz; warnings {}", r["warnings"]);
    assert!(s11 < -6.0, "no clear resonance (|S11| min {s11} dB)");
    assert!((fmin - f_cavity).abs() < 0.06 * f_cavity, "patch resonance {fmin} GHz vs cavity model {f_cavity} GHz");
}

/// Emissions over a ground plane: a small horizontal dipole at 1 m above a metal ground plane,
/// received at 3 m with a 1–4 m height scan. Image theory for a horizontal current gives at most
/// a 2× (6 dB) gain over free space; the scan must land between 0 and +6.5 dB of free space.
#[test]
fn emissions_ground_plane() {
    let run = |gp: bool| -> Value {
        let c = new_core();
        call(&c, "new_model", json!({}));
        call(&c, "add_box", json!({"name": "ArmA", "min": [-40, -2.5, -2.5], "size": [37.5, 5, 5], "material": "Copper C101"}));
        call(&c, "add_box", json!({"name": "ArmB", "min": [2.5, -2.5, -2.5], "size": [37.5, 5, 5], "material": "Copper C101"}));
        call(&c, "add_enclosure", json!({"min": [-112.5, -72.5, -72.5], "size": [225, 145, 145]}));
        call(&c, "set_analysis", json!({"type": "broadband_em", "freq_min_ghz": 0.3, "freq_max_ghz": 1.0, "em_boundary": "open", "freq_steps": 41, "ground_plane": gp, "eut_height_m": 1.0}));
        call(&c, "set_mesh", json!({"element_size_mm": 5}));
        call(&c, "add_lumped_port", json!({"from": [-2.5, -2.5, -2.5], "to": [2.5, -2.5, -2.5], "impedance": 50}));
        solve(&c)
    };
    let (fs, gp) = (run(false), run(true));
    let pick = |r: &Value, pre: &str| -> Vec<(f64, f64)> {
        let s = r["curves"]["series"].as_object().unwrap();
        let (_, v) = s.iter().find(|(k, _)| k.starts_with(pre)).unwrap_or_else(|| panic!("no curve {pre} in {:?}", s.keys().collect::<Vec<_>>()));
        v.as_array().unwrap().iter().map(|p| (p[0].as_f64().unwrap(), p[1].as_f64().unwrap())).collect()
    };
    let (a, b) = (pick(&fs, "E at 3 m"), pick(&gp, "E at 3 m"));
    for (x, y) in a.iter().zip(&b) {
        let d = y.1 - x.1;
        eprintln!("{:.3} GHz: free space {:.1}, ground plane {:.1} dBµV/m (+{d:.1} dB)", x.0, x.1, y.1);
        assert!(d > -0.5 && d < 6.5, "ground-plane gain {d} dB at {} GHz", x.0);
    }
    assert!(qty(&gp, "Margin to FCC class B").is_finite());
}
