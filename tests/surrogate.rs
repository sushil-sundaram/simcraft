//! Gaussian-process surrogate: accuracy on known functions.
use simcraft::surrogate::{Gp, big_phi, expected_improvement};

fn lhs(n: usize, m: usize) -> Vec<Vec<f64>> {
    // simple stratified design with a fixed permutation per dimension
    (0..m).map(|i| (0..n).map(|k| (((i * (2 * k + 3) + k * 7) % m) as f64 + 0.5) / m as f64).collect()).collect()
}

#[test]
fn interpolates_and_generalises_smooth_2d_function() {
    // Branin-like smooth function on [0,1]²
    let f = |u: &[f64]| {
        let (x, y) = (15.0 * u[0] - 5.0, 15.0 * u[1]);
        (y - 5.1 / (4.0 * std::f64::consts::PI.powi(2)) * x * x + 5.0 / std::f64::consts::PI * x - 6.0).powi(2) + 10.0 * (1.0 - 1.0 / (8.0 * std::f64::consts::PI)) * x.cos() + 10.0
    };
    let x = lhs(2, 40);
    let y: Vec<f64> = x.iter().map(|u| f(u)).collect();
    let gp = Gp::fit(x.clone(), &y).unwrap();
    // interpolation at training points (noise-free data)
    let max_train = x.iter().zip(&y).map(|(u, v)| (gp.predict(u).0 - v).abs()).fold(0.0, f64::max);
    // generalisation on a test grid, relative to the function's range
    let mut errs = vec![];
    for i in 0..15 {
        for j in 0..15 {
            let u = [(i as f64 + 0.37) / 15.0, (j as f64 + 0.61) / 15.0];
            errs.push((gp.predict(&u).0 - f(&u)).abs());
        }
    }
    let range = y.iter().cloned().fold(f64::NEG_INFINITY, f64::max) - y.iter().cloned().fold(f64::INFINITY, f64::min);
    let rmse = (errs.iter().map(|e| e * e).sum::<f64>() / errs.len() as f64).sqrt();
    println!("train max err {max_train:.2e}, test RMSE {:.3}% of range, LOO Q² {:.4}, ℓ {:?}", 100.0 * rmse / range, gp.q2, gp.ls);
    assert!(max_train < 1e-2 * range);
    assert!(rmse < 0.03 * range, "test RMSE {rmse}");
    assert!(gp.q2 > 0.95);
}

#[test]
fn irrelevant_input_gets_long_length_scale_and_noise_is_learned() {
    // y depends on x0 only, plus a little noise
    let x = lhs(3, 30);
    let mut s = 12345u64;
    let mut rnd = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };
    let y: Vec<f64> = x.iter().map(|u| (3.0 * u[0]).sin() + 0.01 * rnd()).collect();
    let gp = Gp::fit(x, &y).unwrap();
    let rel = gp.relevance();
    println!("relevance {rel:?}, σn² {:.2e}", gp.sn2);
    assert!(rel[0] > 0.8, "x0 must dominate");
    let (m, sd) = gp.predict(&[0.5, 0.2, 0.9]);
    assert!((m - 1.5f64.sin()).abs() < 0.02 && sd < 0.05);
}

#[test]
fn expected_improvement_and_normal_cdf() {
    assert!((big_phi(0.0) - 0.5).abs() < 1e-7 && (big_phi(1.96) - 0.975).abs() < 1e-4);
    // σ → 0: EI = improvement; mean equal to best: EI = σ φ(0)
    assert!((expected_improvement(1.0, 0.0, 3.0) - 2.0).abs() < 1e-12);
    assert!((expected_improvement(3.0, 1.0, 3.0) - 0.398_942).abs() < 1e-5);
}

mod end_to_end {
    use serde_json::{Value, json};
    use simcraft::session::{Shared, new_core};

    fn call(c: &Shared, t: &str, a: Value) -> Value {
        let o = c.call(t, &a).unwrap_or_else(|e| panic!("{t}: {e:#}"));
        serde_json::from_str(&o.text).unwrap_or(Value::String(o.text))
    }
    fn cantilever() -> Shared {
        let c = new_core();
        call(&c, "add_box", json!({"name": "Beam", "size": [100, 10, 8]}));
        call(&c, "set_mesh", json!({"element_size_mm": 1}));
        call(&c, "add_fixed_support", json!({"faces": ["Beam.xmin"]}));
        call(&c, "add_force", json!({"faces": ["Beam.xmax"], "vector": [0, 0, -100]}));
        c
    }

    /// What-if from 6 designs: the surrogate must predict an unseen thickness like a real solve.
    #[test]
    fn what_if_matches_a_real_solve() {
        let c = cantilever();
        call(&c, "run_parameter_study", json!({"parameters": [{"target": "Beam.size.z", "min": 6, "max": 12}], "responses": ["min:deformation_z", "max:von_mises"], "samples": 6}));
        let q = call(&c, "fit_surrogate", json!({}));
        println!("{}", q["surrogates"]);
        let q2 = q["surrogates"][0]["q2"].as_f64().unwrap();
        let p = call(&c, "predict_surrogate", json!({"values": {"Beam.size.z": 8.7}}));
        let est = p["predictions"][0]["estimate"].as_f64().unwrap();
        let band = p["predictions"][0]["plus_minus_2sigma"].as_f64().unwrap();
        assert!(p["label"].as_str().unwrap().contains("not verified"));
        call(&c, "apply_design", json!({"values": {"Beam.size.z": 8.7}}));
        let real = call(&c, "solve", json!({}))["fields"]["deformation_z"]["min"].as_f64().unwrap();
        let err = (est - real).abs() / real.abs();
        println!("t = 8.7 mm: surrogate {est:.5} ± {band:.5} mm, real solve {real:.5} mm ({:.2}%), Q² {q2:.4}", err * 100.0);
        assert!(q2 > 0.95);
        assert!(err < 0.01, "surrogate error {:.2}%", err * 100.0);
        assert!((est - real).abs() <= band.max(1e-6) * 1.5, "real value outside the predicted band");
        // extrapolation is flagged
        let x = call(&c, "predict_surrogate", json!({"values": {"Beam.size.z": 20}}));
        assert!(x["warning"].as_str().is_some_and(|w| w.contains("outside")));
    }

    /// Bayesian optimisation: lightest beam with σ ≤ 150 MPa in at most 10 real solves.
    #[test]
    fn bayesian_optimisation_finds_constraint_boundary() {
        let c = cantilever();
        let r = call(&c, "surrogate_optimize", json!({
            "parameters": [{"target": "Beam.size.z", "min": 4, "max": 20}],
            "objective": {"response": "mass", "goal": "minimize"},
            "constraints": [{"response": "max:von_mises", "op": "<=", "value": 150}],
            "max_solves": 10
        }));
        println!("{}", r["message"]);
        let runs = r["runs"].as_array().unwrap();
        let best = &runs[r["best_run"].as_u64().unwrap() as usize - 1];
        let t = best["Beam.size.z"].as_f64().unwrap();
        let vm = best["max:von_mises"].as_f64().unwrap();
        println!("{} real solves; best t = {t:.3} mm with σ = {vm:.1} MPa (beam theory t ≥ 6.32 mm)", runs.len());
        assert!(runs.len() <= 10);
        assert!(best["feasible"].as_bool().unwrap() && vm <= 150.0);
        assert!(vm > 140.0, "optimum should sit on the stress limit (σ = {vm})");
        assert!(t > 6.0 && t < 8.0);
    }
}
