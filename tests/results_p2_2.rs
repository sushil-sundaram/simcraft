//! Closed-form validation and robustness tests for P2.2 Results you can trust:
//! - Path plots (`plot_path` + CSV export + hole gaps + invalid inputs)
//! - Cylindrical stresses (`hoop_stress`, `radial_stress`, `axial_stress` + axis/origin)
//! - Mesh-convergence assistant (`mesh_convergence` Richardson extrapolation + cancellation + apply_design)

use serde_json::{Value, json};
use simcraft::{fem, geometry::*, material::library, model::Model, session::{Shared, new_core}, study};
use std::f64::consts::FRAC_1_SQRT_2;
use std::sync::atomic::AtomicBool;

fn call(c: &Shared, t: &str, a: Value) -> Value {
    let o = c.call(t, &a).unwrap_or_else(|e| panic!("{t}: {e:#}"));
    serde_json::from_str(&o.text).unwrap_or(Value::String(o.text))
}

fn lame_model(c: &Shared, axis: &str, hh: f64) {
    let (cut_a, cut_b, sym_faces) = match axis {
        "z" => (
            ([-30, -30, 0], [30, 60, 4]),
            ([-30, -30, 0], [60, 30, 4]),
            ["CutA.xmax", "CutB.ymax", "Pipe.bottom", "Pipe.top"],
        ),
        "x" => (
            ([0, -30, -30], [4, 30, 60]),
            ([0, -30, -30], [4, 60, 30]),
            ["CutA.ymax", "CutB.zmax", "Pipe.bottom", "Pipe.top"],
        ),
        _ => unreachable!(),
    };
    call(c, "add_cylinder", json!({"name": "Pipe", "base": [0, 0, 0], "axis": axis, "radius": 20, "height": 4}));
    call(c, "add_cylinder", json!({"name": "Bore", "base": [0, 0, 0], "axis": axis, "radius": 10, "height": 4, "operation": "subtract"}));
    call(c, "add_box", json!({"name": "CutA", "min": cut_a.0, "size": cut_a.1, "operation": "subtract"}));
    call(c, "add_box", json!({"name": "CutB", "min": cut_b.0, "size": cut_b.1, "operation": "subtract"}));
    call(c, "set_mesh", json!({"element_size_mm": hh}));
    call(c, "add_symmetry", json!({"faces": sym_faces}));
    call(c, "add_pressure", json!({"faces": ["Bore.side"], "value_mpa": 10}));
    call(c, "solve", json!({}));
}

fn lame_exact(ri: f64, ro: f64, p: f64, nu: f64, r: f64) -> (f64, f64, f64) {
    let f = p * ri * ri / (ro * ro - ri * ri);
    let ratio2 = (ro * ro) / (r * r);
    let sigma_t = f * (1.0 + ratio2);
    let sigma_r = f * (1.0 - ratio2);
    let sigma_z = nu * (sigma_t + sigma_r);
    (sigma_r, sigma_t, sigma_z)
}

#[test]
fn lame_path_plots_and_cylindrical_stresses_match_closed_form() {
    let tmp = tempfile::tempdir().unwrap();
    let ws = tmp.path().join("ws");
    let outside = tmp.path().join("outside");
    std::fs::create_dir_all(&ws).unwrap();
    std::fs::create_dir_all(&outside).unwrap();

    let c = new_core();
    c.set_workspace(Some(ws.clone()));
    lame_model(&c, "z", 1.0);

    let nu = library().into_iter().find(|m| m.name == "Structural Steel").unwrap().poisson_ratio;
    let peak_ref = 10.0 * 100.0 / 300.0 * (1.0 + 4.0); // 16.6667 MPa

    // 1. Axis-aligned and 45-degree diagonal paths across the wall
    for (name, from, to) in [
        ("axis-aligned", [10.6, 0.5, 2.0], [19.4, 0.5, 2.0]),
        ("diagonal-45deg", [10.6 * FRAC_1_SQRT_2, 10.6 * FRAC_1_SQRT_2, 2.0], [19.4 * FRAC_1_SQRT_2, 19.4 * FRAC_1_SQRT_2, 2.0]),
    ] {
        let hoop = call(
            &c,
            "plot_path",
            json!({
                "field": "hoop_stress",
                "from": from,
                "to": to,
                "samples": 19,
                "cylindrical_origin": [0, 0, 0],
                "cylindrical_axis": "z",
            }),
        );
        let radial = call(
            &c,
            "plot_path",
            json!({
                "field": "radial_stress",
                "from": from,
                "to": to,
                "samples": 19,
                "cylindrical_origin": [0, 0, 0],
                "cylindrical_axis": "z",
            }),
        );
        assert_eq!(hoop["valid_samples"].as_u64(), Some(19), "{name} hoop missed points: {hoop}");
        assert_eq!(radial["valid_samples"].as_u64(), Some(19), "{name} radial missed points: {radial}");

        let hp = hoop["points"].as_array().unwrap();
        let rp = radial["points"].as_array().unwrap();
        let mut max_err_t = 0.0f64;
        let mut max_err_r_interior = 0.0f64;
        for (hpt, rpt) in hp.iter().zip(rp.iter()) {
            let p = hpt["point_mm"].as_array().unwrap();
            let r = p[0].as_f64().unwrap().hypot(p[1].as_f64().unwrap());
            let (er, et, _) = lame_exact(10.0, 20.0, 10.0, nu, r);
            let gt = hpt["value"].as_f64().unwrap();
            let gr = rpt["value"].as_f64().unwrap();
            max_err_t = max_err_t.max((gt - et).abs());
            if r >= 11.0 {
                max_err_r_interior = max_err_r_interior.max((gr - er).abs());
            }
        }
        assert!(
            max_err_t < 0.015 * peak_ref,
            "{name} max hoop stress error {max_err_t:.3} MPa exceeds 1.5% of {peak_ref:.2}"
        );
        assert!(
            max_err_r_interior < 0.025 * peak_ref,
            "{name} max interior radial stress error {max_err_r_interior:.3} MPa exceeds 2.5% of {peak_ref:.2}"
        );
    }

    // 2. Cylindrical display + probe at 45 degrees (including axial stress at the snapped node)
    call(&c, "set_display", json!({"field": "hoop_stress", "cylindrical_origin": [0, 0, 0], "cylindrical_axis": "z"}));
    let pr = call(&c, "probe", json!({"point": [12.0 * FRAC_1_SQRT_2, 12.0 * FRAC_1_SQRT_2, 2.0]}));
    let np = pr["nearest_node_mm"].as_array().unwrap();
    let r_node = np[0].as_f64().unwrap().hypot(np[1].as_f64().unwrap());
    let (er, et, ez) = lame_exact(10.0, 20.0, 10.0, nu, r_node);
    let gt = pr["values"]["hoop_stress"].as_f64().unwrap();
    let gr = pr["values"]["radial_stress"].as_f64().unwrap();
    let gz = pr["values"]["axial_stress"].as_f64().unwrap();
    assert!((gt - et).abs() < 0.25, "probe hoop {gt} vs exact {et} at r={r_node:.3}");
    assert!((gr - er).abs() < 0.25, "probe radial {gr} vs exact {er} at r={r_node:.3}");
    assert!((gz - ez).abs() < 0.20, "probe axial {gz} vs exact {ez} at r={r_node:.3}");

    // 3. CSV export honors workspace jail and contains valid rows
    let csv_ok = ws.join("lame_hoop.csv");
    let exported = call(
        &c,
        "plot_path",
        json!({
            "field": "hoop_stress",
            "from": [11.0, 1.0, 2.0],
            "to": [18.0, 1.0, 2.0],
            "samples": 8,
            "csv_path": csv_ok.to_string_lossy(),
        }),
    );
    assert!(csv_ok.exists());
    let csv_body = std::fs::read_to_string(&csv_ok).unwrap();
    assert_eq!(csv_body, exported["csv"].as_str().unwrap());
    assert_eq!(csv_body.lines().count(), 9); // header + 8 points

    // Workspace jail & extension checks for csv_path
    assert!(c.call("plot_path", &json!({
        "field": "hoop_stress", "from": [11, 1, 2], "to": [18, 1, 2],
        "csv_path": outside.join("leak.csv").to_string_lossy()
    })).is_err());
    assert!(c.call("plot_path", &json!({
        "field": "hoop_stress", "from": [11, 1, 2], "to": [18, 1, 2],
        "csv_path": ws.join("bad.txt").to_string_lossy()
    })).is_err());
}

#[test]
fn cylindrical_axis_respects_custom_x_axis_and_rejects_zero() {
    let c = new_core();
    lame_model(&c, "x", 1.0);
    let nu = library().into_iter().find(|m| m.name == "Structural Steel").unwrap().poisson_ratio;
    let pr = call(
        &c,
        "probe",
        json!({
            "point": [2.0, 13.0 * FRAC_1_SQRT_2, 13.0 * FRAC_1_SQRT_2],
            "cylindrical_origin": [0, 0, 0],
            "cylindrical_axis": [1, 0, 0],
        }),
    );
    let np = pr["nearest_node_mm"].as_array().unwrap();
    let r_node = np[1].as_f64().unwrap().hypot(np[2].as_f64().unwrap());
    let (er, et, _ez) = lame_exact(10.0, 20.0, 10.0, nu, r_node);
    let gt = pr["values"]["hoop_stress"].as_f64().unwrap();
    let gr = pr["values"]["radial_stress"].as_f64().unwrap();
    assert!((gt - et).abs() < 0.25, "x-axis pipe probe hoop {gt} vs exact {et} at r={r_node:.3}");
    assert!((gr - er).abs() < 0.25, "x-axis pipe probe radial {gr} vs exact {er} at r={r_node:.3}");

    // Zero vector axis rejected
    assert!(c.call("set_display", &json!({"cylindrical_axis": [0, 0, 0]})).is_err());
}

#[test]
fn path_plot_handles_hole_gaps_and_invalid_inputs() {
    let c = new_core();
    call(&c, "add_box", json!({"name": "Plate", "min": [0, 0, 0], "size": [40, 20, 4], "material": "Structural Steel"}));
    call(&c, "add_cylinder", json!({"name": "Hole", "base": [20, 10, 0], "axis": "z", "radius": 4, "height": 4, "operation": "subtract"}));
    call(&c, "set_mesh", json!({"element_size_mm": 2.0}));
    call(&c, "add_fixed_support", json!({"faces": ["Plate.xmin"]}));
    call(&c, "add_force", json!({"faces": ["Plate.xmax"], "vector": [2000, 0, 0]}));
    call(&c, "solve", json!({}));

    let pp = call(
        &c,
        "plot_path",
        json!({
            "field": "von_mises",
            "from": [4.0, 10.0, 2.0],
            "to": [36.0, 10.0, 2.0],
            "samples": 33,
        }),
    );
    let pts = pp["points"].as_array().unwrap();
    assert_eq!(pts.len(), 33);
    let valid = pp["valid_samples"].as_u64().unwrap();
    assert!(valid > 15 && valid < 30, "expected hole gap in valid_samples={valid}");
    // Midpoint (20, 10, 2) is dead center in the r=4 hole -> must be None
    assert!(pts[16]["value"].is_null(), "point inside hole should not be interpolated: {:?}", pts[16]);
    // Both sides outside hole have real values
    assert!(pts[2]["value"].as_f64().is_some());
    assert!(pts[30]["value"].as_f64().is_some());

    // Bad inputs rejected
    assert!(c.call("plot_path", &json!({"field": "von_mises", "from": [5, 5, 2], "to": [5, 5, 2]})).is_err());
    assert!(c.call("plot_path", &json!({"field": "nonexistent_field", "from": [4, 10, 2], "to": [36, 10, 2]})).is_err());
}

#[test]
fn mesh_convergence_cantilever_extrapolates_within_half_percent_and_cancels() {
    let steel = library().into_iter().find(|m| m.name == "Structural Steel").unwrap();
    let (e, nu) = (steel.youngs_modulus_gpa * 1e3, steel.poisson_ratio);
    let (l, b, h, f) = (240.0f64, 12.0f64, 12.0f64, 200.0f64);
    let want = f * l.powi(3) / (3.0 * e * b * h.powi(3) / 12.0) + f * l / (5.0 / 6.0 * e / (2.0 * (1.0 + nu)) * b * h);

    let c = new_core();
    call(&c, "add_box", json!({"name": "Beam", "size": [240, 12, 12], "material": "Structural Steel"}));
    call(&c, "set_mesh", json!({"element_size_mm": 4.0}));
    call(&c, "add_fixed_support", json!({"faces": ["Beam.xmin"]}));
    call(&c, "add_force", json!({"faces": ["Beam.xmax"], "vector": [0, 0, -200]}));

    let rep = call(
        &c,
        "mesh_convergence",
        json!({
            "responses": ["min:deformation_z", "max:von_mises"],
            "base_element_size_mm": 4.0,
            "target_error_pct": 0.5,
        }),
    );
    assert_eq!(rep["kind"].as_str(), Some("convergence"));
    let steps = rep["convergence_steps"].as_array().unwrap();
    assert_eq!(steps.len(), 3);
    assert!(steps[2]["dofs"].as_u64().unwrap() > steps[0]["dofs"].as_u64().unwrap());

    let conv = rep["convergence"].as_array().unwrap();
    let defl = &conv[0];
    assert_eq!(defl["response"].as_str(), Some("min:deformation_z"));
    let ext = -defl["extrapolated"].as_f64().unwrap();
    let err_vs_theory = (ext - want).abs() / want;
    assert!(
        err_vs_theory < 0.005,
        "Richardson-extrapolated deflection {ext:.5} mm differs by {:.3}% from Timoshenko {want:.5} mm",
        100.0 * err_vs_theory
    );
    assert!(defl["monotonic"].as_bool().unwrap());
    let rec = rep["recommended_size_mm"].as_f64().unwrap();
    assert!(rec.is_finite() && rec > 0.0);

    // apply_design updates element size cleanly and undo reverts it
    call(&c, "apply_design", json!({"index": 0}));
    let model = call(&c, "get_model", json!({}));
    let applied_h = model["mesh"]["element_size_mm"].as_f64().unwrap();
    assert!((applied_h - rec).abs() < 1e-3, "applied {applied_h} != recommended {rec}");
    call(&c, "undo", json!({}));
    let undone = call(&c, "get_model", json!({}));
    assert!((undone["mesh"]["element_size_mm"].as_f64().unwrap() - 4.0).abs() < 1e-6);

    // Invalid parameters fail loudly
    assert!(c.call("mesh_convergence", &json!({"base_element_size_mm": -1.0})).is_err());
    assert!(c.call("mesh_convergence", &json!({"target_error_pct": 0.0})).is_err());

    // Cooperative cancellation stops convergence cleanly before solving
    let mut m = Model::default();
    m.bodies.push(Body { name: "Beam".into(), shape: Shape::Box { min: [0.0; 3], size: [120.0, 12.0, 10.0] }, op: Op::Add, material: "Structural Steel".into(), role: Role::Solid });
    let cancel = AtomicBool::new(true);
    let err = study::mesh_convergence(&m, vec!["mass".into()], Some(4.0), 1.0, fem::Progress::cancel_only(&cancel)).unwrap_err();
    assert!(err.is::<fem::Cancelled>(), "unexpected error: {err:#}");
}
