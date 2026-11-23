//! A crashing or cancelled job must leave the app usable.
use serde_json::{Value, json};
use simcraft::session::{Shared, new_core};

fn call(c: &Shared, t: &str, a: Value) -> Value {
    let o = c.call(t, &a).unwrap_or_else(|e| panic!("{t}: {e:#}"));
    serde_json::from_str(&o.text).unwrap_or(Value::String(o.text))
}
fn beam(c: &Shared) {
    call(c, "add_box", json!({"name": "Beam", "size": [100, 10, 10]}));
    call(c, "set_mesh", json!({"element_size_mm": 2.5}));
    call(c, "add_fixed_support", json!({"faces": ["Beam.xmin"]}));
    call(c, "add_force", json!({"faces": ["Beam.xmax"], "vector": [0, 0, -100]}));
}

#[test]
fn solver_panic_becomes_an_error_and_the_app_recovers() {
    // SAFETY: tests in this binary run in one process; the hook only enables the test tool
    unsafe { std::env::set_var("SIMCRAFT_TEST_HOOKS", "1") };
    let c = new_core();
    beam(&c);
    let Err(e) = c.call("__panic", &json!({})) else { panic!("expected an error") };
    let msg = format!("{e:#}");
    println!("{msg}");
    assert!(msg.contains("internal error"));
    assert!(!c.is_busy(), "busy flag must be cleared");
    // the app still solves
    let r = call(&c, "solve", json!({}));
    assert!(r["fields"]["total_deformation"]["max"].as_f64().unwrap() > 0.0);
}

#[test]
fn cancel_stops_a_long_solve() {
    let c = new_core();
    call(&c, "add_enclosure", json!({"min": [0, 0, 0], "size": [200, 2, 100]}));
    call(&c, "set_analysis", json!({"type": "fluid", "max_iterations": 200000}));
    call(&c, "set_mesh", json!({"element_size_mm": 1}));
    call(&c, "add_moving_wall", json!({"faces": ["Enclosure.zmax"], "velocity": [0.01, 0, 0]}));
    call(&c, "add_symmetry", json!({"faces": ["Enclosure.ymin", "Enclosure.ymax"]}));
    let c2 = c.clone();
    let t0 = std::time::Instant::now();
    let h = std::thread::spawn(move || c2.call("solve", &json!({})));
    std::thread::sleep(std::time::Duration::from_millis(800));
    assert!(c.is_busy());
    assert!(call(&c, "cancel", json!({})).as_str().unwrap().contains("Stopping"));
    let r = h.join().unwrap();
    let dt = t0.elapsed().as_secs_f64();
    let Err(err) = r else { panic!("expected the cancelled solve to fail") };
    let msg = format!("{err:#}");
    println!("{msg} after {dt:.2} s");
    assert!(msg.contains("cancelled"));
    assert!(dt < 5.0, "cancel took {dt:.1} s");
    assert!(!c.is_busy());
    assert!(c.session.lock().unwrap().results.is_none(), "a cancelled solve leaves no results");
    // and a new job runs normally
    call(&c, "set_analysis", json!({"max_iterations": 60}));
    assert!(c.call("solve", &json!({})).is_ok());
}

#[test]
fn unsaved_changes_recent_files_and_autosave() {
    let dir = std::env::temp_dir().join(format!("simcraft-cfg-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    // SAFETY: only this test in the binary reads the config location
    unsafe { std::env::set_var("SIMCRAFT_TOKEN_FILE", dir.join("token")) };
    let c = new_core();
    let dirty = |c: &Shared| call(c, "get_status", json!({}))["unsaved_changes"].as_bool().unwrap();
    assert!(!dirty(&c), "a fresh model is clean");
    beam(&c);
    assert!(dirty(&c));
    let proj = dir.join("beam.simcraft").to_string_lossy().to_string();
    call(&c, "save_project", json!({"path": proj}));
    assert!(!dirty(&c));
    assert_eq!(simcraft::session::recent_files().first().map(|p| p.ends_with("beam.simcraft")), Some(true));
    call(&c, "add_gravity", json!({}));
    assert!(dirty(&c));
    // autosave + recovery: the recovered model has the change and counts as unsaved
    c.session.lock().unwrap().autosave().unwrap();
    let c2 = new_core();
    call(&c2, "open_project", json!({"path": simcraft::session::autosave_path().to_string_lossy()}));
    assert_eq!(call(&c2, "get_model", json!({}))["loads"].as_array().unwrap().len(), 3);
    assert!(dirty(&c2), "recovered work is unsaved");
    // reopening the saved project gives the saved state, clean
    call(&c2, "open_project", json!({"path": proj}));
    assert!(!dirty(&c2));
    assert_eq!(call(&c2, "get_model", json!({}))["loads"].as_array().unwrap().len(), 2);
    call(&c2, "new_model", json!({}));
    assert!(!dirty(&c2));
    let _ = std::fs::remove_dir_all(&dir);
}
