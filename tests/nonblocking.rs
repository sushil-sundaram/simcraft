//! Validation tests for P2.A1: cooperative cancellation in parallel loops, non-blocking meshing
//! and session locks, STL BVH loading/querying, and command-based undo/dirty tracking.

use serde_json::{Value, json};
use simcraft::{fem, geometry::*, material, mesh, model::*, radiation, session::{Shared, new_core}};
use std::io::Write;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

fn call(c: &Shared, t: &str, a: Value) -> Value {
    let o = c.call(t, &a).unwrap_or_else(|e| panic!("{t}: {e:#}"));
    serde_json::from_str(&o.text).unwrap_or(Value::String(o.text))
}

#[test]
fn mesh_build_cancels_cleanly_on_large_grid() {
    let mut m = Model::default();
    m.bodies.push(Body {
        name: "Block".into(),
        shape: Shape::Box { min: [0.0; 3], size: [120.0, 120.0, 40.0] },
        op: Op::Add,
        material: "Structural Steel".into(),
        role: Role::Solid,
    });
    // 576,000 cells (>500k) with boundary fitting enabled
    let cancel = Arc::new(AtomicBool::new(true));
    let t0 = Instant::now();
    let err = mesh::build_for_cancellable(&m, 1.0, 2_000_000, &cancel).expect_err("pre-cancelled mesh must abort");
    assert!(err.is::<fem::Cancelled>(), "unexpected error: {err:#}");
    assert!(t0.elapsed().as_secs_f64() < 1.0);

    // Concurrent cancellation during active >500k meshing on a background thread.
    let cancel2 = Arc::new(AtomicBool::new(false));
    let c2 = cancel2.clone();
    let m2 = m.clone();
    let h = std::thread::spawn(move || mesh::build_for_cancellable(&m2, 1.0, 2_000_000, &c2));
    std::thread::sleep(std::time::Duration::from_millis(15));
    cancel2.store(true, Ordering::Relaxed);
    match h.join().unwrap() {
        Ok(_) => {} // completed faster than 15 ms on fast CPU; still valid
        Err(e) => assert!(e.is::<fem::Cancelled>(), "unexpected error: {e:#}"),
    }
    // Normal coarse build succeeds cleanly afterwards.
    let ok = mesh::build_for(&m, 8.0, 100_000).unwrap();
    assert!(!ok.elems.is_empty());
}

#[test]
fn rayon_solvers_cancel_without_poison() {
    let mut m = Model::default();
    m.analysis.kind = AnalysisType::SteadyThermal;
    m.bodies.push(Body {
        name: "Plate".into(),
        shape: Shape::Box { min: [0.0; 3], size: [40.0, 40.0, 4.0] },
        op: Op::Add,
        material: "Aluminum 6061-T6".into(),
        role: Role::Solid,
    });
    m.loads.push(LoadItem { id: 1, load: Load::Temperature { faces: vec!["Plate.zmin".into()], value: 80.0 } });
    m.loads.push(LoadItem { id: 2, load: Load::Radiation { faces: vec!["Plate.zmax".into()], emissivity: 0.9, ambient: 25.0, enclosure: true } });
    let me = mesh::build_for(&m, 2.0, 100_000).unwrap();

    // Cancel inside Radiation::new's rayon::par_iter loop without unwinding/panicking.
    let cancel = AtomicBool::new(true);
    let err = radiation::Radiation::new(&m, &me, fem::Progress::cancel_only(&cancel)).err().expect("cancelled radiation must return Err");
    assert!(err.is::<fem::Cancelled>(), "unexpected error: {err:#}");

    // Cancel inside fem::assemble_structural's rayon::par_iter loop.
    let err = fem::assemble_structural(&m, &me, fem::Progress::cancel_only(&cancel)).err().expect("cancelled assembly must return Err");
    assert!(err.is::<fem::Cancelled>(), "unexpected error: {err:#}");

    // End-to-end Core solve cancellation leaves session unlocked while running and unpoisoned after.
    let c = new_core();
    call(&c, "add_box", json!({"name": "Plate", "size": [100, 10, 10]}));
    call(&c, "set_mesh", json!({"element_size_mm": 2.5}));
    call(&c, "set_analysis", json!({"type": "modal", "modes": 6}));
    call(&c, "add_fixed_support", json!({"faces": ["Plate.xmin"]}));
    let c_bg = c.clone();
    let handle = std::thread::spawn(move || c_bg.call("solve", &json!({})));
    std::thread::sleep(std::time::Duration::from_millis(40));
    // Status query must not block behind session mutex while solve/mesh runs.
    let st = call(&c, "get_status", json!({}));
    assert!(st.is_object());
    let _ = c.call("cancel", &json!({}));
    let res = handle.join().unwrap();
    if let Err(e) = res {
        assert!(format!("{e:#}").contains("cancelled"));
    }
    assert!(!c.is_busy());
    // Session remains usable for subsequent edits & solves.
    assert!(c.call("add_gravity", &json!({})).is_ok());
}

#[test]
fn command_undo_redo_and_dirty_tracking() {
    let dir = std::env::temp_dir().join(format!("simcraft-p2a1-undo-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let proj = dir.join("model.simcraft").to_string_lossy().to_string();
    let c = new_core();
    let dirty = |c: &Shared| call(c, "get_status", json!({}))["unsaved_changes"].as_bool().unwrap();

    assert!(!dirty(&c));
    call(&c, "add_box", json!({"name": "Part", "size": [20, 20, 10]}));
    assert!(dirty(&c));
    call(&c, "save_project", json!({"path": proj}));
    assert!(!dirty(&c));

    // Edit makes model dirty; undoing back to saved state clears dirty without JSON diffing.
    call(&c, "edit_body", json!({"name": "Part", "size": [40, 20, 10]}));
    assert!(dirty(&c));
    call(&c, "undo", json!({}));
    assert!(!dirty(&c), "undoing back to saved revision must mark model clean");
    call(&c, "redo", json!({}));
    assert!(dirty(&c));

    // Non-mutating tools must not create undo entries or flip dirty state.
    call(&c, "undo", json!({}));
    assert!(!dirty(&c));
    call(&c, "set_display", json!({"dark_mode": true}));
    call(&c, "clear_loads", json!({}));
    assert!(!dirty(&c));

    // new_model on modified model is undoable.
    call(&c, "edit_body", json!({"name": "Part", "size": [50, 20, 10]}));
    call(&c, "new_model", json!({}));
    assert!(!dirty(&c));
    assert_eq!(call(&c, "get_model", json!({}))["bodies"].as_array().unwrap().len(), 0);
    call(&c, "undo", json!({}));
    assert!(dirty(&c));
    assert_eq!(call(&c, "get_model", json!({}))["bodies"][0]["shape"]["size"][0], 50.0);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn binary_stl_with_bvh_meshes_cleanly() {
    let dir = std::env::temp_dir().join(format!("simcraft-p2a1-stl-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let stl_path = dir.join("octasphere.stl");

    // Write a binary STL sphere with 512 triangles (radius 10 mm at [10, 10, 10]).
    let n = 8usize;
    let pts = |i: usize, j: usize| -> [f32; 3] {
        let u = std::f64::consts::PI * i as f64 / n as f64;
        let v = std::f64::consts::TAU * j as f64 / (2 * n) as f64;
        [(10.0 + 10.0 * u.sin() * v.cos()) as f32, (10.0 + 10.0 * u.sin() * v.sin()) as f32, (10.0 + 10.0 * u.cos()) as f32]
    };
    let mut tris = Vec::new();
    for i in 0..n {
        for j in 0..2 * n {
            let (a, b, c, d) = (pts(i, j), pts(i + 1, j), pts(i + 1, j + 1), pts(i, j + 1));
            tris.push([a, b, c]);
            tris.push([a, c, d]);
        }
    }
    let mut f = std::fs::File::create(&stl_path).unwrap();
    f.write_all(&[0u8; 80]).unwrap();
    f.write_all(&(tris.len() as u32).to_le_bytes()).unwrap();
    for tri in &tris {
        f.write_all(&[0u8; 12]).unwrap();
        for v in tri {
            for c in v {
                f.write_all(&c.to_le_bytes()).unwrap();
            }
        }
        f.write_all(&[0u8; 2]).unwrap();
    }
    drop(f);

    let data = load_stl(&stl_path.to_string_lossy(), 1.0, [0.0; 3]).unwrap();
    let xs = data.x_crossings(10.1, 10.2);
    assert_eq!(xs.len(), 2, "ray through sphere center must hit twice: {xs:?}");
    let body = Body {
        name: "Sphere".into(),
        shape: Shape::Stl { path: stl_path.to_string_lossy().into(), scale: 1.0, offset: [0.0; 3], data: Some(Arc::new(data)) },
        op: Op::Add,
        material: "Structural Steel".into(),
        role: Role::Solid,
    };
    let me = mesh::build_mesh(&[body], 2.0, &material::library(), false).unwrap();
    assert!(me.elems.len() > 100);
    let _ = std::fs::remove_dir_all(&dir);
}
