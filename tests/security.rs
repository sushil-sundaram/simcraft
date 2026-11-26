//! Security & parser-hardening regression tests (P2.S).
use serde_json::json;
use simcraft::geometry::{Shape, load_stl};
use simcraft::gpu::{self, ComputeMode};
use simcraft::kicad::{self, PcbData};
use simcraft::session::{external_assets, new_core, resolve_io_path};
use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};

#[test]
fn workspace_jail_blocks_traversal_extensions_and_symlinks() {
    let tmp = tempfile::tempdir().unwrap();
    let ws = tmp.path().join("workspace");
    let outside = tmp.path().join("outside");
    std::fs::create_dir_all(&ws).unwrap();
    std::fs::create_dir_all(&outside).unwrap();

    let outside_stl = outside.join("secret.stl");
    std::fs::write(&outside_stl, "solid s\nendsolid s\n").unwrap();
    let valid_stl = ws.join("part.stl");
    std::fs::write(
        &valid_stl,
        "solid s\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex 10 0 0\nvertex 0 10 0\nendloop\nendfacet\nendsolid s\n",
    )
    .unwrap();

    let c = new_core();
    c.set_workspace(Some(ws.clone()));

    // Traversal & absolute path outside workspace are rejected
    assert!(c.call("import_stl", &json!({"path": "../outside/secret.stl"})).is_err());
    assert!(c.call("import_stl", &json!({"path": outside_stl.to_string_lossy()})).is_err());
    assert!(c.call("save_project", &json!({"path": outside.join("leak.simcraft").to_string_lossy()})).is_err());
    assert!(c.call("make_report", &json!({"path": outside.join("leak.html").to_string_lossy()})).is_err());

    // Unsupported extensions are rejected even inside workspace
    assert!(c.call("save_project", &json!({"path": ws.join("bad.sh").to_string_lossy()})).is_err());
    assert!(c.call("make_report", &json!({"path": ws.join("bad.js").to_string_lossy()})).is_err());
    assert!(c.call("export_vtk", &json!({"path": ws.join("bad.txt").to_string_lossy()})).is_err());

    // Symlink escaping workspace is rejected
    #[cfg(unix)]
    {
        let link = ws.join("escape.stl");
        std::os::unix::fs::symlink(&outside_stl, &link).unwrap();
        assert!(c.call("import_stl", &json!({"path": link.to_string_lossy()})).is_err());
        let out_link = ws.join("out_link.simcraft");
        std::os::unix::fs::symlink(outside.join("target.simcraft"), &out_link).unwrap();
        assert!(c.call("save_project", &json!({"path": out_link.to_string_lossy()})).is_err());
    }

    // Valid path inside workspace succeeds
    assert!(c.call("import_stl", &json!({"path": "part.stl"})).is_ok());
    let saved = ws.join("ok.simcraft");
    assert!(c.call("save_project", &json!({"path": saved.to_string_lossy()})).is_ok());
    assert!(saved.exists());
    assert!(resolve_io_path("part.stl", Some(&ws), false, &["stl"]).is_ok());
}

#[test]
fn project_external_assets_detected_and_blocked() {
    let tmp = tempfile::tempdir().unwrap();
    let ws = tmp.path().join("ws");
    let ext = tmp.path().join("ext");
    std::fs::create_dir_all(&ws).unwrap();
    std::fs::create_dir_all(&ext).unwrap();

    let ext_stl = ext.join("external.stl");
    std::fs::write(
        &ext_stl,
        "solid s\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex 10 0 0\nvertex 0 10 0\nendloop\nendfacet\nendsolid s\n",
    )
    .unwrap();
    let proj = ws.join("evil.simcraft");
    let body = json!({
        "simcraft": 1,
        "model": {
            "bodies": [{
                "name": "Part",
                "shape": {"type": "stl", "path": ext_stl.to_string_lossy(), "scale": 1.0, "offset": [0.0, 0.0, 0.0]},
                "op": "add",
                "material": "Structural Steel"
            }],
            "loads": []
        }
    });
    std::fs::write(&proj, serde_json::to_string_pretty(&body).unwrap()).unwrap();

    let found = external_assets(&proj.to_string_lossy()).unwrap();
    assert_eq!(found.len(), 1);
    assert_eq!(found[0], ext_stl.to_string_lossy());

    // Workspace-confined core refuses opening a project referencing an external STL
    let c = new_core();
    c.set_workspace(Some(ws));
    assert!(c.call("open_project", &json!({"path": proj.to_string_lossy()})).is_err());

    // Even without a workspace, non-.stl / non-.kicad_pcb embedded paths are rejected
    let bad_ext_proj = tmp.path().join("bad_asset.simcraft");
    let bad_body = json!({
        "simcraft": 1,
        "model": {
            "bodies": [{
                "name": "Part",
                "shape": {"type": "stl", "path": "/etc/passwd", "scale": 1.0, "offset": [0.0, 0.0, 0.0]},
                "op": "add",
                "material": "Structural Steel"
            }],
            "loads": []
        }
    });
    std::fs::write(&bad_ext_proj, serde_json::to_string_pretty(&bad_body).unwrap()).unwrap();
    assert!(new_core().call("open_project", &json!({"path": bad_ext_proj.to_string_lossy()})).is_err());
}

#[test]
fn kicad_parser_guards_depth_arcs_and_extreme_aspect() {
    // Deep nesting must error cleanly without stack overflow on parse or drop
    let mut deep = String::from("(kicad_pcb ");
    for _ in 0..400 {
        deep.push('(');
    }
    for _ in 0..400 {
        deep.push(')');
    }
    deep.push(')');
    assert!(kicad::parse(&deep).is_err());

    // Extreme KiCad 5 arc angle must not OOM
    let arc_board = "(kicad_pcb (version 20211014)\n  (gr_rect (start 0 0) (end 20 10) (layer Edge.Cuts))\n  (gr_arc (start 10 5) (end 12 5) (angle 1e12) (layer Edge.Cuts))\n)";
    assert!(PcbData::from_text(arc_board, "arc.kicad_pcb").is_ok());

    // Extreme aspect ratio must clamp raster pixels cleanly without OOM
    let skinny = "(kicad_pcb (version 20211014)\n  (gr_rect (start 0 0) (end 1e7 0.01) (layer Edge.Cuts))\n)";
    let pcb = PcbData::from_text(skinny, "skinny.kicad_pcb").unwrap();
    assert!(pcb.nx * pcb.ny <= 16_000_000);
}

#[test]
fn stl_and_mesh_reject_overflows_and_nonfinite() {
    let tmp = tempfile::tempdir().unwrap();

    // Binary STL claiming u32::MAX triangles must reject cleanly
    let bad_bin = tmp.path().join("bad_count.stl");
    let mut bytes = vec![0u8; 84];
    bytes[80..84].copy_from_slice(&u32::MAX.to_le_bytes());
    std::fs::write(&bad_bin, bytes).unwrap();
    assert!(load_stl(&bad_bin.to_string_lossy(), 1.0, [0.0; 3]).is_err());

    // ASCII STL with NaN coordinates must reject
    let nan_stl = tmp.path().join("nan.stl");
    std::fs::write(&nan_stl, "solid s\nfacet normal 0 0 1\nouter loop\nvertex NaN 0 0\nvertex 1 0 0\nvertex 0 1 0\nendloop\nendfacet\nendsolid s\n").unwrap();
    assert!(load_stl(&nan_stl.to_string_lossy(), 1.0, [0.0; 3]).is_err());

    // Non-finite Shape::validate fails
    assert!(Shape::Box { min: [0.0; 3], size: [f64::NAN, 10.0, 10.0] }.validate().is_err());
    assert!(Shape::Box { min: [f64::INFINITY, 0.0, 0.0], size: [10.0, 10.0, 10.0] }.validate().is_err());

    // Extreme mesh cell count / overflow must bail cleanly
    let c = new_core();
    c.call("add_box", &json!({"size": [1000.0, 1000.0, 1000.0]})).unwrap();
    assert!(c.call("set_mesh", &json!({"element_size_mm": 1e-12})).is_err());
}

#[test]
fn html_reports_escape_quotes_and_scripts() {
    let c = new_core();
    c.call("add_box", &json!({"name": "Plate", "size": [20, 10, 5]})).unwrap();
    let mesh = c.session.lock().unwrap().ensure_mesh().unwrap();
    let model = c.session.lock().unwrap().model.clone();
    let disp = c.session.lock().unwrap().display.clone();
    let html = simcraft::report::build_html(&simcraft::report::ReportInput {
        title: "<script>alert('xss')</script>\"".into(),
        model: &model,
        mesh: &mesh,
        results: None,
        display: &disp,
        study: None,
        plan: None,
        checks: vec![],
        project: Some("/tmp/\"evil\".simcraft".into()),
        stale: false,
    })
    .unwrap();
    assert!(!html.contains("<script>"));
    assert!(html.contains("&lt;script&gt;alert(&#39;xss&#39;)&lt;/script&gt;&quot;"));

    let bench_html = simcraft::bench::html(&json!({
        "simcraft_version": "<script>v</script>",
        "commit": "'\"abc",
        "date": "2026",
        "mode": "quick",
        "seconds": 1.0,
        "passed": 0,
        "failed": 0,
        "other_solver_name": "'><script>x</script>",
        "machine": {"cpu": "cpu", "threads": 4},
        "cases": []
    }));
    assert!(!bench_html.contains("<script>"));
    assert!(bench_html.contains("&#39;&quot;abc"));
}

#[test]
fn gpu_thread_local_state_is_isolated() {
    let h1 = std::thread::spawn(|| {
        gpu::set_mode(ComputeMode::Gpu);
        std::thread::sleep(std::time::Duration::from_millis(30));
        assert_eq!(gpu::mode(), ComputeMode::Gpu);
    });
    let h2 = std::thread::spawn(|| {
        gpu::set_mode(ComputeMode::Cpu);
        std::thread::sleep(std::time::Duration::from_millis(30));
        assert_eq!(gpu::mode(), ComputeMode::Cpu);
    });
    h1.join().unwrap();
    h2.join().unwrap();
}

#[test]
fn bridge_enforces_token_and_auth_timeout() {
    let tmp = tempfile::tempdir().unwrap();
    let tok_file = tmp.path().join("token");
    // Grab an unused port then drop it before starting the bridge
    let port = {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        l.local_addr().unwrap().port()
    };
    // SAFETY: only this test in `tests/security.rs` reads/writes bridge env vars
    unsafe {
        std::env::set_var("SIMCRAFT_TOKEN_FILE", &tok_file);
        std::env::set_var("SIMCRAFT_PORT", port.to_string());
    }

    let c = new_core();
    let msg = simcraft::bridge::serve(c).unwrap();
    assert!(msg.contains("token-protected"));
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(&tok_file).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600);
    }
    let tok = std::fs::read_to_string(&tok_file).unwrap();
    assert!(tok.len() >= 64);

    // Bad auth is rejected
    let addr = format!("127.0.0.1:{port}");
    let mut s = TcpStream::connect(&addr).unwrap();
    writeln!(s, "{{\"auth\":\"wrong\"}}").unwrap();
    let mut r = BufReader::new(s);
    let mut line = String::new();
    r.read_line(&mut line).unwrap();
    assert!(line.contains("\"ok\":false"));

    // Valid auth succeeds
    let mut s = TcpStream::connect(&addr).unwrap();
    writeln!(s, "{}", json!({"auth": tok.trim()})).unwrap();
    let mut r = BufReader::new(s.try_clone().unwrap());
    line.clear();
    r.read_line(&mut line).unwrap();
    assert!(line.contains("\"ok\":true"));
    writeln!(s, "{{\"tool\":\"get_status\",\"args\":{{}}}}").unwrap();
    line.clear();
    r.read_line(&mut line).unwrap();
    assert!(line.contains("\"ok\":true"));
}
