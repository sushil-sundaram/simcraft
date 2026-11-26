//! Tests for P2.A2 hardware GPU viewport scene preparation, camera ray/projection math,
//! BVH face picking parity with software rasterizer, enclosure partitioning, deformed section cuts,
//! and MCP `set_display`/`render_image` perspective + translucent enclosure controls.

use serde_json::{Value, json};
use simcraft::geometry::*;
use simcraft::render::{self, Camera, Section};
use simcraft::session::{Shared, new_core};
use std::collections::HashSet;

fn call(c: &Shared, t: &str, a: Value) -> Value {
    let o = c.call(t, &a).unwrap_or_else(|e| panic!("{t}: {e:#}"));
    serde_json::from_str(&o.text).unwrap_or(Value::String(o.text))
}

#[test]
fn camera_project_and_ray_inverse_ortho_and_perspective() {
    let points = [[0.0, 0.0, 0.0], [35.0, -12.0, 8.5], [-18.0, 22.0, -4.0], [100.0, 40.0, 10.0]];
    for perspective in [false, true] {
        let mut cam = Camera { yaw: -1.1, pitch: 0.48, zoom: 3.7, target: [42.0, 15.0, 5.0], perspective };
        let (w, h) = (1024.0, 640.0);
        for &p in &points {
            let (sx, sy, depth) = cam.project(p, w, h);
            assert!(depth > if perspective { 0.0 } else { -1e6 });
            let (orig, dir) = cam.ray(sx, sy, w, h, 500.0);
            let t = dot(sub(p, orig), dir);
            assert!(t > 0.0, "expected positive ray parameter for perspective={perspective}, got {t}");
            let closest = add(orig, scale(dir, t));
            let err = norm(sub(closest, p));
            assert!(err < 1e-6, "ray/project mismatch ({perspective}): err={err} at p={p:?}");
        }
        cam.fit([-10.0, -5.0, -2.0], [110.0, 45.0, 12.0], w, h);
        assert!(cam.zoom > 0.0 && cam.zoom.is_finite());
    }
}

#[test]
fn gpu_scene_builds_once_and_matches_rasterize_picking() {
    let c = new_core();
    call(&c, "add_box", json!({"name": "Beam", "min": [0, 0, 0], "size": [100, 20, 10]}));
    call(&c, "set_mesh", json!({"element_size_mm": 10.0}));
    call(&c, "add_fixed_support", json!({"faces": ["Beam.xmin"]}));
    call(&c, "add_force", json!({"faces": ["Beam.xmax"], "vector": [0, 0, -2500]}));
    call(&c, "solve", json!({}));

    let (mesh, model, results, mut disp) = {
        let s = c.session.lock().unwrap();
        (s.mesh.clone().unwrap(), s.model.clone(), s.results.clone(), s.display.clone())
    };
    disp.show_results = true;
    disp.show_undeformed = true;

    let gs = render::build_gpu_scene(&mesh, &model, results.as_deref(), &disp, &HashSet::new());
    assert!(!gs.vertices.is_empty());
    assert!(!gs.solid_indices.is_empty());
    assert!(gs.enclosure_indices.is_empty());
    assert!(!gs.undeformed_lines.is_empty(), "expected undeformed wireframe lines");
    assert!(gs.vertices.iter().any(|v| v.disp[2].abs() > 1e-4), "expected non-zero GPU vertex displacement");
    assert!(gs.scale_used > 0.0);

    let (w, h) = (720usize, 480usize);
    for perspective in [false, true] {
        let mut cam = Camera { yaw: -60f64.to_radians(), pitch: 28f64.to_radians(), zoom: 1.0, target: [0.0; 3], perspective };
        cam.fit(mesh.bbox.0, mesh.bbox.1, w as f64, h as f64);

        for anim in [0.65, 1.0] {
            let cpu_scene = render::build_scene(&mesh, &model, results.as_deref(), &disp, &HashSet::new(), anim);
            let frame = render::rasterize(&cpu_scene, &cam, w, h, true, false);
            assert_eq!(frame.rgba.len(), w * h * 4);

            // Sample foreground interior pixels (not on silhouette edges) and assert BVH QuadPicker matches rasterizer face_id.
            let mut matched = 0usize;
            for py in (20..h - 20).step_by(24) {
                for px in (20..w - 20).step_by(24) {
                    let fid = frame.face_id[py * w + px];
                    if fid == u32::MAX {
                        continue;
                    }
                    // Skip silhouette/seam pixels whose 4-neighbors differ in face_id
                    let same_neighbors = [
                        frame.face_id[(py - 2) * w + px],
                        frame.face_id[(py + 2) * w + px],
                        frame.face_id[py * w + (px - 2)],
                        frame.face_id[py * w + (px + 2)],
                    ]
                    .into_iter()
                    .all(|n| n == fid);
                    if !same_neighbors {
                        continue;
                    }
                    let picked = gs.picker.pick(&cam, px as f64 + 0.5, py as f64 + 0.5, w as f64, h as f64, anim, false).map(|(f, _)| f);
                    assert_eq!(picked, Some(fid), "picker/rasterizer face mismatch at ({px},{py}) perspective={perspective} anim={anim}");
                    matched += 1;
                }
            }
            assert!(matched >= 20, "expected >= 20 interior matched pixels, got {matched}");
        }
    }
}

#[test]
fn deformed_section_cut_and_enclosure_partitioning() {
    let c = new_core();
    call(&c, "set_analysis", json!({"type": "electrostatic"}));
    call(&c, "add_box", json!({"name": "Chip", "min": [0, 0, 0], "size": [20, 20, 6]}));
    call(&c, "add_enclosure", json!({"name": "AirBox", "material": "Air", "padding": 1.0}));
    call(&c, "set_mesh", json!({"element_size_mm": 6.0}));
    call(&c, "add_voltage", json!({"faces": ["Chip.xmax"], "value": 12.0}));
    call(&c, "add_voltage", json!({"faces": ["AirBox.xmin"], "value": 0.0}));
    call(&c, "solve", json!({}));

    let (mesh, model, results, mut disp) = {
        let s = c.session.lock().unwrap();
        (s.mesh.clone().unwrap(), s.model.clone(), s.results.clone(), s.display.clone())
    };
    disp.show_results = true;
    disp.section = Some(Section { axis: 0, position: 0.5 });

    let gs = render::build_gpu_scene(&mesh, &model, results.as_deref(), &disp, &HashSet::new());
    assert!(gs.has_enclosure);
    assert!(!gs.solid_indices.is_empty());
    assert!(!gs.enclosure_indices.is_empty(), "enclosure quads must be split into enclosure_indices");
    assert!(gs.vertices.iter().any(|v| v.kind > 0.5 && v.kind < 1.5));
    assert!(gs.vertices.iter().any(|v| v.kind < 0.5));

    // Also verify section cut on a structural deformed model preserves non-zero displacement on cut faces.
    let c2 = new_core();
    call(&c2, "add_box", json!({"name": "Bar", "min": [0, 0, 0], "size": [80, 20, 20]}));
    call(&c2, "set_mesh", json!({"element_size_mm": 10.0}));
    call(&c2, "add_fixed_support", json!({"faces": ["Bar.xmin"]}));
    call(&c2, "add_force", json!({"faces": ["Bar.xmax"], "vector": [0, 0, -4000]}));
    call(&c2, "solve", json!({}));
    let (m2, mod2, res2, mut d2) = {
        let s = c2.session.lock().unwrap();
        (s.mesh.clone().unwrap(), s.model.clone(), s.results.clone(), s.display.clone())
    };
    d2.show_results = true;
    d2.section = Some(Section { axis: 1, position: 0.5 });
    let gs2 = render::build_gpu_scene(&m2, &mod2, res2.as_deref(), &d2, &HashSet::new());
    assert!(gs2.vertices.iter().any(|v| v.disp[2].abs() > 1e-4));

    // Verify picking through an enclosure skips near wall and can pick the interior solid (or skip all enclosures when requested).
    let mut cam = Camera::default();
    cam.fit(mesh.bbox.0, mesh.bbox.1, 640.0, 480.0);
    let chip_center = [10.0, 10.0, 3.0];
    let (sx, sy, _) = cam.project(chip_center, 640.0, 480.0);
    let picked = gs.picker.pick(&cam, sx, sy, 640.0, 480.0, 1.0, false).map(|(f, _)| f);
    assert!(picked.is_some(), "expected solid pick through near enclosure wall");
    let fid = picked.unwrap() as usize;
    let body_idx = mesh.elem_body[mesh.faces[fid].elem as usize];
    assert_eq!(body_idx, 0, "expected interior body face before far enclosure wall, got body {body_idx}");
    let picked_skip = gs.picker.pick(&cam, sx, sy, 640.0, 480.0, 1.0, true).map(|(f, _)| f);
    assert_eq!(picked_skip, Some(picked.unwrap()));
}

#[test]
fn mcp_set_display_and_render_image_perspective_and_translucent() {
    let c = new_core();
    call(&c, "set_analysis", json!({"type": "electrostatic"}));
    call(&c, "add_box", json!({"name": "Block", "min": [0, 0, 0], "size": [30, 20, 15]}));
    call(&c, "add_enclosure", json!({"name": "Fluid", "material": "Air", "padding": 0.6}));
    call(&c, "set_display", json!({"perspective": true, "translucent_enclosure": true}));

    {
        let s = c.session.lock().unwrap();
        assert!(s.display.perspective);
        assert!(s.display.translucent_enclosure);
    }

    let ortho = c.call("render_image", &json!({"view": "iso", "width": 640, "height": 420, "perspective": false})).unwrap();
    let persp = c.call("render_image", &json!({"view": "iso", "width": 640, "height": 420, "perspective": true, "translucent_enclosure": true})).unwrap();
    let p_ortho = ortho.image_png.expect("ortho png");
    let p_persp = persp.image_png.expect("persp png");
    assert!(p_ortho.starts_with(b"\x89PNG\r\n\x1a\n") && p_ortho.len() > 256);
    assert!(p_persp.starts_with(b"\x89PNG\r\n\x1a\n") && p_persp.len() > 256);
    assert_ne!(p_ortho, p_persp, "perspective render should produce a distinct image from orthographic");
}
