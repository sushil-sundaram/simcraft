//! Integration tests for P2.U CAE UX overhaul:
//! - Live pre-solve readiness checklist (`preflight_check` & MCP `preflight`)
//! - Per-body hidden/ghost visibility in `GpuScene` & MCP `set_display`/`render_image`
//! - Depth-peeling `QuadPicker::pick_all` front-to-back hit ordering
//! - Copilot live context JSON, offline executable quick-fixes, and issue explainer

use serde_json::{Value, json};
use simcraft::assistant;
use simcraft::model::{PreflightStatus, preflight_check};
use simcraft::render::{self, Camera};
use simcraft::session::{Shared, new_core};
use std::collections::HashSet;

fn call(c: &Shared, t: &str, a: Value) -> Value {
    let o = c.call(t, &a).unwrap_or_else(|e| panic!("{t}: {e:#}"));
    serde_json::from_str(&o.text).unwrap_or(Value::String(o.text))
}

#[test]
fn preflight_diagnostics_cover_structural_fluid_fdtd_and_thermal() {
    let c = new_core();

    // 1. Empty model fails geometry check
    let pf0 = call(&c, "preflight", json!({}));
    assert_eq!(pf0["ready"], Value::Bool(false));
    let items0 = pf0["items"].as_array().unwrap();
    assert!(items0.iter().any(|x| x["code"] == "geometry" && x["status"] == "fail"));

    // 2. Structural box without support fails support, passes after support + force
    call(&c, "add_box", json!({"name": "Bracket", "min": [0, 0, 0], "size": [100, 20, 10]}));
    let pf1 = call(&c, "preflight", json!({}));
    assert_eq!(pf1["ready"], Value::Bool(false));
    assert!(pf1["items"].as_array().unwrap().iter().any(|x| x["code"] == "support" && x["status"] == "fail"));

    call(&c, "add_fixed_support", json!({"faces": ["Bracket.xmin"]}));
    call(&c, "add_force", json!({"faces": ["Bracket.xmax"], "vector": [0, 0, -500]}));
    let pf2 = call(&c, "preflight", json!({}));
    assert_eq!(pf2["ready"], Value::Bool(true));

    // 3. CFD checks enclosure and inlet+outlet pair
    call(&c, "set_analysis", json!({"type": "fluid"}));
    let pf_cfd0 = call(&c, "preflight", json!({}));
    assert_eq!(pf_cfd0["ready"], Value::Bool(false));
    assert!(pf_cfd0["items"].as_array().unwrap().iter().any(|x| x["code"] == "enclosure" && x["status"] == "fail"));

    call(&c, "add_enclosure", json!({"material": "Air", "padding": 0.5}));
    call(&c, "add_inlet", json!({"faces": ["Enclosure.xmin"], "velocity": 1.0}));
    let pf_cfd1 = call(&c, "preflight", json!({}));
    assert_eq!(pf_cfd1["ready"], Value::Bool(false));
    assert!(pf_cfd1["items"].as_array().unwrap().iter().any(|x| x["code"] == "flow_bc" && x["status"] == "fail"));

    call(&c, "add_outlet", json!({"faces": ["Enclosure.xmax"], "pressure": 0.0}));
    let pf_cfd2 = call(&c, "preflight", json!({}));
    assert_eq!(pf_cfd2["ready"], Value::Bool(true));

    // 4. Microwave heating warns when element size exceeds lambda / 10 in water-rich food
    call(&c, "new_model", json!({}));
    call(&c, "add_box", json!({"name": "Food", "min": [0, 0, 0], "size": [40, 40, 20], "material": "Food (water-rich, 2.45 GHz)"}));
    call(&c, "add_enclosure", json!({"material": "Air", "padding": 0.5}));
    call(&c, "set_analysis", json!({"type": "microwave_heating", "frequency_ghz": 2.45}));
    call(&c, "add_port", json!({"faces": ["Enclosure.xmin"], "power": 500.0, "polarization": "z"}));
    call(&c, "set_mesh", json!({"element_size_mm": 5.0}));
    {
        let s = c.session.lock().unwrap();
        let items = preflight_check(&s.model, s.mesh.as_deref(), s.element_size());
        assert!(items.iter().any(|it| it.code == "mesh_wavelength" && it.status == PreflightStatus::Warn));
    }
    call(&c, "set_mesh", json!({"element_size_mm": 1.2}));
    {
        let s = c.session.lock().unwrap();
        let items = preflight_check(&s.model, s.mesh.as_deref(), s.element_size());
        assert!(items.iter().any(|it| it.code == "mesh_wavelength" && it.status == PreflightStatus::Pass));
    }
}

#[test]
fn body_visibility_hidden_and_ghost_scene_and_depth_peel_picking() {
    let c = new_core();
    call(&c, "add_box", json!({"name": "FrontPlate", "min": [0, -40, 0], "size": [40, 10, 40]}));
    call(&c, "add_box", json!({"name": "BackPlate", "min": [0, 20, 0], "size": [40, 10, 40]}));
    call(&c, "set_mesh", json!({"element_size_mm": 10.0}));

    let (mesh, model, mut disp) = {
        let s = c.session.lock().unwrap();
        (s.mesh.clone().unwrap(), s.model.clone(), s.display.clone())
    };
    let mut cam = Camera::default();
    cam.set_view("front"); // looking along +Y from -Y side
    cam.fit(mesh.bbox.0, mesh.bbox.1, 640.0, 480.0);

    // Full model: depth peel hits FrontPlate first, then BackPlate along center ray
    let gs_all = render::build_gpu_scene(&mesh, &model, None, &disp, &HashSet::new());
    let center = [20.0, 0.0, 20.0];
    let (sx, sy, _) = cam.project(center, 640.0, 480.0);
    let peel = gs_all.picker.pick_all(&cam, sx, sy, 640.0, 480.0, 1.0, false);
    assert!(peel.len() >= 2, "expected multiple faces along depth-peel ray, got {}", peel.len());
    assert!(peel.windows(2).all(|w| w[0].1 <= w[1].1), "pick_all must sort front-to-back");
    let bodies_hit: Vec<String> = peel
        .iter()
        .map(|(fid, _, _)| mesh.body_names[mesh.elem_body[mesh.faces[*fid as usize].elem as usize] as usize].clone())
        .collect();
    assert_eq!(bodies_hit.first().map(String::as_str), Some("FrontPlate"));
    assert!(bodies_hit.iter().any(|b| b == "BackPlate"), "depth peel should reach BackPlate behind FrontPlate");

    // Ghosting FrontPlate moves it into ghost_indices and makes single pick click-through to BackPlate
    disp.ghost_bodies.insert("FrontPlate".into());
    let gs_ghost = render::build_gpu_scene(&mesh, &model, None, &disp, &HashSet::new());
    assert!(!gs_ghost.ghost_indices.is_empty(), "ghosted body should populate ghost_indices");
    let picked_thru = gs_ghost.picker.pick(&cam, sx, sy, 640.0, 480.0, 1.0, false).map(|(f, _)| f);
    let thru_body = picked_thru.map(|f| mesh.body_names[mesh.elem_body[mesh.faces[f as usize].elem as usize] as usize].as_str());
    assert_eq!(thru_body, Some("BackPlate"), "single click should pass through ghosted FrontPlate to BackPlate");

    // Hiding FrontPlate removes its geometry from GpuScene & alters isometric render_image PNG
    let mut disp_hidden = disp.clone();
    disp_hidden.ghost_bodies.clear();
    disp_hidden.hidden_bodies.insert("FrontPlate".into());
    let gs_hidden = render::build_gpu_scene(&mesh, &model, None, &disp_hidden, &HashSet::new());
    assert!(gs_hidden.vertices.len() < gs_all.vertices.len(), "hiding body must remove its vertices");

    call(&c, "set_display", json!({"hidden_bodies": ["FrontPlate"]}));
    let png_hidden = c.call("render_image", &json!({"view": "iso", "width": 400, "height": 300})).unwrap().image_png.unwrap();
    call(&c, "set_display", json!({"show_all_bodies": true}));
    let png_shown = c.call("render_image", &json!({"view": "iso", "width": 400, "height": 300})).unwrap().image_png.unwrap();
    assert_ne!(png_hidden, png_shown, "hiding a body should alter rendered image");
}

#[test]
fn copilot_quick_fixes_context_and_issue_explanation() {
    let c = new_core();
    call(&c, "add_box", json!({"name": "Arm", "min": [0, 0, 0], "size": [80, 20, 10]}));
    call(&c, "set_mesh", json!({"element_size_mm": 10.0}));
    let selection = vec!["Arm.xmin".to_string()];

    let (ctx_json, fixes) = {
        let s = c.session.lock().unwrap();
        let m = s.mesh.as_deref();
        (
            assistant::model_context_json(&s.model, m, &selection, s.element_size(), Some("no supports defined")),
            assistant::quick_fixes(&s.model, m, &selection, s.element_size()),
        )
    };
    assert_eq!(ctx_json["analysis"], "static_structural");
    assert_eq!(ctx_json["selected_faces"], json!(["Arm.xmin"]));
    assert!(fixes.iter().any(|fx| fx.tool == "add_fixed_support"), "expected quick-fix for selected support face: {fixes:?}");

    // Apply all proposed offline fixes via Core::call and confirm preflight support check turns Pass
    for fx in fixes {
        c.call(&fx.tool, &fx.args).unwrap();
    }
    let pf = call(&c, "preflight", json!({}));
    assert_eq!(pf["ready"], Value::Bool(true));

    assert!(assistant::explain_issue("no supports defined: rigid-body").to_lowercase().contains("fixed support"));
    assert!(assistant::explain_issue("needs an enclosure").to_lowercase().contains("enclosure"));
    assert!(assistant::explain_issue("Velocity Inlet needs a Pressure Outlet").to_lowercase().contains("outlet"));
}
