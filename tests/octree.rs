//! P2.5 local mesh refinement: 2:1 octree meshes, hanging-node constraints and adaptive ZZ control.
use serde_json::json;
use simcraft::mesh;
use simcraft::model::Model;

fn model(v: serde_json::Value) -> Model {
    serde_json::from_value(v).unwrap()
}

fn plate(refine: serde_json::Value) -> Model {
    model(json!({
        "bodies": [{"name": "Plate", "shape": {"type": "box", "min": [0,0,0], "size": [40, 20, 4]}, "material": "Structural Steel"}],
        "element_size": 2.0,
        "mesh": refine,
    }))
}

#[test]
fn octree_box_basic() {
    let m = plate(json!({"local_refinement": true, "refinements": [{"faces": ["Plate.xmin"], "size_mm": 0.5}]}));
    let me = mesh::build_for(&m, 2.0, 2_000_000).unwrap();
    let uni = mesh::build_for(&plate(json!({})), 2.0, 2_000_000).unwrap();
    eprintln!("octree: {} elems, {} nodes, {} hanging, levels {:?}; uniform {} elems", me.elems.len(), me.nodes.len(), me.hanging.len(),
        (0..=me.max_level).map(|l| me.elem_level.iter().filter(|&&x| x == l).count()).collect::<Vec<_>>(), uni.elems.len());
    assert!(me.octree);
    assert!((me.volume_mm3() - 3200.0).abs() < 1e-6 * 3200.0, "volume {}", me.volume_mm3());
    // every hanging node sits at the weighted average of its parents
    for hn in &me.hanging {
        let s: f64 = hn.parents.iter().map(|p| p.1).sum();
        assert!((s - 1.0).abs() < 1e-12);
        let mut p = [0.0; 3];
        for &(q, w) in &hn.parents { for k in 0..3 { p[k] += w * me.nodes[q as usize][k]; } }
        for k in 0..3 { assert!((p[k] - me.nodes[hn.node as usize][k]).abs() < 1e-9, "hanging {} at {:?} vs {:?}", hn.node, me.nodes[hn.node as usize], p); }
    }
}

use simcraft::session::{Shared, new_core};
use serde_json::Value;

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

fn qty(res: &Value, prefix: &str) -> f64 {
    res["quantities"].as_array().unwrap().iter().find(|q| q["name"].as_str().unwrap_or("").starts_with(prefix)).unwrap_or_else(|| panic!("no '{prefix}' in {res}"))["value"].as_f64().unwrap()
}

/// Patch test: a uniform strain state is reproduced exactly across 2:1 hanging-node transitions.
#[test]
fn structural_patch_test_with_hanging_nodes() {
    let c = new_core();
    call(&c, "new_model", json!({}));
    call(&c, "add_box", json!({"name": "Bar", "size": [20, 10, 6], "material": "Structural Steel"}));
    let m = call(&c, "set_mesh", json!({"element_size_mm": 2, "refinements": [{"faces": ["Bar.xmax", "Bar.ymax"], "size_mm": 0.5}]}));
    eprintln!("{m}");
    assert!(m["octree"]["hanging_nodes"].as_u64().unwrap() > 50, "{m}");
    call(&c, "add_displacement", json!({"faces": ["Bar.xmin"], "x": 0}));
    call(&c, "add_symmetry", json!({"faces": ["Bar.ymin"]}));
    call(&c, "add_symmetry", json!({"faces": ["Bar.zmin"]}));
    call(&c, "add_displacement", json!({"faces": ["Bar.xmax"], "x": 0.02}));
    let r = solve(&c);
    let e_mpa = 200_000.0; // Structural Steel
    let s_exact = e_mpa * 0.02 / 20.0;
    let vm = &r["fields"]["von_mises"];
    let (vmin, vmax) = (vm["min"].as_f64().unwrap(), vm["max"].as_f64().unwrap());
    assert!((vmax - s_exact).abs() < 1e-6 * s_exact && (vmin - s_exact).abs() < 1e-6 * s_exact, "von Mises {vmin}..{vmax} vs exact {s_exact}");
    // lateral contraction −ν ε W is exact too
    let uy = r["fields"]["deformation_y"]["min"].as_f64().unwrap();
    assert!((uy + 0.3 * 0.001 * 10.0).abs() < 1e-9, "uy {uy}");
}
