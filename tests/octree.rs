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

fn refined_bar(c: &Shared, analysis: &str) {
    call(c, "new_model", json!({}));
    call(c, "add_box", json!({"name": "Bar", "size": [20, 10, 6], "material": "Structural Steel"}));
    call(c, "set_analysis", json!({"type": analysis}));
    let m = call(c, "set_mesh", json!({"element_size_mm": 2, "refinements": [{"faces": ["Bar.xmax", "Bar.ymax"], "size_mm": 0.5}]}));
    assert!(m["octree"]["hanging_nodes"].as_u64().unwrap() > 50, "{m}");
}

/// Linear temperature (and constant flux) is exact across hanging nodes.
#[test]
fn thermal_patch_test_with_hanging_nodes() {
    let c = new_core();
    refined_bar(&c, "steady_thermal");
    call(&c, "add_temperature", json!({"faces": ["Bar.xmin"], "value": 0}));
    call(&c, "add_temperature", json!({"faces": ["Bar.xmax"], "value": 100}));
    let r = solve(&c);
    eprintln!("{}", serde_json::to_string(&r["fields"]).unwrap().chars().take(600).collect::<String>());
    eprintln!("{}", serde_json::to_string(&r["reactions"]).unwrap());
    let q = &r["fields"]["heat_flux"];
    let k = 60.5; // Structural Steel W/m·K
    let exact = k * 100.0 / 0.020; // W/m²
    let (qmin, qmax) = (q["min"].as_f64().unwrap(), q["max"].as_f64().unwrap());
    assert!((qmin - exact).abs() < 1e-6 * exact && (qmax - exact).abs() < 1e-6 * exact, "flux {qmin}..{qmax} vs {exact}");
}

/// Linear potential is exact across hanging nodes; the electrode charge balances.
#[test]
fn electrostatic_patch_test_with_hanging_nodes() {
    let c = new_core();
    refined_bar(&c, "electrostatic");
    call(&c, "set_material", json!({"body": "Bar", "material": "FR-4"}));
    call(&c, "add_voltage", json!({"faces": ["Bar.xmin"], "value": 0}));
    call(&c, "add_voltage", json!({"faces": ["Bar.xmax"], "value": 10}));
    let r = solve(&c);
    let e = &r["fields"]["electric_field"];
    let exact = 10.0 / 0.020;
    let (emin, emax) = (e["min"].as_f64().unwrap(), e["max"].as_f64().unwrap());
    assert!((emin - exact).abs() < 1e-6 * exact && (emax - exact).abs() < 1e-6 * exact, "E {emin}..{emax} vs {exact}");
    let q: Vec<f64> = r["reactions"].as_array().unwrap().iter().map(|x| x["charge_c"].as_f64().unwrap()).collect();
    // parallel plate: Q = ε0 εr A V / d (FR-4 εr = 4.4), exact for a linear potential
    let q_exact = 8.854_187_812_8e-12 * 4.4 * 60e-6 * 10.0 / 0.020;
    assert!((q[1] - q_exact).abs() < 1e-4 * q_exact && (q[0] + q[1]).abs() < 1e-9 * q_exact, "charges {q:?} vs {q_exact}");
}

/// Octree and uniform meshes give the same cantilever frequencies within discretisation error.
#[test]
fn modal_on_octree_matches_uniform() {
    let freqs = |refine: bool| -> Vec<f64> {
        let c = new_core();
        call(&c, "new_model", json!({}));
        call(&c, "add_box", json!({"name": "Beam", "size": [60, 6, 6], "material": "Structural Steel"}));
        call(&c, "set_analysis", json!({"type": "modal", "modes": 3}));
        if refine {
            call(&c, "set_mesh", json!({"element_size_mm": 1.5, "refinements": [{"faces": ["Beam.xmin"], "size_mm": 0.75}]}));
        } else {
            call(&c, "set_mesh", json!({"element_size_mm": 1.5}));
        }
        call(&c, "add_fixed_support", json!({"faces": ["Beam.xmin"]}));
        let r = solve(&c);
        r["modes"].as_array().unwrap_or_else(|| panic!("{r}")).iter().map(|m| m["frequency_hz"].as_f64().unwrap()).collect()
    };
    let (u, o) = (freqs(false), freqs(true));
    eprintln!("uniform {u:?}\noctree {o:?}");
    // Euler–Bernoulli first bending: f1 = 0.5596/L² √(EI/ρA)
    let (e, rho, l, a) = (200e9, 7850.0, 0.06, 0.006f64);
    let f1 = 0.5596 / (l * l) * (e * a * a / 12.0 / rho).sqrt();
    for f in [u[0], o[0]] {
        assert!((f - f1).abs() < 0.03 * f1, "f1 {f} vs beam theory {f1}");
    }
    assert!((o[0] - u[0]).abs() < 0.02 * u[0]);
}

/// Transient thermal on an octree mesh tracks the uniform-mesh answer mid-transient.
#[test]
fn transient_thermal_on_octree() {
    let run = |refine: bool| -> f64 {
        let c = new_core();
        call(&c, "new_model", json!({}));
        call(&c, "add_box", json!({"name": "Bar", "size": [20, 10, 6], "material": "Structural Steel"}));
        call(&c, "set_analysis", json!({"type": "transient_thermal", "end_time": 1.5, "time_steps": 30, "initial_temperature": 0}));
        if refine {
            call(&c, "set_mesh", json!({"element_size_mm": 1, "refinements": [{"faces": ["Bar.xmax"], "size_mm": 0.5}]}));
        } else {
            call(&c, "set_mesh", json!({"element_size_mm": 1}));
        }
        call(&c, "add_temperature", json!({"faces": ["Bar.xmax"], "value": 100}));
        solve(&c);
        call(&c, "probe", json!({"point": [14, 5, 3], "mode": 1000}))["values"]["temperature"].as_f64().unwrap()
    };
    let (u, o) = (run(false), run(true));
    eprintln!("uniform {u}, octree {o}");
    assert!(u > 5.0 && u < 95.0, "not mid-transient: {u}");
    assert!((u - o).abs() < 0.03 * u, "uniform {u} vs octree {o}");
}

#[test]
#[ignore]
fn dbg_thermal() {
    let c = new_core();
    refined_bar(&c, "steady_thermal");
    call(&c, "add_temperature", json!({"faces": ["Bar.xmin"], "value": 0}));
    call(&c, "add_temperature", json!({"faces": ["Bar.xmax"], "value": 100}));
    let r = solve(&c);
    for p in [[5.0, 5.0, 3.0], [19.0, 9.5, 3.0], [15.0, 9.75, 5.0], [17.0, 8.0, 1.0], [10.0, 10.0, 6.0]] {
        let v = call(&c, "probe", json!({"point": p}));
        eprintln!("{p:?} -> {}", v);
    }
    eprintln!("{}", r["fields"]["temperature"]);
    eprintln!("{}", r["fields"]["heat_flux"]);
}
