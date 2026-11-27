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


fn kirsch(c: &Shared, mesh: Value) -> (f64, usize) {
    call(c, "new_model", json!({}));
    call(c, "add_box", json!({"name": "Plate", "min": [0, 0, 0], "size": [300, 100, 2]}));
    call(c, "add_cylinder", json!({"name": "Hole", "base": [0, 0, 0], "axis": "z", "radius": 10, "height": 2, "operation": "subtract"}));
    let m = call(c, "set_mesh", mesh);
    call(c, "add_symmetry", json!({"faces": ["Plate.xmin", "Plate.ymin", "Plate.zmin"]}));
    call(c, "add_pressure", json!({"faces": ["Plate.xmax"], "value_mpa": -10}));
    let r = solve(c);
    (r["fields"]["max_principal"]["max"].as_f64().unwrap() / 10.0, m["elements"].as_u64().unwrap() as usize)
}

/// Kirsch strip with a hole: refining only around the hole gives the uniform-fine Kt with ≥ 5× fewer elements.
#[test]
fn kirsch_octree_matches_uniform_with_fewer_elements() {
    let kt_ref = (2.0 + 0.9f64.powi(3)) / 0.9;
    let c = new_core();
    let (kt_u, n_u) = kirsch(&c, json!({"element_size_mm": 1.0}));
    let (kt_o, n_o) = kirsch(&c, json!({"element_size_mm": 2.0, "refinements": [{"faces": ["Hole.side"], "size_mm": 0.5}]}));
    eprintln!("uniform 1 mm: Kt {kt_u:.4} ({n_u} elements); octree 2→0.5 mm: Kt {kt_o:.4} ({n_o} elements); reference {kt_ref:.4}");
    assert!((kt_o - kt_ref).abs() < 0.03 * kt_ref, "octree Kt {kt_o} vs {kt_ref}");
    assert!((kt_o - kt_u).abs() < 0.015 * kt_u, "octree {kt_o} vs uniform {kt_u}");
    assert!(n_u >= 5 * n_o, "uniform {n_u} vs octree {n_o} elements");
}

fn qtys(r: &Value, needle: &str) -> Vec<(String, f64)> {
    r["quantities"].as_array().unwrap().iter().filter(|q| q["name"].as_str().unwrap().contains(needle)).map(|q| (q["name"].as_str().unwrap().to_string(), q["value"].as_f64().unwrap())).collect()
}

/// Adaptive ZZ loop: the estimated error falls pass by pass and beats uniform refinement per DOF.
#[test]
fn adaptive_zz_refines_the_hole() {
    let kt_ref = (2.0 + 0.9f64.powi(3)) / 0.9;
    let c = new_core();
    call(&c, "new_model", json!({}));
    call(&c, "add_box", json!({"name": "Plate", "min": [0, 0, 0], "size": [300, 100, 2]}));
    call(&c, "add_cylinder", json!({"name": "Hole", "base": [0, 0, 0], "axis": "z", "radius": 10, "height": 2, "operation": "subtract"}));
    call(&c, "set_mesh", json!({"element_size_mm": 2.0, "adaptive_passes": 4, "adaptive_error_pct": 1.0, "max_levels": 3}));
    call(&c, "add_symmetry", json!({"faces": ["Plate.xmin", "Plate.ymin", "Plate.zmin"]}));
    call(&c, "add_pressure", json!({"faces": ["Plate.xmax"], "value_mpa": -10}));
    let r = solve(&c);
    let hist = qtys(&r, "Adaptive pass");
    let kt = r["fields"]["max_principal"]["max"].as_f64().unwrap() / 10.0;
    for h in &hist {
        eprintln!("{} = {:.3} %", h.0, h.1);
    }
    eprintln!("Kt {kt:.4} (reference {kt_ref:.4}); warnings {}", r["warnings"]);
    assert!(hist.len() >= 3, "{hist:?}");
    for w in hist.windows(2) {
        assert!(w[1].1 < w[0].1, "error did not fall: {hist:?}");
    }
    assert!((kt - kt_ref).abs() < 0.03 * kt_ref, "adaptive Kt {kt} vs {kt_ref}");
}

/// Energy-norm convergence: adaptive refinement converges faster per DOF than uniform refinement.
#[test]
fn adaptive_converges_faster_than_uniform() {
    let run = |mesh: Value| -> Vec<(f64, f64)> {
        let c = new_core();
        call(&c, "new_model", json!({}));
        call(&c, "add_box", json!({"name": "Plate", "min": [0, 0, 0], "size": [300, 100, 2]}));
        call(&c, "add_cylinder", json!({"name": "Hole", "base": [0, 0, 0], "axis": "z", "radius": 10, "height": 2, "operation": "subtract"}));
        call(&c, "set_mesh", mesh);
        call(&c, "add_symmetry", json!({"faces": ["Plate.xmin", "Plate.ymin", "Plate.zmin"]}));
        call(&c, "add_pressure", json!({"faces": ["Plate.xmax"], "value_mpa": -10}));
        let r = solve(&c);
        let h = qtys(&r, "Adaptive pass");
        if h.is_empty() {
            let dofs = r["info"]["dofs"].as_f64().unwrap();
            return vec![(dofs, qtys(&r, "Estimated discretisation error")[0].1)];
        }
        h.iter().map(|(n, e)| (n.split(", ").nth(1).unwrap().split(' ').next().unwrap().parse::<f64>().unwrap(), *e)).collect()
    };
    let uni: Vec<(f64, f64)> = [2.0, 1.0].iter().flat_map(|hh| run(json!({"element_size_mm": hh}))).collect();
    let ada = run(json!({"element_size_mm": 2.0, "adaptive_passes": 3, "adaptive_error_pct": 0.1, "max_levels": 3}));
    let rate = |p: &[(f64, f64)]| -((p[p.len() - 1].1 / p[0].1).ln() / (p[p.len() - 1].0 / p[0].0).ln());
    eprintln!("uniform {uni:?} rate {:.2}\nadaptive {ada:?} rate {:.2}", rate(&uni), rate(&ada));
    assert!(rate(&ada) > 2.0 * rate(&uni), "adaptive rate {} vs uniform {}", rate(&ada), rate(&uni));
    // the adaptive mesh beats the uniform 1 mm mesh with far fewer unknowns
    let (ua, ea) = *ada.last().unwrap();
    let (uu, eu) = uni[1];
    assert!(ea < eu && ua < 0.5 * uu, "adaptive ({ua}, {ea}) vs uniform ({uu}, {eu})");
}

/// Buckling, plasticity and harmonic response run on octree meshes and agree with uniform meshes.
#[test]
fn other_structural_analyses_on_octree() {
    let run = |analysis: Value, refine: bool, extra: &dyn Fn(&Shared)| -> Value {
        let c = new_core();
        call(&c, "new_model", json!({}));
        call(&c, "add_box", json!({"name": "Col", "size": [6, 6, 60], "material": "Structural Steel"}));
        call(&c, "set_analysis", analysis);
        let mesh = if refine { json!({"element_size_mm": 1.5, "refinements": [{"faces": ["Col.zmin"], "size_mm": 0.75}]}) } else { json!({"element_size_mm": 1.5}) };
        let m = call(&c, "set_mesh", mesh);
        assert_eq!(m.get("octree").is_some(), refine, "{m}");
        call(&c, "add_fixed_support", json!({"faces": ["Col.zmin"]}));
        extra(&c);
        solve(&c)
    };
    let close = |a: f64, b: f64, tol: f64, what: &str| assert!((a - b).abs() <= tol * a.abs().max(b.abs()), "{what}: uniform {a} vs octree {b}");
    // linear buckling: load factor
    let ld = |c: &Shared| { call(c, "add_force", json!({"faces": ["Col.zmax"], "vector": [0, 0, -1000]})); };
    let bf = |r: &Value| r["modes"][0]["load_multiplier"].as_f64().or_else(|| r["modes"][0]["frequency_hz"].as_f64()).unwrap_or_else(|| panic!("{}", r["modes"]));
    let (u, o) = (run(json!({"type": "linear_buckling", "modes": 1}), false, &ld), run(json!({"type": "linear_buckling", "modes": 1}), true, &ld));
    close(bf(&u), bf(&o), 0.03, "buckling");
    // plasticity: tip deflection past yield
    let lp = |c: &Shared| { call(c, "add_force", json!({"faces": ["Col.zmax"], "vector": [400, 0, 0]})); };
    let dx = |r: &Value| r["fields"]["deformation_x"]["max"].as_f64().unwrap();
    let (u, o) = (run(json!({"type": "nonlinear_structural", "load_steps": 5}), false, &lp), run(json!({"type": "nonlinear_structural", "load_steps": 5}), true, &lp));
    let fine = {
        let c = new_core();
        call(&c, "new_model", json!({}));
        call(&c, "add_box", json!({"name": "Col", "size": [6, 6, 60], "material": "Structural Steel"}));
        call(&c, "set_analysis", json!({"type": "nonlinear_structural", "load_steps": 5}));
        call(&c, "set_mesh", json!({"element_size_mm": 0.75}));
        call(&c, "add_fixed_support", json!({"faces": ["Col.zmin"]}));
        lp(&c);
        solve(&c)
    };
    eprintln!("plasticity: uniform 1.5 {} octree {} uniform 0.75 {}", dx(&u), dx(&o), dx(&fine));
    // refining the plastic hinge moves the answer from the coarse towards the fine uniform mesh
    assert!(dx(&o) > dx(&u) && dx(&o) < dx(&fine), "plasticity: {} {} {}", dx(&u), dx(&o), dx(&fine));
    // harmonic response: peak amplitude frequency
    let lh = |c: &Shared| { call(c, "add_force", json!({"faces": ["Col.zmax"], "vector": [10, 0, 0]})); };
    let pk = |r: &Value| r["fields"]["total_deformation"]["max"].as_f64().unwrap();
    let (u, o) = (run(json!({"type": "harmonic_response", "freq_steps": 60}), false, &lh), run(json!({"type": "harmonic_response", "freq_steps": 60}), true, &lh));
    eprintln!("harmonic peak uniform {} octree {}", pk(&u), pk(&o));
    close(pk(&u), pk(&o), 0.15, "harmonic peak");
}

/// Magnetostatics with an air enclosure: refining the magnet converges towards the fine uniform mesh.
#[test]
fn magnetostatic_on_octree() {
    let run = |refine: bool| -> f64 { run_h(refine, 2.5) };
    let fine = run_h(false, 1.25);
    eprintln!("fine uniform {fine}");
    fn run_h(refine: bool, hh: f64) -> f64 {
        let c = new_core();
        call(&c, "new_model", json!({}));
        call(&c, "add_box", json!({"name": "Mag", "min": [-5, -5, -5], "size": [10, 10, 10], "material": "Structural Steel"}));
        call(&c, "set_analysis", json!({"type": "magnetostatic"}));
        call(&c, "add_enclosure", json!({"margin": 15}));
        call(&c, "add_magnet", json!({"body": "Mag", "remanence": [0, 0, 1.2]}));
        let mesh = if refine { json!({"element_size_mm": hh, "refinements": [{"body": "Mag", "size_mm": hh / 2.0}]}) } else { json!({"element_size_mm": hh}) };
        let m = call(&c, "set_mesh", mesh);
        assert_eq!(m.get("octree").is_some(), refine, "{m}");
        let r = solve(&c);
        r["quantities"].as_array().unwrap().iter().find(|q| q["name"] == "Stored magnetic energy").unwrap()["value"].as_f64().unwrap()
    }
    let (u, o) = (run(false), run(true));
    eprintln!("magnetic energy uniform {u} octree {o}");
    // refining the magnet moves the energy from the coarse towards the fine uniform answer
    assert!((o - fine).abs() < (u - fine).abs() * 0.5, "coarse {u}, octree {o}, fine {fine}");
}
