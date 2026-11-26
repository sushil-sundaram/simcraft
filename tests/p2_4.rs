//! P2.4 Assemblies: contact interfaces (bonded, no_separation, frictionless, frictional),
//! two-step bolt pretension, springs, remote forces/moments, remote displacements, and point masses.

use serde_json::{Value, json};
use simcraft::results::Field;
use simcraft::session::{Shared, new_core};

fn call(c: &Shared, tool: &str, args: Value) -> Value {
    let r = c.call(tool, &args).unwrap_or_else(|e| panic!("{tool}({args}) failed: {e:#}"));
    serde_json::from_str(&r.text).unwrap_or_else(|_| json!({ "text": r.text }))
}

fn wait_done(c: &Shared) {
    while c.is_busy() {
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    let st = call(c, "get_status", json!({}));
    if let Some(e) = st["last_error"].as_str() {
        panic!("async job failed: {e}");
    }
}

fn solve(c: &Shared) -> Value {
    let _ = call(c, "solve", json!({}));
    wait_done(c);
    call(c, "get_results", json!({}))
}

fn qty(res: &Value, prefix: &str) -> f64 {
    res["quantities"]
        .as_array()
        .unwrap()
        .iter()
        .find(|q| q["name"].as_str().unwrap_or("").starts_with(prefix))
        .unwrap_or_else(|| panic!("missing quantity starting with '{prefix}' in {res}"))["value"]
        .as_f64()
        .unwrap()
}

fn qty_contains(res: &Value, needle: &str) -> f64 {
    res["quantities"]
        .as_array()
        .unwrap()
        .iter()
        .find(|q| q["name"].as_str().unwrap_or("").contains(needle))
        .unwrap_or_else(|| panic!("missing quantity containing '{needle}' in {res}"))["value"]
        .as_f64()
        .unwrap()
}

fn reaction_on(res: &Value, label: &str) -> [f64; 3] {
    let arr = res["reactions"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["label"].as_str() == Some(label))
        .unwrap_or_else(|| panic!("missing reaction '{label}' in {res}"))["force_n"]
        .as_array()
        .unwrap();
    [arr[0].as_f64().unwrap(), arr[1].as_f64().unwrap(), arr[2].as_f64().unwrap()]
}

fn assert_rel(got: f64, want: f64, tol: f64, ctx: &str) {
    let err = (got - want).abs() / want.abs().max(1e-12);
    assert!(err <= tol, "{ctx}: got {got}, want {want} (rel err {:.2}% > {:.2}%)", err * 100.0, tol * 100.0);
}

#[test]
fn stacked_blocks_separation_vs_no_separation() {
    let make = |kind: &str| {
        let c = new_core();
        call(&c, "add_box", json!({"name": "Base", "min": [0, 0, 0], "size": [60, 12, 12], "material": "Structural Steel"}));
        call(&c, "add_box", json!({"name": "Top", "min": [0, 0, 12], "size": [60, 12, 12], "material": "Structural Steel"}));
        call(&c, "set_mesh", json!({"element_size_mm": 6.0}));
        call(&c, "add_contact", json!({"body_a": "Base", "body_b": "Top", "kind": kind}));
        call(&c, "add_fixed_support", json!({"faces": ["Base.zmin", "Top.xmin"]}));
        call(&c, "add_force", json!({"faces": ["Top.xmax"], "vector": [0, 0, 400]}));
        solve(&c)
    };

    // Frictionless: upward +400 N load lifts Top completely off Base -> free cantilever deflection and ~0 normal force on Base
    let r_sep = make("frictionless");
    let uz_sep = r_sep["fields"]["deformation_z"]["max"].as_f64().unwrap();
    let e = 200e3;
    let i_yy = 12.0 * 12.0_f64.powi(3) / 12.0;
    let uz_cantilever = 400.0 * 60.0_f64.powi(3) / (3.0 * e * i_yy);
    assert_rel(uz_sep, uz_cantilever, 0.05, "frictionless upward separation tip uz vs free cantilever");
    assert!(qty(&r_sep, "Contact nodes open") > 0.0, "expected open nodes when lifting off");
    assert!(qty(&r_sep, "Contact normal force").abs() < 1.0, "expected near-zero compressive normal force on separation");

    // No separation: bottom of Top stays tied normally to Base.zmax -> deflection strongly suppressed
    let r_nosep = make("no_separation");
    let uz_nosep = r_nosep["fields"]["deformation_z"]["max"].as_f64().unwrap();
    assert!(uz_nosep < 0.05 * uz_sep, "no_separation should restrain lift-off (got {uz_nosep} vs {uz_sep})");
    assert_eq!(qty(&r_nosep, "Contact nodes open"), 0.0, "no_separation must never open");

    // Push downward under frictionless contact: Top bears against Base
    let c_down = new_core();
    call(&c_down, "add_box", json!({"name": "Base", "min": [0, 0, 0], "size": [60, 12, 12], "material": "Structural Steel"}));
    call(&c_down, "add_box", json!({"name": "Top", "min": [0, 0, 12], "size": [60, 12, 12], "material": "Structural Steel"}));
    call(&c_down, "set_mesh", json!({"element_size_mm": 6.0}));
    call(&c_down, "add_contact", json!({"body_a": "Base", "body_b": "Top", "kind": "frictionless"}));
    call(&c_down, "add_fixed_support", json!({"faces": ["Base.zmin"]}));
    call(&c_down, "add_displacement", json!({"faces": ["Top.xmin"], "x": 0.0, "y": 0.0}));
    call(&c_down, "add_pressure", json!({"faces": ["Top.zmax"], "value_mpa": 1.0}));
    let r_down = solve(&c_down);
    let expected_fn = 60.0 * 12.0 * 1.0; // 720 N
    assert_rel(qty(&r_down, "Contact normal force"), expected_fn, 0.01, "uniform contact normal force");
    assert_rel(qty(&r_down, "Peak contact pressure"), 1.0, 0.02, "uniform contact pressure");
    assert_rel(reaction_on(&r_down, "Fixed Support")[2], expected_fn, 0.01, "base fixed support reaction");
}

#[test]
fn sliding_block_coulomb_friction() {
    for mu in [0.0, 0.25, 0.5] {
        let c = new_core();
        call(&c, "add_box", json!({"name": "Plane", "min": [0, 0, 0], "size": [60, 30, 10], "material": "Structural Steel"}));
        call(&c, "add_box", json!({"name": "Block", "min": [20, 5, 10], "size": [20, 20, 10], "material": "Structural Steel"}));
        call(&c, "set_mesh", json!({"element_size_mm": 5.0}));
        call(&c, "add_contact", json!({"body_a": "Block", "body_b": "Plane", "kind": "frictional", "friction": mu}));
        call(&c, "add_fixed_support", json!({"faces": ["Plane.zmin"]}));
        call(&c, "add_displacement", json!({"faces": ["Block.xmax"], "x": 0.02, "y": 0.0}));
        call(&c, "add_pressure", json!({"faces": ["Block.zmax"], "value_mpa": 2.0}));
        let r = solve(&c);
        let fn_got = qty(&r, "Contact normal force");
        let ft_got = qty(&r, "Contact tangential force");
        let rx = reaction_on(&r, "Displacement")[0];
        let rx_base = reaction_on(&r, "Fixed Support")[0];
        assert_rel(fn_got, 800.0, 0.01, &format!("normal force mu={mu}"));
        let want_f = mu * 800.0;
        if mu == 0.0 {
            assert!(rx.abs() < 1e-2 && ft_got.abs() < 1e-2, "zero friction should give zero sliding force (rx={rx}, ft={ft_got})");
        } else {
            assert_rel(ft_got, want_f, 0.02, &format!("tangential friction force mu={mu}"));
            assert_rel(rx, want_f, 0.02, &format!("pull reaction vs mu*N mu={mu}"));
            assert_rel(-rx_base, want_f, 0.02, &format!("base fixed support horizontal reaction vs mu*N mu={mu}"));
        }
    }

    // Static sticking regime: Force Fx = 100 N < mu*N = 200 N WITHOUT any horizontal Displacement restraint on Block
    let c_stk = new_core();
    call(&c_stk, "add_box", json!({"name": "Plane", "min": [0, 0, 0], "size": [60, 30, 10], "material": "Structural Steel"}));
    call(&c_stk, "add_box", json!({"name": "Block", "min": [20, 5, 10], "size": [20, 20, 10], "material": "Structural Steel"}));
    call(&c_stk, "set_mesh", json!({"element_size_mm": 5.0}));
    call(&c_stk, "add_contact", json!({"body_a": "Block", "body_b": "Plane", "kind": "frictional", "friction": 0.25}));
    call(&c_stk, "add_fixed_support", json!({"faces": ["Plane.zmin"]}));
    call(&c_stk, "add_pressure", json!({"faces": ["Block.zmax"], "value_mpa": 2.0}));
    call(&c_stk, "add_force", json!({"faces": ["Block.xmin"], "vector": [100.0, 0.0, 0.0]}));
    let r_stk = solve(&c_stk);
    assert_rel(qty(&r_stk, "Contact tangential force"), 100.0, 0.01, "static sticking friction equilibrates Fx = 100 N");
    let rx_stk = reaction_on(&r_stk, "Fixed Support")[0];
    assert_rel(-rx_stk, 100.0, 0.01, "base reaction equilibrates Fx = 100 N without horizontal BC on Block");
    assert!(r_stk["fields"]["deformation_x"]["max"].as_f64().unwrap() < 0.002, "sticking block must stay static (ux < 0.002 mm)");
}

#[test]
fn hertz_cylinder_on_flat_with_fitting() {
    let e_gpa: f64 = 3.0;
    let nu: f64 = 0.35;
    let r_mm: f64 = 40.0;
    let len_mm: f64 = 4.0;
    let f_half: f64 = 800.0;
    let f_prime = (2.0 * f_half) / (len_mm * 1e-3);
    let e_star = (e_gpa * 1e9) / (2.0 * (1.0 - nu * nu));
    let p0_want_mpa = (f_prime * e_star / (std::f64::consts::PI * r_mm * 1e-3)).sqrt() * 1e-6;

    let c = new_core();
    call(&c, "define_material", json!({"name": "HertzPolymer", "youngs_modulus_gpa": e_gpa, "poisson_ratio": nu}));
    call(&c, "add_box", json!({"name": "Base", "min": [0, 0, -24], "size": [28, len_mm, 24], "material": "HertzPolymer"}));
    call(&c, "add_cylinder", json!({"name": "Cyl", "base": [0, 0, r_mm], "axis": "y", "radius": r_mm, "height": len_mm, "material": "HertzPolymer"}));
    call(&c, "add_box", json!({"name": "CutX", "min": [-45, -1, -5], "size": [45, len_mm + 2.0, 90], "operation": "subtract"}));
    call(&c, "add_box", json!({"name": "CutTop", "min": [-5, -1, 20], "size": [50, len_mm + 2.0, 70], "operation": "subtract"}));
    call(&c, "set_mesh", json!({"element_size_mm": 1.0}));
    call(&c, "add_contact", json!({"body_a": "Cyl", "body_b": "Base", "kind": "frictionless"}));
    call(&c, "add_fixed_support", json!({"faces": ["Base.zmin"]}));
    call(&c, "add_symmetry", json!({"faces": ["bbox.xmin", "bbox.ymin", "bbox.ymax"]}));
    call(&c, "add_force", json!({"faces": ["CutTop.zmin"], "vector": [0, 0, -f_half]}));
    let res = solve(&c);

    let peak_p = qty(&res, "Peak contact pressure");
    let fn_got = qty(&res, "Contact normal force");
    assert_rel(fn_got, f_half, 0.01, "Hertz total normal force");
    assert_rel(peak_p, p0_want_mpa, 0.10, "Hertz fitted cylinder-on-flat peak pressure (within 10%)");
    assert!(res["fields"].get(Field::ContactPressure.key()).is_some());
}

#[test]
fn bolted_joint_stiffness_ratio_and_preload_loss() {
    let setup_joint = |preload_n: f64, ext_z: f64| -> (f64, f64, f64) {
        let c = new_core();
        call(&c, "add_box", json!({"name": "PlateBot", "min": [0, 0, 0], "size": [24, 24, 8], "material": "Structural Steel"}));
        call(&c, "add_box", json!({"name": "PlateTop", "min": [0, 0, 8], "size": [24, 24, 8], "material": "Structural Steel"}));
        call(&c, "add_cylinder", json!({"name": "ClearHole", "base": [12, 12, 3], "axis": "z", "radius": 5, "height": 14, "operation": "subtract"}));
        call(&c, "add_cylinder", json!({"name": "Bolt", "base": [12, 12, 0], "axis": "z", "radius": 4, "height": 20, "material": "Structural Steel"}));
        call(&c, "add_box", json!({"name": "BoltHead", "min": [6, 6, 16], "size": [12, 12, 4], "material": "Structural Steel"}));
        call(&c, "set_mesh", json!({"element_size_mm": 2.0}));
        call(&c, "add_contact", json!({"body_a": "PlateBot", "body_b": "PlateTop", "kind": "frictionless"}));
        call(&c, "add_fixed_support", json!({"faces": ["PlateBot.zmin"]}));
        call(&c, "add_displacement", json!({"faces": ["PlateTop.xmax", "PlateTop.xmin"], "x": 0.0, "y": 0.0}));
        call(&c, "add_bolt_pretension", json!({"body": "Bolt", "force_n": preload_n, "cut_mm": 10.0}));
        if ext_z != 0.0 {
            call(&c, "add_force", json!({"faces": ["PlateTop.zmax"], "vector": [0, 0, ext_z]}));
        }
        let r = solve(&c);
        (qty_contains(&r, "force (Step 1)"), qty_contains(&r, "force (Step 2)"), qty_contains(&r, "preload loss"))
    };

    let preload = 10_000.0;
    let (f1_t, f2_t, loss_t) = setup_joint(preload, 4_000.0);
    assert_rel(f1_t, preload, 0.01, "Step 1 bolt preload target");
    assert!(f2_t > f1_t && f2_t < f1_t + 4_000.0, "tension external load splits between bolt and plates (f1={f1_t}, f2={f2_t})");
    assert_rel(loss_t, f1_t - f2_t, 0.002, "preload loss identity");

    let (f1_c, f2_c, loss_c) = setup_joint(preload, -4_000.0);
    assert_rel(f1_c, preload, 0.01, "Step 1 compressive test preload");
    assert!(loss_c > 100.0 && f2_c < f1_c, "compressive working load causes positive preload loss = {loss_c} N");

    // Verify joint stiffness ratio Phi = (F_b2 - F_b1) / F_ext against VDI 2230 delta_p / (delta_s + delta_p):
    // measure bolt+head compliance delta_s and clamped plates compliance delta_p independently on the same mesh!
    let c_bolt = new_core();
    call(&c_bolt, "add_cylinder", json!({"name": "Bolt", "base": [12, 12, 0], "axis": "z", "radius": 4, "height": 20, "material": "Structural Steel"}));
    call(&c_bolt, "add_box", json!({"name": "BoltHead", "min": [6, 6, 16], "size": [12, 12, 4], "material": "Structural Steel"}));
    call(&c_bolt, "set_mesh", json!({"element_size_mm": 2.0}));
    call(&c_bolt, "add_fixed_support", json!({"faces": ["Bolt.bottom"]}));
    call(&c_bolt, "add_force", json!({"faces": ["BoltHead.zmax"], "vector": [0, 0, 1000]}));
    let rb = solve(&c_bolt);
    let delta_s = rb["fields"]["deformation_z"]["max"].as_f64().unwrap() / 1000.0; // mm/N

    let c_plate = new_core();
    call(&c_plate, "add_box", json!({"name": "PlateBot", "min": [0, 0, 0], "size": [24, 24, 8], "material": "Structural Steel"}));
    call(&c_plate, "add_box", json!({"name": "PlateTop", "min": [0, 0, 8], "size": [24, 24, 8], "material": "Structural Steel"}));
    call(&c_plate, "add_cylinder", json!({"name": "ClearHole", "base": [12, 12, 0], "axis": "z", "radius": 5, "height": 16, "operation": "subtract"}));
    call(&c_plate, "set_mesh", json!({"element_size_mm": 2.0}));
    call(&c_plate, "add_fixed_support", json!({"faces": ["PlateBot.zmin"]}));
    // Clamp zone matching BoltHead footprint around z=16
    call(&c_plate, "add_pressure", json!({"faces": ["PlateTop.zmax"], "value_mpa": 1.0}));
    let rp = solve(&c_plate);
    let p_force = (24.0 * 24.0 - std::f64::consts::PI * 25.0) * 1.0;
    let delta_p = (-rp["fields"]["deformation_z"]["min"].as_f64().unwrap()) / p_force; // mm/N
    let phi_vdi = delta_p / (delta_s + delta_p);
    let phi_joint = (f2_t - f1_t) / 4_000.0;
    eprintln!("VDI 2230 check: Phi_joint={phi_joint:.4}, Phi_submodels={phi_vdi:.4}, delta_s={delta_s:.3e}, delta_p={delta_p:.3e}");
    assert!(phi_joint > 0.01 && phi_joint < 0.35, "stiffness factor Phi = {phi_joint:.4} in physical range");
    assert!((phi_joint - phi_vdi).abs() < 0.08, "VDI 2230 stiffness factor Phi={phi_joint:.4} vs isolated compliance={phi_vdi:.4}");

    // Also test prescribed adjustment_mm mode directly
    let c_adj = new_core();
    call(&c_adj, "add_box", json!({"name": "PlateBot", "min": [0, 0, 0], "size": [24, 24, 8], "material": "Structural Steel"}));
    call(&c_adj, "add_box", json!({"name": "PlateTop", "min": [0, 0, 8], "size": [24, 24, 8], "material": "Structural Steel"}));
    call(&c_adj, "add_cylinder", json!({"name": "ClearHole", "base": [12, 12, 3], "axis": "z", "radius": 5, "height": 14, "operation": "subtract"}));
    call(&c_adj, "add_cylinder", json!({"name": "Bolt", "base": [12, 12, 0], "axis": "z", "radius": 4, "height": 20, "material": "Structural Steel"}));
    call(&c_adj, "add_box", json!({"name": "BoltHead", "min": [6, 6, 16], "size": [12, 12, 4], "material": "Structural Steel"}));
    call(&c_adj, "set_mesh", json!({"element_size_mm": 2.0}));
    call(&c_adj, "add_contact", json!({"body_a": "PlateBot", "body_b": "PlateTop", "kind": "frictionless"}));
    call(&c_adj, "add_fixed_support", json!({"faces": ["PlateBot.zmin"]}));
    call(&c_adj, "add_displacement", json!({"faces": ["PlateTop.xmax", "PlateTop.xmin"], "x": 0.0, "y": 0.0}));
    call(&c_adj, "add_bolt_pretension", json!({"body": "Bolt", "adjustment_mm": 0.02, "cut_mm": 10.0}));
    let r_adj = solve(&c_adj);
    assert_rel(qty_contains(&r_adj, "adjustment"), 0.02, 1e-4, "prescribed bolt adjustment_mm locked");
    assert!(qty_contains(&r_adj, "force (Step 1)") > 3000.0, "positive adjustment creates positive clamping preload");
}

#[test]
fn connections_remote_moment_spring_and_point_mass() {
    // 1. Remote Moment on cantilever beam vs Euler-Bernoulli theory
    {
        let c = new_core();
        call(&c, "add_box", json!({"name": "Beam", "min": [0, 0, 0], "size": [120, 12, 12], "material": "Structural Steel"}));
        call(&c, "set_mesh", json!({"element_size_mm": 3.0}));
        call(&c, "add_fixed_support", json!({"faces": ["Beam.xmin"]}));
        call(&c, "add_remote_force", json!({"faces": ["Beam.xmax"], "point": [120, 6, 6], "force": [0, 0, 0], "moment": [0, 20, 0]}));
        let r = solve(&c);
        let uz_got = -r["fields"]["deformation_z"]["min"].as_f64().unwrap();
        let e = 200e9;
        let i_yy = 0.012 * 0.012_f64.powi(3) / 12.0;
        let uz_want = (20.0 * 0.12_f64.powi(2)) / (2.0 * e * i_yy) * 1e3;
        assert_rel(uz_got, uz_want, 0.02, "remote moment cantilever tip deflection vs beam theory");
    }

    // 2. Eccentric Remote Force + Rigid Remote Displacement
    {
        let c = new_core();
        call(&c, "add_box", json!({"name": "Beam", "min": [0, 0, 0], "size": [100, 10, 10], "material": "Structural Steel"}));
        call(&c, "set_mesh", json!({"element_size_mm": 2.5}));
        call(&c, "add_remote_displacement", json!({"faces": ["Beam.xmin"], "point": [0, 5, 5], "x": 0, "y": 0, "z": 0, "rx": 0, "ry": 0, "rz": 0, "rigid": true}));
        call(&c, "add_remote_force", json!({"faces": ["Beam.xmax"], "point": [150, 5, 5], "force": [0, 0, -100]}));
        let r = solve(&c);
        let uz_got = -r["fields"]["deformation_z"]["min"].as_f64().unwrap();
        let e = 200e9;
        let g = e / (2.0 * (1.0 + 0.3));
        let i = 0.01 * 0.01_f64.powi(3) / 12.0;
        let uz_want = (100.0 * 0.1_f64.powi(3) / (3.0 * e * i) + 5.0 * 0.1_f64.powi(2) / (2.0 * e * i) + 100.0 * 0.1 / ((5.0 / 6.0) * g * 1e-4)) * 1e3;
        assert_rel(uz_got, uz_want, 0.025, "eccentric remote force + rigid remote displacement");
        assert_rel(reaction_on(&r, "Remote Displacement")[2], 100.0, 0.01, "remote displacement vertical reaction");
    }

    // 3. Spring + Point Mass (Modal frequency & Static Gravity)
    {
        let k_si: f64 = 40_000.0;
        let m_tot: f64 = 7850.0 * 8e-6 + 1.0;
        let f_want = (k_si / m_tot).sqrt() / (2.0 * std::f64::consts::PI);

        let c = new_core();
        call(&c, "add_box", json!({"name": "Block", "min": [0, 0, 0], "size": [20, 20, 20], "material": "Structural Steel"}));
        call(&c, "set_mesh", json!({"element_size_mm": 5.0}));
        call(&c, "add_displacement", json!({"faces": ["Block.xmin", "Block.xmax"], "x": 0.0}));
        call(&c, "add_displacement", json!({"faces": ["Block.ymin", "Block.ymax"], "y": 0.0}));
        call(&c, "add_spring", json!({"faces": ["Block.zmax"], "stiffness_n_per_mm": 40.0}));
        call(&c, "add_point_mass", json!({"faces": ["Block.zmin"], "mass_kg": 1.0}));
        call(&c, "set_analysis", json!({"type": "modal", "modes": 3}));
        let rm = solve(&c);
        let f1 = rm["modes"][0]["frequency_hz"].as_f64().unwrap();
        assert_rel(f1, f_want, 0.015, "spring-mass modal natural frequency with PointMass");

        call(&c, "set_analysis", json!({"type": "static_structural"}));
        call(&c, "add_gravity", json!({"acceleration": [0, 0, -9.80665]}));
        let rs = solve(&c);
        let uz = -rs["fields"]["deformation_z"]["min"].as_f64().unwrap();
        let sag_want = (m_tot * 9.80665 / k_si) * 1e3;
        assert_rel(uz, sag_want, 0.015, "point-mass static gravity deflection on elastic spring");
    }
}

#[test]
fn p2_4_mcp_undo_rename_roundtrip_and_validation() {
    let c = new_core();
    call(&c, "add_box", json!({"name": "A", "min": [0, 0, 0], "size": [10, 10, 10]}));
    call(&c, "add_box", json!({"name": "B", "min": [0, 0, 10], "size": [10, 10, 10]}));
    call(&c, "add_contact", json!({"body_a": "A", "body_b": "B", "kind": "frictional", "friction": 0.3, "offset_mm": 0.01}));
    call(&c, "add_spring", json!({"faces": ["A.zmax"], "to_faces": ["B.zmin"], "stiffness_n_per_mm": 50.0}));

    let m1 = call(&c, "get_model", json!({}));
    assert_eq!(m1["contacts"].as_array().unwrap().len(), 1);
    assert!(m1["contacts"][0]["summary"].as_str().unwrap().contains("μ=0.3"));

    call(&c, "edit_body", json!({"name": "A", "new_name": "Lower"}));
    let m2 = call(&c, "get_model", json!({}));
    assert_eq!(m2["contacts"][0]["definition"]["body_a"].as_str().unwrap(), "Lower");
    assert_eq!(m2["loads"][0]["definition"]["faces"][0].as_str().unwrap(), "Lower.zmax");

    let tmp = std::env::temp_dir().join("p2_4_roundtrip.simcraft");
    call(&c, "save_project", json!({"path": tmp}));
    call(&c, "undo", json!({}));
    assert_eq!(call(&c, "get_model", json!({}))["contacts"][0]["definition"]["body_a"].as_str().unwrap(), "A");
    call(&c, "open_project", json!({"path": tmp}));
    assert_eq!(call(&c, "get_model", json!({}))["contacts"][0]["definition"]["body_a"].as_str().unwrap(), "Lower");
    let _ = std::fs::remove_file(&tmp);

    assert!(c.call("add_contact", &json!({"body_a": "Lower", "body_b": "B", "kind": "frictional", "friction": -0.1})).is_err());
    assert!(c.call("add_spring", &json!({"faces": ["Lower.zmin"], "stiffness_n_per_mm": -5.0})).is_err());
    assert!(c.call("add_point_mass", &json!({"faces": ["B.zmax"], "mass_kg": 0.0})).is_err());
}
