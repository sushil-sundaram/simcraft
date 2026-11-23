//! Editing after creation, undo/redo and body transforms.
use serde_json::{Value, json};
use simcraft::session::{Shared, new_core};

fn call(c: &Shared, t: &str, a: Value) -> Value {
    let o = c.call(t, &a).unwrap_or_else(|e| panic!("{t}: {e:#}"));
    serde_json::from_str(&o.text).unwrap_or(Value::String(o.text))
}
fn beam() -> Shared {
    let c = new_core();
    call(&c, "add_box", json!({"name": "Beam", "size": [100, 10, 8]}));
    call(&c, "set_mesh", json!({"element_size_mm": 2}));
    call(&c, "add_fixed_support", json!({"faces": ["Beam.xmin"]}));
    call(&c, "add_force", json!({"faces": ["Beam.xmax"], "vector": [0, 0, -100]}));
    c
}
fn tip(c: &Shared) -> f64 {
    call(c, "solve", json!({}))["fields"]["deformation_z"]["min"].as_f64().unwrap().abs()
}
fn size_z(c: &Shared) -> f64 {
    call(c, "get_model", json!({}))["bodies"][0]["shape"]["size"][2].as_f64().unwrap()
}

#[test]
fn edit_body_and_rename_keeps_loads() {
    let c = beam();
    let d8 = tip(&c);
    call(&c, "edit_body", json!({"name": "Beam", "size": [100, 10, 16], "new_name": "Bar"}));
    let m = call(&c, "get_model", json!({}));
    assert_eq!(m["bodies"][0]["name"], "Bar");
    let faces: Vec<String> = m["loads"].as_array().unwrap().iter().map(|l| l["definition"]["faces"][0].as_str().unwrap().to_string()).collect();
    assert_eq!(faces, ["Bar.xmin", "Bar.xmax"]);
    let d16 = tip(&c);
    println!("tip 8 mm {d8:.4}, 16 mm {d16:.4}, ratio {:.2} (theory 8)", d8 / d16);
    assert!((d8 / d16 - 8.0).abs() < 0.6);
    // unknown field is refused with the list of valid ones
    let e = c.call("edit_body", &json!({"name": "Bar", "radius": 3})).err().unwrap();
    assert!(format!("{e:#}").contains("size"));
}

#[test]
fn edit_load_changes_the_reaction() {
    let c = beam();
    let id = call(&c, "get_model", json!({}))["loads"][1]["id"].as_u64().unwrap();
    call(&c, "edit_load", json!({"id": id, "vector": [0, 0, -250]}));
    let r = call(&c, "solve", json!({}));
    let fz = r["reactions"][0]["force_n"][2].as_f64().unwrap();
    assert!((fz - 250.0).abs() < 1e-6, "reaction {fz}");
    assert!(c.call("edit_load", &json!({"id": id, "faces": ["Nope.xmax"]})).is_err(), "bad faces must be refused");
}

#[test]
fn undo_redo() {
    let c = beam();
    call(&c, "edit_body", json!({"name": "Beam", "size": [100, 10, 12]}));
    call(&c, "edit_body", json!({"name": "Beam", "size": [100, 10, 14]}));
    assert_eq!(size_z(&c), 14.0);
    call(&c, "undo", json!({}));
    assert_eq!(size_z(&c), 12.0);
    call(&c, "undo", json!({}));
    assert_eq!(size_z(&c), 8.0);
    call(&c, "redo", json!({}));
    assert_eq!(size_z(&c), 12.0);
    // a new change clears the redo stack
    call(&c, "set_material", json!({"material": "Aluminum 6061-T6"}));
    assert!(c.call("redo", &json!({})).is_err());
    // undo also reverts loads and analysis settings
    call(&c, "set_analysis", json!({"type": "modal"}));
    call(&c, "undo", json!({}));
    assert_eq!(call(&c, "get_model", json!({}))["analysis"]["type"], "static_structural");
    // display changes are not model changes
    let before = call(&c, "get_model", json!({}));
    call(&c, "set_display", json!({"dark_mode": true}));
    call(&c, "undo", json!({}));
    assert_ne!(call(&c, "get_model", json!({}))["bodies"][0]["material"], before["bodies"][0]["material"]);
}

#[test]
fn pattern_mirror_and_copy() {
    let c = new_core();
    call(&c, "add_box", json!({"name": "Fin", "min": [0, 0, 0], "size": [40, 2, 20]}));
    let r = call(&c, "transform_body", json!({"name": "Fin", "mode": "pattern", "translate": [0, 6, 0], "count": 4}));
    assert_eq!(r["bodies"].as_array().unwrap().len(), 3);
    let m = call(&c, "get_model", json!({}));
    let ys: Vec<f64> = m["bodies"].as_array().unwrap().iter().map(|b| b["shape"]["min"][1].as_f64().unwrap()).collect();
    assert_eq!(ys, [0.0, 6.0, 12.0, 18.0]);
    // mirror a box across x = 50: [0,40] → [60,100]
    call(&c, "transform_body", json!({"name": "Fin", "mode": "mirror", "axis": "x", "at": 50}));
    let m = call(&c, "get_model", json!({}));
    let mir = m["bodies"].as_array().unwrap().iter().find(|b| b["name"] == "Fin_mirror").unwrap();
    assert_eq!(mir["shape"]["min"][0].as_f64().unwrap(), 60.0);
    // move, then copy: mass doubles
    let c2 = new_core();
    call(&c2, "add_box", json!({"name": "A", "size": [10, 10, 10]}));
    call(&c2, "set_mesh", json!({"element_size_mm": 1}));
    call(&c2, "add_fixed_support", json!({"faces": ["A.zmin"]}));
    call(&c2, "add_gravity", json!({}));
    let m1 = call(&c2, "solve", json!({}))["info"]["mass_kg"].as_f64().unwrap();
    call(&c2, "transform_body", json!({"name": "A", "mode": "copy", "translate": [20, 0, 0]}));
    call(&c2, "add_fixed_support", json!({"faces": ["A_copy.zmin"]}));
    let m2 = call(&c2, "solve", json!({}))["info"]["mass_kg"].as_f64().unwrap();
    assert!((m2 / m1 - 2.0).abs() < 1e-9, "mass {m1} → {m2}");
    // cylinders along the mirror axis keep their extent
    call(&c2, "add_cylinder", json!({"name": "Pin", "base": [0, 0, 10], "axis": "z", "radius": 2, "height": 5}));
    call(&c2, "transform_body", json!({"name": "Pin", "mode": "mirror", "axis": "z", "at": 10, "copy": false}));
    let pin = call(&c2, "get_model", json!({}))["bodies"].as_array().unwrap().iter().find(|b| b["name"] == "Pin").unwrap().clone();
    assert_eq!(pin["shape"]["base"][2].as_f64().unwrap(), 5.0);
}
