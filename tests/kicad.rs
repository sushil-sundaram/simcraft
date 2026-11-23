//! KiCad import: parser coverage and homogenised copper conduction vs closed form.
use serde_json::{Value, json};
use simcraft::session::{Shared, new_core};

fn call(c: &Shared, t: &str, a: Value) -> Value {
    let o = c.call(t, &a).unwrap_or_else(|e| panic!("{t}: {e:#}"));
    serde_json::from_str(&o.text).unwrap_or(Value::String(o.text))
}
fn close(name: &str, got: f64, want: f64, tol: f64) {
    let err = (got - want).abs() / want.abs();
    println!("{name}: got {got:.6}, theory {want:.6}, error {:.2}%", err * 100.0);
    assert!(err < tol, "{name} off by {:.1}%", err * 100.0);
}
fn write(name: &str, body: &str) -> String {
    let p = std::env::temp_dir().join(format!("simcraft-test-{}-{name}", std::process::id()));
    std::fs::write(&p, body).unwrap();
    p.to_string_lossy().to_string()
}
fn heat(r: &Value, id_index: usize) -> f64 {
    r["reactions"][id_index]["heat_flow_w"].as_f64().unwrap()
}

const KD: f64 = 0.3; // FR-4 conductivity in the library
const KCU: f64 = 385.0;

/// 40 × 10 × 1.6 mm board, F.Cu fully filled (zone), B.Cu empty, optional via grid.
fn strip(vias: &str) -> String {
    format!(r#"(kicad_pcb (version 20240108) (generator "test")
  (general (thickness 1.6))
  (layers (0 "F.Cu" signal) (31 "B.Cu" signal) (44 "Edge.Cuts" user))
  (setup (stackup (layer "F.Cu" (type "copper") (thickness 0.035)) (layer "dielectric 1" (type "core") (thickness 1.53)) (layer "B.Cu" (type "copper") (thickness 0.035))))
  (gr_rect (start 100 50) (end 140 60) (layer "Edge.Cuts") (width 0.1))
  (zone (net 1) (net_name "GND") (layer "F.Cu") (filled_polygon (layer "F.Cu") (pts (xy 99 49) (xy 141 49) (xy 141 61) (xy 99 61))))
  {vias}
)"#)
}

#[test]
fn in_plane_conduction_of_copper_plane() {
    let c = new_core();
    call(&c, "import_kicad", json!({"path": write("strip.kicad_pcb", &strip("")), "components": false}));
    call(&c, "set_analysis", json!({"type": "steady_thermal"}));
    call(&c, "set_mesh", json!({"element_size_mm": 0.5}));
    call(&c, "add_temperature", json!({"faces": ["bbox.xmin"], "value": 10}));
    call(&c, "add_temperature", json!({"faces": ["bbox.xmax"], "value": 0}));
    let r = call(&c, "solve", json!({}));
    // k∥·t = k_d·(t − 2 t_cu) + k_cu·t_cu (F.Cu full) + k_d·t_cu (B.Cu empty)
    let kt = KD * (1.6 - 0.07) + KCU * 0.035 + KD * 0.035;
    let q_theory = kt * 1e-3 * 10e-3 * 10.0 / 40e-3; // (k·t)·W·ΔT/L
    close("in-plane heat flow", heat(&r, 0), q_theory, 0.015);
}

#[test]
fn through_plane_conduction_with_vias() {
    let mut v = String::new();
    let n = 40;
    for i in 0..n {
        v += &format!("(via (at {} {}) (size 0.6) (drill 0.3) (layers \"F.Cu\" \"B.Cu\") (net 1))\n", 101.0 + (i % 20) as f64 * 1.9, 52.5 + (i / 20) as f64 * 5.0);
    }
    let c = new_core();
    call(&c, "import_kicad", json!({"path": write("vias.kicad_pcb", &strip(&v)), "components": false}));
    call(&c, "set_analysis", json!({"type": "steady_thermal"}));
    call(&c, "set_mesh", json!({"element_size_mm": 0.5}));
    call(&c, "add_temperature", json!({"faces": ["Board.top"], "value": 10}));
    call(&c, "add_temperature", json!({"faces": ["Board.bottom"], "value": 0}));
    let r = call(&c, "solve", json!({}));
    // per column: series layers (F.Cu where the via ring is, B.Cu partly ring) ∥ via barrels.
    // B.Cu copper only at via rings (area fraction ≈ n·π·0.3²/A), F.Cu full.
    let a = 40.0 * 10.0;
    let ring = n as f64 * std::f64::consts::PI * 0.3 * 0.3 / a;
    let phi = n as f64 * std::f64::consts::PI * 0.3 * 0.025 / a;
    // columns with a ring on B.Cu are a tiny share: series resistance per unit area
    let r_plain = 1.53 / KD + 0.035 / KCU + 0.035 / KD;
    let r_ring = 1.53 / KD + 0.035 / KCU + 0.035 / KCU;
    let kz_series = 1.6 * ((1.0 - ring) / r_plain + ring / r_ring);
    let kz = (1.0 - phi) * kz_series + phi * KCU;
    let q_theory = kz * a * 1e-6 * 10.0 / 1.6e-3;
    close("through-plane heat flow (vias)", -heat(&r, 1), q_theory, 0.05);
    println!("via copper fraction {phi:.4}, k⊥ = {kz:.3} W/mK (laminate alone {KD})");
    assert!(kz > 3.0 * KD, "0.24 % via copper should roughly quadruple k⊥");
}

/// Parser coverage: rounded outline with a cutout, arc track, rotated footprint, bottom part.
#[test]
fn parses_outline_cutout_and_footprints() {
    let board = r#"(kicad_pcb (version 20241229) (generator "pcbnew")
  (general (thickness 1.2))
  (layers (0 "F.Cu" signal) (4 "In1.Cu" signal) (6 "In2.Cu" signal) (2 "B.Cu" signal) (25 "Edge.Cuts" user))
  (gr_line (start 2 0) (end 48 0) (layer "Edge.Cuts") (stroke (width 0.1)))
  (gr_arc (start 48 0) (mid 49.414 0.586) (end 50 2) (layer "Edge.Cuts") (stroke (width 0.1)))
  (gr_line (start 50 2) (end 50 30) (layer "Edge.Cuts"))
  (gr_line (start 50 30) (end 0 30) (layer "Edge.Cuts"))
  (gr_line (start 0 30) (end 0 2) (layer "Edge.Cuts"))
  (gr_arc (start 0 2) (mid 0.586 0.586) (end 2 0) (layer "Edge.Cuts"))
  (gr_circle (center 40 20) (end 43 20) (layer "Edge.Cuts"))
  (segment (start 5 5) (end 45 5) (width 0.5) (layer "F.Cu") (net 1))
  (arc (start 5 10) (mid 10 15) (end 15 10) (width 0.3) (layer "B.Cu") (net 2))
  (footprint "Package_SO:SOIC-8" (layer "F.Cu") (at 20 15 90)
    (property "Reference" "U7" (at 0 0 0) (layer "F.SilkS"))
    (property "Value" "LM358" (at 0 0 0) (layer "F.Fab"))
    (fp_rect (start -3.7 -2.7) (end 3.7 2.7) (layer "F.CrtYd"))
    (pad "1" smd roundrect (at -2.5 -1.9 90) (size 1.95 0.6) (layers "F.Cu" "F.Paste" "F.Mask"))
    (pad "8" smd roundrect (at 2.5 -1.9 90) (size 1.95 0.6) (layers "F.Cu" "F.Paste" "F.Mask")))
  (footprint "Resistor_THT:R_Axial" (layer "B.Cu") (at 10 25)
    (fp_text reference "R3" (at 0 0) (layer "B.SilkS"))
    (fp_text value "10k" (at 0 0) (layer "B.Fab"))
    (pad "1" thru_hole circle (at 0 0) (size 1.6 1.6) (drill 0.8) (layers "*.Cu" "*.Mask"))
    (pad "2" thru_hole oval (at 7.62 0) (size 1.6 2.4) (drill 0.8) (layers "*.Cu" "*.Mask")))
)"#;
    let d = simcraft::kicad::PcbData::from_text(board, "sample").unwrap();
    println!("{}", d.summary());
    assert_eq!(d.loops.len(), 2, "outline + round cutout");
    assert!((d.size[0] - 50.0).abs() < 0.01 && (d.size[1] - 30.0).abs() < 0.01);
    assert_eq!(d.layers.iter().map(|l| l.0.as_str()).collect::<Vec<_>>(), ["F.Cu", "In1.Cu", "In2.Cu", "B.Cu"]);
    assert!((d.thickness - 1.2).abs() < 1e-9);
    // cutout and rounded corner are outside the board
    // KiCad's circle cutout at (40, 20) lands at (40, 10) once y is flipped (board height 30)
    assert!(!d.inside(40.0, 10.0), "cutout");
    assert!(d.inside(30.0, 10.0) && d.inside(25.0, 15.0));
    // the rounded corners are KiCad's (0,0) and (50,0) → top-left/top-right after the flip
    assert!(!d.inside(0.1, 29.9) && !d.inside(49.9, 29.9), "rounded corners outside");
    assert!(d.inside(0.1, 0.1) && d.inside(49.9, 0.1), "square corners inside");
    let u7 = d.footprints.iter().find(|f| f.reference == "U7").unwrap();
    // courtyard 7.4 × 5.4 rotated by 90° → 5.4 wide in x, 7.4 in y
    assert!(((u7.hi[0] - u7.lo[0]) - 5.4).abs() < 1e-6 && ((u7.hi[1] - u7.lo[1]) - 7.4).abs() < 1e-6, "{u7:?}");
    assert!(!u7.bottom);
    let r3 = d.footprints.iter().find(|f| f.reference == "R3").unwrap();
    assert!(r3.bottom && r3.value == "10k");
    // track 40 × 0.5 mm on F.Cu ≈ 20 mm² of a 1471 mm² board (minus cutout) plus two pads
    let f = d.copper_fraction[0];
    assert!(f > 0.012 && f < 0.02, "F.Cu fraction {f}");
    // through-hole pads put copper on every layer
    assert!(d.copper_fraction[1] > 0.0 && d.copper_fraction[2] > 0.0);
}
