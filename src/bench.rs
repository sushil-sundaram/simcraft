//! Benchmark suite: SimCraft against reference solutions (closed forms, NAFEMS, published
//! experiments and numerical benchmarks), each case at several mesh sizes, with an optional column
//! for Ansys (or any other solver) results. `simcraft bench` writes `bench.json` and a
//! self-contained `bench.html` (error-vs-unknowns plots).
//!
//! Every case drives the public tool API exactly like a user or an AI agent would.

use crate::material::{Material, library};
use crate::session::{Shared, new_core};
use anyhow::{Result, anyhow, bail};
use serde_json::{Value, json};
use std::time::Instant;

pub struct Metric {
    pub id: &'static str,
    pub title: &'static str,
    pub reference: f64,
    pub unit: &'static str,
    /// relative tolerance at the finest size (absolute when `absolute`)
    pub tol: f64,
    pub absolute: bool,
}

pub struct Case {
    pub id: &'static str,
    pub group: &'static str,
    pub title: &'static str,
    pub source: &'static str,
    /// mesh sizes, coarse → fine (element size in mm, or cells across for `cells` cases)
    pub sizes: Vec<f64>,
    pub size_unit: &'static str,
    /// cheap enough for every pull request
    pub quick: bool,
    pub metrics: Vec<Metric>,
    pub run: Box<dyn Fn(f64) -> Result<Outcome> + Send + Sync>,
}

pub struct Outcome {
    pub values: Vec<f64>,
    /// the last solve's `info` (dofs, elements, solver …)
    pub info: Value,
}

fn m(id: &'static str, title: &'static str, reference: f64, unit: &'static str, tol: f64) -> Metric {
    Metric { id, title, reference, unit, tol, absolute: false }
}

fn mat(name: &str) -> Material {
    library().into_iter().find(|x| x.name == name).unwrap_or_else(|| panic!("material {name} missing"))
}

// ------------------------------------------------------------------ tool helpers

fn call(c: &Shared, t: &str, a: Value) -> Result<Value> {
    let o = c.call(t, &a).map_err(|e| anyhow!("{t}: {e:#}"))?;
    Ok(serde_json::from_str(&o.text).unwrap_or(Value::String(o.text)))
}
fn script(c: &Shared, steps: &[(&str, Value)]) -> Result<Value> {
    let mut last = Value::Null;
    for (t, a) in steps {
        last = call(c, t, a.clone())?;
    }
    Ok(last)
}
fn qty(r: &Value, name: &str) -> Result<f64> {
    r["quantities"].as_array().and_then(|a| a.iter().find(|x| x["name"].as_str().is_some_and(|n| n.starts_with(name)))).and_then(|x| x["value"].as_f64()).ok_or_else(|| anyhow!("no quantity '{name}'"))
}
fn field(r: &Value, f: &str, which: &str) -> Result<f64> {
    r["fields"][f][which].as_f64().ok_or_else(|| anyhow!("no field {f}.{which}"))
}
fn probe(c: &Shared, p: [f64; 3], f: &str) -> Result<f64> {
    call(c, "probe", json!({"point": p}))?["values"][f].as_f64().ok_or_else(|| anyhow!("probe: no {f}"))
}
fn info(r: &Value) -> Value {
    r["info"].clone()
}
fn out(values: Vec<f64>, r: &Value) -> Result<Outcome> {
    Ok(Outcome { values, info: info(r) })
}

/// ASCII STL of a closed solid from triangles, each oriented by the outward direction given.
fn stl(tris: &[([f64; 3], [f64; 3], [f64; 3], [f64; 3])]) -> String {
    let mut s = String::from("solid bench\n");
    for (a, b, c, want) in tris {
        let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
        let n = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
        let (b, c) = if n[0] * want[0] + n[1] * want[1] + n[2] * want[2] < 0.0 { (c, b) } else { (b, c) };
        s += &format!("facet normal 0 0 0\nouter loop\nvertex {} {} {}\nvertex {} {} {}\nvertex {} {} {}\nendloop\nendfacet\n", a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2]);
    }
    s + "endsolid bench\n"
}

fn temp_file(name: &str, body: &str) -> Result<String> {
    let p = std::env::temp_dir().join(format!("simcraft-bench-{}-{name}", std::process::id()));
    std::fs::write(&p, body)?;
    Ok(p.to_string_lossy().to_string())
}

const NU_AIR: f64 = 1.81e-5 / 1.204;
const EPS0: f64 = 8.854_187_8e-12;

// ------------------------------------------------------------------ the cases

pub fn cases() -> Vec<Case> {
    let steel = mat("Structural Steel");
    let al = mat("Aluminum 6061-T6");
    let cu = mat("Copper C101");
    let ptfe = mat("PTFE (Teflon)");
    let pi = std::f64::consts::PI;
    let mut v: Vec<Case> = Vec::new();

    // ---------------------------------------------------------------- structural
    {
        let (e, nu) = (steel.youngs_modulus_gpa * 1e3, steel.poisson_ratio);
        let (l, b, h, f) = (120.0f64, 12.0f64, 10.0f64, 500.0f64);
        let want = f * l.powi(3) / (3.0 * e * b * h.powi(3) / 12.0) + f * l / (5.0 / 6.0 * e / (2.0 * (1.0 + nu)) * b * h);
        v.push(Case {
            id: "cantilever",
            group: "Structural",
            title: "Cantilever tip deflection (120 × 12 × 10 mm steel, 500 N)",
            source: "Timoshenko beam theory: FL³/3EI + FL/κGA",
            sizes: vec![2.0, 1.0, 0.5],
            size_unit: "mm",
            quick: true,
            metrics: vec![m("cantilever.tip", "tip deflection", want, "mm", 0.02)],
            run: Box::new(|hh| {
                let c = new_core();
                let r = script(&c, &[
                    ("add_box", json!({"name": "Beam", "size": [120, 12, 10], "material": "Structural Steel"})),
                    ("set_mesh", json!({"element_size_mm": hh})),
                    ("add_fixed_support", json!({"faces": ["Beam.xmin"]})),
                    ("add_force", json!({"faces": ["Beam.xmax"], "vector": [0, 0, -500]})),
                    ("solve", json!({})),
                ])?;
                out(vec![-field(&r, "deformation_z", "min")?], &r)
            }),
        });
    }
    v.push(Case {
        id: "kirsch",
        group: "Structural",
        title: "Strip with a hole in tension, d/W = 0.1: stress concentration",
        source: "Heywood / Howland: Kt(net) = 2 + (1 − d/W)³, on the gross section ÷ (1 − d/W)",
        sizes: vec![2.0, 1.5, 1.0],
        size_unit: "mm",
        quick: true,
        metrics: vec![m("kirsch.kt", "Kt (gross section)", (2.0 + 0.9f64.powi(3)) / 0.9, "", 0.03)],
        run: Box::new(|hh| {
            let c = new_core();
            let r = script(&c, &[
                ("add_box", json!({"name": "Plate", "min": [0, 0, 0], "size": [300, 100, 2]})),
                ("add_cylinder", json!({"name": "Hole", "base": [0, 0, 0], "axis": "z", "radius": 10, "height": 2, "operation": "subtract"})),
                ("set_mesh", json!({"element_size_mm": hh})),
                ("add_symmetry", json!({"faces": ["Plate.xmin", "Plate.ymin", "Plate.zmin"]})),
                ("add_pressure", json!({"faces": ["Plate.xmax"], "value_mpa": -10})),
                ("solve", json!({})),
            ])?;
            out(vec![field(&r, "max_principal", "max")? / 10.0], &r)
        }),
    });
    v.push(Case {
        id: "kirsch_octree",
        group: "Structural",
        title: "Strip with a hole, octree mesh refined 4× at the hole only (P2.5)",
        source: "Heywood / Howland: Kt(net) = 2 + (1 − d/W)³, on the gross section ÷ (1 − d/W)",
        sizes: vec![2.0, 1.0],
        size_unit: "mm",
        quick: true,
        metrics: vec![m("kirsch_octree.kt", "Kt (gross section)", (2.0 + 0.9f64.powi(3)) / 0.9, "", 0.03)],
        run: Box::new(|hh| {
            let c = new_core();
            let r = script(&c, &[
                ("add_box", json!({"name": "Plate", "min": [0, 0, 0], "size": [300, 100, 2]})),
                ("add_cylinder", json!({"name": "Hole", "base": [0, 0, 0], "axis": "z", "radius": 10, "height": 2, "operation": "subtract"})),
                ("set_mesh", json!({"element_size_mm": hh, "refinements": [{"faces": ["Hole.side"], "size_mm": hh / 4.0}]})),
                ("add_symmetry", json!({"faces": ["Plate.xmin", "Plate.ymin", "Plate.zmin"]})),
                ("add_pressure", json!({"faces": ["Plate.xmax"], "value_mpa": -10})),
                ("solve", json!({})),
            ])?;
            out(vec![field(&r, "max_principal", "max")? / 10.0], &r)
        }),
    });
    v.push(Case {
        id: "kirsch_adaptive",
        group: "Structural",
        title: "Strip with a hole, adaptive Zienkiewicz–Zhu refinement to 1 % error (P2.5)",
        source: "Heywood / Howland: Kt(net) = 2 + (1 − d/W)³, on the gross section ÷ (1 − d/W)",
        sizes: vec![2.0],
        size_unit: "mm",
        quick: false,
        metrics: vec![m("kirsch_adaptive.kt", "Kt (gross section)", (2.0 + 0.9f64.powi(3)) / 0.9, "", 0.03)],
        run: Box::new(|hh| {
            let c = new_core();
            let r = script(&c, &[
                ("add_box", json!({"name": "Plate", "min": [0, 0, 0], "size": [300, 100, 2]})),
                ("add_cylinder", json!({"name": "Hole", "base": [0, 0, 0], "axis": "z", "radius": 10, "height": 2, "operation": "subtract"})),
                ("set_mesh", json!({"element_size_mm": hh, "adaptive_passes": 4, "adaptive_error_pct": 1.0})),
                ("add_symmetry", json!({"faces": ["Plate.xmin", "Plate.ymin", "Plate.zmin"]})),
                ("add_pressure", json!({"faces": ["Plate.xmax"], "value_mpa": -10})),
                ("solve", json!({})),
            ])?;
            out(vec![field(&r, "max_principal", "max")? / 10.0], &r)
        }),
    });
    v.push(Case {
        id: "step_kirsch",
        group: "Structural",
        title: "STEP B-rep strip with circular hole in tension, d/W = 0.1: stress concentration",
        source: "Heywood / Howland: Kt(net) = 2 + (1 − d/W)³, imported via AP214 STEP B-rep + cylindrical hole fitting",
        sizes: vec![2.0, 1.5],
        size_unit: "mm",
        quick: true,
        metrics: vec![m("step_kirsch.kt", "Kt (gross section)", (2.0 + 0.9f64.powi(3)) / 0.9, "", 0.03)],
        run: Box::new(|hh| {
            let step_txt = r#"ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('Quarter plate with hole'),'2;1');
FILE_NAME('kirsch.step','2026-01-01T00:00:00',(''),(''),'','','');
FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));
ENDSEC;
DATA;
#1 = APPLICATION_CONTEXT('automotive_design');
#2 = ( GEOMETRIC_REPRESENTATION_CONTEXT(3) GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT((#6)) GLOBAL_UNIT_ASSIGNED_CONTEXT((#3,#4,#5)) REPRESENTATION_CONTEXT('','') );
#3 = ( LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.) );
#4 = ( NAMED_UNIT(*) PLANE_ANGLE_UNIT() SI_UNIT($,.RADIAN.) );
#5 = ( NAMED_UNIT(*) SI_UNIT($,.STERADIAN.) SOLID_ANGLE_UNIT() );
#6 = UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE(1.0E-6),#3,'','');
#7 = APPLICATION_PROTOCOL_DEFINITION('international standard','automotive_design',2000,#1);
#8 = PRODUCT_CONTEXT('',#1,'mechanical');
#9 = PRODUCT_DEFINITION_CONTEXT('part definition',#1,'design');
#10 = PRODUCT('KirschPlate','KirschPlate','',(#8));
#11 = PRODUCT_DEFINITION_FORMATION('','',#10);
#12 = PRODUCT_DEFINITION('design','',#11,#9);
#13 = PRODUCT_DEFINITION_SHAPE('','',#12);
#14 = SHAPE_DEFINITION_REPRESENTATION(#13,#15);
#15 = ADVANCED_BREP_SHAPE_REPRESENTATION('',(#16),#2);
#16 = MANIFOLD_SOLID_BREP('KirschPlate',#17);
#17 = CLOSED_SHELL('',(#300,#301,#302,#303,#304,#305,#306));
#100 = CARTESIAN_POINT('',(0.,0.,0.));
#101 = CARTESIAN_POINT('',(0.,0.,2.));
#102 = CARTESIAN_POINT('',(10.,0.,0.));
#103 = CARTESIAN_POINT('',(300.,0.,0.));
#104 = CARTESIAN_POINT('',(300.,100.,0.));
#105 = CARTESIAN_POINT('',(0.,100.,0.));
#106 = CARTESIAN_POINT('',(0.,10.,0.));
#107 = CARTESIAN_POINT('',(10.,0.,2.));
#108 = CARTESIAN_POINT('',(300.,0.,2.));
#109 = CARTESIAN_POINT('',(300.,100.,2.));
#110 = CARTESIAN_POINT('',(0.,100.,2.));
#111 = CARTESIAN_POINT('',(0.,10.,2.));
#120 = DIRECTION('',(0.,0.,1.));
#121 = DIRECTION('',(0.,0.,-1.));
#122 = DIRECTION('',(1.,0.,0.));
#123 = DIRECTION('',(0.,1.,0.));
#124 = DIRECTION('',(-1.,0.,0.));
#125 = DIRECTION('',(0.,-1.,0.));
#130 = AXIS2_PLACEMENT_3D('',#100,#120,#122);
#131 = AXIS2_PLACEMENT_3D('',#101,#120,#122);
#132 = AXIS2_PLACEMENT_3D('',#100,#121,#122);
#133 = AXIS2_PLACEMENT_3D('',#102,#125,#122);
#134 = AXIS2_PLACEMENT_3D('',#103,#122,#123);
#135 = AXIS2_PLACEMENT_3D('',#104,#123,#124);
#136 = AXIS2_PLACEMENT_3D('',#105,#124,#125);
#140 = VERTEX_POINT('',#102);
#141 = VERTEX_POINT('',#103);
#142 = VERTEX_POINT('',#104);
#143 = VERTEX_POINT('',#105);
#144 = VERTEX_POINT('',#106);
#145 = VERTEX_POINT('',#107);
#146 = VERTEX_POINT('',#108);
#147 = VERTEX_POINT('',#109);
#148 = VERTEX_POINT('',#110);
#149 = VERTEX_POINT('',#111);
#150 = VECTOR('',#122,1.);
#151 = VECTOR('',#123,1.);
#152 = VECTOR('',#124,1.);
#153 = VECTOR('',#125,1.);
#154 = VECTOR('',#120,1.);
#160 = LINE('',#102,#150);
#161 = LINE('',#103,#151);
#162 = LINE('',#104,#152);
#163 = LINE('',#105,#153);
#164 = CIRCLE('',#130,10.);
#170 = LINE('',#107,#150);
#171 = LINE('',#108,#151);
#172 = LINE('',#109,#152);
#173 = LINE('',#110,#153);
#174 = CIRCLE('',#131,10.);
#180 = LINE('',#102,#154);
#181 = LINE('',#103,#154);
#182 = LINE('',#104,#154);
#183 = LINE('',#105,#154);
#184 = LINE('',#106,#154);
#200 = EDGE_CURVE('',#140,#141,#160,.T.);
#201 = EDGE_CURVE('',#141,#142,#161,.T.);
#202 = EDGE_CURVE('',#142,#143,#162,.T.);
#203 = EDGE_CURVE('',#143,#144,#163,.T.);
#204 = EDGE_CURVE('',#140,#144,#164,.T.);
#210 = EDGE_CURVE('',#145,#146,#170,.T.);
#211 = EDGE_CURVE('',#146,#147,#171,.T.);
#212 = EDGE_CURVE('',#147,#148,#172,.T.);
#213 = EDGE_CURVE('',#148,#149,#173,.T.);
#214 = EDGE_CURVE('',#145,#149,#174,.T.);
#220 = EDGE_CURVE('',#140,#145,#180,.T.);
#221 = EDGE_CURVE('',#141,#146,#181,.T.);
#222 = EDGE_CURVE('',#142,#147,#182,.T.);
#223 = EDGE_CURVE('',#143,#148,#183,.T.);
#224 = EDGE_CURVE('',#144,#149,#184,.T.);
#300 = ADVANCED_FACE('',(#310),#400,.T.);
#310 = FACE_OUTER_BOUND('',#320,.T.);
#320 = EDGE_LOOP('',(#330,#331,#332,#333,#334));
#330 = ORIENTED_EDGE('',*,*,#200,.F.);
#331 = ORIENTED_EDGE('',*,*,#204,.T.);
#332 = ORIENTED_EDGE('',*,*,#203,.F.);
#333 = ORIENTED_EDGE('',*,*,#202,.F.);
#334 = ORIENTED_EDGE('',*,*,#201,.F.);
#400 = PLANE('',#132);
#301 = ADVANCED_FACE('',(#311),#401,.T.);
#311 = FACE_OUTER_BOUND('',#321,.T.);
#321 = EDGE_LOOP('',(#335,#336,#337,#338,#339));
#335 = ORIENTED_EDGE('',*,*,#210,.T.);
#336 = ORIENTED_EDGE('',*,*,#211,.T.);
#337 = ORIENTED_EDGE('',*,*,#212,.T.);
#338 = ORIENTED_EDGE('',*,*,#213,.T.);
#339 = ORIENTED_EDGE('',*,*,#214,.F.);
#401 = PLANE('',#131);
#302 = ADVANCED_FACE('',(#312),#402,.T.);
#312 = FACE_OUTER_BOUND('',#322,.T.);
#322 = EDGE_LOOP('',(#340,#341,#342,#343));
#340 = ORIENTED_EDGE('',*,*,#200,.T.);
#341 = ORIENTED_EDGE('',*,*,#221,.T.);
#342 = ORIENTED_EDGE('',*,*,#210,.F.);
#343 = ORIENTED_EDGE('',*,*,#220,.F.);
#402 = PLANE('',#133);
#303 = ADVANCED_FACE('',(#313),#403,.T.);
#313 = FACE_OUTER_BOUND('',#323,.T.);
#323 = EDGE_LOOP('',(#344,#345,#346,#347));
#344 = ORIENTED_EDGE('',*,*,#201,.T.);
#345 = ORIENTED_EDGE('',*,*,#222,.T.);
#346 = ORIENTED_EDGE('',*,*,#211,.F.);
#347 = ORIENTED_EDGE('',*,*,#221,.F.);
#403 = PLANE('',#134);
#304 = ADVANCED_FACE('',(#314),#404,.T.);
#314 = FACE_OUTER_BOUND('',#324,.T.);
#324 = EDGE_LOOP('',(#348,#349,#350,#351));
#348 = ORIENTED_EDGE('',*,*,#202,.T.);
#349 = ORIENTED_EDGE('',*,*,#223,.T.);
#350 = ORIENTED_EDGE('',*,*,#212,.F.);
#351 = ORIENTED_EDGE('',*,*,#222,.F.);
#404 = PLANE('',#135);
#305 = ADVANCED_FACE('',(#315),#405,.T.);
#315 = FACE_OUTER_BOUND('',#325,.T.);
#325 = EDGE_LOOP('',(#352,#353,#354,#355));
#352 = ORIENTED_EDGE('',*,*,#203,.T.);
#353 = ORIENTED_EDGE('',*,*,#224,.T.);
#354 = ORIENTED_EDGE('',*,*,#213,.F.);
#355 = ORIENTED_EDGE('',*,*,#223,.F.);
#405 = PLANE('',#136);
#306 = ADVANCED_FACE('',(#316),#406,.F.);
#316 = FACE_OUTER_BOUND('',#326,.T.);
#326 = EDGE_LOOP('',(#356,#357,#358,#359));
#356 = ORIENTED_EDGE('',*,*,#204,.F.);
#357 = ORIENTED_EDGE('',*,*,#220,.T.);
#358 = ORIENTED_EDGE('',*,*,#214,.T.);
#359 = ORIENTED_EDGE('',*,*,#224,.F.);
#406 = CYLINDRICAL_SURFACE('',#130,10.);
ENDSEC;
END-ISO-10303-21;
"#;
            let path = temp_file("kirsch_step.step", step_txt)?;
            let c = new_core();
            call(&c, "import_step", json!({"path": path, "name": "Plate"}))?;
            call(&c, "set_material", json!({"material": "Structural Steel"}))?;
            call(&c, "set_mesh", json!({"element_size_mm": hh}))?;
            call(&c, "add_symmetry", json!({"faces": ["bbox.xmin", "bbox.ymin", "bbox.zmin"]}))?;
            call(&c, "add_pressure", json!({"faces": ["bbox.xmax"], "value_mpa": -10}))?;
            let r = call(&c, "solve", json!({}))?;
            let _ = std::fs::remove_file(&path);
            out(vec![field(&r, "max_principal", "max")? / 10.0], &r)
        }),
    });
    v.push(Case {
        id: "lame",
        group: "Structural",
        title: "Thick cylinder under 10 MPa internal pressure: bore hoop stress",
        source: "Lamé: σθ = p (ro² + ri²) / (ro² − ri²)",
        sizes: vec![2.0, 1.5, 1.0],
        size_unit: "mm",
        quick: true,
        metrics: vec![m("lame.hoop", "bore hoop stress", 10.0 * 500.0 / 300.0, "MPa", 0.03)],
        run: Box::new(|hh| {
            let c = new_core();
            let r = script(&c, &[
                ("add_cylinder", json!({"name": "Pipe", "base": [0, 0, 0], "axis": "z", "radius": 20, "height": 4})),
                ("add_cylinder", json!({"name": "Bore", "base": [0, 0, 0], "axis": "z", "radius": 10, "height": 4, "operation": "subtract"})),
                ("add_box", json!({"name": "CutX", "min": [-30, -30, 0], "size": [30, 60, 4], "operation": "subtract"})),
                ("add_box", json!({"name": "CutY", "min": [-30, -30, 0], "size": [60, 30, 4], "operation": "subtract"})),
                ("set_mesh", json!({"element_size_mm": hh})),
                ("add_symmetry", json!({"faces": ["CutX.xmax", "CutY.ymax", "Pipe.bottom", "Pipe.top"]})),
                ("add_pressure", json!({"faces": ["Bore.side"], "value_mpa": 10})),
                ("solve", json!({})),
            ])?;
            out(vec![field(&r, "max_principal", "max")?], &r)
        }),
    });
    {
        let c45 = std::f64::consts::FRAC_1_SQRT_2;
        let p0 = 10.0 * 100.0 / 300.0;
        let hoop12 = p0 * (1.0 + 400.0 / 144.0);
        let rad14 = p0 * (1.0 - 400.0 / 196.0);
        v.push(Case {
            id: "lame_path",
            group: "Structural",
            title: "Thick cylinder 45° diagonal path: cylindrical hoop (r=12) and radial (r=14) stress",
            source: "Lamé: σθ = A(1 + ro²/r²), σr = A(1 − ro²/r²) along rotated path",
            sizes: vec![2.0, 1.5, 1.0],
            size_unit: "mm",
            quick: true,
            metrics: vec![
                m("lame_path.hoop12", "hoop stress at r = 12 mm (45°)", hoop12, "MPa", 0.02),
                m("lame_path.rad14", "radial stress at r = 14 mm (45°)", rad14, "MPa", 0.03),
            ],
            run: Box::new(move |hh| {
                let c = new_core();
                let r = script(&c, &[
                    ("add_cylinder", json!({"name": "Pipe", "base": [0, 0, 0], "axis": "z", "radius": 20, "height": 4})),
                    ("add_cylinder", json!({"name": "Bore", "base": [0, 0, 0], "axis": "z", "radius": 10, "height": 4, "operation": "subtract"})),
                    ("add_box", json!({"name": "CutX", "min": [-30, -30, 0], "size": [30, 60, 4], "operation": "subtract"})),
                    ("add_box", json!({"name": "CutY", "min": [-30, -30, 0], "size": [60, 30, 4], "operation": "subtract"})),
                    ("set_mesh", json!({"element_size_mm": hh})),
                    ("add_symmetry", json!({"faces": ["CutX.xmax", "CutY.ymax", "Pipe.bottom", "Pipe.top"]})),
                    ("add_pressure", json!({"faces": ["Bore.side"], "value_mpa": 10})),
                    ("solve", json!({})),
                ])?;
                let hoop = call(&c, "plot_path", json!({"field": "hoop_stress", "from": [12.0 * c45, 12.0 * c45, 2.0], "to": [18.0 * c45, 18.0 * c45, 2.0], "samples": 7, "cylindrical_origin": [0, 0, 0], "cylindrical_axis": "z"}))?;
                let rad = call(&c, "plot_path", json!({"field": "radial_stress", "from": [12.0 * c45, 12.0 * c45, 2.0], "to": [18.0 * c45, 18.0 * c45, 2.0], "samples": 7, "cylindrical_origin": [0, 0, 0], "cylindrical_axis": "z"}))?;
                let h12 = hoop["points"][0]["value"].as_f64().ok_or_else(|| anyhow!("missing r=12 hoop"))?;
                let r14 = rad["points"][2]["value"].as_f64().ok_or_else(|| anyhow!("missing r=14 radial"))?;
                out(vec![h12, r14], &r)
            }),
        });
    }
    {
        let (e, nu) = (steel.youngs_modulus_gpa * 1e3, steel.poisson_ratio);
        let (l, b, h, f) = (240.0f64, 12.0f64, 12.0f64, 200.0f64);
        let want = f * l.powi(3) / (3.0 * e * b * h.powi(3) / 12.0) + f * l / (5.0 / 6.0 * e / (2.0 * (1.0 + nu)) * b * h);
        v.push(Case {
            id: "cantilever_conv",
            group: "Structural",
            title: "Mesh-convergence Richardson extrapolation (240 × 12 × 12 mm cantilever from 3 coarse meshes)",
            source: "Timoshenko beam theory + 3-grid Richardson extrapolation (h, h/1.4, h/2)",
            sizes: vec![6.0, 4.0],
            size_unit: "mm",
            quick: true,
            metrics: vec![m("cantilever_conv.ext", "extrapolated tip deflection", want, "mm", 0.005)],
            run: Box::new(|hh| {
                let c = new_core();
                let rep = script(&c, &[
                    ("add_box", json!({"name": "Beam", "size": [240, 12, 12], "material": "Structural Steel"})),
                    ("set_mesh", json!({"element_size_mm": hh})),
                    ("add_fixed_support", json!({"faces": ["Beam.xmin"]})),
                    ("add_force", json!({"faces": ["Beam.xmax"], "vector": [0, 0, -200]})),
                    ("mesh_convergence", json!({"responses": ["min:deformation_z"], "base_element_size_mm": hh, "target_error_pct": 0.5})),
                ])?;
                let ext = -rep["convergence"][0]["extrapolated"].as_f64().ok_or_else(|| anyhow!("missing extrapolated deflection"))?;
                let fine = &rep["convergence_steps"][2];
                Ok(Outcome { values: vec![ext], info: json!({"dofs": fine["dofs"], "elements": fine["elements"], "solver": "Richardson extrapolation (3 coarse meshes)"}) })
            }),
        });
    }
    v.push(Case {
        id: "nafems_le1",
        group: "Structural",
        title: "NAFEMS LE1: elliptic membrane under outward pressure, σyy at D",
        source: "NAFEMS LE1 (The Standard NAFEMS Benchmarks): σyy(D) = 92.7 MPa",
        sizes: vec![50.0, 35.0, 25.0],
        size_unit: "mm",
        quick: false,
        metrics: vec![m("le1.syy_d", "σyy at D (max principal)", 92.7, "MPa", 0.05)],
        run: Box::new(|hh| {
            // quarter elliptic annulus: inner (x/2000)²+(y/1000)² = 1, outer (x/3250)²+(y/2750)² = 1, 100 mm thick
            let n = 120;
            let pt = |a: f64, b: f64, t: f64, z: f64| [a * t.cos(), b * t.sin(), z];
            let mut tris = Vec::new();
            for i in 0..n {
                let (t0, t1) = (std::f64::consts::FRAC_PI_2 * i as f64 / n as f64, std::f64::consts::FRAC_PI_2 * (i + 1) as f64 / n as f64);
                let (i0, i1, o0, o1) = (pt(2000.0, 1000.0, t0, 0.0), pt(2000.0, 1000.0, t1, 0.0), pt(3250.0, 2750.0, t0, 0.0), pt(3250.0, 2750.0, t1, 0.0));
                let up = |p: [f64; 3]| [p[0], p[1], 100.0];
                for (zf, want) in [(false, [0.0, 0.0, -1.0]), (true, [0.0, 0.0, 1.0])] {
                    let f = |p: [f64; 3]| if zf { up(p) } else { p };
                    tris.push((f(i0), f(o0), f(o1), want));
                    tris.push((f(i0), f(o1), f(i1), want));
                }
                let tm = 0.5 * (t0 + t1);
                let on = [tm.cos() / 3250.0, tm.sin() / 2750.0, 0.0];
                tris.push((o0, o1, up(o1), on));
                tris.push((o0, up(o1), up(o0), on));
                let inn = [-tm.cos() / 2000.0, -tm.sin() / 1000.0, 0.0];
                tris.push((i0, i1, up(i1), inn));
                tris.push((i0, up(i1), up(i0), inn));
            }
            let (a, b) = (pt(2000.0, 1000.0, 0.0, 0.0), pt(3250.0, 2750.0, 0.0, 0.0));
            tris.push((a, b, [b[0], b[1], 100.0], [0.0, -1.0, 0.0]));
            tris.push((a, [b[0], b[1], 100.0], [a[0], a[1], 100.0], [0.0, -1.0, 0.0]));
            let (a, b) = ([0.0, 1000.0, 0.0], [0.0, 2750.0, 0.0]);
            tris.push((a, b, [b[0], b[1], 100.0], [-1.0, 0.0, 0.0]));
            tris.push((a, [b[0], b[1], 100.0], [a[0], a[1], 100.0], [-1.0, 0.0, 0.0]));
            let path = temp_file("le1.stl", &stl(&tris))?;
            let c = new_core();
            call(&c, "import_stl", json!({"path": path, "name": "LE1"}))?;
            call(&c, "set_material", json!({"material": "Structural Steel"}))?;
            call(&c, "set_mesh", json!({"element_size_mm": hh}))?;
            let faces = call(&c, "list_faces", json!({}))?;
            let outer = faces["faces"]
                .as_array()
                .and_then(|a| a.iter().max_by(|p, q| {
                    let r = |x: &Value| x["centroid"][0].as_f64().unwrap_or(0.0).hypot(x["centroid"][1].as_f64().unwrap_or(0.0));
                    r(p).total_cmp(&r(q))
                }))
                .and_then(|x| x["name"].as_str())
                .ok_or_else(|| anyhow!("LE1: no outer face"))?
                .to_string();
            call(&c, "add_symmetry", json!({"faces": ["bbox.xmin", "bbox.ymin", "bbox.zmin"]}))?;
            call(&c, "add_pressure", json!({"faces": [outer], "value_mpa": -10}))?;
            let r = call(&c, "solve", json!({}))?;
            let _ = std::fs::remove_file(&path);
            out(vec![probe(&c, [2000.0, 0.0, 50.0], "max_principal")?], &r)
        }),
    });
    {
        let (e, al_t) = (steel.youngs_modulus_gpa * 1e3, steel.thermal_expansion);
        v.push(Case {
            id: "thermal_stress",
            group: "Structural",
            title: "Bar held between rigid walls, heated by 100 K: axial stress",
            source: "σ = −E α ΔT",
            sizes: vec![5.0, 2.5],
            size_unit: "mm",
            quick: true,
            metrics: vec![m("thermal_stress.sxx", "axial stress (min principal)", -e * al_t * 100.0, "MPa", 0.01)],
            run: Box::new(|hh| {
                let c = new_core();
                let r = script(&c, &[
                    ("add_box", json!({"name": "Bar", "size": [100, 10, 10], "material": "Structural Steel"})),
                    ("set_mesh", json!({"element_size_mm": hh})),
                    ("set_analysis", json!({"type": "thermal_stress", "reference_temperature": 22})),
                    ("add_temperature", json!({"faces": ["all"], "value": 122})),
                    ("add_symmetry", json!({"faces": ["Bar.xmin", "Bar.xmax", "Bar.ymin", "Bar.zmin"]})),
                    ("solve", json!({})),
                ])?;
                out(vec![field(&r, "min_principal", "min")?], &r)
            }),
        });
    }
    {
        let (e, rho) = (steel.youngs_modulus_gpa * 1e9, steel.density);
        let rr = 3.3e-3;
        let f1 = 1.8751f64.powi(2) / std::f64::consts::TAU * (e * rr * rr / 4.0 / rho).sqrt() / 0.08f64.powi(2);
        v.push(Case {
            id: "rod_modal",
            group: "Structural",
            title: "Cantilever round rod (r 3.3, L 80 mm): first bending frequency",
            source: "Euler–Bernoulli: f₁ = (1.8751²/2π)·√(EI/ρA)/L²",
            sizes: vec![1.5, 1.0, 0.75],
            size_unit: "mm",
            quick: true,
            metrics: vec![m("rod_modal.f1", "first frequency", f1, "Hz", 0.02)],
            run: Box::new(|hh| {
                let c = new_core();
                let r = script(&c, &[
                    ("add_cylinder", json!({"name": "Rod", "base": [0, 0, 0], "axis": "x", "radius": 3.3, "height": 80, "material": "Structural Steel"})),
                    ("set_mesh", json!({"element_size_mm": hh})),
                    ("set_analysis", json!({"type": "modal", "modes": 2})),
                    ("add_fixed_support", json!({"faces": ["Rod.bottom"]})),
                    ("solve", json!({})),
                ])?;
                out(vec![r["modes"][0]["frequency_hz"].as_f64().ok_or_else(|| anyhow!("no modes"))?], &r)
            }),
        });
    }
    v.push(Case {
        id: "sliding_friction",
        group: "Structural",
        title: "Sliding block on flat plane with Coulomb friction μ = 0.25",
        source: "Amontons–Coulomb friction: F = μ N = 0.25 × 800 N = 200 N",
        sizes: vec![5.0, 3.33],
        size_unit: "mm",
        quick: true,
        metrics: vec![m("sliding_friction.ft", "tangential friction force", 200.0, "N", 0.02)],
        run: Box::new(|hh| {
            let c = new_core();
            let r = script(&c, &[
                ("add_box", json!({"name": "Plane", "min": [0, 0, 0], "size": [60, 30, 10], "material": "Structural Steel"})),
                ("add_box", json!({"name": "Block", "min": [20, 5, 10], "size": [20, 20, 10], "material": "Structural Steel"})),
                ("set_mesh", json!({"element_size_mm": hh})),
                ("add_contact", json!({"body_a": "Block", "body_b": "Plane", "kind": "frictional", "friction": 0.25})),
                ("add_fixed_support", json!({"faces": ["Plane.zmin"]})),
                ("add_displacement", json!({"faces": ["Block.xmax"], "x": 0.02, "y": 0.0})),
                ("add_pressure", json!({"faces": ["Block.zmax"], "value_mpa": 2.0})),
                ("solve", json!({})),
            ])?;
            out(vec![qty(&r, "Contact tangential force")?], &r)
        }),
    });
    {
        let e_gpa = 3.0;
        let nu = 0.35;
        let r_mm = 40.0;
        let len_mm = 4.0;
        let f_half = 800.0;
        let f_prime = (2.0 * f_half) / (len_mm * 1e-3);
        let e_star = (e_gpa * 1e9) / (2.0 * (1.0 - nu * nu));
        let p0_want = (f_prime * e_star / (pi * r_mm * 1e-3)).sqrt() * 1e-6;
        v.push(Case {
            id: "hertz_contact",
            group: "Structural",
            title: "Hertz cylinder-on-flat fitted contact (R = 40 mm): peak contact pressure",
            source: "Hertz (1882): p₀ = √(F' E* / πR)",
            sizes: vec![1.33, 1.0],
            size_unit: "mm",
            quick: true,
            metrics: vec![m("hertz_contact.p0", "peak contact pressure", p0_want, "MPa", 0.10)],
            run: Box::new(move |hh| {
                let c = new_core();
                let r = script(&c, &[
                    ("define_material", json!({"name": "HertzMat", "youngs_modulus_gpa": e_gpa, "poisson_ratio": nu})),
                    ("add_box", json!({"name": "Base", "min": [0, 0, -24], "size": [28, len_mm, 24], "material": "HertzMat"})),
                    ("add_cylinder", json!({"name": "Cyl", "base": [0, 0, r_mm], "axis": "y", "radius": r_mm, "height": len_mm, "material": "HertzMat"})),
                    ("add_box", json!({"name": "CutX", "min": [-45, -1, -5], "size": [45, len_mm + 2.0, 90], "operation": "subtract"})),
                    ("add_box", json!({"name": "CutTop", "min": [-5, -1, 20], "size": [50, len_mm + 2.0, 70], "operation": "subtract"})),
                    ("set_mesh", json!({"element_size_mm": hh})),
                    ("add_contact", json!({"body_a": "Cyl", "body_b": "Base", "kind": "frictionless"})),
                    ("add_fixed_support", json!({"faces": ["Base.zmin"]})),
                    ("add_symmetry", json!({"faces": ["bbox.xmin", "bbox.ymin", "bbox.ymax"]})),
                    ("add_force", json!({"faces": ["CutTop.zmin"], "vector": [0, 0, -f_half]})),
                    ("solve", json!({})),
                ])?;
                out(vec![qty(&r, "Peak contact pressure")?], &r)
            }),
        });
    }
    {
        let e = steel.youngs_modulus_gpa * 1e9;
        let i_yy = 0.012 * 0.012_f64.powi(3) / 12.0;
        let uz_want = (20.0 * 0.12_f64.powi(2)) / (2.0 * e * i_yy) * 1e3;
        v.push(Case {
            id: "remote_moment",
            group: "Structural",
            title: "Cantilever beam under RBE3 remote moment (20 N·m): tip deflection",
            source: "Euler–Bernoulli: δ = M L² / (2 E I)",
            sizes: vec![4.0, 3.0],
            size_unit: "mm",
            quick: true,
            metrics: vec![m("remote_moment.uz", "tip deflection", uz_want, "mm", 0.02)],
            run: Box::new(|hh| {
                let c = new_core();
                let r = script(&c, &[
                    ("add_box", json!({"name": "Beam", "min": [0, 0, 0], "size": [120, 12, 12], "material": "Structural Steel"})),
                    ("set_mesh", json!({"element_size_mm": hh})),
                    ("add_fixed_support", json!({"faces": ["Beam.xmin"]})),
                    ("add_remote_force", json!({"faces": ["Beam.xmax"], "point": [120, 6, 6], "force": [0, 0, 0], "moment": [0, 20, 0]})),
                    ("solve", json!({})),
                ])?;
                out(vec![-field(&r, "deformation_z", "min")?], &r)
            }),
        });
    }
    {
        let e = steel.youngs_modulus_gpa * 1e3;
        let pcr = pi * pi * e * (10.0f64.powi(4) / 12.0) / (4.0 * 200.0f64.powi(2));
        v.push(Case {
            id: "euler_buckling",
            group: "Structural",
            title: "Fixed–free column 10 × 10 × 200 mm: Euler buckling load",
            source: "Euler: Pcr = π²EI / (4L²)",
            sizes: vec![2.5, 2.0, 1.25],
            size_unit: "mm",
            quick: true,
            metrics: vec![m("buckling.pcr", "critical load", pcr, "N", 0.04)],
            run: Box::new(|hh| {
                let c = new_core();
                let r = script(&c, &[
                    ("add_box", json!({"name": "Col", "min": [0, 0, 0], "size": [10, 10, 200], "material": "Structural Steel"})),
                    ("set_mesh", json!({"element_size_mm": hh})),
                    ("set_analysis", json!({"type": "linear_buckling", "modes": 2})),
                    ("add_fixed_support", json!({"faces": ["Col.zmin"]})),
                    ("add_force", json!({"faces": ["Col.zmax"], "vector": [0, 0, -1000]})),
                    ("solve", json!({})),
                ])?;
                out(vec![1000.0 * r["modes"][0]["load_multiplier"].as_f64().ok_or_else(|| anyhow!("no modes"))?], &r)
            }),
        });
    }
    v.push(Case {
        id: "random_vibration",
        group: "Structural",
        title: "Cantilever under white-noise base acceleration: 1σ tip displacement",
        source: "Miles' equation with the mode-1 tip participation 1.566 (ratio to the closed form)",
        sizes: vec![2.0, 1.0],
        size_unit: "mm",
        quick: false,
        metrics: vec![m("random.tip_ratio", "1σ tip / Miles", 1.0, "", 0.04)],
        run: Box::new(|hh| {
            let c = new_core();
            script(&c, &[
                ("add_box", json!({"name": "Beam", "size": [100, 10, 6], "material": "Structural Steel"})),
                ("set_mesh", json!({"element_size_mm": hh})),
                ("add_fixed_support", json!({"faces": ["Beam.xmin"]})),
                ("set_analysis", json!({"type": "modal", "modes": 2})),
            ])?;
            let f1 = call(&c, "solve", json!({}))?["modes"][0]["frequency_hz"].as_f64().ok_or_else(|| anyhow!("no modes"))?;
            let s_g = 0.01;
            call(&c, "set_analysis", json!({"type": "random_vibration", "table": [[1, s_g], [20000, s_g]], "direction": "z", "damping_ratio": 0.02}))?;
            let r = call(&c, "solve", json!({}))?;
            let w1 = std::f64::consts::TAU * f1;
            let theory = 1.566 * (s_g * 9.80665f64.powi(2) / (8.0 * 0.02 * w1.powi(3))).sqrt() * 1e3;
            out(vec![qty(&r, "1σ max displacement")? / theory], &r)
        }),
    });
    {
        let (e, sy, et) = (steel.youngs_modulus_gpa * 1e3, steel.yield_strength_mpa, steel.tangent_modulus_gpa * 1e3);
        let sigma = sy + et * (0.005 - sy / e);
        v.push(Case {
            id: "plasticity",
            group: "Structural",
            title: "Bar stretched to 0.5 % strain, bilinear hardening: reaction force",
            source: "σ = σy + Et (ε − σy/E), F = σA",
            sizes: vec![5.0, 2.5],
            size_unit: "mm",
            quick: true,
            metrics: vec![m("plasticity.force", "reaction force", sigma * 100.0, "N", 0.01)],
            run: Box::new(|hh| {
                let c = new_core();
                let r = script(&c, &[
                    ("add_box", json!({"name": "Bar", "size": [100, 10, 10], "material": "Structural Steel"})),
                    ("set_mesh", json!({"element_size_mm": hh})),
                    ("set_analysis", json!({"type": "nonlinear_structural", "load_steps": 10})),
                    ("add_displacement", json!({"faces": ["Bar.xmin"], "x": 0})),
                    ("add_displacement", json!({"faces": ["Bar.ymin"], "y": 0})),
                    ("add_displacement", json!({"faces": ["Bar.zmin"], "z": 0})),
                    ("add_displacement", json!({"faces": ["Bar.xmax"], "x": 0.5})),
                    ("solve", json!({})),
                ])?;
                let f = r["reactions"].as_array().and_then(|a| a.iter().find(|x| x["load_id"] == 4)).and_then(|x| x["force_n"][0].as_f64()).ok_or_else(|| anyhow!("no reaction"))?;
                out(vec![f], &r)
            }),
        });
    }

    // ---------------------------------------------------------------- thermal
    v.push(Case {
        id: "nafems_t4",
        group: "Thermal",
        title: "NAFEMS T4: 2-D conduction with convection, temperature at E",
        source: "NAFEMS T4: T(E) = 18.25 °C (k 52 W/m·K, h 750 W/m²·K, 100 °C base)",
        sizes: vec![25.0, 12.5, 10.0],
        size_unit: "mm",
        quick: true,
        metrics: vec![m("t4.te", "temperature at E", 18.25, "°C", 0.02)],
        run: Box::new(|hh| {
            let c = new_core();
            let t = (hh * 2.0).max(20.0);
            let r = script(&c, &[
                ("define_material", json!({"name": "T4", "based_on": "Structural Steel", "thermal_conductivity": 52})),
                ("add_box", json!({"name": "Plate", "size": [600, 1000, t], "material": "T4"})),
                ("set_mesh", json!({"element_size_mm": hh})),
                ("set_analysis", json!({"type": "steady_thermal"})),
                ("add_temperature", json!({"faces": ["Plate.ymin"], "value": 100})),
                ("add_convection", json!({"faces": ["Plate.xmax", "Plate.ymax"], "coefficient": 750, "ambient": 0})),
                ("solve", json!({})),
            ])?;
            out(vec![probe(&c, [600.0, 200.0, t / 2.0], "temperature")?], &r)
        }),
    });
    {
        let k = steel.thermal_conductivity;
        v.push(Case {
            id: "slab_generation",
            group: "Thermal",
            title: "Slab with uniform heat generation between two cold faces: peak rise",
            source: "ΔTmax = q L² / (8k)",
            sizes: vec![5.0, 2.5],
            size_unit: "mm",
            quick: true,
            metrics: vec![m("slab.dt", "peak temperature rise", 1e6 * 0.01 / (8.0 * k), "K", 0.01)],
            run: Box::new(|hh| {
                let c = new_core();
                let r = script(&c, &[
                    ("add_box", json!({"name": "Slab", "size": [100, 10, 10], "material": "Structural Steel"})),
                    ("set_mesh", json!({"element_size_mm": hh})),
                    ("set_analysis", json!({"type": "steady_thermal"})),
                    ("add_temperature", json!({"faces": ["Slab.xmin", "Slab.xmax"], "value": 20})),
                    ("add_heat_generation", json!({"value": 1e6})),
                    ("solve", json!({})),
                ])?;
                out(vec![field(&r, "temperature", "max")? - 20.0], &r)
            }),
        });
    }
    {
        let k = cu.thermal_conductivity;
        v.push(Case {
            id: "pipe_wall",
            group: "Thermal",
            title: "Radial conduction through a quarter copper pipe wall (ri 10, ro 20 mm)",
            source: "Q = (π/2) k L ΔT / ln(ro/ri)",
            sizes: vec![2.0, 1.5, 1.0],
            size_unit: "mm",
            quick: true,
            metrics: vec![m("pipe.q", "heat flow", std::f64::consts::FRAC_PI_2 * k * 4e-3 * 100.0 / 2f64.ln(), "W", 0.01)],
            run: Box::new(|hh| {
                let c = new_core();
                let r = script(&c, &[
                    ("add_cylinder", json!({"name": "Pipe", "base": [0, 0, 0], "axis": "z", "radius": 20, "height": 4, "material": "Copper C101"})),
                    ("add_cylinder", json!({"name": "Bore", "base": [0, 0, 0], "axis": "z", "radius": 10, "height": 4, "operation": "subtract"})),
                    ("add_box", json!({"name": "CutX", "min": [-30, -30, 0], "size": [30, 60, 4], "operation": "subtract"})),
                    ("add_box", json!({"name": "CutY", "min": [-30, -30, 0], "size": [60, 30, 4], "operation": "subtract"})),
                    ("set_mesh", json!({"element_size_mm": hh})),
                    ("set_analysis", json!({"type": "steady_thermal"})),
                    ("add_temperature", json!({"faces": ["Bore.side"], "value": 100})),
                    ("add_temperature", json!({"faces": ["Pipe.side"], "value": 0})),
                    ("solve", json!({})),
                ])?;
                out(vec![r["reactions"][0]["heat_flow_w"].as_f64().ok_or_else(|| anyhow!("no heat flow"))?], &r)
            }),
        });
    }
    {
        let tau = cu.density * cu.specific_heat * 1e-6 / (10.0 * 6e-4);
        v.push(Case {
            id: "lumped_cooling",
            group: "Thermal",
            title: "Copper cube cooling by convection for one time constant",
            source: "Lumped capacitance (Bi ≪ 0.1): T = T∞ + (T₀ − T∞) e^(−t/τ)",
            sizes: vec![2.5, 1.0],
            size_unit: "mm",
            quick: true,
            metrics: vec![m("lumped.t", "temperature at t = τ", 20.0 + 80.0 / std::f64::consts::E, "°C", 0.01)],
            run: Box::new(move |hh| {
                let c = new_core();
                let r = script(&c, &[
                    ("add_box", json!({"name": "Cube", "size": [10, 10, 10], "material": "Copper C101"})),
                    ("set_mesh", json!({"element_size_mm": hh})),
                    ("set_analysis", json!({"type": "transient_thermal", "end_time": tau, "time_steps": 200, "initial_temperature": 100})),
                    ("add_convection", json!({"faces": ["all"], "coefficient": 10, "ambient": 20})),
                    ("solve", json!({})),
                ])?;
                out(vec![field(&r, "temperature", "max")?], &r)
            }),
        });
    }

    {
        let (ta, sig) = (20.0f64 + 273.15, crate::radiation::SIGMA);
        let t = (ta.powi(4) + 2.0 / (0.8 * sig * 6.0 * 4e-4)).powf(0.25) - 273.15;
        v.push(Case {
            id: "radiating_cube",
            group: "Thermal",
            title: "Copper cube dissipating 2 W by radiation only (ε 0.8, 20 °C surroundings)",
            source: "Stefan–Boltzmann: P = εσA(T⁴ − T∞⁴)",
            sizes: vec![5.0, 2.0],
            size_unit: "mm",
            quick: true,
            metrics: vec![m("radcube.dt", "temperature rise", t - 20.0, "K", 0.01)],
            run: Box::new(|hh| {
                let c = new_core();
                let r = script(&c, &[
                    ("add_box", json!({"name": "Cube", "size": [20, 20, 20], "material": "Copper C101"})),
                    ("set_mesh", json!({"element_size_mm": hh})),
                    ("set_analysis", json!({"type": "steady_thermal"})),
                    ("add_heat_generation", json!({"body": "Cube", "power": 2.0})),
                    ("add_radiation", json!({"faces": ["all"], "emissivity": 0.8, "ambient": 20})),
                    ("solve", json!({})),
                ])?;
                out(vec![field(&r, "temperature", "max")? - 20.0], &r)
            }),
        });
        let (t1, t2) = (500.0f64 + 273.15, 100.0f64 + 273.15);
        let f = crate::radiation::parallel_rectangles_vf(100.0, 100.0, 50.0);
        v.push(Case {
            id: "parallel_plates",
            group: "Thermal",
            title: "Black parallel plates 100 × 100 mm, 50 mm apart: radiation exchange (view factors)",
            source: "Q = Aσ(T₁⁴ − F₁₂T₂⁴), F₁₂ from the coaxial-rectangles formula (Howell C-11)",
            sizes: vec![5.0, 2.0],
            size_unit: "mm",
            quick: false,
            metrics: vec![m("plates.q", "heat into the hot plate", 0.01 * sig * (t1.powi(4) - f * t2.powi(4)), "W", 0.02)],
            run: Box::new(|hh| {
                let c = new_core();
                let r = script(&c, &[
                    ("add_box", json!({"name": "Hot", "min": [0, 0, 0], "size": [100, 100, 4], "material": "Copper C101"})),
                    ("add_box", json!({"name": "Cold", "min": [0, 0, 54], "size": [100, 100, 4], "material": "Copper C101"})),
                    ("set_mesh", json!({"element_size_mm": hh})),
                    ("set_analysis", json!({"type": "steady_thermal"})),
                    ("add_temperature", json!({"faces": ["Hot.zmin"], "value": 500})),
                    ("add_temperature", json!({"faces": ["Cold.zmax"], "value": 100})),
                    ("add_radiation", json!({"faces": ["Hot.zmax", "Cold.zmin"], "emissivity": 1.0, "ambient": -273.15, "enclosure": true})),
                    ("solve", json!({})),
                ])?;
                out(vec![r["reactions"][0]["heat_flow_w"].as_f64().ok_or_else(|| anyhow!("no heat flow"))?], &r)
            }),
        });
        v.push(Case {
            id: "conductivity_kt",
            group: "Thermal",
            title: "Slab with k = 50 (1 + 0.005 T) between 200 °C and 0 °C: heat flow",
            source: "Kirchhoff transform: q = (A/L) ∫ k dT",
            sizes: vec![5.0, 2.5],
            size_unit: "mm",
            quick: true,
            metrics: vec![m("kt.q", "heat flow", 1e-4 / 0.05 * 50.0 * (200.0 + 0.0025 * 40000.0), "W", 0.01)],
            run: Box::new(|hh| {
                let c = new_core();
                let r = script(&c, &[
                    ("define_material", json!({"name": "kT", "based_on": "Structural Steel", "thermal_conductivity": 50, "conductivity_table": [[0, 50], [200, 100]]})),
                    ("add_box", json!({"name": "Slab", "size": [50, 10, 10], "material": "kT"})),
                    ("set_mesh", json!({"element_size_mm": hh})),
                    ("set_analysis", json!({"type": "steady_thermal"})),
                    ("add_temperature", json!({"faces": ["Slab.xmin"], "value": 200})),
                    ("add_temperature", json!({"faces": ["Slab.xmax"], "value": 0})),
                    ("solve", json!({})),
                ])?;
                out(vec![r["reactions"][0]["heat_flow_w"].as_f64().ok_or_else(|| anyhow!("no heat flow"))?], &r)
            }),
        });
    }

    // ---------------------------------------------------------------- fluids
    v.push(Case {
        id: "poiseuille",
        group: "Fluids",
        title: "Laminar plane channel (water, 4 mm gap): centreline velocity",
        source: "Plane Poiseuille flow: u_max = 1.5 U_mean",
        sizes: vec![1.0, 0.67, 0.5],
        size_unit: "mm",
        quick: true,
        metrics: vec![m("poiseuille.umax", "u_max / U", 1.5, "", 0.03)],
        run: Box::new(|hh| {
            let c = new_core();
            let r = script(&c, &[
                ("add_enclosure", json!({"min": [0, 0, 0], "size": [60, 4, 10], "material": "Water"})),
                ("set_analysis", json!({"type": "fluid", "max_iterations": 20000})),
                ("set_mesh", json!({"element_size_mm": hh})),
                ("add_inlet", json!({"faces": ["Enclosure.xmin"], "velocity": 0.001})),
                ("add_outlet", json!({"faces": ["Enclosure.xmax"], "pressure": 0})),
                ("add_symmetry", json!({"faces": ["Enclosure.ymin", "Enclosure.ymax"]})),
                ("solve", json!({})),
            ])?;
            out(vec![field(&r, "velocity", "max")? / 0.001], &r)
        }),
    });
    v.push(Case {
        id: "ghia_re100",
        group: "Fluids",
        title: "Lid-driven cavity, Re 100: u on the vertical centreline at y = 0.4531",
        source: "Ghia, Ghia & Shin (1982), Table I: u/U = −0.21090",
        sizes: vec![4.0, 2.5, 2.0],
        size_unit: "mm",
        quick: true,
        metrics: vec![m("ghia.u", "u/U at y = 0.4531", -0.21090, "", 0.04)],
        run: Box::new(|hh| {
            let c = new_core();
            let u = 100.0 * NU_AIR / 0.1;
            let r = script(&c, &[
                ("add_enclosure", json!({"min": [0, 0, 0], "size": [100, hh, 100]})),
                ("set_analysis", json!({"type": "fluid", "max_iterations": 30000})),
                ("set_mesh", json!({"element_size_mm": hh})),
                ("add_moving_wall", json!({"faces": ["Enclosure.zmax"], "velocity": [u, 0, 0]})),
                ("add_symmetry", json!({"faces": ["Enclosure.ymin", "Enclosure.ymax"]})),
                ("solve", json!({})),
            ])?;
            let z = 45.31;
            let lo = (z / hh).floor() * hh;
            let w = (z - lo) / hh;
            let a = probe(&c, [50.0, hh / 2.0, lo], "velocity_x")?;
            let b = probe(&c, [50.0, hh / 2.0, lo + hh], "velocity_x")?;
            out(vec![((1.0 - w) * a + w * b) / u], &r)
        }),
    });
    v.push(Case {
        id: "de_vahl_davis",
        group: "Fluids",
        title: "Differentially heated square cavity, Ra 10⁴: mean Nusselt number",
        source: "de Vahl Davis (1983): Nu = 2.243",
        sizes: vec![20.0, 32.0, 40.0],
        size_unit: "cells",
        quick: false,
        metrics: vec![m("dvd.nu", "Nusselt number", 2.243, "", 0.03)],
        run: Box::new(|cells| {
            let c = new_core();
            let alpha = 0.026 / (1.204 * 1006.0);
            let (beta, dt, g, ra) = (3.4e-3, 10.0, 9.80665, 1e4);
            let l = (ra * NU_AIR * alpha / (g * beta * dt)).cbrt() * 1e3;
            let hh = l / cells;
            script(&c, &[
                ("add_enclosure", json!({"min": [0, 0, 0], "size": [l, hh, l]})),
                ("set_analysis", json!({"type": "conjugate_heat_transfer", "max_iterations": 60000, "reference_temperature": 22})),
                ("set_mesh", json!({"element_size_mm": hh})),
                ("add_symmetry", json!({"faces": ["Enclosure.ymin", "Enclosure.ymax"]})),
            ])?;
            let hot = call(&c, "add_temperature", json!({"faces": ["Enclosure.xmin"], "value": 27}))?;
            call(&c, "add_temperature", json!({"faces": ["Enclosure.xmax"], "value": 17}))?;
            call(&c, "add_gravity", json!({"acceleration": [0, 0, -g]}))?;
            let r = call(&c, "solve", json!({}))?;
            let id = hot["added"]["id"].as_u64().unwrap_or(2);
            let qh = qty(&r, &format!("Heat flow through #{id}"))?;
            out(vec![qh * (l * 1e-3) / (0.026 * dt * (l * 1e-3) * (hh * 1e-3))], &r)
        }),
    });
    v.push(Case {
        id: "dean_channel",
        group: "Fluids",
        title: "Turbulent plane channel, Re 20 000 (LVEL): developed pressure gradient",
        source: "Dean (1978): Cf = 0.073 Re⁻¹ᐟ⁴",
        sizes: vec![1.0],
        size_unit: "mm",
        quick: false,
        metrics: vec![{
            let (ub, d2) = (15.0, 0.02);
            let cf = 0.073 * (ub * d2 / NU_AIR).powf(-0.25);
            m("dean.dpdx", "dp/dx", cf * 0.5 * 1.204 * ub * ub / (d2 / 2.0), "Pa/m", 0.15)
        }],
        run: Box::new(|hh| {
            let c = new_core();
            let r = script(&c, &[
                ("add_enclosure", json!({"min": [0, 0, 0], "size": [1000, hh, 20]})),
                ("set_analysis", json!({"type": "fluid", "turbulence": "lvel", "max_iterations": 60000})),
                ("set_mesh", json!({"element_size_mm": hh})),
                ("add_inlet", json!({"faces": ["Enclosure.xmin"], "velocity": 15.0})),
                ("add_outlet", json!({"faces": ["Enclosure.xmax"], "pressure": 0})),
                ("add_symmetry", json!({"faces": ["Enclosure.ymin", "Enclosure.ymax"]})),
                ("solve", json!({})),
            ])?;
            let dpdx = (probe(&c, [600.0, hh / 2.0, 10.0], "fluid_pressure")? - probe(&c, [900.0, hh / 2.0, 10.0], "fluid_pressure")?) / 0.3;
            out(vec![dpdx], &r)
        }),
    });
    v.push(Case {
        id: "cht_balance",
        group: "Fluids",
        title: "Heated block in a duct (CHT): heat carried away by the air",
        source: "Energy conservation: carried = generated (0.2 W)",
        sizes: vec![1.0],
        size_unit: "mm",
        quick: true,
        metrics: vec![m("cht.q", "heat carried away", 0.2, "W", 0.01)],
        run: Box::new(|hh| {
            let c = new_core();
            let r = script(&c, &[
                ("add_box", json!({"name": "Block", "min": [40, 0, 0], "size": [10, 4, 6], "material": "Aluminum 6061-T6"})),
                ("add_enclosure", json!({"min": [0, 0, 0], "size": [120, 4, 20]})),
                ("set_analysis", json!({"type": "conjugate_heat_transfer", "max_iterations": 40000})),
                ("set_mesh", json!({"element_size_mm": hh})),
                ("add_inlet", json!({"faces": ["Enclosure.xmin"], "velocity": 0.1, "temperature": 20})),
                ("add_outlet", json!({"faces": ["Enclosure.xmax"], "pressure": 0})),
                ("add_symmetry", json!({"faces": ["Enclosure.ymin", "Enclosure.ymax"]})),
                ("add_heat_generation", json!({"body": "Block", "power": 0.2})),
                ("solve", json!({})),
            ])?;
            out(vec![qty(&r, "Heat carried away by the flow")?], &r)
        }),
    });

    // ---------------------------------------------------------------- electromagnetics
    {
        let want = EPS0 * ptfe.relative_permittivity * 0.04 * 0.04 / 0.002;
        v.push(Case {
            id: "capacitor",
            group: "Electromagnetics",
            title: "Parallel-plate capacitor, PTFE 40 × 40 × 2 mm",
            source: "C = ε₀εr A / d",
            sizes: vec![2.0, 1.0, 0.5],
            size_unit: "mm",
            quick: true,
            metrics: vec![m("capacitor.c", "capacitance", want * 1e12, "pF", 0.02)],
            run: Box::new(|hh| {
                let c = new_core();
                let r = script(&c, &[
                    ("add_box", json!({"name": "Dielectric", "size": [40, 40, 2], "material": "PTFE"})),
                    ("set_mesh", json!({"element_size_mm": hh})),
                    ("set_analysis", json!({"type": "electrostatic"})),
                    ("add_voltage", json!({"faces": ["Dielectric.zmax"], "value": 100})),
                    ("add_voltage", json!({"faces": ["Dielectric.zmin"], "value": 0})),
                    ("solve", json!({})),
                ])?;
                out(vec![qty(&r, "Capacitance (from energy)")? * 1e12], &r)
            }),
        });
    }
    {
        let r_th = 0.1 / (cu.electrical_conductivity * 1e-4);
        v.push(Case {
            id: "bar_resistance",
            group: "Electromagnetics",
            title: "Copper bar 100 × 10 × 10 mm: DC resistance",
            source: "R = L / (σA)",
            sizes: vec![5.0, 2.0],
            size_unit: "mm",
            quick: true,
            metrics: vec![m("resistance.r", "resistance", r_th * 1e6, "µΩ", 0.01)],
            run: Box::new(|hh| {
                let c = new_core();
                let r = script(&c, &[
                    ("add_box", json!({"name": "Bar", "size": [100, 10, 10], "material": "Copper C101"})),
                    ("set_mesh", json!({"element_size_mm": hh})),
                    ("set_analysis", json!({"type": "current_flow"})),
                    ("add_voltage", json!({"faces": ["Bar.xmin"], "value": 0.001})),
                    ("add_voltage", json!({"faces": ["Bar.xmax"], "value": 0})),
                    ("solve", json!({})),
                ])?;
                out(vec![qty(&r, "Resistance")? * 1e6], &r)
            }),
        });
    }
    v.push(Case {
        id: "ring_coil",
        group: "Electromagnetics",
        title: "Ring coil (160 A·turns, mean radius 20 mm): field at the centre",
        source: "Thin loop: B = μ₀ I / (2R)",
        sizes: vec![3.0, 2.0],
        size_unit: "mm",
        quick: false,
        metrics: vec![m("coil.b", "B at the centre", 1.256_637_06e-6 * 160.0 / (2.0 * 0.020) * 1e3, "mT", 0.12)],
        run: Box::new(|hh| {
            let c = new_core();
            let r = script(&c, &[
                ("add_cylinder", json!({"name": "Coil", "base": [0, 0, -2], "radius": 22, "height": 4, "material": "Copper C101"})),
                ("add_cylinder", json!({"name": "Bore", "base": [0, 0, -2], "radius": 18, "height": 4, "operation": "subtract"})),
                ("add_enclosure", json!({"padding_mm": [50, 50, 50, 50, 40, 40]})),
                ("set_analysis", json!({"type": "magnetostatic"})),
                ("set_mesh", json!({"element_size_mm": hh})),
                ("add_coil", json!({"body": "Coil", "current_density": 10, "axis": "z", "center": [0, 0, 0]})),
                ("solve", json!({})),
            ])?;
            out(vec![probe(&c, [0.0, 0.0, 0.0], "magnetic_flux_density")? * 1e3], &r)
        }),
    });
    {
        let l0 = 299.792458 / 2.45;
        let lg = l0 / (1.0 - (l0 / 172.0f64).powi(2)).sqrt();
        v.push(Case {
            id: "wr340",
            group: "Electromagnetics",
            title: "WR-340 waveguide at 2.45 GHz: guide wavelength from the standing wave",
            source: "λg = λ₀ / √(1 − (λ₀/2a)²)",
            sizes: vec![2.0],
            size_unit: "mm",
            quick: false,
            metrics: vec![m("wr340.half_lg", "λg/2 (null spacing)", lg / 2.0, "mm", 0.02)],
            run: Box::new(|hh| {
                let c = new_core();
                for (name, min, size) in [("WallY0", [0, -4, -4], [260, 4, 52]), ("WallY1", [0, 86, -4], [260, 4, 52]), ("WallZ0", [0, 0, -4], [260, 86, 4]), ("WallZ1", [0, 0, 44], [260, 86, 4]), ("Short", [252, 0, 0], [8, 86, 44])] {
                    call(&c, "add_box", json!({"name": name, "min": min, "size": size, "material": "Copper C101"}))?;
                }
                let r = script(&c, &[
                    ("add_enclosure", json!({"min": [0, -4, -4], "size": [260, 94, 52]})),
                    ("set_mesh", json!({"element_size_mm": hh})),
                    ("set_analysis", json!({"type": "full_wave_em", "frequency_ghz": 2.45, "em_boundary": "open"})),
                    ("add_port", json!({"faces": ["Enclosure.xmin"], "power": 100, "polarization": "z"})),
                    ("solve", json!({})),
                ])?;
                let xs: Vec<f64> = (10..=125).map(|i| i as f64 * 2.0).collect();
                let e: Vec<f64> = xs.iter().map(|&x| probe(&c, [x, 43.0, 22.0], "electric_field")).collect::<Result<_>>()?;
                let emax = e.iter().cloned().fold(0.0, f64::max);
                let mut nulls = vec![];
                for i in 1..e.len() - 1 {
                    if e[i] < e[i - 1] && e[i] <= e[i + 1] && e[i] < 0.3 * emax {
                        let (a, b, cc) = (e[i - 1].powi(2), e[i].powi(2), e[i + 1].powi(2));
                        nulls.push(xs[i] + 0.5 * (a - cc) / (a - 2.0 * b + cc) * 2.0);
                    }
                }
                if nulls.len() < 2 {
                    bail!("fewer than two standing-wave nulls");
                }
                let n = nulls.len();
                out(vec![nulls[n - 1] - nulls[n - 2]], &r)
            }),
        });
    }
    v.push(Case {
        id: "short_dipole",
        group: "Electromagnetics",
        title: "Short dipole in free space (FDTD + near-to-far-field): directivity and power",
        source: "Hertzian dipole: D = 1.5; radiated = delivered (lossless)",
        sizes: vec![3.0, 2.0],
        size_unit: "mm",
        quick: false,
        metrics: vec![m("dipole.d", "directivity (linear)", 1.5, "", 0.03), m("dipole.power", "(radiated + absorbed) / input", 1.0, "", 0.05)],
        run: Box::new(|hh| {
            let c = new_core();
            let r = script(&c, &[
                ("add_box", json!({"name": "Feed", "min": [-3, -3, -3], "size": [6, 6, 6], "material": "PTFE"})),
                ("add_enclosure", json!({"min": [-60, -60, -60], "size": [120, 120, 120]})),
                ("set_mesh", json!({"element_size_mm": hh})),
                ("set_analysis", json!({"type": "full_wave_em", "frequency_ghz": 2.45, "em_boundary": "open"})),
                ("add_port", json!({"faces": ["Feed.zmax"], "power": 1, "polarization": "x", "profile": "uniform"})),
                ("solve", json!({})),
            ])?;
            out(vec![10f64.powf(qty(&r, "Max directivity")? / 10.0), qty(&r, "Radiated power")? + qty(&r, "Absorbed power")?], &r)
        }),
    });

    // ---------------------------------------------------------------- electronics
    let dc_board = |via: bool| {
        let (seg, tp2) = if via {
            (r#"(segment (start 100 50) (end 110 50) (width 0.5) (layer "F.Cu") (net 1))
  (via (at 110 50) (size 0.6) (drill 0.3) (layers "F.Cu" "B.Cu") (net 1))
  (segment (start 110 50) (end 120 50) (width 0.5) (layer "B.Cu") (net 1))"#, "B.Cu")
        } else {
            (r#"(segment (start 100 50) (end 120 50) (width 0.5) (layer "F.Cu") (net 1))"#, "F.Cu")
        };
        format!(r#"(kicad_pcb (version 20240108) (generator "bench")
  (general (thickness 1.6))
  (layers (0 "F.Cu" signal) (31 "B.Cu" signal) (44 "Edge.Cuts" user))
  (setup (stackup (layer "F.Cu" (type "copper") (thickness 0.035)) (layer "dielectric 1" (type "core") (thickness 1.53)) (layer "B.Cu" (type "copper") (thickness 0.035))))
  (net 0 "") (net 1 "VBUS")
  (gr_rect (start 95 45) (end 125 55) (layer "Edge.Cuts") (width 0.1))
  (footprint "TP" (layer "F.Cu") (at 100 50) (property "Reference" "TP1" (at 0 0 0) (layer "F.SilkS")) (pad "1" smd rect (at 0 0) (size 1 1) (layers "F.Cu") (net 1 "VBUS")))
  (footprint "TP" (layer "{tp2}") (at 120 50) (property "Reference" "TP2" (at 0 0 0) (layer "F.SilkS")) (pad "1" smd rect (at 0 0) (size 1 1) (layers "{tp2}") (net 1 "VBUS")))
  {seg}
)"#)
    };
    let rho = 1.72e-8;
    let track = rho * 9.5e-3 / (0.5e-3 * 35e-6);
    let barrel = rho * 1.6e-3 / (pi * 0.3e-3 * 25e-6);
    for (via, id, title, source, want) in [
        (false, "pcb_track", "PCB DC: 20 mm of 0.5 mm track (35 µm) between two pads", "R = ρL/(w·t) between the pad edges", rho * 19e-3 / (0.5e-3 * 35e-6) * 1e3),
        (true, "pcb_via", "PCB DC: track → via → track to the other layer", "series: two tracks + via barrel ρL/(π d t_plating)", (2.0 * track + barrel) * 1e3),
    ] {
        let text = dc_board(via);
        v.push(Case {
            id,
            group: "Electronics",
            title,
            source,
            sizes: vec![0.0],
            size_unit: "auto",
            quick: true,
            metrics: vec![m(if via { "pcb_via.r" } else { "pcb_track.r" }, "resistance", want, "mΩ", 0.02)],
            run: Box::new(move |_| {
                let r = crate::kicad::dc::dc_drop_text(&text, "bench", None, "TP1.1", "TP2.1", 1.0, 20.0)?;
                Ok(Outcome { values: vec![r["resistance_mohm"].as_f64().unwrap_or(f64::NAN)], info: json!({"dofs": r["unknowns"], "solver": "multigrid CG (sheet network)"}) })
            }),
        });
    }
    {
        let kd = mat("FR-4 (PCB laminate)").thermal_conductivity;
        let kt = kd * (1.6 - 0.07) + 385.0 * 0.035 + kd * 0.035;
        let board = r#"(kicad_pcb (version 20240108) (generator "bench")
  (general (thickness 1.6))
  (layers (0 "F.Cu" signal) (31 "B.Cu" signal) (44 "Edge.Cuts" user))
  (setup (stackup (layer "F.Cu" (type "copper") (thickness 0.035)) (layer "dielectric 1" (type "core") (thickness 1.53)) (layer "B.Cu" (type "copper") (thickness 0.035))))
  (gr_rect (start 100 50) (end 140 60) (layer "Edge.Cuts") (width 0.1))
  (zone (net 1) (net_name "GND") (layer "F.Cu") (filled_polygon (layer "F.Cu") (pts (xy 99 49) (xy 141 49) (xy 141 61) (xy 99 61))))
)"#;
        v.push(Case {
            id: "pcb_plane",
            group: "Electronics",
            title: "KiCad board with a full copper plane: in-plane heat conduction",
            source: "k∥t = Σ kᵢtᵢ; Q = (k∥t)·W·ΔT / L",
            sizes: vec![1.0, 0.5],
            size_unit: "mm",
            quick: true,
            metrics: vec![m("pcb_plane.q", "heat flow", kt * 1e-3 * 10e-3 * 10.0 / 40e-3 * 1e3, "mW", 0.02)],
            run: Box::new(move |hh| {
                let path = temp_file("plane.kicad_pcb", board)?;
                let c = new_core();
                let r = script(&c, &[
                    ("import_kicad", json!({"path": path, "components": false})),
                    ("set_analysis", json!({"type": "steady_thermal"})),
                    ("set_mesh", json!({"element_size_mm": hh})),
                    ("add_temperature", json!({"faces": ["bbox.xmin"], "value": 10})),
                    ("add_temperature", json!({"faces": ["bbox.xmax"], "value": 0})),
                    ("solve", json!({})),
                ]);
                let _ = std::fs::remove_file(&path);
                let r = r?;
                out(vec![r["reactions"][0]["heat_flow_w"].as_f64().ok_or_else(|| anyhow!("no heat flow"))? * 1e3], &r)
            }),
        });
    }
    let _ = (al, ptfe);
    v
}

// ------------------------------------------------------------------ runner

#[derive(Clone)]
pub struct Options {
    pub quick: bool,
    pub filter: Option<String>,
    pub out_dir: String,
    pub ansys: Option<String>,
}

/// Run the suite; returns the report JSON (also written with the HTML to `out_dir`).
pub fn run(opts: &Options, log: &dyn Fn(&str)) -> Result<Value> {
    let t0 = Instant::now();
    let ansys: Value = match &opts.ansys {
        Some(p) => serde_json::from_str(&std::fs::read_to_string(p)?).map_err(|e| anyhow!("{p}: {e}"))?,
        None => json!({}),
    };
    let mut cases_out = Vec::new();
    let (mut pass, mut fail) = (0, 0);
    for case in cases() {
        if opts.quick && !case.quick {
            continue;
        }
        if let Some(f) = &opts.filter {
            let f = f.to_lowercase();
            if !case.id.contains(&f) && !case.group.to_lowercase().contains(&f) {
                continue;
            }
        }
        let sizes: Vec<f64> = if opts.quick { vec![*case.sizes.last().unwrap()] } else { case.sizes.clone() };
        let mut runs = Vec::new();
        for &s in &sizes {
            log(&format!("{:<18} {} = {}", case.id, case.size_unit, s));
            let t = Instant::now();
            // a panicking case must not end the whole run
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| (case.run)(s))).unwrap_or_else(|_| Err(anyhow!("panicked")));
            let secs = t.elapsed().as_secs_f64();
            match r {
                Ok(o) => runs.push(json!({"size": s, "seconds": secs, "values": o.values, "dofs": o.info["dofs"], "elements": o.info["elements"], "solver": o.info["solver"]})),
                Err(e) => runs.push(json!({"size": s, "seconds": secs, "error": format!("{e:#}")})),
            }
        }
        let finest = runs.last().cloned().unwrap_or(Value::Null);
        let metrics: Vec<Value> = case
            .metrics
            .iter()
            .enumerate()
            .map(|(k, mt)| {
                let err_of = |v: f64| if mt.absolute { (v - mt.reference).abs() } else { (v - mt.reference).abs() / mt.reference.abs().max(1e-300) };
                let points: Vec<Value> = runs
                    .iter()
                    .filter_map(|r| {
                        let v = r["values"][k].as_f64()?;
                        Some(json!({"size": r["size"], "value": v, "error": err_of(v), "dofs": r["dofs"], "seconds": r["seconds"]}))
                    })
                    .collect();
                let fin = finest["values"][k].as_f64();
                let ok = fin.is_some_and(|v| err_of(v) <= mt.tol);
                if ok {
                    pass += 1;
                } else {
                    fail += 1;
                }
                // observed order of convergence from the last two sizes
                let order = if points.len() >= 2 && case.size_unit == "mm" {
                    let (a, b) = (&points[points.len() - 2], &points[points.len() - 1]);
                    let (ea, eb) = (a["error"].as_f64().unwrap_or(0.0), b["error"].as_f64().unwrap_or(0.0));
                    let (ha, hb) = (a["size"].as_f64().unwrap_or(1.0), b["size"].as_f64().unwrap_or(1.0));
                    (ea > 0.0 && eb > 0.0 && ha > hb).then(|| (ea / eb).ln() / (ha / hb).ln())
                } else {
                    None
                };
                // a size whose error spikes above both neighbours (and the tolerance) points at a mesh-
                // dependent defect even when the finest size passes; failed sizes count as anomalies too
                let errs: Vec<f64> = points.iter().filter_map(|p| p["error"].as_f64()).collect();
                let mut anomalies: Vec<String> = (1..errs.len().saturating_sub(1))
                    .filter(|&i| errs[i] > mt.tol && errs[i] > 2.0 * errs[i - 1] && errs[i] > 2.0 * errs[i + 1])
                    .map(|i| format!("error spikes to {:.1} % at size {}", 100.0 * errs[i], points[i]["size"]))
                    .collect();
                for r in &runs {
                    if let Some(e) = r["error"].as_str() {
                        anomalies.push(format!("size {} failed: {e}", r["size"]));
                    }
                }
                json!({
                    "id": mt.id, "title": mt.title, "unit": mt.unit, "reference": mt.reference, "tolerance": mt.tol, "absolute": mt.absolute,
                    "value": fin, "error": fin.map(err_of), "pass": ok, "points": points, "observed_order": order, "anomalies": anomalies,
                    "other_solver": ansys.get(mt.id).cloned().unwrap_or(Value::Null),
                })
            })
            .collect();
        cases_out.push(json!({"id": case.id, "group": case.group, "title": case.title, "source": case.source, "size_unit": case.size_unit, "runs": runs, "metrics": metrics}));
    }
    let report = json!({
        "simcraft_version": env!("CARGO_PKG_VERSION"),
        "commit": std::process::Command::new("git").args(["rev-parse", "--short", "HEAD"]).output().ok().and_then(|o| String::from_utf8(o.stdout).ok()).map(|s| s.trim().to_string()),
        "date": std::process::Command::new("date").arg("-u").arg("+%Y-%m-%d %H:%M UTC").output().ok().and_then(|o| String::from_utf8(o.stdout).ok()).map(|s| s.trim().to_string()),
        "machine": machine(),
        "mode": if opts.quick { "quick" } else { "full" },
        "seconds": t0.elapsed().as_secs_f64(),
        "passed": pass,
        "failed": fail,
        "cases": cases_out,
        "other_solver_name": ansys.get("_solver").cloned().unwrap_or(json!("Ansys")),
    });
    std::fs::create_dir_all(&opts.out_dir)?;
    std::fs::write(format!("{}/bench.json", opts.out_dir), serde_json::to_string_pretty(&report)?)?;
    std::fs::write(format!("{}/bench.html", opts.out_dir), html(&report))?;
    Ok(report)
}

fn machine() -> Value {
    let cpu = std::fs::read_to_string("/proc/cpuinfo").ok().and_then(|s| s.lines().find(|l| l.starts_with("model name")).map(|l| l.split(':').nth(1).unwrap_or("").trim().to_string()));
    json!({"cpu": cpu, "threads": rayon::current_num_threads(), "os": std::env::consts::OS})
}

/// Template for results from another solver: metric id → value (and optionally dofs, seconds).
pub fn template() -> Value {
    let mut o = serde_json::Map::new();
    o.insert("_solver".into(), json!("Ansys Mechanical 2025 R2 (fill in)"));
    o.insert("_note".into(), json!("Fill value (same unit as SimCraft), and optionally dofs and seconds, per metric; or a list of {value, dofs, seconds} for several meshes. Unknown metrics are ignored."));
    for c in cases() {
        for mt in &c.metrics {
            o.insert(mt.id.into(), json!({"value": null, "dofs": null, "seconds": null, "unit": mt.unit, "case": c.title}));
        }
    }
    Value::Object(o)
}

// ------------------------------------------------------------------ HTML report

fn esc(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;").replace('"', "&quot;").replace('\'', "&#39;")
}

fn fmt(v: f64) -> String {
    if v == 0.0 || !v.is_finite() {
        return format!("{v}");
    }
    let a = v.abs();
    if (1e-3..1e5).contains(&a) { format!("{:.4}", v).trim_end_matches('0').trim_end_matches('.').to_string() } else { format!("{v:.4e}") }
}

/// log-log error vs unknowns, SimCraft points joined, other solver as diamonds.
fn plot(points: &[Value], other: &Value, tol: f64) -> String {
    let mut pts: Vec<(f64, f64, bool)> = points.iter().filter_map(|p| Some((p["dofs"].as_f64()?, p["error"].as_f64()?.max(1e-6), true))).collect();
    let others: Vec<&Value> = match other {
        Value::Array(a) => a.iter().collect(),
        Value::Object(_) => vec![other],
        _ => vec![],
    };
    let reference = points.first().map(|_| ()).is_some();
    let _ = reference;
    for o in &others {
        if let (Some(d), Some(e)) = (o["dofs"].as_f64(), o["error"].as_f64()) {
            pts.push((d, e.max(1e-6), false));
        }
    }
    if pts.is_empty() {
        return String::from("<div class=noplot>no mesh sweep</div>");
    }
    let (w, h, pad) = (300.0, 170.0, 34.0);
    let lx: Vec<f64> = pts.iter().map(|p| p.0.max(1.0).log10()).collect();
    let ly: Vec<f64> = pts.iter().map(|p| p.1.log10()).chain([tol.max(1e-6).log10()]).collect();
    let (x0, x1) = (lx.iter().cloned().fold(f64::INFINITY, f64::min) - 0.15, lx.iter().cloned().fold(f64::NEG_INFINITY, f64::max) + 0.15);
    let (y0, y1) = ((ly.iter().cloned().fold(f64::INFINITY, f64::min) - 0.3).floor(), (ly.iter().cloned().fold(f64::NEG_INFINITY, f64::max) + 0.3).ceil());
    let sx = |v: f64| pad + (v - x0) / (x1 - x0).max(1e-9) * (w - pad - 8.0);
    let sy = |v: f64| h - 22.0 - (v - y0) / (y1 - y0).max(1e-9) * (h - 30.0);
    let mut s = format!("<svg viewBox='0 0 {w} {h}' class=plot role=img aria-label='error versus unknowns'>");
    let mut y = y0;
    while y <= y1 + 1e-9 {
        s += &format!("<line x1={pad} x2={} y1={:.1} y2={:.1} class=grid /><text x={} y={:.1} class=ax text-anchor=end>{}%</text>", w - 8.0, sy(y), sy(y), pad - 4.0, sy(y) + 3.0, fmt(100.0 * 10f64.powf(y)));
        y += 1.0;
    }
    s += &format!("<line x1={pad} x2={} y1={:.1} y2={:.1} class=tol /><text x={} y={:.1} class=ax>tolerance</text>", w - 8.0, sy(tol.max(1e-6).log10()), sy(tol.max(1e-6).log10()), pad + 4.0, sy(tol.max(1e-6).log10()) - 3.0);
    let own: Vec<(f64, f64)> = pts.iter().filter(|p| p.2).map(|p| (sx(p.0.max(1.0).log10()), sy(p.1.log10()))).collect();
    if own.len() > 1 {
        s += &format!("<polyline class=line points='{}' />", own.iter().map(|(a, b)| format!("{a:.1},{b:.1}")).collect::<Vec<_>>().join(" "));
    }
    for (a, b) in &own {
        s += &format!("<circle cx={a:.1} cy={b:.1} r=3.5 class=dot />");
    }
    for p in pts.iter().filter(|p| !p.2) {
        let (a, b) = (sx(p.0.max(1.0).log10()), sy(p.1.log10()));
        s += &format!("<rect x={:.1} y={:.1} width=7 height=7 transform='rotate(45 {a:.1} {b:.1})' class=other />", a - 3.5, b - 3.5);
    }
    s += &format!("<text x={} y={} class=ax text-anchor=middle>unknowns (log)</text></svg>", (w + pad) / 2.0, h - 4.0);
    s
}

pub fn html(r: &Value) -> String {
    let other = r["other_solver_name"].as_str().unwrap_or("Ansys");
    let mut body = String::new();
    let mut group = String::new();
    for c in r["cases"].as_array().cloned().unwrap_or_default() {
        let g = c["group"].as_str().unwrap_or("");
        if g != group {
            group = g.to_string();
            body += &format!("<h2>{}</h2>", esc(g));
        }
        for mt in c["metrics"].as_array().cloned().unwrap_or_default() {
            let errors: Vec<String> = mt["anomalies"].as_array().cloned().unwrap_or_default().iter().filter_map(|a| a.as_str().map(|e| format!("<p class=err>⚠ {}</p>", esc(e)))).collect();
            let pass = mt["pass"].as_bool().unwrap_or(false);
            let unit = mt["unit"].as_str().unwrap_or("");
            let rows: String = mt["points"]
                .as_array()
                .cloned()
                .unwrap_or_default()
                .iter()
                .map(|p| format!("<tr><td>{}</td><td>{}</td><td>{}</td><td>{:.2} %</td><td>{:.2} s</td></tr>", fmt(p["size"].as_f64().unwrap_or(0.0)), p["dofs"].as_f64().map(|d| format!("{d:.0}")).unwrap_or("–".into()), fmt(p["value"].as_f64().unwrap_or(f64::NAN)), 100.0 * p["error"].as_f64().unwrap_or(f64::NAN), p["seconds"].as_f64().unwrap_or(0.0)))
                .collect();
            let os = &mt["other_solver"];
            let other_cell = match os {
                Value::Object(o) if o.get("value").is_some_and(|v| v.is_number()) => {
                    let v = o["value"].as_f64().unwrap_or(0.0);
                    let e = (v - mt["reference"].as_f64().unwrap_or(0.0)).abs() / mt["reference"].as_f64().unwrap_or(1.0).abs().max(1e-300);
                    format!("<div class=other-res>{}: {} {} ({:.2} % error{})</div>", esc(other), fmt(v), esc(unit), 100.0 * e, o.get("dofs").and_then(|d| d.as_f64()).map(|d| format!(", {d:.0} unknowns")).unwrap_or_default())
                }
                _ => String::new(),
            };
            // give the plot the other solver's error
            let os_plot = match os {
                Value::Object(o) if o.get("value").is_some_and(|v| v.is_number()) => {
                    let v = o["value"].as_f64().unwrap_or(0.0);
                    let e = (v - mt["reference"].as_f64().unwrap_or(0.0)).abs() / mt["reference"].as_f64().unwrap_or(1.0).abs().max(1e-300);
                    json!({"dofs": o.get("dofs"), "error": e})
                }
                _ => Value::Null,
            };
            body += &format!(
                "<section class=case><div class=head><span class='badge {}'>{}</span><h3>{}</h3></div><p class=meta>{} — <b>{}</b>. Reference: {} {} ({}). Tolerance {:.1} %{}.</p>{}<div class=row><div>{}</div><table><thead><tr><th>size ({})</th><th>unknowns</th><th>value</th><th>error</th><th>time</th></tr></thead><tbody>{}</tbody></table></div>{}</section>",
                if !pass { "fail" } else if errors.is_empty() { "pass" } else { "warn" },
                if !pass { "FAIL" } else if errors.is_empty() { "PASS" } else { "PASS ⚠" },
                esc(c["title"].as_str().unwrap_or("")),
                esc(c["id"].as_str().unwrap_or("")),
                esc(mt["title"].as_str().unwrap_or("")),
                fmt(mt["reference"].as_f64().unwrap_or(0.0)),
                esc(unit),
                esc(c["source"].as_str().unwrap_or("")),
                100.0 * mt["tolerance"].as_f64().unwrap_or(0.0),
                mt["observed_order"].as_f64().map(|o| format!("; observed convergence order {o:.1}")).unwrap_or_default(),
                other_cell,
                plot(mt["points"].as_array().map(|a| a.as_slice()).unwrap_or(&[]), &os_plot, mt["tolerance"].as_f64().unwrap_or(0.01)),
                esc(c["size_unit"].as_str().unwrap_or("")),
                rows,
                errors.join(""),
            );
        }
    }
    let m = &r["machine"];
    format!(
        r#"<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1"><title>SimCraft benchmarks</title><style>
:root{{--bg:#f7f8fa;--card:#fff;--text:#1b1f24;--weak:#5b6573;--line:#dde2e8;--ok:#1a7f45;--bad:#c62828;--accent:#e8742a;--other:#3d6fd8}}
@media (prefers-color-scheme:dark){{:root{{--bg:#15181c;--card:#1e2227;--text:#e6e9ed;--weak:#9aa4b1;--line:#2e343b;--ok:#4cc27e;--bad:#ef6b6b;--accent:#f08a45;--other:#7ea2ff}}}}
body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 system-ui,sans-serif}}main{{max-width:1080px;margin:auto;padding:16px}}
h1{{margin:.2em 0}}h2{{margin:1.4em 0 .4em;border-bottom:1px solid var(--line);padding-bottom:4px}}h3{{margin:0;font-size:15px}}
.summary{{display:flex;gap:12px;flex-wrap:wrap;margin:10px 0}}.tile{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 14px}}.tile b{{font-size:22px;display:block}}
.case{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px;margin:10px 0}}.head{{display:flex;gap:10px;align-items:center}}
.badge{{font-size:11px;font-weight:700;padding:2px 7px;border-radius:4px;color:#fff}}.pass{{background:var(--ok)}}.fail{{background:var(--bad)}}.warn{{background:#b7791f}}
.meta{{color:var(--weak);margin:6px 0}}.row{{display:flex;gap:16px;flex-wrap:wrap;align-items:flex-start}}table{{border-collapse:collapse;font-variant-numeric:tabular-nums}}
td,th{{padding:3px 10px;border-bottom:1px solid var(--line);text-align:right}}th{{color:var(--weak);font-weight:600}}
.plot{{width:300px;max-width:100%}}.grid{{stroke:var(--line)}}.tol{{stroke:var(--bad);stroke-dasharray:4 3}}.line{{fill:none;stroke:var(--accent);stroke-width:2}}.dot{{fill:var(--accent)}}.other{{fill:var(--other)}}
.ax{{fill:var(--weak);font-size:10px}}.err{{color:var(--bad)}}.other-res{{color:var(--other);margin:4px 0}}.noplot{{color:var(--weak);width:300px}}
</style></head><body><main><h1>SimCraft benchmarks</h1>
<p class=meta>SimCraft {} · commit {} · {} · {} mode · {} · {} threads · {:.0} s total. Each case runs the public tool API at several mesh sizes; the plot shows the relative error against unknowns (orange: SimCraft, blue diamonds: {} when provided), the dashed line the pass tolerance.</p>
<div class=summary><div class=tile><b>{}</b>passed</div><div class=tile><b>{}</b>failed</div><div class=tile><b>{}</b>cases</div></div>
{}</main></body></html>"#,
        esc(r["simcraft_version"].as_str().unwrap_or("")),
        esc(r["commit"].as_str().unwrap_or("–")),
        esc(r["date"].as_str().unwrap_or("")),
        esc(r["mode"].as_str().unwrap_or("")),
        esc(m["cpu"].as_str().unwrap_or("unknown CPU")),
        m["threads"],
        r["seconds"].as_f64().unwrap_or(0.0),
        esc(other),
        r["passed"],
        r["failed"],
        r["cases"].as_array().map(|a| a.len()).unwrap_or(0),
        body
    )
}
