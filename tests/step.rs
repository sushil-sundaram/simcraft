//! P2.3 STEP CAD import validation: B-rep accuracy, units, assemblies, elementary surfaces,
//! project round-trips, workspace sandboxing, cancellation, and malformed-input handling.

use monstertruck_core::bounding_box::BoundingBox;
use monstertruck_core::cgmath64::{Point2, Point3, Vector3};
use monstertruck_io::step::save::{CompleteStepDisplay, StepMeasurementContext, SiPrefix, StepModel};
use monstertruck_modeling::{Face, Plane, Solid, Wire, builder, primitive};
use serde_json::{Value, json};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use tempfile::tempdir;

use simcraft::geometry::{Body, Shape};
use simcraft::mesh;
use simcraft::session::{Session, Shared, external_assets, new_core};
use simcraft::step::{
    StepSurfaceAnalytic, StepSurfaceKind, detect_length_unit_mm, load_step_bodies,
};
use simcraft::study;

fn call(c: &Shared, tool: &str, args: Value) -> Value {
    let out = c
        .call(tool, &args)
        .unwrap_or_else(|e| panic!("tool '{tool}' failed: {e:#}"));
    serde_json::from_str(&out.text).unwrap_or(json!({"text": out.text}))
}

fn make_plate_with_hole_step() -> String {
    let rect_w: Wire = primitive::rect(
        BoundingBox::from_iter([Point2::new(0.0, 0.0), Point2::new(100.0, 40.0)]),
        Plane::new(
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        ),
    );
    let hole_w: Wire = primitive::circle(
        Point3::new(58.0, 20.0, 0.0),
        Point3::new(50.0, 20.0, 0.0),
        -Vector3::unit_z(),
        4,
    );
    let face: Face = builder::try_attach_plane(vec![rect_w, hole_w]).unwrap();
    let solid: Solid = builder::extrude(&face, Vector3::new(0.0, 0.0, 10.0));
    let cs = solid.compress();
    format!(
        "{}",
        CompleteStepDisplay::new(StepModel::from(&cs), Default::default())
    )
}

fn make_box_step_with_units(lx: f64, ly: f64, lz: f64, prefix: SiPrefix, inches: bool) -> String {
    let solid: Solid = primitive::cuboid(BoundingBox::from_iter([
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(lx, ly, lz),
    ]));
    let cs = solid.compress();
    let raw = format!(
        "{}",
        CompleteStepDisplay::new(
            StepModel::from(&cs).with_measurement_context(StepMeasurementContext {
                length_prefix: prefix,
                distance_accuracy_value: 1.0e-6,
            }),
            Default::default()
        )
    );
    if !inches {
        return raw;
    }
    raw.replace(
        "#12 = ( LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.) );",
        "#12 = ( CONVERSION_BASED_UNIT('INCH',#9001) LENGTH_UNIT() NAMED_UNIT(#9002) );\n\
         #9001 = LENGTH_MEASURE_WITH_UNIT(LENGTH_MEASURE(25.4),#9003);\n\
         #9002 = DIMENSIONAL_EXPONENTS(1.,0.,0.,0.,0.,0.,0.);\n\
         #9003 = ( LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.) );",
    )
}

/// Hand-crafted AP214 multi-part assembly STEP containing 3 distinct solid products
/// positioned via `NEXT_ASSEMBLY_USAGE_OCCURRENCE` and `ITEM_DEFINED_TRANSFORMATION`
/// (including a 90-degree rotation + translation so transforms are exercised strictly):
/// - `BarLeft`:   [0..40] x [0..10] x [0..10]
/// - `MidBlock`:  local [0..10] x [-20..0] x [0..10], rotated +90 deg around Z & shifted +40 X -> [40..60] x [0..10] x [0..10]
/// - `RightCap`:  local [0..40] x [0..10] x [0..10], translated +60 X -> [60..100] x [0..10] x [0..10]
fn make_ap214_assembly_step() -> String {
    // Build cleanly with sequential Entity IDs using `monstertruck-io` single solids wrapped in AP214 assembly,
    // by generating 2 cuboids and connecting them via AP214 assembly entities referencing a root product.
    let s1_sol: Solid = primitive::cuboid(BoundingBox::from_iter([Point3::new(0.0, 0.0, 0.0), Point3::new(40.0, 10.0, 10.0)]));
    let s2_sol: Solid = primitive::cuboid(BoundingBox::from_iter([Point3::new(0.0, -20.0, 0.0), Point3::new(10.0, 0.0, 10.0)]));
    let s1 = s1_sol.compress();
    let s2 = s2_sol.compress();
    let mut next_id = 10000usize;
    let mut alloc = || {
        let id = next_id;
        next_id += 1;
        id
    };

    // Extract DATA body of a StepModel and rebase entity numbers cleanly by offset.
    let rebase = |step_txt: &str, off: usize| -> (String, usize) {
        let data = step_txt.split("DATA;\n").nth(1).unwrap().split("ENDSEC;").next().unwrap();
        let mut out = String::new();
        let mut solid_id = 0usize;
        for raw_line in data.lines() {
            let line = raw_line.trim();
            if line.is_empty() {
                continue;
            }
            if line.contains("MANIFOLD_SOLID_BREP")
                && let Some(num) = line.strip_prefix('#').and_then(|s| s.split(|c: char| !c.is_ascii_digit()).next())
            {
                solid_id = num.parse::<usize>().unwrap() + off;
            }
            // Shift every `#<digits>` token
            let bytes = line.as_bytes();
            let mut i = 0;
            let mut line_out = String::new();
            while i < bytes.len() {
                if bytes[i] == b'#' {
                    let j = i + 1;
                    let mut k = j;
                    while k < bytes.len() && bytes[k].is_ascii_digit() {
                        k += 1;
                    }
                    if k > j {
                        let n: usize = line[j..k].parse().unwrap();
                        line_out.push('#');
                        line_out.push_str(&(n + off).to_string());
                        i = k;
                        continue;
                    }
                }
                line_out.push(bytes[i] as char);
                i += 1;
            }
            // Strip top-level preamble/product lines (`<= 24` in monstertruck save) from rebased blocks;
            // we emit clean shared context & explicit AP214 product graph below!
            let id_orig: usize = line
                .strip_prefix('#')
                .and_then(|s| s.split(|c: char| !c.is_ascii_digit()).next())
                .and_then(|n| n.parse().ok())
                .unwrap_or(0);
            if id_orig >= 16 {
                out.push_str(&line_out);
                out.push('\n');
            }
        }
        (out, solid_id)
    };

    let t1 = format!("{}", CompleteStepDisplay::new(StepModel::from(&s1), Default::default()));
    let t2 = format!("{}", CompleteStepDisplay::new(StepModel::from(&s2), Default::default()));
    let (b1_geom, b1_solid) = rebase(&t1, 1000);
    let (b2_geom, b2_solid) = rebase(&t2, 3000);
    let (b3_geom, b3_solid) = rebase(&t1, 5000);

    let mut data = String::from(
        "#1 = APPLICATION_CONTEXT('automotive_design');\n\
         #2 = ( GEOMETRIC_REPRESENTATION_CONTEXT(3) GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT((#6)) GLOBAL_UNIT_ASSIGNED_CONTEXT((#3,#4,#5)) REPRESENTATION_CONTEXT('','') );\n\
         #3 = ( LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.) );\n\
         #4 = ( NAMED_UNIT(*) PLANE_ANGLE_UNIT() SI_UNIT($,.RADIAN.) );\n\
         #5 = ( NAMED_UNIT(*) SI_UNIT($,.STERADIAN.) SOLID_ANGLE_UNIT() );\n\
         #6 = UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE(1.0E-6),#3,'distance_accuracy_value','');\n\
         #7 = APPLICATION_PROTOCOL_DEFINITION('international standard','automotive_design',2000,#1);\n\
         #8 = PRODUCT_CONTEXT('',#1,'mechanical');\n\
         #9 = PRODUCT_DEFINITION_CONTEXT('part definition',#1,'design');\n\
         #10 = CARTESIAN_POINT('',(0.,0.,0.));\n\
         #11 = DIRECTION('',(0.,0.,1.));\n\
         #12 = DIRECTION('',(1.,0.,0.));\n\
         #13 = AXIS2_PLACEMENT_3D('origin',#10,#11,#12);\n",
    );

    // Helper emitting a complete product node + optional solid representation.
    let mut make_part = |id: &str, name: &str, solid_opt: Option<usize>| -> (usize, usize) {
        let p = alloc();
        let pdf = alloc();
        let pd = alloc();
        let pds = alloc();
        let sr = alloc();
        let sdr = alloc();
        let items = match solid_opt {
            Some(sid) => format!("(#{sid})"),
            None => "(#13)".to_string(),
        };
        data.push_str(&format!(
            "#{p} = PRODUCT('{id}','{name}','',(#8));\n\
             #{pdf} = PRODUCT_DEFINITION_FORMATION('','',#{p});\n\
             #{pd} = PRODUCT_DEFINITION('design','',#{pdf},#9);\n\
             #{pds} = PRODUCT_DEFINITION_SHAPE('','',#{pd});\n\
             #{sr} = SHAPE_REPRESENTATION('',{items},#2);\n\
             #{sdr} = SHAPE_DEFINITION_REPRESENTATION(#{pds},#{sr});\n"
        ));
        (pd, sr)
    };

    let (root_pd, root_sr) = make_part("ROOT", "Assembly", None);
    let (p1_pd, p1_sr) = make_part("P1", "BarLeft", Some(b1_solid));
    let (p2_pd, p2_sr) = make_part("P2", "MidRotated", Some(b2_solid));
    let (p3_pd, p3_sr) = make_part("P3", "RightCap", Some(b3_solid));

    let mut link_child = |inst_name: &str, child_pd: usize, child_sr: usize, tx: f64, ty: f64, tz: f64, rot_z_90: bool| {
        let pt = alloc();
        let dz = alloc();
        let dx = alloc();
        let ax = alloc();
        let idtf = alloc();
        let srr = alloc();
        let nauo = alloc();
        let pds = alloc();
        let cdsr = alloc();
        let ref_x = if rot_z_90 { "(0.,1.,0.)" } else { "(1.,0.,0.)" };
        data.push_str(&format!(
            "#{pt} = CARTESIAN_POINT('',({tx},{ty},{tz}));\n\
             #{dz} = DIRECTION('',(0.,0.,1.));\n\
             #{dx} = DIRECTION('',{ref_x});\n\
             #{ax} = AXIS2_PLACEMENT_3D('',#{pt},#{dz},#{dx});\n\
             #{idtf} = ITEM_DEFINED_TRANSFORMATION('','',#13,#{ax});\n\
             #{srr} = ( REPRESENTATION_RELATIONSHIP('','',#{child_sr},#{root_sr}) REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION(#{idtf}) SHAPE_REPRESENTATION_RELATIONSHIP() );\n\
             #{nauo} = NEXT_ASSEMBLY_USAGE_OCCURRENCE('{inst_name}','{inst_name}','',#{root_pd},#{child_pd},$);\n\
             #{pds} = PRODUCT_DEFINITION_SHAPE('','',#{nauo});\n\
             #{cdsr} = CONTEXT_DEPENDENT_SHAPE_REPRESENTATION(#{srr},#{pds});\n"
        ));
    };

    link_child("BarLeft", p1_pd, p1_sr, 0.0, 0.0, 0.0, false);
    // Rotate [0..10] x [-20..0] by +90 deg around Z -> [0..20] x [0..10]; translate +40 X -> [40..60] x [0..10]!
    link_child("MidRotated", p2_pd, p2_sr, 40.0, 0.0, 0.0, true);
    link_child("RightCap", p3_pd, p3_sr, 60.0, 0.0, 0.0, false);

    data.push_str(&b1_geom);
    data.push_str(&b2_geom);
    data.push_str(&b3_geom);

    format!(
        "ISO-10303-21;\nHEADER;\nFILE_DESCRIPTION(('Assembly test'),'2;1');\nFILE_NAME('assy.step','2026-01-01T00:00:00',(''),(''),'','','');\nFILE_SCHEMA(('AUTOMOTIVE_DESIGN'));\nENDSEC;\nDATA;\n{data}ENDSEC;\nEND-ISO-10303-21;\n"
    )
}

/// Complete AP214 solid containing a true `CYLINDRICAL_SURFACE`, `SPHERICAL_SURFACE`,
/// or `CONICAL_SURFACE` (`r0=10 -> r1=5, h=10`) inside a closed `MANIFOLD_SOLID_BREP`!
fn make_elementary_cylinder_step() -> String {
    r#"ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('Cylinder'),'2;1');
FILE_NAME('cyl.step','2026-01-01T00:00:00',(''),(''),'','','');
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
#10 = PRODUCT('Cyl','Cyl','',(#8));
#11 = PRODUCT_DEFINITION_FORMATION('','',#10);
#12 = PRODUCT_DEFINITION('design','',#11,#9);
#13 = PRODUCT_DEFINITION_SHAPE('','',#12);
#14 = SHAPE_DEFINITION_REPRESENTATION(#13,#15);
#15 = ADVANCED_BREP_SHAPE_REPRESENTATION('',(#16),#2);
#16 = MANIFOLD_SOLID_BREP('Cyl',#17);
#17 = CLOSED_SHELL('',(#20,#30,#40));
#100 = CARTESIAN_POINT('',(10.,20.,0.));
#101 = CARTESIAN_POINT('',(10.,20.,15.));
#102 = CARTESIAN_POINT('',(15.,20.,0.));
#103 = CARTESIAN_POINT('',(15.,20.,15.));
#104 = DIRECTION('',(0.,0.,1.));
#105 = DIRECTION('',(1.,0.,0.));
#106 = DIRECTION('',(0.,0.,-1.));
#107 = AXIS2_PLACEMENT_3D('',#100,#104,#105);
#108 = AXIS2_PLACEMENT_3D('',#101,#104,#105);
#109 = AXIS2_PLACEMENT_3D('',#100,#106,#105);
#110 = VERTEX_POINT('',#102);
#111 = VERTEX_POINT('',#103);
#112 = CIRCLE('',#107,5.);
#113 = CIRCLE('',#108,5.);
#114 = VECTOR('',#104,1.);
#115 = LINE('',#102,#114);
#116 = EDGE_CURVE('',#110,#110,#112,.T.);
#117 = EDGE_CURVE('',#111,#111,#113,.T.);
#118 = EDGE_CURVE('',#110,#111,#115,.T.);
#20 = ADVANCED_FACE('',(#21),#22,.T.);
#21 = FACE_OUTER_BOUND('',#23,.T.);
#22 = CYLINDRICAL_SURFACE('',#107,5.);
#23 = EDGE_LOOP('',(#24,#25,#26,#27));
#24 = ORIENTED_EDGE('',*,*,#116,.T.);
#25 = ORIENTED_EDGE('',*,*,#118,.T.);
#26 = ORIENTED_EDGE('',*,*,#117,.F.);
#27 = ORIENTED_EDGE('',*,*,#118,.F.);
#30 = ADVANCED_FACE('',(#31),#32,.T.);
#31 = FACE_OUTER_BOUND('',#33,.T.);
#32 = PLANE('',#108);
#33 = EDGE_LOOP('',(#34));
#34 = ORIENTED_EDGE('',*,*,#117,.T.);
#40 = ADVANCED_FACE('',(#41),#42,.T.);
#41 = FACE_OUTER_BOUND('',#43,.T.);
#42 = PLANE('',#109);
#43 = EDGE_LOOP('',(#44));
#44 = ORIENTED_EDGE('',*,*,#116,.F.);
ENDSEC;
END-ISO-10303-21;
"#.to_string()
}

fn make_elementary_cone_step() -> String {
    r#"ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('Cone frustum'),'2;1');
FILE_NAME('cone.step','2026-01-01T00:00:00',(''),(''),'','','');
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
#10 = PRODUCT('Frustum','Frustum','',(#8));
#11 = PRODUCT_DEFINITION_FORMATION('','',#10);
#12 = PRODUCT_DEFINITION('design','',#11,#9);
#13 = PRODUCT_DEFINITION_SHAPE('','',#12);
#14 = SHAPE_DEFINITION_REPRESENTATION(#13,#15);
#15 = ADVANCED_BREP_SHAPE_REPRESENTATION('',(#16),#2);
#16 = MANIFOLD_SOLID_BREP('Frustum',#17);
#17 = CLOSED_SHELL('',(#20,#30,#40));
#100 = CARTESIAN_POINT('',(0.,0.,0.));
#101 = CARTESIAN_POINT('',(0.,0.,10.));
#102 = CARTESIAN_POINT('',(10.,0.,0.));
#103 = CARTESIAN_POINT('',(5.,0.,10.));
#104 = DIRECTION('',(0.,0.,1.));
#105 = DIRECTION('',(1.,0.,0.));
#106 = DIRECTION('',(0.,0.,-1.));
#107 = DIRECTION('',(-0.447213595499958,0.,0.894427190999916));
#108 = AXIS2_PLACEMENT_3D('',#100,#104,#105);
#109 = AXIS2_PLACEMENT_3D('',#101,#104,#105);
#110 = AXIS2_PLACEMENT_3D('',#100,#106,#105);
#111 = VERTEX_POINT('',#102);
#112 = VERTEX_POINT('',#103);
#113 = CIRCLE('',#108,10.);
#114 = CIRCLE('',#109,5.);
#115 = VECTOR('',#107,1.);
#116 = LINE('',#102,#115);
#117 = EDGE_CURVE('',#111,#111,#113,.T.);
#118 = EDGE_CURVE('',#112,#112,#114,.T.);
#119 = EDGE_CURVE('',#111,#112,#116,.T.);
#20 = ADVANCED_FACE('',(#21),#22,.T.);
#21 = FACE_OUTER_BOUND('',#23,.T.);
#22 = CONICAL_SURFACE('',#108,10.,-0.4636476090008061);
#23 = EDGE_LOOP('',(#24,#25,#26,#27));
#24 = ORIENTED_EDGE('',*,*,#117,.T.);
#25 = ORIENTED_EDGE('',*,*,#119,.T.);
#26 = ORIENTED_EDGE('',*,*,#118,.F.);
#27 = ORIENTED_EDGE('',*,*,#119,.F.);
#30 = ADVANCED_FACE('',(#31),#32,.T.);
#31 = FACE_OUTER_BOUND('',#33,.T.);
#32 = PLANE('',#109);
#33 = EDGE_LOOP('',(#34));
#34 = ORIENTED_EDGE('',*,*,#118,.T.);
#40 = ADVANCED_FACE('',(#41),#42,.T.);
#41 = FACE_OUTER_BOUND('',#43,.T.);
#42 = PLANE('',#110);
#43 = EDGE_LOOP('',(#44));
#44 = ORIENTED_EDGE('',*,*,#117,.F.);
ENDSEC;
END-ISO-10303-21;
"#.to_string()
}

fn make_elementary_sphere_step() -> String {
    r#"ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('Sphere'),'2;1');
FILE_NAME('sphere.step','2026-01-01T00:00:00',(''),(''),'','','');
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
#10 = PRODUCT('Ball','Ball','',(#8));
#11 = PRODUCT_DEFINITION_FORMATION('','',#10);
#12 = PRODUCT_DEFINITION('design','',#11,#9);
#13 = PRODUCT_DEFINITION_SHAPE('','',#12);
#14 = SHAPE_DEFINITION_REPRESENTATION(#13,#15);
#15 = ADVANCED_BREP_SHAPE_REPRESENTATION('',(#16),#2);
#16 = MANIFOLD_SOLID_BREP('Ball',#17);
#17 = CLOSED_SHELL('',(#20));
#100 = CARTESIAN_POINT('',(5.,-3.,8.));
#101 = DIRECTION('',(0.,0.,1.));
#102 = DIRECTION('',(1.,0.,0.));
#103 = AXIS2_PLACEMENT_3D('',#100,#101,#102);
#20 = ADVANCED_FACE('',(),#21,.T.);
#21 = SPHERICAL_SURFACE('',#103,12.);
ENDSEC;
END-ISO-10303-21;
"#.to_string()
}

#[test]
fn plate_with_hole_step_matches_analytic_volume_and_csg_kt() {
    let dir = tempdir().unwrap();
    let step_path = dir.path().join("plate_hole.step");
    std::fs::write(&step_path, make_plate_with_hole_step()).unwrap();

    // 1. STEP import & analytical volume + curved hole fitting check
    let c_step = new_core();
    let res = call(&c_step, "import_step", json!({"path": step_path, "name": "Plate"}));
    assert_eq!(res["detected_unit"], "mm");
    assert_eq!(res["body_count"], 1);
    let vol_step = res["total_volume_mm3"].as_f64().unwrap();
    let vol_exact = (100.0 * 40.0 - std::f64::consts::PI * 64.0) * 10.0;
    let rel_vol_err = (vol_step - vol_exact).abs() / vol_exact;
    assert!(
        rel_vol_err < 0.001,
        "STEP plate volume {vol_step} differs from exact {vol_exact} by {:.4}%",
        rel_vol_err * 100.0
    );

    call(&c_step, "set_mesh", json!({"element_size_mm": 2.5}));
    call(&c_step, "add_fixed_support", json!({"faces": ["bbox.xmin"]}));
    call(&c_step, "add_pressure", json!({"faces": ["bbox.xmax"], "value_mpa": -10.0}));
    let solve_step = call(&c_step, "solve", json!({}));

    // Verify boundary fitting placed hole perimeter nodes directly onto r = 8.0 mm circle!
    let (hole_nodes, max_r_err) = {
        let s = c_step.session.lock().unwrap();
        let m = s.mesh.as_ref().unwrap();
        let mut cnt = 0usize;
        let mut max_err = 0.0f64;
        for f in &m.faces {
            if f.outer {
                continue;
            }
            let cx = f.nodes.iter().map(|&n| m.nodes[n as usize][0]).sum::<f64>() / 4.0;
            let cy = f.nodes.iter().map(|&n| m.nodes[n as usize][1]).sum::<f64>() / 4.0;
            let r_c = (cx - 50.0).hypot(cy - 20.0);
            if r_c > 7.2 && r_c < 8.8 && f.center[2] > 1.0 && f.center[2] < 9.0 {
                cnt += 1;
                for &ni in &f.nodes {
                    let p = m.nodes[ni as usize];
                    let err = ((p[0] - 50.0).hypot(p[1] - 20.0) - 8.0).abs();
                    max_err = max_err.max(err);
                }
            }
        }
        (cnt, max_err)
    };
    assert!(hole_nodes >= 16, "expected fitted hole faces, found {hole_nodes}");
    assert!(
        max_r_err < 1e-6,
        "fitted STEP hole surface radial error {max_r_err:.3e} mm exceeds 1e-6 mm"
    );

    // 2. Compare max von Mises stress against SimCraft's CSG Box - Cylinder reference!
    let c_csg = new_core();
    call(&c_csg, "add_box", json!({"name": "Plate", "min": [0, 0, 0], "size": [100, 40, 10]}));
    call(&c_csg, "add_cylinder", json!({"name": "Hole", "base": [50, 20, -1], "axis": "z", "radius": 8.0, "height": 12.0, "operation": "subtract"}));
    call(&c_csg, "set_mesh", json!({"element_size_mm": 2.5}));
    call(&c_csg, "add_fixed_support", json!({"faces": ["bbox.xmin"]}));
    call(&c_csg, "add_pressure", json!({"faces": ["bbox.xmax"], "value_mpa": -10.0}));
    let solve_csg = call(&c_csg, "solve", json!({}));

    let vm_step = solve_step["fields"]["von_mises"]["max"].as_f64().unwrap();
    let vm_csg = solve_csg["fields"]["von_mises"]["max"].as_f64().unwrap();
    let rel_diff = (vm_step - vm_csg).abs() / vm_csg;
    assert!(
        rel_diff < 0.02,
        "STEP vs CSG plate-with-hole max von Mises diverged: STEP={vm_step:.3} MPa vs CSG={vm_csg:.3} MPa ({:.2}%)",
        rel_diff * 100.0
    );
}

#[test]
fn step_units_metres_and_inches_scale_exact_to_mm() {
    let dir = tempdir().unwrap();

    // Metre file: 0.060 m x 0.020 m x 0.010 m -> 60 x 20 x 10 mm
    let m_path = dir.path().join("block_m.stp");
    let m_txt = make_box_step_with_units(0.060, 0.020, 0.010, SiPrefix::None, false);
    assert!((detect_length_unit_mm(&m_txt).unwrap().0 - 1000.0).abs() < 1e-9);
    std::fs::write(&m_path, m_txt).unwrap();

    let cm = new_core();
    let rm = call(&cm, "import_step", json!({"path": m_path}));
    assert_eq!(rm["detected_unit"], "m");
    assert!((rm["unit_scale_mm"].as_f64().unwrap() - 1000.0).abs() < 1e-9);
    let b = &rm["bodies"][0]["bounds_mm"];
    assert!((b["max"][0].as_f64().unwrap() - 60.0).abs() < 1e-4);
    assert!((b["max"][1].as_f64().unwrap() - 20.0).abs() < 1e-4);
    assert!((b["max"][2].as_f64().unwrap() - 10.0).abs() < 1e-4);
    assert!((rm["total_volume_mm3"].as_f64().unwrap() - 12_000.0).abs() < 1e-2);

    // Inch file: 2.0 in x 1.0 in x 0.5 in -> 50.8 x 25.4 x 12.7 mm, plus user_scale=2 -> 101.6 x 50.8 x 25.4
    let in_path = dir.path().join("block_in.step");
    let in_txt = make_box_step_with_units(2.0, 1.0, 0.5, SiPrefix::Milli, true);
    let (usc, ulabel, _) = detect_length_unit_mm(&in_txt).unwrap();
    assert!((usc - 25.4).abs() < 1e-9);
    assert_eq!(ulabel, "inch");
    std::fs::write(&in_path, in_txt).unwrap();

    let cin = new_core();
    let rin = call(&cin, "import_step", json!({"path": in_path, "scale": 2.0}));
    let bi = &rin["bodies"][0]["bounds_mm"];
    assert!((bi["max"][0].as_f64().unwrap() - 101.6).abs() < 1e-4);
    assert!((bi["max"][1].as_f64().unwrap() - 50.8).abs() < 1e-4);
    assert!((bi["max"][2].as_f64().unwrap() - 25.4).abs() < 1e-4);
}

#[test]
fn real_ap214_assembly_preserves_parts_placements_and_bonded_continuity() {
    let dir = tempdir().unwrap();
    let assy_path = dir.path().join("bar_assembly.step");
    std::fs::write(&assy_path, make_ap214_assembly_step()).unwrap();

    let c = new_core();
    let res = call(&c, "import_step", json!({"path": assy_path}));
    assert_eq!(res["body_count"], 3, "expected 3 assembly bodies: {res}");
    let bodies = res["bodies"].as_array().unwrap();
    let names: Vec<&str> = bodies.iter().map(|b| b["name"].as_str().unwrap()).collect();
    assert_eq!(names, vec!["BarLeft", "MidRotated", "RightCap"]);

    // Verify BarLeft [0..40], MidRotated [40..60] (after 90 deg rotation + translation!), RightCap [60..100]
    let chk = |i: usize, x0: f64, x1: f64| {
        let b = &bodies[i]["bounds_mm"];
        assert!((b["min"][0].as_f64().unwrap() - x0).abs() < 1e-4, "body {i} min x: {b}");
        assert!((b["max"][0].as_f64().unwrap() - x1).abs() < 1e-4, "body {i} max x: {b}");
        assert!((b["min"][1].as_f64().unwrap() - 0.0).abs() < 1e-4, "body {i} min y: {b}");
        assert!((b["max"][1].as_f64().unwrap() - 10.0).abs() < 1e-4, "body {i} max y: {b}");
    };
    chk(0, 0.0, 40.0);
    chk(1, 40.0, 60.0);
    chk(2, 60.0, 100.0);

    // Axial bar tension across the 3 bonded STEP assembly bodies: delta = FL / (EA)
    call(&c, "set_material", json!({"material": "Structural Steel"}));
    call(&c, "set_mesh", json!({"element_size_mm": 5.0}));
    call(&c, "add_fixed_support", json!({"faces": ["bbox.xmin"]}));
    call(&c, "add_force", json!({"faces": ["bbox.xmax"], "vector": [10000.0, 0.0, 0.0]}));
    let solve = call(&c, "solve", json!({}));
    let u_max = solve["fields"]["total_deformation"]["max"].as_f64().unwrap();
    // F = 10000 N, L = 100 mm, A = 100 mm^2, E = 200 GPa = 200_000 MPa -> delta = 0.05 mm
    assert!(
        (u_max - 0.05).abs() / 0.05 < 0.03,
        "bonded 3-part STEP assembly elongation {u_max} mm vs exact 0.05 mm"
    );
}

#[test]
fn elementary_cylindrical_spherical_and_conical_step_surfaces_fit_exact() {
    let dir = tempdir().unwrap();

    // 1. Cylinder: center (10, 20, z), R = 5, z in [0, 15]
    let cyl_path = dir.path().join("cyl.step");
    std::fs::write(&cyl_path, make_elementary_cylinder_step()).unwrap();
    let cyl_parts = load_step_bodies(cyl_path.to_str().unwrap(), 1.0, [0.0; 3], None).unwrap();
    let cyl = &cyl_parts[0];
    let cyl_idx = cyl
        .faces
        .iter()
        .position(|f| f.kind == StepSurfaceKind::Cylinder)
        .expect("expected StepSurfaceKind::Cylinder");
    assert!(cyl.faces.iter().any(|f| f.kind == StepSurfaceKind::Plane));
    let q_cyl = cyl.project(cyl_idx, [14.6, 20.0, 7.5]).unwrap();
    assert!(((q_cyl[0] - 10.0).hypot(q_cyl[1] - 20.0) - 5.0).abs() < 1e-11);
    assert!((q_cyl[2] - 7.5).abs() < 1e-11);
    let exact_cyl_vol = std::f64::consts::PI * 25.0 * 15.0;
    assert!((cyl.volume_mm3 - exact_cyl_vol).abs() / exact_cyl_vol < 0.002);

    // 2. Sphere: center (5, -3, 8), R = 12
    let sph_path = dir.path().join("sphere.step");
    std::fs::write(&sph_path, make_elementary_sphere_step()).unwrap();
    let sph_parts = load_step_bodies(sph_path.to_str().unwrap(), 1.0, [0.0; 3], None).unwrap();
    let sph = &sph_parts[0];
    let sph_idx = sph
        .faces
        .iter()
        .position(|f| f.kind == StepSurfaceKind::Sphere)
        .expect("expected StepSurfaceKind::Sphere");
    let q_sph = sph.project(sph_idx, [16.5, -3.0, 11.0]).unwrap();
    let dist_sph = ((q_sph[0] - 5.0).powi(2) + (q_sph[1] + 3.0).powi(2) + (q_sph[2] - 8.0).powi(2)).sqrt();
    assert!((dist_sph - 12.0).abs() < 1e-11);
    let exact_sph_vol = 4.0 / 3.0 * std::f64::consts::PI * 12.0f64.powi(3);
    assert!((sph.volume_mm3 - exact_sph_vol).abs() / exact_sph_vol < 0.005);

    // 3. Cone frustum: axis Z, r(0)=10 -> r(10)=5 (half-angle tan_a = 0.5)
    let cone_path = dir.path().join("cone.step");
    std::fs::write(&cone_path, make_elementary_cone_step()).unwrap();
    let cone_parts = load_step_bodies(cone_path.to_str().unwrap(), 1.0, [0.0; 3], None).unwrap();
    let cone = &cone_parts[0];
    let cone_idx = cone
        .faces
        .iter()
        .position(|f| f.kind == StepSurfaceKind::Cone)
        .expect("expected StepSurfaceKind::Cone");
    let q_cone = cone.project(cone_idx, [8.2, 0.0, 4.2]).unwrap();
    let expected_r_at_z = 10.0 - 0.5 * q_cone[2];
    let actual_r = q_cone[0].hypot(q_cone[1]);
    assert!((actual_r - expected_r_at_z).abs() < 1e-11);
    let exact_cone_vol = std::f64::consts::PI * 10.0 / 3.0 * (100.0 + 50.0 + 25.0);
    assert!((cone.volume_mm3 - exact_cone_vol).abs() / exact_cone_vol < 0.002);

    // 4. Analytic torus projection fixed-point check
    let torus = StepSurfaceAnalytic::Torus {
        center: [0.0, 0.0, 0.0],
        axis: [0.0, 0.0, 1.0],
        major_r: 20.0,
        minor_r: 4.0,
        outward: 1.0,
    };
    let q_tor = torus.project([23.0, 0.0, 2.0]).unwrap();
    let tube_d = (q_tor[0].hypot(q_tor[1]) - 20.0).hypot(q_tor[2]);
    assert!((tube_d - 4.0).abs() < 1e-12);
}

#[test]
fn step_round_trip_project_save_open_edit_transform_and_undo() {
    let dir = tempdir().unwrap();
    let ws = dir.path().join("ws");
    let ext = dir.path().join("outside");
    std::fs::create_dir_all(&ws).unwrap();
    std::fs::create_dir_all(&ext).unwrap();

    let step_in = ws.join("part.step");
    let step_outside = ext.join("other.step");
    let txt = make_plate_with_hole_step();
    std::fs::write(&step_in, &txt).unwrap();
    std::fs::write(&step_outside, &txt).unwrap();

    let c = new_core();
    c.set_workspace(Some(ws.clone()));
    call(&c, "import_step", json!({"path": step_in, "name": "Part"}));

    // Edit offset + transform_body move + copy + undo/redo
    call(&c, "edit_body", json!({"name": "Part", "offset": [10.0, 0.0, 0.0]}));
    call(&c, "transform_body", json!({"name": "Part", "mode": "move", "translate": [5.0, 0.0, 0.0]}));
    call(&c, "transform_body", json!({"name": "Part", "mode": "copy", "translate": [0.0, 50.0, 0.0]}));
    let model_now = call(&c, "get_model", json!({}));
    assert_eq!(model_now["bodies"].as_array().unwrap().len(), 2);
    call(&c, "undo", json!({}));
    let model_undo = call(&c, "get_model", json!({}));
    assert_eq!(model_undo["bodies"].as_array().unwrap().len(), 1);
    assert!((model_undo["bodies"][0]["shape"]["offset"][0].as_f64().unwrap() - 15.0).abs() < 1e-9);

    // Project save + reload inside workspace restores StepData cleanly
    let proj_path = ws.join("plate.simcraft");
    call(&c, "save_project", json!({"path": proj_path}));
    assert!(external_assets(proj_path.to_str().unwrap()).unwrap().is_empty());

    let c2 = new_core();
    c2.set_workspace(Some(ws.clone()));
    call(&c2, "open_project", json!({"path": proj_path}));
    let s2 = c2.session.lock().unwrap();
    assert!(matches!(&s2.model.bodies[0].shape, Shape::Step { data: Some(_), .. }));
    // Parameter study apply preserves & translates/reloads StepData cleanly
    let p_target = study::Parameter {
        target: "Part.offset.z".into(),
        min: 0.0,
        max: 20.0,
        steps: None,
    };
    let applied = study::apply(&s2.model, &[p_target], &[12.0]).unwrap();
    assert!((applied.bodies[0].shape.bounds().0[2] - 12.0).abs() < 1e-6);
    drop(s2);

    // Project referencing STEP outside its folder flags external_assets!
    let mut s_ext = Session::default();
    let d_ext = Arc::new(load_step_bodies(step_outside.to_str().unwrap(), 1.0, [0.0; 3], None).unwrap().swap_remove(0));
    s_ext.model.bodies.push(Body {
        name: "Ext".into(),
        shape: Shape::Step {
            path: step_outside.to_string_lossy().into(),
            part_index: 0,
            part_name: "Ext".into(),
            scale: 1.0,
            offset: [0.0; 3],
            data: Some(d_ext),
        },
        op: Default::default(),
        material: simcraft::geometry::default_material(),
        role: Default::default(),
    });
    let ext_proj = ws.join("untrusted.simcraft");
    s_ext.save(ext_proj.to_str().unwrap()).unwrap();
    let ext_list = external_assets(ext_proj.to_str().unwrap()).unwrap();
    assert_eq!(ext_list.len(), 1);
    assert!(ext_list[0].contains("other.step"));

    // Sanity check mesh builds from loaded StepData
    assert!(mesh::build_for(&s_ext.model, 5.0, 200_000).is_ok());
}

#[test]
fn step_security_sandbox_extensions_and_corrupt_input_safety() {
    let dir = tempdir().unwrap();
    let ws = dir.path().join("ws");
    let secret = dir.path().join("secret.step");
    std::fs::create_dir_all(&ws).unwrap();
    std::fs::write(&secret, make_plate_with_hole_step()).unwrap();

    let c = new_core();
    c.set_workspace(Some(ws.clone()));

    // 1. Path traversal outside SIMCRAFT_WORKSPACE blocked
    assert!(c.call("import_step", &json!({"path": "../secret.step"})).is_err());

    // 2. Wrong extension rejected
    let bad_ext = ws.join("not_step.stl");
    std::fs::write(&bad_ext, make_plate_with_hole_step()).unwrap();
    assert!(c.call("import_step", &json!({"path": bad_ext})).is_err());

    // 3. Empty / truncated / garbage / NaN STEP inputs error cleanly (never panic)
    for (name, body) in [
        ("empty.step", ""),
        ("garbage.stp", "not a step file at all"),
        ("truncated.step", "ISO-10303-21;\nHEADER;\nENDSEC;\nDATA;\nENDSEC;\nEND-ISO-10303-21;\n"),
        ("nan.step", "ISO-10303-21;\nHEADER;\nENDSEC;\nDATA;\n#1 = CARTESIAN_POINT('',(NaN,0.,0.));\nENDSEC;\nEND-ISO-10303-21;\n"),
    ] {
        let p = ws.join(name);
        std::fs::write(&p, body).unwrap();
        assert!(c.call("import_step", &json!({"path": p})).is_err(), "{name} should fail cleanly");
    }

    // 4. Negative or NaN scale rejected
    let ok_step = ws.join("ok.step");
    std::fs::write(&ok_step, make_plate_with_hole_step()).unwrap();
    assert!(c.call("import_step", &json!({"path": ok_step, "scale": 0.0})).is_err());
    assert!(c.call("import_step", &json!({"path": ok_step, "scale": -2.0})).is_err());

    // 5. Cancellation flag aborts loading cleanly
    let cancel = AtomicBool::new(true);
    let cancelled = load_step_bodies(ok_step.to_str().unwrap(), 1.0, [0.0; 3], Some(&cancel));
    assert!(cancelled.is_err());
}
