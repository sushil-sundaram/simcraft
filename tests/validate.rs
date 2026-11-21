use simcraft::{fem, geometry::*, material, mesh, model::*};

fn beam(h: f64) -> (Model, mesh::Mesh) {
    let mut m = Model::default();
    m.bodies.push(Body { name: "beam".into(), shape: Shape::Box { min: [0.0; 3], size: [100.0, 10.0, 10.0] }, op: Op::Add, material: "Structural Steel".into() });
    m.loads.push(LoadItem { id: 1, load: Load::FixedSupport { faces: vec!["beam.xmin".into()] } });
    m.loads.push(LoadItem { id: 2, load: Load::Force { faces: vec!["beam.xmax".into()], vector: [0.0, 0.0, -100.0] } });
    let me = mesh::build_mesh(&m.bodies, h, &material::library()).unwrap();
    (m, me)
}

#[test]
fn cantilever_static_and_modal() {
    let (mut m, me) = beam(2.5);
    let r = fem::solve(&m, &me, &|_, _| {}).unwrap();
    let tip = r.disp.as_ref().unwrap().iter().map(|d| d[2]).fold(0.0, f64::min);
    println!("tip deflection {tip} mm (theory -0.200 + shear); reaction {:?}; iters {}", r.reactions, r.info.iterations);
    assert!((tip + 0.2).abs() < 0.02, "tip {tip}");
    assert!((r.reactions[0].value[2] - 100.0).abs() < 0.5);
    let vmax = r.vm.as_ref().unwrap().iter().cloned().fold(0.0, f64::max);
    println!("max von Mises {vmax} MPa (beam theory at root 60 MPa)");
    m.analysis.kind = AnalysisType::Modal;
    let r = fem::solve(&m, &me, &|_, _| {}).unwrap();
    let f: Vec<f64> = r.modes.iter().map(|m| m.frequency_hz).collect();
    println!("modal freqs {f:?} (theory first bending 815 Hz)");
    assert!((f[0] - 815.0).abs() / 815.0 < 0.05);
}

#[test]
fn thermal_bar() {
    let mut m = Model::default();
    m.analysis.kind = AnalysisType::SteadyThermal;
    m.bodies.push(Body { name: "bar".into(), shape: Shape::Box { min: [0.0; 3], size: [100.0, 10.0, 10.0] }, op: Op::Add, material: "Copper C101".into() });
    m.loads.push(LoadItem { id: 1, load: Load::Temperature { faces: vec!["bar.xmin".into()], value: 100.0 } });
    m.loads.push(LoadItem { id: 2, load: Load::Temperature { faces: vec!["bar.xmax".into()], value: 0.0 } });
    let me = mesh::build_mesh(&m.bodies, 2.5, &material::library()).unwrap();
    let r = fem::solve(&m, &me, &|_, _| {}).unwrap();
    // Q = k A dT / L = 391 * 1e-4 * 100 / 0.1 = 39.1 W
    println!("heat flows {:?}", r.reactions);
    assert!((r.reactions[0].value[0] - 39.1).abs() < 0.4);
}

#[test]
fn free_thermal_expansion_no_stress() {
    let mut m = Model::default();
    m.analysis.kind = AnalysisType::ThermalStress;
    m.bodies.push(Body { name: "b".into(), shape: Shape::Box { min: [0.0; 3], size: [50.0, 10.0, 10.0] }, op: Op::Add, material: "Structural Steel".into() });
    m.loads.push(LoadItem { id: 1, load: Load::Temperature { faces: vec!["all".into()], value: 122.0 } });
    m.loads.push(LoadItem { id: 2, load: Load::Displacement { faces: vec!["b.xmin".into()], x: Some(0.0), y: None, z: None } });
    m.loads.push(LoadItem { id: 3, load: Load::Displacement { faces: vec!["b.ymin".into()], x: None, y: Some(0.0), z: None } });
    m.loads.push(LoadItem { id: 4, load: Load::Displacement { faces: vec!["b.zmin".into()], x: None, y: None, z: Some(0.0) } });
    let me = mesh::build_mesh(&m.bodies, 2.5, &material::library()).unwrap();
    let r = fem::solve(&m, &me, &|_, _| {}).unwrap();
    let ux = r.disp.as_ref().unwrap().iter().map(|d| d[0]).fold(0.0, f64::max);
    let vm = r.vm.as_ref().unwrap().iter().cloned().fold(0.0, f64::max);
    println!("ux {ux} mm (theory 0.06), vm {vm}");
    assert!((ux - 0.06).abs() < 1e-3 && vm < 1.0);
}
