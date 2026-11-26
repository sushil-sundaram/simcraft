//! Right Properties panel: collapsible header, selected face/body summary, unified unit-aware
//! primitive & load forms (used for both creation and editing, replacing raw JSON editing).

use super::theme::{UnitSystem, card, drag_scaled, primary, xyz, xyz_scaled};
use super::{BodyDraft, EditTarget, Form, Forms, LoadDraft, LoadKind, SimApp};
use crate::geometry::{Axis, Op, Role, Shape, V3};
use crate::mesh::Mesh;
use crate::model::{AnalysisType, CoilMode, Load};
use crate::session::Shared;
use eframe::egui::{self, RichText};
use serde_json::{Value, json};
use std::sync::Arc;

fn load_to_draft(id: u32, load: &Load) -> LoadDraft {
    let mut f = Forms::default();
    let mut faces = load.faces().to_vec();
    let mut body = String::new();
    let kind = match load {
        Load::FixedSupport { .. } => LoadKind::Fixed,
        Load::Displacement { x, y, z, .. } => {
            f.dcomp = [
                (x.is_some(), x.unwrap_or(0.0)),
                (y.is_some(), y.unwrap_or(0.0)),
                (z.is_some(), z.unwrap_or(0.0)),
            ];
            LoadKind::Displacement
        }
        Load::CompressionOnly { .. } => LoadKind::CompressionOnly,
        Load::Symmetry { .. } => LoadKind::Symmetry,
        Load::Force { vector, .. } => {
            f.vector = *vector;
            LoadKind::Force
        }
        Load::Pressure { value_mpa, .. } => {
            f.value = *value_mpa;
            LoadKind::Pressure
        }
        Load::Gravity { acceleration } => {
            f.grav = *acceleration;
            LoadKind::Gravity
        }
        Load::Temperature { value, .. } => {
            f.value = *value;
            LoadKind::Temperature
        }
        Load::HeatFlux { value, .. } => {
            f.value = *value;
            LoadKind::HeatFlux
        }
        Load::Convection { coefficient, ambient, .. } => {
            f.coef = *coefficient;
            f.ambient = *ambient;
            LoadKind::Convection
        }
        Load::Radiation { emissivity, ambient, enclosure, .. } => {
            f.rad_eps = *emissivity;
            f.ambient = *ambient;
            f.rad_enclosure = *enclosure;
            LoadKind::Radiation
        }
        Load::HeatGeneration { value, body: b, power } => {
            f.heat_watts = power.is_some();
            f.heat_power = power.unwrap_or(5.0);
            f.value = *value;
            f.heat_body = b.clone().unwrap_or_default();
            LoadKind::HeatGen
        }
        Load::Voltage { value, .. } => {
            f.value = *value;
            LoadKind::Voltage
        }
        Load::Current { value, .. } => {
            f.value = *value;
            LoadKind::Current
        }
        Load::Coil { body: b, current_density, axis, mode, .. } => {
            body = b.clone();
            f.value = *current_density;
            f.coil_axis = axis.idx();
            f.coil_axial = *mode == CoilMode::Axial;
            LoadKind::Coil
        }
        Load::Magnet { body: b, remanence } => {
            body = b.clone();
            f.remanence = *remanence;
            LoadKind::Magnet
        }
        Load::Inlet { velocity, temperature, .. } => {
            f.velocity = *velocity;
            if let Some(t) = temperature {
                f.ambient = *t;
            }
            LoadKind::Inlet
        }
        Load::MovingWall { velocity, .. } => {
            f.wall_v = *velocity;
            LoadKind::MovingWall
        }
        Load::Outlet { pressure, .. } => {
            f.outlet_p = *pressure;
            LoadKind::Outlet
        }
        Load::Port { power, polarization, profile, .. } => {
            f.port_power = *power;
            f.port_pol = polarization.idx();
            f.port_te10 = profile != "uniform";
            LoadKind::Port
        }
    };
    faces.shrink_to_fit();
    LoadDraft { id, kind, f, faces, body }
}

fn load_from_draft(k: LoadKind, f: &Forms, faces: Vec<String>, body: &str, is_cht: bool) -> Load {
    let ax = |i: usize| [Axis::X, Axis::Y, Axis::Z][i.min(2)];
    match k {
        LoadKind::Fixed => Load::FixedSupport { faces },
        LoadKind::Displacement => Load::Displacement {
            faces,
            x: f.dcomp[0].0.then_some(f.dcomp[0].1),
            y: f.dcomp[1].0.then_some(f.dcomp[1].1),
            z: f.dcomp[2].0.then_some(f.dcomp[2].1),
        },
        LoadKind::CompressionOnly => Load::CompressionOnly { faces },
        LoadKind::Symmetry => Load::Symmetry { faces },
        LoadKind::Force => Load::Force { faces, vector: f.vector },
        LoadKind::Pressure => Load::Pressure { faces, value_mpa: f.value },
        LoadKind::Gravity => Load::Gravity { acceleration: f.grav },
        LoadKind::Temperature => Load::Temperature { faces, value: f.value },
        LoadKind::HeatFlux => Load::HeatFlux { faces, value: f.value },
        LoadKind::Convection => Load::Convection { faces, coefficient: f.coef, ambient: f.ambient },
        LoadKind::Radiation => Load::Radiation { faces, emissivity: f.rad_eps, ambient: f.ambient, enclosure: f.rad_enclosure },
        LoadKind::HeatGen => Load::HeatGeneration {
            value: if f.heat_watts { 0.0 } else { f.value },
            body: (!f.heat_body.trim().is_empty()).then(|| f.heat_body.clone()),
            power: f.heat_watts.then_some(f.heat_power),
        },
        LoadKind::Voltage => Load::Voltage { faces, value: f.value },
        LoadKind::Current => Load::Current { faces, value: f.value },
        LoadKind::Coil => Load::Coil {
            body: body.to_string(),
            current_density: f.value,
            axis: ax(f.coil_axis),
            mode: if f.coil_axial { CoilMode::Axial } else { CoilMode::Circumferential },
            center: None,
        },
        LoadKind::Magnet => Load::Magnet { body: body.to_string(), remanence: f.remanence },
        LoadKind::Inlet => Load::Inlet { faces, velocity: f.velocity, temperature: is_cht.then_some(f.ambient) },
        LoadKind::MovingWall => Load::MovingWall { faces, velocity: f.wall_v },
        LoadKind::Outlet => Load::Outlet { faces, pressure: f.outlet_p },
        LoadKind::Port => Load::Port {
            faces,
            power: f.port_power,
            polarization: ax(f.port_pol),
            profile: if f.port_te10 { "te10" } else { "uniform" }.into(),
        },
    }
}

fn load_tool_args(k: LoadKind, f: &Forms, sel: &[String], body: &str, is_cht: bool) -> (&'static str, Value) {
    let fa = json!(sel);
    match k {
        LoadKind::Fixed => ("add_fixed_support", json!({"faces": fa})),
        LoadKind::Displacement => {
            let mut a = json!({"faces": fa});
            for (i, c) in ["x", "y", "z"].iter().enumerate() {
                if f.dcomp[i].0 {
                    a[*c] = json!(f.dcomp[i].1);
                }
            }
            ("add_displacement", a)
        }
        LoadKind::Force => ("add_force", json!({"faces": fa, "vector": f.vector})),
        LoadKind::Pressure => ("add_pressure", json!({"faces": fa, "value_mpa": f.value})),
        LoadKind::Gravity => ("add_gravity", json!({"acceleration": f.grav})),
        LoadKind::Temperature => ("add_temperature", json!({"faces": fa, "value": f.value})),
        LoadKind::HeatFlux => ("add_heat_flux", json!({"faces": fa, "value": f.value})),
        LoadKind::Convection => ("add_convection", json!({"faces": fa, "coefficient": f.coef, "ambient": f.ambient})),
        LoadKind::Radiation => ("add_radiation", json!({"faces": fa, "emissivity": f.rad_eps, "ambient": f.ambient, "enclosure": f.rad_enclosure})),
        LoadKind::HeatGen => {
            if f.heat_watts {
                ("add_heat_generation", json!({"power": f.heat_power, "body": f.heat_body}))
            } else {
                ("add_heat_generation", json!({"value": f.value, "body": f.heat_body}))
            }
        }
        LoadKind::CompressionOnly => ("add_compression_only_support", json!({"faces": fa})),
        LoadKind::Symmetry => ("add_symmetry", json!({"faces": fa})),
        LoadKind::Voltage => ("add_voltage", json!({"faces": fa, "value": f.value})),
        LoadKind::Current => ("add_current", json!({"faces": fa, "value": f.value})),
        LoadKind::Coil => ("add_coil", json!({"body": body, "current_density": f.value, "axis": (["x", "y", "z"])[f.coil_axis], "mode": (if f.coil_axial { "axial" } else { "circumferential" })})),
        LoadKind::Magnet => ("add_magnet", json!({"body": body, "remanence": f.remanence})),
        LoadKind::Inlet => {
            let mut a = json!({"faces": fa, "velocity": f.velocity});
            if is_cht {
                a["temperature"] = json!(f.ambient);
            }
            ("add_inlet", a)
        }
        LoadKind::MovingWall => ("add_moving_wall", json!({"faces": fa, "velocity": f.wall_v})),
        LoadKind::Outlet => ("add_outlet", json!({"faces": fa, "pressure": f.outlet_p})),
        LoadKind::Port => ("add_port", json!({"faces": fa, "power": f.port_power, "polarization": (["x", "y", "z"])[f.port_pol], "profile": if f.port_te10 { "te10" } else { "uniform" }})),
    }
}

fn load_definition_ui(ui: &mut egui::Ui, p: &super::theme::Pal, units: UnitSystem, core: &Shared, k: LoadKind, f: &mut Forms, id_salt: &str) {
    match k {
        LoadKind::Fixed => {
            ui.label(RichText::new("Fixes all translations on the faces.").color(p.weak));
        }
        LoadKind::Displacement => {
            for (i, c) in ["X", "Y", "Z"].iter().enumerate() {
                ui.horizontal(|ui| {
                    ui.checkbox(&mut f.dcomp[i].0, *c);
                    if f.dcomp[i].0 {
                        drag_scaled(ui, &mut f.dcomp[i].1, units.len_scale(), units.len_suffix(), 0.01, None);
                    } else {
                        ui.label(RichText::new("free").color(p.weak));
                    }
                });
            }
        }
        LoadKind::Force => {
            ui.label(RichText::new(format!("Total force ({})", units.force_suffix().trim())).small().color(p.weak));
            xyz_scaled(ui, &format!("{id_salt}-force"), &mut f.vector, units.force_scale(), "", 1.0);
        }
        LoadKind::Pressure => {
            drag_scaled(ui, &mut f.value, units.stress_scale(), units.stress_suffix(), 0.01, None);
            ui.label(RichText::new("Positive pushes into the surface.").small().color(p.weak));
        }
        LoadKind::Gravity => {
            ui.label(RichText::new("Acceleration (m/s²)").small().color(p.weak));
            xyz(ui, &format!("{id_salt}-grav"), &mut f.grav, "", 0.1);
        }
        LoadKind::Temperature => {
            ui.add(egui::DragValue::new(&mut f.value).speed(0.5).suffix(" °C"));
        }
        LoadKind::HeatFlux => {
            ui.add(egui::DragValue::new(&mut f.value).speed(10.0).suffix(" W/m²"));
            ui.label(RichText::new("Positive flows into the body.").small().color(p.weak));
        }
        LoadKind::Convection => {
            egui::Grid::new(format!("{id_salt}-conv")).num_columns(2).show(ui, |ui| {
                ui.label("Film coeff.");
                ui.add(egui::DragValue::new(&mut f.coef).speed(0.5).range(0.0..=1e6).suffix(" W/m²K"));
                ui.end_row();
                ui.label("Ambient");
                ui.add(egui::DragValue::new(&mut f.ambient).speed(0.5).suffix(" °C"));
                ui.end_row();
            });
        }
        LoadKind::Radiation => {
            egui::Grid::new(format!("{id_salt}-rad")).num_columns(2).show(ui, |ui| {
                ui.label("Emissivity ε");
                ui.add(egui::DragValue::new(&mut f.rad_eps).speed(0.01).range(0.01..=1.0))
                    .on_hover_text("Polished aluminium ≈ 0.05, oxidised metal 0.3–0.6, anodised / painted / FR-4 0.8–0.95");
                ui.end_row();
                ui.label("Surroundings");
                ui.add(egui::DragValue::new(&mut f.ambient).speed(0.5).suffix(" °C"));
                ui.end_row();
            });
            ui.checkbox(&mut f.rad_enclosure, "Faces see each other (enclosure)")
                .on_hover_text("Surface-to-surface view-factor exchange inside an enclosure.");
            ui.label(RichText::new("Non-linear: the solver iterates.").small().color(p.weak));
        }
        LoadKind::HeatGen => {
            ui.horizontal(|ui| {
                ui.selectable_value(&mut f.heat_watts, true, "Total power");
                ui.selectable_value(&mut f.heat_watts, false, "Power density");
            });
            if f.heat_watts {
                ui.add(egui::DragValue::new(&mut f.heat_power).speed(0.1).range(0.0..=1e9).suffix(" W"));
            } else {
                ui.add(egui::DragValue::new(&mut f.value).speed(1000.0).suffix(" W/m³"));
            }
            let bodies: Vec<String> = core
                .session
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .model
                .bodies
                .iter()
                .filter(|b| b.role == Role::Solid && b.op == Op::Add)
                .map(|b| b.name.clone())
                .collect();
            ui.horizontal(|ui| {
                ui.label("In");
                let shown = if f.heat_body.is_empty() { "Whole model".to_string() } else { f.heat_body.clone() };
                egui::ComboBox::from_id_salt(format!("{id_salt}-hb")).width(170.0).selected_text(shown).show_ui(ui, |ui| {
                    ui.selectable_value(&mut f.heat_body, String::new(), "Whole model");
                    for b in &bodies {
                        ui.selectable_value(&mut f.heat_body, b.clone(), b);
                    }
                });
            });
            ui.label(
                RichText::new(if f.heat_watts { "Spread evenly over the body's volume." } else { "Power [W] = value × volume." })
                    .small()
                    .color(p.weak),
            );
        }
        LoadKind::CompressionOnly => {
            ui.label(RichText::new("Rests on a rigid surface: pushes resisted, tension lifts off.").color(p.weak));
        }
        LoadKind::Symmetry => {
            ui.label(RichText::new("Structural: zero normal motion. Fluid: free-slip wall.").color(p.weak));
        }
        LoadKind::Voltage => {
            ui.add(egui::DragValue::new(&mut f.value).speed(0.1).suffix(" V"));
        }
        LoadKind::Current => {
            ui.add(egui::DragValue::new(&mut f.value).speed(0.1).suffix(" A"));
            ui.label(RichText::new("Enters the body through the faces; ground another face with 0 V.").small().color(p.weak));
        }
        LoadKind::Coil => {
            ui.add(egui::DragValue::new(&mut f.value).speed(0.1).suffix(" A/mm²"));
            ui.horizontal(|ui| {
                ui.label("Axis");
                for (i, a) in ["X", "Y", "Z"].iter().enumerate() {
                    ui.selectable_value(&mut f.coil_axis, i, *a);
                }
            });
            ui.horizontal(|ui| {
                ui.selectable_value(&mut f.coil_axial, false, "Around axis (winding)");
                ui.selectable_value(&mut f.coil_axial, true, "Along axis (bar)");
            });
        }
        LoadKind::Magnet => {
            ui.label(RichText::new("Remanence Br (T)").small().color(p.weak));
            xyz(ui, &format!("{id_salt}-br"), &mut f.remanence, "", 0.05);
        }
        LoadKind::Inlet => {
            ui.add(egui::DragValue::new(&mut f.velocity).speed(0.01).suffix(" m/s"));
            ui.label(RichText::new("Uniform velocity into the domain.").small().color(p.weak));
            if core.session.lock().unwrap_or_else(std::sync::PoisonError::into_inner).model.analysis.kind == AnalysisType::ConjugateHeatTransfer {
                ui.horizontal(|ui| {
                    ui.label("Inflow temp.");
                    ui.add(egui::DragValue::new(&mut f.ambient).speed(0.5).suffix(" °C"));
                });
            }
        }
        LoadKind::MovingWall => {
            ui.label(RichText::new("Wall velocity (m/s), tangential to the faces").small().color(p.weak));
            xyz(ui, &format!("{id_salt}-mw"), &mut f.wall_v, " m/s", 0.01);
        }
        LoadKind::Outlet => {
            ui.add(egui::DragValue::new(&mut f.outlet_p).speed(1.0).suffix(" Pa"));
        }
        LoadKind::Port => {
            ui.add(egui::DragValue::new(&mut f.port_power).speed(5.0).range(0.001..=1e7).suffix(" W"));
            ui.horizontal(|ui| {
                ui.label("E-field along");
                for (i, a) in ["X", "Y", "Z"].iter().enumerate() {
                    ui.selectable_value(&mut f.port_pol, i, *a);
                }
            });
            ui.checkbox(&mut f.port_te10, "Waveguide (TE10) profile");
            ui.label(RichText::new("Pick an air/dielectric opening; E-field axis must lie in the face.").small().color(p.weak));
        }
    }
}

fn first_selected_centroid(mesh: Option<&Arc<Mesh>>, selected: &std::collections::HashSet<u32>) -> Option<V3> {
    let m = mesh?;
    let p = selected.iter().filter_map(|&i| m.patches.get(i as usize)).next()?;
    Some(p.centroid)
}

impl SimApp {
    pub(super) fn item_editor(&mut self, ui: &mut egui::Ui, mesh: Option<&Arc<Mesh>>) {
        let p = self.p;
        let units = self.units;
        let (model, lib, hidden, ghost) = {
            let s = self.core.session.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            (
                s.model.clone(),
                s.model.materials().into_iter().map(|m| m.name).collect::<Vec<_>>(),
                s.display.hidden_bodies.clone(),
                s.display.ghost_bodies.clone(),
            )
        };
        let (key, fresh_target): (String, EditTarget) = if let Some(n) = &self.sel_body {
            let Some(b) = model.bodies.iter().find(|b| &b.name == n) else {
                self.sel_body = None;
                return;
            };
            (
                format!("b:{n}"),
                EditTarget::Body(BodyDraft {
                    orig_name: b.name.clone(),
                    name: b.name.clone(),
                    op: b.op,
                    role: b.role,
                    material: b.material.clone(),
                    shape: b.shape.clone(),
                }),
            )
        } else {
            let id = self.sel_load.unwrap();
            let Some(l) = model.loads.iter().find(|l| l.id == id) else {
                self.sel_load = None;
                return;
            };
            (format!("l:{id}"), EditTarget::Load(load_to_draft(id, &l.load)))
        };

        if !matches!(&self.edit_target, Some((k, _)) if *k == key) {
            self.edit_target = Some((key.clone(), fresh_target.clone()));
        }

        let sel_faces = self.selected_names(mesh);
        let snap_pt = first_selected_centroid(mesh, &self.selected);
        let mut apply_body: Option<BodyDraft> = None;
        let mut apply_load: Option<LoadDraft> = None;
        let mut action: Option<(&str, Value)> = None;
        let mut vis_action: Option<(String, Option<u8>)> = None; // Some(0/1/2) or None for isolate
        let mut close_editor = false;

        card(ui, &p, |ui| {
            let (_, target) = self.edit_target.as_mut().unwrap();
            match (target, &fresh_target) {
                (EditTarget::Body(bd), EditTarget::Body(orig)) => {
                    ui.horizontal(|ui| {
                        ui.label(RichText::new(format!("Body · {}", orig.orig_name)).strong());
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            let is_h = hidden.contains(&orig.orig_name);
                            let is_g = ghost.contains(&orig.orig_name);
                            if ui.selectable_label(is_h, "🚫").on_hover_text("Hide body").clicked() {
                                vis_action = Some((orig.orig_name.clone(), Some(if is_h { 0 } else { 2 })));
                            }
                            if ui.selectable_label(is_g, "👻").on_hover_text("Ghost (translucent)").clicked() {
                                vis_action = Some((orig.orig_name.clone(), Some(if is_g { 0 } else { 1 })));
                            }
                            if model.bodies.len() > 1 && ui.small_button("Isolate").clicked() {
                                vis_action = Some((orig.orig_name.clone(), None));
                            }
                        });
                    });
                    ui.add_space(4.0);
                    ui.horizontal(|ui| {
                        ui.label("Name");
                        ui.add(egui::TextEdit::singleline(&mut bd.name).desired_width(f32::INFINITY));
                    });
                    ui.horizontal(|ui| {
                        ui.label("Material");
                        egui::ComboBox::from_id_salt("eb-mat").width(180.0).selected_text(&bd.material).show_ui(ui, |ui| {
                            for m in &lib {
                                ui.selectable_value(&mut bd.material, m.clone(), m);
                            }
                        });
                    });
                    if bd.role == Role::Solid {
                        ui.horizontal(|ui| {
                            ui.label("Operation");
                            ui.selectable_value(&mut bd.op, Op::Add, "Add (solid)");
                            ui.selectable_value(&mut bd.op, Op::Subtract, "Subtract (cut)");
                        });
                    }
                    ui.separator();
                    let ls = units.len_scale();
                    let lsf = units.len_suffix();
                    match &mut bd.shape {
                        Shape::Box { min, size } => {
                            ui.label(RichText::new(format!("Corner ({})", lsf.trim())).small().color(p.weak));
                            xyz_scaled(ui, "eb-bmin", min, ls, "", 0.5);
                            if let Some(c) = snap_pt {
                                if ui.small_button("📍 Snap corner to selected face center").clicked() {
                                    *min = c;
                                }
                            }
                            ui.label(RichText::new(format!("Size ({})", lsf.trim())).small().color(p.weak));
                            xyz_scaled(ui, "eb-bsize", size, ls, "", 0.5);
                        }
                        Shape::Cylinder { base, axis, radius, height } => {
                            ui.label(RichText::new(format!("Base centre ({})", lsf.trim())).small().color(p.weak));
                            xyz_scaled(ui, "eb-cbase", base, ls, "", 0.5);
                            if let Some(c) = snap_pt {
                                if ui.small_button("📍 Snap base to selected face center").clicked() {
                                    *base = c;
                                }
                            }
                            ui.horizontal(|ui| {
                                ui.label("Axis");
                                for a in [Axis::X, Axis::Y, Axis::Z] {
                                    ui.selectable_value(axis, a, format!("{a:?}"));
                                }
                            });
                            egui::Grid::new("eb-cyl").num_columns(2).show(ui, |ui| {
                                ui.label("Radius");
                                drag_scaled(ui, radius, ls, lsf, 0.2, Some((0.01, 1e6)));
                                ui.end_row();
                                ui.label("Height");
                                drag_scaled(ui, height, ls, lsf, 0.5, Some((0.01, 1e6)));
                                ui.end_row();
                            });
                        }
                        Shape::Sphere { center, radius } => {
                            ui.label(RichText::new(format!("Centre ({})", lsf.trim())).small().color(p.weak));
                            xyz_scaled(ui, "eb-sc", center, ls, "", 0.5);
                            if let Some(c) = snap_pt {
                                if ui.small_button("📍 Snap centre to selected face center").clicked() {
                                    *center = c;
                                }
                            }
                            ui.horizontal(|ui| {
                                ui.label("Radius");
                                drag_scaled(ui, radius, ls, lsf, 0.2, Some((0.01, 1e6)));
                            });
                        }
                        Shape::Stl { path, scale, offset, .. } => {
                            let fname = std::path::Path::new(path).file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| path.clone());
                            ui.label(RichText::new(format!("STL: {fname}")).small().color(p.weak));
                            ui.horizontal(|ui| {
                                ui.label("Scale");
                                ui.add(egui::DragValue::new(scale).speed(0.01).range(1e-4..=1e4));
                            });
                            ui.label(RichText::new(format!("Offset ({})", lsf.trim())).small().color(p.weak));
                            xyz_scaled(ui, "eb-soff", offset, ls, "", 0.5);
                        }
                        Shape::Pcb { path, offset, .. } => {
                            let fname = std::path::Path::new(path).file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| path.clone());
                            ui.label(RichText::new(format!("KiCad PCB: {fname}")).small().color(p.weak));
                            ui.label(RichText::new(format!("Offset ({})", lsf.trim())).small().color(p.weak));
                            xyz_scaled(ui, "eb-poff", offset, ls, "", 0.5);
                        }
                    }
                    ui.label(RichText::new("Tip: drag X/Y/Z arrows in the 3D viewport to move.").small().color(p.weak));
                    let differs = bd.name != orig.name || bd.op != orig.op || bd.material != orig.material || serde_json::to_value(&bd.shape).ok() != serde_json::to_value(&orig.shape).ok();
                    ui.add_space(6.0);
                    ui.horizontal(|ui| {
                        if ui.add_enabled(differs, primary(&p, "Apply")).clicked() {
                            apply_body = Some(bd.clone());
                        }
                        if ui.add_enabled(differs, egui::Button::new("Revert")).clicked() {
                            *bd = orig.clone();
                        }
                        if ui.button("Close").clicked() {
                            close_editor = true;
                        }
                    });
                }
                (EditTarget::Load(ld), EditTarget::Load(orig)) => {
                    ui.horizontal(|ui| {
                        ui.label(RichText::new(ld.kind.icon()).size(16.0).color(ld.kind.color()));
                        ui.label(RichText::new(format!("Load #{} · {}", ld.id, ld.kind.title())).strong());
                    });
                    ui.add_space(4.0);
                    if ld.kind.needs_body() {
                        let bodies: Vec<String> = model.bodies.iter().filter(|b| b.role == Role::Solid && b.op == Op::Add).map(|b| b.name.clone()).collect();
                        ui.horizontal(|ui| {
                            ui.label("Body");
                            egui::ComboBox::from_id_salt("el-body").width(180.0).selected_text(&ld.body).show_ui(ui, |ui| {
                                for b in &bodies {
                                    ui.selectable_value(&mut ld.body, b.clone(), b);
                                }
                            });
                        });
                    }
                    if ld.kind.needs_faces() {
                        ui.label(RichText::new("Scoped faces").small().strong().color(p.weak));
                        if ld.faces.is_empty() {
                            ui.label(RichText::new("No faces — pick faces in 3D and click '+ Add selected'.").small().color(p.warn));
                        } else {
                            let mut rm = None;
                            ui.horizontal_wrapped(|ui| {
                                for (idx, fnm) in ld.faces.iter().enumerate() {
                                    if ui.small_button(format!("{fnm}  ✖")).clicked() {
                                        rm = Some(idx);
                                    }
                                }
                            });
                            if let Some(i) = rm {
                                ld.faces.remove(i);
                            }
                        }
                        if !sel_faces.is_empty() {
                            ui.horizontal(|ui| {
                                if ui.small_button(format!("+ Add {} selected", sel_faces.len())).clicked() {
                                    for s in &sel_faces {
                                        if !ld.faces.contains(s) {
                                            ld.faces.push(s.clone());
                                        }
                                    }
                                }
                                if ui.small_button("Replace with selected").clicked() {
                                    ld.faces = sel_faces.clone();
                                }
                            });
                        }
                    }
                    ui.separator();
                    load_definition_ui(ui, &p, units, &self.core, ld.kind, &mut ld.f, "el");
                    let differs = ld.faces != orig.faces || ld.body != orig.body || ld.f != orig.f;
                    let valid = (!ld.kind.needs_faces() || !ld.faces.is_empty()) && (!ld.kind.needs_body() || !ld.body.is_empty());
                    ui.add_space(6.0);
                    ui.horizontal(|ui| {
                        if ui.add_enabled(differs && valid, primary(&p, "Apply")).clicked() {
                            apply_load = Some(ld.clone());
                        }
                        if ui.add_enabled(differs, egui::Button::new("Revert")).clicked() {
                            *ld = orig.clone();
                        }
                        if ui.button("Close").clicked() {
                            close_editor = true;
                        }
                    });
                }
                _ => {}
            }
        });

        if close_editor {
            self.sel_body = None;
            self.sel_load = None;
            self.edit_target = None;
            return;
        }

        if let Some((bname, vis)) = vis_action {
            match vis {
                Some(m) => self.set_body_vis(&bname, m),
                None => self.isolate_body(&bname),
            }
        }

        if let Some(name) = self.sel_body.clone() {
            ui.add_space(8.0);
            card(ui, &p, |ui| {
                ui.label(RichText::new("Transform").strong());
                let x = &mut self.xform;
                ui.horizontal(|ui| {
                    for (m, l) in [("move", "Move"), ("copy", "Copy"), ("pattern", "Pattern"), ("mirror", "Mirror")] {
                        ui.selectable_value(&mut x.0, m.to_string(), l);
                    }
                });
                let ls = units.len_scale();
                let lsf = units.len_suffix();
                if x.0 == "mirror" {
                    ui.horizontal(|ui| {
                        ui.label("Plane");
                        for (i, a) in ["X", "Y", "Z"].iter().enumerate() {
                            ui.selectable_value(&mut x.3, i, *a);
                        }
                        ui.label("=");
                        drag_scaled(ui, &mut x.4, ls, lsf, 0.5, None);
                    });
                    ui.checkbox(&mut x.5, "Keep the original (mirrored copy)");
                } else {
                    ui.label(RichText::new(if x.0 == "pattern" { format!("Step ({})", lsf.trim()) } else { format!("Offset ({})", lsf.trim()) }).small().color(p.weak));
                    xyz_scaled(ui, "xf-off", &mut x.1, ls, "", 0.5);
                    if x.0 == "pattern" {
                        ui.horizontal(|ui| {
                            ui.label("Instances");
                            ui.add(egui::DragValue::new(&mut x.2).range(2..=200));
                        });
                    }
                }
                if ui.add(primary(&p, "Apply transform")).clicked() {
                    let a = match x.0.as_str() {
                        "mirror" => json!({"name": name, "mode": "mirror", "axis": (["x", "y", "z"][x.3]), "at": x.4, "copy": x.5}),
                        "pattern" => json!({"name": name, "mode": "pattern", "translate": x.1, "count": x.2}),
                        m => json!({"name": name, "mode": m, "translate": x.1}),
                    };
                    action = Some(("transform_body", a));
                }
            });
        }

        if let Some(bd) = apply_body {
            let mut a = json!({"name": bd.orig_name});
            if bd.name != bd.orig_name {
                a["new_name"] = json!(bd.name);
            }
            a["material"] = json!(bd.material);
            if bd.role == Role::Solid {
                a["operation"] = json!(if bd.op == Op::Add { "add" } else { "subtract" });
            }
            if let Ok(Value::Object(map)) = serde_json::to_value(&bd.shape) {
                for (k, v) in map {
                    if !matches!(k.as_str(), "type" | "path") {
                        a[k] = v;
                    }
                }
            }
            if self.run("edit_body", a) {
                self.sel_body = Some(bd.name);
                self.edit_target = None;
            }
        }
        if let Some(ld) = apply_load {
            let is_cht = model.analysis.kind == AnalysisType::ConjugateHeatTransfer;
            let orig_load = model.loads.iter().find(|l| l.id == ld.id).map(|l| &l.load);
            let new_load = load_from_draft(ld.kind, &ld.f, ld.faces.clone(), &ld.body, is_cht);
            let new_v = serde_json::to_value(&new_load).unwrap_or(json!({}));
            let old_v = orig_load.and_then(|l| serde_json::to_value(l).ok()).unwrap_or(json!({}));
            let mut a = json!({"id": ld.id});
            for (k, val) in new_v.as_object().into_iter().flatten() {
                if k != "kind" && old_v.get(k) != Some(val) {
                    a[k] = val.clone();
                }
            }
            if self.run("edit_load", a) {
                self.edit_target = None;
            }
        }
        if let Some((t, a)) = action {
            if self.run(t, a) {
                self.edit_target = None;
                self.need_fit = true;
            }
        }
    }

    pub(super) fn properties(&mut self, ui: &mut egui::Ui) {
        let p = self.p;
        let units = self.units;
        let mesh = self.current_mesh();
        let snap_pt = first_selected_centroid(mesh.as_ref(), &self.selected);
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.label(RichText::new("PROPERTIES").size(11.0).strong().color(p.weak));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.small_button("»").on_hover_text("Collapse Properties panel").clicked() {
                    self.props_open = false;
                }
            });
        });
        ui.add_space(2.0);
        let name_row = |ui: &mut egui::Ui, name: &mut String| {
            ui.horizontal(|ui| {
                ui.label("Name");
                ui.add(egui::TextEdit::singleline(name).hint_text("auto").desired_width(f32::INFINITY));
            });
        };
        let op_args = |sub: bool| if sub { "subtract" } else { "add" };
        let ls = units.len_scale();
        let lsf = units.len_suffix();
        match self.form {
            Form::None if self.sel_body.is_some() || self.sel_load.is_some() => self.item_editor(ui, mesh.as_ref()),
            Form::None => {
                card(ui, &p, |ui| {
                    let sel = self.selected_names(mesh.as_ref());
                    if sel.is_empty() {
                        ui.label(RichText::new("Nothing selected").strong());
                        ui.label(RichText::new("Pick a tool from the ribbon, click a face, or switch to Body mode in the viewport toolbar.").color(p.weak));
                    } else {
                        ui.label(RichText::new(format!("{} face(s) selected", sel.len())).strong());
                        for n in &sel {
                            ui.label(format!("• {n}"));
                        }
                        if let Some(m) = &mesh {
                            let area_mm2: f64 = self.selected.iter().filter_map(|&i| m.patches.get(i as usize)).map(|x| x.area_mm2).sum();
                            let area_u = area_mm2 * units.area_scale();
                            ui.label(RichText::new(format!("Area ≈ {area_u:.2} {}", units.area_suffix())).color(p.weak));
                        }
                    }
                });
                ui.add_space(10.0);
                card(ui, &p, |ui| {
                    ui.label(RichText::new("Navigation & Shortcuts").strong());
                    for (k, v) in [
                        ("Left drag", "Orbit"),
                        ("Right / Middle / Space+drag", "Pan"),
                        ("Pinch / Wheel", "Zoom (or 2-finger Pan)"),
                        ("Click / Shift+click", "Select face or body"),
                        ("Right click", "Depth-peel & body actions"),
                        ("F · 1–6 · P · H · Del · Esc", "Fit · Views · Persp · Hide · Del"),
                    ] {
                        ui.horizontal(|ui| {
                            ui.label(RichText::new(k).small().color(p.weak));
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| ui.label(RichText::new(v).small()));
                        });
                    }
                });
            }
            Form::Box => {
                ui.heading(if self.f.subtract { "Pocket" } else { "Box" });
                card(ui, &p, |ui| {
                    name_row(ui, &mut self.f.name);
                    ui.label(RichText::new(format!("Corner ({})", lsf.trim())).small().color(p.weak));
                    xyz_scaled(ui, "bmin", &mut self.f.bmin, ls, "", 0.5);
                    if let Some(c) = snap_pt {
                        if ui.small_button("📍 Snap corner to selected face center").clicked() {
                            self.f.bmin = c;
                        }
                    }
                    ui.label(RichText::new(format!("Size ({})", lsf.trim())).small().color(p.weak));
                    xyz_scaled(ui, "bsize", &mut self.f.bsize, ls, "", 0.5);
                    ui.checkbox(&mut self.f.subtract, "Cut from existing bodies");
                    ui.label(RichText::new("Tip: drag X/Y/Z handles in the 3D preview.").small().color(p.weak));
                });
                ui.add_space(8.0);
                if ui.add(primary(&p, "Create")).clicked() {
                    let mut a = json!({"min": self.f.bmin, "size": self.f.bsize, "operation": op_args(self.f.subtract)});
                    if !self.f.name.trim().is_empty() {
                        a["name"] = json!(self.f.name.trim());
                    }
                    if self.run("add_box", a) {
                        self.f.name.clear();
                        self.need_fit = true;
                    }
                }
            }
            Form::Cylinder => {
                ui.heading(if self.f.subtract { "Hole" } else { "Cylinder" });
                card(ui, &p, |ui| {
                    name_row(ui, &mut self.f.name);
                    ui.label(RichText::new(format!("Base centre ({})", lsf.trim())).small().color(p.weak));
                    xyz_scaled(ui, "cbase", &mut self.f.cbase, ls, "", 0.5);
                    if let Some(c) = snap_pt {
                        if ui.small_button("📍 Snap base to selected face center").clicked() {
                            self.f.cbase = c;
                        }
                    }
                    ui.horizontal(|ui| {
                        ui.label("Axis");
                        for (i, a) in ["X", "Y", "Z"].iter().enumerate() {
                            ui.selectable_value(&mut self.f.caxis, i, *a);
                        }
                    });
                    egui::Grid::new("cyl").num_columns(2).show(ui, |ui| {
                        ui.label("Radius");
                        drag_scaled(ui, &mut self.f.cradius, ls, lsf, 0.2, Some((0.01, 1e6)));
                        ui.end_row();
                        ui.label("Height");
                        drag_scaled(ui, &mut self.f.cheight, ls, lsf, 0.5, Some((0.01, 1e6)));
                        ui.end_row();
                    });
                    ui.checkbox(&mut self.f.subtract, "Cut from existing bodies");
                });
                ui.add_space(8.0);
                if ui.add(primary(&p, "Create")).clicked() {
                    let axis = ["x", "y", "z"][self.f.caxis];
                    let mut a = json!({"base": self.f.cbase, "axis": axis, "radius": self.f.cradius, "height": self.f.cheight, "operation": op_args(self.f.subtract)});
                    if !self.f.name.trim().is_empty() {
                        a["name"] = json!(self.f.name.trim());
                    }
                    if self.run("add_cylinder", a) {
                        self.f.name.clear();
                        self.need_fit = true;
                    }
                }
            }
            Form::Sphere => {
                ui.heading("Sphere");
                card(ui, &p, |ui| {
                    name_row(ui, &mut self.f.name);
                    ui.label(RichText::new(format!("Centre ({})", lsf.trim())).small().color(p.weak));
                    xyz_scaled(ui, "sc", &mut self.f.scenter, ls, "", 0.5);
                    if let Some(c) = snap_pt {
                        if ui.small_button("📍 Snap centre to selected face center").clicked() {
                            self.f.scenter = c;
                        }
                    }
                    ui.horizontal(|ui| {
                        ui.label("Radius");
                        drag_scaled(ui, &mut self.f.sradius, ls, lsf, 0.2, Some((0.01, 1e6)));
                    });
                    ui.checkbox(&mut self.f.subtract, "Cut from existing bodies");
                });
                ui.add_space(8.0);
                if ui.add(primary(&p, "Create")).clicked() {
                    let mut a = json!({"center": self.f.scenter, "radius": self.f.sradius, "operation": op_args(self.f.subtract)});
                    if !self.f.name.trim().is_empty() {
                        a["name"] = json!(self.f.name.trim());
                    }
                    if self.run("add_sphere", a) {
                        self.f.name.clear();
                        self.need_fit = true;
                    }
                }
            }
            Form::Load(k) => self.load_form(ui, k, mesh.as_ref()),
            Form::Study | Form::Optimize | Form::Convergence | Form::Topology => {
                let f = self.form;
                self.opt_form(ui, f);
            }
            Form::Enclosure => {
                ui.heading("Enclosure");
                card(ui, &p, |ui| {
                    ui.label(RichText::new("The air or fluid region around your parts, used by electrostatic, magnetostatic and fluid-flow analyses.").color(p.weak));
                    ui.horizontal(|ui| {
                        ui.label("Fluid");
                        for f in ["Air", "Water", "Engine Oil (SAE 30)"] {
                            ui.selectable_value(&mut self.f.fluid, f.to_string(), f.split(' ').next().unwrap());
                        }
                    });
                    ui.checkbox(&mut self.f.enc_box, "Explicit box").on_hover_text("Give the enclosure's corner and size instead of padding around the parts");
                    if self.f.enc_box {
                        ui.label(RichText::new(format!("Corner ({})", lsf.trim())).small().color(p.weak));
                        xyz_scaled(ui, "emin", &mut self.f.enc_min, ls, "", 0.5);
                        ui.label(RichText::new(format!("Size ({})", lsf.trim())).small().color(p.weak));
                        xyz_scaled(ui, "esize", &mut self.f.enc_size, ls, "", 0.5);
                    } else {
                        ui.horizontal(|ui| {
                            ui.label("Padding");
                            ui.add(egui::Slider::new(&mut self.f.padding, 0.1..=3.0).suffix("× part size"));
                        });
                    }
                });
                ui.add_space(8.0);
                let args = if self.f.enc_box { json!({"material": self.f.fluid, "min": self.f.enc_min, "size": self.f.enc_size}) } else { json!({"material": self.f.fluid, "padding": self.f.padding}) };
                if ui.add(primary(&p, "Create")).clicked() && self.run("add_enclosure", args) {
                    self.need_fit = true;
                }
            }
        }
        if self.form != Form::None {
            ui.add_space(4.0);
            if ui.add(egui::Button::new("Close").frame_when_inactive(false)).clicked() {
                self.form = Form::None;
            }
        }
    }

    fn load_form(&mut self, ui: &mut egui::Ui, k: LoadKind, mesh: Option<&Arc<Mesh>>) {
        let p = self.p;
        ui.horizontal(|ui| {
            ui.label(RichText::new(k.icon()).size(18.0).color(k.color()));
            ui.heading(k.title());
        });
        let sel = self.selected_names(mesh);
        if k.needs_body() {
            let bodies: Vec<String> = self
                .core
                .session
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .model
                .bodies
                .iter()
                .filter(|b| b.role == Role::Solid && b.op == Op::Add)
                .map(|b| b.name.clone())
                .collect();
            if !bodies.contains(&self.f.body) {
                self.f.body = bodies.first().cloned().unwrap_or_default();
            }
            card(ui, &p, |ui| {
                ui.label(RichText::new("Body").strong());
                egui::ComboBox::from_id_salt("loadbody").width(200.0).selected_text(&self.f.body).show_ui(ui, |ui| {
                    for b in &bodies {
                        ui.selectable_value(&mut self.f.body, b.clone(), b);
                    }
                });
            });
            ui.add_space(6.0);
        }
        if k.needs_faces() {
            card(ui, &p, |ui| {
                ui.label(RichText::new("Scope").strong());
                if sel.is_empty() {
                    ui.label(RichText::new("Click faces in the viewport (Shift+click for more).").color(p.warn));
                } else {
                    for n in &sel {
                        ui.label(format!("• {n}"));
                    }
                    if ui.small_button("Clear").clicked() {
                        self.selected.clear();
                        self.sel_version += 1;
                    }
                }
            });
            ui.add_space(6.0);
        }
        card(ui, &p, |ui| {
            ui.label(RichText::new("Definition").strong());
            load_definition_ui(ui, &p, self.units, &self.core, k, &mut self.f, "cr");
        });
        ui.add_space(8.0);
        let ready = (!k.needs_faces() || !sel.is_empty()) && (!k.needs_body() || !self.f.body.is_empty());
        if ui.add_enabled(ready, primary(&p, "Apply")).clicked() {
            let is_cht = self.core.session.lock().unwrap_or_else(std::sync::PoisonError::into_inner).model.analysis.kind == AnalysisType::ConjugateHeatTransfer;
            let (tool, args) = load_tool_args(k, &self.f, &sel, &self.f.body, is_cht);
            if self.run(tool, args) {
                self.selected.clear();
                self.sel_version += 1;
            }
        }
    }
}
