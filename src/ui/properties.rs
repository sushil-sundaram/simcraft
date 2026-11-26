//! Right Properties panel: selected face summary, primitive creation forms, item editor, and load builder.

use super::theme::{card, json_editor, primary, xyz};
use super::{Form, LoadKind, SimApp};
use crate::geometry::Op;
use crate::mesh::Mesh;
use crate::model::AnalysisType;
use eframe::egui::{self, RichText};
use serde_json::{Value, json};
use std::sync::Arc;

impl SimApp {
    pub(super) fn item_editor(&mut self, ui: &mut egui::Ui) {
        let p = self.p;
        let (model, lib) = {
            let s = self.core.session.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            (s.model.clone(), s.model.materials().into_iter().map(|m| m.name).collect::<Vec<_>>())
        };
        let (key, fresh) = if let Some(n) = &self.sel_body {
            let Some(b) = model.bodies.iter().find(|b| &b.name == n) else {
                self.sel_body = None;
                return;
            };
            let mut v = serde_json::to_value(&b.shape).unwrap_or(json!({}));
            v["name"] = json!(b.name);
            v["material"] = json!(b.material);
            v["operation"] = json!(if b.op == Op::Add { "add" } else { "subtract" });
            (format!("b:{n}"), v)
        } else {
            let id = self.sel_load.unwrap();
            let Some(l) = model.loads.iter().find(|l| l.id == id) else {
                self.sel_load = None;
                return;
            };
            (format!("l:{id}"), serde_json::to_value(&l.load).unwrap_or(json!({})))
        };
        let stale = !matches!(&self.edit_val, Some((k, _)) if *k == key);
        if stale {
            self.edit_val = Some((key.clone(), fresh.clone()));
        }
        let mut apply = None;
        let mut action: Option<(&str, Value)> = None;
        card(ui, &p, |ui| {
            let title = if self.sel_body.is_some() {
                format!("Body · {}", fresh["name"].as_str().unwrap_or(""))
            } else {
                format!("Load #{} · {}", self.sel_load.unwrap(), model.loads.iter().find(|l| Some(l.id) == self.sel_load).map(|l| l.load.label()).unwrap_or(""))
            };
            ui.label(RichText::new(title).strong());
            ui.add_space(4.0);
            let (_, v) = self.edit_val.as_mut().unwrap();
            json_editor(ui, v, &lib, &key);
            ui.add_space(6.0);
            let differs = *v != fresh;
            ui.horizontal(|ui| {
                if ui.add_enabled(differs, primary(&p, "Apply")).clicked() {
                    apply = Some(v.clone());
                }
                if ui.add_enabled(differs, egui::Button::new("Revert")).clicked() {
                    *v = fresh.clone();
                }
                if ui.button("Close").clicked() {
                    self.sel_body = None;
                    self.sel_load = None;
                }
            });
        });
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
                if x.0 == "mirror" {
                    ui.horizontal(|ui| {
                        ui.label("Plane");
                        for (i, a) in ["X", "Y", "Z"].iter().enumerate() {
                            ui.selectable_value(&mut x.3, i, *a);
                        }
                        ui.label("=");
                        ui.add(egui::DragValue::new(&mut x.4).speed(0.5).suffix(" mm"));
                    });
                    ui.checkbox(&mut x.5, "Keep the original (mirrored copy)");
                } else {
                    ui.horizontal(|ui| {
                        ui.label(if x.0 == "pattern" { "Step" } else { "Offset" });
                        for c in x.1.iter_mut() {
                            ui.add(egui::DragValue::new(c).speed(0.5).suffix(" mm"));
                        }
                    });
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
        if let Some(v) = apply {
            if let Some(old) = self.sel_body.clone() {
                let mut a = json!({"name": old});
                for (k, val) in v.as_object().into_iter().flatten() {
                    if matches!(k.as_str(), "type" | "path") || fresh.get(k) == Some(val) {
                        continue;
                    }
                    if k == "name" {
                        a["new_name"] = val.clone();
                    } else {
                        a[k] = val.clone();
                    }
                }
                if self.run("edit_body", a) {
                    if let Some(n) = v["name"].as_str() {
                        self.sel_body = Some(n.to_string());
                    }
                    self.edit_val = None;
                }
            } else if let Some(id) = self.sel_load {
                let mut a = json!({"id": id});
                for (k, val) in v.as_object().into_iter().flatten() {
                    if k != "kind" && fresh.get(k) != Some(val) {
                        a[k] = val.clone();
                    }
                }
                if self.run("edit_load", a) {
                    self.edit_val = None;
                }
            }
        }
        if let Some((t, a)) = action {
            if self.run(t, a) {
                self.need_fit = true;
            }
        }
    }

    pub(super) fn properties(&mut self, ui: &mut egui::Ui) {
        let p = self.p;
        let mesh = self.current_mesh();
        ui.add_space(4.0);
        ui.label(RichText::new("PROPERTIES").size(11.0).strong().color(p.weak));
        ui.add_space(2.0);
        let name_row = |ui: &mut egui::Ui, name: &mut String| {
            ui.horizontal(|ui| {
                ui.label("Name");
                ui.add(egui::TextEdit::singleline(name).hint_text("auto").desired_width(f32::INFINITY));
            });
        };
        let op_args = |sub: bool| if sub { "subtract" } else { "add" };
        match self.form {
            Form::None if self.sel_body.is_some() || self.sel_load.is_some() => self.item_editor(ui),
            Form::None => {
                card(ui, &p, |ui| {
                    let sel = self.selected_names(mesh.as_ref());
                    if sel.is_empty() {
                        ui.label(RichText::new("Nothing selected").strong());
                        ui.label(RichText::new("Pick a tool from the ribbon, or click a face in the viewport.").color(p.weak));
                    } else {
                        ui.label(RichText::new(format!("{} face(s) selected", sel.len())).strong());
                        for n in &sel {
                            ui.label(format!("• {n}"));
                        }
                        if let Some(m) = &mesh {
                            let area: f64 = self.selected.iter().filter_map(|&i| m.patches.get(i as usize)).map(|x| x.area_mm2).sum();
                            ui.label(RichText::new(format!("Area ≈ {area:.1} mm²")).color(p.weak));
                        }
                    }
                });
                ui.add_space(10.0);
                card(ui, &p, |ui| {
                    ui.label(RichText::new("Navigation").strong());
                    for (k, v) in [("Left drag", "Orbit"), ("Right / middle drag", "Pan"), ("Wheel", "Zoom"), ("Click", "Select face"), ("Shift + click", "Add to selection")] {
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
                    ui.label(RichText::new("Corner (mm)").small().color(p.weak));
                    xyz(ui, "bmin", &mut self.f.bmin, "", 0.5);
                    ui.label(RichText::new("Size (mm)").small().color(p.weak));
                    xyz(ui, "bsize", &mut self.f.bsize, "", 0.5);
                    ui.checkbox(&mut self.f.subtract, "Cut from existing bodies");
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
                    ui.label(RichText::new("Base centre (mm)").small().color(p.weak));
                    xyz(ui, "cbase", &mut self.f.cbase, "", 0.5);
                    ui.horizontal(|ui| {
                        ui.label("Axis");
                        for (i, a) in ["X", "Y", "Z"].iter().enumerate() {
                            ui.selectable_value(&mut self.f.caxis, i, *a);
                        }
                    });
                    egui::Grid::new("cyl").num_columns(2).show(ui, |ui| {
                        ui.label("Radius");
                        ui.add(egui::DragValue::new(&mut self.f.cradius).speed(0.2).range(0.01..=1e6).suffix(" mm"));
                        ui.end_row();
                        ui.label("Height");
                        ui.add(egui::DragValue::new(&mut self.f.cheight).speed(0.5).range(0.01..=1e6).suffix(" mm"));
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
                    ui.label(RichText::new("Centre (mm)").small().color(p.weak));
                    xyz(ui, "sc", &mut self.f.scenter, "", 0.5);
                    ui.horizontal(|ui| {
                        ui.label("Radius");
                        ui.add(egui::DragValue::new(&mut self.f.sradius).speed(0.2).range(0.01..=1e6).suffix(" mm"));
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
                    ui.checkbox(&mut self.f.enc_box, "Explicit box").on_hover_text("Give the enclosure's corner and size instead of padding around the parts (e.g. λ/4 of air for radiation)");
                    if self.f.enc_box {
                        ui.label(RichText::new("Corner (mm)").small().color(p.weak));
                        xyz(ui, "emin", &mut self.f.enc_min, "", 0.5);
                        ui.label(RichText::new("Size (mm)").small().color(p.weak));
                        xyz(ui, "esize", &mut self.f.enc_size, "", 0.5);
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
            let bodies: Vec<String> = self.core.session.lock().unwrap_or_else(std::sync::PoisonError::into_inner).model.bodies.iter().filter(|b| b.role == crate::geometry::Role::Solid && b.op == Op::Add).map(|b| b.name.clone()).collect();
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
            match k {
                LoadKind::Fixed => {
                    ui.label(RichText::new("Fixes all translations on the faces.").color(p.weak));
                }
                LoadKind::Displacement => {
                    for (i, c) in ["X", "Y", "Z"].iter().enumerate() {
                        ui.horizontal(|ui| {
                            ui.checkbox(&mut self.f.dcomp[i].0, *c);
                            if self.f.dcomp[i].0 {
                                ui.add(egui::DragValue::new(&mut self.f.dcomp[i].1).speed(0.01).suffix(" mm"));
                            } else {
                                ui.label(RichText::new("free").color(p.weak));
                            }
                        });
                    }
                }
                LoadKind::Force => {
                    ui.label(RichText::new("Total force (N)").small().color(p.weak));
                    xyz(ui, "force", &mut self.f.vector, "", 1.0);
                }
                LoadKind::Pressure => {
                    ui.add(egui::DragValue::new(&mut self.f.value).speed(0.01).suffix(" MPa"));
                    ui.label(RichText::new("Positive pushes into the surface.").small().color(p.weak));
                }
                LoadKind::Gravity => {
                    ui.label(RichText::new("Acceleration (m/s²)").small().color(p.weak));
                    xyz(ui, "grav", &mut self.f.grav, "", 0.1);
                }
                LoadKind::Temperature => {
                    ui.add(egui::DragValue::new(&mut self.f.value).speed(0.5).suffix(" °C"));
                }
                LoadKind::HeatFlux => {
                    ui.add(egui::DragValue::new(&mut self.f.value).speed(10.0).suffix(" W/m²"));
                    ui.label(RichText::new("Positive flows into the body.").small().color(p.weak));
                }
                LoadKind::Convection => {
                    egui::Grid::new("conv").num_columns(2).show(ui, |ui| {
                        ui.label("Film coeff.");
                        ui.add(egui::DragValue::new(&mut self.f.coef).speed(0.5).range(0.0..=1e6).suffix(" W/m²K"));
                        ui.end_row();
                        ui.label("Ambient");
                        ui.add(egui::DragValue::new(&mut self.f.ambient).speed(0.5).suffix(" °C"));
                        ui.end_row();
                    });
                }
                LoadKind::Radiation => {
                    egui::Grid::new("rad").num_columns(2).show(ui, |ui| {
                        ui.label("Emissivity ε");
                        ui.add(egui::DragValue::new(&mut self.f.rad_eps).speed(0.01).range(0.01..=1.0)).on_hover_text("Polished aluminium ≈ 0.05, oxidised metal 0.3–0.6, anodised / painted / FR-4 0.8–0.95");
                        ui.end_row();
                        ui.label("Surroundings");
                        ui.add(egui::DragValue::new(&mut self.f.ambient).speed(0.5).suffix(" °C"));
                        ui.end_row();
                    });
                    ui.checkbox(&mut self.f.rad_enclosure, "Faces see each other (enclosure)").on_hover_text("Surface-to-surface: these faces (and those of other enclosure radiation loads) exchange radiation by view factors; parts block each other. Off = each face radiates to the surroundings only.");
                    ui.label(RichText::new("Non-linear: the solver iterates. Combine with Convection on the same faces for natural cooling.").small().color(p.weak));
                }
                LoadKind::HeatGen => {
                    ui.horizontal(|ui| {
                        ui.selectable_value(&mut self.f.heat_watts, true, "Total power");
                        ui.selectable_value(&mut self.f.heat_watts, false, "Power density");
                    });
                    if self.f.heat_watts {
                        ui.add(egui::DragValue::new(&mut self.f.heat_power).speed(0.1).range(0.0..=1e9).suffix(" W"));
                    } else {
                        ui.add(egui::DragValue::new(&mut self.f.value).speed(1000.0).suffix(" W/m³"));
                    }
                    let bodies: Vec<String> = self.core.session.lock().unwrap_or_else(std::sync::PoisonError::into_inner).model.bodies.iter().filter(|b| b.role == crate::geometry::Role::Solid && b.op == Op::Add).map(|b| b.name.clone()).collect();
                    ui.horizontal(|ui| {
                        ui.label("In");
                        let shown = if self.f.heat_body.is_empty() { "Whole model".to_string() } else { self.f.heat_body.clone() };
                        egui::ComboBox::from_id_salt("heatbody").width(170.0).selected_text(shown).show_ui(ui, |ui| {
                            ui.selectable_value(&mut self.f.heat_body, String::new(), "Whole model");
                            for b in &bodies {
                                ui.selectable_value(&mut self.f.heat_body, b.clone(), b);
                            }
                        });
                    });
                    ui.label(RichText::new(if self.f.heat_watts { "Spread evenly over the body's volume." } else { "Power [W] = value × volume (e.g. 2 W in a 10×10×2 mm chip = 10⁷ W/m³)." }).small().color(p.weak));
                }
                LoadKind::CompressionOnly => {
                    ui.label(RichText::new("Rests on a rigid surface: pushes are resisted, pulling lifts off (frictionless).").color(p.weak));
                }
                LoadKind::Symmetry => {
                    ui.label(RichText::new("Structural: zero normal motion. Fluid: free-slip wall.").color(p.weak));
                }
                LoadKind::Voltage => {
                    ui.add(egui::DragValue::new(&mut self.f.value).speed(0.1).suffix(" V"));
                }
                LoadKind::Current => {
                    ui.add(egui::DragValue::new(&mut self.f.value).speed(0.1).suffix(" A"));
                    ui.label(RichText::new("Enters the body through the faces; ground another face with 0 V.").small().color(p.weak));
                }
                LoadKind::Coil => {
                    ui.add(egui::DragValue::new(&mut self.f.value).speed(0.1).suffix(" A/mm²"));
                    ui.horizontal(|ui| {
                        ui.label("Axis");
                        for (i, a) in ["X", "Y", "Z"].iter().enumerate() {
                            ui.selectable_value(&mut self.f.coil_axis, i, *a);
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.selectable_value(&mut self.f.coil_axial, false, "Around axis (winding)");
                        ui.selectable_value(&mut self.f.coil_axial, true, "Along axis (bar)");
                    });
                }
                LoadKind::Magnet => {
                    ui.label(RichText::new("Remanence Br (T)").small().color(p.weak));
                    xyz(ui, "br", &mut self.f.remanence, "", 0.05);
                }
                LoadKind::Inlet => {
                    ui.add(egui::DragValue::new(&mut self.f.velocity).speed(0.01).suffix(" m/s"));
                    ui.label(RichText::new("Uniform velocity into the domain.").small().color(p.weak));
                    if self.core.session.lock().unwrap_or_else(std::sync::PoisonError::into_inner).model.analysis.kind == AnalysisType::ConjugateHeatTransfer {
                        ui.horizontal(|ui| {
                            ui.label("Inflow temperature");
                            ui.add(egui::DragValue::new(&mut self.f.ambient).speed(0.5).suffix(" °C"));
                        });
                    }
                }
                LoadKind::MovingWall => {
                    ui.label(RichText::new("Wall velocity (m/s), tangential to the faces").small().color(p.weak));
                    xyz(ui, "mw", &mut self.f.wall_v, " m/s", 0.01);
                }
                LoadKind::Outlet => {
                    ui.add(egui::DragValue::new(&mut self.f.outlet_p).speed(1.0).suffix(" Pa"));
                }
                LoadKind::Port => {
                    ui.add(egui::DragValue::new(&mut self.f.port_power).speed(5.0).range(0.001..=1e7).suffix(" W"));
                    ui.horizontal(|ui| {
                        ui.label("E-field along");
                        for (i, a) in ["X", "Y", "Z"].iter().enumerate() {
                            ui.selectable_value(&mut self.f.port_pol, i, *a);
                        }
                    });
                    ui.checkbox(&mut self.f.port_te10, "Waveguide (TE10) profile");
                    ui.label(RichText::new("Pick the feed opening, usually an Enclosure wall face. The E-field direction must lie in the face (WR-340: along its short side).").small().color(p.weak));
                }
            }
        });
        ui.add_space(8.0);
        let ready = (!k.needs_faces() || !sel.is_empty()) && (!k.needs_body() || !self.f.body.is_empty());
        if ui.add_enabled(ready, primary(&p, "Apply")).clicked() {
            let fa = json!(sel);
            let (tool, args) = match k {
                LoadKind::Fixed => ("add_fixed_support", json!({"faces": fa})),
                LoadKind::Displacement => {
                    let mut a = json!({"faces": fa});
                    for (i, c) in ["x", "y", "z"].iter().enumerate() {
                        if self.f.dcomp[i].0 {
                            a[*c] = json!(self.f.dcomp[i].1);
                        }
                    }
                    ("add_displacement", a)
                }
                LoadKind::Force => ("add_force", json!({"faces": fa, "vector": self.f.vector})),
                LoadKind::Pressure => ("add_pressure", json!({"faces": fa, "value_mpa": self.f.value})),
                LoadKind::Gravity => ("add_gravity", json!({"acceleration": self.f.grav})),
                LoadKind::Temperature => ("add_temperature", json!({"faces": fa, "value": self.f.value})),
                LoadKind::HeatFlux => ("add_heat_flux", json!({"faces": fa, "value": self.f.value})),
                LoadKind::Convection => ("add_convection", json!({"faces": fa, "coefficient": self.f.coef, "ambient": self.f.ambient})),
                LoadKind::Radiation => ("add_radiation", json!({"faces": fa, "emissivity": self.f.rad_eps, "ambient": self.f.ambient, "enclosure": self.f.rad_enclosure})),
                LoadKind::HeatGen => {
                    if self.f.heat_watts {
                        ("add_heat_generation", json!({"power": self.f.heat_power, "body": self.f.heat_body}))
                    } else {
                        ("add_heat_generation", json!({"value": self.f.value, "body": self.f.heat_body}))
                    }
                }
                LoadKind::CompressionOnly => ("add_compression_only_support", json!({"faces": fa})),
                LoadKind::Symmetry => ("add_symmetry", json!({"faces": fa})),
                LoadKind::Voltage => ("add_voltage", json!({"faces": fa, "value": self.f.value})),
                LoadKind::Current => ("add_current", json!({"faces": fa, "value": self.f.value})),
                LoadKind::Coil => ("add_coil", json!({"body": self.f.body, "current_density": self.f.value, "axis": (["x", "y", "z"])[self.f.coil_axis], "mode": (if self.f.coil_axial { "axial" } else { "circumferential" })})),
                LoadKind::Magnet => ("add_magnet", json!({"body": self.f.body, "remanence": self.f.remanence})),
                LoadKind::Inlet => {
                    let mut a = json!({"faces": fa, "velocity": self.f.velocity});
                    if self.core.session.lock().unwrap_or_else(std::sync::PoisonError::into_inner).model.analysis.kind == AnalysisType::ConjugateHeatTransfer {
                        a["temperature"] = json!(self.f.ambient);
                    }
                    ("add_inlet", a)
                }
                LoadKind::MovingWall => ("add_moving_wall", json!({"faces": fa, "velocity": self.f.wall_v})),
                LoadKind::Outlet => ("add_outlet", json!({"faces": fa, "pressure": self.f.outlet_p})),
                LoadKind::Port => ("add_port", json!({"faces": fa, "power": self.f.port_power, "polarization": (["x", "y", "z"])[self.f.port_pol], "profile": if self.f.port_te10 { "te10" } else { "uniform" }})),
            };
            if self.run(tool, args) {
                self.selected.clear();
                self.sel_version += 1;
            }
        }
    }
}
