//! Left Model Browser: Geometry bodies, Loads/Supports, Mesh statistics, and Solution summary.

use super::{Form, SimApp};
use crate::geometry::Op;
use crate::model::AnalysisType;
use crate::render;
use eframe::egui::{self, Color32, RichText, Sense, vec2};
use serde_json::json;

impl SimApp {
    pub(super) fn browser(&mut self, ui: &mut egui::Ui) {
        let p = self.p;
        let (model, mesh, res, disp, stale) = {
            let s = self.core.session.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            (s.model.clone(), if s.mesh_valid() { s.mesh.clone() } else { None }, s.results.clone(), s.display.clone(), s.results_stale)
        };
        ui.add_space(4.0);
        ui.label(RichText::new("BROWSER").size(11.0).strong().color(p.weak));
        ui.add_space(2.0);
        let del = |ui: &mut egui::Ui| ui.add(egui::Button::new(RichText::new("✖").size(10.0).color(p.weak)).frame_when_inactive(false)).on_hover_text("Delete").clicked();
        egui::ScrollArea::vertical().show(ui, |ui| {
            egui::CollapsingHeader::new(RichText::new("📦  Geometry").strong()).default_open(true).show(ui, |ui| {
                if model.bodies.is_empty() {
                    ui.label(RichText::new("No bodies yet").color(p.weak));
                }
                let mut mats: Vec<&str> = model.bodies.iter().filter(|b| b.op == Op::Add).map(|b| b.material.as_str()).collect();
                mats.dedup();
                if !mats.is_empty() {
                    ui.label(RichText::new(format!("🎨 {}", mats.join(", "))).small().color(p.weak));
                }
                for b in &model.bodies {
                    ui.horizontal(|ui| {
                        let (icon, c) = if b.op == Op::Add { ("■", p.accent) } else { ("□", p.weak) };
                        ui.label(RichText::new(icon).color(c));
                        let tip = format!("{}\nMaterial: {}\n{}", b.name, b.material, serde_json::to_string(&b.shape).unwrap_or_default());
                        let mut lbl = RichText::new(&b.name);
                        if b.op == Op::Subtract {
                            lbl = lbl.color(p.weak);
                        }
                        let picked = self.sel_body.as_deref() == Some(b.name.as_str());
                        if picked {
                            lbl = lbl.strong().color(p.accent);
                        }
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if del(ui) {
                                self.run("remove_body", json!({"name": b.name}));
                            }
                            if b.op == Op::Subtract {
                                ui.label(RichText::new("cut").small().color(p.weak));
                            }
                            ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                                if ui.add(egui::Label::new(lbl).truncate().sense(Sense::click())).on_hover_text(format!("{tip}\nClick to edit")).clicked() {
                                    self.sel_body = if picked { None } else { Some(b.name.clone()) };
                                    self.sel_load = None;
                                    self.form = Form::None;
                                }
                            });
                        });
                    });
                }
            });
            egui::CollapsingHeader::new(RichText::new(format!("🔧  {}", model.analysis.kind.label())).strong()).default_open(true).show(ui, |ui| {
                if model.loads.is_empty() {
                    ui.label(RichText::new("No supports or loads").color(p.weak));
                }
                for l in &model.loads {
                    ui.horizontal(|ui| {
                        let c = render::load_color(&l.load);
                        let (r, _) = ui.allocate_exact_size(vec2(9.0, 9.0), Sense::hover());
                        ui.painter().circle_filled(r.center(), 4.5, Color32::from_rgb(c[0], c[1], c[2]));
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if del(ui) {
                                self.run("remove_load", json!({"id": l.id}));
                            }
                            ui.label(RichText::new(format!("#{}", l.id)).small().color(p.weak));
                            ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                                let picked = self.sel_load == Some(l.id);
                                let txt = if picked { RichText::new(l.load.label()).strong().color(p.accent) } else { RichText::new(l.load.label()) };
                                if ui.add(egui::Label::new(txt).truncate().sense(Sense::click())).on_hover_text(format!("#{} {}\nClick to edit", l.id, l.load.summary())).clicked() {
                                    self.sel_load = if picked { None } else { Some(l.id) };
                                    self.sel_body = None;
                                    self.form = Form::None;
                                }
                            });
                        });
                    });
                }
            });
            egui::CollapsingHeader::new(RichText::new("▦  Mesh").strong()).default_open(true).show(ui, |ui| match &mesh {
                Some(m) => {
                    ui.label(format!("{} elements", m.elems.len()));
                    ui.label(RichText::new(format!("{} nodes · {} mm", m.nodes.len(), m.h)).color(p.weak));
                    for w in &m.warnings {
                        ui.label(RichText::new(format!("⚠ {w}")).small().color(p.warn));
                    }
                }
                None => {
                    ui.label(RichText::new("not generated").color(p.weak));
                }
            });
            egui::CollapsingHeader::new(RichText::new("📊  Solution").strong()).default_open(true).show(ui, |ui| match &res {
                Some(r) => {
                    if stale {
                        ui.label(RichText::new("⚠ setup changed — solve again").color(p.warn));
                    }
                    ui.label(RichText::new(format!("{} DOF · {:.2} s · {}", r.info.dofs, r.info.seconds, r.info.solver)).small().color(p.weak));
                    ui.label(RichText::new(format!("Mass {:.4} kg", r.info.mass_kg)).small().color(p.weak));
                    ui.add_space(4.0);
                    if let Some(m) = mesh.as_ref() {
                        let sum = r.summary_with(m, disp.cylindrical);
                        if let Some(fields) = sum["fields"].as_object() {
                            egui::Grid::new("resgrid").num_columns(2).spacing([8.0, 2.0]).show(ui, |ui| {
                                let rows = [
                                    ("total_deformation", "Max deformation", "max"),
                                    ("von_mises", "Max von Mises", "max"),
                                    ("max_principal", "Max principal", "max"),
                                    ("min_principal", "Min principal", "min"),
                                    ("hoop_stress", "Max hoop stress", "max"),
                                    ("radial_stress", "Min radial stress", "min"),
                                    ("safety_factor", "Min safety factor", "min"),
                                    ("temperature", "Max temperature", "max"),
                                    ("temperature", "Min temperature", "min"),
                                    ("heat_flux", "Max heat flux", "max"),
                                ];
                                for (key, label, which) in rows {
                                    if let Some(v) = fields.get(key) {
                                        let val = v[which].as_f64().unwrap_or(0.0);
                                        let mut t = RichText::new(format!("{} {}", render::fmt_val(val), v["unit"].as_str().unwrap_or(""))).small().strong();
                                        if key == "safety_factor" && val < 1.0 {
                                            t = t.color(p.err);
                                        }
                                        ui.label(RichText::new(label).small());
                                        ui.label(t);
                                        ui.end_row();
                                    }
                                }
                            });
                        }
                    }
                    for i in 0..r.modes.len() {
                        if ui.link(r.step_label(i)).clicked() {
                            self.run("set_display", json!({"mode": i + 1, "show_results": true}));
                        }
                    }
                    for qn in &r.quantities {
                        ui.horizontal(|ui| {
                            ui.add(egui::Label::new(RichText::new(&qn.name).small()).truncate());
                            ui.label(RichText::new(format!("{} {}", render::fmt_val(qn.value), qn.unit)).small().strong());
                        });
                    }
                    for rc in &r.reactions {
                        use AnalysisType::*;
                        let t = match r.analysis {
                            SteadyThermal | TransientThermal => format!("#{} heat flow {:.4} W", rc.load_id, rc.value[0]),
                            Electrostatic => format!("#{} charge {} C", rc.load_id, render::fmt_val(rc.value[0])),
                            CurrentFlow | ThermalElectric => format!("#{} current {} A", rc.load_id, render::fmt_val(rc.value[0])),
                            Fluid => format!("#{} flow {} m³/s, p̄ {} Pa", rc.load_id, render::fmt_val(rc.value[0]), render::fmt_val(rc.value[1])),
                            _ => format!("#{} reaction ({:.1}, {:.1}, {:.1}) N", rc.load_id, rc.value[0], rc.value[1], rc.value[2]),
                        };
                        ui.label(RichText::new(t).small().color(p.weak));
                    }
                    for w in &r.warnings {
                        ui.label(RichText::new(format!("⚠ {w}")).small().color(p.warn));
                    }
                }
                None => {
                    ui.label(RichText::new("not solved").color(p.weak));
                }
            });
        });
    }
}
