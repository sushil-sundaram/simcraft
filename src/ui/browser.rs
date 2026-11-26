//! Left Model Browser: collapsible panel, search filter, grouped KiCad PCB components,
//! per-body visibility (show/ghost/hide), multi-body batch actions, loads, mesh, and solution summary.

use super::theme::pal;
use super::{Form, SimApp};
use crate::geometry::{Body, Op, Role, Shape};
use crate::model::AnalysisType;
use crate::render;
use eframe::egui::{self, Color32, RichText, Sense, vec2};
use serde_json::json;
use std::collections::{HashMap, HashSet};

fn group_pcb_components(bodies: &[Body]) -> (HashMap<String, Vec<usize>>, HashSet<usize>) {
    let mut grouped: HashMap<String, Vec<usize>> = HashMap::new();
    let mut is_child: HashSet<usize> = HashSet::new();
    for parent in bodies {
        let Shape::Pcb { data: Some(pcb), .. } = &parent.shape else { continue };
        let refs: HashSet<&str> = pcb.footprints.iter().map(|f| f.reference.as_str()).collect();
        let mut kids = Vec::new();
        for (idx, b) in bodies.iter().enumerate() {
            if b.name == parent.name || b.role != Role::Solid {
                continue;
            }
            if matches!(b.shape, Shape::Box { .. }) && (refs.contains(b.name.as_str()) || refs.iter().any(|r| b.name.starts_with(&format!("{r}_")))) {
                kids.push(idx);
                is_child.insert(idx);
            }
        }
        if !kids.is_empty() {
            grouped.insert(parent.name.clone(), kids);
        }
    }
    (grouped, is_child)
}

impl SimApp {
    fn body_row(&mut self, ui: &mut egui::Ui, b: &Body, hidden: &HashSet<String>, ghost: &HashSet<String>, indent: bool) -> (Option<(String, bool)>, Option<String>) {
        let p = self.p;
        let del = |ui: &mut egui::Ui| ui.add(egui::Button::new(RichText::new("✖").size(10.0).color(p.weak)).frame_when_inactive(false)).on_hover_text("Delete").clicked();
        let mut clicked_vis = None;
        let mut clicked_del = None;
        ui.horizontal(|ui| {
            if indent {
                ui.add_space(12.0);
            }
            let mut checked = self.sel_bodies.contains(&b.name);
            if ui.checkbox(&mut checked, "").on_hover_text("Select for batch material / hide / delete").changed() {
                if checked {
                    self.sel_bodies.insert(b.name.clone());
                } else {
                    self.sel_bodies.remove(&b.name);
                }
            }
            let vis_mode = if hidden.contains(&b.name) { 2 } else if ghost.contains(&b.name) { 1 } else { 0 };
            let (vicon, vtip, vcol) = match vis_mode {
                1 => ("👻", "Ghosted (translucent, click-through) — click to hide", p.accent),
                2 => ("🚫", "Hidden — click to show", p.err),
                _ => ("👁", "Visible — click to ghost", p.weak),
            };
            if ui.add(egui::Button::new(RichText::new(vicon).size(11.0).color(vcol)).frame_when_inactive(false)).on_hover_text(vtip).clicked() {
                clicked_vis = Some((b.name.clone(), vis_mode));
            }
            let (icon, c) = if b.role == Role::Enclosure {
                ("⬚", Color32::from_rgb(80, 175, 225))
            } else if b.op == Op::Add {
                ("■", p.accent)
            } else {
                ("□", p.weak)
            };
            ui.label(RichText::new(icon).color(c));
            let tip = format!("{}\nMaterial: {}\n{}", b.name, b.material, serde_json::to_string(&b.shape).unwrap_or_default());
            let mut lbl = RichText::new(&b.name);
            if vis_mode == 2 {
                lbl = lbl.strikethrough().color(p.weak);
            } else if b.op == Op::Subtract {
                lbl = lbl.color(p.weak);
            }
            let picked = self.sel_body.as_deref() == Some(b.name.as_str());
            if picked {
                lbl = lbl.strong().color(p.accent);
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if del(ui) {
                    clicked_del = Some(b.name.clone());
                }
                if b.op == Op::Subtract {
                    ui.label(RichText::new("cut").small().color(p.weak));
                }
                ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                    if ui.add(egui::Label::new(lbl).truncate().sense(Sense::click())).on_hover_text(format!("{tip}\nClick to edit")).clicked() {
                        self.sel_body = if picked { None } else { Some(b.name.clone()) };
                        self.sel_load = None;
                        self.edit_target = None;
                        self.form = Form::None;
                        self.props_open = true;
                    }
                });
            });
        });
        (clicked_vis.map(|(n, m)| (n, m == 0)), clicked_del)
    }

    pub(super) fn browser(&mut self, ui: &mut egui::Ui) {
        let p = self.p;
        let _ = pal(self.dark);
        let (model, mesh, res, disp, stale) = {
            let s = self.core.session.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            (s.model.clone(), if s.mesh_valid() { s.mesh.clone() } else { None }, s.results.clone(), s.display.clone(), s.results_stale)
        };
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.label(RichText::new("BROWSER").size(11.0).strong().color(p.weak));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.small_button("«").on_hover_text("Collapse Browser panel").clicked() {
                    self.browser_open = false;
                }
                if (!disp.hidden_bodies.is_empty() || !disp.ghost_bodies.is_empty()) && ui.small_button("👁 Show all").on_hover_text("Show all hidden/ghost bodies (Shift+H)").clicked() {
                    self.show_all_bodies();
                }
            });
        });
        ui.horizontal(|ui| {
            ui.add(egui::TextEdit::singleline(&mut self.browser_filter).hint_text("🔍 Filter bodies & loads…").desired_width(ui.available_width() - 26.0));
            if !self.browser_filter.is_empty() && ui.small_button("✖").on_hover_text("Clear filter").clicked() {
                self.browser_filter.clear();
            }
        });
        ui.add_space(2.0);
        let q = self.browser_filter.trim().to_lowercase();
        let mats_lib = model.materials();
        if !self.sel_bodies.is_empty() {
            let names: Vec<String> = model.bodies.iter().filter(|b| self.sel_bodies.contains(&b.name)).map(|b| b.name.clone()).collect();
            if !names.is_empty() {
                egui::Frame::new().fill(p.card).stroke(eframe::egui::Stroke::new(1.0, p.border)).corner_radius(6).inner_margin(6).show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.label(RichText::new(format!("{} selected", names.len())).small().strong().color(p.accent));
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui.small_button("Clear").clicked() {
                                self.sel_bodies.clear();
                            }
                            if ui.small_button("Hide").clicked() {
                                let mut h = disp.hidden_bodies.clone();
                                for n in &names {
                                    h.insert(n.clone());
                                }
                                let hv: Vec<_> = h.into_iter().collect();
                                self.run("set_display", json!({"hidden_bodies": hv}));
                            }
                            if ui.small_button("🗑").on_hover_text("Delete all selected bodies").clicked() {
                                for n in &names {
                                    self.run("remove_body", json!({"name": n}));
                                }
                                self.sel_bodies.clear();
                            }
                        });
                    });
                    ui.horizontal(|ui| {
                        egui::ComboBox::from_id_salt("batch-mat").width(145.0).selected_text(&self.batch_material).show_ui(ui, |ui| {
                            for m in &mats_lib {
                                ui.selectable_value(&mut self.batch_material, m.name.clone(), &m.name);
                            }
                        });
                        if ui.small_button("Apply mat").clicked() {
                            for n in &names {
                                self.run("set_material", json!({"body": n, "material": self.batch_material}));
                            }
                        }
                    });
                });
                ui.add_space(4.0);
            }
        }

        let del_load = |ui: &mut egui::Ui| ui.add(egui::Button::new(RichText::new("✖").size(10.0).color(p.weak)).frame_when_inactive(false)).on_hover_text("Delete").clicked();
        let mut vis_cycles = Vec::new();
        let mut del_bodies = Vec::new();

        egui::ScrollArea::vertical().show(ui, |ui| {
            egui::CollapsingHeader::new(RichText::new(format!("📦  Geometry ({})", model.bodies.len())).strong()).default_open(true).show(ui, |ui| {
                if model.bodies.is_empty() {
                    ui.label(RichText::new("No bodies yet").color(p.weak));
                }
                let mut mats: Vec<&str> = model.bodies.iter().filter(|b| b.op == Op::Add).map(|b| b.material.as_str()).collect();
                mats.dedup();
                if !mats.is_empty() {
                    ui.label(RichText::new(format!("🎨 {}", mats.join(", "))).small().color(p.weak));
                }
                let (grouped, is_child) = group_pcb_components(&model.bodies);
                for (bi, b) in model.bodies.iter().enumerate() {
                    if is_child.contains(&bi) {
                        continue;
                    }
                    let kids = grouped.get(&b.name);
                    let match_self = q.is_empty() || b.name.to_lowercase().contains(&q) || b.material.to_lowercase().contains(&q);
                    let matching_kids: Vec<usize> = kids
                        .into_iter()
                        .flatten()
                        .copied()
                        .filter(|&ki| q.is_empty() || model.bodies[ki].name.to_lowercase().contains(&q) || model.bodies[ki].material.to_lowercase().contains(&q))
                        .collect();
                    if !match_self && matching_kids.is_empty() {
                        continue;
                    }
                    let (cv, cd) = self.body_row(ui, b, &disp.hidden_bodies, &disp.ghost_bodies, false);
                    if let Some((n, _)) = cv {
                        vis_cycles.push(n);
                    }
                    if let Some(d) = cd {
                        del_bodies.push(d);
                    }
                    if let Some(all_kids) = kids {
                        let shown_kids = if q.is_empty() { all_kids.as_slice() } else { matching_kids.as_slice() };
                        if !shown_kids.is_empty() {
                            egui::CollapsingHeader::new(RichText::new(format!("🧩 {} components", all_kids.len())).small().color(p.weak))
                                .id_salt(format!("pcb-kids-{}", b.name))
                                .default_open(!q.is_empty())
                                .show(ui, |ui| {
                                    for &ki in shown_kids {
                                        let kb = &model.bodies[ki];
                                        let (kcv, kcd) = self.body_row(ui, kb, &disp.hidden_bodies, &disp.ghost_bodies, true);
                                        if let Some((n, _)) = kcv {
                                            vis_cycles.push(n);
                                        }
                                        if let Some(d) = kcd {
                                            del_bodies.push(d);
                                        }
                                    }
                                });
                        }
                    }
                }
            });
            egui::CollapsingHeader::new(RichText::new(format!("🔧  {}", model.analysis.kind.label())).strong()).default_open(true).show(ui, |ui| {
                if model.loads.is_empty() {
                    ui.label(RichText::new("No supports or loads").color(p.weak));
                }
                for l in &model.loads {
                    if !q.is_empty() && !l.load.label().to_lowercase().contains(&q) && !l.load.summary().to_lowercase().contains(&q) {
                        continue;
                    }
                    ui.horizontal(|ui| {
                        let c = render::load_color(&l.load);
                        let (r, _) = ui.allocate_exact_size(vec2(9.0, 9.0), Sense::hover());
                        ui.painter().circle_filled(r.center(), 4.5, Color32::from_rgb(c[0], c[1], c[2]));
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if del_load(ui) {
                                self.run("remove_load", json!({"id": l.id}));
                            }
                            ui.label(RichText::new(format!("#{}", l.id)).small().color(p.weak));
                            ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                                let picked = self.sel_load == Some(l.id);
                                let txt = if picked { RichText::new(l.load.label()).strong().color(p.accent) } else { RichText::new(l.load.label()) };
                                if ui.add(egui::Label::new(txt).truncate().sense(Sense::click())).on_hover_text(format!("#{} {}\nClick to edit", l.id, l.load.summary())).clicked() {
                                    self.sel_load = if picked { None } else { Some(l.id) };
                                    self.sel_body = None;
                                    self.edit_target = None;
                                    self.form = Form::None;
                                    self.props_open = true;
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

        for n in vis_cycles {
            self.cycle_body_vis(&n);
        }
        for d in del_bodies {
            self.run("remove_body", json!({"name": d}));
        }
    }
}
