//! Dockable Workbench bottom drawer (and optional detached windows) for Charts & Path Plot,
//! Studies & Convergence, PCB DC Drop, AI Simulation Planner, and in-app User Guide.

use super::theme::{card, line_chart, primary, render_md};
use super::{DrawerTab, PcbUi, SimApp};
use crate::model::AnalysisType;
use crate::render;
use eframe::egui::{self, Color32, RichText, vec2};
use serde_json::{Value, json};
use std::sync::Arc;

impl SimApp {
    pub(super) fn poll_background_jobs(&mut self, ctx: &egui::Context) {
        let opt_finished = self.opt.job.as_ref().map(|job| job.lock().unwrap_or_else(std::sync::PoisonError::into_inner).take());
        match opt_finished {
            Some(Some(r)) => {
                self.opt.error = r.err();
                self.opt.show_results = self.opt.error.is_none();
                self.opt.job = None;
                if self.opt.show_results {
                    self.open_drawer(DrawerTab::Study);
                }
            }
            Some(None) => ctx.request_repaint_after(std::time::Duration::from_millis(250)),
            None => {}
        }
        let sig = self.core.session.lock().unwrap_or_else(std::sync::PoisonError::into_inner).study.as_ref().map(|st| format!("{}|{}|{}", st.kind, st.message, st.seconds));
        if let Some(sig) = sig {
            if sig != self.opt.seen {
                self.opt.seen = sig;
                self.opt.show_results = true;
                self.opt.whatif.clear();
                self.opt.whatif_key.clear();
                self.open_drawer(DrawerTab::Study);
            }
        }

        if let Some(job) = &self.planner.job {
            let done = job.lock().unwrap_or_else(std::sync::PoisonError::into_inner).take();
            if let Some(r) = done {
                self.planner.job = None;
                self.planner.error = r.err();
            } else {
                ctx.request_repaint_after(std::time::Duration::from_millis(200));
            }
        }

        if let Some((_cancel, slot)) = &self.pcb.nets_job {
            let done = slot.lock().unwrap_or_else(std::sync::PoisonError::into_inner).take();
            if let Some((done_path, r)) = done {
                self.pcb.nets_job = None;
                if done_path == self.pcb.path {
                    match r {
                        Ok(nets) => {
                            if nets.is_empty() {
                                self.pcb.error = Some("This board has no nets with two or more pads".into());
                            } else if self.pcb.net.is_empty() && self.pcb.from.is_empty() && self.pcb.to.is_empty() {
                                if let Some(i) = super::default_net(&nets) {
                                    self.pcb.net = nets[i].0.clone();
                                    self.pcb.from = nets[i].1.first().cloned().unwrap_or_default();
                                    self.pcb.to = nets[i].1.last().cloned().unwrap_or_default();
                                }
                            }
                            self.pcb.nets = nets;
                        }
                        Err(e) => self.pcb.error = Some(e),
                    }
                }
            } else {
                ctx.request_repaint_after(std::time::Duration::from_millis(80));
            }
        }
        if let Some(job) = &self.pcb.job {
            let done = job.lock().unwrap_or_else(std::sync::PoisonError::into_inner).take();
            if let Some(r) = done {
                match r {
                    Ok(v) => {
                        self.pcb.result = Some(v);
                        self.pcb.error = None;
                    }
                    Err(e) => {
                        self.pcb.result = None;
                        self.pcb.error = Some(e);
                    }
                }
                self.pcb.job = None;
            } else {
                ctx.request_repaint_after(std::time::Duration::from_millis(100));
            }
        }
    }

    pub(super) fn tab_enabled(&self, tab: DrawerTab) -> bool {
        match tab {
            DrawerTab::Charts => self.show_chart,
            DrawerTab::Study => self.opt.show_results,
            DrawerTab::Pcb => self.pcb.open,
            DrawerTab::Planner => self.planner.open,
            DrawerTab::Help => self.help.open,
        }
    }

    pub(super) fn open_drawer(&mut self, tab: DrawerTab) {
        match tab {
            DrawerTab::Charts => self.show_chart = true,
            DrawerTab::Study => self.opt.show_results = true,
            DrawerTab::Pcb => self.pcb.open = true,
            DrawerTab::Planner => self.planner.open = true,
            DrawerTab::Help => self.help.open = true,
        }
        if !self.drawer.detached.contains(&tab) {
            self.drawer.active = tab;
            self.drawer.open = true;
        }
    }

    pub(super) fn close_drawer_tab(&mut self, tab: DrawerTab) {
        match tab {
            DrawerTab::Charts => self.show_chart = false,
            DrawerTab::Study => self.opt.show_results = false,
            DrawerTab::Pcb => {
                self.pcb.open = false;
                if let Some((cancel, _)) = self.pcb.nets_job.take() {
                    cancel.store(true, std::sync::atomic::Ordering::Relaxed);
                }
            }
            DrawerTab::Planner => self.planner.open = false,
            DrawerTab::Help => self.help.open = false,
        }
        self.drawer.detached.remove(&tab);
        if self.drawer.active == tab {
            if let Some(next) = DrawerTab::ALL.into_iter().find(|&t| self.tab_enabled(t) && !self.drawer.detached.contains(&t)) {
                self.drawer.active = next;
            } else {
                self.drawer.open = false;
            }
        }
    }

    pub(super) fn workbench_ui(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        let docked: Vec<DrawerTab> = DrawerTab::ALL
            .into_iter()
            .filter(|&t| self.tab_enabled(t) && !self.drawer.detached.contains(&t))
            .collect();

        if self.drawer.open && !docked.is_empty() {
            if !docked.contains(&self.drawer.active) {
                self.drawer.active = docked[0];
            }
            let p = self.p;
            let bar = egui::Frame::new().fill(p.panel).inner_margin(egui::Margin::symmetric(12, 8)).stroke(egui::Stroke::new(1.0, p.border));
            let active = self.drawer.active;
            egui::Panel::bottom("workbench")
                .frame(bar)
                .resizable(true)
                .default_size(270.0)
                .min_size(160.0)
                .max_size((ctx.content_rect().height() * 0.62).max(240.0))
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        for &t in &docked {
                            let on = t == active;
                            if ui.selectable_label(on, t.label()).clicked() {
                                self.drawer.active = t;
                            }
                        }
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui.small_button("✖").on_hover_text("Close this panel").clicked() {
                                self.close_drawer_tab(active);
                            }
                            if ui.small_button("⎯").on_hover_text("Minimize Workbench drawer").clicked() {
                                self.drawer.open = false;
                            }
                            if ui.small_button("↗ Pop out").on_hover_text("Detach this panel into a separate window").clicked() {
                                self.drawer.detached.insert(active);
                                if let Some(next) = docked.iter().copied().find(|&x| x != active) {
                                    self.drawer.active = next;
                                } else {
                                    self.drawer.open = false;
                                }
                            }
                        });
                    });
                    ui.separator();
                    let cur = self.drawer.active;
                    if self.tab_enabled(cur) && !self.drawer.detached.contains(&cur) {
                        egui::ScrollArea::vertical().id_salt(("wb-scroll", cur)).auto_shrink([false, false]).show(ui, |ui| {
                            self.draw_drawer_tab(&ctx, ui, cur);
                        });
                    }
                });
        }

        let detached: Vec<DrawerTab> = DrawerTab::ALL.into_iter().filter(|&t| self.tab_enabled(t) && self.drawer.detached.contains(&t)).collect();
        for tab in detached {
            let p = self.p;
            let mut open = true;
            let mut dock_back = false;
            let default_sz = match tab {
                DrawerTab::Help => vec2(880.0, 600.0),
                DrawerTab::Study => vec2(720.0, 420.0),
                _ => vec2(620.0, 440.0),
            };
            egui::Window::new(RichText::new(tab.label()).strong())
                .open(&mut open)
                .resizable(true)
                .collapsible(false)
                .default_size(default_sz)
                .frame(egui::Frame::window(&ctx.global_style()).fill(p.card).inner_margin(12))
                .show(&ctx, |ui| {
                    ui.horizontal(|ui| {
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui.small_button("⬇ Dock").on_hover_text("Dock back into Workbench drawer").clicked() {
                                dock_back = true;
                            }
                        });
                    });
                    egui::ScrollArea::vertical().id_salt(("wb-pop", tab)).auto_shrink([false, false]).show(ui, |ui| {
                        self.draw_drawer_tab(&ctx, ui, tab);
                    });
                });
            if dock_back {
                self.drawer.detached.remove(&tab);
                self.drawer.active = tab;
                self.drawer.open = true;
            } else if !open {
                self.close_drawer_tab(tab);
            }
        }
    }

    fn draw_drawer_tab(&mut self, ctx: &egui::Context, ui: &mut egui::Ui, tab: DrawerTab) {
        match tab {
            DrawerTab::Charts => self.draw_charts(ui),
            DrawerTab::Study => self.draw_study(ui),
            DrawerTab::Pcb => self.draw_pcb(ctx, ui),
            DrawerTab::Planner => self.draw_planner(ctx, ui),
            DrawerTab::Help => self.draw_help(ui),
        }
    }

    fn draw_charts(&mut self, ui: &mut egui::Ui) {
        let (res, mesh, disp) = {
            let s = self.core.session.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            (s.results.clone(), s.mesh.clone(), s.display.clone())
        };
        let (Some(r), Some(m)) = (res, mesh) else {
            self.show_chart = false;
            return;
        };
        let rid = Arc::as_ptr(&r) as u64;
        if self.path.inited_res != rid {
            self.path.inited_res = rid;
            let (lo, hi) = m.bbox;
            let ym = 0.5 * (lo[1] + hi[1]);
            let zm = 0.5 * (lo[2] + hi[2]);
            self.path.from = [lo[0], ym, zm];
            self.path.to = [hi[0], ym, zm];
            self.path.cache_key.clear();
        }
        if r.history.is_empty() && self.path.tab == 0 {
            self.path.tab = 1;
        }
        let p = self.p;
        if !r.history.is_empty() {
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.path.tab, 0, "📈 History");
                ui.selectable_value(&mut self.path.tab, 1, "📏 Path plot");
            });
            ui.separator();
        } else {
            self.path.tab = 1;
        }
        if self.path.tab == 0 {
            let label = if r.history_label.is_empty() { "value".to_string() } else { r.history_label.clone() };
            let xl = if r.history_x.is_empty() { "step".to_string() } else { r.history_x.clone() };
            let sz = vec2(ui.available_width().max(300.0), (ui.available_height() - 10.0).max(170.0));
            line_chart(ui, &p, &[(label, r.history.clone())], &xl, sz);
            return;
        }
        let avail = r.available_fields();
        let field = self.path.field.filter(|f| r.has(*f)).unwrap_or_else(|| if r.has(disp.field) { disp.field } else { avail[0] });
        ui.horizontal(|ui| {
            ui.label("Field");
            let mut picked = field;
            egui::ComboBox::from_id_salt("path-field").width(200.0).selected_text(field.label()).show_ui(ui, |ui| {
                for &f in &avail {
                    ui.selectable_value(&mut picked, f, f.label());
                }
            });
            self.path.field = Some(picked);
            ui.label("Samples");
            ui.add(egui::DragValue::new(&mut self.path.samples).range(2..=500));
            ui.label(RichText::new("A").strong().color(p.accent));
            super::theme::xyz(ui, "path-a", &mut self.path.from, " mm", 0.5);
            ui.label(RichText::new("B").strong().color(p.ok));
            super::theme::xyz(ui, "path-b", &mut self.path.to, " mm", 0.5);
        });
        ui.horizontal(|ui| {
            let pick_lbl = match self.path.picking {
                1 => "Click start point A in 3D…",
                2 => "Click end point B in 3D…",
                _ => "🎯 Pick 2 points",
            };
            if ui.selectable_label(self.path.picking > 0, pick_lbl).clicked() {
                self.path.picking = if self.path.picking > 0 { 0 } else { 1 };
            }
            let two_probes = self.probes.iter().rev().filter(|(_, id)| *id == rid).take(2).map(|(n, _)| *n).collect::<Vec<_>>();
            if ui.add_enabled(two_probes.len() == 2, egui::Button::new("Use last 2 probes")).clicked() {
                self.path.from = m.nodes[two_probes[1] as usize];
                self.path.to = m.nodes[two_probes[0] as usize];
            }
            if let Some(pp) = &self.path.plot {
                if ui.button("💾 Export CSV…").clicked() {
                    if let Some(path) = rfd::FileDialog::new().add_filter("CSV", &["csv"]).set_file_name("path-plot.csv").save_file() {
                        match std::fs::write(&path, pp.to_csv()) {
                            Ok(()) => self.toast = Some((format!("Saved {}", path.display()), false, 4.0)),
                            Err(e) => self.toast = Some((format!("CSV error: {e}"), true, 6.0)),
                        }
                    }
                }
                if ui.button("📋 Copy CSV").clicked() {
                    ui.ctx().copy_text(pp.to_csv());
                    self.toast = Some(("Copied path CSV to clipboard".into(), false, 3.0));
                }
            }
        });
        let f_use = self.path.field.filter(|f| r.has(*f)).unwrap_or(field);
        let key = format!("{rid}|{}|{}|{:?}|{:?}|{:?}|{}", f_use.key(), disp.mode, disp.cylindrical, self.path.from, self.path.to, self.path.samples);
        if key != self.path.cache_key {
            self.path.cache_key = key;
            match r.path_plot(&m, f_use, disp.mode, disp.cylindrical, self.path.from, self.path.to, self.path.samples) {
                Ok(pp) => {
                    self.path.plot = Some(pp);
                    self.path.error = None;
                }
                Err(e) => {
                    self.path.plot = None;
                    self.path.error = Some(format!("{e:#}"));
                }
            }
        }
        if let Some(e) = &self.path.error {
            ui.label(RichText::new(format!("⚠ {e}")).color(p.err));
        } else if let Some(pp) = &self.path.plot {
            let series_pts: Vec<(f64, f64)> = pp.points.iter().filter_map(|pt| pt.value.map(|v| (pt.distance_mm, v))).collect();
            let title = if pp.unit.is_empty() { pp.label.clone() } else { format!("{} [{}]", pp.label, pp.unit) };
            let sz = vec2(ui.available_width().max(300.0), (ui.available_height() - 26.0).max(160.0));
            line_chart(ui, &p, &[(title, series_pts)], "distance along path [mm]", sz);
            if let (Some(mn), Some(mx), Some(mean)) = (pp.min, pp.max, pp.mean) {
                ui.label(RichText::new(format!("min {} · max {} · mean {} {} · length {} mm ({}/{} inside)", render::fmt_val(mn), render::fmt_val(mx), render::fmt_val(mean), pp.unit, render::fmt_val(pp.length_mm), pp.valid_samples, pp.points.len())).small().color(p.weak));
            }
        }
    }

    fn draw_study(&mut self, ui: &mut egui::Ui) {
        let Some(st) = self.core.session.lock().unwrap_or_else(std::sync::PoisonError::into_inner).study.clone() else {
            ui.label("No study results yet.");
            return;
        };
        let p = self.p;
        let mut actions: Vec<(String, Value)> = Vec::new();
        ui.label(RichText::new(&st.message).strong());
        ui.label(RichText::new(format!("{:.1} s", st.seconds)).small().color(p.weak));
        ui.add_space(6.0);
        if st.kind == "convergence" {
            if let Some(rec) = st.recommended_size_mm {
                ui.horizontal(|ui| {
                    if ui.add(primary(&p, &format!("Apply {rec} mm mesh"))).clicked() {
                        actions.push(("apply_design".into(), json!({"values": {"element_size_mm": rec}})));
                    }
                    if let Some(t) = st.target_error_pct {
                        ui.label(RichText::new(format!("Target error <= {t}% (set element_size_mm then Solve)")).small().color(p.weak));
                    }
                });
                ui.add_space(6.0);
            }
            egui::Grid::new("conv-summary").striped(true).show(ui, |ui| {
                for h in ["Response", "Fine value", "Extrapolated", "Order p", "Est. err %", "GCI %", "Rec. h (mm)", "Status"] {
                    ui.label(RichText::new(h).strong());
                }
                ui.end_row();
                for rep in &st.convergence {
                    ui.label(RichText::new(&rep.response).strong());
                    ui.label(rep.values.last().and_then(|v| *v).map(render::fmt_val).unwrap_or_else(|| "—".into()));
                    ui.label(rep.extrapolated.map(render::fmt_val).unwrap_or_else(|| "—".into()));
                    ui.label(rep.observed_order.map(|x| format!("{x:.2}")).unwrap_or_else(|| "est. 2".into()));
                    ui.label(rep.fine_error_pct.map(|x| format!("{x:.2}%")).unwrap_or_else(|| "—".into()));
                    ui.label(rep.gci_fine_pct.map(|x| format!("{x:.2}%")).unwrap_or_else(|| "—".into()));
                    ui.label(rep.recommended_size_mm.map(render::fmt_val).unwrap_or_else(|| "—".into()));
                    ui.label(RichText::new(&rep.note).small().color(if rep.monotonic { p.ok } else { p.warn }));
                    ui.end_row();
                }
            });
            ui.add_space(8.0);
            egui::Grid::new("conv-steps").striped(true).show(ui, |ui| {
                ui.label(RichText::new("h (mm)").strong());
                ui.label(RichText::new("Elements").strong());
                ui.label(RichText::new("DOFs").strong());
                for r in &st.responses {
                    ui.label(RichText::new(r).strong());
                }
                ui.end_row();
                for s in &st.convergence_steps {
                    ui.label(render::fmt_val(s.element_size_mm));
                    ui.label(s.elements.to_string());
                    ui.label(s.dofs.to_string());
                    for v in &s.responses {
                        ui.label(v.map(render::fmt_val).unwrap_or_else(|| "—".into()));
                    }
                    ui.end_row();
                }
            });
            ui.add_space(6.0);
            for (ri, rname) in st.responses.iter().enumerate() {
                let pts: Vec<(f64, f64)> = st.convergence_steps.iter().filter_map(|s| Some((s.element_size_mm, s.responses.get(ri).copied().flatten()?))).collect();
                if pts.len() >= 2 {
                    line_chart(ui, &p, &[(rname.clone(), pts)], "element size h [mm]", vec2(ui.available_width().max(300.0), 140.0));
                    ui.add_space(4.0);
                }
            }
        } else if st.kind == "topology" {
            egui::Grid::new("topohist").striped(true).num_columns(3).show(ui, |ui| {
                ui.label(RichText::new("Iteration").strong());
                ui.label(RichText::new("Compliance (J)").strong());
                ui.label(RichText::new("Volume").strong());
                ui.end_row();
                let stride = (st.history.len() / 10).max(1);
                for (i, c, v) in st.history.iter().step_by(stride) {
                    ui.label(i.to_string());
                    ui.label(render::fmt_val(*c));
                    ui.label(format!("{:.0}%", v * 100.0));
                    ui.end_row();
                }
            });
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                if ui.button("Show density").clicked() {
                    actions.push(("set_display".into(), json!({"field": "density", "show_results": true})));
                }
                if let Some(path) = &st.stl_path {
                    if ui.add(primary(&p, "Open optimized shape")).on_hover_text(path).clicked() {
                        let mat = self.core.session.lock().unwrap_or_else(std::sync::PoisonError::into_inner).model.bodies.iter().find(|b| b.role == crate::geometry::Role::Solid).map(|b| b.material.clone()).unwrap_or_default();
                        actions.push(("new_model".into(), json!({})));
                        actions.push(("import_stl".into(), json!({"path": path, "name": "Optimized"})));
                        if !mat.is_empty() {
                            actions.push(("set_material".into(), json!({"material": mat})));
                        }
                    }
                    ui.label(RichText::new(path).small().color(p.weak));
                }
            });
        } else {
            if st.runs.iter().filter(|r| r.error.is_none()).count() >= 3 {
                egui::CollapsingHeader::new(RichText::new("🔮 What-if (surrogate estimate)").strong()).default_open(true).show(ui, |ui| {
                    if self.opt.whatif.len() != st.parameters.len() {
                        self.opt.whatif = st.best.and_then(|b| st.runs.get(b)).map(|r| r.values.clone()).unwrap_or_else(|| st.parameters.iter().map(|q| (q.min + q.max) / 2.0).collect());
                    }
                    for (k, q) in st.parameters.iter().enumerate() {
                        ui.horizontal(|ui| {
                            ui.label(RichText::new(&q.target).color(p.accent));
                            ui.add(egui::Slider::new(&mut self.opt.whatif[k], q.min..=q.max));
                        });
                    }
                    let key = format!("{}|{}|{:?}", st.runs.len(), st.seconds, self.opt.whatif);
                    if key != self.opt.whatif_key {
                        self.opt.whatif_key = key;
                        let vals: serde_json::Map<String, Value> = st.parameters.iter().zip(&self.opt.whatif).map(|(q, v)| (q.target.clone(), json!(v))).collect();
                        self.opt.whatif_pred = self.core.call("predict_surrogate", &json!({"values": vals})).ok().and_then(|o| serde_json::from_str(&o.text).ok());
                    }
                    if let Some(pred) = &self.opt.whatif_pred {
                        egui::Grid::new("whatif").num_columns(2).show(ui, |ui| {
                            for row in pred["predictions"].as_array().into_iter().flatten() {
                                ui.label(row["response"].as_str().unwrap_or(""));
                                match row["estimate"].as_f64() {
                                    Some(v) => ui.label(RichText::new(format!("{} ± {}", render::fmt_val(v), render::fmt_val(row["plus_minus_2sigma"].as_f64().unwrap_or(0.0)))).strong()),
                                    None => ui.label("—"),
                                };
                                ui.end_row();
                            }
                        });
                    }
                    ui.horizontal(|ui| {
                        ui.label(RichText::new("Surrogate estimate, not verified (±2σ).").small().color(p.warn));
                        if ui.small_button("Verify with a real solve").on_hover_text("Applies these values to the model and solves").clicked() {
                            let vals: serde_json::Map<String, Value> = st.parameters.iter().zip(&self.opt.whatif).map(|(q, v)| (q.target.clone(), json!(v))).collect();
                            actions.push(("apply_design".into(), json!({"values": vals})));
                            actions.push(("__solve".into(), json!({})));
                        }
                    });
                });
                ui.add_space(6.0);
            }
            if st.parameters.len() == 1 && st.runs.iter().filter(|r| r.error.is_none()).count() >= 2 {
                egui::CollapsingHeader::new(RichText::new("📈 Chart").strong()).default_open(false).show(ui, |ui| {
                    for (ri, name) in st.responses.iter().enumerate() {
                        let mut pts: Vec<(f64, f64)> = st.runs.iter().filter_map(|r| Some((r.values[0], r.responses.get(ri).copied().flatten()?))).collect();
                        pts.sort_by(|a, b| a.0.total_cmp(&b.0));
                        line_chart(ui, &p, &[(name.clone(), pts)], &st.parameters[0].target, vec2(ui.available_width().max(300.0), 150.0));
                        ui.add_space(4.0);
                    }
                });
                ui.add_space(6.0);
            }
            egui::Grid::new("studytable").striped(true).show(ui, |ui| {
                ui.label(RichText::new("#").strong());
                for q in &st.parameters {
                    ui.label(RichText::new(&q.target).strong().color(p.accent));
                }
                for r in &st.responses {
                    ui.label(RichText::new(r).strong());
                }
                ui.label("");
                ui.end_row();
                for (i, run) in st.runs.iter().enumerate() {
                    let best = st.best == Some(i);
                    let mark = |t: String| if best { RichText::new(t).strong().color(p.ok) } else { RichText::new(t) };
                    ui.label(mark(format!("{}{}", i + 1, if best { " ★" } else { "" })));
                    for v in &run.values {
                        ui.label(mark(render::fmt_val(*v)));
                    }
                    for v in &run.responses {
                        ui.label(mark(v.map(render::fmt_val).unwrap_or_else(|| "—".into())));
                    }
                    ui.horizontal(|ui| {
                        if (st.kind == "optimization" || st.kind == "surrogate_optimization") && !run.feasible {
                            ui.label(RichText::new("violates").small().color(p.warn));
                        }
                        if let Some(e) = &run.error {
                            ui.label(RichText::new("failed").small().color(p.err)).on_hover_text(e);
                        } else if ui.small_button("Apply").clicked() {
                            actions.push(("apply_design".into(), json!({"run": i + 1})));
                        }
                    });
                    ui.end_row();
                }
            });
        }
        for (t, a) in actions {
            if t == "__solve" {
                self.solve();
                continue;
            }
            self.run(&t, a);
            if t == "import_stl" {
                self.need_fit = true;
            }
        }
    }

    pub(super) fn open_pcb_dc(&mut self, ctx: &egui::Context, path: &str) {
        if self.pcb.path == path && (!self.pcb.nets.is_empty() || self.pcb.nets_job.is_some()) {
            self.open_drawer(DrawerTab::Pcb);
            return;
        }
        if let Some((old_cancel, _)) = self.pcb.nets_job.take() {
            old_cancel.store(true, std::sync::atomic::Ordering::Relaxed);
        }
        let cancel = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let slot: super::NetsJobSlot = Arc::new(std::sync::Mutex::new(None));
        self.pcb = PcbUi {
            open: true,
            path: path.to_string(),
            current: self.pcb.current,
            temp: self.pcb.temp,
            nets_job: Some((cancel.clone(), slot.clone())),
            ..Default::default()
        };
        let path_owned = path.to_string();
        let ctx_clone = ctx.clone();
        std::thread::spawn(move || {
            let r = crate::kicad::dc::net_pads(&path_owned).map_err(|e| format!("{e:#}"));
            if !cancel.load(std::sync::atomic::Ordering::Relaxed) {
                *slot.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = Some((path_owned, r));
                ctx_clone.request_repaint();
            }
        });
        self.open_drawer(DrawerTab::Pcb);
    }

    pub(super) fn start_pcb_job(&mut self) {
        let f = &mut self.pcb;
        let slot = Arc::new(std::sync::Mutex::new(None));
        let (path, net, from, to, cur, temp) = (f.path.clone(), f.net.clone(), f.from.clone(), f.to.clone(), f.current, f.temp);
        let out = slot.clone();
        std::thread::spawn(move || {
            let r = std::panic::catch_unwind(|| crate::kicad::dc::dc_drop(&path, Some(&net), &from, &to, cur, temp))
                .unwrap_or_else(|_| Err(anyhow::anyhow!("internal error in the DC solver")))
                .map_err(|e| format!("{e:#}"));
            *out.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = Some(r);
        });
        f.job = Some(slot);
        f.error = None;
    }

    fn draw_pcb(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
        let p = self.p;
        let mut go = false;
        let mut help = false;
        let mut cancel_nets = false;
        let f = &mut self.pcb;
        ui.horizontal(|ui| {
            ui.label(RichText::new("Current through one net between two pads: resistance, IR drop, current density and IPC-2221 track heating.").small().color(p.weak));
            if ui.small_button("ⓘ").on_hover_text("How the DC drop is computed").clicked() {
                help = true;
            }
        });
        ui.add_space(4.0);
        if f.nets_job.is_some() {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label(RichText::new("Reading KiCad board and extracting nets…").color(p.weak));
                if ui.small_button("Cancel").clicked() {
                    cancel_nets = true;
                }
            });
            ui.add_space(4.0);
        }
        egui::Grid::new("pcbdc-in").num_columns(2).spacing([10.0, 6.0]).show(ui, |ui| {
            ui.label("Net");
            ui.horizontal(|ui| {
                let cur = f.nets.iter().find(|(n, _)| *n == f.net).map(|(n, p)| format!("{n}  ({} pads)", p.len())).unwrap_or_else(|| "choose…".into());
                let mut picked: Option<String> = None;
                egui::ComboBox::from_id_salt("pcbdc-net").width(240.0).height(360.0).selected_text(cur).show_ui(ui, |ui| {
                    ui.add(egui::TextEdit::singleline(&mut f.filter).hint_text("🔎 filter nets").desired_width(220.0));
                    let q = f.filter.to_lowercase();
                    for (n, pads) in f.nets.iter().filter(|(n, _)| q.is_empty() || n.to_lowercase().contains(&q)).take(300) {
                        if ui.selectable_label(*n == f.net, format!("{n}  ({} pads)", pads.len())).clicked() {
                            picked = Some(n.clone());
                        }
                    }
                });
                if let Some(n) = picked {
                    if n != f.net {
                        let pads = f.nets.iter().find(|(m, _)| *m == n).map(|x| x.1.clone()).unwrap_or_default();
                        f.from = pads.first().cloned().unwrap_or_default();
                        f.to = pads.last().cloned().unwrap_or_default();
                        f.net = n;
                        f.result = None;
                    }
                }
            });
            ui.end_row();
            let pads: Vec<String> = f.nets.iter().find(|(n, _)| *n == f.net).map(|x| x.1.clone()).unwrap_or_default();
            for (label, which) in [("From pad (source)", 0), ("To pad (sink)", 1)] {
                ui.label(label);
                ui.horizontal(|ui| {
                    let val = if which == 0 { &mut f.from } else { &mut f.to };
                    egui::ComboBox::from_id_salt(("pcbdc-pad", which)).width(140.0).height(320.0).selected_text(val.clone()).show_ui(ui, |ui| {
                        for pd in &pads {
                            ui.selectable_value(val, pd.clone(), pd);
                        }
                    });
                    if which == 1 && ui.small_button("⇄").on_hover_text("Swap source and sink").clicked() {
                        std::mem::swap(&mut f.from, &mut f.to);
                    }
                });
                ui.end_row();
            }
            ui.label("Current");
            ui.add(egui::DragValue::new(&mut f.current).speed(0.05).range(0.001..=1000.0).suffix(" A"));
            ui.end_row();
            ui.label("Copper temperature");
            ui.add(egui::DragValue::new(&mut f.temp).speed(1.0).range(-60.0..=250.0).suffix(" °C")).on_hover_text("Copper resistivity rises 0.393 % per °C");
            ui.end_row();
        });
        ui.add_space(6.0);
        let running = f.job.is_some() || f.nets_job.is_some();
        ui.horizontal(|ui| {
            let ok = !f.from.is_empty() && !f.to.is_empty() && f.from != f.to && !running;
            go = ui.add_enabled(ok, primary(&p, "Calculate")).clicked();
            if f.job.is_some() {
                ui.spinner();
                ui.label(RichText::new("solving the copper network…").color(p.weak));
            } else if f.from == f.to && !f.from.is_empty() {
                ui.label(RichText::new("pick two different pads").color(p.warn));
            }
        });
        if cancel_nets {
            if let Some((c, _)) = f.nets_job.take() {
                c.store(true, std::sync::atomic::Ordering::Relaxed);
            }
        }
        if let Some(e) = &f.error {
            ui.add_space(4.0);
            ui.label(RichText::new(format!("⚠ {e}")).color(p.err));
        }
        if let Some(r) = &f.result {
            ui.add_space(6.0);
            ui.separator();
            let num = |k: &str| r[k].as_f64().unwrap_or(f64::NAN);
            let at = |k: &str| r[k].as_array().map(|a| format!("({:.2}, {:.2}) mm", a[0].as_f64().unwrap_or(0.0), a[1].as_f64().unwrap_or(0.0))).unwrap_or_default();
            egui::Grid::new("pcbdc-out").num_columns(2).spacing([16.0, 4.0]).show(ui, |ui| {
                for (label, value) in [
                    ("Resistance", format!("{:.3} mΩ", num("resistance_mohm"))),
                    ("Voltage drop", format!("{:.2} mV at {} A", num("voltage_drop_mv"), crate::results::float(num("current_a")))),
                    ("Power in the copper", format!("{:.4} W", num("power_w"))),
                    ("Peak current density", format!("{:.1} A/mm²  on {} at {}", num("max_current_density_a_per_mm2"), r["max_current_density_layer"].as_str().unwrap_or(""), at("max_current_density_at_mm"))),
                    ("Largest via current", if num("vias_and_holes") > 0.0 { format!("{:.3} A at {}", num("max_via_current_a"), at("max_via_at_mm")) } else { "no vias in the path".into() }),
                ] {
                    ui.label(RichText::new(label).color(p.weak));
                    ui.label(RichText::new(value).strong());
                    ui.end_row();
                }
            });
            ui.add_space(6.0);
            ui.label(RichText::new("Tracks carrying the most current").strong());
            egui::Grid::new("pcbdc-tracks").striped(true).num_columns(7).spacing([12.0, 3.0]).show(ui, |ui| {
                for h in ["Layer", "Width", "Length", "Current", "J", "IPC ΔT", "Width for 10 °C"] {
                    ui.label(RichText::new(h).small().strong());
                }
                ui.end_row();
                for t in r["tracks_by_current"].as_array().cloned().unwrap_or_default() {
                    let g = |k: &str| t[k].as_f64().unwrap_or(0.0);
                    let rise = g("ipc2221_temperature_rise_c");
                    let short = g("length_mm") < crate::kicad::dc::SHORT_MM;
                    let col = if short { p.weak } else if rise > 20.0 { p.err } else if rise > 10.0 { p.warn } else { p.text };
                    ui.label(t["layer"].as_str().unwrap_or(""));
                    ui.label(format!("{:.3} mm", g("width_mm")));
                    ui.label(format!("{:.1} mm", g("length_mm")));
                    ui.label(format!("{:.3} A", g("current_a")));
                    ui.label(format!("{:.1} A/mm²", g("current_density_a_per_mm2")));
                    ui.label(RichText::new(format!("{rise:.1} °C{}", if short { " *" } else { "" })).color(col)).on_hover_text(if short { "Neck-down shorter than 2 mm: the copper around it carries heat away, so it runs cooler than IPC-2221's long-track figure" } else { "IPC-2221 rise for this current in a long track of this width" });
                    ui.label(format!("{:.3} mm", g("ipc2221_width_for_10c_mm")));
                    ui.end_row();
                }
            });
            for w in r["warnings"].as_array().cloned().unwrap_or_default() {
                ui.label(RichText::new(format!("⚠ {}", w.as_str().unwrap_or(""))).small().color(p.warn));
            }
            ui.horizontal(|ui| {
                ui.label(RichText::new(format!("{} unknowns on a {:.3} mm raster · DC, no skin effect · IPC rise per track alone · * short neck-down", r["unknowns"], num("raster_mm"))).small().color(p.weak));
                if ui.small_button("Copy JSON").clicked() {
                    ctx.copy_text(serde_json::to_string_pretty(r).unwrap_or_default());
                }
            });
        }
        if help {
            self.open_help("DC current in a net");
        }
        if go {
            self.start_pcb_job();
        }
        if cancel_nets {
            self.close_drawer_tab(DrawerTab::Pcb);
        }
    }

    fn draw_planner(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
        let p = self.p;
        if self.planner.models.is_none() {
            self.planner.models = Some(crate::assistant::list_models().map_err(|e| format!("{e:#}")));
            if let Some(Ok(m)) = &self.planner.models {
                self.planner.model = crate::assistant::default_model(m).unwrap_or_default();
            }
        }
        let busy = self.planner.job.is_some();
        let plan = self.core.session.lock().unwrap_or_else(std::sync::PoisonError::into_inner).plan.clone();
        let now = ctx.input(|i| i.time);
        let mut actions: Vec<(String, Value)> = Vec::new();
        ui.label(RichText::new("Describe what you want to simulate. A local AI model (running on this computer through Ollama) drafts a to-do list and flags anything SimCraft cannot do yet.").color(p.weak));
        ui.add_space(6.0);
        ui.add(egui::TextEdit::multiline(&mut self.planner.goal).hint_text("e.g. simulate EMI on my PCB · how hot does food get in a microwave · will this bracket bend under 2 kN").desired_rows(2).desired_width(f32::INFINITY));
        ui.add_space(6.0);
        ui.horizontal(|ui| {
            match &self.planner.models {
                Some(Ok(models)) if !models.is_empty() => {
                    ui.label("Model");
                    egui::ComboBox::from_id_salt("planner-model").width(220.0).selected_text(&self.planner.model).show_ui(ui, |ui| {
                        for m in models {
                            ui.selectable_value(&mut self.planner.model, m.clone(), m);
                        }
                    });
                }
                Some(Ok(_)) => {
                    ui.label(RichText::new("No local models installed — run: ollama pull qwen2.5:7b").color(p.warn));
                }
                Some(Err(e)) => {
                    ui.label(RichText::new(format!("⚠ {e}")).small().color(p.warn));
                }
                None => {}
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let ready = !busy && !self.planner.goal.trim().is_empty() && !self.planner.model.is_empty();
                if ui.add_enabled(ready, primary(&p, "Draft to-do list")).clicked() {
                    let slot = Arc::new(std::sync::Mutex::new(None));
                    let (core, goal, model, out) = (self.core.clone(), self.planner.goal.clone(), self.planner.model.clone(), slot.clone());
                    std::thread::spawn(move || {
                        let r = core.call("plan_simulation", &json!({"goal": goal, "model": model})).map(|_| ()).map_err(|e| format!("{e:#}"));
                        *out.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = Some(r);
                    });
                    self.planner.job = Some(slot);
                    self.planner.error = None;
                }
                if busy {
                    ui.add(egui::Spinner::new());
                    ui.label(RichText::new(format!("Thinking locally with {}…", self.planner.model)).color(p.weak));
                }
            });
        });
        if let Some(e) = &self.planner.error {
            ui.label(RichText::new(format!("✖ {e}")).color(p.err));
        }
        if let Some(plan) = plan {
            ui.add_space(6.0);
            ui.separator();
            ui.horizontal_wrapped(|ui| {
                ui.label(RichText::new(&plan.title).size(16.0).strong());
                let (txt, col) = match plan.feasibility.as_str() {
                    "supported" => ("Fully supported", p.ok),
                    "partial" => ("Partly supported", p.warn),
                    _ => ("Not supported yet", p.err),
                };
                ui.label(RichText::new(format!(" {txt} ")).small().strong().color(Color32::WHITE).background_color(col));
            });
            ui.label(RichText::new(&plan.summary).color(p.weak));
            if !plan.physics.is_empty() {
                ui.label(RichText::new(format!("Physics: {}", plan.physics.join(" · "))).small().color(p.weak));
            }
            if let Some(a) = AnalysisType::all().into_iter().find(|a| a.key() == plan.analysis) {
                ui.horizontal(|ui| {
                    ui.label(RichText::new(format!("Start with: {}", a.label())).strong());
                    if ui.small_button("Set up this analysis").clicked() {
                        actions.push(("set_analysis".into(), json!({"type": a.key()})));
                    }
                });
            }
            ui.add_space(6.0);
            let done = plan.steps.iter().filter(|s| s.done).count();
            ui.label(RichText::new(format!("TO-DO  ·  {done}/{} done", plan.steps.len())).size(11.0).strong().color(p.weak));
            for (i, st) in plan.steps.iter().enumerate() {
                card(ui, &p, |ui| {
                    ui.horizontal(|ui| {
                        let mut d = st.done;
                        if ui.checkbox(&mut d, "").changed() {
                            actions.push(("set_plan_step".into(), json!({"index": i + 1, "done": d})));
                        }
                        let mut t = RichText::new(format!("{}. {}", i + 1, st.task)).strong();
                        if st.done {
                            t = t.strikethrough().color(p.weak);
                        }
                        ui.add(egui::Label::new(t).wrap());
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            let (b, c) = match st.status.as_str() {
                                "workaround" => ("workaround", p.warn),
                                "missing" => ("missing in SimCraft", p.err),
                                _ => ("ready", p.ok),
                            };
                            ui.label(RichText::new(b).small().strong().color(c));
                        });
                    });
                    if !st.detail.is_empty() {
                        ui.label(RichText::new(&st.detail).small().color(p.weak));
                    }
                });
                ui.add_space(4.0);
            }
            if !plan.inputs_needed.is_empty() {
                ui.add_space(6.0);
                ui.label(RichText::new("YOU WILL NEED").size(11.0).strong().color(p.weak));
                for x in &plan.inputs_needed {
                    ui.label(format!("• {x}"));
                }
            }
            if !plan.missing_capabilities.is_empty() {
                ui.add_space(6.0);
                ui.label(RichText::new("NOT YET IN SIMCRAFT").size(11.0).strong().color(p.err));
                for x in &plan.missing_capabilities {
                    ui.label(RichText::new(format!("• {x}")).small());
                }
            }
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                let copied = now - self.planner.copied_at < 1.8;
                if ui.button(if copied { "✔ Copied" } else { "📋 Copy as Markdown" }).clicked() {
                    ctx.copy_text(crate::assistant::to_markdown(&plan));
                    self.planner.copied_at = now;
                }
                ui.label(RichText::new(format!("Drafted by {} — check the steps; small local models can be wrong.", plan.model)).small().italics().color(p.weak));
            });
        }
        for (t, a) in actions {
            self.run(&t, a);
        }
    }

    pub(super) fn open_help(&mut self, topic: &str) {
        if let Some(i) = crate::help::find(topic) {
            self.help.section = i;
        }
        self.help.query.clear();
        self.help.scroll_top = true;
        self.open_drawer(DrawerTab::Help);
    }

    pub(super) fn context_help(&mut self) {
        let kind = self.core.session.lock().unwrap_or_else(std::sync::PoisonError::into_inner).model.analysis.kind;
        let topic = match self.tab {
            super::Tab::Geometry => "4".to_string(),
            super::Tab::Material => "5".to_string(),
            super::Tab::Mesh => "6".to_string(),
            super::Tab::Setup if self.form != super::Form::None => "8".to_string(),
            super::Tab::Setup | super::Tab::Solve => crate::help::analysis_topic(&kind.key()).to_string(),
            super::Tab::Results => "9".to_string(),
            super::Tab::Optimize => "10".to_string(),
            super::Tab::Help => "2".to_string(),
        };
        self.open_help(&topic);
    }

    fn draw_help(&mut self, ui: &mut egui::Ui) {
        let p = self.p;
        let secs = crate::help::sections();
        let mut goto: Option<usize> = None;
        ui.horizontal(|ui| {
            let r = ui.add(egui::TextEdit::singleline(&mut self.help.query).hint_text("🔎 Search the guide (e.g. heat sink, fatigue, undo)").desired_width(360.0));
            if self.help.focus_search {
                r.request_focus();
                self.help.focus_search = false;
            }
            if !self.help.query.is_empty() && ui.small_button("✖").clicked() {
                self.help.query.clear();
            }
            ui.label(RichText::new("F1 opens the topic for what you are doing").small().color(p.weak));
        });
        ui.separator();
        let h = ui.available_height().max(140.0);
        ui.horizontal_top(|ui| {
            ui.vertical(|ui| {
                ui.set_width(240.0);
                egui::ScrollArea::vertical().id_salt("help-toc").max_height(h).show(ui, |ui| {
                    if self.help.query.trim().len() >= 2 {
                        let hits = crate::help::search(&self.help.query, 25);
                        if hits.is_empty() {
                            ui.label(RichText::new("No matches").color(p.weak));
                        }
                        for (i, _, snip) in hits {
                            let on = self.help.section == i;
                            if ui.selectable_label(on, RichText::new(&secs[i].title).strong()).clicked() {
                                goto = Some(i);
                            }
                            ui.label(RichText::new(snip).small().color(p.weak));
                            ui.add_space(4.0);
                        }
                    } else {
                        for (i, s) in secs.iter().enumerate().filter(|(i, s)| s.level >= 2 || crate::help::is_tutorial(*i)) {
                            if s.level == 1 {
                                ui.add_space(8.0);
                                if ui.selectable_label(i == self.help.section, RichText::new(s.title.to_uppercase()).small().strong().color(p.ok)).clicked() {
                                    goto = Some(i);
                                }
                                continue;
                            }
                            let on = crate::help::span(self.help.section).contains(&i) && (s.level == 3 || i == self.help.section) || i == self.help.section;
                            let txt = if s.level == 2 { RichText::new(&s.title).strong() } else { RichText::new(&s.title).small() };
                            ui.horizontal(|ui| {
                                ui.add_space(if s.level == 3 { 14.0 } else { 0.0 });
                                if ui.selectable_label(on, txt).clicked() {
                                    goto = Some(i);
                                }
                            });
                        }
                    }
                });
            });
            ui.separator();
            ui.vertical(|ui| {
                let mut sa = egui::ScrollArea::vertical().id_salt(("help-body", self.help.section)).max_height((h - 30.0).max(100.0)).auto_shrink([false, false]);
                if self.help.scroll_top {
                    sa = sa.vertical_scroll_offset(0.0);
                    self.help.scroll_top = false;
                }
                let hl = self.help.query.trim().to_string();
                sa.show(ui, |ui| {
                    for j in crate::help::span(self.help.section) {
                        let s = &secs[j];
                        let size = match s.level {
                            1 => 20.0,
                            2 => 18.0,
                            _ => 15.0,
                        };
                        ui.add_space(if j == self.help.section { 0.0 } else { 10.0 });
                        ui.label(RichText::new(&s.title).size(size).strong().color(p.text));
                        ui.add_space(4.0);
                        render_md(ui, &p, &s.body, &hl);
                    }
                });
                ui.horizontal(|ui| {
                    let chapters: Vec<usize> = secs.iter().enumerate().filter(|(_, s)| s.level <= 2).map(|(i, _)| i).collect();
                    let cur = chapters.iter().rposition(|&c| c <= self.help.section).unwrap_or(0);
                    if ui.add_enabled(cur > 0, egui::Button::new("◀ Previous")).clicked() {
                        goto = Some(chapters[cur - 1]);
                    }
                    if ui.add_enabled(cur + 1 < chapters.len(), egui::Button::new("Next ▶")).clicked() {
                        goto = Some(chapters[cur + 1]);
                    }
                });
            });
        });
        if let Some(i) = goto {
            self.help.section = i;
            self.help.scroll_top = true;
        }
    }
}
