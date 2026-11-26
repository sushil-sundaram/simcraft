//! Focused dialogs and safety modals: MCP connect setup, material editor, fatigue evaluation, and save/open confirmation.

use super::theme::{json_editor, primary};
use super::{Pending, SimApp};
use crate::model::AnalysisType;
use eframe::egui::{self, Align2, Color32, RichText, Stroke, vec2};
use serde_json::json;

pub(super) fn mcp_connect_panel(app: &mut SimApp, ctx: &egui::Context, ui: &mut egui::Ui) {
    let p = app.p;
    let exe = std::env::current_exe().map(|x| x.to_string_lossy().to_string()).unwrap_or("simcraft".into());
    let cmd = format!("claude mcp add --scope user simcraft -- {exe} mcp");
    let js = format!("{{\n  \"mcpServers\": {{\n    \"simcraft\": {{\n      \"command\": \"{exe}\",\n      \"args\": [\"mcp\"]\n    }}\n  }}\n}}");
    let ok = app.bridge.starts_with("MCP bridge listening");
    let now = ctx.input(|i| i.time);
    let mut copy: Option<(&'static str, String)> = None;

    ui.label(RichText::new("Let Claude Code, Antigravity or any MCP client build models, run solves and inspect results live.").color(p.weak));
    ui.add_space(8.0);
    egui::Frame::new()
        .fill(if ok { p.ok.gamma_multiply(0.12) } else { p.warn.gamma_multiply(0.12) })
        .corner_radius(8)
        .inner_margin(egui::Margin::symmetric(12, 8))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.horizontal(|ui| {
                ui.label(RichText::new(if ok { "●" } else { "○" }).color(if ok { p.ok } else { p.warn }));
                ui.label(RichText::new(if ok { "Ready — agents connected to SimCraft drive this window live" } else { "Bridge offline — another SimCraft window may already be using the port" }).strong());
            });
            ui.label(RichText::new(&app.bridge).small().color(p.weak));
        });
    ui.add_space(10.0);
    let mut block = |ui: &mut egui::Ui, step: &str, title: &str, hint: &str, text: &str, key: &'static str| {
        ui.horizontal(|ui| {
            ui.label(RichText::new(step).strong().color(p.accent));
            ui.label(RichText::new(title).strong());
        });
        ui.label(RichText::new(hint).small().color(p.weak));
        ui.add_space(4.0);
        egui::Frame::new()
            .fill(if p.dark { Color32::from_rgb(22, 24, 28) } else { Color32::from_rgb(244, 245, 248) })
            .stroke(Stroke::new(1.0, p.border))
            .corner_radius(8)
            .inner_margin(egui::Margin::symmetric(10, 8))
            .show(ui, |ui| {
                ui.set_width(ui.available_width());
                ui.horizontal_top(|ui| {
                    let copied = app.copied.is_some_and(|(k, t)| k == key && now - t < 1.8);
                    let label = if copied { "✔ Copied" } else { "📋 Copy" };
                    let btn = ui.with_layout(egui::Layout::right_to_left(egui::Align::Min), |ui| {
                        let b = ui.add(egui::Button::new(RichText::new(label).small()).min_size(vec2(78.0, 24.0)));
                        ui.with_layout(egui::Layout::left_to_right(egui::Align::Min), |ui| {
                            egui::ScrollArea::horizontal().id_salt(key).show(ui, |ui| {
                                ui.add(egui::Label::new(RichText::new(text).monospace().size(12.0)).extend());
                            });
                        });
                        b
                    });
                    if btn.inner.clicked() {
                        copy = Some((key, text.to_string()));
                    }
                });
            });
        ui.add_space(10.0);
    };
    block(ui, "1", "Claude Code", "Run once in a terminal; SimCraft is then available in every Claude Code session.", &cmd, "claude");
    block(ui, "2", "Antigravity, Cursor and other MCP clients", "Paste into the client's MCP config (Antigravity: Agent panel ▸ ⋯ ▸ MCP Servers ▸ Manage ▸ View raw config).", &js, "json");
    ui.label(RichText::new("Try asking: “In SimCraft, make an aluminium bracket with a hole, fix one end, load the other and show me the stress.”").small().italics().color(p.weak));

    if let Some((k, t)) = copy {
        ctx.copy_text(t);
        app.copied = Some((k, now));
        ctx.request_repaint_after(std::time::Duration::from_millis(1900));
    }
}

impl SimApp {
    pub(super) fn mcp_dialog(&mut self, ctx: &egui::Context) {
        if !self.show_mcp {
            return;
        }
        let p = self.p;
        let mut open = true;
        egui::Window::new(RichText::new("Connect an AI agent").strong())
            .open(&mut open)
            .collapsible(false)
            .resizable(false)
            .default_width(560.0)
            .anchor(Align2::CENTER_CENTER, vec2(0.0, 0.0))
            .frame(egui::Frame::window(&ctx.global_style()).fill(p.card).inner_margin(18))
            .show(ctx, |ui| {
                ui.set_width(560.0);
                mcp_connect_panel(self, ctx, ui);
            });
        if !open {
            self.show_mcp = false;
        }
    }

    pub(super) fn fatigue_window(&mut self, ctx: &egui::Context) {
        if !self.f.fat_open {
            return;
        }
        let kind = self.core.session.lock().unwrap_or_else(std::sync::PoisonError::into_inner).results.as_ref().map(|r| r.analysis);
        let mut open = true;
        let mut go = false;
        egui::Window::new("Fatigue (stress-life)").open(&mut open).resizable(false).collapsible(false).show(ctx, |ui| {
            let f = &mut self.f;
            match kind {
                Some(AnalysisType::RandomVibration) => {
                    ui.label("Random vibration: Steinberg 3-band method (1σ / 2σ / 3σ).");
                    ui.horizontal(|ui| {
                        ui.label("Exposure time");
                        ui.add(egui::DragValue::new(&mut f.fat_exposure_h).speed(0.1).range(0.001..=1e6).suffix(" h"));
                    });
                }
                Some(AnalysisType::HarmonicResponse | AnalysisType::ResponseSpectrum) => {
                    ui.label("Stresses are amplitudes: treated as fully reversed cycles.");
                    ui.horizontal(|ui| {
                        ui.label("Design life");
                        let sp = f.fat_life * 0.01;
                        ui.add(egui::DragValue::new(&mut f.fat_life).speed(sp).range(1.0..=1e12).suffix(" cycles"));
                    });
                }
                _ => {
                    ui.label("The solved load is applied cyclically.");
                    ui.horizontal(|ui| {
                        ui.label("Loading");
                        for (k, l) in [("fully_reversed", "Fully reversed"), ("zero_based", "Zero-based"), ("ratio", "Ratio R")] {
                            ui.selectable_value(&mut f.fat_loading, k.to_string(), l);
                        }
                    });
                    if f.fat_loading == "ratio" {
                        ui.horizontal(|ui| {
                            ui.label("R = σmin/σmax");
                            ui.add(egui::DragValue::new(&mut f.fat_ratio).speed(0.01).range(-5.0..=0.99));
                        });
                    }
                    ui.horizontal(|ui| {
                        ui.label("Mean stress");
                        for k in ["goodman", "gerber", "soderberg", "none"] {
                            let l = format!("{}{}", k[..1].to_uppercase(), &k[1..]);
                            ui.selectable_value(&mut f.fat_correction, k.to_string(), l);
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label("Design life");
                        let sp = f.fat_life * 0.01;
                        ui.add(egui::DragValue::new(&mut f.fat_life).speed(sp).range(1.0..=1e12).suffix(" cycles"));
                    });
                }
            }
            ui.horizontal(|ui| {
                ui.label("Load scale");
                ui.add(egui::DragValue::new(&mut f.fat_scale).speed(0.01).range(0.001..=1000.0).prefix("× "));
            });
            ui.label(RichText::new("S-N data comes from the material (σf', b, endurance limit); missing values are estimated from the ultimate strength.").small().weak());
            ui.add_space(4.0);
            go = ui.button("Evaluate fatigue").clicked();
        });
        if go {
            let f = &self.f;
            let ok = self.run(
                "fatigue_life",
                json!({"loading": f.fat_loading, "ratio": f.fat_ratio, "correction": f.fat_correction, "scale": f.fat_scale, "design_life": f.fat_life, "exposure_time_s": f.fat_exposure_h * 3600.0}),
            );
            if ok {
                self.toast = Some(("Fatigue evaluated — showing the fatigue safety factor".into(), false, 3.5));
                open = false;
            }
        }
        self.f.fat_open = open && self.f.fat_open;
    }

    pub(super) fn material_window(&mut self, ctx: &egui::Context) {
        let Some((base, mut v)) = self.mat_edit.clone() else { return };
        let p = self.p;
        let mut open = true;
        let mut save = false;
        egui::Window::new(RichText::new("✏ Material").strong()).open(&mut open).resizable(false).collapsible(false).show(ctx, |ui| {
            ui.label(RichText::new(format!("Based on {base}. Units: E in GPa, strengths in MPa, density kg/m³, k W/m·K, α 1/K, c J/kg·K, σ S/m.")).small().color(p.weak));
            ui.add_space(4.0);
            egui::ScrollArea::vertical().max_height(420.0).show(ui, |ui| {
                json_editor(ui, &mut v, &[], "mated");
            });
            ui.add_space(6.0);
            save = ui.add(primary(&p, "Save material")).clicked();
        });
        if save {
            let mut a = v.clone();
            a["based_on"] = json!(base);
            let name = v["name"].as_str().unwrap_or("").to_string();
            if self.run("define_material", a) {
                self.f.material = name;
                self.mat_edit = None;
                return;
            }
        }
        self.mat_edit = if open { Some((base, v)) } else { None };
    }

    pub(super) fn safety_ui(&mut self, ctx: &egui::Context) {
        if let Some((path, assets)) = self.confirm_external.clone() {
            let p = self.p;
            let mut choice = None;
            egui::Modal::new(egui::Id::NULL.with("open-external")).show(ctx, |ui| {
                ui.set_width(440.0);
                ui.label(RichText::new("Project references external files").strong().size(15.0));
                ui.label(RichText::new("This project loads STL or KiCad files outside its folder:").color(p.weak));
                ui.add_space(6.0);
                for a in assets.iter().take(6) {
                    ui.label(RichText::new(format!("• {a}")).small().monospace());
                }
                if assets.len() > 6 {
                    ui.label(RichText::new(format!("… and {} more", assets.len() - 6)).small().color(p.weak));
                }
                ui.add_space(10.0);
                ui.horizontal(|ui| {
                    if ui.add(primary(&p, "Open anyway")).clicked() {
                        choice = Some(true);
                    }
                    if ui.button("Cancel").clicked() {
                        choice = Some(false);
                    }
                });
            });
            match choice {
                Some(true) => {
                    self.confirm_external = None;
                    self.run("open_project", json!({"path": path}));
                    self.need_fit = true;
                }
                Some(false) => self.confirm_external = None,
                None => {}
            }
        }
        if ctx.input(|i| i.viewport().close_requested()) && !self.allow_close {
            if self.dirty() {
                ctx.send_viewport_cmd(egui::ViewportCommand::CancelClose);
                self.confirm = Some(Pending::Quit);
            } else {
                let _ = std::fs::remove_file(crate::session::autosave_path());
            }
        }
        let now = ctx.input(|i| i.time);
        if now - self.last_autosave > 120.0 {
            self.last_autosave = now;
            let s = self.core.session.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            if s.is_dirty() && !s.model.bodies.is_empty() {
                let _ = s.autosave();
            }
        }
        ctx.request_repaint_after(std::time::Duration::from_secs(30));
        if !self.recovery_offered {
            self.recovery_offered = true;
            if crate::session::autosave_path().exists() {
                self.toast = Some(("SimCraft found unsaved work from last time — File ▸ Recover autosave".into(), false, 10.0));
            }
        }
        let Some(pending) = self.confirm.clone() else { return };
        let p = self.p;
        let name = self.core.session.lock().unwrap_or_else(std::sync::PoisonError::into_inner).project_path.clone().and_then(|p| std::path::Path::new(&p).file_name().map(|n| n.to_string_lossy().to_string())).unwrap_or_else(|| "this model".into());
        let mut choice = None;
        egui::Modal::new(egui::Id::NULL.with("save-changes")).show(ctx, |ui| {
            ui.set_width(380.0);
            ui.label(RichText::new(format!("Save changes to {name}?")).strong().size(15.0));
            ui.label(RichText::new("Your changes will be lost if you don't save them.").color(p.weak));
            ui.add_space(10.0);
            ui.horizontal(|ui| {
                if ui.add(primary(&p, "Save")).clicked() {
                    choice = Some(0);
                }
                if ui.button("Don't save").clicked() {
                    choice = Some(1);
                }
                if ui.button("Cancel").clicked() {
                    choice = Some(2);
                }
            });
        });
        match choice {
            Some(0) => {
                let has = self.core.session.lock().unwrap_or_else(std::sync::PoisonError::into_inner).project_path.is_some();
                if has {
                    self.run("save_project", json!({}));
                } else {
                    self.save_as();
                }
                self.confirm = None;
                if !self.dirty() {
                    self.perform(pending);
                }
            }
            Some(1) => {
                self.confirm = None;
                self.perform(pending);
            }
            Some(2) => self.confirm = None,
            _ => {}
        }
    }
}
