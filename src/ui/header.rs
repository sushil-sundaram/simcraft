//! Top header bar, File/Edit/Examples menus, theme switcher, and Workbench toggle.

use super::theme::{UnitSystem, logo};
use super::{DrawerTab, Pending, SimApp};
use eframe::egui::{self, RichText, Sense};
use serde_json::json;

impl SimApp {
    pub(super) fn header(&mut self, ui: &mut egui::Ui) {
        let p = self.p;
        // Right-hand items (theme, MCP status, project name) are laid out first; the menus get the
        // remaining width and scroll sideways in narrow windows instead of drawing over them.
        ui.horizontal(|ui| {
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let icon = if self.dark { "☀" } else { "🌙" };
                let tip = if self.dark { "Switch to light mode" } else { "Switch to dark mode" };
                if ui.add(egui::Button::new(RichText::new(icon).size(15.0)).frame_when_inactive(false)).on_hover_text(tip).clicked() {
                    let d = !self.dark;
                    self.run("set_display", json!({"dark_mode": d}));
                }
                let ok = self.bridge.starts_with("MCP bridge listening");
                let (c, t) = if ok { (p.ok, "● MCP ready") } else { (p.warn, "○ MCP bridge off") };
                if ui.add(egui::Label::new(RichText::new(t).size(11.5).color(c)).sense(Sense::click())).on_hover_text("Connect an AI agent").clicked() {
                    self.show_mcp = true;
                }
                let name = self.core.session.lock().unwrap_or_else(std::sync::PoisonError::into_inner).project_path.clone();
                if let Some(n) = name {
                    let f = std::path::Path::new(&n).file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or(n);
                    ui.label(RichText::new(f).size(12.0).color(p.weak));
                }
                ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                    egui::ScrollArea::horizontal().id_salt("header-left").scroll_bar_visibility(egui::scroll_area::ScrollBarVisibility::AlwaysHidden).show(ui, |ui| {
                        ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Extend);
                        ui.horizontal(|ui| {
                        logo(ui, 24.0);
                        ui.label(RichText::new("SimCraft").size(15.0).strong().color(p.text));
                        if self.dirty() {
                            ui.label(RichText::new("● unsaved").small().color(p.warn)).on_hover_text("The model has changes that are not saved (autosaved every 2 minutes)");
                        }
                        ui.add_space(10.0);
                        egui::MenuBar::new().ui(ui, |ui| {
                            ui.menu_button("File", |ui| self.file_menu(ui));
                            ui.menu_button("Edit", |ui| {
                                let (nu, nr) = {
                                    let s = self.core.session.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                                    (s.undo.len(), s.redo.len())
                                };
                                if ui.add_enabled(nu > 0, egui::Button::new("Undo").shortcut_text("Ctrl+Z")).clicked() {
                                    ui.close();
                                    self.run("undo", json!({}));
                                }
                                if ui.add_enabled(nr > 0, egui::Button::new("Redo").shortcut_text("Ctrl+Shift+Z")).clicked() {
                                    ui.close();
                                    self.run("redo", json!({}));
                                }
                            });
                            ui.menu_button("Examples", |ui| {
                                if ui.button("🔩  Cantilever bracket with hole").clicked() {
                                    ui.close();
                                    self.guard(Pending::Example("bracket"));
                                }
                                if ui.button("🌡  Heat sink (thermal)").clicked() {
                                    ui.close();
                                    self.guard(Pending::Example("heatsink"));
                                }
                                if ui.button("🎵  Tuning fork (modal)").clicked() {
                                    ui.close();
                                    self.guard(Pending::Example("fork"));
                                }
                                if ui.button("🏛  Column buckling").clicked() {
                                    ui.close();
                                    self.guard(Pending::Example("buckling"));
                                }
                                ui.separator();
                                if ui.button("⚡  Parallel-plate capacitor (electrostatic)").clicked() {
                                    ui.close();
                                    self.guard(Pending::Example("capacitor"));
                                }
                                if ui.button("🔌  Bus bar Joule heating (thermal-electric)").clicked() {
                                    ui.close();
                                    self.guard(Pending::Example("busbar"));
                                }
                                if ui.button("🌀  Coil with iron core (magnetostatic)").clicked() {
                                    ui.close();
                                    self.guard(Pending::Example("coil"));
                                }
                                if ui.button("📡  Microwave oven heating (full-wave EM)").clicked() {
                                    ui.close();
                                    self.guard(Pending::Example("microwave"));
                                }
                                ui.separator();
                                if ui.button("🔥  Heat sink in a duct (conjugate heat transfer)").clicked() {
                                    ui.close();
                                    self.guard(Pending::Example("heatsink_cht"));
                                }
                                if ui.button("💨  Flow past a cylinder (CFD)").clicked() {
                                    ui.close();
                                    self.guard(Pending::Example("cfd"));
                                }
                            });
                            if ui.button("Connect AI").on_hover_text("Control SimCraft from Claude Code, Antigravity or any MCP client").clicked() {
                                self.show_mcp = true;
                            }
                            if ui.button("✨ Copilot").on_hover_text("Unified AI simulation copilot, quick-fixes, tool runner, and MCP setup").clicked() {
                                self.open_drawer(DrawerTab::Planner);
                            }
                            if ui.selectable_label(self.drawer.open, "🗂 Workbench").on_hover_text("Toggle docked bottom drawer (Charts, Studies, PCB DC, Copilot, Guide)").clicked() {
                                self.drawer.open = !self.drawer.open;
                            }
                            ui.separator();
                            if ui.selectable_label(self.browser_open, "📦").on_hover_text("Show / hide left Browser panel").clicked() {
                                self.browser_open = !self.browser_open;
                            }
                            if ui.selectable_label(self.props_open, "⚙").on_hover_text("Show / hide right Properties panel").clicked() {
                                self.props_open = !self.props_open;
                            }
                            egui::ComboBox::from_id_salt("hdr-units").width(96.0).selected_text(self.units.label()).show_ui(ui, |ui| {
                                for u in UnitSystem::ALL {
                                    if ui.selectable_label(self.units == u, u.label()).clicked() {
                                        self.units = u;
                                        self.persist_prefs();
                                    }
                                }
                            }).response.on_hover_text("Display & input unit system preference (model solves cleanly in SI)");
                            if ui.selectable_label(self.trackpad_pan, "🖱 Pan").on_hover_text("Trackpad mode: two-finger scroll pans; pinch or Ctrl+scroll zooms (also Space+drag anywhere)").clicked() {
                                self.trackpad_pan = !self.trackpad_pan;
                                self.persist_prefs();
                            }
                        });
                        });
                    });
                });
            });
        });
    }

    fn file_menu(&mut self, ui: &mut egui::Ui) {
        if ui.button("New").clicked() {
            ui.close();
            self.guard(Pending::New);
        }
        if ui.button("Open project…").clicked() {
            ui.close();
            self.guard(Pending::Open);
        }
        let recent = crate::session::recent_files();
        ui.add_enabled_ui(!recent.is_empty(), |ui| {
            ui.menu_button("Open recent", |ui| {
                for path in recent {
                    let name = std::path::Path::new(&path).file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_else(|| path.clone());
                    if ui.button(name).on_hover_text(&path).clicked() {
                        ui.close();
                        self.guard(Pending::OpenPath(path));
                    }
                }
            });
        });
        if crate::session::autosave_path().exists() && ui.button("Recover autosave").on_hover_text("Open the model SimCraft saved automatically (e.g. before a crash)").clicked() {
            ui.close();
            self.guard(Pending::OpenPath(crate::session::autosave_path().to_string_lossy().to_string()));
        }
        if ui.button("Save").clicked() {
            ui.close();
            let has = self.core.session.lock().unwrap_or_else(std::sync::PoisonError::into_inner).project_path.is_some();
            if has {
                self.run("save_project", json!({}));
            } else {
                self.save_as();
            }
        }
        if ui.button("Save as…").clicked() {
            ui.close();
            self.save_as();
        }
        ui.separator();
        if ui.button("Import STEP…").clicked() {
            ui.close();
            self.import_step();
        }
        if ui.button("Import STL…").clicked() {
            ui.close();
            self.import_stl();
        }
        if ui.button("Import KiCad board…").clicked() {
            ui.close();
            self.import_kicad();
        }
        if ui.button("Export results (VTK)…").clicked() {
            ui.close();
            if let Some(p) = rfd::FileDialog::new().add_filter("VTK", &["vtk"]).set_file_name("results.vtk").save_file() {
                self.run("export_vtk", json!({"path": p.to_string_lossy()}));
            }
        }
        if ui.button("Export report (HTML)…").clicked() {
            ui.close();
            if let Some(p) = rfd::FileDialog::new().add_filter("HTML report", &["html"]).set_file_name("simcraft-report.html").save_file() {
                self.run("make_report", json!({"path": p.to_string_lossy()}));
            }
        }
        if ui.button("Save image (PNG)…").clicked() {
            ui.close();
            if let Some(p) = rfd::FileDialog::new().add_filter("PNG", &["png"]).set_file_name("simcraft.png").save_file() {
                match self.core.call("render_image", &json!({"width": 1400, "height": 900})) {
                    Ok(o) => {
                        if let Some(png) = o.image_png {
                            let _ = std::fs::write(&p, png);
                            self.toast = Some((format!("Saved {}", p.display()), false, 4.0));
                        }
                    }
                    Err(e) => self.toast = Some((format!("{e:#}"), true, 6.0)),
                }
            }
        }
        ui.separator();
        if ui.button("Quit").clicked() {
            ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
        }
    }
}
