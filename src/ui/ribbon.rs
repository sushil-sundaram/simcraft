//! Top ribbon tabs (Geometry, Material, Setup, Mesh, Solve, Results, Optimize, Help) and opt forms.

use super::theme::{MAT_KEYS, card, group, primary, tool};
use super::{DrawerTab, Form, LoadKind, SimApp, Tab};
use crate::model::AnalysisType;
use crate::render;
use crate::results::Field;
use eframe::egui::{self, Color32, RichText, Sense, Stroke, pos2, vec2};
use serde_json::{Value, json};

impl SimApp {
    pub(super) fn ribbon(&mut self, ui: &mut egui::Ui) {
        let p = self.p;
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 18.0;
            for (t, name) in [(Tab::Geometry, "GEOMETRY"), (Tab::Material, "MATERIAL"), (Tab::Setup, "SETUP"), (Tab::Mesh, "MESH"), (Tab::Solve, "SOLVE"), (Tab::Results, "RESULTS"), (Tab::Optimize, "OPTIMIZE"), (Tab::Help, "HELP")] {
                let on = self.tab == t;
                let r = ui.add(egui::Label::new(RichText::new(name).size(12.0).strong().color(if on { p.accent } else { p.weak })).sense(Sense::click()));
                if on {
                    let y = r.rect.bottom() + 3.0;
                    ui.painter().line_segment([pos2(r.rect.left(), y), pos2(r.rect.right(), y)], Stroke::new(2.5, p.accent));
                }
                if r.on_hover_cursor(egui::CursorIcon::PointingHand).clicked() {
                    self.tab = t;
                }
            }
        });
        ui.add_space(6.0);
        let (kind, has_enc, mut show_mesh, mut persp, mut glass) = {
            let s = self.core.session.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            (s.model.analysis.kind, s.model.has_enclosure(), s.display.show_mesh, s.display.perspective, s.display.translucent_enclosure)
        };
        ui.horizontal(|ui| match self.tab {
            Tab::Geometry => {
                group(ui, &p, "CREATE", |ui| {
                    if tool(ui, &p, "⬛", "Box", None, true) {
                        self.f.subtract = false;
                        self.form = Form::Box;
                    }
                    if tool(ui, &p, "🛢", "Cylinder", None, true) {
                        self.f.subtract = false;
                        self.form = Form::Cylinder;
                    }
                    if tool(ui, &p, "⚫", "Sphere", None, true) {
                        self.f.subtract = false;
                        self.form = Form::Sphere;
                    }
                });
                group(ui, &p, "MODIFY", |ui| {
                    if tool(ui, &p, "⭕", "Hole", Some(p.weak), true) {
                        self.f.subtract = true;
                        self.form = Form::Cylinder;
                    }
                    if tool(ui, &p, "⬜", "Pocket", Some(p.weak), true) {
                        self.f.subtract = true;
                        self.form = Form::Box;
                    }
                });
                group(ui, &p, "INSERT", |ui| {
                    if tool(ui, &p, "📂", "Import STL", Some(p.weak), true) {
                        self.import_stl();
                    }
                    if tool(ui, &p, "🟩", "KiCad PCB", Some(Color32::from_rgb(70, 160, 90)), true) {
                        self.import_kicad();
                    }
                    if tool(ui, &p, "⬚", "Enclosure", Some(Color32::from_rgb(90, 170, 230)), true) {
                        self.form = Form::Enclosure;
                    }
                });
            }
            Tab::Material => {
                let lib = self.core.session.lock().unwrap_or_else(std::sync::PoisonError::into_inner).model.materials();
                group(ui, &p, "ASSIGN", |ui| {
                    ui.vertical(|ui| {
                        ui.add_space(6.0);
                        egui::ComboBox::from_id_salt("matall").width(230.0).selected_text(&self.f.material).show_ui(ui, |ui| {
                            for m in &lib {
                                ui.selectable_value(&mut self.f.material, m.name.clone(), &m.name);
                            }
                        });
                        if ui.add(primary(&p, "Assign to all bodies")).clicked() {
                            self.run("set_material", json!({"material": self.f.material}));
                        }
                        if ui.button("✏ New / edit material…").on_hover_text("Create a custom material from the selected one, or change a custom material").clicked() {
                            if let Some(m) = lib.iter().find(|m| m.name == self.f.material) {
                                let full = serde_json::to_value(m).unwrap_or(json!({}));
                                let mut v = json!({"name": if crate::material::library().iter().any(|x| x.name == m.name) { format!("{} (custom)", m.name) } else { m.name.clone() }});
                                for k in MAT_KEYS {
                                    if let Some(x) = full.get(*k) {
                                        v[*k] = x.clone();
                                    } else if k.ends_with("_table") {
                                        v[*k] = json!([]);
                                    }
                                }
                                self.mat_edit = Some((m.name.clone(), v));
                            }
                        }
                    });
                });
                if let Some(m) = lib.iter().find(|m| m.name == self.f.material) {
                    group(ui, &p, "PROPERTIES", |ui| {
                        egui::Grid::new("matprops").num_columns(4).spacing([14.0, 2.0]).show(ui, |ui| {
                            let kv = |ui: &mut egui::Ui, k: &str, v: String| {
                                ui.label(RichText::new(k).small().color(p.weak));
                                ui.label(RichText::new(v).small());
                            };
                            kv(ui, "Young's modulus", format!("{} GPa", m.youngs_modulus_gpa));
                            kv(ui, "Yield", format!("{} MPa", m.yield_strength_mpa));
                            ui.end_row();
                            kv(ui, "Poisson's ratio", format!("{}", m.poisson_ratio));
                            kv(ui, "Ultimate", format!("{} MPa", m.ultimate_strength_mpa));
                            ui.end_row();
                            kv(ui, "Density", format!("{} kg/m³", m.density));
                            kv(ui, "Conductivity", format!("{} W/m·K", m.thermal_conductivity));
                            ui.end_row();
                        });
                    });
                }
            }
            Tab::Setup => {
                group(ui, &p, "ANALYSIS", |ui| {
                    ui.vertical(|ui| {
                        ui.add_space(12.0);
                        let mut k = kind;
                        egui::ComboBox::from_id_salt("analysis").width(230.0).selected_text(k.label()).height(520.0).show_ui(ui, |ui| {
                            let mut last = "";
                            for a in AnalysisType::all() {
                                if a.group() != last {
                                    last = a.group();
                                    ui.label(RichText::new(last.to_uppercase()).size(10.0).strong().color(p.weak));
                                }
                                ui.selectable_value(&mut k, a, a.label());
                            }
                        });
                        if k != kind {
                            self.run("set_analysis", json!({"type": k.key()}));
                        }
                    });
                });
                for (name, kinds) in [("STRUCTURAL", &LoadKind::STRUCTURAL[..]), ("THERMAL", &LoadKind::THERMAL[..]), ("ELECTROMAGNETIC", &LoadKind::ELECTRO[..]), ("FLUID", &LoadKind::FLUID[..])] {
                    let usable: Vec<LoadKind> = kinds.iter().copied().filter(|k| k.applies(kind)).collect();
                    let cfd = matches!(kind, AnalysisType::Fluid | AnalysisType::ConjugateHeatTransfer);
                    if usable.is_empty() || name == "FLUID" && !cfd || name == "STRUCTURAL" && cfd {
                        continue;
                    }
                    group(ui, &p, name, |ui| {
                        for lk in usable {
                            if tool(ui, &p, lk.icon(), lk.label(), Some(lk.color()), true) {
                                self.form = Form::Load(lk);
                            }
                        }
                    });
                }
                if kind.requires_enclosure() && !has_enc {
                    ui.vertical(|ui| {
                        ui.add_space(16.0);
                        ui.label(RichText::new("⚠ This analysis needs an enclosure").color(p.warn));
                        if ui.button("⬚  Add enclosure").clicked() {
                            self.form = Form::Enclosure;
                        }
                    });
                }
            }
            Tab::Mesh => {
                group(ui, &p, "ELEMENT SIZE", |ui| {
                    ui.vertical(|ui| {
                        ui.add_space(8.0);
                        ui.add(egui::DragValue::new(&mut self.f.mesh_size).speed(0.1).range(0.05..=1000.0).suffix(" mm"));
                        ui.label(RichText::new("smaller = more accurate").small().color(p.weak));
                    });
                    if tool(ui, &p, "▦", "Generate", None, true) {
                        self.run("set_mesh", json!({"element_size_mm": self.f.mesh_size}));
                    }
                    if tool(ui, &p, "✨", "Auto", None, true) {
                        self.run("set_mesh", json!({}));
                        self.f.mesh_size = self.core.session.lock().unwrap_or_else(std::sync::PoisonError::into_inner).element_size();
                    }
                });
                group(ui, &p, "DISPLAY", |ui| {
                    ui.vertical(|ui| {
                        ui.add_space(8.0);
                        if ui.checkbox(&mut show_mesh, "Element edges").changed() {
                            self.run("set_display", json!({"show_mesh": show_mesh}));
                        }
                        if ui.checkbox(&mut persp, "Perspective camera").changed() {
                            self.run("set_display", json!({"perspective": persp}));
                        }
                        if has_enc && ui.checkbox(&mut glass, "Translucent enclosure").changed() {
                            self.run("set_display", json!({"translucent_enclosure": glass}));
                        }
                    });
                });
            }
            Tab::Solve => {
                let busy = self.core.is_busy();
                group(ui, &p, "RUN", |ui| {
                    ui.vertical(|ui| {
                        ui.add_space(8.0);
                        if busy {
                            let b = egui::Button::new(RichText::new("■  Stop").size(15.0).strong().color(Color32::WHITE)).fill(p.err).min_size(vec2(120.0, 40.0));
                            if ui.add(b).on_hover_text("Stop the running solve or study").clicked() {
                                let _ = self.core.call("cancel", &json!({}));
                            }
                        } else {
                            let b = egui::Button::new(RichText::new("⚡  Solve").size(15.0).strong().color(Color32::WHITE)).fill(p.accent).min_size(vec2(120.0, 40.0));
                            if ui.add(b).clicked() {
                                self.solve();
                            }
                        }
                    });
                });
                let board = self.core.session.lock().unwrap_or_else(std::sync::PoisonError::into_inner).pcb_path();
                if let Some(path) = board {
                    group(ui, &p, "PCB", |ui| {
                        let on = self.pcb.open;
                        if tool(ui, &p, "🔌", "DC drop", Some(if on { p.accent } else { Color32::from_rgb(70, 160, 90) }), true) {
                            self.open_pcb_dc(ui.ctx(), &path);
                        }
                    });
                }
                group(ui, &p, "SETTINGS", |ui| {
                    ui.vertical(|ui| {
                        ui.add_space(4.0);
                        ui.horizontal(|ui| {
                            ui.label(RichText::new(kind.label()).strong());
                            if ui.small_button("ⓘ").on_hover_text("How this analysis works (F1)").clicked() {
                                self.open_help(crate::help::analysis_topic(&kind.key()));
                            }
                        });
                        let st = self.core.session.lock().unwrap_or_else(std::sync::PoisonError::into_inner).model.analysis.clone();
                        let num = |ui: &mut egui::Ui, label: &str, v: f64, key: &str, speed: f64, suffix: &str, run: &mut Vec<(String, f64)>| {
                            let mut x = v;
                            ui.horizontal(|ui| {
                                ui.label(label);
                                if ui.add(egui::DragValue::new(&mut x).speed(speed).suffix(suffix.to_string())).changed() {
                                    run.push((key.to_string(), x));
                                }
                            });
                        };
                        let mut changes: Vec<(String, f64)> = Vec::new();
                        if kind.transient() {
                            num(ui, "End time", st.end_time, "end_time", st.end_time * 0.02, " s", &mut changes);
                            num(ui, "Time steps", st.time_steps as f64, "time_steps", 1.0, "", &mut changes);
                        }
                        if matches!(kind, AnalysisType::FullWaveEm | AnalysisType::MicrowaveHeating) {
                            num(ui, "Frequency", st.frequency_hz / 1e9, "frequency_ghz", 0.01, " GHz", &mut changes);
                            num(ui, "Max RF periods", st.max_periods as f64, "max_periods", 10.0, "", &mut changes);
                        }
                        if matches!(kind, AnalysisType::TransientThermal | AnalysisType::MicrowaveHeating) {
                            num(ui, "Initial temp.", st.initial_temperature, "initial_temperature", 0.5, " °C", &mut changes);
                        }
                        if matches!(kind, AnalysisType::TransientStructural | AnalysisType::HarmonicResponse | AnalysisType::ResponseSpectrum | AnalysisType::RandomVibration) {
                            num(ui, "Damping ratio", st.damping_ratio, "damping_ratio", 0.005, "", &mut changes);
                        }
                        if kind == AnalysisType::NonlinearStructural {
                            num(ui, "Load steps", st.load_steps as f64, "load_steps", 1.0, "", &mut changes);
                        }
                        if matches!(kind, AnalysisType::Fluid | AnalysisType::ConjugateHeatTransfer) {
                            ui.horizontal(|ui| {
                                ui.label("Turbulence");
                                for (t, l, tip) in [("laminar", "Laminar", "Low speeds, small gaps (Re below ~2000)"), ("lvel", "LVEL", "Algebraic turbulence model for electronics enclosures, fans and ducts")] {
                                    if ui.selectable_label(st.turbulence == t, l).on_hover_text(tip).clicked() && st.turbulence != t {
                                        self.run("set_analysis", json!({"turbulence": t}));
                                    }
                                }
                            });
                        }
                        if matches!(kind, AnalysisType::Fluid | AnalysisType::ConjugateHeatTransfer) {
                            ui.horizontal(|ui| {
                                ui.label("Scheme");
                                for (t, l, tip) in [
                                    ("auto", "Auto", "Implicit SIMPLE (fast, no time-step limit); switches to Projection if the flow will not settle"),
                                    ("simple", "SIMPLE", "Implicit SIMPLE only"),
                                    ("projection", "Projection", "Explicit pseudo-time marching: follows unsteady flows and time-averages them"),
                                ] {
                                    if ui.selectable_label(st.cfd_scheme == t, l).on_hover_text(tip).clicked() && st.cfd_scheme != t {
                                        self.run("set_analysis", json!({"cfd_scheme": t}));
                                    }
                                }
                            });
                            num(ui, "Max iterations", st.max_iterations as f64, "max_iterations", 50.0, "", &mut changes);
                        }
                        if kind == AnalysisType::LinearBuckling {
                            num(ui, "Buckling modes", st.modes as f64, "modes", 1.0, "", &mut changes);
                        }
                        if kind == AnalysisType::HarmonicResponse {
                            num(ui, "From (0 = auto)", st.freq_min.unwrap_or(0.0), "freq_min", 1.0, " Hz", &mut changes);
                            num(ui, "To (0 = auto)", st.freq_max.unwrap_or(0.0), "freq_max", 10.0, " Hz", &mut changes);
                            num(ui, "Frequency points", st.freq_steps as f64, "freq_steps", 5.0, "", &mut changes);
                        }
                        for (k, v) in changes {
                            self.run("set_analysis", json!({ k: v }));
                        }
                        ui.horizontal(|ui| {
                            ui.label("Compute");
                            let cur = st.compute;
                            for (m, label, tip) in [
                                (crate::gpu::ComputeMode::Auto, "Auto", "CPU direct solver for small/medium models, GPU for large iterative solves"),
                                (crate::gpu::ComputeMode::Cpu, "CPU", "Never use the GPU"),
                                (crate::gpu::ComputeMode::Gpu, "GPU", "Run every linear solve on the GPU (falls back to CPU if unavailable)"),
                            ] {
                                if ui.selectable_label(cur == m, label).on_hover_text(tip).clicked() && cur != m {
                                    let key = serde_json::to_value(m).unwrap();
                                    self.run("set_analysis", json!({"compute": key}));
                                }
                            }
                        });
                        if kind == AnalysisType::Modal {
                            let mut m = self.core.session.lock().unwrap_or_else(std::sync::PoisonError::into_inner).model.analysis.modes as u32;
                            ui.horizontal(|ui| {
                                ui.label("Modes");
                                if ui.add(egui::DragValue::new(&mut m).range(1..=20)).changed() {
                                    self.run("set_analysis", json!({"modes": m}));
                                }
                            });
                        }
                        if matches!(kind, AnalysisType::FullWaveEm | AnalysisType::MicrowaveHeating) {
                            ui.horizontal(|ui| {
                                ui.label("Outer boundary");
                                for (b, l, tip) in [("pec", "Metal cavity", "The enclosure walls are metal (microwave oven, shielded box)"), ("open", "Open", "Absorbing boundary: waves leave the enclosure (antennas, radiated emissions)")] {
                                    if ui.selectable_label(st.em_boundary == b, l).on_hover_text(tip).clicked() && st.em_boundary != b {
                                        self.run("set_analysis", json!({"em_boundary": b}));
                                    }
                                }
                            });
                        }
                        if matches!(kind, AnalysisType::ResponseSpectrum | AnalysisType::RandomVibration) {
                            let dir = serde_json::to_value(st.direction).ok().and_then(|v| v.as_str().map(String::from)).unwrap_or_default();
                            ui.horizontal(|ui| {
                                ui.label("Base direction");
                                for d in ["x", "y", "z"] {
                                    if ui.selectable_label(dir == d, d.to_uppercase()).clicked() && dir != d {
                                        self.run("set_analysis", json!({"direction": d}));
                                    }
                                }
                            });
                            if kind == AnalysisType::ResponseSpectrum {
                                ui.horizontal(|ui| {
                                    ui.label("Combination");
                                    for (c, tip) in [("srss", "Square root of sum of squares (well-separated modes)"), ("cqc", "Complete quadratic combination (closely spaced modes)")] {
                                        if ui.selectable_label(st.combination == c, c.to_uppercase()).on_hover_text(tip).clicked() && st.combination != c {
                                            self.run("set_analysis", json!({"combination": c}));
                                        }
                                    }
                                });
                            }
                        }
                        if matches!(kind, AnalysisType::StaticStructural | AnalysisType::ThermalStress | AnalysisType::SteadyThermal | AnalysisType::TransientThermal | AnalysisType::Modal | AnalysisType::HarmonicResponse | AnalysisType::ResponseSpectrum | AnalysisType::RandomVibration | AnalysisType::TransientStructural) {
                            let mut fit = st.boundary_fit;
                            if ui.checkbox(&mut fit, "Fit curved surfaces").on_hover_text("Moves the surface nodes onto the true holes, fillets and off-grid faces (with a fitted surface layer) for accurate peak stresses, heat flows and frequencies. Off = plain voxels. Models with a KiCad board always use plain voxels.").changed() {
                                self.run("set_analysis", json!({"boundary_fit": fit}));
                            }
                        }
                        if kind == AnalysisType::ThermalStress {
                            let mut t = self.core.session.lock().unwrap_or_else(std::sync::PoisonError::into_inner).model.analysis.reference_temperature;
                            ui.horizontal(|ui| {
                                ui.label("Reference temp.");
                                if ui.add(egui::DragValue::new(&mut t).suffix(" °C")).changed() {
                                    self.run("set_analysis", json!({"reference_temperature": t}));
                                }
                            });
                        }
                    });
                });
                if matches!(kind, AnalysisType::ResponseSpectrum | AnalysisType::RandomVibration) {
                    let (st_table, spectrum) = (self.core.session.lock().unwrap_or_else(std::sync::PoisonError::into_inner).model.analysis.table.clone(), kind == AnalysisType::ResponseSpectrum);
                    if st_table != self.f.table_src {
                        self.f.table_text = st_table.iter().map(|r| format!("{} {}", r[0], r[1])).collect::<Vec<_>>().join("\n");
                        self.f.table_src = st_table;
                    }
                    group(ui, &p, if spectrum { "SPECTRUM (Hz  g)" } else { "PSD (Hz  g²/Hz)" }, |ui| {
                        ui.horizontal(|ui| {
                            egui::ScrollArea::vertical().id_salt("table").max_height(62.0).show(ui, |ui| {
                                ui.add(egui::TextEdit::multiline(&mut self.f.table_text).desired_width(150.0).desired_rows(3).font(egui::TextStyle::Monospace).hint_text("10 0.01\n2000 0.01"));
                            });
                            ui.vertical(|ui| {
                                if ui.button("Apply").on_hover_text("One 'frequency value' pair per line; log-log interpolated").clicked() {
                                    let pts: Result<Vec<[f64; 2]>, String> = self
                                        .f
                                        .table_text
                                        .lines()
                                        .filter(|l| !l.trim().is_empty())
                                        .map(|l| {
                                            let v: Vec<f64> = l.split(|c: char| c == ',' || c.is_whitespace()).filter(|x| !x.is_empty()).map(|x| x.parse::<f64>()).collect::<Result<_, _>>().map_err(|_| format!("bad number in '{l}'"))?;
                                            if v.len() != 2 { Err(format!("'{l}': need frequency and value")) } else { Ok([v[0], v[1]]) }
                                        })
                                        .collect();
                                    match pts {
                                        Ok(pts) => {
                                            self.run("set_analysis", json!({"table": pts}));
                                        }
                                        Err(e) => self.toast = Some((e, true, 6.0)),
                                    }
                                }
                                if ui.button("Example").on_hover_text(if spectrum { "Flat 1 g spectrum" } else { "NAVMAT-style 6 Grms profile" }).clicked() {
                                    let ex = if spectrum { json!([[1, 1.0], [10000, 1.0]]) } else { json!([[20, 0.01], [80, 0.04], [350, 0.04], [2000, 0.007]]) };
                                    self.run("set_analysis", json!({"table": ex}));
                                }
                            });
                        });
                    });
                }
            }
            Tab::Results => self.results_ribbon(ui),
            Tab::Help => {
                let mut open: Option<&str> = None;
                group(ui, &p, "GUIDE", |ui| {
                    if tool(ui, &p, "📖", "User guide", Some(p.accent), true) {
                        open = Some("2");
                    }
                    if tool(ui, &p, "🚀", "Quick start", None, true) {
                        open = Some("3");
                    }
                    if tool(ui, &p, "🔎", "Search", None, true) {
                        open = Some("");
                    }
                    if tool(ui, &p, "⌨", "Shortcuts", None, true) {
                        open = Some("Keyboard and mouse");
                    }
                });
                group(ui, &p, "TUTORIALS", |ui| {
                    for (icon, label, topic) in [("🔩", "Bracket", "Tutorial 1"), ("🌡", "Heat sink", "Tutorial 2"), ("🎵", "Vibration", "Tutorial 3"), ("🟩", "KiCad board", "Tutorial 4"), ("🎯", "Optimize", "Tutorial 5"), ("📡", "Emissions", "Tutorial 6"), ("🤖", "AI agent", "Tutorial 7")] {
                        if tool(ui, &p, icon, label, Some(p.ok), true) {
                            open = Some(topic);
                        }
                    }
                });
                group(ui, &p, "REFERENCE", |ui| {
                    for (icon, label, topic) in [("⚙", "Analyses", "7"), ("📌", "Loads", "8"), ("📊", "Results", "9"), ("📏", "Units", "14"), ("🩺", "Problems", "16"), ("⚠", "Limits", "17")] {
                        if tool(ui, &p, icon, label, None, true) {
                            open = Some(topic);
                        }
                    }
                });
                group(ui, &p, "AI & SCRIPTS", |ui| {
                    if tool(ui, &p, "🤖", "MCP / Python", None, true) {
                        open = Some("13");
                    }
                    if tool(ui, &p, "✨", "Planner", None, true) {
                        open = Some("12");
                    }
                });
                group(ui, &p, "ABOUT", |ui| {
                    ui.vertical(|ui| {
                        ui.add_space(4.0);
                        ui.label(RichText::new(format!("SimCraft {}", env!("CARGO_PKG_VERSION"))).strong());
                        ui.label(RichText::new("MIT licence · F1 opens help anywhere").small().color(p.weak));
                        ui.hyperlink_to(RichText::new("github.com/sushil-sundaram/simcraft").small(), "https://github.com/sushil-sundaram/simcraft");
                    });
                });
                if let Some(t) = open {
                    if t.is_empty() {
                        self.open_drawer(DrawerTab::Help);
                        self.help.focus_search = true;
                    } else {
                        self.open_help(t);
                    }
                }
            }
            Tab::Optimize => {
                group(ui, &p, "STUDIES", |ui| {
                    if tool(ui, &p, "📊", "Parameters", None, true) {
                        self.form = Form::Study;
                    }
                    if tool(ui, &p, "🎯", "Optimize", None, true) {
                        self.form = Form::Optimize;
                    }
                    if tool(ui, &p, "📐", "Convergence", None, true) {
                        self.form = Form::Convergence;
                    }
                });
                group(ui, &p, "SHAPE", |ui| {
                    if tool(ui, &p, "🦴", "Topology", Some(Color32::from_rgb(120, 170, 255)), kind == AnalysisType::StaticStructural) {
                        self.form = Form::Topology;
                    }
                });
                group(ui, &p, "RESULTS", |ui| {
                    let has = self.core.session.lock().unwrap_or_else(std::sync::PoisonError::into_inner).study.is_some();
                    if tool(ui, &p, "📋", "Study results", Some(p.weak), has) {
                        self.open_drawer(DrawerTab::Study);
                    }
                });
            }
        });
    }

    fn results_ribbon(&mut self, ui: &mut egui::Ui) {
        let p = self.p;
        let (res, disp, with_enc) = {
            let s = self.core.session.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            (s.results.clone(), s.display.clone(), s.mesh.as_ref().is_some_and(|m| m.with_enclosure))
        };
        let Some(r) = res else {
            ui.add_space(18.0);
            ui.label(RichText::new("No results yet — open SOLVE and press Solve.").color(p.weak));
            return;
        };
        let groups: [(&str, &[(Field, &str, &str)]); 7] = [
            ("DEFORMATION", &[(Field::TotalDeformation, "📐", "Total"), (Field::DeformationX, "X", "Dir. X"), (Field::DeformationY, "Y", "Dir. Y"), (Field::DeformationZ, "Z", "Dir. Z")]),
            ("STRESS", &[(Field::VonMises, "🔶", "Von Mises"), (Field::MaxPrincipal, "⬆", "Max Princ."), (Field::MinPrincipal, "⬇", "Min Princ."), (Field::RadialStress, "◎", "Radial"), (Field::HoopStress, "⭕", "Hoop"), (Field::AxialStress, "↕", "Axial"), (Field::SafetyFactor, "🛡", "Safety"), (Field::PlasticStrain, "〰", "Plastic ε")]),
            ("THERMAL", &[(Field::Temperature, "🌡", "Temperature"), (Field::HeatFlux, "☀", "Heat Flux")]),
            ("ELECTRIC", &[(Field::ElectricPotential, "⚡", "Potential"), (Field::ElectricField, "↯", "E-Field"), (Field::CurrentDensity, "🔌", "Current"), (Field::JouleHeat, "🔥", "Joule Heat")]),
            ("RF", &[(Field::AbsorbedPower, "♨", "Absorbed"), (Field::Sar, "🩺", "SAR")]),
            ("MAGNETIC", &[(Field::MagneticFluxDensity, "🧲", "B Field"), (Field::MagneticFieldIntensity, "🌀", "H Field")]),
            ("FLUID", &[(Field::Velocity, "💨", "Velocity"), (Field::VelocityX, "X", "Vx"), (Field::VelocityY, "Y", "Vy"), (Field::VelocityZ, "Z", "Vz"), (Field::FluidPressure, "⏲", "Pressure")]),
        ];
        for (g, fields) in groups {
            if !fields.iter().any(|f| r.has(f.0)) {
                continue;
            }
            group(ui, &p, g, |ui| {
                for (f, icon, label) in fields.iter().filter(|f| r.has(f.0)) {
                    let on = disp.field == *f && disp.show_results;
                    let col = if on { p.accent } else { p.weak };
                    if tool(ui, &p, icon, label, Some(col), true) {
                        self.run("set_display", json!({"field": f.key(), "show_results": true}));
                    }
                }
            });
        }
        if r.stress.is_some() {
            group(ui, &p, "FATIGUE", |ui| {
                for (f, icon, label) in [(Field::FatigueSafety, "🛡", "Safety"), (Field::FatigueLife, "⏳", "Life"), (Field::FatigueDamage, "💥", "Damage")] {
                    if r.has(f) {
                        let on = disp.field == f && disp.show_results;
                        if tool(ui, &p, icon, label, Some(if on { p.accent } else { p.weak }), true) {
                            self.run("set_display", json!({"field": f.key(), "show_results": true}));
                        }
                    }
                }
                if tool(ui, &p, "🔁", "Evaluate", None, true) {
                    self.f.fat_open = true;
                }
            });
        }
        if disp.field.cyl_comp().is_some() {
            group(ui, &p, "CYLINDRICAL AXIS", |ui| {
                ui.vertical(|ui| {
                    let mut origin = disp.cylindrical.origin;
                    let dir = disp.cylindrical.direction;
                    ui.horizontal(|ui| {
                        ui.label(RichText::new("Axis").small());
                        for (ax, lbl, dv) in [(0, "X", [1.0, 0.0, 0.0]), (1, "Y", [0.0, 1.0, 0.0]), (2, "Z", [0.0, 0.0, 1.0])] {
                            let on = dir[ax].abs() > 0.9 && dir[(ax + 1) % 3].abs() < 0.2 && dir[(ax + 2) % 3].abs() < 0.2;
                            if ui.selectable_label(on, lbl).clicked() {
                                self.run("set_display", json!({"cylindrical_axis": dv}));
                            }
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label(RichText::new("Origin").small());
                        let mut changed = false;
                        for (i, k) in ["X", "Y", "Z"].iter().enumerate() {
                            changed |= ui.add(egui::DragValue::new(&mut origin[i]).speed(0.5).prefix(format!("{k}:")).max_decimals(2)).changed();
                        }
                        if changed {
                            self.run("set_display", json!({"cylindrical_origin": origin}));
                        }
                    });
                });
            });
        }
        group(ui, &p, "TOOLS", |ui| {
            let charts_open = self.drawer.is_showing(DrawerTab::Charts);
            if tool(ui, &p, "📈", "History", Some(if charts_open && self.path.tab == 0 { p.accent } else { p.weak }), !r.history.is_empty()) {
                if charts_open && self.path.tab == 0 {
                    self.close_drawer_tab(DrawerTab::Charts);
                } else {
                    self.path.tab = 0;
                    self.open_drawer(DrawerTab::Charts);
                }
            }
            if tool(ui, &p, "📏", "Path", Some(if charts_open && self.path.tab == 1 { p.accent } else { p.weak }), true) {
                if charts_open && self.path.tab == 1 {
                    self.close_drawer_tab(DrawerTab::Charts);
                } else {
                    self.path.tab = 1;
                    self.open_drawer(DrawerTab::Charts);
                }
            }
            if tool(ui, &p, "🎯", "Probe", Some(if self.probe_mode { p.accent } else { p.weak }), true) {
                self.probe_mode = !self.probe_mode;
                self.toast = Some((if self.probe_mode { "Probe: click the model to read the value at the nearest node (click again to add more)" } else { "Probe off" }.into(), false, 4.0));
            }
            if !self.probes.is_empty() && tool(ui, &p, "✖", "Clear", None, true) {
                self.probes.clear();
            }
        });
        group(ui, &p, "EXPORT", |ui| {
            if tool(ui, &p, "📄", "Report", None, true) {
                if let Some(path) = rfd::FileDialog::new().add_filter("HTML report", &["html"]).set_file_name("simcraft-report.html").save_file() {
                    self.run("make_report", json!({"path": path.to_string_lossy()}));
                }
            }
        });
        group(ui, &p, "VIEW", |ui| {
            ui.vertical(|ui| {
                let steps = r.n_steps();
                if !r.modes.is_empty() {
                    egui::ComboBox::from_id_salt("mode").width(190.0).selected_text(r.step_label(disp.mode)).show_ui(ui, |ui| {
                        for i in 0..steps {
                            if ui.selectable_label(disp.mode == i, r.step_label(i)).clicked() {
                                self.run("set_display", json!({"mode": i + 1}));
                            }
                        }
                    });
                } else if steps > 1 {
                    let mut i = disp.mode.min(steps - 1);
                    ui.horizontal(|ui| {
                        if ui.add(egui::Slider::new(&mut i, 0..=steps - 1).show_value(false)).changed() {
                            self.run("set_display", json!({"mode": i + 1}));
                        }
                        ui.label(RichText::new(r.step_label(i)).small());
                    });
                }
                if with_enc || disp.section.is_some() {
                    ui.horizontal(|ui| {
                        let mut on = disp.section.is_some();
                        if ui.checkbox(&mut on, "Cut plane").changed() {
                            self.run("set_display", json!({"section": if on { "y" } else { "off" }}));
                        }
                        if let Some(sec) = disp.section {
                            for (ax, name) in ["x", "y", "z"].iter().enumerate() {
                                if ui.selectable_label(sec.axis == ax, name.to_uppercase()).clicked() {
                                    self.run("set_display", json!({"section": name, "section_position": sec.position}));
                                }
                            }
                            let mut pos = sec.position;
                            if ui.add(egui::Slider::new(&mut pos, 0.0..=1.0).show_value(false)).changed() {
                                self.run("set_display", json!({"section_position": pos}));
                            }
                        }
                    });
                }
                if r.deformation(disp.mode).is_some() {
                    let cur = match disp.deformation_scale {
                        None => "Auto scale".to_string(),
                        Some(x) if x == 0.0 => "Undeformed".into(),
                        Some(x) if x == 1.0 => "True scale".into(),
                        Some(x) => format!("{}×", render::fmt_val(x)),
                    };
                    egui::ComboBox::from_id_salt("dscale").width(160.0).selected_text(cur).show_ui(ui, |ui| {
                        for (lbl, v) in [("Undeformed", json!(0.0)), ("True scale", json!(1.0)), ("Auto scale", json!("auto"))] {
                            if ui.selectable_label(false, lbl).clicked() {
                                self.run("set_display", json!({"deformation_scale": v}));
                            }
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.checkbox(&mut self.animate, "Animate");
                        let mut show = disp.show_results;
                        if ui.checkbox(&mut show, "Contours").changed() {
                            self.run("set_display", json!({"show_results": show}));
                        }
                    });
                }
            });
        });
    }

    /// Forms for parameter studies, optimization, convergence, and topology optimization.
    pub(super) fn opt_form(&mut self, ui: &mut egui::Ui, form: Form) {
        let p = self.p;
        let model = self.core.session.lock().unwrap_or_else(std::sync::PoisonError::into_inner).model.clone();
        let targets = crate::study::list_targets(&model);
        let mut responses = crate::study::suggested_responses(&model);
        for r in self.opt.responses.iter().chain(std::iter::once(&self.opt.objective)) {
            if !r.is_empty() && !responses.contains(r) {
                responses.push(r.clone());
            }
        }
        let busy = self.opt.job.is_some() || self.core.is_busy();
        let title = match form {
            Form::Study => "📊 Parameter study",
            Form::Optimize => "🎯 Optimize design",
            Form::Convergence => "📐 Mesh convergence",
            _ => "🦴 Topology optimization",
        };
        ui.heading(title);
        let response_combo = |ui: &mut egui::Ui, id: String, value: &mut String| {
            egui::ComboBox::from_id_salt(id).width(190.0).selected_text(value.clone()).show_ui(ui, |ui| {
                for r in &responses {
                    ui.selectable_value(value, r.clone(), r);
                }
            });
        };
        if matches!(form, Form::Study | Form::Optimize) {
            card(ui, &p, |ui| {
                ui.label(RichText::new("Parameters (what to vary)").strong());
                let mut remove = None;
                for (i, (t, lo, hi, steps)) in self.opt.params.iter_mut().enumerate() {
                    ui.horizontal(|ui| {
                        egui::ComboBox::from_id_salt(("ptarget", i)).width(150.0).selected_text(t.clone()).show_ui(ui, |ui| {
                            for (name, cur) in &targets {
                                ui.selectable_value(t, name.clone(), format!("{name}  ({})", render::fmt_val(*cur)));
                            }
                        });
                        if ui.small_button("✖").clicked() {
                            remove = Some(i);
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label(RichText::new("from").small());
                        ui.add(egui::DragValue::new(lo).speed(0.1));
                        ui.label(RichText::new("to").small());
                        ui.add(egui::DragValue::new(hi).speed(0.1));
                        if form == Form::Study {
                            ui.label(RichText::new("steps").small());
                            ui.add(egui::DragValue::new(steps).range(2..=20));
                        }
                    });
                }
                if let Some(i) = remove {
                    self.opt.params.remove(i);
                }
                if ui.button("＋ Add parameter").clicked() {
                    if let Some((t, cur)) = targets.iter().find(|(t, _)| !self.opt.params.iter().any(|q| &q.0 == t)) {
                        let (lo, hi) = if *cur > 0.0 { (cur * 0.5, cur * 1.5) } else if *cur < 0.0 { (cur * 1.5, cur * 0.5) } else { (-1.0, 1.0) };
                        self.opt.params.push((t.clone(), lo, hi, 4));
                    }
                }
            });
            ui.add_space(6.0);
        }
        match form {
            Form::Study => {
                card(ui, &p, |ui| {
                    ui.label(RichText::new("Responses (what to record)").strong());
                    for r in &responses {
                        let mut on = self.opt.responses.contains(r);
                        if ui.checkbox(&mut on, r).changed() {
                            if on {
                                self.opt.responses.push(r.clone());
                            } else {
                                self.opt.responses.retain(|x| x != r);
                            }
                        }
                    }
                });
                let n: usize = self.opt.params.iter().map(|q| q.3.max(2)).product();
                ui.label(RichText::new(format!("{n} designs will be solved")).small().color(p.weak));
                let ready = !busy && !self.opt.params.is_empty() && !self.opt.responses.is_empty();
                if ui.add_enabled(ready, primary(&p, "Run study")).clicked() {
                    let params: Vec<Value> = self.opt.params.iter().map(|(t, lo, hi, st)| json!({"target": t, "min": lo, "max": hi, "steps": st})).collect();
                    self.start_job("run_parameter_study", json!({"parameters": params, "responses": self.opt.responses}));
                }
            }
            Form::Optimize => {
                card(ui, &p, |ui| {
                    ui.label(RichText::new("Objective").strong());
                    ui.horizontal(|ui| {
                        ui.selectable_value(&mut self.opt.maximize, false, "Minimize");
                        ui.selectable_value(&mut self.opt.maximize, true, "Maximize");
                    });
                    response_combo(ui, "objective".into(), &mut self.opt.objective);
                    ui.add_space(6.0);
                    ui.label(RichText::new("Constraints").strong());
                    let mut remove = None;
                    for (i, (r, le, v)) in self.opt.constraints.iter_mut().enumerate() {
                        ui.horizontal(|ui| {
                            response_combo(ui, format!("con{i}"), r);
                            ui.selectable_value(le, true, "≤");
                            ui.selectable_value(le, false, "≥");
                            ui.add(egui::DragValue::new(v).speed(0.1));
                            if ui.small_button("✖").clicked() {
                                remove = Some(i);
                            }
                        });
                    }
                    if let Some(i) = remove {
                        self.opt.constraints.remove(i);
                    }
                    if ui.button("＋ Add constraint").clicked() {
                        self.opt.constraints.push(("min:safety_factor".into(), false, 2.0));
                    }
                    ui.horizontal(|ui| {
                        ui.label("Method");
                        ui.selectable_value(&mut self.opt.bayes, true, "Surrogate (Bayesian)").on_hover_text("Gaussian-process surrogates pick each next design; every proposal is verified with a real solve. Fewest solves.");
                        ui.selectable_value(&mut self.opt.bayes, false, "Direct search").on_hover_text("Latin hypercube + Nelder–Mead on real solves only.");
                    });
                    ui.horizontal(|ui| {
                        ui.label(if self.opt.bayes { "Max real solves" } else { "Max evaluations" });
                        ui.add(egui::DragValue::new(&mut self.opt.max_evals).range(5..=400));
                    });
                });
                let ready = !busy && !self.opt.params.is_empty() && !self.opt.objective.is_empty();
                if ui.add_enabled(ready, primary(&p, "Optimize")).clicked() {
                    let params: Vec<Value> = self.opt.params.iter().map(|(t, lo, hi, _)| json!({"target": t, "min": lo, "max": hi})).collect();
                    let cons: Vec<Value> = self.opt.constraints.iter().map(|(r, le, v)| json!({"response": r, "op": if *le { "<=" } else { ">=" }, "value": v})).collect();
                    let goal = json!({"response": self.opt.objective, "goal": if self.opt.maximize { "maximize" } else { "minimize" }});
                    if self.opt.bayes {
                        self.start_job("surrogate_optimize", json!({"parameters": params, "objective": goal, "constraints": cons, "max_solves": self.opt.max_evals}));
                    } else {
                        self.start_job("optimize_design", json!({"parameters": params, "objective": goal, "constraints": cons, "max_evaluations": self.opt.max_evals}));
                    }
                }
            }
            Form::Convergence => {
                card(ui, &p, |ui| {
                    ui.label(RichText::new("Solves at h, h/1.4 and h/2, Richardson-extrapolates each response, and estimates discretisation error and required element size.").color(p.weak));
                    ui.add_space(4.0);
                    let cur_h = self.core.session.lock().unwrap_or_else(std::sync::PoisonError::into_inner).element_size();
                    ui.checkbox(&mut self.opt.conv_custom_h, format!("Custom coarsest h (auto: {} mm)", render::fmt_val(cur_h)));
                    if self.opt.conv_custom_h {
                        ui.horizontal(|ui| {
                            ui.label("Base h");
                            ui.add(egui::DragValue::new(&mut self.opt.conv_base_h).speed(0.1).range(0.05..=500.0).suffix(" mm"));
                        });
                    }
                    ui.horizontal(|ui| {
                        ui.label("Target error");
                        ui.add(egui::DragValue::new(&mut self.opt.conv_target_err).speed(0.05).range(0.05..=25.0).suffix(" %"));
                    });
                    ui.add_space(4.0);
                    ui.label(RichText::new("Responses (optional; defaults to main physics responses)").strong());
                    for r in &responses {
                        let mut on = self.opt.responses.contains(r);
                        if ui.checkbox(&mut on, r).changed() {
                            if on {
                                self.opt.responses.push(r.clone());
                            } else {
                                self.opt.responses.retain(|x| x != r);
                            }
                        }
                    }
                });
                if ui.add_enabled(!busy, primary(&p, "Run convergence")).clicked() {
                    let mut args = json!({"responses": self.opt.responses, "target_error_pct": self.opt.conv_target_err});
                    if self.opt.conv_custom_h {
                        args["base_element_size_mm"] = json!(self.opt.conv_base_h);
                    }
                    self.start_job("mesh_convergence", args);
                }
            }
            _ => {
                card(ui, &p, |ui| {
                    ui.label(RichText::new("Keeps the stiffest layout of material for your supports and loads. Regions under loads and supports stay solid.").color(p.weak));
                    ui.horizontal(|ui| {
                        ui.label("Keep");
                        ui.add(egui::Slider::new(&mut self.opt.vf, 0.1..=0.8).custom_formatter(|v, _| format!("{:.0}%", v * 100.0)));
                        ui.label("of the material");
                    });
                    ui.horizontal(|ui| {
                        ui.label("Iterations");
                        ui.add(egui::DragValue::new(&mut self.opt.iters).range(5..=200));
                    });
                });
                if ui.add_enabled(!busy, primary(&p, "Run topology")).clicked() {
                    self.start_job("topology_optimize", json!({"volume_fraction": self.opt.vf, "iterations": self.opt.iters}));
                }
            }
        }
        if busy {
            ui.horizontal(|ui| {
                ui.add(egui::Spinner::new());
                ui.label(RichText::new("Running — progress in the status bar").color(p.weak));
            });
        }
        if let Some(e) = &self.opt.error {
            ui.label(RichText::new(format!("✖ {e}")).color(p.err));
        }
    }
}
