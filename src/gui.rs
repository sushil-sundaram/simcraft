//! Desktop UI: ribbon tabs (Geometry · Material · Setup · Mesh · Solve · Results), model browser,
//! properties panel and a 3D viewport. Every action goes through the same tool API the MCP uses,
//! so the GUI and AI agents always see the same model.

use crate::geometry::{Op, add, dot, scale};
use crate::mesh::Mesh;
use crate::model::*;
use crate::render::{self, Camera};
use crate::results::Field;
use crate::session::Shared;
use eframe::egui::{self, Align2, Color32, CornerRadius, FontId, Pos2, Rect, RichText, Sense, Stroke, TextStyle, pos2, vec2};
use serde_json::{Value, json};
use std::collections::HashSet;
use std::sync::Arc;

// ---------------------------------------------------------------- theme

#[derive(Clone, Copy)]
struct Pal {
    dark: bool,
    panel: Color32,
    ribbon: Color32,
    card: Color32,
    hover: Color32,
    text: Color32,
    weak: Color32,
    border: Color32,
    accent: Color32,
    ok: Color32,
    warn: Color32,
    err: Color32,
}

fn pal(dark: bool) -> Pal {
    if dark {
        Pal {
            dark,
            panel: Color32::from_rgb(30, 32, 37),
            ribbon: Color32::from_rgb(37, 40, 46),
            card: Color32::from_rgb(43, 46, 53),
            hover: Color32::from_rgb(55, 59, 68),
            text: Color32::from_rgb(228, 230, 235),
            weak: Color32::from_rgb(145, 151, 162),
            border: Color32::from_rgb(56, 60, 69),
            accent: Color32::from_rgb(245, 130, 50),
            ok: Color32::from_rgb(90, 200, 120),
            warn: Color32::from_rgb(235, 175, 60),
            err: Color32::from_rgb(255, 110, 110),
        }
    } else {
        Pal {
            dark,
            panel: Color32::from_rgb(248, 249, 250),
            ribbon: Color32::from_rgb(255, 255, 255),
            card: Color32::from_rgb(255, 255, 255),
            hover: Color32::from_rgb(236, 239, 243),
            text: Color32::from_rgb(32, 34, 38),
            weak: Color32::from_rgb(112, 118, 128),
            border: Color32::from_rgb(222, 225, 230),
            accent: Color32::from_rgb(232, 112, 32),
            ok: Color32::from_rgb(30, 150, 70),
            warn: Color32::from_rgb(190, 120, 0),
            err: Color32::from_rgb(200, 40, 40),
        }
    }
}

fn install_style(ctx: &egui::Context) {
    ctx.all_styles_mut(|s| {
        let p = pal(s.visuals.dark_mode);
        s.spacing.item_spacing = vec2(8.0, 6.0);
        s.spacing.button_padding = vec2(10.0, 4.0);
        s.spacing.interact_size.y = 24.0;
        s.text_styles.insert(TextStyle::Body, FontId::proportional(13.5));
        s.text_styles.insert(TextStyle::Button, FontId::proportional(13.5));
        s.text_styles.insert(TextStyle::Small, FontId::proportional(11.5));
        s.text_styles.insert(TextStyle::Heading, FontId::proportional(17.0));
        let v = &mut s.visuals;
        v.panel_fill = p.panel;
        v.window_fill = p.card;
        v.extreme_bg_color = if p.dark { Color32::from_rgb(22, 24, 28) } else { Color32::WHITE };
        v.faint_bg_color = p.hover;
        v.hyperlink_color = p.accent;
        v.selection.bg_fill = p.accent.gamma_multiply(if p.dark { 0.45 } else { 0.30 });
        v.selection.stroke = Stroke::new(1.0, p.accent);
        v.window_corner_radius = CornerRadius::same(10);
        v.window_stroke = Stroke::new(1.0, p.border);
        for w in [&mut v.widgets.noninteractive, &mut v.widgets.inactive, &mut v.widgets.hovered, &mut v.widgets.active, &mut v.widgets.open] {
            w.corner_radius = CornerRadius::same(6);
        }
        v.widgets.noninteractive.bg_stroke = Stroke::new(1.0, p.border);
        v.widgets.inactive.weak_bg_fill = p.hover;
        v.widgets.inactive.bg_fill = p.hover;
    });
}

fn settings_path() -> Option<std::path::PathBuf> {
    let base = std::env::var_os("XDG_CONFIG_HOME").map(std::path::PathBuf::from).or_else(|| std::env::var_os("HOME").map(|h| std::path::PathBuf::from(h).join(".config")))?;
    Some(base.join("simcraft").join("settings.json"))
}
fn load_dark_pref() -> Option<bool> {
    let t = std::fs::read_to_string(settings_path()?).ok()?;
    serde_json::from_str::<Value>(&t).ok()?.get("dark_mode")?.as_bool()
}
fn save_dark_pref(dark: bool) {
    if let Some(p) = settings_path() {
        let _ = std::fs::create_dir_all(p.parent().unwrap());
        let _ = std::fs::write(p, json!({"dark_mode": dark}).to_string());
    }
}

// ---------------------------------------------------------------- state

#[derive(PartialEq, Clone, Copy)]
enum Tab {
    Geometry,
    Material,
    Setup,
    Mesh,
    Solve,
    Results,
}

#[derive(PartialEq, Clone, Copy, Debug)]
enum LoadKind {
    Fixed,
    Displacement,
    Force,
    Pressure,
    Gravity,
    Temperature,
    HeatFlux,
    Convection,
    HeatGen,
}
impl LoadKind {
    fn label(self) -> &'static str {
        match self {
            LoadKind::Fixed => "Fixed",
            LoadKind::Displacement => "Displacement",
            LoadKind::Force => "Force",
            LoadKind::Pressure => "Pressure",
            LoadKind::Gravity => "Gravity",
            LoadKind::Temperature => "Temperature",
            LoadKind::HeatFlux => "Heat Flux",
            LoadKind::Convection => "Convection",
            LoadKind::HeatGen => "Heat Gen.",
        }
    }
    fn title(self) -> &'static str {
        match self {
            LoadKind::Fixed => "Fixed Support",
            LoadKind::HeatGen => "Internal Heat Generation",
            k => k.label(),
        }
    }
    fn icon(self) -> &'static str {
        match self {
            LoadKind::Fixed => "📌",
            LoadKind::Displacement => "↔",
            LoadKind::Force => "➡",
            LoadKind::Pressure => "⬇",
            LoadKind::Gravity => "🌍",
            LoadKind::Temperature => "🌡",
            LoadKind::HeatFlux => "☀",
            LoadKind::Convection => "💨",
            LoadKind::HeatGen => "🔥",
        }
    }
    fn color(self) -> Color32 {
        let l = match self {
            LoadKind::Fixed => Load::FixedSupport { faces: vec![] },
            LoadKind::Displacement => Load::Displacement { faces: vec![], x: None, y: None, z: None },
            LoadKind::Force => Load::Force { faces: vec![], vector: [0.0; 3] },
            LoadKind::Pressure => Load::Pressure { faces: vec![], value_mpa: 0.0 },
            LoadKind::Temperature => Load::Temperature { faces: vec![], value: 0.0 },
            LoadKind::HeatFlux => Load::HeatFlux { faces: vec![], value: 0.0 },
            LoadKind::Convection => Load::Convection { faces: vec![], coefficient: 0.0, ambient: 0.0 },
            LoadKind::Gravity => return Color32::from_rgb(70, 150, 90),
            LoadKind::HeatGen => return Color32::from_rgb(235, 90, 40),
        };
        let c = render::load_color(&l);
        Color32::from_rgb(c[0], c[1], c[2])
    }
    fn needs_faces(self) -> bool {
        !matches!(self, LoadKind::Gravity | LoadKind::HeatGen)
    }
}

#[derive(PartialEq, Clone, Copy)]
enum Form {
    None,
    Box,
    Cylinder,
    Sphere,
    Load(LoadKind),
}

struct Forms {
    name: String,
    subtract: bool,
    bmin: [f64; 3],
    bsize: [f64; 3],
    cbase: [f64; 3],
    caxis: usize,
    cradius: f64,
    cheight: f64,
    scenter: [f64; 3],
    sradius: f64,
    vector: [f64; 3],
    value: f64,
    coef: f64,
    ambient: f64,
    dcomp: [(bool, f64); 3],
    grav: [f64; 3],
    material: String,
    mesh_size: f64,
}

impl Default for Forms {
    fn default() -> Self {
        Forms {
            name: String::new(),
            subtract: false,
            bmin: [0.0; 3],
            bsize: [100.0, 40.0, 10.0],
            cbase: [50.0, 20.0, 0.0],
            caxis: 2,
            cradius: 8.0,
            cheight: 10.0,
            scenter: [0.0; 3],
            sradius: 20.0,
            vector: [0.0, 0.0, -1000.0],
            value: 1.0,
            coef: 20.0,
            ambient: 22.0,
            dcomp: [(true, 0.0), (false, 0.0), (false, 0.0)],
            grav: [0.0, 0.0, -9.80665],
            material: "Structural Steel".into(),
            mesh_size: 2.0,
        }
    }
}

#[derive(PartialEq, Clone)]
struct RenderKey {
    rev: u64,
    cam: Camera,
    w: usize,
    h: usize,
    sel: u64,
    anim: i64,
    mesh: usize,
    dark: bool,
}

pub struct SimApp {
    core: Shared,
    cam: Camera,
    tab: Tab,
    form: Form,
    f: Forms,
    tex: Option<egui::TextureHandle>,
    key: Option<RenderKey>,
    face_buf: Vec<u32>,
    buf_size: (usize, usize),
    render_scale: f32,
    selected: HashSet<u32>,
    sel_version: u64,
    mesh_ptr: usize,
    mesh_err: Option<(u64, String)>,
    bridge: String,
    animate: bool,
    need_fit: bool,
    toast: Option<(String, bool, f64)>,
    legend: Option<((f64, f64), f64)>,
    dark: bool,
    p: Pal,
}

// ---------------------------------------------------------------- widgets

/// Flat ribbon tool button: big icon above a short label.
fn tool(ui: &mut egui::Ui, p: &Pal, icon: &str, label: &str, color: Option<Color32>, enabled: bool) -> bool {
    let (rect, resp) = ui.allocate_exact_size(vec2(70.0, 56.0), if enabled { Sense::click() } else { Sense::hover() });
    let painter = ui.painter_at(rect);
    if enabled && resp.hovered() {
        painter.rect_filled(rect, 7.0, p.hover);
    }
    let a = if enabled { 1.0 } else { 0.35 };
    let ic = color.unwrap_or(p.accent).gamma_multiply(a);
    painter.text(rect.center_top() + vec2(0.0, 19.0), Align2::CENTER_CENTER, icon, FontId::proportional(21.0), ic);
    painter.text(rect.center_bottom() - vec2(0.0, 12.0), Align2::CENTER_CENTER, label, FontId::proportional(11.5), p.text.gamma_multiply(a));
    let resp = if enabled { resp.on_hover_cursor(egui::CursorIcon::PointingHand) } else { resp };
    enabled && resp.clicked()
}

/// A captioned group of ribbon tools followed by a divider.
fn group(ui: &mut egui::Ui, p: &Pal, caption: &str, add: impl FnOnce(&mut egui::Ui)) {
    ui.vertical(|ui| {
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 2.0;
            add(ui);
        });
        ui.label(RichText::new(caption).size(10.0).color(p.weak).strong());
    });
    let (r, _) = ui.allocate_exact_size(vec2(1.0, 66.0), Sense::hover());
    ui.painter().line_segment([r.center_top(), r.center_bottom()], Stroke::new(1.0, p.border));
}

fn card<R>(ui: &mut egui::Ui, p: &Pal, add: impl FnOnce(&mut egui::Ui) -> R) -> R {
    egui::Frame::new().fill(p.card).stroke(Stroke::new(1.0, p.border)).corner_radius(8).inner_margin(10).show(ui, |ui| {
        ui.set_width(ui.available_width());
        add(ui)
    }).inner
}

fn xyz(ui: &mut egui::Ui, id: &str, v: &mut [f64; 3], suffix: &str, speed: f64) {
    egui::Grid::new(id).num_columns(6).spacing([4.0, 4.0]).show(ui, |ui| {
        for (i, c) in ["X", "Y", "Z"].iter().enumerate() {
            ui.label(RichText::new(*c).small());
            ui.add(egui::DragValue::new(&mut v[i]).speed(speed).suffix(suffix));
        }
        ui.end_row();
    });
}

fn primary(p: &Pal, text: &str) -> egui::Button<'static> {
    egui::Button::new(RichText::new(text.to_string()).color(Color32::WHITE).strong()).fill(p.accent).min_size(vec2(110.0, 30.0))
}

// ---------------------------------------------------------------- app

impl SimApp {
    pub fn new(cc: &eframe::CreationContext<'_>, core: Shared, bridge: String) -> Self {
        install_style(&cc.egui_ctx);
        let dark = load_dark_pref().unwrap_or_else(|| cc.egui_ctx.system_theme() == Some(egui::Theme::Dark));
        cc.egui_ctx.set_theme(if dark { egui::ThemePreference::Dark } else { egui::ThemePreference::Light });
        core.session.lock().unwrap().display.dark = dark;
        *core.gui_ctx.lock().unwrap() = Some(cc.egui_ctx.clone());
        let ctx = cc.egui_ctx.clone();
        *core.notify.lock().unwrap() = Some(Box::new(move || ctx.request_repaint()));
        SimApp {
            core,
            cam: Camera::default(),
            tab: Tab::Geometry,
            form: Form::None,
            f: Forms::default(),
            tex: None,
            key: None,
            face_buf: vec![],
            buf_size: (0, 0),
            render_scale: 1.0,
            selected: HashSet::new(),
            sel_version: 0,
            mesh_ptr: 0,
            mesh_err: None,
            bridge,
            animate: false,
            need_fit: true,
            toast: None,
            legend: None,
            dark,
            p: pal(dark),
        }
    }

    /// Run a tool and surface the result as a toast.
    fn run(&mut self, tool: &str, args: Value) -> bool {
        match self.core.call(tool, &args) {
            Ok(o) => {
                let short: String = o.text.lines().next().unwrap_or("").chars().take(140).collect();
                if !short.starts_with('{') {
                    self.toast = Some((short, false, 3.5));
                }
                true
            }
            Err(e) => {
                self.toast = Some((format!("{e:#}"), true, 8.0));
                false
            }
        }
    }

    fn selected_names(&self, mesh: Option<&Arc<Mesh>>) -> Vec<String> {
        let Some(m) = mesh else { return vec![] };
        let mut v: Vec<String> = self.selected.iter().filter_map(|&p| m.patches.get(p as usize).map(|x| x.name.clone())).collect();
        v.sort();
        v
    }

    fn current_mesh(&self) -> Option<Arc<Mesh>> {
        let s = self.core.session.lock().unwrap();
        if s.mesh_valid() { s.mesh.clone() } else { None }
    }

    fn solve(&mut self) {
        if self.core.is_busy() {
            return;
        }
        let core = self.core.clone();
        std::thread::spawn(move || {
            let _ = core.call("solve", &json!({}));
        });
        self.tab = Tab::Results;
    }

    fn sync_theme(&mut self, ctx: &egui::Context) {
        let want = self.core.session.lock().unwrap().display.dark;
        if want != self.dark {
            self.dark = want;
            self.p = pal(want);
            ctx.set_theme(if want { egui::ThemePreference::Dark } else { egui::ThemePreference::Light });
            save_dark_pref(want);
        }
    }

    // ------------------------------------------------ header (logo, menus, theme)
    fn header(&mut self, ui: &mut egui::Ui) {
        let p = self.p;
        ui.horizontal(|ui| {
            ui.label(RichText::new("◆").size(18.0).color(p.accent));
            ui.label(RichText::new("SimCraft").size(15.0).strong().color(p.text));
            ui.add_space(10.0);
            egui::MenuBar::new().ui(ui, |ui| {
                ui.menu_button("File", |ui| self.file_menu(ui));
                ui.menu_button("Examples", |ui| {
                    if ui.button("🔩  Cantilever bracket with hole").clicked() {
                        ui.close();
                        self.example_bracket();
                    }
                    if ui.button("🌡  Heat sink (thermal)").clicked() {
                        ui.close();
                        self.example_heatsink();
                    }
                    if ui.button("🎵  Tuning fork (modal)").clicked() {
                        ui.close();
                        self.example_fork();
                    }
                });
                ui.menu_button("MCP", |ui| self.mcp_menu(ui));
            });
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let icon = if self.dark { "☀" } else { "🌙" };
                let tip = if self.dark { "Switch to light mode" } else { "Switch to dark mode" };
                if ui.add(egui::Button::new(RichText::new(icon).size(15.0)).frame_when_inactive(false)).on_hover_text(tip).clicked() {
                    let d = !self.dark;
                    self.run("set_display", json!({"dark_mode": d}));
                }
                let ok = self.bridge.starts_with("MCP bridge listening");
                let (c, t) = if ok { (p.ok, "● MCP ready") } else { (p.warn, "○ MCP bridge off") };
                ui.label(RichText::new(t).size(11.5).color(c)).on_hover_text(&self.bridge);
                let name = self.core.session.lock().unwrap().project_path.clone();
                if let Some(n) = name {
                    let f = std::path::Path::new(&n).file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or(n);
                    ui.label(RichText::new(f).size(12.0).color(p.weak));
                }
            });
        });
    }

    fn file_menu(&mut self, ui: &mut egui::Ui) {
        if ui.button("New").clicked() {
            ui.close();
            self.run("new_model", json!({}));
            self.need_fit = true;
        }
        if ui.button("Open project…").clicked() {
            ui.close();
            if let Some(p) = rfd::FileDialog::new().add_filter("SimCraft", &["simcraft", "json"]).pick_file() {
                self.run("open_project", json!({"path": p.to_string_lossy()}));
                self.need_fit = true;
            }
        }
        if ui.button("Save").clicked() {
            ui.close();
            let has = self.core.session.lock().unwrap().project_path.is_some();
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
        if ui.button("Import STL…").clicked() {
            ui.close();
            self.import_stl();
        }
        if ui.button("Export results (VTK)…").clicked() {
            ui.close();
            if let Some(p) = rfd::FileDialog::new().add_filter("VTK", &["vtk"]).set_file_name("results.vtk").save_file() {
                self.run("export_vtk", json!({"path": p.to_string_lossy()}));
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

    fn mcp_menu(&mut self, ui: &mut egui::Ui) {
        ui.set_min_width(380.0);
        ui.label(RichText::new(&self.bridge).small());
        ui.separator();
        let exe = std::env::current_exe().map(|p| p.to_string_lossy().to_string()).unwrap_or("simcraft".into());
        let cmd = format!("claude mcp add simcraft -- {exe} mcp");
        ui.label("Claude Code");
        ui.code(&cmd);
        if ui.button("📋  Copy Claude Code command").clicked() {
            ui.ctx().copy_text(cmd.clone());
            ui.close();
        }
        ui.add_space(4.0);
        ui.label("Antigravity / Cursor / other MCP clients (mcp_config.json)");
        let js = format!("{{\n  \"mcpServers\": {{\n    \"simcraft\": {{\n      \"command\": \"{exe}\",\n      \"args\": [\"mcp\"]\n    }}\n  }}\n}}");
        ui.code(&js);
        if ui.button("📋  Copy JSON config").clicked() {
            ui.ctx().copy_text(js);
            ui.close();
        }
    }

    // ------------------------------------------------ ribbon
    fn ribbon(&mut self, ui: &mut egui::Ui) {
        let p = self.p;
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 18.0;
            for (t, name) in [(Tab::Geometry, "GEOMETRY"), (Tab::Material, "MATERIAL"), (Tab::Setup, "SETUP"), (Tab::Mesh, "MESH"), (Tab::Solve, "SOLVE"), (Tab::Results, "RESULTS")] {
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
        let kind = self.core.session.lock().unwrap().model.analysis.kind;
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
                });
            }
            Tab::Material => {
                let lib = self.core.session.lock().unwrap().model.materials();
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
                        egui::ComboBox::from_id_salt("analysis").width(180.0).selected_text(k.label()).show_ui(ui, |ui| {
                            for a in AnalysisType::all() {
                                ui.selectable_value(&mut k, a, a.label());
                            }
                        });
                        if k != kind {
                            self.run("set_analysis", json!({"type": serde_json::to_value(k).unwrap()}));
                        }
                    });
                });
                group(ui, &p, "STRUCTURAL", |ui| {
                    for lk in [LoadKind::Fixed, LoadKind::Displacement, LoadKind::Force, LoadKind::Pressure, LoadKind::Gravity] {
                        let en = match kind {
                            AnalysisType::SteadyThermal => false,
                            AnalysisType::Modal => matches!(lk, LoadKind::Fixed | LoadKind::Displacement),
                            _ => true,
                        };
                        if tool(ui, &p, lk.icon(), lk.label(), Some(lk.color()), en) {
                            self.form = Form::Load(lk);
                        }
                    }
                });
                group(ui, &p, "THERMAL", |ui| {
                    for lk in [LoadKind::Temperature, LoadKind::HeatFlux, LoadKind::Convection, LoadKind::HeatGen] {
                        if tool(ui, &p, lk.icon(), lk.label(), Some(lk.color()), kind.thermal()) {
                            self.form = Form::Load(lk);
                        }
                    }
                });
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
                        self.f.mesh_size = self.core.session.lock().unwrap().element_size();
                    }
                });
                group(ui, &p, "DISPLAY", |ui| {
                    let mut sm = self.core.session.lock().unwrap().display.show_mesh;
                    ui.vertical(|ui| {
                        ui.add_space(14.0);
                        if ui.checkbox(&mut sm, "Element edges").changed() {
                            self.run("set_display", json!({"show_mesh": sm}));
                        }
                    });
                });
            }
            Tab::Solve => {
                let busy = self.core.is_busy();
                group(ui, &p, "RUN", |ui| {
                    ui.vertical(|ui| {
                        ui.add_space(8.0);
                        let b = egui::Button::new(RichText::new("⚡  Solve").size(15.0).strong().color(Color32::WHITE)).fill(p.accent).min_size(vec2(120.0, 40.0));
                        if ui.add_enabled(!busy, b).clicked() {
                            self.solve();
                        }
                    });
                });
                group(ui, &p, "SETTINGS", |ui| {
                    ui.vertical(|ui| {
                        ui.add_space(4.0);
                        ui.label(RichText::new(kind.label()).strong());
                        if kind == AnalysisType::Modal {
                            let mut m = self.core.session.lock().unwrap().model.analysis.modes as u32;
                            ui.horizontal(|ui| {
                                ui.label("Modes");
                                if ui.add(egui::DragValue::new(&mut m).range(1..=20)).changed() {
                                    self.run("set_analysis", json!({"modes": m}));
                                }
                            });
                        }
                        if kind == AnalysisType::ThermalStress {
                            let mut t = self.core.session.lock().unwrap().model.analysis.reference_temperature;
                            ui.horizontal(|ui| {
                                ui.label("Reference temp.");
                                if ui.add(egui::DragValue::new(&mut t).suffix(" °C")).changed() {
                                    self.run("set_analysis", json!({"reference_temperature": t}));
                                }
                            });
                        }
                    });
                });
            }
            Tab::Results => self.results_ribbon(ui),
        });
    }

    fn results_ribbon(&mut self, ui: &mut egui::Ui) {
        let p = self.p;
        let (res, disp) = {
            let s = self.core.session.lock().unwrap();
            (s.results.clone(), s.display.clone())
        };
        let Some(r) = res else {
            ui.add_space(18.0);
            ui.label(RichText::new("No results yet — open SOLVE and press Solve.").color(p.weak));
            return;
        };
        let groups: [(&str, &[(Field, &str, &str)]); 3] = [
            ("DEFORMATION", &[(Field::TotalDeformation, "📐", "Total"), (Field::DeformationX, "X", "Dir. X"), (Field::DeformationY, "Y", "Dir. Y"), (Field::DeformationZ, "Z", "Dir. Z")]),
            ("STRESS", &[(Field::VonMises, "🔶", "Von Mises"), (Field::MaxPrincipal, "⬆", "Max Princ."), (Field::MinPrincipal, "⬇", "Min Princ."), (Field::SafetyFactor, "🛡", "Safety")]),
            ("THERMAL", &[(Field::Temperature, "🌡", "Temperature"), (Field::HeatFlux, "☀", "Heat Flux")]),
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
        group(ui, &p, "VIEW", |ui| {
            ui.vertical(|ui| {
                if r.analysis == AnalysisType::Modal {
                    egui::ComboBox::from_id_salt("mode").width(160.0).selected_text(format!("Mode {} — {:.2} Hz", disp.mode + 1, r.modes.get(disp.mode).map(|m| m.frequency_hz).unwrap_or(0.0))).show_ui(ui, |ui| {
                        for (i, m) in r.modes.iter().enumerate() {
                            if ui.selectable_label(disp.mode == i, format!("Mode {} — {:.2} Hz", i + 1, m.frequency_hz)).clicked() {
                                self.run("set_display", json!({"mode": i + 1}));
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

    fn save_as(&mut self) {
        if let Some(p) = rfd::FileDialog::new().add_filter("SimCraft", &["simcraft"]).set_file_name("model.simcraft").save_file() {
            self.run("save_project", json!({"path": p.to_string_lossy()}));
        }
    }

    fn import_stl(&mut self) {
        if let Some(p) = rfd::FileDialog::new().add_filter("STL", &["stl", "STL"]).pick_file() {
            self.run("import_stl", json!({"path": p.to_string_lossy()}));
            self.need_fit = true;
        }
    }

    // ------------------------------------------------ browser
    fn browser(&mut self, ui: &mut egui::Ui) {
        let p = self.p;
        let (model, mesh, res, stale) = {
            let s = self.core.session.lock().unwrap();
            (s.model.clone(), if s.mesh_valid() { s.mesh.clone() } else { None }, s.results.clone(), s.results_stale)
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
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if del(ui) {
                                self.run("remove_body", json!({"name": b.name}));
                            }
                            if b.op == Op::Subtract {
                                ui.label(RichText::new("cut").small().color(p.weak));
                            }
                            ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                                ui.add(egui::Label::new(lbl).truncate()).on_hover_text(tip);
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
                                ui.add(egui::Label::new(l.load.label()).truncate()).on_hover_text(format!("#{} {}", l.id, l.load.summary()));
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
                        let sum = r.summary(m);
                        if let Some(fields) = sum["fields"].as_object() {
                            egui::Grid::new("resgrid").num_columns(2).spacing([8.0, 2.0]).show(ui, |ui| {
                                let rows = [
                                    ("total_deformation", "Max deformation", "max"),
                                    ("von_mises", "Max von Mises", "max"),
                                    ("max_principal", "Max principal", "max"),
                                    ("min_principal", "Min principal", "min"),
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
                    for (i, md) in r.modes.iter().enumerate() {
                        if ui.link(format!("Mode {} — {:.2} Hz", i + 1, md.frequency_hz)).clicked() {
                            self.run("set_display", json!({"mode": i + 1, "show_results": true}));
                        }
                    }
                    for rc in &r.reactions {
                        let t = if r.analysis == AnalysisType::SteadyThermal {
                            format!("#{} heat flow {:.4} W", rc.load_id, rc.value[0])
                        } else {
                            format!("#{} reaction ({:.1}, {:.1}, {:.1}) N", rc.load_id, rc.value[0], rc.value[1], rc.value[2])
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

    // ------------------------------------------------ properties
    fn properties(&mut self, ui: &mut egui::Ui) {
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
                LoadKind::HeatGen => {
                    ui.add(egui::DragValue::new(&mut self.f.value).speed(1000.0).suffix(" W/m³"));
                }
            }
        });
        ui.add_space(8.0);
        let ready = !k.needs_faces() || !sel.is_empty();
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
                LoadKind::HeatGen => ("add_heat_generation", json!({"value": self.f.value})),
            };
            if self.run(tool, args) {
                self.selected.clear();
                self.sel_version += 1;
            }
        }
    }

    // ------------------------------------------------ viewport
    fn viewport(&mut self, ui: &mut egui::Ui) {
        let p = self.p;
        let rect = ui.available_rect_before_wrap();
        let resp = ui.allocate_rect(rect, Sense::click_and_drag());
        let painter = ui.painter_at(rect);

        let (mesh, model, results, display, rev, view_req, stale) = {
            let mut s = self.core.session.lock().unwrap();
            let mesh = if s.model.bodies.is_empty() || self.mesh_err.as_ref().is_some_and(|(r, _)| *r == s.geom_rev) {
                None
            } else {
                match s.ensure_mesh() {
                    Ok(m) => {
                        self.mesh_err = None;
                        Some(m)
                    }
                    Err(e) => {
                        self.mesh_err = Some((s.geom_rev, format!("{e:#}")));
                        None
                    }
                }
            };
            (mesh, s.model.clone(), s.results.clone(), s.display.clone(), s.revision, s.view_request.take(), s.results_stale)
        };

        let (bt, bb) = render::background(self.dark);
        let c32 = |c: [f32; 3]| Color32::from_rgb((c[0] * 255.0) as u8, (c[1] * 255.0) as u8, (c[2] * 255.0) as u8);
        let Some(mesh) = mesh else {
            painter.rect_filled(rect, 0.0, c32(bb));
            let (title, sub) = match &self.mesh_err {
                Some((_, e)) => ("⚠  Geometry problem".to_string(), e.clone()),
                None => ("Start a new simulation".into(), "GEOMETRY ▸ Box / Cylinder / Import STL      or      Examples ▸ Cantilever bracket".into()),
            };
            painter.text(rect.center() - vec2(0.0, 14.0), Align2::CENTER_CENTER, title, FontId::proportional(22.0), p.text);
            painter.text(rect.center() + vec2(0.0, 16.0), Align2::CENTER_CENTER, sub, FontId::proportional(13.0), p.weak);
            let _ = bt;
            return;
        };
        let mptr = Arc::as_ptr(&mesh) as usize;
        if mptr != self.mesh_ptr {
            self.mesh_ptr = mptr;
            self.selected.clear();
            self.sel_version += 1;
        }
        {
            let names = self.selected_names(Some(&mesh));
            let mut s = self.core.session.lock().unwrap();
            if s.selection != names {
                s.selection = names;
            }
        }

        if let Some(v) = view_req {
            if v != "fit" {
                self.cam.set_view(&v);
            }
            self.need_fit = true;
        }
        if self.need_fit {
            // leave room for the legend card on the left
            let reserve = if display.show_results && rect.width() > 760.0 { 240.0 } else { 0.0 };
            self.cam.fit(mesh.bbox.0, mesh.bbox.1, (rect.width() - reserve) as f64, rect.height() as f64);
            self.cam.pan(-(reserve as f64) / 2.0, 0.0);
            self.need_fit = false;
        }

        // ---- input
        let dragging = resp.dragged();
        if resp.dragged_by(egui::PointerButton::Primary) {
            let d = resp.drag_delta();
            self.cam.orbit(d.x as f64, d.y as f64);
        } else if resp.dragged_by(egui::PointerButton::Secondary) || resp.dragged_by(egui::PointerButton::Middle) {
            let d = resp.drag_delta();
            self.cam.pan(d.x as f64, d.y as f64);
        }
        if resp.hovered() {
            let dy = ui.input(|i| i.smooth_scroll_delta.y);
            if dy != 0.0 {
                let old = self.cam.zoom;
                let new = (old * (dy as f64 * 0.003).exp()).clamp(1e-4, 1e6);
                if let Some(hp) = resp.hover_pos() {
                    let (r, u, _) = self.cam.basis();
                    let k = 1.0 / old - 1.0 / new;
                    let ox = (hp.x - rect.center().x) as f64 * k;
                    let oy = (rect.center().y - hp.y) as f64 * k;
                    self.cam.target = add(self.cam.target, add(scale(r, ox), scale(u, oy)));
                }
                self.cam.zoom = new;
            }
        }

        // ---- animation
        let anim_on = self.animate && display.show_results && results.as_ref().is_some_and(|r| r.deformation(display.mode).is_some());
        let anim = if anim_on {
            ui.ctx().request_repaint();
            (ui.input(|i| i.time) * std::f64::consts::TAU * 0.7).sin()
        } else {
            1.0
        };

        // ---- render
        let ppp = ui.ctx().pixels_per_point();
        self.render_scale = ppp * if dragging || anim_on { 0.6 } else { 1.0 };
        let w = ((rect.width() * self.render_scale) as usize).max(16);
        let h = ((rect.height() * self.render_scale) as usize).max(16);
        let key = RenderKey { rev, cam: self.cam, w, h, sel: self.sel_version, anim: (anim * 1000.0) as i64, mesh: mptr, dark: self.dark };
        if self.key.as_ref() != Some(&key) {
            let scene = render::build_scene(&mesh, &model, results.as_deref(), &display, &self.selected, anim);
            let mut cam = self.cam;
            cam.zoom *= self.render_scale as f64;
            let frame = render::rasterize(&scene, &cam, w, h, display.show_mesh && !anim_on, self.dark);
            let img = egui::ColorImage::from_rgba_unmultiplied([w, h], &frame.rgba);
            match &mut self.tex {
                Some(t) => t.set(img, egui::TextureOptions::LINEAR),
                None => self.tex = Some(ui.ctx().load_texture("viewport", img, egui::TextureOptions::LINEAR)),
            }
            self.face_buf = frame.face_id;
            self.buf_size = (w, h);
            self.key = Some(key);
            self.legend = scene.range.map(|r| (r, scene.scale_used));
        }
        if let Some(t) = &self.tex {
            painter.image(t.id(), rect, Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)), Color32::WHITE);
        }

        // ---- picking + hover
        let pick = |pos: Pos2| -> Option<u32> {
            let x = ((pos.x - rect.min.x) * self.render_scale) as usize;
            let y = ((pos.y - rect.min.y) * self.render_scale) as usize;
            let (bw, bh) = self.buf_size;
            (x < bw && y < bh).then(|| self.face_buf[y * bw + x]).filter(|f| *f != u32::MAX)
        };
        if resp.clicked() {
            if let Some(pos) = resp.interact_pointer_pos() {
                let shift = ui.input(|i| i.modifiers.shift || i.modifiers.command);
                match pick(pos) {
                    Some(fid) => {
                        let patch = mesh.faces[fid as usize].patch;
                        if shift {
                            if !self.selected.remove(&patch) {
                                self.selected.insert(patch);
                            }
                        } else if self.selected.len() == 1 && self.selected.contains(&patch) {
                            self.selected.clear();
                        } else {
                            self.selected.clear();
                            self.selected.insert(patch);
                        }
                    }
                    None if !shift => self.selected.clear(),
                    None => {}
                }
                self.sel_version += 1;
            }
        }
        let text_col = if self.dark { Color32::from_gray(235) } else { Color32::from_gray(25) };
        let card_fill = p.card.gamma_multiply(0.92);
        if let (Some(hp), false) = (resp.hover_pos(), dragging) {
            if let Some(fid) = pick(hp) {
                if let Some(pi) = mesh.patches.get(mesh.faces[fid as usize].patch as usize) {
                    let galley = painter.layout_no_wrap(pi.name.clone(), FontId::proportional(12.0), text_col);
                    let r = Rect::from_min_size(hp + vec2(14.0, 12.0), galley.size() + vec2(12.0, 6.0));
                    painter.rect(r, 5.0, card_fill, Stroke::new(1.0, p.border), egui::StrokeKind::Inside);
                    painter.galley(r.min + vec2(6.0, 3.0), galley, text_col);
                }
            }
        }

        // ---- title + legend card
        let o = rect.left_top() + vec2(14.0, 14.0);
        if display.show_results {
            if let (Some(r), Some(((mn, mx), sc))) = (&results, self.legend) {
                let f = if r.has(display.field) { display.field } else { r.available_fields()[0] };
                let mut title = f.label().to_string();
                let mut unit = f.unit().to_string();
                if r.analysis == AnalysisType::Modal {
                    if let Some(m) = r.modes.get(display.mode) {
                        title = format!("Mode {} · {:.2} Hz", display.mode + 1, m.frequency_hz);
                        unit = "normalised".into();
                    }
                }
                let mut sub = model.analysis.kind.label().to_string();
                if sc > 0.0 {
                    sub += &format!("  ·  scale ×{}", render::fmt_val(sc));
                }
                let compact = rect.width() < 700.0 || rect.height() < 560.0;
                let bh = if compact { 14.0 } else { 22.0 };
                let card_r = Rect::from_min_size(o, vec2(if compact { 170.0 } else { 220.0 }, 62.0 + bh * 9.0 + 14.0));
                painter.rect(card_r, 10.0, p.card.gamma_multiply(0.80), Stroke::new(1.0, p.border), egui::StrokeKind::Inside);
                let i = card_r.min + vec2(12.0, 10.0);
                painter.text(i, Align2::LEFT_TOP, &title, FontId::proportional(14.0), text_col);
                painter.text(i + vec2(0.0, 19.0), Align2::LEFT_TOP, format!("{unit}   {sub}"), FontId::proportional(11.0), p.weak);
                if stale {
                    painter.text(i + vec2(0.0, 33.0), Align2::LEFT_TOP, "⚠ results stale — solve again", FontId::proportional(11.0), p.warn);
                }
                let top = i + vec2(0.0, 52.0);
                for b in 0..9 {
                    let c = render::BANDS[8 - b];
                    let rr = Rect::from_min_size(top + vec2(0.0, b as f32 * bh), vec2(16.0, bh));
                    painter.rect_filled(rr, 0.0, Color32::from_rgb(c[0], c[1], c[2]));
                }
                painter.rect_stroke(Rect::from_min_size(top, vec2(16.0, bh * 9.0)), 2.0, Stroke::new(1.0, p.border), egui::StrokeKind::Outside);
                for b in 0..=9 {
                    let mut v = mx - (mx - mn) * b as f64 / 9.0;
                    if v.abs() < 1e-9 * mx.abs().max(mn.abs()) {
                        v = 0.0;
                    }
                    let tag = match b {
                        0 => "  max",
                        9 => "  min",
                        _ => "",
                    };
                    painter.text(top + vec2(24.0, b as f32 * bh), Align2::LEFT_CENTER, format!("{}{tag}", render::fmt_val(v)), FontId::monospace(if compact { 10.0 } else { 11.5 }), text_col);
                }
            }
        } else {
            let t = format!("{}  ·  model", model.analysis.kind.label());
            let galley = painter.layout_no_wrap(t, FontId::proportional(13.0), text_col);
            let r = Rect::from_min_size(o, galley.size() + vec2(20.0, 12.0));
            painter.rect(r, 8.0, card_fill, Stroke::new(1.0, p.border), egui::StrokeKind::Inside);
            painter.galley(r.min + vec2(10.0, 6.0), galley, text_col);
        }

        // ---- axis triad
        let (r, u, _) = self.cam.basis();
        let c0 = Pos2::new(rect.left() + 52.0, rect.bottom() - 52.0);
        painter.circle(c0, 40.0, card_fill, Stroke::new(1.0, p.border));
        for (k, (name, col)) in [("X", Color32::from_rgb(225, 70, 70)), ("Y", Color32::from_rgb(70, 180, 90)), ("Z", Color32::from_rgb(70, 120, 235))].iter().enumerate() {
            let mut e = [0.0; 3];
            e[k] = 1.0;
            let d = vec2((dot(e, r) * 26.0) as f32, -(dot(e, u) * 26.0) as f32);
            painter.line_segment([c0, c0 + d], Stroke::new(2.5, *col));
            painter.text(c0 + d * 1.28, Align2::CENTER_CENTER, *name, FontId::proportional(11.0), *col);
        }

        // ---- view toolbar (top right)
        let views = [("⛶", "fit", "Fit all"), ("◩", "iso", "Isometric"), ("▣", "front", "Front"), ("⬒", "top", "Top"), ("◨", "right", "Right"), ("◧", "left", "Left")];
        let tb = Rect::from_min_size(pos2(rect.right() - 14.0 - views.len() as f32 * 30.0 - 8.0, rect.top() + 14.0), vec2(views.len() as f32 * 30.0 + 8.0, 34.0));
        painter.rect(tb, 10.0, card_fill, Stroke::new(1.0, p.border), egui::StrokeKind::Inside);
        for (i, (icon, view, tip)) in views.iter().enumerate() {
            let br = Rect::from_min_size(tb.min + vec2(4.0 + i as f32 * 30.0, 4.0), vec2(28.0, 26.0));
            let r = ui.interact(br, ui.scope_id().with(("view", i)), Sense::click()).on_hover_text(*tip);
            if r.hovered() {
                painter.rect_filled(br, 6.0, p.hover);
            }
            painter.text(br.center(), Align2::CENTER_CENTER, *icon, FontId::proportional(15.0), text_col);
            if r.clicked() {
                if *view != "fit" {
                    self.cam.set_view(view);
                }
                self.need_fit = true;
                ui.ctx().request_repaint();
            }
        }
    }

    // ------------------------------------------------ status bar
    fn status(&mut self, ui: &mut egui::Ui) {
        let p = self.p;
        let pr = self.core.progress.lock().unwrap().clone();
        let (last, err) = {
            let s = self.core.session.lock().unwrap();
            (s.log.last().cloned().unwrap_or_default(), s.last_error.clone())
        };
        ui.horizontal(|ui| {
            if pr.running {
                ui.add(egui::ProgressBar::new(pr.fraction as f32).desired_width(240.0).desired_height(14.0).text(RichText::new(pr.message.clone()).small()).fill(p.accent));
                ui.ctx().request_repaint_after(std::time::Duration::from_millis(100));
            }
            match err {
                Some(e) => ui.label(RichText::new(format!("✖ {e}")).small().color(p.err)),
                None => ui.label(RichText::new(last).small().color(p.weak)),
            };
        });
    }

    // ------------------------------------------------ examples
    fn script(&mut self, steps: &[(&str, Value)]) {
        for (t, a) in steps {
            if !self.run(t, a.clone()) {
                return;
            }
        }
        self.need_fit = true;
        self.form = Form::None;
        self.tab = Tab::Solve;
        self.toast = Some(("Example loaded — press ⚡ Solve".into(), false, 4.0));
    }
    fn example_bracket(&mut self) {
        self.script(&[
            ("new_model", json!({})),
            ("add_box", json!({"name": "Bracket", "min": [0, 0, 0], "size": [120, 40, 10]})),
            ("add_cylinder", json!({"name": "Hole", "base": [80, 20, 0], "axis": "z", "radius": 8, "height": 10, "operation": "subtract"})),
            ("set_material", json!({"material": "Aluminum 6061-T6"})),
            ("set_mesh", json!({"element_size_mm": 2})),
            ("set_analysis", json!({"type": "static_structural"})),
            ("add_fixed_support", json!({"faces": ["Bracket.xmin"]})),
            ("add_force", json!({"faces": ["Bracket.xmax"], "vector": [0, 0, -500]})),
        ]);
    }
    fn example_heatsink(&mut self) {
        let mut steps = vec![("new_model", json!({})), ("add_box", json!({"name": "Base", "min": [0, 0, 0], "size": [60, 40, 5]}))];
        for i in 0..6 {
            steps.push(("add_box", json!({"name": format!("Fin{}", i + 1), "min": [2.5 + i as f64 * 10.0, 0, 5], "size": [5, 40, 30]})));
        }
        steps.extend([
            ("set_material", json!({"material": "Aluminum 6061-T6"})),
            ("set_mesh", json!({"element_size_mm": 1.25})),
            ("set_analysis", json!({"type": "steady_thermal"})),
            ("add_heat_flux", json!({"faces": ["Base.zmin"], "value": 20000})),
            ("add_convection", json!({"faces": ["all"], "coefficient": 25, "ambient": 22})),
        ]);
        self.script(&steps);
    }
    fn example_fork(&mut self) {
        self.script(&[
            ("new_model", json!({})),
            ("add_box", json!({"name": "Handle", "min": [-3, -3, 0], "size": [6, 6, 40]})),
            ("add_box", json!({"name": "Yoke", "min": [-12, -3, 40], "size": [24, 6, 6]})),
            ("add_box", json!({"name": "TineA", "min": [-12, -3, 46], "size": [6, 6, 80]})),
            ("add_box", json!({"name": "TineB", "min": [6, -3, 46], "size": [6, 6, 80]})),
            ("set_material", json!({"material": "Stainless Steel 304"})),
            ("set_mesh", json!({"element_size_mm": 1.5})),
            ("set_analysis", json!({"type": "modal", "modes": 6})),
            ("add_fixed_support", json!({"faces": ["Handle.zmin"]})),
        ]);
    }
}

impl eframe::App for SimApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.sync_theme(ui.ctx());
        let p = self.p;
        let bar = |fill: Color32| egui::Frame::new().fill(fill).inner_margin(egui::Margin::symmetric(12, 6)).stroke(Stroke::new(1.0, p.border));
        egui::Panel::top("header").frame(bar(p.ribbon)).show(ui, |ui| self.header(ui));
        egui::Panel::top("ribbon").frame(bar(p.ribbon).inner_margin(egui::Margin { left: 12, right: 12, top: 8, bottom: 6 })).show(ui, |ui| self.ribbon(ui));
        egui::Panel::bottom("status").frame(bar(p.panel).inner_margin(egui::Margin::symmetric(12, 4))).show(ui, |ui| self.status(ui));
        egui::Panel::left("browser").frame(bar(p.panel).inner_margin(10)).resizable(true).default_size(260.0).show(ui, |ui| self.browser(ui));
        egui::Panel::right("props").frame(bar(p.panel).inner_margin(10)).resizable(true).default_size(290.0).show(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| self.properties(ui));
        });
        egui::CentralPanel::default().frame(egui::Frame::new()).show(ui, |ui| self.viewport(ui));

        if let Some((msg, err, ttl)) = &mut self.toast {
            *ttl -= (ui.input(|i| i.stable_dt) as f64).min(0.1);
            let ctx = ui.ctx().clone();
            let (stripe, msg, err) = (if *err { p.err } else { p.ok }, msg.clone(), *err);
            egui::Area::new(egui::Id::new("toast")).anchor(Align2::CENTER_BOTTOM, vec2(0.0, -44.0)).show(&ctx, |ui| {
                egui::Frame::new().fill(p.card).stroke(Stroke::new(1.0, p.border)).corner_radius(10).inner_margin(egui::Margin { left: 14, right: 14, top: 9, bottom: 9 }).shadow(egui::epaint::Shadow { offset: [0, 4], blur: 16, spread: 0, color: Color32::from_black_alpha(50) }).show(ui, |ui| {
                    ui.set_max_width(620.0);
                    ui.horizontal(|ui| {
                        ui.label(RichText::new(if err { "✖" } else { "✔" }).color(stripe).strong());
                        ui.label(RichText::new(msg).color(p.text));
                    });
                });
            });
            ui.ctx().request_repaint_after(std::time::Duration::from_millis(100));
            if *ttl <= 0.0 {
                self.toast = None;
            }
        }
    }
}

pub fn run(core: Shared, open: Option<String>) -> eframe::Result<()> {
    let bridge = match crate::bridge::serve(core.clone()) {
        Ok(s) => s,
        Err(e) => e,
    };
    if let Some(p) = open {
        let _ = core.call("open_project", &json!({"path": p}));
    }
    let opts = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_title("SimCraft").with_app_id("simcraft").with_inner_size([1440.0, 880.0]).with_min_inner_size([960.0, 600.0]),
        ..Default::default()
    };
    eframe::run_native("SimCraft", opts, Box::new(move |cc| Ok(Box::new(SimApp::new(cc, core, bridge)))))
}
