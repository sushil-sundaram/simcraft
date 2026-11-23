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
    Optimize,
}

#[derive(PartialEq, Clone, Copy, Debug)]
enum LoadKind {
    Fixed,
    Displacement,
    CompressionOnly,
    Symmetry,
    Force,
    Pressure,
    Gravity,
    Temperature,
    HeatFlux,
    Convection,
    HeatGen,
    Voltage,
    Current,
    Coil,
    Magnet,
    Inlet,
    Outlet,
    Port,
    MovingWall,
}
impl LoadKind {
    const STRUCTURAL: [LoadKind; 7] = [LoadKind::Fixed, LoadKind::Displacement, LoadKind::CompressionOnly, LoadKind::Symmetry, LoadKind::Force, LoadKind::Pressure, LoadKind::Gravity];
    const THERMAL: [LoadKind; 4] = [LoadKind::Temperature, LoadKind::HeatFlux, LoadKind::Convection, LoadKind::HeatGen];
    const ELECTRO: [LoadKind; 5] = [LoadKind::Voltage, LoadKind::Current, LoadKind::Coil, LoadKind::Magnet, LoadKind::Port];
    const FLUID: [LoadKind; 5] = [LoadKind::Inlet, LoadKind::Outlet, LoadKind::Symmetry, LoadKind::MovingWall, LoadKind::Gravity];
    fn label(self) -> &'static str {
        match self {
            LoadKind::Fixed => "Fixed",
            LoadKind::Displacement => "Displacement",
            LoadKind::CompressionOnly => "Contact",
            LoadKind::Symmetry => "Symmetry",
            LoadKind::Force => "Force",
            LoadKind::Pressure => "Pressure",
            LoadKind::Gravity => "Gravity",
            LoadKind::Temperature => "Temperature",
            LoadKind::HeatFlux => "Heat Flux",
            LoadKind::Convection => "Convection",
            LoadKind::HeatGen => "Heat Gen.",
            LoadKind::Voltage => "Voltage",
            LoadKind::Current => "Current",
            LoadKind::Coil => "Coil",
            LoadKind::Magnet => "Magnet",
            LoadKind::Inlet => "Inlet",
            LoadKind::Outlet => "Outlet",
            LoadKind::Port => "RF Port",
            LoadKind::MovingWall => "Moving Wall",
        }
    }
    fn title(self) -> &'static str {
        self.sample().label()
    }
    fn icon(self) -> &'static str {
        match self {
            LoadKind::Fixed => "📌",
            LoadKind::Displacement => "↔",
            LoadKind::CompressionOnly => "⏏",
            LoadKind::Symmetry => "◫",
            LoadKind::Force => "➡",
            LoadKind::Pressure => "⬇",
            LoadKind::Gravity => "🌍",
            LoadKind::Temperature => "🌡",
            LoadKind::HeatFlux => "☀",
            LoadKind::Convection => "💨",
            LoadKind::HeatGen => "🔥",
            LoadKind::Voltage => "⚡",
            LoadKind::Current => "🔌",
            LoadKind::Coil => "🌀",
            LoadKind::Magnet => "🧲",
            LoadKind::Inlet => "⇥",
            LoadKind::Outlet => "⇤",
            LoadKind::Port => "📡",
            LoadKind::MovingWall => "⏩",
        }
    }
    fn sample(self) -> Load {
        let f = vec![];
        match self {
            LoadKind::Fixed => Load::FixedSupport { faces: f },
            LoadKind::Displacement => Load::Displacement { faces: f, x: None, y: None, z: None },
            LoadKind::CompressionOnly => Load::CompressionOnly { faces: f },
            LoadKind::Symmetry => Load::Symmetry { faces: f },
            LoadKind::Force => Load::Force { faces: f, vector: [0.0; 3] },
            LoadKind::Pressure => Load::Pressure { faces: f, value_mpa: 0.0 },
            LoadKind::Gravity => Load::Gravity { acceleration: [0.0; 3] },
            LoadKind::Temperature => Load::Temperature { faces: f, value: 0.0 },
            LoadKind::HeatFlux => Load::HeatFlux { faces: f, value: 0.0 },
            LoadKind::Convection => Load::Convection { faces: f, coefficient: 0.0, ambient: 0.0 },
            LoadKind::HeatGen => Load::HeatGeneration { value: 0.0, body: None, power: None },
            LoadKind::Voltage => Load::Voltage { faces: f, value: 0.0 },
            LoadKind::Current => Load::Current { faces: f, value: 0.0 },
            LoadKind::Coil => Load::Coil { body: String::new(), current_density: 0.0, axis: Default::default(), mode: Default::default(), center: None },
            LoadKind::Magnet => Load::Magnet { body: String::new(), remanence: [0.0; 3] },
            LoadKind::Inlet => Load::Inlet { faces: f, velocity: 0.0, temperature: None },
            LoadKind::Outlet => Load::Outlet { faces: f, pressure: 0.0 },
            LoadKind::MovingWall => Load::MovingWall { faces: f, velocity: [0.0; 3] },
            LoadKind::Port => Load::Port { faces: f, power: 0.0, polarization: Default::default(), profile: "te10".into() },
        }
    }
    fn color(self) -> Color32 {
        let c = match self {
            LoadKind::Gravity => [70, 150, 90],
            LoadKind::HeatGen => [235, 90, 40],
            k => render::load_color(&k.sample()),
        };
        Color32::from_rgb(c[0], c[1], c[2])
    }
    fn needs_faces(self) -> bool {
        !matches!(self, LoadKind::Gravity | LoadKind::HeatGen | LoadKind::Coil | LoadKind::Magnet)
    }
    fn needs_body(self) -> bool {
        matches!(self, LoadKind::Coil | LoadKind::Magnet)
    }
    fn applies(self, kind: AnalysisType) -> bool {
        self.sample().applies_to(kind)
    }
}

#[derive(PartialEq, Clone, Copy)]
enum Form {
    None,
    Box,
    Cylinder,
    Sphere,
    Enclosure,
    Study,
    Optimize,
    Topology,
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
    body: String,
    coil_axis: usize,
    coil_axial: bool,
    remanence: [f64; 3],
    velocity: f64,
    outlet_p: f64,
    padding: f64,
    fluid: String,
    /// Body for heat generation ("" = whole model).
    heat_body: String,
    /// Spectrum / PSD table being edited ("Hz value" per line) and the model table it came from.
    table_text: String,
    table_src: Vec<[f64; 2]>,
    fat_open: bool,
    fat_loading: String,
    fat_ratio: f64,
    fat_correction: String,
    fat_scale: f64,
    fat_life: f64,
    fat_exposure_h: f64,
    port_power: f64,
    heat_watts: bool,
    heat_power: f64,
    wall_v: [f64; 3],
    port_pol: usize,
    port_te10: bool,
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
            body: String::new(),
            coil_axis: 2,
            coil_axial: false,
            remanence: [0.0, 0.0, 1.3],
            velocity: 1.0,
            outlet_p: 0.0,
            padding: 0.5,
            fluid: "Air".into(),
            heat_body: String::new(),
            table_text: String::new(),
            table_src: Vec::new(),
            fat_open: false,
            fat_loading: "fully_reversed".into(),
            fat_ratio: 0.1,
            fat_correction: "goodman".into(),
            fat_scale: 1.0,
            fat_life: 1e6,
            fat_exposure_h: 1.0,
            port_power: 800.0,
            heat_watts: true,
            heat_power: 5.0,
            wall_v: [0.1, 0.0, 0.0],
            port_pol: 2,
            port_te10: true,
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
    show_mcp: bool,
    planner: Planner,
    opt: OptUi,
    copied: Option<(&'static str, f64)>,
}

/// State of the OPTIMIZE tab forms and the study results window.
#[derive(Default)]
struct OptUi {
    /// (target, min, max, steps)
    params: Vec<(String, f64, f64, usize)>,
    responses: Vec<String>,
    objective: String,
    maximize: bool,
    /// (response, is "<=", value)
    constraints: Vec<(String, bool, f64)>,
    max_evals: usize,
    vf: f64,
    iters: usize,
    show_results: bool,
    job: Option<Arc<std::sync::Mutex<Option<Result<(), String>>>>>,
    error: Option<String>,
    /// Signature of the last study shown, so new results (from the GUI or an AI agent) pop up.
    seen: String,
}

/// State of the local-model simulation planner window.
#[derive(Default)]
struct Planner {
    open: bool,
    goal: String,
    models: Option<Result<Vec<String>, String>>,
    model: String,
    job: Option<Arc<std::sync::Mutex<Option<Result<(), String>>>>>,
    error: Option<String>,
    copied_at: f64,
}

// ---------------------------------------------------------------- widgets

/// The SimCraft logo (assets/simcraft-64.png), drawn as an image.
fn logo(ui: &mut egui::Ui, size: f32) {
    let id = egui::Id::new("simcraft-logo-tex");
    let tex = ui.ctx().data_mut(|d| d.get_temp::<egui::TextureHandle>(id)).unwrap_or_else(|| {
        let img = decode_png(include_bytes!("../assets/simcraft-64.png")).unwrap_or_else(|| egui::ColorImage::filled([1, 1], Color32::TRANSPARENT));
        let t = ui.ctx().load_texture("simcraft-logo", img, egui::TextureOptions::LINEAR);
        ui.ctx().data_mut(|d| d.insert_temp(id, t.clone()));
        t
    });
    let (r, _) = ui.allocate_exact_size(vec2(size, size), Sense::hover());
    ui.painter().image(tex.id(), r, Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)), Color32::WHITE);
}

fn decode_png(bytes: &[u8]) -> Option<egui::ColorImage> {
    let mut dec = png::Decoder::new(std::io::Cursor::new(bytes));
    dec.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = dec.read_info().ok()?;
    let mut buf = vec![0; reader.output_buffer_size()?];
    let info = reader.next_frame(&mut buf).ok()?;
    let (w, h) = (info.width as usize, info.height as usize);
    let rgba: Vec<u8> = match info.color_type {
        png::ColorType::Rgba => buf[..w * h * 4].to_vec(),
        png::ColorType::Rgb => buf[..w * h * 3].chunks(3).flat_map(|c| [c[0], c[1], c[2], 255]).collect(),
        png::ColorType::GrayscaleAlpha => buf[..w * h * 2].chunks(2).flat_map(|c| [c[0], c[0], c[0], c[1]]).collect(),
        _ => buf[..w * h].iter().flat_map(|&g| [g, g, g, 255]).collect(),
    };
    Some(egui::ColorImage::from_rgba_unmultiplied([w, h], &rgba))
}

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
            show_mcp: std::env::var_os("SIMCRAFT_OPEN_CONNECT").is_some(),
            opt: OptUi { max_evals: 30, vf: 0.4, iters: 40, objective: "mass".into(), ..Default::default() },
            planner: Planner { open: std::env::var_os("SIMCRAFT_OPEN_PLANNER").is_some(), ..Default::default() },
            copied: None,
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
            logo(ui, 24.0);
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
                    if ui.button("🏛  Column buckling").clicked() {
                        ui.close();
                        self.example_buckling();
                    }
                    ui.separator();
                    if ui.button("⚡  Parallel-plate capacitor (electrostatic)").clicked() {
                        ui.close();
                        self.example_capacitor();
                    }
                    if ui.button("🔌  Bus bar Joule heating (thermal-electric)").clicked() {
                        ui.close();
                        self.example_busbar();
                    }
                    if ui.button("🌀  Coil with iron core (magnetostatic)").clicked() {
                        ui.close();
                        self.example_coil();
                    }
                    if ui.button("📡  Microwave oven heating (full-wave EM)").clicked() {
                        ui.close();
                        self.example_microwave();
                    }
                    ui.separator();
                    if ui.button("🔥  Heat sink in a duct (conjugate heat transfer)").clicked() {
                        ui.close();
                        self.example_heatsink_cht();
                    }
                    if ui.button("💨  Flow past a cylinder (CFD)").clicked() {
                        ui.close();
                        self.example_cfd();
                    }
                });
                if ui.button("Connect AI").on_hover_text("Control SimCraft from Claude Code, Antigravity or any MCP client").clicked() {
                    self.show_mcp = true;
                }
                if ui.button("✨ Planner").on_hover_text("Describe what you want to simulate and get a to-do list from a local AI model").clicked() {
                    self.planner.open = true;
                }
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
                if ui.add(egui::Label::new(RichText::new(t).size(11.5).color(c)).sense(Sense::click())).on_hover_text("Connect an AI agent").clicked() {
                    self.show_mcp = true;
                }
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

    fn start_job(&mut self, tool: &str, args: Value) {
        let slot = Arc::new(std::sync::Mutex::new(None));
        let (core, out, tool) = (self.core.clone(), slot.clone(), tool.to_string());
        std::thread::spawn(move || {
            let r = core.call(&tool, &args).map(|_| ()).map_err(|e| format!("{e:#}"));
            *out.lock().unwrap() = Some(r);
        });
        self.opt.job = Some(slot);
        self.opt.error = None;
    }

    /// Forms for parameter studies, optimization and topology optimization.
    fn opt_form(&mut self, ui: &mut egui::Ui, form: Form) {
        let p = self.p;
        let model = self.core.session.lock().unwrap().model.clone();
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
        if form != Form::Topology {
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
                        ui.label("Max evaluations");
                        ui.add(egui::DragValue::new(&mut self.opt.max_evals).range(5..=400));
                    });
                });
                let ready = !busy && !self.opt.params.is_empty() && !self.opt.objective.is_empty();
                if ui.add_enabled(ready, primary(&p, "Optimize")).clicked() {
                    let params: Vec<Value> = self.opt.params.iter().map(|(t, lo, hi, _)| json!({"target": t, "min": lo, "max": hi})).collect();
                    let cons: Vec<Value> = self.opt.constraints.iter().map(|(r, le, v)| json!({"response": r, "op": if *le { "<=" } else { ">=" }, "value": v})).collect();
                    self.start_job(
                        "optimize_design",
                        json!({"parameters": params, "objective": {"response": self.opt.objective, "goal": if self.opt.maximize { "maximize" } else { "minimize" }}, "constraints": cons, "max_evaluations": self.opt.max_evals}),
                    );
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

    /// Table of study runs with the best design highlighted.
    fn study_window(&mut self, ctx: &egui::Context) {
        let finished = self.opt.job.as_ref().map(|job| job.lock().unwrap().take());
        match finished {
            Some(Some(r)) => {
                self.opt.error = r.err();
                self.opt.show_results = self.opt.error.is_none();
                self.opt.job = None;
            }
            Some(None) => ctx.request_repaint_after(std::time::Duration::from_millis(250)),
            None => {}
        }
        let Some(st) = self.core.session.lock().unwrap().study.clone() else { return };
        let sig = format!("{}|{}|{}", st.kind, st.message, st.seconds);
        if sig != self.opt.seen {
            self.opt.seen = sig;
            self.opt.show_results = true;
        }
        if !self.opt.show_results {
            return;
        }
        let p = self.p;
        let mut open = true;
        let mut actions: Vec<(String, Value)> = Vec::new();
        let title = match st.kind.as_str() {
            "optimization" => "🎯 Optimization results",
            "topology" => "🦴 Topology optimization",
            _ => "📊 Parameter study",
        };
        egui::Window::new(RichText::new(title).strong())
            .open(&mut open)
            .resizable(true)
            .default_size(vec2(720.0, 420.0))
            .frame(egui::Frame::window(&ctx.global_style()).fill(p.card).inner_margin(14))
            .show(ctx, |ui| {
                ui.label(RichText::new(&st.message).strong());
                ui.label(RichText::new(format!("{:.1} s", st.seconds)).small().color(p.weak));
                ui.add_space(6.0);
                if st.kind == "topology" {
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
                                let mat = self.core.session.lock().unwrap().model.bodies.iter().find(|b| b.role == crate::geometry::Role::Solid).map(|b| b.material.clone()).unwrap_or_default();
                                actions.push(("new_model".into(), json!({})));
                                actions.push(("import_stl".into(), json!({"path": path, "name": "Optimized"})));
                                if !mat.is_empty() {
                                    actions.push(("set_material".into(), json!({"material": mat})));
                                }
                            }
                            ui.label(RichText::new(path).small().color(p.weak));
                        }
                    });
                    return;
                }
                egui::ScrollArea::both().auto_shrink([false, false]).show(ui, |ui| {
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
                                if st.kind == "optimization" && !run.feasible {
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
                });
            });
        for (t, a) in actions {
            self.run(&t, a);
            if t == "import_stl" {
                self.need_fit = true;
            }
        }
        if !open {
            self.opt.show_results = false;
        }
    }

    /// Simulation planner: goal → to-do list from a local model (Ollama).
    fn planner_window(&mut self, ctx: &egui::Context) {
        if !self.planner.open {
            return;
        }
        let p = self.p;
        if self.planner.models.is_none() {
            self.planner.models = Some(crate::assistant::list_models().map_err(|e| format!("{e:#}")));
            if let Some(Ok(m)) = &self.planner.models {
                self.planner.model = crate::assistant::default_model(m).unwrap_or_default();
            }
        }
        // finished job?
        let mut finished = None;
        if let Some(job) = &self.planner.job {
            if let Some(r) = job.lock().unwrap().take() {
                finished = Some(r);
            }
        }
        if let Some(r) = finished {
            self.planner.job = None;
            self.planner.error = r.err();
        }
        let busy = self.planner.job.is_some();
        if busy {
            ctx.request_repaint_after(std::time::Duration::from_millis(200));
        }
        let plan = self.core.session.lock().unwrap().plan.clone();
        let now = ctx.input(|i| i.time);
        let mut open = true;
        let mut actions: Vec<(String, Value)> = Vec::new();
        egui::Window::new(RichText::new("✨ Simulation Planner").strong())
            .open(&mut open)
            .collapsible(false)
            .resizable(true)
            .default_size(vec2(620.0, 640.0))
            .default_pos(ctx.content_rect().center() - vec2(310.0, 320.0))
            .frame(egui::Frame::window(&ctx.global_style()).fill(p.card).inner_margin(16))
            .show(ctx, |ui| {
                ui.label(RichText::new("Describe what you want to simulate. A local AI model (running on this computer through Ollama) drafts a to-do list and flags anything SimCraft cannot do yet.").color(p.weak));
                ui.add_space(8.0);
                ui.add(egui::TextEdit::multiline(&mut self.planner.goal).hint_text("e.g. simulate EMI on my PCB · how hot does food get in a microwave · will this bracket bend under 2 kN").desired_rows(3).desired_width(f32::INFINITY));
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
                                *out.lock().unwrap() = Some(r);
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
                ui.add_space(10.0);
                let Some(plan) = plan else {
                    return;
                };
                ui.separator();
                egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
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
                    ui.add_space(8.0);
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
                            ui.ctx().copy_text(crate::assistant::to_markdown(&plan));
                            self.planner.copied_at = now;
                        }
                        ui.label(RichText::new(format!("Drafted by {} — check the steps; small local models can be wrong.", plan.model)).small().italics().color(p.weak));
                    });
                });
            });
        for (t, a) in actions {
            self.run(&t, a);
        }
        if !open {
            self.planner.open = false;
        }
    }

    /// "Connect an AI agent" dialog: bridge status plus copy-ready setup for MCP clients.
    fn mcp_dialog(&mut self, ctx: &egui::Context) {
        if !self.show_mcp {
            return;
        }
        let p = self.p;
        let exe = std::env::current_exe().map(|x| x.to_string_lossy().to_string()).unwrap_or("simcraft".into());
        let cmd = format!("claude mcp add --scope user simcraft -- {exe} mcp");
        let js = format!("{{\n  \"mcpServers\": {{\n    \"simcraft\": {{\n      \"command\": \"{exe}\",\n      \"args\": [\"mcp\"]\n    }}\n  }}\n}}");
        let ok = self.bridge.starts_with("MCP bridge listening");
        let now = ctx.input(|i| i.time);
        let mut open = true;
        let mut copy: Option<(&'static str, String)> = None;
        egui::Window::new(RichText::new("Connect an AI agent").strong())
            .open(&mut open)
            .collapsible(false)
            .resizable(false)
            .default_width(560.0)
            .anchor(Align2::CENTER_CENTER, vec2(0.0, 0.0))
            .frame(egui::Frame::window(&ctx.global_style()).fill(p.card).inner_margin(18))
            .show(ctx, |ui| {
                ui.set_width(560.0);
                ui.label(RichText::new("Let Claude Code, Antigravity or any MCP client build models, run solves and look at results in this window.").color(p.weak));
                ui.add_space(10.0);
                // status
                egui::Frame::new().fill(if ok { p.ok.gamma_multiply(0.12) } else { p.warn.gamma_multiply(0.12) }).corner_radius(8).inner_margin(egui::Margin::symmetric(12, 8)).show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    ui.horizontal(|ui| {
                        ui.label(RichText::new(if ok { "●" } else { "○" }).color(if ok { p.ok } else { p.warn }));
                        ui.label(RichText::new(if ok { "Ready — agents connected to SimCraft drive this window live" } else { "Bridge offline — another SimCraft window may already be using the port" }).strong());
                    });
                    ui.label(RichText::new(&self.bridge).small().color(p.weak));
                });
                ui.add_space(14.0);
                let mut block = |ui: &mut egui::Ui, step: &str, title: &str, hint: &str, text: &str, key: &'static str, multiline: bool| {
                    ui.horizontal(|ui| {
                        ui.label(RichText::new(step).strong().color(p.accent));
                        ui.label(RichText::new(title).strong());
                    });
                    ui.label(RichText::new(hint).small().color(p.weak));
                    ui.add_space(4.0);
                    egui::Frame::new().fill(if p.dark { Color32::from_rgb(22, 24, 28) } else { Color32::from_rgb(244, 245, 248) }).stroke(Stroke::new(1.0, p.border)).corner_radius(8).inner_margin(egui::Margin::symmetric(10, 8)).show(ui, |ui| {
                        ui.set_width(ui.available_width());
                        ui.horizontal_top(|ui| {
                            let copied = self.copied.is_some_and(|(k, t)| k == key && now - t < 1.8);
                            let label = if copied { "✔ Copied" } else { "📋 Copy" };
                            let btn = ui.with_layout(egui::Layout::right_to_left(egui::Align::Min), |ui| {
                                let b = ui.add(egui::Button::new(RichText::new(label).small()).min_size(vec2(78.0, 24.0)));
                                ui.with_layout(egui::Layout::left_to_right(egui::Align::Min), |ui| {
                                    let _ = multiline;
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
                    ui.add_space(14.0);
                };
                block(ui, "1", "Claude Code", "Run once in a terminal; SimCraft is then available in every Claude Code session.", &cmd, "claude", false);
                block(ui, "2", "Antigravity, Cursor and other MCP clients", "Paste into the client's MCP config (Antigravity: Agent panel ▸ ⋯ ▸ MCP Servers ▸ Manage ▸ View raw config).", &js, "json", true);
                ui.label(RichText::new("Try asking: “In SimCraft, make an aluminium bracket with a hole, fix one end, load the other and show me the stress.”").small().italics().color(p.weak));
            });
        if let Some((k, t)) = copy {
            ctx.copy_text(t);
            self.copied = Some((k, now));
            ctx.request_repaint_after(std::time::Duration::from_millis(1900));
        }
        if !open {
            self.show_mcp = false;
        }
    }

    // ------------------------------------------------ ribbon
    fn ribbon(&mut self, ui: &mut egui::Ui) {
        let p = self.p;
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 18.0;
            for (t, name) in [(Tab::Geometry, "GEOMETRY"), (Tab::Material, "MATERIAL"), (Tab::Setup, "SETUP"), (Tab::Mesh, "MESH"), (Tab::Solve, "SOLVE"), (Tab::Results, "RESULTS"), (Tab::Optimize, "OPTIMIZE")] {
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
                    if tool(ui, &p, "🟩", "KiCad PCB", Some(Color32::from_rgb(70, 160, 90)), true) {
                        self.import_kicad();
                    }
                    if tool(ui, &p, "⬚", "Enclosure", Some(Color32::from_rgb(90, 170, 230)), true) {
                        self.form = Form::Enclosure;
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
                if kind.requires_enclosure() && !self.core.session.lock().unwrap().model.has_enclosure() {
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
                        let st = self.core.session.lock().unwrap().model.analysis.clone();
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
                            let mut m = self.core.session.lock().unwrap().model.analysis.modes as u32;
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
                        if kind == AnalysisType::StaticStructural {
                            let mut fit = st.boundary_fit;
                            if ui.checkbox(&mut fit, "Fit curved surfaces").on_hover_text("Moves the surface nodes onto the true holes, fillets and off-grid faces (with a fitted surface layer) for accurate peak stresses. Off = plain voxels.").changed() {
                                self.run("set_analysis", json!({"boundary_fit": fit}));
                            }
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
                if matches!(kind, AnalysisType::ResponseSpectrum | AnalysisType::RandomVibration) {
                    let (st_table, spectrum) = (self.core.session.lock().unwrap().model.analysis.table.clone(), kind == AnalysisType::ResponseSpectrum);
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
            Tab::Optimize => {
                let kind = self.core.session.lock().unwrap().model.analysis.kind;
                group(ui, &p, "STUDIES", |ui| {
                    if tool(ui, &p, "📊", "Parameters", None, true) {
                        self.form = Form::Study;
                    }
                    if tool(ui, &p, "🎯", "Optimize", None, true) {
                        self.form = Form::Optimize;
                    }
                });
                group(ui, &p, "SHAPE", |ui| {
                    if tool(ui, &p, "🦴", "Topology", Some(Color32::from_rgb(120, 170, 255)), kind == AnalysisType::StaticStructural) {
                        self.form = Form::Topology;
                    }
                });
                group(ui, &p, "RESULTS", |ui| {
                    let has = self.core.session.lock().unwrap().study.is_some();
                    if tool(ui, &p, "📋", "Study results", Some(p.weak), has) {
                        self.opt.show_results = true;
                    }
                });
            }
        });
    }

    fn fatigue_window(&mut self, ctx: &egui::Context) {
        if !self.f.fat_open {
            return;
        }
        let kind = self.core.session.lock().unwrap().results.as_ref().map(|r| r.analysis);
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
        let groups: [(&str, &[(Field, &str, &str)]); 7] = [
            ("DEFORMATION", &[(Field::TotalDeformation, "📐", "Total"), (Field::DeformationX, "X", "Dir. X"), (Field::DeformationY, "Y", "Dir. Y"), (Field::DeformationZ, "Z", "Dir. Z")]),
            ("STRESS", &[(Field::VonMises, "🔶", "Von Mises"), (Field::MaxPrincipal, "⬆", "Max Princ."), (Field::MinPrincipal, "⬇", "Min Princ."), (Field::SafetyFactor, "🛡", "Safety"), (Field::PlasticStrain, "〰", "Plastic ε")]),
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
                if self.core.session.lock().unwrap().mesh.as_ref().is_some_and(|m| m.with_enclosure) || disp.section.is_some() {
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

    fn save_as(&mut self) {
        if let Some(p) = rfd::FileDialog::new().add_filter("SimCraft", &["simcraft"]).set_file_name("model.simcraft").save_file() {
            self.run("save_project", json!({"path": p.to_string_lossy()}));
        }
    }

    fn import_kicad(&mut self) {
        if let Some(p) = rfd::FileDialog::new().add_filter("KiCad board", &["kicad_pcb"]).pick_file() {
            let path = p.to_string_lossy().to_string();
            match self.core.call("import_kicad", &json!({"path": path})) {
                Ok(o) => {
                    let v: Value = serde_json::from_str(&o.text).unwrap_or(Value::Null);
                    let parts = v["components"].as_array().map(|a| a.len()).unwrap_or(0);
                    let warn = v["warnings"].as_array().and_then(|w| w.first()).and_then(|w| w.as_str()).map(|w| format!(" ⚠ {w}")).unwrap_or_default();
                    self.toast = Some((format!("Imported {} × {} mm board with {parts} components. Add Heat Generation (W) on the parts and a convection or airflow boundary.{warn}", v["size_mm"][0], v["size_mm"][1]), !warn.is_empty(), 8.0));
                    // the copper must be resolved through the thickness: suggest ≤ half the board
                    let t = v["thickness_mm"].as_f64().unwrap_or(1.6);
                    let cur = self.core.session.lock().unwrap().model.element_size;
                    if cur.is_none_or(|h| h > t / 2.0) {
                        self.run("set_mesh", json!({"element_size_mm": (t / 2.0 * 10.0).floor() / 10.0}));
                        self.f.mesh_size = (t / 2.0 * 10.0).floor() / 10.0;
                    }
                }
                Err(e) => self.toast = Some((format!("{e:#}"), true, 8.0)),
            }
            self.need_fit = true;
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
            Form::Study | Form::Optimize | Form::Topology => {
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
                    ui.horizontal(|ui| {
                        ui.label("Padding");
                        ui.add(egui::Slider::new(&mut self.f.padding, 0.1..=3.0).suffix("× part size"));
                    });
                });
                ui.add_space(8.0);
                if ui.add(primary(&p, "Create")).clicked() && self.run("add_enclosure", json!({"material": self.f.fluid, "padding": self.f.padding})) {
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
            let bodies: Vec<String> = self.core.session.lock().unwrap().model.bodies.iter().filter(|b| b.role == crate::geometry::Role::Solid && b.op == Op::Add).map(|b| b.name.clone()).collect();
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
                    let bodies: Vec<String> = self.core.session.lock().unwrap().model.bodies.iter().filter(|b| b.role == crate::geometry::Role::Solid && b.op == Op::Add).map(|b| b.name.clone()).collect();
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
                    if self.core.session.lock().unwrap().model.analysis.kind == AnalysisType::ConjugateHeatTransfer {
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
                    if self.core.session.lock().unwrap().model.analysis.kind == AnalysisType::ConjugateHeatTransfer {
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
                if !r.modes.is_empty() {
                    title = r.step_label(display.mode);
                    if f.is_deformation() {
                        unit = "normalised".into();
                    }
                } else if r.n_steps() > 1 {
                    title = format!("{title} · {}", r.step_label(display.mode));
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
                    let c = render::BANDS[if f.reversed_scale() { b } else { 8 - b }];
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
        let tbw = views.len() as f32 * 30.0 + 8.0;
        // top-right normally; bottom-right when the viewport is too narrow to share the top with the legend
        let tby = if rect.width() < 640.0 { rect.bottom() - 48.0 } else { rect.top() + 14.0 };
        let tb = Rect::from_min_size(pos2(rect.right() - 14.0 - tbw, tby), vec2(tbw, 34.0));
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
    fn example_buckling(&mut self) {
        self.script(&[
            ("new_model", json!({})),
            ("add_box", json!({"name": "Column", "min": [0, 0, 0], "size": [10, 10, 200]})),
            ("set_mesh", json!({"element_size_mm": 2})),
            ("set_analysis", json!({"type": "linear_buckling", "modes": 3})),
            ("add_fixed_support", json!({"faces": ["Column.zmin"]})),
            ("add_force", json!({"faces": ["Column.zmax"], "vector": [0, 0, -1000]})),
        ]);
    }
    fn example_capacitor(&mut self) {
        self.script(&[
            ("new_model", json!({})),
            ("add_box", json!({"name": "TopPlate", "min": [0, 0, 6], "size": [40, 40, 2], "material": "Copper"})),
            ("add_box", json!({"name": "BottomPlate", "min": [0, 0, 0], "size": [40, 40, 2], "material": "Copper"})),
            ("add_box", json!({"name": "Dielectric", "min": [0, 0, 2], "size": [40, 40, 4], "material": "FR-4"})),
            ("add_enclosure", json!({"material": "Air", "padding": 0.4})),
            ("set_analysis", json!({"type": "electrostatic"})),
            ("set_mesh", json!({"element_size_mm": 1})),
            ("add_voltage", json!({"faces": ["TopPlate"], "value": 100})),
            ("add_voltage", json!({"faces": ["BottomPlate"], "value": 0})),
        ]);
    }
    fn example_busbar(&mut self) {
        self.script(&[
            ("new_model", json!({})),
            ("add_box", json!({"name": "BusBar", "min": [0, 0, 0], "size": [150, 20, 5], "material": "Copper"})),
            ("add_cylinder", json!({"name": "BoltA", "base": [12, 10, 0], "radius": 4, "height": 5, "operation": "subtract"})),
            ("add_cylinder", json!({"name": "BoltB", "base": [138, 10, 0], "radius": 4, "height": 5, "operation": "subtract"})),
            ("set_mesh", json!({"element_size_mm": 1})),
            ("set_analysis", json!({"type": "thermal_electric"})),
            ("add_current", json!({"faces": ["BoltA.side"], "value": 400})),
            ("add_voltage", json!({"faces": ["BoltB.side"], "value": 0})),
            ("add_convection", json!({"faces": ["all"], "coefficient": 15, "ambient": 25})),
        ]);
    }
    fn example_coil(&mut self) {
        self.script(&[
            ("new_model", json!({})),
            ("add_cylinder", json!({"name": "Coil", "base": [0, 0, -15], "radius": 22, "height": 30, "material": "Copper"})),
            ("add_cylinder", json!({"name": "Bore", "base": [0, 0, -15], "radius": 14, "height": 30, "operation": "subtract"})),
            ("add_cylinder", json!({"name": "Core", "base": [0, 0, -25], "radius": 10, "height": 50, "material": "Electrical Steel (M19)"})),
            ("add_enclosure", json!({"material": "Air", "padding": 0.6})),
            ("set_analysis", json!({"type": "magnetostatic"})),
            ("set_mesh", json!({"element_size_mm": 2.5})),
            ("add_coil", json!({"body": "Coil", "current_density": 3, "axis": "z", "mode": "circumferential"})),
        ]);
    }
    fn example_microwave(&mut self) {
        self.script(&[
            ("new_model", json!({})),
            ("add_box", json!({"name": "Tray", "min": [30, 30, 12], "size": [180, 180, 6], "material": "Microwave Glass Tray"})),
            ("add_box", json!({"name": "Food", "min": [80, 85, 18], "size": [80, 70, 40], "material": "Food (water-rich, 2.45 GHz)"})),
            ("add_enclosure", json!({"material": "Air", "min": [0, 0, 0], "size": [240, 240, 160]})),
            ("set_analysis", json!({"type": "microwave_heating", "frequency_ghz": 2.45, "em_boundary": "pec", "end_time": 30, "time_steps": 30, "initial_temperature": 5})),
            ("set_mesh", json!({"element_size_mm": 1.6})),
            ("add_port", json!({"faces": ["Enclosure.xmin"], "power": 800, "polarization": "z", "profile": "te10"})),
        ]);
    }
    fn example_heatsink_cht(&mut self) {
        let mut calls: Vec<(&str, Value)> = vec![
            ("new_model", json!({})),
            ("add_box", json!({"name": "Chip", "min": [12.5, 12.5, 0], "size": [15, 15, 2], "material": "Copper C101"})),
            ("add_box", json!({"name": "Base", "min": [0, 0, 2], "size": [40, 40, 4], "material": "Aluminum 6061-T6"})),
        ];
        let fins: Vec<(String, f64)> = (0..6).map(|i| (format!("Fin{}", i + 1), 1.0 + i as f64 * 7.2)).collect();
        for (name, y) in &fins {
            calls.push(("add_box", json!({"name": name, "min": [0, y, 6], "size": [40, 2, 25], "material": "Aluminum 6061-T6"})));
        }
        calls.extend([
            ("add_enclosure", json!({"material": "Air", "min": [-40, -10, 0], "size": [160, 60, 45]})),
            ("set_analysis", json!({"type": "conjugate_heat_transfer", "turbulence": "lvel", "reference_temperature": 25, "max_iterations": 6000})),
            ("set_mesh", json!({"element_size_mm": 2})),
            ("add_inlet", json!({"faces": ["Enclosure.xmin"], "velocity": 2.0, "temperature": 25})),
            ("add_outlet", json!({"faces": ["Enclosure.xmax"], "pressure": 0})),
            ("add_heat_generation", json!({"body": "Chip", "power": 10.0})),
        ]);
        self.script(&calls);
    }
    fn example_cfd(&mut self) {
        self.script(&[
            ("new_model", json!({})),
            ("add_cylinder", json!({"name": "Cylinder", "base": [0, 0, 0], "axis": "y", "radius": 10, "height": 20, "material": "Aluminum 6061-T6"})),
            ("add_enclosure", json!({"material": "Water", "padding_mm": [40, 120, 0, 0, 40, 40]})),
            ("set_analysis", json!({"type": "fluid", "max_iterations": 3000})),
            ("set_mesh", json!({"element_size_mm": 2})),
            ("add_inlet", json!({"faces": ["Enclosure.xmin"], "velocity": 0.002})),
            ("add_outlet", json!({"faces": ["Enclosure.xmax"], "pressure": 0})),
            ("add_symmetry", json!({"faces": ["Enclosure.ymin", "Enclosure.ymax", "Enclosure.zmin", "Enclosure.zmax"]})),
        ]);
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

        let ctx = ui.ctx().clone();
        self.mcp_dialog(&ctx);
        self.planner_window(&ctx);
        self.study_window(&ctx);
        self.fatigue_window(&ctx);
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
        viewport: {
            let mut v = egui::ViewportBuilder::default().with_title("SimCraft").with_app_id("simcraft").with_inner_size([1440.0, 880.0]).with_min_inner_size([960.0, 600.0]);
            if let Ok(icon) = eframe::icon_data::from_png_bytes(include_bytes!("../assets/simcraft-256.png")) {
                v = v.with_icon(Arc::new(icon));
            }
            v
        },
        ..Default::default()
    };
    eframe::run_native("SimCraft", opts, Box::new(move |cc| Ok(Box::new(SimApp::new(cc, core, bridge)))))
}
