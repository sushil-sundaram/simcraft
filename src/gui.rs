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
    Help,
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

/// Material fields editable in the app (the ones define_material accepts).
const MAT_KEYS: &[&str] = &[
    "youngs_modulus_gpa", "poisson_ratio", "density", "yield_strength_mpa", "ultimate_strength_mpa", "thermal_conductivity", "thermal_expansion", "specific_heat",
    "relative_permittivity", "electrical_conductivity", "relative_permeability", "loss_tangent", "fatigue_strength_coefficient_mpa", "fatigue_exponent", "endurance_limit_mpa",
];

/// Edit the fields of a JSON object in place: numbers, [x,y,z] vectors, text, face lists, switches.
/// Returns true when something changed. `mats` feeds the material picker.
fn json_editor(ui: &mut egui::Ui, v: &mut Value, mats: &[String], id: &str) -> bool {
    let mut changed = false;
    let Some(obj) = v.as_object_mut() else { return false };
    // identity fields first, then the rest alphabetically
    let mut keys: Vec<String> = obj.keys().cloned().collect();
    let rank = |k: &str| ["name", "material", "operation"].iter().position(|x| *x == k).unwrap_or(9);
    keys.sort_by(|a, b| rank(a).cmp(&rank(b)).then(a.cmp(b)));
    egui::Grid::new(id).num_columns(2).spacing(vec2(8.0, 4.0)).show(ui, |ui| {
        for k in &keys {
            let val = obj.get_mut(k).unwrap();
            if matches!(k.as_str(), "type" | "kind" | "path" | "id") || val.is_null() {
                continue;
            }
            ui.label(k.replace('_', " "));
            match val {
                Value::Number(n) => {
                    let mut x = n.as_f64().unwrap_or(0.0);
                    let sp = if x == 0.0 { 0.01 } else { x.abs() * 0.01 };
                    // significant digits, so small values like 1.2e-5 stay readable
                    if ui.add(egui::DragValue::new(&mut x).speed(sp).custom_formatter(|v, _| render::fmt_val(v))).changed() {
                        *val = json!(x);
                        changed = true;
                    }
                }
                Value::Array(a) if a.len() == 3 && a.iter().all(|e| e.is_number()) => {
                    ui.horizontal(|ui| {
                        for e in a.iter_mut() {
                            let mut x = e.as_f64().unwrap_or(0.0);
                            let sp = (x.abs() * 0.01).max(0.05);
                            if ui.add(egui::DragValue::new(&mut x).speed(sp)).changed() {
                                *e = json!(x);
                                changed = true;
                            }
                        }
                    });
                }
                Value::Array(a) if a.iter().all(|e| e.is_string()) => {
                    let mut t = a.iter().filter_map(|e| e.as_str()).collect::<Vec<_>>().join(", ");
                    if ui.add(egui::TextEdit::singleline(&mut t).desired_width(170.0)).on_hover_text("comma-separated face names").changed() {
                        *val = json!(t.split(',').map(|x| x.trim()).filter(|x| !x.is_empty()).collect::<Vec<_>>());
                        changed = true;
                    }
                }
                Value::Bool(b) => {
                    changed |= ui.checkbox(b, "").changed();
                }
                Value::String(t) if k == "material" => {
                    let mut cur = t.clone();
                    egui::ComboBox::from_id_salt(format!("{id}-mat")).width(170.0).selected_text(&cur).show_ui(ui, |ui| {
                        for m in mats {
                            ui.selectable_value(&mut cur, m.clone(), m);
                        }
                    });
                    if cur != *t {
                        *t = cur;
                        changed = true;
                    }
                }
                Value::String(t) if matches!(k.as_str(), "axis" | "polarization" | "direction") => {
                    ui.horizontal(|ui| {
                        for ax in ["x", "y", "z"] {
                            if ui.selectable_label(t == ax, ax.to_uppercase()).clicked() && t != ax {
                                *t = ax.to_string();
                                changed = true;
                            }
                        }
                    });
                }
                Value::String(t) if k == "operation" => {
                    ui.horizontal(|ui| {
                        for o in ["add", "subtract"] {
                            if ui.selectable_label(t == o, o).clicked() && t != o {
                                *t = o.to_string();
                                changed = true;
                            }
                        }
                    });
                }
                Value::String(t) => {
                    changed |= ui.add(egui::TextEdit::singleline(t).desired_width(170.0)).changed();
                }
                _ => {
                    ui.label(RichText::new("—").weak());
                }
            }
            ui.end_row();
        }
    });
    changed
}

/// Simple line chart: series of (x, y) points with axes, ticks and a hover readout.
fn line_chart(ui: &mut egui::Ui, p: &Pal, series: &[(String, Vec<(f64, f64)>)], xlabel: &str, size: egui::Vec2) {
    let (rect, resp) = ui.allocate_exact_size(size, Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 6.0, p.card);
    let pts: Vec<(f64, f64)> = series.iter().flat_map(|s| s.1.iter().copied()).filter(|q| q.0.is_finite() && q.1.is_finite()).collect();
    if pts.len() < 2 {
        painter.text(rect.center(), Align2::CENTER_CENTER, "no data", FontId::proportional(13.0), p.weak);
        return;
    }
    let (mut x0, mut x1, mut y0, mut y1) = (f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY);
    for (x, y) in &pts {
        x0 = x0.min(*x);
        x1 = x1.max(*x);
        y0 = y0.min(*y);
        y1 = y1.max(*y);
    }
    if (y1 - y0).abs() < 1e-30 {
        y0 -= 1.0;
        y1 += 1.0;
    }
    if (x1 - x0).abs() < 1e-30 {
        x1 += 1.0;
    }
    let pad = (y1 - y0) * 0.06;
    let (y0, y1) = (y0 - pad, y1 + pad);
    let plot = Rect::from_min_max(rect.min + vec2(62.0, 10.0), rect.max - vec2(30.0, 30.0));
    let sx = |x: f64| plot.min.x + ((x - x0) / (x1 - x0)) as f32 * plot.width();
    let sy = |y: f64| plot.max.y - ((y - y0) / (y1 - y0)) as f32 * plot.height();
    let axis = Stroke::new(1.0, p.border);
    painter.line_segment([plot.left_bottom(), plot.right_bottom()], axis);
    painter.line_segment([plot.left_bottom(), plot.left_top()], axis);
    for i in 0..=4 {
        let fy = y0 + (y1 - y0) * i as f64 / 4.0;
        let fx = x0 + (x1 - x0) * i as f64 / 4.0;
        painter.text(pos2(plot.min.x - 6.0, sy(fy)), Align2::RIGHT_CENTER, render::fmt_val(fy), FontId::proportional(10.0), p.weak);
        painter.text(pos2(sx(fx), plot.max.y + 4.0), Align2::CENTER_TOP, render::fmt_val(fx), FontId::proportional(10.0), p.weak);
        painter.line_segment([pos2(plot.min.x, sy(fy)), pos2(plot.max.x, sy(fy))], Stroke::new(0.5, p.border.gamma_multiply(0.5)));
    }
    painter.text(pos2(plot.center().x, rect.max.y - 2.0), Align2::CENTER_BOTTOM, xlabel, FontId::proportional(11.0), p.weak);
    let colors = [p.accent, Color32::from_rgb(70, 170, 230), Color32::from_rgb(120, 200, 90), Color32::from_rgb(220, 90, 160)];
    for (si, (name, s)) in series.iter().enumerate() {
        let c = colors[si % colors.len()];
        let line: Vec<Pos2> = s.iter().filter(|q| q.0.is_finite() && q.1.is_finite()).map(|(x, y)| pos2(sx(*x), sy(*y))).collect();
        painter.add(egui::Shape::line(line.clone(), Stroke::new(2.0, c)));
        if line.len() <= 60 {
            for q in &line {
                painter.circle_filled(*q, 2.5, c);
            }
        }
        painter.text(plot.min + vec2(8.0, 6.0 + 14.0 * si as f32), Align2::LEFT_TOP, name, FontId::proportional(11.0), c);
    }
    if let Some(hp) = resp.hover_pos() {
        if plot.contains(hp) {
            let x = x0 + ((hp.x - plot.min.x) / plot.width()) as f64 * (x1 - x0);
            if let Some((bx, by)) = series.first().and_then(|s| s.1.iter().min_by(|a, b| (a.0 - x).abs().total_cmp(&(b.0 - x).abs())).copied()) {
                painter.circle_stroke(pos2(sx(bx), sy(by)), 4.0, Stroke::new(1.5, p.text));
                painter.text(pos2(sx(bx) + 8.0, sy(by) - 8.0), Align2::LEFT_BOTTOM, format!("{}, {}", render::fmt_val(bx), render::fmt_val(by)), FontId::proportional(11.0), p.text);
            }
        }
    }
}

/// Help window state.
#[derive(Default)]
struct HelpUi {
    open: bool,
    section: usize,
    query: String,
    focus_search: bool,
    scroll_top: bool,
}

/// Inline markdown (**bold**, `code`, [link](url)) as a wrapped layout job; `hl` is highlighted.
fn md_inline(text: &str, p: &Pal, size: f32, hl: &str) -> egui::text::LayoutJob {
    use egui::text::{LayoutJob, TextFormat};
    let base = TextFormat { font_id: FontId::proportional(size), color: p.text.gamma_multiply(0.86), ..Default::default() };
    let mut segs: Vec<(String, TextFormat)> = Vec::new();
    let (mut bold, mut code, mut ital) = (false, false, false);
    let mut cur = String::new();
    let fmt = |bold: bool, code: bool, ital: bool| {
        let mut f = base.clone();
        f.italics = ital;
        if code {
            f.font_id = FontId::monospace(size - 1.0);
            f.background = p.border.gamma_multiply(0.5);
            f.color = p.text;
        } else if bold {
            f.color = p.text;
            f.font_id = FontId::proportional(size + 0.3);
        }
        f
    };
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if !code && c == '*' && chars.get(i + 1) == Some(&'*') {
            segs.push((std::mem::take(&mut cur), fmt(bold, code, ital)));
            bold = !bold;
            i += 2;
            continue;
        }
        if c == '`' {
            segs.push((std::mem::take(&mut cur), fmt(bold, code, ital)));
            code = !code;
            i += 1;
            continue;
        }
        // *italic*: a single star touching a word on the inside
        if !code && c == '*' && chars.get(i + 1).is_some_and(|n| if ital { true } else { !n.is_whitespace() }) && (ital || i == 0 || !chars[i - 1].is_alphanumeric()) {
            segs.push((std::mem::take(&mut cur), fmt(bold, code, ital)));
            ital = !ital;
            i += 1;
            continue;
        }
        if !code && c == '[' {
            if let Some(close) = chars[i..].iter().position(|&x| x == ']').map(|k| i + k) {
                if chars.get(close + 1) == Some(&'(') {
                    if let Some(end) = chars[close..].iter().position(|&x| x == ')').map(|k| close + k) {
                        segs.push((std::mem::take(&mut cur), fmt(bold, code, ital)));
                        let mut f = fmt(bold, code, ital);
                        f.color = p.accent;
                        segs.push((chars[i + 1..close].iter().collect(), f));
                        i = end + 1;
                        continue;
                    }
                }
            }
        }
        cur.push(c);
        i += 1;
    }
    segs.push((cur, fmt(bold, code, ital)));
    let mut job = LayoutJob::default();
    let hl_l = hl.to_lowercase();
    for (t, f) in segs.into_iter().filter(|s| !s.0.is_empty()) {
        if hl_l.len() < 2 {
            job.append(&t, 0.0, f);
            continue;
        }
        // highlight every occurrence of the search text
        let low = t.to_lowercase();
        let mut last = 0;
        for (pos, _) in low.match_indices(&hl_l) {
            if pos < last || !t.is_char_boundary(pos) || !t.is_char_boundary(pos + hl_l.len()) {
                continue;
            }
            job.append(&t[last..pos], 0.0, f.clone());
            let mut h = f.clone();
            h.background = Color32::from_rgba_unmultiplied(250, 200, 60, 110);
            job.append(&t[pos..pos + hl_l.len()], 0.0, h);
            last = pos + hl_l.len();
        }
        job.append(&t[last..], 0.0, f);
    }
    job
}

/// Render a markdown section body: paragraphs, bullets, numbered lists, tables and code blocks.
fn render_md(ui: &mut egui::Ui, p: &Pal, md: &str, hl: &str) {
    let lines: Vec<&str> = md.lines().collect();
    let mut i = 0;
    let wrap = |ui: &mut egui::Ui, job: egui::text::LayoutJob| {
        ui.add(egui::Label::new(job).wrap());
    };
    while i < lines.len() {
        let line = lines[i];
        let t = line.trim();
        if t.is_empty() {
            ui.add_space(6.0);
            i += 1;
            continue;
        }
        if t.chars().all(|c| c == '-') && t.len() >= 3 {
            ui.separator();
            i += 1;
            continue;
        }
        if t.starts_with("```") {
            let mut code = Vec::new();
            i += 1;
            while i < lines.len() && !lines[i].trim_start().starts_with("```") {
                code.push(lines[i]);
                i += 1;
            }
            i += 1;
            egui::Frame::new().fill(p.border.gamma_multiply(0.35)).corner_radius(6.0).inner_margin(8.0).show(ui, |ui| {
                ui.add(egui::Label::new(RichText::new(code.join("\n")).monospace().size(12.5)).wrap());
            });
            ui.add_space(4.0);
            continue;
        }
        if t.starts_with('|') {
            let mut rows: Vec<Vec<String>> = Vec::new();
            while i < lines.len() && lines[i].trim().starts_with('|') {
                let cells: Vec<String> = lines[i].trim().trim_matches('|').split('|').map(|c| c.trim().to_string()).collect();
                if !cells.iter().all(|c| c.chars().all(|ch| ch == '-' || ch == ':') && !c.is_empty()) {
                    rows.push(cells);
                }
                i += 1;
            }
            let ncol = rows.iter().map(|r| r.len()).max().unwrap_or(1);
            // column widths in proportion to their (capped) text length
            let lens: Vec<f32> = (0..ncol).map(|k| rows.iter().map(|r| r.get(k).map(|c| c.chars().count()).unwrap_or(0)).max().unwrap_or(1).clamp(6, 60) as f32).collect();
            let total: f32 = lens.iter().sum();
            let avail = ui.available_width() - 14.0 * ncol as f32 - 8.0;
            let widths: Vec<f32> = lens.iter().map(|l| (avail * l / total).max(60.0)).collect();
            egui::Grid::new(ui.next_auto_id()).striped(true).num_columns(ncol).spacing(vec2(14.0, 4.0)).show(ui, |ui| {
                for (ri, row) in rows.iter().enumerate() {
                    for (k, c) in row.iter().enumerate() {
                        let txt = if ri == 0 { format!("**{c}**") } else { c.clone() };
                        let w = widths.get(k).copied().unwrap_or(80.0);
                        ui.vertical(|ui| {
                            ui.set_width(w);
                            ui.add(egui::Label::new(md_inline(&txt, p, 12.5, hl)).wrap());
                        });
                    }
                    ui.end_row();
                }
            });
            ui.add_space(4.0);
            continue;
        }
        let indent = line.len() - line.trim_start().len();
        let bullet = t.strip_prefix("- ").or_else(|| t.strip_prefix("* "));
        let numbered = t.split_once(". ").filter(|(n, _)| !n.is_empty() && n.len() <= 3 && n.chars().all(|c| c.is_ascii_digit()));
        if bullet.is_some() || numbered.is_some() {
            let (mark, mut text) = match (bullet, numbered) {
                (Some(b), _) => ("•".to_string(), b.to_string()),
                (_, Some((n, rest))) => (format!("{n}."), rest.to_string()),
                _ => unreachable!(),
            };
            i += 1;
            // continuation lines of this item
            while i < lines.len() {
                let nt = lines[i].trim();
                let nind = lines[i].len() - lines[i].trim_start().len();
                if nt.is_empty() || nt.starts_with("- ") || nt.starts_with("* ") || nt.starts_with('|') || nt.starts_with("```") || nind <= indent || nt.split_once(". ").is_some_and(|(n, _)| n.chars().all(|c| c.is_ascii_digit()) && !n.is_empty()) {
                    break;
                }
                text.push(' ');
                text.push_str(nt);
                i += 1;
            }
            ui.horizontal_top(|ui| {
                ui.add_space(8.0 + indent as f32 * 6.0);
                ui.label(RichText::new(mark).color(p.accent));
                wrap(ui, md_inline(&text, p, 13.5, hl));
            });
            continue;
        }
        // paragraph: join lines until a blank line or another block
        let mut text = t.to_string();
        i += 1;
        while i < lines.len() {
            let nt = lines[i].trim();
            let numbered_next = nt.split_once(". ").is_some_and(|(n, _)| !n.is_empty() && n.len() <= 3 && n.chars().all(|c| c.is_ascii_digit()));
            if nt.is_empty() || nt.starts_with("- ") || nt.starts_with("* ") || nt.starts_with('|') || nt.starts_with("```") || nt.starts_with('#') || numbered_next {
                break;
            }
            text.push(' ');
            text.push_str(nt);
            i += 1;
        }
        wrap(ui, md_inline(&text, p, 13.5, hl));
    }
}

/// An action that discards the current model; asks first when there are unsaved changes.
#[derive(Clone, Debug)]
enum Pending {
    New,
    Open,
    OpenPath(String),
    Example(&'static str),
    Quit,
}

pub struct SimApp {
    core: Shared,
    cam: Camera,
    tab: Tab,
    form: Form,
    f: Forms,
    tex: Option<egui::TextureHandle>,
    key: Option<RenderKey>,
    /// action waiting for the "save changes?" answer
    confirm: Option<Pending>,
    /// body / load picked in the browser, and the editable copy shown in Properties ("b:Name" / "l:3")
    sel_body: Option<String>,
    sel_load: Option<u32>,
    edit_val: Option<(String, Value)>,
    /// transform panel: mode, translate, count, mirror axis, plane, copy
    xform: (String, [f64; 3], usize, usize, f64, bool),
    /// material editor: (material it is based on, editable fields incl. name)
    mat_edit: Option<(String, Value)>,
    show_chart: bool,
    help: HelpUi,
    probe_mode: bool,
    /// probed nodes, with the results revision they belong to
    probes: Vec<(u32, u64)>,
    allow_close: bool,
    last_autosave: f64,
    recovery_offered: bool,
    /// Scene (geometry + colours) cached across camera moves: (rev, sel, anim, mesh) → scene
    scene: Option<((u64, u64, i64, usize), Arc<render::Scene>)>,
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
    /// optimiser: Bayesian (surrogate) instead of direct search
    bayes: bool,
    /// what-if slider values (parameter order of the current study) and the last prediction
    whatif: Vec<f64>,
    whatif_key: String,
    whatif_pred: Option<Value>,
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
        core.session.lock().unwrap_or_else(std::sync::PoisonError::into_inner).display.dark = dark;
        *core.gui_ctx.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = Some(cc.egui_ctx.clone());
        let ctx = cc.egui_ctx.clone();
        *core.notify.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = Some(Box::new(move || ctx.request_repaint()));
        SimApp {
            core,
            cam: Camera::default(),
            tab: Tab::Geometry,
            form: Form::None,
            f: Forms::default(),
            tex: None,
            key: None,
            confirm: None,
            sel_body: None,
            sel_load: None,
            edit_val: None,
            xform: ("move".into(), [10.0, 0.0, 0.0], 3, 0, 0.0, true),
            mat_edit: None,
            show_chart: false,
            help: HelpUi::default(),
            probe_mode: false,
            probes: Vec::new(),
            allow_close: false,
            last_autosave: 0.0,
            recovery_offered: false,
            scene: None,
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
            opt: OptUi { max_evals: 12, bayes: true, vf: 0.4, iters: 40, objective: "mass".into(), ..Default::default() },
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
        let s = self.core.session.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
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
        let want = self.core.session.lock().unwrap_or_else(std::sync::PoisonError::into_inner).display.dark;
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
                let name = self.core.session.lock().unwrap_or_else(std::sync::PoisonError::into_inner).project_path.clone();
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
            *out.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = Some(r);
        });
        self.opt.job = Some(slot);
        self.opt.error = None;
    }

    /// Forms for parameter studies, optimization and topology optimization.
    fn opt_form(&mut self, ui: &mut egui::Ui, form: Form) {
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
        let finished = self.opt.job.as_ref().map(|job| job.lock().unwrap_or_else(std::sync::PoisonError::into_inner).take());
        match finished {
            Some(Some(r)) => {
                self.opt.error = r.err();
                self.opt.show_results = self.opt.error.is_none();
                self.opt.job = None;
            }
            Some(None) => ctx.request_repaint_after(std::time::Duration::from_millis(250)),
            None => {}
        }
        let Some(st) = self.core.session.lock().unwrap_or_else(std::sync::PoisonError::into_inner).study.clone() else { return };
        let sig = format!("{}|{}|{}", st.kind, st.message, st.seconds);
        if sig != self.opt.seen {
            self.opt.seen = sig;
            self.opt.show_results = true;
            self.opt.whatif.clear();
            self.opt.whatif_key.clear();
        }
        if !self.opt.show_results {
            return;
        }
        let p = self.p;
        let mut open = true;
        let mut actions: Vec<(String, Value)> = Vec::new();
        let title = match st.kind.as_str() {
            "optimization" => "🎯 Optimization results",
            "surrogate_optimization" => "🎯 Surrogate optimization results",
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
                    return;
                }
                // ---- what-if from the surrogate
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
                // ---- charts for one-parameter studies
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
                });
            });
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
            if let Some(r) = job.lock().unwrap_or_else(std::sync::PoisonError::into_inner).take() {
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
        let plan = self.core.session.lock().unwrap_or_else(std::sync::PoisonError::into_inner).plan.clone();
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
        let kind = self.core.session.lock().unwrap_or_else(std::sync::PoisonError::into_inner).model.analysis.kind;
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
                if kind.requires_enclosure() && !self.core.session.lock().unwrap_or_else(std::sync::PoisonError::into_inner).model.has_enclosure() {
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
                    let mut sm = self.core.session.lock().unwrap_or_else(std::sync::PoisonError::into_inner).display.show_mesh;
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
                        self.help.open = true;
                        self.help.focus_search = true;
                    } else {
                        self.open_help(t);
                    }
                }
            }
            Tab::Optimize => {
                let kind = self.core.session.lock().unwrap_or_else(std::sync::PoisonError::into_inner).model.analysis.kind;
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
                    let has = self.core.session.lock().unwrap_or_else(std::sync::PoisonError::into_inner).study.is_some();
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

    fn results_ribbon(&mut self, ui: &mut egui::Ui) {
        let p = self.p;
        let (res, disp) = {
            let s = self.core.session.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
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
        group(ui, &p, "TOOLS", |ui| {
            if tool(ui, &p, "📈", "History", Some(if self.show_chart { p.accent } else { p.weak }), !r.history.is_empty()) {
                self.show_chart = !self.show_chart;
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
                if self.core.session.lock().unwrap_or_else(std::sync::PoisonError::into_inner).mesh.as_ref().is_some_and(|m| m.with_enclosure) || disp.section.is_some() {
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

    fn open_help(&mut self, topic: &str) {
        if let Some(i) = crate::help::find(topic) {
            self.help.section = i;
        }
        self.help.query.clear();
        self.help.open = true;
        self.help.scroll_top = true;
    }

    /// Help for what the user is looking at (F1).
    fn context_help(&mut self) {
        let kind = self.core.session.lock().unwrap_or_else(std::sync::PoisonError::into_inner).model.analysis.kind;
        let topic = match self.tab {
            Tab::Geometry => "4".to_string(),
            Tab::Material => "5".to_string(),
            Tab::Mesh => "6".to_string(),
            Tab::Setup if self.form != Form::None => "8".to_string(),
            Tab::Setup | Tab::Solve => crate::help::analysis_topic(&kind.key()).to_string(),
            Tab::Results => "9".to_string(),
            Tab::Optimize => "10".to_string(),
            Tab::Help => "2".to_string(),
        };
        self.open_help(&topic);
    }

    /// The user guide inside the app: contents / search on the left, the section on the right.
    fn help_window(&mut self, ctx: &egui::Context) {
        if !self.help.open {
            return;
        }
        let p = self.p;
        let secs = crate::help::sections();
        let mut open = true;
        let mut goto: Option<usize> = None;
        egui::Window::new(RichText::new("📖 SimCraft help").strong())
            .open(&mut open)
            .resizable(true)
            .collapsible(false)
            .default_size(vec2(940.0, 660.0))
            .default_pos(ctx.content_rect().center() - vec2(470.0, 330.0))
            .frame(egui::Frame::window(&ctx.global_style()).fill(p.card).inner_margin(12))
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    let r = ui.add(egui::TextEdit::singleline(&mut self.help.query).hint_text("🔎 Search the guide (e.g. heat sink, fatigue, undo)").desired_width(380.0));
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
                let h = ui.available_height();
                ui.horizontal_top(|ui| {
                    // contents or search results
                    ui.vertical(|ui| {
                        ui.set_width(250.0);
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
                                for (i, s) in secs.iter().enumerate().filter(|(_, s)| s.level >= 2) {
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
                    // the section with its subsections
                    ui.vertical(|ui| {
                        let mut sa = egui::ScrollArea::vertical().id_salt(("help-body", self.help.section)).max_height(h - 34.0).auto_shrink([false, false]);
                        if self.help.scroll_top {
                            sa = sa.vertical_scroll_offset(0.0);
                            self.help.scroll_top = false;
                        }
                        let hl = self.help.query.trim().to_string();
                        sa.show(ui, |ui| {
                            for j in crate::help::span(self.help.section) {
                                let s = &secs[j];
                                let size = match s.level {
                                    1 => 22.0,
                                    2 => 19.0,
                                    _ => 15.5,
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
            });
        if let Some(i) = goto {
            self.help.section = i;
            self.help.scroll_top = true;
        }
        self.help.open = open;
    }

    /// History chart of the current results (transient, harmonic sweep, load steps, convergence).
    fn chart_window(&mut self, ctx: &egui::Context) {
        if !self.show_chart {
            return;
        }
        let Some(r) = self.core.session.lock().unwrap_or_else(std::sync::PoisonError::into_inner).results.clone() else {
            self.show_chart = false;
            return;
        };
        if r.history.is_empty() {
            self.show_chart = false;
            return;
        }
        let p = self.p;
        let mut open = true;
        egui::Window::new(RichText::new("📈 History").strong()).open(&mut open).resizable(true).default_size(vec2(560.0, 340.0)).show(ctx, |ui| {
            let label = if r.history_label.is_empty() { "value".to_string() } else { r.history_label.clone() };
            let xl = if r.history_x.is_empty() { "step".to_string() } else { r.history_x.clone() };
            let sz = vec2(ui.available_width().max(300.0), (ui.available_height() - 6.0).max(200.0));
            line_chart(ui, &p, &[(label, r.history.clone())], &xl, sz);
        });
        self.show_chart = open;
    }

    /// Custom material editor (saves through define_material).
    fn material_window(&mut self, ctx: &egui::Context) {
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

    /// Properties panel for the body or load picked in the browser.
    fn item_editor(&mut self, ui: &mut egui::Ui) {
        let p = self.p;
        let (model, lib) = {
            let s = self.core.session.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            (s.model.clone(), s.model.materials().into_iter().map(|m| m.name).collect::<Vec<_>>())
        };
        // the editable copy follows the selection (and is refreshed after undo / external edits)
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
            let title = if self.sel_body.is_some() { format!("Body · {}", fresh["name"].as_str().unwrap_or("")) } else { format!("Load #{} · {}", self.sel_load.unwrap(), model.loads.iter().find(|l| Some(l.id) == self.sel_load).map(|l| l.load.label()).unwrap_or("")) };
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
        // transform tools for bodies
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

    fn dirty(&self) -> bool {
        self.core.session.lock().unwrap_or_else(std::sync::PoisonError::into_inner).is_dirty()
    }
    /// Run an action that replaces the model, asking to save unsaved changes first.
    fn guard(&mut self, a: Pending) {
        if self.dirty() {
            self.confirm = Some(a);
        } else {
            self.perform(a);
        }
    }
    fn perform(&mut self, a: Pending) {
        match a {
            Pending::New => {
                self.run("new_model", json!({}));
            }
            Pending::Open => {
                if let Some(p) = rfd::FileDialog::new().add_filter("SimCraft", &["simcraft", "json"]).pick_file() {
                    self.run("open_project", json!({"path": p.to_string_lossy()}));
                }
            }
            Pending::OpenPath(p) => {
                self.run("open_project", json!({"path": p}));
            }
            Pending::Example(name) => match name {
                "bracket" => self.example_bracket(),
                "heatsink" => self.example_heatsink(),
                "fork" => self.example_fork(),
                "buckling" => self.example_buckling(),
                "capacitor" => self.example_capacitor(),
                "busbar" => self.example_busbar(),
                "coil" => self.example_coil(),
                "microwave" => self.example_microwave(),
                "heatsink_cht" => self.example_heatsink_cht(),
                _ => self.example_cfd(),
            },
            Pending::Quit => {
                let _ = std::fs::remove_file(crate::session::autosave_path());
                self.allow_close = true;
                if let Some(ctx) = self.core.gui_ctx.lock().unwrap_or_else(std::sync::PoisonError::into_inner).clone() {
                    ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                }
            }
        }
        self.need_fit = true;
    }
    /// "Save changes?" dialog, autosave and crash-recovery offer.
    fn safety_ui(&mut self, ctx: &egui::Context) {
        // window close with unsaved work
        if ctx.input(|i| i.viewport().close_requested()) && !self.allow_close {
            if self.dirty() {
                ctx.send_viewport_cmd(egui::ViewportCommand::CancelClose);
                self.confirm = Some(Pending::Quit);
            } else {
                let _ = std::fs::remove_file(crate::session::autosave_path());
            }
        }
        // autosave every 2 minutes while there are unsaved changes
        let now = ctx.input(|i| i.time);
        if now - self.last_autosave > 120.0 {
            self.last_autosave = now;
            let s = self.core.session.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            if s.is_dirty() && !s.model.bodies.is_empty() {
                let _ = s.autosave();
            }
        }
        ctx.request_repaint_after(std::time::Duration::from_secs(30));
        // offer recovery once at start-up
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
        egui::Modal::new(egui::Id::new("save-changes")).show(ctx, |ui| {
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
                    let cur = self.core.session.lock().unwrap_or_else(std::sync::PoisonError::into_inner).model.element_size;
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
            let s = self.core.session.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
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

    // ------------------------------------------------ viewport
    fn viewport(&mut self, ui: &mut egui::Ui) {
        let p = self.p;
        let rect = ui.available_rect_before_wrap();
        let resp = ui.allocate_rect(rect, Sense::click_and_drag());
        let painter = ui.painter_at(rect);

        let (mesh, model, results, display, rev, view_req, stale) = {
            let mut s = self.core.session.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
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
            let mut s = self.core.session.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
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
            let skey = (key.rev, key.sel, key.anim, key.mesh);
            let scene = match &self.scene {
                Some((k, sc)) if *k == skey => sc.clone(),
                _ => {
                    let sc = Arc::new(render::build_scene(&mesh, &model, results.as_deref(), &display, &self.selected, anim));
                    self.scene = Some((skey, sc.clone()));
                    sc
                }
            };
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
        let res_id = results.as_ref().map(|r| Arc::as_ptr(r) as u64).unwrap_or(0);
        let probing = self.probe_mode && display.show_results && results.is_some();
        // screen position of a node (deformed like the picture)
        let node_screen = |n: usize| -> Pos2 {
            let mut q = mesh.nodes[n];
            if let (Some(r), Some((_, sc))) = (&results, self.legend) {
                if let Some(d) = r.deformation(display.mode) {
                    if d.len() == mesh.nodes.len() && sc > 0.0 {
                        q = add(q, scale(d[n], sc * anim));
                    }
                }
            }
            let (x, y, _) = self.cam.project(q, rect.width() as f64, rect.height() as f64);
            rect.min + vec2(x as f32, y as f32)
        };
        if probing && resp.clicked() {
            if let (Some(pos), Some(fid)) = (resp.interact_pointer_pos(), resp.interact_pointer_pos().and_then(|q| pick(q))) {
                let n = mesh.faces[fid as usize].nodes.iter().copied().min_by(|a, b| node_screen(*a as usize).distance(pos).total_cmp(&node_screen(*b as usize).distance(pos))).unwrap();
                self.probes.retain(|(_, rid)| *rid == res_id);
                if !self.probes.iter().any(|(m, _)| *m == n) {
                    self.probes.push((n, res_id));
                    if self.probes.len() > 8 {
                        self.probes.remove(0);
                    }
                }
            }
        } else if resp.clicked() {
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

        // ---- probe markers
        if display.show_results {
            if let Some(r) = &results {
                let f = if r.has(display.field) { display.field } else { r.available_fields()[0] };
                if let Some(vals) = r.scalar(f, display.mode) {
                    for (n, rid) in &self.probes {
                        if *rid != res_id || *n as usize >= vals.len() {
                            continue;
                        }
                        let sp = node_screen(*n as usize);
                        if !rect.contains(sp) {
                            continue;
                        }
                        painter.circle(sp, 4.5, p.accent, Stroke::new(1.5, Color32::WHITE));
                        let txt = format!("{} {}", render::fmt_val(vals[*n as usize]), f.unit());
                        let galley = painter.layout_no_wrap(txt, FontId::monospace(12.0), Color32::WHITE);
                        let br = Rect::from_min_size(sp + vec2(8.0, -22.0), galley.size() + vec2(10.0, 6.0));
                        painter.rect_filled(br, 4.0, Color32::from_black_alpha(190));
                        painter.galley(br.min + vec2(5.0, 3.0), galley, Color32::WHITE);
                    }
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
        let pr = self.core.progress.lock().unwrap_or_else(std::sync::PoisonError::into_inner).clone();
        let (last, err) = {
            let s = self.core.session.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            (s.log.last().cloned().unwrap_or_default(), s.last_error.clone())
        };
        ui.horizontal(|ui| {
            if pr.running {
                ui.add(egui::ProgressBar::new(pr.fraction as f32).desired_width(240.0).desired_height(14.0).text(RichText::new(pr.message.clone()).small()).fill(p.accent));
                if ui.small_button(RichText::new("■ Stop").color(p.err)).on_hover_text("Stop the running solve or study").clicked() {
                    let _ = self.core.call("cancel", &json!({}));
                }
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
        // test hook (`__ui`, only with SIMCRAFT_TEST_HOOKS): drive GUI-only state for screenshots
        let req = self.core.session.lock().unwrap_or_else(std::sync::PoisonError::into_inner).ui_request.take();
        if let Some(q) = req {
            if let Some(b) = q["select_body"].as_str() {
                self.sel_body = Some(b.to_string());
                self.sel_load = None;
                self.form = Form::None;
            }
            if let Some(id) = q["select_load"].as_u64() {
                self.sel_load = Some(id as u32);
                self.sel_body = None;
                self.form = Form::None;
            }
            if let Some(m) = q["edit_material"].as_str() {
                self.f.material = m.to_string();
                let lib = self.core.session.lock().unwrap_or_else(std::sync::PoisonError::into_inner).model.materials();
                if let Some(mat) = lib.iter().find(|x| x.name == m) {
                    let full = serde_json::to_value(mat).unwrap_or(json!({}));
                    let mut v = json!({"name": format!("{} (custom)", mat.name)});
                    for k in MAT_KEYS {
                        if let Some(x) = full.get(*k) {
                            v[*k] = x.clone();
                        }
                    }
                    self.mat_edit = Some((mat.name.clone(), v));
                }
            }
            if let Some(t) = q["help"].as_str() {
                self.open_help(t);
            }
            if let Some(t) = q["help_search"].as_str() {
                self.help.open = true;
                self.help.query = t.to_string();
            }
            if q["show_chart"].as_bool() == Some(true) {
                self.show_chart = true;
            }
            if let Some(n) = q["probe_node"].as_u64() {
                let rid = self.core.session.lock().unwrap_or_else(std::sync::PoisonError::into_inner).results.as_ref().map(|r| Arc::as_ptr(r) as u64).unwrap_or(0);
                self.probes.push((n as u32, rid));
                self.probe_mode = true;
            }
            if let Some(t) = q["tab"].as_str() {
                self.tab = match t {
                    "results" => Tab::Results,
                    "material" => Tab::Material,
                    "solve" => Tab::Solve,
                    "help" => Tab::Help,
                    _ => self.tab,
                };
            }
        }
        self.safety_ui(&ctx);
        self.chart_window(&ctx);
        self.help_window(&ctx);
        if ctx.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::F1)) {
            self.context_help();
        }
        self.material_window(&ctx);
        if !ctx.egui_wants_keyboard_input() {
            let (undo, redo) = ctx.input_mut(|i| {
                let redo = i.consume_shortcut(&egui::KeyboardShortcut::new(egui::Modifiers::COMMAND | egui::Modifiers::SHIFT, egui::Key::Z)) || i.consume_shortcut(&egui::KeyboardShortcut::new(egui::Modifiers::COMMAND, egui::Key::Y));
                let undo = i.consume_shortcut(&egui::KeyboardShortcut::new(egui::Modifiers::COMMAND, egui::Key::Z));
                (undo, redo)
            });
            if undo {
                self.run("undo", json!({}));
            }
            if redo {
                self.run("redo", json!({}));
            }
        }
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
