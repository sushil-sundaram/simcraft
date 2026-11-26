//! Theme palette, typography/styles, dark-mode persistence, reusable widgets, charts, and markdown.

use crate::render;
use eframe::egui::{self, Align2, Color32, CornerRadius, FontId, Pos2, Rect, RichText, Sense, Stroke, TextStyle, pos2, vec2};
use serde_json::{Value, json};

#[derive(Clone, Copy)]
pub struct Pal {
    pub dark: bool,
    pub panel: Color32,
    pub ribbon: Color32,
    pub card: Color32,
    pub hover: Color32,
    pub text: Color32,
    pub weak: Color32,
    pub border: Color32,
    pub accent: Color32,
    pub ok: Color32,
    pub warn: Color32,
    pub err: Color32,
}

pub fn pal(dark: bool) -> Pal {
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

pub fn install_style(ctx: &egui::Context) {
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

pub fn load_dark_pref() -> Option<bool> {
    let t = std::fs::read_to_string(settings_path()?).ok()?;
    serde_json::from_str::<Value>(&t).ok()?.get("dark_mode")?.as_bool()
}

pub fn save_dark_pref(dark: bool) {
    if let Some(p) = settings_path() {
        let _ = std::fs::create_dir_all(p.parent().unwrap());
        let _ = std::fs::write(p, json!({"dark_mode": dark}).to_string());
    }
}

/// Material fields editable in the app (the ones define_material accepts).
pub const MAT_KEYS: &[&str] = &[
    "youngs_modulus_gpa", "poisson_ratio", "density", "yield_strength_mpa", "ultimate_strength_mpa", "thermal_conductivity", "thermal_expansion", "specific_heat",
    "relative_permittivity", "electrical_conductivity", "relative_permeability", "loss_tangent", "fatigue_strength_coefficient_mpa", "fatigue_exponent", "endurance_limit_mpa",
    "conductivity_table", "specific_heat_table",
];

/// Edit the fields of a JSON object in place: numbers, [x,y,z] vectors, text, face lists, switches.
pub fn json_editor(ui: &mut egui::Ui, v: &mut Value, mats: &[String], id: &str) -> bool {
    let mut changed = false;
    let Some(obj) = v.as_object_mut() else { return false };
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
                    if ui.add(egui::DragValue::new(&mut x).speed(sp).custom_formatter(|v, _| render::fmt_val(v))).changed() {
                        *val = json!(x);
                        changed = true;
                    }
                }
                Value::Array(a) if k.ends_with("_table") => {
                    let mut t = a.iter().filter_map(|r| Some(format!("{}: {}", render::fmt_val(r[0].as_f64()?), render::fmt_val(r[1].as_f64()?)))).collect::<Vec<_>>().join("; ");
                    let resp = ui.add(egui::TextEdit::singleline(&mut t).desired_width(170.0).hint_text("°C: value; °C: value")).on_hover_text("Value against temperature, e.g. 20: 167; 200: 180 (linear between points, constant outside). Empty = constant.");
                    if resp.changed() {
                        let rows: Option<Vec<[f64; 2]>> = t
                            .split(';')
                            .filter(|x| !x.trim().is_empty())
                            .map(|p| {
                                let (a, b) = p.split_once(':')?;
                                Some([a.trim().parse().ok()?, b.trim().parse().ok()?])
                            })
                            .collect();
                        if let Some(rows) = rows {
                            *val = json!(rows);
                            changed = true;
                        }
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
pub fn line_chart(ui: &mut egui::Ui, p: &Pal, series: &[(String, Vec<(f64, f64)>)], xlabel: &str, size: egui::Vec2) {
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

/// Inline markdown (**bold**, `code`, [link](url)) as a wrapped layout job; `hl` is highlighted.
pub fn md_inline(text: &str, p: &Pal, size: f32, hl: &str) -> egui::text::LayoutJob {
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
pub fn render_md(ui: &mut egui::Ui, p: &Pal, md: &str, hl: &str) {
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

/// The SimCraft logo (assets/simcraft-64.png), drawn as an image.
pub fn logo(ui: &mut egui::Ui, size: f32) {
    let id = egui::Id::NULL.with("simcraft-logo-tex");
    let tex = ui.ctx().data_mut(|d| d.get_temp::<egui::TextureHandle>(id)).unwrap_or_else(|| {
        let img = decode_png(include_bytes!("../../assets/simcraft-64.png")).unwrap_or_else(|| egui::ColorImage::filled([1, 1], Color32::TRANSPARENT));
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
pub fn tool(ui: &mut egui::Ui, p: &Pal, icon: &str, label: &str, color: Option<Color32>, enabled: bool) -> bool {
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
pub fn group(ui: &mut egui::Ui, p: &Pal, caption: &str, add: impl FnOnce(&mut egui::Ui)) {
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

pub fn card<R>(ui: &mut egui::Ui, p: &Pal, add: impl FnOnce(&mut egui::Ui) -> R) -> R {
    egui::Frame::new().fill(p.card).stroke(Stroke::new(1.0, p.border)).corner_radius(8).inner_margin(10).show(ui, |ui| {
        ui.set_width(ui.available_width());
        add(ui)
    }).inner
}

pub fn xyz(ui: &mut egui::Ui, id: &str, v: &mut [f64; 3], suffix: &str, speed: f64) {
    egui::Grid::new(id).num_columns(6).spacing([4.0, 4.0]).show(ui, |ui| {
        for (i, c) in ["X", "Y", "Z"].iter().enumerate() {
            ui.label(RichText::new(*c).small());
            ui.add(egui::DragValue::new(&mut v[i]).speed(speed).suffix(suffix));
        }
        ui.end_row();
    });
}

pub fn primary(p: &Pal, text: &str) -> egui::Button<'static> {
    egui::Button::new(RichText::new(text.to_string()).color(Color32::WHITE).strong()).fill(p.accent).min_size(vec2(110.0, 30.0))
}
