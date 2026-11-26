//! 3D viewport overlays: load/BC glyphs, hover perimeter highlighting, live primitive/transform
//! ghost previews with 3D axis drag gizmos, and interactive ViewCube.

use super::{EditTarget, Form, LoadKind, SelectMode, SimApp};
use crate::geometry::{Shape, V3, add, cross, dot, norm, normalize, scale, sub};
use crate::mesh::Mesh;
use crate::model::{CoilMode, Load, Model};
use crate::render::{self, Camera};
use eframe::egui::{self, Align2, Color32, FontId, Pos2, Rect, Sense, Stroke, pos2, vec2};
use serde_json::json;
use std::collections::{HashMap, HashSet};

fn world_to_screen(cam: &Camera, rect: Rect, q: V3) -> Pos2 {
    let (x, y, _) = cam.project(q, rect.width() as f64, rect.height() as f64);
    rect.min + vec2(x as f32, y as f32)
}

fn world_dir_screen(cam: &Camera, rect: Rect, origin: V3, dir: V3) -> Option<(Pos2, egui::Vec2)> {
    let u = normalize(dir);
    if norm(u) < 1e-12 {
        return None;
    }
    let step = (40.0 / cam.zoom.max(1e-6)).max(0.5);
    let p0 = world_to_screen(cam, rect, origin);
    let p1 = world_to_screen(cam, rect, add(origin, scale(u, step)));
    let d = p1 - p0;
    let len = d.length();
    if len < 2.0 {
        return None;
    }
    Some((p0, d / len))
}

fn draw_arrow_2d(painter: &egui::Painter, tip_base: Pos2, tip_end: Pos2, stroke: Stroke, head_px: f32) {
    let d = tip_end - tip_base;
    let len = d.length();
    if len < 3.0 {
        return;
    }
    let dir = d / len;
    let perp = vec2(-dir.y, dir.x);
    painter.line_segment([tip_base, tip_end], stroke);
    let b1 = tip_end - dir * head_px + perp * (head_px * 0.45);
    let b2 = tip_end - dir * head_px - perp * (head_px * 0.45);
    painter.add(egui::Shape::convex_polygon(vec![tip_end, b1, b2], stroke.color, Stroke::NONE));
}

fn draw_arrow_3d(cam: &Camera, rect: Rect, painter: &egui::Painter, origin: V3, dir: V3, len_px: f32, inward: bool, col: Color32, label: Option<&str>) {
    let Some((p0, sdir)) = world_dir_screen(cam, rect, origin, dir) else { return };
    if !rect.contains(p0) {
        return;
    }
    let (start, end) = if inward { (p0 + sdir * len_px, p0) } else { (p0, p0 + sdir * len_px) };
    draw_arrow_2d(painter, start, end, Stroke::new(2.2, col), 8.0);
    if let Some(lbl) = label {
        let lp = if inward { start + sdir * 6.0 } else { end + sdir * 6.0 };
        painter.text(lp, Align2::CENTER_CENTER, lbl, FontId::proportional(10.5), col);
    }
}

fn patch_anchors(mesh: &Mesh, names: &[String], hidden_bodies: &HashSet<String>) -> Vec<(V3, V3)> {
    let want: HashSet<&str> = names.iter().map(String::as_str).collect();
    mesh.patches
        .iter()
        .filter(|p| {
            let body = p.name.split('.').next().unwrap_or("");
            want.contains(p.name.as_str()) && !hidden_bodies.contains(body)
        })
        .map(|p| (p.centroid, p.normal))
        .collect()
}

fn body_center_and_radius(model: &Model, mesh: &Mesh, body_name: Option<&str>) -> (V3, f64) {
    if let Some(bn) = body_name {
        if let Some(b) = model.bodies.iter().find(|b| b.name.eq_ignore_ascii_case(bn)) {
            if let Some((lo, hi)) = shape_bbox(&b.shape) {
                return (scale(add(lo, hi), 0.5), norm(sub(hi, lo)) * 0.5);
            }
        }
    }
    (scale(add(mesh.bbox.0, mesh.bbox.1), 0.5), norm(sub(mesh.bbox.1, mesh.bbox.0)) * 0.4)
}

pub(super) fn draw_load_glyphs(
    cam: &Camera,
    rect: Rect,
    painter: &egui::Painter,
    model: &Model,
    mesh: &Mesh,
    selected_load: Option<u32>,
    hidden_bodies: &HashSet<String>,
) {
    for item in &model.loads {
        let rgb = render::load_color(&item.load);
        let mut col = Color32::from_rgb(rgb[0], rgb[1], rgb[2]);
        let is_sel = selected_load == Some(item.id);
        if is_sel {
            col = Color32::from_rgb(255, 220, 40);
        }
        let tag = is_sel.then(|| format!("#{}", item.id));
        if let Some(b) = item.load.body() {
            if hidden_bodies.contains(b) {
                continue;
            }
        }
        let anchors = patch_anchors(mesh, item.load.faces(), hidden_bodies);
        let visible_anchor = |pt: V3| -> bool {
            let sp = world_to_screen(cam, rect, pt);
            rect.contains(sp)
        };
        match &item.load {
            Load::FixedSupport { .. } => {
                for (c, _) in anchors {
                    if !visible_anchor(c) {
                        continue;
                    }
                    let sp = world_to_screen(cam, rect, c);
                    let pts = vec![sp, sp + vec2(-6.0, 9.0), sp + vec2(6.0, 9.0)];
                    painter.add(egui::Shape::convex_polygon(pts, col.gamma_multiply(0.8), Stroke::new(1.5, Color32::WHITE)));
                    painter.line_segment([sp + vec2(-7.0, 12.0), sp + vec2(7.0, 12.0)], Stroke::new(1.8, col));
                }
            }
            Load::Displacement { x, y, z, .. } => {
                for (c, _) in anchors {
                    for (ax, opt) in [(0, *x), (1, *y), (2, *z)] {
                        if let Some(v) = opt {
                            let mut d = [0.0; 3];
                            d[ax] = if v < 0.0 { -1.0 } else { 1.0 };
                            draw_arrow_3d(cam, rect, painter, c, d, 22.0, false, col, None);
                        }
                    }
                }
            }
            Load::CompressionOnly { .. } => {
                for (c, n) in anchors {
                    draw_arrow_3d(cam, rect, painter, c, n, 24.0, true, col, tag.as_deref());
                }
            }
            Load::Symmetry { .. } => {
                for (c, n) in anchors {
                    if let Some((p0, sd)) = world_dir_screen(cam, rect, c, n) {
                        if rect.contains(p0) {
                            let perp = vec2(-sd.y, sd.x);
                            painter.line_segment([p0 - perp * 10.0, p0 + perp * 10.0], Stroke::new(2.2, col));
                            painter.line_segment([p0, p0 + sd * 10.0], Stroke::new(1.5, col));
                        }
                    }
                }
            }
            Load::Force { vector, .. } => {
                for (c, _) in anchors {
                    draw_arrow_3d(cam, rect, painter, c, *vector, 36.0, false, col, tag.as_deref());
                }
            }
            Load::Pressure { value_mpa, .. } => {
                for (c, n) in anchors {
                    draw_arrow_3d(cam, rect, painter, c, n, 28.0, *value_mpa >= 0.0, col, tag.as_deref());
                }
            }
            Load::Gravity { acceleration } => {
                let (c, _) = body_center_and_radius(model, mesh, None);
                draw_arrow_3d(cam, rect, painter, c, *acceleration, 42.0, false, col, Some("g"));
            }
            Load::HeatFlux { value, .. } => {
                for (c, n) in anchors {
                    draw_arrow_3d(cam, rect, painter, c, n, 26.0, *value >= 0.0, col, Some("q"));
                }
            }
            Load::Temperature { .. } => {
                for (c, _) in anchors {
                    let sp = world_to_screen(cam, rect, c);
                    if rect.contains(sp) {
                        painter.circle(sp, 5.5, col, Stroke::new(1.5, Color32::WHITE));
                        painter.text(sp + vec2(8.0, -8.0), Align2::LEFT_CENTER, "T", FontId::proportional(10.5), col);
                    }
                }
            }
            Load::Convection { .. } => {
                for (c, n) in anchors {
                    draw_arrow_3d(cam, rect, painter, c, n, 22.0, false, col, Some("h"));
                }
            }
            Load::Radiation { .. } => {
                for (c, n) in anchors {
                    draw_arrow_3d(cam, rect, painter, c, n, 26.0, false, col, Some("ε"));
                }
            }
            Load::HeatGeneration { body, .. } => {
                let (c, _) = body_center_and_radius(model, mesh, body.as_deref());
                let sp = world_to_screen(cam, rect, c);
                if rect.contains(sp) {
                    painter.circle(sp, 6.5, col.gamma_multiply(0.85), Stroke::new(1.5, Color32::WHITE));
                    painter.text(sp + vec2(9.0, -9.0), Align2::LEFT_CENTER, "Q", FontId::proportional(11.0), col);
                }
            }
            Load::Voltage { .. } => {
                for (c, _) in anchors {
                    let sp = world_to_screen(cam, rect, c);
                    if rect.contains(sp) {
                        painter.circle(sp, 5.5, col, Stroke::new(1.5, Color32::WHITE));
                        painter.text(sp + vec2(8.0, -8.0), Align2::LEFT_CENTER, "V", FontId::proportional(10.5), col);
                    }
                }
            }
            Load::Current { value, .. } => {
                for (c, n) in anchors {
                    draw_arrow_3d(cam, rect, painter, c, n, 28.0, *value >= 0.0, col, Some("I"));
                }
            }
            Load::Coil { body, axis, mode, center, .. } => {
                let (mut c, r) = body_center_and_radius(model, mesh, Some(body));
                if let Some(cc) = center {
                    c = *cc;
                }
                let mut ax = [0.0; 3];
                ax[axis.idx()] = 1.0;
                if *mode == CoilMode::Axial {
                    draw_arrow_3d(cam, rect, painter, c, ax, 36.0, false, col, Some("J"));
                } else {
                    let u = normalize(cross(ax, if ax[2].abs() < 0.9 { [0.0, 0.0, 1.0] } else { [1.0, 0.0, 0.0] }));
                    let v = cross(ax, u);
                    let rr = (r * 0.45).max(3.0);
                    let pts: Vec<Pos2> = (0..=18)
                        .map(|i| {
                            let th = (i as f64 / 18.0) * std::f64::consts::TAU;
                            world_to_screen(cam, rect, add(c, add(scale(u, rr * th.cos()), scale(v, rr * th.sin()))))
                        })
                        .collect();
                    for w in pts.windows(2) {
                        painter.line_segment([w[0], w[1]], Stroke::new(2.0, col));
                    }
                    if pts.len() >= 2 {
                        draw_arrow_2d(painter, pts[0], pts[2], Stroke::new(2.0, col), 7.0);
                    }
                }
            }
            Load::Magnet { body, remanence } => {
                let (c, _) = body_center_and_radius(model, mesh, Some(body));
                draw_arrow_3d(cam, rect, painter, c, *remanence, 34.0, false, col, Some("B"));
            }
            Load::Inlet { .. } => {
                for (c, n) in anchors {
                    draw_arrow_3d(cam, rect, painter, c, n, 34.0, true, col, Some("In"));
                }
            }
            Load::Outlet { .. } => {
                for (c, n) in anchors {
                    draw_arrow_3d(cam, rect, painter, c, n, 34.0, false, col, Some("Out"));
                }
            }
            Load::MovingWall { velocity, .. } => {
                for (c, _) in anchors {
                    draw_arrow_3d(cam, rect, painter, c, *velocity, 30.0, false, col, Some("Vw"));
                }
            }
            Load::Port { polarization, .. } => {
                let mut e = [0.0; 3];
                e[polarization.idx()] = 1.0;
                for (c, _) in anchors {
                    draw_arrow_3d(cam, rect, painter, c, e, 30.0, false, col, Some("E"));
                }
            }
        }
    }
}

pub(super) fn draw_hover_highlight(
    app: &mut SimApp,
    cam: &Camera,
    rect: Rect,
    painter: &egui::Painter,
    mesh: &Mesh,
    hover_fid: Option<u32>,
) {
    let Some(fid) = hover_fid.filter(|&f| (f as usize) < mesh.faces.len()) else { return };
    let face = &mesh.faces[fid as usize];
    let body_mode = app.select_mode == SelectMode::Body;
    let target_id = if body_mode {
        mesh.elem_body.get(face.elem as usize).copied().unwrap_or(0) as u32
    } else {
        face.patch
    };
    let key = (app.mesh_ptr, body_mode, target_id);

    let rebuild = app.hover_cache.as_ref().is_none_or(|(k, _)| *k != key);
    if rebuild {
        let mut counts: HashMap<(u32, u32), (u8, u32, u32)> = HashMap::new();
        for f in &mesh.faces {
            let matches = if body_mode {
                mesh.elem_body.get(f.elem as usize).copied().unwrap_or(u16::MAX) as u32 == target_id
            } else {
                f.patch == target_id
            };
            if !matches {
                continue;
            }
            for i in 0..4 {
                let a = f.nodes[i];
                let b = f.nodes[(i + 1) % 4];
                let ek = if a < b { (a, b) } else { (b, a) };
                counts
                    .entry(ek)
                    .and_modify(|e| e.0 = e.0.saturating_add(1))
                    .or_insert((1, a, b));
            }
        }
        let edges: Vec<[V3; 2]> = counts
            .into_values()
            .filter(|(c, _, _)| *c == 1)
            .map(|(_, a, b)| [mesh.nodes[a as usize], mesh.nodes[b as usize]])
            .collect();
        app.hover_cache = Some((key, edges));
    }

    if let Some((_, edges)) = &app.hover_cache {
        let col = if body_mode { Color32::from_rgb(80, 195, 255) } else { Color32::from_rgb(255, 195, 65) };
        let stroke = Stroke::new(2.0, col);
        for &[a, b] in edges {
            let pa = world_to_screen(cam, rect, a);
            let pb = world_to_screen(cam, rect, b);
            if rect.contains(pa) || rect.contains(pb) {
                painter.line_segment([pa, pb], stroke);
            }
        }
    }
}

fn shape_bbox(s: &Shape) -> Option<(V3, V3)> {
    match s {
        Shape::Box { min, size } => Some((*min, add(*min, *size))),
        Shape::Cylinder { base, axis, radius, height } => {
            let ax = axis.idx();
            let (mut lo, mut hi) = ([base[0] - *radius, base[1] - *radius, base[2] - *radius], [base[0] + *radius, base[1] + *radius, base[2] + *radius]);
            lo[ax] = base[ax].min(base[ax] + *height);
            hi[ax] = base[ax].max(base[ax] + *height);
            Some((lo, hi))
        }
        Shape::Sphere { center, radius } => Some((sub(*center, [*radius; 3]), add(*center, [*radius; 3]))),
        Shape::Stl { data, offset, .. } => data.as_ref().map(|d| (add(d.min, *offset), add(d.max, *offset))),
        Shape::Pcb { data, offset, .. } => data.as_ref().map(|d| (*offset, add(*offset, [d.size[0], d.size[1], d.thickness]))),
    }
}

fn draw_wire_box(cam: &Camera, rect: Rect, painter: &egui::Painter, lo: V3, hi: V3, stroke: Stroke, fill: Option<Color32>) {
    let corners: [Pos2; 8] = std::array::from_fn(|i| {
        let p = [
            if i & 1 == 0 { lo[0] } else { hi[0] },
            if i & 2 == 0 { lo[1] } else { hi[1] },
            if i & 4 == 0 { lo[2] } else { hi[2] },
        ];
        world_to_screen(cam, rect, p)
    });
    if let Some(f) = fill {
        let faces = [[0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4], [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5]];
        for q in faces {
            painter.add(egui::Shape::convex_polygon(vec![corners[q[0]], corners[q[1]], corners[q[2]], corners[q[3]]], f, Stroke::NONE));
        }
    }
    let edges = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)];
    for (a, b) in edges {
        painter.line_segment([corners[a], corners[b]], stroke);
    }
}

fn draw_wire_cyl(cam: &Camera, rect: Rect, painter: &egui::Painter, base: V3, axis: usize, r: f64, h: f64, stroke: Stroke) {
    let u = (axis + 1) % 3;
    let v = (axis + 2) % 3;
    let n = 20usize;
    let ring = |z_off: f64| -> Vec<Pos2> {
        (0..=n)
            .map(|i| {
                let th = (i as f64 / n as f64) * std::f64::consts::TAU;
                let mut p = base;
                p[axis] += z_off;
                p[u] += r * th.cos();
                p[v] += r * th.sin();
                world_to_screen(cam, rect, p)
            })
            .collect()
    };
    let bot = ring(0.0);
    let top = ring(h);
    for pts in [&bot, &top] {
        for w in pts.windows(2) {
            painter.line_segment([w[0], w[1]], stroke);
        }
    }
    for i in [0, 5, 10, 15] {
        painter.line_segment([bot[i], top[i]], stroke);
    }
}

fn draw_wire_sphere(cam: &Camera, rect: Rect, painter: &egui::Painter, c: V3, r: f64, stroke: Stroke) {
    for (u, v) in [(0, 1), (1, 2), (0, 2)] {
        let pts: Vec<Pos2> = (0..=24)
            .map(|i| {
                let th = (i as f64 / 24.0) * std::f64::consts::TAU;
                let mut p = c;
                p[u] += r * th.cos();
                p[v] += r * th.sin();
                world_to_screen(cam, rect, p)
            })
            .collect();
        for w in pts.windows(2) {
            painter.line_segment([w[0], w[1]], stroke);
        }
    }
}

pub(super) fn draw_ghost_and_gizmo(
    app: &mut SimApp,
    ui: &egui::Ui,
    cam: &Camera,
    rect: Rect,
    painter: &egui::Painter,
    mesh: Option<&Mesh>,
) {
    let mut gizmo_anchor: Option<V3> = None;
    let ghost_stroke = Stroke::new(2.0, Color32::from_rgb(70, 210, 255));
    let sub_stroke = Stroke::new(2.0, Color32::from_rgb(255, 130, 65));
    let ghost_fill = Some(Color32::from_rgba_unmultiplied(70, 195, 255, 26));

    match app.form {
        Form::Box => {
            let lo = app.f.bmin;
            let hi = add(lo, app.f.bsize);
            draw_wire_box(cam, rect, painter, lo, hi, if app.f.subtract { sub_stroke } else { ghost_stroke }, ghost_fill);
            gizmo_anchor = Some(scale(add(lo, hi), 0.5));
        }
        Form::Cylinder => {
            draw_wire_cyl(cam, rect, painter, app.f.cbase, app.f.caxis, app.f.cradius, app.f.cheight, if app.f.subtract { sub_stroke } else { ghost_stroke });
            gizmo_anchor = Some(app.f.cbase);
        }
        Form::Sphere => {
            draw_wire_sphere(cam, rect, painter, app.f.scenter, app.f.sradius, if app.f.subtract { sub_stroke } else { ghost_stroke });
            gizmo_anchor = Some(app.f.scenter);
        }
        Form::Enclosure if app.f.enc_box => {
            let lo = app.f.enc_min;
            let hi = add(lo, app.f.enc_size);
            draw_wire_box(cam, rect, painter, lo, hi, ghost_stroke, ghost_fill);
            gizmo_anchor = Some(scale(add(lo, hi), 0.5));
        }
        Form::None => {
            if let Some((_, EditTarget::Body(bd))) = &app.edit_target {
                match &bd.shape {
                    Shape::Box { min, size } => {
                        let hi = add(*min, *size);
                        draw_wire_box(cam, rect, painter, *min, hi, ghost_stroke, None);
                        gizmo_anchor = Some(scale(add(*min, hi), 0.5));
                    }
                    Shape::Cylinder { base, axis, radius, height } => {
                        draw_wire_cyl(cam, rect, painter, *base, axis.idx(), *radius, *height, ghost_stroke);
                        gizmo_anchor = Some(*base);
                    }
                    Shape::Sphere { center, radius } => {
                        draw_wire_sphere(cam, rect, painter, *center, *radius, ghost_stroke);
                        gizmo_anchor = Some(*center);
                    }
                    Shape::Stl { .. } | Shape::Pcb { .. } => {
                        if let Some((lo, hi)) = shape_bbox(&bd.shape) {
                            draw_wire_box(cam, rect, painter, lo, hi, ghost_stroke, None);
                            gizmo_anchor = Some(scale(add(lo, hi), 0.5));
                        }
                    }
                }
                if app.xform.0 == "move" || app.xform.0 == "copy" {
                    let s = app.core.session.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                    if let Some(orig_b) = s.model.bodies.iter().find(|b| b.name == bd.orig_name) {
                        if let Some((lo, hi)) = shape_bbox(&orig_b.shape) {
                            let d = app.xform.1;
                            if norm(d) > 1e-6 {
                                draw_wire_box(cam, rect, painter, add(lo, d), add(hi, d), Stroke::new(1.5, Color32::from_rgb(245, 190, 60)), None);
                            }
                        }
                    }
                }
            }
        }
        _ => {}
    }

    // Also render PCB DC track hotspot pinpoint if present
    if let Some((pt, label)) = &app.highlight_point {
        let sp = world_to_screen(cam, rect, *pt);
        if rect.contains(sp) {
            painter.circle(sp, 7.0, Color32::from_rgba_unmultiplied(255, 85, 45, 220), Stroke::new(2.0, Color32::WHITE));
            painter.text(sp + vec2(10.0, -10.0), Align2::LEFT_CENTER, label, FontId::proportional(11.5), Color32::from_rgb(255, 190, 90));
        }
    }

    let Some(anchor) = gizmo_anchor else {
        app.gizmo_busy = false;
        return;
    };
    let origin_px = world_to_screen(cam, rect, anchor);
    if !rect.contains(origin_px) {
        app.gizmo_busy = false;
        return;
    }
    let axis_cols = [Color32::from_rgb(235, 75, 75), Color32::from_rgb(75, 195, 95), Color32::from_rgb(80, 140, 245)];
    let axis_names = ["X", "Y", "Z"];
    let handle_len_mm = (48.0 / cam.zoom.max(1e-6)).max(2.0);
    let mut busy = false;
    for k in 0..3 {
        let mut unit = [0.0; 3];
        unit[k] = 1.0;
        let tip_world = add(anchor, scale(unit, handle_len_mm));
        let tip_px = world_to_screen(cam, rect, tip_world);
        let tip_end = if (tip_px - origin_px).length() > 1.0 {
            origin_px + (tip_px - origin_px).normalized() * 44.0
        } else {
            continue;
        };
        let h_rect = Rect::from_center_size(tip_end, vec2(22.0, 22.0));
        let resp = ui.interact(h_rect, ui.scope_id().with(("gizmo", k)), Sense::click_and_drag())
            .on_hover_cursor(egui::CursorIcon::Grab)
            .on_hover_text(format!("Drag to move along {} axis", axis_names[k]));
        let active = resp.hovered() || resp.dragged();
        busy |= active;
        let col = if active { Color32::YELLOW } else { axis_cols[k] };
        draw_arrow_2d(painter, origin_px, tip_end, Stroke::new(if active { 3.0 } else { 2.2 }, col), 8.5);
        painter.text(tip_end + (tip_end - origin_px).normalized() * 10.0, Align2::CENTER_CENTER, axis_names[k], FontId::proportional(11.0), col);

        if resp.dragged() {
            let p1_unit = world_to_screen(cam, rect, add(anchor, unit));
            let s_unit = p1_unit - origin_px;
            let l2 = s_unit.length_sq();
            if l2 > 1e-6 {
                let d_mm = (s_unit.dot(resp.drag_delta()) / l2) as f64;
                if d_mm.abs() > 1e-9 && d_mm.is_finite() {
                    match app.form {
                        Form::Box => app.f.bmin[k] += d_mm,
                        Form::Cylinder => app.f.cbase[k] += d_mm,
                        Form::Sphere => app.f.scenter[k] += d_mm,
                        Form::Enclosure if app.f.enc_box => app.f.enc_min[k] += d_mm,
                        Form::None => {
                            if let Some((_, EditTarget::Body(bd))) = &mut app.edit_target {
                                match &mut bd.shape {
                                    Shape::Box { min, .. } => min[k] += d_mm,
                                    Shape::Cylinder { base, .. } => base[k] += d_mm,
                                    Shape::Sphere { center, .. } => center[k] += d_mm,
                                    Shape::Stl { offset, .. } | Shape::Pcb { offset, .. } => offset[k] += d_mm,
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    }
    app.gizmo_busy = busy;
    let _ = mesh;
}

pub(super) fn draw_view_cube(app: &mut SimApp, ui: &egui::Ui, rect: Rect, painter: &egui::Painter) {
    let center = pos2(rect.left() + 148.0, rect.bottom() - 52.0);
    if rect.width() < 260.0 {
        return;
    }
    let p = app.p;
    let card_fill = p.card.gamma_multiply(0.92);
    painter.circle(center, 40.0, card_fill, Stroke::new(1.0, p.border));

    let (r, u, f) = app.cam.basis();
    let eye = scale(f, -1.0);
    let project = |v: V3| -> Pos2 { center + vec2((dot(v, r) * 18.0) as f32, -(dot(v, u) * 18.0) as f32) };

    struct CubeFace {
        label: &'static str,
        view: &'static str,
        normal: V3,
        corners: [V3; 4],
    }
    let faces = [
        CubeFace { label: "FRONT", view: "front", normal: [0.0, -1.0, 0.0], corners: [[-1.0, -1.0, -1.0], [1.0, -1.0, -1.0], [1.0, -1.0, 1.0], [-1.0, -1.0, 1.0]] },
        CubeFace { label: "BACK", view: "back", normal: [0.0, 1.0, 0.0], corners: [[1.0, 1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, 1.0, 1.0], [1.0, 1.0, 1.0]] },
        CubeFace { label: "RIGHT", view: "right", normal: [1.0, 0.0, 0.0], corners: [[1.0, -1.0, -1.0], [1.0, 1.0, -1.0], [1.0, 1.0, 1.0], [1.0, -1.0, 1.0]] },
        CubeFace { label: "LEFT", view: "left", normal: [-1.0, 0.0, 0.0], corners: [[-1.0, 1.0, -1.0], [-1.0, -1.0, -1.0], [-1.0, -1.0, 1.0], [-1.0, 1.0, 1.0]] },
        CubeFace { label: "TOP", view: "top", normal: [0.0, 0.0, 1.0], corners: [[-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, 1.0], [-1.0, 1.0, 1.0]] },
        CubeFace { label: "BOT", view: "bottom", normal: [0.0, 0.0, -1.0], corners: [[-1.0, 1.0, -1.0], [1.0, 1.0, -1.0], [1.0, -1.0, -1.0], [-1.0, -1.0, -1.0]] },
    ];
    let mut visible: Vec<(f64, &CubeFace)> = faces
        .iter()
        .filter_map(|cf| {
            let d = dot(cf.normal, eye);
            (d > 0.05).then_some((d, cf))
        })
        .collect();
    visible.sort_by(|a, b| a.0.total_cmp(&b.0));

    let text_col = if app.dark { Color32::from_gray(230) } else { Color32::from_gray(30) };
    for (_, cf) in visible {
        let pts: Vec<Pos2> = cf.corners.iter().map(|c| project(*c)).collect();
        let fc = project(cf.normal);
        let hit = ui.interact(Rect::from_center_size(fc, vec2(20.0, 20.0)), ui.scope_id().with(("vc", cf.view)), Sense::click())
            .on_hover_text(format!("Orient view: {}", cf.label));
        let shade = if hit.hovered() {
            p.accent.gamma_multiply(0.35)
        } else if app.dark {
            Color32::from_rgb(56, 60, 70)
        } else {
            Color32::from_rgb(230, 234, 240)
        };
        painter.add(egui::Shape::convex_polygon(pts, shade, Stroke::new(1.0, p.border)));
        painter.text(fc, Align2::CENTER_CENTER, cf.label, FontId::proportional(8.0), if hit.hovered() { p.accent } else { text_col });
        if hit.clicked() {
            app.cam.set_view(cf.view);
            app.need_fit = true;
            ui.ctx().request_repaint();
        }
    }
    // Corner isometric quick-snap dots
    for (view, corner) in [("iso", [1.0, -1.0, 1.0]), ("iso_back", [-1.0, 1.0, 1.0])] {
        if dot(corner, eye) > 0.1 {
            let cp = project(corner);
            let cr = ui.interact(Rect::from_center_size(cp, vec2(12.0, 12.0)), ui.scope_id().with(("vci", view)), Sense::click())
                .on_hover_text("Snap to isometric view");
            painter.circle(cp, if cr.hovered() { 4.2 } else { 3.0 }, if cr.hovered() { p.accent } else { p.border }, Stroke::NONE);
            if cr.clicked() {
                app.cam.set_view(view);
                app.need_fit = true;
                ui.ctx().request_repaint();
            }
        }
    }
}

pub(super) fn draw_context_menu(app: &mut SimApp, ui: &egui::Ui, mesh: &Mesh, model: &Model) {
    let Some(pos) = app.ctx_menu_pos else { return };
    let mut close = false;
    let p = app.p;
    let hits = app.ctx_hits.clone();
    let resp = egui::Area::new(egui::Id::NULL.with("vp-ctx-menu"))
        .fixed_pos(pos)
        .order(egui::Order::Foreground)
        .show(ui.ctx(), |ui| {
            egui::Frame::popup(ui.style()).fill(p.card).stroke(Stroke::new(1.0, p.border)).show(ui, |ui| {
                ui.set_min_width(210.0);
                // Group depth-peel hits by patch (preserving front-to-back order)
                let mut seen_patches = HashSet::new();
                let mut unique_faces = Vec::new();
                for &(fid, t, _) in &hits {
                    if let Some(f) = mesh.faces.get(fid as usize) {
                        if seen_patches.insert(f.patch) {
                            let body_idx = mesh.elem_body.get(f.elem as usize).copied().unwrap_or(0) as usize;
                            let bname = mesh.body_names.get(body_idx).cloned().unwrap_or_default();
                            let pname = mesh.patches.get(f.patch as usize).map(|x| x.name.clone()).unwrap_or_default();
                            unique_faces.push((f.patch, bname, pname, t));
                            if unique_faces.len() >= 8 {
                                break;
                            }
                        }
                    }
                }
                if !unique_faces.is_empty() {
                    ui.label(egui::RichText::new("Select Behind (Depth Peel)").small().strong().color(p.weak));
                    for (idx, (patch, _bname, pname, _t)) in unique_faces.iter().enumerate() {
                        let lbl = if idx == 0 { format!("  {pname}") } else { format!("↳ {pname} (#{})", idx + 1) };
                        if ui.button(lbl).clicked() {
                            app.selected.clear();
                            app.selected.insert(*patch);
                            app.sel_version += 1;
                            close = true;
                        }
                    }
                    ui.separator();
                    let (_, front_body, _, _) = &unique_faces[0];
                    if !front_body.is_empty() {
                        if ui.button(format!("✏ Edit body '{front_body}'")).clicked() {
                            app.sel_body = Some(front_body.clone());
                            app.sel_load = None;
                            app.form = Form::None;
                            app.props_open = true;
                            close = true;
                        }
                        if ui.button(format!("👻 Ghost '{front_body}'")).clicked() {
                            app.set_body_vis(front_body, 1);
                            close = true;
                        }
                        if ui.button(format!("🚫 Hide '{front_body}'")).clicked() {
                            app.set_body_vis(front_body, 2);
                            close = true;
                        }
                        if model.bodies.len() > 1 && ui.button(format!("🎯 Isolate '{front_body}'")).clicked() {
                            app.isolate_body(front_body);
                            close = true;
                        }
                        ui.separator();
                    }
                }
                if !model.bodies.is_empty() && ui.button("👁 Show all hidden/ghost bodies").clicked() {
                    app.show_all_bodies();
                    close = true;
                }
                if ui.button("⛶ Fit view (F)").clicked() {
                    app.need_fit = true;
                    close = true;
                }
                if !unique_faces.is_empty() && ui.button("🎯 Apply Fixed / Load…").clicked() {
                    app.selected.clear();
                    app.selected.insert(unique_faces[0].0);
                    app.sel_version += 1;
                    app.tab = super::Tab::Setup;
                    app.form = Form::Load(LoadKind::Fixed);
                    app.props_open = true;
                    close = true;
                }
                let _ = json!({});
            });
        })
        .response;
    if close || (ui.input(|i| i.pointer.any_click()) && !resp.hovered()) {
        app.ctx_menu_pos = None;
    }
}
