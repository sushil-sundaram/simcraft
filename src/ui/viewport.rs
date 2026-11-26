//! Central 3D viewport: non-blocking meshing coordinator, hardware GL PaintCallback (with
//! software rasterizer fallback), BVH face picking, probes/path overlay, legend, and view toolbar.

use super::gl_viewport::{DrawParams, UploadKey};
use super::overlays::{draw_context_menu, draw_ghost_and_gizmo, draw_hover_highlight, draw_load_glyphs, draw_view_cube};
use super::{DrawerTab, Form, MeshJob, RenderKey, SelectMode, SimApp};
use crate::geometry::{add, dot, scale};
use crate::render;
use eframe::egui::{self, Align2, Color32, FontId, Pos2, Rect, RichText, Sense, Stroke, pos2, vec2};
use serde_json::json;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

impl SimApp {
    pub(super) fn viewport(&mut self, ui: &mut egui::Ui) {
        let p = self.p;
        let rect = ui.available_rect_before_wrap();
        let resp = ui.allocate_rect(rect, Sense::click_and_drag());
        let painter = ui.painter_at(rect);

        if let Some(job) = &self.mesh_job {
            let done = job.done.lock().unwrap_or_else(std::sync::PoisonError::into_inner).take();
            if let Some(res) = done {
                let finished = self.mesh_job.take().unwrap();
                let cancelled = finished.cancel.load(Ordering::Relaxed);
                match res {
                    Ok(m) if !cancelled => {
                        let mut s = self.core.session.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                        if s.mesh_key() == finished.key {
                            let _ = s.install_mesh(m, finished.key.0, (finished.key.1, finished.key.2));
                            self.mesh_err = None;
                        }
                    }
                    Ok(_) => {}
                    Err(_) if cancelled => {}
                    Err(e) => {
                        self.mesh_err = Some((finished.key, e));
                    }
                }
            }
        }

        let (mesh, model, results, display, rev, view_req, stale) = {
            let mut s = self.core.session.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            // nothing to mesh, or the mesh is current: no background job needed
            if s.model.bodies.is_empty() || s.mesh_valid() {
                if let Some(old) = self.mesh_job.take() {
                    old.cancel.store(true, Ordering::Relaxed);
                }
                self.mesh_err = None;
            } else {
                let key = s.mesh_key();
                if self.mesh_err.as_ref().is_some_and(|(k, _)| *k != key) {
                    self.mesh_err = None;
                }
                if self.mesh_err.is_none() && self.mesh_job.as_ref().is_none_or(|j| j.key != key) {
                    if let Some(old) = self.mesh_job.take() {
                        old.cancel.store(true, Ordering::Relaxed);
                    }
                    let model = s.model.clone();
                    let h = s.element_size();
                    let limit = crate::mesh::max_cells_for(&model);
                    let cancel = Arc::new(AtomicBool::new(false));
                    let done = Arc::new(std::sync::Mutex::new(None));
                    let (c2, d2, ctx2) = (cancel.clone(), done.clone(), ui.ctx().clone());
                    std::thread::spawn(move || {
                        let r = crate::mesh::build_for_cancellable(&model, h, limit, &c2).map_err(|e| format!("{e:#}"));
                        *d2.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = Some(r);
                        ctx2.request_repaint();
                    });
                    self.mesh_job = Some(MeshJob { key, cancel, done });
                }
            }
            let mesh = if s.model.bodies.is_empty() { None } else { s.mesh.clone() };
            (mesh, s.model.clone(), s.results.clone(), s.display.clone(), s.revision, s.view_request.take(), s.results_stale)
        };
        if self.mesh_job.is_some() {
            ui.ctx().request_repaint_after(std::time::Duration::from_millis(80));
        }
        self.cam.perspective = display.perspective;

        let (_, bb) = render::background(self.dark);
        let c32 = |c: [f32; 3]| Color32::from_rgb((c[0] * 255.0) as u8, (c[1] * 255.0) as u8, (c[2] * 255.0) as u8);
        let ghost_form = matches!(self.form, Form::Box | Form::Cylinder | Form::Sphere | Form::Enclosure);
        let Some(mesh) = mesh else {
            painter.rect_filled(rect, 0.0, c32(bb));
            if self.mesh_job.is_some() {
                painter.text(rect.center() - vec2(0.0, 22.0), Align2::CENTER_CENTER, "⚙  Meshing geometry…", FontId::proportional(20.0), p.text);
                painter.text(rect.center() + vec2(0.0, 4.0), Align2::CENTER_CENTER, "Window stays responsive while building the mesh", FontId::proportional(13.0), p.weak);
                let btn = Rect::from_center_size(rect.center() + vec2(0.0, 34.0), vec2(110.0, 26.0));
                if ui.put(btn, egui::Button::new("Cancel meshing")).clicked()
                    && let Some(j) = self.mesh_job.take() {
                        j.cancel.store(true, Ordering::Relaxed);
                        self.mesh_err = Some((j.key, "Meshing cancelled.".into()));
                    }
            } else if ghost_form {
                if resp.dragged_by(egui::PointerButton::Primary) && !ui.input(|i| i.key_down(egui::Key::Space)) {
                    let d = resp.drag_delta();
                    self.cam.orbit(d.x as f64, d.y as f64);
                } else if resp.dragged_by(egui::PointerButton::Secondary) || resp.dragged_by(egui::PointerButton::Middle) || (resp.dragged_by(egui::PointerButton::Primary) && ui.input(|i| i.key_down(egui::Key::Space))) {
                    let d = resp.drag_delta();
                    self.cam.pan(d.x as f64, d.y as f64);
                }
                draw_ghost_and_gizmo(self, ui, &self.cam.clone(), rect, &painter, None);
                draw_view_cube(self, ui, rect, &painter);
                painter.text(rect.left_top() + vec2(14.0, 14.0), Align2::LEFT_TOP, "Live ghost preview — drag X/Y/Z handles or click Create", FontId::proportional(13.0), p.weak);
            } else {
                let (title, sub) = match &self.mesh_err {
                    Some((_, e)) => ("⚠  Geometry / mesh notice".to_string(), e.clone()),
                    None => ("Start a new simulation".into(), "GEOMETRY ▸ Box / Cylinder / Import STL      or      Examples ▸ Cantilever bracket".into()),
                };
                painter.text(rect.center() - vec2(0.0, 14.0), Align2::CENTER_CENTER, title, FontId::proportional(22.0), p.text);
                painter.text(rect.center() + vec2(0.0, 16.0), Align2::CENTER_CENTER, sub, FontId::proportional(13.0), p.weak);
                if self.mesh_err.is_some() {
                    let btn = Rect::from_center_size(rect.center() + vec2(0.0, 46.0), vec2(100.0, 26.0));
                    if ui.put(btn, egui::Button::new("Retry mesh")).clicked() {
                        self.mesh_err = None;
                    }
                }
            }
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
            let reserve = if display.show_results && rect.width() > 760.0 { 240.0 } else { 0.0 };
            self.cam.fit(mesh.bbox.0, mesh.bbox.1, (rect.width() - reserve) as f64, rect.height() as f64);
            self.cam.pan(-(reserve as f64) / 2.0, 0.0);
            self.need_fit = false;
        }

        // ---- input (orbit / pan / zoom + trackpad Space+drag & pinch gestures)
        let dragging = resp.dragged() && !self.gizmo_busy;
        let space_down = ui.input(|i| i.key_down(egui::Key::Space));
        if resp.dragged_by(egui::PointerButton::Primary) && !self.gizmo_busy {
            let d = resp.drag_delta();
            if space_down {
                self.cam.pan(d.x as f64, d.y as f64);
            } else {
                self.cam.orbit(d.x as f64, d.y as f64);
            }
        } else if resp.dragged_by(egui::PointerButton::Secondary) || resp.dragged_by(egui::PointerButton::Middle) {
            let d = resp.drag_delta();
            self.cam.pan(d.x as f64, d.y as f64);
        }
        if resp.hovered() {
            let (scroll, mods, zoom_pinch, hover_pos) = ui.input(|i| (i.smooth_scroll_delta, i.modifiers, i.zoom_delta(), i.pointer.hover_pos()));
            let zoom_around = |cam: &mut render::Camera, factor: f64| {
                let old = cam.zoom;
                let new = (old * factor).clamp(1e-4, 1e6);
                if let Some(hp) = hover_pos.or(resp.hover_pos()) {
                    let (r, u, _) = cam.basis();
                    let k = 1.0 / old - 1.0 / new;
                    let ox = (hp.x - rect.center().x) as f64 * k;
                    let oy = (rect.center().y - hp.y) as f64 * k;
                    cam.target = add(cam.target, add(scale(r, ox), scale(u, oy)));
                }
                cam.zoom = new;
            };
            if (zoom_pinch - 1.0).abs() > 1e-4 {
                zoom_around(&mut self.cam, zoom_pinch as f64);
            }
            if scroll.x != 0.0 || scroll.y != 0.0 {
                // trackpad mode: two-finger scroll pans (Ctrl / Cmd + scroll zooms);
                // mouse mode: Shift + scroll pans, plain scroll zooms
                let pan = if self.trackpad_pan { !(mods.ctrl || mods.command) } else { mods.shift };
                if pan {
                    self.cam.pan(scroll.x as f64, scroll.y as f64);
                } else if scroll.y != 0.0 {
                    zoom_around(&mut self.cam, (scroll.y as f64 * 0.003).exp());
                }
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

        // ---- render (hardware GL viewport when available, else software rasterizer fallback)
        let mut gpu_scene_opt: Option<Arc<render::GpuScene>> = None;
        if let Some(gl_vp) = &self.gl_vp {
            let gkey = (rev, self.sel_version, mptr, self.dark);
            let gs = match &self.gpu_scene {
                Some((k, sc)) if *k == gkey => sc.clone(),
                _ => {
                    let sc = Arc::new(render::build_gpu_scene(&mesh, &model, results.as_deref(), &display, &self.selected));
                    self.gpu_scene = Some((gkey, sc.clone()));
                    sc
                }
            };
            self.legend = gs.range.map(|r| (r, gs.scale_used));
            painter.add(gl_vp.paint_callback(
                rect,
                DrawParams {
                    cam: self.cam,
                    w_pts: rect.width(),
                    h_pts: rect.height(),
                    anim,
                    show_mesh: display.show_mesh,
                    dark: self.dark,
                    translucent_enclosure: display.translucent_enclosure,
                    scene: gs.clone(),
                    upload_key: UploadKey { rev, sel: self.sel_version, mesh: mptr, dark: self.dark },
                },
            ));
            gpu_scene_opt = Some(gs);
        } else {
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
        }

        // ---- picking + hover + right-click depth peel
        let clicked_pos = resp.clicked().then(|| resp.interact_pointer_pos()).flatten();
        let right_clicked_pos = resp.secondary_clicked().then(|| resp.interact_pointer_pos()).flatten();
        let hover_pos = (!dragging).then(|| resp.hover_pos()).flatten();
        if let Some(rc_pos) = right_clicked_pos {
            self.ctx_menu_pos = Some(rc_pos);
            self.ctx_hits = if let Some(gs) = &gpu_scene_opt {
                gs.picker.pick_all(&self.cam, (rc_pos.x - rect.min.x) as f64, (rc_pos.y - rect.min.y) as f64, rect.width() as f64, rect.height() as f64, anim, false)
            } else {
                let gs = render::build_gpu_scene(&mesh, &model, results.as_deref(), &display, &self.selected);
                gs.picker.pick_all(&self.cam, (rc_pos.x - rect.min.x) as f64, (rc_pos.y - rect.min.y) as f64, rect.width() as f64, rect.height() as f64, anim, false)
            };
        }
        let (clicked_fid, hover_fid) = {
            let pick = |pos: Pos2| -> Option<u32> {
                if let Some(gs) = &gpu_scene_opt {
                    gs.picker.pick(&self.cam, (pos.x - rect.min.x) as f64, (pos.y - rect.min.y) as f64, rect.width() as f64, rect.height() as f64, anim, false).map(|(f, _)| f)
                } else {
                    let x = ((pos.x - rect.min.x) * self.render_scale) as usize;
                    let y = ((pos.y - rect.min.y) * self.render_scale) as usize;
                    let (bw, bh) = self.buf_size;
                    (x < bw && y < bh).then(|| self.face_buf[y * bw + x]).filter(|f| *f != u32::MAX)
                }
            };
            (clicked_pos.and_then(pick), hover_pos.and_then(pick))
        };
        let res_id = results.as_ref().map(|r| Arc::as_ptr(r) as u64).unwrap_or(0);
        let probing = self.probe_mode && display.show_results && results.is_some();
        let path_picking = self.path.picking > 0 && display.show_results && results.is_some();
        let cam = self.cam;
        let legend = self.legend;
        let node_screen = |n: usize| -> Pos2 {
            let mut q = mesh.nodes[n];
            if let (Some(r), Some((_, sc))) = (&results, legend)
                && let Some(d) = r.deformation(display.mode)
                    && d.len() == mesh.nodes.len() && sc > 0.0 {
                        q = add(q, scale(d[n], sc * anim));
                    }
            let (x, y, _) = cam.project(q, rect.width() as f64, rect.height() as f64);
            rect.min + vec2(x as f32, y as f32)
        };
        let world_screen = |q: [f64; 3]| -> Pos2 {
            let (x, y, _) = cam.project(q, rect.width() as f64, rect.height() as f64);
            rect.min + vec2(x as f32, y as f32)
        };
        if (probing || path_picking) && clicked_pos.is_some() {
            if let (Some(pos), Some(fid)) = (clicked_pos, clicked_fid) {
                let n = mesh.faces[fid as usize].nodes.iter().copied().min_by(|a, b| node_screen(*a as usize).distance(pos).total_cmp(&node_screen(*b as usize).distance(pos))).unwrap();
                if path_picking {
                    if self.path.picking == 1 {
                        self.path.from = mesh.nodes[n as usize];
                        self.path.picking = 2;
                        self.toast = Some(("Path start A set — click end point B on the model".into(), false, 4.0));
                    } else {
                        self.path.to = mesh.nodes[n as usize];
                        self.path.picking = 0;
                        self.path.tab = 1;
                        self.open_drawer(DrawerTab::Charts);
                        self.toast = Some(("Path end B set".into(), false, 3.0));
                    }
                } else {
                    self.probes.retain(|(_, rid)| *rid == res_id);
                    if !self.probes.iter().any(|(m, _)| *m == n) {
                        self.probes.push((n, res_id));
                        if self.probes.len() > 8 {
                            self.probes.remove(0);
                        }
                    }
                }
            }
        } else if clicked_pos.is_some() {
            let shift = ui.input(|i| i.modifiers.shift || i.modifiers.command);
            match clicked_fid {
                Some(fid) if self.select_mode == SelectMode::Body => {
                    let face = &mesh.faces[fid as usize];
                    let bi = mesh.elem_body.get(face.elem as usize).copied().unwrap_or(0) as usize;
                    if let Some(bname) = mesh.body_names.get(bi).cloned() {
                        if shift {
                            if !self.sel_bodies.remove(&bname) {
                                self.sel_bodies.insert(bname.clone());
                            }
                            self.sel_body = self.sel_bodies.iter().next().cloned();
                        } else {
                            self.sel_bodies.clear();
                            self.sel_bodies.insert(bname.clone());
                            self.sel_body = Some(bname.clone());
                            self.sel_load = None;
                            self.form = Form::None;
                            self.props_open = true;
                        }
                        self.selected.clear();
                        let active_bodies: std::collections::HashSet<&str> = self.sel_bodies.iter().map(String::as_str).collect();
                        for f2 in &mesh.faces {
                            let b2 = mesh.elem_body.get(f2.elem as usize).copied().unwrap_or(0) as usize;
                            if mesh.body_names.get(b2).is_some_and(|n| active_bodies.contains(n.as_str())) {
                                self.selected.insert(f2.patch);
                            }
                        }
                    }
                }
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
                None if !shift => {
                    self.selected.clear();
                    self.sel_bodies.clear();
                }
                None => {}
            }
            self.sel_version += 1;
        }
        let cam_copy = self.cam;
        draw_hover_highlight(self, &cam_copy, rect, &painter, &mesh, hover_fid);
        if self.show_glyphs && !display.show_results {
            draw_load_glyphs(&cam_copy, rect, &painter, &model, &mesh, self.sel_load, &display.hidden_bodies);
        }
        draw_ghost_and_gizmo(self, ui, &cam_copy, rect, &painter, Some(&mesh));
        let text_col = if self.dark { Color32::from_gray(235) } else { Color32::from_gray(25) };
        let card_fill = p.card.gamma_multiply(0.92);
        if let (Some(hp), Some(fid)) = (hover_pos, hover_fid)
            && let Some(pi) = mesh.patches.get(mesh.faces[fid as usize].patch as usize) {
                let bi = mesh.elem_body.get(mesh.faces[fid as usize].elem as usize).copied().unwrap_or(0) as usize;
                let tip = if self.select_mode == SelectMode::Body {
                    format!("Body: {}  ({})", mesh.body_names.get(bi).map(String::as_str).unwrap_or(""), pi.name)
                } else {
                    pi.name.clone()
                };
                let galley = painter.layout_no_wrap(tip, FontId::proportional(12.0), text_col);
                let r = Rect::from_min_size(hp + vec2(14.0, 12.0), galley.size() + vec2(12.0, 6.0));
                painter.rect(r, 5.0, card_fill, Stroke::new(1.0, p.border), egui::StrokeKind::Inside);
                painter.galley(r.min + vec2(6.0, 3.0), galley, text_col);
            }

        // ---- probe markers & path line
        if display.show_results
            && let Some(r) = &results {
                let f = if r.has(display.field) { display.field } else { r.available_fields()[0] };
                if let Some(vals) = r.scalar_at(&mesh, f, display.mode, display.cylindrical) {
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
                if (self.show_chart && self.path.tab == 1 && self.path.inited_res == res_id) || self.path.picking > 0 {
                    let a = world_screen(self.path.from);
                    let b = world_screen(self.path.to);
                    painter.line_segment([a, b], Stroke::new(2.5, p.accent));
                    for (pt, lbl, col) in [(a, "A", p.accent), (b, "B", p.ok)] {
                        painter.circle(pt, 5.0, col, Stroke::new(1.5, Color32::WHITE));
                        painter.text(pt + vec2(7.0, -14.0), Align2::LEFT_CENTER, lbl, FontId::monospace(12.0), Color32::WHITE);
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

        // ---- axis triad + interactive ViewCube + right-click context menu
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
        draw_view_cube(self, ui, rect, &painter);

        // ---- view & selection toolbar (top right)
        let has_enc = mesh.with_enclosure;
        let views = [("⛶", "fit", "Fit all (F)"), ("◩", "iso", "Isometric (1)"), ("▣", "front", "Front (2)"), ("⬒", "top", "Top (4)"), ("◨", "right", "Right (5)"), ("◧", "left", "Left (6)")];
        let extra_pills = 2 + if has_enc { 1 } else { 0 } + 1; // Face/Body, Persp, Glyphs, (Glass)
        // Layout by available width: one row top-right (wide), one row bottom-right beside the
        // triad and ViewCube (medium), or two stacked rows above them (narrow) — never over them.
        let views_w = views.len() as f32 * 30.0 + 8.0;
        let pills_w = extra_pills as f32 * 50.0 + 8.0;
        let corner = 200.0; // triad + ViewCube band at the bottom left
        let (tb, pb) = if rect.width() >= 760.0 || rect.width() >= views_w + pills_w + corner + 28.0 {
            let y = if rect.width() >= 760.0 { rect.top() + 14.0 } else { rect.bottom() - 48.0 };
            let tb = Rect::from_min_size(pos2(rect.right() - 14.0 - views_w - pills_w, y), vec2(views_w, 34.0));
            (tb, Rect::from_min_size(pos2(tb.right(), y), vec2(pills_w, 34.0)))
        } else {
            let x = |w: f32| (rect.right() - 14.0 - w).max(rect.left() + 8.0);
            let y0 = rect.bottom() - 104.0 - 2.0 * 40.0;
            (Rect::from_min_size(pos2(x(views_w), y0), vec2(views_w, 34.0)), Rect::from_min_size(pos2(x(pills_w), y0 + 40.0), vec2(pills_w, 34.0)))
        };
        if (tb.right() - pb.left()).abs() < 0.5 && (tb.top() - pb.top()).abs() < 0.5 {
            painter.rect(tb.union(pb), 10.0, card_fill, Stroke::new(1.0, p.border), egui::StrokeKind::Inside);
        } else {
            for r in [tb, pb] {
                painter.rect(r, 10.0, card_fill, Stroke::new(1.0, p.border), egui::StrokeKind::Inside);
            }
        }
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
        let mut ex = pb.min.x + 4.0;
        let pill = |ex: &mut f32, id: &str, lbl: &str, active: bool, tip: &str| -> bool {
            let pr = Rect::from_min_size(pos2(*ex, pb.min.y + 4.0), vec2(48.0, 26.0));
            let r = ui.interact(pr, ui.scope_id().with(id), Sense::click()).on_hover_text(tip);
            if active {
                painter.rect_filled(pr, 6.0, p.accent.gamma_multiply(0.22));
            } else if r.hovered() {
                painter.rect_filled(pr, 6.0, p.hover);
            }
            painter.text(pr.center(), Align2::CENTER_CENTER, lbl, FontId::proportional(11.5), if active { p.accent } else { text_col });
            *ex += 50.0;
            r.clicked()
        };
        if pill(&mut ex, "selmode", if self.select_mode == SelectMode::Body { "Body" } else { "Face" }, self.select_mode == SelectMode::Body, "Toggle Face / Body viewport selection") {
            self.select_mode = match self.select_mode {
                SelectMode::Face => SelectMode::Body,
                SelectMode::Body => SelectMode::Face,
            };
        }
        if pill(&mut ex, "glyphs", "Glyphs", self.show_glyphs, "Show / hide 3D load & BC glyphs") {
            self.show_glyphs = !self.show_glyphs;
        }
        if pill(&mut ex, "persp", if display.perspective { "Persp" } else { "Ortho" }, display.perspective, "Toggle Orthographic / Perspective projection (P)") {
            self.run("set_display", json!({"perspective": !display.perspective}));
        }
        if has_enc && pill(&mut ex, "glass", "Glass", display.translucent_enclosure, "Toggle translucent glass enclosure") {
            self.run("set_display", json!({"translucent_enclosure": !display.translucent_enclosure}));
        }
        draw_context_menu(self, ui, &mesh, &model);

        if self.mesh_job.is_some() {
            let pill = Rect::from_center_size(pos2(rect.center().x, rect.bottom() - 24.0), vec2(210.0, 28.0));
            painter.rect(pill, 8.0, card_fill, Stroke::new(1.0, p.border), egui::StrokeKind::Inside);
            painter.text(pill.left_center() + vec2(12.0, 0.0), Align2::LEFT_CENTER, "⚙  Updating mesh…", FontId::proportional(12.5), text_col);
            let cbtn = Rect::from_min_size(pos2(pill.right() - 62.0, pill.top() + 4.0), vec2(54.0, 20.0));
            if ui.put(cbtn, egui::Button::new(RichText::new("Cancel").small())).clicked()
                && let Some(j) = self.mesh_job.take() {
                    j.cancel.store(true, Ordering::Relaxed);
                    self.mesh_err = Some((j.key, "Meshing cancelled.".into()));
                }
        }
    }
}
