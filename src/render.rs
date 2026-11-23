//! Software renderer shared by the GUI viewport and the MCP `render_image` tool.
//! Z-buffered rasterisation, ANSYS-style banded contours, mesh edges and a face-id buffer
//! used for picking.

use rayon::prelude::*;
use crate::geometry::*;
use crate::mesh::Mesh;
use crate::model::{Load, Model};
use crate::results::{Field, Results};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct Camera {
    pub yaw: f64,
    pub pitch: f64,
    /// pixels per mm
    pub zoom: f64,
    pub target: V3,
}

impl Default for Camera {
    fn default() -> Self {
        Camera { yaw: -60f64.to_radians(), pitch: 28f64.to_radians(), zoom: 4.0, target: [0.0; 3] }
    }
}

pub const VIEWS: [&str; 8] = ["iso", "front", "back", "left", "right", "top", "bottom", "iso_back"];

impl Camera {
    pub fn set_view(&mut self, name: &str) -> bool {
        let (y, p) = match name {
            "iso" => (-60.0, 28.0),
            "iso_back" => (120.0, 28.0),
            "front" => (-90.0, 0.0),
            "back" => (90.0, 0.0),
            "right" => (0.0, 0.0),
            "left" => (180.0, 0.0),
            "top" => (-90.0, 89.9),
            "bottom" => (-90.0, -89.9),
            _ => return false,
        };
        self.yaw = f64::to_radians(y);
        self.pitch = f64::to_radians(p);
        true
    }
    pub fn basis(&self) -> (V3, V3, V3) {
        let p = self.pitch.clamp(-1.5706, 1.5706);
        let eye = [p.cos() * self.yaw.cos(), p.cos() * self.yaw.sin(), p.sin()];
        let fwd = scale(eye, -1.0);
        let right = normalize(cross(fwd, [0.0, 0.0, 1.0]));
        let up = cross(right, fwd);
        (right, up, fwd)
    }
    pub fn fit(&mut self, lo: V3, hi: V3, w: f64, h: f64) {
        self.target = scale(add(lo, hi), 0.5);
        let (r, u, _) = self.basis();
        let (mut sx, mut sy) = (1e-9f64, 1e-9f64);
        for i in 0..8 {
            let c = [if i & 1 == 0 { lo[0] } else { hi[0] }, if i & 2 == 0 { lo[1] } else { hi[1] }, if i & 4 == 0 { lo[2] } else { hi[2] }];
            let d = sub(c, self.target);
            sx = sx.max(dot(d, r).abs());
            sy = sy.max(dot(d, u).abs());
        }
        self.zoom = 0.84 * (w / (2.0 * sx)).min(h / (2.0 * sy));
    }
    pub fn project(&self, p: V3, w: f64, h: f64) -> (f64, f64, f64) {
        let (r, u, f) = self.basis();
        let d = sub(p, self.target);
        (w / 2.0 + dot(d, r) * self.zoom, h / 2.0 - dot(d, u) * self.zoom, dot(d, f))
    }
    pub fn orbit(&mut self, dx: f64, dy: f64) {
        self.yaw -= dx * 0.008;
        self.pitch = (self.pitch + dy * 0.008).clamp(-1.5706, 1.5706);
    }
    pub fn pan(&mut self, dx: f64, dy: f64) {
        let (r, u, _) = self.basis();
        self.target = add(self.target, add(scale(r, -dx / self.zoom), scale(u, dy / self.zoom)));
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Display {
    /// Show results contour (true) or the model with supports/loads (false).
    pub show_results: bool,
    pub field: Field,
    /// Mode index (0-based) for modal results.
    pub mode: usize,
    /// None = automatic (ANSYS-like ~5% of model size); 1.0 = true scale; 0 = undeformed.
    pub deformation_scale: Option<f64>,
    pub show_mesh: bool,
    pub show_undeformed: bool,
    /// Dark viewport / UI theme.
    #[serde(default)]
    pub dark: bool,
    /// Cut-plane view through the domain.
    #[serde(default)]
    pub section: Option<Section>,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct Section {
    /// 0 = x, 1 = y, 2 = z
    pub axis: usize,
    /// 0..1 across the domain
    pub position: f64,
}
impl Default for Display {
    fn default() -> Self {
        Display { show_results: false, field: Field::TotalDeformation, mode: 0, deformation_scale: None, show_mesh: true, show_undeformed: false, dark: false, section: None }
    }
}

/// ANSYS-like 9-band rainbow.
pub const BANDS: [[u8; 3]; 9] = [
    [0, 0, 255],
    [0, 178, 255],
    [0, 255, 255],
    [0, 255, 178],
    [0, 255, 0],
    [178, 255, 0],
    [255, 255, 0],
    [255, 178, 0],
    [255, 0, 0],
];

pub fn band_color(t: f64) -> [u8; 3] {
    let i = ((t.clamp(0.0, 1.0) * 9.0).floor() as usize).min(8);
    BANDS[i]
}

pub fn load_color(l: &Load) -> [u8; 3] {
    match l {
        Load::FixedSupport { .. } => [70, 90, 220],
        Load::Displacement { .. } => [150, 90, 220],
        Load::Force { .. } => [225, 40, 40],
        Load::Pressure { .. } => [235, 120, 20],
        Load::Temperature { .. } => [230, 60, 120],
        Load::HeatFlux { .. } => [230, 190, 20],
        Load::Convection { .. } => [20, 180, 200],
        Load::CompressionOnly { .. } => [40, 160, 230],
        Load::Symmetry { .. } => [120, 130, 210],
        Load::Voltage { .. } => [170, 60, 220],
        Load::Current { .. } => [210, 80, 190],
        Load::Inlet { .. } => [40, 190, 90],
        Load::Outlet { .. } => [150, 200, 40],
        Load::Coil { .. } => [200, 120, 40],
        Load::Magnet { .. } => [200, 50, 80],
        _ => [160, 160, 160],
    }
}

pub struct Scene {
    /// Quad corner positions (mm, deformed) — 4 per face.
    pub pos: Vec<V3>,
    /// Normalised scalar per corner (NaN = use colour).
    pub t: Vec<f64>,
    pub color: Vec<[u8; 3]>,
    pub face: Vec<u32>,
    /// Per quad: 0 = solid surface, 1 = enclosure wall (drawn only from inside), 2 = section cut.
    pub kind: Vec<u8>,
    pub undeformed: Vec<V3>,
    pub range: Option<(f64, f64)>,
    /// Colour scale reversed (low values red) — see `Field::reversed_scale`.
    pub reversed: bool,
    pub scale_used: f64,
    pub lo: V3,
    pub hi: V3,
}

/// Automatic deformation scale (max displacement shown as ~5% of model diagonal).
pub fn auto_scale(mesh: &Mesh, d: &[V3]) -> f64 {
    let maxd = d.iter().map(|v| norm(*v)).fold(0.0, f64::max);
    if maxd < 1e-300 {
        return 0.0;
    }
    let diag = norm(sub(mesh.bbox.1, mesh.bbox.0));
    let s = 0.05 * diag / maxd;
    // round to 2 significant digits like ANSYS ("2.4e+002 (Auto Scale)")
    let p = 10f64.powf(s.log10().floor() - 1.0);
    (s / p).round() * p
}

pub fn build_scene(
    mesh: &Mesh,
    model: &Model,
    results: Option<&Results>,
    disp: &Display,
    highlight: &HashSet<u32>,
    anim: f64,
) -> Scene {
    let use_res = disp.show_results && results.is_some();
    let mut range = None;
    let mut values: Option<Vec<f64>> = None;
    let mut defl: Option<&[V3]> = None;
    let mut scale_used = 0.0;
    if use_res {
        let r = results.unwrap();
        let field = if r.has(disp.field) { disp.field } else { r.available_fields().first().copied().unwrap_or(Field::TotalDeformation) };
        if let Some(v) = r.scalar(field, disp.mode) {
            let mn = v.iter().cloned().fold(f64::INFINITY, f64::min);
            let mx = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            range = Some((mn, mx));
            values = Some(v);
        }
        if let Some(d) = r.deformation(disp.mode) {
            if d.len() == mesh.nodes.len() {
                scale_used = disp.deformation_scale.unwrap_or_else(|| auto_scale(mesh, d));
                defl = Some(d);
            }
        }
    }
    // face -> load overlay colour
    let mut face_color: Vec<Option<[u8; 3]>> = vec![None; mesh.faces.len()];
    if !use_res {
        for it in &model.loads {
            if let Ok(fs) = mesh.resolve_faces(it.load.faces()) {
                for f in fs {
                    face_color[f as usize] = Some(load_color(&it.load));
                }
            }
        }
    }
    let n = mesh.faces.len() * 4;
    let mut s = Scene {
        pos: Vec::with_capacity(n),
        t: Vec::with_capacity(n),
        color: Vec::with_capacity(n),
        face: Vec::with_capacity(mesh.faces.len()),
        kind: Vec::with_capacity(mesh.faces.len()),
        undeformed: Vec::new(),
        range,
        reversed: disp.field.reversed_scale(),
        scale_used,
        lo: [f64::INFINITY; 3],
        hi: [f64::NEG_INFINITY; 3],
    };
    if !use_res {
        for it in &model.loads {
            if let Some(b) = it.load.body() {
                if let Ok(els) = mesh.body_elems(b) {
                    let set: HashSet<u32> = els.iter().map(|&e| e as u32).collect();
                    for (fi, f) in mesh.faces.iter().enumerate() {
                        if !f.outer && set.contains(&f.elem) && face_color[fi].is_none() {
                            face_color[fi] = Some(load_color(&it.load));
                        }
                    }
                }
            }
        }
    }
    let section = disp.section.map(|sec| {
        let a = sec.axis.min(2);
        (a, mesh.bbox.0[a] + sec.position.clamp(0.0, 1.0) * (mesh.bbox.1[a] - mesh.bbox.0[a]))
    });
    let defl = if section.is_some() { None } else { defl };
    let rev = disp.field.reversed_scale();
    let norm_t = |v: f64| {
        let t = match range {
            Some((mn, mx)) if mx - mn > 1e-30 * mx.abs().max(1.0) => (v - mn) / (mx - mn),
            _ => 0.5,
        };
        if rev { 1.0 - t } else { t }
    };
    let k = scale_used * anim;
    for (fi, f) in mesh.faces.iter().enumerate() {
        if let Some((a, pc)) = section {
            // cut away the half in front of the plane
            if !f.outer && mesh.elem_center(f.elem as usize)[a] < pc {
                continue;
            }
        }
        let hl = highlight.contains(&f.patch);
        for &nid in &f.nodes {
            let base = mesh.nodes[nid as usize];
            let p = match defl {
                Some(d) => add(base, scale(d[nid as usize], k)),
                None => base,
            };
            for c in 0..3 {
                s.lo[c] = s.lo[c].min(p[c]);
                s.hi[c] = s.hi[c].max(p[c]);
            }
            s.pos.push(p);
            if disp.show_undeformed && defl.is_some() {
                s.undeformed.push(base);
            }
            let (t, col) = match (&values, range) {
                (Some(v), Some(_)) => {
                    let t = norm_t(v[nid as usize]);
                    (t, [0, 0, 0])
                }
                _ => {
                    let base = if f.outer { if disp.dark { [70, 78, 92] } else { [214, 222, 234] } } else { [178, 182, 190] };
                    let c = if hl { [40, 200, 90] } else { face_color[fi].unwrap_or(base) };
                    (f64::NAN, c)
                }
            };
            s.t.push(t);
            s.color.push(col);
        }
        s.face.push(fi as u32);
        s.kind.push(if f.outer { 1 } else { 0 });
    }
    // section quads through every element cut by the plane
    if let Some((a, pc)) = section {
        let (b, c) = ((a + 1) % 3, (a + 2) % 3);
        let h = mesh.h;
        for (e, en) in mesh.elems.iter().enumerate() {
            let lower = mesh.origin[a] + mesh.elem_cell[e][a] as f64 * h;
            if !(pc >= lower && pc < lower + h) {
                continue;
            }
            let w = (pc - lower) / h;
            for (ob, oc) in [(0usize, 0usize), (1, 0), (1, 1), (0, 1)] {
                let pick = |da: usize| {
                    let mut d = [0usize; 3];
                    d[a] = da;
                    d[b] = ob;
                    d[c] = oc;
                    en[crate::mesh::CORNER.iter().position(|q| *q == d).unwrap()] as usize
                };
                let (n0, n1) = (pick(0), pick(1));
                let mut pos = mesh.nodes[n0];
                pos[a] = pc;
                s.pos.push(pos);
                for q in 0..3 {
                    s.lo[q] = s.lo[q].min(pos[q]);
                    s.hi[q] = s.hi[q].max(pos[q]);
                }
                match &values {
                    Some(v) => {
                        s.t.push(norm_t((1.0 - w) * v[n0] + w * v[n1]));
                        s.color.push([0, 0, 0]);
                    }
                    None => {
                        s.t.push(f64::NAN);
                        let solid = mesh.elem_region[e] == crate::mesh::REGION_SOLID;
                        s.color.push(if solid { [160, 166, 176] } else if disp.dark { [60, 90, 120] } else { [190, 215, 240] });
                    }
                }
            }
            s.face.push(u32::MAX);
            s.kind.push(2);
        }
    }
    s
}

pub struct Frame {
    pub w: usize,
    pub h: usize,
    pub rgba: Vec<u8>,
    pub face_id: Vec<u32>,
}

pub fn background(dark: bool) -> ([f32; 3], [f32; 3]) {
    if dark { ([0.17, 0.19, 0.23], [0.075, 0.08, 0.095]) } else { ([0.80, 0.85, 0.92], [0.975, 0.98, 0.99]) }
}

pub fn rasterize(scene: &Scene, cam: &Camera, w: usize, h: usize, show_mesh: bool, dark: bool) -> Frame {
    let (bg_top, bg_bottom) = background(dark);
    let mut rgba = vec![0u8; w * h * 4];
    rgba.par_chunks_mut(w * 4).enumerate().for_each(|(y, row)| {
        let t = y as f32 / h.max(1) as f32;
        let c: Vec<u8> = (0..3).map(|k| ((bg_top[k] * (1.0 - t) + bg_bottom[k] * t) * 255.0) as u8).collect();
        for px in row.chunks_mut(4) {
            px[..3].copy_from_slice(&c);
            px[3] = 255;
        }
    });
    let mut zbuf = vec![f64::INFINITY; w * h];
    let mut face_id = vec![u32::MAX; w * h];
    let (_, _, fwd) = cam.basis();
    let light = normalize(sub(scale(fwd, -1.0), add(scale(cam.basis().0, 0.35), scale(cam.basis().1, -0.45))));
    let (wf, hf) = (w as f64, h as f64);
    let proj: Vec<(f64, f64, f64)> = scene.pos.par_iter().map(|p| cam.project(*p, wf, hf)).collect();

    // ghost of undeformed shape (wireframe-ish) drawn first, faint
    if !scene.undeformed.is_empty() {
        for q in scene.undeformed.chunks(4) {
            for e in 0..4 {
                let a = cam.project(q[e], wf, hf);
                let b = cam.project(q[(e + 1) % 4], wf, hf);
                draw_line(&mut rgba, w, h, a.0, a.1, b.0, b.1, [120, 120, 130]);
            }
        }
    }

    // per-quad setup (visibility, shading, edges, screen rows) in parallel
    struct Q {
        qi: usize,
        shade: f64,
        draw_edges: bool,
        thick: f64,
        y0: i64,
        y1: i64,
    }
    let quads: Vec<Q> = proj
        .par_chunks(4)
        .enumerate()
        .filter_map(|(qi, quad)| {
            let p = &scene.pos[qi * 4..qi * 4 + 4];
            let nrm = normalize(cross(sub(p[1], p[0]), sub(p[3], p[0])));
            match scene.kind.get(qi).copied().unwrap_or(0) {
                0 if dot(nrm, fwd) > 0.05 => return None,  // back face of a solid
                1 if dot(nrm, fwd) < -0.05 => return None, // near enclosure wall: see through it
                _ => {}
            }
            let shade = 0.38 + 0.62 * dot(nrm, light).abs();
            let edge_px = {
                let a = (quad[1].0 - quad[0].0).hypot(quad[1].1 - quad[0].1);
                let b = (quad[3].0 - quad[0].0).hypot(quad[3].1 - quad[0].1);
                a.min(b).max(1e-6)
            };
            let y0 = quad.iter().map(|a| a.1).fold(f64::INFINITY, f64::min).floor().max(0.0) as i64;
            let y1 = quad.iter().map(|a| a.1).fold(f64::NEG_INFINITY, f64::max).ceil().min(hf - 1.0) as i64;
            (y0 <= y1).then_some(Q { qi, shade, draw_edges: show_mesh && edge_px > 6.0, thick: 0.9 / edge_px, y0, y1 })
        })
        .collect();
    // horizontal bands rendered in parallel; each band owns its rows of every buffer
    let band = (h / (rayon::current_num_threads() * 4).max(1)).max(8);
    let nb = h.div_ceil(band);
    let mut bins: Vec<Vec<u32>> = vec![Vec::new(); nb];
    for (k, q) in quads.iter().enumerate() {
        for b in (q.y0 as usize / band)..=((q.y1 as usize / band).min(nb - 1)) {
            bins[b].push(k as u32);
        }
    }
    rgba.par_chunks_mut(band * w * 4)
        .zip(zbuf.par_chunks_mut(band * w))
        .zip(face_id.par_chunks_mut(band * w))
        .enumerate()
        .for_each(|(bi, ((rgba, zbuf), face_id))| {
            let row0 = (bi * band) as i64;
            let row1 = row0 + (zbuf.len() / w) as i64 - 1;
            let uv = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
            for &k in &bins[bi] {
                let q = &quads[k as usize];
                let qi = q.qi;
                let quad = &proj[qi * 4..qi * 4 + 4];
                for tri in [[0usize, 1, 2], [0, 2, 3]] {
                    let v = [quad[tri[0]], quad[tri[1]], quad[tri[2]]];
                    let minx = v.iter().map(|a| a.0).fold(f64::INFINITY, f64::min).floor().max(0.0) as i64;
                    let maxx = v.iter().map(|a| a.0).fold(f64::NEG_INFINITY, f64::max).ceil().min(wf - 1.0) as i64;
                    let miny = (v.iter().map(|a| a.1).fold(f64::INFINITY, f64::min).floor().max(0.0) as i64).max(row0);
                    let maxy = (v.iter().map(|a| a.1).fold(f64::NEG_INFINITY, f64::max).ceil().min(hf - 1.0) as i64).min(row1);
                    if minx > maxx || miny > maxy {
                        continue;
                    }
                    let area = (v[1].0 - v[0].0) * (v[2].1 - v[0].1) - (v[2].0 - v[0].0) * (v[1].1 - v[0].1);
                    if area.abs() < 1e-12 {
                        continue;
                    }
                    for py in miny..=maxy {
                        for px in minx..=maxx {
                            let (x, y) = (px as f64 + 0.5, py as f64 + 0.5);
                            let w0 = ((v[1].0 - x) * (v[2].1 - y) - (v[2].0 - x) * (v[1].1 - y)) / area;
                            let w1 = ((v[2].0 - x) * (v[0].1 - y) - (v[0].0 - x) * (v[2].1 - y)) / area;
                            let w2 = 1.0 - w0 - w1;
                            if w0 < -1e-9 || w1 < -1e-9 || w2 < -1e-9 {
                                continue;
                            }
                            let z = w0 * v[0].2 + w1 * v[1].2 + w2 * v[2].2;
                            let idx = (py - row0) as usize * w + px as usize;
                            if z >= zbuf[idx] {
                                continue;
                            }
                            zbuf[idx] = z;
                            face_id[idx] = scene.face[qi];
                            let g = |k: usize| qi * 4 + tri[k];
                            let t0 = scene.t[g(0)];
                            let mut col = if t0.is_nan() {
                                let c = scene.color[g(0)];
                                [c[0] as f64, c[1] as f64, c[2] as f64]
                            } else {
                                let t = w0 * t0 + w1 * scene.t[g(1)] + w2 * scene.t[g(2)];
                                let c = band_color(t);
                                [c[0] as f64, c[1] as f64, c[2] as f64]
                            };
                            for c in col.iter_mut() {
                                *c *= q.shade;
                            }
                            if q.draw_edges {
                                let u = w0 * uv[tri[0]].0 + w1 * uv[tri[1]].0 + w2 * uv[tri[2]].0;
                                let vv = w0 * uv[tri[0]].1 + w1 * uv[tri[1]].1 + w2 * uv[tri[2]].1;
                                let e = u.min(1.0 - u).min(vv).min(1.0 - vv);
                                if e < q.thick {
                                    for c in col.iter_mut() {
                                        *c *= 0.55;
                                    }
                                }
                            }
                            let o = idx * 4;
                            rgba[o] = col[0] as u8;
                            rgba[o + 1] = col[1] as u8;
                            rgba[o + 2] = col[2] as u8;
                        }
                    }
                }
            }
        });
    Frame { w, h, rgba, face_id }
}

fn draw_line(img: &mut [u8], w: usize, h: usize, x0: f64, y0: f64, x1: f64, y1: f64, c: [u8; 3]) {
    let n = ((x1 - x0).abs().max((y1 - y0).abs()).ceil() as usize).max(1);
    for i in 0..=n {
        let t = i as f64 / n as f64;
        let x = (x0 + (x1 - x0) * t) as i64;
        let y = (y0 + (y1 - y0) * t) as i64;
        if x >= 0 && y >= 0 && (x as usize) < w && (y as usize) < h {
            let o = (y as usize * w + x as usize) * 4;
            img[o..o + 3].copy_from_slice(&c);
        }
    }
}

// ---------------- PNG output with legend ----------------

const FONT: &[(char, [u8; 7])] = &[
    ('0', [0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E]),
    ('1', [0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E]),
    ('2', [0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F]),
    ('3', [0x1F, 0x02, 0x04, 0x02, 0x01, 0x11, 0x0E]),
    ('4', [0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02]),
    ('5', [0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E]),
    ('6', [0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E]),
    ('7', [0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08]),
    ('8', [0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E]),
    ('9', [0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C]),
    ('.', [0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C]),
    ('-', [0x00, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x00]),
    ('+', [0x00, 0x04, 0x04, 0x1F, 0x04, 0x04, 0x00]),
    ('e', [0x00, 0x00, 0x0E, 0x11, 0x1F, 0x10, 0x0E]),
    ('M', [0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11]),
    ('P', [0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10]),
    ('a', [0x00, 0x00, 0x0E, 0x01, 0x0F, 0x11, 0x0F]),
    ('m', [0x00, 0x00, 0x1A, 0x15, 0x15, 0x11, 0x11]),
    ('C', [0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E]),
    ('W', [0x11, 0x11, 0x11, 0x15, 0x15, 0x15, 0x0A]),
    ('/', [0x01, 0x01, 0x02, 0x04, 0x08, 0x10, 0x10]),
    ('2', [0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F]),
    ('H', [0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11]),
    ('z', [0x00, 0x00, 0x1F, 0x02, 0x04, 0x08, 0x1F]),
    ('s', [0x00, 0x00, 0x0F, 0x10, 0x0E, 0x01, 0x1E]),
    ('V', [0x11, 0x11, 0x11, 0x11, 0x11, 0x0A, 0x04]),
    ('A', [0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11]),
    ('T', [0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04]),
    ('K', [0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11]),
    ('J', [0x07, 0x02, 0x02, 0x02, 0x02, 0x12, 0x0C]),
    ('N', [0x11, 0x19, 0x15, 0x13, 0x11, 0x11, 0x11]),
    ('°', [0x0C, 0x12, 0x12, 0x0C, 0x00, 0x00, 0x00]),
    ('²', [0x0C, 0x02, 0x04, 0x0E, 0x00, 0x00, 0x00]),
    ('³', [0x0E, 0x02, 0x06, 0x02, 0x0E, 0x00, 0x00]),
    (' ', [0; 7]),
];

fn draw_text(img: &mut [u8], w: usize, h: usize, x: usize, y: usize, s: &str, sc: usize, col: [u8; 3]) {
    let mut cx = x;
    for ch in s.chars() {
        if let Some((_, rows)) = FONT.iter().find(|(c, _)| *c == ch) {
            for (ry, row) in rows.iter().enumerate() {
                for rx in 0..5 {
                    if row & (0x10 >> rx) != 0 {
                        for dy in 0..sc {
                            for dx in 0..sc {
                                let (px, py) = (cx + rx * sc + dx, y + ry * sc + dy);
                                if px < w && py < h {
                                    let o = (py * w + px) * 4;
                                    img[o..o + 3].copy_from_slice(&col);
                                }
                            }
                        }
                    }
                }
            }
        }
        cx += 6 * sc;
    }
}

pub fn fmt_val(v: f64) -> String {
    if v == 0.0 {
        return "0".into();
    }
    let a = v.abs();
    if (1e-3..1e5).contains(&a) {
        let d = (4 - a.log10().floor() as i32).clamp(0, 6) as usize;
        format!("{:.*}", d, v)
    } else {
        format!("{:.3e}", v)
    }
}

pub fn render_png(scene: &Scene, cam: &Camera, w: usize, h: usize, show_mesh: bool, unit: &str, dark: bool) -> Result<Vec<u8>> {
    let tc = if dark { [225, 228, 235] } else { [20, 20, 30] };
    let ss = 2; // supersampling
    let mut c2 = *cam;
    c2.zoom *= ss as f64;
    let big = rasterize(scene, &c2, w * ss, h * ss, show_mesh, dark);
    let mut img = vec![0u8; w * h * 4];
    for y in 0..h {
        for x in 0..w {
            for c in 0..4 {
                let mut acc = 0u32;
                for dy in 0..ss {
                    for dx in 0..ss {
                        acc += big.rgba[((y * ss + dy) * w * ss + x * ss + dx) * 4 + c] as u32;
                    }
                }
                img[(y * w + x) * 4 + c] = (acc / (ss * ss) as u32) as u8;
            }
        }
    }
    if let Some((mn, mx)) = scene.range {
        let (lx, ly, bw, bh) = (14usize, 30usize, 22usize, 22usize);
        let sc = if w >= 900 { 2 } else { 1 };
        for b in 0..9 {
            let col = if scene.reversed { BANDS[b] } else { BANDS[8 - b] };
            for y in ly + b * bh..ly + (b + 1) * bh {
                for x in lx..lx + bw {
                    if x < w && y < h {
                        let o = (y * w + x) * 4;
                        img[o..o + 3].copy_from_slice(&col);
                    }
                }
            }
        }
        for b in 0..=9 {
            let mut v = mx - (mx - mn) * b as f64 / 9.0;
            if v.abs() < 1e-9 * mx.abs().max(mn.abs()) {
                v = 0.0;
            }
            draw_text(&mut img, w, h, lx + bw + 6, (ly + b * bh).saturating_sub(3), &fmt_val(v), sc, tc);
        }
        draw_text(&mut img, w, h, lx, 10, unit, sc, tc);
    }
    let mut out = Vec::new();
    {
        let mut enc = png::Encoder::new(&mut out, w as u32, h as u32);
        enc.set_color(png::ColorType::Rgba);
        enc.set_depth(png::BitDepth::Eight);
        let mut wr = enc.write_header()?;
        wr.write_image_data(&img)?;
    }
    Ok(out)
}
