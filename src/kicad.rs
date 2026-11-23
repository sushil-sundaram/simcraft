//! KiCad PCB import (`.kicad_pcb`, KiCad 5–10 S-expression format).
//!
//! The board becomes one body (`Shape::Pcb`): the Edge.Cuts outline (with cutouts) extruded over
//! the stack-up thickness. Copper (tracks, arcs, pads, filled zones, via rings) is rasterised per
//! copper layer at ~0.1 mm; via barrels are kept as a plated-copper area map. 35 µm copper cannot
//! be resolved by the voxel mesh, so each mesh cell column gets a homogenised **orthotropic**
//! material:
//!
//! * in-plane  k∥ = Σ kᵢ tᵢ / t            (copper layers with their local copper fraction, FR-4)
//! * through   k⊥ = t / Σ (tᵢ / kᵢ), then in parallel with the via barrels' copper area fraction
//! * ρ·c_p by volume; stiffness and expansion from the laminate.
//!
//! Values are scaled for the voxelised board thickness so conductances and heat capacity match the
//! real board. Footprints can become component boxes (named by reference) for heat sources.

use crate::geometry::V3;
use crate::material::Material;
use anyhow::{Result, anyhow, bail};
use std::collections::BTreeMap;

const K_CU: f64 = 385.0;
const RHO_CU: f64 = 8940.0;
const CP_CU: f64 = 385.0;
/// via barrel plating thickness [mm]
const PLATING: f64 = 0.025;

// ------------------------------------------------------------------ S-expressions

#[derive(Debug, Clone)]
pub enum Sx {
    Atom(String),
    List(Vec<Sx>),
}

impl Sx {
    fn head(&self) -> Option<&str> {
        match self {
            Sx::List(v) => match v.first() {
                Some(Sx::Atom(a)) => Some(a.as_str()),
                _ => None,
            },
            _ => None,
        }
    }
    fn items(&self) -> &[Sx] {
        match self {
            Sx::List(v) => v,
            _ => &[],
        }
    }
    /// Child lists with the given head.
    fn all<'a>(&'a self, name: &'a str) -> impl Iterator<Item = &'a Sx> + 'a {
        self.items().iter().filter(move |c| c.head() == Some(name))
    }
    fn get(&self, name: &str) -> Option<&Sx> {
        self.items().iter().find(|c| c.head() == Some(name))
    }
    fn atom(&self, i: usize) -> Option<&str> {
        match self.items().get(i) {
            Some(Sx::Atom(a)) => Some(a.as_str()),
            _ => None,
        }
    }
    fn num(&self, i: usize) -> Option<f64> {
        self.atom(i).and_then(|a| a.parse().ok())
    }
    /// (name x y [angle]) → (x, y, angle)
    fn xy(&self, name: &str) -> Option<(f64, f64, f64)> {
        let s = self.get(name)?;
        Some((s.num(1)?, s.num(2)?, s.num(3).unwrap_or(0.0)))
    }
    fn str_of(&self, name: &str) -> Option<&str> {
        self.get(name).and_then(|s| s.atom(1))
    }
    /// all string atoms of a (layers ...) list, or the single (layer ...)
    fn layers(&self) -> Vec<String> {
        if let Some(l) = self.get("layers") {
            return l.items()[1..].iter().filter_map(|a| if let Sx::Atom(s) = a { Some(s.clone()) } else { None }).collect();
        }
        self.str_of("layer").map(|s| vec![s.to_string()]).unwrap_or_default()
    }
}

pub fn parse(text: &str) -> Result<Sx> {
    let b = text.as_bytes();
    let mut i = 0usize;
    let mut stack: Vec<Vec<Sx>> = vec![Vec::new()];
    while i < b.len() {
        let c = b[i];
        match c {
            b'(' => {
                stack.push(Vec::new());
                i += 1;
            }
            b')' => {
                let done = stack.pop().ok_or_else(|| anyhow!("unbalanced ')'"))?;
                stack.last_mut().ok_or_else(|| anyhow!("unbalanced ')'"))?.push(Sx::List(done));
                i += 1;
            }
            b'"' => {
                let mut s = Vec::new();
                i += 1;
                while i < b.len() && b[i] != b'"' {
                    if b[i] == b'\\' && i + 1 < b.len() {
                        i += 1;
                        s.push(match b[i] {
                            b'n' => b'\n',
                            b't' => b'\t',
                            x => x,
                        });
                    } else {
                        s.push(b[i]);
                    }
                    i += 1;
                }
                i += 1;
                stack.last_mut().unwrap().push(Sx::Atom(String::from_utf8_lossy(&s).into_owned()));
            }
            c if c.is_ascii_whitespace() => i += 1,
            _ => {
                let st = i;
                while i < b.len() && !b[i].is_ascii_whitespace() && b[i] != b'(' && b[i] != b')' {
                    i += 1;
                }
                stack.last_mut().unwrap().push(Sx::Atom(String::from_utf8_lossy(&b[st..i]).into_owned()));
            }
        }
    }
    if stack.len() != 1 {
        bail!("unbalanced parentheses in the board file");
    }
    stack.pop().unwrap().into_iter().find(|s| s.head() == Some("kicad_pcb")).ok_or_else(|| anyhow!("not a KiCad board (no kicad_pcb)"))
}

// ------------------------------------------------------------------ board model

type P2 = [f64; 2];

#[derive(Debug, Clone)]
pub struct Footprint {
    pub reference: String,
    pub value: String,
    pub footprint: String,
    pub bottom: bool,
    /// axis-aligned body extent in board coordinates (mm): min, max
    pub lo: P2,
    pub hi: P2,
}

/// Imported board: outline, stack-up and copper coverage maps.
#[derive(Debug)]
pub struct PcbData {
    pub path: String,
    /// closed outline loops in board coordinates (mm, y up, min corner at the origin)
    pub loops: Vec<Vec<P2>>,
    pub size: P2,
    pub thickness: f64,
    /// copper layers: name, thickness [mm]
    pub layers: Vec<(String, f64)>,
    pub footprints: Vec<Footprint>,
    /// raster: pixel size [mm], dims
    pub res: f64,
    pub nx: usize,
    pub ny: usize,
    /// per copper layer: summed-area table of covered pixels ((nx+1)·(ny+1))
    sat: Vec<Vec<u32>>,
    /// summed-area table of via-barrel copper area [mm²]
    via_sat: Vec<f64>,
    pub copper_fraction: Vec<f64>,
    pub n_vias: usize,
    pub n_tracks: usize,
    pub warnings: Vec<String>,
}

fn rot(x: f64, y: f64, deg: f64) -> P2 {
    // KiCad: positive angle = counter-clockwise on screen with y pointing down
    let (s, c) = deg.to_radians().sin_cos();
    [x * c + y * s, -x * s + y * c]
}

fn arc_points(a: P2, m: P2, b: P2) -> Vec<P2> {
    // circle through three points
    let d = 2.0 * (a[0] * (m[1] - b[1]) + m[0] * (b[1] - a[1]) + b[0] * (a[1] - m[1]));
    if d.abs() < 1e-12 {
        return vec![a, b];
    }
    let sq = |p: P2| p[0] * p[0] + p[1] * p[1];
    let ux = (sq(a) * (m[1] - b[1]) + sq(m) * (b[1] - a[1]) + sq(b) * (a[1] - m[1])) / d;
    let uy = (sq(a) * (b[0] - m[0]) + sq(m) * (a[0] - b[0]) + sq(b) * (m[0] - a[0])) / d;
    let r = ((a[0] - ux).powi(2) + (a[1] - uy).powi(2)).sqrt();
    let ang = |p: P2| (p[1] - uy).atan2(p[0] - ux);
    let (a0, am, a1) = (ang(a), ang(m), ang(b));
    let tau = std::f64::consts::TAU;
    let norm = |x: f64| ((x % tau) + tau) % tau;
    // go from a0 to a1 through am
    let ccw = norm(am - a0) < norm(a1 - a0);
    let sweep = if ccw { norm(a1 - a0) } else { -norm(a0 - a1) };
    let n = ((sweep.abs() * r / 0.2).ceil() as usize).clamp(4, 180);
    (0..=n).map(|i| {
        let t = a0 + sweep * i as f64 / n as f64;
        [ux + r * t.cos(), uy + r * t.sin()]
    }).collect()
}

/// Polyline of an outline/graphic primitive (file coordinates, y down).
fn graphic(g: &Sx) -> Option<Vec<P2>> {
    let p = |n: &str| g.xy(n).map(|(x, y, _)| [x, y]);
    match g.head()? {
        "gr_line" | "fp_line" => Some(vec![p("start")?, p("end")?]),
        "gr_rect" | "fp_rect" => {
            let (a, b) = (p("start")?, p("end")?);
            Some(vec![a, [b[0], a[1]], b, [a[0], b[1]], a])
        }
        "gr_circle" | "fp_circle" => {
            let (c, e) = (p("center")?, p("end")?);
            let r = ((e[0] - c[0]).powi(2) + (e[1] - c[1]).powi(2)).sqrt();
            Some((0..=72).map(|i| {
                let t = std::f64::consts::TAU * i as f64 / 72.0;
                [c[0] + r * t.cos(), c[1] + r * t.sin()]
            }).collect())
        }
        "gr_arc" | "fp_arc" => {
            if let Some(m) = p("mid") {
                Some(arc_points(p("start")?, m, p("end")?))
            } else {
                // KiCad 5: (start = centre) (end = start point) (angle)
                let (c, s) = (p("start")?, p("end")?);
                let ang = g.get("angle")?.num(1)?;
                let r = ((s[0] - c[0]).powi(2) + (s[1] - c[1]).powi(2)).sqrt();
                let a0 = (s[1] - c[1]).atan2(s[0] - c[0]);
                let n = ((ang.abs() / 5.0).ceil() as usize).max(2);
                Some((0..=n).map(|i| {
                    let t = a0 + ang.to_radians() * i as f64 / n as f64;
                    [c[0] + r * t.cos(), c[1] + r * t.sin()]
                }).collect())
            }
        }
        "gr_poly" | "fp_poly" => {
            let pts: Vec<P2> = g.get("pts")?.all("xy").filter_map(|q| Some([q.num(1)?, q.num(2)?])).collect();
            let mut v = pts.clone();
            if let Some(f) = pts.first() {
                v.push(*f);
            }
            Some(v)
        }
        _ => None,
    }
}

/// Join polyline pieces into closed loops by matching end points.
fn join_loops(mut pieces: Vec<Vec<P2>>, tol: f64) -> (Vec<Vec<P2>>, usize) {
    let close = |a: P2, b: P2| (a[0] - b[0]).abs() < tol && (a[1] - b[1]).abs() < tol;
    let mut loops = Vec::new();
    let mut open = 0;
    while let Some(mut cur) = pieces.pop() {
        loop {
            if cur.len() > 2 && close(cur[0], *cur.last().unwrap()) {
                loops.push(cur);
                break;
            }
            let end = *cur.last().unwrap();
            if let Some(i) = pieces.iter().position(|p| close(p[0], end) || close(*p.last().unwrap(), end)) {
                let mut p = pieces.swap_remove(i);
                if !close(p[0], end) {
                    p.reverse();
                }
                cur.extend_from_slice(&p[1..]);
            } else {
                open += 1;
                break;
            }
        }
    }
    (loops, open)
}

fn point_in_loops(loops: &[Vec<P2>], x: f64, y: f64) -> bool {
    let mut inside = false;
    for l in loops {
        let n = l.len();
        for i in 0..n {
            let (a, b) = (l[i], l[(i + 1) % n]);
            if (a[1] > y) != (b[1] > y) && x < a[0] + (y - a[1]) / (b[1] - a[1]) * (b[0] - a[0]) {
                inside = !inside;
            }
        }
    }
    inside
}

/// Bit raster of one copper layer.
struct Raster {
    nx: usize,
    ny: usize,
    res: f64,
    x0: f64,
    y0: f64,
    bits: Vec<u8>,
}

impl Raster {
    fn px(&self, x: f64) -> f64 {
        (x - self.x0) / self.res - 0.5
    }
    fn py(&self, y: f64) -> f64 {
        (y - self.y0) / self.res - 0.5
    }
    fn range(&self, lo: f64, hi: f64, n: usize) -> (usize, usize) {
        (lo.floor().max(0.0) as usize, (hi.ceil().max(0.0) as usize).min(n.saturating_sub(1)))
    }
    fn centre(&self, i: usize, j: usize) -> P2 {
        [self.x0 + (i as f64 + 0.5) * self.res, self.y0 + (j as f64 + 0.5) * self.res]
    }
    /// thick segment (capsule)
    fn capsule(&mut self, a: P2, b: P2, w: f64) {
        let r = w / 2.0;
        let (i0, i1) = self.range(self.px(a[0].min(b[0]) - r), self.px(a[0].max(b[0]) + r), self.nx);
        let (j0, j1) = self.range(self.py(a[1].min(b[1]) - r), self.py(a[1].max(b[1]) + r), self.ny);
        let d = [b[0] - a[0], b[1] - a[1]];
        let l2 = (d[0] * d[0] + d[1] * d[1]).max(1e-18);
        for j in j0..=j1 {
            for i in i0..=i1 {
                // 2×2 sub-samples per pixel: bit (sy·2 + sx)
                let c0 = self.centre(i, j);
                let mut mask = 0u8;
                for sb in 0..4 {
                    let c = [c0[0] + if sb & 1 == 1 { 0.25 } else { -0.25 } * self.res, c0[1] + if sb & 2 == 2 { 0.25 } else { -0.25 } * self.res];
                    let t = (((c[0] - a[0]) * d[0] + (c[1] - a[1]) * d[1]) / l2).clamp(0.0, 1.0);
                    let q = [a[0] + t * d[0] - c[0], a[1] + t * d[1] - c[1]];
                    if q[0] * q[0] + q[1] * q[1] <= r * r {
                        mask |= 1 << sb;
                    }
                }
                self.bits[i + j * self.nx] |= mask;
            }
        }
    }
    /// filled polygon (even-odd)
    fn polygon(&mut self, pts: &[P2]) {
        if pts.len() < 3 {
            return;
        }
        let (mut ylo, mut yhi) = (f64::INFINITY, f64::NEG_INFINITY);
        for p in pts {
            ylo = ylo.min(p[1]);
            yhi = yhi.max(p[1]);
        }
        let (j0, j1) = self.range(self.py(ylo), self.py(yhi), self.ny);
        let n = pts.len();
        let mut xs: Vec<f64> = Vec::new();
        for j in j0..=j1 {
            for sy in 0..2usize {
                let y = self.centre(0, j)[1] + if sy == 1 { 0.25 } else { -0.25 } * self.res;
                xs.clear();
                for k in 0..n {
                    let (a, b) = (pts[k], pts[(k + 1) % n]);
                    if (a[1] > y) != (b[1] > y) {
                        xs.push(a[0] + (y - a[1]) / (b[1] - a[1]) * (b[0] - a[0]));
                    }
                }
                xs.sort_by(|a, b| a.total_cmp(b));
                for pair in xs.chunks(2) {
                    if pair.len() < 2 {
                        continue;
                    }
                    // sub-columns at x = (i + 0.25)·res and (i + 0.75)·res
                    for sx in 0..2usize {
                        let off = if sx == 1 { 0.75 } else { 0.25 };
                        let lo = ((pair[0] - self.x0) / self.res - off).ceil().max(0.0);
                        let hi = ((pair[1] - self.x0) / self.res - off).floor();
                        if hi < lo {
                            continue;
                        }
                        let bit = 1u8 << (sy * 2 + sx);
                        for i in (lo as usize)..=(hi as usize).min(self.nx - 1) {
                            self.bits[i + j * self.nx] |= bit;
                        }
                    }
                }
            }
        }
    }
    fn sat(&self) -> Vec<u32> {
        let w = self.nx + 1;
        let mut s = vec![0u32; w * (self.ny + 1)];
        for j in 0..self.ny {
            let mut row = 0u32;
            for i in 0..self.nx {
                row += self.bits[i + j * self.nx].count_ones();
                s[(i + 1) + (j + 1) * w] = s[(i + 1) + j * w] + row;
            }
        }
        s
    }
}

/// Pad copper as polygon or capsule in board (file) coordinates.
enum PadGeo {
    Poly(Vec<P2>),
    Capsule(P2, P2, f64),
}

fn pad_geometry(fx: f64, fy: f64, frot: f64, pad: &Sx) -> Option<PadGeo> {
    let (px, py, prot) = pad.xy("at")?;
    let size = pad.get("size")?;
    let (w, h) = (size.num(1)?, size.num(2).unwrap_or(size.num(1)?));
    let c = {
        let r = rot(px, py, frot);
        [fx + r[0], fy + r[1]]
    };
    // pad angle in the file already includes the footprint rotation (KiCad ≥ 5)
    let ang = prot;
    let shape = pad.atom(3).unwrap_or("rect");
    match shape {
        "circle" => Some(PadGeo::Capsule(c, c, w)),
        "oval" => {
            let (len, wid, along_x) = if w >= h { (w - h, h, true) } else { (h - w, w, false) };
            let d = if along_x { rot(len / 2.0, 0.0, ang) } else { rot(0.0, len / 2.0, ang) };
            Some(PadGeo::Capsule([c[0] - d[0], c[1] - d[1]], [c[0] + d[0], c[1] + d[1]], wid))
        }
        _ => {
            let corners = [[-w / 2.0, -h / 2.0], [w / 2.0, -h / 2.0], [w / 2.0, h / 2.0], [-w / 2.0, h / 2.0]];
            Some(PadGeo::Poly(corners.iter().map(|p| {
                let r = rot(p[0], p[1], ang);
                [c[0] + r[0], c[1] + r[1]]
            }).collect()))
        }
    }
}

impl PcbData {
    pub fn load(path: &str) -> Result<PcbData> {
        let text = std::fs::read_to_string(path).map_err(|e| anyhow!("reading {path}: {e}"))?;
        Self::from_text(&text, path)
    }

    pub fn from_text(text: &str, path: &str) -> Result<PcbData> {
        let root = parse(text)?;
        let mut warnings = Vec::new();
        // ---- copper layers and stack-up
        let mut layer_names: Vec<String> = Vec::new();
        if let Some(ls) = root.get("layers") {
            for l in ls.items().iter().skip(1) {
                if let Some(name) = l.atom(1) {
                    if name.ends_with(".Cu") {
                        layer_names.push(name.to_string());
                    }
                }
            }
        }
        // F.Cu first, inner layers in order, B.Cu last
        let order = |n: &str| -> i64 {
            if n == "F.Cu" {
                -1
            } else if n == "B.Cu" {
                1_000
            } else {
                n.trim_start_matches("In").trim_end_matches(".Cu").parse::<i64>().unwrap_or(500)
            }
        };
        layer_names.sort_by_key(|n| order(n));
        layer_names.dedup();
        if layer_names.is_empty() {
            layer_names = vec!["F.Cu".into(), "B.Cu".into()];
        }
        let mut cu_t: BTreeMap<String, f64> = BTreeMap::new();
        let mut stack_total = 0.0;
        if let Some(st) = root.get("setup").and_then(|s| s.get("stackup")) {
            for l in st.all("layer") {
                let t = l.get("thickness").and_then(|t| t.num(1)).unwrap_or(0.0);
                stack_total += t;
                if let Some(name) = l.atom(1) {
                    if name.ends_with(".Cu") {
                        cu_t.insert(name.to_string(), t);
                    }
                }
            }
        }
        let thickness = root.get("general").and_then(|g| g.get("thickness")).and_then(|t| t.num(1)).filter(|t| *t > 0.0).unwrap_or(if stack_total > 0.2 { stack_total } else { 1.6 });
        let layers: Vec<(String, f64)> = layer_names.iter().map(|n| (n.clone(), cu_t.get(n).copied().filter(|t| *t > 0.0).unwrap_or(0.035))).collect();

        // ---- outline
        let mut pieces: Vec<Vec<P2>> = Vec::new();
        for g in root.items() {
            if g.str_of("layer") == Some("Edge.Cuts") {
                if let Some(p) = graphic(g) {
                    pieces.push(p);
                }
            }
        }
        let footprints_sx: Vec<&Sx> = root.all("footprint").chain(root.all("module")).collect();
        for fp in &footprints_sx {
            let (fx, fy, frot) = fp.xy("at").unwrap_or((0.0, 0.0, 0.0));
            for g in fp.items() {
                if g.str_of("layer") == Some("Edge.Cuts") {
                    if let Some(p) = graphic(g) {
                        pieces.push(p.into_iter().map(|q| {
                            let r = rot(q[0], q[1], frot);
                            [fx + r[0], fy + r[1]]
                        }).collect());
                    }
                }
            }
        }
        let (mut loops, open) = join_loops(pieces, 0.02);
        if open > 0 {
            warnings.push(format!("{open} Edge.Cuts pieces do not close into a loop and were ignored"));
        }

        // ---- copper primitives (file coordinates)
        let mut tracks: Vec<(String, P2, P2, f64)> = Vec::new();
        for s in root.all("segment") {
            if let (Some((x0, y0, _)), Some((x1, y1, _)), Some(w)) = (s.xy("start"), s.xy("end"), s.get("width").and_then(|w| w.num(1))) {
                tracks.push((s.str_of("layer").unwrap_or("F.Cu").to_string(), [x0, y0], [x1, y1], w));
            }
        }
        for s in root.all("arc") {
            if let (Some((a0, a1, _)), Some((m0, m1, _)), Some((b0, b1, _)), Some(w)) = (s.xy("start"), s.xy("mid"), s.xy("end"), s.get("width").and_then(|w| w.num(1))) {
                let pts = arc_points([a0, a1], [m0, m1], [b0, b1]);
                for p in pts.windows(2) {
                    tracks.push((s.str_of("layer").unwrap_or("F.Cu").to_string(), p[0], p[1], w));
                }
            }
        }
        let n_tracks = root.all("segment").count() + root.all("arc").count();
        let mut vias: Vec<(P2, f64, f64)> = Vec::new();
        for v in root.all("via") {
            if let (Some((x, y, _)), Some(d)) = (v.xy("at"), v.get("size").and_then(|s| s.num(1))) {
                let drill = v.get("drill").and_then(|s| s.num(1)).unwrap_or(d * 0.5);
                vias.push(([x, y], d, drill));
            }
        }
        let mut zones: Vec<(String, Vec<P2>)> = Vec::new();
        for z in root.all("zone") {
            for fp in z.all("filled_polygon") {
                let layer = fp.str_of("layer").map(String::from).or_else(|| z.layers().first().cloned()).unwrap_or_default();
                let pts: Vec<P2> = fp.get("pts").map(|p| p.all("xy").filter_map(|q| Some([q.num(1)?, q.num(2)?])).collect()).unwrap_or_default();
                zones.push((layer, pts));
            }
        }
        let unfilled = root.all("zone").filter(|z| z.get("filled_polygon").is_none()).count();
        if unfilled > 0 {
            warnings.push(format!("{unfilled} copper zones have no fill: press B (Fill all zones) in KiCad and save, or their copper is missing"));
        }
        let mut pads: Vec<(Vec<String>, PadGeo)> = Vec::new();
        let mut footprints = Vec::new();
        for fp in &footprints_sx {
            let (fx, fy, frot) = fp.xy("at").unwrap_or((0.0, 0.0, 0.0));
            let bottom = fp.str_of("layer") == Some("B.Cu");
            let prop = |name: &str| -> Option<String> {
                fp.all("property").find(|p| p.atom(1) == Some(name)).and_then(|p| p.atom(2)).map(String::from).or_else(|| fp.all("fp_text").find(|t| t.atom(1) == Some(&name.to_lowercase())).and_then(|t| t.atom(2)).map(String::from))
            };
            let mut lo = [f64::INFINITY; 2];
            let mut hi = [f64::NEG_INFINITY; 2];
            let mut grow = |q: P2| {
                lo = [lo[0].min(q[0]), lo[1].min(q[1])];
                hi = [hi[0].max(q[0]), hi[1].max(q[1])];
            };
            let to_world = |q: P2| {
                let r = rot(q[0], q[1], frot);
                [fx + r[0], fy + r[1]]
            };
            // body extent: courtyard, else fabrication outline, else pads
            let mut found = false;
            for want in [["F.CrtYd", "B.CrtYd"], ["F.Fab", "B.Fab"]] {
                for g in fp.items() {
                    if let Some(l) = g.str_of("layer") {
                        if want.contains(&l) {
                            if let Some(pts) = graphic(g) {
                                for q in pts {
                                    grow(to_world(q));
                                    found = true;
                                }
                            }
                        }
                    }
                }
                if found {
                    break;
                }
            }
            for pad in fp.all("pad") {
                let ls = pad.layers();
                if let Some(geo) = pad_geometry(fx, fy, frot, pad) {
                    if !found {
                        match &geo {
                            PadGeo::Poly(p) => p.iter().for_each(|q| grow(*q)),
                            PadGeo::Capsule(a, b, w) => {
                                grow([a[0] - w / 2.0, a[1] - w / 2.0]);
                                grow([b[0] + w / 2.0, b[1] + w / 2.0]);
                            }
                        }
                    }
                    if ls.iter().any(|l| l.ends_with(".Cu")) {
                        pads.push((ls, geo));
                    }
                }
            }
            if lo[0].is_finite() {
                footprints.push(Footprint {
                    reference: prop("Reference").unwrap_or_else(|| "?".into()),
                    value: prop("Value").unwrap_or_default(),
                    footprint: fp.atom(1).unwrap_or("").to_string(),
                    bottom,
                    lo,
                    hi,
                });
            }
        }

        // ---- frame: outline (or copper extent) → board coordinates (y up, origin at min corner)
        let mut lo = [f64::INFINITY; 2];
        let mut hi = [f64::NEG_INFINITY; 2];
        let mut ext = |p: P2| {
            lo = [lo[0].min(p[0]), lo[1].min(p[1])];
            hi = [hi[0].max(p[0]), hi[1].max(p[1])];
        };
        if loops.is_empty() {
            warnings.push("no closed board outline on Edge.Cuts: using the copper extent plus 1 mm".into());
            for t in &tracks {
                ext(t.1);
                ext(t.2);
            }
            for z in &zones {
                z.1.iter().for_each(|p| ext(*p));
            }
            for f in &footprints {
                ext(f.lo);
                ext(f.hi);
            }
            if !lo[0].is_finite() {
                bail!("the board has no outline and no copper — nothing to import");
            }
            lo = [lo[0] - 1.0, lo[1] - 1.0];
            hi = [hi[0] + 1.0, hi[1] + 1.0];
            loops.push(vec![lo, [hi[0], lo[1]], hi, [lo[0], hi[1]]]);
        } else {
            for l in &loops {
                l.iter().for_each(|p| ext(*p));
            }
        }
        let (fx0, fy1) = (lo[0], hi[1]);
        let tf = |p: P2| -> P2 { [p[0] - fx0, fy1 - p[1]] };
        let size = [hi[0] - lo[0], hi[1] - lo[1]];
        if size[0] <= 0.0 || size[1] <= 0.0 {
            bail!("degenerate board outline");
        }
        let loops: Vec<Vec<P2>> = loops.into_iter().map(|l| l.into_iter().map(tf).collect()).collect();

        // ---- rasterise copper
        let area = size[0] * size[1];
        let res = (area / 4.0e6).sqrt().max(0.05).max(0.1f64.min(area.sqrt() / 50.0));
        let (nx, ny) = ((size[0] / res).ceil() as usize + 1, (size[1] / res).ceil() as usize + 1);
        let new_r = || Raster { nx, ny, res, x0: 0.0, y0: 0.0, bits: vec![0u8; nx * ny] };
        let mut rasters: Vec<Raster> = layers.iter().map(|_| new_r()).collect();
        let lidx = |name: &str| -> Vec<usize> {
            if name == "*.Cu" {
                (0..layers.len()).collect()
            } else if name == "F&B.Cu" {
                vec![0, layers.len() - 1]
            } else {
                layers.iter().position(|l| l.0 == name).into_iter().collect()
            }
        };
        for (layer, a, b, w) in &tracks {
            for li in lidx(layer) {
                rasters[li].capsule(tf(*a), tf(*b), *w);
            }
        }
        for (layer, pts) in &zones {
            let p: Vec<P2> = pts.iter().map(|q| tf(*q)).collect();
            for li in lidx(layer) {
                rasters[li].polygon(&p);
            }
        }
        for (ls, geo) in &pads {
            let mut targets: Vec<usize> = ls.iter().flat_map(|l| lidx(l)).collect();
            targets.sort_unstable();
            targets.dedup();
            for li in targets {
                match geo {
                    PadGeo::Poly(p) => rasters[li].polygon(&p.iter().map(|q| tf(*q)).collect::<Vec<_>>()),
                    PadGeo::Capsule(a, b, w) => rasters[li].capsule(tf(*a), tf(*b), *w),
                }
            }
        }
        let mut via_area = vec![0.0f64; nx * ny];
        for (c, d, drill) in &vias {
            let c = tf(*c);
            for r in rasters.iter_mut() {
                r.capsule(c, c, *d);
            }
            let i = ((c[0] / res).floor().max(0.0) as usize).min(nx - 1);
            let j = ((c[1] / res).floor().max(0.0) as usize).min(ny - 1);
            via_area[i + j * nx] += std::f64::consts::PI * drill * PLATING;
        }
        // clip copper to the board outline
        let inside: Vec<bool> = (0..nx * ny).map(|k| {
            let c = [(k % nx) as f64 * res + res / 2.0, (k / nx) as f64 * res + res / 2.0];
            point_in_loops(&loops, c[0], c[1])
        }).collect();
        for r in rasters.iter_mut() {
            for (b, ins) in r.bits.iter_mut().zip(&inside) {
                if !ins {
                    *b = 0;
                }
            }
        }
        let n_in = inside.iter().filter(|b| **b).count().max(1);
        let copper_fraction: Vec<f64> = rasters.iter().map(|r| r.bits.iter().map(|b| b.count_ones() as f64).sum::<f64>() / (4.0 * n_in as f64)).collect();
        let sat: Vec<Vec<u32>> = rasters.iter().map(|r| r.sat()).collect();
        let w = nx + 1;
        let mut via_sat = vec![0.0f64; w * (ny + 1)];
        for j in 0..ny {
            let mut row = 0.0;
            for i in 0..nx {
                row += via_area[i + j * nx];
                via_sat[(i + 1) + (j + 1) * w] = via_sat[(i + 1) + j * w] + row;
            }
        }
        let footprints = footprints
            .into_iter()
            .map(|f| {
                let (a, b) = (tf(f.lo), tf(f.hi));
                Footprint { lo: [a[0].min(b[0]), a[1].min(b[1])], hi: [a[0].max(b[0]), a[1].max(b[1])], ..f }
            })
            .collect();
        Ok(PcbData { path: path.to_string(), loops, size, thickness, layers, footprints, res, nx, ny, sat, via_sat, copper_fraction, n_vias: vias.len(), n_tracks, warnings })
    }

    /// Board coordinates contain (x, y) (inside the outline, outside cutouts)?
    pub fn inside(&self, x: f64, y: f64) -> bool {
        point_in_loops(&self.loops, x, y)
    }

    fn box_sum(&self, s: &[u32], x0: f64, y0: f64, x1: f64, y1: f64) -> (f64, f64) {
        let w = self.nx + 1;
        let cl = |v: f64, n: usize| ((v / self.res).round().max(0.0) as usize).min(n);
        let (i0, i1, j0, j1) = (cl(x0, self.nx), cl(x1, self.nx), cl(y0, self.ny), cl(y1, self.ny));
        let n = 4.0 * ((i1 - i0) * (j1 - j0)) as f64;
        if n == 0.0 {
            return (0.0, 0.0);
        }
        let v = s[i1 + j1 * w] as f64 + s[i0 + j0 * w] as f64 - s[i0 + j1 * w] as f64 - s[i1 + j0 * w] as f64;
        (v, n)
    }

    /// Copper fraction of each layer and via-barrel area fraction over the column [x0,x1]×[y0,y1].
    pub fn column(&self, x0: f64, y0: f64, x1: f64, y1: f64) -> (Vec<f64>, f64) {
        let f = self.sat.iter().map(|s| {
            let (v, n) = self.box_sum(s, x0, y0, x1, y1);
            if n > 0.0 { v / n } else { 0.0 }
        }).collect();
        let w = self.nx + 1;
        let cl = |v: f64, n: usize| ((v / self.res).round().max(0.0) as usize).min(n);
        let (i0, i1, j0, j1) = (cl(x0, self.nx), cl(x1, self.nx), cl(y0, self.ny), cl(y1, self.ny));
        let va = self.via_sat[i1 + j1 * w] + self.via_sat[i0 + j0 * w] - self.via_sat[i0 + j1 * w] - self.via_sat[i1 + j0 * w];
        let area = ((x1 - x0) * (y1 - y0)).max(1e-12);
        (f, (va / area).min(1.0))
    }

    /// Homogenised orthotropic material for a cell column of size h centred at (x, y); `t_vox` is the
    /// voxelised board thickness (properties are scaled so the real board's conductance and heat
    /// capacity are kept).
    pub fn material_at(&self, body: &str, base: &Material, x: f64, y: f64, h: f64, t_vox: f64) -> Material {
        let (f, via) = self.column(x - h / 2.0, y - h / 2.0, x + h / 2.0, y + h / 2.0);
        let t = self.thickness;
        let kd = base.thermal_conductivity;
        let tcu: f64 = self.layers.iter().map(|l| l.1).sum();
        let td = (t - tcu).max(0.1 * t);
        let mut kxy = kd * td;
        let mut rz = td / kd;
        let mut rhocp = base.density * base.specific_heat * td;
        let mut rho = base.density * td;
        for (fi, (_, tl)) in f.iter().zip(&self.layers) {
            let kl = fi * K_CU + (1.0 - fi) * kd;
            kxy += kl * tl;
            rz += tl / kl;
            rhocp += (fi * RHO_CU * CP_CU + (1.0 - fi) * base.density * base.specific_heat) * tl;
            rho += (fi * RHO_CU + (1.0 - fi) * base.density) * tl;
        }
        let tt = td + tcu;
        kxy /= tt;
        let kz = (1.0 - via) * tt / rz + via * K_CU;
        rhocp /= tt;
        rho /= tt;
        // scale to the voxel thickness: same in-plane conductance (k·t), same through resistance (t/k)
        let s = t / t_vox;
        let (kxy, kz, rho) = (kxy * s, kz / s, rho * s);
        let cp = rhocp * s / rho;
        // quantise (2 % steps on a log scale) so a board needs only a few dozen materials
        let q = |v: f64| (v.ln() / 1.02f64.ln()).round();
        let dq = |b: f64| 1.02f64.powf(b);
        let (bxy, bz) = (q(kxy), q(kz));
        let mut m = base.clone();
        m.name = format!("{body} · k∥ {:.3} · k⊥ {:.3}", dq(bxy), dq(bz));
        m.thermal_conductivity = dq(bxy);
        m.conductivity_xyz = Some([dq(bxy), dq(bxy), dq(bz)]);
        m.density = rho;
        m.specific_heat = cp;
        m
    }

    pub fn summary(&self) -> serde_json::Value {
        serde_json::json!({
            "size_mm": [crate::results::float(self.size[0]), crate::results::float(self.size[1])],
            "thickness_mm": self.thickness,
            "copper_layers": self.layers.iter().zip(&self.copper_fraction).map(|((n, t), f)| serde_json::json!({"layer": n, "thickness_mm": t, "copper_coverage_percent": crate::results::float(100.0 * f)})).collect::<Vec<_>>(),
            "outline_loops": self.loops.len(),
            "tracks": self.n_tracks,
            "vias": self.n_vias,
            "footprints": self.footprints.len(),
            "warnings": self.warnings,
        })
    }
}

/// z range of the board body: [z0, z0 + thickness].
pub fn board_bounds(d: &PcbData, offset: V3) -> (V3, V3) {
    (offset, [offset[0] + d.size[0], offset[1] + d.size[1], offset[2] + d.thickness])
}
