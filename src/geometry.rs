//! CAD-lite geometry: constructive solid geometry from primitives and STL imports.
//! All lengths are in millimetres.

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

pub type V3 = [f64; 3];

pub fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
pub fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
pub fn scale(a: V3, s: f64) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}
pub fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
pub fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
pub fn norm(a: V3) -> f64 {
    dot(a, a).sqrt()
}
pub fn normalize(a: V3) -> V3 {
    let n = norm(a);
    if n > 0.0 { scale(a, 1.0 / n) } else { a }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum Axis {
    X,
    Y,
    #[default]
    Z,
}
impl Axis {
    pub fn idx(self) -> usize {
        match self {
            Axis::X => 0,
            Axis::Y => 1,
            Axis::Z => 2,
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum Op {
    #[default]
    Add,
    Subtract,
}

/// Triangle bounding-volume hierarchy (median centroid split on longest axis) for fast STL queries.
#[derive(Debug, Default)]
struct BvhNode {
    lo: V3,
    hi: V3,
    /// Inner nodes use `left` and `left + 1` with `count == 0`; leaves use `order[start..start + count]`.
    left: usize,
    start: usize,
    count: usize,
}

#[derive(Debug, Default)]
struct StlBvh {
    order: Vec<u32>,
    nodes: Vec<BvhNode>,
}

const STL_BVH_LEAF: usize = 8;

#[inline]
fn tri_centroid(t: &[V3; 3], ax: usize) -> f64 {
    (t[0][ax] + t[1][ax] + t[2][ax]) / 3.0
}

#[inline]
fn aabb_dist2(lo: V3, hi: V3, p: V3) -> f64 {
    let mut d2 = 0.0;
    for k in 0..3 {
        let d = if p[k] < lo[k] { lo[k] - p[k] } else if p[k] > hi[k] { p[k] - hi[k] } else { 0.0 };
        d2 += d * d;
    }
    d2
}

impl StlBvh {
    fn build(tris: &[[V3; 3]]) -> Self {
        if tris.is_empty() {
            return Self::default();
        }
        let mut b = Self { order: (0..tris.len() as u32).collect(), nodes: Vec::with_capacity(tris.len() / 2 + 1) };
        let count = b.order.len();
        b.nodes.push(BvhNode { lo: [0.0; 3], hi: [0.0; 3], left: 0, start: 0, count });
        b.split(0, tris);
        b
    }

    fn split(&mut self, ni: usize, tris: &[[V3; 3]]) {
        let (start, count) = (self.nodes[ni].start, self.nodes[ni].count);
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for &t in &self.order[start..start + count] {
            for v in &tris[t as usize] {
                for k in 0..3 {
                    lo[k] = lo[k].min(v[k]);
                    hi[k] = hi[k].max(v[k]);
                }
            }
        }
        self.nodes[ni].lo = lo;
        self.nodes[ni].hi = hi;
        if count <= STL_BVH_LEAF {
            return;
        }
        let ax = (0..3).max_by(|&a, &b| (hi[a] - lo[a]).total_cmp(&(hi[b] - lo[b]))).unwrap();
        let half = count / 2;
        self.order[start..start + count].select_nth_unstable_by(half, |&a, &b| {
            tri_centroid(&tris[a as usize], ax).total_cmp(&tri_centroid(&tris[b as usize], ax))
        });
        let left = self.nodes.len();
        self.nodes.push(BvhNode { lo: [0.0; 3], hi: [0.0; 3], left: 0, start, count: half });
        self.nodes.push(BvhNode { lo: [0.0; 3], hi: [0.0; 3], left: 0, start: start + half, count: count - half });
        self.nodes[ni].left = left;
        self.nodes[ni].count = 0;
        self.split(left, tris);
        self.split(left + 1, tris);
    }

    /// Best-first branch-and-bound traversal maintaining `limit_d2` (best squared distance found so far).
    fn visit_closest(&self, p: V3, mut leaf: impl FnMut(usize, &mut f64)) {
        if self.nodes.is_empty() {
            return;
        }
        let mut limit_d2 = f64::INFINITY;
        let mut stack = Vec::with_capacity(64);
        stack.push(0usize);
        let prune = |d2: f64, lim: f64| lim.is_finite() && d2 > lim * (1.0 + 1e-12) + 1e-24;
        while let Some(ni) = stack.pop() {
            let n = &self.nodes[ni];
            if prune(aabb_dist2(n.lo, n.hi, p), limit_d2) {
                continue;
            }
            if n.count > 0 {
                for &t in &self.order[n.start..n.start + n.count] {
                    leaf(t as usize, &mut limit_d2);
                }
            } else {
                let l = n.left;
                let r = n.left + 1;
                let dl = aabb_dist2(self.nodes[l].lo, self.nodes[l].hi, p);
                let dr = aabb_dist2(self.nodes[r].lo, self.nodes[r].hi, p);
                if dl <= dr {
                    if !prune(dr, limit_d2) {
                        stack.push(r);
                    }
                    if !prune(dl, limit_d2) {
                        stack.push(l);
                    }
                } else {
                    if !prune(dl, limit_d2) {
                        stack.push(l);
                    }
                    if !prune(dr, limit_d2) {
                        stack.push(r);
                    }
                }
            }
        }
    }
}

/// Triangle soup loaded from an STL file plus its segmentation into smooth faces.
#[derive(Debug, Default)]
pub struct StlData {
    pub tris: Vec<[V3; 3]>,
    pub normals: Vec<V3>,
    pub segment: Vec<u32>,
    pub n_segments: usize,
    pub min: V3,
    pub max: V3,
    bvh: StlBvh,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Shape {
    Box {
        min: V3,
        size: V3,
    },
    Cylinder {
        base: V3,
        #[serde(default)]
        axis: Axis,
        radius: f64,
        height: f64,
    },
    Sphere {
        center: V3,
        radius: f64,
    },
    Stl {
        path: String,
        #[serde(default = "one")]
        scale: f64,
        #[serde(default)]
        offset: V3,
        #[serde(skip)]
        data: Option<Arc<StlData>>,
    },
    /// Printed circuit board imported from KiCad: outline extruded over the board thickness, with
    /// homogenised copper (see `kicad.rs`). `offset` = position of the board's min corner.
    Pcb {
        path: String,
        #[serde(default)]
        offset: V3,
        #[serde(skip)]
        data: Option<Arc<crate::kicad::PcbData>>,
    },
}
fn one() -> f64 {
    1.0
}

/// Solid bodies are the parts being analysed; an enclosure is the surrounding air / fluid box
/// used by electrostatic, magnetostatic and CFD analyses.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    #[default]
    Solid,
    Enclosure,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Body {
    pub name: String,
    pub shape: Shape,
    #[serde(default)]
    pub op: Op,
    #[serde(default = "default_material")]
    pub material: String,
    #[serde(default)]
    pub role: Role,
}
pub fn default_material() -> String {
    "Structural Steel".to_string()
}

impl Shape {
    /// Exact volume [mm³] of primitive shapes (None for STL).
    pub fn volume(&self) -> Option<f64> {
        match self {
            Shape::Box { size, .. } => Some(size[0] * size[1] * size[2]),
            Shape::Cylinder { radius, height, .. } => Some(std::f64::consts::PI * radius * radius * height),
            Shape::Sphere { radius, .. } => Some(4.0 / 3.0 * std::f64::consts::PI * radius.powi(3)),
            Shape::Stl { .. } | Shape::Pcb { .. } => None,
        }
    }
    pub fn bounds(&self) -> (V3, V3) {
        match self {
            Shape::Box { min, size } => (*min, add(*min, *size)),
            Shape::Cylinder { base, axis, radius, height } => {
                let a = axis.idx();
                let mut lo = *base;
                let mut hi = *base;
                for k in 0..3 {
                    if k == a {
                        hi[k] += height;
                    } else {
                        lo[k] -= radius;
                        hi[k] += radius;
                    }
                }
                (lo, hi)
            }
            Shape::Sphere { center, radius } => {
                (sub(*center, [*radius; 3]), add(*center, [*radius; 3]))
            }
            Shape::Stl { data, .. } => match data {
                Some(d) => (d.min, d.max),
                None => ([0.0; 3], [0.0; 3]),
            },
            Shape::Pcb { data, offset, .. } => match data {
                Some(d) => crate::kicad::board_bounds(d, *offset),
                None => (*offset, *offset),
            },
        }
    }

    pub fn validate(&self) -> Result<()> {
        let finite = |a: &[f64]| a.iter().all(|v| v.is_finite());
        match self {
            Shape::Box { min, size } => {
                if !finite(min) || !finite(size) || size.iter().any(|s| *s <= 0.0) {
                    bail!("box corner and size must be finite, and size must be positive in every direction");
                }
            }
            Shape::Cylinder { base, radius, height, .. } => {
                if !finite(base) || !radius.is_finite() || !height.is_finite() || *radius <= 0.0 || *height <= 0.0 {
                    bail!("cylinder base, radius and height must be finite and positive");
                }
            }
            Shape::Sphere { center, radius } => {
                if !finite(center) || !radius.is_finite() || *radius <= 0.0 {
                    bail!("sphere center and radius must be finite and positive");
                }
            }
            Shape::Stl { data, path, scale, offset } => {
                if !scale.is_finite() || *scale <= 0.0 || !finite(offset) {
                    bail!("STL '{path}' scale must be positive and offset finite");
                }
                if data.is_none() {
                    bail!("STL '{path}' is not loaded");
                }
            }
            Shape::Pcb { data, path, offset } => {
                if !finite(offset) {
                    bail!("KiCad board '{path}' offset must be finite");
                }
                if data.is_none() {
                    bail!("KiCad board '{path}' is not loaded");
                }
            }
        }
        Ok(())
    }

    /// Point membership (analytic shapes only; STL uses column ray casting).
    pub fn contains(&self, p: V3) -> bool {
        match self {
            Shape::Box { min, size } => (0..3).all(|k| p[k] >= min[k] && p[k] <= min[k] + size[k]),
            Shape::Cylinder { base, axis, radius, height } => {
                let a = axis.idx();
                let t = p[a] - base[a];
                if t < 0.0 || t > *height {
                    return false;
                }
                let (b, c) = ((a + 1) % 3, (a + 2) % 3);
                let r2 = (p[b] - base[b]).powi(2) + (p[c] - base[c]).powi(2);
                r2 <= radius * radius
            }
            Shape::Sphere { center, radius } => {
                let d = sub(p, *center);
                dot(d, d) <= radius * radius
            }
            Shape::Stl { .. } => false,
            Shape::Pcb { data, offset, .. } => match data {
                Some(d) => p[2] >= offset[2] && p[2] <= offset[2] + d.thickness && d.inside(p[0] - offset[0], p[1] - offset[1]),
                None => false,
            },
        }
    }

    pub fn patch_names(&self) -> Vec<String> {
        match self {
            Shape::Box { .. } => ["xmin", "xmax", "ymin", "ymax", "zmin", "zmax"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
            Shape::Cylinder { .. } => ["side", "bottom", "top"].iter().map(|s| s.to_string()).collect(),
            Shape::Sphere { .. } => vec!["surface".to_string()],
            Shape::Stl { data, .. } => {
                let n = data.as_ref().map(|d| d.n_segments).unwrap_or(0);
                (0..n).map(|i| format!("face{i}")).collect()
            }
            Shape::Pcb { .. } => ["edge", "bottom", "top"].iter().map(|s| s.to_string()).collect(),
        }
    }

    /// Closest point on surface patch `patch` (planes and cylinder/sphere surfaces are treated as
    /// unbounded so alternating projections onto neighbouring patches converge to their edges).
    /// None for shapes without an analytic surface (KiCad boards).
    pub fn project(&self, patch: usize, p: V3) -> Option<V3> {
        match self {
            Shape::Box { min, size } => {
                let a = patch / 2;
                let mut q = p;
                q[a] = if patch % 2 == 0 { min[a] } else { min[a] + size[a] };
                Some(q)
            }
            Shape::Cylinder { base, axis, radius, height } => {
                let a = axis.idx();
                let mut q = p;
                match patch {
                    0 => {
                        let (b, c) = ((a + 1) % 3, (a + 2) % 3);
                        let (rb, rc) = (p[b] - base[b], p[c] - base[c]);
                        let r = (rb * rb + rc * rc).sqrt();
                        if r < 1e-12 {
                            return None;
                        }
                        q[b] = base[b] + rb / r * radius;
                        q[c] = base[c] + rc / r * radius;
                    }
                    1 => q[a] = base[a],
                    _ => q[a] = base[a] + height,
                }
                Some(q)
            }
            Shape::Sphere { center, radius } => {
                let d = sub(p, *center);
                let r = norm(d);
                (r > 1e-12).then(|| add(*center, scale(d, radius / r)))
            }
            Shape::Stl { data, .. } => {
                let d = data.as_ref()?;
                let mut best: Option<(f64, V3)> = None;
                d.bvh.visit_closest(p, |i, limit_d2| {
                    if d.segment[i] as usize != patch {
                        return;
                    }
                    let q = closest_point_triangle(p, &d.tris[i]);
                    let dv = sub(p, q);
                    let d2 = dot(dv, dv);
                    if d2 < *limit_d2 {
                        *limit_d2 = d2;
                        best = Some((d2, q));
                    }
                });
                best.map(|b| b.1)
            }
            Shape::Pcb { .. } => None,
        }
    }

    /// Best matching surface patch for a voxel boundary face at `p` with outward normal `n`.
    /// Returns (patch index, score); lower score is better.
    pub fn best_patch(&self, p: V3, n: V3, h: f64, subtract: bool) -> (usize, f64) {
        let sign = if subtract { -1.0 } else { 1.0 };
        let score = |dist: f64, pn: V3| {
            let d = (dot(pn, n) * sign).clamp(-1.0, 1.0);
            if d < -0.2 { f64::INFINITY } else { dist + h * (1.0 - d) }
        };
        // flat axis-aligned surfaces only take voxel faces that face the same way: a staircase face
        // perpendicular to a plane belongs to the curved or other flat surface it steps along, even
        // when it is close to the plane (thin layers next to a snapped grid plane)
        let flat = |dist: f64, pn: V3| {
            let d = (dot(pn, n) * sign).clamp(-1.0, 1.0);
            if d < 0.5 { f64::INFINITY } else { dist + h * (1.0 - d) }
        };
        match self {
            Shape::Box { min, size } => {
                let mut best = (0, f64::INFINITY);
                for i in 0..6 {
                    let a = i / 2;
                    let plane = if i % 2 == 0 { min[a] } else { min[a] + size[a] };
                    let mut d2 = (p[a] - plane).powi(2);
                    for k in 0..3 {
                        if k != a {
                            let o = (min[k] - p[k]).max(p[k] - min[k] - size[k]).max(0.0);
                            d2 += o * o;
                        }
                    }
                    let mut pn = [0.0; 3];
                    pn[a] = if i % 2 == 0 { -1.0 } else { 1.0 };
                    let s = flat(d2.sqrt(), pn);
                    if s < best.1 {
                        best = (i, s);
                    }
                }
                best
            }
            Shape::Cylinder { base, axis, radius, height } => {
                let a = axis.idx();
                let (b, c) = ((a + 1) % 3, (a + 2) % 3);
                let t = p[a] - base[a];
                let rb = p[b] - base[b];
                let rc = p[c] - base[c];
                let r = (rb * rb + rc * rc).sqrt();
                let mut radial = [0.0; 3];
                if r > 1e-12 {
                    radial[b] = rb / r;
                    radial[c] = rc / r;
                }
                let dt_out = (-t).max(t - height).max(0.0);
                let side = score(((r - radius).powi(2) + dt_out * dt_out).sqrt(), radial);
                let dr_out = (r - radius).max(0.0);
                let mut down = [0.0; 3];
                down[a] = -1.0;
                let mut up = [0.0; 3];
                up[a] = 1.0;
                let bottom = flat((t * t + dr_out * dr_out).sqrt(), down);
                let top = flat(((t - height).powi(2) + dr_out * dr_out).sqrt(), up);
                let mut best = (0, side);
                if bottom < best.1 {
                    best = (1, bottom);
                }
                if top < best.1 {
                    best = (2, top);
                }
                best
            }
            Shape::Sphere { center, radius } => {
                let d = sub(p, *center);
                (0, score((norm(d) - radius).abs(), normalize(d)))
            }
            Shape::Pcb { data, offset, .. } => {
                let t = data.as_ref().map(|d| d.thickness).unwrap_or(0.0);
                let bottom = score((p[2] - offset[2]).abs(), [0.0, 0.0, -1.0]);
                let top = score((p[2] - offset[2] - t).abs(), [0.0, 0.0, 1.0]);
                let side = if n[2].abs() < 0.5 { h * 0.25 } else { f64::INFINITY };
                let mut best = (0, side);
                if bottom < best.1 {
                    best = (1, bottom);
                }
                if top < best.1 {
                    best = (2, top);
                }
                best
            }
            Shape::Stl { data, .. } => {
                let Some(d) = data else { return (0, f64::INFINITY) };
                let mut best = (0, f64::INFINITY);
                d.bvh.visit_closest(p, |i, limit_d2| {
                    let q = closest_point_triangle(p, &d.tris[i]);
                    let dist = norm(sub(p, q));
                    if dist > best.1 {
                        return;
                    }
                    let nn = d.normals[i];
                    let axis_flat = nn.iter().any(|c| c.abs() > 0.999);
                    let s = if axis_flat { flat(dist, nn) } else { score(dist, nn) };
                    if s < best.1 {
                        best = (d.segment[i] as usize, s);
                        *limit_d2 = best.1 * best.1;
                    }
                });
                best
            }
        }
    }
}

/// Closest point on triangle (Ericson, Real-Time Collision Detection 5.1.5).
pub fn closest_point_triangle(p: V3, t: &[V3; 3]) -> V3 {
    let (a, b, c) = (t[0], t[1], t[2]);
    let ab = sub(b, a);
    let ac = sub(c, a);
    let ap = sub(p, a);
    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }
    let bp = sub(p, b);
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return add(a, scale(ab, v));
    }
    let cp = sub(p, c);
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return add(a, scale(ac, w));
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return add(b, scale(sub(c, b), w));
    }
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    add(a, add(scale(ab, v), scale(ac, w)))
}

const MAX_STL_BYTES: u64 = 256 * 1024 * 1024;
const MAX_STL_TRIS: usize = 5_000_000;

/// Load an STL (binary or ASCII), apply scale + offset and segment it into smooth faces.
pub fn load_stl(path: &str, scale_f: f64, offset: V3) -> Result<StlData> {
    if !scale_f.is_finite() || scale_f <= 0.0 || !offset.iter().all(|v| v.is_finite()) {
        bail!("STL scale must be finite and positive, and offset must be finite");
    }
    let meta = std::fs::metadata(path).with_context(|| format!("reading STL {path}"))?;
    if meta.len() > MAX_STL_BYTES {
        bail!("STL '{path}' is too large ({} MB; limit {} MB)", meta.len() / (1024 * 1024), MAX_STL_BYTES / (1024 * 1024));
    }
    let bytes = std::fs::read(path).with_context(|| format!("reading STL {path}"))?;
    let mut tris: Vec<[V3; 3]> = Vec::new();
    let binary_count = (bytes.len() >= 84).then(|| u32::from_le_bytes([bytes[80], bytes[81], bytes[82], bytes[83]]) as usize);
    let is_binary = binary_count.is_some_and(|n| n.checked_mul(50).and_then(|b| b.checked_add(84)) == Some(bytes.len()));
    if is_binary {
        let n = binary_count.unwrap();
        if n > MAX_STL_TRIS {
            bail!("STL '{path}' has too many triangles ({n}; limit {MAX_STL_TRIS})");
        }
        tris.reserve(n);
        for i in 0..n {
            let o = 84 + i * 50 + 12;
            let mut t = [[0.0; 3]; 3];
            for (v, tv) in t.iter_mut().enumerate() {
                for (k, tk) in tv.iter_mut().enumerate() {
                    let q = o + v * 12 + k * 4;
                    let val = f32::from_le_bytes([bytes[q], bytes[q + 1], bytes[q + 2], bytes[q + 3]]) as f64;
                    if !val.is_finite() {
                        bail!("STL '{path}' contains non-finite vertex coordinates");
                    }
                    *tk = val;
                }
            }
            tris.push(t);
        }
    } else {
        let text = String::from_utf8_lossy(&bytes);
        let mut cur: Vec<V3> = Vec::new();
        for line in text.lines() {
            let mut it = line.split_whitespace();
            if it.next() == Some("vertex") {
                let v: Vec<f64> = it.filter_map(|s| s.parse::<f64>().ok().filter(|x| x.is_finite())).collect();
                if v.len() == 3 {
                    cur.push([v[0], v[1], v[2]]);
                }
                if cur.len() == 3 {
                    if tris.len() >= MAX_STL_TRIS {
                        bail!("STL '{path}' has too many triangles (limit {MAX_STL_TRIS})");
                    }
                    tris.push([cur[0], cur[1], cur[2]]);
                    cur.clear();
                }
            }
        }
    }
    if tris.is_empty() {
        bail!("STL '{path}' contains no triangles");
    }
    for t in tris.iter_mut() {
        for v in t.iter_mut() {
            *v = add(scale(*v, scale_f), offset);
            if !v.iter().all(|c| c.is_finite()) {
                bail!("STL '{path}' coordinates overflow after scaling");
            }
        }
    }
    let normals: Vec<V3> = tris
        .iter()
        .map(|t| normalize(cross(sub(t[1], t[0]), sub(t[2], t[0]))))
        .collect();
    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];
    for t in &tris {
        for v in t {
            for k in 0..3 {
                min[k] = min[k].min(v[k]);
                max[k] = max[k].max(v[k]);
            }
        }
    }
    // Weld vertices and build edge adjacency for region growing.
    let diag = norm(sub(max, min)).max(1e-9);
    let q = diag * 1e-6;
    let key = |v: V3| ((v[0] / q).round() as i64, (v[1] / q).round() as i64, (v[2] / q).round() as i64);
    let mut vid: HashMap<(i64, i64, i64), u32> = HashMap::new();
    let mut tv: Vec<[u32; 3]> = Vec::with_capacity(tris.len());
    for t in &tris {
        let mut ids = [0u32; 3];
        for (k, v) in t.iter().enumerate() {
            let n = vid.len() as u32;
            ids[k] = *vid.entry(key(*v)).or_insert(n);
        }
        tv.push(ids);
    }
    let mut edges: HashMap<(u32, u32), Vec<u32>> = HashMap::new();
    for (i, ids) in tv.iter().enumerate() {
        for k in 0..3 {
            let (a, b) = (ids[k], ids[(k + 1) % 3]);
            edges.entry((a.min(b), a.max(b))).or_default().push(i as u32);
        }
    }
    let cos_tol = 25f64.to_radians().cos();
    let mut segment = vec![u32::MAX; tris.len()];
    let mut n_seg = 0u32;
    for seed in 0..tris.len() {
        if segment[seed] != u32::MAX {
            continue;
        }
        segment[seed] = n_seg;
        let mut stack = vec![seed as u32];
        while let Some(t) = stack.pop() {
            let ids = tv[t as usize];
            for k in 0..3 {
                let (a, b) = (ids[k], ids[(k + 1) % 3]);
                if let Some(nb) = edges.get(&(a.min(b), a.max(b))) {
                    for &o in nb {
                        if segment[o as usize] == u32::MAX
                            && dot(normals[t as usize], normals[o as usize]) >= cos_tol
                        {
                            segment[o as usize] = n_seg;
                            stack.push(o);
                        }
                    }
                }
            }
        }
        n_seg += 1;
    }
    let bvh = StlBvh::build(&tris);
    Ok(StlData { tris, normals, segment, n_segments: n_seg as usize, min, max, bvh })
}

impl StlData {
    /// Inside test along a +X ray using crossing parity, filling a column of cells.
    /// Returns sorted x positions of surface crossings at (y, z).
    pub fn x_crossings(&self, y: f64, z: f64) -> Vec<f64> {
        let mut xs = Vec::new();
        if self.bvh.nodes.is_empty() {
            return xs;
        }
        let mut stack = Vec::with_capacity(64);
        stack.push(0usize);
        while let Some(ni) = stack.pop() {
            let n = &self.bvh.nodes[ni];
            if y < n.lo[1] || y > n.hi[1] || z < n.lo[2] || z > n.hi[2] {
                continue;
            }
            if n.count > 0 {
                for &ti in &self.bvh.order[n.start..n.start + n.count] {
                    let t = &self.tris[ti as usize];
                    let (a, b, c) = (t[0], t[1], t[2]);
                    // barycentric in the yz-plane
                    let d = (b[1] - a[1]) * (c[2] - a[2]) - (c[1] - a[1]) * (b[2] - a[2]);
                    if d.abs() < 1e-18 {
                        continue;
                    }
                    let l1 = ((y - a[1]) * (c[2] - a[2]) - (c[1] - a[1]) * (z - a[2])) / d;
                    let l2 = ((b[1] - a[1]) * (z - a[2]) - (y - a[1]) * (b[2] - a[2])) / d;
                    let l0 = 1.0 - l1 - l2;
                    if l0 >= 0.0 && l1 >= 0.0 && l2 >= 0.0 {
                        xs.push(l0 * a[0] + l1 * b[0] + l2 * c[0]);
                    }
                }
            } else {
                stack.push(n.left);
                stack.push(n.left + 1);
            }
        }
        xs.sort_by(|a, b| a.total_cmp(b));
        xs.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
        xs
    }
}

#[cfg(test)]
mod stl_bvh_tests {
    use super::*;

    fn brute_x_crossings(d: &StlData, y: f64, z: f64) -> Vec<f64> {
        let mut xs = Vec::new();
        for t in &d.tris {
            let (a, b, c) = (t[0], t[1], t[2]);
            let det = (b[1] - a[1]) * (c[2] - a[2]) - (c[1] - a[1]) * (b[2] - a[2]);
            if det.abs() < 1e-18 {
                continue;
            }
            let l1 = ((y - a[1]) * (c[2] - a[2]) - (c[1] - a[1]) * (z - a[2])) / det;
            let l2 = ((b[1] - a[1]) * (z - a[2]) - (y - a[1]) * (b[2] - a[2])) / det;
            let l0 = 1.0 - l1 - l2;
            if l0 >= 0.0 && l1 >= 0.0 && l2 >= 0.0 {
                xs.push(l0 * a[0] + l1 * b[0] + l2 * c[0]);
            }
        }
        xs.sort_by(|a, b| a.total_cmp(b));
        xs.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
        xs
    }

    #[test]
    fn stl_bvh_matches_bruteforce_queries() {
        let mut tris = Vec::new();
        let mut normals = Vec::new();
        let mut segment = Vec::new();
        // Subdivided octasphere with 512 triangles across 4 patches.
        let n = 8usize;
        let pts = |i: usize, j: usize| -> V3 {
            let u = std::f64::consts::PI * i as f64 / n as f64;
            let v = std::f64::consts::TAU * j as f64 / (2 * n) as f64;
            [10.0 * u.sin() * v.cos(), 10.0 * u.sin() * v.sin(), 10.0 * u.cos()]
        };
        for i in 0..n {
            for j in 0..2 * n {
                let a = pts(i, j);
                let b = pts(i + 1, j);
                let c = pts(i + 1, j + 1);
                let d = pts(i, j + 1);
                let seg = ((j / n) * 2 + (i >= n / 2) as usize) as u32;
                for tri in [[a, b, c], [a, c, d]] {
                    let cr = cross(sub(tri[1], tri[0]), sub(tri[2], tri[0]));
                    if norm(cr) > 1e-12 {
                        normals.push(normalize(cr));
                        tris.push(tri);
                        segment.push(seg);
                    }
                }
            }
        }
        let bvh = StlBvh::build(&tris);
        let stl = StlData { tris, normals, segment, n_segments: 4, min: [-10.0; 3], max: [10.0; 3], bvh };
        let shape = Shape::Stl { path: "s.stl".into(), scale: 1.0, offset: [0.0; 3], data: Some(Arc::new(stl)) };
        let Shape::Stl { data: Some(d), .. } = &shape else { unreachable!() };
        for &(y, z) in &[(0.13, 0.27), (-4.5, 3.1), (9.9, 9.9), (0.0, -7.2)] {
            assert_eq!(d.x_crossings(y, z), brute_x_crossings(d, y, z));
        }
        for p in [[12.3, 1.5, -2.0], [0.5, -15.0, 4.0], [-6.2, 6.2, 6.2]] {
            for patch in 0..4 {
                let brute = d.tris.iter().enumerate()
                    .filter(|(i, _)| d.segment[*i] as usize == patch)
                    .map(|(_, t)| closest_point_triangle(p, t))
                    .min_by(|a, b| norm(sub(p, *a)).total_cmp(&norm(sub(p, *b))));
                match (shape.project(patch, p), brute) {
                    (Some(g), Some(b)) => {
                        let dg = norm(sub(p, g));
                        let db = norm(sub(p, b));
                        assert!((dg - db).abs() < 1e-12, "patch {patch} p {p:?}: bvh dist {dg} vs brute {db}");
                    }
                    (None, None) => {}
                    (g, b) => panic!("mismatch: bvh {g:?} brute {b:?}"),
                }
            }
            let nr = normalize(p);
            let got = shape.best_patch(p, nr, 1.0, false);
            let mut want = (0, f64::INFINITY);
            for (i, t) in d.tris.iter().enumerate() {
                let dist = norm(sub(p, closest_point_triangle(p, t)));
                let nn = d.normals[i];
                let dp = dot(nn, nr).clamp(-1.0, 1.0);
                let lim = if nn.iter().any(|c| c.abs() > 0.999) { 0.5 } else { -0.2 };
                let s = if dp < lim { f64::INFINITY } else { dist + 1.0 - dp };
                if s < want.1 {
                    want = (d.segment[i] as usize, s);
                }
            }
            assert_eq!(got.0, want.0);
            assert!((got.1 - want.1).abs() < 1e-12);
        }
    }
}
