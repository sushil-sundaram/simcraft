//! Thermal radiation on boundary faces.
//!
//! * **To ambient** (`Load::Radiation`, `enclosure: false`): q = εσ(T⁴ − T∞⁴), linearised per
//!   iteration as h_r (T − T∞) with h_r = εσ(T² + T∞²)(T + T∞) — exact at convergence.
//! * **Surface to surface** (`enclosure: true`): the selected faces are grouped into patches; view
//!   factors F_ij come from cosine-weighted ray casting against every boundary face of the model
//!   (so parts block each other's view), with reciprocity enforced; rays that escape — or hit a face
//!   without a radiation load — see the ambient. Grey diffuse radiosity
//!   J = εσT⁴ + (1 − ε)(F J + F_amb σT∞⁴) is factorised once (it does not depend on T); the net flux
//!   q = J − G is linearised as q* + 4εσT*³ (T − T*).
//!
//! Temperatures are °C in the solvers and Kelvin here.

use crate::fem::face_shares;
use crate::geometry::*;
use crate::mesh::Mesh;
use crate::model::{Load, Model};
use anyhow::{Result, bail};
use rayon::prelude::*;

pub const SIGMA: f64 = 5.670_374_419e-8;
pub const KELVIN: f64 = 273.15;
/// most patches for surface-to-surface radiation (the radiosity matrix is dense)
const MAX_PATCHES: usize = 1500;
const RAYS_PER_PATCH: usize = 1024;

struct Face {
    nodes: [u32; 4],
    /// consistent nodal areas ∫Nᵢ dA [m²]
    a: [f64; 4],
}

struct AmbFace {
    f: Face,
    eps: f64,
    ta: f64,
    load: usize,
}

struct S2s {
    faces: Vec<(Face, usize)>,
    area: Vec<f64>,
    eps: Vec<f64>,
    ta: Vec<f64>,
    load: Vec<usize>,
    /// F[i·n + j], and the fraction that reaches the ambient
    f: Vec<f64>,
    famb: Vec<f64>,
    /// LU factors of (I − diag(1 − ε) F) with row permutation
    lu: Vec<f64>,
    perm: Vec<usize>,
}

pub struct Radiation {
    amb: Vec<AmbFace>,
    s2s: Option<S2s>,
    /// load id and label per radiation load (index = load position in the model)
    loads: Vec<(u32, String)>,
    pub patches: usize,
    pub rays: usize,
}

/// Linearised radiation terms per node: add `diag` to the conduction matrix diagonal and `rhs`
/// to the load vector (both in W/K and W, temperatures in °C).
pub struct Linear {
    pub diag: Vec<f64>,
    pub rhs: Vec<f64>,
}

// ------------------------------------------------------------------ ray casting

struct Tri {
    v: [V3; 3],
    face: u32,
}

struct BNode {
    lo: V3,
    hi: V3,
    /// leaf: (start, count) into `order`; inner: children at `left`, `left + 1`
    left: usize,
    start: usize,
    count: usize,
}

struct Bvh {
    tris: Vec<Tri>,
    order: Vec<usize>,
    nodes: Vec<BNode>,
}

impl Bvh {
    fn new(tris: Vec<Tri>) -> Bvh {
        let mut b = Bvh { order: (0..tris.len()).collect(), tris, nodes: Vec::new() };
        let n = b.order.len();
        b.nodes.push(BNode { lo: [0.0; 3], hi: [0.0; 3], left: 0, start: 0, count: n });
        b.split(0);
        b
    }
    fn bounds(&self, start: usize, count: usize) -> (V3, V3) {
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for &t in &self.order[start..start + count] {
            for v in &self.tris[t].v {
                for k in 0..3 {
                    lo[k] = lo[k].min(v[k]);
                    hi[k] = hi[k].max(v[k]);
                }
            }
        }
        (lo, hi)
    }
    fn split(&mut self, ni: usize) {
        let (start, count) = (self.nodes[ni].start, self.nodes[ni].count);
        let (lo, hi) = self.bounds(start, count);
        self.nodes[ni].lo = lo;
        self.nodes[ni].hi = hi;
        if count <= 4 {
            return;
        }
        let ax = (0..3).max_by(|&a, &b| (hi[a] - lo[a]).total_cmp(&(hi[b] - lo[b]))).unwrap();
        let cen = |t: usize| (self.tris[t].v[0][ax] + self.tris[t].v[1][ax] + self.tris[t].v[2][ax]) / 3.0;
        let mut idx: Vec<usize> = self.order[start..start + count].to_vec();
        idx.sort_by(|&a, &b| cen(a).total_cmp(&cen(b)));
        self.order[start..start + count].copy_from_slice(&idx);
        let half = count / 2;
        let left = self.nodes.len();
        self.nodes.push(BNode { lo: [0.0; 3], hi: [0.0; 3], left: 0, start, count: half });
        self.nodes.push(BNode { lo: [0.0; 3], hi: [0.0; 3], left: 0, start: start + half, count: count - half });
        self.nodes[ni].left = left;
        self.nodes[ni].count = 0;
        self.split(left);
        self.split(left + 1);
    }
    /// nearest hit: (distance, face)
    fn hit(&self, o: V3, d: V3) -> Option<(f64, u32)> {
        let inv = d.map(|v| 1.0 / v);
        let mut best: Option<(f64, u32)> = None;
        let mut stack = vec![0usize];
        while let Some(ni) = stack.pop() {
            let n = &self.nodes[ni];
            // slab test
            let (mut t0, mut t1) = (0.0f64, best.map(|b| b.0).unwrap_or(f64::INFINITY));
            for k in 0..3 {
                let (a, b) = ((n.lo[k] - o[k]) * inv[k], (n.hi[k] - o[k]) * inv[k]);
                let (a, b) = if a < b { (a, b) } else { (b, a) };
                t0 = t0.max(a);
                t1 = t1.min(b);
            }
            if t0 > t1 {
                continue;
            }
            if n.count > 0 {
                for &t in &self.order[n.start..n.start + n.count] {
                    if let Some(s) = ray_tri(o, d, &self.tris[t].v) {
                        if best.is_none_or(|b| s < b.0) {
                            best = Some((s, self.tris[t].face));
                        }
                    }
                }
            } else {
                stack.push(n.left);
                stack.push(n.left + 1);
            }
        }
        best
    }
}

/// Möller–Trumbore, both sides; distance along the (unit) ray.
fn ray_tri(o: V3, d: V3, v: &[V3; 3]) -> Option<f64> {
    let e1 = sub(v[1], v[0]);
    let e2 = sub(v[2], v[0]);
    let p = cross(d, e2);
    let det = dot(e1, p);
    if det.abs() < 1e-14 {
        return None;
    }
    let inv = 1.0 / det;
    let s = sub(o, v[0]);
    let u = dot(s, p) * inv;
    if !(-1e-9..=1.0 + 1e-9).contains(&u) {
        return None;
    }
    let q = cross(s, e1);
    let w = dot(d, q) * inv;
    if w < -1e-9 || u + w > 1.0 + 1e-9 {
        return None;
    }
    let t = dot(e2, q) * inv;
    (t > 1e-9).then_some(t)
}

// ------------------------------------------------------------------ setup

/// View factor between two coaxial parallel rectangles a × b at distance c (Hottel / Howell C-11).
pub fn parallel_rectangles_vf(a: f64, b: f64, c: f64) -> f64 {
    let (x, y) = (a / c, b / c);
    let (x1, y1) = (1.0 + x * x, 1.0 + y * y);
    2.0 / (std::f64::consts::PI * x * y)
        * ((x1 * y1 / (1.0 + x * x + y * y)).sqrt().ln() + x * y1.sqrt() * (x / y1.sqrt()).atan() + y * x1.sqrt() * (y / x1.sqrt()).atan() - x * x.atan() - y * y.atan())
}

impl Radiation {
    /// Radiation boundary conditions of the model (None when there are none).
    pub fn new(model: &Model, mesh: &Mesh, prog: &dyn Fn(&str)) -> Result<Option<Radiation>> {
        let mut amb = Vec::new();
        let mut s2s_faces: Vec<(usize, f64, f64, usize)> = Vec::new(); // face, eps, ta, load
        let mut loads = Vec::new();
        for it in &model.loads {
            let Load::Radiation { faces, emissivity, ambient, enclosure } = &it.load else { continue };
            if !(*emissivity > 0.0 && *emissivity <= 1.0) {
                bail!("radiation #{}: emissivity must be in (0, 1]", it.id);
            }
            loads.push((it.id, it.load.label().to_string()));
            for fi in mesh.resolve_faces(faces)? {
                let fi = fi as usize;
                if mesh.faces[fi].outer {
                    continue;
                }
                if *enclosure {
                    s2s_faces.push((fi, *emissivity, ambient + KELVIN, loads.len() - 1));
                } else {
                    let a = face_shares(mesh, fi).1;
                    amb.push(AmbFace { f: Face { nodes: mesh.faces[fi].nodes, a }, eps: *emissivity, ta: ambient + KELVIN, load: loads.len() - 1 });
                }
            }
        }
        if loads.is_empty() {
            return Ok(None);
        }
        let mut rad = Radiation { amb, s2s: None, loads, patches: 0, rays: 0 };
        if !s2s_faces.is_empty() {
            prog("radiation view factors");
            rad.s2s = Some(Self::view_factors(mesh, &s2s_faces, &mut rad.patches, &mut rad.rays)?);
        }
        Ok(Some(rad))
    }

    fn view_factors(mesh: &Mesh, sel: &[(usize, f64, f64, usize)], npatch: &mut usize, nrays: &mut usize) -> Result<S2s> {
        let h = mesh.h;
        // outward normal (unit) and centre of every selected face
        let info: Vec<(V3, V3, f64)> = sel
            .iter()
            .map(|&(fi, ..)| {
                let (v, a) = face_shares(mesh, fi);
                let n = v.iter().fold([0.0; 3], |s, x| add(s, *x));
                let c = scale(mesh.faces[fi].nodes.iter().fold([0.0; 3], |s, &k| add(s, mesh.nodes[k as usize])), 0.25);
                (normalize(n), c, a.iter().sum::<f64>())
            })
            .collect();
        // patches: faces of one load, one coarse cell and one dominant normal direction
        let mut cell = 2.0 * h;
        let mut keys: Vec<(usize, [i64; 3], usize)>;
        loop {
            keys = sel
                .iter()
                .zip(&info)
                .map(|(&(_, _, _, load), (n, c, _))| {
                    let ax = (0..3).max_by(|&a, &b| n[a].abs().total_cmp(&n[b].abs())).unwrap();
                    let dir = ax * 2 + (n[ax] > 0.0) as usize;
                    (load, [0, 1, 2].map(|k| (c[k] / cell).floor() as i64), dir)
                })
                .collect();
            let mut u = keys.clone();
            u.sort();
            u.dedup();
            if u.len() <= MAX_PATCHES {
                break;
            }
            cell *= 1.5;
        }
        let mut uniq = keys.clone();
        uniq.sort();
        uniq.dedup();
        let np = uniq.len();
        let pid: Vec<usize> = keys.iter().map(|k| uniq.binary_search(k).unwrap()).collect();
        let mut area = vec![0.0; np];
        let (mut eps, mut ta, mut load) = (vec![0.0; np], vec![0.0; np], vec![0usize; np]);
        let mut members: Vec<Vec<usize>> = vec![Vec::new(); np];
        for (k, &(_, e, t, l)) in sel.iter().enumerate() {
            let p = pid[k];
            area[p] += info[k].2;
            eps[p] = e;
            ta[p] = t;
            load[p] = l;
            members[p].push(k);
        }
        // BVH over every solid boundary face of the model (blocking included)
        let mut face_patch = vec![usize::MAX; mesh.faces.len()];
        for (k, &(fi, ..)) in sel.iter().enumerate() {
            face_patch[fi] = pid[k];
        }
        let mut tris = Vec::new();
        for (fi, f) in mesh.faces.iter().enumerate() {
            if f.outer {
                continue;
            }
            let q = f.nodes.map(|n| mesh.nodes[n as usize]);
            tris.push(Tri { v: [q[0], q[1], q[2]], face: fi as u32 });
            tris.push(Tri { v: [q[0], q[2], q[3]], face: fi as u32 });
        }
        let bvh = Bvh::new(tris);
        // cosine-weighted rays from each patch, spread over its faces by area
        let counts: Vec<(Vec<(usize, f64)>, f64, usize)> = (0..np)
            .into_par_iter()
            .map(|p| {
                let mut seed = 0x9E37_79B9_7F4A_7C15u64 ^ (p as u64).wrapping_mul(0x2545_F491_4F6C_DD1D);
                let mut rnd = || {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    (seed >> 11) as f64 / (1u64 << 53) as f64
                };
                let mut hits: Vec<(usize, f64)> = Vec::new();
                let mut amb = 0.0;
                let mut total = 0usize;
                for &k in &members[p] {
                    let (fi, ..) = sel[k];
                    let (n, _, a) = info[k];
                    let nr = ((RAYS_PER_PATCH as f64 * a / area[p]).round() as usize).max(4);
                    let q = mesh.faces[fi].nodes.map(|m| mesh.nodes[m as usize]);
                    let t1 = normalize(if n[0].abs() < 0.9 { cross(n, [1.0, 0.0, 0.0]) } else { cross(n, [0.0, 1.0, 0.0]) });
                    let t2 = cross(n, t1);
                    for _ in 0..nr {
                        let (u, v) = (rnd(), rnd());
                        let pt = add(add(scale(q[0], (1.0 - u) * (1.0 - v)), scale(q[1], u * (1.0 - v))), add(scale(q[2], u * v), scale(q[3], (1.0 - u) * v)));
                        let (r1, r2) = (rnd(), rnd());
                        let phi = std::f64::consts::TAU * r1;
                        let st = r2.sqrt();
                        let d = add(add(scale(t1, phi.cos() * st), scale(t2, phi.sin() * st)), scale(n, (1.0 - r2).sqrt()));
                        let o = add(pt, scale(n, 1e-4 * h));
                        total += 1;
                        match bvh.hit(o, d) {
                            Some((_, f)) if face_patch[f as usize] != usize::MAX => {
                                let j = face_patch[f as usize];
                                match hits.iter_mut().find(|x| x.0 == j) {
                                    Some(x) => x.1 += 1.0,
                                    None => hits.push((j, 1.0)),
                                }
                            }
                            // escapes, or hits a face that does not take part: the ambient
                            _ => amb += 1.0,
                        }
                    }
                }
                (hits, amb, total)
            })
            .collect();
        *nrays = counts.iter().map(|c| c.2).sum();
        let mut f = vec![0.0; np * np];
        for (i, (hits, _, total)) in counts.iter().enumerate() {
            for &(j, c) in hits {
                f[i * np + j] = c / *total as f64;
            }
        }
        // reciprocity A_i F_ij = A_j F_ji, then the rest of each row sees the ambient
        for i in 0..np {
            for j in i + 1..np {
                let s = 0.5 * (area[i] * f[i * np + j] + area[j] * f[j * np + i]);
                f[i * np + j] = s / area[i];
                f[j * np + i] = s / area[j];
            }
        }
        let mut famb = vec![0.0; np];
        for i in 0..np {
            let row: f64 = f[i * np..(i + 1) * np].iter().sum();
            if row > 1.0 {
                for v in &mut f[i * np..(i + 1) * np] {
                    *v /= row;
                }
            }
            famb[i] = (1.0 - row).max(0.0);
        }
        // LU of (I − diag(1 − ε) F)
        let mut lu = vec![0.0; np * np];
        for i in 0..np {
            for j in 0..np {
                lu[i * np + j] = if i == j { 1.0 } else { 0.0 } - (1.0 - eps[i]) * f[i * np + j];
            }
        }
        let mut perm: Vec<usize> = (0..np).collect();
        for k in 0..np {
            let piv = (k..np).max_by(|&a, &b| lu[a * np + k].abs().total_cmp(&lu[b * np + k].abs())).unwrap();
            if piv != k {
                for j in 0..np {
                    lu.swap(k * np + j, piv * np + j);
                }
                perm.swap(k, piv);
            }
            let d = lu[k * np + k];
            if d.abs() < 1e-14 {
                bail!("radiosity system is singular");
            }
            let (head, tail) = lu.split_at_mut((k + 1) * np);
            let rowk = &head[k * np..];
            tail.par_chunks_mut(np).for_each(|ri| {
                let m = ri[k] / d;
                if m != 0.0 {
                    ri[k] = m;
                    for j in k + 1..np {
                        ri[j] -= m * rowk[j];
                    }
                }
            });
        }
        *npatch = np;
        let faces = sel.iter().enumerate().map(|(k, &(fi, ..))| (Face { nodes: mesh.faces[fi].nodes, a: face_shares(mesh, fi).1 }, pid[k])).collect();
        Ok(S2s { faces, area, eps, ta, load, f, famb, lu, perm })
    }

    /// Net outgoing flux q [W/m²] and patch temperature [K] of every enclosure patch at nodal
    /// temperatures `t` [°C].
    fn s2s_flux(&self, t: &[f64]) -> Option<(Vec<f64>, Vec<f64>)> {
        let s = self.s2s.as_ref()?;
        let np = s.area.len();
        let mut tp = vec![0.0; np];
        for (fc, p) in &s.faces {
            for k in 0..4 {
                tp[*p] += fc.a[k] * (t[fc.nodes[k] as usize] + KELVIN);
            }
        }
        for p in 0..np {
            tp[p] /= s.area[p].max(1e-300);
        }
        // J = (I − (1−ε)F)⁻¹ [εσT⁴ + (1−ε) F_amb σ T∞⁴]
        let b: Vec<f64> = (0..np).map(|i| s.eps[i] * SIGMA * tp[i].powi(4) + (1.0 - s.eps[i]) * s.famb[i] * SIGMA * s.ta[i].powi(4)).collect();
        let mut y: Vec<f64> = s.perm.iter().map(|&p| b[p]).collect();
        for i in 0..np {
            let mut v = y[i];
            for j in 0..i {
                v -= s.lu[i * np + j] * y[j];
            }
            y[i] = v;
        }
        for i in (0..np).rev() {
            let mut v = y[i];
            for j in i + 1..np {
                v -= s.lu[i * np + j] * y[j];
            }
            y[i] = v / s.lu[i * np + i];
        }
        let j = y;
        let q: Vec<f64> = (0..np)
            .into_par_iter()
            .map(|i| {
                let g: f64 = (0..np).map(|k| s.f[i * np + k] * j[k]).sum::<f64>() + s.famb[i] * SIGMA * s.ta[i].powi(4);
                j[i] - g
            })
            .collect();
        Some((q, tp))
    }

    /// Linearised terms at the current nodal temperatures `t` [°C].
    pub fn linearize(&self, t: &[f64]) -> Linear {
        let nn = t.len();
        let mut diag = vec![0.0; nn];
        let mut rhs = vec![0.0; nn];
        for af in &self.amb {
            for k in 0..4 {
                let n = af.f.nodes[k] as usize;
                let tk = (t[n] + KELVIN).max(1.0);
                // secant coefficient: h (T − T∞) = εσ (T⁴ − T∞⁴) exactly
                let hr = af.eps * SIGMA * (tk * tk + af.ta * af.ta) * (tk + af.ta);
                diag[n] += hr * af.f.a[k];
                rhs[n] += hr * af.f.a[k] * (af.ta - KELVIN);
            }
        }
        if let (Some(s), Some((q, tp))) = (&self.s2s, self.s2s_flux(t)) {
            for (fc, p) in &s.faces {
                let p = *p;
                let hp = 4.0 * s.eps[p] * SIGMA * tp[p].powi(3);
                for k in 0..4 {
                    let n = fc.nodes[k] as usize;
                    diag[n] += hp * fc.a[k];
                    rhs[n] += (hp * (tp[p] - KELVIN) - q[p]) * fc.a[k];
                }
            }
        }
        Linear { diag, rhs }
    }

    /// Net heat leaving through each radiation load [W] at nodal temperatures `t` [°C]:
    /// (load id, label, W).
    pub fn heat_flows(&self, t: &[f64]) -> Vec<(u32, String, f64)> {
        let mut w = vec![0.0; self.loads.len()];
        for af in &self.amb {
            for k in 0..4 {
                let tk = t[af.f.nodes[k] as usize] + KELVIN;
                w[af.load] += af.eps * SIGMA * (tk.powi(4) - af.ta.powi(4)) * af.f.a[k];
            }
        }
        if let (Some(s), Some((q, _))) = (&self.s2s, self.s2s_flux(t)) {
            for p in 0..s.area.len() {
                w[s.load[p]] += q[p] * s.area[p];
            }
        }
        self.loads.iter().zip(w).map(|((id, l), v)| (*id, l.clone(), v)).collect()
    }

    /// Net heat lost by all enclosure patches to the ambient [W] (the rest is exchanged).
    pub fn enclosure_to_ambient(&self, t: &[f64]) -> Option<f64> {
        let (q, _) = self.s2s_flux(t)?;
        let s = self.s2s.as_ref()?;
        Some((0..s.area.len()).map(|p| q[p] * s.area[p]).sum())
    }
}
