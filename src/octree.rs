//! 2:1-balanced octree voxel mesher and hanging-node constraint handler for FE solvers.
//!
//! Structural, thermal, modal, vibration, buckling, plasticity, and field (electrostatic,
//! current, magnetostatic) solvers support locally refined octree meshes where adjacent leaves
//! differ by at most one refinement level across every vertex, edge, and face (26-neighbor balance).
//! Hanging nodes along coarse–fine transitions are eliminated algebraically via affine constraints
//! `u_full = C * u_master` (`K_red = C^T K C`, `f_red = C^T f`), preserving SPD structure, exact
//! patch tests, mass conservation, and C^0 continuity.

use crate::fem::check_cancel;
use crate::geometry::{Body, Op, Role, Shape, StlData, V3, add, norm, normalize, scale, sub};
use crate::material::Material;
use crate::mesh::{BFace, CORNER, Mesh, PatchInfo, REGION_ENCLOSURE, REGION_SOLID, dir_normal};
use crate::model::Model;
use crate::sparse::NodeSparse;

struct PendingBolt {
    load_id: u32,
    bi: usize,
    ax: usize,
    cut_val: f64,
}
use anyhow::{Result, bail};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::AtomicBool;

/// Affine constraint on a hanging mesh node: `u[node] = Σ_k weight_k * u[parent_k]`, where all
/// `parent_k` are independent (master) nodes and `Σ weight_k = 1`.
#[derive(Clone, Debug)]
pub struct HangingNode {
    pub node: u32,
    pub parents: Vec<(u32, f64)>,
}

/// Condensed master-node mapping for hanging-node elimination (`u_full = C * u_master`).
///
/// When `hanging` is empty (uniform grids), every node maps to itself with weight 1.0 and
/// zero extra overhead.
#[derive(Clone, Debug)]
pub struct HangingMap {
    pub nn: usize,
    pub n_master: usize,
    /// `u32::MAX` for constrained hanging nodes; `0..n_master` for master mesh nodes.
    pub to_master: Vec<u32>,
    /// Original mesh node id `0..nn` for each master `0..n_master`.
    pub masters: Vec<u32>,
    /// CSR rows of `C` over mesh nodes `0..nn` (`(master_idx, weight)`).
    pub ptr: Vec<usize>,
    pub ent: Vec<(u32, f64)>,
    pub has_hanging: bool,
}

impl HangingMap {
    pub fn build(nn: usize, hanging: &[HangingNode]) -> Self {
        if hanging.is_empty() {
            return Self {
                nn,
                n_master: nn,
                to_master: (0..nn as u32).collect(),
                masters: (0..nn as u32).collect(),
                ptr: (0..=nn).collect(),
                ent: (0..nn).map(|i| (i as u32, 1.0)).collect(),
                has_hanging: false,
            };
        }
        let mut hmap: HashMap<u32, &[(u32, f64)]> = HashMap::with_capacity(hanging.len());
        for h in hanging {
            hmap.insert(h.node, &h.parents);
        }
        let mut to_master = vec![u32::MAX; nn];
        let mut masters = Vec::with_capacity(nn.saturating_sub(hanging.len()));
        for i in 0..nn {
            if !hmap.contains_key(&(i as u32)) {
                to_master[i] = masters.len() as u32;
                masters.push(i as u32);
            }
        }
        let n_master = masters.len();
        let mut ptr = Vec::with_capacity(nn + 1);
        let mut ent = Vec::with_capacity(nn + hanging.len() * 3);
        ptr.push(0);
        for i in 0..nn {
            if let Some(pars) = hmap.get(&(i as u32)) {
                for &(p, w) in *pars {
                    let mp = to_master[p as usize];
                    debug_assert_ne!(mp, u32::MAX, "hanging node {i} references non-master parent {p}");
                    if mp != u32::MAX && w != 0.0 {
                        ent.push((mp, w));
                    }
                }
            } else {
                ent.push((to_master[i], 1.0));
            }
            ptr.push(ent.len());
        }
        Self { nn, n_master, to_master, masters, ptr, ent, has_hanging: true }
    }

    #[inline]
    pub fn total_nodes(&self, aux_nodes: usize) -> usize {
        self.n_master + aux_nodes
    }

    /// Map a node id (`0..nn` or auxiliary pilot `nn..nn+aux`) to `(master_idx, weight)` pairs.
    #[inline]
    pub fn push_terms(&self, node: u32, out: &mut Vec<(u32, f64)>) {
        let idx = node as usize;
        if idx < self.nn {
            out.extend_from_slice(&self.ent[self.ptr[idx]..self.ptr[idx + 1]]);
        } else {
            out.push(((self.n_master + (idx - self.nn)) as u32, 1.0));
        }
    }

    /// Condense element stiffness scatter `K_red += C_e^T (scale * ke) C_e` onto `k` (bs = 1 or 3).
    pub fn scatter_elem(&self, k: &mut NodeSparse, en: &[u32; 8], ke: &[f64], scale_val: f64) {
        if !self.has_hanging {
            k.add_element(en, ke, scale_val);
            return;
        }
        let all_master = en.iter().all(|&n| self.to_master[n as usize] != u32::MAX);
        if all_master {
            let mapped = en.map(|n| self.to_master[n as usize]);
            k.add_element(&mapped, ke, scale_val);
            return;
        }
        let bs = k.bs;
        let mut uniq: Vec<u32> = Vec::with_capacity(16);
        for &n in en {
            for &(m, _) in &self.ent[self.ptr[n as usize]..self.ptr[n as usize + 1]] {
                if !uniq.contains(&m) {
                    uniq.push(m);
                }
            }
        }
        let nu = uniq.len();
        // Local weights: for each corner a in 0..8 and unique master u in 0..nu
        let mut cw = vec![0.0f64; 8 * nu];
        for (a, &n) in en.iter().enumerate() {
            for &(m, w) in &self.ent[self.ptr[n as usize]..self.ptr[n as usize + 1]] {
                let u = uniq.iter().position(|&x| x == m).unwrap();
                cw[a * nu + u] += w;
            }
        }
        for ui in 0..nu {
            let mi = uniq[ui] as usize;
            for uj in 0..nu {
                let mj = uniq[uj] as usize;
                if bs == 1 {
                    let mut val = 0.0;
                    for a in 0..8 {
                        let wa = cw[a * nu + ui];
                        if wa == 0.0 {
                            continue;
                        }
                        for b in 0..8 {
                            let wb = cw[b * nu + uj];
                            if wb != 0.0 {
                                val += wa * wb * ke[a * 8 + b];
                            }
                        }
                    }
                    if val != 0.0 {
                        k.add_block(mi, mj, &[val * scale_val]);
                    }
                } else {
                    let mut blk = [0.0f64; 9];
                    for a in 0..8 {
                        let wa = cw[a * nu + ui];
                        if wa == 0.0 {
                            continue;
                        }
                        for b in 0..8 {
                            let wb = cw[b * nu + uj];
                            if wb == 0.0 {
                                continue;
                            }
                            let ww = wa * wb * scale_val;
                            for c in 0..3 {
                                let row = (a * 3 + c) * 24 + b * 3;
                                for d in 0..3 {
                                    blk[c * 3 + d] += ww * ke[row + d];
                                }
                            }
                        }
                    }
                    if blk.iter().any(|&v| v != 0.0) {
                        k.add_block(mi, mj, &blk);
                    }
                }
            }
        }
    }

    /// Add `C_a^T * blk * C_b` between two possibly-hanging or auxiliary nodes `(na, nb)`.
    pub fn add_pair_block(&self, k: &mut NodeSparse, na: u32, nb: u32, blk: &[f64]) {
        if !self.has_hanging {
            k.add_block(na as usize, nb as usize, blk);
            return;
        }
        let bs2 = k.bs * k.bs;
        let mut ta = Vec::with_capacity(4);
        let mut tb = Vec::with_capacity(4);
        self.push_terms(na, &mut ta);
        self.push_terms(nb, &mut tb);
        let mut tmp = vec![0.0; bs2];
        for &(ma, wa) in &ta {
            for &(mb, wb) in &tb {
                let s = wa * wb;
                if s != 0.0 {
                    for (q, &v) in blk[..bs2].iter().enumerate() {
                        tmp[q] = v * s;
                    }
                    k.add_block(ma as usize, mb as usize, &tmp);
                }
            }
        }
    }

    /// Extra graph edges required in master node numbering for `NodeSparse::new_with_edges`.
    pub fn extra_edges(&self, elems: &[[u32; 8]], raw_edges: &[(u32, u32)]) -> Vec<(u32, u32)> {
        let mut out = Vec::new();
        let mut uniq = Vec::with_capacity(16);
        if self.has_hanging {
            for en in elems {
                if en.iter().all(|&n| self.to_master[n as usize] != u32::MAX) {
                    continue;
                }
                uniq.clear();
                for &n in en {
                    for &(m, _) in &self.ent[self.ptr[n as usize]..self.ptr[n as usize + 1]] {
                        if !uniq.contains(&m) {
                            uniq.push(m);
                        }
                    }
                }
                for i in 0..uniq.len() {
                    for j in (i + 1)..uniq.len() {
                        let (a, b) = (uniq[i], uniq[j]);
                        out.push(if a < b { (a, b) } else { (b, a) });
                    }
                }
            }
        }
        let mut ta = Vec::with_capacity(4);
        let mut tb = Vec::with_capacity(4);
        for &(a, b) in raw_edges {
            if !self.has_hanging {
                out.push((a, b));
                continue;
            }
            ta.clear();
            tb.clear();
            self.push_terms(a, &mut ta);
            self.push_terms(b, &mut tb);
            for &(ma, _) in &ta {
                for &(mb, _) in &tb {
                    if ma != mb {
                        out.push(if ma < mb { (ma, mb) } else { (mb, ma) });
                    }
                }
            }
            for i in 0..ta.len() {
                for j in (i + 1)..ta.len() {
                    let (x, y) = (ta[i].0, ta[j].0);
                    if x != y {
                        out.push(if x < y { (x, y) } else { (y, x) });
                    }
                }
            }
            for i in 0..tb.len() {
                for j in (i + 1)..tb.len() {
                    let (x, y) = (tb[i].0, tb[j].0);
                    if x != y {
                        out.push(if x < y { (x, y) } else { (y, x) });
                    }
                }
            }
        }
        out.sort_unstable();
        out.dedup();
        out
    }

    /// Condense forces: `f_red = C^T f_full` (including `aux_nodes` past `nn`).
    pub fn restrict_vec(&self, bs: usize, aux_nodes: usize, full: &[f64]) -> Vec<f64> {
        if !self.has_hanging {
            return full.to_vec();
        }
        let mut out = vec![0.0; (self.n_master + aux_nodes) * bs];
        for n in 0..self.nn {
            for &(m, w) in &self.ent[self.ptr[n]..self.ptr[n + 1]] {
                for c in 0..bs {
                    out[m as usize * bs + c] += w * full[n * bs + c];
                }
            }
        }
        let start_full = self.nn * bs;
        let start_red = self.n_master * bs;
        for i in 0..(aux_nodes * bs) {
            if start_full + i < full.len() {
                out[start_red + i] = full[start_full + i];
            }
        }
        out
    }

    /// Expand solution DOFs: `u_full = C * u_master` (plus auxiliary DOFs past `n_master`).
    pub fn prolong_vec(&self, bs: usize, aux_nodes: usize, red: &[f64]) -> Vec<f64> {
        if !self.has_hanging {
            return red.to_vec();
        }
        let mut out = vec![0.0; (self.nn + aux_nodes) * bs];
        for n in 0..self.nn {
            for &(m, w) in &self.ent[self.ptr[n]..self.ptr[n + 1]] {
                for c in 0..bs {
                    out[n * bs + c] += w * red[m as usize * bs + c];
                }
            }
        }
        let start_full = self.nn * bs;
        let start_red = self.n_master * bs;
        for i in 0..(aux_nodes * bs) {
            if start_red + i < red.len() {
                out[start_full + i] = red[start_red + i];
            }
        }
        out
    }

    /// Map Dirichlet values and dof owners to master DOFs (`0..n_master + aux_nodes`).
    pub fn condense_fixed(&self, bs: usize, aux_nodes: usize, fixed_full: &[Option<f64>], owner_full: &[u32]) -> (Vec<Option<f64>>, Vec<u32>) {
        if !self.has_hanging {
            return (fixed_full.to_vec(), owner_full.to_vec());
        }
        let total = (self.n_master + aux_nodes) * bs;
        let mut fx = vec![None; total];
        let mut own = vec![0u32; total];
        for (m, &orig) in self.masters.iter().enumerate() {
            for c in 0..bs {
                let s = orig as usize * bs + c;
                let d = m * bs + c;
                if s < fixed_full.len() {
                    fx[d] = fixed_full[s];
                }
                if s < owner_full.len() {
                    own[d] = owner_full[s];
                }
            }
        }
        for i in 0..(aux_nodes * bs) {
            let s = self.nn * bs + i;
            let d = self.n_master * bs + i;
            if s < fixed_full.len() {
                fx[d] = fixed_full[s];
            }
            if s < owner_full.len() {
                own[d] = owner_full[s];
            }
        }
        (fx, own)
    }

    /// Condense diagonal lumped quantities (mass, heat capacity): row-sum lumping preserves total.
    pub fn condense_diag(&self, bs: usize, aux_nodes: usize, diag_full: &[f64]) -> Vec<f64> {
        self.restrict_vec(bs, aux_nodes, diag_full)
    }
}

/// Optional spatial target hints carried between adaptive Zienkiewicz–Zhu refinement passes.
#[derive(Clone, Debug, Default)]
pub struct AdaptiveHint {
    /// Regions `(box_lo, box_hi, target_level)` forced to at least `target_level`.
    pub regions: Vec<(V3, V3, u8)>,
}

/// Spatial refinement target computed from user body/face sizing, automatic features, or ZZ hints.
struct RefineZone {
    lo: V3,
    hi: V3,
    target_level: u8,
    kind: ZoneKind,
}

enum ZoneKind {
    Box,
    Body,
    Patch { bi: usize, patch: usize, band: f64 },
}

fn box_overlaps(a_lo: V3, a_hi: V3, b_lo: V3, b_hi: V3) -> bool {
    (0..3).all(|k| a_hi[k] >= b_lo[k] && a_lo[k] <= b_hi[k])
}

fn level_for_size(base_h: f64, size_mm: f64, max_levels: u8) -> u8 {
    if !(size_mm.is_finite() && size_mm > 0.0) || size_mm >= base_h {
        return 0;
    }
    ((base_h / size_mm).log2().ceil().max(0.0) as u8).min(max_levels)
}

fn collect_refine_zones(
    bodies: &[Body],
    base_h: f64,
    max_levels: u8,
    model: Option<&Model>,
    hint: Option<&AdaptiveHint>,
) -> Result<Vec<RefineZone>> {
    let mut zones = Vec::new();
    let h_min_floor = base_h / (1usize << max_levels) as f64;
    if let Some(m) = model {
        for r in &m.mesh.refinements {
            if !r.size_mm.is_finite() || r.size_mm <= 0.0 {
                bail!("mesh refinement size_mm must be finite and positive");
            }
            let lvl = level_for_size(base_h, r.size_mm, max_levels);
            if lvl == 0 {
                continue;
            }
            let cell_target = base_h / (1usize << lvl) as f64;
            if let Some(bname) = &r.body
                && r.faces.is_empty() {
                    let Some(bi) = bodies.iter().position(|b| b.name.eq_ignore_ascii_case(bname.trim())) else {
                        bail!("mesh refinement references unknown body '{bname}'");
                    };
                    let (lo, hi) = bodies[bi].shape.bounds();
                    zones.push(RefineZone { lo, hi, target_level: lvl, kind: ZoneKind::Body });
                }
            for sel in &r.faces {
                let sl = sel.trim().to_lowercase();
                let mut matched = false;
                if let Some(side) = sl.strip_prefix("bbox.") {
                    let names = ["xmin", "xmax", "ymin", "ymax", "zmin", "zmax"];
                    let Some(d) = names.iter().position(|n| *n == side) else {
                        bail!("unknown bbox side '{side}', use one of {names:?}");
                    };
                    matched = true;
                    let ax = d / 2;
                    let (dom_lo, dom_hi) = crate::mesh::domain_bounds(bodies, false);
                    let mut zlo = dom_lo;
                    let mut zhi = dom_hi;
                    let target_x = if d % 2 == 0 { dom_lo[ax] } else { dom_hi[ax] };
                    let pad = 1.8 * cell_target;
                    zlo[ax] = target_x - pad;
                    zhi[ax] = target_x + pad;
                    zones.push(RefineZone { lo: zlo, hi: zhi, target_level: lvl, kind: ZoneKind::Box });
                } else if sl == "all" {
                    matched = true;
                    for (bi, b) in bodies.iter().enumerate() {
                        if b.role != Role::Solid {
                            continue;
                        }
                        let (lo, hi) = b.shape.bounds();
                        let pad = 2.0 * cell_target;
                        for (lp, _) in b.shape.patch_names().iter().enumerate() {
                            zones.push(RefineZone {
                                lo: sub(lo, [pad; 3]),
                                hi: add(hi, [pad; 3]),
                                target_level: lvl,
                                kind: ZoneKind::Patch { bi, patch: lp, band: 1.5 * cell_target },
                            });
                        }
                    }
                } else {
                    for (bi, b) in bodies.iter().enumerate() {
                        let bname = b.name.to_lowercase();
                        let (lo, hi) = b.shape.bounds();
                        let pad = 2.0 * cell_target;
                        if sl == bname && !r.faces.is_empty() {
                            matched = true;
                            zones.push(RefineZone { lo, hi, target_level: lvl, kind: ZoneKind::Body });
                        } else {
                            for (lp, p) in b.shape.patch_names().iter().enumerate() {
                                if format!("{bname}.{}", p.to_lowercase()) == sl {
                                    matched = true;
                                    zones.push(RefineZone {
                                        lo: sub(lo, [pad; 3]),
                                        hi: add(hi, [pad; 3]),
                                        target_level: lvl,
                                        kind: ZoneKind::Patch { bi, patch: lp, band: 1.8 * cell_target },
                                    });
                                }
                            }
                        }
                    }
                }
                if !matched {
                    bail!("mesh refinement selector '{sel}' matched no body or face");
                }
            }
        }

        // Automatic feature detection: curved holes/cylinders/spheres/tori, small STEP faces, thin walls
        if m.mesh.feature_refinement {
            for (bi, b) in bodies.iter().enumerate() {
                if b.role != Role::Solid {
                    continue;
                }
                let (blo, bhi) = b.shape.bounds();
                let sp = [bhi[0] - blo[0], bhi[1] - blo[1], bhi[2] - blo[2]];
                match &b.shape {
                    Shape::Cylinder { radius, .. } => {
                        let target = (radius / 3.0).max(h_min_floor);
                        let lvl = level_for_size(base_h, target, max_levels);
                        if lvl > 0 {
                            let pad = 2.0 * (base_h / (1usize << lvl) as f64);
                            zones.push(RefineZone {
                                lo: sub(blo, [pad; 3]),
                                hi: add(bhi, [pad; 3]),
                                target_level: lvl,
                                kind: ZoneKind::Patch { bi, patch: 0, band: 1.8 * (base_h / (1usize << lvl) as f64) },
                            });
                        }
                    }
                    Shape::Sphere { radius, .. } => {
                        let target = (radius / 3.0).max(h_min_floor);
                        let lvl = level_for_size(base_h, target, max_levels);
                        if lvl > 0 {
                            let pad = 2.0 * (base_h / (1usize << lvl) as f64);
                            zones.push(RefineZone {
                                lo: sub(blo, [pad; 3]),
                                hi: add(bhi, [pad; 3]),
                                target_level: lvl,
                                kind: ZoneKind::Patch { bi, patch: 0, band: 1.8 * (base_h / (1usize << lvl) as f64) },
                            });
                        }
                    }
                    Shape::Box { .. } => {
                        let min_dim = sp[0].min(sp[1]).min(sp[2]);
                        if min_dim > 1e-4 && min_dim < 1.5 * base_h {
                            let target = (min_dim / 2.0).max(h_min_floor);
                            let lvl = level_for_size(base_h, target, max_levels);
                            if lvl > 0 {
                                zones.push(RefineZone { lo: blo, hi: bhi, target_level: lvl, kind: ZoneKind::Body });
                            }
                        }
                    }
                    Shape::Step { data: Some(sd), .. } => {
                        for (pi, r, plo, phi) in sd.feature_patches() {
                            let target = (r / 3.0).max(h_min_floor);
                            let lvl = level_for_size(base_h, target, max_levels);
                            if lvl > 0 {
                                let csz = base_h / (1usize << lvl) as f64;
                                let pad = 2.0 * csz;
                                zones.push(RefineZone {
                                    lo: sub(plo, [pad; 3]),
                                    hi: add(phi, [pad; 3]),
                                    target_level: lvl,
                                    kind: ZoneKind::Patch { bi, patch: pi, band: 1.8 * csz },
                                });
                            }
                        }
                    }
                    Shape::Stl { data: Some(sd), .. } => {
                        for (pi, (plo, phi)) in sd.patch_bounds().into_iter().enumerate() {
                            let diag = norm(sub(phi, plo));
                            if diag > 1e-4 && diag < 4.0 * base_h {
                                let target = (diag / 4.0).max(h_min_floor);
                                let lvl = level_for_size(base_h, target, max_levels);
                                if lvl > 0 {
                                    let csz = base_h / (1usize << lvl) as f64;
                                    let pad = 2.0 * csz;
                                    zones.push(RefineZone {
                                        lo: sub(plo, [pad; 3]),
                                        hi: add(phi, [pad; 3]),
                                        target_level: lvl,
                                        kind: ZoneKind::Patch { bi, patch: pi, band: 1.8 * csz },
                                    });
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    if let Some(ah) = hint {
        for &(lo, hi, lvl) in &ah.regions {
            let l = lvl.min(max_levels);
            if l > 0 {
                zones.push(RefineZone { lo, hi, target_level: l, kind: ZoneKind::Box });
            }
        }
    }
    Ok(zones)
}

fn desired_level(zones: &[RefineZone], bodies: &[Body], clo: V3, chi: V3, h_cell: f64, max_levels: u8) -> u8 {
    let ctr = scale(add(clo, chi), 0.5);
    let half_diag = 0.87 * h_cell;
    let mut best = 0u8;
    for z in zones {
        if z.target_level <= best || !box_overlaps(clo, chi, z.lo, z.hi) {
            continue;
        }
        let hit = match z.kind {
            ZoneKind::Box | ZoneKind::Body => true,
            ZoneKind::Patch { bi, patch, band } => {
                if h_cell > 2.5 * band {
                    true
                } else {
                    bodies[bi]
                        .shape
                        .project(patch, ctr)
                        .is_some_and(|q| norm(sub(q, ctr)) <= half_diag + band && (0..3).all(|k| q[k] >= z.lo[k] - 1e-4 && q[k] <= z.hi[k] + 1e-4))
                }
            }
        };
        if hit {
            best = z.target_level.min(max_levels);
        }
    }
    best
}

#[inline]
fn pack_key(level: u8, ijk: [u32; 3]) -> u64 {
    ((level as u64) << 60) | ((ijk[0] as u64) << 40) | ((ijk[1] as u64) << 20) | (ijk[2] as u64)
}

#[derive(Clone, Debug)]
struct OctCell {
    level: u8,
    ijk: [u32; 3],
    alive: bool,
    occ: i32,
    region: u8,
    owner: u16,
    /// Bonded split group (0 for enclosure or default bonded group).
    cgroup: usize,
}

fn mesh_data_of(shape: &Shape) -> Option<&StlData> {
    match shape {
        Shape::Stl { data: Some(d), .. } => Some(d.as_ref()),
        Shape::Step { data: Some(d), .. } => Some(&d.mesh),
        _ => None,
    }
}

struct RayCache {
    h_tiny: f64,
    /// Cache keyed by body index -> quantized `(y, z)` -> 4 crossing vectors around tiny offsets
    rays: Vec<HashMap<(i64, i64), [Vec<f64>; 4]>>,
}

impl RayCache {
    fn new(nb: usize, h_tiny: f64) -> Self {
        Self { h_tiny, rays: vec![HashMap::new(); nb] }
    }

    fn point_inside_body(&mut self, bi: usize, body: &Body, p: V3) -> bool {
        if let Some(d) = mesh_data_of(&body.shape) {
            let (blo, bhi) = body.shape.bounds();
            let eps = self.h_tiny * 1e-3;
            if (0..3).any(|k| p[k] < blo[k] - eps || p[k] > bhi[k] + eps) {
                return false;
            }
            let ky = (p[1] * 1e6).round() as i64;
            let kz = (p[2] * 1e6).round() as i64;
            let a = self.h_tiny * 1.1e-4;
            let b = self.h_tiny * 0.7e-4;
            let cross4 = self.rays[bi].entry((ky, kz)).or_insert_with(|| {
                [(a, b), (-a, -b), (a, -b), (-a, b)].map(|(dy, dz)| d.x_crossings(p[1] + dy, p[2] + dz))
            });
            cross4.iter().any(|xs| xs.iter().filter(|&&q| q < p[0]).count() % 2 == 1)
        } else {
            body.shape.contains(p)
        }
    }

    fn point_in_conservative(&mut self, bi: usize, body: &Body, p: V3, strict_sub: bool, eps: f64) -> bool {
        if let Some(d) = mesh_data_of(&body.shape) {
            let (blo, bhi) = body.shape.bounds();
            if (0..3).any(|k| p[k] < blo[k] - eps || p[k] > bhi[k] + eps) {
                return false;
            }
            let ky = (p[1] * 1e6).round() as i64;
            let kz = (p[2] * 1e6).round() as i64;
            let a = self.h_tiny * 1.1e-4;
            let b = self.h_tiny * 0.7e-4;
            let cross4 = self.rays[bi].entry((ky, kz)).or_insert_with(|| {
                [(a, b), (-a, -b), (a, -b), (-a, b)].map(|(dy, dz)| d.x_crossings(p[1] + dy, p[2] + dz))
            });
            cross4.iter().any(|xs| {
                let odd = |t: f64| xs.iter().filter(|&&q| q < t).count() % 2 == 1;
                if strict_sub {
                    odd(p[0] - eps) && odd(p[0] + eps)
                } else {
                    odd(p[0] - eps) || odd(p[0] + eps)
                }
            })
        } else if strict_sub {
            (0..6).all(|q| {
                let mut t = p;
                t[q / 2] += if q % 2 == 0 { -eps } else { eps };
                body.shape.contains(t)
            })
        } else {
            body.shape.contains(p)
        }
    }
}

/// Build a 2:1-balanced octree mesh supporting FE solvers, boundary fitting, contacts, and ZZ adaptivity.
#[allow(clippy::too_many_arguments)]
pub fn build_octree_ex_model_cancellable(
    bodies: &[Body],
    h: f64,
    materials: &[Material],
    with_enclosure: bool,
    max_cells: usize,
    conservative: bool,
    model: Option<&Model>,
    hint: Option<&AdaptiveHint>,
    cancel: &AtomicBool,
) -> Result<Mesh> {
    let pair_key = |a: usize, b: usize| if a < b { (a, b) } else { (b, a) };
    let find_body = |name: &str| bodies.iter().position(|b| b.name.eq_ignore_ascii_case(name.trim()));
    let mut non_bonded: HashSet<(usize, usize)> = HashSet::new();
    let mut protected_bodies: HashSet<usize> = HashSet::new();
    let mut pending_bolts: Vec<PendingBolt> = Vec::new();
    if let Some(mdl) = model {
        let resolve_contact_body = |bname: &str, fnames: &[String]| -> Option<usize> {
            if !bname.trim().is_empty() {
                return find_body(bname);
            }
            fnames.first().and_then(|s| s.split('.').next()).and_then(find_body)
        };
        for c in &mdl.contacts {
            if !c.kind.splits_mesh() {
                continue;
            }
            if let (Some(ba), Some(bb)) = (resolve_contact_body(&c.body_a, &c.faces_a), resolve_contact_body(&c.body_b, &c.faces_b))
                && ba != bb {
                    non_bonded.insert(pair_key(ba, bb));
                    protected_bodies.insert(ba);
                    protected_bodies.insert(bb);
                }
        }
        for it in &mdl.loads {
            if let Some(bname) = it.load.body()
                && let Some(bi) = find_body(bname) {
                    protected_bodies.insert(bi);
                }
            for f in it.load.faces() {
                if let Some(prefix) = f.split('.').next()
                    && let Some(bi) = find_body(prefix) {
                        protected_bodies.insert(bi);
                    }
            }
            if let crate::model::Load::BoltPretension { body, axis, cut_mm, .. } = &it.load
                && let Some(bi) = find_body(body) {
                    let ax = axis.map(|a| a.idx()).unwrap_or_else(|| match &bodies[bi].shape {
                        Shape::Cylinder { axis, .. } => axis.idx(),
                        _ => {
                            let (blo, bhi) = bodies[bi].shape.bounds();
                            let d = [bhi[0] - blo[0], bhi[1] - blo[1], bhi[2] - blo[2]];
                            if d[0] >= d[1] && d[0] >= d[2] { 0 } else if d[1] >= d[2] { 1 } else { 2 }
                        }
                    });
                    let (blo, bhi) = bodies[bi].shape.bounds();
                    let cut_val = cut_mm.unwrap_or(0.5 * (blo[ax] + bhi[ax]));
                    pending_bolts.push(PendingBolt { load_id: it.id, bi, ax, cut_val });
                }
        }
    }

    let has_solid = bodies.iter().any(|b| b.op == Op::Add && b.role == Role::Solid);
    let has_encl = bodies.iter().any(|b| b.op == Op::Add && b.role == Role::Enclosure);
    let with_enclosure = with_enclosure && has_encl;
    if !has_solid && !with_enclosure {
        bail!("no geometry: add at least one body (box, cylinder, sphere, STEP or STL)");
    }
    for b in bodies {
        b.shape.validate()?;
        if matches!(&b.shape, Shape::Stl { data: None, .. } | Shape::Step { data: None, .. }) {
            bail!("CAD body '{}' is not loaded", b.name);
        }
    }
    if !h.is_finite() || h <= 0.0 {
        bail!("element size must be finite and positive");
    }
    let (lo, hi) = crate::mesh::domain_bounds(bodies, with_enclosure);
    let size = sub(hi, lo);
    if !lo.iter().chain(&hi).chain(&size).all(|v| v.is_finite()) {
        bail!("geometry bounds are not finite");
    }

    let mut max_levels = model.map(|m| m.mesh.max_levels).unwrap_or(3).clamp(1, 5);
    let mut dims0 = [0usize; 3];
    let grid_cap = max_cells.saturating_mul(4);
    for k in 0..3 {
        let count = ((size[k] / h) - 1e-6).ceil().max(1.0);
        if !count.is_finite() || count > grid_cap as f64 {
            bail!("element size {h} mm is too small for this geometry; use a larger element size");
        }
        dims0[k] = count as usize;
    }
    // Ensure fine integer coordinates fit safely in 20 bits (< 1_048_576).
    while max_levels > 1 && dims0.iter().any(|&d| (d << max_levels) >= (1 << 20)) {
        max_levels -= 1;
    }
    let origin = lo; // anchor min corner so symmetry/origin planes remain grid-aligned

    // Disjoint bonded contact groups
    let mut cgroup_of_body = vec![0usize; bodies.len()];
    let _n_cgroups = if non_bonded.is_empty() {
        1
    } else {
        let mut par: Vec<usize> = (0..bodies.len()).collect();
        let find = |par: &mut Vec<usize>, mut x: usize| -> usize {
            while par[x] != x {
                par[x] = par[par[x]];
                x = par[x];
            }
            x
        };
        for a in 0..bodies.len() {
            if bodies[a].role != Role::Solid || bodies[a].op != Op::Add {
                continue;
            }
            for b in (a + 1)..bodies.len() {
                if bodies[b].role != Role::Solid || bodies[b].op != Op::Add {
                    continue;
                }
                if !non_bonded.contains(&pair_key(a, b)) {
                    let (ra, rb) = (find(&mut par, a), find(&mut par, b));
                    if ra != rb {
                        par[rb] = ra;
                    }
                }
            }
        }
        let mut map = HashMap::new();
        for bi in 0..bodies.len() {
            if bodies[bi].role == Role::Solid && bodies[bi].op == Op::Add {
                let r = find(&mut par, bi);
                let nxt = map.len();
                cgroup_of_body[bi] = *map.entry(r).or_insert(nxt);
            }
        }
        map.len().max(1)
    };

    let zones = collect_refine_zones(bodies, h, max_levels, model, hint)?;

    // Ensure bolt cut regions are reasonably refined along the cut plane
    let bolt_zones: Vec<(usize, V3, V3, u8)> = pending_bolts
        .iter()
        .map(|pb| {
            let (mut blo, mut bhi) = bodies[pb.bi].shape.bounds();
            blo[pb.ax] = pb.cut_val - 0.5 * h;
            bhi[pb.ax] = pb.cut_val + 0.5 * h;
            (pb.bi, blo, bhi, 1u8.min(max_levels))
        })
        .collect();

    // Build initial octree covering additive domain bounding boxes
    let mut leaves: Vec<OctCell> = Vec::with_capacity(dims0[0] * dims0[1] * dims0[2] * 2);
    let mut leaf_map: HashMap<u64, usize> = HashMap::new();
    let add_boxes: Vec<(V3, V3)> = bodies
        .iter()
        .filter(|b| b.op == Op::Add && (b.role == Role::Solid || (with_enclosure && b.role == Role::Enclosure)))
        .map(|b| b.shape.bounds())
        .collect();

    let cell_bounds = |level: u8, ijk: [u32; 3]| -> (V3, V3, f64) {
        let hs = h / (1usize << level) as f64;
        let clo = [origin[0] + ijk[0] as f64 * hs, origin[1] + ijk[1] as f64 * hs, origin[2] + ijk[2] as f64 * hs];
        let chi = [clo[0] + hs, clo[1] + hs, clo[2] + hs];
        (clo, chi, hs)
    };

    let insert_leaf = |leaves: &mut Vec<OctCell>, map: &mut HashMap<u64, usize>, level: u8, ijk: [u32; 3]| -> usize {
        let idx = leaves.len();
        leaves.push(OctCell { level, ijk, alive: true, occ: -1, region: REGION_SOLID, owner: 0, cgroup: 0 });
        map.insert(pack_key(level, ijk), idx);
        idx
    };

    let split_leaf = |idx: usize, leaves: &mut Vec<OctCell>, map: &mut HashMap<u64, usize>| -> [usize; 8] {
        let (lvl, p) = (leaves[idx].level, leaves[idx].ijk);
        leaves[idx].alive = false;
        map.remove(&pack_key(lvl, p));
        let cl = lvl + 1;
        let mut ch = [0usize; 8];
        for (o, c) in CORNER.iter().enumerate() {
            let cijk = [p[0] * 2 + c[0] as u32, p[1] * 2 + c[1] as u32, p[2] * 2 + c[2] as u32];
            let ci = leaves.len();
            leaves.push(OctCell { level: cl, ijk: cijk, alive: true, occ: -1, region: REGION_SOLID, owner: 0, cgroup: 0 });
            map.insert(pack_key(cl, cijk), ci);
            ch[o] = ci;
        }
        ch
    };

    // Top-down recursive subdivision
    let mut stack: Vec<(u8, [u32; 3])> = Vec::with_capacity(dims0[0] * dims0[1] * dims0[2]);
    for k in 0..dims0[2] as u32 {
        for j in 0..dims0[1] as u32 {
            for i in 0..dims0[0] as u32 {
                stack.push((0, [i, j, k]));
            }
        }
    }
    let mut steps_done = 0usize;
    while let Some((lvl, ijk)) = stack.pop() {
        steps_done += 1;
        if steps_done & 2047 == 0 {
            check_cancel(cancel)?;
            if leaf_map.len() > grid_cap {
                bail!("octree mesh exceeded element budget ({grid_cap} cells); increase sizes or lower max_levels");
            }
        }
        let (clo, chi, hs) = cell_bounds(lvl, ijk);
        let overlaps_any = add_boxes.iter().any(|&(blo, bhi)| box_overlaps(clo, chi, blo, bhi));
        let mut req = if overlaps_any {
            desired_level(&zones, bodies, clo, chi, hs, max_levels)
        } else {
            0
        };
        if overlaps_any {
            for &(_, blo, bhi, blvl) in &bolt_zones {
                if blvl > req && box_overlaps(clo, chi, blo, bhi) {
                    req = blvl;
                }
            }
        }
        if lvl < req && lvl < max_levels {
            let cl = lvl + 1;
            for c in CORNER {
                stack.push((cl, [ijk[0] * 2 + c[0] as u32, ijk[1] * 2 + c[1] as u32, ijk[2] * 2 + c[2] as u32]));
            }
        } else {
            insert_leaf(&mut leaves, &mut leaf_map, lvl, ijk);
        }
    }

    // Helper: enforce 26-neighbor 2:1 balance across all active leaves
    let balance_2_1 = |leaves: &mut Vec<OctCell>, map: &mut HashMap<u64, usize>, cancel: &AtomicBool| -> Result<()> {
        for l in (2..=max_levels).rev() {
            check_cancel(cancel)?;
            let at_l: Vec<[u32; 3]> = leaves.iter().filter(|c| c.alive && c.level == l).map(|c| c.ijk).collect();
            for ijk in at_l {
                let p = [ijk[0] / 2, ijk[1] / 2, ijk[2] / 2];
                let lim = [dims0[0] << (l - 1), dims0[1] << (l - 1), dims0[2] << (l - 1)];
                for dz in -1isize..=1 {
                    let nz = p[2] as isize + dz;
                    if nz < 0 || nz as usize >= lim[2] {
                        continue;
                    }
                    for dy in -1isize..=1 {
                        let ny = p[1] as isize + dy;
                        if ny < 0 || ny as usize >= lim[1] {
                            continue;
                        }
                        for dx in -1isize..=1 {
                            if dx == 0 && dy == 0 && dz == 0 {
                                continue;
                            }
                            let nx = p[0] as isize + dx;
                            if nx < 0 || nx as usize >= lim[0] {
                                continue;
                            }
                            let target = [nx as u32, ny as u32, nz as u32];
                            loop {
                                let mut coarse: Option<(u8, [u32; 3], usize)> = None;
                                for c in 0..=(l - 2) {
                                    let sh = (l - 1 - c) as u32;
                                    let aijk = [target[0] >> sh, target[1] >> sh, target[2] >> sh];
                                    if let Some(&idx) = map.get(&pack_key(c, aijk)) {
                                        coarse = Some((c, aijk, idx));
                                        break;
                                    }
                                }
                                let Some((_, _, idx)) = coarse else { break };
                                split_leaf(idx, leaves, map);
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    };
    balance_2_1(&mut leaves, &mut leaf_map, cancel)?;

    // Evaluate CSG occupancy on one leaf
    let h_min = h / (1usize << max_levels) as f64;
    let mut rcache = RayCache::new(bodies.len(), h_min);
    let mat_index = |name: &str| -> i32 { materials.iter().position(|m| m.name == name).unwrap_or(0) as i32 };
    let mut body_mats: Vec<i32> = bodies.iter().map(|b| mat_index(&b.material)).collect();

    // Cache conservative corner masks `(solid_group_bitmask, sub_hit)` keyed by finest integer coords
    let fin_sh = |lvl: u8| (max_levels - lvl) as u32;
    let mut corner_cache: HashMap<u64, u64> = HashMap::new(); // bit i in 0..n_cgroups = corner inside solid group i

    let eval_cell = |leaf: &mut OctCell, rcache: &mut RayCache, corner_cache: &mut HashMap<u64, u64>, use_cons: bool| {
        let (clo, chi, hs) = cell_bounds(leaf.level, leaf.ijk);
        let ctr = scale(add(clo, chi), 0.5);
        leaf.occ = -1;
        leaf.region = REGION_SOLID;
        leaf.owner = 0;
        leaf.cgroup = 0;
        let mut encl_occ = -1i32;
        let mut encl_own = 0u16;
        if with_enclosure {
            for (bi, b) in bodies.iter().enumerate().filter(|(_, b)| b.role == Role::Enclosure && b.op == Op::Add) {
                if rcache.point_inside_body(bi, b, ctr) {
                    leaf.occ = body_mats[bi];
                    leaf.region = REGION_ENCLOSURE;
                    leaf.owner = bi as u16;
                    encl_occ = leaf.occ;
                    encl_own = leaf.owner;
                }
            }
        }
        for (bi, b) in bodies.iter().enumerate().filter(|(_, b)| b.role == Role::Solid) {
            if !rcache.point_inside_body(bi, b, ctr) {
                continue;
            }
            match b.op {
                Op::Add => {
                    leaf.occ = body_mats[bi];
                    leaf.region = REGION_SOLID;
                    leaf.owner = bi as u16;
                    leaf.cgroup = cgroup_of_body[bi];
                }
                Op::Subtract => {
                    if leaf.region == REGION_SOLID && leaf.occ >= 0 {
                        leaf.occ = encl_occ;
                        leaf.region = REGION_ENCLOSURE;
                        leaf.owner = encl_own;
                        leaf.cgroup = 0;
                    }
                }
            }
        }
        if use_cons && conservative && leaf.occ >= 0 && leaf.region == REGION_SOLID {
            let s = 1u32 << fin_sh(leaf.level);
            let eps = 1e-6 * h_min;
            let cg = leaf.cgroup;
            let all_in = CORNER.iter().all(|c| {
                let lat = [(leaf.ijk[0] + c[0] as u32) * s, (leaf.ijk[1] + c[1] as u32) * s, (leaf.ijk[2] + c[2] as u32) * s];
                let ckey = ((lat[0] as u64) << 40) | ((lat[1] as u64) << 20) | (lat[2] as u64);
                let mask = *corner_cache.entry(ckey).or_insert_with(|| {
                    let p = [origin[0] + lat[0] as f64 * h_min, origin[1] + lat[1] as f64 * h_min, origin[2] + lat[2] as f64 * h_min];
                    let mut m = 0u64;
                    for (bi, body) in bodies.iter().enumerate() {
                        if body.role != Role::Solid {
                            continue;
                        }
                        if body.op == Op::Add {
                            let g = cgroup_of_body[bi];
                            if g < 64 && (m & (1 << g)) == 0 && rcache.point_in_conservative(bi, body, p, false, eps) {
                                m |= 1 << g;
                            }
                        } else if body.op == Op::Subtract && m != 0 && rcache.point_in_conservative(bi, body, p, true, eps) {
                            m = 0;
                        }
                    }
                    m
                });
                cg < 64 && (mask & (1 << cg)) != 0
            });
            let _ = hs;
            if !all_in {
                leaf.occ = encl_occ;
                if encl_occ >= 0 {
                    leaf.region = REGION_ENCLOSURE;
                    leaf.owner = encl_own;
                }
                leaf.cgroup = 0;
            }
        }
    };

    for idx in 0..leaves.len() {
        if leaves[idx].alive {
            let mut copy = leaves[idx].clone();
            eval_cell(&mut copy, &mut rcache, &mut corner_cache, true);
            leaves[idx] = copy;
        }
    }

    // Protect bodies whose conservative drop removed all cells
    let mut warnings = Vec::new();
    if conservative {
        let mut has = vec![0usize; bodies.len()];
        let mut had_center = vec![0usize; bodies.len()];
        for l in &leaves {
            if l.alive && l.occ >= 0 && l.region == REGION_SOLID {
                has[l.owner as usize] += 1;
            }
        }
        let mut need_restore = vec![false; bodies.len()];
        for (bi, b) in bodies.iter().enumerate() {
            if b.role == Role::Solid && b.op == Op::Add && has[bi] == 0 {
                // Re-check centroid occupancy on existing leaves for this vanished body
                for l in &leaves {
                    if !l.alive {
                        continue;
                    }
                    let (clo, chi, _) = cell_bounds(l.level, l.ijk);
                    let ctr = scale(add(clo, chi), 0.5);
                    if rcache.point_inside_body(bi, b, ctr) {
                        had_center[bi] += 1;
                    }
                }
                if had_center[bi] > 0 {
                    need_restore[bi] = true;
                    warnings.push(format!("{} thinner than a local cell: kept centroid cells unfitted", b.name));
                }
            }
        }
        if need_restore.iter().any(|&v| v) {
            for idx in 0..leaves.len() {
                if !leaves[idx].alive {
                    continue;
                }
                let mut copy = leaves[idx].clone();
                eval_cell(&mut copy, &mut rcache, &mut corner_cache, false);
                if copy.occ >= 0 && copy.region == REGION_SOLID && need_restore[copy.owner as usize] {
                    leaves[idx] = copy;
                }
            }
        }
    }

    // Eliminate partial boundary splits across L -> L+1 transitions so all exposed BFace quads are watertight
    let neighbor_lookup = |lvl: u8, ijk: [isize; 3], map: &HashMap<u64, usize>, leaves: &[OctCell]| -> Option<usize> {
        let lim = [dims0[0] << lvl, dims0[1] << lvl, dims0[2] << lvl];
        if (0..3).any(|a| ijk[a] < 0 || ijk[a] as usize >= lim[a]) {
            return None;
        }
        let u = [ijk[0] as u32, ijk[1] as u32, ijk[2] as u32];
        for c in (0..=lvl).rev() {
            let sh = (lvl - c) as u32;
            if let Some(&id) = map.get(&pack_key(c, [u[0] >> sh, u[1] >> sh, u[2] >> sh]))
                && leaves[id].alive {
                    return Some(id);
                }
        }
        None
    };

    for _pass in 0..=max_levels {
        check_cancel(cancel)?;
        let mut to_split = Vec::new();
        for (idx, lf) in leaves.iter().enumerate() {
            if !lf.alive || lf.occ < 0 || lf.level >= max_levels {
                continue;
            }
            let cl = lf.level + 1;
            let [ix, iy, iz] = lf.ijk;
            let mut split_me = false;
            for dir in 0..6u8 {
                let a = (dir / 2) as usize;
                let plus = dir % 2 == 1;
                let (b, cax) = ((a + 1) % 3, (a + 2) % 3);
                let mut same_count = 0usize;
                let mut any_finer = false;
                for ob in 0..2u32 {
                    for oc in 0..2u32 {
                        let mut n = [ix as isize * 2, iy as isize * 2, iz as isize * 2];
                        n[a] += if plus { 2 } else { -1 };
                        n[b] += ob as isize;
                        n[cax] += oc as isize;
                        if let Some(nb_id) = neighbor_lookup(cl, n, &leaf_map, &leaves) {
                            let nb = &leaves[nb_id];
                            if nb.level > lf.level {
                                any_finer = true;
                            }
                            let match_solid = lf.region == REGION_SOLID && nb.occ >= 0 && nb.region == REGION_SOLID && nb.cgroup == lf.cgroup;
                            let match_encl = lf.region == REGION_ENCLOSURE && nb.occ >= 0;
                            if match_solid || match_encl {
                                same_count += 1;
                            }
                        }
                    }
                }
                if any_finer && same_count > 0 && same_count < 4 {
                    split_me = true;
                    break;
                }
            }
            if split_me {
                to_split.push(idx);
            }
        }
        if to_split.is_empty() {
            break;
        }
        for idx in to_split {
            if !leaves[idx].alive {
                continue;
            }
            let children = split_leaf(idx, &mut leaves, &mut leaf_map);
            for ch in children {
                let mut copy = leaves[ch].clone();
                eval_cell(&mut copy, &mut rcache, &mut corner_cache, true);
                leaves[ch] = copy;
            }
        }
        let prev_len = leaves.len();
        balance_2_1(&mut leaves, &mut leaf_map, cancel)?;
        for idx in prev_len..leaves.len() {
            if leaves[idx].alive {
                let mut copy = leaves[idx].clone();
                eval_cell(&mut copy, &mut rcache, &mut corner_cache, true);
                leaves[idx] = copy;
            }
        }
    }

    // Connected components over face-adjacent active cells (handling same, coarser, and 4 finer neighbors)
    let mut comp = vec![u32::MAX; leaves.len()];
    let mut comp_sizes = Vec::new();
    let mut comp_prot = Vec::new();
    for start in 0..leaves.len() {
        if !leaves[start].alive || leaves[start].occ < 0 || comp[start] != u32::MAX {
            continue;
        }
        let id = comp_sizes.len() as u32;
        let mut cnt = 0usize;
        let mut prot = false;
        let mut q = vec![start];
        comp[start] = id;
        while let Some(u) = q.pop() {
            let lf = &leaves[u];
            cnt += 1;
            if lf.region == REGION_SOLID && protected_bodies.contains(&(lf.owner as usize)) {
                prot = true;
            }
            for dir in 0..6u8 {
                let a = (dir / 2) as usize;
                let plus = dir % 2 == 1;
                let (b, cax) = ((a + 1) % 3, (a + 2) % 3);
                // Probe at finest sub-level if finer children might exist, or single probe if at max_levels
                if lf.level < max_levels {
                    let cl = lf.level + 1;
                    let [ix, iy, iz] = lf.ijk;
                    for ob in 0..2isize {
                        for oc in 0..2isize {
                            let mut n = [ix as isize * 2, iy as isize * 2, iz as isize * 2];
                            n[a] += if plus { 2 } else { -1 };
                            n[b] += ob;
                            n[cax] += oc;
                            if let Some(nb) = neighbor_lookup(cl, n, &leaf_map, &leaves)
                                && leaves[nb].occ >= 0 && comp[nb] == u32::MAX {
                                    comp[nb] = id;
                                    q.push(nb);
                                }
                        }
                    }
                } else {
                    let mut n = [lf.ijk[0] as isize, lf.ijk[1] as isize, lf.ijk[2] as isize];
                    n[a] += if plus { 1 } else { -1 };
                    if let Some(nb) = neighbor_lookup(lf.level, n, &leaf_map, &leaves)
                        && leaves[nb].occ >= 0 && comp[nb] == u32::MAX {
                            comp[nb] = id;
                            q.push(nb);
                        }
                }
            }
        }
        comp_sizes.push(cnt);
        comp_prot.push(prot);
    }
    if comp_sizes.is_empty() {
        bail!("geometry produced an empty octree mesh; check element sizes and bodies");
    }
    let largest = *comp_sizes.iter().max().unwrap();
    let keep: Vec<bool> = comp_sizes.iter().zip(&comp_prot).map(|(&s, &p)| (s * 50 >= largest && s >= 2) || p).collect();
    let dropped: usize = comp_sizes.iter().zip(&keep).filter(|(_, k)| !**k).map(|(s, _)| s).sum();
    if dropped > 0 {
        warnings.push(format!("removed {dropped} disconnected fragment cell(s)"));
    }
    for idx in 0..leaves.len() {
        if leaves[idx].alive && leaves[idx].occ >= 0 && !keep[comp[idx] as usize] {
            leaves[idx].occ = -1;
        }
    }

    // Collect final occupied active leaves in deterministic order
    let active_ids: Vec<usize> = (0..leaves.len()).filter(|&i| leaves[i].alive && leaves[i].occ >= 0).collect();
    if active_ids.len() > max_cells {
        bail!("octree mesh has {} elements (limit {}); increase sizes or lower max_levels", active_ids.len(), max_cells);
    }
    let mut leaf_to_elem = vec![u32::MAX; leaves.len()];
    for (ei, &lid) in active_ids.iter().enumerate() {
        leaf_to_elem[lid] = ei as u32;
    }

    // Bolt pretension cut coordinates snapped to finest integer plane inside each bolt body
    struct BoltCutOct {
        load_id: u32,
        bi: usize,
        ax: usize,
        k_cut_lat: u32,
        cut_mm: f64,
    }
    let mut bolt_cuts = Vec::new();
    for pb in &pending_bolts {
        let mut lat_lo = u32::MAX;
        let mut lat_hi = 0u32;
        for &lid in &active_ids {
            let lf = &leaves[lid];
            if lf.region == REGION_SOLID && lf.owner as usize == pb.bi {
                let s = 1u32 << fin_sh(lf.level);
                lat_lo = lat_lo.min(lf.ijk[pb.ax] * s);
                lat_hi = lat_hi.max((lf.ijk[pb.ax] + 1) * s);
            }
        }
        if lat_lo == u32::MAX || lat_hi <= lat_lo {
            bail!("bolt pretension #{}: bolt body '{}' has no solid elements", pb.load_id, bodies[pb.bi].name);
        }
        let target = ((pb.cut_val - origin[pb.ax]) / h_min).round().clamp(lat_lo as f64 + 1.0, lat_hi as f64 - 1.0) as u32;
        bolt_cuts.push(BoltCutOct {
            load_id: pb.load_id,
            bi: pb.bi,
            ax: pb.ax,
            k_cut_lat: target,
            cut_mm: origin[pb.ax] + target as f64 * h_min,
        });
    }

    // Vertex deduplication & non-bonded contact / bolt-cut cluster splitting, keyed on (cgroup, [x_lat, y_lat, z_lat])
    let mut nodes: Vec<V3> = Vec::new();
    let mut node_lat: Vec<[u32; 3]> = Vec::new();
    // Map: (lat_x, lat_y, lat_z) -> list of (split_signature, node_id) where split_signature is (cgroup, bolt_mask)
    let mut split_clusters: HashMap<[u32; 3], Vec<(usize, u32, u32)>> = HashMap::new();
    // Also map: (cgroup, lat_x, lat_y, lat_z) -> node_id for interior/non-bolt hanging detection
    let mut group_node_map: HashMap<(usize, [u32; 3]), u32> = HashMap::new();

    let mut elems: Vec<[u32; 8]> = Vec::with_capacity(active_ids.len());
    let mut elem_mat: Vec<u16> = Vec::with_capacity(active_ids.len());
    let mut elem_region: Vec<u8> = Vec::with_capacity(active_ids.len());
    let mut elem_body: Vec<u16> = Vec::with_capacity(active_ids.len());
    let mut elem_cell: Vec<[u32; 3]> = Vec::with_capacity(active_ids.len());
    let mut elem_level: Vec<u8> = Vec::with_capacity(active_ids.len());
    let mut elem_size_mm: Vec<f64> = Vec::with_capacity(active_ids.len());

    let _ = &mut body_mats;
    for &lid in &active_ids {
        let lf = &leaves[lid];
        let s = 1u32 << fin_sh(lf.level);
        let hs = h / (1usize << lf.level) as f64;
        let cg = if lf.region == REGION_SOLID { lf.cgroup } else { 0 };
        let ob = lf.owner as usize;
        let mut en = [0u32; 8];
        for (l, c) in CORNER.iter().enumerate() {
            let plat = [(lf.ijk[0] + c[0] as u32) * s, (lf.ijk[1] + c[1] as u32) * s, (lf.ijk[2] + c[2] as u32) * s];
            let mut bsig = 0u32;
            for (bi_cut, bc) in bolt_cuts.iter().enumerate() {
                if lf.region == REGION_SOLID && ob == bc.bi && plat[bc.ax] == bc.k_cut_lat {
                    let plus_side = lf.ijk[bc.ax] * s >= bc.k_cut_lat;
                    if plus_side {
                        bsig |= 1u32 << (bi_cut.min(31));
                    }
                }
            }
            let clusters = split_clusters.entry(plat).or_default();
            let nid = if let Some(cl) = clusters.iter().find(|&&(sc, sb, _)| sc == cg && sb == bsig) {
                cl.2
            } else {
                let id = nodes.len() as u32;
                nodes.push([origin[0] + plat[0] as f64 * h_min, origin[1] + plat[1] as f64 * h_min, origin[2] + plat[2] as f64 * h_min]);
                node_lat.push(plat);
                clusters.push((cg, bsig, id));
                if bsig == 0 {
                    group_node_map.insert((cg, plat), id);
                }
                id
            };
            en[l] = nid;
        }
        elems.push(en);
        elem_mat.push(lf.occ as u16);
        elem_region.push(lf.region);
        elem_body.push(lf.owner);
        elem_cell.push([lf.ijk[0] * s, lf.ijk[1] * s, lf.ijk[2] * s]);
        elem_level.push(lf.level);
        elem_size_mm.push(hs);
    }

    // Detect hanging nodes on coarse element edges and faces (integer midpoints in same bonded group)
    const EDGES: [(usize, usize); 12] = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)];
    const FACES: [[usize; 4]; 6] = [[0, 3, 7, 4], [1, 2, 6, 5], [0, 1, 5, 4], [3, 2, 6, 7], [0, 1, 2, 3], [4, 5, 6, 7]];
    let mut raw_hang: HashMap<u32, Vec<(u32, f64)>> = HashMap::new();
    for (ei, &lid) in active_ids.iter().enumerate() {
        let lf = &leaves[lid];
        let s = 1u32 << fin_sh(lf.level);
        if s < 2 {
            continue;
        }
        let cg = if lf.region == REGION_SOLID { lf.cgroup } else { 0 };
        let en = elems[ei];
        for &(a, b) in &EDGES {
            let pa = node_lat[en[a] as usize];
            let pb = node_lat[en[b] as usize];
            let mid = [(pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2, (pa[2] + pb[2]) / 2];
            if let Some(&hn) = group_node_map.get(&(cg, mid))
                && hn != en[a] && hn != en[b] {
                    raw_hang.entry(hn).or_insert_with(|| vec![(en[a], 0.5), (en[b], 0.5)]);
                }
        }
        for q in FACES {
            let p0 = node_lat[en[q[0]] as usize];
            let p1 = node_lat[en[q[1]] as usize];
            let p2 = node_lat[en[q[2]] as usize];
            let p3 = node_lat[en[q[3]] as usize];
            let ctr = [
                (p0[0] + p1[0] + p2[0] + p3[0]) / 4,
                (p0[1] + p1[1] + p2[1] + p3[1]) / 4,
                (p0[2] + p1[2] + p2[2] + p3[2]) / 4,
            ];
            if let Some(&hn) = group_node_map.get(&(cg, ctr))
                && !q.iter().any(|&c| en[c] == hn) {
                    raw_hang.entry(hn).or_insert_with(|| q.iter().map(|&c| (en[c], 0.25)).collect());
                }
        }
    }

    // Flatten any transitive hanging chains strictly to master nodes (guaranteed acyclic by level ordering)
    let mut hanging: Vec<HangingNode> = Vec::with_capacity(raw_hang.len());
    let mut h_keys: Vec<u32> = raw_hang.keys().copied().collect();
    h_keys.sort_unstable();
    for hn in h_keys {
        let mut cur = raw_hang[&hn].clone();
        for _ in 0..8 {
            let mut changed = false;
            let mut next: Vec<(u32, f64)> = Vec::new();
            for &(p, w) in &cur {
                if let Some(gp) = raw_hang.get(&p) {
                    changed = true;
                    for &(gpp, gw) in gp {
                        if let Some(ex) = next.iter_mut().find(|x| x.0 == gpp) {
                            ex.1 += w * gw;
                        } else {
                            next.push((gpp, w * gw));
                        }
                    }
                } else if let Some(ex) = next.iter_mut().find(|x| x.0 == p) {
                    ex.1 += w;
                } else {
                    next.push((p, w));
                }
            }
            cur = next;
            if !changed {
                break;
            }
        }
        cur.retain(|x| x.1.abs() > 1e-12);
        hanging.push(HangingNode { node: hn, parents: cur });
    }

    // Bolt pretension cut pairs on octree faces crossing `k_cut_lat`
    let mut pretensions = Vec::with_capacity(bolt_cuts.len());
    for (bi_cut, bc) in bolt_cuts.iter().enumerate() {
        let mut pmap: HashMap<(u32, u32), f64> = HashMap::new();
        let (ax, b_ax, c_ax) = (bc.ax, (bc.ax + 1) % 3, (bc.ax + 2) % 3);
        let bit = 1u32 << bi_cut.min(31);
        for (ei, &lid) in active_ids.iter().enumerate() {
            let lf = &leaves[lid];
            let s = 1u32 << fin_sh(lf.level);
            if lf.region != REGION_SOLID || lf.owner as usize != bc.bi || (lf.ijk[ax] + 1) * s != bc.k_cut_lat {
                continue;
            }
            // Element sits immediately on the minus side of `k_cut_lat`
            let hs_m = elem_size_mm[ei] * 1e-3;
            let share = 0.25 * hs_m * hs_m;
            for (ob, oc) in [(0usize, 0usize), (1, 0), (1, 1), (0, 1)] {
                let mut dm = [0usize; 3];
                dm[ax] = 1;
                dm[b_ax] = ob;
                dm[c_ax] = oc;
                let lm = CORNER.iter().position(|&p| p == dm).unwrap();
                let nm = elems[ei][lm];
                let plat = node_lat[nm as usize];
                if let Some(cl) = split_clusters.get(&plat)
                    && let Some(&(_, _, np)) = cl.iter().find(|&&(sc, sb, _)| sc == lf.cgroup && (sb & bit) != 0) {
                        *pmap.entry((nm, np)).or_insert(0.0) += share;
                    }
            }
        }
        let mut pairs: Vec<crate::mesh::PretensionPair> = pmap.into_iter().map(|((minus, plus), area_m2)| crate::mesh::PretensionPair { minus, plus, area_m2 }).collect();
        pairs.sort_by_key(|p| (p.minus, p.plus));
        let mut normal = [0.0; 3];
        normal[ax] = 1.0;
        pretensions.push(crate::mesh::PretensionSection { load_id: bc.load_id, body_idx: bc.bi, normal, cut_mm: bc.cut_mm, pairs });
    }

    // Boundary faces across exposed solid and enclosure faces
    let mut faces = Vec::new();
    let mut split_face = Vec::new();
    let order = [(0usize, 0usize), (1, 0), (1, 1), (0, 1)];
    for (el, &lid) in active_ids.iter().enumerate() {
        let lf = &leaves[lid];
        let en = elems[el];
        let hs = elem_size_mm[el];
        let blo = [
            origin[0] + (lf.ijk[0] as f64) * hs,
            origin[1] + (lf.ijk[1] as f64) * hs,
            origin[2] + (lf.ijk[2] as f64) * hs,
        ];
        for dir in 0..6u8 {
            let a = (dir / 2) as usize;
            let plus = dir % 2 == 1;
            let mut n = [lf.ijk[0] as isize, lf.ijk[1] as isize, lf.ijk[2] as isize];
            n[a] += if plus { 1 } else { -1 };
            // Because partial_boundary_split removed mixed L+1 covers, querying one sub-quadrant or same-level
            // tells us the complete coverage status across face `dir`.
            let ncell = if lf.level < max_levels {
                let mut probe = [lf.ijk[0] as isize * 2, lf.ijk[1] as isize * 2, lf.ijk[2] as isize * 2];
                probe[a] += if plus { 2 } else { -1 };
                neighbor_lookup(lf.level + 1, probe, &leaf_map, &leaves).filter(|&id| leaves[id].occ >= 0)
            } else {
                neighbor_lookup(lf.level, n, &leaf_map, &leaves).filter(|&id| leaves[id].occ >= 0)
            };
            let n_active = ncell.is_some();
            let n_solid = ncell.is_some_and(|q| leaves[q].region == REGION_SOLID);
            let is_split = n_solid && ncell.is_some_and(|nc| non_bonded.contains(&pair_key(lf.owner as usize, leaves[nc].owner as usize)));
            let outer = match lf.region {
                REGION_SOLID if !n_solid || is_split => false,
                REGION_ENCLOSURE if !n_active => true,
                _ => continue,
            };
            let (b, cax) = ((a + 1) % 3, (a + 2) % 3);
            let mut quad = [0u32; 4];
            for (q, (ob, oc)) in order.iter().enumerate() {
                let mut d = [0usize; 3];
                d[a] = plus as usize;
                d[b] = *ob;
                d[cax] = *oc;
                let cl = CORNER.iter().position(|&p| p == d).unwrap();
                quad[q] = en[cl];
            }
            if !plus {
                quad.swap(1, 3);
            }
            let mut ctr = [blo[0] + 0.5 * hs, blo[1] + 0.5 * hs, blo[2] + 0.5 * hs];
            ctr[a] = blo[a] + if plus { hs } else { 0.0 };
            faces.push(BFace { elem: el as u32, nodes: quad, dir, patch: 0, center: ctr, outer });
            split_face.push(is_split);
        }
    }

    // Patch assignment (weighted by true quad area hs^2)
    let mut offsets = Vec::new();
    let mut names = Vec::new();
    for b in bodies {
        offsets.push(names.len());
        for p in b.shape.patch_names() {
            names.push(format!("{}.{}", b.name, p));
        }
    }
    let assign: Vec<u32> = faces
        .par_iter()
        .enumerate()
        .map(|(fi, f)| -> Result<u32> {
            if fi & 1023 == 0 {
                check_cancel(cancel)?;
            }
            let n = dir_normal(f.dir);
            let hf = elem_size_mm[f.elem as usize];
            let want = if f.outer { Role::Enclosure } else { Role::Solid };
            let owner_bi = elem_body[f.elem as usize] as usize;
            let is_split = split_face[fi];
            let mut best = (0usize, f64::INFINITY);
            for (bi, b) in bodies.iter().enumerate() {
                if b.role != want {
                    continue;
                }
                if !f.outer {
                    if is_split && bi != owner_bi && b.op != Op::Subtract {
                        continue;
                    }
                    if !is_split && non_bonded.contains(&pair_key(bi, owner_bi)) {
                        continue;
                    }
                }
                let (pi, s) = b.shape.best_patch(f.center, n, hf, b.op == Op::Subtract);
                if s < best.1 {
                    best = (offsets[bi] + pi, s);
                }
            }
            Ok(best.0 as u32)
        })
        .collect::<Result<Vec<u32>>>()?;

    let mut remap = vec![u32::MAX; names.len().max(1)];
    let mut patches: Vec<PatchInfo> = Vec::new();
    let mut patch_src: Vec<(u32, u32)> = Vec::new();
    for (f, &g) in faces.iter_mut().zip(&assign) {
        let g = g as usize;
        if remap[g] == u32::MAX {
            remap[g] = patches.len() as u32;
            let bi = offsets.iter().rposition(|&o| o <= g).unwrap_or(0);
            patch_src.push((bi as u32, (g - offsets[bi]) as u32));
            patches.push(PatchInfo { name: names.get(g).cloned().unwrap_or_else(|| "surface".into()), area_mm2: 0.0, centroid: [0.0; 3], normal: [0.0; 3], n_faces: 0 });
        }
        f.patch = remap[g];
        let hf = elem_size_mm[f.elem as usize];
        let area = hf * hf;
        let p = &mut patches[f.patch as usize];
        p.n_faces += 1;
        p.area_mm2 += area;
        p.centroid = add(p.centroid, scale(f.center, area));
        p.normal = add(p.normal, scale(dir_normal(f.dir), area));
    }
    for p in patches.iter_mut() {
        if p.area_mm2 > 0.0 {
            p.centroid = scale(p.centroid, 1.0 / p.area_mm2);
        }
        p.normal = normalize(p.normal);
        for v in p.centroid.iter_mut().chain(p.normal.iter_mut()) {
            *v = (*v * 1000.0).round() / 1000.0;
        }
    }

    let mut mesh = Mesh {
        h,
        origin,
        dims: [dims0[0] << max_levels, dims0[1] << max_levels, dims0[2] << max_levels],
        nodes,
        elems,
        elem_mat,
        elem_region,
        elem_body,
        elem_cell,
        materials: materials.to_vec(),
        faces,
        patches,
        warnings,
        bbox: (lo, hi),
        body_names: bodies.iter().map(|b| b.name.clone()).collect(),
        with_enclosure,
        patch_src,
        fitted: Vec::new(),
        contacts: Vec::new(),
        pretensions,
        octree: true,
        max_level: max_levels,
        elem_level,
        elem_size_mm,
        hanging,
        node_lat,
    };
    if !conservative
        && let Some(mdl) = model {
            mesh.build_contact_pairs(mdl)?;
        }
    Ok(mesh)
}

impl Mesh {
    /// Hanging-node constraints with every parent resolved to an independent (master) node.
    fn resolved_hanging(&self) -> HashMap<u32, Vec<(u32, f64)>> {
        let raw: HashMap<u32, &[(u32, f64)]> = self.hanging.iter().map(|h| (h.node, h.parents.as_slice())).collect();
        let mut out = HashMap::with_capacity(raw.len());
        for (&n, pars) in &raw {
            let mut terms: Vec<(u32, f64)> = pars.to_vec();
            // chains (a parent that is itself hanging) resolve in at most a few rounds
            for _ in 0..8 {
                if !terms.iter().any(|(p, _)| raw.contains_key(p)) {
                    break;
                }
                let mut next: Vec<(u32, f64)> = Vec::with_capacity(terms.len() * 2);
                for &(p, w) in &terms {
                    match raw.get(&p) {
                        Some(pp) => next.extend(pp.iter().map(|&(q, v)| (q, v * w))),
                        None => next.push((p, w)),
                    }
                }
                next.sort_unstable_by_key(|t| t.0);
                next.dedup_by(|b, a| {
                    if a.0 == b.0 {
                        a.1 += b.1;
                        true
                    } else {
                        false
                    }
                });
                terms = next;
            }
            terms.retain(|t| t.1 != 0.0);
            out.insert(n, terms);
        }
        out
    }

    /// Resolved hanging-node constraints to attach to assembled matrices (`k.hang = mesh.hang_ctx()`):
    /// the linear [`crate::sparse::Solver`] then solves `CᵀKC q = Cᵀf` and returns `u = C q`.
    pub fn hang_ctx(&self) -> Option<std::sync::Arc<crate::sparse::Hanging>> {
        if self.hanging.is_empty() {
            return None;
        }
        let mut nodes: Vec<(u32, Vec<(u32, f64)>)> = self.resolved_hanging().into_iter().collect();
        nodes.sort_unstable_by_key(|t| t.0);
        Some(std::sync::Arc::new(crate::sparse::Hanging::new(nodes)))
    }

    /// Condense a right-hand side (or a lumped diagonal) in place: `f_parent += w·f_hanging`,
    /// `f_hanging = 0`. Totals are preserved; idempotent.
    pub fn condense_rhs(&self, f: &mut [f64], bs: usize) {
        if let Some(h) = self.hang_ctx() {
            h.condense_rhs(f, bs);
        }
    }

    /// Hanging dofs are never prescribed directly: they follow their parents.
    pub fn release_hanging(&self, fixed: &mut [Option<f64>], bs: usize) {
        for hn in &self.hanging {
            for c in 0..bs {
                if let Some(f) = fixed.get_mut(hn.node as usize * bs + c) {
                    *f = None;
                }
            }
        }
    }

    /// Set every hanging dof to the interpolation of its parents (`u = C u_master`).
    pub fn expand_solution(&self, u: &mut [f64], bs: usize) {
        if let Some(h) = self.hang_ctx() {
            h.expand(u, bs);
        }
    }

    /// Whether a node is a hanging (constrained) node.
    pub fn hanging_set(&self) -> HashSet<u32> {
        self.hanging.iter().map(|h| h.node).collect()
    }
}

/// Zienkiewicz–Zhu error estimate of a solved static structural or steady thermal model.
pub struct ZzEstimate {
    /// Relative energy-norm error ‖e‖ / √(‖u‖² + ‖e‖²) [%].
    pub error_pct: f64,
    /// Squared element error contributions (same units as the energy norm).
    pub elem_err2: Vec<f64>,
}

/// ZZ estimate: element-centre (superconvergent) stresses or heat fluxes are averaged to the
/// nodes (volume-weighted), and each element's error is the energy-norm distance between that
/// recovered field and its own constant centre value.
pub fn zz_estimate(mesh: &Mesh, res: &crate::results::Results) -> Option<ZzEstimate> {
    let ne = mesh.elems.len();
    let cg = crate::fem::center_grads();
    // centre gradients of the 8 shape functions [1/mm]
    let grads = |e: usize| -> Vec<[f64; 3]> {
        if mesh.is_fitted(e) {
            crate::fem::center_grads_general(&mesh.elem_nodes(e))
        } else {
            let h = mesh.elem_h_mm(e);
            cg.iter().map(|g| g.map(|v| v / h)).collect()
        }
    };
    let solid = |e: usize| mesh.elem_region[e] == REGION_SOLID;
    // per element: (field vector, energy weight 1/modulus)
    let (ncomp, vals): (usize, Vec<(Vec<f64>, f64)>) = if let Some(u) = &res.disp {
        let v = (0..ne)
            .into_par_iter()
            .map(|e| {
                let m = &mesh.materials[mesh.elem_mat[e] as usize];
                let g = grads(e);
                let mut du = [[0.0; 3]; 3]; // du[i][j] = ∂u_i/∂x_j
                for (a, &n) in mesh.elems[e].iter().enumerate() {
                    for i in 0..3 {
                        for j in 0..3 {
                            du[i][j] += u[n as usize][i] * g[a][j];
                        }
                    }
                }
                let (ee, nu) = (m.youngs_modulus_gpa * 1e3, m.poisson_ratio);
                let lam = ee * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
                let mu = ee / (2.0 * (1.0 + nu));
                let tr = du[0][0] + du[1][1] + du[2][2];
                let s = vec![
                    lam * tr + 2.0 * mu * du[0][0],
                    lam * tr + 2.0 * mu * du[1][1],
                    lam * tr + 2.0 * mu * du[2][2],
                    mu * (du[0][1] + du[1][0]),
                    mu * (du[1][2] + du[2][1]),
                    mu * (du[0][2] + du[2][0]),
                ];
                (s, 1.0 / ee.max(1e-12))
            })
            .collect();
        (6, v)
    } else if let Some(t) = &res.temp {
        let v = (0..ne)
            .into_par_iter()
            .map(|e| {
                let m = &mesh.materials[mesh.elem_mat[e] as usize];
                let g = grads(e);
                let mut q = vec![0.0; 3];
                for (a, &n) in mesh.elems[e].iter().enumerate() {
                    for c in 0..3 {
                        q[c] -= m.k_axis(c) * t[n as usize] * g[a][c] * 1e3;
                    }
                }
                (q, 1.0 / m.thermal_conductivity.max(1e-12))
            })
            .collect();
        (3, v)
    } else {
        return None;
    };
    let nn = mesh.nodes.len();
    let mut acc = vec![0.0; nn * ncomp];
    let mut wsum = vec![0.0; nn];
    let vol: Vec<f64> = (0..ne).map(|e| mesh.elem_volume(e)).collect();
    for e in (0..ne).filter(|&e| solid(e)) {
        for &n in &mesh.elems[e] {
            let n = n as usize;
            wsum[n] += vol[e];
            for c in 0..ncomp {
                acc[n * ncomp + c] += vol[e] * vals[e].0[c];
            }
        }
    }
    // shear components weigh twice in σ:σ
    let wc = |c: usize| if ncomp == 6 && c >= 3 { 2.0 } else { 1.0 };
    let mut elem_err2 = vec![0.0; ne];
    let (mut err2, mut norm2) = (0.0, 0.0);
    for e in (0..ne).filter(|&e| solid(e)) {
        let (s, w) = &vals[e];
        let mut ee = 0.0;
        for &n in &mesh.elems[e] {
            let n = n as usize;
            for c in 0..ncomp {
                let d = acc[n * ncomp + c] / wsum[n].max(1e-300) - s[c];
                ee += wc(c) * d * d;
            }
        }
        let ee = ee * w * vol[e] / 8.0;
        elem_err2[e] = ee;
        err2 += ee;
        norm2 += (0..ncomp).map(|c| wc(c) * s[c] * s[c]).sum::<f64>() * w * vol[e];
    }
    if !(norm2 > 0.0) {
        return None;
    }
    Some(ZzEstimate { error_pct: 100.0 * (err2 / (norm2 + err2)).sqrt(), elem_err2 })
}

/// Dörfler marking: the smallest set of elements carrying `theta` of the squared error, each
/// asked one level finer (up to `max_levels`). Returns the number of marked elements.
pub fn mark_for_refinement(mesh: &Mesh, est: &ZzEstimate, theta: f64, max_levels: u8, hint: &mut AdaptiveHint) -> usize {
    let total: f64 = est.elem_err2.iter().sum();
    let mut order: Vec<usize> = (0..est.elem_err2.len()).filter(|&e| est.elem_err2[e] > 0.0).collect();
    order.sort_unstable_by(|&a, &b| est.elem_err2[b].total_cmp(&est.elem_err2[a]));
    let mut acc = 0.0;
    let mut marked = 0;
    for e in order {
        if acc >= theta * total {
            break;
        }
        acc += est.elem_err2[e];
        let lvl = mesh.elem_level.get(e).copied().unwrap_or(0);
        if lvl >= max_levels {
            continue;
        }
        // the element's own cell (pre-fitting lattice box), shrunk so neighbours are not caught
        let x = mesh.elem_nodes(e);
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for p in &x {
            for k in 0..3 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
        let pad = 0.05 * mesh.elem_h_mm(e);
        hint.regions.push((add(lo, [pad; 3]), sub(hi, [pad; 3]), lvl + 1));
        marked += 1;
    }
    marked
}
