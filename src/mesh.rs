//! Voxel hexahedral mesher. Every element is a cube of side `h`, so a single element
//! matrix per material is reused for the whole model.
//!
//! Two meshing modes:
//! * solids only (structural, thermal, current flow)
//! * solids + enclosure (electrostatic, magnetostatic, CFD): the enclosure body (air, water…)
//!   fills the space around the solids and is meshed as region 1.

use crate::geometry::*;
use crate::material::Material;
use anyhow::{Result, bail};
use rayon::prelude::*;
use serde::Serialize;

#[derive(Clone, Debug)]
pub struct BFace {
    pub elem: u32,
    /// Corner node ids, counter-clockwise seen from outside.
    pub nodes: [u32; 4],
    /// 0..6 => -x,+x,-y,+y,-z,+z
    pub dir: u8,
    pub patch: u32,
    pub center: V3,
    /// Outer boundary of the enclosure (not a solid surface).
    pub outer: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct PatchInfo {
    pub name: String,
    pub area_mm2: f64,
    pub centroid: V3,
    pub normal: V3,
    pub n_faces: usize,
}

pub const REGION_SOLID: u8 = 0;
pub const REGION_ENCLOSURE: u8 = 1;

#[derive(Clone, Debug)]
pub struct Mesh {
    pub h: f64,
    pub origin: V3,
    pub dims: [usize; 3],
    pub nodes: Vec<V3>,
    pub elems: Vec<[u32; 8]>,
    pub elem_mat: Vec<u16>,
    /// REGION_SOLID or REGION_ENCLOSURE
    pub elem_region: Vec<u8>,
    /// Index into the model's body list (the body that owns the cell).
    pub elem_body: Vec<u16>,
    pub elem_cell: Vec<[u32; 3]>,
    pub materials: Vec<Material>,
    pub faces: Vec<BFace>,
    pub patches: Vec<PatchInfo>,
    pub warnings: Vec<String>,
    /// Bounds of the meshed domain.
    pub bbox: (V3, V3),
    pub body_names: Vec<String>,
    pub with_enclosure: bool,
    /// Per patch: (body index, patch index within the body's shape).
    pub patch_src: Vec<(u32, u32)>,
    /// Per element: true when boundary fitting moved its nodes (general hexahedron, not a cube).
    /// Empty for plain voxel meshes.
    pub fitted: Vec<bool>,
}

/// Local hex node index for corner offset (dx,dy,dz).
pub const CORNER: [[usize; 3]; 8] = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]];

pub fn dir_normal(dir: u8) -> V3 {
    let mut n = [0.0; 3];
    n[(dir / 2) as usize] = if dir % 2 == 0 { -1.0 } else { 1.0 };
    n
}

pub const MAX_CELLS: usize = 400_000;
/// Largest mesh for analyses solved by multigrid (static structural, thermal stress, steady
/// thermal): the stiffness is applied matrix-free, so memory grows only with the node count.
pub const MAX_CELLS_LARGE: usize = 2_000_000;

/// Element limit for the model's current analysis.
pub fn max_cells_for(model: &crate::model::Model) -> usize {
    use crate::model::AnalysisType::*;
    match model.analysis.kind {
        FullWaveEm | MicrowaveHeating => MAX_CELLS_FDTD,
        StaticStructural | ThermalStress | SteadyThermal => MAX_CELLS_LARGE,
        _ => MAX_CELLS,
    }
}

pub fn auto_size(bodies: &[Body], with_enclosure: bool) -> f64 {
    let (lo, hi) = domain_bounds(bodies, with_enclosure);
    let d = sub(hi, lo);
    let maxd = d[0].max(d[1]).max(d[2]).max(1e-6);
    let mut h = maxd / if with_enclosure { 50.0 } else { 40.0 };
    let vol = d[0].max(h) * d[1].max(h) * d[2].max(h);
    let budget = if with_enclosure { 90_000.0 } else { 120_000.0 };
    while vol / (h * h * h) > budget {
        h *= 1.15;
    }
    nice(h)
}

fn nice(h: f64) -> f64 {
    let p = 10f64.powf(h.log10().floor());
    let m = h / p;
    let r = if m < 1.5 { 1.0 } else if m < 2.25 { 2.0 } else if m < 3.5 { 2.5 } else if m < 7.5 { 5.0 } else { 10.0 };
    r * p
}

fn union_bounds<'a>(it: impl Iterator<Item = &'a Body>) -> Option<(V3, V3)> {
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    let mut any = false;
    for b in it {
        any = true;
        let (l, h) = b.shape.bounds();
        for k in 0..3 {
            lo[k] = lo[k].min(l[k]);
            hi[k] = hi[k].max(h[k]);
        }
    }
    any.then_some((lo, hi))
}

/// Bounds of the solid (additive) bodies.
pub fn additive_bounds(bodies: &[Body]) -> (V3, V3) {
    union_bounds(bodies.iter().filter(|b| b.op == Op::Add && b.role == Role::Solid)).unwrap_or(([0.0; 3], [0.0; 3]))
}

pub fn domain_bounds(bodies: &[Body], with_enclosure: bool) -> (V3, V3) {
    if with_enclosure {
        if let Some(b) = union_bounds(bodies.iter().filter(|b| b.op == Op::Add)) {
            return b;
        }
    }
    additive_bounds(bodies)
}

pub fn build_mesh(bodies: &[Body], h: f64, materials: &[Material], with_enclosure: bool) -> Result<Mesh> {
    build_mesh_max(bodies, h, materials, with_enclosure, MAX_CELLS)
}

/// Largest mesh for explicit full-wave (FDTD) analyses, which need no matrix factorisation.
pub const MAX_CELLS_FDTD: usize = 12_000_000;

pub fn build_mesh_max(bodies: &[Body], h: f64, materials: &[Material], with_enclosure: bool, max_cells: usize) -> Result<Mesh> {
    build_mesh_ex(bodies, h, materials, with_enclosure, max_cells, false)
}

/// `conservative`: keep only cells whose 8 corners are all inside the solid (for boundary
/// fitting: every surface node then lies inside the true surface and moves outward onto it).
/// Coordinates of the axis-aligned flat faces of the bodies, per axis (box faces, cylinder end caps,
/// flat axis-aligned STL faces), for snapping grid lines onto them.
fn axis_planes(bodies: &[Body]) -> [Vec<f64>; 3] {
    let mut p: [Vec<f64>; 3] = [vec![], vec![], vec![]];
    for b in bodies.iter().filter(|b| b.role == Role::Solid) {
        match &b.shape {
            Shape::Box { min, size } => {
                for a in 0..3 {
                    p[a].push(min[a]);
                    p[a].push(min[a] + size[a]);
                }
            }
            Shape::Cylinder { base, axis, height, .. } => {
                let a = axis.idx();
                p[a].push(base[a]);
                p[a].push(base[a] + height);
            }
            Shape::Stl { data: Some(d), .. } => {
                for seg in 0..d.n_segments {
                    let tris: Vec<usize> = (0..d.tris.len()).filter(|&t| d.segment[t] as usize == seg).collect();
                    let Some(&t0) = tris.first() else { continue };
                    for a in 0..3 {
                        let c = d.tris[t0][0][a];
                        let tol = 1e-9 * (d.max[a] - d.min[a]).abs().max(1.0);
                        if tris.iter().all(|&t| d.tris[t].iter().all(|v| (v[a] - c).abs() <= tol)) {
                            p[a].push(c);
                        }
                    }
                }
            }
            _ => {}
        }
    }
    for v in p.iter_mut() {
        v.sort_by(|a, b| a.total_cmp(b));
        v.dedup_by(|a, b| (*a - *b).abs() < 1e-12);
    }
    p
}

pub fn build_mesh_ex(bodies: &[Body], h: f64, materials: &[Material], with_enclosure: bool, max_cells: usize, conservative: bool) -> Result<Mesh> {
    let has_solid = bodies.iter().any(|b| b.op == Op::Add && b.role == Role::Solid);
    let mut restored: Vec<String> = Vec::new();
    let has_encl = bodies.iter().any(|b| b.op == Op::Add && b.role == Role::Enclosure);
    let with_enclosure = with_enclosure && has_encl;
    if !has_solid && !with_enclosure {
        bail!("no geometry: add at least one body (box, cylinder, sphere or STL)");
    }
    for b in bodies {
        b.shape.validate()?;
    }
    if !(h > 0.0) {
        bail!("element size must be positive");
    }
    let (lo, hi) = domain_bounds(bodies, with_enclosure);
    let size = sub(hi, lo);
    let mut dims = [0usize; 3];
    let mut origin = [0.0; 3];
    for k in 0..3 {
        dims[k] = ((size[k] / h) - 1e-6).ceil().max(1.0) as usize;
        origin[k] = lo[k] - (dims[k] as f64 * h - size[k]) / 2.0;
    }
    let ncell = dims[0] * dims[1] * dims[2];
    if ncell > max_cells * 4 {
        bail!("element size {h} mm is too small for this geometry ({ncell} grid cells); use a larger element size");
    }
    let [nx, ny, nz] = dims;
    let cid = |i: usize, j: usize, k: usize| i + nx * (j + ny * k);
    // grid lines per axis. For boundary-fitted meshes, a grid line within 0.45 h of an axis-aligned
    // flat face (box faces, cylinder end caps, flat STL faces) is moved onto it, so those faces are
    // meshed exactly by slightly stretched or squashed boxes instead of a fitted pillow layer. Only
    // curved and oblique surfaces are then pillowed, and a part whose thickness is not a whole
    // number of cells keeps full-thickness elements (no thin core between two pillow layers).
    let mut lines: [Vec<f64>; 3] = [0, 1, 2].map(|a| (0..=dims[a]).map(|i| origin[a] + i as f64 * h).collect());
    let mut moved: [Vec<bool>; 3] = [0, 1, 2].map(|a| vec![false; dims[a] + 1]);
    if conservative {
        let planes = axis_planes(bodies);
        for a in 0..3 {
            let mut cands: Vec<(f64, usize, f64)> = planes[a]
                .iter()
                .filter_map(|&c| {
                    let k = ((c - origin[a]) / h).round();
                    if k < 0.0 || k > dims[a] as f64 {
                        return None;
                    }
                    let delta = c - (origin[a] + k * h);
                    (delta.abs() <= 0.45 * h).then_some((delta.abs(), k as usize, c))
                })
                .collect();
            cands.sort_by(|p, q| p.0.total_cmp(&q.0));
            for (dl, k, c) in cands {
                if moved[a][k] || dl <= 1e-9 * h {
                    moved[a][k] |= dl <= 1e-9 * h; // already on the grid: keep it there
                    continue;
                }
                let lo_ok = k == 0 || c - lines[a][k - 1] >= 0.3 * h;
                let hi_ok = k == dims[a] || lines[a][k + 1] - c >= 0.3 * h;
                if lo_ok && hi_ok {
                    lines[a][k] = c;
                    moved[a][k] = true;
                }
            }
            // lines that sit on a plane without moving do not make their elements general
            for k in 0..=dims[a] {
                if moved[a][k] && (lines[a][k] - (origin[a] + k as f64 * h)).abs() <= 1e-9 * h {
                    moved[a][k] = false;
                }
            }
        }
    }
    let lines = &lines;
    let center = |i: usize, j: usize, k: usize| [0.5 * (lines[0][i] + lines[0][i + 1]), 0.5 * (lines[1][j] + lines[1][j + 1]), 0.5 * (lines[2][k] + lines[2][k + 1])];
    let mat_index = |name: &str| -> i32 { materials.iter().position(|m| m.name == name).unwrap_or(0) as i32 };

    // membership of each body on the grid
    let inside_of = |body: &Body| -> Result<Vec<bool>> {
        let mut v = vec![false; ncell];
        match &body.shape {
            Shape::Stl { data: Some(d), .. } => {
                let cols: Vec<(usize, usize, Vec<f64>)> = (0..ny * nz)
                    .into_par_iter()
                    .map(|jk| {
                        let (j, k) = (jk % ny, jk / ny);
                        let c = center(0, j, k);
                        (j, k, d.x_crossings(c[1] + h * 1.1e-4, c[2] + h * 0.7e-4))
                    })
                    .collect();
                for (j, k, xs) in cols {
                    if xs.len() < 2 {
                        continue;
                    }
                    for i in 0..nx {
                        let x = center(i, j, k)[0];
                        if xs.iter().filter(|&&q| q < x).count() % 2 == 1 {
                            v[cid(i, j, k)] = true;
                        }
                    }
                }
            }
            Shape::Stl { .. } => bail!("STL body '{}' is not loaded", body.name),
            shape => {
                v.par_iter_mut().enumerate().for_each(|(c, o)| {
                    let (i, j, k) = (c % nx, (c / nx) % ny, c / (nx * ny));
                    *o = shape.contains(center(i, j, k));
                });
            }
        }
        Ok(v)
    };

    // occupancy: material (-1 empty), region, owning body
    let mut occ = vec![-1i32; ncell];
    let mut region = vec![REGION_SOLID; ncell];
    let mut owner = vec![0u16; ncell];
    let mut encl_mat = vec![-1i32; ncell];
    let mut encl_owner = vec![0u16; ncell];
    if with_enclosure {
        for (bi, body) in bodies.iter().enumerate().filter(|(_, b)| b.role == Role::Enclosure && b.op == Op::Add) {
            let ins = inside_of(body)?;
            let m = mat_index(&body.material);
            for c in 0..ncell {
                if ins[c] {
                    occ[c] = m;
                    region[c] = REGION_ENCLOSURE;
                    owner[c] = bi as u16;
                    encl_mat[c] = m;
                    encl_owner[c] = bi as u16;
                }
            }
        }
    }
    // homogenised PCB materials are created per cell column and appended to the material list
    let mut extra_mats: Vec<Material> = Vec::new();
    for (bi, body) in bodies.iter().enumerate().filter(|(_, b)| b.role == Role::Solid) {
        let ins = inside_of(body)?;
        let m = mat_index(&body.material);
        let pcb = match &body.shape {
            Shape::Pcb { data: Some(d), offset, .. } if body.op == Op::Add => {
                let nzc = (0..nz).filter(|&k| {
                    let z = center(0, 0, k)[2];
                    z >= offset[2] && z <= offset[2] + d.thickness
                }).count();
                if nzc == 0 {
                    bail!("element size {h} mm is too large for the {:.2} mm board '{}': use at most {:.2} mm (half the thickness or less is better)", d.thickness, body.name, d.thickness);
                }
                let base = materials.get(m.max(0) as usize).cloned().unwrap_or_else(|| materials[0].clone());
                Some((d.clone(), *offset, nzc as f64 * h, base))
            }
            _ => None,
        };
        let mut col_cache: std::collections::HashMap<(usize, usize), i32> = std::collections::HashMap::new();
        for c in 0..ncell {
            if !ins[c] {
                continue;
            }
            let m = match &pcb {
                Some((d, off, t_vox, base)) => {
                    let (i, j) = (c % nx, (c / nx) % ny);
                    *col_cache.entry((i, j)).or_insert_with(|| {
                        let p = center(i, j, 0);
                        let mat = d.material_at(&body.name, base, p[0] - off[0], p[1] - off[1], h, *t_vox);
                        match materials.iter().chain(extra_mats.iter()).position(|x| x.name == mat.name) {
                            Some(k) => k as i32,
                            None => {
                                extra_mats.push(mat);
                                (materials.len() + extra_mats.len() - 1) as i32
                            }
                        }
                    })
                }
                None => m,
            };
            match body.op {
                Op::Add => {
                    occ[c] = m;
                    region[c] = REGION_SOLID;
                    owner[c] = bi as u16;
                }
                Op::Subtract => {
                    if region[c] == REGION_SOLID && occ[c] >= 0 {
                        occ[c] = encl_mat[c];
                        region[c] = REGION_ENCLOSURE;
                        owner[c] = encl_owner[c];
                    }
                }
            }
        }
    }

    if conservative {
        let (gx, gy, gz) = (nx + 1, ny + 1, nz + 1);
        let nid = |i: usize, j: usize, k: usize| i + gx * (j + gy * k);
        let node_pt = |i: usize, j: usize, k: usize| [lines[0][i], lines[1][j], lines[2][k]];
        let eps = 1e-6 * h;
        // membership of grid nodes (boundary counts as inside for added bodies, as outside for cuts)
        let node_in = |body: &Body, strict: bool| -> Vec<bool> {
            let mut v = vec![false; gx * gy * gz];
            match &body.shape {
                Shape::Stl { data: Some(d), .. } => {
                    // a node on the surface counts as inside: rays just beside it on all four
                    // sides in y/z (a single offset ray misses nodes on the top and y-max faces)
                    let cols: Vec<(usize, usize, Vec<Vec<f64>>)> = (0..gy * gz)
                        .into_par_iter()
                        .map(|jk| {
                            let (j, k) = (jk % gy, jk / gy);
                            let p = node_pt(0, j, k);
                            let (a, b) = (h * 1.1e-4, h * 0.7e-4);
                            (j, k, [(a, b), (-a, -b), (a, -b), (-a, b)].iter().map(|(dy, dz)| d.x_crossings(p[1] + dy, p[2] + dz)).collect())
                        })
                        .collect();
                    for (j, k, rays) in cols {
                        for i in 0..gx {
                            let x = node_pt(i, j, k)[0];
                            let inside = rays.iter().any(|xs| {
                                let odd = |t: f64| xs.iter().filter(|&&q| q < t).count() % 2 == 1;
                                odd(x - eps) || odd(x + eps)
                            });
                            if inside {
                                v[nid(i, j, k)] = true;
                            }
                        }
                    }
                }
                shape => {
                    v.par_iter_mut().enumerate().for_each(|(g, o)| {
                        let p = node_pt(g % gx, (g / gx) % gy, g / (gx * gy));
                        *o = if strict {
                            (0..6).all(|q| {
                                let mut t = p;
                                t[q / 2] += if q % 2 == 0 { -eps } else { eps };
                                shape.contains(t)
                            })
                        } else {
                            shape.contains(p)
                        };
                    });
                }
            }
            v
        };
        let mut solid = vec![false; gx * gy * gz];
        for b in bodies.iter().filter(|b| b.role == Role::Solid && b.op == Op::Add) {
            for (s, v) in solid.iter_mut().zip(node_in(b, false)) {
                *s |= v;
            }
        }
        for b in bodies.iter().filter(|b| b.role == Role::Solid && b.op == Op::Subtract) {
            for (s, v) in solid.iter_mut().zip(node_in(b, true)) {
                if v {
                    *s = false;
                }
            }
        }
        let mut dropped = 0usize;
        let before = (occ.clone(), region.clone(), owner.clone());
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let c = cid(i, j, k);
                    if occ[c] < 0 || region[c] != REGION_SOLID {
                        continue;
                    }
                    let all = CORNER.iter().all(|o| solid[nid(i + o[0], j + o[1], k + o[2])]);
                    if !all {
                        occ[c] = if with_enclosure { encl_mat[c] } else { -1 };
                        if with_enclosure && encl_mat[c] >= 0 {
                            region[c] = REGION_ENCLOSURE;
                            owner[c] = encl_owner[c];
                        }
                        dropped += 1;
                    }
                }
            }
        }
        // a body that keeps no whole cell (thinner than a cell somewhere it matters) would vanish:
        // give it back its centre-inside cells, unfitted, and say so
        if dropped > 0 {
            let nb = bodies.len();
            let (mut had, mut has) = (vec![0usize; nb], vec![0usize; nb]);
            for c in 0..ncell {
                if before.0[c] >= 0 && before.1[c] == REGION_SOLID {
                    had[before.2[c] as usize] += 1;
                }
                if occ[c] >= 0 && region[c] == REGION_SOLID {
                    has[owner[c] as usize] += 1;
                }
            }
            for bi in 0..nb {
                if had[bi] > 0 && has[bi] == 0 {
                    for c in 0..ncell {
                        if before.0[c] >= 0 && before.1[c] == REGION_SOLID && before.2[c] as usize == bi {
                            occ[c] = before.0[c];
                            region[c] = before.1[c];
                            owner[c] = before.2[c];
                        }
                    }
                    restored.push(bodies[bi].name.clone());
                }
            }
        }
    }

    // Connected components (face adjacency); drop tiny floating fragments.
    let mut warnings = Vec::new();
    if !restored.is_empty() {
        warnings.push(format!("{} thinner than a whole cell at {h} mm: meshed with plain voxels (not fitted to the surface) — use a smaller element size for accurate results there", restored.join(", ")));
    }
    let mut comp = vec![u32::MAX; ncell];
    let mut sizes = Vec::new();
    for s in 0..ncell {
        if occ[s] < 0 || comp[s] != u32::MAX {
            continue;
        }
        let id = sizes.len() as u32;
        let mut cnt = 0usize;
        let mut stack = vec![s];
        comp[s] = id;
        while let Some(c) = stack.pop() {
            cnt += 1;
            let (i, j, k) = ((c % nx) as isize, ((c / nx) % ny) as isize, (c / (nx * ny)) as isize);
            for (di, dj, dk) in [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)] {
                let (ii, jj, kk) = (i + di, j + dj, k + dk);
                if ii < 0 || jj < 0 || kk < 0 || ii >= nx as isize || jj >= ny as isize || kk >= nz as isize {
                    continue;
                }
                let n = cid(ii as usize, jj as usize, kk as usize);
                if occ[n] >= 0 && comp[n] == u32::MAX {
                    comp[n] = id;
                    stack.push(n);
                }
            }
        }
        sizes.push(cnt);
    }
    if sizes.is_empty() {
        bail!("geometry produced an empty mesh; check that bodies are not fully subtracted and the element size is smaller than the part");
    }
    let largest = *sizes.iter().max().unwrap();
    let keep: Vec<bool> = sizes.iter().map(|&s| s * 50 >= largest && s >= 2).collect();
    let dropped: usize = sizes.iter().zip(&keep).filter(|(_, k)| !**k).map(|(s, _)| s).sum();
    if dropped > 0 {
        warnings.push(format!("removed {dropped} disconnected fragment cell(s)"));
    }
    if !with_enclosure && keep.iter().filter(|k| **k).count() > 1 {
        warnings.push("model contains several disconnected parts; each needs its own supports".into());
    }
    for c in 0..ncell {
        if occ[c] >= 0 && !keep[comp[c] as usize] {
            occ[c] = -1;
        }
    }

    // Nodes and elements
    let (gx, gy) = (nx + 1, ny + 1);
    let gid = |i: usize, j: usize, k: usize| i + gx * (j + gy * k);
    let mut node_map = vec![u32::MAX; gx * gy * (nz + 1)];
    let mut nodes = Vec::new();
    let (mut elems, mut elem_mat, mut elem_region, mut elem_body, mut elem_cell) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut elem_of_cell = vec![u32::MAX; ncell];
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let c = cid(i, j, k);
                if occ[c] < 0 {
                    continue;
                }
                let mut en = [0u32; 8];
                for (l, d) in CORNER.iter().enumerate() {
                    let g = gid(i + d[0], j + d[1], k + d[2]);
                    if node_map[g] == u32::MAX {
                        node_map[g] = nodes.len() as u32;
                        nodes.push([lines[0][i + d[0]], lines[1][j + d[1]], lines[2][k + d[2]]]);
                    }
                    en[l] = node_map[g];
                }
                elem_of_cell[c] = elems.len() as u32;
                elems.push(en);
                elem_mat.push(occ[c] as u16);
                elem_region.push(region[c]);
                elem_body.push(owner[c]);
                elem_cell.push([i as u32, j as u32, k as u32]);
            }
        }
    }
    if elems.len() > max_cells {
        bail!("mesh has {} elements (limit {}); increase the element size", elems.len(), max_cells);
    }

    // Boundary faces: solid surfaces (solid next to non-solid) and the enclosure's outer boundary.
    let mut faces = Vec::new();
    let order = [(0usize, 0usize), (1, 0), (1, 1), (0, 1)];
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let c = cid(i, j, k);
                if occ[c] < 0 {
                    continue;
                }
                let ijk = [i, j, k];
                for dir in 0..6u8 {
                    let a = (dir / 2) as usize;
                    let plus = dir % 2 == 1;
                    let mut n = [i as isize, j as isize, k as isize];
                    n[a] += if plus { 1 } else { -1 };
                    let inside = n[a] >= 0 && (n[a] as usize) < dims[a];
                    let ncell = inside.then(|| cid(n[0] as usize, n[1] as usize, n[2] as usize));
                    let n_active = ncell.is_some_and(|q| occ[q] >= 0);
                    let n_solid = ncell.is_some_and(|q| occ[q] >= 0 && region[q] == REGION_SOLID);
                    let outer = match region[c] {
                        REGION_SOLID if !n_solid => false,
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
                        quad[q] = node_map[gid(ijk[0] + d[0], ijk[1] + d[1], ijk[2] + d[2])];
                    }
                    if !plus {
                        quad.swap(1, 3);
                    }
                    let mut ctr = center(i, j, k);
                    ctr[a] = lines[a][ijk[a] + plus as usize];
                    faces.push(BFace { elem: elem_of_cell[c], nodes: quad, dir, patch: 0, center: ctr, outer });
                }
            }
        }
    }

    // Patch (CAD face) assignment
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
        .map(|f| {
            let n = dir_normal(f.dir);
            let want = if f.outer { Role::Enclosure } else { Role::Solid };
            let mut best = (0usize, f64::INFINITY);
            for (bi, b) in bodies.iter().enumerate() {
                if b.role != want {
                    continue;
                }
                let (pi, s) = b.shape.best_patch(f.center, n, h, b.op == Op::Subtract);
                if s < best.1 {
                    best = (offsets[bi] + pi, s);
                }
            }
            best.0 as u32
        })
        .collect();
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
        let p = &mut patches[f.patch as usize];
        p.n_faces += 1;
        p.area_mm2 += h * h;
        p.centroid = add(p.centroid, f.center);
        p.normal = add(p.normal, dir_normal(f.dir));
    }
    for p in patches.iter_mut() {
        p.centroid = scale(p.centroid, 1.0 / p.n_faces as f64);
        p.normal = normalize(p.normal);
        for v in p.centroid.iter_mut().chain(p.normal.iter_mut()) {
            *v = (*v * 1000.0).round() / 1000.0;
        }
    }

    // elements on snapped grid lines are boxes of another size: general hexahedra
    let snapped_elems: Vec<bool> = if moved.iter().any(|m| m.iter().any(|x| *x)) {
        elem_cell.iter().map(|c| (0..3).any(|a| moved[a][c[a] as usize] || moved[a][c[a] as usize + 1])).collect()
    } else {
        Vec::new()
    };
    Ok(Mesh {
        h,
        origin,
        dims,
        nodes,
        elems,
        elem_mat,
        elem_region,
        elem_body,
        elem_cell,
        materials: materials.iter().cloned().chain(extra_mats).collect(),
        faces,
        patches,
        warnings,
        bbox: (lo, hi),
        body_names: bodies.iter().map(|b| b.name.clone()).collect(),
        with_enclosure,
        patch_src,
        fitted: snapped_elems,
    })
}

/// Mesh for a model: voxel mesh, boundary-fitted when the analysis asks for it.
pub fn build_for(model: &crate::model::Model, h: f64, max_cells: usize) -> Result<Mesh> {
    let fit = model.mesh_fitted();
    let mut m = build_mesh_ex(&model.bodies, h, &model.materials(), model.mesh_with_enclosure(), max_cells, fit)?;
    if fit {
        m.fit_boundary(&model.bodies);
    }
    Ok(m)
}

/// Smallest corner det J (relative to the undistorted cube) a fitted element may have.
pub const MIN_QUALITY: f64 = 0.05;

/// Outcome of boundary fitting.
#[derive(Debug, Default, Clone)]
pub struct FitReport {
    pub moved_nodes: usize,
    pub max_move_mm: f64,
    pub fitted_elements: usize,
    pub relaxed_elements: usize,
}

/// det J of the trilinear map of hex `x` (corners in CORNER order) at ξ ∈ [-1, 1]³.
pub fn hex_det_j(x: &[V3; 8], xi: V3) -> f64 {
    let mut j = [[0.0f64; 3]; 3];
    for (i, c) in CORNER.iter().enumerate() {
        let s = c.map(|v| 2.0 * v as f64 - 1.0);
        let f = |k: usize| 1.0 + s[k] * xi[k];
        let g = [s[0] * f(1) * f(2) / 8.0, f(0) * s[1] * f(2) / 8.0, f(0) * f(1) * s[2] / 8.0];
        for r in 0..3 {
            for c2 in 0..3 {
                j[r][c2] += x[i][r] * g[c2];
            }
        }
    }
    j[0][0] * (j[1][1] * j[2][2] - j[1][2] * j[2][1]) - j[0][1] * (j[1][0] * j[2][2] - j[1][2] * j[2][0]) + j[0][2] * (j[1][0] * j[2][1] - j[1][1] * j[2][0])
}

impl Mesh {
    pub fn is_fitted(&self, e: usize) -> bool {
        self.fitted.get(e).copied().unwrap_or(false)
    }
    pub fn elem_nodes(&self, e: usize) -> [V3; 8] {
        self.elems[e].map(|n| self.nodes[n as usize])
    }
    /// Element volume [mm³] (h³ for cubes).
    pub fn elem_volume(&self, e: usize) -> f64 {
        if !self.is_fitted(e) {
            return self.h.powi(3);
        }
        let x = self.elem_nodes(e);
        let g = 0.577_350_269_189_625_8;
        let mut v = 0.0;
        for a in [-g, g] {
            for b in [-g, g] {
                for c in [-g, g] {
                    v += hex_det_j(&x, [a, b, c]);
                }
            }
        }
        v
    }

    /// Move surface nodes onto the true surfaces of the bodies (boundary fitting) and relax the
    /// next layers inward, so curved and off-grid surfaces are represented exactly at the nodes.
    /// Elements that would become too distorted are relaxed back towards the voxel shape.
    pub fn fit_boundary(&mut self, bodies: &[Body]) -> FitReport {
        let h = self.h;
        // elements already general (on snapped grid lines) stay general
        let pre_fitted = std::mem::take(&mut self.fitted);
        // ---- which faces need fitting: those with a node off their true surface (curved surfaces,
        // planes that miss the grid). Decided per face, not per surface: a flat face that lies on a
        // (snapped) grid plane stays as it is even when a few faces of the same surface step off it
        // next to a hole, otherwise it would get a zero-thickness pillow that inverts at the edges.
        let n_old = self.nodes.len();
        let face_needs: Vec<bool> = self
            .faces
            .iter()
            .map(|f| {
                if f.outer {
                    return false;
                }
                let (bi, lp) = self.patch_src[f.patch as usize];
                let Some(b) = bodies.get(bi as usize) else { return false };
                f.nodes.iter().any(|&n| {
                    let p0 = self.nodes[n as usize];
                    b.shape.project(lp as usize, p0).is_some_and(|q| norm(sub(q, p0)) > 1e-6 * h)
                })
            })
            .collect();
        // ---- pillow layer on those faces: a new hexahedron whose outer nodes are fitted to the
        // surface. No element then has more than one face on a curved surface, so staircase corners
        // cannot collapse.
        let mut prime: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
        let mut shrink_dir: Vec<V3> = vec![[0.0; 3]; n_old];
        let mut plane_normals: Vec<Vec<V3>> = vec![Vec::new(); n_old];
        let solid_faces: Vec<usize> = (0..self.faces.len()).filter(|&f| face_needs[f]).collect();
        for (_, f) in self.faces.iter().enumerate().filter(|(fi, f)| !f.outer && !face_needs[*fi]) {
            for &n in &f.nodes {
                let nr = dir_normal(f.dir);
                if !plane_normals[n as usize].contains(&nr) {
                    plane_normals[n as usize].push(nr);
                }
            }
        }
        for &fi in &solid_faces {
            let f = self.faces[fi].clone();
            let nrm = dir_normal(f.dir);
            for &n in &f.nodes {
                shrink_dir[n as usize] = add(shrink_dir[n as usize], nrm);
                if let std::collections::hash_map::Entry::Vacant(v) = prime.entry(n) {
                    v.insert(self.nodes.len() as u32);
                    self.nodes.push(self.nodes[n as usize]);
                }
            }
        }
        // edges (pairs of old nodes) used by pillowed faces → count, to find exposed pillow sides
        let ekey = |a: u32, b: u32| if a < b { (a, b) } else { (b, a) };
        let mut edge_use: std::collections::HashMap<(u32, u32), u32> = std::collections::HashMap::new();
        for &fi in &solid_faces {
            let nds = self.faces[fi].nodes;
            for k in 0..4 {
                *edge_use.entry(ekey(nds[k], nds[(k + 1) % 4])).or_insert(0) += 1;
            }
        }
        // non-pillowed faces by edge (to give exposed pillow sides their patch)
        let mut plain_by_edge: std::collections::HashMap<(u32, u32), usize> = std::collections::HashMap::new();
        for (fi, f) in self.faces.iter().enumerate() {
            if f.outer || face_needs[fi] {
                continue;
            }
            for k in 0..4 {
                plain_by_edge.insert(ekey(f.nodes[k], f.nodes[(k + 1) % 4]), fi);
            }
        }
        let mut new_faces: Vec<BFace> = Vec::new();
        for &fi in &solid_faces {
            let f = self.faces[fi].clone();
            let e = f.elem as usize;
            let top = f.nodes.map(|n| prime[&n]);
            // face nodes are counter-clockwise seen from outside: bottom (voxel face) → top (outer)
            self.elems.push([f.nodes[0], f.nodes[1], f.nodes[2], f.nodes[3], top[0], top[1], top[2], top[3]]);
            self.elem_mat.push(self.elem_mat[e]);
            self.elem_region.push(self.elem_region[e]);
            self.elem_body.push(self.elem_body[e]);
            self.elem_cell.push(self.elem_cell[e]);
            let pe = (self.elems.len() - 1) as u32;
            // exposed side faces where the pillow meets an unfitted (grid-aligned) patch
            for k in 0..4 {
                let (a, b) = (f.nodes[k], f.nodes[(k + 1) % 4]);
                if edge_use.get(&ekey(a, b)).copied().unwrap_or(0) >= 2 {
                    continue;
                }
                let Some(&pf) = plain_by_edge.get(&ekey(a, b)) else { continue };
                let plain = self.faces[pf].clone();
                let mut quad = [a, b, prime[&b], prime[&a]];
                let pts = quad.map(|n| self.nodes[n as usize]);
                let nrm = cross(sub(pts[2], pts[0]), sub(pts[3], pts[1]));
                let want = dir_normal(plain.dir);
                // degenerate now (zero thickness); orient by the side's own direction from the face
                let side_out = sub(add(pts[0], pts[1]), add(self.nodes[f.nodes[(k + 2) % 4] as usize], self.nodes[f.nodes[(k + 3) % 4] as usize]));
                let ok = if norm(nrm) > 1e-12 { dot(nrm, want) > 0.0 } else { dot(cross(sub(pts[1], pts[0]), dir_normal(f.dir)), side_out) > 0.0 };
                if !ok {
                    quad = [b, a, prime[&a], prime[&b]];
                }
                new_faces.push(BFace { elem: pe, nodes: quad, dir: plain.dir, patch: plain.patch, center: [0.0; 3], outer: false });
            }
            let face = &mut self.faces[fi];
            face.elem = pe;
            face.nodes = top;
        }
        self.faces.extend(new_faces);
        let nn = self.nodes.len();
        // patches touching each (outer, pillow) surface node
        let mut node_patches: Vec<Vec<u32>> = vec![Vec::new(); nn];
        for f in self.faces.iter().filter(|f| !f.outer) {
            for &n in &f.nodes {
                let v = &mut node_patches[n as usize];
                if !v.contains(&f.patch) {
                    v.push(f.patch);
                }
            }
        }
        let base = self.nodes.clone();
        let mut d = vec![[0.0f64; 3]; nn];
        let mut pinned = vec![false; nn];
        for (&old, _) in prime.iter() {
            pinned[old as usize] = true;
        }
        for n in 0..nn {
            let ps = &node_patches[n];
            if ps.is_empty() {
                continue;
            }
            pinned[n] = true;
            // closest point on the touching patches (alternating projections converge to the
            // edge where two patches meet). With a conservative voxelisation every surface node
            // is inside the solid, so moves are outward and keep their order along the surface.
            let mut q = base[n];
            let mut ok = true;
            for _ in 0..10 {
                for &p in ps {
                    let (bi, lp) = self.patch_src[p as usize];
                    match bodies.get(bi as usize).and_then(|b| b.shape.project(lp as usize, q)) {
                        Some(t) => q = t,
                        None => ok = false,
                    }
                }
            }
            let mv = sub(q, base[n]);
            if ok && norm(mv) <= 1.5 * h {
                d[n] = mv;
            }
        }
        // tangential smoothing of fitted surface nodes on single-patch surfaces: staircase steps leave
        // uneven spacing along the surface; average with surface neighbours and re-project
        {
            let mut snbr: Vec<Vec<u32>> = vec![Vec::new(); nn];
            for f in self.faces.iter().filter(|f| !f.outer) {
                for k in 0..4 {
                    let (a, b) = (f.nodes[k], f.nodes[(k + 1) % 4]);
                    if !snbr[a as usize].contains(&b) {
                        snbr[a as usize].push(b);
                    }
                    if !snbr[b as usize].contains(&a) {
                        snbr[b as usize].push(a);
                    }
                }
            }
            for _ in 0..4 {
                let cur: Vec<V3> = (0..nn).map(|n| add(base[n], d[n])).collect();
                for n in 0..nn {
                    let ps = &node_patches[n];
                    if ps.len() != 1 || snbr[n].is_empty() || norm(d[n]) == 0.0 {
                        continue;
                    }
                    let mut avg = [0.0; 3];
                    for &m in &snbr[n] {
                        avg = add(avg, cur[m as usize]);
                    }
                    avg = scale(avg, 1.0 / snbr[n].len() as f64);
                    let target = add(scale(cur[n], 0.5), scale(avg, 0.5));
                    let (bi, lp) = self.patch_src[ps[0] as usize];
                    if let Some(q) = bodies.get(bi as usize).and_then(|b| b.shape.project(lp as usize, target)) {
                        let mv = sub(q, base[n]);
                        if norm(mv) <= 1.5 * h {
                            d[n] = mv;
                        }
                    }
                }
            }
        }
        // the voxel surface nodes step inward along their pillowed faces' normals (kept inside any
        // unfitted plane they also lie on) so the pillow layer is at least 0.4 h thick. Only the
        // missing thickness is taken: where the true surface is already far from the voxel face the
        // pillow is thick enough, and stepping in anyway would crush a thin core between two
        // pillowed faces (a plate whose thickness is not a whole number of cells).
        {
            for (&old, &pr) in prime.iter() {
                let o = old as usize;
                let mut sd = shrink_dir[o];
                for pnrm in &plane_normals[o] {
                    sd = sub(sd, scale(*pnrm, dot(sd, *pnrm)));
                }
                let l = norm(sd);
                if l > 1e-9 {
                    let u = scale(sd, 1.0 / l);
                    // pillow thickness the surface move already gives along this direction
                    let gap = dot(d[pr as usize], u).max(0.0);
                    let t = (0.4 * h - gap).max(0.1 * h);
                    d[o] = scale(u, -t);
                }
            }
        }
        // node neighbours along element edges
        const EDGES: [(usize, usize); 12] = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)];
        let mut nbr: Vec<Vec<u32>> = vec![Vec::new(); nn];
        for en in &self.elems {
            for (a, b) in EDGES {
                let (x, y) = (en[a], en[b]);
                if !nbr[x as usize].contains(&y) {
                    nbr[x as usize].push(y);
                }
                if !nbr[y as usize].contains(&x) {
                    nbr[y as usize].push(x);
                }
            }
        }
        // relax the displacement into two interior layers
        for _ in 0..2 {
            let prev = d.clone();
            for n in 0..nn {
                if pinned[n] {
                    continue;
                }
                let moved: Vec<V3> = nbr[n].iter().map(|&m| prev[m as usize]).filter(|v| norm(*v) > 0.0).collect();
                if !moved.is_empty() {
                    let mut s = [0.0; 3];
                    for v in &nbr[n] {
                        s = add(s, prev[*v as usize]);
                    }
                    d[n] = scale(s, 0.5 / nbr[n].len() as f64);
                }
            }
        }
        // keep every element valid: min corner det J ≥ MIN_QUALITY × the cube's
        let ne = self.elems.len();
        let cube = (h / 2.0).powi(3);
        let corners: Vec<V3> = CORNER.iter().map(|c| c.map(|v| 2.0 * v as f64 - 1.0)).collect();
        let quality = |e: usize, d: &[V3]| -> f64 {
            let x = self.elems[e].map(|n| add(base[n as usize], d[n as usize]));
            corners.iter().map(|xi| hex_det_j(&x, *xi)).fold(f64::INFINITY, f64::min) / cube
        };
        let touched = |e: usize, d: &[V3]| self.elems[e].iter().any(|&n| norm(d[n as usize]) > 0.0);
        let mut relaxed = vec![false; ne];
        for round in 0..8 {
            let bad: Vec<usize> = (0..ne).filter(|&e| touched(e, &d) && quality(e, &d) < MIN_QUALITY).collect();
            if bad.is_empty() {
                break;
            }
            for &e in &bad {
                relaxed[e] = true;
                for &n in &self.elems[e] {
                    let n = n as usize;
                    if prime.contains_key(&(n as u32)) {
                        // voxel-surface node: keep ≥ 0.1 h of the inward step so the pillow has thickness
                        let l = norm(d[n]);
                        if l > 0.1 * h {
                            d[n] = scale(d[n], (0.5f64).max(0.1 * h / l));
                        }
                    } else {
                        d[n] = if round >= 6 { [0.0; 3] } else { scale(d[n], 0.5) };
                    }
                }
            }
        }
        let mut rep = FitReport::default();
        for n in 0..nn {
            let m = norm(d[n]);
            if m > 0.0 {
                rep.moved_nodes += 1;
                rep.max_move_mm = rep.max_move_mm.max(m);
                self.nodes[n] = add(base[n], d[n]);
            }
        }
        self.fitted = (0..ne).map(|e| touched(e, &d) || e >= ne - solid_faces.len() || pre_fitted.get(e).copied().unwrap_or(false)).collect();
        rep.fitted_elements = self.fitted.iter().filter(|f| **f).count();
        rep.relaxed_elements = relaxed.iter().filter(|r| **r).count();
        // faces and patches follow the nodes
        for f in self.faces.iter_mut() {
            f.center = scale(f.nodes.iter().fold([0.0; 3], |a, &n| add(a, self.nodes[n as usize])), 0.25);
        }
        for p in self.patches.iter_mut() {
            p.area_mm2 = 0.0;
        }
        for f in &self.faces {
            let q = f.nodes.map(|n| self.nodes[n as usize]);
            let a = scale(cross(sub(q[2], q[0]), sub(q[3], q[1])), 0.5);
            self.patches[f.patch as usize].area_mm2 += norm(a);
        }
        rep
    }

    /// Resolve face selectors into boundary face indices.
    /// Selectors: "body.patch" (CAD face), "body" (all faces of a body), "bbox.xmin" .. "bbox.zmax",
    /// "all" (whole outer surface).
    pub fn resolve_faces(&self, sel: &[String]) -> Result<Vec<u32>> {
        let mut out = Vec::new();
        for s in sel {
            let sl = s.trim().to_lowercase();
            let before = out.len();
            if sl == "all" {
                out.extend(self.faces.iter().enumerate().filter(|(_, f)| !f.outer).map(|(i, _)| i as u32));
            } else if let Some(side) = sl.strip_prefix("bbox.") {
                let names = ["xmin", "xmax", "ymin", "ymax", "zmin", "zmax"];
                let Some(d) = names.iter().position(|n| *n == side) else {
                    bail!("unknown bbox side '{side}', use one of {names:?}");
                };
                let a = d / 2;
                let solid = |f: &&BFace| !f.outer && f.dir as usize == d;
                let ext = self.faces.iter().filter(solid).map(|f| f.center[a]).fold(if d % 2 == 0 { f64::INFINITY } else { f64::NEG_INFINITY }, |acc, v| if d % 2 == 0 { acc.min(v) } else { acc.max(v) });
                for (i, f) in self.faces.iter().enumerate() {
                    if !f.outer && f.dir as usize == d && (f.center[a] - ext).abs() < self.h * 0.25 {
                        out.push(i as u32);
                    }
                }
            } else {
                let ids: Vec<u32> = self
                    .patches
                    .iter()
                    .enumerate()
                    .filter(|(_, p)| {
                        let pl = p.name.to_lowercase();
                        pl == sl || pl.split('.').next() == Some(sl.as_str())
                    })
                    .map(|(i, _)| i as u32)
                    .collect();
                for (i, f) in self.faces.iter().enumerate() {
                    if ids.contains(&f.patch) {
                        out.push(i as u32);
                    }
                }
            }
            if out.len() == before {
                let avail: Vec<&str> = self.patches.iter().map(|p| p.name.as_str()).collect();
                bail!("face selector '{s}' matched nothing in the current mesh. Available faces: {avail:?} (also bbox.xmin..bbox.zmax, all)");
            }
        }
        out.sort_unstable();
        out.dedup();
        Ok(out)
    }

    /// Elements belonging to a body (by name, case-insensitive).
    pub fn body_elems(&self, name: &str) -> Result<Vec<usize>> {
        let Some(bi) = self.body_names.iter().position(|b| b.eq_ignore_ascii_case(name)) else {
            bail!("no body named '{name}' (bodies: {:?})", self.body_names);
        };
        let v: Vec<usize> = (0..self.elems.len()).filter(|&e| self.elem_body[e] as usize == bi && self.elem_region[e] == REGION_SOLID).collect();
        if v.is_empty() {
            bail!("body '{name}' has no elements in this mesh");
        }
        Ok(v)
    }

    pub fn n_dofs_structural(&self) -> usize {
        self.nodes.len() * 3
    }

    pub fn nearest_node(&self, p: V3) -> usize {
        self.nodes
            .par_iter()
            .enumerate()
            .map(|(i, n)| (i, dot(sub(*n, p), sub(*n, p))))
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map(|x| x.0)
            .unwrap_or(0)
    }

    pub fn volume_mm3(&self) -> f64 {
        self.elem_region.iter().filter(|r| **r == REGION_SOLID).count() as f64 * self.h.powi(3)
    }

    pub fn elem_center(&self, e: usize) -> V3 {
        let c = self.elem_cell[e];
        [self.origin[0] + (c[0] as f64 + 0.5) * self.h, self.origin[1] + (c[1] as f64 + 0.5) * self.h, self.origin[2] + (c[2] as f64 + 0.5) * self.h]
    }
}
