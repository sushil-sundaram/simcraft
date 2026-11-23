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
}

/// Local hex node index for corner offset (dx,dy,dz).
pub const CORNER: [[usize; 3]; 8] = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]];

pub fn dir_normal(dir: u8) -> V3 {
    let mut n = [0.0; 3];
    n[(dir / 2) as usize] = if dir % 2 == 0 { -1.0 } else { 1.0 };
    n
}

pub const MAX_CELLS: usize = 400_000;

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
    let has_solid = bodies.iter().any(|b| b.op == Op::Add && b.role == Role::Solid);
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
    let center = |i: usize, j: usize, k: usize| [origin[0] + (i as f64 + 0.5) * h, origin[1] + (j as f64 + 0.5) * h, origin[2] + (k as f64 + 0.5) * h];
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

    // Connected components (face adjacency); drop tiny floating fragments.
    let mut warnings = Vec::new();
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
                        nodes.push([origin[0] + (i + d[0]) as f64 * h, origin[1] + (j + d[1]) as f64 * h, origin[2] + (k + d[2]) as f64 * h]);
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
                    ctr[a] += if plus { h / 2.0 } else { -h / 2.0 };
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
    for (f, &g) in faces.iter_mut().zip(&assign) {
        let g = g as usize;
        if remap[g] == u32::MAX {
            remap[g] = patches.len() as u32;
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
    })
}

impl Mesh {
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
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
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
