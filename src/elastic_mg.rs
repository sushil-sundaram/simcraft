//! Matrix-free multigrid-preconditioned CG for linear elasticity on the voxel mesh.
//!
//! * Fine level: never assembled. K·x is applied element by element (node-gathered, in
//!   coloured so each colour scatters in parallel): voxel cubes share one unit stiffness per
//!   material, scaled by E·h; boundary-fitted elements store their difference from the cube
//!   (symmetric, f32). Memory is ~O(nodes), so 2 M-element models fit.
//! * Coarse levels live on the voxel lattice with spacing 2h, 4h, …: trilinear prolongation P,
//!   Galerkin operators Pᵀ A P. The first coarse operator is built element by element — a cube
//!   inside a coarse cell contributes E·h·Pₒᵀ U Pₒ, precomputed for each of the 8 octants — and
//!   further levels by sparse triple products.
//! * Dirichlet dofs: the prolongation is masked there, so corrections never move a support and the
//!   coarse operators see exactly the constrained stiffness.
//! * Smoother: Chebyshev (degree 3 fine, 2 coarse) on the Jacobi-scaled operator; coarsest level:
//!   direct Cholesky.
//!
//! Typically 10–25 CG iterations instead of thousands for diagonal preconditioning: 1.07 M DOF in
//! 3 s (Jacobi-PCG: 60 s CPU, 10 s GPU), 5.6 M DOF / 1.8 M elements in ~28 s on a laptop.

use crate::fem::{ElemUnit, elem_xyz_m, hex_elastic_general};
use crate::mesh::{CORNER, Mesh};
use crate::sparse::{Factor, NodeSparse};
use anyhow::{Result, bail};
use rayon::prelude::*;

const CHEB_DEGREE: usize = 3;
const CHEB_COARSE: usize = 2;
const COARSEST_NODES: usize = 3000;

/// Reusable work vectors: the V-cycle needs a handful of multi-megabyte vectors per call, and
/// allocating (and page-faulting) them fresh every iteration costs as much as a matrix product.
#[derive(Default)]
struct Pool(std::sync::Mutex<Vec<Vec<f64>>>);

impl Pool {
    /// A vector of length n with unspecified contents.
    fn take(&self, n: usize) -> Vec<f64> {
        let mut g = self.0.lock().unwrap_or_else(|e| e.into_inner());
        match g.iter().position(|v| v.len() == n) {
            Some(i) => g.swap_remove(i),
            None => vec![0.0; n],
        }
    }
    fn zeros(&self, n: usize) -> Vec<f64> {
        let mut v = self.take(n);
        v.par_iter_mut().for_each(|x| *x = 0.0);
        v
    }
    fn give(&self, v: Vec<f64>) {
        self.0.lock().unwrap_or_else(|e| e.into_inner()).push(v);
    }
}

trait Op: Sync {
    fn n(&self) -> usize;
    fn apply(&self, x: &[f64], y: &mut [f64]);
    fn dinv(&self) -> &[f64];
    fn lmax(&self) -> f64;
}

/// Interpolation from a coarse level (columns) to a fine level (rows), node based, with its
/// transpose for restriction.
struct Interp {
    ptr: Vec<usize>,
    ent: Vec<(u32, f64)>,
    tptr: Vec<usize>,
    tent: Vec<(u32, f64)>,
    nc: usize,
}

impl Interp {
    fn new(rows: Vec<Vec<(u32, f64)>>, nc: usize) -> Interp {
        let mut ptr = vec![0usize];
        let mut ent = Vec::new();
        let mut cnt = vec![0usize; nc + 1];
        for r in &rows {
            for &(c, w) in r {
                ent.push((c, w));
                cnt[c as usize + 1] += 1;
            }
            ptr.push(ent.len());
        }
        for i in 0..nc {
            cnt[i + 1] += cnt[i];
        }
        let tptr = cnt.clone();
        let mut fill = cnt;
        let mut tent = vec![(0u32, 0.0); ent.len()];
        for (i, r) in rows.iter().enumerate() {
            for &(c, w) in r {
                tent[fill[c as usize]] = (i as u32, w);
                fill[c as usize] += 1;
            }
        }
        Interp { ptr, ent, tptr, tent, nc }
    }
    /// x_f += P x_c (dofs where `mask` is true are left alone)
    fn prolong_add(&self, xc: &[f64], xf: &mut [f64], mask: Option<&[bool]>) {
        xf.par_chunks_mut(3).enumerate().for_each(|(i, xi)| {
            for c in 0..3 {
                if mask.is_some_and(|m| m[i * 3 + c]) {
                    continue;
                }
                let mut s = 0.0;
                for &(k, w) in &self.ent[self.ptr[i]..self.ptr[i + 1]] {
                    s += w * xc[k as usize * 3 + c];
                }
                xi[c] += s;
            }
        });
    }
    /// r_c = Pᵀ r_f (masked dofs excluded)
    fn restrict(&self, rf: &[f64], mask: Option<&[bool]>, pool: &Pool) -> Vec<f64> {
        let mut rc = pool.take(self.nc * 3);
        rc.par_chunks_mut(3).enumerate().for_each(|(k, rk)| {
            rk.fill(0.0);
            for &(i, w) in &self.tent[self.tptr[k]..self.tptr[k + 1]] {
                for c in 0..3 {
                    let d = i as usize * 3 + c;
                    if !mask.is_some_and(|m| m[d]) {
                        rk[c] += w * rf[d];
                    }
                }
            }
        });
        rc
    }
}

/// Packed index of (r, q) in the upper triangle of a symmetric 24×24 matrix.
#[inline]
fn sym(r: usize, q: usize) -> usize {
    let (i, j) = if r <= q { (r, q) } else { (q, r) };
    i * 24 - i * (i + 1) / 2 + j
}
const SYM24: usize = 300;

/// Fine-level operator: K applied element by element; identity on Dirichlet dofs.
pub struct FineOp<'a> {
    mesh: &'a Mesh,
    nptr: Vec<usize>,
    nel: Vec<(u32, u8)>,
    /// per element: E·h, the multiplier of its material's unit-cube matrix
    scale: Vec<f64>,
    /// per element: u32::MAX = plain cube, else index into `delta`
    gidx: Vec<u32>,
    units: Vec<Vec<f64>>,
    /// boundary-fitted elements: (K_general − h·U) for E = 1, symmetric-packed in f32. Fitted
    /// elements are mildly distorted cubes, so the rounding is ~1e-8 of their stiffness while the
    /// storage is a quarter of full f64 matrices.
    delta: Vec<f32>,
    pub fixed: Vec<bool>,
    dinv: Vec<f64>,
    lmax: f64,
    /// elements grouped by colour: no two elements of one colour share a node
    colours: Vec<Vec<u32>>,
}

/// Greedy element colouring so each colour can scatter K_e x_e in parallel without conflicts.
fn colour_elements(mesh: &Mesh) -> Vec<Vec<u32>> {
    let mut used = vec![0u64; mesh.nodes.len()];
    let mut colours: Vec<Vec<u32>> = Vec::new();
    for (e, en) in mesh.elems.iter().enumerate() {
        let taken = en.iter().fold(0u64, |m, &n| m | used[n as usize]);
        let c = (!taken).trailing_zeros() as usize;
        assert!(c < 64, "element colouring needs more than 64 colours");
        for &n in en {
            used[n as usize] |= 1 << c;
        }
        if colours.len() <= c {
            colours.resize(c + 1, Vec::new());
        }
        colours[c].push(e as u32);
    }
    colours
}

impl<'a> FineOp<'a> {
    pub fn new(mesh: &'a Mesh, units: &[ElemUnit], fixed: Vec<bool>) -> FineOp<'a> {
        let nn = mesh.nodes.len();
        let ne = mesh.elems.len();
        let h = mesh.h * 1e-3;
        let mut cnt = vec![0usize; nn + 1];
        for en in &mesh.elems {
            for &n in en {
                cnt[n as usize + 1] += 1;
            }
        }
        for i in 0..nn {
            cnt[i + 1] += cnt[i];
        }
        let nptr = cnt.clone();
        let mut fill = cnt;
        let mut nel = vec![(0u32, 0u8); nptr[nn]];
        for (e, en) in mesh.elems.iter().enumerate() {
            for (a, &n) in en.iter().enumerate() {
                nel[fill[n as usize]] = (e as u32, a as u8);
                fill[n as usize] += 1;
            }
        }
        let uk: Vec<Vec<f64>> = units.iter().map(|u| u.k.clone()).collect();
        let fitted: Vec<usize> = (0..ne).filter(|&e| mesh.is_fitted(e)).collect();
        let mut delta = vec![0.0f32; fitted.len() * SYM24];
        delta.par_chunks_mut(SYM24).zip(&fitted).for_each(|(d, &e)| {
            let k = hex_elastic_general(&elem_xyz_m(mesh, e), mesh.materials[mesh.elem_mat[e] as usize].poisson_ratio).k;
            let u = &uk[mesh.elem_mat[e] as usize];
            for r in 0..24 {
                for q in r..24 {
                    d[sym(r, q)] = (k[r * 24 + q] - h * u[r * 24 + q]) as f32;
                }
            }
        });
        let mut gidx = vec![u32::MAX; ne];
        for (g, &e) in fitted.iter().enumerate() {
            gidx[e] = g as u32;
        }
        let scale: Vec<f64> = (0..ne).map(|e| mesh.materials[mesh.elem_mat[e] as usize].youngs_modulus_gpa * 1e9 * h).collect();
        let colours = colour_elements(mesh);
        let mut op = FineOp { mesh, nptr, nel, scale, gidx, units: uk, delta, fixed, dinv: vec![], lmax: 1.0, colours };
        let mut diag = vec![0.0; nn * 3];
        diag.par_chunks_mut(3).enumerate().for_each(|(n, dn)| {
            for &(e, a) in &op.nel[op.nptr[n]..op.nptr[n + 1]] {
                for c in 0..3 {
                    let r = a as usize * 3 + c;
                    dn[c] += op.entry(e as usize, r, r);
                }
            }
        });
        op.dinv = diag.iter().zip(&op.fixed).map(|(d, f)| if *f || *d <= 0.0 { 1.0 } else { 1.0 / d }).collect();
        op.lmax = estimate_lmax(&op);
        op
    }

    /// Entry (r, q) of element e's stiffness [N/m].
    #[inline]
    fn entry(&self, e: usize, r: usize, q: usize) -> f64 {
        let u = self.units[self.mesh.elem_mat[e] as usize][r * 24 + q] * self.scale[e];
        match self.gidx[e] {
            u32::MAX => u,
            g => u + self.delta[g as usize * SYM24 + sym(r, q)] as f64 * self.scale[e] / (self.mesh.h * 1e-3),
        }
    }

    /// Full 24×24 stiffness of element e [N/m].
    pub fn elem_matrix(&self, e: usize) -> Vec<f64> {
        let mut k = vec![0.0; 576];
        for r in 0..24 {
            for q in 0..24 {
                k[r * 24 + q] = self.entry(e, r, q);
            }
        }
        k
    }

    /// y = K x on every row (no constraint handling): element by element, one colour at a time.
    pub fn apply_k(&self, x: &[f64], y: &mut [f64]) {
        let inv_h = 1.0 / (self.mesh.h * 1e-3);
        y.par_iter_mut().for_each(|v| *v = 0.0);
        let y_len = y.len();
        let yp = SyncPtr(y.as_mut_ptr());
        for col in &self.colours {
            col.par_iter().with_min_len(64).for_each(|&e| {
                let p = yp;
                let e = e as usize;
                let en = &self.mesh.elems[e];
                let mut xe = [0.0f64; 24];
                for b in 0..8 {
                    xe[b * 3..b * 3 + 3].copy_from_slice(&x[en[b] as usize * 3..en[b] as usize * 3 + 3]);
                }
                let k = &self.units[self.mesh.elem_mat[e] as usize];
                let s = self.scale[e];
                let mut ye = [0.0f64; 24];
                for (r, yr) in ye.iter_mut().enumerate() {
                    let row = &k[r * 24..r * 24 + 24];
                    *yr = s * row.iter().zip(&xe).map(|(a, b)| a * b).sum::<f64>();
                }
                let g = self.gidx[e];
                if g != u32::MAX {
                    let d = &self.delta[g as usize * SYM24..(g as usize + 1) * SYM24];
                    let sd = s * inv_h;
                    let mut i = 0;
                    for r in 0..24 {
                        // diagonal then the upper part of row r; mirror into the lower part
                        let v = d[i] as f64 * sd;
                        ye[r] += v * xe[r];
                        i += 1;
                        for q in r + 1..24 {
                            let v = d[i] as f64 * sd;
                            ye[r] += v * xe[q];
                            ye[q] += v * xe[r];
                            i += 1;
                        }
                    }
                }
                for b in 0..8 {
                    let base = en[b] as usize * 3;
                    debug_assert!(base + 2 < y_len);
                    for c in 0..3 {
                        // SAFETY: elements of one colour share no node, so writes are disjoint
                        unsafe { *p.0.add(base + c) += ye[b * 3 + c] };
                    }
                }
            });
        }
    }
}

impl Op for FineOp<'_> {
    fn n(&self) -> usize {
        self.mesh.nodes.len() * 3
    }
    /// Constrained operator for vectors that vanish on Dirichlet dofs: K x on free rows, x on
    /// fixed rows.
    fn apply(&self, x: &[f64], y: &mut [f64]) {
        self.apply_k(x, y);
        y.par_iter_mut().zip(&self.fixed).enumerate().for_each(|(i, (yi, f))| {
            if *f {
                *yi = x[i];
            }
        });
    }
    fn dinv(&self) -> &[f64] {
        &self.dinv
    }
    fn lmax(&self) -> f64 {
        self.lmax
    }
}

struct Level {
    a: NodeSparse,
    dinv: Vec<f64>,
    lmax: f64,
    /// lattice coordinates (in this level's spacing)
    lat: Vec<[u32; 3]>,
}

impl Op for Level {
    fn n(&self) -> usize {
        self.a.n()
    }
    fn apply(&self, x: &[f64], y: &mut [f64]) {
        self.a.matvec(x, y);
    }
    fn dinv(&self) -> &[f64] {
        &self.dinv
    }
    fn lmax(&self) -> f64 {
        self.lmax
    }
}

fn estimate_lmax(op: &dyn Op) -> f64 {
    let n = op.n();
    let dinv = op.dinv();
    let mut seed = 0x9E3779B97F4A7C15u64;
    let mut v: Vec<f64> = (0..n)
        .map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f64 / (1u64 << 53) as f64 + 0.1
        })
        .collect();
    let mut w = vec![0.0; n];
    let mut lam = 1.0;
    for _ in 0..12 {
        let nv = v.par_iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-300);
        v.par_iter_mut().for_each(|x| *x /= nv);
        op.apply(&v, &mut w);
        w.par_iter_mut().zip(dinv).for_each(|(x, d)| *x *= d);
        lam = w.par_iter().zip(&v).map(|(a, b)| a * b).sum::<f64>();
        std::mem::swap(&mut v, &mut w);
    }
    lam.max(1e-12) * 1.1
}

/// Chebyshev smoothing of A x = b on [λmax/20, λmax] (Jacobi-scaled). `zero` = x starts at 0.
fn chebyshev(op: &dyn Op, b: &[f64], x: &mut [f64], zero: bool, degree: usize, pool: &Pool) {
    let n = op.n();
    let dinv = op.dinv();
    let hi = op.lmax();
    let lo = hi / 20.0;
    let theta = 0.5 * (hi + lo);
    let delta = 0.5 * (hi - lo);
    let mut r = pool.take(n);
    r.par_iter_mut().zip(b).for_each(|(r, b)| *r = *b);
    let mut t = pool.take(n);
    if !zero {
        op.apply(x, &mut t);
        r.par_iter_mut().zip(&t).for_each(|(r, t)| *r -= t);
    }
    let sigma = theta / delta;
    let mut rho = 1.0 / sigma;
    let mut d = pool.take(n);
    d.par_iter_mut().zip(&r).zip(dinv).for_each(|((d, r), di)| *d = r * di / theta);
    for k in 0..degree {
        x.par_iter_mut().zip(&d).for_each(|(x, d)| *x += d);
        if k + 1 == degree {
            break;
        }
        op.apply(&d, &mut t);
        r.par_iter_mut().zip(&t).for_each(|(r, t)| *r -= t);
        let rho1 = 1.0 / (2.0 * sigma - rho);
        let (c1, c2) = (rho1 * rho, 2.0 * rho1 / delta);
        d.par_iter_mut().zip(&r).zip(dinv).for_each(|((d, r), di)| *d = c1 * *d + c2 * r * di);
        rho = rho1;
    }
    pool.give(r);
    pool.give(t);
    pool.give(d);
}

/// Unsafe shared write access for scatters whose rows are provably disjoint (parity colouring).
#[derive(Clone, Copy)]
struct SyncPtr(*mut f64);
unsafe impl Sync for SyncPtr {}
unsafe impl Send for SyncPtr {}

/// Add a 3×3 block to (row node `i`, column node `j`) of a block-sparse matrix.
#[inline]
fn block_index(a: &NodeSparse, i: usize, j: u32) -> Option<(usize, usize)> {
    let nb = &a.adj[a.ptr[i]..a.ptr[i + 1]];
    let k = nb.binary_search(&j).ok()?;
    Some((a.ptr[i] * 9 + k * 3, nb.len() * 3))
}

fn build_matrix(rows: Vec<Vec<u32>>) -> NodeSparse {
    let mut ptr = vec![0usize];
    let mut adj = Vec::new();
    for (i, mut r) in rows.into_iter().enumerate() {
        r.push(i as u32);
        r.sort_unstable();
        r.dedup();
        adj.extend_from_slice(&r);
        ptr.push(adj.len());
    }
    let vals = vec![0.0; adj.len() * 9];
    NodeSparse { bs: 3, ptr, adj, vals, hang: None }
}

/// Put 1 on the diagonal of dofs that no fine dof reaches (all-constrained coarse nodes).
fn fix_empty_rows(a: &mut NodeSparse) {
    let d = a.diag();
    let scale = d.iter().cloned().fold(0.0f64, f64::max).max(1e-300);
    for (dof, v) in d.iter().enumerate() {
        if *v <= 1e-9 * scale {
            let (n, c) = (dof / 3, dof % 3);
            if let Some((base, stride)) = block_index(a, n, n as u32) {
                a.vals[base + c * stride + c] = scale * 1e-3;
            }
        }
    }
}

fn make_level(a: NodeSparse, lat: Vec<[u32; 3]>) -> Level {
    let d = a.diag();
    let dinv = d.iter().map(|v| if *v > 0.0 { 1.0 / v } else { 1.0 }).collect();
    let mut l = Level { a, dinv, lmax: 1.0, lat };
    l.lmax = estimate_lmax(&l);
    l
}

pub struct ElasticMg<'a> {
    pub fine: FineOp<'a>,
    p0: Interp,
    levels: Vec<Level>,
    /// interpolation from level k+1 to level k (k ≥ 1 counted in `levels`)
    ps: Vec<Interp>,
    coarse: Factor,
    pool: Pool,
}

/// Trilinear weights of the 8 corners of a cell for local coordinates t ∈ [0,1]³.
fn tri_weights(t: [f64; 3]) -> [f64; 8] {
    let mut w = [0.0; 8];
    for (a, c) in CORNER.iter().enumerate() {
        w[a] = (0..3).map(|k| if c[k] == 1 { t[k] } else { 1.0 - t[k] }).product();
    }
    w
}

impl<'a> ElasticMg<'a> {
    pub fn new(mesh: &'a Mesh, units: &[ElemUnit], fixed: Vec<bool>) -> Result<ElasticMg<'a>> {
        let prof = std::env::var_os("SIMCRAFT_PROFILE").is_some();
        let t0 = std::time::Instant::now();
        let tick = |what: &str| {
            if prof {
                eprintln!("[mg] {what}: {:.2} s", t0.elapsed().as_secs_f64());
            }
        };
        let fine = FineOp::new(mesh, units, fixed);
        tick("fine operator");
        let nn = mesh.nodes.len();
        let h = mesh.h;
        // ---------------- level 1 nodes: lattice points of spacing 2h
        let ncell = mesh.dims.map(|d| d.div_ceil(2).max(1));
        let side = [ncell[0] + 1, ncell[1] + 1, ncell[2] + 1];
        let lid = |c: [usize; 3]| c[0] + side[0] * (c[1] + side[1] * c[2]);
        let node_cell = |p: [f64; 3]| -> ([usize; 3], [f64; 3]) {
            let mut cell = [0usize; 3];
            let mut t = [0.0; 3];
            for k in 0..3 {
                let x = (p[k] - mesh.origin[k]) / (2.0 * h);
                let c = (x.floor().max(0.0) as usize).min(ncell[k] - 1);
                cell[k] = c;
                t[k] = (x - c as f64).clamp(0.0, 1.0);
            }
            (cell, t)
        };
        let mut map = vec![u32::MAX; side[0] * side[1] * side[2]];
        let mut lat1: Vec<[u32; 3]> = Vec::new();
        let mut rows0: Vec<Vec<(u32, f64)>> = Vec::with_capacity(nn);
        for p in &mesh.nodes {
            let (cell, t) = node_cell(*p);
            let w = tri_weights(t);
            let mut r = Vec::with_capacity(8);
            for (a, c) in CORNER.iter().enumerate() {
                if w[a] > 1e-12 {
                    let q = [cell[0] + c[0], cell[1] + c[1], cell[2] + c[2]];
                    let id = &mut map[lid(q)];
                    if *id == u32::MAX {
                        *id = lat1.len() as u32;
                        lat1.push([q[0] as u32, q[1] as u32, q[2] as u32]);
                    }
                    r.push((*id, w[a]));
                }
            }
            rows0.push(r);
        }
        let n1 = lat1.len();
        let p0 = Interp::new(rows0, n1);
        tick("interpolation");

        // ---------------- level 1 operator, element by element
        // simple elements: unit cubes without constrained dofs, grouped by coarse cell
        let ne = mesh.elems.len();
        let simple: Vec<bool> = (0..ne).map(|e| !mesh.is_fitted(e) && mesh.elems[e].iter().all(|&n| (0..3).all(|c| !fine.fixed[n as usize * 3 + c]))).collect();
        let ccell = |e: usize| -> usize {
            let c = mesh.elem_cell[e];
            (c[0] / 2) as usize + ncell[0] * ((c[1] / 2) as usize + ncell[1] * (c[2] / 2) as usize)
        };
        let mut by_cell: Vec<(usize, u32)> = (0..ne).filter(|&e| simple[e]).map(|e| (ccell(e), e as u32)).collect();
        by_cell.par_sort_unstable();
        // coarse corner ids of a coarse cell
        let corners = |ci: usize| -> [u32; 8] {
            let cx = ci % ncell[0];
            let cy = (ci / ncell[0]) % ncell[1];
            let cz = ci / (ncell[0] * ncell[1]);
            CORNER.map(|c| map[lid([cx + c[0], cy + c[1], cz + c[2]])])
        };
        // octant matrices Pₒᵀ U Pₒ per material (24×24 over the coarse cell corners)
        let oct: Vec<Vec<Vec<f64>>> = fine
            .units
            .iter()
            .map(|u| {
                (0..8)
                    .map(|o| {
                        let oc = CORNER[o];
                        let mut p = [[0.0; 8]; 8]; // p[fine local a][coarse corner A]
                        for (a, c) in CORNER.iter().enumerate() {
                            let t = [(oc[0] + c[0]) as f64 / 2.0, (oc[1] + c[1]) as f64 / 2.0, (oc[2] + c[2]) as f64 / 2.0];
                            p[a] = tri_weights(t);
                        }
                        // M[A c][B d] = Σ_{a,b} p[a][A] U[a c][b d] p[b][B]
                        let mut up = vec![0.0; 24 * 24]; // U P: [a c][B d]
                        for r in 0..24 {
                            for bb in 0..8 {
                                for d in 0..3 {
                                    let mut s = 0.0;
                                    for b in 0..8 {
                                        s += u[r * 24 + b * 3 + d] * p[b][bb];
                                    }
                                    up[r * 24 + bb * 3 + d] = s;
                                }
                            }
                        }
                        let mut m = vec![0.0; 24 * 24];
                        for aa in 0..8 {
                            for c in 0..3 {
                                for col in 0..24 {
                                    let mut s = 0.0;
                                    for a in 0..8 {
                                        s += p[a][aa] * up[(a * 3 + c) * 24 + col];
                                    }
                                    m[(aa * 3 + c) * 24 + col] = s;
                                }
                            }
                        }
                        m
                    })
                    .collect()
            })
            .collect();
        // groups of simple elements per coarse cell
        let mut groups: Vec<(usize, usize, usize)> = Vec::new(); // (cell, start, end)
        let mut s0 = 0;
        for i in 1..=by_cell.len() {
            if i == by_cell.len() || by_cell[i].0 != by_cell[s0].0 {
                groups.push((by_cell[s0].0, s0, i));
                s0 = i;
            }
        }
        // complex elements: explicit P_eᵀ K_e P_e with constrained dofs removed
        let complex: Vec<usize> = (0..ne).filter(|&e| !simple[e]).collect();
        let complex_blocks: Vec<(Vec<u32>, Vec<f64>)> = complex
            .par_iter()
            .map(|&e| {
                let en = &mesh.elems[e];
                let k = fine.elem_matrix(e);
                let mut ids: Vec<u32> = en.iter().flat_map(|&n| p0.ent[p0.ptr[n as usize]..p0.ptr[n as usize + 1]].iter().map(|x| x.0)).collect();
                ids.sort_unstable();
                ids.dedup();
                let m = ids.len();
                // P_e: 24 × 3m, masked on constrained fine dofs
                let mut pe = vec![0.0; 24 * 3 * m];
                for (a, &n) in en.iter().enumerate() {
                    for &(cid, w) in &p0.ent[p0.ptr[n as usize]..p0.ptr[n as usize + 1]] {
                        let j = ids.binary_search(&cid).unwrap();
                        for c in 0..3 {
                            if !fine.fixed[n as usize * 3 + c] {
                                pe[(a * 3 + c) * 3 * m + j * 3 + c] = w;
                            }
                        }
                    }
                }
                let mm = 3 * m;
                let mut kp = vec![0.0; 24 * mm];
                for r in 0..24 {
                    for q in 0..24 {
                        let kv = k[r * 24 + q];
                        if kv != 0.0 {
                            for col in 0..mm {
                                kp[r * mm + col] += kv * pe[q * mm + col];
                            }
                        }
                    }
                }
                let mut out = vec![0.0; mm * mm];
                for r in 0..24 {
                    for i in 0..mm {
                        let pv = pe[r * mm + i];
                        if pv != 0.0 {
                            for j in 0..mm {
                                out[i * mm + j] += pv * kp[r * mm + j];
                            }
                        }
                    }
                }
                (ids, out)
            })
            .collect();
        tick(&format!("{} complex element blocks", complex.len()));
        // sparsity pattern
        let mut rows: Vec<Vec<u32>> = vec![Vec::new(); n1];
        for &(ci, _, _) in &groups {
            let cn = corners(ci);
            for &a in &cn {
                if a != u32::MAX {
                    rows[a as usize].extend(cn.iter().filter(|&&b| b != u32::MAX));
                }
            }
        }
        for (ids, _) in &complex_blocks {
            for &a in ids {
                rows[a as usize].extend_from_slice(ids);
            }
        }
        let mut a1 = build_matrix(rows);
        tick("level-1 pattern");
        // scatter simple cells by parity colour (cells of one colour share no corner)
        let ptr = SyncPtr(a1.vals.as_mut_ptr());
        for colour in 0..8usize {
            let a1r = &a1;
            groups.par_iter().filter(|g| {
                let ci = g.0;
                let (cx, cy, cz) = (ci % ncell[0], (ci / ncell[0]) % ncell[1], ci / (ncell[0] * ncell[1]));
                (cx % 2) | ((cy % 2) << 1) | ((cz % 2) << 2) == colour
            }).for_each(|&(ci, s, t)| {
                let p = ptr;
                let mut m = [0.0f64; 576];
                for &(_, e) in &by_cell[s..t] {
                    let e = e as usize;
                    let c = mesh.elem_cell[e];
                    let o = ((c[0] % 2) + 2 * (c[1] % 2) + 4 * (c[2] % 2)) as usize;
                    // octant index in CORNER order
                    let o = CORNER.iter().position(|q| q[0] == o % 2 && q[1] == (o / 2) % 2 && q[2] == o / 4).unwrap();
                    let u = &oct[mesh.elem_mat[e] as usize][o];
                    let sc = fine.scale[e];
                    for (mv, uv) in m.iter_mut().zip(u) {
                        *mv += sc * uv;
                    }
                }
                let cn = corners(ci);
                for (aa, &ra) in cn.iter().enumerate() {
                    if ra == u32::MAX {
                        continue;
                    }
                    for (bb, &rb) in cn.iter().enumerate() {
                        if rb == u32::MAX {
                            continue;
                        }
                        let (base, stride) = block_index(a1r, ra as usize, rb).unwrap();
                        debug_assert!(base + 2 * stride + 2 < a1r.vals.len());
                        for c in 0..3 {
                            for d in 0..3 {
                                // SAFETY: rows of one colour's cells are disjoint
                                unsafe { *p.0.add(base + c * stride + d) += m[(aa * 3 + c) * 24 + bb * 3 + d] };
                            }
                        }
                    }
                }
            });
        }
        for (ids, out) in &complex_blocks {
            let mm = ids.len() * 3;
            for (i, &ra) in ids.iter().enumerate() {
                for (j, &rb) in ids.iter().enumerate() {
                    let (base, stride) = block_index(&a1, ra as usize, rb).unwrap();
                    for c in 0..3 {
                        for d in 0..3 {
                            a1.vals[base + c * stride + d] += out[(i * 3 + c) * mm + j * 3 + d];
                        }
                    }
                }
            }
        }
        fix_empty_rows(&mut a1);
        tick("level-1 values");
        let mut levels = vec![make_level(a1, lat1)];
        tick("level-1 smoother");
        let mut ps = Vec::new();
        // ---------------- further levels: lattice coarsening + sparse triple products
        loop {
            let cur = levels.last().unwrap();
            let nf = cur.lat.len();
            if nf <= COARSEST_NODES || levels.len() >= 10 {
                break;
            }
            let mx = cur.lat.iter().fold([0u32; 3], |m, l| [m[0].max(l[0]), m[1].max(l[1]), m[2].max(l[2])]);
            let side = [(mx[0] / 2 + 2) as usize, (mx[1] / 2 + 2) as usize, (mx[2] / 2 + 2) as usize];
            let lid = |c: [u32; 3]| c[0] as usize + side[0] * (c[1] as usize + side[1] * c[2] as usize);
            let mut map = vec![u32::MAX; side[0] * side[1] * side[2]];
            let mut lat: Vec<[u32; 3]> = Vec::new();
            let mut prow: Vec<Vec<(u32, f64)>> = Vec::with_capacity(nf);
            for l in &cur.lat {
                let opts: Vec<Vec<(u32, f64)>> = (0..3).map(|k| if l[k] % 2 == 0 { vec![(l[k] / 2, 1.0)] } else { vec![(l[k] / 2, 0.5), (l[k] / 2 + 1, 0.5)] }).collect();
                let mut r = Vec::with_capacity(8);
                for &(x, wx) in &opts[0] {
                    for &(y, wy) in &opts[1] {
                        for &(z, wz) in &opts[2] {
                            let id = &mut map[lid([x, y, z])];
                            if *id == u32::MAX {
                                *id = lat.len() as u32;
                                lat.push([x, y, z]);
                            }
                            r.push((*id, wx * wy * wz));
                        }
                    }
                }
                prow.push(r);
            }
            let nc = lat.len();
            if nc as f64 > 0.6 * nf as f64 {
                break;
            }
            let p = Interp::new(prow, nc);
            // A_c = Pᵀ A P, row by row in parallel
            let a = &cur.a;
            let rows: Vec<Vec<(u32, [f64; 9])>> = (0..nc)
                .into_par_iter()
                .map(|ic| {
                    let mut acc: Vec<(u32, [f64; 9])> = Vec::with_capacity(32);
                    for &(i, wi) in &p.tent[p.tptr[ic]..p.tptr[ic + 1]] {
                        let i = i as usize;
                        let nb = &a.adj[a.ptr[i]..a.ptr[i + 1]];
                        let deg = nb.len();
                        let base = a.ptr[i] * 9;
                        for (k, &j) in nb.iter().enumerate() {
                            let mut blk = [0.0; 9];
                            for c in 0..3 {
                                for d in 0..3 {
                                    blk[c * 3 + d] = a.vals[base + c * deg * 3 + k * 3 + d];
                                }
                            }
                            for &(jc, wj) in &p.ent[p.ptr[j as usize]..p.ptr[j as usize + 1]] {
                                let w = wi * wj;
                                let slot = match acc.iter().position(|x| x.0 == jc) {
                                    Some(s) => s,
                                    None => {
                                        acc.push((jc, [0.0; 9]));
                                        acc.len() - 1
                                    }
                                };
                                for q in 0..9 {
                                    acc[slot].1[q] += w * blk[q];
                                }
                            }
                        }
                    }
                    acc.sort_unstable_by_key(|x| x.0);
                    acc
                })
                .collect();
            let mut ptr = vec![0usize];
            let mut adj = Vec::new();
            for r in &rows {
                adj.extend(r.iter().map(|x| x.0));
                ptr.push(adj.len());
            }
            let mut vals = vec![0.0; adj.len() * 9];
            for (ic, r) in rows.iter().enumerate() {
                let deg = r.len();
                let base = ptr[ic] * 9;
                for (k, (_, blk)) in r.iter().enumerate() {
                    for c in 0..3 {
                        for d in 0..3 {
                            vals[base + c * deg * 3 + k * 3 + d] = blk[c * 3 + d];
                        }
                    }
                }
            }
            let mut ac = NodeSparse { bs: 3, ptr, adj, vals, hang: None };
            fix_empty_rows(&mut ac);
            ps.push(p);
            levels.push(make_level(ac, lat));
            tick(&format!("level {} ({} nodes)", levels.len(), levels.last().unwrap().lat.len()));
        }
        // coarse operators can be singular where coarse nodes outnumber the free fine dofs near
        // supports (those combinations prolongate to zero): a tiny shift keeps the factor SPD
        let mut last = levels.last().unwrap().a.clone();
        for (dof, d) in last.diag().iter().enumerate() {
            let (n, c) = (dof / 3, dof % 3);
            if let Some((base, stride)) = block_index(&last, n, n as u32) {
                last.vals[base + c * stride + c] += 1e-8 * d.abs();
            }
        }
        let coarse = match Factor::new(&last, usize::MAX) {
            Some(f) => f?,
            None => bail!("multigrid: coarsest level too large"),
        };
        tick("coarse factor");
        Ok(ElasticMg { fine, p0, levels, ps, coarse, pool: Pool::default() })
    }

    pub fn levels(&self) -> usize {
        self.levels.len() + 1
    }

    fn vcycle_level(&self, l: usize, b: &[f64]) -> Vec<f64> {
        let pool = &self.pool;
        if l + 1 == self.levels.len() {
            return self.coarse.solve(b);
        }
        let lev = &self.levels[l];
        let n = b.len();
        let mut x = pool.zeros(n);
        chebyshev(lev, b, &mut x, true, CHEB_COARSE, pool);
        let mut r = pool.take(n);
        lev.apply(&x, &mut r);
        r.par_iter_mut().zip(b).for_each(|(r, b)| *r = b - *r);
        let p = &self.ps[l];
        let rc = p.restrict(&r, None, pool);
        pool.give(r);
        let ec = self.vcycle_level(l + 1, &rc);
        pool.give(rc);
        p.prolong_add(&ec, &mut x, None);
        pool.give(ec);
        chebyshev(lev, b, &mut x, false, CHEB_COARSE, pool);
        x
    }

    /// One V-cycle from zero for the fine level, into `x`.
    fn precondition_into(&self, b: &[f64], x: &mut [f64]) {
        let pool = &self.pool;
        let n = b.len();
        let mask = Some(&self.fine.fixed[..]);
        x.par_iter_mut().for_each(|v| *v = 0.0);
        chebyshev(&self.fine, b, x, true, CHEB_DEGREE, pool);
        let mut r = pool.take(n);
        self.fine.apply(x, &mut r);
        r.par_iter_mut().zip(b).for_each(|(r, b)| *r = b - *r);
        let rc = self.p0.restrict(&r, mask, pool);
        pool.give(r);
        let ec = self.vcycle_level(0, &rc);
        pool.give(rc);
        self.p0.prolong_add(&ec, x, mask);
        pool.give(ec);
        chebyshev(&self.fine, b, x, false, CHEB_DEGREE, pool);
    }

    fn precondition(&self, b: &[f64]) -> Vec<f64> {
        let mut x = vec![0.0; b.len()];
        self.precondition_into(b, &mut x);
        x
    }

    /// Solve K u = f with u prescribed on the constrained dofs (`ug` holds those values, zero
    /// elsewhere). Returns CG iterations; `u` receives the solution.
    pub fn solve(&self, f: &[f64], ug: &[f64], u: &mut [f64], tol: f64, max_it: usize, progress: &(dyn Fn(f64) -> Result<()> + Sync)) -> Result<usize> {
        let n = f.len();
        let fixed = &self.fine.fixed;
        // move prescribed displacements to the right-hand side
        let mut kg = vec![0.0; n];
        self.fine.apply_k(ug, &mut kg);
        let mut r: Vec<f64> = (0..n).map(|i| if fixed[i] { 0.0 } else { f[i] - kg[i] }).collect();
        let mut x = vec![0.0; n]; // free part of the solution
        if std::env::var_os("SIMCRAFT_PROFILE").is_some() {
            let mut y = vec![0.0; n];
            let t = std::time::Instant::now();
            for _ in 0..5 {
                self.fine.apply(&r, &mut y);
            }
            eprintln!("[mg] fine apply {:.1} ms", t.elapsed().as_secs_f64() * 200.0);
            for (l, lev) in self.levels.iter().enumerate() {
                let v = vec![1.0; lev.n()];
                let mut w = vec![0.0; lev.n()];
                let t = std::time::Instant::now();
                for _ in 0..5 {
                    lev.apply(&v, &mut w);
                }
                eprintln!("[mg] level {} matvec {:.1} ms ({} nodes, {} blocks)", l + 1, t.elapsed().as_secs_f64() * 200.0, lev.lat.len(), lev.a.adj.len());
            }
            let t = std::time::Instant::now();
            let _ = self.precondition(&r);
            eprintln!("[mg] V-cycle {:.1} ms", t.elapsed().as_secs_f64() * 1000.0);
            let t = std::time::Instant::now();
            let _ = self.p0.restrict(&r, Some(&self.fine.fixed), &self.pool);
            eprintln!("[mg] restrict {:.1} ms", t.elapsed().as_secs_f64() * 1000.0);
        }
        let dot = |a: &[f64], b: &[f64]| a.par_iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
        let r0 = dot(&r, &r).sqrt();
        let mut its = 0;
        if r0 > 0.0 {
            let mut z = self.precondition(&r);
            let mut p = z.clone();
            let mut rz = dot(&r, &z);
            let mut ap = vec![0.0; n];
            let (lr0, ltol) = (1.0f64.ln(), tol.ln());
            loop {
                its += 1;
                self.fine.apply(&p, &mut ap);
                let pap = dot(&p, &ap);
                if !(pap > 0.0) {
                    bail!("solver broke down (matrix not positive definite): the model is probably under-constrained. Add supports so it cannot move freely.");
                }
                let al = rz / pap;
                x.par_iter_mut().zip(&p).for_each(|(x, p)| *x += al * p);
                r.par_iter_mut().zip(&ap).for_each(|(r, q)| *r -= al * q);
                let rn = dot(&r, &r).sqrt() / r0;
                if !rn.is_finite() {
                    bail!("multigrid CG diverged");
                }
                if its % 2 == 0 {
                    progress(((lr0 - rn.max(1e-300).ln()) / (lr0 - ltol)).clamp(0.0, 1.0))?;
                }
                if rn < tol {
                    break;
                }
                if its >= max_it {
                    bail!("multigrid CG did not converge in {max_it} iterations (residual {rn:.1e})");
                }
                self.precondition_into(&r, &mut z);
                let rz2 = dot(&r, &z);
                let beta = rz2 / rz;
                rz = rz2;
                p.par_iter_mut().zip(&z).for_each(|(p, z)| *p = z + beta * *p);
            }
        }
        for i in 0..n {
            u[i] = if fixed[i] { ug[i] } else { x[i] };
        }
        Ok(its)
    }
}
