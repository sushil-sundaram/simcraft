//! Aggregation multigrid for the CFD pressure Poisson equation on the voxel grid.
//!
//! Levels are built by merging 2×2×2 blocks of cells (any mix of fluid cells, so solids and
//! irregular domains are fine); coarse operators are Galerkin products Pᵀ A P with piecewise-
//! constant prolongation. One symmetric V-cycle (damped-Jacobi pre/post smoothing, direct solve on
//! the coarsest level) preconditions conjugate gradients, which typically converges in 10–25
//! iterations instead of hundreds for diagonal preconditioning.

use crate::fem::check_cancel;
use crate::sparse::{Factor, NodeSparse};
use anyhow::{Result, anyhow, bail};
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::atomic::AtomicBool;

const CH: usize = 8192;

/// Apply `f(i, &mut out[i])` over all entries: chunked in parallel for large vectors, serial
/// otherwise (these loops are memory-bound and tiny; fine-grained tasks cost more than they save).
fn each(out: &mut [f64], f: impl Fn(usize, &mut f64) + Sync) {
    if out.len() < 4 * CH {
        out.iter_mut().enumerate().for_each(|(i, o)| f(i, o));
    } else {
        out.par_chunks_mut(CH).enumerate().for_each(|(c, ch)| {
            let base = c * CH;
            ch.iter_mut().enumerate().for_each(|(k, o)| f(base + k, o));
        });
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    if a.len() < 4 * CH {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    } else {
        a.par_chunks(CH).zip(b.par_chunks(CH)).map(|(x, y)| x.iter().zip(y).map(|(p, q)| p * q).sum::<f64>()).sum()
    }
}

struct Level {
    ptr: Vec<usize>,
    col: Vec<u32>,
    val: Vec<f64>,
    dinv: Vec<f64>,
    /// fine cell → aggregate on the next level
    agg: Vec<u32>,
}

impl Level {
    fn n(&self) -> usize {
        self.dinv.len()
    }
    #[inline]
    fn row(&self, i: usize, x: &[f64]) -> f64 {
        let mut s = 0.0;
        for k in self.ptr[i]..self.ptr[i + 1] {
            s += self.val[k] * x[self.col[k] as usize];
        }
        s
    }
    fn matvec(&self, x: &[f64], y: &mut [f64]) {
        each(y, |i, yi| *yi = self.row(i, x));
    }
    /// One damped-Jacobi sweep x ← x + ω D⁻¹ (b − A x) (Jacobi needs the old x: write to `tmp`).
    fn jacobi(&self, x: &mut Vec<f64>, b: &[f64], w: f64, tmp: &mut Vec<f64>) {
        {
            let xr = &*x;
            each(tmp, |i, t| *t = xr[i] + w * self.dinv[i] * (b[i] - self.row(i, xr)));
        }
        std::mem::swap(x, tmp);
    }
}

pub struct Multigrid {
    levels: Vec<Level>,
    coarse: Factor,
    coarse_a: NodeSparse,
    /// over-correction of the coarse-grid update (unsmoothed aggregation under-weights it)
    alpha: f64,
    omega: f64,
}

const SMOOTH: usize = 2;

impl Multigrid {
    /// `a`: symmetric positive definite matrix (bs = 1) of the cells; `coords`: grid cell index of
    /// each row.
    pub fn new(a: &NodeSparse, coords: &[[u32; 3]]) -> Result<Multigrid> {
        if a.bs != 1 {
            bail!("multigrid needs a scalar matrix");
        }
        let n0 = a.ptr.len() - 1;
        let mut levels = Vec::new();
        let (mut ptr, mut col, mut val) = (a.ptr.clone(), a.adj.clone(), a.vals.clone());
        let mut xyz: Vec<[u32; 3]> = coords.to_vec();
        let mut n = n0;
        loop {
            let dinv: Vec<f64> = (0..n)
                .map(|i| {
                    let d = (ptr[i]..ptr[i + 1]).find(|&k| col[k] as usize == i).map(|k| val[k]).unwrap_or(0.0);
                    if d > 0.0 { 1.0 / d } else { 1.0 }
                })
                .collect();
            if n <= 400 || levels.len() >= 12 {
                // coarsest level: direct factorisation
                let mut lists_ptr = vec![0usize];
                let mut adj = Vec::with_capacity(col.len());
                let mut vals = Vec::with_capacity(col.len());
                for i in 0..n {
                    let mut row: Vec<(u32, f64)> = (ptr[i]..ptr[i + 1]).map(|k| (col[k], val[k])).collect();
                    row.sort_unstable_by_key(|e| e.0);
                    for (c, v) in row {
                        adj.push(c);
                        vals.push(v);
                    }
                    lists_ptr.push(adj.len());
                }
                let coarse_a = NodeSparse { bs: 1, ptr: lists_ptr, adj, vals, hang: None, condensed: false };
                let coarse = Factor::new(&coarse_a, usize::MAX).ok_or_else(|| anyhow!("coarse grid too large"))??;
                levels.push(Level { ptr, col, val, dinv, agg: vec![] });
                return Ok(Multigrid { levels, coarse, coarse_a, alpha: 1.6, omega: 0.7 });
            }
            // aggregate 2×2×2 blocks
            let mut map: HashMap<[u32; 3], u32> = HashMap::with_capacity(n / 6);
            let mut agg = vec![0u32; n];
            let mut cxyz: Vec<[u32; 3]> = Vec::with_capacity(n / 6);
            for i in 0..n {
                let k = [xyz[i][0] / 2, xyz[i][1] / 2, xyz[i][2] / 2];
                let len = map.len() as u32;
                let id = *map.entry(k).or_insert_with(|| {
                    cxyz.push(k);
                    len
                });
                agg[i] = id;
            }
            let nc = cxyz.len();
            if nc as f64 > 0.7 * n as f64 {
                // coarsening stalled (thin geometry): stop here with a direct solve one level up
                levels.push(Level { ptr: ptr.clone(), col: col.clone(), val: val.clone(), dinv, agg: vec![] });
                let coarse_a = NodeSparse { bs: 1, ptr: ptr.clone(), adj: col.clone(), vals: val.clone(), hang: None, condensed: false };
                let coarse = Factor::new(&coarse_a, usize::MAX).ok_or_else(|| anyhow!("coarse grid too large"))??;
                return Ok(Multigrid { levels, coarse, coarse_a, alpha: 1.6, omega: 0.7 });
            }
            // Galerkin coarse operator Pᵀ A P
            let mut rows: Vec<HashMap<u32, f64>> = vec![HashMap::new(); nc];
            for i in 0..n {
                let ci = agg[i] as usize;
                for k in ptr[i]..ptr[i + 1] {
                    *rows[ci].entry(agg[col[k] as usize]).or_insert(0.0) += val[k];
                }
            }
            let mut cptr = vec![0usize];
            let mut ccol = Vec::new();
            let mut cval = Vec::new();
            for row in rows {
                let mut r: Vec<(u32, f64)> = row.into_iter().filter(|e| e.1 != 0.0).collect();
                r.sort_unstable_by_key(|e| e.0);
                for (c, v) in r {
                    ccol.push(c);
                    cval.push(v);
                }
                cptr.push(ccol.len());
            }
            levels.push(Level { ptr, col, val, dinv, agg });
            ptr = cptr;
            col = ccol;
            val = cval;
            xyz = cxyz;
            n = nc;
        }
    }

    pub fn levels(&self) -> usize {
        self.levels.len()
    }

    /// x ≈ A⁻¹ b by one V-cycle from zero.
    fn vcycle(&self, l: usize, b: &[f64]) -> Vec<f64> {
        let lev = &self.levels[l];
        if l + 1 == self.levels.len() {
            return self.coarse.solve(b);
        }
        let n = lev.n();
        let w = self.omega;
        let mut x = vec![0.0; n];
        each(&mut x, |i, xi| *xi = w * b[i] * lev.dinv[i]);
        let mut tmp = vec![0.0; n];
        for _ in 1..SMOOTH {
            lev.jacobi(&mut x, b, w, &mut tmp);
        }
        // restrict the residual
        let nc = self.levels[l + 1].n();
        let mut rc = vec![0.0; nc];
        for i in 0..n {
            rc[lev.agg[i] as usize] += b[i] - lev.row(i, &x);
        }
        let ec = self.vcycle(l + 1, &rc);
        let a = self.alpha;
        each(&mut x, |i, xi| *xi += a * ec[lev.agg[i] as usize]);
        for _ in 0..SMOOTH {
            lev.jacobi(&mut x, b, w, &mut tmp);
        }
        x
    }

    /// One V-cycle from zero: an approximate inverse, for use as a preconditioner.
    pub fn precondition(&self, r: &[f64]) -> Vec<f64> {
        self.vcycle(0, r)
    }

    /// Multigrid-preconditioned CG; `x` is the initial guess. Converges to `tol` relative to the
    /// initial residual. Returns iterations.
    pub fn solve(&self, b: &[f64], x: &mut [f64], tol: f64, max_it: usize) -> Result<usize> {
        self.solve_cancellable(b, x, tol, max_it, &crate::fem::NEVER_CANCEL)
    }

    pub fn solve_cancellable(&self, b: &[f64], x: &mut [f64], tol: f64, max_it: usize, cancel: &AtomicBool) -> Result<usize> {
        let lev = &self.levels[0];
        let n = lev.n();
        let mut r = vec![0.0; n];
        {
            let xr = &*x;
            each(&mut r, |i, ri| *ri = b[i] - lev.row(i, xr));
        }
        let r0 = dot(&r, &r).sqrt();
        let bn = dot(b, b).sqrt();
        if r0 == 0.0 || r0 < 1e-14 * bn {
            return Ok(0);
        }
        let mut z = self.vcycle(0, &r);
        let mut p = z.clone();
        let mut rz = dot(&r, &z);
        let mut ap = vec![0.0; n];
        for it in 1..=max_it {
            if it % 4 == 0 {
                check_cancel(cancel)?;
            }
            lev.matvec(&p, &mut ap);
            let alpha = rz / dot(&p, &ap);
            each(x, |i, xi| *xi += alpha * p[i]);
            each(&mut r, |i, ri| *ri -= alpha * ap[i]);
            let rn = dot(&r, &r).sqrt();
            if !rn.is_finite() {
                bail!("multigrid CG diverged");
            }
            if rn < tol * r0 {
                return Ok(it);
            }
            z = self.vcycle(0, &r);
            let rz2 = dot(&r, &z);
            let beta = rz2 / rz;
            rz = rz2;
            each(&mut p, |i, pi| *pi = z[i] + beta * *pi);
        }
        bail!("pressure solve did not converge ({max_it} multigrid-CG iterations)")
    }
}

#[allow(dead_code)]
fn _keep(m: &Multigrid) -> usize {
    m.coarse_a.n()
}
