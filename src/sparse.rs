//! Node-block sparse matrices, preconditioned conjugate gradients and small dense helpers.

use anyhow::{Result, bail};
use rayon::prelude::*;

/// Sparse matrix whose sparsity follows node adjacency, with `bs` dofs per node.
/// Row (node n, comp c) holds deg(n)*bs entries laid out as [neighbour][comp].
pub struct NodeSparse {
    pub bs: usize,
    pub ptr: Vec<usize>,
    pub adj: Vec<u32>,
    pub vals: Vec<f64>,
}

impl NodeSparse {
    pub fn new(n_nodes: usize, elems: &[[u32; 8]], bs: usize) -> Self {
        let mut lists: Vec<Vec<u32>> = vec![Vec::new(); n_nodes];
        for e in elems {
            for &a in e {
                lists[a as usize].extend_from_slice(e);
            }
        }
        let mut ptr = Vec::with_capacity(n_nodes + 1);
        let mut adj = Vec::new();
        ptr.push(0);
        for l in lists.iter_mut() {
            l.sort_unstable();
            l.dedup();
            adj.extend_from_slice(l);
            ptr.push(adj.len());
        }
        let vals = vec![0.0; adj.len() * bs * bs];
        NodeSparse { bs, ptr, adj, vals }
    }

    pub fn n(&self) -> usize {
        (self.ptr.len() - 1) * self.bs
    }

    #[inline]
    fn row_start(&self, node: usize, c: usize) -> usize {
        let deg = self.ptr[node + 1] - self.ptr[node];
        self.ptr[node] * self.bs * self.bs + c * deg * self.bs
    }

    #[inline]
    pub fn find(&self, row_node: usize, col_node: u32) -> usize {
        let s = &self.adj[self.ptr[row_node]..self.ptr[row_node + 1]];
        s.binary_search(&col_node).expect("sparsity pattern")
    }

    /// Scatter a dense element matrix (size 8*bs square, row-major) scaled by `s`.
    pub fn add_element(&mut self, en: &[u32; 8], ke: &[f64], s: f64) {
        let bs = self.bs;
        let m = 8 * bs;
        for a in 0..8 {
            let na = en[a] as usize;
            for b in 0..8 {
                let pos = self.find(na, en[b]);
                for c in 0..bs {
                    let rs = self.row_start(na, c) + pos * bs;
                    for d in 0..bs {
                        self.vals[rs + d] += s * ke[(a * bs + c) * m + b * bs + d];
                    }
                }
            }
        }
    }

    pub fn add_diag(&mut self, dof: usize, v: f64) {
        let (n, c) = (dof / self.bs, dof % self.bs);
        let pos = self.find(n, n as u32);
        let i = self.row_start(n, c) + pos * self.bs + c;
        self.vals[i] += v;
    }

    pub fn diag(&self) -> Vec<f64> {
        (0..self.n())
            .map(|dof| {
                let (n, c) = (dof / self.bs, dof % self.bs);
                let pos = self.find(n, n as u32);
                self.vals[self.row_start(n, c) + pos * self.bs + c]
            })
            .collect()
    }

    pub fn matvec(&self, x: &[f64], y: &mut [f64]) {
        let bs = self.bs;
        let body = |(n, yn): (usize, &mut [f64])| {
            let nb = &self.adj[self.ptr[n]..self.ptr[n + 1]];
            for (c, yc) in yn.iter_mut().enumerate() {
                let rs = self.row_start(n, c);
                let row = &self.vals[rs..rs + nb.len() * bs];
                let mut acc = 0.0;
                for (k, &m) in nb.iter().enumerate() {
                    let xm = &x[m as usize * bs..m as usize * bs + bs];
                    for d in 0..bs {
                        acc += row[k * bs + d] * xm[d];
                    }
                }
                *yc = acc;
            }
        };
        if y.len() < PAR_MIN {
            y.chunks_mut(bs).enumerate().for_each(body);
        } else {
            y.par_chunks_mut(bs).enumerate().with_min_len(512).for_each(body);
        }
    }

    /// Apply Dirichlet conditions by symmetric elimination: modifies `f`, zeroes rows/cols and
    /// puts 1 on the diagonal.
    pub fn apply_dirichlet(&mut self, fixed: &[Option<f64>], f: &mut [f64]) {
        let bs = self.bs;
        // RHS correction using the original matrix (symmetric => use row r).
        for (r, g) in fixed.iter().enumerate() {
            let Some(g) = g else { continue };
            if *g == 0.0 {
                continue;
            }
            let (n, c) = (r / bs, r % bs);
            let rs = self.row_start(n, c);
            for (k, &m) in self.adj[self.ptr[n]..self.ptr[n + 1]].iter().enumerate() {
                for d in 0..bs {
                    f[m as usize * bs + d] -= self.vals[rs + k * bs + d] * g;
                }
            }
        }
        // zero rows and columns
        let nn = self.ptr.len() - 1;
        let ptr = &self.ptr;
        let adj = &self.adj;
        let chunks: Vec<(usize, usize)> = (0..nn).map(|n| (ptr[n] * bs * bs, ptr[n + 1] * bs * bs)).collect();
        let mut slices: Vec<&mut [f64]> = Vec::with_capacity(nn);
        let mut rest: &mut [f64] = &mut self.vals;
        let mut off = 0;
        for &(s, e) in &chunks {
            let (_, r) = rest.split_at_mut(s - off);
            let (mine, r2) = r.split_at_mut(e - s);
            slices.push(mine);
            rest = r2;
            off = e;
        }
        slices.into_par_iter().enumerate().for_each(|(n, blk)| {
            let nb = &adj[ptr[n]..ptr[n + 1]];
            let deg = nb.len();
            for c in 0..bs {
                let row_fixed = fixed[n * bs + c].is_some();
                for (k, &m) in nb.iter().enumerate() {
                    for d in 0..bs {
                        let col = m as usize * bs + d;
                        let idx = c * deg * bs + k * bs + d;
                        if row_fixed || fixed[col].is_some() {
                            blk[idx] = if row_fixed && col == n * bs + c { 1.0 } else { 0.0 };
                        }
                    }
                }
            }
        });
        for (r, g) in fixed.iter().enumerate() {
            if let Some(g) = g {
                f[r] = *g;
            }
        }
    }
}

pub const PAR_MIN: usize = 30_000;

pub fn dotp(a: &[f64], b: &[f64]) -> f64 {
    if a.len() < PAR_MIN {
        return a.iter().zip(b).map(|(x, y)| x * y).sum();
    }
    a.par_chunks(8192).zip(b.par_chunks(8192)).map(|(x, y)| x.iter().zip(y).map(|(p, q)| p * q).sum::<f64>()).sum()
}

/// y[i] = f(y[i], x[i]) in parallel for large vectors.
pub fn zip_apply(y: &mut [f64], x: &[f64], f: impl Fn(&mut f64, f64) + Sync + Send) {
    if y.len() < PAR_MIN {
        y.iter_mut().zip(x).for_each(|(a, b)| f(a, *b));
    } else {
        y.par_chunks_mut(8192).zip(x.par_chunks(8192)).for_each(|(ya, xa)| ya.iter_mut().zip(xa).for_each(|(a, b)| f(a, *b)));
    }
}

pub struct CgStats {
    pub iterations: usize,
    pub residual: f64,
}

/// Jacobi preconditioned conjugate gradients. `x` holds the initial guess on input.
pub fn pcg(
    a: &NodeSparse,
    b: &[f64],
    x: &mut [f64],
    tol: f64,
    max_iter: usize,
    progress: &dyn Fn(f64),
) -> Result<CgStats> {
    let n = b.len();
    let dinv: Vec<f64> = a.diag().iter().map(|d| if d.abs() > 0.0 { 1.0 / d } else { 1.0 }).collect();
    let mut r = vec![0.0; n];
    a.matvec(x, &mut r);
    zip_apply(&mut r, b, |ri, bi| *ri = bi - *ri);
    let bnorm = dotp(b, b).sqrt().max(1e-300);
    let mut z: Vec<f64> = r.iter().zip(&dinv).map(|(r, d)| r * d).collect();
    let mut p = z.clone();
    let mut rz = dotp(&r, &z);
    let mut ap = vec![0.0; n];
    let r0 = dotp(&r, &r).sqrt() / bnorm;
    if r0 < tol {
        return Ok(CgStats { iterations: 0, residual: r0 });
    }
    let log_r0 = r0.ln();
    let log_tol = tol.ln();
    for it in 1..=max_iter {
        a.matvec(&p, &mut ap);
        let pap = dotp(&p, &ap);
        if pap <= 0.0 || !pap.is_finite() {
            bail!("solver broke down (matrix not positive definite): the model is probably under-constrained. Add supports so it cannot move freely.");
        }
        let alpha = rz / pap;
        zip_apply(x, &p, |xi, pi| *xi += alpha * pi);
        zip_apply(&mut r, &ap, |ri, api| *ri -= alpha * api);
        let rn = dotp(&r, &r).sqrt() / bnorm;
        if it % 25 == 0 {
            progress(((log_r0 - rn.max(1e-300).ln()) / (log_r0 - log_tol)).clamp(0.0, 1.0));
        }
        if rn < tol {
            return Ok(CgStats { iterations: it, residual: rn });
        }
        z.iter_mut().zip(&r).zip(&dinv).for_each(|((zi, ri), di)| *zi = ri * di);
        let rz_new = dotp(&r, &z);
        let beta = rz_new / rz;
        rz = rz_new;
        zip_apply(&mut p, &z, |pi, zi| *pi = zi + beta * *pi);
    }
    let rn = dotp(&r, &r).sqrt() / bnorm;
    if rn < tol * 1e3 {
        return Ok(CgStats { iterations: max_iter, residual: rn });
    }
    bail!("solver did not converge (residual {rn:.2e}); the model may be under-constrained or ill-conditioned")
}

// ---------- small dense linear algebra ----------

pub fn dense_inverse(a: &[f64], n: usize) -> Vec<f64> {
    let mut m = a.to_vec();
    let mut inv = vec![0.0; n * n];
    for i in 0..n {
        inv[i * n + i] = 1.0;
    }
    for col in 0..n {
        let piv = (col..n).max_by(|&i, &j| m[i * n + col].abs().partial_cmp(&m[j * n + col].abs()).unwrap()).unwrap();
        if piv != col {
            for k in 0..n {
                m.swap(col * n + k, piv * n + k);
                inv.swap(col * n + k, piv * n + k);
            }
        }
        let d = m[col * n + col];
        for k in 0..n {
            m[col * n + k] /= d;
            inv[col * n + k] /= d;
        }
        for i in 0..n {
            if i != col {
                let f = m[i * n + col];
                if f != 0.0 {
                    for k in 0..n {
                        m[i * n + k] -= f * m[col * n + k];
                        inv[i * n + k] -= f * inv[col * n + k];
                    }
                }
            }
        }
    }
    inv
}

/// Cyclic Jacobi eigen-decomposition of a symmetric matrix. Returns (eigenvalues, eigenvectors
/// stored column-wise in row-major n*n).
pub fn sym_eigen(a: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut m = a.to_vec();
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    for _sweep in 0..100 {
        let off: f64 = (0..n).flat_map(|i| (0..n).filter(move |&j| j != i).map(move |j| (i, j))).map(|(i, j)| m[i * n + j].powi(2)).sum();
        let diag: f64 = (0..n).map(|i| m[i * n + i].powi(2)).sum();
        if off <= 1e-24 * diag.max(1e-300) {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                let apq = m[p * n + q];
                if apq.abs() < 1e-300 {
                    continue;
                }
                let theta = (m[q * n + q] - m[p * n + p]) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let mkp = m[k * n + p];
                    let mkq = m[k * n + q];
                    m[k * n + p] = c * mkp - s * mkq;
                    m[k * n + q] = s * mkp + c * mkq;
                }
                for k in 0..n {
                    let mpk = m[p * n + k];
                    let mqk = m[q * n + k];
                    m[p * n + k] = c * mpk - s * mqk;
                    m[q * n + k] = s * mpk + c * mqk;
                }
                for k in 0..n {
                    let vkp = v[k * n + p];
                    let vkq = v[k * n + q];
                    v[k * n + p] = c * vkp - s * vkq;
                    v[k * n + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    ((0..n).map(|i| m[i * n + i]).collect(), v)
}

/// Cholesky factor (lower) of SPD matrix.
pub fn cholesky(a: &[f64], n: usize) -> Result<Vec<f64>> {
    let mut l = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if s <= 0.0 {
                    bail!("matrix not positive definite");
                }
                l[i * n + i] = s.sqrt();
            } else {
                l[i * n + j] = s / l[j * n + j];
            }
        }
    }
    Ok(l)
}

// ---------- direct solver: RCM-ordered skyline (envelope) Cholesky ----------

/// Reverse Cuthill-McKee ordering of the node graph. Returns perm[new] = old.
fn rcm(ptr: &[usize], adj: &[u32]) -> Vec<usize> {
    let n = ptr.len() - 1;
    let deg = |i: usize| ptr[i + 1] - ptr[i];
    let mut visited = vec![false; n];
    let mut order = Vec::with_capacity(n);
    let bfs_last = |start: usize| -> usize {
        let mut seen = vec![false; n];
        let mut q = std::collections::VecDeque::from([start]);
        seen[start] = true;
        let mut last = start;
        while let Some(v) = q.pop_front() {
            last = v;
            for &w in &adj[ptr[v]..ptr[v + 1]] {
                if !seen[w as usize] {
                    seen[w as usize] = true;
                    q.push_back(w as usize);
                }
            }
        }
        last
    };
    for s0 in 0..n {
        if visited[s0] {
            continue;
        }
        // pseudo-peripheral start node
        let start = bfs_last(bfs_last(s0));
        let start = if visited[start] { s0 } else { start };
        let mut q = std::collections::VecDeque::from([start]);
        visited[start] = true;
        while let Some(v) = q.pop_front() {
            order.push(v);
            let mut nb: Vec<usize> = adj[ptr[v]..ptr[v + 1]].iter().map(|&w| w as usize).filter(|&w| !visited[w]).collect();
            nb.sort_by_key(|&w| deg(w));
            for w in nb {
                visited[w] = true;
                q.push_back(w);
            }
        }
    }
    order.reverse();
    order
}

pub struct Factor {
    bs: usize,
    /// perm[new dof] = old dof
    perm: Vec<usize>,
    first: Vec<usize>,
    start: Vec<usize>,
    l: Vec<f64>,
}

impl Factor {
    /// Number of stored entries the factor would need (None if over budget).
    pub fn new(a: &NodeSparse, max_entries: usize) -> Option<Result<Factor>> {
        let bs = a.bs;
        let nn = a.ptr.len() - 1;
        let nperm = rcm(&a.ptr, &a.adj);
        let mut inv = vec![0usize; nn];
        for (new, &old) in nperm.iter().enumerate() {
            inv[old] = new;
        }
        let n = nn * bs;
        // envelope: first column per (new) dof row
        let mut first = vec![0usize; n];
        for newn in 0..nn {
            let old = nperm[newn];
            let minn = a.adj[a.ptr[old]..a.ptr[old + 1]].iter().map(|&m| inv[m as usize]).min().unwrap_or(newn);
            for c in 0..bs {
                first[newn * bs + c] = minn * bs;
            }
        }
        let mut start = vec![0usize; n + 1];
        for i in 0..n {
            start[i + 1] = start[i] + (i - first[i] + 1);
        }
        if start[n] > max_entries {
            return None;
        }
        let mut l = vec![0.0; start[n]];
        // scatter A (lower triangle) into the envelope
        for old in 0..nn {
            let newr = inv[old];
            let deg = a.ptr[old + 1] - a.ptr[old];
            for c in 0..bs {
                let i = newr * bs + c;
                let rs = a.ptr[old] * bs * bs + c * deg * bs;
                for (k, &m) in a.adj[a.ptr[old]..a.ptr[old + 1]].iter().enumerate() {
                    for d in 0..bs {
                        let j = inv[m as usize] * bs + d;
                        if j <= i {
                            l[start[i] + j - first[i]] = a.vals[rs + k * bs + d];
                        }
                    }
                }
            }
        }
        // row-oriented envelope Cholesky
        for i in 0..n {
            let fi = first[i];
            for j in fi..i {
                let fj = first[j];
                let k0 = fi.max(fj);
                let ri = &l[start[i] + k0 - fi..start[i] + j - fi];
                let rj = &l[start[j] + k0 - fj..start[j] + j - fj];
                let s: f64 = ri.iter().zip(rj).map(|(a, b)| a * b).sum();
                let ljj = l[start[j + 1] - 1];
                let idx = start[i] + j - fi;
                l[idx] = (l[idx] - s) / ljj;
            }
            let row = &l[start[i]..start[i + 1] - 1];
            let s: f64 = row.iter().map(|v| v * v).sum();
            let d = l[start[i + 1] - 1] - s;
            if d <= 0.0 || !d.is_finite() {
                return Some(Err(anyhow::anyhow!(
                    "stiffness matrix is singular: the model is under-constrained (add supports so it cannot move or rotate freely)"
                )));
            }
            l[start[i + 1] - 1] = d.sqrt();
        }
        let perm: Vec<usize> = (0..n).map(|i| nperm[i / bs] * bs + i % bs).collect();
        Some(Ok(Factor { bs, perm, first, start, l }))
    }

    pub fn solve(&self, b: &[f64]) -> Vec<f64> {
        let _ = self.bs;
        let n = self.perm.len();
        let mut y: Vec<f64> = self.perm.iter().map(|&o| b[o]).collect();
        for i in 0..n {
            let fi = self.first[i];
            let row = &self.l[self.start[i]..self.start[i + 1] - 1];
            let s: f64 = row.iter().zip(&y[fi..i]).map(|(a, b)| a * b).sum();
            y[i] = (y[i] - s) / self.l[self.start[i + 1] - 1];
        }
        for i in (0..n).rev() {
            y[i] /= self.l[self.start[i + 1] - 1];
            let fi = self.first[i];
            let yi = y[i];
            let row = &self.l[self.start[i]..self.start[i + 1] - 1];
            for (k, v) in row.iter().enumerate() {
                y[fi + k] -= v * yi;
            }
        }
        let mut x = vec![0.0; n];
        for (i, &o) in self.perm.iter().enumerate() {
            x[o] = y[i];
        }
        x
    }
}

/// Max envelope entries for the direct solver (~1.2 GB of f64 at the limit).
pub const DIRECT_MAX: usize = 150_000_000;

pub enum Solver {
    Direct(Factor),
    Iterative,
}

impl Solver {
    pub fn new(a: &NodeSparse) -> Result<Solver> {
        if std::env::var("SIMCRAFT_ITERATIVE").is_ok() {
            return Ok(Solver::Iterative);
        }
        match Factor::new(a, DIRECT_MAX) {
            Some(f) => Ok(Solver::Direct(f?)),
            None => Ok(Solver::Iterative),
        }
    }
    pub fn name(&self) -> &'static str {
        match self {
            Solver::Direct(_) => "direct sparse Cholesky",
            Solver::Iterative => "Jacobi-PCG iterative",
        }
    }
    /// Solve a x = b; `x` holds the initial guess for the iterative path.
    pub fn solve(&self, a: &NodeSparse, b: &[f64], x: &mut [f64], tol: f64, progress: &dyn Fn(f64)) -> Result<usize> {
        match self {
            Solver::Direct(f) => {
                let r = f.solve(b);
                x.copy_from_slice(&r);
                Ok(0)
            }
            Solver::Iterative => Ok(pcg(a, b, x, tol, 30_000, progress)?.iterations),
        }
    }
}
