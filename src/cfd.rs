//! Steady incompressible flow (Navier–Stokes) on the voxel grid, with optional conjugate heat
//! transfer, Boussinesq buoyancy and the LVEL turbulence model.
//!
//! Flow, default: SIMPLE (Patankar) — implicit under-relaxed momentum, Rhie–Chow face velocities, pressure
//! correction (preconditioned by a periodically refreshed factor / multigrid); no time-step limit. Fallback (and `cfd_scheme = projection`) for flows that do not
//! settle: cell-centred velocities with face normal velocities, pseudo-time projection (Chorin):
//! explicit advection (conservative, hybrid central/upwind by cell Péclet number) and diffusion
//! with a variable effective viscosity, interpolation to faces, pressure Poisson (direct or
//! warm-started PCG), face correction; marched until the velocity stops changing.
//!
//! Energy (conjugate_heat_transfer): steady advection–diffusion of temperature over every cell —
//! fluid and solid — with harmonic-mean conductances at every face (so solid/fluid interfaces are
//! coupled), Patankar's hybrid scheme for advection, heat sources in solids and thermal loads on
//! boundary faces; solved implicitly (BiCGSTAB) every few flow iterations so buoyancy
//! (ρ ≈ ρ₀[1 − β(T − T₀)]) and the flow stay coupled.
//!
//! Turbulence (LVEL, Agonafer et al. 1996): ν_eff/ν = dy⁺/du⁺ from Spalding's law of the wall,
//! with the local speed and the distance to the nearest wall; k_eff = k + ρc_p ν_t / Pr_t.
//!
//! Boundaries: no-slip walls (default on solids and enclosure walls), moving walls, free-slip
//! (Symmetry), velocity inlets (with temperature), pressure outlets.

use crate::fem::Prog;
use crate::mesh::{Mesh, REGION_ENCLOSURE};
use crate::model::*;
use crate::results::*;
use crate::sparse::*;
use anyhow::{Result, bail};
use rayon::prelude::*;
use std::collections::HashMap;

const KAPPA: f64 = 0.41;
const E_WALL: f64 = 8.6;
const PR_T: f64 = 0.9;

#[derive(Clone, Copy, PartialEq, Debug)]
enum Bc {
    Fluid(usize),
    Wall,
    Moving([f64; 3]),
    Slip,
    /// inflow speed [m/s], inflow temperature [°C]
    Inlet(f64, f64),
    Outlet(f64),
}

/// LVEL: effective-to-molecular viscosity ratio from Spalding's law, given Re_L = |u| L / ν.
pub fn lvel_ratio(re_l: f64) -> f64 {
    lvel_law(re_l).0
}

/// Spalding's law at Re_L = |u| L / ν: (local ratio dy⁺/du⁺ for the bulk, wall-function secant
/// ratio y⁺/u⁺ for the shear between a wall and the first cell centre).
pub fn lvel_law(re_l: f64) -> (f64, f64) {
    if re_l <= 1e-9 {
        return (1.0, 1.0);
    }
    let spalding = |up: f64| {
        let k = KAPPA * up;
        up + ((k).exp() - 1.0 - k - k * k / 2.0 - k * k * k / 6.0) / E_WALL
    };
    let dyp = |up: f64| {
        let k = KAPPA * up;
        1.0 + KAPPA * ((k).exp() - 1.0 - k - k * k / 2.0) / E_WALL
    };
    // solve u⁺·y⁺(u⁺) = Re_L (monotonic) by Newton from a laminar/log guess
    let mut up = re_l.sqrt().min((re_l.max(1.0)).ln() / KAPPA + 5.0).max(1e-6);
    for _ in 0..60 {
        let f = up * spalding(up) - re_l;
        let df = spalding(up) + up * dyp(up);
        let step = f / df;
        up = (up - step).max(up * 0.2);
        if step.abs() < 1e-10 * up {
            break;
        }
    }
    (dyp(up), (re_l / (up * up)).max(1.0))
}

/// Jacobi-preconditioned BiCGSTAB on a CSR matrix, restarted on breakdown. Returns iterations.
fn bicgstab(ptr: &[usize], col: &[u32], val: &[f64], b: &[f64], x: &mut [f64], tol: f64, max_it: usize) -> Result<usize> {
    let mut total = 0;
    for _ in 0..8 {
        match bicgstab_once(ptr, col, val, b, x, tol, max_it, false) {
            Ok((it, true)) => return Ok(total + it),
            Ok((it, false)) => total += it, // breakdown: restart from the current iterate
            Err(e) => return Err(e),
        }
    }
    bail!("energy equation did not converge (BiCGSTAB kept breaking down)")
}

/// One BiCGSTAB cycle: (iterations, converged). `false` = breakdown, x holds the best iterate.
fn bicgstab_once(ptr: &[usize], col: &[u32], val: &[f64], b: &[f64], x: &mut [f64], tol: f64, max_it: usize, loose: bool) -> Result<(usize, bool)> {
    let n = b.len();
    let dinv: Vec<f64> = (0..n)
        .map(|i| {
            let d = (ptr[i]..ptr[i + 1]).find(|&k| col[k] as usize == i).map(|k| val[k]).unwrap_or(1.0);
            if d.abs() > 0.0 { 1.0 / d } else { 1.0 }
        })
        .collect();
    let mv = |v: &[f64], out: &mut [f64]| {
        out.par_iter_mut().enumerate().for_each(|(i, o)| {
            let mut s = 0.0;
            for k in ptr[i]..ptr[i + 1] {
                s += val[k] * v[col[k] as usize];
            }
            *o = s;
        });
    };
    let dot = |a: &[f64], b: &[f64]| a.par_iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
    let bn = dot(b, b).sqrt().max(1e-300);
    let mut r = vec![0.0; n];
    mv(x, &mut r);
    r.par_iter_mut().zip(b).for_each(|(r, b)| *r = b - *r);
    if dot(&r, &r).sqrt() / bn < tol {
        return Ok((0, true));
    }
    let r0 = r.clone();
    let (mut rho, mut alpha, mut omega) = (1.0f64, 1.0f64, 1.0f64);
    let mut v = vec![0.0; n];
    let mut p = vec![0.0; n];
    let mut y = vec![0.0; n];
    let mut s = vec![0.0; n];
    let mut z = vec![0.0; n];
    let mut t = vec![0.0; n];
    for it in 1..=max_it {
        let rho1 = dot(&r0, &r);
        if rho1.abs() < 1e-30 * dot(&r0, &r0) {
            return Ok((it, false));
        }
        let beta = (rho1 / rho) * (alpha / omega);
        rho = rho1;
        p.par_iter_mut().enumerate().for_each(|(i, pi)| *pi = r[i] + beta * (*pi - omega * v[i]));
        y.par_iter_mut().enumerate().for_each(|(i, yi)| *yi = p[i] * dinv[i]);
        mv(&y, &mut v);
        let r0v = dot(&r0, &v);
        if r0v.abs() < 1e-300 {
            return Ok((it, false));
        }
        alpha = rho / r0v;
        s.par_iter_mut().enumerate().for_each(|(i, si)| *si = r[i] - alpha * v[i]);
        if dot(&s, &s).sqrt() / bn < tol {
            x.par_iter_mut().enumerate().for_each(|(i, xi)| *xi += alpha * y[i]);
            return Ok((it, true));
        }
        z.par_iter_mut().enumerate().for_each(|(i, zi)| *zi = s[i] * dinv[i]);
        mv(&z, &mut t);
        let tt = dot(&t, &t);
        omega = if tt > 0.0 { dot(&t, &s) / tt } else { 0.0 };
        x.par_iter_mut().enumerate().for_each(|(i, xi)| *xi += alpha * y[i] + omega * z[i]);
        r.par_iter_mut().enumerate().for_each(|(i, ri)| *ri = s[i] - omega * t[i]);
        let rn = dot(&r, &r).sqrt() / bn;
        if !rn.is_finite() {
            bail!("BiCGSTAB diverged");
        }
        if rn < tol {
            return Ok((it, true));
        }
        if omega == 0.0 {
            return Ok((it, false));
        }
    }
    if loose {
        return Ok((max_it, true));
    }
    bail!("energy equation did not converge")
}

/// Inexact BiCGSTAB for the momentum equations of SIMPLE: reduce the residual of the current
/// iterate by `reduce` (at most `max_it` iterations; outer iterations fix the rest).
fn bicgstab_reduce(ptr: &[usize], col: &[u32], val: &[f64], b: &[f64], x: &mut [f64], reduce: f64, max_it: usize) -> Result<usize> {
    let r0: f64 = (0..b.len())
        .into_par_iter()
        .map(|i| {
            let mut s = b[i];
            for k in ptr[i]..ptr[i + 1] {
                s -= val[k] * x[col[k] as usize];
            }
            s * s
        })
        .sum::<f64>()
        .sqrt();
    let bn = b.par_iter().map(|v| v * v).sum::<f64>().sqrt();
    if r0 == 0.0 || bn == 0.0 {
        return Ok(0);
    }
    let mut total = 0;
    for _ in 0..3 {
        let (it, ok) = bicgstab_once(ptr, col, val, b, x, reduce * r0 / bn, max_it, true)?;
        total += it;
        if ok {
            break;
        }
    }
    Ok(total)
}

/// Preconditioned CG to `tol` relative to the initial residual (best effort after `max_it`).
fn pcg_prec(a: &NodeSparse, b: &[f64], x: &mut [f64], tol: f64, max_it: usize, prec: &dyn Fn(&[f64]) -> Vec<f64>) -> Result<usize> {
    let n = b.len();
    let mut r = vec![0.0; n];
    a.matvec(x, &mut r);
    r.par_iter_mut().zip(b).for_each(|(r, b)| *r = b - *r);
    let dot = |u: &[f64], v: &[f64]| u.par_iter().zip(v).map(|(p, q)| p * q).sum::<f64>();
    let r0 = dot(&r, &r).sqrt();
    if r0 == 0.0 {
        return Ok(0);
    }
    let mut z = prec(&r);
    let mut p = z.clone();
    let mut rz = dot(&r, &z);
    let mut ap = vec![0.0; n];
    for it in 1..=max_it {
        a.matvec(&p, &mut ap);
        let pap = dot(&p, &ap);
        if !(pap > 0.0) {
            return Ok(it);
        }
        let al = rz / pap;
        x.par_iter_mut().zip(&p).for_each(|(x, p)| *x += al * p);
        r.par_iter_mut().zip(&ap).for_each(|(r, q)| *r -= al * q);
        let rn = dot(&r, &r).sqrt();
        if !rn.is_finite() {
            bail!("pressure-correction solve diverged");
        }
        if rn < tol * r0 {
            return Ok(it);
        }
        z = prec(&r);
        let rz2 = dot(&r, &z);
        let beta = rz2 / rz;
        rz = rz2;
        p.par_iter_mut().zip(&z).for_each(|(p, z)| *p = z + beta * *p);
    }
    Ok(max_it)
}

/// Thermal boundary condition on an outer face of a cell.
#[derive(Clone, Copy)]
enum Tbc {
    Fixed(f64, usize),
    Flux(f64),
    Conv(f64, f64, usize),
    /// radiation to ambient: emissivity, ambient °C, load index
    Rad(f64, f64, usize),
}

/// Secant radiation coefficient εσ(T² + T∞²)(T + T∞) [W/(m²·K)] for temperatures in °C.
fn h_rad(eps: f64, t: f64, ta: f64) -> f64 {
    let (a, b) = ((t + crate::radiation::KELVIN).max(1.0), ta + crate::radiation::KELVIN);
    eps * crate::radiation::SIGMA * (a * a + b * b) * (a + b)
}

pub fn solve(model: &Model, mesh: &Mesh, prog: Prog, res: &mut Results) -> Result<()> {
    let st = &model.analysis;
    let energy = st.kind == AnalysisType::ConjugateHeatTransfer;
    let lvel = match st.turbulence.as_str() {
        "laminar" => false,
        "lvel" => true,
        t => bail!("turbulence must be laminar or lvel, not '{t}'"),
    };
    let h = mesh.h * 1e-3;
    let [nx, ny, nz] = mesh.dims;
    let ncell = nx * ny * nz;
    let cid = |c: [u32; 3]| c[0] as usize + nx * (c[1] as usize + ny * c[2] as usize);
    let mut elem_of = vec![u32::MAX; ncell];
    for (e, c) in mesh.elem_cell.iter().enumerate() {
        elem_of[cid(*c)] = e as u32;
    }
    // neighbouring element of e across direction d (None at the domain boundary)
    let neighbour = |e: usize, d: usize| -> Option<usize> {
        let c = mesh.elem_cell[e];
        let a = d / 2;
        let mut q = [c[0] as i64, c[1] as i64, c[2] as i64];
        q[a] += if d % 2 == 1 { 1 } else { -1 };
        if q[a] < 0 || q[a] as usize >= mesh.dims[a] {
            return None;
        }
        let ne = elem_of[cid([q[0] as u32, q[1] as u32, q[2] as u32])];
        (ne != u32::MAX).then_some(ne as usize)
    };
    let fluid_elems: Vec<usize> = (0..mesh.elems.len()).filter(|&e| mesh.elem_region[e] == REGION_ENCLOSURE).collect();
    if fluid_elems.is_empty() {
        bail!("no fluid region: add an enclosure (GEOMETRY ▸ Enclosure) filled with Air or Water");
    }
    let fmat = mesh.materials[mesh.elem_mat[fluid_elems[0]] as usize].clone();
    if !fmat.is_fluid || fmat.dynamic_viscosity <= 0.0 {
        bail!("the enclosure material '{}' is not a fluid: use Air, Water or Engine Oil", fmat.name);
    }
    let (rho, mu) = (fmat.density, fmat.dynamic_viscosity);
    let nu = mu / rho;
    let cp = fmat.specific_heat;
    let t_ref = st.reference_temperature;
    let nf = fluid_elems.len();
    let mut fidx = vec![usize::MAX; mesh.elems.len()];
    for (i, &e) in fluid_elems.iter().enumerate() {
        fidx[e] = i;
    }

    // ---------------------------------------------------------------- boundary conditions
    let mut face_bc: HashMap<(u32, u8), Bc> = HashMap::new();
    let mut load_faces: Vec<(u32, String, Vec<(u32, u8)>)> = Vec::new();
    let mut uref: f64 = 0.0;
    let mut has_inlet = false;
    let mut has_outlet = false;
    for it in &model.loads {
        let bc = match &it.load {
            Load::Inlet { velocity, temperature, .. } => {
                uref = uref.max(velocity.abs());
                has_inlet = true;
                Bc::Inlet(*velocity, temperature.unwrap_or(t_ref))
            }
            Load::Outlet { pressure, .. } => {
                has_outlet = true;
                Bc::Outlet(*pressure)
            }
            Load::Symmetry { .. } => Bc::Slip,
            Load::MovingWall { velocity, .. } => {
                uref = uref.max((velocity[0].powi(2) + velocity[1].powi(2) + velocity[2].powi(2)).sqrt());
                Bc::Moving(*velocity)
            }
            _ => continue,
        };
        let fs = mesh.resolve_faces(it.load.faces())?;
        let keys: Vec<(u32, u8)> = fs.iter().map(|&f| (mesh.faces[f as usize].elem, mesh.faces[f as usize].dir)).collect();
        for k in &keys {
            face_bc.insert(*k, bc);
        }
        if matches!(bc, Bc::Inlet(..) | Bc::Outlet(_)) {
            load_faces.push((it.id, it.load.label().to_string(), keys));
        }
    }
    // gravity / buoyancy
    let gravity = model.loads.iter().find_map(|l| match &l.load {
        Load::Gravity { acceleration } => Some(*acceleration),
        _ => None,
    });
    let beta = fmat.thermal_expansion;
    let buoyant = energy && gravity.is_some();
    if !has_inlet && uref == 0.0 && !buoyant {
        bail!("fluid analysis needs something to drive the flow: a Velocity Inlet (+ Pressure Outlet), a Moving Wall, or — in conjugate heat transfer — gravity for natural convection");
    }
    if has_inlet && !has_outlet {
        bail!("a Velocity Inlet needs a Pressure Outlet (e.g. 0 Pa on Enclosure.xmax)");
    }

    let mut nb = vec![[Bc::Wall; 6]; nf];
    for (i, &e) in fluid_elems.iter().enumerate() {
        for d in 0..6usize {
            nb[i][d] = match neighbour(e, d) {
                None => *face_bc.get(&(e as u32, d as u8)).unwrap_or(&Bc::Wall),
                Some(ne) if fidx[ne] != usize::MAX => Bc::Fluid(fidx[ne]),
                // solid neighbour: its face points back at us
                Some(ne) => *face_bc.get(&(ne as u32, (d ^ 1) as u8)).unwrap_or(&Bc::Wall),
            };
        }
    }

    // ---------------------------------------------------------------- wall distance (LVEL)
    let centre = |e: usize| {
        let c = mesh.elem_cell[e];
        [(c[0] as f64 + 0.5) * h, (c[1] as f64 + 0.5) * h, (c[2] as f64 + 0.5) * h]
    };
    let wall_dist: Vec<f64> = if lvel {
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;
        let mut best = vec![f64::INFINITY; nf];
        let mut wp = vec![[0.0f64; 3]; nf];
        let mut heap = BinaryHeap::new();
        for i in 0..nf {
            for d in 0..6 {
                if matches!(nb[i][d], Bc::Wall | Bc::Moving(_)) && 0.5 * h < best[i] {
                    let mut p = centre(fluid_elems[i]);
                    p[d / 2] += if d % 2 == 1 { 0.5 * h } else { -0.5 * h };
                    best[i] = 0.5 * h;
                    wp[i] = p;
                }
            }
            if best[i].is_finite() {
                heap.push(Reverse(((best[i] * 1e12) as u64, i)));
            }
        }
        while let Some(Reverse((dq, i))) = heap.pop() {
            if (dq as f64) > best[i] * 1e12 + 1.0 {
                continue;
            }
            for d in 0..6 {
                if let Bc::Fluid(j) = nb[i][d] {
                    let c = centre(fluid_elems[j]);
                    let dist = ((c[0] - wp[i][0]).powi(2) + (c[1] - wp[i][1]).powi(2) + (c[2] - wp[i][2]).powi(2)).sqrt();
                    if dist < best[j] * (1.0 - 1e-9) {
                        best[j] = dist;
                        wp[j] = wp[i];
                        heap.push(Reverse(((dist * 1e12) as u64, j)));
                    }
                }
            }
        }
        // no wall at all (e.g. all slip/inlet): use the domain size
        let dmax = (nx.max(ny).max(nz)) as f64 * h;
        best.into_iter().map(|b| if b.is_finite() { b } else { dmax }).collect()
    } else {
        vec![]
    };
    let mut nu_eff = vec![nu; nf];
    // wall-function viscosity for faces between a fluid cell and a wall (secant of the log law)
    let mut nu_wall = vec![nu; nf];

    // ---------------------------------------------------------------- pressure matrix
    let mut ptr = vec![0usize];
    let mut adj: Vec<u32> = Vec::new();
    for i in 0..nf {
        let mut row: Vec<u32> = vec![i as u32];
        for d in 0..6 {
            if let Bc::Fluid(j) = nb[i][d] {
                row.push(j as u32);
            }
        }
        row.sort_unstable();
        adj.extend_from_slice(&row);
        ptr.push(adj.len());
    }
    let mut a = NodeSparse { bs: 1, ptr: ptr.clone(), adj: adj.clone(), vals: vec![0.0; adj.len()] };
    let mut outlet_rhs = vec![0.0; nf];
    for i in 0..nf {
        let row = &adj[ptr[i]..ptr[i + 1]];
        let di = row.binary_search(&(i as u32)).unwrap();
        for d in 0..6 {
            match nb[i][d] {
                Bc::Fluid(j) => {
                    let dj = row.binary_search(&(j as u32)).unwrap();
                    a.vals[ptr[i] + di] += 1.0;
                    a.vals[ptr[i] + dj] -= 1.0;
                }
                Bc::Outlet(po) => {
                    a.vals[ptr[i] + di] += 2.0;
                    outlet_rhs[i] += 2.0 * po;
                }
                _ => {}
            }
        }
    }
    if !has_outlet {
        // closed domain: pressure is defined up to a constant — pin one cell
        let di = adj[ptr[0]..ptr[1]].binary_search(&0).unwrap();
        a.vals[ptr[0] + di] += 1.0;
    }
    // pressure solver: direct / PCG for small grids, multigrid-preconditioned CG for large ones
    let (psolver, mg) = if nf > 20_000 {
        let coords: Vec<[u32; 3]> = fluid_elems.iter().map(|&e| mesh.elem_cell[e]).collect();
        (None, Some(crate::multigrid::Multigrid::new(&a, &coords)?))
    } else {
        (Some(Solver::new(&a)?), None)
    };

    // ---------------------------------------------------------------- energy system layout
    let ne = mesh.elems.len();
    let fluid_k = fmat.thermal_conductivity;
    let elem_k: Vec<[f64; 3]> = (0..ne).map(|e| {
        let m = &mesh.materials[mesh.elem_mat[e] as usize];
        [m.k_axis(0), m.k_axis(1), m.k_axis(2)]
    }).collect();
    let mut tbc: HashMap<(u32, u8), Tbc> = HashMap::new();
    let mut q_elem = vec![0.0f64; ne];
    let mut thermal_loads: Vec<(u32, String)> = Vec::new();
    let mut has_sink = false;
    if energy {
        for (li, it) in model.loads.iter().enumerate() {
            match &it.load {
                Load::Temperature { faces, value } => {
                    has_sink = true;
                    thermal_loads.push((it.id, it.load.label().to_string()));
                    for f in mesh.resolve_faces(faces)? {
                        let bf = &mesh.faces[f as usize];
                        tbc.insert((bf.elem, bf.dir), Tbc::Fixed(*value, li));
                    }
                }
                Load::Convection { faces, coefficient, ambient } => {
                    has_sink = true;
                    thermal_loads.push((it.id, it.load.label().to_string()));
                    for f in mesh.resolve_faces(faces)? {
                        let bf = &mesh.faces[f as usize];
                        tbc.insert((bf.elem, bf.dir), Tbc::Conv(*coefficient, *ambient, li));
                    }
                }
                Load::Radiation { faces, emissivity, ambient, enclosure } => {
                    has_sink = true;
                    thermal_loads.push((it.id, it.load.label().to_string()));
                    if *enclosure {
                        res.warnings.push(format!("radiation #{}: surface-to-surface radiation is not modelled in conjugate heat transfer (the fluid region is between the faces); treated as radiation to the ambient", it.id));
                    }
                    for f in mesh.resolve_faces(faces)? {
                        let bf = &mesh.faces[f as usize];
                        tbc.insert((bf.elem, bf.dir), Tbc::Rad(*emissivity, *ambient, li));
                    }
                }
                Load::HeatFlux { faces, value } => {
                    for f in mesh.resolve_faces(faces)? {
                        let bf = &mesh.faces[f as usize];
                        tbc.insert((bf.elem, bf.dir), Tbc::Flux(*value));
                    }
                }
                Load::HeatGeneration { .. } => {
                    let (els, q) = crate::fem::heat_source(model, mesh, &it.load)?;
                    for e in els {
                        q_elem[e] += q;
                    }
                }
                _ => {}
            }
        }
        if !has_sink && !has_inlet {
            bail!("conjugate heat transfer needs a way for heat to leave: a Velocity Inlet/Outlet, a Temperature or a Convection boundary");
        }
        let interior = tbc.keys().filter(|(e, d)| neighbour(*e as usize, *d as usize).is_some()).count();
        if interior > 0 {
            res.warnings.push(format!("{interior} thermal-load faces lie between a solid and the fluid; the solid–fluid heat exchange there is computed, so those loads are ignored. Put heat sources in the parts with Heat Generation."));
            tbc.retain(|(e, d), _| neighbour(*e as usize, *d as usize).is_none());
        }
    }
    let mut temp = vec![t_ref; ne];
    // CSR pattern for the energy matrix (all cells)
    let (eptr, ecol) = {
        let mut eptr = vec![0usize];
        let mut ecol: Vec<u32> = Vec::new();
        if energy {
            for e in 0..ne {
                let mut row = vec![e as u32];
                for d in 0..6 {
                    if let Some(n2) = neighbour(e, d) {
                        row.push(n2 as u32);
                    }
                }
                row.sort_unstable();
                ecol.extend_from_slice(&row);
                eptr.push(ecol.len());
            }
        }
        (eptr, ecol)
    };
    let rcp = rho * cp;
    let a2 = h * h;
    // conductivity of a cell (fluid: + turbulent part)
    let k_of = |e: usize, nu_eff: &[f64]| -> [f64; 3] {
        let i = fidx[e];
        if i != usize::MAX { [fluid_k + if lvel { rcp * (nu_eff[i] - nu).max(0.0) / PR_T } else { 0.0 }; 3] } else { elem_k[e] }
    };

    // ---------------------------------------------------------------- iteration state
    let mut uf = vec![[0.0f64; 6]; nf];
    let mut uc = vec![[0.0f64; 3]; nf];
    let mut p = vec![0.0f64; nf];
    let mut ustar = vec![[0.0f64; 3]; nf];
    let mut rhs = vec![0.0; nf];
    let max_it = st.max_iterations.clamp(50, 200_000);
    let mut iters = 0usize;
    let mut change = f64::INFINITY;
    let mut t_change = f64::INFINITY;
    let mut cg_total = 0usize;
    let mut bicg_total = 0usize;
    let energy_every = 10usize;
    let g = gravity.unwrap_or([0.0, 0.0, -9.80665]);
    let lchar = {
        let (lo, hi) = crate::mesh::additive_bounds(&model.bodies);
        let l = (hi[0] - lo[0]).max(hi[1] - lo[1]).max(hi[2] - lo[2]);
        if l > 0.0 { l * 1e-3 } else { (nx.max(ny).max(nz)) as f64 * h }
    };

    let ldomain = (nx.max(ny).max(nz)) as f64 * h;
    const TOL: f64 = 1e-4;
    // ghost value of velocity component `comp` across face d of fluid cell i
    let ghost = |i: usize, d: usize, comp: usize, uc: &[[f64; 3]]| -> f64 {
        let a = d / 2;
        match nb[i][d] {
            Bc::Fluid(j) => uc[j][comp],
            Bc::Wall => -uc[i][comp],
            Bc::Moving(v) => {
                if comp == a {
                    -uc[i][comp]
                } else {
                    2.0 * v[comp] - uc[i][comp]
                }
            }
            Bc::Slip => {
                if comp == a {
                    -uc[i][comp]
                } else {
                    uc[i][comp]
                }
            }
            Bc::Inlet(v, _) => {
                let vin = if comp == a { if d % 2 == 1 { -v } else { v } } else { 0.0 };
                2.0 * vin - uc[i][comp]
            }
            Bc::Outlet(_) => uc[i][comp],
        }
    };

    // energy assembly + solve with the current face velocities
    let solve_energy = |uf: &[[f64; 6]], nu_eff: &[f64], nu_wall: &[f64], temp: &mut Vec<f64>, reduce: Option<f64>| -> Result<usize> {
        let mut val = vec![0.0f64; ecol.len()];
        let mut b = vec![0.0f64; ne];
        let kc: Vec<[f64; 3]> = (0..ne).map(|e| k_of(e, nu_eff)).collect();
        let kw: Vec<[f64; 3]> = (0..ne).map(|e| k_of(e, nu_wall)).collect();
        for e in 0..ne {
            let row = &ecol[eptr[e]..eptr[e + 1]];
            let di = eptr[e] + row.binary_search(&(e as u32)).unwrap();
            let fi = fidx[e];
            b[e] += q_elem[e] * h * h * h;
            for d in 0..6 {
                let out = if d % 2 == 1 { 1.0 } else { -1.0 };
                match neighbour(e, d) {
                    Some(n2) => {
                        // fluid–solid faces use the wall-function conductivity on the fluid side
                        let (k1, k2) = if (fi != usize::MAX) != (fidx[n2] != usize::MAX) { (kw[e][d / 2], kw[n2][d / 2]) } else { (kc[e][d / 2], kc[n2][d / 2]) };
                        let kh = 2.0 * k1 * k2 / (k1 + k2).max(1e-300);
                        let dcond = kh * h;
                        let f = if fi != usize::MAX && fidx[n2] != usize::MAX { rcp * uf[fi][d] * out * a2 } else { 0.0 };
                        let anb = (-f).max(dcond - f / 2.0).max(0.0);
                        let dj = eptr[e] + row.binary_search(&(n2 as u32)).unwrap();
                        val[dj] -= anb;
                        val[di] += anb + f;
                    }
                    None => {
                        let key = (e as u32, d as u8);
                        if fi != usize::MAX {
                            match nb[fi][d] {
                                Bc::Inlet(_, tin) => {
                                    let f = rcp * uf[fi][d] * out * a2;
                                    let anb = (-f).max(2.0 * kc[e][d / 2] * h - f / 2.0).max(0.0);
                                    val[di] += anb + f;
                                    b[e] += anb * tin;
                                    continue;
                                }
                                Bc::Outlet(_) => {
                                    let f = rcp * uf[fi][d] * out * a2;
                                    let anb = (-f).max(0.0); // backflow enters at the reference temperature
                                    val[di] += anb + f;
                                    b[e] += anb * t_ref;
                                    continue;
                                }
                                _ => {}
                            }
                        }
                        match tbc.get(&key) {
                            Some(Tbc::Fixed(tw, _)) => {
                                let c = 2.0 * kw[e][d / 2] * h;
                                val[di] += c;
                                b[e] += c * tw;
                            }
                            Some(Tbc::Conv(hc, tamb, _)) => {
                                let c = 1.0 / (1.0 / (2.0 * kw[e][d / 2] * h) + 1.0 / (hc * a2));
                                val[di] += c;
                                b[e] += c * tamb;
                            }
                            Some(Tbc::Rad(eps, ta, _)) => {
                                let c = 1.0 / (1.0 / (2.0 * kw[e][d / 2] * h) + 1.0 / (h_rad(*eps, temp[e], *ta) * a2));
                                val[di] += c;
                                b[e] += c * ta;
                            }
                            Some(Tbc::Flux(q)) => b[e] += q * a2,
                            None => {}
                        }
                    }
                }
            }
        }
        match reduce {
            Some(r) => bicgstab_reduce(&eptr, &ecol, &val, &b, temp, r, 25),
            None => bicgstab(&eptr, &ecol, &val, &b, temp, 1e-9, 5000),
        }
    };

    let mut last_energy_t: Vec<f64> = temp.clone();
    let mut t_pressure = 0.0f64;
    let (mut avg_uf, mut avg_uc, mut avg_p, mut avg_t) = (vec![[0.0f64; 6]; nf], vec![[0.0f64; 3]; nf], vec![0.0f64; nf], vec![0.0f64; ne]);
    let (mut avg_n, mut sim_time, mut avg_start_time) = (0usize, 0.0f64, 0.0f64);
    let t_loop = std::time::Instant::now();
    let profile = std::env::var_os("SIMCRAFT_PROFILE").is_some();

    // ---------------------------------------------------------------- SIMPLE (implicit, steady)
    // Momentum solved implicitly (hybrid advection, under-relaxed), face velocities by Rhie–Chow
    // interpolation with Majumdar's relaxation correction, SIMPLE pressure correction. No time-
    // step limit, so fine meshes and slow (diffusion-dominated) flows converge in a few hundred
    // outer iterations. Flows that do not settle (vortex shedding, plumes) fall back to the
    // projection method below, which time-averages them.
    let scheme = st.cfd_scheme.as_str();
    let mut simple_ok = false;
    let mut simple_iters = 0usize;
    let mut mom_total = 0usize;
    let mut fallback_note: Option<String> = None;
    if scheme != "projection" {
        // classic SIMPLE relaxation (Patankar): robust for cavities, buoyancy and LVEL alike
        let (alpha, alpha_p) = (0.7f64, 0.3f64);
        let af = h * h;
        let vol = h * h * h;
        let budget = if scheme == "simple" { max_it } else { max_it.min(2500) };
        let mut stalled = false;
        let dpos: Vec<usize> = (0..nf).map(|i| ptr[i] + adj[ptr[i]..ptr[i + 1]].binary_search(&(i as u32)).unwrap()).collect();
        let npos: Vec<[usize; 6]> = (0..nf)
            .map(|i| {
                let mut q = [usize::MAX; 6];
                for d in 0..6 {
                    if let Bc::Fluid(j) = nb[i][d] {
                        q[d] = ptr[i] + adj[ptr[i]..ptr[i + 1]].binary_search(&(j as u32)).unwrap();
                    }
                }
                q
            })
            .collect();
        // cell pressure gradient; boundary faces: outlet value (or 0 for corrections), else p_i
        let cell_grad = |p: &[f64], correction: bool| -> Vec<[f64; 3]> {
            (0..nf)
                .into_par_iter()
                .map(|i| {
                    let pf = |d: usize| match nb[i][d] {
                        Bc::Fluid(j) => 0.5 * (p[i] + p[j]),
                        Bc::Outlet(po) => {
                            if correction {
                                0.0
                            } else {
                                po
                            }
                        }
                        _ => p[i],
                    };
                    [(pf(1) - pf(0)) / h, (pf(3) - pf(2)) / h, (pf(5) - pf(4)) / h]
                })
                .collect()
        };
        let mut val = vec![0.0f64; adj.len()];
        let mut mval = vec![0.0f64; adj.len()];
        let mut dcoef = vec![0.0f64; nf];
        let mut pc = vec![0.0f64; nf];
        let mut xb = vec![0.0f64; nf];
        let mut bb = vec![0.0f64; nf];
        let mut best = (f64::INFINITY, 0usize);
        let mut du = f64::INFINITY;
        let mut mres = f64::INFINITY;
        let mut t_change_s = f64::INFINITY;
        let coords: Vec<[u32; 3]> = fluid_elems.iter().map(|&e| mesh.elem_cell[e]).collect();
        // fixed boundary face velocities (axis components)
        for i in 0..nf {
            for d in 0..6 {
                if let Bc::Inlet(v, _) = nb[i][d] {
                    uf[i][d] = if d % 2 == 1 { -v } else { v };
                }
            }
        }
        // pressure-correction preconditioner, rebuilt every few iterations (d changes slowly)
        enum Pre {
            None,
            Direct(crate::sparse::Factor),
            Mg(crate::multigrid::Multigrid),
        }
        let mut pre = Pre::None;
        let mut last_pcg = 0usize;
        for it in 1..=budget {
            simple_iters = it;
            let umax = uc.par_iter().map(|u| u[0].abs().max(u[1].abs()).max(u[2].abs())).reduce(|| 0.0, f64::max);
            let uscale = umax.max(uref).max(1e-9);
            if lvel {
                nu_eff.par_iter_mut().zip(nu_wall.par_iter_mut()).enumerate().for_each(|(i, (v, w))| {
                    let sp = (uc[i][0].powi(2) + uc[i][1].powi(2) + uc[i][2].powi(2)).sqrt();
                    *v = nu * lvel_law(sp * wall_dist[i] / nu).0;
                    *w = nu * lvel_law(sp * 0.5 * h / nu).1;
                });
            }
            if buoyant {
                bicg_total += solve_energy(&uf, &nu_eff, &nu_wall, &mut temp, Some(0.1))?;
                let tr = temp.iter().cloned().fold(f64::NEG_INFINITY, f64::max) - temp.iter().cloned().fold(f64::INFINITY, f64::min);
                t_change_s = temp.iter().zip(&last_energy_t).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max) / tr.max(1e-6);
                last_energy_t.clone_from(&temp);
            }
            // ---- momentum coefficients (same off-diagonals for all three components)
            let gp = cell_grad(&p, false);
            val.iter_mut().for_each(|v| *v = 0.0);
            let mut extra = vec![[0.0f64; 3]; nf];
            let mut src = vec![[0.0f64; 3]; nf];
            for i in 0..nf {
                for d in 0..6 {
                    let out = if d % 2 == 1 { 1.0 } else { -1.0 };
                    let ax = d / 2;
                    let f = uf[i][d] * out * af; // outward volume flux
                    match nb[i][d] {
                        Bc::Fluid(j) => {
                            let dd = 0.5 * (nu_eff[i] + nu_eff[j]) * af / h;
                            let anb = (-f).max(dd - f / 2.0).max(0.0);
                            val[dpos[i]] += anb + f;
                            val[npos[i][d]] -= anb;
                        }
                        Bc::Wall | Bc::Moving(_) => {
                            let c = nu_wall[i] * af / (h / 2.0);
                            for comp in 0..3 {
                                extra[i][comp] += c;
                                if let Bc::Moving(v) = nb[i][d]
                                    && comp != ax {
                                        src[i][comp] += c * v[comp];
                                    }
                            }
                        }
                        Bc::Slip => extra[i][ax] += nu_eff[i] * af / (h / 2.0),
                        Bc::Inlet(..) => {
                            let dd = nu_eff[i] * af / (h / 2.0);
                            let anb = (-f).max(dd - f / 2.0).max(0.0);
                            val[dpos[i]] += anb + f;
                            src[i][ax] += anb * uf[i][d];
                        }
                        Bc::Outlet(_) => {
                            val[dpos[i]] += f.max(0.0);
                            for comp in 0..3 {
                                src[i][comp] += (-f).max(0.0) * uc[i][comp];
                            }
                        }
                    }
                }
                for comp in 0..3 {
                    src[i][comp] -= vol / rho * gp[i][comp];
                    if buoyant {
                        src[i][comp] -= vol * beta * (temp[fluid_elems[i]] - t_ref) * g[comp];
                    }
                }
            }
            // ---- solve each component (under-relaxed)
            let base_diag: Vec<f64> = dpos.iter().map(|&k| val[k]).collect();
            let mut apr = vec![0.0f64; nf]; // mean relaxed diagonal
            for comp in 0..3 {
                for i in 0..nf {
                    let ac = (base_diag[i] + extra[i][comp]).max(1e-300) / alpha;
                    val[dpos[i]] = ac;
                    bb[i] = src[i][comp] + (1.0 - alpha) * ac * uc[i][comp];
                    xb[i] = uc[i][comp];
                    apr[i] += ac / 3.0;
                }
                mom_total += bicgstab_reduce(&ptr, &adj, &val, &bb, &mut xb, 0.05, 60)?;
                for i in 0..nf {
                    ustar[i][comp] = xb[i];
                }
            }
            // SIMPLE: d = V / (ρ a_P/α)
            for i in 0..nf {
                dcoef[i] = vol / (rho * apr[i].max(1e-300));
            }
            // ---- Rhie–Chow face velocities
            let uf_old = uf.clone();
            uf.par_iter_mut().enumerate().for_each(|(i, ufi)| {
                for d in 0..6 {
                    let ax = d / 2;
                    let up = d % 2 == 1;
                    match nb[i][d] {
                        Bc::Fluid(j) => {
                            let ub = 0.5 * (ustar[i][ax] + ustar[j][ax]);
                            let db = 0.5 * (dcoef[i] + dcoef[j]);
                            let gb = 0.5 * (gp[i][ax] + gp[j][ax]);
                            let gf = if up { (p[j] - p[i]) / h } else { (p[i] - p[j]) / h };
                            ufi[d] = ub - db * (gf - gb) + (1.0 - alpha) * (uf_old[i][d] - 0.5 * (uc[i][ax] + uc[j][ax]));
                        }
                        Bc::Outlet(po) => {
                            let gf = if up { (po - p[i]) / (h / 2.0) } else { (p[i] - po) / (h / 2.0) };
                            ufi[d] = ustar[i][ax] - dcoef[i] * (gf - gp[i][ax]) + (1.0 - alpha) * (uf_old[i][d] - uc[i][ax]);
                        }
                        _ => {}
                    }
                }
            });
            // ---- pressure correction
            let imb: Vec<f64> = (0..nf).into_par_iter().map(|i| (0..6).map(|d| uf[i][d] * if d % 2 == 1 { af } else { -af }).sum()).collect();
            mval.iter_mut().for_each(|v| *v = 0.0);
            for i in 0..nf {
                for d in 0..6 {
                    match nb[i][d] {
                        Bc::Fluid(j) => {
                            let c = 0.5 * (dcoef[i] + dcoef[j]) * af / h;
                            mval[dpos[i]] += c;
                            mval[npos[i][d]] -= c;
                        }
                        Bc::Outlet(_) => mval[dpos[i]] += dcoef[i] * af / (h / 2.0),
                        _ => {}
                    }
                }
            }
            if !has_outlet {
                mval[dpos[0]] += mval[dpos[0]].max(1e-300);
            }
            let mm = NodeSparse { bs: 1, ptr: ptr.clone(), adj: adj.clone(), vals: mval.clone() };
            let rhs_c: Vec<f64> = imb.iter().map(|v| -v).collect();
            pc.iter_mut().for_each(|v| *v = 0.0);
            let tp = std::time::Instant::now();
            if last_pcg > 8 || matches!(pre, Pre::None) {
                pre = match (nf <= 20_000).then(|| crate::sparse::Factor::new(&mm, crate::sparse::DIRECT_MAX)).flatten() {
                    Some(f) => Pre::Direct(f?),
                    None => Pre::Mg(crate::multigrid::Multigrid::new(&mm, &coords)?),
                };
            }
            let prec = |r: &[f64]| -> Vec<f64> {
                match &pre {
                    Pre::Direct(f) => f.solve(r),
                    Pre::Mg(m) => m.precondition(r),
                    Pre::None => r.to_vec(),
                }
            };
            last_pcg = pcg_prec(&mm, &rhs_c, &mut pc, 1e-3, 200, &prec)?;
            cg_total += last_pcg;
            t_pressure += tp.elapsed().as_secs_f64();
            // ---- corrections
            for i in 0..nf {
                p[i] += alpha_p * pc[i];
            }
            let gpc = cell_grad(&pc, true);
            uf.par_iter_mut().enumerate().for_each(|(i, ufi)| {
                for d in 0..6 {
                    let up = d % 2 == 1;
                    match nb[i][d] {
                        Bc::Fluid(j) => {
                            let gf = if up { (pc[j] - pc[i]) / h } else { (pc[i] - pc[j]) / h };
                            ufi[d] -= 0.5 * (dcoef[i] + dcoef[j]) * gf;
                        }
                        Bc::Outlet(_) => {
                            let gf = if up { -pc[i] / (h / 2.0) } else { pc[i] / (h / 2.0) };
                            ufi[d] -= dcoef[i] * gf;
                        }
                        _ => {}
                    }
                }
            });
            du = uc
                .par_iter_mut()
                .enumerate()
                .map(|(i, u)| {
                    let mut c = 0.0f64;
                    for q in 0..3 {
                        let nv = ustar[i][q] - dcoef[i] * gpc[i][q];
                        c = c.max((nv - u[q]).abs());
                        u[q] = nv;
                    }
                    c
                })
                .reduce(|| 0.0, f64::max)
                / uscale;
            mres = imb.iter().fold(0.0f64, |m, v| m.max(v.abs())) / (uscale * af);
            if profile && it % 50 == 0 {
                eprintln!("[simple] it {it}: du {du:.2e}, mass {mres:.2e}, dT {t_change_s:.1e}, {:.1} ms/it ({:.1} ms pressure)", 1e3 * t_loop.elapsed().as_secs_f64() / it as f64, 1e3 * t_pressure / it as f64);
            }
            if it % 10 == 0 {
                let conv = (-(du.max(1e-12)).log10()).clamp(0.0, 6.0) / 6.0;
                prog.report((0.97 * conv).max(0.97 * it as f64 / budget as f64).min(0.97), &format!("SIMPLE iteration {it}, change {du:.2e}, continuity {mres:.2e}"))?;
            } else {
                prog.check()?;
            }
            if !(du.is_finite() && mres.is_finite()) {
                stalled = true;
                break;
            }
            if du < 1e-5 && mres < 1e-4 && it > 10 && (!buoyant || t_change_s < 1e-6) {
                simple_ok = true;
                break;
            }
            let worst = du.max(mres * 0.1);
            if worst < 0.5 * best.0 {
                best = (worst, it);
            }
            // stalled (oscillating, unsteady flow): hand over to the projection method
            if scheme != "simple" && it > best.1 + 400 {
                stalled = true;
                break;
            }
        }
        if !simple_ok {
            let why = if du.is_finite() { format!("it did not settle (change {du:.1e}, continuity {mres:.1e} after {simple_iters} iterations)") } else { "it diverged".to_string() };
            if scheme == "simple" || !stalled {
                if !du.is_finite() {
                    bail!("CFD (SIMPLE) diverged; try the projection scheme, a coarser mesh or a lower inlet velocity");
                }
                res.warnings.push(format!("SIMPLE is not fully converged: {why}. Raise max iterations or use the projection scheme for unsteady flows."));
                simple_ok = true; // keep the result
            } else {
                fallback_note = Some(format!("SIMPLE {why}; solved with the projection method instead"));
                uf.iter_mut().for_each(|v| *v = [0.0; 6]);
                uc.iter_mut().for_each(|v| *v = [0.0; 3]);
                p.iter_mut().for_each(|v| *v = 0.0);
                temp.iter_mut().for_each(|v| *v = t_ref);
                last_energy_t.clone_from(&temp);
                nu_eff.iter_mut().for_each(|v| *v = nu);
                nu_wall.iter_mut().for_each(|v| *v = nu);
            }
        }
    }
    while !simple_ok && iters < max_it {
        iters += 1;
        let umax = uc.par_iter().map(|u| u[0].abs().max(u[1].abs()).max(u[2].abs())).reduce(|| 0.0, f64::max);
        let uscale = umax.max(uref).max(1e-9);
        let numax = nu_eff.iter().cloned().fold(nu, f64::max);
        let dt = (0.35 * h / umax.max(uref).max(1e-12)).min(0.15 * h * h / numax);

        // LVEL viscosity update
        if lvel && (iters % 5 == 1) {
            nu_eff.par_iter_mut().zip(nu_wall.par_iter_mut()).enumerate().for_each(|(i, (v, w))| {
                let sp = (uc[i][0].powi(2) + uc[i][1].powi(2) + uc[i][2].powi(2)).sqrt();
                let (local, _) = lvel_law(sp * wall_dist[i] / nu);
                *v = nu * local;
                // wall faces are half a cell from the centre
                *w = nu * lvel_law(sp * 0.5 * h / nu).1;
            });
        }
        // energy (implicit, quasi-steady) every few iterations
        // two-way coupling only with buoyancy; forced convection solves energy once at the end
        if buoyant && (iters % energy_every == 1 || iters == 1) {
            let tb = std::time::Instant::now();
            let its = solve_energy(&uf, &nu_eff, &nu_wall, &mut temp, None)?;
            bicg_total += its;
            if std::env::var_os("SIMCRAFT_PROFILE").is_some() {
                eprintln!("[cht] it {iters}: energy {its} BiCGSTAB its, {:.3} s", tb.elapsed().as_secs_f64());
            }
            let tr = temp.iter().cloned().fold(f64::NEG_INFINITY, f64::max) - temp.iter().cloned().fold(f64::INFINITY, f64::min);
            t_change = temp.iter().zip(&last_energy_t).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max) / tr.max(1e-6);
            last_energy_t.clone_from(&temp);
        }

        // 1. cell tendency
        let temp_ref = &temp;
        ustar.par_iter_mut().enumerate().for_each(|(i, us)| {
            for comp in 0..3 {
                let phi = uc[i][comp];
                let mut adv = 0.0;
                let mut dif = 0.0;
                for d in 0..6 {
                    let out = if d % 2 == 1 { 1.0 } else { -1.0 };
                    let un = uf[i][d] * out; // outward normal velocity
                    let gv = ghost(i, d, comp, &uc);
                    let nuf = match nb[i][d] {
                        Bc::Fluid(j) => 0.5 * (nu_eff[i] + nu_eff[j]),
                        Bc::Wall | Bc::Moving(_) => nu_wall[i],
                        _ => nu_eff[i],
                    };
                    let face_val = match nb[i][d] {
                        Bc::Fluid(_) => {
                            if un.abs() * h <= 2.0 * nuf {
                                0.5 * (phi + gv)
                            } else if un > 0.0 {
                                phi
                            } else {
                                gv
                            }
                        }
                        Bc::Inlet(..) => 0.5 * (phi + gv),
                        _ => phi,
                    };
                    adv += un * (face_val - phi) / h;
                    dif += nuf * (gv - phi) / (h * h);
                }
                let body = if buoyant { -beta * (temp_ref[fluid_elems[i]] - t_ref) * g[comp] } else { 0.0 };
                us[comp] = phi + dt * (-adv + dif + body);
            }
        });
        // 2. provisional face velocities
        uf.par_iter_mut().enumerate().for_each(|(i, ufi)| {
            for d in 0..6 {
                let ax = d / 2;
                ufi[d] = match nb[i][d] {
                    Bc::Fluid(j) => 0.5 * (ustar[i][ax] + ustar[j][ax]),
                    Bc::Wall | Bc::Slip | Bc::Moving(_) => 0.0,
                    Bc::Inlet(v, _) => {
                        if d % 2 == 1 {
                            -v
                        } else {
                            v
                        }
                    }
                    Bc::Outlet(_) => ustar[i][ax],
                };
            }
        });
        // 3. pressure Poisson
        rhs.par_iter_mut().enumerate().for_each(|(i, r)| {
            let div = (uf[i][1] - uf[i][0] + uf[i][3] - uf[i][2] + uf[i][5] - uf[i][4]) / h;
            *r = -(rho * h * h / dt) * div + outlet_rhs[i];
        });
        let tp = std::time::Instant::now();
        cg_total += match (&psolver, &mg) {
            (Some(s), _) => s.solve(&a, &rhs, &mut p, 1e-7, &prog.check_scalar())?,
            (None, Some(m)) => m.solve_cancellable(&rhs, &mut p, 1e-3, 500, prog.cancel)?,
            _ => unreachable!(),
        };
        t_pressure += tp.elapsed().as_secs_f64();
        let snapshot = if profile && iters.is_multiple_of(200) { Some(uc.clone()) } else { None };
        // 4. correct faces, recompute cell velocities
        let k = dt / rho;
        let pr = &p;
        change = uf
            .par_iter_mut()
            .zip(uc.par_iter_mut())
            .enumerate()
            .map(|(i, (ufi, uci))| {
                let mut corr = [0.0f64; 6];
                for d in 0..6 {
                    let sgn = if d % 2 == 1 { 1.0 } else { -1.0 };
                    corr[d] = match nb[i][d] {
                        Bc::Fluid(j) => -k * sgn * (pr[j] - pr[i]) / h,
                        Bc::Outlet(po) => -k * sgn * (po - pr[i]) / (h / 2.0),
                        _ => 0.0,
                    };
                    ufi[d] += corr[d];
                }
                // cell velocity: its own provisional value plus the mean face pressure correction
                // (re-averaging the face velocities would smooth the field like an extra viscosity h²/4Δt)
                let us = ustar[i];
                let new = [us[0] + 0.5 * (corr[0] + corr[1]), us[1] + 0.5 * (corr[2] + corr[3]), us[2] + 0.5 * (corr[4] + corr[5])];
                let mut c = 0.0f64;
                for q in 0..3 {
                    c = c.max((new[q] - uci[q]).abs());
                }
                *uci = new;
                c
            })
            .reduce(|| 0.0, f64::max);
        // relative velocity change per flow-through time of the domain: (Δu/Δt)·L / U²
        change = change / dt * ldomain / (uscale * uscale);
        if let Some(old) = &snapshot {
            let (mut bi, mut bv) = (0usize, 0.0f64);
            for i in 0..nf {
                for q in 0..3 {
                    let d = (uc[i][q] - old[i][q]).abs();
                    if d > bv {
                        bv = d;
                        bi = i;
                    }
                }
            }
            let c = mesh.elem_cell[fluid_elems[bi]];
            eprintln!("[cfd]   max change {bv:.3e} m/s at cell {:?} of {:?}, u = {:?}, ν_eff/ν = {:.1}, bc {:?}", c, mesh.dims, uc[bi], nu_eff[bi] / nu, nb[bi]);
        }
        if profile && iters.is_multiple_of(200) {
            eprintln!("[cfd] it {iters}: {:.1} ms/it ({:.1} ms pressure), dt {dt:.2e}, change {change:.2e}, cg {cg_total}", 1e3 * t_loop.elapsed().as_secs_f64() / iters as f64, 1e3 * t_pressure / iters as f64);
        }
        if iters.is_multiple_of(20) {
            let conv = (1.0f64.ln() - change.max(1e-12).ln()).max(0.0) / (1.0f64.ln() - TOL.ln()) * 10.0;
            let frac = (iters as f64 / max_it as f64).max(1.0 - conv / 10.0).min(0.97);
            let tmsg = if buoyant { format!(", ΔT {:.1e}", t_change) } else { String::new() };
            prog.report(frac, &format!("CFD iteration {iters}, change {change:.2e}{tmsg}"))?;
        } else {
            prog.check()?;
        }
        if !change.is_finite() {
            bail!("CFD diverged; try a coarser mesh or a lower inlet velocity");
        }
        if change < TOL && iters > 30 && (!buoyant || t_change < 1e-5) {
            break;
        }
        // not settling: average the (unsteady) flow over the rest of the run
        if iters * 10 >= max_it * 6 {
            if avg_n == 0 {
                avg_start_time = sim_time;
            }
            avg_n += 1;
            for i in 0..nf {
                for d in 0..6 {
                    avg_uf[i][d] += uf[i][d];
                }
                for q in 0..3 {
                    avg_uc[i][q] += uc[i][q];
                }
                avg_p[i] += p[i];
            }
            if buoyant {
                for (a, t) in avg_t.iter_mut().zip(&temp) {
                    *a += t;
                }
            }
        }
        sim_time += dt;
    }
    let converged = simple_ok || (change < TOL && (!buoyant || t_change < 1e-5));
    if !converged && avg_n > 0 {
        let k = 1.0 / avg_n as f64;
        for i in 0..nf {
            for d in 0..6 {
                uf[i][d] = avg_uf[i][d] * k;
            }
            for q in 0..3 {
                uc[i][q] = avg_uc[i][q] * k;
            }
            p[i] = avg_p[i] * k;
        }
        if buoyant {
            for (t, a) in temp.iter_mut().zip(&avg_t) {
                *t = a * k;
            }
        }
        let span = sim_time - avg_start_time;
        let through = span * uref.max(1e-12) / ldomain;
        res.warnings.push(format!(
            "the flow is unsteady (vortex shedding or plumes; change still {change:.1e} after {iters} iterations): velocities, pressures and temperatures are averaged over the last {avg_n} iterations ({span:.3} s{}). Raise max iterations for a longer average.",
            if uref > 0.0 { format!(", ≈ {through:.1} flow-through times") } else { String::new() }
        ));
        res.note = "time-averaged unsteady flow".into();
    }
    if energy {
        bicg_total += solve_energy(&uf, &nu_eff, &nu_wall, &mut temp, None)?;
        // radiation is non-linear in T: re-linearise until the temperatures settle
        if tbc.values().any(|b| matches!(b, Tbc::Rad(..))) {
            for _ in 0..60 {
                let old = temp.clone();
                bicg_total += solve_energy(&uf, &nu_eff, &nu_wall, &mut temp, None)?;
                if temp.iter().zip(&old).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max) < 1e-4 {
                    break;
                }
            }
        }
    }
    if let Some(n) = &fallback_note {
        res.warnings.push(n.clone());
    }
    res.info.iterations = if simple_ok { simple_iters } else { iters };
    res.info.dofs = nf * 4 + if energy { ne } else { 0 };
    res.info.solver = format!(
        "{} ({}{}), {}{}",
        if simple_ok { format!("SIMPLE, {mom_total} momentum BiCGSTAB iterations") } else { "projection method".into() },
        if lvel { "LVEL turbulence" } else { "laminar" },
        if buoyant { ", buoyancy" } else { "" },
        match (&psolver, &mg) {
            _ if simple_ok => format!("{cg_total} pressure-correction PCG iterations"),
            (_, Some(m)) => format!("{m_levels}-level multigrid CG, {cg_total} pressure iterations", m_levels = m.levels()),
            (Some(_), _) if cg_total > 0 => format!("{cg_total} pressure CG iterations"),
            (Some(s), _) => s.name(),
            _ => String::new(),
        },
        if energy { format!(", energy BiCGSTAB {bicg_total} iterations") } else { String::new() }
    );

    // ---------------------------------------------------------------- forces and flows
    let mut force = [0.0f64; 3];
    for i in 0..nf {
        for d in 0..6 {
            if !matches!(nb[i][d], Bc::Wall | Bc::Moving(_)) {
                continue;
            }
            let e = fluid_elems[i];
            if neighbour(e, d).is_none() {
                continue; // enclosure wall, not a part
            }
            let ax = d / 2;
            let sgn = if d % 2 == 1 { 1.0 } else { -1.0 };
            force[ax] += p[i] * h * h * sgn;
            for comp in 0..3 {
                if comp != ax {
                    force[comp] += rho * nu_wall[i] * h * h * uc[i][comp] / (h / 2.0);
                }
            }
        }
    }
    res.quantity("Force on solids X (drag)", force[0], "N");
    res.quantity("Force on solids Y", force[1], "N");
    res.quantity("Force on solids Z (lift)", force[2], "N");
    let ure = if uref > 0.0 { uref } else { uc.iter().map(|u| (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt()).fold(0.0, f64::max) };
    res.quantity("Reynolds number (reference velocity, part size)", ure * lchar / nu, "");
    let mut key_to_cell: HashMap<(u32, u8), (usize, usize)> = HashMap::new();
    for i in 0..nf {
        for d in 0..6 {
            if matches!(nb[i][d], Bc::Inlet(..) | Bc::Outlet(_)) {
                key_to_cell.insert((fluid_elems[i] as u32, d as u8), (i, d));
            }
        }
    }
    let mut carried = 0.0;
    for (id, label, keys) in &load_faces {
        let (mut q, mut pa, mut n) = (0.0f64, 0.0f64, 0.0f64);
        for k in keys {
            if let Some(&(i, d)) = key_to_cell.get(k) {
                let out = if d % 2 == 1 { 1.0 } else { -1.0 };
                let qf = uf[i][d] * out * h * h;
                q += qf;
                pa += p[i];
                n += 1.0;
                if energy {
                    let tf = match nb[i][d] {
                        Bc::Inlet(_, tin) => tin,
                        _ if qf < 0.0 => t_ref,
                        _ => temp[fluid_elems[i]],
                    };
                    carried += rcp * qf * tf;
                }
            }
        }
        res.reactions.push(Reaction { load_id: *id, label: label.clone(), value: [float(q), float(pa / n.max(1.0)), 0.0] });
        if label == "Velocity Inlet" {
            res.quantity("Inlet pressure", pa / n.max(1.0), "Pa");
        }
    }

    // ---------------------------------------------------------------- energy results
    if energy {
        let generated: f64 = q_elem.iter().sum::<f64>() * h * h * h;
        let kc: Vec<[f64; 3]> = (0..ne).map(|e| k_of(e, &nu_wall)).collect();
        let mut through: Vec<f64> = vec![0.0; model.loads.len()];
        let mut flux_in = 0.0;
        for (&(e, d), bc) in &tbc {
            let e = e as usize;
            match *bc {
                Tbc::Fixed(tw, li) => through[li] += 2.0 * kc[e][d as usize / 2] * h * (tw - temp[e]),
                Tbc::Conv(hc, ta, li) => through[li] += (ta - temp[e]) / (1.0 / (2.0 * kc[e][d as usize / 2] * h) + 1.0 / (hc * a2)),
                Tbc::Rad(eps, ta, li) => through[li] += (ta - temp[e]) / (1.0 / (2.0 * kc[e][d as usize / 2] * h) + 1.0 / (h_rad(eps, temp[e], ta) * a2)),
                Tbc::Flux(q) => flux_in += q * a2,
            }
        }
        let mut boundary_in = flux_in;
        for (li, it) in model.loads.iter().enumerate() {
            if matches!(it.load, Load::Temperature { .. } | Load::Convection { .. } | Load::Radiation { .. }) {
                res.quantity(&format!("Heat flow through #{} ({})", it.id, it.load.label()), through[li], "W");
                boundary_in += through[li];
            }
        }
        if has_inlet {
            res.quantity("Heat carried away by the flow", carried, "W");
        }
        if generated > 0.0 {
            res.quantity("Heat generated", generated, "W");
        }
        let total_in = generated + boundary_in.max(0.0) + if carried < 0.0 { -carried } else { 0.0 };
        let total_out = (if carried > 0.0 { carried } else { 0.0 }) + (-boundary_in).max(0.0);
        if total_in > 0.0 {
            res.quantity("Energy balance (out / in)", 100.0 * total_out / total_in, "%");
        }
        // per-body temperatures
        let mut per: std::collections::BTreeMap<String, (f64, f64, f64)> = std::collections::BTreeMap::new();
        for e in 0..ne {
            if fidx[e] != usize::MAX {
                continue;
            }
            let name = mesh.body_names.get(mesh.elem_body[e] as usize).cloned().unwrap_or_default();
            let x = per.entry(name).or_insert((f64::NEG_INFINITY, 0.0, 0.0));
            x.0 = x.0.max(temp[e]);
            x.1 += temp[e];
            x.2 += 1.0;
        }
        for (b, (mx, s, n)) in per {
            res.quantity(&format!("Max temperature of {b}"), mx, "°C");
            res.quantity(&format!("Mean temperature of {b}"), s / n, "°C");
        }
        let tfl = fluid_elems.iter().map(|&e| temp[e]).fold(f64::NEG_INFINITY, f64::max);
        res.quantity("Max fluid temperature", tfl, "°C");
        // nodal temperature: solid cells win at interfaces so parts read their own temperature
        let nn = mesh.nodes.len();
        let (mut ts, mut cs, mut tf, mut cf) = (vec![0.0; nn], vec![0u32; nn], vec![0.0; nn], vec![0u32; nn]);
        for (e, en) in mesh.elems.iter().enumerate() {
            for &n in en {
                let n = n as usize;
                if fidx[e] == usize::MAX {
                    ts[n] += temp[e];
                    cs[n] += 1;
                } else {
                    tf[n] += temp[e];
                    cf[n] += 1;
                }
            }
        }
        res.temp = Some((0..nn).map(|n| if cs[n] > 0 { ts[n] / cs[n] as f64 } else if cf[n] > 0 { tf[n] / cf[n] as f64 } else { t_ref }).collect());
        if lvel {
            let ratio = nu_eff.iter().cloned().fold(0.0, f64::max) / nu;
            res.quantity("Peak turbulent viscosity ratio ν_eff/ν", ratio, "");
        }
    } else if lvel {
        res.quantity("Peak turbulent viscosity ratio ν_eff/ν", nu_eff.iter().cloned().fold(0.0, f64::max) / nu, "");
    }

    // ---------------------------------------------------------------- nodal flow fields
    let nn = mesh.nodes.len();
    let mut vs = vec![0.0; nn];
    let mut vc = [vec![0.0; nn], vec![0.0; nn], vec![0.0; nn]];
    let mut ps = vec![0.0; nn];
    let mut cnt = vec![0u32; nn];
    for (i, &e) in fluid_elems.iter().enumerate() {
        let v = (uc[i][0].powi(2) + uc[i][1].powi(2) + uc[i][2].powi(2)).sqrt();
        for &n in &mesh.elems[e] {
            let n = n as usize;
            vs[n] += v;
            for c in 0..3 {
                vc[c][n] += uc[i][c];
            }
            ps[n] += p[i];
            cnt[n] += 1;
        }
    }
    for n in 0..nn {
        if cnt[n] > 0 {
            let c = cnt[n] as f64;
            vs[n] /= c;
            ps[n] /= c;
            for v in vc.iter_mut() {
                v[n] /= c;
            }
        }
    }
    res.extra.insert(Field::Velocity, vs);
    let [vx, vy, vz] = vc;
    res.extra.insert(Field::VelocityX, vx);
    res.extra.insert(Field::VelocityY, vy);
    res.extra.insert(Field::VelocityZ, vz);
    res.extra.insert(Field::FluidPressure, ps);
    prog.report(1.0, "done")?;
    Ok(())
}
