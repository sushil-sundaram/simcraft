//! DC current flow in one net of a KiCad board: resistance between two pads, IR drop, current
//! density hot spots, Joule power, and IPC-2221 temperature rise of the tracks that carry it.
//!
//! The net's copper (tracks, arcs, filled zones, pads, via pads) is rasterised per layer with 2×2
//! sub-samples; each pixel is a node of a sheet-resistance network (conductance σ·t between
//! neighbours, harmonic mean of their coverage). Via barrels and plated holes connect the layers
//! (σ·π·d·t_plating / Δz, spread over the via pad's pixels). The source pad is held at 1 V and the
//! sink pad at 0 V; the network is solved with multigrid CG and scaled to the requested current.

use super::*;
use crate::sparse::NodeSparse;
use serde_json::{Value, json};

/// Copper resistivity at 20 °C [Ω·m] and its temperature coefficient [1/K].
const RHO20: f64 = 1.72e-8;
const ALPHA_CU: f64 = 0.00393;
const MAX_PIXELS: usize = 6_000_000;
/// Tracks shorter than this [mm] are neck-downs: IPC-2221's long-track rise overstates their heating.
pub const SHORT_MM: f64 = 2.0;

/// IPC-2221 temperature rise [°C] of a conductor carrying `amps` (cross-section in mm², outer layer
/// or inner layer).
pub fn ipc2221_rise(amps: f64, area_mm2: f64, outer: bool) -> f64 {
    let k = if outer { 0.048 } else { 0.024 };
    let a_mil2 = area_mm2 / (0.0254 * 0.0254);
    (amps.abs() / (k * a_mil2.powf(0.725))).powf(1.0 / 0.44)
}

/// IPC-2221 track width [mm] for a temperature rise `dt` at `amps` with copper thickness `t` mm.
pub fn ipc2221_width(amps: f64, t: f64, dt: f64, outer: bool) -> f64 {
    let k = if outer { 0.048 } else { 0.024 };
    let a_mil2 = (amps.abs() / (k * dt.powf(0.44))).powf(1.0 / 0.725);
    a_mil2 * 0.0254 * 0.0254 / t
}

struct NetPad {
    reference: String,
    number: String,
    layers: Vec<String>,
    geo: PadGeo,
    drill: Option<f64>,
    centre: P2,
}

fn net_name(s: &Sx, table: &BTreeMap<String, String>) -> Option<String> {
    if let Some(n) = s.str_of("net_name") {
        return Some(n.to_string());
    }
    let n = s.get("net")?;
    if let Some(name) = n.atom(2) {
        return Some(name.to_string());
    }
    let a = n.atom(1)?;
    Some(table.get(a).cloned().unwrap_or_else(|| a.to_string()))
}

/// Solve the DC flow of `current` amps from pad `from` to pad `to` ("U1.3" = reference U1, pad 3).
pub fn dc_drop(path: &str, net: Option<&str>, from: &str, to: &str, current: f64, temperature_c: f64) -> Result<Value> {
    let text = std::fs::read_to_string(path).map_err(|e| anyhow!("reading {path}: {e}"))?;
    dc_drop_text(&text, path, net, from, to, current, temperature_c)
}

pub fn dc_drop_text(text: &str, path: &str, net: Option<&str>, from: &str, to: &str, current: f64, temperature_c: f64) -> Result<Value> {
    if !(current > 0.0) {
        bail!("current must be > 0 A");
    }
    let board = PcbData::from_text(text, path)?;
    let root = parse(text)?;
    let table: BTreeMap<String, String> = root.all("net").filter_map(|n| Some((n.atom(1)?.to_string(), n.atom(2).unwrap_or("").to_string()))).collect();
    let layers = &board.layers;
    let nl = layers.len();
    let lidx = |name: &str| -> Vec<usize> {
        if name == "*.Cu" {
            (0..nl).collect()
        } else if name == "F&B.Cu" {
            vec![0, nl - 1]
        } else {
            layers.iter().position(|l| l.0 == name).into_iter().collect()
        }
    };
    // ---- pads (all, to find the terminals and their net)
    let mut pads: Vec<(NetPad, Option<String>)> = Vec::new();
    for fp in root.all("footprint").chain(root.all("module")) {
        let (fx, fy, frot) = fp.xy("at").unwrap_or((0.0, 0.0, 0.0));
        let reference = fp.all("property").find(|p| p.atom(1) == Some("Reference")).and_then(|p| p.atom(2)).or_else(|| fp.all("fp_text").find(|t| t.atom(1) == Some("reference")).and_then(|t| t.atom(2))).unwrap_or("?").to_string();
        for pad in fp.all("pad") {
            let ls: Vec<String> = pad.layers().into_iter().filter(|l| l.ends_with(".Cu")).collect();
            if ls.is_empty() {
                continue;
            }
            let Some(geo) = pad_geometry(fx, fy, frot, pad) else { continue };
            let drill = pad.get("drill").and_then(|d| d.num(1).or_else(|| d.num(2)));
            let (px, py, _) = pad.xy("at").unwrap_or((0.0, 0.0, 0.0));
            let r = rot(px, py, frot);
            pads.push((NetPad { reference: reference.clone(), number: pad.atom(1).unwrap_or("").to_string(), layers: ls, geo, drill, centre: [fx + r[0], fy + r[1]] }, net_name(pad, &table)));
        }
    }
    let find_pad = |spec: &str| -> Result<usize> {
        let (r, n) = spec.rsplit_once('.').ok_or_else(|| anyhow!("pad '{spec}': use REFERENCE.PAD, e.g. U1.3"))?;
        pads.iter().position(|(p, _)| p.reference.eq_ignore_ascii_case(r) && p.number == n).ok_or_else(|| anyhow!("pad '{spec}' not found on the board"))
    };
    let (ia, ib) = (find_pad(from)?, find_pad(to)?);
    let net = match net {
        Some(n) => n.to_string(),
        None => pads[ia].1.clone().filter(|n| !n.is_empty()).ok_or_else(|| anyhow!("pad {from} has no net: give 'net'"))?,
    };
    for (i, spec) in [(ia, from), (ib, to)] {
        if let Some(n) = &pads[i].1 {
            if !n.is_empty() && n != &net {
                bail!("pad {spec} is on net '{n}', not '{net}'");
            }
        }
    }
    // ---- copper of the net
    let mut tracks: Vec<(String, P2, P2, f64)> = Vec::new();
    for s in root.all("segment") {
        if net_name(s, &table).as_deref() == Some(net.as_str()) {
            if let (Some((x0, y0, _)), Some((x1, y1, _)), Some(w)) = (s.xy("start"), s.xy("end"), s.get("width").and_then(|w| w.num(1))) {
                tracks.push((s.str_of("layer").unwrap_or("F.Cu").to_string(), [x0, y0], [x1, y1], w));
            }
        }
    }
    for s in root.all("arc") {
        if net_name(s, &table).as_deref() == Some(net.as_str()) {
            if let (Some((a0, a1, _)), Some((m0, m1, _)), Some((b0, b1, _)), Some(w)) = (s.xy("start"), s.xy("mid"), s.xy("end"), s.get("width").and_then(|w| w.num(1))) {
                for p in arc_points([a0, a1], [m0, m1], [b0, b1]).windows(2) {
                    tracks.push((s.str_of("layer").unwrap_or("F.Cu").to_string(), p[0], p[1], w));
                }
            }
        }
    }
    let mut vias: Vec<(P2, f64, f64, usize, usize)> = Vec::new();
    for v in root.all("via") {
        if net_name(v, &table).as_deref() != Some(net.as_str()) {
            continue;
        }
        if let (Some((x, y, _)), Some(d)) = (v.xy("at"), v.get("size").and_then(|s| s.num(1))) {
            let drill = v.get("drill").and_then(|s| s.num(1)).unwrap_or(d * 0.5);
            let span: Vec<usize> = v.layers().iter().flat_map(|l| lidx(l)).collect();
            let (l0, l1) = (span.iter().copied().min().unwrap_or(0), span.iter().copied().max().unwrap_or(nl - 1));
            vias.push(([x, y], d, drill, l0, l1));
        }
    }
    let mut zones: Vec<(String, Vec<P2>)> = Vec::new();
    for z in root.all("zone") {
        if net_name(z, &table).as_deref() != Some(net.as_str()) {
            continue;
        }
        for fp in z.all("filled_polygon") {
            let layer = fp.str_of("layer").map(String::from).or_else(|| z.layers().first().cloned()).unwrap_or_default();
            let pts: Vec<P2> = fp.get("pts").map(|p| p.all("xy").filter_map(|q| Some([q.num(1)?, q.num(2)?])).collect()).unwrap_or_default();
            zones.push((layer, pts));
        }
    }
    let net_pads: Vec<usize> = (0..pads.len()).filter(|&i| i == ia || i == ib || pads[i].1.as_deref() == Some(net.as_str())).collect();
    // ---- raster frame around the net's copper
    let mut lo = [f64::INFINITY; 2];
    let mut hi = [f64::NEG_INFINITY; 2];
    let mut grow = |p: P2, r: f64| {
        lo = [lo[0].min(p[0] - r), lo[1].min(p[1] - r)];
        hi = [hi[0].max(p[0] + r), hi[1].max(p[1] + r)];
    };
    let mut wmin = f64::INFINITY;
    for t in &tracks {
        grow(t.1, t.3);
        grow(t.2, t.3);
        wmin = wmin.min(t.3);
    }
    for z in &zones {
        z.1.iter().for_each(|p| grow(*p, 0.0));
    }
    for v in &vias {
        grow(v.0, v.1);
        wmin = wmin.min(v.1);
    }
    for &i in &net_pads {
        match &pads[i].0.geo {
            PadGeo::Poly(p) => p.iter().for_each(|q| grow(*q, 0.0)),
            PadGeo::Capsule(a, b, w) => {
                grow(*a, *w);
                grow(*b, *w);
                wmin = wmin.min(*w);
            }
        }
    }
    if !wmin.is_finite() {
        wmin = 0.5;
    }
    let (sx, sy) = (hi[0] - lo[0], hi[1] - lo[1]);
    let mut res = (wmin / 4.0).clamp(0.02, 0.25);
    while (sx / res) * (sy / res) * nl as f64 > MAX_PIXELS as f64 {
        res *= 1.25;
    }
    let (nx, ny) = ((sx / res).ceil() as usize + 2, (sy / res).ceil() as usize + 2);
    let (x0, y0) = (lo[0] - res, lo[1] - res);
    let new_r = || Raster { nx, ny, res, x0, y0, bits: vec![0u8; nx * ny] };
    let mut cu: Vec<Raster> = (0..nl).map(|_| new_r()).collect();
    for (layer, a, b, w) in &tracks {
        for li in lidx(layer) {
            cu[li].capsule(*a, *b, *w);
        }
    }
    for (layer, pts) in &zones {
        for li in lidx(layer) {
            cu[li].polygon(pts);
        }
    }
    let paint = |r: &mut Raster, geo: &PadGeo| match geo {
        PadGeo::Poly(p) => r.polygon(p),
        PadGeo::Capsule(a, b, w) => r.capsule(*a, *b, *w),
    };
    for &i in &net_pads {
        for l in pads[i].0.layers.iter().flat_map(|l| lidx(l)) {
            paint(&mut cu[l], &pads[i].0.geo);
        }
    }
    for v in &vias {
        for r in cu.iter_mut().take(v.4 + 1).skip(v.3) {
            r.capsule(v.0, v.0, v.1);
        }
    }
    // terminal masks
    let term = |i: usize| -> Vec<Vec<bool>> {
        let ls: Vec<usize> = pads[i].0.layers.iter().flat_map(|l| lidx(l)).collect();
        (0..nl)
            .map(|l| {
                if !ls.contains(&l) {
                    return vec![false; nx * ny];
                }
                let mut r = new_r();
                paint(&mut r, &pads[i].0.geo);
                r.bits.iter().map(|b| *b != 0).collect()
            })
            .collect()
    };
    let (src, snk) = (term(ia), term(ib));
    // ---- network
    let rho = RHO20 * (1.0 + ALPHA_CU * (temperature_c - 20.0));
    let sigma = 1.0 / rho;
    let gsheet: Vec<f64> = layers.iter().map(|l| sigma * l.1 * 1e-3).collect();
    let cov = |l: usize, k: usize| cu[l].bits[k].count_ones() as f64 / 4.0;
    let np = nx * ny;
    let mut id = vec![u32::MAX; nl * np];
    let mut coords: Vec<[u32; 3]> = Vec::new();
    for l in 0..nl {
        for k in 0..np {
            if cu[l].bits[k] != 0 {
                id[l * np + k] = coords.len() as u32;
                coords.push([(k % nx) as u32, (k / nx) as u32, l as u32]);
            }
        }
    }
    let n = coords.len();
    let mut edges: Vec<(u32, u32, f64)> = Vec::new();
    for l in 0..nl {
        for j in 0..ny {
            for i in 0..nx {
                let k = i + j * nx;
                let a = id[l * np + k];
                if a == u32::MAX {
                    continue;
                }
                for (di, dj) in [(1usize, 0usize), (0, 1)] {
                    if i + di >= nx || j + dj >= ny {
                        continue;
                    }
                    let k2 = (i + di) + (j + dj) * nx;
                    let b = id[l * np + k2];
                    if b != u32::MAX {
                        let (f1, f2) = (cov(l, k), cov(l, k2));
                        edges.push((a, b, gsheet[l] * 2.0 * f1 * f2 / (f1 + f2)));
                    }
                }
            }
        }
    }
    let dz = if nl > 1 { board.thickness * 1e-3 / (nl - 1) as f64 } else { 1.0 };
    // barrels: via pads and plated pad holes; (centre, pad diameter, drill, l0, l1)
    let mut barrels: Vec<(P2, f64, f64, usize, usize)> = vias.clone();
    for &i in &net_pads {
        if let Some(d) = pads[i].0.drill {
            let ls: Vec<usize> = pads[i].0.layers.iter().flat_map(|l| lidx(l)).collect();
            if ls.len() > 1 {
                let w = match &pads[i].0.geo {
                    PadGeo::Capsule(_, _, w) => *w,
                    PadGeo::Poly(_) => d * 1.5,
                };
                barrels.push((pads[i].0.centre, w.max(d), d, *ls.iter().min().unwrap(), *ls.iter().max().unwrap()));
            }
        }
    }
    let mut barrel_edges: Vec<Vec<(usize, usize)>> = Vec::new(); // per barrel: edge index ranges per layer pair
    for (c, d, drill, l0, l1) in &barrels {
        let mut ranges = Vec::new();
        let r = (d / 2.0).max(0.75 * res);
        let (i0, i1) = (((c[0] - r - x0) / res).floor().max(0.0) as usize, (((c[0] + r - x0) / res).ceil() as usize).min(nx - 1));
        let (j0, j1) = (((c[1] - r - y0) / res).floor().max(0.0) as usize, (((c[1] + r - y0) / res).ceil() as usize).min(ny - 1));
        let ring: Vec<usize> = (j0..=j1)
            .flat_map(|j| (i0..=i1).map(move |i| (i, j)))
            .filter(|&(i, j)| {
                let p = [x0 + (i as f64 + 0.5) * res, y0 + (j as f64 + 0.5) * res];
                (p[0] - c[0]).powi(2) + (p[1] - c[1]).powi(2) <= r * r
            })
            .map(|(i, j)| i + j * nx)
            .collect();
        for l in *l0..*l1 {
            let pairs: Vec<(u32, u32)> = ring.iter().filter_map(|&k| {
                let (a, b) = (id[l * np + k], id[(l + 1) * np + k]);
                (a != u32::MAX && b != u32::MAX).then_some((a, b))
            }).collect();
            if pairs.is_empty() {
                continue;
            }
            let g = sigma * std::f64::consts::PI * drill * 1e-3 * PLATING * 1e-3 / dz / pairs.len() as f64;
            let s0 = edges.len();
            for (a, b) in pairs {
                edges.push((a, b, g));
            }
            ranges.push((s0, edges.len()));
        }
        barrel_edges.push(ranges);
    }
    // ---- terminals and connectivity
    let mut fixed: Vec<Option<f64>> = vec![None; n];
    for l in 0..nl {
        for k in 0..np {
            let a = id[l * np + k];
            if a == u32::MAX {
                continue;
            }
            if src[l][k] {
                fixed[a as usize] = Some(1.0);
            } else if snk[l][k] {
                fixed[a as usize] = Some(0.0);
            }
        }
    }
    if !fixed.iter().any(|f| *f == Some(1.0)) || !fixed.iter().any(|f| *f == Some(0.0)) {
        bail!("pads {from} and {to} have no copper of net '{net}' on their layers");
    }
    let mut nbr: Vec<Vec<(u32, f64)>> = vec![Vec::new(); n];
    for &(a, b, g) in &edges {
        nbr[a as usize].push((b, g));
        nbr[b as usize].push((a, g));
    }
    let mut seen = vec![false; n];
    let mut stack: Vec<usize> = (0..n).filter(|&i| fixed[i] == Some(1.0)).collect();
    stack.iter().for_each(|&i| seen[i] = true);
    while let Some(i) = stack.pop() {
        for &(j, _) in &nbr[i] {
            if !seen[j as usize] {
                seen[j as usize] = true;
                stack.push(j as usize);
            }
        }
    }
    if !(0..n).any(|i| seen[i] && fixed[i] == Some(0.0)) {
        bail!("pads {from} and {to} are not connected by copper of net '{net}' (check unfilled zones or missing tracks)");
    }
    // ---- linear system on the connected copper (Dirichlet rows eliminated symmetrically)
    let mut map = vec![u32::MAX; n];
    let mut nodes: Vec<usize> = Vec::new();
    for i in 0..n {
        if seen[i] {
            map[i] = nodes.len() as u32;
            nodes.push(i);
        }
    }
    let m = nodes.len();
    let mut ptr = vec![0usize];
    let mut adj: Vec<u32> = Vec::new();
    let mut vals: Vec<f64> = Vec::new();
    let mut rhs = vec![0.0; m];
    for (r, &i) in nodes.iter().enumerate() {
        let mut row: Vec<(u32, f64)> = vec![(r as u32, 0.0)];
        if let Some(v) = fixed[i] {
            row[0].1 = 1.0;
            rhs[r] = v;
        } else {
            for &(j, g) in &nbr[i] {
                row[0].1 += g;
                match fixed[j as usize] {
                    Some(v) => rhs[r] += g * v,
                    None => row.push((map[j as usize], -g)),
                }
            }
        }
        row.sort_unstable_by_key(|e| e.0);
        let mut merged: Vec<(u32, f64)> = Vec::with_capacity(row.len());
        for e in row {
            match merged.last_mut() {
                Some(last) if last.0 == e.0 => last.1 += e.1,
                _ => merged.push(e),
            }
        }
        for (c, v) in merged {
            adj.push(c);
            vals.push(v);
        }
        ptr.push(adj.len());
    }
    let a = NodeSparse { bs: 1, ptr, adj, vals };
    let mut vsol: Vec<f64> = (0..m).map(|r| fixed[nodes[r]].unwrap_or(0.5)).collect();
    // layers are coupled only through via barrels: keep them apart in the aggregation (the layer
    // coordinate is spread so 2×2×2 blocks never merge two layers)
    let crd: Vec<[u32; 3]> = nodes.iter().map(|&i| [coords[i][0], coords[i][1], coords[i][2] << 16]).collect();
    let mg = crate::multigrid::Multigrid::new(&a, &crd)?;
    let its = mg.solve(&rhs, &mut vsol, 1e-9, 3000)?;
    let mut v = vec![0.0; n];
    for (r, &i) in nodes.iter().enumerate() {
        v[i] = vsol[r];
    }
    // ---- results (1 V solution → scale to the requested current)
    let i1: f64 = (0..n).filter(|&i| seen[i] && fixed[i] == Some(1.0)).map(|i| nbr[i].iter().filter(|(j, _)| fixed[*j as usize] != Some(1.0)).map(|(j, g)| g * (1.0 - v[*j as usize])).sum::<f64>()).sum();
    if !(i1 > 0.0) {
        bail!("no current flows between {from} and {to}");
    }
    let r_ohm = 1.0 / i1;
    let vs = current * r_ohm; // volts per unit solution
    let to_board = |p: P2| [p[0] - board.frame[0], board.frame[1] - p[1]];
    // sheet current vector K = −G□ ∇V [A/m] per pixel
    let kvec = |l: usize, i: usize, j: usize| -> [f64; 2] {
        let at = |ii: usize, jj: usize| -> Option<f64> {
            let a = id[l * np + ii + jj * nx];
            (a != u32::MAX && seen[a as usize]).then(|| v[a as usize])
        };
        let Some(c) = at(i, j) else { return [0.0; 2] };
        let d = |p: Option<f64>, q: Option<f64>| match (p, q) {
            (Some(p), Some(q)) => (q - p) / (2.0 * res * 1e-3),
            (Some(p), None) => (c - p) / (res * 1e-3),
            (None, Some(q)) => (q - c) / (res * 1e-3),
            _ => 0.0,
        };
        let gx = d(if i > 0 { at(i - 1, j) } else { None }, if i + 1 < nx { at(i + 1, j) } else { None });
        let gy = d(if j > 0 { at(i, j - 1) } else { None }, if j + 1 < ny { at(i, j + 1) } else { None });
        // a partly covered edge pixel carries current only in its copper share
        let f = cov(l, i + j * nx);
        [-gsheet[l] * f * vs * gx, -gsheet[l] * f * vs * gy]
    };
    let kmag = |l: usize, i: usize, j: usize| -> f64 {
        let k = kvec(l, i, j);
        (k[0] * k[0] + k[1] * k[1]).sqrt()
    };
    let mut jmax = (0.0f64, 0usize, [0.0; 2]);
    for l in 0..nl {
        for j in 0..ny {
            for i in 0..nx {
                let k = i + j * nx;
                let a = id[l * np + k];
                // skip pixels inside the terminal pads (equipotential by construction)
                if a == u32::MAX || !seen[a as usize] || fixed[a as usize].is_some() || cov(l, k) < 1.0 {
                    continue;
                }
                let jd = kmag(l, i, j) / (layers[l].1 * 1e-3) / 1e6; // A/mm²
                if jd > jmax.0 {
                    jmax = (jd, l, [x0 + (i as f64 + 0.5) * res, y0 + (j as f64 + 0.5) * res]);
                }
            }
        }
    }
    let outer = |l: usize| l == 0 || l + 1 == nl;
    let sample = |l: usize, p: P2| -> [f64; 2] {
        let i = (((p[0] - x0) / res).floor().max(0.0) as usize).min(nx - 1);
        let j = (((p[1] - y0) / res).floor().max(0.0) as usize).min(ny - 1);
        kvec(l, i, j)
    };
    // current through a track: K·t̂ integrated across its width at the middle of the segment
    let through = |l: usize, a: P2, b: P2, w: f64| -> f64 {
        let len = ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt().max(1e-12);
        let t = [(b[0] - a[0]) / len, (b[1] - a[1]) / len];
        let nrm = [-t[1], t[0]];
        let m = [(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0];
        let n = ((w / (res / 2.0)).ceil() as usize).max(4);
        let ds = w / n as f64;
        (0..n)
            .map(|q| {
                let s = -w / 2.0 + (q as f64 + 0.5) * ds;
                let k = sample(l, [m[0] + s * nrm[0], m[1] + s * nrm[1]]);
                (k[0] * t[0] + k[1] * t[1]) * ds * 1e-3
            })
            .sum::<f64>()
            .abs()
    };
    let mut segs: Vec<Value> = Vec::new();
    for (layer, a, b, w) in &tracks {
        let len = ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt();
        if len < (*w).max(2.0 * res) {
            continue; // stubs shorter than their width carry no meaningful "track current"
        }
        for l in lidx(layer) {
            let amps = through(l, *a, *b, *w);
            let t = layers[l].1;
            segs.push(json!({
                "layer": layer, "width_mm": w, "length_mm": crate::results::float(len),
                "from_mm": to_board(*a).map(crate::results::float), "to_mm": to_board(*b).map(crate::results::float),
                "current_a": crate::results::float(amps),
                "current_density_a_per_mm2": crate::results::float(amps / (w * t)),
                "ipc2221_temperature_rise_c": crate::results::float(ipc2221_rise(amps, w * t, outer(l))),
                "ipc2221_width_for_10c_mm": crate::results::float(ipc2221_width(amps, t, 10.0, outer(l))),
            }));
        }
    }
    segs.sort_by(|p, q| q["current_a"].as_f64().unwrap_or(0.0).total_cmp(&p["current_a"].as_f64().unwrap_or(0.0)));
    segs.truncate(10);
    // via currents: max over the layer pairs of each barrel
    let mut via_list: Vec<(f64, P2)> = barrels
        .iter()
        .zip(&barrel_edges)
        .map(|(b, ranges)| {
            let imax = ranges.iter().map(|&(s0, s1)| edges[s0..s1].iter().map(|&(p, q, g)| g * (v[p as usize] - v[q as usize])).sum::<f64>().abs()).fold(0.0, f64::max);
            (imax * vs, b.0)
        })
        .collect();
    via_list.sort_by(|p, q| q.0.total_cmp(&p.0));
    // IPC-2221 is for long tracks: a neck-down of a few mm is cooled by the copper around it
    let worst_rise = segs.iter().filter(|s| s["length_mm"].as_f64().unwrap_or(0.0) >= SHORT_MM).filter_map(|s| s["ipc2221_temperature_rise_c"].as_f64()).fold(0.0, f64::max);
    let mut warnings = board.warnings.clone();
    if res > wmin / 2.5 {
        warnings.push(format!("raster {res:.3} mm is coarse for the narrowest copper ({wmin:.3} mm): results for thin tracks are approximate"));
    }
    if worst_rise > 20.0 {
        warnings.push(format!("a track would heat by ~{worst_rise:.0} °C (IPC-2221): widen it or add parallel copper"));
    }
    Ok(json!({
        "net": net,
        "from": from,
        "to": to,
        "current_a": current,
        "copper_temperature_c": temperature_c,
        "resistance_mohm": crate::results::float(r_ohm * 1e3),
        "voltage_drop_mv": crate::results::float(current * r_ohm * 1e3),
        "power_w": crate::results::float(current * current * r_ohm),
        "max_current_density_a_per_mm2": crate::results::float(jmax.0),
        "max_current_density_at_mm": to_board(jmax.2).map(crate::results::float),
        "max_current_density_layer": layers[jmax.1].0,
        "tracks_by_current": segs,
        "vias_and_holes": barrels.len(),
        "max_via_current_a": crate::results::float(via_list.first().map(|v| v.0).unwrap_or(0.0)),
        "max_via_at_mm": via_list.first().map(|v| to_board(v.1).map(crate::results::float)),
        "raster_mm": crate::results::float(res),
        "unknowns": m,
        "solver_iterations": its,
        "warnings": warnings,
        "note": "DC sheet-resistance model of the net's copper (tracks, zones, pads, via barrels); track current = sheet current integrated across the track at its middle; IPC-2221 temperature rise per track for the current it carries alone (long tracks; segments shorter than 2 mm are neck-downs cooled by the copper around them).",
    }))
}

/// Nets of a board with their pads ("REF.PAD"), for choosing `pcb_dc` terminals. Nets with fewer
/// than two pads are left out; sorted by name.
pub fn net_pads(path: &str) -> Result<Vec<(String, Vec<String>)>> {
    let text = std::fs::read_to_string(path).map_err(|e| anyhow!("reading {path}: {e}"))?;
    net_pads_text(&text)
}

pub fn net_pads_text(text: &str) -> Result<Vec<(String, Vec<String>)>> {
    let root = parse(text)?;
    let table: BTreeMap<String, String> = root.all("net").filter_map(|n| Some((n.atom(1)?.to_string(), n.atom(2).unwrap_or("").to_string()))).collect();
    let mut nets: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for fp in root.all("footprint").chain(root.all("module")) {
        let reference = fp.all("property").find(|p| p.atom(1) == Some("Reference")).and_then(|p| p.atom(2)).or_else(|| fp.all("fp_text").find(|t| t.atom(1) == Some("reference")).and_then(|t| t.atom(2))).unwrap_or("?").to_string();
        for pad in fp.all("pad") {
            if !pad.layers().iter().any(|l| l.ends_with(".Cu")) {
                continue;
            }
            if let Some(n) = net_name(pad, &table).filter(|n| !n.is_empty()) {
                nets.entry(n).or_default().push(format!("{reference}.{}", pad.atom(1).unwrap_or("")));
            }
        }
    }
    let mut out: Vec<(String, Vec<String>)> = nets.into_iter().filter(|(_, p)| p.len() >= 2).collect();
    for (_, p) in out.iter_mut() {
        p.sort_by(|a, b| natural(a).cmp(&natural(b)));
        p.dedup();
    }
    Ok(out)
}

/// Sort key that orders "U2.10" after "U2.9".
fn natural(s: &str) -> Vec<(String, u64)> {
    let mut out = Vec::new();
    let mut text = String::new();
    let mut num = String::new();
    for ch in s.chars() {
        if ch.is_ascii_digit() {
            num.push(ch);
        } else {
            if !num.is_empty() {
                out.push((std::mem::take(&mut text), num.parse().unwrap_or(0)));
                num.clear();
            }
            text.push(ch);
        }
    }
    out.push((text, num.parse().unwrap_or(0)));
    out
}
