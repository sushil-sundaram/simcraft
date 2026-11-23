//! Parameter studies and design optimization.
//!
//! * Parameters point at numbers in the model with readable targets: `Bracket.size.z`,
//!   `Hole.radius`, `load2.vector.z`, `element_size`.
//! * Responses are results: `mass`, `volume`, `max:von_mises`, `min:safety_factor`,
//!   `frequency:1`, `load_multiplier:1`, `reaction:1:z`, `quantity:Capacitance`.
//! * A study evaluates designs on copies of the model (the user's model is never changed until a
//!   design is applied). Optimization = Latin-hypercube exploration followed by a bounded
//!   Nelder–Mead search with quadratic penalties for constraints.

use crate::fem;
use crate::geometry::{Shape, load_stl};
use crate::mesh;
use crate::model::{Load, Model};
use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::sync::Arc;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Parameter {
    pub target: String,
    pub min: f64,
    pub max: f64,
    /// Grid points for a parameter study (default 5).
    #[serde(default)]
    pub steps: Option<usize>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Objective {
    pub response: String,
    /// "minimize" or "maximize"
    #[serde(default = "minimize")]
    pub goal: String,
}
fn minimize() -> String {
    "minimize".into()
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Constraint {
    pub response: String,
    /// "<=" or ">="
    pub op: String,
    pub value: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
pub struct Run {
    pub values: Vec<f64>,
    pub responses: Vec<Option<f64>>,
    #[serde(default)]
    pub feasible: bool,
    #[serde(default)]
    pub error: Option<String>,
    /// "doe" or "search"
    #[serde(default)]
    pub phase: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
pub struct Study {
    /// "parameter_study", "optimization" or "topology"
    pub kind: String,
    pub parameters: Vec<Parameter>,
    pub responses: Vec<String>,
    #[serde(default)]
    pub objective: Option<Objective>,
    #[serde(default)]
    pub constraints: Vec<Constraint>,
    pub runs: Vec<Run>,
    #[serde(default)]
    pub best: Option<usize>,
    #[serde(default)]
    pub message: String,
    /// Topology optimization: exported shape.
    #[serde(default)]
    pub stl_path: Option<String>,
    /// Topology optimization: (iteration, compliance [J], volume fraction).
    #[serde(default)]
    pub history: Vec<(usize, f64, f64)>,
    #[serde(default)]
    pub seconds: f64,
}

// ------------------------------------------------------------------ parameter targets

fn shape_fields(shape: &Shape) -> Vec<String> {
    let v3 = |n: &str| ["x", "y", "z"].iter().map(|c| format!("{n}.{c}")).collect::<Vec<_>>();
    match shape {
        Shape::Box { .. } => [v3("min"), v3("size")].concat(),
        Shape::Cylinder { .. } => [v3("base"), vec!["radius".into(), "height".into()]].concat(),
        Shape::Sphere { .. } => [v3("center"), vec!["radius".into()]].concat(),
        Shape::Stl { .. } => [vec!["scale".into()], v3("offset")].concat(),
        Shape::Pcb { .. } => v3("offset"),
    }
}

fn load_fields(l: &Load) -> Vec<String> {
    let v3 = |n: &str| ["x", "y", "z"].iter().map(|c| format!("{n}.{c}")).collect::<Vec<_>>();
    match l {
        Load::Displacement { .. } => vec!["x".into(), "y".into(), "z".into()],
        Load::Force { .. } => v3("vector"),
        Load::Pressure { .. } => vec!["value_mpa".into()],
        Load::Gravity { .. } => v3("acceleration"),
        Load::Temperature { .. } | Load::HeatFlux { .. } | Load::HeatGeneration { .. } | Load::Voltage { .. } | Load::Current { .. } => vec!["value".into()],
        Load::Convection { .. } => vec!["coefficient".into(), "ambient".into()],
        Load::Coil { .. } => vec!["current_density".into()],
        Load::Magnet { .. } => v3("remanence"),
        Load::Inlet { .. } => vec!["velocity".into()],
        Load::Outlet { .. } => vec!["pressure".into()],
        _ => vec![],
    }
}

/// Every numeric target in the model, with its current value.
pub fn list_targets(model: &Model) -> Vec<(String, f64)> {
    let mut out = Vec::new();
    for b in &model.bodies {
        for f in shape_fields(&b.shape) {
            let t = format!("{}.{f}", b.name);
            if let Ok(v) = get_value(model, &t) {
                out.push((t, v));
            }
        }
    }
    for l in &model.loads {
        for f in load_fields(&l.load) {
            let t = format!("load{}.{f}", l.id);
            out.push((t.clone(), get_value(model, &t).unwrap_or(0.0)));
        }
    }
    out.push(("element_size".into(), model.element_size.unwrap_or_else(|| mesh::auto_size(&model.bodies, model.mesh_with_enclosure()))));
    out
}

/// JSON path (into the serialized model) for a readable target.
fn path_of(model: &Model, target: &str) -> Result<Vec<Value>> {
    let t = target.trim();
    if t.eq_ignore_ascii_case("element_size") {
        return Ok(vec![json!("element_size")]);
    }
    let comp = |s: &str| -> Option<usize> {
        match s {
            "x" | "0" => Some(0),
            "y" | "1" => Some(1),
            "z" | "2" => Some(2),
            _ => None,
        }
    };
    let parts: Vec<&str> = t.split('.').collect();
    if let Some(id) = parts[0].strip_prefix("load").and_then(|n| n.trim_start_matches('#').parse::<u32>().ok()) {
        let j = model.loads.iter().position(|l| l.id == id).ok_or_else(|| anyhow!("no load with id {id}"))?;
        let field = parts.get(1).ok_or_else(|| anyhow!("target '{t}' needs a field, e.g. load{id}.vector.z"))?;
        if !load_fields(&model.loads[j].load).iter().any(|f| f.split('.').next() == Some(*field)) {
            bail!("load {id} ({}) has no numeric field '{field}'; options: {:?}", model.loads[j].load.label(), load_fields(&model.loads[j].load));
        }
        let mut p = vec![json!("loads"), json!(j), json!(field)];
        if let Some(c) = parts.get(2) {
            p.push(json!(comp(c).ok_or_else(|| anyhow!("component must be x, y or z"))?));
        }
        return Ok(p);
    }
    let i = model.bodies.iter().position(|b| b.name.eq_ignore_ascii_case(parts[0])).ok_or_else(|| {
        anyhow!("'{}' is not a body, load (loadN) or element_size; bodies: {:?}", parts[0], model.bodies.iter().map(|b| b.name.clone()).collect::<Vec<_>>())
    })?;
    let field = parts.get(1).ok_or_else(|| anyhow!("target '{t}' needs a field, e.g. {}.size.z", parts[0]))?;
    let fields = shape_fields(&model.bodies[i].shape);
    if !fields.iter().any(|f| f.split('.').next() == Some(*field)) {
        bail!("body '{}' has no field '{field}'; options: {fields:?}", model.bodies[i].name);
    }
    let mut p = vec![json!("bodies"), json!(i), json!("shape"), json!(field)];
    if let Some(c) = parts.get(2) {
        p.push(json!(comp(c).ok_or_else(|| anyhow!("component must be x, y or z"))?));
    }
    Ok(p)
}

fn walk<'a>(v: &'a mut Value, path: &[Value]) -> Option<&'a mut Value> {
    let mut cur = v;
    for k in path {
        cur = match k {
            Value::String(s) => cur.get_mut(s.as_str())?,
            Value::Number(n) => cur.get_mut(n.as_u64()? as usize)?,
            _ => return None,
        };
    }
    Some(cur)
}

pub fn get_value(model: &Model, target: &str) -> Result<f64> {
    let path = path_of(model, target)?;
    let mut v = serde_json::to_value(model)?;
    walk(&mut v, &path).and_then(|x| x.as_f64()).ok_or_else(|| anyhow!("target '{target}' has no numeric value"))
}

/// Copy of `model` with the parameters set to `values`.
pub fn apply(model: &Model, params: &[Parameter], values: &[f64]) -> Result<Model> {
    let mut v = serde_json::to_value(model)?;
    for (p, x) in params.iter().zip(values) {
        let path = path_of(model, &p.target)?;
        let slot = walk(&mut v, &path).ok_or_else(|| anyhow!("cannot set '{}'", p.target))?;
        *slot = json!(x);
    }
    let mut m: Model = serde_json::from_value(v).context("applying parameters")?;
    // STL data is not serialized: carry it over, reloading when scale/offset changed
    for (nb, ob) in m.bodies.iter_mut().zip(&model.bodies) {
        if let (Shape::Stl { path, scale, offset, data }, Shape::Stl { scale: os, offset: oo, data: od, .. }) = (&mut nb.shape, &ob.shape) {
            *data = if scale == os && offset == oo { od.clone() } else { Some(Arc::new(load_stl(path, *scale, *offset)?)) };
        }
    }
    Ok(m)
}

// ------------------------------------------------------------------ responses

/// Responses that make sense for the model's current analysis.
pub fn suggested_responses(model: &Model) -> Vec<String> {
    use crate::model::AnalysisType::*;
    let mut r = vec!["mass".to_string(), "volume".into()];
    let k = model.analysis.kind;
    let f = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
    r.extend(match k {
        StaticStructural | TransientStructural | NonlinearStructural | ThermalStress => f(&["max:von_mises", "min:safety_factor", "max:total_deformation", "min:deformation_z", "max:max_principal"]),
        Modal => f(&["frequency:1", "frequency:2", "frequency:3"]),
        LinearBuckling => f(&["load_multiplier:1", "max:von_mises"]),
        HarmonicResponse => f(&["quantity:Peak displacement amplitude", "quantity:Peak response frequency", "max:von_mises"]),
        ResponseSpectrum => f(&["max:total_deformation", "max:von_mises", "quantity:Cumulative effective mass"]),
        RandomVibration => f(&["quantity:3σ max displacement", "quantity:3σ max von Mises", "quantity:Zero-crossing frequency"]),
        SteadyThermal | TransientThermal => f(&["max:temperature", "min:temperature", "max:heat_flux"]),
        Electrostatic => f(&["quantity:Capacitance (from energy)", "max:electric_field"]),
        CurrentFlow | ThermalElectric => f(&["quantity:Resistance", "quantity:Total Joule heating power", "max:current_density", "max:temperature"]),
        Magnetostatic => f(&["quantity:Peak |B| in solids", "quantity:Stored magnetic energy", "max:magnetic_flux_density"]),
        FullWaveEm => f(&["quantity:Absorbed power", "quantity:Absorbed fraction", "max:electric_field", "max:absorbed_power"]),
        MicrowaveHeating => f(&["max:temperature", "min:temperature", "quantity:Absorbed power", "quantity:Temperature spread"]),
        ConjugateHeatTransfer => f(&["max:temperature", "quantity:Heat carried away by the flow", "max:velocity", "quantity:Inlet pressure"]),
        Fluid => f(&["quantity:Force on solids X (drag)", "max:velocity", "max:fluid_pressure"]),
    });
    for l in model.loads.iter().filter(|l| l.load.is_support()) {
        r.push(format!("reaction:{}", l.id));
    }
    r
}

/// Extract a response from a solved design (summary JSON from Results::summary).
pub fn response(summary: &Value, mass: f64, volume_mm3: f64, key: &str) -> Result<f64> {
    let k = key.trim();
    let bad = || anyhow!("unknown response '{k}' (e.g. mass, max:von_mises, min:safety_factor, frequency:1, load_multiplier:1, reaction:1, quantity:<name>)");
    if k == "mass" {
        return Ok(mass);
    }
    if k == "volume" {
        return Ok(volume_mm3);
    }
    let (kind, rest) = k.split_once(':').ok_or_else(bad)?;
    match kind {
        "max" | "min" => summary["fields"][rest][kind].as_f64().ok_or_else(|| anyhow!("field '{rest}' is not in the results of this analysis")),
        "frequency" | "load_multiplier" => {
            let i: usize = rest.parse().map_err(|_| bad())?;
            let key = if kind == "frequency" { "frequency_hz" } else { "load_multiplier" };
            summary["modes"].get(i.saturating_sub(1)).and_then(|m| m[key].as_f64()).ok_or_else(|| anyhow!("no {kind} for mode {i}"))
        }
        "quantity" => summary["quantities"]
            .as_array()
            .and_then(|q| q.iter().find(|x| x["name"].as_str().is_some_and(|n| n.to_lowercase().starts_with(&rest.to_lowercase()))))
            .and_then(|x| x["value"].as_f64())
            .ok_or_else(|| anyhow!("quantity '{rest}' not reported by this analysis")),
        "reaction" => {
            let (id, comp) = match rest.split_once(':') {
                Some((a, b)) => (a, Some(b)),
                None => (rest, None),
            };
            let id: u64 = id.parse().map_err(|_| bad())?;
            let r = summary["reactions"].as_array().and_then(|a| a.iter().find(|x| x["load_id"].as_u64() == Some(id))).ok_or_else(|| anyhow!("no reaction for load {id}"))?;
            match comp {
                None => r["magnitude_n"].as_f64().or_else(|| r["heat_flow_w"].as_f64()).or_else(|| r["current_a"].as_f64()).ok_or_else(bad),
                Some(c) => {
                    let idx = match c {
                        "x" => 0,
                        "y" => 1,
                        "z" => 2,
                        _ => return Err(bad()),
                    };
                    r["force_n"][idx].as_f64().ok_or_else(bad)
                }
            }
        }
        _ => Err(bad()),
    }
}

/// Solve one design and return its responses.
pub fn evaluate(model: &Model, responses: &[String]) -> Result<Vec<Option<f64>>> {
    let h = model.element_size.unwrap_or_else(|| mesh::auto_size(&model.bodies, model.mesh_with_enclosure()));
    let m = mesh::build_for(model, h, mesh::MAX_CELLS)?;
    let res = fem::solve(model, &m, &|_, _| {})?;
    let sum = res.summary(&m);
    Ok(responses.iter().map(|k| response(&sum, res.info.mass_kg, m.volume_mm3(), k).ok()).collect())
}

fn validate(model: &Model, params: &[Parameter]) -> Result<()> {
    if params.is_empty() {
        bail!("add at least one parameter");
    }
    for p in params {
        path_of(model, &p.target)?;
        if !(p.max > p.min) {
            bail!("parameter '{}': max must be greater than min", p.target);
        }
        if p.target.eq_ignore_ascii_case("element_size") && p.min <= 0.0 {
            bail!("element_size must be positive");
        }
    }
    Ok(())
}

pub type StudyProg<'a> = &'a dyn Fn(f64, &str);

/// Geometry is meshed with cubes, so dimensions effectively change in whole-element steps.
fn snap_note(model: &Model, params: &[Parameter]) -> String {
    let geometric = params.iter().any(|p| !p.target.starts_with("load") && !p.target.eq_ignore_ascii_case("element_size"));
    if !geometric || model.mesh_fitted() {
        return String::new();
    }
    let h = model.element_size.unwrap_or_else(|| mesh::auto_size(&model.bodies, model.mesh_with_enclosure()));
    format!(". Note: dimensions are meshed in whole elements of {} mm, so nearby values can give identical results; use a smaller element size for finer steps", crate::results::float(h))
}

fn eval_run(model: &Model, params: &[Parameter], values: Vec<f64>, responses: &[String], phase: &str) -> Run {
    match apply(model, params, &values).and_then(|m| evaluate(&m, responses)) {
        Ok(r) => Run { values, responses: r, feasible: true, error: None, phase: phase.into() },
        Err(e) => Run { values, responses: vec![None; responses.len()], feasible: false, error: Some(format!("{e:#}")), phase: phase.into() },
    }
}

/// Latin-hypercube samples in [0,1]^n (deterministic).
fn lhs(n: usize, m: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut s = seed;
    let mut rnd = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut cols: Vec<Vec<f64>> = (0..n)
        .map(|_| {
            let mut c: Vec<f64> = (0..m).map(|i| (i as f64 + 0.5) / m as f64).collect();
            for i in (1..m).rev() {
                let j = (rnd() * (i + 1) as f64) as usize;
                c.swap(i, j.min(i));
            }
            c
        })
        .collect();
    (0..m).map(|i| cols.iter_mut().map(|c| c[i]).collect()).collect()
}

fn denorm(params: &[Parameter], u: &[f64]) -> Vec<f64> {
    params.iter().zip(u).map(|(p, x)| p.min + (p.max - p.min) * x.clamp(0.0, 1.0)).collect()
}

/// Parameter study: full grid (≤ 400 designs) or Latin hypercube with `samples` designs.
pub fn parameter_study(model: &Model, params: Vec<Parameter>, responses: Vec<String>, samples: Option<usize>, prog: StudyProg) -> Result<Study> {
    validate(model, &params)?;
    if responses.is_empty() {
        bail!("add at least one response (e.g. max:von_mises)");
    }
    let t0 = std::time::Instant::now();
    let designs: Vec<Vec<f64>> = match samples {
        Some(m) => lhs(params.len(), m.clamp(2, 400), 0x5EED).into_iter().map(|u| denorm(&params, &u)).collect(),
        None => {
            let steps: Vec<usize> = params.iter().map(|p| p.steps.unwrap_or(if params.len() == 1 { 6 } else { 4 }).clamp(2, 50)).collect();
            let total: usize = steps.iter().product();
            if total > 400 {
                bail!("{total} designs is too many for a grid; reduce steps or use samples (Latin hypercube)");
            }
            let mut out = Vec::with_capacity(total);
            for mut k in 0..total {
                let mut u = Vec::new();
                for &s in &steps {
                    u.push((k % s) as f64 / (s - 1) as f64);
                    k /= s;
                }
                out.push(denorm(&params, &u));
            }
            out
        }
    };
    let mut runs = Vec::new();
    let n = designs.len();
    for (i, d) in designs.into_iter().enumerate() {
        prog(i as f64 / n as f64, &format!("design {}/{n}", i + 1));
        runs.push(eval_run(model, &params, d, &responses, "doe"));
    }
    let ok = runs.iter().filter(|r| r.error.is_none()).count();
    Ok(Study {
        kind: "parameter_study".into(),
        message: format!("{ok} of {n} designs solved{}", snap_note(model, &params)),
        parameters: params,
        responses,
        runs,
        seconds: t0.elapsed().as_secs_f64(),
        ..Default::default()
    })
}

fn violation(c: &[Constraint], vals: &[Option<f64>], offset: usize) -> Option<f64> {
    let mut v = 0.0;
    for (i, ct) in c.iter().enumerate() {
        let x = vals[offset + i]?;
        let scale = ct.value.abs().max(1e-12);
        let g = match ct.op.as_str() {
            "<=" | "<" => (x - ct.value) / scale,
            _ => (ct.value - x) / scale,
        };
        v += g.max(0.0).powi(2);
    }
    Some(v)
}

/// Design optimization with constraints.
pub fn optimize(model: &Model, params: Vec<Parameter>, objective: Objective, constraints: Vec<Constraint>, max_evals: usize, prog: StudyProg) -> Result<Study> {
    validate(model, &params)?;
    for c in &constraints {
        if !matches!(c.op.as_str(), "<=" | ">=" | "<" | ">") {
            bail!("constraint operator must be <= or >=");
        }
    }
    let sign = match objective.goal.as_str() {
        "minimize" | "min" => 1.0,
        "maximize" | "max" => -1.0,
        g => bail!("goal must be minimize or maximize (got '{g}')"),
    };
    let t0 = std::time::Instant::now();
    let n = params.len();
    let budget = max_evals.clamp(n + 3, 400);
    let mut responses = vec![objective.response.clone()];
    responses.extend(constraints.iter().map(|c| c.response.clone()));
    let mut runs: Vec<Run> = Vec::new();
    let mut scale = None::<f64>;

    // merit: objective (normalised) + quadratic constraint penalty
    let merit = |run: &Run, scale: f64| -> f64 {
        match (run.responses[0], violation(&constraints, &run.responses, 1)) {
            (Some(o), Some(v)) => sign * o / scale + 1e3 * v,
            _ => f64::INFINITY,
        }
    };
    let mut cache: Vec<(Vec<f64>, f64)> = Vec::new();
    let mut eval = |u: &[f64], phase: &str, runs: &mut Vec<Run>, scale: &mut Option<f64>| -> f64 {
        let u: Vec<f64> = u.iter().map(|x| x.clamp(0.0, 1.0)).collect();
        if let Some((_, f)) = cache.iter().find(|(c, _)| c.iter().zip(&u).all(|(a, b)| (a - b).abs() < 1e-9)) {
            return *f;
        }
        let i = runs.len();
        prog(i as f64 / budget as f64, &format!("design {}/{budget}", i + 1));
        let mut run = eval_run(model, &params, denorm(&params, &u), &responses, phase);
        run.feasible = run.error.is_none() && violation(&constraints, &run.responses, 1).is_some_and(|v| v <= 1e-9);
        if scale.is_none() {
            if let Some(o) = run.responses[0] {
                *scale = Some(o.abs().max(1e-12));
            }
        }
        let f = merit(&run, scale.unwrap_or(1.0));
        runs.push(run);
        cache.push((u, f));
        f
    };

    // 1. exploration
    let m0 = (2 * n + 2).max(5).min(budget / 2).max(2);
    let mut best: (Vec<f64>, f64) = (vec![0.5; n], f64::INFINITY);
    for u in lhs(n, m0, 0xC0FFEE) {
        let f = eval(&u, "doe", &mut runs, &mut scale);
        if f < best.1 {
            best = (u, f);
        }
    }
    // 2. bounded Nelder–Mead from the best sample
    let mut simplex: Vec<(Vec<f64>, f64)> = vec![best.clone()];
    for i in 0..n {
        let mut u = best.0.clone();
        u[i] = if u[i] + 0.2 <= 1.0 { u[i] + 0.2 } else { u[i] - 0.2 };
        let f = eval(&u, "search", &mut runs, &mut scale);
        simplex.push((u, f));
    }
    while runs.len() < budget {
        simplex.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let size: f64 = simplex.iter().skip(1).map(|(u, _)| u.iter().zip(&simplex[0].0).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max)).fold(0.0, f64::max);
        if size < 2e-3 {
            break;
        }
        let worst = simplex[n].clone();
        let centroid: Vec<f64> = (0..n).map(|k| simplex[..n].iter().map(|(u, _)| u[k]).sum::<f64>() / n as f64).collect();
        let pt = |t: f64| -> Vec<f64> { centroid.iter().zip(&worst.0).map(|(c, w)| (c + t * (c - w)).clamp(0.0, 1.0)).collect() };
        let xr = pt(1.0);
        let fr = eval(&xr, "search", &mut runs, &mut scale);
        if fr < simplex[0].1 {
            let xe = pt(2.0);
            let fe = if runs.len() < budget { eval(&xe, "search", &mut runs, &mut scale) } else { f64::INFINITY };
            simplex[n] = if fe < fr { (xe, fe) } else { (xr, fr) };
        } else if fr < simplex[n - 1].1 {
            simplex[n] = (xr, fr);
        } else {
            let xc = pt(if fr < worst.1 { 0.5 } else { -0.5 });
            let fc = eval(&xc, "search", &mut runs, &mut scale);
            if fc < worst.1.min(fr) {
                simplex[n] = (xc, fc);
            } else {
                // shrink towards the best
                let b = simplex[0].0.clone();
                for j in 1..=n {
                    if runs.len() >= budget {
                        break;
                    }
                    let u: Vec<f64> = simplex[j].0.iter().zip(&b).map(|(x, y)| y + 0.5 * (x - y)).collect();
                    let f = eval(&u, "search", &mut runs, &mut scale);
                    simplex[j] = (u, f);
                }
            }
        }
    }

    // best feasible design (or least-violating one)
    let s = scale.unwrap_or(1.0);
    let best_idx = runs
        .iter()
        .enumerate()
        .filter(|(_, r)| r.feasible)
        .min_by(|a, b| (sign * a.1.responses[0].unwrap()).total_cmp(&(sign * b.1.responses[0].unwrap())))
        .map(|(i, _)| i)
        .or_else(|| runs.iter().enumerate().min_by(|a, b| merit(a.1, s).partial_cmp(&merit(b.1, s)).unwrap_or(std::cmp::Ordering::Equal)).map(|(i, _)| i));
    let feasible = best_idx.is_some_and(|i| runs[i].feasible);
    let message = match best_idx {
        Some(i) => {
            let vals: Vec<String> = params.iter().zip(&runs[i].values).map(|(p, v)| format!("{} = {}", p.target, crate::results::float(*v))).collect();
            format!(
                "{} design after {} evaluations: {} → {} = {}",
                if feasible { "Best feasible" } else { "No design met every constraint; least-violating" },
                runs.len(),
                vals.join(", "),
                objective.response,
                runs[i].responses[0].map(crate::results::float).map(|v| v.to_string()).unwrap_or("?".into())
            ) + &snap_note(model, &params)
        }
        None => "no design could be evaluated".into(),
    };
    Ok(Study {
        kind: "optimization".into(),
        parameters: params,
        responses,
        objective: Some(objective),
        constraints,
        runs,
        best: best_idx,
        message,
        seconds: t0.elapsed().as_secs_f64(),
        ..Default::default()
    })
}

// ------------------------------------------------------------------ surrogates

/// Gaussian-process response surfaces fitted to a study's solved designs.
pub struct SurrogateSet {
    pub parameters: Vec<Parameter>,
    pub responses: Vec<String>,
    /// one GP per response (None when fewer than 3 designs have that response)
    pub gps: Vec<Option<crate::surrogate::Gp>>,
    pub n_runs: usize,
}

fn norm_values(params: &[Parameter], v: &[f64]) -> Vec<f64> {
    params.iter().zip(v).map(|(p, x)| (x - p.min) / (p.max - p.min)).collect()
}

pub fn fit_surrogates(st: &Study) -> Result<SurrogateSet> {
    if st.kind == "topology" || st.parameters.is_empty() {
        bail!("surrogates need a parameter study or optimization (run_parameter_study / optimize_design / surrogate_optimize) first");
    }
    let mut gps = Vec::new();
    for ri in 0..st.responses.len() {
        let pts: Vec<(Vec<f64>, f64)> = st
            .runs
            .iter()
            .filter(|r| r.error.is_none())
            .filter_map(|r| r.responses.get(ri).copied().flatten().map(|y| (norm_values(&st.parameters, &r.values), y)))
            .filter(|(_, y)| y.is_finite())
            .collect();
        // drop duplicate designs (keep the first)
        let mut x: Vec<Vec<f64>> = Vec::new();
        let mut y = Vec::new();
        for (u, v) in pts {
            if !x.iter().any(|w| w.iter().zip(&u).all(|(a, b)| (a - b).abs() < 1e-9)) {
                x.push(u);
                y.push(v);
            }
        }
        gps.push(if x.len() >= 3 { Some(crate::surrogate::Gp::fit(x, &y)?) } else { None });
    }
    Ok(SurrogateSet { parameters: st.parameters.clone(), responses: st.responses.clone(), gps, n_runs: st.runs.len() })
}

impl SurrogateSet {
    /// (mean, std) per response for real parameter values.
    pub fn predict(&self, values: &[f64]) -> Vec<Option<(f64, f64)>> {
        let u = norm_values(&self.parameters, values);
        self.gps.iter().map(|g| g.as_ref().map(|g| g.predict(&u))).collect()
    }
}

/// Bayesian (surrogate-based) optimisation: Gaussian processes on the objective and every
/// constraint, next design = argmax expected improvement × probability of feasibility, each
/// proposal verified with a real solve. `prior` runs (same parameters) are reused.
pub fn surrogate_optimize(model: &Model, params: Vec<Parameter>, objective: Objective, constraints: Vec<Constraint>, max_solves: usize, prior: Vec<Run>, prog: StudyProg) -> Result<Study> {
    use crate::surrogate::{Gp, big_phi, expected_improvement, nelder_mead};
    validate(model, &params)?;
    for c in &constraints {
        if !matches!(c.op.as_str(), "<=" | ">=" | "<" | ">") {
            bail!("constraint operator must be <= or >=");
        }
    }
    let sign = match objective.goal.as_str() {
        "minimize" | "min" => 1.0,
        "maximize" | "max" => -1.0,
        g => bail!("goal must be minimize or maximize (got '{g}')"),
    };
    let t0 = std::time::Instant::now();
    let n = params.len();
    let mut responses = vec![objective.response.clone()];
    responses.extend(constraints.iter().map(|c| c.response.clone()));
    let budget = max_solves.clamp(2, 200);
    let mut runs: Vec<Run> = prior;
    let mut solved = 0usize;
    let feasible = |r: &Run| r.error.is_none() && violation(&constraints, &r.responses, 1).is_some_and(|v| v <= 1e-9);
    let solve = |u: &[f64], phase: &str, runs: &mut Vec<Run>, solved: &mut usize| {
        prog(*solved as f64 / budget as f64, &format!("verifying design {}/{budget} with a real solve", *solved + 1));
        let mut r = eval_run(model, &params, denorm(&params, u), &responses, phase);
        r.feasible = feasible(&r);
        runs.push(r);
        *solved += 1;
    };
    // initial design if there is too little data
    let ok_runs = |runs: &Vec<Run>| runs.iter().filter(|r| r.error.is_none()).count();
    if ok_runs(&runs) < n + 2 {
        let m0 = (2 * n + 2).min(budget.saturating_sub(1)).max(2);
        for u in lhs(n, m0, 0xBA7E5) {
            if solved >= budget {
                break;
            }
            solve(&u, "doe", &mut runs, &mut solved);
        }
    }
    let mut note = String::new();
    while solved < budget {
        // fit surrogates to every verified design
        let data: Vec<&Run> = runs.iter().filter(|r| r.error.is_none() && r.responses.iter().all(|v| v.is_some())).collect();
        if data.len() < 3 {
            let u: Vec<f64> = lhs(n, 1, 0xF00D + solved as u64).remove(0);
            solve(&u, "doe", &mut runs, &mut solved);
            continue;
        }
        let x: Vec<Vec<f64>> = data.iter().map(|r| norm_values(&params, &r.values)).collect();
        let fobj = Gp::fit(x.clone(), &data.iter().map(|r| sign * r.responses[0].unwrap()).collect::<Vec<_>>())?;
        let fcon: Vec<Gp> = (0..constraints.len()).map(|k| Gp::fit(x.clone(), &data.iter().map(|r| r.responses[k + 1].unwrap()).collect::<Vec<_>>())).collect::<Result<_>>()?;
        let best_feasible = data.iter().filter(|r| feasible(r)).map(|r| sign * r.responses[0].unwrap()).fold(f64::INFINITY, f64::min);
        let acq = |u: &[f64]| -> f64 {
            let mut pof = 1.0;
            for (c, g) in constraints.iter().zip(&fcon) {
                let (m, s) = g.predict(u);
                let s = s.max(1e-12);
                pof *= match c.op.as_str() {
                    "<=" | "<" => big_phi((c.value - m) / s),
                    _ => big_phi((m - c.value) / s),
                };
            }
            if best_feasible.is_finite() {
                let (m, s) = fobj.predict(u);
                expected_improvement(m, s, best_feasible) * pof
            } else {
                pof
            }
        };
        // candidates: space-filling set, then local refinement of the best few
        let mut cands: Vec<(Vec<f64>, f64)> = lhs(n, 1500, 0xACE + solved as u64).into_iter().map(|u| {
            let a = acq(&u);
            (u, a)
        }).collect();
        cands.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut best = cands[0].clone();
        for c in cands.iter().take(3) {
            let (u, f) = nelder_mead(&|u: &[f64]| -acq(u), &c.0, 0.05, 0.0, 1.0, 120 * n);
            if -f > best.1 {
                best = (u, -f);
            }
        }
        let scale = if best_feasible.is_finite() { best_feasible.abs().max(1e-12) } else { 1.0 };
        if best_feasible.is_finite() && best.1 < 1e-7 * scale {
            note = format!(" Converged: expected improvement below 1e-7 after {solved} solves.");
            break;
        }
        // never re-solve (nearly) the same design
        let dup = runs.iter().any(|r| norm_values(&params, &r.values).iter().zip(&best.0).all(|(a, b)| (a - b).abs() < 1e-3));
        let u = if dup { cands.iter().find(|c| !runs.iter().any(|r| norm_values(&params, &r.values).iter().zip(&c.0).all(|(a, b)| (a - b).abs() < 1e-3))).map(|c| c.0.clone()).unwrap_or(best.0) } else { best.0 };
        solve(&u, "search", &mut runs, &mut solved);
    }
    let best_idx = runs
        .iter()
        .enumerate()
        .filter(|(_, r)| r.feasible)
        .min_by(|a, b| (sign * a.1.responses[0].unwrap()).total_cmp(&(sign * b.1.responses[0].unwrap())))
        .map(|(i, _)| i);
    let message = match best_idx {
        Some(i) => {
            let vals: Vec<String> = params.iter().zip(&runs[i].values).map(|(p, v)| format!("{} = {}", p.target, crate::results::float(*v))).collect();
            format!(
                "Best verified design after {solved} real solves ({} designs in total): {} → {} = {}.{note}",
                runs.len(),
                vals.join(", "),
                objective.response,
                runs[i].responses[0].map(crate::results::float).map(|v| v.to_string()).unwrap_or("?".into())
            ) + &snap_note(model, &params)
        }
        None => format!("No verified design met every constraint after {solved} solves; widen the ranges or relax the constraints.{note}"),
    };
    Ok(Study {
        kind: "surrogate_optimization".into(),
        parameters: params,
        responses,
        objective: Some(objective),
        constraints,
        runs,
        best: best_idx,
        message,
        seconds: t0.elapsed().as_secs_f64(),
        ..Default::default()
    })
}
