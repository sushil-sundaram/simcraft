//! Simulation planner: turns a goal ("simulate EMI on my PCB") into a SimCraft to-do list using a
//! local LLM served by Ollama. The model is grounded with SimCraft's real capabilities and must
//! answer in a fixed JSON schema; a rule-based gap check flags physics SimCraft cannot do yet.

use crate::model::AnalysisType;
use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

const MAX_LLM_BYTES: u64 = 8 * 1024 * 1024;

pub fn base_url() -> String {
    std::env::var("SIMCRAFT_LLM_URL").unwrap_or_else(|_| "http://127.0.0.1:11434".into())
}

fn validated_base_url() -> Result<String> {
    let raw = base_url();
    let trimmed = raw.trim().trim_end_matches('/');
    let rest = trimmed
        .strip_prefix("http://")
        .or_else(|| trimmed.strip_prefix("https://"))
        .ok_or_else(|| anyhow!("SIMCRAFT_LLM_URL must use http:// or https://"))?;
    let host_port = rest.split('/').next().unwrap_or("");
    if host_port.is_empty() {
        bail!("SIMCRAFT_LLM_URL is missing a host");
    }
    let local_only = std::env::var("SIMCRAFT_LOCAL_ONLY_LLM").is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true"));
    if local_only {
        let host = if host_port.starts_with('[') {
            host_port.split(']').next().map(|h| h.trim_start_matches('[')).unwrap_or("")
        } else {
            host_port.split(':').next().unwrap_or("")
        };
        if !matches!(host.to_ascii_lowercase().as_str(), "127.0.0.1" | "localhost" | "::1") {
            bail!("SIMCRAFT_LOCAL_ONLY_LLM requires a loopback SIMCRAFT_LLM_URL host, got '{host}'");
        }
    }
    Ok(trimmed.to_string())
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ToolAction {
    pub tool: String,
    #[serde(default)]
    pub args: Value,
    #[serde(default)]
    pub reason: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Step {
    pub task: String,
    #[serde(default)]
    pub detail: String,
    /// ready | workaround | missing
    #[serde(default)]
    pub status: String,
    #[serde(default)]
    pub done: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
pub struct Plan {
    #[serde(default)]
    pub goal: String,
    #[serde(default)]
    pub title: String,
    #[serde(default)]
    pub summary: String,
    #[serde(default)]
    pub physics: Vec<String>,
    /// supported | partial | not_supported
    #[serde(default)]
    pub feasibility: String,
    /// SimCraft analysis key to start with, or "none".
    #[serde(default)]
    pub analysis: String,
    #[serde(default)]
    pub steps: Vec<Step>,
    #[serde(default)]
    pub actions: Vec<ToolAction>,
    #[serde(default)]
    pub inputs_needed: Vec<String>,
    #[serde(default)]
    pub missing_capabilities: Vec<String>,
    #[serde(default)]
    pub model: String,
}

/// Models installed in the local Ollama server.
pub fn list_models() -> Result<Vec<String>> {
    let url = validated_base_url()?;
    let agent: ureq::Agent = ureq::Agent::config_builder().timeout_global(Some(std::time::Duration::from_secs(10))).build().into();
    let v: Value = agent
        .get(format!("{url}/api/tags"))
        .call()
        .map_err(|e| anyhow!("cannot reach the local model server at {url} ({e}). Install Ollama (https://ollama.com) and run `ollama pull qwen2.5:7b`"))?
        .body_mut()
        .with_config()
        .limit(MAX_LLM_BYTES)
        .read_json()?;
    Ok(v["models"].as_array().map(|a| a.iter().filter_map(|m| m["name"].as_str().map(String::from)).collect()).unwrap_or_default())
}

/// Pick a sensible default: prefer strong instruction-following models, then the largest one.
pub fn default_model(models: &[String]) -> Option<String> {
    let prefs = ["qwen3", "qwen2.5", "llama3.3", "llama3.1", "gemma3", "mistral-small", "phi4", "llama3.2-vision", "llama3.2", "mistral", "olmo2"];
    for p in prefs {
        if let Some(m) = models.iter().find(|m| m.starts_with(p)) {
            return Some(m.clone());
        }
    }
    models.first().cloned()
}

/// What SimCraft can do, written for the model.
pub fn capabilities() -> String {
    let analyses: Vec<String> = AnalysisType::all().iter().map(|a| format!("- {} (key `{}`, group {})", a.label(), a.key(), a.group())).collect();
    let mats: Vec<String> = crate::material::library().iter().map(|m| m.name.clone()).collect();
    format!(
        "SimCraft is a desktop finite-element simulation app (units mm, N, MPa, °C).\n\
GEOMETRY: boxes, cylinders, spheres combined by add/subtract (holes, pockets), closed STL import (e.g. from CAD). \
An optional ENCLOSURE (air, water or oil box) surrounds the parts for electrostatic, magnetostatic and fluid analyses. Voxel hexahedral mesh with a user-set element size.\n\
ANALYSES:\n{}\n\
LOADS & SUPPORTS: fixed support, prescribed displacement, compression-only support (contact with a rigid surface), symmetry, force, pressure, gravity, \
temperature, heat flux, convection, internal heat generation (whole model or one body), base excitation tables (shock spectrum in g, PSD in g²/Hz), voltage, DC current, coil current density (in a body), permanent magnet (in a body), \
fluid velocity inlet (with inflow temperature), pressure outlet, moving wall.\n\
CFD: steady incompressible flow, laminar or LVEL turbulence (electronics enclosures, ducts, fans as velocity inlets); conjugate_heat_transfer \
solves flow and temperature in fluid AND solids together (heat sinks, PCB/chip cooling, natural convection with add_gravity / buoyancy).\n\
PCB: import_kicad reads a KiCad .kicad_pcb board (outline, copper layers, zones, vias → orthotropic board conductivity; footprints → component bodies named U1, Q1… with power in W). \
Use it for board/component temperatures (steady_thermal with convection, or conjugate_heat_transfer with airflow).\n\
RESULTS: deformation, von Mises / principal stress, safety factor, plastic strain, modal frequencies, buckling load factors, transient histories, frequency-response curves, spectrum peaks, random-vibration 1σ/3σ responses, \
stress-life fatigue life / damage / fatigue safety factor (Basquin S-N with Goodman/Gerber/Soderberg, Steinberg for random) from any stress result, temperature, heat flux, \
electric potential and field, capacitance, current density, Joule heat, resistance, magnetic flux density B and H, fluid velocity, pressure, drag/lift.\n\
MATERIALS: {}; custom materials can be defined (E, ν, density, yield, conductivity, permittivity, electrical conductivity, permeability, viscosity).\n\
FULL-WAVE EM: full_wave_em (FDTD at one frequency: E field, absorbed power density, SAR, power per body; metal cavity or open absorbing boundary; RF port with a power in W) \
and microwave_heating (full_wave_em then transient heating of the solids: food in a microwave oven, dielectric heating, RF absorption). Element size ≤ wavelength in the densest material / 10 \
(food at 2.45 GHz: ≤ 1.7 mm). Materials have a loss tangent; 'Food (water-rich, 2.45 GHz)' and Water are lossy.\n\
THERMAL RADIATION: add_radiation (emissivity, surroundings °C) to the surroundings, or enclosure: true for surface-to-surface exchange by view factors \
(fins, lids, sealed boxes); in conjugate_heat_transfer to the surroundings only. Materials may have conductivity / specific-heat tables vs temperature.\n\
EMI: full_wave_em with em_boundary open reports radiated power, directivity / gain and the field at 3 m / 10 m against FCC / CISPR 32 class B (free-space estimate).\n\
NOT AVAILABLE (must be marked missing): S-parameters and broadband sweeps, sub-cell PCB traces in FDTD, \
time-harmonic eddy currents and induction heating, contact between two deformable bodies and friction, compressible / supersonic flow, transient (time-accurate) CFD, \
surface-to-surface radiation across a CFD fluid, phase change (melting/boiling/evaporation), moisture transport, strain-life (low-cycle) fatigue and crack growth, composites layups, \
acoustics, electrochemistry, moving or rotating parts. \
Workarounds are allowed when honest, e.g. an estimated absorbed power applied as internal heat generation, or a convection coefficient instead of CFD.\n\
WORKFLOW in SimCraft: GEOMETRY tab (bodies) → MATERIAL tab → SETUP tab (analysis + loads/supports, click faces) → MESH tab → SOLVE → RESULTS (contours, cut plane, probe). \
Everything can also be driven by an AI agent through the SimCraft MCP tools.",
        analyses.join("\n"),
        mats.join(", ")
    )
}

/// Rule-based check for physics SimCraft cannot do, so small models cannot gloss over it.
pub fn known_gaps(goal: &str) -> Vec<String> {
    let g = goal.to_lowercase();
    let words: Vec<&str> = g.split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()).collect();
    // short acronyms match whole words only ("emi" must not match "emissivity")
    let has = |ws: &[&str]| ws.iter().any(|w| if w.len() <= 3 { words.contains(w) } else { g.contains(w) });
    let mut out = Vec::new();
    if has(&["emi", "emc", "antenna", "radiated", "far field", "far-field", "s-parameter", "radar", "cispr"]) {
        out.push("S-parameters / broadband sweeps and sub-cell PCB trace models — full_wave_em runs one frequency (its far field and FCC / CISPR margins are free-space estimates).".into());
    }
    if has(&["induction", "eddy"]) {
        out.push("Time-harmonic eddy currents / induction heating (low-frequency magnetic losses in conductors).".into());
    }
    if has(&["turntable", "rotating plate"]) {
        out.push("Microwave turntable rotation (approximate by averaging several food positions).".into());
    }
    if has(&["food", "cook", "thaw", "defrost", "melt", "boil", "evaporat", "steam", "freez"]) {
        out.push("Temperature-dependent dielectric properties and phase change (thawing, boiling, moisture loss) for food or water-rich materials (thermal conductivity / specific heat tables are available).".into());
    }
    if has(&["high speed", "supersonic", "compressib", "shock wave", "mach"]) {
        out.push("Compressible / high-speed CFD (SimCraft's CFD is incompressible).".into());
    }
    if has(&["radiat"]) && has(&["heat", "thermal", "cool", "hot"]) && has(&["airflow", "fan", "cfd", "flow"]) {
        out.push("Surface-to-surface radiation across the air in a CFD model (conjugate heat transfer radiates to the surroundings only; a thermal model without flow supports both).".into());
    }
    if has(&["crack", "low-cycle", "low cycle", "strain-life", "weld fatigue"]) {
        out.push("Crack growth / strain-life (low-cycle) fatigue (stress-life S-N fatigue is available).".into());
    }
    if has(&["friction", "bolted joint", "two parts touching", "gear", "press fit", "snap fit"]) {
        out.push("Contact between deformable bodies with friction (only contact against a rigid surface is available).".into());
    }
    if has(&["acoustic", "noise", "sound", "vibration of air"]) {
        out.push("Acoustics solver.".into());
    }
    if has(&["rotat", "spinning", "moving", "turntable", "motor"]) {
        out.push("Moving / rotating parts (e.g. a microwave turntable or motor rotor) — approximate with several static positions.".into());
    }
    out
}

/// True when a step explicitly runs one of SimCraft's own analyses.
fn names_simcraft_analysis(text: &str) -> bool {
    let t = text.to_lowercase();
    ["transient thermal", "steady-state thermal", "steady state thermal", "thermal analysis", "static structural", "modal analysis", "buckling analysis",
        "thermal-stress", "thermal stress", "electrostatic analysis", "current flow", "thermal-electric", "magnetostatic analysis", "fluid flow analysis", "cfd analysis"]
        .iter()
        .any(|k| t.contains(k))
}

/// True when a step only involves things SimCraft can do (small models often mislabel these).
fn only_supported(text: &str) -> bool {
    let t = text.to_lowercase();
    let gap = ["emi", "emc", "antenna", "far field", "far-field", "s-param", "eddy", "induction", "phase change", "melt", "boil",
        "evaporat", "moisture", "supersonic", "compressib", "crack", "friction", "acoustic", "temperature-dependent", "temperature dependent", "rotat", "moving"];
    if gap.iter().any(|g| t.contains(g)) {
        return false;
    }
    let ok = ["full-wave", "full wave", "fdtd", "microwave", "rf port", "absorbed power", "sar", "dielectric heating", "modal", "vibration", "harmonic", "response spectrum", "shock", "psd", "random", "fatigue", "transient thermal", "steady-state thermal", "steady state thermal", "thermal analysis", "heat conduction", "conduction", "convection", "static structural",
        "stress", "deformation", "buckling", "plastic", "mesh", "material", "geometry", "import", "stl", "results", "temperature", "heat flux", "heat generation", "electrostatic",
        "capacitance", "current flow", "joule", "magnetostatic", "laminar", "turbulen", "lvel", "conjugate", "heat sink", "heatsink", "natural convection", "buoyan", "fluid flow", "probe", "contour"];
    ok.iter().any(|k| t.contains(k))
}

fn schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "summary": {"type": "string"},
            "physics": {"type": "array", "items": {"type": "string"}},
            "feasibility": {"type": "string", "enum": ["supported", "partial", "not_supported"]},
            "analysis": {"type": "string"},
            "steps": {"type": "array", "items": {"type": "object", "properties": {
                "task": {"type": "string"},
                "detail": {"type": "string"},
                "status": {"type": "string", "enum": ["ready", "workaround", "missing"]}
            }, "required": ["task", "detail", "status"]}},
            "actions": {"type": "array", "items": {"type": "object", "properties": {
                "tool": {"type": "string"},
                "args": {"type": "object"},
                "reason": {"type": "string"}
            }, "required": ["tool", "args", "reason"]}},
            "inputs_needed": {"type": "array", "items": {"type": "string"}},
            "missing_capabilities": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["title", "summary", "physics", "feasibility", "analysis", "steps", "inputs_needed", "missing_capabilities"]
    })
}

/// Compact snapshot of the active SimCraft session for local AI Copilot grounding.
pub fn model_context_json(
    model: &crate::model::Model,
    mesh: Option<&crate::mesh::Mesh>,
    selection: &[String],
    elem_size_mm: f64,
    last_error: Option<&str>,
) -> Value {
    let preflight = crate::model::preflight_check(model, mesh, elem_size_mm);
    let bodies: Vec<Value> = model.bodies.iter().map(|b| json!({"name": b.name, "material": b.material, "op": b.op, "role": b.role})).collect();
    let loads: Vec<Value> = model.loads.iter().map(|l| json!({"id": l.id, "label": l.load.label(), "summary": l.load.summary()})).collect();
    let faces: Vec<&str> = mesh.map(|m| m.patches.iter().take(48).map(|p| p.name.as_str()).collect()).unwrap_or_default();
    json!({
        "analysis": model.analysis.kind.key(),
        "element_size_mm": elem_size_mm,
        "bodies": bodies,
        "loads": loads,
        "selected_faces": selection,
        "faces_sample": faces,
        "preflight": preflight,
        "last_error": last_error,
    })
}

/// Rule-based executable quick-fix tool suggestions derived from preflight & current selection.
pub fn quick_fixes(
    model: &crate::model::Model,
    mesh: Option<&crate::mesh::Mesh>,
    selection: &[String],
    elem_size_mm: f64,
) -> Vec<ToolAction> {
    let items = crate::model::preflight_check(model, mesh, elem_size_mm);
    let mut actions = Vec::new();
    for it in items {
        if it.status == crate::model::PreflightStatus::Pass {
            continue;
        }
        match it.code.as_str() {
            "geometry" => actions.push(ToolAction {
                tool: "add_box".into(),
                args: json!({"name": "Part", "min": [0, 0, 0], "size": [100, 40, 10]}),
                reason: "Create a starter 100×40×10 mm box".into(),
            }),
            "enclosure" => actions.push(ToolAction {
                tool: "add_enclosure".into(),
                args: json!({"material": "Air", "padding": 0.5}),
                reason: "Add an Air enclosure around existing bodies".into(),
            }),
            "support" if !selection.is_empty() => actions.push(ToolAction {
                tool: "add_fixed_support".into(),
                args: json!({"faces": selection}),
                reason: format!("Fix the {} currently selected face(s)", selection.len()),
            }),
            "thermal_bc" if !selection.is_empty() => actions.push(ToolAction {
                tool: "add_convection".into(),
                args: json!({"faces": selection, "coefficient": 20.0, "ambient": 22.0}),
                reason: "Add natural convection (20 W/m²K) on selected face(s)".into(),
            }),
            "mesh_wavelength" if model.analysis.frequency_hz > 0.0 => {
                let max_er = model.materials().iter()
                    .filter(|m| model.bodies.iter().any(|b| b.material == m.name))
                    .map(|m| m.relative_permittivity)
                    .fold(1.0f64, f64::max)
                    .max(1.0);
                let rec = ((299_792_458.0 / model.analysis.frequency_hz) * 1000.0 / max_er.sqrt() / 12.0 * 100.0).round() / 100.0;
                actions.push(ToolAction {
                    tool: "set_mesh".into(),
                    args: json!({"element_size_mm": rec.max(0.1)}),
                    reason: format!("Refine element size to {rec:.2} mm (≤ λ/12 for FDTD stability)"),
                });
            }
            _ => {}
        }
    }
    actions
}

/// Actionable diagnosis hint for common solver errors or preflight warnings.
pub fn explain_issue(error_or_msg: &str) -> String {
    let e = error_or_msg.to_lowercase();
    if e.contains("no supports") || e.contains("rigid-body") {
        "The structure can translate or rotate freely in 3D. Select a mounting face and apply Fixed Support (or constrain X/Y/Z with Displacement).".into()
    } else if e.contains("enclosure") {
        "Fluid, electrostatic, magnetostatic, and FDTD analyses solve in the surrounding space. Add an Enclosure (Air/Water) from GEOMETRY ▸ Enclosure.".into()
    } else if e.contains("outlet") {
        "Incompressible CFD needs a fixed-pressure exit paired with a Velocity Inlet. Apply Outlet (0 Pa) on the downstream enclosure face.".into()
    } else if e.contains("rf port") || e.contains("no power") {
        "FDTD needs an RF Port placed on a dielectric/air face (not entirely covered by metal) plus either a lossy body or open boundary.".into()
    } else if e.contains("thermal model needs") {
        "Steady thermal solves need at least one Temperature, Convection, or Radiation path so heat can equilibrate.".into()
    } else {
        format!("Check the SETUP pre-flight checklist and Mesh size: {error_or_msg}")
    }
}

const INSTRUCTIONS: &str = "You are SimCraft's simulation copilot & planning assistant. The user describes a goal, asks about their active model, or troubleshoots an error. \
Produce a practical, ordered to-do list (6 to 10 steps) for doing it IN SIMCRAFT, AND optionally `actions`: concrete MCP tool calls `{tool, args, reason}` when the user asks to set up, modify, or fix the live model (use real body/face names from `LIVE CONTEXT` when available). \
Rules: use only the capabilities listed; mark a step `ready` if SimCraft can do it now, `workaround` if an approximation inside SimCraft is used (say what is approximated), \
`missing` if it needs a capability SimCraft does not have. Never pretend a missing capability exists. \
`analysis` must be one of the analysis keys listed (the first analysis to set up) or `none`. \
Put concrete numbers and units in `detail` when sensible (element sizes, typical material values, load magnitudes). \
Each `task` is a short imperative phrase naming the action (e.g. \"Model the PCB stack-up\", \"Apply the estimated absorbed power as heat generation\"), never a bare tab name like \"setup\". \
SUPPORTed physics that must never be marked missing: heat conduction, convection boundaries, transient heating/cooling, internal heat generation, linear and plastic stress, vibration (modal, harmonic, shock response spectrum, random PSD), high-cycle fatigue, buckling, \
electrostatics, DC current and Joule heating, magnetostatics, laminar flow, full-wave EM at one frequency (FDTD) and microwave/dielectric heating. \
Pick physically sensible materials from the library (food or water-rich items ≈ Water; PCBs: import the KiCad board with import_kicad when available, otherwise FR-4 with Copper). Use realistic element sizes (a few mm for parts of tens of mm). \
Do not add steps for settings SimCraft handles automatically (solver tolerances, convergence settings). \
Choosing the analysis: temperature changing over time (heating, cooling, warm-up) → transient_thermal; steady temperatures → steady_thermal; \
stress/deflection under loads → static_structural; natural frequencies → modal; sinusoidal excitation / unbalance → harmonic_response; shock or seismic spectrum → response_spectrum; \
random vibration / PSD / transport or launch vibration → random_vibration (then fatigue); impact or time-varying loads → transient_structural; yielding → nonlinear_structural; \
slender parts in compression → linear_buckling; heat-induced stress → thermal_stress; capacitance or E-field → electrostatic; DC current/resistance → current_flow; \
current heating → thermal_electric; microwave ovens, RF/dielectric heating → microwave_heating; RF fields, shielding, cavity modes → full_wave_em; coils/magnets → magnetostatic; electronics cooling, heat sinks, natural convection → conjugate_heat_transfer; liquid/air flow → fluid. \
`inputs_needed` lists what the user must provide (dimensions, CAD/STL, material data, operating conditions). \
`missing_capabilities` lists features SimCraft would need to do the job properly. Be concise and specific.";

/// Ask the local model for a plan (without live context).
pub fn plan(goal: &str, model: &str) -> Result<Plan> {
    plan_with_context(goal, model, None)
}

/// Ask the local model for a plan and optional executable tool actions grounded with live model context.
pub fn plan_with_context(goal: &str, model: &str, context: Option<&Value>) -> Result<Plan> {
    if goal.trim().is_empty() {
        bail!("describe what you want to simulate");
    }
    let gaps = known_gaps(goal);
    let mut system = format!("{INSTRUCTIONS}\n\nSIMCRAFT CAPABILITIES\n{}", capabilities());
    if let Some(ctx) = context {
        system += &format!("\n\nLIVE SESSION CONTEXT\n{}", serde_json::to_string_pretty(ctx).unwrap_or_default());
    }
    if !gaps.is_empty() {
        system += &format!("\n\nCAPABILITY CHECK for this request — these are NOT available in SimCraft and the plan must mark the related steps `missing` or `workaround`:\n- {}", gaps.join("\n- "));
    }
    let body = json!({
        "model": model,
        "stream": false,
        "format": schema(),
        "options": {"temperature": 0.2, "num_ctx": 8192},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": goal}
        ]
    });
    let url = validated_base_url()?;
    let agent: ureq::Agent = ureq::Agent::config_builder().timeout_global(Some(std::time::Duration::from_secs(600))).build().into();
    let resp: Value = agent
        .post(format!("{url}/api/chat"))
        .send_json(&body)
        .map_err(|e| anyhow!("local model request failed ({e}); is Ollama running and is '{model}' installed?"))?
        .body_mut()
        .with_config()
        .limit(MAX_LLM_BYTES)
        .read_json()
        .context("reading the model response")?;
    let content = resp["message"]["content"].as_str().ok_or_else(|| anyhow!("unexpected model response: {resp}"))?;
    let mut p: Plan = serde_json::from_str(content).with_context(|| format!("the model did not return a valid plan: {}", &content[..content.len().min(300)]))?;
    p.goal = goal.trim().to_string();
    p.model = model.to_string();
    // validate / repair
    if !AnalysisType::all().iter().any(|a| a.key() == p.analysis) {
        p.analysis = "none".into();
    }
    for s in p.steps.iter_mut() {
        if !matches!(s.status.as_str(), "ready" | "workaround" | "missing") {
            s.status = "ready".into();
        }
        let text = format!("{} {}", s.task, s.detail);
        if s.status == "missing" {
            if only_supported(&text) {
                s.status = "ready".into();
            } else if names_simcraft_analysis(&text) {
                // it runs a SimCraft analysis, only approximating the missing physics
                s.status = "workaround".into();
            }
        }
        s.done = false;
    }
    // replace the model's paraphrases of known gaps with the precise wording
    let words = |t: &str| -> std::collections::HashSet<String> {
        t.to_lowercase().split(|c: char| !c.is_alphanumeric()).filter(|w| w.len() > 3 && !matches!(*w, "with" | "solver" | "simulation" | "properties" | "material")).map(String::from).collect()
    };
    let gap_words: Vec<std::collections::HashSet<String>> = gaps.iter().map(|g| words(g)).collect();
    p.missing_capabilities.retain(|m| {
        let mw = words(m);
        !gap_words.iter().any(|gw| mw.intersection(gw).count() >= 2 || (!mw.is_empty() && mw.is_subset(gw)))
    });
    let mut merged = gaps;
    merged.append(&mut p.missing_capabilities);
    merged.dedup();
    p.missing_capabilities = merged;
    let any_missing = p.steps.iter().any(|s| s.status == "missing") || !p.missing_capabilities.is_empty();
    if p.feasibility == "supported" && any_missing {
        p.feasibility = "partial".into();
    }
    // a plan with a SimCraft analysis to run and steps that work is at least partly supported
    let usable = p.steps.iter().any(|s| s.status == "workaround") && p.analysis != "none";
    if p.feasibility == "not_supported" && usable {
        p.feasibility = "partial".into();
    }
    if !matches!(p.feasibility.as_str(), "supported" | "partial" | "not_supported") {
        p.feasibility = if any_missing { "partial".into() } else { "supported".into() };
    }
    Ok(p)
}

pub fn to_markdown(p: &Plan) -> String {
    let mut s = format!("# {}\n\n_Goal:_ {}\n\n{}\n\n**Feasibility in SimCraft:** {}", p.title, p.goal, p.summary, p.feasibility.replace('_', " "));
    if p.analysis != "none" && !p.analysis.is_empty() {
        s += &format!(" · start with **{}**", p.analysis);
    }
    if !p.physics.is_empty() {
        s += &format!("\n\n**Physics:** {}", p.physics.join(", "));
    }
    s += "\n\n## To-do\n";
    for st in &p.steps {
        let tag = match st.status.as_str() {
            "workaround" => " _(workaround)_",
            "missing" => " **(missing in SimCraft)**",
            _ => "",
        };
        s += &format!("- [{}] **{}**{tag} — {}\n", if st.done { "x" } else { " " }, st.task, st.detail);
    }
    if !p.inputs_needed.is_empty() {
        s += "\n## You will need\n";
        for i in &p.inputs_needed {
            s += &format!("- {i}\n");
        }
    }
    if !p.missing_capabilities.is_empty() {
        s += "\n## Not yet in SimCraft\n";
        for m in &p.missing_capabilities {
            s += &format!("- {m}\n");
        }
    }
    s += &format!("\n_Plan drafted locally by {}._\n", p.model);
    s
}
