//! Shared application state and the tool API used by both the GUI and the MCP server.

use crate::fem;
use crate::geometry::*;
use crate::material::{self, Material};
use crate::mesh::{self, Mesh};
use crate::model::*;
use crate::render::{self, Camera, Display};
use crate::results::{Field, Results};
use anyhow::{Context, Result, anyhow, bail};
use serde_json::{Value, json};
use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

pub struct Session {
    pub model: Model,
    pub mesh: Option<Arc<Mesh>>,
    mesh_rev: u64,
    pub geom_rev: u64,
    pub results: Option<Arc<Results>>,
    pub results_stale: bool,
    /// Bumped on every change; the GUI uses it to refresh.
    pub revision: u64,
    pub display: Display,
    /// View preset requested through the API, consumed by the GUI.
    pub view_request: Option<String>,
    pub selection: Vec<String>,
    pub log: Vec<String>,
    pub project_path: Option<String>,
    pub last_error: Option<String>,
}

impl Default for Session {
    fn default() -> Self {
        Session {
            model: Model::default(),
            mesh: None,
            mesh_rev: u64::MAX,
            geom_rev: 0,
            results: None,
            results_stale: false,
            revision: 1,
            display: Display::default(),
            view_request: None,
            selection: vec![],
            log: vec!["SimCraft ready.".into()],
            project_path: None,
            last_error: None,
        }
    }
}

impl Session {
    pub fn touch_geometry(&mut self) {
        self.geom_rev += 1;
        self.results = None;
        self.results_stale = false;
        self.display.show_results = false;
        self.selection.clear();
        self.revision += 1;
    }
    pub fn touch_setup(&mut self) {
        if self.results.is_some() {
            self.results_stale = true;
        }
        self.revision += 1;
    }
    pub fn log(&mut self, s: impl Into<String>) {
        let s = s.into();
        self.log.push(s);
        if self.log.len() > 300 {
            self.log.remove(0);
        }
        self.revision += 1;
    }
    pub fn element_size(&self) -> f64 {
        self.model.element_size.unwrap_or_else(|| mesh::auto_size(&self.model.bodies))
    }
    pub fn mesh_valid(&self) -> bool {
        self.mesh.is_some() && self.mesh_rev == self.geom_rev
    }
    pub fn ensure_mesh(&mut self) -> Result<Arc<Mesh>> {
        if self.mesh_valid() {
            return Ok(self.mesh.clone().unwrap());
        }
        let h = self.element_size();
        let m = Arc::new(mesh::build_mesh(&self.model.bodies, h, &self.model.materials())?);
        self.mesh = Some(m.clone());
        self.mesh_rev = self.geom_rev;
        self.revision += 1;
        Ok(m)
    }
    fn unique_name(&self, base: &str) -> String {
        let clean: String = base.chars().map(|c| if c.is_alphanumeric() || c == '_' || c == '-' { c } else { '_' }).collect();
        let clean = if clean.is_empty() { "Body".to_string() } else { clean };
        if !self.model.bodies.iter().any(|b| b.name.eq_ignore_ascii_case(&clean)) {
            return clean;
        }
        (2..).map(|i| format!("{clean}{i}")).find(|n| !self.model.bodies.iter().any(|b| b.name.eq_ignore_ascii_case(n))).unwrap()
    }
    fn default_name(&self, kind: &str) -> String {
        let n = self.model.bodies.len() + 1;
        self.unique_name(&format!("{kind}{n}"))
    }
    pub fn add_body(&mut self, name: Option<&str>, kind: &str, shape: Shape, op: Op, material: Option<&str>) -> Result<String> {
        shape.validate()?;
        let name = match name {
            Some(n) if !n.trim().is_empty() => self.unique_name(n),
            _ => self.default_name(kind),
        };
        let material = match material {
            Some(m) => self.resolve_material(m)?.name,
            None => self
                .model
                .bodies
                .iter()
                .rev()
                .find(|b| b.op == Op::Add)
                .map(|b| b.material.clone())
                .unwrap_or_else(default_material),
        };
        self.model.bodies.push(Body { name: name.clone(), shape, op, material });
        self.touch_geometry();
        self.log(format!("Added body '{name}' ({})", if op == Op::Add { "add" } else { "subtract" }));
        Ok(name)
    }
    pub fn resolve_material(&self, name: &str) -> Result<Material> {
        let lib = self.model.materials();
        material::find(&lib, name).ok_or_else(|| {
            anyhow!("unknown material '{name}'. Available: {:?}", lib.iter().map(|m| m.name.clone()).collect::<Vec<_>>())
        })
    }
    pub fn add_load(&mut self, load: Load) -> Result<u32> {
        // validate faces against the current mesh
        if !load.faces().is_empty() {
            let m = self.ensure_mesh()?;
            m.resolve_faces(load.faces())?;
        }
        let id = self.model.next_id.max(1);
        self.model.next_id = id + 1;
        let label = load.label();
        let summary = load.summary();
        self.model.loads.push(LoadItem { id, load });
        self.touch_setup();
        self.log(format!("Added {label} #{id}: {summary}"));
        Ok(id)
    }

    pub fn model_json(&self) -> Value {
        let bodies: Vec<Value> = self
            .model
            .bodies
            .iter()
            .map(|b| json!({"name": b.name, "operation": b.op, "material": b.material, "shape": b.shape}))
            .collect();
        let loads: Vec<Value> = self
            .model
            .loads
            .iter()
            .map(|l| json!({"id": l.id, "type": l.load.label(), "summary": l.load.summary(), "definition": l.load}))
            .collect();
        let mesh = if self.mesh_valid() {
            let m = self.mesh.as_ref().unwrap();
            json!({"element_size_mm": m.h, "elements": m.elems.len(), "nodes": m.nodes.len(), "faces": m.patches.len(), "volume_mm3": m.volume_mm3(), "warnings": m.warnings})
        } else {
            json!({"element_size_mm": self.model.element_size, "status": "not generated (generated automatically on solve)"})
        };
        json!({
            "units": "mm, N, MPa, °C, s",
            "bodies": bodies,
            "analysis": {"type": self.model.analysis.kind, "label": self.model.analysis.kind.label(), "modes": self.model.analysis.modes, "reference_temperature_c": self.model.analysis.reference_temperature},
            "loads": loads,
            "mesh": mesh,
            "results": if self.results.is_some() { if self.results_stale { "available (stale: setup changed since solve)" } else { "available" } } else { "none" },
            "display": self.display,
            "gui_selection": self.selection,
            "project_path": self.project_path,
        })
    }

    pub fn save(&self, path: &str) -> Result<()> {
        let v = json!({"simcraft": 1, "model": self.model, "display": self.display});
        std::fs::write(path, serde_json::to_string_pretty(&v)?)?;
        Ok(())
    }
    pub fn open(&mut self, path: &str) -> Result<()> {
        let txt = std::fs::read_to_string(path).with_context(|| format!("reading {path}"))?;
        let v: Value = serde_json::from_str(&txt)?;
        let mut model: Model = serde_json::from_value(v.get("model").cloned().unwrap_or(v.clone()))?;
        for b in model.bodies.iter_mut() {
            if let Shape::Stl { path, scale, offset, data } = &mut b.shape {
                *data = Some(Arc::new(load_stl(path, *scale, *offset)?));
            }
        }
        self.model = model;
        if let Some(d) = v.get("display") {
            if let Ok(d) = serde_json::from_value(d.clone()) {
                self.display = d;
            }
        }
        self.mesh = None;
        self.touch_geometry();
        self.project_path = Some(path.to_string());
        self.view_request = Some("fit".into());
        self.log(format!("Opened {path}"));
        Ok(())
    }
}

#[derive(Default, Clone)]
pub struct Progress {
    pub running: bool,
    pub fraction: f64,
    pub message: String,
}

pub struct Output {
    pub text: String,
    pub image_png: Option<Vec<u8>>,
}
impl Output {
    fn json(v: Value) -> Output {
        Output { text: serde_json::to_string_pretty(&v).unwrap(), image_png: None }
    }
    fn text(s: impl Into<String>) -> Output {
        Output { text: s.into(), image_png: None }
    }
}

pub struct Core {
    pub session: Mutex<Session>,
    pub progress: Mutex<Progress>,
    busy: AtomicBool,
    pub notify: Mutex<Option<Box<dyn Fn() + Send + Sync>>>,
    /// Set by the GUI so `screenshot_gui` can capture the window.
    pub gui_ctx: Mutex<Option<eframe::egui::Context>>,
}

pub type Shared = Arc<Core>;

pub fn new_core() -> Shared {
    Arc::new(Core { session: Mutex::new(Session::default()), progress: Mutex::new(Progress::default()), busy: AtomicBool::new(false), notify: Mutex::new(None), gui_ctx: Mutex::new(None) })
}

// ---------- argument helpers ----------
fn v3(a: &Value, k: &str) -> Result<V3> {
    let v = a.get(k).ok_or_else(|| anyhow!("missing '{k}' ([x, y, z])"))?;
    let arr = v.as_array().ok_or_else(|| anyhow!("'{k}' must be an array [x, y, z]"))?;
    if arr.len() != 3 {
        bail!("'{k}' must have 3 numbers");
    }
    let mut o = [0.0; 3];
    for i in 0..3 {
        o[i] = arr[i].as_f64().ok_or_else(|| anyhow!("'{k}' must contain numbers"))?;
    }
    Ok(o)
}
fn v3_or(a: &Value, k: &str, d: V3) -> Result<V3> {
    if a.get(k).is_some_and(|v| !v.is_null()) { v3(a, k) } else { Ok(d) }
}
fn num(a: &Value, k: &str) -> Result<f64> {
    a.get(k).and_then(|v| v.as_f64()).ok_or_else(|| anyhow!("missing number '{k}'"))
}
fn opt_num(a: &Value, k: &str) -> Option<f64> {
    a.get(k).and_then(|v| v.as_f64())
}
fn opt_str<'a>(a: &'a Value, k: &str) -> Option<&'a str> {
    a.get(k).and_then(|v| v.as_str())
}
fn faces(a: &Value) -> Result<Vec<String>> {
    match a.get("faces") {
        Some(Value::String(s)) => Ok(vec![s.clone()]),
        Some(Value::Array(v)) => {
            let f: Vec<String> = v.iter().filter_map(|x| x.as_str().map(String::from)).collect();
            if f.is_empty() {
                bail!("'faces' is empty");
            }
            Ok(f)
        }
        _ => bail!("missing 'faces' (array of face names from list_faces)"),
    }
}
fn op(a: &Value) -> Result<Op> {
    match opt_str(a, "operation").unwrap_or("add") {
        "add" | "union" | "join" => Ok(Op::Add),
        "subtract" | "cut" | "remove" => Ok(Op::Subtract),
        o => bail!("operation must be 'add' or 'subtract', got '{o}'"),
    }
}
fn axis(a: &Value) -> Result<Axis> {
    match opt_str(a, "axis").unwrap_or("z").to_lowercase().as_str() {
        "x" => Ok(Axis::X),
        "y" => Ok(Axis::Y),
        "z" => Ok(Axis::Z),
        o => bail!("axis must be x, y or z, got '{o}'"),
    }
}
fn field_arg(a: &Value) -> Result<Option<Field>> {
    match opt_str(a, "field") {
        None => Ok(None),
        Some(s) => serde_json::from_value(json!(s)).map(Some).map_err(|_| {
            anyhow!("unknown field '{s}'. Use one of {:?}", Field::all().iter().map(|f| f.key()).collect::<Vec<_>>())
        }),
    }
}

impl Core {
    fn changed(&self) {
        if let Some(f) = self.notify.lock().unwrap().as_ref() {
            f();
        }
    }

    pub fn is_busy(&self) -> bool {
        self.busy.load(Ordering::SeqCst)
    }

    /// Execute a tool by name. Used by the GUI, the TCP bridge and the MCP server.
    pub fn call(&self, tool: &str, args: &Value) -> Result<Output> {
        let r = self.call_inner(tool, args);
        {
            let mut s = self.session.lock().unwrap();
            match &r {
                Ok(_) => s.last_error = None,
                Err(e) => {
                    s.last_error = Some(format!("{e:#}"));
                    s.log(format!("Error in {tool}: {e:#}"));
                }
            }
        }
        self.changed();
        r
    }

    fn call_inner(&self, tool: &str, a: &Value) -> Result<Output> {
        let a = if a.is_null() { &json!({}) } else { a };
        if tool == "solve" {
            return self.solve();
        }
        if tool == "screenshot_gui" {
            return self.screenshot_gui();
        }
        if self.is_busy() && !matches!(tool, "get_model" | "get_results" | "list_faces" | "list_materials" | "render_image" | "probe" | "get_status" | "set_display" | "get_selection") {
            bail!("a solve is running; wait for it to finish (use get_status)");
        }
        let mut guard = self.session.lock().unwrap();
        let s = &mut *guard;
        match tool {
            "get_model" => Ok(Output::json(s.model_json())),
            "get_status" => {
                let p = self.progress.lock().unwrap().clone();
                Ok(Output::json(json!({"solving": p.running, "progress": p.fraction, "message": p.message, "last_error": s.last_error, "recent_log": s.log.iter().rev().take(8).rev().collect::<Vec<_>>()})))
            }
            "new_model" => {
                let rev = s.revision;
                *s = Session::default();
                s.revision = rev + 1;
                s.geom_rev = rev + 1;
                s.view_request = Some("fit".into());
                s.log("New model");
                Ok(Output::text("Started a new empty model."))
            }
            "add_box" => {
                let shape = Shape::Box { min: v3_or(a, "min", [0.0; 3])?, size: v3(a, "size")? };
                let n = s.add_body(opt_str(a, "name"), "Box", shape, op(a)?, opt_str(a, "material"))?;
                s.view_request = Some("fit".into());
                Ok(Output::text(format!("Added box '{n}'. Its faces are {n}.xmin, {n}.xmax, {n}.ymin, {n}.ymax, {n}.zmin, {n}.zmax.")))
            }
            "add_cylinder" => {
                let shape = Shape::Cylinder { base: v3_or(a, "base", [0.0; 3])?, axis: axis(a)?, radius: num(a, "radius")?, height: num(a, "height")? };
                let n = s.add_body(opt_str(a, "name"), "Cylinder", shape, op(a)?, opt_str(a, "material"))?;
                s.view_request = Some("fit".into());
                Ok(Output::text(format!("Added cylinder '{n}'. Its faces are {n}.side, {n}.bottom, {n}.top.")))
            }
            "add_sphere" => {
                let shape = Shape::Sphere { center: v3_or(a, "center", [0.0; 3])?, radius: num(a, "radius")? };
                let n = s.add_body(opt_str(a, "name"), "Sphere", shape, op(a)?, opt_str(a, "material"))?;
                s.view_request = Some("fit".into());
                Ok(Output::text(format!("Added sphere '{n}'. Its face is {n}.surface.")))
            }
            "import_stl" => {
                let path = opt_str(a, "path").ok_or_else(|| anyhow!("missing 'path'"))?.to_string();
                let sc = opt_num(a, "scale").unwrap_or(1.0);
                let off = v3_or(a, "offset", [0.0; 3])?;
                let data = Arc::new(load_stl(&path, sc, off)?);
                let nseg = data.n_segments;
                let ntri = data.tris.len();
                let stem = std::path::Path::new(&path).file_stem().and_then(|x| x.to_str()).unwrap_or("Part").to_string();
                let shape = Shape::Stl { path: path.clone(), scale: sc, offset: off, data: Some(data) };
                let n = s.add_body(Some(opt_str(a, "name").unwrap_or(&stem)), "Part", shape, op(a)?, opt_str(a, "material"))?;
                s.view_request = Some("fit".into());
                Ok(Output::text(format!("Imported '{path}' as body '{n}' ({ntri} triangles, {nseg} smooth faces). Use list_faces to see face names.")))
            }
            "remove_body" => {
                let name = opt_str(a, "name").ok_or_else(|| anyhow!("missing 'name'"))?;
                let before = s.model.bodies.len();
                s.model.bodies.retain(|b| !b.name.eq_ignore_ascii_case(name));
                if s.model.bodies.len() == before {
                    bail!("no body named '{name}'");
                }
                s.touch_geometry();
                s.log(format!("Removed body '{name}'"));
                Ok(Output::text(format!("Removed body '{name}'. Note: loads that referenced its faces may now fail; check get_model.")))
            }
            "list_materials" => {
                let lib = s.model.materials();
                Ok(Output::json(json!(lib)))
            }
            "define_material" => {
                let name = opt_str(a, "name").ok_or_else(|| anyhow!("missing 'name'"))?;
                let base = opt_str(a, "based_on").map(|b| s.resolve_material(b)).transpose()?.unwrap_or_else(|| material::library()[0].clone());
                let mut m = base;
                m.name = name.to_string();
                let set = |k: &str, f: &mut f64| {
                    if let Some(v) = opt_num(a, k) {
                        *f = v;
                    }
                };
                set("youngs_modulus_gpa", &mut m.youngs_modulus_gpa);
                set("poisson_ratio", &mut m.poisson_ratio);
                set("density", &mut m.density);
                set("yield_strength_mpa", &mut m.yield_strength_mpa);
                set("ultimate_strength_mpa", &mut m.ultimate_strength_mpa);
                set("thermal_conductivity", &mut m.thermal_conductivity);
                set("thermal_expansion", &mut m.thermal_expansion);
                set("specific_heat", &mut m.specific_heat);
                if !(m.poisson_ratio > -1.0 && m.poisson_ratio < 0.5) || m.youngs_modulus_gpa <= 0.0 {
                    bail!("invalid material: need E > 0 and -1 < poisson < 0.5");
                }
                s.model.custom_materials.retain(|x| x.name != m.name);
                s.model.custom_materials.push(m.clone());
                s.touch_geometry();
                Ok(Output::json(json!({"defined": m})))
            }
            "set_material" => {
                let m = s.resolve_material(opt_str(a, "material").ok_or_else(|| anyhow!("missing 'material'"))?)?;
                let targets: Option<Vec<String>> = a.get("bodies").and_then(|b| b.as_array()).map(|v| v.iter().filter_map(|x| x.as_str().map(|s| s.to_lowercase())).collect());
                let mut n = 0;
                for b in s.model.bodies.iter_mut() {
                    if targets.as_ref().is_none_or(|t| t.contains(&b.name.to_lowercase())) {
                        b.material = m.name.clone();
                        n += 1;
                    }
                }
                if n == 0 {
                    bail!("no matching bodies");
                }
                s.touch_geometry();
                s.log(format!("Material '{}' assigned to {n} body(ies)", m.name));
                Ok(Output::text(format!("Assigned {} to {n} body(ies).", m.name)))
            }
            "set_analysis" => {
                if let Some(t) = opt_str(a, "type") {
                    s.model.analysis.kind = serde_json::from_value(json!(t)).map_err(|_| anyhow!("type must be static_structural, steady_thermal, modal or thermal_stress"))?;
                }
                if let Some(m) = opt_num(a, "modes") {
                    s.model.analysis.modes = (m as usize).clamp(1, 20);
                }
                if let Some(t) = opt_num(a, "reference_temperature") {
                    s.model.analysis.reference_temperature = t;
                }
                s.touch_setup();
                s.results = None;
                s.display.show_results = false;
                let l = s.model.analysis.kind.label();
                s.log(format!("Analysis set to {l}"));
                Ok(Output::text(format!("Analysis: {l}.")))
            }
            "list_faces" => {
                let m = s.ensure_mesh()?;
                Ok(Output::json(json!({
                    "faces": m.patches,
                    "bbox_mm": {"min": m.bbox.0, "max": m.bbox.1},
                    "note": "Use these names in 'faces'. Also accepted: bbox.xmin|xmax|ymin|ymax|zmin|zmax (extreme planar faces), a body name (all its faces), or 'all'."
                })))
            }
            "add_fixed_support" => Ok(load_out(s.add_load(Load::FixedSupport { faces: faces(a)? })?)),
            "add_displacement" => {
                let l = Load::Displacement { faces: faces(a)?, x: opt_num(a, "x"), y: opt_num(a, "y"), z: opt_num(a, "z") };
                Ok(load_out(s.add_load(l)?))
            }
            "add_force" => Ok(load_out(s.add_load(Load::Force { faces: faces(a)?, vector: v3(a, "vector")? })?)),
            "add_pressure" => Ok(load_out(s.add_load(Load::Pressure { faces: faces(a)?, value_mpa: num(a, "value_mpa")? })?)),
            "add_gravity" => Ok(load_out(s.add_load(Load::Gravity { acceleration: v3_or(a, "acceleration", [0.0, 0.0, -9.80665])? })?)),
            "add_temperature" => Ok(load_out(s.add_load(Load::Temperature { faces: faces(a)?, value: num(a, "value")? })?)),
            "add_heat_flux" => Ok(load_out(s.add_load(Load::HeatFlux { faces: faces(a)?, value: num(a, "value")? })?)),
            "add_convection" => {
                let l = Load::Convection { faces: faces(a)?, coefficient: num(a, "coefficient")?, ambient: opt_num(a, "ambient").unwrap_or(22.0) };
                Ok(load_out(s.add_load(l)?))
            }
            "add_heat_generation" => Ok(load_out(s.add_load(Load::HeatGeneration { value: num(a, "value")? })?)),
            "remove_load" => {
                let id = num(a, "id")? as u32;
                let before = s.model.loads.len();
                s.model.loads.retain(|l| l.id != id);
                if before == s.model.loads.len() {
                    bail!("no load with id {id}");
                }
                s.touch_setup();
                s.log(format!("Removed load #{id}"));
                Ok(Output::text(format!("Removed load #{id}.")))
            }
            "clear_loads" => {
                s.model.loads.clear();
                s.touch_setup();
                Ok(Output::text("All supports and loads removed."))
            }
            "set_mesh" => {
                let es = opt_num(a, "element_size_mm");
                if let Some(e) = es {
                    if e <= 0.0 {
                        bail!("element_size_mm must be positive");
                    }
                }
                s.model.element_size = es;
                s.touch_geometry();
                let m = s.ensure_mesh()?;
                s.log(format!("Mesh: {} elements, {} nodes, size {} mm", m.elems.len(), m.nodes.len(), m.h));
                Ok(Output::json(json!({"element_size_mm": m.h, "elements": m.elems.len(), "nodes": m.nodes.len(), "dofs": m.nodes.len()*3, "warnings": m.warnings})))
            }
            "get_results" => {
                let r = s.results.clone().ok_or_else(|| anyhow!("no results yet; run solve"))?;
                let m = s.mesh.clone().unwrap();
                let mut v = r.summary(&m);
                if s.results_stale {
                    v["stale"] = json!("setup changed after this solve; run solve again");
                }
                Ok(Output::json(v))
            }
            "probe" => {
                let r = s.results.clone().ok_or_else(|| anyhow!("no results yet; run solve"))?;
                let m = s.mesh.clone().unwrap();
                let p = v3(a, "point")?;
                let n = m.nearest_node(p);
                let mut vals = serde_json::Map::new();
                let mode = opt_num(a, "mode").map(|x| (x as usize).saturating_sub(1)).unwrap_or(0);
                for f in r.available_fields() {
                    if let Some(v) = r.scalar(f, mode) {
                        vals.insert(f.key(), json!(crate::results::float(v[n])));
                    }
                }
                Ok(Output::json(json!({"requested_point_mm": p, "nearest_node_mm": m.nodes[n], "values": vals})))
            }
            "set_display" => {
                if let Some(f) = field_arg(a)? {
                    s.display.field = f;
                    s.display.show_results = true;
                }
                if let Some(b) = a.get("show_results").and_then(|v| v.as_bool()) {
                    s.display.show_results = b;
                }
                if let Some(m) = opt_num(a, "mode") {
                    s.display.mode = (m as usize).saturating_sub(1);
                }
                match a.get("deformation_scale") {
                    Some(Value::String(x)) if x == "auto" => s.display.deformation_scale = None,
                    Some(v) if v.is_number() => s.display.deformation_scale = v.as_f64(),
                    _ => {}
                }
                if let Some(b) = a.get("show_mesh").and_then(|v| v.as_bool()) {
                    s.display.show_mesh = b;
                }
                if let Some(b) = a.get("dark_mode").and_then(|v| v.as_bool()) {
                    s.display.dark = b;
                }
                if let Some(v) = opt_str(a, "view") {
                    if !render::VIEWS.contains(&v) && v != "fit" {
                        bail!("view must be one of {:?} or fit", render::VIEWS);
                    }
                    s.view_request = Some(v.to_string());
                }
                s.revision += 1;
                Ok(Output::json(json!({"display": s.display})))
            }
            "render_image" => {
                let mesh = s.ensure_mesh()?;
                let mut d = s.display.clone();
                if let Some(f) = field_arg(a)? {
                    d.field = f;
                    d.show_results = true;
                }
                if let Some(b) = a.get("show_results").and_then(|v| v.as_bool()) {
                    d.show_results = b;
                }
                if let Some(m) = opt_num(a, "mode") {
                    d.mode = (m as usize).saturating_sub(1);
                }
                if let Some(x) = opt_num(a, "deformation_scale") {
                    d.deformation_scale = Some(x);
                }
                let w = opt_num(a, "width").unwrap_or(900.0).clamp(200.0, 2000.0) as usize;
                let h = opt_num(a, "height").unwrap_or(620.0).clamp(150.0, 2000.0) as usize;
                let view = opt_str(a, "view").unwrap_or("iso");
                let res = s.results.clone();
                let hl: HashSet<u32> = HashSet::new();
                let scene = render::build_scene(&mesh, &s.model, res.as_deref(), &d, &hl, 1.0);
                let mut cam = Camera::default();
                if !cam.set_view(view) {
                    bail!("unknown view '{view}', use {:?}", render::VIEWS);
                }
                cam.fit(scene.lo, scene.hi, w as f64, h as f64);
                let (title, unit) = match (&res, d.show_results) {
                    (Some(r), true) => {
                        let f = if r.has(d.field) { d.field } else { r.available_fields()[0] };
                        let mut t = format!("{} [{}]", f.label(), f.unit());
                        if r.analysis == AnalysisType::Modal {
                            if let Some(m) = r.modes.get(d.mode) {
                                t = format!("Mode {} — {:.2} Hz (shape normalised)", d.mode + 1, m.frequency_hz);
                            }
                        }
                        (t, f.unit().to_string())
                    }
                    _ => ("Model (blue = supports, red = forces, orange = pressure, pink = temperature, cyan = convection, yellow = heat flux)".to_string(), String::new()),
                };
                let png = render::render_png(&scene, &cam, w, h, d.show_mesh, &unit, d.dark)?;
                let mut text = format!("{title}; view {view}");
                if let Some((mn, mx)) = scene.range {
                    text += &format!("; legend min {} max {}", render::fmt_val(mn), render::fmt_val(mx));
                }
                if scene.scale_used > 0.0 {
                    text += &format!("; deformation scale {}x", render::fmt_val(scene.scale_used));
                }
                if s.results_stale {
                    text += " (results are stale)";
                }
                Ok(Output { text, image_png: Some(png) })
            }
            "export_vtk" => {
                let path = opt_str(a, "path").ok_or_else(|| anyhow!("missing 'path'"))?;
                let m = s.ensure_mesh()?;
                crate::vtk::export(path, &m, s.results.as_deref())?;
                s.log(format!("Exported VTK {path}"));
                Ok(Output::text(format!("Wrote {path} (open with ParaView).")))
            }
            "save_project" => {
                let path = opt_str(a, "path").map(String::from).or(s.project_path.clone()).ok_or_else(|| anyhow!("missing 'path'"))?;
                s.save(&path)?;
                s.project_path = Some(path.clone());
                s.log(format!("Saved {path}"));
                Ok(Output::text(format!("Saved project to {path}.")))
            }
            "open_project" => {
                let path = opt_str(a, "path").ok_or_else(|| anyhow!("missing 'path'"))?;
                s.open(path)?;
                Ok(Output::json(s.model_json()))
            }
            "get_selection" => Ok(Output::json(json!({"selected_faces": s.selection}))),
            _ => bail!("unknown tool '{tool}'"),
        }
    }

    fn screenshot_gui(&self) -> Result<Output> {
        use eframe::egui;
        let ctx = self.gui_ctx.lock().unwrap().clone().ok_or_else(|| anyhow!("the SimCraft window is not open (use launch_gui first); render_image works headless"))?;
        let (tx, rx) = std::sync::mpsc::channel();
        let mut img = None;
        for _attempt in 0..6 {
            let tx = tx.clone();
            ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(egui::ScreenshotCallback::new(move |img| {
                let _ = tx.send(img);
            })));
            ctx.send_viewport_cmd(egui::ViewportCommand::RequestPaintWhileHidden);
            ctx.request_repaint();
            if let Ok(i) = rx.recv_timeout(std::time::Duration::from_millis(1500)) {
                img = Some(i);
                break;
            }
        }
        let img = img.ok_or_else(|| anyhow!("timed out waiting for the SimCraft window to repaint (is it minimised?)"))?;
        let [w, h] = img.size;
        let rgba: Vec<u8> = img.pixels.iter().flat_map(|c| c.to_srgba_unmultiplied()).collect();
        let mut out = Vec::new();
        {
            let mut enc = png::Encoder::new(&mut out, w as u32, h as u32);
            enc.set_color(png::ColorType::Rgba);
            enc.set_depth(png::BitDepth::Eight);
            enc.write_header()?.write_image_data(&rgba)?;
        }
        Ok(Output { text: format!("SimCraft window screenshot ({w}x{h})"), image_png: Some(out) })
    }

    fn solve(&self) -> Result<Output> {
        if self.busy.swap(true, Ordering::SeqCst) {
            bail!("a solve is already running");
        }
        let out = self.solve_inner();
        self.busy.store(false, Ordering::SeqCst);
        {
            let mut p = self.progress.lock().unwrap();
            p.running = false;
        }
        out
    }

    fn solve_inner(&self) -> Result<Output> {
        let (model, mesh, rev) = {
            let mut s = self.session.lock().unwrap();
            let lbl = s.model.analysis.kind.label();
            s.log(format!("Solving {lbl}…"));
            let m = s.ensure_mesh()?;
            (s.model.clone(), m, s.geom_rev)
        };
        *self.progress.lock().unwrap() = Progress { running: true, fraction: 0.0, message: "starting".into() };
        self.changed();
        let prog = |f: f64, msg: &str| {
            let mut p = self.progress.lock().unwrap();
            p.fraction = f;
            p.message = msg.to_string();
            drop(p);
            self.changed();
        };
        let res = fem::solve(&model, &mesh, &prog)?;
        let mut s = self.session.lock().unwrap();
        if s.geom_rev != rev {
            bail!("geometry changed during the solve; solve again");
        }
        let summary = res.summary(&mesh);
        s.display.field = match res.analysis {
            AnalysisType::SteadyThermal => Field::Temperature,
            AnalysisType::Modal => Field::TotalDeformation,
            _ => if s.display.field == Field::Temperature || !res.has(s.display.field) { Field::VonMises } else { s.display.field },
        };
        if !res.has(s.display.field) {
            s.display.field = res.available_fields()[0];
        }
        s.display.show_results = true;
        s.display.mode = 0;
        let info = res.info.clone();
        s.results = Some(Arc::new(res));
        s.results_stale = false;
        s.revision += 1;
        let how = if info.iterations > 0 { format!("{}, {} iterations", info.solver, info.iterations) } else { info.solver.clone() };
        s.log(format!("Solved in {:.2} s — {} elements, {} DOF ({how})", info.seconds, info.elements, info.dofs));
        Ok(Output::json(summary))
    }
}

fn load_out(id: u32) -> Output {
    Output::text(format!("Added load #{id}."))
}

// ---------- tool catalogue (MCP schemas) ----------

pub fn tool_defs() -> Vec<Value> {
    let v3s = |d: &str| json!({"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3, "description": d});
    let faces = json!({"type": "array", "items": {"type": "string"}, "description": "Face names from list_faces (e.g. \"Plate.zmin\", \"Hole.side\"), a body name for all its faces, bbox.xmin..bbox.zmax for the extreme planar faces, or \"all\"."});
    let op = json!({"type": "string", "enum": ["add", "subtract"], "description": "add = union material, subtract = cut a hole/pocket. Bodies are applied in order."});
    let mat = json!({"type": "string", "description": "Material name (see list_materials). Default: previous body's material or Structural Steel."});
    let name = json!({"type": "string", "description": "Body name (used as prefix of its face names)."});
    let t = |n: &str, d: &str, props: Value, req: &[&str]| json!({"name": n, "description": d, "inputSchema": {"type": "object", "properties": props, "required": req}});
    vec![
        t("get_model", "Get the whole model: bodies, materials, analysis type, supports/loads, mesh status, display state and faces the user has selected in the GUI.", json!({}), &[]),
        t("get_status", "Solver progress, last error and recent log lines.", json!({}), &[]),
        t("new_model", "Discard everything and start an empty model.", json!({}), &[]),
        t("add_box", "Add a rectangular block (mm). Faces: <name>.xmin/xmax/ymin/ymax/zmin/zmax.", json!({"name": name, "min": v3s("Min corner [x,y,z] mm (default origin)"), "size": v3s("Size [dx,dy,dz] mm"), "operation": op, "material": mat}), &["size"]),
        t("add_cylinder", "Add a cylinder (mm) or, with operation=subtract, drill a hole. Faces: <name>.side/bottom/top.", json!({"name": name, "base": v3s("Centre of the bottom cap [x,y,z] mm"), "axis": {"type": "string", "enum": ["x", "y", "z"]}, "radius": {"type": "number"}, "height": {"type": "number", "description": "Length along the axis from base"}, "operation": op, "material": mat}), &["radius", "height"]),
        t("add_sphere", "Add a sphere (mm). Face: <name>.surface.", json!({"name": name, "center": v3s("Centre mm"), "radius": {"type": "number"}, "operation": op, "material": mat}), &["radius"]),
        t("import_stl", "Import a closed STL solid (e.g. exported from SolidCraft / Fusion 360). Smooth regions become faces <name>.face0, face1 …", json!({"path": {"type": "string", "description": "Absolute path to .stl"}, "scale": {"type": "number", "description": "Multiply coordinates (use 1000 if the STL is in metres)"}, "offset": v3s("Translation mm"), "name": name, "operation": op, "material": mat}), &["path"]),
        t("remove_body", "Delete a body by name.", json!({"name": {"type": "string"}}), &["name"]),
        t("list_materials", "List the engineering materials library with properties.", json!({}), &[]),
        t("define_material", "Create or overwrite a custom material. Unspecified properties come from 'based_on' (default Structural Steel).", json!({"name": {"type": "string"}, "based_on": {"type": "string"}, "youngs_modulus_gpa": {"type": "number"}, "poisson_ratio": {"type": "number"}, "density": {"type": "number", "description": "kg/m³"}, "yield_strength_mpa": {"type": "number"}, "ultimate_strength_mpa": {"type": "number"}, "thermal_conductivity": {"type": "number", "description": "W/(m·K)"}, "thermal_expansion": {"type": "number", "description": "1/K"}, "specific_heat": {"type": "number"}}), &["name"]),
        t("set_material", "Assign a material to bodies (all bodies if 'bodies' omitted).", json!({"material": {"type": "string"}, "bodies": {"type": "array", "items": {"type": "string"}}}), &["material"]),
        t("set_analysis", "Choose the analysis system.", json!({"type": {"type": "string", "enum": ["static_structural", "steady_thermal", "modal", "thermal_stress"]}, "modes": {"type": "integer", "description": "Number of modes for modal (1-20, default 6)"}, "reference_temperature": {"type": "number", "description": "Stress-free temperature °C (default 22)"}}), &[]),
        t("list_faces", "Mesh the geometry if needed and list selectable faces with area, centroid and outward normal. Call this before adding supports or loads.", json!({}), &[]),
        t("add_fixed_support", "Fix faces in all directions.", json!({"faces": faces}), &["faces"]),
        t("add_displacement", "Prescribe displacement (mm) on faces; omit a component to leave it free (e.g. z=0 for a roller).", json!({"faces": faces, "x": {"type": "number"}, "y": {"type": "number"}, "z": {"type": "number"}}), &["faces"]),
        t("add_force", "Total force vector in N, spread evenly over the faces.", json!({"faces": faces, "vector": v3s("[Fx,Fy,Fz] N")}), &["faces", "vector"]),
        t("add_pressure", "Normal pressure in MPa (positive pushes into the surface).", json!({"faces": faces, "value_mpa": {"type": "number"}}), &["faces", "value_mpa"]),
        t("add_gravity", "Body acceleration in m/s² (default standard earth gravity -Z).", json!({"acceleration": v3s("[ax,ay,az] m/s²")}), &[]),
        t("add_temperature", "Fixed temperature (°C) on faces. Thermal / thermal-stress analyses.", json!({"faces": faces, "value": {"type": "number"}}), &["faces", "value"]),
        t("add_heat_flux", "Heat flux into the faces, W/m².", json!({"faces": faces, "value": {"type": "number"}}), &["faces", "value"]),
        t("add_convection", "Convection film coefficient W/(m²·K) with ambient temperature °C.", json!({"faces": faces, "coefficient": {"type": "number"}, "ambient": {"type": "number"}}), &["faces", "coefficient"]),
        t("add_heat_generation", "Internal heat generation in W/m³ for the whole part.", json!({"value": {"type": "number"}}), &["value"]),
        t("remove_load", "Remove a support/load by id (ids are shown in get_model).", json!({"id": {"type": "integer"}}), &["id"]),
        t("clear_loads", "Remove all supports and loads.", json!({}), &[]),
        t("set_mesh", "Set the element size in mm (omit for automatic) and generate the mesh.", json!({"element_size_mm": {"type": "number"}}), &[]),
        t("solve", "Mesh (if needed) and solve the current analysis. Returns min/max of every result, reactions and modal frequencies.", json!({}), &[]),
        t("get_results", "Summary of the last solution: min/max with locations, reaction forces / heat flows, natural frequencies.", json!({}), &[]),
        t("probe", "Result values at the node nearest to a point (mm).", json!({"point": v3s("[x,y,z] mm"), "mode": {"type": "integer", "description": "1-based mode number for modal"}}), &["point"]),
        t("set_display", "Change what the GUI viewport shows.", json!({"field": {"type": "string", "enum": Field::all().iter().map(|f| f.key()).collect::<Vec<_>>()}, "show_results": {"type": "boolean"}, "mode": {"type": "integer"}, "deformation_scale": {"description": "number, or \"auto\"; 0 = undeformed, 1 = true scale"}, "show_mesh": {"type": "boolean"}, "dark_mode": {"type": "boolean", "description": "Dark UI and viewport theme"}, "view": {"type": "string", "enum": ["iso", "front", "back", "left", "right", "top", "bottom", "iso_back", "fit"]}}), &[]),
        t("render_image", "Render a PNG picture of the model or of a result contour so you can see it.", json!({"field": {"type": "string", "enum": Field::all().iter().map(|f| f.key()).collect::<Vec<_>>()}, "show_results": {"type": "boolean", "description": "false = show geometry with supports/loads"}, "mode": {"type": "integer"}, "view": {"type": "string", "enum": render::VIEWS}, "deformation_scale": {"type": "number"}, "width": {"type": "integer"}, "height": {"type": "integer"}}), &[]),
        t("export_vtk", "Write mesh + results to a legacy .vtk file for ParaView.", json!({"path": {"type": "string"}}), &["path"]),
        t("save_project", "Save the project (.simcraft JSON).", json!({"path": {"type": "string"}}), &[]),
        t("open_project", "Open a .simcraft project.", json!({"path": {"type": "string"}}), &["path"]),
        t("screenshot_gui", "Screenshot of the whole SimCraft window (ribbon, browser, viewport) when the GUI is open.", json!({}), &[]),
        t("get_selection", "Faces the user has currently picked in the GUI viewport.", json!({}), &[]),
    ]
}
