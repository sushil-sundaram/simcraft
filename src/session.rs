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
    /// Test hook: GUI state requested by the hidden `__ui` tool (consumed by the window).
    pub ui_request: Option<Value>,
    /// Model snapshots before each change (newest last) and undone ones, for undo / redo.
    pub undo: Vec<Model>,
    pub redo: Vec<Model>,
    undo_revs: Vec<u64>,
    redo_revs: Vec<u64>,
    /// Monotonic revision tag of the current `Model` and the last saved revision (None if recovered/unsaved).
    pub model_rev: u64,
    next_model_rev: u64,
    saved_rev: Option<u64>,
    pub model: Model,
    pub mesh: Option<Arc<Mesh>>,
    mesh_rev: u64,
    mesh_mode: (bool, bool),
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
    /// To-do list from the simulation planner.
    pub plan: Option<crate::assistant::Plan>,
    /// Surrogates fitted to the current study: (runs, parameters, responses) key and the set.
    pub surrogate: Option<((usize, Vec<crate::study::Parameter>, Vec<String>), Arc<crate::study::SurrogateSet>)>,
    /// Last parameter study / optimization / topology run.
    pub study: Option<crate::study::Study>,
}

impl Default for Session {
    fn default() -> Self {
        Session {
            model: Model::default(),
            mesh: None,
            mesh_rev: u64::MAX,
            mesh_mode: (false, false),
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
            plan: None,
            undo: Vec::new(),
            ui_request: None,
            redo: Vec::new(),
            undo_revs: Vec::new(),
            redo_revs: Vec::new(),
            model_rev: 0,
            next_model_rev: 1,
            saved_rev: Some(0),
            study: None,
            surrogate: None,
        }
    }
}

impl Session {
    fn bump_model_rev(&mut self) {
        self.model_rev = self.next_model_rev;
        self.next_model_rev += 1;
    }
    pub fn invalidate_geometry(&mut self) {
        self.geom_rev += 1;
        self.results = None;
        self.results_stale = false;
        self.display.show_results = false;
        self.selection.clear();
        self.revision += 1;
    }
    pub fn touch_geometry(&mut self) {
        self.invalidate_geometry();
        self.bump_model_rev();
    }
    pub fn touch_setup(&mut self) {
        if self.results.is_some() {
            self.results_stale = true;
        }
        self.revision += 1;
        self.bump_model_rev();
    }
    pub fn mesh_key(&self) -> (u64, bool, bool, u64) {
        (self.geom_rev, self.model.mesh_with_enclosure(), self.model.mesh_fitted(), self.element_size().to_bits())
    }
    pub fn install_mesh(&mut self, m: Mesh, rev: u64, mode: (bool, bool)) -> Option<Arc<Mesh>> {
        if self.geom_rev != rev || mode.0 != self.model.mesh_with_enclosure() {
            return None;
        }
        let m = Arc::new(m);
        if self.results.as_ref().is_some_and(|r| r.node_len().is_some_and(|n| n != m.nodes.len())) {
            self.results = None;
        }
        self.mesh_mode = mode;
        self.mesh = Some(m.clone());
        self.mesh_rev = self.geom_rev;
        self.revision += 1;
        Some(m)
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
        self.model.element_size.unwrap_or_else(|| {
            let geo = mesh::auto_size(&self.model.bodies, self.model.mesh_with_enclosure());
            // imported boards need ≥ 2 cells through their thickness
            let pcb_t = self.model.bodies.iter().filter_map(|b| match &b.shape {
                Shape::Pcb { data: Some(d), .. } => Some(d.thickness),
                _ => None,
            }).fold(f64::INFINITY, f64::min);
            let geo = if pcb_t.is_finite() { geo.min(((pcb_t / 2.0) * 20.0).floor() / 20.0) } else { geo };
            if !self.fdtd() {
                return geo;
            }
            // full-wave: ≥ 10 cells per wavelength in the densest dielectric, ≤ ~8 M cells
            let mats = self.model.materials();
            let f = self.model.analysis.frequency_hz;
            let eps = self
                .model
                .bodies
                .iter()
                .filter_map(|b| crate::material::find(&mats, &b.material))
                .filter(|m| m.electrical_conductivity < 1e4)
                .map(|m| m.relative_permittivity)
                .fold(1.0, f64::max);
            let lam = 299_792_458.0 / (f * eps.sqrt()) * 1e3;
            let (lo, hi) = mesh::domain_bounds(&self.model.bodies, self.model.mesh_with_enclosure());
            let vol = (hi[0] - lo[0]) * (hi[1] - lo[1]) * (hi[2] - lo[2]);
            let floor = (vol / 8e6).cbrt();
            let h = (lam / 10.0).max(floor);
            (h * 10.0).floor() / 10.0
        })
    }
    fn fdtd(&self) -> bool {
        matches!(self.model.analysis.kind, AnalysisType::FullWaveEm | AnalysisType::MicrowaveHeating)
    }
    /// File of the first imported KiCad board.
    pub fn pcb_path(&self) -> Option<String> {
        self.model.bodies.iter().find_map(|b| match &b.shape {
            Shape::Pcb { path, .. } => Some(path.clone()),
            _ => None,
        })
    }
    pub fn mesh_valid(&self) -> bool {
        self.mesh.as_ref().is_some_and(|m| m.elems.len() <= mesh::max_cells_for(&self.model)) && self.mesh_rev == self.geom_rev && self.mesh_mode == (self.model.mesh_with_enclosure(), self.model.mesh_fitted())
    }
    pub fn ensure_mesh(&mut self) -> Result<Arc<Mesh>> {
        if self.mesh_valid() {
            return Ok(self.mesh.clone().unwrap());
        }
        let h = self.element_size();
        let mode = (self.model.mesh_with_enclosure(), self.model.mesh_fitted());
        let limit = mesh::max_cells_for(&self.model);
        let rev = self.geom_rev;
        let m = mesh::build_for(&self.model, h, limit)?;
        Ok(self.install_mesh(m, rev, mode).expect("session lock held"))
    }
    /// Plain voxel mesh (no boundary fitting), e.g. for topology optimisation on the voxel grid.
    /// Becomes the session mesh so results display on it; the next fitted solve rebuilds.
    pub fn ensure_plain_mesh(&mut self) -> Result<Arc<Mesh>> {
        let m = self.ensure_mesh()?;
        if m.fitted.is_empty() {
            return Ok(m);
        }
        let h = self.element_size();
        let m = Arc::new(mesh::build_mesh_max(&self.model.bodies, h, &self.model.materials(), self.model.mesh_with_enclosure(), mesh::MAX_CELLS)?);
        if self.results.as_ref().is_some_and(|r| r.node_len().is_some_and(|n| n != m.nodes.len())) {
            self.results = None;
        }
        self.mesh_mode = (self.model.mesh_with_enclosure(), false);
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
        self.add_body_role(name, kind, shape, op, material, Role::Solid)
    }
    pub fn add_body_role(&mut self, name: Option<&str>, kind: &str, shape: Shape, op: Op, material: Option<&str>, role: Role) -> Result<String> {
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
                .find(|b| b.op == Op::Add && b.role == Role::Solid)
                .map(|b| b.material.clone())
                .unwrap_or_else(default_material),
        };
        self.model.bodies.push(Body { name: name.clone(), shape, op, material, role });
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
        if let Some(b) = load.body() {
            if !self.model.bodies.iter().any(|x| x.name.eq_ignore_ascii_case(b) && x.role == Role::Solid) {
                bail!("no solid body named '{b}' (bodies: {:?})", self.model.bodies.iter().map(|x| x.name.clone()).collect::<Vec<_>>());
            }
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

    /// True when the model differs from the last saved / opened / new one.
    pub fn is_dirty(&self) -> bool {
        self.saved_rev != Some(self.model_rev)
    }
    pub fn mark_saved(&mut self) {
        self.saved_rev = Some(self.model_rev);
    }
    /// Write the model to the autosave file (does not change the project path or saved state).
    pub fn autosave(&self) -> Result<()> {
        let p = autosave_path();
        if let Some(d) = p.parent() {
            std::fs::create_dir_all(d)?;
        }
        self.save(&p.to_string_lossy())
    }
    pub fn save(&self, path: &str) -> Result<()> {
        let v = json!({"simcraft": 1, "model": self.model, "display": self.display, "plan": self.plan});
        std::fs::write(path, serde_json::to_string_pretty(&v)?)?;
        Ok(())
    }
    pub fn open(&mut self, path: &str) -> Result<()> {
        self.open_in_workspace(path, None)
    }

    pub fn open_in_workspace(&mut self, path: &str, workspace: Option<&std::path::Path>) -> Result<()> {
        let resolved = resolve_io_path(path, workspace, false, &["simcraft", "json"])?;
        let txt = std::fs::read_to_string(&resolved).with_context(|| format!("reading {path}"))?;
        let v: Value = serde_json::from_str(&txt)?;
        let mut model: Model = serde_json::from_value(v.get("model").cloned().unwrap_or(v.clone()))?;
        for b in model.bodies.iter_mut() {
            if let Shape::Stl { path: sp, scale, offset, data } = &mut b.shape {
                let asset = resolve_io_path(sp, workspace, false, &["stl"])?;
                *data = Some(Arc::new(load_stl(&asset.to_string_lossy(), *scale, *offset)?));
            }
            if let Shape::Pcb { path: pp, data, .. } = &mut b.shape {
                let asset = resolve_io_path(pp, workspace, false, &["kicad_pcb"])?;
                *data = Some(Arc::new(crate::kicad::PcbData::load(&asset.to_string_lossy())?));
            }
        }
        self.model = model;
        if let Some(d) = v.get("display") {
            if let Ok(d) = serde_json::from_value::<Display>(d.clone()) {
                let dark = self.display.dark;
                self.display = d;
                self.display.dark = dark; // keep the user's theme, not the one saved with the file
            }
        }
        self.plan = v.get("plan").and_then(|p| serde_json::from_value(p.clone()).ok());
        self.mesh = None;
        self.touch_geometry();
        self.project_path = Some(resolved.to_string_lossy().to_string());
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

pub const MAX_PROJECT_BYTES: u64 = 64 * 1024 * 1024;

/// Resolve and validate a file path for read/write tools.
/// Checks for NUL bytes, validates the file extension against `allowed_exts` (when non-empty),
/// and enforces containment inside `workspace` (resolving symlinks and `..`) when set.
pub fn resolve_io_path(raw: &str, workspace: Option<&std::path::Path>, write: bool, allowed_exts: &[&str]) -> Result<std::path::PathBuf> {
    if raw.is_empty() || raw.contains('\0') {
        bail!("invalid file path");
    }
    let candidate = std::path::Path::new(raw);
    if !allowed_exts.is_empty() {
        let ext = candidate.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
        if !allowed_exts.iter().any(|a| *a == ext) {
            bail!("unsupported file extension for '{raw}' (expected: {})", allowed_exts.iter().map(|e| format!(".{e}")).collect::<Vec<_>>().join(", "));
        }
    }
    let Some(root) = workspace else {
        return Ok(candidate.to_path_buf());
    };
    let canon_root = std::fs::canonicalize(root).with_context(|| format!("invalid workspace directory {}", root.display()))?;
    let joined = if candidate.is_absolute() { candidate.to_path_buf() } else { canon_root.join(candidate) };
    if !write {
        let canon = std::fs::canonicalize(&joined).with_context(|| format!("reading {raw}"))?;
        if !canon.starts_with(&canon_root) {
            bail!("path '{raw}' is outside workspace {}", canon_root.display());
        }
        return Ok(canon);
    }
    if let Ok(meta) = std::fs::symlink_metadata(&joined) {
        if meta.file_type().is_symlink() {
            let target = std::fs::canonicalize(&joined).with_context(|| format!("resolving {raw}"))?;
            if !target.starts_with(&canon_root) {
                bail!("path '{raw}' points outside workspace {}", canon_root.display());
            }
        }
    }
    // Walk up to the deepest existing ancestor so we can canonicalize it and validate relative tail.
    let mut existing = joined.as_path();
    let mut tail: Vec<std::ffi::OsString> = Vec::new();
    while !existing.exists() {
        let Some(name) = existing.file_name() else {
            bail!("invalid output path '{raw}'");
        };
        tail.push(name.to_os_string());
        let Some(p) = existing.parent() else {
            bail!("invalid output path '{raw}'");
        };
        existing = p;
    }
    let mut out = std::fs::canonicalize(existing).with_context(|| format!("resolving parent of {raw}"))?;
    if !out.starts_with(&canon_root) {
        bail!("path '{raw}' is outside workspace {}", canon_root.display());
    }
    for comp in tail.into_iter().rev() {
        if comp == ".." {
            bail!("path '{raw}' contains invalid traversal");
        }
        out.push(comp);
    }
    Ok(out)
}

/// External STL / KiCad PCB paths referenced inside a `.simcraft` project that lie outside the
/// project's own directory (or its subdirectories).
pub fn external_assets(project_path: &str) -> Result<Vec<String>> {
    let p = std::path::Path::new(project_path);
    let meta = std::fs::metadata(p).with_context(|| format!("reading {project_path}"))?;
    if meta.len() > MAX_PROJECT_BYTES {
        bail!("project file '{project_path}' is too large ({} MB)", meta.len() / (1024 * 1024));
    }
    let txt = std::fs::read_to_string(p).with_context(|| format!("reading {project_path}"))?;
    let v: Value = serde_json::from_str(&txt)?;
    let model: Model = serde_json::from_value(v.get("model").cloned().unwrap_or(v))?;
    let base = p.parent().and_then(|d| std::fs::canonicalize(d).ok());
    let mut out = Vec::new();
    for b in &model.bodies {
        let asset = match &b.shape {
            Shape::Stl { path, .. } | Shape::Pcb { path, .. } => Some(path.as_str()),
            _ => None,
        };
        if let Some(ap) = asset {
            let ap_path = std::path::Path::new(ap);
            let inside = match (&base, std::fs::canonicalize(ap_path).ok()) {
                (Some(dir), Some(canon)) => canon.starts_with(dir),
                (Some(dir), None) if !ap_path.is_absolute() => !ap_path.components().any(|c| matches!(c, std::path::Component::ParentDir)) && dir.join(ap_path).starts_with(dir),
                _ => false,
            };
            if !inside && !out.iter().any(|x| x == ap) {
                out.push(ap.to_string());
            }
        }
    }
    Ok(out)
}

pub struct Core {
    pub session: Mutex<Session>,
    pub progress: Mutex<Progress>,
    busy: AtomicBool,
    /// Set by `cancel` (Stop button): the running job stops at its next progress update.
    cancel: AtomicBool,
    pub notify: Mutex<Option<Box<dyn Fn() + Send + Sync>>>,
    /// Set by the GUI so `screenshot_gui` can capture the window.
    pub gui_ctx: Mutex<Option<eframe::egui::Context>>,
    /// Optional root directory confining file-taking tools (`SIMCRAFT_WORKSPACE` / `--workspace`).
    workspace: Mutex<Option<std::path::PathBuf>>,
}

impl Core {
    pub fn set_workspace(&self, dir: Option<std::path::PathBuf>) {
        *self.workspace.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = dir;
    }
    pub fn workspace_dir(&self) -> Option<std::path::PathBuf> {
        self.workspace.lock().unwrap_or_else(std::sync::PoisonError::into_inner).clone()
    }
}

pub type Shared = Arc<Core>;

/// Shape moved by d [mm].
fn shift_shape(shape: &Shape, d: [f64; 3]) -> Result<Shape> {
    let add = |p: [f64; 3]| [p[0] + d[0], p[1] + d[1], p[2] + d[2]];
    Ok(match shape.clone() {
        Shape::Box { min, size } => Shape::Box { min: add(min), size },
        Shape::Cylinder { base, axis, radius, height } => Shape::Cylinder { base: add(base), axis, radius, height },
        Shape::Sphere { center, radius } => Shape::Sphere { center: add(center), radius },
        Shape::Stl { path, scale, offset, .. } => {
            let offset = add(offset);
            let data = Some(Arc::new(load_stl(&path, scale, offset)?));
            Shape::Stl { path, scale, offset, data }
        }
        Shape::Pcb { path, offset, data } => Shape::Pcb { path, offset: add(offset), data },
    })
}

/// Shape mirrored across the plane {axis = at}.
fn mirror_shape(shape: &Shape, a: usize, at: f64) -> Result<Shape> {
    Ok(match shape.clone() {
        Shape::Box { mut min, size } => {
            min[a] = 2.0 * at - (min[a] + size[a]);
            Shape::Box { min, size }
        }
        Shape::Cylinder { mut base, axis, radius, height } => {
            base[a] = 2.0 * at - base[a] - if axis.idx() == a { height } else { 0.0 };
            Shape::Cylinder { base, axis, radius, height }
        }
        Shape::Sphere { mut center, radius } => {
            center[a] = 2.0 * at - center[a];
            Shape::Sphere { center, radius }
        }
        _ => bail!("mirroring works on boxes, cylinders and spheres (STL parts and boards cannot be mirrored)"),
    })
}

fn config_dir() -> std::path::PathBuf {
    crate::bridge::token_path().parent().map(|p| p.to_path_buf()).unwrap_or_else(std::env::temp_dir)
}
/// Autosave file (crash recovery).
pub fn autosave_path() -> std::path::PathBuf {
    config_dir().join("autosave.simcraft")
}
/// Recently opened / saved projects, newest first (existing files only).
pub fn recent_files() -> Vec<String> {
    std::fs::read_to_string(config_dir().join("recent.json"))
        .ok()
        .and_then(|t| serde_json::from_str::<Vec<String>>(&t).ok())
        .unwrap_or_default()
        .into_iter()
        .filter(|p| std::path::Path::new(p).exists())
        .collect()
}
fn push_recent(path: &str) {
    let abs = std::fs::canonicalize(path).map(|p| p.to_string_lossy().to_string()).unwrap_or_else(|_| path.to_string());
    let mut v: Vec<String> = recent_files().into_iter().filter(|p| *p != abs).collect();
    v.insert(0, abs);
    v.truncate(10);
    let _ = std::fs::create_dir_all(config_dir());
    let _ = std::fs::write(config_dir().join("recent.json"), serde_json::to_string_pretty(&v).unwrap_or_default());
}

/// Clears the busy flag and the progress bar when a job ends — also when it panics or is cancelled.
struct JobGuard<'a>(&'a Core);
impl Drop for JobGuard<'_> {
    fn drop(&mut self) {
        self.0.busy.store(false, Ordering::SeqCst);
        self.0.cancel.store(false, Ordering::SeqCst);
        self.0.progress.lock().unwrap_or_else(std::sync::PoisonError::into_inner).running = false;
        self.0.changed();
    }
}

pub fn new_core() -> Shared {
    let workspace = std::env::var_os("SIMCRAFT_WORKSPACE").filter(|v| !v.is_empty()).map(std::path::PathBuf::from);
    Arc::new(Core { session: Mutex::new(Session::default()), progress: Mutex::new(Progress::default()), busy: AtomicBool::new(false), cancel: AtomicBool::new(false), notify: Mutex::new(None), gui_ctx: Mutex::new(None), workspace: Mutex::new(workspace) })
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
        let x = arr[i].as_f64().ok_or_else(|| anyhow!("'{k}' must contain finite numbers"))?;
        if !x.is_finite() {
            bail!("'{k}' must contain finite numbers");
        }
        o[i] = x;
    }
    Ok(o)
}
fn v3_or(a: &Value, k: &str, d: V3) -> Result<V3> {
    if a.get(k).is_some_and(|v| !v.is_null()) { v3(a, k) } else { Ok(d) }
}
fn num(a: &Value, k: &str) -> Result<f64> {
    let x = a.get(k).and_then(|v| v.as_f64()).ok_or_else(|| anyhow!("missing number '{k}'"))?;
    if !x.is_finite() {
        bail!("'{k}' must be finite");
    }
    Ok(x)
}
fn opt_num(a: &Value, k: &str) -> Option<f64> {
    a.get(k).and_then(|v| v.as_f64()).filter(|x| x.is_finite())
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

fn tool_may_mutate(tool: &str) -> bool {
    matches!(
        tool,
        "new_model" | "add_box" | "add_cylinder" | "add_sphere" | "import_stl" | "import_kicad"
            | "add_enclosure" | "remove_body" | "define_material" | "set_material"
            | "set_analysis" | "set_mesh" | "edit_body" | "edit_load" | "transform_body"
            | "remove_load" | "clear_loads" | "apply_design" | "open_project"
            | "add_fixed_support" | "add_displacement" | "add_force" | "add_pressure"
            | "add_gravity" | "add_temperature" | "add_heat_flux" | "add_convection"
            | "add_radiation" | "add_heat_generation" | "add_compression_only_support"
            | "add_symmetry" | "add_voltage" | "add_current" | "add_coil" | "add_magnet"
            | "add_inlet" | "add_moving_wall" | "add_port" | "add_outlet"
    )
}

impl Core {
    fn changed(&self) {
        if let Some(f) = self.notify.lock().unwrap_or_else(std::sync::PoisonError::into_inner).as_ref() {
            f();
        }
    }

    pub fn is_busy(&self) -> bool {
        self.busy.load(Ordering::SeqCst)
    }

    /// Execute a tool by name. Used by the GUI, the TCP bridge and the MCP server.
    pub fn call(&self, tool: &str, args: &Value) -> Result<Output> {
        // Explicit command-level snapshot (O(1) dirty check; skips read-only queries and solvers).
        let before = if tool_may_mutate(tool) {
            let s = self.session.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            Some((s.model_rev, s.model.clone()))
        } else {
            None
        };
        let r = self.call_inner(tool, args);
        {
            let mut s = self.session.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            if let (Ok(_), Some((rev_before, model_before))) = (&r, before) {
                if s.model_rev != rev_before {
                    if tool == "open_project" {
                        s.undo.clear();
                        s.undo_revs.clear();
                    } else {
                        s.undo.push(model_before);
                        s.undo_revs.push(rev_before);
                        if s.undo.len() > 100 {
                            s.undo.remove(0);
                            s.undo_revs.remove(0);
                        }
                    }
                    s.redo.clear();
                    s.redo_revs.clear();
                }
            }
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
        if tool == "__panic" && std::env::var_os("SIMCRAFT_TEST_HOOKS").is_some() {
            // test hook: a solver that panics half-way
            return self.run_job("solve", || -> Result<Output> {
                let v: Vec<f64> = vec![];
                let i = v.len() + 3;
                Ok(Output::text(format!("{}", v[i])))
            });
        }
        if tool == "__ui" && std::env::var_os("SIMCRAFT_TEST_HOOKS").is_some() {
            let mut s = self.session.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            s.ui_request = Some(a.clone());
            s.revision += 1;
            drop(s);
            self.changed();
            return Ok(Output::text("ok"));
        }
        if tool == "cancel" {
            if !self.is_busy() {
                return Ok(Output::text("Nothing is running."));
            }
            self.cancel.store(true, Ordering::SeqCst);
            return Ok(Output::text("Stopping the running job at its next progress step…"));
        }
        if tool == "screenshot_gui" {
            return self.screenshot_gui();
        }
        if matches!(tool, "run_parameter_study" | "optimize_design" | "topology_optimize" | "surrogate_optimize" | "mesh_convergence") {
            return self.study_job(tool, a);
        }
        if tool == "plan_simulation" {
            return self.plan_simulation(a);
        }
        if tool == "list_local_models" {
            let models = crate::assistant::list_models()?;
            return Ok(Output::json(json!({"server": crate::assistant::base_url(), "models": models, "default": crate::assistant::default_model(&models)})));
        }
        if self.is_busy() && !matches!(tool, "get_model" | "get_help" | "get_results" | "list_faces" | "list_materials" | "list_tools" | "render_image" | "probe" | "plot_path" | "get_status" | "set_display" | "get_selection" | "get_plan" | "set_plan_step" | "get_study" | "list_parameter_targets") {
            bail!("a solve is running; wait for it to finish (use get_status)");
        }
        let ws = self.workspace_dir();
        let ws_ref = ws.as_deref();
        let mut guard = self.session.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let s = &mut *guard;
        match tool {
            "get_model" => Ok(Output::json(s.model_json())),
            "list_tools" => Ok(Output::json(json!(tool_defs()))),
            "get_status" => {
                let p = self.progress.lock().unwrap_or_else(std::sync::PoisonError::into_inner).clone();
                Ok(Output::json(json!({"solving": p.running, "progress": p.fraction, "message": p.message, "last_error": s.last_error, "unsaved_changes": s.is_dirty(), "project": s.project_path, "recent_log": s.log.iter().rev().take(8).rev().collect::<Vec<_>>()})))
            }
            "new_model" => {
                let rev = s.revision;
                let next = s.next_model_rev;
                let empty = s.model_rev == 0 && s.model.bodies.is_empty() && s.model.loads.is_empty() && s.model.custom_materials.is_empty() && s.model.element_size.is_none();
                // the theme and edge display are user preferences, not part of the model
                let (dark, show_mesh) = (s.display.dark, s.display.show_mesh);
                let plan = s.plan.take();
                *s = Session::default();
                s.plan = plan;
                s.display.dark = dark;
                s.display.show_mesh = show_mesh;
                s.revision = rev + 1;
                s.geom_rev = rev + 1;
                if empty {
                    s.next_model_rev = next;
                } else {
                    s.model_rev = next;
                    s.next_model_rev = next + 1;
                    s.saved_rev = Some(next);
                }
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
                let raw = opt_str(a, "path").ok_or_else(|| anyhow!("missing 'path'"))?;
                let resolved = resolve_io_path(raw, ws_ref, false, &["stl"])?;
                let path = resolved.to_string_lossy().to_string();
                let sc = opt_num(a, "scale").unwrap_or(1.0);
                let off = v3_or(a, "offset", [0.0; 3])?;
                let data = Arc::new(load_stl(&path, sc, off)?);
                let nseg = data.n_segments;
                let ntri = data.tris.len();
                let stem = resolved.file_stem().and_then(|x| x.to_str()).unwrap_or("Part").to_string();
                let shape = Shape::Stl { path: path.clone(), scale: sc, offset: off, data: Some(data) };
                let n = s.add_body(Some(opt_str(a, "name").unwrap_or(&stem)), "Part", shape, op(a)?, opt_str(a, "material"))?;
                s.view_request = Some("fit".into());
                Ok(Output::text(format!("Imported '{path}' as body '{n}' ({ntri} triangles, {nseg} smooth faces). Use list_faces to see face names.")))
            }
            "pcb_nets" => {
                let raw = match opt_str(a, "path") {
                    Some(p) => p.to_string(),
                    None => s.pcb_path().ok_or_else(|| anyhow!("give 'path' (.kicad_pcb) or import a board first"))?,
                };
                let resolved = resolve_io_path(&raw, ws_ref, false, &["kicad_pcb"])?;
                let path = resolved.to_string_lossy().to_string();
                let nets = crate::kicad::dc::net_pads(&path)?;
                Ok(Output::json(json!({"path": path, "nets": nets.iter().map(|(n, p)| json!({"net": n, "pads": p})).collect::<Vec<_>>()})))
            }
            "pcb_dc" => {
                let raw = match opt_str(a, "path") {
                    Some(p) => p.to_string(),
                    None => s.pcb_path().ok_or_else(|| anyhow!("give 'path' (.kicad_pcb) or import a board first"))?,
                };
                let resolved = resolve_io_path(&raw, ws_ref, false, &["kicad_pcb"])?;
                let path = resolved.to_string_lossy().to_string();
                let from = opt_str(a, "from").ok_or_else(|| anyhow!("missing 'from' (pad as REFERENCE.PAD, e.g. U1.3)"))?;
                let to = opt_str(a, "to").ok_or_else(|| anyhow!("missing 'to' (pad as REFERENCE.PAD, e.g. J1.1)"))?;
                let current = opt_num(a, "current").unwrap_or(1.0);
                let temp = opt_num(a, "temperature_c").unwrap_or(20.0);
                let r = crate::kicad::dc::dc_drop(&path, opt_str(a, "net"), from, to, current, temp)?;
                Ok(Output::json(r))
            }
            "import_kicad" => {
                let raw = opt_str(a, "path").ok_or_else(|| anyhow!("missing 'path' (.kicad_pcb file)"))?;
                let resolved = resolve_io_path(raw, ws_ref, false, &["kicad_pcb"])?;
                let path = resolved.to_string_lossy().to_string();
                let data = std::sync::Arc::new(crate::kicad::PcbData::load(&path)?);
                let off = v3_or(a, "offset", [0.0; 3])?;
                let bname = s.add_body(Some(opt_str(a, "name").unwrap_or("Board")), "Board", Shape::Pcb { path: path.clone(), offset: off, data: Some(data.clone()) }, Op::Add, Some(opt_str(a, "material").unwrap_or("FR-4 (PCB laminate)")))?;
                let with_parts = a.get("components").and_then(|v| v.as_bool()).unwrap_or(true);
                let h_default = opt_num(a, "component_height_mm").unwrap_or(1.0);
                let min_area = opt_num(a, "min_component_area_mm2").unwrap_or(4.0);
                let heights = a.get("heights").and_then(|v| v.as_object()).cloned().unwrap_or_default();
                let powers = a.get("power").and_then(|v| v.as_object()).cloned().unwrap_or_default();
                let mut created = Vec::new();
                let mut skipped = 0usize;
                let mut loads = Vec::new();
                if with_parts {
                    for fp in &data.footprints {
                        let size = [fp.hi[0] - fp.lo[0], fp.hi[1] - fp.lo[1]];
                        let listed = heights.contains_key(&fp.reference) || powers.contains_key(&fp.reference);
                        if size[0] <= 0.0 || size[1] <= 0.0 || (!listed && size[0] * size[1] < min_area) {
                            skipped += 1;
                            continue;
                        }
                        let ht = heights.get(&fp.reference).and_then(|v| v.as_f64()).unwrap_or(h_default).max(0.05);
                        let z = if fp.bottom { off[2] - ht } else { off[2] + data.thickness };
                        let shape = Shape::Box { min: [off[0] + fp.lo[0], off[1] + fp.lo[1], z], size: [size[0], size[1], ht] };
                        let n = s.add_body(Some(&fp.reference), "Part", shape, Op::Add, Some("IC Package (mold compound)"))?;
                        if let Some(p) = powers.get(&fp.reference).and_then(|v| v.as_f64()) {
                            let id = s.add_load(Load::HeatGeneration { value: 0.0, body: Some(n.clone()), power: Some(p) })?;
                            loads.push(json!({"id": id, "body": n, "power_w": p}));
                        }
                        created.push(json!({"body": n, "value": fp.value, "footprint": fp.footprint, "side": if fp.bottom { "bottom" } else { "top" }, "size_mm": [crate::results::float(size[0]), crate::results::float(size[1]), ht]}));
                    }
                }
                let missing: Vec<&String> = powers.keys().filter(|r| !data.footprints.iter().any(|f| &f.reference == *r)).collect();
                s.view_request = Some("fit".into());
                s.log(format!("Imported KiCad board '{path}' ({} components)", created.len()));
                let mut out = data.summary();
                out["board_body"] = json!(bname);
                out["components"] = json!(created);
                out["small_footprints_skipped"] = json!(skipped);
                out["heat_loads"] = json!(loads);
                if !missing.is_empty() {
                    out["power_references_not_found"] = json!(missing);
                }
                out["note"] = json!(format!("Board faces: {bname}.top, {bname}.bottom, {bname}.edge. Copper is homogenised into orthotropic conductivity per cell — use an element size ≤ half the board thickness ({:.2} mm) through the board, e.g. set_mesh 0.8 for a 1.6 mm board. Thermal analyses (steady/transient thermal, conjugate heat transfer) use it; add_heat_generation with power on component bodies.", data.thickness / 2.0));
                Ok(Output::json(out))
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
                set("fatigue_strength_coefficient_mpa", &mut m.fatigue_strength_coefficient_mpa);
                set("fatigue_exponent", &mut m.fatigue_exponent);
                set("endurance_limit_mpa", &mut m.endurance_limit_mpa);
                set("relative_permittivity", &mut m.relative_permittivity);
                set("electrical_conductivity", &mut m.electrical_conductivity);
                set("relative_permeability", &mut m.relative_permeability);
                set("loss_tangent", &mut m.loss_tangent);
                let table = |k: &str| -> Result<Option<Vec<[f64; 2]>>> {
                    let Some(v) = a.get(k) else { return Ok(None) };
                    let rows: Vec<[f64; 2]> = serde_json::from_value(v.clone()).map_err(|_| anyhow!("{k} must be [[temperature °C, value], …]"))?;
                    if rows.iter().any(|r| !r[0].is_finite() || !r[1].is_finite() || !(r[1] > 0.0)) || rows.windows(2).any(|w| w[1][0] <= w[0][0]) {
                        bail!("{k}: temperatures must increase and values must be finite and > 0");
                    }
                    Ok(Some(rows))
                };
                if let Some(t) = table("conductivity_table")? {
                    if let Some(first) = t.first() {
                        if !t.iter().any(|r| r[1] == m.thermal_conductivity) {
                            m.thermal_conductivity = first[1];
                        }
                    }
                    m.conductivity_table = t;
                }
                if let Some(t) = table("specific_heat_table")? {
                    m.specific_heat_table = t;
                }
                let finite_ok = [
                    m.youngs_modulus_gpa, m.poisson_ratio, m.density, m.yield_strength_mpa, m.ultimate_strength_mpa,
                    m.thermal_conductivity, m.thermal_expansion, m.specific_heat, m.relative_permittivity,
                    m.electrical_conductivity, m.relative_permeability, m.loss_tangent,
                ].iter().all(|v| v.is_finite());
                if !finite_ok {
                    bail!("invalid material: all numeric properties must be finite");
                }
                if m.relative_permittivity < 1.0 || m.electrical_conductivity < 0.0 || m.relative_permeability <= 0.0 || m.loss_tangent < 0.0 {
                    bail!("invalid electromagnetic properties: need εr ≥ 1, σ ≥ 0, μr > 0, tan δ ≥ 0");
                }
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
                let mut changed = false;
                for b in s.model.bodies.iter_mut() {
                    if targets.as_ref().is_none_or(|t| t.contains(&b.name.to_lowercase())) {
                        if b.material != m.name {
                            b.material = m.name.clone();
                            changed = true;
                        }
                        n += 1;
                    }
                }
                if n == 0 {
                    bail!("no matching bodies");
                }
                if changed {
                    s.touch_geometry();
                }
                s.log(format!("Material '{}' assigned to {n} body(ies)", m.name));
                Ok(Output::text(format!("Assigned {} to {n} body(ies).", m.name)))
            }
            "set_analysis" => {
                if let Some(t) = opt_str(a, "type") {
                    s.model.analysis.kind = serde_json::from_value(json!(t)).map_err(|_| anyhow!("type must be one of {:?}", AnalysisType::all().iter().map(|a| a.key()).collect::<Vec<_>>()))?;
                }
                let st = &mut s.model.analysis;
                if let Some(v) = opt_num(a, "end_time") {
                    st.end_time = v.max(1e-9);
                }
                if let Some(v) = opt_num(a, "time_steps") {
                    st.time_steps = (v as usize).clamp(1, 5000);
                }
                if let Some(v) = opt_num(a, "initial_temperature") {
                    st.initial_temperature = v;
                }
                if let Some(v) = opt_num(a, "damping_ratio") {
                    st.damping_ratio = v.clamp(0.0, 1.0);
                }
                if let Some(v) = opt_num(a, "load_steps") {
                    st.load_steps = (v as usize).clamp(1, 200);
                }
                if let Some(v) = opt_num(a, "max_iterations") {
                    st.max_iterations = (v as usize).clamp(50, 100_000);
                }
                if let Some(v) = opt_num(a, "freq_min") {
                    st.freq_min = if v > 0.0 { Some(v) } else { None };
                }
                if let Some(v) = opt_num(a, "freq_max") {
                    st.freq_max = if v > 0.0 { Some(v) } else { None };
                }
                if let Some(v) = opt_num(a, "freq_steps") {
                    st.freq_steps = (v as usize).clamp(20, 5000);
                }
                if let Some(t) = a.get("table") {
                    let pts: Vec<[f64; 2]> = serde_json::from_value(t.clone()).map_err(|_| anyhow!("table must be [[frequency_hz, value], ...]"))?;
                    if pts.iter().any(|p| p[0] <= 0.0 || p[1] < 0.0) {
                        bail!("table frequencies must be > 0 and values ≥ 0");
                    }
                    st.table = pts;
                }
                if let Some(d) = opt_str(a, "direction") {
                    st.direction = serde_json::from_value(json!(d.to_lowercase())).map_err(|_| anyhow!("direction must be x, y or z"))?;
                }
                if let Some(c) = opt_str(a, "combination") {
                    if !matches!(c.to_lowercase().as_str(), "srss" | "cqc") {
                        bail!("combination must be srss or cqc");
                    }
                    st.combination = c.to_lowercase();
                }
                if let Some(b) = a.get("boundary_fit").and_then(|v| v.as_bool()) {
                    st.boundary_fit = b;
                }
                if let Some(t) = opt_str(a, "cfd_scheme") {
                    let t = t.to_lowercase();
                    if !matches!(t.as_str(), "auto" | "simple" | "projection") {
                        bail!("cfd_scheme must be auto, simple or projection");
                    }
                    st.cfd_scheme = t;
                }
                if let Some(t) = opt_str(a, "turbulence") {
                    let t = t.to_lowercase();
                    if !matches!(t.as_str(), "laminar" | "lvel") {
                        bail!("turbulence must be laminar or lvel");
                    }
                    st.turbulence = t;
                }
                if let Some(v) = opt_num(a, "frequency_hz") {
                    if !(v > 0.0) {
                        bail!("frequency_hz must be > 0");
                    }
                    st.frequency_hz = v;
                }
                if let Some(v) = opt_num(a, "frequency_ghz") {
                    if !(v > 0.0) {
                        bail!("frequency_ghz must be > 0");
                    }
                    st.frequency_hz = v * 1e9;
                }
                if let Some(b) = opt_str(a, "em_boundary") {
                    let b = b.to_lowercase();
                    if !matches!(b.as_str(), "pec" | "open") {
                        bail!("em_boundary must be pec (closed metal cavity) or open (absorbing)");
                    }
                    st.em_boundary = b;
                }
                if let Some(v) = opt_num(a, "max_periods") {
                    st.max_periods = (v as usize).clamp(24, 20_000);
                }
                if let Some(v) = opt_str(a, "compute") {
                    st.compute = serde_json::from_value(json!(v)).map_err(|_| anyhow!("compute must be auto, cpu or gpu"))?;
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
                let mut note = String::new();
                if s.model.analysis.kind.requires_enclosure() && !s.model.has_enclosure() {
                    note = " It needs an enclosure (air/fluid box): call add_enclosure.".into();
                }
                Ok(Output::json(json!({"analysis": l, "settings": s.model.analysis, "note": note})))
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
            "add_radiation" => {
                let eps = opt_num(a, "emissivity").unwrap_or(0.9);
                if !(eps > 0.0 && eps <= 1.0) {
                    bail!("emissivity must be in (0, 1] (polished metal ≈ 0.05, anodised / painted ≈ 0.8–0.95)");
                }
                let l = Load::Radiation { faces: faces(a)?, emissivity: eps, ambient: opt_num(a, "ambient").unwrap_or(22.0), enclosure: a.get("enclosure").and_then(|v| v.as_bool()).unwrap_or(false) };
                Ok(load_out(s.add_load(l)?))
            }
            "add_heat_generation" => {
                let power = opt_num(a, "power");
                let value = match (opt_num(a, "value"), power) {
                    (Some(v), _) => v,
                    (None, Some(_)) => 0.0,
                    (None, None) => bail!("give 'power' (W) or 'value' (W/m³)"),
                };
                if power.is_some_and(|p| p < 0.0) {
                    bail!("power must be ≥ 0");
                }
                Ok(load_out(s.add_load(Load::HeatGeneration { value, body: opt_str(a, "body").filter(|b| !b.is_empty()).map(String::from), power })?))
            }
            "add_compression_only_support" => Ok(load_out(s.add_load(Load::CompressionOnly { faces: faces(a)? })?)),
            "add_symmetry" => Ok(load_out(s.add_load(Load::Symmetry { faces: faces(a)? })?)),
            "add_voltage" => Ok(load_out(s.add_load(Load::Voltage { faces: faces(a)?, value: num(a, "value")? })?)),
            "add_current" => Ok(load_out(s.add_load(Load::Current { faces: faces(a)?, value: num(a, "value")? })?)),
            "add_coil" => {
                let mode = match opt_str(a, "mode").unwrap_or("circumferential") {
                    "axial" => CoilMode::Axial,
                    "circumferential" => CoilMode::Circumferential,
                    o => bail!("mode must be 'axial' or 'circumferential', got '{o}'"),
                };
                let body = opt_str(a, "body").ok_or_else(|| anyhow!("missing 'body'"))?.to_string();
                let center = if a.get("center").is_some_and(|v| !v.is_null()) { Some(v3(a, "center")?) } else { None };
                Ok(load_out(s.add_load(Load::Coil { body, current_density: num(a, "current_density")?, axis: axis(a)?, mode, center })?))
            }
            "add_magnet" => {
                let body = opt_str(a, "body").ok_or_else(|| anyhow!("missing 'body'"))?.to_string();
                Ok(load_out(s.add_load(Load::Magnet { body, remanence: v3(a, "remanence")? })?))
            }
            "add_inlet" => Ok(load_out(s.add_load(Load::Inlet { faces: faces(a)?, velocity: num(a, "velocity")?, temperature: opt_num(a, "temperature") })?)),
            "add_moving_wall" => Ok(load_out(s.add_load(Load::MovingWall { faces: faces(a)?, velocity: v3(a, "velocity")? })?)),
            "add_port" => {
                let power = num(a, "power")?;
                if !(power > 0.0) {
                    bail!("power must be > 0 W");
                }
                let pol = opt_str(a, "polarization").ok_or_else(|| anyhow!("missing 'polarization' (x, y or z: the E-field direction, lying in the port face)"))?;
                let polarization = serde_json::from_value(json!(pol.to_lowercase())).map_err(|_| anyhow!("polarization must be x, y or z"))?;
                let profile = opt_str(a, "profile").unwrap_or("te10").to_lowercase();
                if !matches!(profile.as_str(), "te10" | "uniform") {
                    bail!("profile must be te10 or uniform");
                }
                Ok(load_out(s.add_load(Load::Port { faces: faces(a)?, power, polarization, profile })?))
            }
            "add_outlet" => Ok(load_out(s.add_load(Load::Outlet { faces: faces(a)?, pressure: opt_num(a, "pressure").unwrap_or(0.0) })?)),
            "add_enclosure" => {
                let (lo, hi) = mesh::additive_bounds(&s.model.bodies);
                if lo == hi && !a.get("size").is_some() {
                    bail!("add the solid bodies first, or give min/size for the enclosure");
                }
                let shape = if a.get("size").is_some_and(|v| !v.is_null()) {
                    Shape::Box { min: v3_or(a, "min", [0.0; 3])?, size: v3(a, "size")? }
                } else {
                    // padding as a fraction of the part size per side (default 50%), or explicit per-axis padding in mm
                    let d = sub(hi, lo);
                    let pad: [f64; 6] = match a.get("padding_mm").and_then(|v| v.as_array()) {
                        Some(arr) if arr.len() == 6 => {
                            let mut p = [0.0; 6];
                            for i in 0..6 {
                                p[i] = arr[i].as_f64().unwrap_or(0.0);
                            }
                            p
                        }
                        _ => {
                            let f = opt_num(a, "padding").unwrap_or(0.5);
                            let m = d[0].max(d[1]).max(d[2]) * f;
                            [m; 6]
                        }
                    };
                    let min = [lo[0] - pad[0], lo[1] - pad[2], lo[2] - pad[4]];
                    let size = [d[0] + pad[0] + pad[1], d[1] + pad[2] + pad[3], d[2] + pad[4] + pad[5]];
                    Shape::Box { min, size }
                };
                s.model.bodies.retain(|b| b.role != Role::Enclosure);
                let mat = opt_str(a, "material").unwrap_or("Air");
                let mat = s.resolve_material(mat)?.name;
                let n = s.add_body_role(Some(opt_str(a, "name").unwrap_or("Enclosure")), "Enclosure", shape, Op::Add, Some(&mat), Role::Enclosure)?;
                s.view_request = Some("fit".into());
                Ok(Output::text(format!(
                    "Added enclosure '{n}' ({mat}). Its outer faces are {n}.xmin … {n}.zmax (use them for inlets/outlets). It is only meshed for electrostatic, magnetostatic and fluid analyses."
                )))
            }
            "get_help" => {
                use crate::help;
                if let Some(q) = opt_str(a, "query") {
                    let hits: Vec<Value> = help::search(q, 6).into_iter().map(|(i, score, snip)| json!({"section": help::sections()[i].title, "score": crate::results::float(score), "snippet": snip})).collect();
                    if hits.is_empty() {
                        return Ok(Output::text(format!("Nothing in the guide matches '{q}'. Try fewer or different words, or get_help without arguments for the contents.")));
                    }
                    return Ok(Output::json(json!({"query": q, "results": hits, "next": "get_help {topic: <section>} for the full text"})));
                }
                if let Some(t) = opt_str(a, "topic") {
                    let i = help::find(t).ok_or_else(|| anyhow!("no help topic '{t}'. Use get_help without arguments for the contents, or get_help {{query}} to search"))?;
                    return Ok(Output::text(help::section_text(i)));
                }
                let toc: Vec<Value> = help::sections().iter().filter(|x| x.level >= 2).map(|x| json!({"section": x.title, "level": x.level})).collect();
                Ok(Output::json(json!({"guide": "SimCraft User Guide", "contents": toc, "next": "get_help {topic: '7'} or {topic: 'meshing'} for a section, {query: 'heat sink'} to search"})))
            }
            "undo" | "redo" => {
                let undo = tool == "undo";
                let prev = if undo { s.undo.pop().zip(s.undo_revs.pop()) } else { s.redo.pop().zip(s.redo_revs.pop()) };
                let Some((prev, prev_rev)) = prev else { bail!("nothing to {tool}") };
                let cur = std::mem::replace(&mut s.model, prev);
                let cur_rev = std::mem::replace(&mut s.model_rev, prev_rev);
                if undo {
                    s.redo.push(cur);
                    s.redo_revs.push(cur_rev);
                } else {
                    s.undo.push(cur);
                    s.undo_revs.push(cur_rev);
                }
                s.invalidate_geometry();
                let msg = format!("{} — {} more undo, {} redo available", if undo { "Undid the last change" } else { "Redid the change" }, s.undo.len(), s.redo.len());
                s.log(msg.clone());
                Ok(Output::text(msg))
            }
            "edit_body" => {
                let name = opt_str(a, "name").ok_or_else(|| anyhow!("missing 'name'"))?.to_string();
                let bi = s.model.bodies.iter().position(|b| b.name.eq_ignore_ascii_case(&name)).ok_or_else(|| anyhow!("no body named '{name}'"))?;
                let old = s.model.bodies[bi].clone();
                let mut b = old.clone();
                // shape fields
                let mut sj = serde_json::to_value(&b.shape)?;
                let mut changed_shape = false;
                for key in ["min", "size", "base", "axis", "radius", "height", "center", "scale", "offset"] {
                    if let Some(v) = a.get(key) {
                        if sj.get(key).is_none() {
                            bail!("a {} body has no '{key}' (it has: {})", sj["type"].as_str().unwrap_or("?"), sj.as_object().map(|o| o.keys().filter(|k| *k != "type" && *k != "path").cloned().collect::<Vec<_>>().join(", ")).unwrap_or_default());
                        }
                        sj[key] = v.clone();
                        changed_shape = true;
                    }
                }
                if changed_shape {
                    let mut shape: Shape = serde_json::from_value(sj).map_err(|e| anyhow!("bad shape values: {e}"))?;
                    match (&mut shape, &old.shape) {
                        (Shape::Stl { path, scale, offset, data }, _) => *data = Some(Arc::new(load_stl(path, *scale, *offset)?)),
                        (Shape::Pcb { data, .. }, Shape::Pcb { data: od, .. }) => *data = od.clone(),
                        _ => {}
                    }
                    shape.validate()?;
                    b.shape = shape;
                }
                if let Some(m) = opt_str(a, "material") {
                    b.material = s.resolve_material(m)?.name;
                }
                if let Some(o) = opt_str(a, "operation") {
                    b.op = match o {
                        "add" => Op::Add,
                        "subtract" => Op::Subtract,
                        _ => bail!("operation must be add or subtract"),
                    };
                }
                let new_name = opt_str(a, "new_name").map(|n| n.trim().to_string()).filter(|n| !n.is_empty() && *n != old.name);
                if let Some(n) = &new_name {
                    if !n.chars().all(|c| c.is_alphanumeric() || c == '_' || c == '-') {
                        bail!("names may contain letters, digits, _ and - only");
                    }
                    if s.model.bodies.iter().any(|x| x.name.eq_ignore_ascii_case(n)) {
                        bail!("a body named '{n}' already exists");
                    }
                    b.name = n.clone();
                }
                s.model.bodies[bi] = b.clone();
                if let Some(n) = &new_name {
                    // keep loads pointing at the renamed body
                    let (op, np) = (format!("{}.", old.name), format!("{n}."));
                    for it in s.model.loads.iter_mut() {
                        let mut v = serde_json::to_value(&it.load)?;
                        if let Some(fs) = v.get_mut("faces").and_then(|f| f.as_array_mut()) {
                            for f in fs.iter_mut() {
                                if let Some(t) = f.as_str() {
                                    if t == old.name {
                                        *f = json!(n);
                                    } else if let Some(rest) = t.strip_prefix(&op) {
                                        *f = json!(format!("{np}{rest}"));
                                    }
                                }
                            }
                        }
                        if v.get("body").and_then(|x| x.as_str()) == Some(old.name.as_str()) {
                            v["body"] = json!(n);
                        }
                        it.load = serde_json::from_value(v)?;
                    }
                }
                s.touch_geometry();
                s.log(format!("Edited body '{}'", b.name));
                Ok(Output::json(json!({"edited": b.name, "shape": b.shape, "material": b.material, "operation": b.op})))
            }
            "edit_load" => {
                let id = num(a, "id")? as u32;
                let li = s.model.loads.iter().position(|l| l.id == id).ok_or_else(|| anyhow!("no load with id {id}"))?;
                let mut v = serde_json::to_value(&s.model.loads[li].load)?;
                let mut n = 0;
                for (k, val) in a.as_object().into_iter().flatten() {
                    if k == "id" || k == "kind" {
                        continue;
                    }
                    if v.get(k).is_none() && !matches!(k.as_str(), "body" | "temperature" | "power" | "profile") {
                        bail!("a {} load has no '{k}' (it has: {})", s.model.loads[li].load.label(), v.as_object().map(|o| o.keys().filter(|x| *x != "kind").cloned().collect::<Vec<_>>().join(", ")).unwrap_or_default());
                    }
                    v[k] = val.clone();
                    n += 1;
                }
                if n == 0 {
                    bail!("give the fields to change, e.g. {{\"id\": {id}, \"value\": 5}}");
                }
                let load: Load = serde_json::from_value(v).map_err(|e| anyhow!("bad load values: {e}"))?;
                if !load.faces().is_empty() {
                    let m = s.ensure_mesh()?;
                    m.resolve_faces(load.faces())?;
                }
                if let Some(b) = load.body() {
                    if !s.model.bodies.iter().any(|x| x.name.eq_ignore_ascii_case(b)) {
                        bail!("no body named '{b}'");
                    }
                }
                s.model.loads[li].load = load.clone();
                s.touch_setup();
                s.log(format!("Edited load #{id}"));
                Ok(Output::json(json!({"edited": id, "load": load, "summary": load.summary()})))
            }
            "transform_body" => {
                let name = opt_str(a, "name").ok_or_else(|| anyhow!("missing 'name'"))?.to_string();
                let src = s.model.bodies.iter().find(|b| b.name.eq_ignore_ascii_case(&name)).cloned().ok_or_else(|| anyhow!("no body named '{name}'"))?;
                let mode = opt_str(a, "mode").unwrap_or("move");
                let mut made: Vec<String> = Vec::new();
                match mode {
                    "move" | "copy" => {
                        let d = v3(a, "translate")?;
                        let shape = shift_shape(&src.shape, d)?;
                        if mode == "move" {
                            let b = s.model.bodies.iter_mut().find(|b| b.name == src.name).unwrap();
                            b.shape = shape;
                            made.push(src.name.clone());
                        } else {
                            let n = s.unique_name(&format!("{}_copy", src.name));
                            s.model.bodies.push(Body { name: n.clone(), shape, ..src.clone() });
                            made.push(n);
                        }
                    }
                    "pattern" => {
                        let d = v3(a, "translate")?;
                        let count = opt_num(a, "count").unwrap_or(2.0) as usize;
                        if !(2..=200).contains(&count) {
                            bail!("count must be 2..200 (total instances including the original)");
                        }
                        for i in 1..count {
                            let shape = shift_shape(&src.shape, [d[0] * i as f64, d[1] * i as f64, d[2] * i as f64])?;
                            let n = s.unique_name(&format!("{}_{}", src.name, i + 1));
                            s.model.bodies.push(Body { name: n.clone(), shape, ..src.clone() });
                            made.push(n);
                        }
                    }
                    "mirror" => {
                        let ax = axis(a)?.idx();
                        let at = opt_num(a, "at").unwrap_or(0.0);
                        let shape = mirror_shape(&src.shape, ax, at)?;
                        if a.get("copy").and_then(|v| v.as_bool()).unwrap_or(true) {
                            let n = s.unique_name(&format!("{}_mirror", src.name));
                            s.model.bodies.push(Body { name: n.clone(), shape, ..src.clone() });
                            made.push(n);
                        } else {
                            let b = s.model.bodies.iter_mut().find(|b| b.name == src.name).unwrap();
                            b.shape = shape;
                            made.push(src.name.clone());
                        }
                    }
                    m => bail!("mode must be move, copy, pattern or mirror (got '{m}')"),
                }
                s.touch_geometry();
                s.view_request = Some("fit".into());
                s.log(format!("{mode}: {}", made.join(", ")));
                Ok(Output::json(json!({"mode": mode, "bodies": made, "note": "copies have no loads of their own: add supports/loads to their faces if needed"})))
            }
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
                if !s.model.loads.is_empty() {
                    s.model.loads.clear();
                    s.touch_setup();
                }
                Ok(Output::text("All supports and loads removed."))
            }
            "set_mesh" => {
                let es = opt_num(a, "element_size_mm");
                if let Some(e) = es {
                    if e <= 0.0 {
                        bail!("element_size_mm must be positive");
                    }
                }
                if s.model.element_size != es {
                    s.model.element_size = es;
                    s.touch_geometry();
                }
                let m = s.ensure_mesh()?;
                s.log(format!("Mesh: {} elements, {} nodes, size {} mm", m.elems.len(), m.nodes.len(), m.h));
                Ok(Output::json(json!({"element_size_mm": m.h, "elements": m.elems.len(), "nodes": m.nodes.len(), "dofs": m.nodes.len()*3, "warnings": m.warnings})))
            }
            "get_results" => {
                let r = s.results.clone().ok_or_else(|| anyhow!("no results yet; run solve"))?;
                let m = s.mesh.clone().unwrap();
                let mut v = r.summary_with(&m, s.display.cylindrical);
                if s.results_stale {
                    v["stale"] = json!("setup changed after this solve; run solve again");
                }
                Ok(Output::json(v))
            }
            "probe" => {
                let r = s.results.clone().ok_or_else(|| anyhow!("no results yet; run solve"))?;
                let m = s.mesh.clone().unwrap();
                let p = v3(a, "point")?;
                let mut cyl = s.display.cylindrical;
                apply_cylindrical(&mut cyl, a)?;
                let n = m.nearest_node(p);
                let mut vals = serde_json::Map::new();
                let mode = opt_num(a, "mode").map(|x| (x as usize).saturating_sub(1)).unwrap_or(0);
                for f in r.available_fields() {
                    if let Some(v) = r.scalar_at(&m, f, mode, cyl) {
                        vals.insert(f.key(), json!(crate::results::float(v[n])));
                    }
                }
                Ok(Output::json(json!({"requested_point_mm": p, "nearest_node_mm": m.nodes[n], "values": vals})))
            }
            "plot_path" => {
                let r = s.results.clone().ok_or_else(|| anyhow!("no results yet; run solve"))?;
                let m = s.mesh.clone().unwrap();
                let from = v3(a, "from")?;
                let to = v3(a, "to")?;
                let samples = opt_num(a, "samples").unwrap_or(50.0).clamp(2.0, 500.0) as usize;
                let mode = opt_num(a, "mode").map(|x| (x as usize).saturating_sub(1)).unwrap_or(s.display.mode);
                let mut cyl = s.display.cylindrical;
                apply_cylindrical(&mut cyl, a)?;
                let field = field_arg(a)?.unwrap_or_else(|| if r.has(s.display.field) { s.display.field } else { r.available_fields()[0] });
                let pp = r.path_plot(&m, field, mode, cyl, from, to, samples)?;
                let csv = pp.to_csv();
                let saved = if let Some(p) = opt_str(a, "csv_path") {
                    let resolved = resolve_io_path(p, ws_ref, true, &["csv"])?;
                    std::fs::write(&resolved, &csv).with_context(|| format!("writing path CSV {}", resolved.display()))?;
                    Some(resolved.to_string_lossy().to_string())
                } else {
                    None
                };
                let mut out = serde_json::to_value(&pp)?;
                out["csv"] = json!(csv);
                if let Some(p) = saved {
                    out["csv_path"] = json!(p);
                }
                Ok(Output::json(out))
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
                if let Some(b) = a.get("perspective").and_then(|v| v.as_bool()) {
                    s.display.perspective = b;
                }
                if let Some(b) = a.get("translucent_enclosure").and_then(|v| v.as_bool()) {
                    s.display.translucent_enclosure = b;
                }
                apply_section(&mut s.display, a)?;
                apply_cylindrical(&mut s.display.cylindrical, a)?;
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
                if let Some(b) = a.get("perspective").and_then(|v| v.as_bool()) {
                    d.perspective = b;
                }
                if let Some(b) = a.get("translucent_enclosure").and_then(|v| v.as_bool()) {
                    d.translucent_enclosure = b;
                }
                apply_section(&mut d, a)?;
                apply_cylindrical(&mut d.cylindrical, a)?;
                let w = opt_num(a, "width").unwrap_or(900.0).clamp(200.0, 2000.0) as usize;
                let h = opt_num(a, "height").unwrap_or(620.0).clamp(150.0, 2000.0) as usize;
                let view = opt_str(a, "view").unwrap_or("iso");
                let res = s.results.clone();
                let hl: HashSet<u32> = HashSet::new();
                let scene = render::build_scene(&mesh, &s.model, res.as_deref(), &d, &hl, 1.0);
                let mut cam = Camera::default();
                cam.perspective = d.perspective;
                if !cam.set_view(view) {
                    bail!("unknown view '{view}', use {:?}", render::VIEWS);
                }
                cam.fit(scene.lo, scene.hi, w as f64, h as f64);
                let (title, unit) = match (&res, d.show_results) {
                    (Some(r), true) => {
                        let f = if r.has(d.field) { d.field } else { r.available_fields()[0] };
                        let mut t = format!("{} [{}]", f.label(), f.unit());
                        if r.n_steps() > 1 {
                            t = format!("{t} — {}", r.step_label(d.mode));
                        }
                        (t, f.unit().to_string())
                    }
                    _ => ("Model (blue = supports, red = forces, orange = pressure, pink = temperature, cyan = convection, yellow = heat flux, purple = voltage/current, green = inlet/outlet; enclosure drawn as the far walls)".to_string(), String::new()),
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
                let raw = opt_str(a, "path").ok_or_else(|| anyhow!("missing 'path'"))?;
                let resolved = resolve_io_path(raw, ws_ref, true, &["vtk"])?;
                let path = resolved.to_string_lossy().to_string();
                let m = s.ensure_mesh()?;
                crate::vtk::export(&path, &m, s.results.as_deref())?;
                s.log(format!("Exported VTK {path}"));
                Ok(Output::text(format!("Wrote {path} (open with ParaView).")))
            }
            "save_project" => {
                let raw = opt_str(a, "path").map(String::from).or(s.project_path.clone()).ok_or_else(|| anyhow!("missing 'path'"))?;
                let resolved = resolve_io_path(&raw, ws_ref, true, &["simcraft", "json"])?;
                let path = resolved.to_string_lossy().to_string();
                s.save(&path)?;
                s.project_path = Some(path.clone());
                s.mark_saved();
                push_recent(&path);
                let _ = std::fs::remove_file(autosave_path());
                s.log(format!("Saved {path}"));
                Ok(Output::text(format!("Saved project to {path}.")))
            }
            "open_project" => {
                let raw = opt_str(a, "path").ok_or_else(|| anyhow!("missing 'path'"))?;
                let is_autosave = std::path::Path::new(raw) == autosave_path();
                let is_handoff = a.get("handoff").and_then(|v| v.as_bool()).unwrap_or(false)
                    || std::env::var_os("SIMCRAFT_HANDOFF_FILE").is_some_and(|h| std::path::Path::new(raw) == std::path::Path::new(&h));
                // Autosave/handoff paths live outside SIMCRAFT_WORKSPACE, while embedded assets still obey ws_ref.
                let proj_ws = if is_autosave || is_handoff { None } else { ws_ref };
                let resolved = resolve_io_path(raw, proj_ws, false, &["simcraft", "json"])?;
                let path = resolved.to_string_lossy().to_string();
                s.open_in_workspace(&path, ws_ref)?;
                s.mark_saved();
                if !is_autosave && !is_handoff {
                    push_recent(&path);
                    s.project_path = Some(path);
                } else {
                    // a recovered autosave or MCP handoff has no user project file yet
                    s.project_path = None;
                    s.saved_rev = None;
                    if is_handoff {
                        let _ = std::fs::remove_file(&resolved);
                    }
                }
                Ok(Output::json(s.model_json()))
            }
            "get_selection" => Ok(Output::json(json!({"selected_faces": s.selection}))),
            "list_parameter_targets" => {
                let targets: Vec<Value> = crate::study::list_targets(&s.model).into_iter().map(|(t, v)| json!({"target": t, "current": crate::results::float(v)})).collect();
                Ok(Output::json(json!({"parameters": targets, "responses": crate::study::suggested_responses(&s.model),
                    "note": "responses: mass, volume, max:<field>, min:<field>, frequency:N, load_multiplier:N, reaction:<load id>[:x|y|z], quantity:<name>"})))
            }
            "make_report" => {
                let mesh = s.ensure_mesh()?;
                let checks: Vec<crate::study::Constraint> = serde_json::from_value(a.get("checks").cloned().unwrap_or(json!([]))).map_err(|e| anyhow!("bad 'checks': {e}"))?;
                let title = opt_str(a, "title").map(String::from).unwrap_or_else(|| {
                    let name = s.project_path.as_deref().and_then(|p| std::path::Path::new(p).file_stem()).map(|x| x.to_string_lossy().to_string()).unwrap_or_else(|| "SimCraft".into());
                    format!("{name} — {} report", s.model.analysis.kind.label())
                });
                let html = crate::report::build_html(&crate::report::ReportInput {
                    title,
                    model: &s.model,
                    mesh: &mesh,
                    results: s.results.as_deref(),
                    display: &s.display,
                    study: s.study.as_ref(),
                    plan: s.plan.as_ref(),
                    checks,
                    project: s.project_path.clone(),
                    stale: s.results_stale,
                })?;
                let path = match opt_str(a, "path") {
                    Some(p) => resolve_io_path(p, ws_ref, true, &["html", "htm"])?.to_string_lossy().to_string(),
                    None => {
                        let dir = if let Some(ws_dir) = ws_ref {
                            std::fs::canonicalize(ws_dir).unwrap_or_else(|_| ws_dir.to_path_buf())
                        } else {
                            match &s.project_path {
                                Some(p) => std::path::Path::new(p).parent().map(|d| d.to_path_buf()).unwrap_or_else(std::env::temp_dir),
                                None => std::path::PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| ".".into())).join("SimCraft"),
                            }
                        };
                        std::fs::create_dir_all(&dir)?;
                        let stamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
                        dir.join(format!("report-{stamp}.html")).to_string_lossy().to_string()
                    }
                };
                std::fs::write(&path, &html)?;
                s.log(format!("Report written to {path}"));
                Ok(Output::text(format!("Report written to {path} ({} KB). Open it in a browser; print to PDF from there.", html.len() / 1024)))
            }
            "fatigue_life" => {
                let mesh = s.ensure_mesh()?;
                let res = s.results.clone().ok_or_else(|| anyhow!("no results yet; solve first"))?;
                if s.results_stale {
                    bail!("results are out of date; solve again before the fatigue evaluation");
                }
                let d = crate::fatigue::FatigueOptions::default();
                let o = crate::fatigue::FatigueOptions {
                    loading: opt_str(a, "loading").unwrap_or(&d.loading).to_string(),
                    ratio: opt_num(a, "ratio").unwrap_or(d.ratio),
                    correction: opt_str(a, "correction").unwrap_or(&d.correction).to_string(),
                    scale: opt_num(a, "scale").unwrap_or(d.scale),
                    design_life: opt_num(a, "design_life").unwrap_or(d.design_life),
                    exposure_s: opt_num(a, "exposure_time_s").unwrap_or(d.exposure_s),
                };
                let o = crate::fatigue::FatigueOptions { loading: if a.get("ratio").is_some() && a.get("loading").is_none() { "ratio".into() } else { o.loading.clone() }, ..o };
                let out = crate::fatigue::apply(&res, &mesh, &o)?;
                let qs: Vec<Value> = out.quantities.iter().filter(|q| q.name.starts_with("Fatigue")).map(|q| json!({"name": q.name, "value": crate::results::float(q.value), "unit": q.unit})).collect();
                let warns: Vec<String> = out.warnings.iter().filter(|w| w.starts_with("Fatigue")).cloned().collect();
                s.results = Some(Arc::new(out));
                s.display.field = Field::FatigueSafety;
                s.display.show_results = true;
                s.revision += 1;
                s.log("Fatigue evaluated".to_string());
                Ok(Output::json(json!({"quantities": qs, "warnings": warns, "fields": ["fatigue_life", "fatigue_damage", "fatigue_safety"]})))
            }
            "fit_surrogate" | "predict_surrogate" => {
                let st = s.study.clone().ok_or_else(|| anyhow!("no study yet: run run_parameter_study, optimize_design or surrogate_optimize first"))?;
                let key = (st.runs.len(), st.parameters.clone(), st.responses.clone());
                let fresh = s.surrogate.as_ref().is_some_and(|(k, _)| *k == key);
                if !fresh {
                    let set = crate::study::fit_surrogates(&st)?;
                    s.surrogate = Some((key, Arc::new(set)));
                }
                let set = s.surrogate.as_ref().unwrap().1.clone();
                if tool == "fit_surrogate" {
                    let rows: Vec<Value> = set.responses.iter().zip(&set.gps).map(|(r, g)| match g {
                        Some(g) => json!({
                            "response": r,
                            "designs": g.x.len(),
                            "loo_rmse": crate::results::float(g.loo_rmse),
                            "q2": crate::results::float(g.q2),
                            "quality": if g.q2 > 0.9 { "good" } else if g.q2 > 0.6 { "fair — add designs" } else { "poor — add designs or narrow the ranges" },
                            "relevance": set.parameters.iter().zip(g.relevance()).map(|(p, w)| json!({"parameter": p.target, "weight": crate::results::float(w)})).collect::<Vec<_>>(),
                        }),
                        None => json!({"response": r, "note": "fewer than 3 solved designs"}),
                    }).collect();
                    return Ok(Output::json(json!({"surrogates": rows, "from_designs": set.n_runs, "note": "Gaussian-process response surfaces of the current study. Q² is the leave-one-out prediction quality (1 = perfect); relevance shows which parameters drive each response."})));
                }
                // prediction at the given values (missing parameters: middle of their range)
                let vals = a.get("values").ok_or_else(|| anyhow!("missing 'values' {{parameter target: value}}"))?;
                let mut x = Vec::new();
                let mut outside = Vec::new();
                for p in &set.parameters {
                    let v = match vals {
                        Value::Object(m) => m.get(&p.target).and_then(|v| v.as_f64()),
                        Value::Array(arr) => arr.get(x.len()).and_then(|v| v.as_f64()),
                        _ => None,
                    }
                    .unwrap_or((p.min + p.max) / 2.0);
                    if v < p.min - 1e-9 || v > p.max + 1e-9 {
                        outside.push(p.target.clone());
                    }
                    x.push(v);
                }
                let pred = set.predict(&x);
                let rows: Vec<Value> = set.responses.iter().zip(pred).zip(&set.gps).map(|((r, p), g)| match p {
                    Some((m, sd)) => json!({"response": r, "estimate": crate::results::float(m), "plus_minus_2sigma": crate::results::float(2.0 * sd), "loo_rmse": crate::results::float(g.as_ref().unwrap().loo_rmse)}),
                    None => json!({"response": r, "estimate": null}),
                }).collect();
                let mut out = json!({
                    "label": "surrogate estimate, not verified — use apply_design with these values and solve to confirm",
                    "values": set.parameters.iter().zip(&x).map(|(p, v)| (p.target.clone(), json!(crate::results::float(*v)))).collect::<serde_json::Map<String, Value>>(),
                    "predictions": rows,
                });
                if !outside.is_empty() {
                    out["warning"] = json!(format!("{} outside the studied range: extrapolation is unreliable", outside.join(", ")));
                }
                Ok(Output::json(out))
            }
            "get_study" => match &s.study {
                Some(st) => Ok(Output::json(study_json(st))),
                None => Ok(Output::text("No study yet. Use run_parameter_study, optimize_design or topology_optimize.")),
            },
            "apply_design" => {
                let st = s.study.clone().ok_or_else(|| anyhow!("no study yet"))?;
                if st.kind == "topology" {
                    bail!("for topology results use the exported STL (import_stl {:?})", st.stl_path);
                }
                if st.kind == "convergence" {
                    let h = a
                        .get("values")
                        .and_then(|v| v.get("element_size_mm").or_else(|| v.get("element_size")))
                        .and_then(|v| v.as_f64())
                        .or(st.recommended_size_mm)
                        .ok_or_else(|| anyhow!("convergence study has no recommended element size"))?;
                    if !h.is_finite() || h <= 0.0 {
                        bail!("element_size_mm must be positive");
                    }
                    s.model.element_size = Some(h);
                    s.touch_geometry();
                    s.log(format!("Applied recommended mesh size {h} mm"));
                    return Ok(Output::text(format!("Set element_size_mm = {h}. Solve to compute on the refined mesh.")));
                }
                if let Some(Value::Object(vals)) = a.get("values") {
                    let mut v = Vec::new();
                    for p in &st.parameters {
                        v.push(match vals.get(&p.target).and_then(|x| x.as_f64()) {
                            Some(x) => x,
                            None => crate::study::get_value(&s.model, &p.target)?,
                        });
                    }
                    let m = crate::study::apply(&s.model, &st.parameters, &v)?;
                    s.model = m;
                    s.touch_geometry();
                    let desc: Vec<String> = st.parameters.iter().zip(&v).map(|(p, x)| format!("{} = {}", p.target, crate::results::float(*x))).collect();
                    s.log(format!("Applied design: {}", desc.join(", ")));
                    return Ok(Output::text(format!("Applied {}. Solve to verify the surrogate estimate.", desc.join(", "))));
                }
                let idx = match opt_num(a, "run") {
                    Some(r) => (r as usize).checked_sub(1).ok_or_else(|| anyhow!("run numbers start at 1"))?,
                    None => st.best.ok_or_else(|| anyhow!("this study has no best design; give 'run'"))?,
                };
                let run = st.runs.get(idx).ok_or_else(|| anyhow!("run must be 1..={}", st.runs.len()))?;
                let m = crate::study::apply(&s.model, &st.parameters, &run.values)?;
                s.model = m;
                s.touch_geometry();
                let desc: Vec<String> = st.parameters.iter().zip(&run.values).map(|(p, v)| format!("{} = {}", p.target, crate::results::float(*v))).collect();
                s.log(format!("Applied design #{}: {}", idx + 1, desc.join(", ")));
                Ok(Output::text(format!("Applied design #{} to the model: {}. Solve to see its full results.", idx + 1, desc.join(", "))))
            }
            "get_plan" => match &s.plan {
                Some(p) => Ok(Output { text: format!("{}\n\n---\n{}", crate::assistant::to_markdown(p), serde_json::to_string(p)?), image_png: None }),
                None => Ok(Output::text("No plan yet. Use plan_simulation with a goal, or the Planner in the app.")),
            },
            "set_plan_step" => {
                let i = num(a, "index")? as usize;
                let done = a.get("done").and_then(|v| v.as_bool()).unwrap_or(true);
                let p = s.plan.as_mut().ok_or_else(|| anyhow!("no plan yet"))?;
                let n = p.steps.len();
                let st = p.steps.get_mut(i.wrapping_sub(1)).ok_or_else(|| anyhow!("step index must be 1..={n}"))?;
                st.done = done;
                s.revision += 1;
                Ok(Output::text(format!("Step {i} marked {}.", if done { "done" } else { "not done" })))
            }
            _ => bail!("unknown tool '{tool}'"),
        }
    }

    /// Long-running optimization jobs: run on a copy of the model without holding the lock.
    fn study_job(&self, tool: &str, a: &Value) -> Result<Output> {
        self.run_job(tool, || self.study_inner(tool, a))
    }

    fn study_inner(&self, tool: &str, a: &Value) -> Result<Output> {
        *self.progress.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = Progress { running: true, fraction: 0.0, message: "starting study".into() };
        self.changed();
        let cb = |f: f64, msg: &str| {
            let mut p = self.progress.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            p.fraction = f;
            p.message = msg.to_string();
            drop(p);
            self.changed();
        };
        let prog = fem::Progress::new(&self.cancel, &cb);
        let (model, mesh, project) = if tool == "topology_optimize" {
            let (m, mesh, _) = self.prepare_mesh(true, &cb)?;
            let proj = self.session.lock().unwrap_or_else(std::sync::PoisonError::into_inner).project_path.clone();
            (m, Some(mesh), proj)
        } else {
            let s = self.session.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            (s.model.clone(), None, s.project_path.clone())
        };
        let parse_params = || -> Result<Vec<crate::study::Parameter>> {
            serde_json::from_value(a.get("parameters").cloned().ok_or_else(|| anyhow!("missing 'parameters' [{{target, min, max}}]"))?).map_err(|e| anyhow!("bad 'parameters': {e}"))
        };
        let t0 = std::time::Instant::now();
        let study = match tool {
            "run_parameter_study" => {
                let responses: Vec<String> = serde_json::from_value(a.get("responses").cloned().unwrap_or(json!([]))).map_err(|e| anyhow!("bad 'responses': {e}"))?;
                let samples = opt_num(a, "samples").map(|v| v as usize);
                crate::study::parameter_study(&model, parse_params()?, responses, samples, prog)?
            }
            "surrogate_optimize" => {
                let objective: crate::study::Objective = serde_json::from_value(a.get("objective").cloned().ok_or_else(|| anyhow!("missing 'objective' {{response, goal}}"))?).map_err(|e| anyhow!("bad 'objective': {e}"))?;
                let constraints: Vec<crate::study::Constraint> = serde_json::from_value(a.get("constraints").cloned().unwrap_or(json!([]))).map_err(|e| anyhow!("bad 'constraints': {e}"))?;
                let budget = opt_num(a, "max_solves").unwrap_or(12.0) as usize;
                let params = parse_params()?;
                // reuse the previous study's verified designs when it covered the same question
                let mut want = vec![objective.response.clone()];
                want.extend(constraints.iter().map(|c| c.response.clone()));
                let prior: Vec<crate::study::Run> = {
                    let s = self.session.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                    match &s.study {
                        Some(st) if st.parameters.iter().map(|p| (&p.target, p.min, p.max)).eq(params.iter().map(|p| (&p.target, p.min, p.max))) && a.get("reuse").and_then(|v| v.as_bool()).unwrap_or(true) => {
                            let idx: Option<Vec<usize>> = want.iter().map(|w| st.responses.iter().position(|r| r == w)).collect();
                            match idx {
                                Some(idx) => st.runs.iter().map(|r| crate::study::Run { values: r.values.clone(), responses: idx.iter().map(|&i| r.responses.get(i).copied().flatten()).collect(), feasible: false, error: r.error.clone(), phase: "prior".into() }).collect(),
                                None => vec![],
                            }
                        }
                        _ => vec![],
                    }
                };
                crate::study::surrogate_optimize(&model, params, objective, constraints, budget, prior, prog)?
            }
            "optimize_design" => {
                let objective: crate::study::Objective = serde_json::from_value(a.get("objective").cloned().ok_or_else(|| anyhow!("missing 'objective' {{response, goal}}"))?).map_err(|e| anyhow!("bad 'objective': {e}"))?;
                let constraints: Vec<crate::study::Constraint> = serde_json::from_value(a.get("constraints").cloned().unwrap_or(json!([]))).map_err(|e| anyhow!("bad 'constraints': {e}"))?;
                let budget = opt_num(a, "max_evaluations").unwrap_or(30.0) as usize;
                crate::study::optimize(&model, parse_params()?, objective, constraints, budget, prog)?
            }
            "mesh_convergence" => {
                let responses: Vec<String> = serde_json::from_value(a.get("responses").cloned().unwrap_or(json!([]))).map_err(|e| anyhow!("bad 'responses': {e}"))?;
                let base_h = opt_num(a, "base_element_size_mm");
                let target = opt_num(a, "target_error_pct").unwrap_or(1.0);
                crate::study::mesh_convergence(&model, responses, base_h, target, prog)?
            }
            _ => {
                let mesh = mesh.unwrap();
                let ws = self.workspace_dir();
                let ws_ref = ws.as_deref();
                let path = match opt_str(a, "export_path") {
                    Some(p) => resolve_io_path(p, ws_ref, true, &["stl"])?.to_string_lossy().to_string(),
                    None => {
                        let dir = if let Some(ws_dir) = ws_ref {
                            std::fs::canonicalize(ws_dir).unwrap_or_else(|_| ws_dir.to_path_buf())
                        } else {
                            match &project {
                                Some(p) => std::path::Path::new(p).parent().map(|d| d.to_path_buf()).unwrap_or_else(std::env::temp_dir),
                                None => {
                                    let d = std::path::PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| ".".into())).join("SimCraft");
                                    std::fs::create_dir_all(&d)?;
                                    d
                                }
                            }
                        };
                        let stamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
                        dir.join(format!("optimized-{stamp}.stl")).to_string_lossy().to_string()
                    }
                };
                let settings = crate::topo::TopoSettings {
                    volume_fraction: opt_num(a, "volume_fraction").unwrap_or(0.4),
                    iterations: opt_num(a, "iterations").unwrap_or(40.0) as usize,
                    filter_radius_mm: opt_num(a, "filter_radius_mm"),
                    penal: 3.0,
                };
                let tr = crate::topo::run(&model, &mesh, &settings, prog)?;
                let ntri = crate::topo::write_stl(&mesh, &tr.density, 0.5, &path)?;
                let mut s = self.session.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                s.results = Some(Arc::new(tr.results));
                s.results_stale = false;
                s.display.field = Field::Density;
                s.display.show_results = true;
                s.display.deformation_scale = Some(0.0);
                s.revision += 1;
                s.log(format!("Topology optimization: kept {} of {} elements, shape saved to {path}", tr.kept_elements, mesh.elems.len()));
                drop(s);
                crate::study::Study {
                    kind: "topology".into(),
                    message: format!(
                        "Kept {} of {} elements ({:.0}%). Compliance {:.4e} → {:.4e} J. Shape exported ({ntri} triangles) to {path}",
                        tr.kept_elements,
                        mesh.elems.len(),
                        tr.kept_elements as f64 / mesh.elems.len() as f64 * 100.0,
                        tr.history.first().map(|h| h.1).unwrap_or(0.0),
                        tr.history.last().map(|h| h.1).unwrap_or(0.0)
                    ),
                    stl_path: Some(path),
                    history: tr.history,
                    seconds: t0.elapsed().as_secs_f64(),
                    ..Default::default()
                }
            }
        };
        let out = study_json(&study);
        let mut s = self.session.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        s.log(study.message.clone());
        s.study = Some(study);
        s.revision += 1;
        Ok(Output::json(out))
    }

    fn plan_simulation(&self, a: &Value) -> Result<Output> {
        let goal = opt_str(a, "goal").ok_or_else(|| anyhow!("missing 'goal' (what you want to simulate)"))?.to_string();
        let model = match opt_str(a, "model") {
            Some(m) => m.to_string(),
            None => {
                let models = crate::assistant::list_models()?;
                crate::assistant::default_model(&models).ok_or_else(|| anyhow!("no local models installed; run `ollama pull qwen2.5:7b`"))?
            }
        };
        *self.progress.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = Progress { running: true, fraction: 0.0, message: format!("Planning with {model}…") };
        self.changed();
        let r = crate::assistant::plan(&goal, &model);
        self.progress.lock().unwrap_or_else(std::sync::PoisonError::into_inner).running = false;
        let p = r?;
        let md = crate::assistant::to_markdown(&p);
        {
            let mut s = self.session.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            s.plan = Some(p);
            s.revision += 1;
            s.log(format!("Planner: drafted a to-do list with {model}"));
        }
        Ok(Output::text(md))
    }

    fn screenshot_gui(&self) -> Result<Output> {
        use eframe::egui;
        // the bridge comes up slightly before the window finishes opening: wait for it
        let mut ctx = None;
        for _ in 0..50 {
            ctx = self.gui_ctx.lock().unwrap_or_else(std::sync::PoisonError::into_inner).clone();
            if ctx.is_some() || self.notify.lock().unwrap_or_else(std::sync::PoisonError::into_inner).is_none() && !crate::bridge::gui_running() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(200));
        }
        let ctx = ctx.ok_or_else(|| anyhow!("the SimCraft window is not open (use launch_gui first); render_image works headless"))?;
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
        self.run_job("solve", || self.solve_inner())
    }

    /// Run a long job: one at a time, cancellable, and a panic inside a solver becomes an error
    /// instead of leaving the app stuck "busy".
    fn run_job(&self, what: &str, f: impl FnOnce() -> Result<Output>) -> Result<Output> {
        if self.busy.swap(true, Ordering::SeqCst) {
            bail!("a solve or study is already running (stop it with the Stop button / cancel)");
        }
        self.cancel.store(false, Ordering::SeqCst);
        let _guard = JobGuard(self);
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
            Ok(Err(e)) if e.is::<fem::Cancelled>() => {
                self.session.lock().unwrap_or_else(std::sync::PoisonError::into_inner).log(format!("{what} cancelled"));
                bail!("{what} cancelled")
            }
            Ok(r) => r,
            Err(payload) => {
                let msg = payload.downcast_ref::<String>().cloned().or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_else(|| "unknown error".into());
                bail!("internal error during {what}: {msg}. The model is unchanged; please report this (with the project file) if it repeats")
            }
        }
    }

    /// Build or fetch the session mesh outside the session lock so meshing stays responsive and cancellable.
    fn prepare_mesh(&self, plain: bool, cb: &(dyn Fn(f64, &str) + Sync)) -> Result<(Model, Arc<Mesh>, u64)> {
        let (model, h, limit, mode, rev) = {
            let s = self.session.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            if s.mesh_valid() {
                let m = s.mesh.clone().unwrap();
                if !plain || m.fitted.is_empty() {
                    return Ok((s.model.clone(), m, s.geom_rev));
                }
            }
            let limit = if plain { mesh::MAX_CELLS } else { mesh::max_cells_for(&s.model) };
            let mode = (s.model.mesh_with_enclosure(), !plain && s.model.mesh_fitted());
            (s.model.clone(), s.element_size(), limit, mode, s.geom_rev)
        };
        fem::check_cancel(&self.cancel)?;
        cb(0.0, "meshing");
        let built = if plain {
            mesh::build_mesh_ex_cancellable(&model.bodies, h, &model.materials(), mode.0, limit, false, &self.cancel)?
        } else {
            mesh::build_for_cancellable(&model, h, limit, &self.cancel)?
        };
        let mut s = self.session.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let m = s.install_mesh(built, rev, mode).ok_or_else(|| anyhow!("geometry changed during meshing; solve again"))?;
        Ok((s.model.clone(), m, rev))
    }

    fn solve_inner(&self) -> Result<Output> {
        {
            let mut s = self.session.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            let lbl = s.model.analysis.kind.label();
            s.log(format!("Solving {lbl}…"));
        }
        *self.progress.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = Progress { running: true, fraction: 0.0, message: "starting".into() };
        self.changed();
        let cb = |f: f64, msg: &str| {
            let mut p = self.progress.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            p.fraction = f;
            p.message = msg.to_string();
            drop(p);
            self.changed();
        };
        let (model, mesh, rev) = self.prepare_mesh(false, &cb)?;
        let prog = fem::Progress::new(&self.cancel, &cb);
        let res = fem::solve(&model, &mesh, prog)?;
        let mut s = self.session.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if s.geom_rev != rev {
            bail!("geometry changed during the solve; solve again");
        }
        let summary = res.summary(&mesh);
        use AnalysisType::*;
        s.display.field = match res.analysis {
            SteadyThermal | TransientThermal | ThermalElectric => Field::Temperature,
            Modal | LinearBuckling | TransientStructural => Field::TotalDeformation,
            NonlinearStructural => Field::VonMises,
            Electrostatic | FullWaveEm => Field::ElectricField,
            MicrowaveHeating => Field::Temperature,
            CurrentFlow => Field::CurrentDensity,
            Magnetostatic => Field::MagneticFluxDensity,
            Fluid => Field::Velocity,
            _ => if !res.has(s.display.field) || !s.display.field.is_deformation() && s.display.field != Field::VonMises { Field::VonMises } else { s.display.field },
        };
        if res.analysis.uses_enclosure() && res.analysis != AnalysisType::MicrowaveHeating && mesh.with_enclosure && s.display.section.is_none() {
            s.display.section = Some(render::Section { axis: 1, position: 0.5 });
            s.view_request = Some("front".into());
        }
        if !res.has(s.display.field) {
            s.display.field = res.available_fields()[0];
        }
        s.display.show_results = true;
        s.display.mode = if res.frames.is_empty() { 0 } else { res.frames.len() - 1 };
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
        t("cancel", "Stop the running solve or study (it stops at its next progress step; results of a cancelled solve are discarded).", json!({}), &[]),
        t("get_status", "Solver progress, last error and recent log lines.", json!({}), &[]),
        t("new_model", "Discard everything and start an empty model.", json!({}), &[]),
        t("add_box", "Add a rectangular block (mm). Faces: <name>.xmin/xmax/ymin/ymax/zmin/zmax.", json!({"name": name, "min": v3s("Min corner [x,y,z] mm (default origin)"), "size": v3s("Size [dx,dy,dz] mm"), "operation": op, "material": mat}), &["size"]),
        t("add_cylinder", "Add a cylinder (mm) or, with operation=subtract, drill a hole. Faces: <name>.side/bottom/top.", json!({"name": name, "base": v3s("Centre of the bottom cap [x,y,z] mm"), "axis": {"type": "string", "enum": ["x", "y", "z"]}, "radius": {"type": "number"}, "height": {"type": "number", "description": "Length along the axis from base"}, "operation": op, "material": mat}), &["radius", "height"]),
        t("add_sphere", "Add a sphere (mm). Face: <name>.surface.", json!({"name": name, "center": v3s("Centre mm"), "radius": {"type": "number"}, "operation": op, "material": mat}), &["radius"]),
        t("import_kicad", "Import a KiCad PCB (.kicad_pcb, KiCad 5–10): board outline and stack-up become a 'Board' body whose copper (tracks, pads, filled zones, vias) is homogenised per cell into orthotropic thermal conductivity; footprints larger than min_component_area_mm2 become component boxes named by reference (U1, Q2…). 'power' {reference: watts} adds heat sources. Then add convection/temperature boundaries (thermal) or an air enclosure + inlet (conjugate_heat_transfer). Board lies in XY, z from 0 to the thickness; top parts above, bottom parts below. Save the board in KiCad with zones filled.", json!({"path": {"type": "string"}, "name": {"type": "string", "description": "board body name (default Board)"}, "power": {"type": "object", "additionalProperties": {"type": "number"}, "description": "e.g. {\"U1\": 1.5, \"Q1\": 0.8} watts"}, "heights": {"type": "object", "additionalProperties": {"type": "number"}, "description": "component heights in mm by reference"}, "components": {"type": "boolean", "description": "create component boxes (default true)"}, "component_height_mm": {"type": "number", "description": "default component height (1 mm)"}, "min_component_area_mm2": {"type": "number", "description": "skip footprints smaller than this unless listed in power/heights (default 4)"}, "offset": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}, "material": {"type": "string", "description": "laminate (default FR-4)"}}), &["path"]),
        t("pcb_nets", "List the nets of a KiCad board with their pads (REFERENCE.PAD), to choose 'from' / 'to' for pcb_dc. Uses the imported board unless 'path' is given.", json!({"path": {"type": "string"}}), &[]),
        t("pcb_dc", "DC current in one net of a KiCad board: resistance and voltage drop between two pads, Joule power, peak current density and where it occurs, current and IPC-2221 temperature rise of the tracks that carry it, and via currents. Pads are REFERENCE.PAD (e.g. U1.3 → J2.1). Uses the imported board's file unless 'path' is given.", json!({"path": {"type": "string", "description": ".kicad_pcb (default: the imported board)"}, "net": {"type": "string", "description": "net name (default: the net of 'from')"}, "from": {"type": "string", "description": "source pad, e.g. U1.3"}, "to": {"type": "string", "description": "sink pad, e.g. J2.1"}, "current": {"type": "number", "description": "amps (default 1)"}, "temperature_c": {"type": "number", "description": "copper temperature for resistivity (default 20 °C)"}}), &["from", "to"]),
        t("import_stl", "Import a closed STL solid (e.g. exported from SolidCraft / Fusion 360). Smooth regions become faces <name>.face0, face1 …", json!({"path": {"type": "string", "description": "Absolute path to .stl"}, "scale": {"type": "number", "description": "Multiply coordinates (use 1000 if the STL is in metres)"}, "offset": v3s("Translation mm"), "name": name, "operation": op, "material": mat}), &["path"]),
        t("edit_body", "Change an existing body: any shape field (box: min, size; cylinder: base, axis, radius, height; sphere: center, radius; STL: scale, offset; board: offset), material, operation (add/subtract) or new_name (loads on it follow the rename). Undoable.", json!({"name": {"type": "string"}, "new_name": {"type": "string"}, "material": {"type": "string"}, "operation": {"type": "string", "enum": ["add", "subtract"]}, "min": v3s("box corner mm"), "size": v3s("box size mm"), "base": v3s("cylinder base centre mm"), "axis": {"type": "string", "enum": ["x", "y", "z"]}, "radius": {"type": "number"}, "height": {"type": "number"}, "center": v3s("sphere centre mm"), "scale": {"type": "number"}, "offset": v3s("STL / board offset mm")}), &["name"]),
        t("edit_load", "Change an existing support or load by id (see get_model): any of its fields, e.g. {id: 3, vector: [0, 0, -500]}, {id: 2, value_mpa: 4}, {id: 5, faces: [\"Plate.xmax\"]}. Undoable.", json!({"id": {"type": "integer"}}), &["id"]),
        t("transform_body", "Move, copy, pattern or mirror a body. move/copy/pattern use 'translate' [dx, dy, dz] mm (pattern: 'count' instances in total, each step 'translate' further); mirror uses 'axis' and the plane coordinate 'at' (copy: true by default). Copies get new names and no loads.", json!({"name": {"type": "string"}, "mode": {"type": "string", "enum": ["move", "copy", "pattern", "mirror"]}, "translate": v3s("offset mm"), "count": {"type": "integer"}, "axis": {"type": "string", "enum": ["x", "y", "z"]}, "at": {"type": "number", "description": "mirror plane coordinate mm"}, "copy": {"type": "boolean"}}), &["name", "mode"]),
        t("get_help", "Read the SimCraft user guide: no arguments = table of contents; 'topic' = one section with its subsections (a number like '7' or a name like 'meshing', 'fatigue', 'KiCad'); 'query' = search the whole guide (ranked sections with snippets). Use it to learn how an analysis, load or result works.", json!({"topic": {"type": "string"}, "query": {"type": "string"}}), &[]),
        t("undo", "Undo the last change to the model (geometry, materials, loads, analysis settings).", json!({}), &[]),
        t("redo", "Redo the last undone change.", json!({}), &[]),
        t("remove_body", "Delete a body by name.", json!({"name": {"type": "string"}}), &["name"]),
        t("list_materials", "List the engineering materials library with properties.", json!({}), &[]),
        t("define_material", "Create or overwrite a custom material. Unspecified properties come from 'based_on' (default Structural Steel).", json!({"name": {"type": "string"}, "based_on": {"type": "string"}, "fatigue_strength_coefficient_mpa": {"type": "number", "description": "Basquin σf' (0 = estimate)"}, "fatigue_exponent": {"type": "number", "description": "Basquin b, negative (e.g. -0.09)"}, "endurance_limit_mpa": {"type": "number"}, "relative_permittivity": {"type": "number", "description": "εr"}, "electrical_conductivity": {"type": "number", "description": "S/m"}, "relative_permeability": {"type": "number", "description": "μr"}, "loss_tangent": {"type": "number", "description": "dielectric tan δ at the RF frequency (full-wave EM)"}, "youngs_modulus_gpa": {"type": "number"}, "poisson_ratio": {"type": "number"}, "density": {"type": "number", "description": "kg/m³"}, "yield_strength_mpa": {"type": "number"}, "ultimate_strength_mpa": {"type": "number"}, "thermal_conductivity": {"type": "number", "description": "W/(m·K)"}, "thermal_expansion": {"type": "number", "description": "1/K"}, "specific_heat": {"type": "number"}, "conductivity_table": {"type": "array", "items": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2}, "description": "k vs T: [[°C, W/(m·K)], …] (linear between points)"}, "specific_heat_table": {"type": "array", "items": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2}, "description": "c vs T: [[°C, J/(kg·K)], …]"}}), &["name"]),
        t("set_material", "Assign a material to bodies (all bodies if 'bodies' omitted).", json!({"material": {"type": "string"}, "bodies": {"type": "array", "items": {"type": "string"}}}), &["material"]),
        t("set_analysis", "Choose the analysis system and its settings. Structural: static_structural, modal, linear_buckling, transient_structural, nonlinear_structural (plasticity), harmonic_response (forces/pressures are amplitudes), response_spectrum and random_vibration (base excitation; supports only, give 'table' and 'direction'). Follow with fatigue_life for S-N fatigue. Thermal: steady_thermal, transient_thermal, thermal_stress. Electromagnetic: electrostatic, current_flow, thermal_electric, magnetostatic (needs enclosure). Full-wave EM: full_wave_em (FDTD at one frequency: E field, absorbed power, SAR; needs an enclosure = the cavity or air box, and an RF port via add_port) and microwave_heating (full_wave_em + transient heating of the solids over end_time). Fluids: fluid (needs enclosure) and conjugate_heat_transfer (flow + temperature in fluid and solids: electronics cooling, heat sinks; add_heat_generation in the parts, add_inlet with temperature, add_gravity for natural convection/buoyancy).", json!({"type": {"type": "string", "enum": AnalysisType::all().iter().map(|a| a.key()).collect::<Vec<_>>()}, "modes": {"type": "integer", "description": "Number of modes for modal / buckling (default 6)"}, "reference_temperature": {"type": "number", "description": "Stress-free temperature °C (default 22)"}, "end_time": {"type": "number", "description": "Transient end time [s] (default 1)"}, "time_steps": {"type": "integer", "description": "Transient time steps (default 50)"}, "initial_temperature": {"type": "number", "description": "Transient thermal start temperature °C"}, "damping_ratio": {"type": "number", "description": "Transient structural damping ratio (default 0.02)"}, "load_steps": {"type": "integer", "description": "Nonlinear load steps (default 10)"}, "max_iterations": {"type": "integer", "description": "CFD pseudo-time iterations (default 3000)"}, "boundary_fit": {"type": "boolean", "description": "Structural, thermal, modal and vibration analyses: move surface nodes onto the true curved/off-grid surfaces for accurate peak stresses, heat flows and frequencies (default true; buckling, plasticity, EM, CFD and PCB models use plain voxels)"}, "turbulence": {"type": "string", "enum": ["laminar", "lvel"], "description": "CFD turbulence model: laminar (default) or lvel (algebraic wall-distance model, good for electronics enclosures)"}, "cfd_scheme": {"type": "string", "enum": ["auto", "simple", "projection"], "description": "CFD solution scheme: auto (default: implicit SIMPLE, switching to the time-averaging projection method for flows that do not settle), simple (SIMPLE only), projection (explicit pseudo-time projection)"}, "freq_min": {"type": "number", "description": "Harmonic sweep start [Hz] (default 0.5·f1)"}, "freq_max": {"type": "number", "description": "Harmonic sweep end [Hz] (default 1.5·highest mode used)"}, "freq_steps": {"type": "integer", "description": "Harmonic frequency points (default 300; resonances are refined automatically)"}, "table": {"type": "array", "items": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2}, "description": "[[Hz, value], ...] — response spectrum: acceleration in g; random vibration: acceleration PSD in g²/Hz (log-log interpolated)"}, "direction": {"type": "string", "enum": ["x", "y", "z"], "description": "Base-excitation direction for response spectrum / random vibration"}, "combination": {"type": "string", "enum": ["srss", "cqc"], "description": "Response spectrum modal combination (default srss)"}, "frequency_ghz": {"type": "number", "description": "Full-wave EM / microwave heating excitation frequency in GHz (default 2.45)"}, "em_boundary": {"type": "string", "enum": ["pec", "open"], "description": "Full-wave EM outer boundary: pec = the enclosure is a closed metal cavity (oven), open = absorbing (radiation, EMI)"}, "max_periods": {"type": "integer", "description": "Full-wave EM: max RF periods to reach steady state (default 400)"}, "compute": {"type": "string", "enum": ["auto", "cpu", "gpu"], "description": "Where linear solves run: auto = CPU direct for small models, GPU for large iterative solves; cpu; gpu = always GPU (falls back to CPU if unavailable)"}}), &[]),
        t("list_faces", "Mesh the geometry if needed and list selectable faces with area, centroid and outward normal. Call this before adding supports or loads.", json!({}), &[]),
        t("add_fixed_support", "Fix faces in all directions.", json!({"faces": faces}), &["faces"]),
        t("add_displacement", "Prescribe displacement (mm) on faces; omit a component to leave it free (e.g. z=0 for a roller).", json!({"faces": faces, "x": {"type": "number"}, "y": {"type": "number"}, "z": {"type": "number"}}), &["faces"]),
        t("add_force", "Total force vector in N, spread evenly over the faces.", json!({"faces": faces, "vector": v3s("[Fx,Fy,Fz] N")}), &["faces", "vector"]),
        t("add_pressure", "Normal pressure in MPa (positive pushes into the surface).", json!({"faces": faces, "value_mpa": {"type": "number"}}), &["faces", "value_mpa"]),
        t("add_gravity", "Body acceleration in m/s² (default standard earth gravity -Z).", json!({"acceleration": v3s("[ax,ay,az] m/s²")}), &[]),
        t("add_temperature", "Fixed temperature (°C) on faces. Thermal / thermal-stress analyses.", json!({"faces": faces, "value": {"type": "number"}}), &["faces", "value"]),
        t("add_heat_flux", "Heat flux into the faces, W/m².", json!({"faces": faces, "value": {"type": "number"}}), &["faces", "value"]),
        t("add_radiation", "Thermal radiation from faces (thermal, thermal stress, transient, microwave heating, conjugate heat transfer): emissivity ε (polished Al 0.05, oxidised metal 0.3–0.6, anodised / painted / FR-4 0.8–0.95) to surroundings at 'ambient' °C. enclosure: true makes all enclosure faces exchange radiation with each other (view factors by ray casting; parts block the view), the rest of their view seeing the ambient — for heat sinks, fins facing each other, sealed boxes. Conjugate heat transfer supports radiation to ambient on outer faces only.", json!({"faces": faces, "emissivity": {"type": "number", "description": "0–1 (default 0.9)"}, "ambient": {"type": "number", "description": "surroundings °C (default 22)"}, "enclosure": {"type": "boolean", "description": "surface-to-surface between all enclosure faces (default false: to ambient only)"}}), &["faces"]),
        t("add_convection", "Convection film coefficient W/(m²·K) with ambient temperature °C.", json!({"faces": faces, "coefficient": {"type": "number"}, "ambient": {"type": "number"}}), &["faces", "coefficient"]),
        t("add_heat_generation", "Internal heat generation in one body (a chip, a resistor, the food, a coil) or in all solids when 'body' is omitted. Give either 'power' in W (spread over the body; exact on any mesh — easiest for electronics) or 'value' in W/m³ (power = value × body volume).", json!({"power": {"type": "number", "description": "total heat [W]"}, "value": {"type": "number", "description": "power density [W/m³]"}, "body": {"type": "string", "description": "solid body name; omit for the whole model"}}), &[]),
        t("add_compression_only_support", "Frictionless contact with a rigid surface on these faces: blocks motion into the surface, allows lift-off (static structural).", json!({"faces": faces}), &["faces"]),
        t("add_symmetry", "Symmetry plane: zero normal displacement (structural) or free-slip wall (fluid).", json!({"faces": faces}), &["faces"]),
        t("add_voltage", "Fixed electric potential [V] on faces (electrostatic, current flow, thermal-electric).", json!({"faces": faces, "value": {"type": "number"}}), &["faces", "value"]),
        t("add_current", "Total DC current [A] entering the body through faces (current flow / thermal-electric). Needs a Voltage (ground) elsewhere.", json!({"faces": faces, "value": {"type": "number"}}), &["faces", "value"]),
        t("add_coil", "Coil current density [A/mm²] in a solid body (magnetostatic). mode=circumferential for a winding around the axis, axial for a straight conductor.", json!({"body": {"type": "string"}, "current_density": {"type": "number"}, "axis": {"type": "string", "enum": ["x", "y", "z"]}, "mode": {"type": "string", "enum": ["circumferential", "axial"]}, "center": v3s("Point on the coil axis mm (default body centre)")}), &["body", "current_density"]),
        t("add_magnet", "Permanent magnet: remanent flux density vector [T] in a body (e.g. [0,0,1.3] for N42 magnetised along +Z). Use material 'NdFeB Magnet (N42)'.", json!({"body": {"type": "string"}, "remanence": v3s("[Brx, Bry, Brz] T")}), &["body", "remanence"]),
        t("add_inlet", "Fluid velocity inlet [m/s] flowing into the domain through faces (usually Enclosure.xmin). For conjugate heat transfer, 'temperature' is the inflow temperature (default: reference temperature, 22 °C).", json!({"faces": faces, "velocity": {"type": "number"}, "temperature": {"type": "number", "description": "°C"}}), &["faces", "velocity"]),
        t("add_moving_wall", "No-slip wall sliding tangentially with the given velocity [m/s] (lid-driven cavity, conveyor belt). Fluid and conjugate heat transfer analyses.", json!({"faces": faces, "velocity": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3, "description": "m/s"}}), &["faces", "velocity"]),
        t("add_port", "RF / microwave feed for full_wave_em and microwave_heating: a current sheet on the faces (e.g. a waveguide opening on the cavity wall, usually an Enclosure face) delivering 'power' watts with the electric field along 'polarization'. profile te10 (default) = half-sine across the face like a rectangular waveguide's TE10 mode; uniform = constant.", json!({"faces": faces, "power": {"type": "number", "description": "delivered power [W], e.g. 800 for a domestic oven"}, "polarization": {"type": "string", "enum": ["x", "y", "z"], "description": "E-field direction; must lie in the face (for a WR-340 feed: along the short side)"}, "profile": {"type": "string", "enum": ["te10", "uniform"]}}), &["faces", "power", "polarization"]),
        t("add_outlet", "Fluid pressure outlet [Pa] (usually Enclosure.xmax, 0 Pa).", json!({"faces": faces, "pressure": {"type": "number"}}), &["faces"]),
        t("add_enclosure", "Add the surrounding air/fluid box used by electrostatic, magnetostatic and CFD analyses. By default it pads the parts by 50% of their size on every side; give padding (fraction), padding_mm [xmin,xmax,ymin,ymax,zmin,zmax] or explicit min/size. Its faces are Enclosure.xmin … Enclosure.zmax.", json!({"material": {"type": "string", "description": "Air (default), Water or Engine Oil"}, "padding": {"type": "number"}, "padding_mm": {"type": "array", "items": {"type": "number"}, "minItems": 6, "maxItems": 6}, "min": v3s("Explicit min corner mm"), "size": v3s("Explicit size mm"), "name": {"type": "string"}}), &[]),
        t("remove_load", "Remove a support/load by id (ids are shown in get_model).", json!({"id": {"type": "integer"}}), &["id"]),
        t("clear_loads", "Remove all supports and loads.", json!({}), &[]),
        t("set_mesh", "Set the element size in mm (omit for automatic) and generate the mesh.", json!({"element_size_mm": {"type": "number"}}), &[]),
        t("solve", "Mesh (if needed) and solve the current analysis. Returns min/max of every result, reactions and modal frequencies.", json!({}), &[]),
        t("get_results", "Summary of the last solution: min/max with locations, reaction forces / heat flows, natural frequencies.", json!({}), &[]),
        t("probe", "Result values at the node nearest to a point (mm), including radial/hoop/axial stresses around the cylindrical axis.", json!({"point": v3s("[x,y,z] mm"), "mode": {"type": "integer", "description": "1-based mode number for modal"}, "cylindrical_origin": v3s("Origin on cylindrical axis mm (optional)"), "cylindrical_axis": {"description": "'x', 'y', 'z' or [dx,dy,dz] (optional)"}}), &["point"]),
        t("plot_path", "Sample any result field along a straight line segment from 'from' to 'to' [x,y,z] mm, returning sample points, min/max/mean and CSV (optionally written to csv_path).", json!({"from": v3s("Start [x,y,z] mm"), "to": v3s("End [x,y,z] mm"), "field": {"type": "string", "enum": Field::all().iter().map(|f| f.key()).collect::<Vec<_>>()}, "samples": {"type": "integer", "description": "2..=500 points (default 50)"}, "mode": {"type": "integer"}, "cylindrical_origin": v3s("Cylindrical origin mm (optional)"), "cylindrical_axis": {"description": "'x', 'y', 'z' or [dx,dy,dz] (optional)"}, "csv_path": {"type": "string", "description": "optional .csv output file path"}}), &["from", "to"]),
        t("set_display", "Change what the GUI viewport shows (including cylindrical coordinate origin/axis, perspective projection, and translucent enclosure walls).", json!({"field": {"type": "string", "enum": Field::all().iter().map(|f| f.key()).collect::<Vec<_>>()}, "show_results": {"type": "boolean"}, "mode": {"type": "integer"}, "deformation_scale": {"description": "number, or \"auto\"; 0 = undeformed, 1 = true scale"}, "show_mesh": {"type": "boolean"}, "dark_mode": {"type": "boolean", "description": "Dark UI and viewport theme"}, "perspective": {"type": "boolean", "description": "Use 3D perspective camera (false = orthographic)"}, "translucent_enclosure": {"type": "boolean", "description": "Render enclosure walls as translucent glass"}, "section": {"type": "string", "enum": ["off", "x", "y", "z"], "description": "Cut-plane view through the domain (great for fields in air and fluid flow)"}, "section_position": {"type": "number", "description": "0..1 across the domain (default 0.5)"}, "cylindrical_origin": v3s("Origin [x,y,z] mm for radial/hoop/axial stresses (default [0,0,0])"), "cylindrical_axis": {"description": "'x', 'y', 'z' or direction [dx,dy,dz] for cylindrical stresses (default 'z')"}, "view": {"type": "string", "enum": ["iso", "front", "back", "left", "right", "top", "bottom", "iso_back", "fit"]}}), &[]),
        t("render_image", "Render a PNG picture of the model or of a result contour so you can see it.", json!({"field": {"type": "string", "enum": Field::all().iter().map(|f| f.key()).collect::<Vec<_>>()}, "show_results": {"type": "boolean", "description": "false = show geometry with supports/loads"}, "mode": {"type": "integer"}, "view": {"type": "string", "enum": render::VIEWS}, "deformation_scale": {"type": "number"}, "section": {"type": "string", "enum": ["off", "x", "y", "z"]}, "section_position": {"type": "number"}, "cylindrical_origin": v3s("Cylindrical origin mm (optional)"), "cylindrical_axis": {"description": "'x', 'y', 'z' or [dx,dy,dz] (optional)"}, "width": {"type": "integer"}, "height": {"type": "integer"}}), &[]),
        t("export_vtk", "Write mesh + results to a legacy .vtk file for ParaView.", json!({"path": {"type": "string"}}), &["path"]),
        t("save_project", "Save the project (.simcraft JSON).", json!({"path": {"type": "string"}}), &[]),
        t("open_project", "Open a .simcraft project.", json!({"path": {"type": "string"}}), &["path"]),
        t("screenshot_gui", "Screenshot of the whole SimCraft window (ribbon, browser, viewport) when the GUI is open.", json!({}), &[]),
        t("list_parameter_targets", "Numbers in the current model that studies can vary (e.g. Bracket.size.z, Hole.radius, load2.vector.z, element_size) with current values, plus suggested responses.", json!({}), &[]),
        t("run_parameter_study", "Sweep parameters (full grid, or Latin hypercube with 'samples') and tabulate responses such as mass, max:von_mises, min:safety_factor, frequency:1.", json!({"parameters": {"type": "array", "items": {"type": "object", "properties": {"target": {"type": "string"}, "min": {"type": "number"}, "max": {"type": "number"}, "steps": {"type": "integer"}}, "required": ["target", "min", "max"]}}, "responses": {"type": "array", "items": {"type": "string"}}, "samples": {"type": "integer", "description": "Latin-hypercube sample count instead of a grid"}}), &["parameters", "responses"]),
        t("mesh_convergence", "Mesh-convergence assistant: solve at h, h/1.4 and h/2, Richardson-extrapolate chosen responses, report discretisation error and element size needed for target accuracy.", json!({"responses": {"type": "array", "items": {"type": "string"}, "description": "responses to check (default suggested physics responses)"}, "base_element_size_mm": {"type": "number", "description": "coarsest element size h mm (default current mesh size)"}, "target_error_pct": {"type": "number", "description": "target discretisation error % (default 1.0)"}}), &[]),
        t("surrogate_optimize", "Bayesian optimisation with surrogates: Gaussian-process response surfaces of the objective and constraints choose each next design (expected improvement × probability of meeting the constraints), and every proposal is verified with a real solve — usually finds the optimum in far fewer solves than optimize_design. Reuses the current study's designs when the parameters match.", json!({"parameters": {"type": "array", "items": {"type": "object", "properties": {"target": {"type": "string"}, "min": {"type": "number"}, "max": {"type": "number"}}, "required": ["target", "min", "max"]}}, "objective": {"type": "object", "properties": {"response": {"type": "string"}, "goal": {"type": "string", "enum": ["minimize", "maximize"]}}, "required": ["response"]}, "constraints": {"type": "array", "items": {"type": "object", "properties": {"response": {"type": "string"}, "op": {"type": "string", "enum": ["<=", ">="]}, "value": {"type": "number"}}, "required": ["response", "op", "value"]}}, "max_solves": {"type": "integer", "description": "real solves allowed (default 12)"}, "reuse": {"type": "boolean", "description": "reuse matching designs of the current study (default true)"}}), &["parameters", "objective"]),
        t("fit_surrogate", "Fit Gaussian-process response surfaces to the current study's designs and report their quality (leave-one-out RMSE and Q²) and which parameters drive each response.", json!({}), &[]),
        t("predict_surrogate", "Instant what-if: estimate every study response at new parameter values from the surrogate (with ±2σ uncertainty), without solving. Labelled 'not verified'; confirm with apply_design {values} + solve.", json!({"values": {"type": "object", "additionalProperties": {"type": "number"}, "description": "{parameter target: value}; missing parameters use the middle of their range"}}), &["values"]),
        t("optimize_design", "Find parameter values that minimise or maximise a response subject to constraints, e.g. minimise mass with min:safety_factor >= 2. Latin-hypercube exploration + bounded Nelder-Mead. Use apply_design to adopt the result.", json!({"parameters": {"type": "array", "items": {"type": "object", "properties": {"target": {"type": "string"}, "min": {"type": "number"}, "max": {"type": "number"}}, "required": ["target", "min", "max"]}}, "objective": {"type": "object", "properties": {"response": {"type": "string"}, "goal": {"type": "string", "enum": ["minimize", "maximize"]}}, "required": ["response"]}, "constraints": {"type": "array", "items": {"type": "object", "properties": {"response": {"type": "string"}, "op": {"type": "string", "enum": ["<=", ">="]}, "value": {"type": "number"}}, "required": ["response", "op", "value"]}}, "max_evaluations": {"type": "integer", "description": "default 30"}}), &["parameters", "objective"]),
        t("topology_optimize", "SIMP topology optimization of the current Static Structural setup: keeps the stiffest material layout at the given volume fraction; regions under loads/supports stay solid. Shows the density field and exports the shape as an STL.", json!({"volume_fraction": {"type": "number", "description": "fraction of material to keep (default 0.4)"}, "iterations": {"type": "integer", "description": "default 40"}, "filter_radius_mm": {"type": "number", "description": "default 1.6 × element size"}, "export_path": {"type": "string"}}), &[]),
        t("fatigue_life", "Stress-life (S-N) fatigue of the current stress result: adds Fatigue Life [cycles], Fatigue Damage and Fatigue Safety Factor fields. Static results: the solved load is cycled (fully_reversed, zero_based or stress ratio R) with a mean-stress correction. Harmonic / response spectrum results are amplitudes (fully reversed). Random vibration uses Steinberg's 3-band method over exposure_time_s. S-N data comes from the material (σf', b, endurance limit) or is estimated from the ultimate strength.", json!({"loading": {"type": "string", "enum": ["fully_reversed", "zero_based", "ratio"]}, "ratio": {"type": "number", "description": "stress ratio σmin/σmax for loading=ratio (e.g. 0.1)"}, "correction": {"type": "string", "enum": ["goodman", "gerber", "soderberg", "none"], "description": "mean-stress correction (default goodman)"}, "scale": {"type": "number", "description": "multiplier on the solved stresses (default 1)"}, "design_life": {"type": "number", "description": "required cycles for the safety factor (default 1e6)"}, "exposure_time_s": {"type": "number", "description": "random vibration exposure time in seconds (default 3600)"}}), &[]),
        t("make_report", "Write a self-contained HTML report (model, materials, loads, mesh, rendered views, results, pass/fail checks, study tables and charts). Default check: yield safety factor >= 1.", json!({"path": {"type": "string", "description": "default: next to the project, or ~/SimCraft/report-<time>.html"}, "title": {"type": "string"}, "checks": {"type": "array", "items": {"type": "object", "properties": {"response": {"type": "string"}, "op": {"type": "string", "enum": ["<=", ">="]}, "value": {"type": "number"}}, "required": ["response", "op", "value"]}, "description": "e.g. [{response: 'max:von_mises', op: '<=', value: 150}]"}}), &[]),
        t("get_study", "Results of the last parameter study / optimization / topology run.", json!({}), &[]),
        t("apply_design", "Apply a study design to the model (default: the optimizer's best; 'run' = 1-based row; or 'values' {parameter target: value} for any design, e.g. one you explored with predict_surrogate).", json!({"run": {"type": "integer"}, "values": {"type": "object", "additionalProperties": {"type": "number"}}}), &[]),
        t("plan_simulation", "Draft a step-by-step to-do list for simulating something in SimCraft (e.g. 'EMI on my PCB', 'food heating in a microwave') using a LOCAL model via Ollama. Marks each step ready / workaround / missing-in-SimCraft and lists inputs and missing capabilities. Takes 10-120 s.", json!({"goal": {"type": "string"}, "model": {"type": "string", "description": "Ollama model name (default: best installed)"}}), &["goal"]),
        t("get_plan", "The current simulation to-do list (markdown + JSON), including which steps are done.", json!({}), &[]),
        t("set_plan_step", "Tick (or untick) a to-do step by its 1-based index.", json!({"index": {"type": "integer"}, "done": {"type": "boolean"}}), &["index"]),
        t("list_local_models", "List the local LLMs available to the planner (Ollama).", json!({}), &[]),
        t("get_selection", "Faces the user has currently picked in the GUI viewport.", json!({}), &[]),
    ]
}

fn apply_section(d: &mut Display, a: &Value) -> Result<()> {
    if let Some(sec) = opt_str(a, "section") {
        d.section = match sec {
            "off" | "none" => None,
            "x" => Some(render::Section { axis: 0, position: 0.5 }),
            "y" => Some(render::Section { axis: 1, position: 0.5 }),
            "z" => Some(render::Section { axis: 2, position: 0.5 }),
            o => bail!("section must be off, x, y or z (got '{o}')"),
        };
    }
    if let (Some(p), Some(sec)) = (opt_num(a, "section_position"), d.section.as_mut()) {
        sec.position = p.clamp(0.0, 1.0);
    }
    Ok(())
}

fn apply_cylindrical(cyl: &mut crate::results::CylindricalAxis, a: &Value) -> Result<()> {
    if a.get("cylindrical_origin").is_some() {
        cyl.origin = v3(a, "cylindrical_origin")?;
    }
    if let Some(v) = a.get("cylindrical_axis") {
        cyl.direction = if let Some(s) = v.as_str() {
            match s.to_lowercase().as_str() {
                "x" => [1.0, 0.0, 0.0],
                "y" => [0.0, 1.0, 0.0],
                "z" => [0.0, 0.0, 1.0],
                o => bail!("cylindrical_axis must be x, y, z or [dx,dy,dz] (got '{o}')"),
            }
        } else {
            let dir = v3(a, "cylindrical_axis")?;
            if !(norm(dir) > 1e-12) {
                bail!("cylindrical_axis direction must be non-zero");
            }
            dir
        };
    }
    Ok(())
}

fn study_json(st: &crate::study::Study) -> Value {
    let rows: Vec<Value> = st
        .runs
        .iter()
        .enumerate()
        .map(|(i, r)| {
            let mut o = serde_json::Map::new();
            o.insert("run".into(), json!(i + 1));
            for (p, v) in st.parameters.iter().zip(&r.values) {
                o.insert(p.target.clone(), json!(crate::results::float(*v)));
            }
            for (k, v) in st.responses.iter().zip(&r.responses) {
                o.insert(k.clone(), json!(v.map(crate::results::float)));
            }
            if st.kind == "optimization" || st.kind == "surrogate_optimization" {
                o.insert("feasible".into(), json!(r.feasible));
            }
            if let Some(e) = &r.error {
                o.insert("error".into(), json!(e));
            }
            Value::Object(o)
        })
        .collect();
    let mut v = json!({"kind": st.kind, "message": st.message, "seconds": crate::results::float(st.seconds)});
    if !rows.is_empty() {
        v["runs"] = json!(rows);
    }
    if let Some(b) = st.best {
        v["best_run"] = json!(b + 1);
    }
    if let Some(p) = &st.stl_path {
        v["stl_path"] = json!(p);
        let stride = (st.history.len() / 12).max(1);
        v["history"] = json!(st.history.iter().step_by(stride).map(|(i, c, f)| json!({"iteration": i, "compliance_j": crate::results::float(*c), "volume_fraction": crate::results::float(*f)})).collect::<Vec<_>>());
    }
    if !st.convergence.is_empty() {
        v["convergence_steps"] = json!(st.convergence_steps);
        v["convergence"] = json!(st.convergence);
        v["target_error_pct"] = json!(st.target_error_pct);
        v["recommended_size_mm"] = json!(st.recommended_size_mm);
    }
    v
}
