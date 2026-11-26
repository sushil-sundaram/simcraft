//! Desktop UI module shell: application state, dockable layout, lifecycle, examples, and entry point.

mod browser;
mod charts;
mod dialogs;
mod gl_viewport;
mod header;
mod overlays;
mod properties;
mod ribbon;
mod theme;
mod viewport;

use crate::geometry::V3;
use crate::mesh::Mesh;
use crate::model::*;
use crate::render::{self, Camera, GpuScene};

#[derive(PartialEq, Clone)]
pub(super) struct RenderKey {
    pub rev: u64,
    pub cam: Camera,
    pub w: usize,
    pub h: usize,
    pub sel: u64,
    pub anim: i64,
    pub mesh: usize,
    pub dark: bool,
}
use crate::results::Field;
use crate::session::Shared;
use eframe::egui::{self, Align2, Color32, RichText, Stroke, vec2};
use gl_viewport::GlViewport;
use serde_json::{Value, json};
use std::collections::HashSet;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use theme::{Pal, UnitSystem, install_fonts, install_style, load_prefs, pal, save_dark_pref, save_prefs};

#[derive(PartialEq, Clone, Copy)]
pub(super) enum Tab {
    Geometry,
    Material,
    Setup,
    Mesh,
    Solve,
    Results,
    Optimize,
    Help,
}

#[derive(PartialEq, Clone, Copy, Debug)]
pub(super) enum LoadKind {
    Fixed,
    Displacement,
    CompressionOnly,
    Symmetry,
    Force,
    Pressure,
    Gravity,
    Temperature,
    HeatFlux,
    Convection,
    Radiation,
    HeatGen,
    Voltage,
    Current,
    Coil,
    Magnet,
    Inlet,
    Outlet,
    Port,
    MovingWall,
    BoltPretension,
    Spring,
    RemoteForce,
    RemoteDisplacement,
    PointMass,
}

impl LoadKind {
    pub(super) const STRUCTURAL: [LoadKind; 12] = [
        LoadKind::Fixed,
        LoadKind::Displacement,
        LoadKind::CompressionOnly,
        LoadKind::Symmetry,
        LoadKind::Force,
        LoadKind::Pressure,
        LoadKind::Gravity,
        LoadKind::BoltPretension,
        LoadKind::Spring,
        LoadKind::RemoteForce,
        LoadKind::RemoteDisplacement,
        LoadKind::PointMass,
    ];
    pub(super) const THERMAL: [LoadKind; 5] = [LoadKind::Temperature, LoadKind::HeatFlux, LoadKind::Convection, LoadKind::Radiation, LoadKind::HeatGen];
    pub(super) const ELECTRO: [LoadKind; 5] = [LoadKind::Voltage, LoadKind::Current, LoadKind::Coil, LoadKind::Magnet, LoadKind::Port];
    pub(super) const FLUID: [LoadKind; 5] = [LoadKind::Inlet, LoadKind::Outlet, LoadKind::Symmetry, LoadKind::MovingWall, LoadKind::Gravity];

    pub(super) fn label(self) -> &'static str {
        match self {
            LoadKind::Fixed => "Fixed",
            LoadKind::Displacement => "Displacement",
            LoadKind::CompressionOnly => "Contact",
            LoadKind::Symmetry => "Symmetry",
            LoadKind::Force => "Force",
            LoadKind::Pressure => "Pressure",
            LoadKind::Gravity => "Gravity",
            LoadKind::Temperature => "Temperature",
            LoadKind::HeatFlux => "Heat Flux",
            LoadKind::Convection => "Convection",
            LoadKind::Radiation => "Radiation",
            LoadKind::HeatGen => "Heat Gen.",
            LoadKind::Voltage => "Voltage",
            LoadKind::Current => "Current",
            LoadKind::Coil => "Coil",
            LoadKind::Magnet => "Magnet",
            LoadKind::Inlet => "Inlet",
            LoadKind::Outlet => "Outlet",
            LoadKind::Port => "RF Port",
            LoadKind::MovingWall => "Moving Wall",
            LoadKind::BoltPretension => "Bolt Pretension",
            LoadKind::Spring => "Spring",
            LoadKind::RemoteForce => "Remote Force",
            LoadKind::RemoteDisplacement => "Remote Disp.",
            LoadKind::PointMass => "Point Mass",
        }
    }
    pub(super) fn title(self) -> &'static str {
        self.sample().label()
    }
    pub(super) fn icon(self) -> &'static str {
        match self {
            LoadKind::Fixed => "📌",
            LoadKind::Displacement => "↔",
            LoadKind::CompressionOnly => "⏏",
            LoadKind::Symmetry => "◫",
            LoadKind::Force => "➡",
            LoadKind::Pressure => "⬇",
            LoadKind::Gravity => "🌍",
            LoadKind::Temperature => "🌡",
            LoadKind::HeatFlux => "☀",
            LoadKind::Convection => "💨",
            LoadKind::Radiation => "🔆",
            LoadKind::HeatGen => "🔥",
            LoadKind::Voltage => "⚡",
            LoadKind::Current => "🔌",
            LoadKind::Coil => "🌀",
            LoadKind::Magnet => "🧲",
            LoadKind::Inlet => "⇥",
            LoadKind::Outlet => "⇤",
            LoadKind::Port => "📡",
            LoadKind::MovingWall => "⏩",
            LoadKind::BoltPretension => "🔩",
            LoadKind::Spring => "➰",
            LoadKind::RemoteForce => "↗",
            LoadKind::RemoteDisplacement => "🎯",
            LoadKind::PointMass => "⚖",
        }
    }
    pub(super) fn sample(self) -> Load {
        let f = vec![];
        match self {
            LoadKind::Fixed => Load::FixedSupport { faces: f },
            LoadKind::Displacement => Load::Displacement { faces: f, x: None, y: None, z: None },
            LoadKind::CompressionOnly => Load::CompressionOnly { faces: f },
            LoadKind::Symmetry => Load::Symmetry { faces: f },
            LoadKind::Force => Load::Force { faces: f, vector: [0.0; 3] },
            LoadKind::Pressure => Load::Pressure { faces: f, value_mpa: 0.0 },
            LoadKind::Gravity => Load::Gravity { acceleration: [0.0; 3] },
            LoadKind::Temperature => Load::Temperature { faces: f, value: 0.0 },
            LoadKind::HeatFlux => Load::HeatFlux { faces: f, value: 0.0 },
            LoadKind::Convection => Load::Convection { faces: f, coefficient: 0.0, ambient: 0.0 },
            LoadKind::Radiation => Load::Radiation { faces: f, emissivity: 0.9, ambient: 0.0, enclosure: false },
            LoadKind::HeatGen => Load::HeatGeneration { value: 0.0, body: None, power: None },
            LoadKind::Voltage => Load::Voltage { faces: f, value: 0.0 },
            LoadKind::Current => Load::Current { faces: f, value: 0.0 },
            LoadKind::Coil => Load::Coil { body: String::new(), current_density: 0.0, axis: Default::default(), mode: Default::default(), center: None },
            LoadKind::Magnet => Load::Magnet { body: String::new(), remanence: [0.0; 3] },
            LoadKind::Inlet => Load::Inlet { faces: f, velocity: 0.0, temperature: None },
            LoadKind::Outlet => Load::Outlet { faces: f, pressure: 0.0 },
            LoadKind::MovingWall => Load::MovingWall { faces: f, velocity: [0.0; 3] },
            LoadKind::Port => Load::Port { faces: f, power: 0.0, polarization: Default::default(), profile: "te10".into() },
            LoadKind::BoltPretension => Load::BoltPretension { body: String::new(), force_n: Some(1000.0), adjustment_mm: None, axis: None, cut_mm: None },
            LoadKind::Spring => Load::Spring { faces: f, to_faces: None, stiffness_n_per_mm: 100.0, anchor: None, preload_n: 0.0 },
            LoadKind::RemoteForce => Load::RemoteForce { faces: f, point: [0.0; 3], force: [0.0; 3], moment: [0.0; 3] },
            LoadKind::RemoteDisplacement => Load::RemoteDisplacement { faces: f, point: [0.0; 3], x: None, y: None, z: None, rx: None, ry: None, rz: None, rigid: false },
            LoadKind::PointMass => Load::PointMass { faces: f, mass_kg: 1.0, point: None },
        }
    }
    pub(super) fn color(self) -> Color32 {
        let c = match self {
            LoadKind::Gravity => [70, 150, 90],
            LoadKind::HeatGen => [235, 90, 40],
            k => render::load_color(&k.sample()),
        };
        Color32::from_rgb(c[0], c[1], c[2])
    }
    pub(super) fn needs_faces(self) -> bool {
        !matches!(self, LoadKind::Gravity | LoadKind::HeatGen | LoadKind::Coil | LoadKind::Magnet | LoadKind::BoltPretension)
    }
    pub(super) fn needs_body(self) -> bool {
        matches!(self, LoadKind::Coil | LoadKind::Magnet | LoadKind::BoltPretension)
    }
    pub(super) fn applies(self, kind: AnalysisType) -> bool {
        self.sample().applies_to(kind)
    }
}

#[derive(PartialEq, Clone, Copy)]
pub(super) enum Form {
    None,
    Box,
    Cylinder,
    Sphere,
    Enclosure,
    Contact,
    Study,
    Optimize,
    Convergence,
    Topology,
    Load(LoadKind),
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct Forms {
    pub name: String,
    pub subtract: bool,
    pub bmin: [f64; 3],
    pub bsize: [f64; 3],
    pub cbase: [f64; 3],
    pub caxis: usize,
    pub cradius: f64,
    pub cheight: f64,
    pub scenter: [f64; 3],
    pub sradius: f64,
    pub vector: [f64; 3],
    pub value: f64,
    pub coef: f64,
    pub ambient: f64,
    pub rad_eps: f64,
    pub rad_enclosure: bool,
    pub dcomp: [(bool, f64); 3],
    pub grav: [f64; 3],
    pub material: String,
    pub mesh_size: f64,
    pub body: String,
    pub coil_axis: usize,
    pub coil_axial: bool,
    pub remanence: [f64; 3],
    pub velocity: f64,
    pub outlet_p: f64,
    pub padding: f64,
    pub enc_box: bool,
    pub enc_min: [f64; 3],
    pub enc_size: [f64; 3],
    pub fluid: String,
    pub heat_body: String,
    pub table_text: String,
    pub table_src: Vec<[f64; 2]>,
    pub fat_open: bool,
    pub fat_loading: String,
    pub fat_ratio: f64,
    pub fat_correction: String,
    pub fat_scale: f64,
    pub fat_life: f64,
    pub fat_exposure_h: f64,
    pub port_power: f64,
    pub heat_watts: bool,
    pub heat_power: f64,
    pub wall_v: [f64; 3],
    pub port_pol: usize,
    pub port_te10: bool,
    pub contact_a: String,
    pub contact_b: String,
    pub contact_kind: usize,
    pub contact_mu: f64,
    pub contact_off: f64,
    pub bolt_by_force: bool,
    pub bolt_force: f64,
    pub bolt_adj: f64,
    pub spring_k: f64,
    pub spring_pre: f64,
    pub spring_to: String,
    pub remote_pt: [f64; 3],
    pub remote_moment: [f64; 3],
    pub remote_rot: [(bool, f64); 3],
    pub remote_rigid: bool,
    pub mass_offset: bool,
}

impl Default for Forms {
    fn default() -> Self {
        Forms {
            name: String::new(),
            subtract: false,
            bmin: [0.0; 3],
            bsize: [100.0, 40.0, 10.0],
            cbase: [50.0, 20.0, 0.0],
            caxis: 2,
            cradius: 8.0,
            cheight: 10.0,
            scenter: [0.0; 3],
            sradius: 20.0,
            vector: [0.0, 0.0, -1000.0],
            value: 1.0,
            coef: 20.0,
            rad_eps: 0.9,
            rad_enclosure: false,
            ambient: 22.0,
            dcomp: [(true, 0.0), (false, 0.0), (false, 0.0)],
            grav: [0.0, 0.0, -9.80665],
            material: "Structural Steel".into(),
            mesh_size: 2.0,
            body: String::new(),
            coil_axis: 2,
            coil_axial: false,
            remanence: [0.0, 0.0, 1.3],
            velocity: 1.0,
            outlet_p: 0.0,
            padding: 0.5,
            enc_box: false,
            enc_min: [-50.0, -50.0, -50.0],
            enc_size: [100.0, 100.0, 100.0],
            fluid: "Air".into(),
            heat_body: String::new(),
            table_text: String::new(),
            table_src: Vec::new(),
            fat_open: false,
            fat_loading: "fully_reversed".into(),
            fat_ratio: 0.1,
            fat_correction: "goodman".into(),
            fat_scale: 1.0,
            fat_life: 1e6,
            fat_exposure_h: 1.0,
            port_power: 800.0,
            heat_watts: true,
            heat_power: 5.0,
            wall_v: [0.1, 0.0, 0.0],
            port_pol: 2,
            port_te10: true,
            contact_a: String::new(),
            contact_b: String::new(),
            contact_kind: 2,
            contact_mu: 0.2,
            contact_off: 0.0,
            bolt_by_force: true,
            bolt_force: 1000.0,
            bolt_adj: 0.05,
            spring_k: 100.0,
            spring_pre: 0.0,
            spring_to: String::new(),
            remote_pt: [0.0; 3],
            remote_moment: [0.0; 3],
            remote_rot: [(false, 0.0); 3],
            remote_rigid: false,
            mass_offset: false,
        }
    }
}

#[derive(PartialEq, Clone, Copy, Eq, Hash, Debug)]
pub(super) enum DrawerTab {
    Charts,
    Study,
    Pcb,
    Planner,
    Help,
}

impl DrawerTab {
    pub(super) const ALL: [DrawerTab; 5] = [DrawerTab::Charts, DrawerTab::Study, DrawerTab::Pcb, DrawerTab::Planner, DrawerTab::Help];
    pub(super) fn label(self) -> &'static str {
        match self {
            DrawerTab::Charts => "📈 Charts & Path",
            DrawerTab::Study => "📊 Studies",
            DrawerTab::Pcb => "🔌 PCB DC",
            DrawerTab::Planner => "✨ Copilot",
            DrawerTab::Help => "📖 Guide",
        }
    }
}

pub(super) struct DrawerState {
    pub open: bool,
    pub active: DrawerTab,
    pub detached: HashSet<DrawerTab>,
}

impl Default for DrawerState {
    fn default() -> Self {
        Self { open: false, active: DrawerTab::Charts, detached: HashSet::new() }
    }
}

impl DrawerState {
    pub(super) fn is_showing(&self, tab: DrawerTab) -> bool {
        (self.open && self.active == tab && !self.detached.contains(&tab)) || self.detached.contains(&tab)
    }
}

#[derive(Clone)]
pub(super) enum Pending {
    New,
    Open,
    OpenPath(String),
    Example(&'static str),
    Quit,
}

pub(super) type MeshKey = (u64, bool, bool, u64);

pub(super) struct MeshJob {
    pub key: MeshKey,
    pub cancel: Arc<AtomicBool>,
    pub done: Arc<std::sync::Mutex<Option<Result<Mesh, String>>>>,
}

pub(super) type NetsJobSlot = Arc<std::sync::Mutex<Option<(String, Result<Vec<(String, Vec<String>)>, String>)>>>;

#[derive(Default)]
pub(super) struct PcbUi {
    pub open: bool,
    pub path: String,
    pub nets: Vec<(String, Vec<String>)>,
    pub nets_job: Option<(Arc<AtomicBool>, NetsJobSlot)>,
    pub filter: String,
    pub net: String,
    pub from: String,
    pub to: String,
    pub current: f64,
    pub temp: f64,
    pub job: Option<Arc<std::sync::Mutex<Option<Result<Value, String>>>>>,
    pub result: Option<Value>,
    pub error: Option<String>,
}

pub(super) fn default_net(nets: &[(String, Vec<String>)]) -> Option<usize> {
    let rail = |n: &str| {
        let u = n.to_uppercase();
        ["VBUS", "VIN", "VCC", "VDD", "PWR", "+", "12V", "5V", "3V3", "VBAT"].iter().any(|k| u.contains(k)) && !u.contains("GND")
    };
    nets.iter().position(|(n, _)| rail(n)).or_else(|| (0..nets.len()).max_by_key(|&i| nets[i].1.len()))
}

#[derive(Default)]
pub(super) struct HelpUi {
    pub open: bool,
    pub section: usize,
    pub query: String,
    pub focus_search: bool,
    pub scroll_top: bool,
}

#[derive(Default)]
pub(super) struct PathUi {
    pub tab: usize,
    pub from: [f64; 3],
    pub to: [f64; 3],
    pub samples: usize,
    pub field: Option<Field>,
    pub picking: u8,
    pub inited_res: u64,
    pub cache_key: String,
    pub plot: Option<crate::results::PathPlot>,
    pub error: Option<String>,
}

#[derive(Default)]
pub(super) struct OptUi {
    pub params: Vec<(String, f64, f64, usize)>,
    pub responses: Vec<String>,
    pub objective: String,
    pub maximize: bool,
    pub constraints: Vec<(String, bool, f64)>,
    pub max_evals: usize,
    pub bayes: bool,
    pub whatif: Vec<f64>,
    pub whatif_key: String,
    pub whatif_pred: Option<Value>,
    pub vf: f64,
    pub iters: usize,
    pub conv_custom_h: bool,
    pub conv_base_h: f64,
    pub conv_target_err: f64,
    pub show_results: bool,
    pub job: Option<Arc<std::sync::Mutex<Option<Result<(), String>>>>>,
    pub error: Option<String>,
    pub seen: String,
}

#[derive(PartialEq, Eq, Clone, Copy, Debug, Default)]
pub(super) enum SelectMode {
    #[default]
    Face,
    Body,
}

#[derive(Clone, Debug)]
pub(super) struct BodyDraft {
    pub orig_name: String,
    pub name: String,
    pub op: crate::geometry::Op,
    pub role: crate::geometry::Role,
    pub material: String,
    pub shape: crate::geometry::Shape,
}

#[derive(Clone, Debug)]
pub(super) struct LoadDraft {
    pub id: u32,
    pub kind: LoadKind,
    pub f: Forms,
    pub faces: Vec<String>,
    pub body: String,
}

#[derive(Clone, Debug)]
pub(super) enum EditTarget {
    Body(BodyDraft),
    Load(Box<LoadDraft>),
}

pub(super) type ModelsJobSlot = Arc<std::sync::Mutex<Option<Result<Vec<String>, String>>>>;

#[derive(Default)]
pub(super) struct Planner {
    pub open: bool,
    pub mode: usize,
    pub goal: String,
    pub models: Option<Result<Vec<String>, String>>,
    pub models_job: Option<ModelsJobSlot>,
    pub model: String,
    pub job: Option<Arc<std::sync::Mutex<Option<Result<(), String>>>>>,
    pub error: Option<String>,
    pub copied_at: f64,
    pub executed_actions: HashSet<usize>,
}

pub struct SimApp {
    pub(super) core: Shared,
    pub(super) cam: Camera,
    pub(super) tab: Tab,
    pub(super) form: Form,
    pub(super) f: Forms,
    pub(super) gl_vp: Option<GlViewport>,
    pub(super) gpu_scene: Option<((u64, u64, usize, bool), Arc<GpuScene>)>,
    pub(super) tex: Option<egui::TextureHandle>,
    pub(super) key: Option<RenderKey>,
    pub(super) confirm: Option<Pending>,
    pub(super) confirm_external: Option<(String, Vec<String>)>,
    pub(super) sel_body: Option<String>,
    pub(super) sel_bodies: HashSet<String>,
    pub(super) batch_material: String,
    pub(super) sel_load: Option<u32>,
    pub(super) edit_target: Option<(String, EditTarget)>,
    pub(super) xform: (String, [f64; 3], usize, usize, f64, bool),
    pub(super) mat_edit: Option<(String, Value)>,
    pub(super) show_chart: bool,
    pub(super) help: HelpUi,
    pub(super) pcb: PcbUi,
    pub(super) probe_mode: bool,
    pub(super) probes: Vec<(u32, u64)>,
    pub(super) allow_close: bool,
    pub(super) last_autosave: f64,
    pub(super) recovery_offered: bool,
    pub(super) scene: Option<((u64, u64, i64, usize), Arc<render::Scene>)>,
    pub(super) face_buf: Vec<u32>,
    pub(super) buf_size: (usize, usize),
    pub(super) render_scale: f32,
    pub(super) selected: HashSet<u32>,
    pub(super) sel_version: u64,
    pub(super) mesh_ptr: usize,
    pub(super) mesh_job: Option<MeshJob>,
    pub(super) mesh_err: Option<(MeshKey, String)>,
    pub(super) bridge: String,
    pub(super) animate: bool,
    pub(super) need_fit: bool,
    pub(super) toast: Option<(String, bool, f64)>,
    pub(super) legend: Option<((f64, f64), f64)>,
    pub(super) dark: bool,
    pub(super) p: Pal,
    pub(super) units: UnitSystem,
    pub(super) trackpad_pan: bool,
    pub(super) show_glyphs: bool,
    pub(super) select_mode: SelectMode,
    pub(super) browser_open: bool,
    pub(super) props_open: bool,
    pub(super) browser_filter: String,
    pub(super) hover_cache: Option<((usize, bool, u32), Vec<[V3; 2]>)>,
    pub(super) ctx_menu_pos: Option<egui::Pos2>,
    pub(super) ctx_hits: Vec<(u32, f64, V3)>,
    pub(super) gizmo_busy: bool,
    pub(super) gizmo_drag: Option<(usize, f64, V3)>,
    pub(super) highlight_point: Option<(V3, String)>,
    pub(super) show_mcp: bool,
    pub(super) planner: Planner,
    pub(super) opt: OptUi,
    pub(super) path: PathUi,
    pub(super) drawer: DrawerState,
    pub(super) copied: Option<(&'static str, f64)>,
}

impl SimApp {
    pub fn new(cc: &eframe::CreationContext<'_>, core: Shared, bridge: String) -> Self {
        install_fonts(&cc.egui_ctx);
        install_style(&cc.egui_ctx);
        let prefs = load_prefs();
        let dark = prefs.dark_mode.unwrap_or_else(|| cc.egui_ctx.system_theme() == Some(egui::Theme::Dark));
        cc.egui_ctx.set_theme(if dark { egui::ThemePreference::Dark } else { egui::ThemePreference::Light });
        core.session.lock().unwrap_or_else(std::sync::PoisonError::into_inner).display.dark = dark;
        *core.gui_ctx.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = Some(cc.egui_ctx.clone());
        let ctx = cc.egui_ctx.clone();
        *core.notify.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = Some(Box::new(move || ctx.request_repaint()));
        let gl_vp = cc.gl.as_ref().and_then(GlViewport::try_new);
        let open_planner = std::env::var_os("SIMCRAFT_OPEN_PLANNER").is_some();
        let mut drawer = DrawerState::default();
        if open_planner {
            drawer.open = true;
            drawer.active = DrawerTab::Planner;
        }
        SimApp {
            core,
            cam: Camera::default(),
            tab: Tab::Geometry,
            form: Form::None,
            f: Forms::default(),
            gl_vp,
            gpu_scene: None,
            tex: None,
            key: None,
            confirm: None,
            confirm_external: None,
            sel_body: None,
            sel_bodies: HashSet::new(),
            batch_material: "Aluminum 6061-T6".into(),
            sel_load: None,
            edit_target: None,
            xform: ("move".into(), [10.0, 0.0, 0.0], 3, 0, 0.0, true),
            mat_edit: None,
            show_chart: false,
            help: HelpUi::default(),
            pcb: PcbUi { current: 1.0, temp: 20.0, ..Default::default() },
            probe_mode: false,
            probes: Vec::new(),
            allow_close: false,
            last_autosave: 0.0,
            recovery_offered: false,
            scene: None,
            face_buf: vec![],
            buf_size: (0, 0),
            render_scale: 1.0,
            selected: HashSet::new(),
            sel_version: 0,
            mesh_ptr: 0,
            mesh_job: None,
            mesh_err: None,
            bridge,
            animate: false,
            need_fit: true,
            toast: None,
            legend: None,
            dark,
            p: pal(dark),
            units: prefs.unit_system,
            trackpad_pan: prefs.trackpad_pan,
            show_glyphs: true,
            select_mode: SelectMode::Face,
            browser_open: true,
            props_open: true,
            browser_filter: String::new(),
            hover_cache: None,
            ctx_menu_pos: None,
            ctx_hits: Vec::new(),
            gizmo_busy: false,
            gizmo_drag: None,
            highlight_point: None,
            show_mcp: std::env::var_os("SIMCRAFT_OPEN_CONNECT").is_some(),
            opt: OptUi { max_evals: 12, bayes: true, vf: 0.4, iters: 40, conv_base_h: 4.0, conv_target_err: 1.0, objective: "mass".into(), ..Default::default() },
            path: PathUi { samples: 50, to: [100.0, 0.0, 0.0], ..Default::default() },
            planner: Planner { open: open_planner, ..Default::default() },
            drawer,
            copied: None,
        }
    }

    pub(super) fn persist_prefs(&self) {
        save_prefs(&theme::UiPrefs {
            dark_mode: Some(self.dark),
            unit_system: self.units,
            trackpad_pan: self.trackpad_pan,
        });
    }

    pub(super) fn set_body_vis(&mut self, name: &str, mode: u8) {
        let (mut hidden, mut ghost) = {
            let s = self.core.session.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            (s.display.hidden_bodies.iter().cloned().collect::<Vec<_>>(), s.display.ghost_bodies.iter().cloned().collect::<Vec<_>>())
        };
        hidden.retain(|x| x != name);
        ghost.retain(|x| x != name);
        match mode {
            1 => ghost.push(name.to_string()),
            2 => hidden.push(name.to_string()),
            _ => {}
        }
        self.run("set_display", json!({"hidden_bodies": hidden, "ghost_bodies": ghost}));
    }

    pub(super) fn cycle_body_vis(&mut self, name: &str) {
        let mode = {
            let s = self.core.session.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            if s.display.hidden_bodies.contains(name) {
                0 // Hidden -> Visible
            } else if s.display.ghost_bodies.contains(name) {
                2 // Ghost -> Hidden
            } else {
                1 // Visible -> Ghost
            }
        };
        self.set_body_vis(name, mode);
    }

    pub(super) fn isolate_body(&mut self, name: &str) {
        let names: Vec<String> = {
            let s = self.core.session.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            s.model.bodies.iter().filter(|b| b.name != name).map(|b| b.name.clone()).collect()
        };
        self.run("set_display", json!({"hidden_bodies": names, "ghost_bodies": []}));
    }

    pub(super) fn show_all_bodies(&mut self) {
        self.run("set_display", json!({"show_all_bodies": true}));
    }

    pub(super) fn run(&mut self, tool: &str, args: Value) -> bool {
        if matches!(tool, "new_model" | "open_project") {
            self.edit_target = None;
            self.sel_body = None;
            self.sel_bodies.clear();
            self.sel_load = None;
            self.highlight_point = None;
            self.ctx_menu_pos = None;
            self.gizmo_drag = None;
            self.planner.executed_actions.clear();
        }
        match self.core.call(tool, &args) {
            Ok(o) => {
                let short: String = o.text.lines().next().unwrap_or("").chars().take(140).collect();
                if !short.starts_with('{') {
                    self.toast = Some((short, false, 3.5));
                }
                true
            }
            Err(e) => {
                self.toast = Some((format!("{e:#}"), true, 8.0));
                false
            }
        }
    }

    pub(super) fn selected_names(&self, mesh: Option<&Arc<Mesh>>) -> Vec<String> {
        let Some(m) = mesh else { return vec![] };
        let mut v: Vec<String> = self.selected.iter().filter_map(|&p| m.patches.get(p as usize).map(|x| x.name.clone())).collect();
        v.sort();
        v
    }

    pub(super) fn current_mesh(&self) -> Option<Arc<Mesh>> {
        let s = self.core.session.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        if s.mesh_valid() { s.mesh.clone() } else { None }
    }

    pub(super) fn solve(&mut self) {
        if self.core.is_busy() {
            return;
        }
        let core = self.core.clone();
        std::thread::spawn(move || {
            let _ = core.call("solve", &json!({}));
        });
        self.tab = Tab::Results;
    }

    fn sync_theme(&mut self, ctx: &egui::Context) {
        let want = self.core.session.lock().unwrap_or_else(std::sync::PoisonError::into_inner).display.dark;
        if want != self.dark {
            self.dark = want;
            self.p = pal(want);
            ctx.set_theme(if want { egui::ThemePreference::Dark } else { egui::ThemePreference::Light });
            save_dark_pref(want);
        }
    }

    pub(super) fn start_job(&mut self, tool: &str, args: Value) {
        let slot = Arc::new(std::sync::Mutex::new(None));
        let (core, out, tool) = (self.core.clone(), slot.clone(), tool.to_string());
        std::thread::spawn(move || {
            let r = core.call(&tool, &args).map(|_| ()).map_err(|e| format!("{e:#}"));
            *out.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = Some(r);
        });
        self.opt.job = Some(slot);
        self.opt.error = None;
    }

    pub(super) fn dirty(&self) -> bool {
        self.core.session.lock().unwrap_or_else(std::sync::PoisonError::into_inner).is_dirty()
    }

    pub(super) fn guard(&mut self, a: Pending) {
        if self.dirty() {
            self.confirm = Some(a);
        } else {
            self.perform(a);
        }
    }

    pub(super) fn perform(&mut self, a: Pending) {
        match a {
            Pending::New => {
                self.run("new_model", json!({}));
            }
            Pending::Open => {
                if let Some(p) = rfd::FileDialog::new().add_filter("SimCraft", &["simcraft", "json"]).pick_file() {
                    let s = p.to_string_lossy().to_string();
                    self.open_checked(&s);
                }
            }
            Pending::OpenPath(p) => {
                self.open_checked(&p);
            }
            Pending::Example(name) => match name {
                "bracket" => self.example_bracket(),
                "heatsink" => self.example_heatsink(),
                "fork" => self.example_fork(),
                "buckling" => self.example_buckling(),
                "capacitor" => self.example_capacitor(),
                "busbar" => self.example_busbar(),
                "coil" => self.example_coil(),
                "microwave" => self.example_microwave(),
                "heatsink_cht" => self.example_heatsink_cht(),
                _ => self.example_cfd(),
            },
            Pending::Quit => {
                let _ = std::fs::remove_file(crate::session::autosave_path());
                self.allow_close = true;
                if let Some(ctx) = self.core.gui_ctx.lock().unwrap_or_else(std::sync::PoisonError::into_inner).clone() {
                    ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                }
            }
        }
        self.need_fit = true;
    }

    fn open_checked(&mut self, path: &str) {
        if std::path::Path::new(path) != crate::session::autosave_path()
            && let Ok(ext) = crate::session::external_assets(path)
                && !ext.is_empty() {
                    self.confirm_external = Some((path.to_string(), ext));
                    return;
                }
        self.run("open_project", json!({"path": path}));
    }

    pub(super) fn save_as(&mut self) {
        if let Some(p) = rfd::FileDialog::new().add_filter("SimCraft", &["simcraft"]).set_file_name("model.simcraft").save_file() {
            self.run("save_project", json!({"path": p.to_string_lossy()}));
        }
    }

    pub(super) fn import_kicad(&mut self) {
        if let Some(p) = rfd::FileDialog::new().add_filter("KiCad board", &["kicad_pcb"]).pick_file() {
            let path = p.to_string_lossy().to_string();
            match self.core.call("import_kicad", &json!({"path": path})) {
                Ok(o) => {
                    let v: Value = serde_json::from_str(&o.text).unwrap_or(Value::Null);
                    let parts = v["components"].as_array().map(|a| a.len()).unwrap_or(0);
                    let warn = v["warnings"].as_array().and_then(|w| w.first()).and_then(|w| w.as_str()).map(|w| format!(" ⚠ {w}")).unwrap_or_default();
                    self.toast = Some((format!("Imported {} × {} mm board with {parts} components. Add Heat Generation (W) on the parts and a convection or airflow boundary.{warn}", v["size_mm"][0], v["size_mm"][1]), !warn.is_empty(), 8.0));
                    let t = v["thickness_mm"].as_f64().unwrap_or(1.6);
                    let cur = self.core.session.lock().unwrap_or_else(std::sync::PoisonError::into_inner).model.element_size;
                    if cur.is_none_or(|h| h > t / 2.0) {
                        self.run("set_mesh", json!({"element_size_mm": (t / 2.0 * 10.0).floor() / 10.0}));
                        self.f.mesh_size = (t / 2.0 * 10.0).floor() / 10.0;
                    }
                }
                Err(e) => self.toast = Some((format!("{e:#}"), true, 8.0)),
            }
            self.need_fit = true;
        }
    }

    pub(super) fn import_stl(&mut self) {
        if let Some(p) = rfd::FileDialog::new().add_filter("STL", &["stl", "STL"]).pick_file() {
            self.run("import_stl", json!({"path": p.to_string_lossy()}));
            self.need_fit = true;
        }
    }

    pub(super) fn import_step(&mut self) {
        if self.core.is_busy() {
            self.toast = Some(("Wait for the running job to finish first.".into(), true, 4.0));
            return;
        }
        if let Some(p) = rfd::FileDialog::new().add_filter("STEP CAD", &["step", "stp", "STEP", "STP"]).pick_file() {
            let core = self.core.clone();
            let args = json!({"path": p.to_string_lossy()});
            std::thread::spawn(move || {
                let _ = core.call("import_step", &args);
            });
            self.need_fit = true;
        }
    }

    fn status(&mut self, ui: &mut egui::Ui) {
        let p = self.p;
        let pr = self.core.progress.lock().unwrap_or_else(std::sync::PoisonError::into_inner).clone();
        let (last, err) = {
            let s = self.core.session.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            (s.log.last().cloned().unwrap_or_default(), s.last_error.clone())
        };
        ui.horizontal(|ui| {
            if pr.running {
                ui.add(egui::ProgressBar::new(pr.fraction as f32).desired_width(240.0).desired_height(14.0).text(RichText::new(pr.message.clone()).small()).fill(p.accent));
                if ui.small_button(RichText::new("■ Stop").color(p.err)).on_hover_text("Stop the running solve or study").clicked() {
                    let _ = self.core.call("cancel", &json!({}));
                }
                ui.ctx().request_repaint_after(std::time::Duration::from_millis(100));
            }
            match err {
                Some(e) => ui.label(RichText::new(format!("✖ {e}")).small().color(p.err)),
                None => ui.label(RichText::new(last).small().color(p.weak)),
            };
        });
    }

    fn script(&mut self, steps: &[(&str, Value)]) {
        for (t, a) in steps {
            if !self.run(t, a.clone()) {
                return;
            }
        }
        self.need_fit = true;
        self.form = Form::None;
        self.tab = Tab::Solve;
        self.toast = Some(("Example loaded — press ⚡ Solve".into(), false, 4.0));
    }
    fn example_bracket(&mut self) {
        self.script(&[
            ("new_model", json!({})),
            ("add_box", json!({"name": "Bracket", "min": [0, 0, 0], "size": [120, 40, 10]})),
            ("add_cylinder", json!({"name": "Hole", "base": [80, 20, 0], "axis": "z", "radius": 8, "height": 10, "operation": "subtract"})),
            ("set_material", json!({"material": "Aluminum 6061-T6"})),
            ("set_mesh", json!({"element_size_mm": 2})),
            ("set_analysis", json!({"type": "static_structural"})),
            ("add_fixed_support", json!({"faces": ["Bracket.xmin"]})),
            ("add_force", json!({"faces": ["Bracket.xmax"], "vector": [0, 0, -500]})),
        ]);
    }
    fn example_heatsink(&mut self) {
        let mut steps = vec![("new_model", json!({})), ("add_box", json!({"name": "Base", "min": [0, 0, 0], "size": [60, 40, 5]}))];
        for i in 0..6 {
            steps.push(("add_box", json!({"name": format!("Fin{}", i + 1), "min": [2.5 + i as f64 * 10.0, 0, 5], "size": [5, 40, 30]})));
        }
        steps.extend([
            ("set_material", json!({"material": "Aluminum 6061-T6"})),
            ("set_mesh", json!({"element_size_mm": 1.25})),
            ("set_analysis", json!({"type": "steady_thermal"})),
            ("add_heat_flux", json!({"faces": ["Base.zmin"], "value": 20000})),
            ("add_convection", json!({"faces": ["all"], "coefficient": 25, "ambient": 22})),
        ]);
        self.script(&steps);
    }
    fn example_buckling(&mut self) {
        self.script(&[
            ("new_model", json!({})),
            ("add_box", json!({"name": "Column", "min": [0, 0, 0], "size": [10, 10, 200]})),
            ("set_mesh", json!({"element_size_mm": 2})),
            ("set_analysis", json!({"type": "linear_buckling", "modes": 3})),
            ("add_fixed_support", json!({"faces": ["Column.zmin"]})),
            ("add_force", json!({"faces": ["Column.zmax"], "vector": [0, 0, -1000]})),
        ]);
    }
    fn example_capacitor(&mut self) {
        self.script(&[
            ("new_model", json!({})),
            ("add_box", json!({"name": "TopPlate", "min": [0, 0, 6], "size": [40, 40, 2], "material": "Copper"})),
            ("add_box", json!({"name": "BottomPlate", "min": [0, 0, 0], "size": [40, 40, 2], "material": "Copper"})),
            ("add_box", json!({"name": "Dielectric", "min": [0, 0, 2], "size": [40, 40, 4], "material": "FR-4"})),
            ("add_enclosure", json!({"material": "Air", "padding": 0.4})),
            ("set_analysis", json!({"type": "electrostatic"})),
            ("set_mesh", json!({"element_size_mm": 1})),
            ("add_voltage", json!({"faces": ["TopPlate"], "value": 100})),
            ("add_voltage", json!({"faces": ["BottomPlate"], "value": 0})),
        ]);
    }
    fn example_busbar(&mut self) {
        self.script(&[
            ("new_model", json!({})),
            ("add_box", json!({"name": "BusBar", "min": [0, 0, 0], "size": [150, 20, 5], "material": "Copper"})),
            ("add_cylinder", json!({"name": "BoltA", "base": [12, 10, 0], "radius": 4, "height": 5, "operation": "subtract"})),
            ("add_cylinder", json!({"name": "BoltB", "base": [138, 10, 0], "radius": 4, "height": 5, "operation": "subtract"})),
            ("set_mesh", json!({"element_size_mm": 1})),
            ("set_analysis", json!({"type": "thermal_electric"})),
            ("add_current", json!({"faces": ["BoltA.side"], "value": 400})),
            ("add_voltage", json!({"faces": ["BoltB.side"], "value": 0})),
            ("add_convection", json!({"faces": ["all"], "coefficient": 15, "ambient": 25})),
        ]);
    }
    fn example_coil(&mut self) {
        self.script(&[
            ("new_model", json!({})),
            ("add_cylinder", json!({"name": "Coil", "base": [0, 0, -15], "radius": 22, "height": 30, "material": "Copper"})),
            ("add_cylinder", json!({"name": "Bore", "base": [0, 0, -15], "radius": 14, "height": 30, "operation": "subtract"})),
            ("add_cylinder", json!({"name": "Core", "base": [0, 0, -25], "radius": 10, "height": 50, "material": "Electrical Steel (M19)"})),
            ("add_enclosure", json!({"material": "Air", "padding": 0.6})),
            ("set_analysis", json!({"type": "magnetostatic"})),
            ("set_mesh", json!({"element_size_mm": 2.5})),
            ("add_coil", json!({"body": "Coil", "current_density": 3, "axis": "z", "mode": "circumferential"})),
        ]);
    }
    fn example_microwave(&mut self) {
        self.script(&[
            ("new_model", json!({})),
            ("add_box", json!({"name": "Tray", "min": [30, 30, 12], "size": [180, 180, 6], "material": "Microwave Glass Tray"})),
            ("add_box", json!({"name": "Food", "min": [80, 85, 18], "size": [80, 70, 40], "material": "Food (water-rich, 2.45 GHz)"})),
            ("add_enclosure", json!({"material": "Air", "min": [0, 0, 0], "size": [240, 240, 160]})),
            ("set_analysis", json!({"type": "microwave_heating", "frequency_ghz": 2.45, "em_boundary": "pec", "end_time": 30, "time_steps": 30, "initial_temperature": 5})),
            ("set_mesh", json!({"element_size_mm": 1.6})),
            ("add_port", json!({"faces": ["Enclosure.xmin"], "power": 800, "polarization": "z", "profile": "te10"})),
        ]);
    }
    fn example_heatsink_cht(&mut self) {
        let mut calls: Vec<(&str, Value)> = vec![
            ("new_model", json!({})),
            ("add_box", json!({"name": "Chip", "min": [12.5, 12.5, 0], "size": [15, 15, 2], "material": "Copper C101"})),
            ("add_box", json!({"name": "Base", "min": [0, 0, 2], "size": [40, 40, 4], "material": "Aluminum 6061-T6"})),
        ];
        let fins: Vec<(String, f64)> = (0..6).map(|i| (format!("Fin{}", i + 1), 1.0 + i as f64 * 7.2)).collect();
        for (name, y) in &fins {
            calls.push(("add_box", json!({"name": name, "min": [0, y, 6], "size": [40, 2, 25], "material": "Aluminum 6061-T6"})));
        }
        calls.extend([
            ("add_enclosure", json!({"material": "Air", "min": [-40, -10, 0], "size": [160, 60, 45]})),
            ("set_analysis", json!({"type": "conjugate_heat_transfer", "turbulence": "lvel", "reference_temperature": 25, "max_iterations": 6000})),
            ("set_mesh", json!({"element_size_mm": 2})),
            ("add_inlet", json!({"faces": ["Enclosure.xmin"], "velocity": 2.0, "temperature": 25})),
            ("add_outlet", json!({"faces": ["Enclosure.xmax"], "pressure": 0})),
            ("add_heat_generation", json!({"body": "Chip", "power": 10.0})),
        ]);
        self.script(&calls);
    }
    fn example_cfd(&mut self) {
        self.script(&[
            ("new_model", json!({})),
            ("add_cylinder", json!({"name": "Cylinder", "base": [0, 0, 0], "axis": "y", "radius": 10, "height": 20, "material": "Aluminum 6061-T6"})),
            ("add_enclosure", json!({"material": "Water", "padding_mm": [40, 120, 0, 0, 40, 40]})),
            ("set_analysis", json!({"type": "fluid", "max_iterations": 3000})),
            ("set_mesh", json!({"element_size_mm": 2})),
            ("add_inlet", json!({"faces": ["Enclosure.xmin"], "velocity": 0.002})),
            ("add_outlet", json!({"faces": ["Enclosure.xmax"], "pressure": 0})),
            ("add_symmetry", json!({"faces": ["Enclosure.ymin", "Enclosure.ymax", "Enclosure.zmin", "Enclosure.zmax"]})),
        ]);
    }
    fn example_fork(&mut self) {
        self.script(&[
            ("new_model", json!({})),
            ("add_box", json!({"name": "Handle", "min": [-3, -3, 0], "size": [6, 6, 40]})),
            ("add_box", json!({"name": "Yoke", "min": [-12, -3, 40], "size": [24, 6, 6]})),
            ("add_box", json!({"name": "TineA", "min": [-12, -3, 46], "size": [6, 6, 80]})),
            ("add_box", json!({"name": "TineB", "min": [6, -3, 46], "size": [6, 6, 80]})),
            ("set_material", json!({"material": "Stainless Steel 304"})),
            ("set_mesh", json!({"element_size_mm": 1.5})),
            ("set_analysis", json!({"type": "modal", "modes": 6})),
            ("add_fixed_support", json!({"faces": ["Handle.zmin"]})),
        ]);
    }
}

impl eframe::App for SimApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.sync_theme(ui.ctx());
        let p = self.p;
        let ctx = ui.ctx().clone();
        self.poll_background_jobs(&ctx);

        let bar = |fill: Color32| egui::Frame::new().fill(fill).inner_margin(egui::Margin::symmetric(12, 6)).stroke(Stroke::new(1.0, p.border));
        egui::Panel::top("header").frame(bar(p.ribbon)).show(ui, |ui| self.header(ui));
        egui::Panel::top("ribbon").frame(bar(p.ribbon).inner_margin(egui::Margin { left: 12, right: 12, top: 8, bottom: 6 })).show(ui, |ui| self.ribbon(ui));
        egui::Panel::bottom("status").frame(bar(p.panel).inner_margin(egui::Margin::symmetric(12, 4))).show(ui, |ui| self.status(ui));
        if self.browser_open {
            egui::Panel::left("browser").frame(bar(p.panel).inner_margin(10)).resizable(true).default_size(260.0).show(ui, |ui| self.browser(ui));
        }
        if self.props_open {
            egui::Panel::right("props").frame(bar(p.panel).inner_margin(10)).resizable(true).default_size(290.0).show(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| self.properties(ui));
            });
        }

        let req = self.core.session.lock().unwrap_or_else(std::sync::PoisonError::into_inner).ui_request.take();
        if let Some(q) = req {
            if let Some(b) = q["select_body"].as_str() {
                self.sel_body = Some(b.to_string());
                self.sel_load = None;
                self.form = Form::None;
            }
            if let Some(id) = q["select_load"].as_u64() {
                self.sel_load = Some(id as u32);
                self.sel_body = None;
                self.form = Form::None;
            }
            if let Some(m) = q["edit_material"].as_str() {
                self.f.material = m.to_string();
                let lib = self.core.session.lock().unwrap_or_else(std::sync::PoisonError::into_inner).model.materials();
                if let Some(mat) = lib.iter().find(|x| x.name == m) {
                    let full = serde_json::to_value(mat).unwrap_or(json!({}));
                    let mut v = json!({"name": format!("{} (custom)", mat.name)});
                    for k in theme::MAT_KEYS {
                        if let Some(x) = full.get(*k) {
                            v[*k] = x.clone();
                        } else if k.ends_with("_table") {
                            v[*k] = json!([]);
                        }
                    }
                    self.mat_edit = Some((mat.name.clone(), v));
                }
            }
            if let Some(t) = q["help"].as_str() {
                self.open_help(t);
            }
            if let Some(d) = q.get("pcb_dc") {
                let path = self.core.session.lock().unwrap_or_else(std::sync::PoisonError::into_inner).pcb_path();
                if let Some(path) = path {
                    self.open_pcb_dc(&ctx, &path);
                    for (k, slot) in [("net", &mut self.pcb.net), ("from", &mut self.pcb.from), ("to", &mut self.pcb.to)] {
                        if let Some(v) = d[k].as_str() {
                            *slot = v.to_string();
                        }
                    }
                    if let Some(c) = d["current"].as_f64() {
                        self.pcb.current = c;
                    }
                    if d["calculate"].as_bool() == Some(true) {
                        self.start_pcb_job();
                    }
                }
            }
            if let Some(t) = q["help_search"].as_str() {
                self.open_drawer(DrawerTab::Help);
                self.help.query = t.to_string();
            }
            if q["show_chart"].as_bool() == Some(true) {
                self.open_drawer(DrawerTab::Charts);
            }
            if let Some(n) = q["probe_node"].as_u64() {
                let rid = self.core.session.lock().unwrap_or_else(std::sync::PoisonError::into_inner).results.as_ref().map(|r| Arc::as_ptr(r) as u64).unwrap_or(0);
                self.probes.push((n as u32, rid));
                self.probe_mode = true;
            }
            if let Some(t) = q["tab"].as_str() {
                self.tab = match t {
                    "results" => Tab::Results,
                    "material" => Tab::Material,
                    "solve" => Tab::Solve,
                    "help" => Tab::Help,
                    _ => self.tab,
                };
            }
        }

        self.workbench_ui(ui);
        egui::CentralPanel::default().frame(egui::Frame::new()).show(ui, |ui| self.viewport(ui));

        self.mcp_dialog(&ctx);
        self.safety_ui(&ctx);
        if ctx.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::F1)) {
            self.context_help();
        }
        self.material_window(&ctx);
        if !ctx.egui_wants_keyboard_input() {
            let (undo, redo, key_fit, key_persp, key_del, key_esc, key_hide, key_show_all, view_key) = ctx.input_mut(|i| {
                let redo = i.consume_shortcut(&egui::KeyboardShortcut::new(egui::Modifiers::COMMAND | egui::Modifiers::SHIFT, egui::Key::Z)) || i.consume_shortcut(&egui::KeyboardShortcut::new(egui::Modifiers::COMMAND, egui::Key::Y));
                let undo = i.consume_shortcut(&egui::KeyboardShortcut::new(egui::Modifiers::COMMAND, egui::Key::Z));
                let fit = i.consume_key(egui::Modifiers::NONE, egui::Key::F);
                let persp = i.consume_key(egui::Modifiers::NONE, egui::Key::P);
                let del = i.consume_key(egui::Modifiers::NONE, egui::Key::Delete) || i.consume_key(egui::Modifiers::NONE, egui::Key::Backspace);
                let esc = i.consume_key(egui::Modifiers::NONE, egui::Key::Escape);
                let show_all = i.consume_key(egui::Modifiers::SHIFT, egui::Key::H);
                let hide = i.consume_key(egui::Modifiers::NONE, egui::Key::H);
                let mut vk = None;
                for (k, v) in [
                    (egui::Key::Num1, "iso"),
                    (egui::Key::Num2, "front"),
                    (egui::Key::Num3, "back"),
                    (egui::Key::Num4, "top"),
                    (egui::Key::Num5, "right"),
                    (egui::Key::Num6, "left"),
                ] {
                    if i.consume_key(egui::Modifiers::NONE, k) {
                        vk = Some(v);
                    }
                }
                (undo, redo, fit, persp, del, esc, hide, show_all, vk)
            });
            if undo {
                self.run("undo", json!({}));
            }
            if redo {
                self.run("redo", json!({}));
            }
            if key_fit {
                self.need_fit = true;
            }
            if let Some(v) = view_key {
                self.cam.set_view(v);
                self.need_fit = true;
            }
            if key_persp {
                let next = !self.core.session.lock().unwrap_or_else(std::sync::PoisonError::into_inner).display.perspective;
                self.run("set_display", json!({"perspective": next}));
            }
            if key_show_all {
                self.show_all_bodies();
            } else if key_hide
                && let Some(b) = self.sel_body.clone() {
                    self.set_body_vis(&b, 2);
                }
            if key_del {
                if let Some(id) = self.sel_load.take() {
                    self.edit_target = None;
                    self.run("remove_load", json!({"id": id}));
                } else if !self.sel_bodies.is_empty() {
                    let names: Vec<String> = self.sel_bodies.drain().collect();
                    self.sel_body = None;
                    self.edit_target = None;
                    for n in names {
                        self.run("remove_body", json!({"name": n}));
                    }
                } else if let Some(b) = self.sel_body.take() {
                    self.edit_target = None;
                    self.run("remove_body", json!({"name": b}));
                }
            }
            if key_esc {
                self.ctx_menu_pos = None;
                self.probe_mode = false;
                self.path.picking = 0;
                if self.form != Form::None {
                    self.form = Form::None;
                } else if self.sel_body.is_some() || self.sel_load.is_some() {
                    self.sel_body = None;
                    self.sel_load = None;
                    self.edit_target = None;
                } else if !self.selected.is_empty() || !self.sel_bodies.is_empty() {
                    self.selected.clear();
                    self.sel_bodies.clear();
                    self.sel_version += 1;
                }
            }
        }
        self.fatigue_window(&ctx);

        if let Some((msg, err, ttl)) = &mut self.toast {
            *ttl -= (ui.input(|i| i.stable_dt) as f64).min(0.1);
            let (stripe, msg, err) = (if *err { p.err } else { p.ok }, msg.clone(), *err);
            egui::Area::new(egui::Id::NULL.with("toast")).anchor(Align2::CENTER_BOTTOM, vec2(0.0, -44.0)).show(&ctx, |ui| {
                egui::Frame::new().fill(p.card).stroke(Stroke::new(1.0, p.border)).corner_radius(10).inner_margin(egui::Margin { left: 14, right: 14, top: 9, bottom: 9 }).shadow(egui::epaint::Shadow { offset: [0, 4], blur: 16, spread: 0, color: Color32::from_black_alpha(50) }).show(ui, |ui| {
                    ui.set_max_width(620.0);
                    ui.horizontal(|ui| {
                        ui.label(RichText::new(if err { "✖" } else { "✔" }).color(stripe).strong());
                        ui.label(RichText::new(msg).color(p.text));
                    });
                });
            });
            ui.ctx().request_repaint_after(std::time::Duration::from_millis(100));
            if *ttl <= 0.0 {
                self.toast = None;
            }
        }
    }

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        if let Some(gl_vp) = &self.gl_vp {
            gl_vp.destroy();
        }
    }
}

pub fn run(core: Shared, open: Option<String>) -> eframe::Result<()> {
    let bridge = match crate::bridge::serve(core.clone()) {
        Ok(s) => s,
        Err(e) => e,
    };
    if let Some(p) = open {
        let handoff = std::env::var_os("SIMCRAFT_HANDOFF_FILE").is_some_and(|h| std::path::Path::new(&p) == std::path::Path::new(&h));
        let _ = core.call("open_project", &json!({"path": p, "handoff": handoff}));
    }
    let opts = eframe::NativeOptions {
        viewport: {
            let mut v = egui::ViewportBuilder::default().with_title("SimCraft").with_app_id("simcraft").with_inner_size([1440.0, 880.0]).with_min_inner_size([960.0, 600.0]);
            if let Ok(icon) = eframe::icon_data::from_png_bytes(include_bytes!("../../assets/simcraft-256.png")) {
                v = v.with_icon(Arc::new(icon));
            }
            v
        },
        ..Default::default()
    };
    eframe::run_native("SimCraft", opts, Box::new(move |cc| Ok(Box::new(SimApp::new(cc, core, bridge)))))
}
