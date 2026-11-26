//! Project model: geometry, materials, analysis settings, supports and loads.

use crate::geometry::{Axis, Body, V3};
use crate::material::{self, Material};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Default, Hash)]
#[serde(rename_all = "snake_case")]
pub enum AnalysisType {
    #[default]
    StaticStructural,
    Modal,
    LinearBuckling,
    TransientStructural,
    NonlinearStructural,
    HarmonicResponse,
    ResponseSpectrum,
    RandomVibration,
    SteadyThermal,
    TransientThermal,
    ThermalStress,
    Electrostatic,
    CurrentFlow,
    ThermalElectric,
    Magnetostatic,
    FullWaveEm,
    MicrowaveHeating,
    Fluid,
    ConjugateHeatTransfer,
}

impl AnalysisType {
    pub fn label(self) -> &'static str {
        use AnalysisType::*;
        match self {
            StaticStructural => "Static Structural",
            Modal => "Modal",
            LinearBuckling => "Linear Buckling",
            TransientStructural => "Transient Structural",
            NonlinearStructural => "Nonlinear Structural (Plasticity)",
            HarmonicResponse => "Harmonic Response",
            ResponseSpectrum => "Response Spectrum",
            RandomVibration => "Random Vibration (PSD)",
            SteadyThermal => "Steady-State Thermal",
            TransientThermal => "Transient Thermal",
            ThermalStress => "Thermal-Stress",
            Electrostatic => "Electrostatic",
            CurrentFlow => "DC Current Flow",
            ThermalElectric => "Thermal-Electric (Joule Heating)",
            Magnetostatic => "Magnetostatic",
            FullWaveEm => "Full-Wave EM (FDTD)",
            MicrowaveHeating => "Microwave Heating",
            Fluid => "Fluid Flow (CFD)",
            ConjugateHeatTransfer => "Conjugate Heat Transfer (CFD + thermal)",
        }
    }
    pub fn group(self) -> &'static str {
        use AnalysisType::*;
        match self {
            StaticStructural | Modal | LinearBuckling | TransientStructural | NonlinearStructural | HarmonicResponse | ResponseSpectrum | RandomVibration => "Structural",
            SteadyThermal | TransientThermal | ThermalStress => "Thermal",
            Electrostatic | CurrentFlow | ThermalElectric | Magnetostatic | FullWaveEm | MicrowaveHeating => "Electromagnetic",
            Fluid | ConjugateHeatTransfer => "Fluids",
        }
    }
    pub fn all() -> [AnalysisType; 19] {
        use AnalysisType::*;
        [StaticStructural, Modal, LinearBuckling, TransientStructural, NonlinearStructural, HarmonicResponse, ResponseSpectrum, RandomVibration, SteadyThermal, TransientThermal, ThermalStress, Electrostatic, CurrentFlow, ThermalElectric, Magnetostatic, FullWaveEm, MicrowaveHeating, Fluid, ConjugateHeatTransfer]
    }
    pub fn key(self) -> String {
        serde_json::to_value(self).unwrap().as_str().unwrap().to_string()
    }
    /// Uses structural loads (forces, supports).
    pub fn structural(self) -> bool {
        use AnalysisType::*;
        matches!(self, StaticStructural | Modal | LinearBuckling | TransientStructural | NonlinearStructural | HarmonicResponse | ResponseSpectrum | RandomVibration | ThermalStress)
    }
    /// Uses thermal loads.
    pub fn thermal(self) -> bool {
        use AnalysisType::*;
        matches!(self, SteadyThermal | TransientThermal | ThermalStress | ThermalElectric | MicrowaveHeating | ConjugateHeatTransfer)
    }
    pub fn electric(self) -> bool {
        use AnalysisType::*;
        matches!(self, Electrostatic | CurrentFlow | ThermalElectric)
    }
    /// Meshes the enclosure (air / fluid box) when one exists.
    pub fn uses_enclosure(self) -> bool {
        matches!(self, AnalysisType::Electrostatic | AnalysisType::Magnetostatic | AnalysisType::Fluid | AnalysisType::ConjugateHeatTransfer | AnalysisType::FullWaveEm | AnalysisType::MicrowaveHeating)
    }
    pub fn requires_enclosure(self) -> bool {
        matches!(self, AnalysisType::Magnetostatic | AnalysisType::Fluid | AnalysisType::ConjugateHeatTransfer | AnalysisType::FullWaveEm | AnalysisType::MicrowaveHeating)
    }
    pub fn transient(self) -> bool {
        matches!(self, AnalysisType::TransientStructural | AnalysisType::TransientThermal | AnalysisType::MicrowaveHeating)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AnalysisSettings {
    #[serde(rename = "type", default)]
    pub kind: AnalysisType,
    /// Number of modes (modal) or buckling modes.
    #[serde(default = "six")]
    pub modes: usize,
    /// Stress-free reference temperature [°C].
    #[serde(default = "ref_t")]
    pub reference_temperature: f64,
    /// Transient end time [s].
    #[serde(default = "end_t")]
    pub end_time: f64,
    /// Number of time steps for transient analyses.
    #[serde(default = "steps")]
    pub time_steps: usize,
    /// Initial temperature for transient thermal [°C].
    #[serde(default = "ref_t")]
    pub initial_temperature: f64,
    /// Damping ratio for transient structural (fraction of critical).
    #[serde(default = "damp")]
    pub damping_ratio: f64,
    /// Load steps for nonlinear analysis.
    #[serde(default = "ten")]
    pub load_steps: usize,
    /// Harmonic response: frequency range [Hz] (None = automatic around the modes) and points.
    #[serde(default)]
    pub freq_min: Option<f64>,
    #[serde(default)]
    pub freq_max: Option<f64>,
    #[serde(default = "freq_steps")]
    pub freq_steps: usize,
    /// Response spectrum [Hz, acceleration in g] or random-vibration PSD [Hz, g²/Hz].
    #[serde(default)]
    pub table: Vec<[f64; 2]>,
    /// Base-excitation direction for spectrum / random analyses.
    #[serde(default)]
    pub direction: Axis,
    /// Modal combination for response spectrum: "srss" or "cqc".
    #[serde(default = "srss")]
    pub combination: String,
    /// Maximum pseudo-time iterations for CFD.
    #[serde(default = "cfd_it")]
    pub max_iterations: usize,
    /// Static structural: move surface nodes onto the true (curved / off-grid) surfaces.
    #[serde(default = "yes")]
    pub boundary_fit: bool,
    /// CFD turbulence model: "laminar" or "lvel" (algebraic wall-distance model for electronics).
    #[serde(default = "laminar")]
    pub turbulence: String,
    /// CFD scheme: "auto" (SIMPLEC, projection for flows that do not settle), "simple", "projection".
    #[serde(default = "auto")]
    pub cfd_scheme: String,
    /// Full-wave EM: excitation frequency [Hz].
    #[serde(default = "mw_freq")]
    pub frequency_hz: f64,
    /// Full-wave EM outer boundary: "pec" (metal cavity walls) or "open" (absorbing layer).
    #[serde(default = "pec")]
    pub em_boundary: String,
    /// Full-wave EM: maximum number of RF periods to reach steady state.
    #[serde(default = "max_periods")]
    pub max_periods: usize,
    /// Where linear solves run: auto (GPU for large iterative solves), cpu or gpu.
    #[serde(default)]
    pub compute: crate::gpu::ComputeMode,
}
fn six() -> usize {
    6
}
fn ten() -> usize {
    10
}
fn ref_t() -> f64 {
    22.0
}
fn end_t() -> f64 {
    1.0
}
fn steps() -> usize {
    50
}
fn damp() -> f64 {
    0.02
}
fn freq_steps() -> usize {
    300
}
fn yes() -> bool {
    true
}
fn auto() -> String {
    "auto".into()
}
fn laminar() -> String {
    "laminar".into()
}
fn mw_freq() -> f64 {
    2.45e9
}
fn pec() -> String {
    "pec".into()
}
fn max_periods() -> usize {
    400
}
fn te10() -> String {
    "te10".into()
}
fn srss() -> String {
    "srss".into()
}
fn cfd_it() -> usize {
    3000
}
impl Default for AnalysisSettings {
    fn default() -> Self {
        Self {
            kind: AnalysisType::default(),
            modes: 6,
            reference_temperature: 22.0,
            end_time: 1.0,
            time_steps: 50,
            initial_temperature: 22.0,
            damping_ratio: 0.02,
            load_steps: 10,
            max_iterations: 3000,
            compute: Default::default(),
            freq_min: None,
            freq_max: None,
            freq_steps: 300,
            table: vec![],
            direction: Axis::Z,
            combination: "srss".into(),
            turbulence: "laminar".into(),
            cfd_scheme: "auto".into(),
            boundary_fit: true,
            frequency_hz: 2.45e9,
            em_boundary: "pec".into(),
            max_periods: 400,
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum CoilMode {
    /// Current flows along the coil axis (straight conductor / bus bar).
    Axial,
    /// Current circulates around the axis (solenoid / winding).
    #[default]
    Circumferential,
}

/// Body-to-body structural interface behaviour (default when omitted: bonded / merged nodes).
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ContactKind {
    #[default]
    Bonded,
    NoSeparation,
    Frictionless,
    Frictional,
}

impl ContactKind {
    pub fn label(self) -> &'static str {
        match self {
            ContactKind::Bonded => "Bonded",
            ContactKind::NoSeparation => "No Separation",
            ContactKind::Frictionless => "Frictionless",
            ContactKind::Frictional => "Frictional",
        }
    }
    pub fn splits_mesh(self) -> bool {
        !matches!(self, ContactKind::Bonded)
    }
}

/// Pairwise contact definition between two bodies (or subsets of their faces).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ContactPair {
    pub id: u32,
    #[serde(default)]
    pub body_a: String,
    #[serde(default)]
    pub body_b: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub faces_a: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub faces_b: Vec<String>,
    #[serde(default)]
    pub kind: ContactKind,
    /// Coulomb friction coefficient μ ≥ 0 (used when `kind == Frictional`).
    #[serde(default = "default_friction")]
    pub friction: f64,
    /// Normal offset [mm] subtracted from geometric gap (>0 = interference / shrink fit; <0 = initial clearance).
    #[serde(default)]
    pub offset_mm: f64,
}
fn default_friction() -> f64 {
    0.2
}

impl ContactPair {
    pub fn summary(&self) -> String {
        let a = if !self.faces_a.is_empty() { self.faces_a.join(", ") } else { self.body_a.clone() };
        let b = if !self.faces_b.is_empty() { self.faces_b.join(", ") } else { self.body_b.clone() };
        let extra = match self.kind {
            ContactKind::Frictional => format!(", μ={}", self.friction),
            _ => String::new(),
        };
        let off = if self.offset_mm != 0.0 { format!(", offset={} mm", self.offset_mm) } else { String::new() };
        format!("{} ↔ {} ({}{}{})", a, b, self.kind.label(), extra, off)
    }
}

/// Units: lengths/displacements mm, forces N, pressure MPa, temperature °C,
/// heat flux W/m², film coefficient W/(m² K), heat generation W/m³, acceleration m/s²,
/// voltage V, current A, current density A/mm², remanence T, velocity m/s, fluid pressure Pa.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Load {
    FixedSupport { faces: Vec<String> },
    Displacement {
        faces: Vec<String>,
        #[serde(default)]
        x: Option<f64>,
        #[serde(default)]
        y: Option<f64>,
        #[serde(default)]
        z: Option<f64>,
    },
    /// Frictionless contact with a rigid surface: blocks motion into it, allows lift-off.
    CompressionOnly { faces: Vec<String> },
    /// Structural: zero normal displacement. CFD: free-slip wall. Thermal: adiabatic.
    Symmetry { faces: Vec<String> },
    Force { faces: Vec<String>, vector: V3 },
    Pressure { faces: Vec<String>, value_mpa: f64 },
    Gravity {
        #[serde(default = "g")]
        acceleration: V3,
    },
    Temperature { faces: Vec<String>, value: f64 },
    HeatFlux { faces: Vec<String>, value: f64 },
    Convection { faces: Vec<String>, coefficient: f64, ambient: f64 },
    /// Thermal radiation from the faces: to the surroundings at `ambient` °C, or — with
    /// `enclosure` — between all enclosure faces (view factors), the rest of their view seeing the
    /// ambient.
    Radiation {
        faces: Vec<String>,
        emissivity: f64,
        ambient: f64,
        #[serde(default)]
        enclosure: bool,
    },
    /// Internal heat [W/m³] in one body, or in the whole model when `body` is omitted. With
    /// `power` [W] the total is spread over the body's cells (exact on any mesh) and `value` is unused.
    HeatGeneration {
        value: f64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        body: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        power: Option<f64>,
    },
    Voltage { faces: Vec<String>, value: f64 },
    /// Total current [A] entering the body through the faces.
    Current { faces: Vec<String>, value: f64 },
    /// Current density [A/mm²] in a body (a coil or conductor).
    Coil {
        body: String,
        current_density: f64,
        #[serde(default)]
        axis: Axis,
        #[serde(default)]
        mode: CoilMode,
        /// Point on the coil axis [mm]; default = body centre.
        #[serde(default)]
        center: Option<V3>,
    },
    /// Permanent magnet: remanent flux density vector [T].
    Magnet { body: String, remanence: V3 },
    /// Fluid inlet: uniform velocity [m/s] flowing into the domain.
    Inlet {
        faces: Vec<String>,
        velocity: f64,
        /// Inflow temperature [°C] for conjugate heat transfer (default: the reference temperature).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        temperature: Option<f64>,
    },
    /// No-slip wall moving tangentially with `velocity` [m/s] (lid-driven cavities, belts).
    MovingWall { faces: Vec<String>, velocity: V3 },
    /// Fluid outlet at a fixed static pressure [Pa].
    Outlet {
        faces: Vec<String>,
        #[serde(default)]
        pressure: f64,
    },
    /// RF feed (full-wave EM): a current sheet on the faces driving the electric field along
    /// `polarization`, delivering `power` [W]. `profile` "te10" (half-sine across the face, like a
    /// rectangular waveguide feed) or "uniform".
    Port {
        faces: Vec<String>,
        power: f64,
        polarization: Axis,
        #[serde(default = "te10")]
        profile: String,
    },
    /// Bolt pretension across an interior cut plane of `body`. Specify either `force_n` [N]
    /// (Step 1 target preload, then locked for Step 2) or `adjustment_mm` [mm] (shortening Δ).
    BoltPretension {
        body: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        force_n: Option<f64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        adjustment_mm: Option<f64>,
        /// Cut normal axis (None = auto: cylinder axis or longest bounding-box axis of `body`).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        axis: Option<Axis>,
        /// Cut coordinate along `axis` [mm] (None = body midpoint).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cut_mm: Option<f64>,
    },
    /// Elastic spring connecting `faces` to ground (when `to_faces` is None) or to `to_faces`.
    Spring {
        faces: Vec<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        to_faces: Option<Vec<String>>,
        stiffness_n_per_mm: f64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        anchor: Option<V3>,
        #[serde(default)]
        preload_n: f64,
    },
    /// Remote force [N] and/or moment [N·m] applied at `point` [mm] via RBE3 weighted distribution on `faces`.
    RemoteForce {
        faces: Vec<String>,
        point: V3,
        #[serde(default)]
        force: V3,
        #[serde(default)]
        moment: V3,
    },
    /// Remote point constraint/displacement at `point` [mm] coupled to `faces` (rigid RBE2 or distributing RBE3).
    /// Translations in mm, rotations in rad.
    RemoteDisplacement {
        faces: Vec<String>,
        point: V3,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        x: Option<f64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        y: Option<f64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        z: Option<f64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        rx: Option<f64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        ry: Option<f64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        rz: Option<f64>,
        #[serde(default)]
        rigid: bool,
    },
    /// Concentrated mass [kg] attached to `faces` (at `point` [mm] or face centroid). Participates in Gravity and dynamic/modal analyses.
    PointMass {
        faces: Vec<String>,
        mass_kg: f64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        point: Option<V3>,
    },
}
fn g() -> V3 {
    [0.0, 0.0, -9.80665]
}

impl Load {
    pub fn label(&self) -> &'static str {
        match self {
            Load::FixedSupport { .. } => "Fixed Support",
            Load::Displacement { .. } => "Displacement",
            Load::CompressionOnly { .. } => "Compression-Only Support",
            Load::Symmetry { .. } => "Symmetry",
            Load::Force { .. } => "Force",
            Load::Pressure { .. } => "Pressure",
            Load::Gravity { .. } => "Standard Earth Gravity",
            Load::Temperature { .. } => "Temperature",
            Load::HeatFlux { .. } => "Heat Flux",
            Load::Convection { .. } => "Convection",
            Load::Radiation { enclosure: true, .. } => "Radiation (enclosure)",
            Load::Radiation { .. } => "Radiation",
            Load::HeatGeneration { .. } => "Internal Heat Generation",
            Load::Voltage { .. } => "Voltage",
            Load::Current { .. } => "Current",
            Load::Coil { .. } => "Coil Current",
            Load::Magnet { .. } => "Permanent Magnet",
            Load::Inlet { .. } => "Velocity Inlet",
            Load::Outlet { .. } => "Pressure Outlet",
            Load::Port { .. } => "RF Port",
            Load::MovingWall { .. } => "Moving Wall",
            Load::BoltPretension { .. } => "Bolt Pretension",
            Load::Spring { .. } => "Elastic Spring",
            Load::RemoteForce { .. } => "Remote Force",
            Load::RemoteDisplacement { .. } => "Remote Displacement",
            Load::PointMass { .. } => "Point Mass",
        }
    }
    pub fn faces(&self) -> &[String] {
        match self {
            Load::FixedSupport { faces }
            | Load::Displacement { faces, .. }
            | Load::CompressionOnly { faces }
            | Load::Symmetry { faces }
            | Load::Force { faces, .. }
            | Load::Pressure { faces, .. }
            | Load::Temperature { faces, .. }
            | Load::HeatFlux { faces, .. }
            | Load::Convection { faces, .. }
            | Load::Radiation { faces, .. }
            | Load::Voltage { faces, .. }
            | Load::Current { faces, .. }
            | Load::Inlet { faces, .. }
            | Load::Outlet { faces, .. }
            | Load::Port { faces, .. }
            | Load::MovingWall { faces, .. }
            | Load::Spring { faces, .. }
            | Load::RemoteForce { faces, .. }
            | Load::RemoteDisplacement { faces, .. }
            | Load::PointMass { faces, .. } => faces,
            _ => &[],
        }
    }
    pub fn body(&self) -> Option<&str> {
        match self {
            Load::Coil { body, .. } | Load::Magnet { body, .. } | Load::BoltPretension { body, .. } => Some(body),
            Load::HeatGeneration { body: Some(b), .. } => Some(b),
            _ => None,
        }
    }
    pub fn is_structural(&self) -> bool {
        matches!(
            self,
            Load::FixedSupport { .. }
                | Load::Displacement { .. }
                | Load::CompressionOnly { .. }
                | Load::Symmetry { .. }
                | Load::Force { .. }
                | Load::Pressure { .. }
                | Load::Gravity { .. }
                | Load::BoltPretension { .. }
                | Load::Spring { .. }
                | Load::RemoteForce { .. }
                | Load::RemoteDisplacement { .. }
                | Load::PointMass { .. }
        )
    }
    pub fn is_support(&self) -> bool {
        matches!(
            self,
            Load::FixedSupport { .. }
                | Load::Displacement { .. }
                | Load::CompressionOnly { .. }
                | Load::Symmetry { .. }
                | Load::RemoteDisplacement { .. }
        )
    }
    pub fn is_thermal(&self) -> bool {
        matches!(self, Load::Temperature { .. } | Load::HeatFlux { .. } | Load::Convection { .. } | Load::Radiation { .. } | Load::HeatGeneration { .. })
    }
    /// Whether this load takes part in the given analysis.
    pub fn applies_to(&self, kind: AnalysisType) -> bool {
        use AnalysisType::*;
        match kind {
            StaticStructural | NonlinearStructural | TransientStructural | LinearBuckling => self.is_structural(),
            Modal | ResponseSpectrum | RandomVibration => {
                self.is_support() || matches!(self, Load::Spring { .. } | Load::PointMass { .. } | Load::BoltPretension { .. })
            }
            HarmonicResponse => self.is_structural() && !matches!(self, Load::Gravity { .. }),
            SteadyThermal | TransientThermal => self.is_thermal(),
            ThermalStress => self.is_structural() || self.is_thermal(),
            Electrostatic => matches!(self, Load::Voltage { .. }),
            CurrentFlow => matches!(self, Load::Voltage { .. } | Load::Current { .. }),
            ThermalElectric => matches!(self, Load::Voltage { .. } | Load::Current { .. }) || self.is_thermal(),
            Magnetostatic => matches!(self, Load::Coil { .. } | Load::Magnet { .. }),
            FullWaveEm => matches!(self, Load::Port { .. }),
            MicrowaveHeating => matches!(self, Load::Port { .. }) || self.is_thermal(),
            Fluid => matches!(self, Load::Inlet { .. } | Load::Outlet { .. } | Load::Symmetry { .. } | Load::MovingWall { .. }),
            ConjugateHeatTransfer => matches!(self, Load::Inlet { .. } | Load::Outlet { .. } | Load::Symmetry { .. } | Load::MovingWall { .. } | Load::Gravity { .. }) || self.is_thermal(),
        }
    }
    pub fn summary(&self) -> String {
        let f = |v: &[String]| v.join(", ");
        match self {
            Load::FixedSupport { faces } | Load::CompressionOnly { faces } | Load::Symmetry { faces } => format!("on {}", f(faces)),
            Load::Displacement { faces, x, y, z } => {
                let c = |o: &Option<f64>| o.map(|v| format!("{v}")).unwrap_or("free".into());
                format!("({}, {}, {}) mm on {}", c(x), c(y), c(z), f(faces))
            }
            Load::Force { faces, vector } => format!("({}, {}, {}) N on {}", vector[0], vector[1], vector[2], f(faces)),
            Load::Pressure { faces, value_mpa } => format!("{value_mpa} MPa on {}", f(faces)),
            Load::Gravity { acceleration: a } => format!("({}, {}, {}) m/s²", a[0], a[1], a[2]),
            Load::Temperature { faces, value } => format!("{value} °C on {}", f(faces)),
            Load::HeatFlux { faces, value } => format!("{value} W/m² on {}", f(faces)),
            Load::Convection { faces, coefficient, ambient } => format!("h={coefficient} W/m²K, T∞={ambient} °C on {}", f(faces)),
            Load::Radiation { faces, emissivity, ambient, enclosure } => format!("ε={emissivity}, T∞={ambient} °C{} on {}", if *enclosure { ", surface-to-surface" } else { "" }, f(faces)),
            Load::HeatGeneration { value, body, power } => match (power, body) {
                (Some(p), Some(b)) => format!("{p} W in {b}"),
                (Some(p), None) => format!("{p} W in all solids"),
                (None, Some(b)) => format!("{value} W/m³ in {b}"),
                (None, None) => format!("{value} W/m³"),
            },
            Load::Voltage { faces, value } => format!("{value} V on {}", f(faces)),
            Load::Current { faces, value } => format!("{value} A into {}", f(faces)),
            Load::Coil { body, current_density, axis, mode, .. } => {
                format!("{current_density} A/mm² in {body} ({:?} about {:?})", mode, axis)
            }
            Load::Magnet { body, remanence: r } => format!("Br = ({}, {}, {}) T in {body}", r[0], r[1], r[2]),
            Load::Inlet { faces, velocity, temperature } => match temperature {
                Some(t) => format!("{velocity} m/s at {t} °C into {}", f(faces)),
                None => format!("{velocity} m/s into {}", f(faces)),
            },
            Load::MovingWall { faces, velocity: v } => format!("({}, {}, {}) m/s on {}", v[0], v[1], v[2], f(faces)),
            Load::Outlet { faces, pressure } => format!("{pressure} Pa on {}", f(faces)),
            Load::Port { faces, power, polarization, profile } => format!("{power} W, E along {:?} ({profile}) on {}", polarization, f(faces)),
            Load::BoltPretension { body, force_n, adjustment_mm, axis, .. } => {
                let ax = axis.map(|a| format!(" ({a:?})")).unwrap_or_default();
                match (force_n, adjustment_mm) {
                    (Some(fn_), _) => format!("{fn_} N preload on {body}{ax}"),
                    (None, Some(dz)) => format!("{dz} mm adjustment on {body}{ax}"),
                    (None, None) => format!("on {body}{ax}"),
                }
            }
            Load::Spring { faces, to_faces, stiffness_n_per_mm, preload_n, .. } => {
                let target = to_faces.as_ref().map(|v| f(v)).unwrap_or_else(|| "ground".into());
                let pre = if *preload_n != 0.0 { format!(", preload {preload_n} N") } else { String::new() };
                format!("k={stiffness_n_per_mm} N/mm{} ({} ↔ {target})", pre, f(faces))
            }
            Load::RemoteForce { faces, point, force, moment } => {
                format!("F=({}, {}, {}) N, M=({}, {}, {}) N·m at ({}, {}, {}) mm on {}", force[0], force[1], force[2], moment[0], moment[1], moment[2], point[0], point[1], point[2], f(faces))
            }
            Load::RemoteDisplacement { faces, point, x, y, z, rx, ry, rz, rigid } => {
                let c = |o: &Option<f64>| o.map(|v| format!("{v}")).unwrap_or("free".into());
                let kind = if *rigid { "rigid" } else { "distributing" };
                format!("({kind}) u=({},{},{}) mm θ=({},{},{}) rad at ({},{},{}) on {}", c(x), c(y), c(z), c(rx), c(ry), c(rz), point[0], point[1], point[2], f(faces))
            }
            Load::PointMass { faces, mass_kg, point } => match point {
                Some(p) => format!("{mass_kg} kg at ({}, {}, {}) mm on {}", p[0], p[1], p[2], f(faces)),
                None => format!("{mass_kg} kg on {}", f(faces)),
            },
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LoadItem {
    pub id: u32,
    #[serde(flatten)]
    pub load: Load,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Model {
    #[serde(default)]
    pub bodies: Vec<Body>,
    #[serde(default)]
    pub custom_materials: Vec<Material>,
    #[serde(default)]
    pub analysis: AnalysisSettings,
    #[serde(default)]
    pub loads: Vec<LoadItem>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub contacts: Vec<ContactPair>,
    /// Element size [mm]; None = automatic.
    #[serde(default)]
    pub element_size: Option<f64>,
    #[serde(default)]
    pub next_id: u32,
}

impl Default for Model {
    fn default() -> Self {
        Model { bodies: vec![], custom_materials: vec![], analysis: AnalysisSettings::default(), loads: vec![], contacts: vec![], element_size: None, next_id: 1 }
    }
}

impl Model {
    pub fn materials(&self) -> Vec<Material> {
        let mut lib = material::library();
        for m in &self.custom_materials {
            if let Some(e) = lib.iter_mut().find(|x| x.name == m.name) {
                *e = m.clone();
            } else {
                lib.push(m.clone());
            }
        }
        lib
    }
    pub fn has_enclosure(&self) -> bool {
        self.bodies.iter().any(|b| b.role == crate::geometry::Role::Enclosure)
    }
    /// Whether the current analysis uses a boundary-fitted mesh: linear structural, thermal,
    /// thermal-stress, modal and vibration analyses (not buckling, plasticity, fields, CFD, FDTD).
    /// Models with an imported PCB stay on plain voxels: its homogenised copper conductivity is
    /// already scaled to the voxel layer thickness.
    pub fn mesh_fitted(&self) -> bool {
        use AnalysisType::*;
        self.analysis.boundary_fit
            && !self.mesh_with_enclosure()
            && !self.bodies.iter().any(|b| matches!(b.shape, crate::geometry::Shape::Pcb { .. }))
            && matches!(self.analysis.kind, StaticStructural | ThermalStress | SteadyThermal | TransientThermal | Modal | HarmonicResponse | ResponseSpectrum | RandomVibration | TransientStructural)
    }
    /// Whether the mesh for the current analysis includes the enclosure region.
    pub fn mesh_with_enclosure(&self) -> bool {
        self.analysis.kind.uses_enclosure() && self.has_enclosure()
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PreflightStatus {
    Pass,
    Warn,
    Fail,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct PreflightItem {
    pub code: String,
    pub label: String,
    pub message: String,
    pub status: PreflightStatus,
    /// Suggested UI tab: "geometry" | "material" | "setup" | "mesh" | "solve".
    pub target_tab: String,
}

/// Live pre-solve readiness checklist checked against solver requirements.
pub fn preflight_check(model: &Model, mesh: Option<&crate::mesh::Mesh>, elem_size_mm: f64) -> Vec<PreflightItem> {
    use crate::geometry::{Op, Role};
    use AnalysisType::*;
    let mut out = Vec::new();
    let item = |code: &str, label: &str, message: String, status: PreflightStatus, target_tab: &str| PreflightItem {
        code: code.into(),
        label: label.into(),
        message,
        status,
        target_tab: target_tab.into(),
    };

    let solids: Vec<&crate::geometry::Body> = model.bodies.iter().filter(|b| b.role == Role::Solid && b.op == Op::Add).collect();
    if solids.is_empty() {
        out.push(item("geometry", "Needs Geometry", "Add at least one solid body (Box, Cylinder, Sphere, or STL).".into(), PreflightStatus::Fail, "geometry"));
    } else {
        out.push(item("geometry", "Geometry", format!("{} additive solid body(s) defined.", solids.len()), PreflightStatus::Pass, "geometry"));
    }

    let mats = model.materials();
    let resolved_mats: Vec<&Material> = solids.iter().filter_map(|b| mats.iter().find(|m| m.name == b.material)).collect();
    if solids.is_empty() {
        out.push(item("material", "Material", "Waiting for solid geometry.".into(), PreflightStatus::Warn, "material"));
    } else if resolved_mats.len() < solids.len() {
        out.push(item("material", "Unknown Material", "One or more bodies reference a missing material.".into(), PreflightStatus::Fail, "material"));
    } else {
        out.push(item("material", "Material", solids.iter().map(|b| b.material.as_str()).collect::<std::collections::HashSet<_>>().into_iter().collect::<Vec<_>>().join(", "), PreflightStatus::Pass, "material"));
    }

    let kind = model.analysis.kind;
    let enc = model.bodies.iter().find(|b| b.role == Role::Enclosure);
    if kind.requires_enclosure() {
        match enc {
            None => out.push(item("enclosure", "Needs Enclosure", format!("{} requires an Air/Fluid enclosure around the parts.", kind.label()), PreflightStatus::Fail, "geometry")),
            Some(eb) => {
                if matches!(kind, Fluid | ConjugateHeatTransfer) {
                    let is_fluid = mats.iter().find(|m| m.name == eb.material).is_some_and(|m| m.is_fluid && m.dynamic_viscosity > 0.0);
                    if !is_fluid {
                        out.push(item("enclosure", "Enclosure Fluid", format!("Enclosure '{}' uses '{}', which is not a fluid (use Air, Water, or Engine Oil).", eb.name, eb.material), PreflightStatus::Fail, "material"));
                    } else {
                        out.push(item("enclosure", "Fluid Enclosure", format!("{} ({})", eb.name, eb.material), PreflightStatus::Pass, "geometry"));
                    }
                } else {
                    out.push(item("enclosure", "Enclosure", format!("{} ({})", eb.name, eb.material), PreflightStatus::Pass, "geometry"));
                }
            }
        }
    }

    let has = |pred: &dyn Fn(&Load) -> bool| model.loads.iter().any(|l| pred(&l.load));
    let has_support = has(&|l| {
        matches!(
            l,
            Load::FixedSupport { .. }
                | Load::Displacement { .. }
                | Load::CompressionOnly { .. }
                | Load::Symmetry { .. }
                | Load::Spring { to_faces: None, .. }
        ) || matches!(
            l,
            Load::RemoteDisplacement { x, y, z, rx, ry, rz, .. }
                if [x, y, z, rx, ry, rz].iter().any(|c| c.is_some())
        )
    });
    let has_struct_load = has(&|l| {
        matches!(
            l,
            Load::Force { .. }
                | Load::Pressure { .. }
                | Load::Gravity { .. }
                | Load::BoltPretension { .. }
                | Load::RemoteForce { .. }
        ) || matches!(l, Load::Spring { preload_n, .. } if *preload_n != 0.0)
            || matches!(l, Load::Displacement { x, y, z, .. } if [x, y, z].iter().any(|c| c.is_some_and(|v| v != 0.0)))
            || matches!(
                l,
                Load::RemoteDisplacement { x, y, z, rx, ry, rz, .. }
                    if [x, y, z, rx, ry, rz].iter().any(|c| c.is_some_and(|v| v != 0.0))
            )
    });
    let has_thermal_bc = has(&|l| matches!(l, Load::Temperature { .. } | Load::Convection { .. } | Load::Radiation { .. }));
    let has_thermal_src = has(&|l| matches!(l, Load::HeatFlux { .. } | Load::HeatGeneration { .. } | Load::Temperature { .. }));

    match kind {
        StaticStructural | NonlinearStructural | LinearBuckling | TransientStructural => {
            if !has_support {
                out.push(item("support", "Needs Support", "Add a Fixed Support, Displacement, or Symmetry restraint.".into(), PreflightStatus::Fail, "setup"));
            } else {
                out.push(item("support", "Support", "Structural restraint defined.".into(), PreflightStatus::Pass, "setup"));
            }
            if !has_struct_load {
                out.push(item("excitation", "Needs Load", "Add a Force, Pressure, Gravity, or non-zero Displacement.".into(), PreflightStatus::Warn, "setup"));
            }
        }
        Modal => {
            if !has_support {
                out.push(item("support", "Needs Support", "Add a Fixed Support or Displacement to remove rigid-body modes.".into(), PreflightStatus::Fail, "setup"));
            } else {
                out.push(item("support", "Support", "Modal restraint defined.".into(), PreflightStatus::Pass, "setup"));
            }
        }
        HarmonicResponse => {
            if !has_support {
                out.push(item("support", "Needs Support", "Add a Fixed Support or Displacement restraint.".into(), PreflightStatus::Fail, "setup"));
            }
            if !has(&|l| matches!(l, Load::Force { .. } | Load::Pressure { .. })) {
                out.push(item("excitation", "Needs Force / Pressure", "Harmonic response needs at least one Force or Pressure amplitude.".into(), PreflightStatus::Fail, "setup"));
            } else if has_support {
                out.push(item("excitation", "Harmonic Load", "Support and harmonic excitation defined.".into(), PreflightStatus::Pass, "setup"));
            }
        }
        ResponseSpectrum | RandomVibration => {
            if !has_support {
                out.push(item("support", "Needs Support", "Add a Fixed Support or Displacement restraint.".into(), PreflightStatus::Fail, "setup"));
            }
            if model.analysis.table.is_empty() {
                out.push(item("excitation", "Needs Table", "Define the excitation table (Hz vs g or g²/Hz) in Setup/Solve.".into(), PreflightStatus::Fail, "setup"));
            } else if has_support {
                out.push(item("excitation", "Excitation Table", format!("{} table point(s).", model.analysis.table.len()), PreflightStatus::Pass, "setup"));
            }
        }
        SteadyThermal | TransientThermal => {
            if !has_thermal_bc {
                out.push(item("thermal_bc", "Needs Thermal BC", "Add a Temperature, Convection, or Radiation boundary condition.".into(), PreflightStatus::Fail, "setup"));
            } else if !has_thermal_src {
                out.push(item("thermal_bc", "Thermal Load", "Thermal boundary present; add Heat Flux / Heat Generation if needed.".into(), PreflightStatus::Warn, "setup"));
            } else {
                out.push(item("thermal_bc", "Thermal BCs", "Thermal source and sink defined.".into(), PreflightStatus::Pass, "setup"));
            }
        }
        ThermalStress => {
            if !has_support {
                out.push(item("support", "Needs Support", "Add a Fixed Support or Displacement restraint.".into(), PreflightStatus::Fail, "setup"));
            }
            if !has_thermal_bc {
                out.push(item("thermal_bc", "Needs Thermal BC", "Add a Temperature, Convection, or Radiation boundary condition.".into(), PreflightStatus::Fail, "setup"));
            } else if has_support {
                out.push(item("thermal_bc", "Thermal + Support", "Structural and thermal BCs defined.".into(), PreflightStatus::Pass, "setup"));
            }
        }
        Electrostatic => {
            if !has(&|l| matches!(l, Load::Voltage { .. })) {
                out.push(item("excitation", "Needs Voltage", "Add at least two Voltage boundaries (e.g. +V and 0 V ground).".into(), PreflightStatus::Fail, "setup"));
            } else {
                out.push(item("excitation", "Voltage BCs", "Voltage boundary defined.".into(), PreflightStatus::Pass, "setup"));
            }
        }
        CurrentFlow | ThermalElectric => {
            let has_v = has(&|l| matches!(l, Load::Voltage { .. }));
            let has_i = has(&|l| matches!(l, Load::Current { .. }));
            if !has_v {
                out.push(item("excitation", "Needs Voltage Ref", "Ground at least one face with a 0 V Voltage boundary.".into(), PreflightStatus::Fail, "setup"));
            } else if !has_i && model.loads.iter().filter(|l| matches!(l.load, Load::Voltage { .. })).count() < 2 {
                out.push(item("excitation", "Needs Current / 2nd V", "Add a Current load or second Voltage boundary.".into(), PreflightStatus::Warn, "setup"));
            } else {
                out.push(item("excitation", "Electric BCs", "Current path defined.".into(), PreflightStatus::Pass, "setup"));
            }
        }
        Magnetostatic => {
            if !has(&|l| matches!(l, Load::Coil { .. } | Load::Magnet { .. })) {
                out.push(item("excitation", "Needs Coil / Magnet", "Add a Coil or Permanent Magnet excitation.".into(), PreflightStatus::Fail, "setup"));
            } else {
                out.push(item("excitation", "Magnetic Source", "Coil or Magnet defined.".into(), PreflightStatus::Pass, "setup"));
            }
        }
        Fluid | ConjugateHeatTransfer => {
            let has_in = has(&|l| matches!(l, Load::Inlet { .. }));
            let has_out = has(&|l| matches!(l, Load::Outlet { .. }));
            let has_wall = has(&|l| matches!(l, Load::MovingWall { .. }));
            let has_grav = kind == ConjugateHeatTransfer && has(&|l| matches!(l, Load::Gravity { .. }));
            if !has_in && !has_wall && !has_grav {
                out.push(item("flow_bc", "Needs Flow Drive", "Add a Velocity Inlet (+ Pressure Outlet), Moving Wall, or Gravity (natural convection).".into(), PreflightStatus::Fail, "setup"));
            } else if has_in && !has_out {
                out.push(item("flow_bc", "Needs Outlet", "A Velocity Inlet requires a Pressure Outlet (e.g. 0 Pa).".into(), PreflightStatus::Fail, "setup"));
            } else if kind == ConjugateHeatTransfer && !has_in && !has_out && !has_thermal_bc {
                out.push(item("flow_bc", "Needs Heat Sink", "CHT needs an Inlet/Outlet, Temperature, or Convection boundary for heat to leave.".into(), PreflightStatus::Fail, "setup"));
            } else {
                out.push(item("flow_bc", "Flow BCs", "Inflow / boundary conditions defined.".into(), PreflightStatus::Pass, "setup"));
            }
        }
        FullWaveEm | MicrowaveHeating => {
            if !has(&|l| matches!(l, Load::Port { .. })) {
                out.push(item("excitation", "Needs RF Port", "Add an RF Port where electromagnetic power enters.".into(), PreflightStatus::Fail, "setup"));
            } else {
                out.push(item("excitation", "RF Port", "RF excitation port defined.".into(), PreflightStatus::Pass, "setup"));
            }
            if model.analysis.frequency_hz > 0.0 && elem_size_mm > 0.0 {
                let max_er = resolved_mats.iter().map(|m| m.relative_permittivity).fold(1.0f64, f64::max).max(1.0);
                let lambda_mm = (299_792_458.0 / model.analysis.frequency_hz) * 1000.0 / max_er.sqrt();
                let limit_mm = lambda_mm / 10.0;
                if elem_size_mm > limit_mm {
                    out.push(item(
                        "mesh_wavelength",
                        "Mesh > λ/10",
                        format!("Element size {elem_size_mm:.2} mm exceeds λ/10 ({limit_mm:.2} mm at {:.2} GHz, εr={max_er:.1}).", model.analysis.frequency_hz / 1e9),
                        PreflightStatus::Warn,
                        "mesh",
                    ));
                } else {
                    out.push(item("mesh_wavelength", "Mesh ≤ λ/10", format!("{elem_size_mm:.2} mm ≤ {limit_mm:.2} mm"), PreflightStatus::Pass, "mesh"));
                }
            }
        }
    }

    if let Some(m) = mesh {
        if !m.warnings.is_empty() {
            out.push(item("mesh", "Mesh Warning", m.warnings[0].clone(), PreflightStatus::Warn, "mesh"));
        } else {
            out.push(item("mesh", "Mesh Ready", format!("{} elements ({} mm)", m.elems.len(), m.h), PreflightStatus::Pass, "mesh"));
        }
    }
    out
}
