//! Project model: geometry, materials, analysis settings, supports and loads.

use crate::geometry::{Body, V3};
use crate::material::{self, Material};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum AnalysisType {
    #[default]
    StaticStructural,
    SteadyThermal,
    Modal,
    ThermalStress,
}
impl AnalysisType {
    pub fn label(self) -> &'static str {
        match self {
            AnalysisType::StaticStructural => "Static Structural",
            AnalysisType::SteadyThermal => "Steady-State Thermal",
            AnalysisType::Modal => "Modal",
            AnalysisType::ThermalStress => "Thermal-Stress",
        }
    }
    pub fn all() -> [AnalysisType; 4] {
        [Self::StaticStructural, Self::SteadyThermal, Self::Modal, Self::ThermalStress]
    }
    pub fn structural(self) -> bool {
        !matches!(self, AnalysisType::SteadyThermal)
    }
    pub fn thermal(self) -> bool {
        matches!(self, AnalysisType::SteadyThermal | AnalysisType::ThermalStress)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AnalysisSettings {
    #[serde(rename = "type", default)]
    pub kind: AnalysisType,
    /// Number of modes for modal analysis.
    #[serde(default = "six")]
    pub modes: usize,
    /// Stress-free reference temperature [°C].
    #[serde(default = "ref_t")]
    pub reference_temperature: f64,
}
fn six() -> usize {
    6
}
fn ref_t() -> f64 {
    22.0
}
impl Default for AnalysisSettings {
    fn default() -> Self {
        Self { kind: AnalysisType::default(), modes: 6, reference_temperature: 22.0 }
    }
}

/// Units: lengths/displacements mm, forces N, pressure MPa, temperature °C,
/// heat flux W/m², film coefficient W/(m² K), heat generation W/m³, acceleration m/s².
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
    Force { faces: Vec<String>, vector: V3 },
    Pressure { faces: Vec<String>, value_mpa: f64 },
    Gravity {
        #[serde(default = "g")]
        acceleration: V3,
    },
    Temperature { faces: Vec<String>, value: f64 },
    HeatFlux { faces: Vec<String>, value: f64 },
    Convection { faces: Vec<String>, coefficient: f64, ambient: f64 },
    HeatGeneration { value: f64 },
}
fn g() -> V3 {
    [0.0, 0.0, -9.80665]
}

impl Load {
    pub fn label(&self) -> &'static str {
        match self {
            Load::FixedSupport { .. } => "Fixed Support",
            Load::Displacement { .. } => "Displacement",
            Load::Force { .. } => "Force",
            Load::Pressure { .. } => "Pressure",
            Load::Gravity { .. } => "Standard Earth Gravity",
            Load::Temperature { .. } => "Temperature",
            Load::HeatFlux { .. } => "Heat Flux",
            Load::Convection { .. } => "Convection",
            Load::HeatGeneration { .. } => "Internal Heat Generation",
        }
    }
    pub fn faces(&self) -> &[String] {
        match self {
            Load::FixedSupport { faces }
            | Load::Displacement { faces, .. }
            | Load::Force { faces, .. }
            | Load::Pressure { faces, .. }
            | Load::Temperature { faces, .. }
            | Load::HeatFlux { faces, .. }
            | Load::Convection { faces, .. } => faces,
            _ => &[],
        }
    }
    pub fn is_structural(&self) -> bool {
        matches!(
            self,
            Load::FixedSupport { .. } | Load::Displacement { .. } | Load::Force { .. } | Load::Pressure { .. } | Load::Gravity { .. }
        )
    }
    pub fn is_support(&self) -> bool {
        matches!(self, Load::FixedSupport { .. } | Load::Displacement { .. })
    }
    pub fn summary(&self) -> String {
        let f = |v: &[String]| v.join(", ");
        match self {
            Load::FixedSupport { faces } => format!("on {}", f(faces)),
            Load::Displacement { faces, x, y, z } => {
                let c = |o: &Option<f64>| o.map(|v| format!("{v}")).unwrap_or("free".into());
                format!("({}, {}, {}) mm on {}", c(x), c(y), c(z), f(faces))
            }
            Load::Force { faces, vector } => {
                format!("({}, {}, {}) N on {}", vector[0], vector[1], vector[2], f(faces))
            }
            Load::Pressure { faces, value_mpa } => format!("{value_mpa} MPa on {}", f(faces)),
            Load::Gravity { acceleration: a } => format!("({}, {}, {}) m/s²", a[0], a[1], a[2]),
            Load::Temperature { faces, value } => format!("{value} °C on {}", f(faces)),
            Load::HeatFlux { faces, value } => format!("{value} W/m² on {}", f(faces)),
            Load::Convection { faces, coefficient, ambient } => {
                format!("h={coefficient} W/m²K, T∞={ambient} °C on {}", f(faces))
            }
            Load::HeatGeneration { value } => format!("{value} W/m³"),
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
    /// Element size [mm]; None = automatic.
    #[serde(default)]
    pub element_size: Option<f64>,
    #[serde(default)]
    pub next_id: u32,
}

impl Default for Model {
    fn default() -> Self {
        Model {
            bodies: vec![],
            custom_materials: vec![],
            analysis: AnalysisSettings::default(),
            loads: vec![],
            element_size: None,
            next_id: 1,
        }
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
}
