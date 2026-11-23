//! Engineering materials library (SI units unless noted).

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Material {
    pub name: String,
    /// Young's modulus [GPa]
    pub youngs_modulus_gpa: f64,
    pub poisson_ratio: f64,
    /// Density [kg/m^3]
    pub density: f64,
    /// Tensile yield strength [MPa]
    pub yield_strength_mpa: f64,
    /// Ultimate tensile strength [MPa]
    pub ultimate_strength_mpa: f64,
    /// Thermal conductivity [W/(m K)]
    pub thermal_conductivity: f64,
    /// Coefficient of thermal expansion [1/K]
    pub thermal_expansion: f64,
    /// Specific heat [J/(kg K)]
    pub specific_heat: f64,
    /// Plastic tangent modulus [GPa] for bilinear hardening (0 = 1% of E).
    #[serde(default)]
    pub tangent_modulus_gpa: f64,
    /// Relative permittivity εr [-]
    #[serde(default = "one")]
    pub relative_permittivity: f64,
    /// Electrical conductivity σ [S/m]
    #[serde(default)]
    pub electrical_conductivity: f64,
    /// Relative magnetic permeability μr [-]
    #[serde(default = "one")]
    pub relative_permeability: f64,
    /// Dynamic viscosity μ [Pa·s] (fluids)
    #[serde(default)]
    pub dynamic_viscosity: f64,
    /// True for fluids (air, water, oil): usable as a CFD / field enclosure.
    #[serde(default)]
    pub is_fluid: bool,
    /// Basquin fatigue strength coefficient σf' [MPa] (0 = estimate from ultimate strength).
    #[serde(default)]
    pub fatigue_strength_coefficient_mpa: f64,
    /// Basquin exponent b (negative; 0 = estimate).
    #[serde(default)]
    pub fatigue_exponent: f64,
    /// Endurance limit [MPa] (0 = estimate for ferrous metals, none otherwise).
    #[serde(default)]
    pub endurance_limit_mpa: f64,
    /// Dielectric loss tangent tan δ = ε''/ε' at RF/microwave frequencies (full-wave EM).
    #[serde(default)]
    pub loss_tangent: f64,
    /// Orthotropic thermal conductivity [kx, ky, kz] W/(m·K) (homogenised PCBs); None = isotropic.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub conductivity_xyz: Option<[f64; 3]>,
}
fn one() -> f64 {
    1.0
}

/// Fatigue S-N parameters (σf' MPa, b, endurance MPa or 0) and whether they were estimated.
pub struct SnCurve {
    pub sigma_f: f64,
    pub b: f64,
    pub endurance: f64,
    pub estimated: bool,
}

impl Material {
    /// Basquin S-N curve σa = σf'(2N)^b. Missing data is estimated from the ultimate strength
    /// (σf' ≈ Su + 345 MPa, b ≈ −0.085; ferrous endurance limit ≈ 0.5 Su ≤ 700 MPa).
    pub fn sn_curve(&self) -> SnCurve {
        let ferrous = ["steel", "iron"].iter().any(|k| self.name.to_lowercase().contains(k));
        let est = self.fatigue_strength_coefficient_mpa <= 0.0 || self.fatigue_exponent >= 0.0;
        SnCurve {
            sigma_f: if self.fatigue_strength_coefficient_mpa > 0.0 { self.fatigue_strength_coefficient_mpa } else { self.ultimate_strength_mpa + 345.0 },
            b: if self.fatigue_exponent < 0.0 { self.fatigue_exponent } else { -0.085 },
            endurance: if self.endurance_limit_mpa > 0.0 {
                self.endurance_limit_mpa
            } else if ferrous {
                (0.5 * self.ultimate_strength_mpa).min(700.0)
            } else {
                0.0
            },
            estimated: est || (self.endurance_limit_mpa <= 0.0 && ferrous),
        }
    }
    pub fn tangent_modulus(&self) -> f64 {
        if self.tangent_modulus_gpa > 0.0 { self.tangent_modulus_gpa } else { self.youngs_modulus_gpa * 0.01 }
    }
}

fn m(name: &str, e: f64, nu: f64, rho: f64, sy: f64, su: f64, k: f64, a: f64, cp: f64) -> Material {
    Material {
        name: name.into(),
        youngs_modulus_gpa: e,
        poisson_ratio: nu,
        density: rho,
        yield_strength_mpa: sy,
        ultimate_strength_mpa: su,
        thermal_conductivity: k,
        thermal_expansion: a,
        specific_heat: cp,
        tangent_modulus_gpa: 0.0,
        relative_permittivity: 1.0,
        electrical_conductivity: 0.0,
        relative_permeability: 1.0,
        dynamic_viscosity: 0.0,
        is_fluid: false,
        fatigue_strength_coefficient_mpa: 0.0,
        fatigue_exponent: 0.0,
        endurance_limit_mpa: 0.0,
        loss_tangent: 0.0,
        conductivity_xyz: None,
    }
}

impl Material {
    /// Electromagnetic properties: εr, σ [S/m], μr.
    fn em(mut self, eps: f64, sigma: f64, mu: f64) -> Self {
        self.relative_permittivity = eps;
        self.electrical_conductivity = sigma;
        self.relative_permeability = mu;
        self
    }
    fn fluid(mut self, viscosity: f64) -> Self {
        self.is_fluid = true;
        self.dynamic_viscosity = viscosity;
        self
    }
    /// Thermal conductivity along axis a (0 = x, 1 = y, 2 = z).
    pub fn k_axis(&self, a: usize) -> f64 {
        self.conductivity_xyz.map(|k| k[a]).unwrap_or(self.thermal_conductivity)
    }
    fn loss(mut self, tan_d: f64) -> Self {
        self.loss_tangent = tan_d;
        self
    }
    /// Effective conductivity at frequency f [Hz]: σ + ωε₀εr·tanδ.
    pub fn sigma_at(&self, f: f64) -> f64 {
        self.electrical_conductivity + std::f64::consts::TAU * f * 8.854187817e-12 * self.relative_permittivity * self.loss_tangent
    }
    fn tangent(mut self, et: f64) -> Self {
        self.tangent_modulus_gpa = et;
        self
    }
}

pub fn library() -> Vec<Material> {
    vec![
        m("Structural Steel", 200.0, 0.30, 7850.0, 250.0, 460.0, 60.5, 1.2e-5, 434.0).em(1.0, 1.45e6, 100.0).tangent(1.45),
        m("Stainless Steel 304", 193.0, 0.29, 8000.0, 215.0, 505.0, 16.2, 1.73e-5, 500.0).em(1.0, 1.39e6, 1.008).tangent(2.0),
        m("Aluminum 6061-T6", 68.9, 0.33, 2700.0, 276.0, 310.0, 167.0, 2.36e-5, 896.0).em(1.0, 2.5e7, 1.0).tangent(0.7),
        m("Aluminum 7075-T6", 71.7, 0.33, 2810.0, 503.0, 572.0, 130.0, 2.34e-5, 960.0).em(1.0, 1.9e7, 1.0).tangent(0.7),
        m("Titanium Ti-6Al-4V", 113.8, 0.342, 4430.0, 880.0, 950.0, 6.7, 8.6e-6, 526.0).em(1.0, 5.8e5, 1.0).tangent(1.1),
        m("Copper C101", 115.0, 0.31, 8940.0, 69.0, 220.0, 391.0, 1.7e-5, 385.0).em(1.0, 5.8e7, 1.0).tangent(1.2),
        m("Brass C360", 97.0, 0.31, 8500.0, 310.0, 385.0, 115.0, 2.05e-5, 380.0).em(1.0, 1.5e7, 1.0),
        m("Gray Cast Iron", 110.0, 0.28, 7200.0, 170.0, 240.0, 52.0, 1.1e-5, 447.0).em(1.0, 1.5e6, 200.0),
        m("Electrical Steel (M19)", 200.0, 0.30, 7650.0, 350.0, 450.0, 25.0, 1.2e-5, 460.0).em(1.0, 2.0e6, 4000.0),
        m("NdFeB Magnet (N42)", 160.0, 0.24, 7500.0, 75.0, 75.0, 9.0, 5.0e-6, 440.0).em(1.0, 6.7e5, 1.05),
        m("Concrete", 30.0, 0.18, 2300.0, 3.0, 3.0, 1.7, 1.0e-5, 880.0).em(4.5, 0.01, 1.0).loss(0.1),
        m("ABS Plastic", 2.2, 0.35, 1050.0, 40.0, 44.0, 0.17, 9.0e-5, 1400.0).em(3.0, 1e-14, 1.0).loss(0.005),
        m("PLA Plastic", 3.5, 0.36, 1240.0, 50.0, 60.0, 0.13, 6.8e-5, 1800.0).em(3.0, 1e-14, 1.0).loss(0.01),
        m("Nylon PA12", 1.7, 0.39, 1010.0, 48.0, 50.0, 0.25, 1.0e-4, 1700.0).em(3.5, 1e-12, 1.0).loss(0.02),
        m("PTFE (Teflon)", 0.5, 0.46, 2200.0, 12.0, 25.0, 0.25, 1.2e-4, 1000.0).em(2.1, 1e-16, 1.0).loss(0.0002),
        m("FR-4 (PCB laminate)", 22.0, 0.14, 1850.0, 300.0, 300.0, 0.3, 1.4e-5, 1100.0).em(4.4, 1e-14, 1.0).loss(0.02),
        m("Carbon Fiber (quasi-iso)", 70.0, 0.10, 1600.0, 600.0, 600.0, 5.0, 2.0e-6, 1000.0).em(10.0, 1e4, 1.0),
        m("Wood (Oak, along grain)", 11.0, 0.35, 750.0, 50.0, 90.0, 0.17, 5.0e-6, 2000.0).em(2.5, 1e-10, 1.0).loss(0.05),
        m("Glass", 70.0, 0.22, 2500.0, 50.0, 50.0, 1.0, 9.0e-6, 840.0).em(6.0, 1e-12, 1.0).loss(0.005),
        m("Air", 1e-4, 0.3, 1.204, 0.0, 0.0, 0.026, 3.4e-3, 1006.0).em(1.0006, 0.0, 1.0).fluid(1.81e-5),
        m("Water", 1e-4, 0.3, 998.0, 0.0, 0.0, 0.6, 2.1e-4, 4182.0).em(78.0, 0.05, 1.0).loss(0.157).fluid(1.0e-3),
        // dielectric data at 2.45 GHz, ~20 °C (meat/vegetables ≈ εr 50, ε'' 15; potato 57/17)
        m("Food (water-rich, 2.45 GHz)", 1e-4, 0.3, 1050.0, 0.0, 0.0, 0.5, 2.0e-4, 3600.0).em(52.0, 0.0, 1.0).loss(0.3),
        // epoxy mould compound of IC packages / generic SMD component body
        m("IC Package (mold compound)", 24.0, 0.3, 1900.0, 100.0, 120.0, 0.8, 1.2e-5, 880.0).em(4.0, 1e-12, 1.0).loss(0.01),
        m("Microwave Glass Tray", 70.0, 0.22, 2500.0, 50.0, 50.0, 1.0, 9.0e-6, 840.0).em(6.0, 1e-12, 1.0).loss(0.002),
        m("Engine Oil (SAE 30)", 1e-4, 0.3, 875.0, 0.0, 0.0, 0.145, 7.0e-4, 1900.0).em(2.2, 1e-12, 1.0).fluid(0.29).loss(0.002),
    ]
}

pub fn find(lib: &[Material], name: &str) -> Option<Material> {
    let n = name.to_lowercase();
    lib.iter()
        .find(|m| m.name.to_lowercase() == n)
        .or_else(|| lib.iter().find(|m| m.name.to_lowercase().contains(&n)))
        .cloned()
}
