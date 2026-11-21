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
    }
}

pub fn library() -> Vec<Material> {
    vec![
        m("Structural Steel", 200.0, 0.30, 7850.0, 250.0, 460.0, 60.5, 1.2e-5, 434.0),
        m("Stainless Steel 304", 193.0, 0.29, 8000.0, 215.0, 505.0, 16.2, 1.73e-5, 500.0),
        m("Aluminum 6061-T6", 68.9, 0.33, 2700.0, 276.0, 310.0, 167.0, 2.36e-5, 896.0),
        m("Aluminum 7075-T6", 71.7, 0.33, 2810.0, 503.0, 572.0, 130.0, 2.34e-5, 960.0),
        m("Titanium Ti-6Al-4V", 113.8, 0.342, 4430.0, 880.0, 950.0, 6.7, 8.6e-6, 526.0),
        m("Copper C101", 115.0, 0.31, 8940.0, 69.0, 220.0, 391.0, 1.7e-5, 385.0),
        m("Brass C360", 97.0, 0.31, 8500.0, 310.0, 385.0, 115.0, 2.05e-5, 380.0),
        m("Gray Cast Iron", 110.0, 0.28, 7200.0, 170.0, 240.0, 52.0, 1.1e-5, 447.0),
        m("Concrete", 30.0, 0.18, 2300.0, 3.0, 3.0, 1.7, 1.0e-5, 880.0),
        m("ABS Plastic", 2.2, 0.35, 1050.0, 40.0, 44.0, 0.17, 9.0e-5, 1400.0),
        m("PLA Plastic", 3.5, 0.36, 1240.0, 50.0, 60.0, 0.13, 6.8e-5, 1800.0),
        m("Nylon PA12", 1.7, 0.39, 1010.0, 48.0, 50.0, 0.25, 1.0e-4, 1700.0),
        m("Carbon Fiber (quasi-iso)", 70.0, 0.10, 1600.0, 600.0, 600.0, 5.0, 2.0e-6, 1000.0),
        m("Wood (Oak, along grain)", 11.0, 0.35, 750.0, 50.0, 90.0, 0.17, 5.0e-6, 2000.0),
        m("Glass", 70.0, 0.22, 2500.0, 50.0, 50.0, 1.0, 9.0e-6, 840.0),
    ]
}

pub fn find(lib: &[Material], name: &str) -> Option<Material> {
    let n = name.to_lowercase();
    lib.iter()
        .find(|m| m.name.to_lowercase() == n)
        .or_else(|| lib.iter().find(|m| m.name.to_lowercase().contains(&n)))
        .cloned()
}
