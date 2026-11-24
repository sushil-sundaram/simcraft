//! The benchmark suite itself: well-formed cases, comparison template and report rendering.
use serde_json::json;
use simcraft::bench;

#[test]
fn cases_are_well_formed() {
    let cases = bench::cases();
    let mut ids = std::collections::HashSet::new();
    for c in &cases {
        assert!(ids.insert(c.id), "duplicate case id {}", c.id);
        assert!(!c.sizes.is_empty() && !c.metrics.is_empty() && !c.source.is_empty(), "{}", c.id);
        for m in &c.metrics {
            assert!(m.reference.is_finite() && m.tol > 0.0, "{}", m.id);
            assert!(ids.insert(m.id), "duplicate metric id {}", m.id);
        }
    }
    assert!(cases.len() >= 24, "{} cases", cases.len());
    assert!(cases.iter().filter(|c| c.quick).count() >= 15);
    // the comparison template lists every metric
    let t = bench::template();
    for c in &cases {
        for m in &c.metrics {
            assert!(t.get(m.id).is_some(), "template misses {}", m.id);
        }
    }
}

#[test]
fn report_renders_with_another_solver() {
    let r = json!({
        "simcraft_version": "test", "mode": "full", "seconds": 1.0, "passed": 1, "failed": 0, "machine": {"cpu": "x", "threads": 4},
        "other_solver_name": "Ansys",
        "cases": [{"id": "cantilever", "group": "Structural", "title": "Cantilever", "source": "beam theory", "size_unit": "mm",
            "runs": [{"size": 2.0}, {"size": 1.0}],
            "metrics": [{"id": "cantilever.tip", "title": "tip", "unit": "mm", "reference": 1.0, "tolerance": 0.02, "value": 1.005, "error": 0.005, "pass": true,
                "points": [{"size": 2.0, "value": 1.03, "error": 0.03, "dofs": 1000, "seconds": 0.1}, {"size": 1.0, "value": 1.005, "error": 0.005, "dofs": 8000, "seconds": 0.3}],
                "other_solver": {"value": 1.002, "dofs": 5000}}]}]
    });
    let h = bench::html(&r);
    assert!(h.contains("<svg") && h.contains("PASS") && h.contains("Ansys: 1.002 mm (0.20 % error, 5000 unknowns)"), "{}", &h[..h.len().min(400)]);
}
