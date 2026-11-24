//! In-app help: guide parsing, topic lookup and search.
use serde_json::{Value, json};
use simcraft::help;
use simcraft::session::new_core;

#[test]
fn guide_is_split_into_sections() {
    let s = help::sections();
    let chapters: Vec<&str> = s.iter().filter(|x| x.level == 2).map(|x| x.title.as_str()).collect();
    println!("{} sections, {} chapters", s.len(), chapters.len());
    assert!(chapters.len() >= 17, "{chapters:?}");
    assert!(chapters[0].starts_with("1. "));
    // a '#' comment inside a code block is not a heading
    assert!(!s.iter().any(|x| x.title.starts_with("install Ollama")));
    // every chapter has text
    assert!(s.iter().filter(|x| x.level == 2).all(|x| !x.body.trim().is_empty() || help::span(s.iter().position(|y| y.title == x.title).unwrap()).len() > 1));
}

#[test]
fn topics_and_search() {
    let s = help::sections();
    let t = |q: &str| help::find(q).map(|i| s[i].title.clone()).unwrap_or_default();
    assert!(t("7").starts_with("7. "));
    assert!(t("meshing").contains("Meshing"));
    assert_eq!(t("Fluids"), "Fluids");
    assert_eq!(t(help::analysis_topic("microwave_heating")), "Full-wave electromagnetics (RF and microwaves)");
    assert_eq!(t(help::analysis_topic("magnetostatic")), "Electromagnetic");
    // a chapter's text includes its subsections
    let seven = help::section_text(help::find("7").unwrap());
    assert!(seven.contains("### Fluids") && seven.contains("### Structural"));
    // search ranks the relevant section first
    let hits = help::search("heat sink duct", 3);
    println!("{:?}", hits.iter().map(|h| &s[h.0].title).collect::<Vec<_>>());
    assert_eq!(s[hits[0].0].title, "Fluids");
    assert!(help::search("zzzqqq", 3).is_empty());
}

#[test]
fn get_help_tool() {
    let c = new_core();
    let call = |a: Value| c.call("get_help", &a).unwrap().text;
    let toc: Value = serde_json::from_str(&call(json!({}))).unwrap();
    assert!(toc["contents"].as_array().unwrap().len() > 20);
    assert!(call(json!({"topic": "fatigue"})).to_lowercase().contains("goodman"));
    let q: Value = serde_json::from_str(&call(json!({"query": "undo"}))).unwrap();
    assert!(q["results"].as_array().is_some_and(|r| !r.is_empty()));
    assert!(c.call("get_help", &json!({"topic": "quantum chromodynamics"})).is_err());
}
