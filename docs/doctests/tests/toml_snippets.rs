//! Every TOML example in the documentation is checked against the code: pipelines must plan,
//! loose steps must pass their operation's parameter check, and dataset specs (or fragments of
//! them) must validate. Other TOML (Cargo manifests) is skipped.

#![allow(clippy::unwrap_used)]

use std::path::{Path, PathBuf};

use modewise_dataset::DatasetSpec;
use modewise_operations::builtin_registry;
use modewise_pipeline::{PipelineSpec, Plan};

fn markdown_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() && !path.ends_with("_build") && !path.ends_with("doctests") {
            markdown_files(&path, out);
        } else if path.extension().is_some_and(|e| e == "md") {
            out.push(path);
        }
    }
}

/// The ```toml blocks of a markdown file, with the line each starts on.
fn toml_blocks(text: &str) -> Vec<(usize, String)> {
    let mut blocks = Vec::new();
    let mut current: Option<(usize, String)> = None;
    for (i, line) in text.lines().enumerate() {
        match (&mut current, line.trim_start()) {
            (None, "```toml") => current = Some((i + 1, String::new())),
            (Some(_), "```") => blocks.extend(current.take()),
            (Some((_, body)), _) => {
                body.push_str(line);
                body.push('\n');
            }
            _ => {}
        }
    }
    blocks
}

const STUB_INPUT: &str = "[[input]]\nname = \"bold\"\npattern = \"sub-{subject}.h5\"\nformat = \"hdf5\"\ndataset = \"d\"\nsampling_rate_hz = 1.0\n";

fn check(block: &str) -> Result<&'static str, String> {
    let has = |table: &str| block.lines().any(|l| l.trim_start().starts_with(table));
    if has("[pipeline]") {
        let spec = PipelineSpec::from_toml(block).map_err(|e| e.to_string())?;
        Plan::new(&spec, &builtin_registry()).map_err(|e| e.to_string())?;
        return Ok("pipeline");
    }
    if has("[[step]]") {
        let value: toml::Value = toml::from_str(block).map_err(|e| e.to_string())?;
        for step in value["step"].as_array().into_iter().flatten() {
            let op = step["op"].as_str().ok_or("a step needs `op`")?;
            let operation = builtin_registry()
                .get(op)
                .ok_or(format!("unknown operation {op:?}"))?;
            let params = step
                .get("params")
                .map_or(serde_json::json!({}), |p| serde_json::to_value(p).unwrap());
            operation.resolve(params)?;
        }
        return Ok("steps");
    }
    if has("root") || has("[[input]]") || has("[channels]") || has("[participants]") {
        let mut spec = String::new();
        if !has("root") {
            spec.push_str("root = \".\"\n");
        }
        // Keys before the first table must stay at the top.
        let (head, tables) = match block.find("\n[") {
            Some(i) if !block.starts_with('[') => block.split_at(i + 1),
            _ if block.starts_with('[') => ("", block),
            _ => (block, ""),
        };
        spec.push_str(head);
        if !has("[[input]]") {
            spec.push_str(STUB_INPUT);
        }
        spec.push_str(tables);
        DatasetSpec::from_toml(&spec).map_err(|e| e.to_string())?;
        return Ok("dataset");
    }
    Ok("skipped")
}

#[test]
fn every_toml_example_matches_the_code() {
    let docs = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let mut files = Vec::new();
    markdown_files(docs, &mut files);
    let (mut checked, mut failures) = (0, Vec::new());
    for file in &files {
        for (line, block) in toml_blocks(&std::fs::read_to_string(file).unwrap()) {
            match check(&block) {
                Ok("skipped") => {}
                Ok(_) => checked += 1,
                Err(e) => failures.push(format!(
                    "{}:{line}: {e}",
                    file.strip_prefix(docs).unwrap().display()
                )),
            }
        }
    }
    assert!(
        failures.is_empty(),
        "documentation TOML out of date:\n{}",
        failures.join("\n")
    );
    assert!(checked >= 6, "only {checked} TOML examples were checked");
}

#[test]
fn broken_examples_are_caught() {
    let pipeline = "[pipeline]\nname = \"x\"\ninputs = [\"bold\"]\n\n[[step]]\nid = \"m\"\nop = \"nope\"\ninput = \"bold\"\n";
    assert!(check(pipeline).unwrap_err().contains("nope"));
    let step = "[[step]]\nid = \"p\"\nop = \"spectrum\"\ninput = \"bold\"\nparams = { windw = \"hann\" }\n";
    assert!(check(step).unwrap_err().contains("windw"));
    let dataset = "[[input]]\nname = \"bold\"\npattern = \"x\"\nformat = \"csv\"\n";
    assert!(check(dataset).is_err());
    assert_eq!(check("[dependencies]\nfoo = \"1\"\n"), Ok("skipped"));
    assert_eq!(
        check(
            "root = \".\"\n[[input]]\nname = \"bold\"\npattern = \"x\"\nformat = \"hdf5\"\ndataset = \"d\"\nsampling_rate_hz = 1.0\n"
        ),
        Ok("dataset")
    );
}
