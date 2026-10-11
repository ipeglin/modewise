//! Every built-in operation documents itself: a summary and a described JSON Schema for its
//! parameters, from which docs/reference/operations.md is generated.

#![allow(clippy::unwrap_used)]

use modewise_operations::builtin_registry;
use serde_json::Value;

/// Properties of a schema, following `$ref`s into `$defs`.
fn properties<'a>(schema: &'a Value, node: &'a Value) -> Vec<(String, &'a Value)> {
    let node = match node.get("$ref").and_then(Value::as_str) {
        Some(r) => &schema["$defs"][r.trim_start_matches("#/$defs/")],
        None => node,
    };
    node.get("properties")
        .and_then(Value::as_object)
        .map(|p| p.iter().map(|(k, v)| (k.clone(), v)).collect())
        .unwrap_or_default()
}

#[test]
fn every_operation_has_a_summary_and_described_parameters() {
    let registry = builtin_registry();
    let mut missing = Vec::new();
    for name in registry.names() {
        let op = registry.get(name).unwrap();
        if op.summary().is_empty() {
            missing.push(format!("{name}: summary"));
        }
        let schema = op.params_schema();
        if schema.get("type").is_none() {
            missing.push(format!("{name}: params_schema"));
            continue;
        }
        for (field, spec) in properties(&schema, &schema) {
            if spec
                .get("description")
                .and_then(Value::as_str)
                .is_none_or(str::is_empty)
            {
                missing.push(format!("{name}.{field}: description"));
            }
        }
    }
    assert!(missing.is_empty(), "undocumented:\n{}", missing.join("\n"));
}

/// docs/reference/operations.md is generated from the operations; it must match the code.
#[test]
fn the_committed_reference_is_current() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../docs/reference/operations.md"
    );
    let generated = modewise_operations::reference_markdown();
    if std::env::var_os("MODEWISE_UPDATE_DOCS").is_some() {
        std::fs::create_dir_all(std::path::Path::new(path).parent().unwrap()).unwrap();
        std::fs::write(path, &generated).unwrap();
    }
    let committed = std::fs::read_to_string(path).unwrap_or_default();
    assert!(
        committed == generated,
        "docs/reference/operations.md is stale; run MODEWISE_UPDATE_DOCS=1 cargo test -p modewise_operations --test reference"
    );
}

#[test]
fn the_catalogue_lists_every_operation_with_ports_and_schema() {
    let catalogue = modewise_operations::catalogue();
    let ops = catalogue["operations"].as_array().unwrap();
    assert_eq!(ops.len(), builtin_registry().names().count());
    let spectrum = ops.iter().find(|o| o["name"] == "spectrum").unwrap();
    assert_eq!(spectrum["inputs"][0]["name"], "signal");
    assert_eq!(spectrum["outputs"][0]["kinds"][0], "time_frequency");
    assert_eq!(
        spectrum["params"]["properties"]["window"]["default"],
        "boxcar"
    );
    let mvmd = ops.iter().find(|o| o["name"] == "mvmd").unwrap();
    assert!(
        mvmd["params"]["properties"].get("noise_sd").is_none(),
        "na_mvmd's noise settings are not mvmd's"
    );
    let na = ops.iter().find(|o| o["name"] == "na_mvmd").unwrap();
    assert_eq!(na["params"]["properties"]["noise_sd"]["default"], 0.8);
    let eemd = ops.iter().find(|o| o["name"] == "eemd").unwrap();
    assert_eq!(eemd["params"]["properties"]["stop"]["default"], "fixed");
    assert!(eemd["params"]["properties"].get("epsilon").is_none());
}
