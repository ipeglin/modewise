//! Self-description of the built-in operations: a JSON catalogue for tools and agents, and the
//! generated reference page (docs/reference/operations.md).

use modewise_pipeline::{Operation, Port};
use serde_json::{Value, json};

use crate::builtin_registry;

fn ports(ports: &[Port]) -> Value {
    ports.iter().map(|p| json!({"name": p.name, "kinds": p.kinds.iter().map(ToString::to_string).collect::<Vec<_>>()})).collect()
}

fn entry(name: &str, op: &dyn Operation) -> Value {
    let d = op.descriptor();
    json!({
        "name": name,
        "version": d.version,
        "summary": op.summary(),
        "inputs": ports(d.inputs),
        "outputs": ports(d.outputs),
        "params": op.params_schema(),
    })
}

/// Every built-in operation: name, version, summary, ports and parameter JSON Schema.
pub fn catalogue() -> Value {
    let registry = builtin_registry();
    let operations: Vec<Value> = registry
        .names()
        .filter_map(|n| registry.get(n).map(|op| entry(n, op.as_ref())))
        .collect();
    json!({ "operations": operations })
}

fn resolve<'a>(schema: &'a Value, spec: &'a Value) -> &'a Value {
    match spec.get("$ref").and_then(Value::as_str) {
        Some(r) => &schema["$defs"][r.trim_start_matches("#/$defs/")],
        None => spec,
    }
}

/// A short human type for a schema node.
fn type_name(schema: &Value, spec: &Value) -> String {
    let spec = resolve(schema, spec);
    if let Some(values) = spec.get("enum").and_then(Value::as_array) {
        return values
            .iter()
            .map(|v| format!("`{}`", v.as_str().unwrap_or_default()))
            .collect::<Vec<_>>()
            .join(", ");
    }
    if let Some(options) = spec
        .get("anyOf")
        .or_else(|| spec.get("oneOf"))
        .and_then(Value::as_array)
    {
        let names: Vec<String> = options
            .iter()
            .filter(|o| o.get("type") != Some(&json!("null")))
            .map(|o| type_name(schema, o))
            .collect();
        return names.join(" or ");
    }
    let kind = match spec.get("type") {
        Some(Value::Array(kinds)) => kinds
            .iter()
            .filter_map(Value::as_str)
            .find(|k| *k != "null")
            .unwrap_or("value"),
        Some(Value::String(kind)) => kind.as_str(),
        _ => "value",
    };
    match kind {
        "array" => format!(
            "list of {}",
            spec.get("items")
                .map_or("values".into(), |i| type_name(schema, i))
        ),
        "object" if spec.get("properties").is_some() => {
            // Sorted, so the page does not depend on serde_json's map order.
            let mut keys: Vec<&String> = spec["properties"]
                .as_object()
                .map(|p| p.keys().collect())
                .unwrap_or_default();
            keys.sort();
            format!(
                "table ({})",
                keys.iter()
                    .map(|k| format!("`{k}`"))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        }
        "object" => "table".into(),
        "number" => "number".into(),
        other => other.into(),
    }
}

fn cell(text: &str) -> String {
    text.replace('|', "\\|").replace('\n', " ")
}

fn port_line(ports: &Value) -> String {
    ports
        .as_array()
        .map(|ps| {
            ps.iter()
                .map(|p| {
                    let kinds: Vec<&str> = p["kinds"]
                        .as_array()
                        .map(|k| k.iter().filter_map(Value::as_str).collect())
                        .unwrap_or_default();
                    format!(
                        "`{}` ({})",
                        p["name"].as_str().unwrap_or_default(),
                        kinds.join(" or ")
                    )
                })
                .collect::<Vec<_>>()
                .join(", ")
        })
        .unwrap_or_default()
}

/// The reference page for every built-in operation, generated from `catalogue()`.
pub fn reference_markdown() -> String {
    let mut out = String::from(
        "# Operation reference\n\n\
         <!-- Generated from the operations' own schemas by `modewise ops --markdown`; do not edit.\n\
         Regenerate with `MODEWISE_UPDATE_DOCS=1 cargo test -p modewise_operations --test reference`. -->\n\n\
         Every step in a pipeline names one of these operations. Parameters go in the step's `params` table; \
         anything not listed is rejected, and missing parameters take the defaults shown. \
         `modewise ops --json` gives the same information as JSON Schema, for tools and agents.\n",
    );
    for op in catalogue()["operations"].as_array().into_iter().flatten() {
        let name = op["name"].as_str().unwrap_or_default();
        out.push_str(&format!(
            "\n## `{name}`\n\n{}\n\n",
            op["summary"].as_str().unwrap_or_default()
        ));
        out.push_str(&format!(
            "- **Inputs:** {}\n- **Outputs:** {}\n",
            port_line(&op["inputs"]),
            port_line(&op["outputs"])
        ));
        let schema = &op["params"];
        let required: Vec<&str> = schema["required"]
            .as_array()
            .map(|r| r.iter().filter_map(Value::as_str).collect())
            .unwrap_or_default();
        let Some(properties) = schema["properties"].as_object().filter(|p| !p.is_empty()) else {
            out.push_str("\nNo parameters.\n");
            continue;
        };
        let mut fields: Vec<(&String, &Value)> = properties.iter().collect();
        fields.sort_by_key(|(k, _)| (!required.contains(&k.as_str()), k.as_str()));
        out.push_str("\n| Parameter | Type | Default | Meaning |\n|---|---|---|---|\n");
        for (field, spec) in fields {
            let default = match spec.get("default") {
                _ if required.contains(&field.as_str()) => "required".to_string(),
                Some(Value::Null) | None => "—".to_string(),
                Some(v) => format!("`{v}`"),
            };
            let meaning = spec
                .get("description")
                .and_then(Value::as_str)
                .unwrap_or_default();
            out.push_str(&format!(
                "| `{field}` | {} | {default} | {} |\n",
                cell(&type_name(schema, spec)),
                cell(meaning)
            ));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn types_read_from_schema_nodes() {
        let schema = json!({"$defs": {"Side": {"enum": ["left", "right"], "type": "string"}}});
        let name = |spec: Value| type_name(&schema, &spec);
        assert_eq!(name(json!({"$ref": "#/$defs/Side"})), "`left`, `right`");
        assert_eq!(name(json!({"type": ["integer", "null"]})), "integer");
        assert_eq!(
            name(json!({"type": "array", "items": {"type": "string"}})),
            "list of string"
        );
        assert_eq!(
            name(json!({"type": "object", "additionalProperties": {"type": "string"}})),
            "table"
        );
        assert_eq!(
            name(json!({"type": "object", "properties": {"b": {}, "a": {}}})),
            "table (`a`, `b`)"
        );
        assert_eq!(
            name(json!({"anyOf": [{"$ref": "#/$defs/Side"}, {"type": "null"}]})),
            "`left`, `right`"
        );
        assert_eq!(name(json!({})), "value");
        assert_eq!(cell("a | b\nc"), "a \\| b c");
    }
}
