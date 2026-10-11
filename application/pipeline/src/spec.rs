use std::collections::BTreeMap;

use serde::Deserialize;
use serde_json::{Map, Value};

use crate::{ArtifactKind, PipelineError};

/// A pipeline as the user writes it, usually in TOML.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PipelineSpec {
    pub pipeline: PipelineHeader,
    #[serde(default, rename = "step")]
    pub steps: Vec<StepSpec>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PipelineHeader {
    pub name: String,
    /// Data supplied when the pipeline runs: a list of names (all series), or a table of
    /// name = kind, for example `{ bold = "series", events = "events" }`.
    #[serde(default)]
    pub inputs: InputDecl,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum InputDecl {
    Series(Vec<String>),
    Typed(BTreeMap<String, ArtifactKind>),
}

impl Default for InputDecl {
    fn default() -> Self {
        Self::Series(Vec::new())
    }
}

impl InputDecl {
    /// Every input with its kind.
    pub fn kinds(&self) -> Vec<(String, ArtifactKind)> {
        match self {
            Self::Series(names) => names
                .iter()
                .map(|n| (n.clone(), ArtifactKind::Series))
                .collect(),
            Self::Typed(map) => map.iter().map(|(n, k)| (n.clone(), *k)).collect(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StepSpec {
    pub id: String,
    pub op: String,
    #[serde(default)]
    pub input: InputSpec,
    #[serde(default)]
    pub params: Map<String, Value>,
    /// Store this step's outputs. With `false` they live only in memory during a run: steps that
    /// use them still run and are cached, but large intermediates (for example modes before band
    /// powers) take no disk space. They are recomputed only when a later step needs them.
    #[serde(default = "keep_by_default")]
    pub keep: bool,
}

fn keep_by_default() -> bool {
    true
}

/// `input = "x"` for single-input operations, or `input = { port = "x" }`.
/// A reference is a pipeline input name, a step id, or `step.output`.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum InputSpec {
    Single(String),
    Named(BTreeMap<String, String>),
}

impl Default for InputSpec {
    fn default() -> Self {
        Self::Named(BTreeMap::new())
    }
}

impl PipelineSpec {
    pub fn from_toml(text: &str) -> Result<Self, PipelineError> {
        toml::from_str(text).map_err(|e| PipelineError::Spec(e.to_string()))
    }

    pub fn from_json(text: &str) -> Result<Self, PipelineError> {
        serde_json::from_str(text).map_err(|e| PipelineError::Spec(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_single_and_named_inputs() {
        let spec = PipelineSpec::from_toml(
            r#"
            [pipeline]
            name = "demo"
            inputs = ["bold"]

            [[step]]
            id = "modes"
            op = "mvmd"
            input = "bold"
            params = { num_modes = 4, alpha = 1000.0 }

            [[step]]
            id = "mix"
            op = "combine"
            input = { left = "modes", right = "bold" }
            "#,
        )
        .unwrap();
        assert_eq!(
            spec.pipeline.inputs.kinds(),
            [("bold".to_string(), ArtifactKind::Series)]
        );
        assert!(matches!(&spec.steps[0].input, InputSpec::Single(r) if r == "bold"));
        assert_eq!(spec.steps[0].params["num_modes"], 4);
        assert!(matches!(&spec.steps[1].input, InputSpec::Named(m) if m.len() == 2));
    }

    #[test]
    fn parses_json() {
        let spec = PipelineSpec::from_json(
            r#"{"pipeline": {"name": "j", "inputs": ["x"]},
                "step": [{"id": "a", "op": "scale", "input": "x", "params": {"factor": 2}}]}"#,
        )
        .unwrap();
        assert_eq!(spec.steps[0].params["factor"], 2);
    }

    #[test]
    fn parses_typed_inputs() {
        let spec = PipelineSpec::from_toml(
            "[pipeline]\nname = \"t\"\ninputs = { bold = \"series\", trials = \"events\" }\n",
        )
        .unwrap();
        assert_eq!(
            spec.pipeline.inputs.kinds(),
            [
                ("bold".to_string(), ArtifactKind::Series),
                ("trials".to_string(), ArtifactKind::Events)
            ]
        );
        assert!(
            PipelineSpec::from_toml("[pipeline]\nname = \"t\"\ninputs = { bold = \"sound\" }\n")
                .is_err()
        );
    }

    #[test]
    fn rejects_invalid_json() {
        assert!(matches!(
            PipelineSpec::from_json("{not json"),
            Err(PipelineError::Spec(_))
        ));
    }

    #[test]
    fn rejects_unknown_fields() {
        let err = PipelineSpec::from_toml("[pipeline]\nname = \"x\"\ntypo = 1\n").unwrap_err();
        assert!(matches!(err, PipelineError::Spec(_)));
    }
}
