use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::ArtifactKey;

/// What a run did, for provenance: every step's operation, parameters and data keys.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunManifest {
    pub pipeline: String,
    pub modewise_version: String,
    pub inputs: BTreeMap<String, ArtifactKey>,
    pub steps: Vec<StepRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepRecord {
    pub id: String,
    pub op: String,
    pub op_version: String,
    pub params: Value,
    pub inputs: BTreeMap<String, ArtifactKey>,
    pub outputs: BTreeMap<String, ArtifactKey>,
    pub cached: bool,
    /// False for steps with `keep = false`: their outputs were not written to the store.
    #[serde(default = "stored_by_default")]
    pub stored: bool,
    pub seconds: f64,
}

fn stored_by_default() -> bool {
    true
}

impl RunManifest {
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }

    /// Every step output, by `step.output`.
    pub fn outputs(&self) -> BTreeMap<String, ArtifactKey> {
        self.steps
            .iter()
            .flat_map(|s| {
                s.outputs
                    .iter()
                    .map(|(o, k)| (format!("{}.{o}", s.id), k.clone()))
            })
            .collect()
    }
}
