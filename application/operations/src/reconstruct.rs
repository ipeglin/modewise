use modewise_pipeline::{
    Artifact, ArtifactKind, Inputs, Operation, OperationDescriptor, Outputs, Port, resolve_params,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::input;

#[derive(Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct Params {
    /// Mode indices to sum, counted from the lowest frequency.
    modes: Vec<usize>,
}

/// Sums chosen modes back into a signal, for example to keep one frequency band.
pub struct ReconstructOperation;

impl Operation for ReconstructOperation {
    fn summary(&self) -> &'static str {
        "Sums chosen modes of a decomposition back into a series, for example to keep one band."
    }

    fn params_schema(&self) -> serde_json::Value {
        modewise_pipeline::schema_of::<Params>()
    }

    fn descriptor(&self) -> OperationDescriptor {
        OperationDescriptor {
            name: "reconstruct",
            version: env!("CARGO_PKG_VERSION"),
            inputs: &[Port {
                name: "modes",
                kinds: &[ArtifactKind::Decomposition],
            }],
            outputs: &[Port {
                name: "signal",
                kinds: &[ArtifactKind::Series],
            }],
        }
    }

    fn resolve(&self, params: Value) -> Result<Value, String> {
        resolve_params::<Params>(params).map(|(_, resolved)| resolved)
    }

    fn run(&self, params: &Value, inputs: &Inputs) -> Result<Outputs, String> {
        let p: Params = serde_json::from_value(params.clone()).map_err(|e| e.to_string())?;
        let Artifact::Decomposition(modes) = input(inputs, "modes")? else {
            return Err("input \"modes\" must be a decomposition".into());
        };
        let signal = modes.reconstruct(&p.modes).map_err(|e| e.to_string())?;
        Ok(Outputs::from([("signal".into(), Artifact::Series(signal))]))
    }
}
