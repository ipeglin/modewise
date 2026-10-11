use modewise_connectivity::Measure;
use modewise_fc::{Source, functional_connectivity};
use modewise_pipeline::{
    Artifact, ArtifactKind, Inputs, Operation, OperationDescriptor, Outputs, Port, resolve_params,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::input;

#[derive(Serialize, Deserialize, JsonSchema, Clone, Copy)]
#[serde(rename_all = "lowercase")]
enum Kind {
    Pearson,
    Plv,
}

#[derive(Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct Params {
    /// `pearson` (correlation) or `plv` (phase locking value).
    measure: Kind,
    /// Return atanh(r) (pearson only); the diagonal becomes NaN.
    #[serde(default)]
    fisher_z: bool,
}

impl Params {
    fn measure(&self) -> Result<Measure, String> {
        match (self.measure, self.fisher_z) {
            (Kind::Pearson, fisher_z) => Ok(Measure::Pearson { fisher_z }),
            (Kind::Plv, false) => Ok(Measure::PhaseLockingValue),
            (Kind::Plv, true) => Err("fisher_z applies to pearson only".into()),
        }
    }
}

/// Functional connectivity per mode and per epoch (docs/connectivity.md).
pub struct FunctionalConnectivityOperation;

impl Operation for FunctionalConnectivityOperation {
    fn summary(&self) -> &'static str {
        "Functional connectivity per component (and per epoch): Pearson correlation or phase locking value (docs/connectivity.md)."
    }

    fn params_schema(&self) -> serde_json::Value {
        modewise_pipeline::schema_of::<Params>()
    }

    fn descriptor(&self) -> OperationDescriptor {
        OperationDescriptor {
            name: "functional_connectivity",
            version: env!("CARGO_PKG_VERSION"),
            inputs: &[Port {
                name: "data",
                kinds: &[
                    ArtifactKind::Series,
                    ArtifactKind::Decomposition,
                    ArtifactKind::Epochs,
                ],
            }],
            outputs: &[Port {
                name: "matrices",
                kinds: &[ArtifactKind::FunctionalConnectivity],
            }],
        }
    }

    fn resolve(&self, params: Value) -> Result<Value, String> {
        let (typed, resolved) = resolve_params::<Params>(params)?;
        typed.measure()?;
        Ok(resolved)
    }

    fn run(&self, params: &Value, inputs: &Inputs) -> Result<Outputs, String> {
        let measure = serde_json::from_value::<Params>(params.clone())
            .map_err(|e| e.to_string())?
            .measure()?;
        let source = match input(inputs, "data")? {
            Artifact::Series(s) => Source::Series(s),
            Artifact::Decomposition(d) => Source::Decomposition(d),
            Artifact::Epochs(e) => Source::Epochs(e),
            other => return Err(format!("input \"data\" cannot be a {}", other.kind())),
        };
        let matrices = functional_connectivity(source, measure).map_err(|e| e.to_string())?;
        Ok(Outputs::from([(
            "matrices".into(),
            Artifact::FunctionalConnectivity(matrices),
        )]))
    }
}
