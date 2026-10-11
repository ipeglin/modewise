use modewise_decomposition::ModeDecomposer;
use modewise_pipeline::{
    Artifact, ArtifactKind, Inputs, Operation, OperationDescriptor, Outputs, Port, resolve_params,
};
use modewise_wavelet::{Boundary, ModwtMra};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::input;

#[derive(Serialize, Deserialize, JsonSchema, Clone, Copy)]
#[serde(rename_all = "snake_case")]
enum BoundaryName {
    Periodic,
    Reflection,
}

#[derive(Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct Params {
    /// sym4 is MATLAB's default and Percival and Walden's LA(8).
    #[serde(default = "sym4")]
    wavelet: String,
    /// Deepest level; by default the deepest whose filter fits the series.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    levels: Option<usize>,
    /// `periodic` (as the MODWT is defined) or `reflection` (fewer edge effects for non-periodic series).
    #[serde(default = "periodic")]
    boundary: BoundaryName,
}

fn sym4() -> String {
    "sym4".into()
}

fn periodic() -> BoundaryName {
    BoundaryName::Periodic
}

impl Params {
    fn method(&self) -> Result<ModwtMra, String> {
        let boundary = match self.boundary {
            BoundaryName::Periodic => Boundary::Periodic,
            BoundaryName::Reflection => Boundary::Reflection,
        };
        ModwtMra::new(&self.wavelet, self.levels, boundary).map_err(|e| e.to_string())
    }
}

/// `modwt_mra`: wavelet multiresolution analysis of each channel.
pub struct ModwtMraOperation;

impl Operation for ModwtMraOperation {
    fn summary(&self) -> &'static str {
        "MODWT multiresolution analysis: octave-band details and a smooth that sum to each channel (docs/wavelets.md)."
    }

    fn params_schema(&self) -> serde_json::Value {
        modewise_pipeline::schema_of::<Params>()
    }

    fn descriptor(&self) -> OperationDescriptor {
        OperationDescriptor {
            name: "modwt_mra",
            version: env!("CARGO_PKG_VERSION"),
            inputs: &[Port {
                name: "signal",
                kinds: &[ArtifactKind::Series],
            }],
            outputs: &[Port {
                name: "modes",
                kinds: &[ArtifactKind::Decomposition],
            }],
        }
    }

    fn resolve(&self, params: Value) -> Result<Value, String> {
        let (typed, resolved) = resolve_params::<Params>(params)?;
        typed.method()?;
        Ok(resolved)
    }

    fn run(&self, params: &Value, inputs: &Inputs) -> Result<Outputs, String> {
        let method = serde_json::from_value::<Params>(params.clone())
            .map_err(|e| e.to_string())?
            .method()?;
        let Artifact::Series(signal) = input(inputs, "signal")? else {
            return Err("input \"signal\" must be a series".into());
        };
        let modes = method.decompose(signal).map_err(|e| e.to_string())?;
        Ok(Outputs::from([(
            "modes".into(),
            Artifact::Decomposition(modes),
        )]))
    }
}
