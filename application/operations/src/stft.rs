use modewise_pipeline::{
    Artifact, ArtifactKind, Inputs, Operation, OperationDescriptor, Outputs, Port, resolve_params,
};
use modewise_spectral::TimeFrequencyTransform;
use modewise_stft::Stft;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::input;

#[derive(Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct Params {
    /// Window length in samples (periodic Hann).
    window_len: usize,
    /// Step between frames in samples.
    hop: usize,
}

pub struct StftOperation;

impl Operation for StftOperation {
    fn summary(&self) -> &'static str {
        "Short-time Fourier transform power with a periodic Hann window."
    }

    fn params_schema(&self) -> serde_json::Value {
        modewise_pipeline::schema_of::<Params>()
    }

    fn descriptor(&self) -> OperationDescriptor {
        OperationDescriptor {
            name: "stft",
            version: env!("CARGO_PKG_VERSION"),
            inputs: &[Port {
                name: "signal",
                kinds: &[ArtifactKind::Series],
            }],
            outputs: &[Port {
                name: "power",
                kinds: &[ArtifactKind::TimeFrequency],
            }],
        }
    }

    fn resolve(&self, params: Value) -> Result<Value, String> {
        let (typed, resolved) = resolve_params::<Params>(params)?;
        Stft::new(typed.window_len, typed.hop).map_err(|e| e.to_string())?;
        Ok(resolved)
    }

    fn run(&self, params: &Value, inputs: &Inputs) -> Result<Outputs, String> {
        let p: Params = serde_json::from_value(params.clone()).map_err(|e| e.to_string())?;
        let method = Stft::new(p.window_len, p.hop).map_err(|e| e.to_string())?;
        let Artifact::Series(signal) = input(inputs, "signal")? else {
            return Err("input \"signal\" must be a series".into());
        };
        let map = method.transform(signal).map_err(|e| e.to_string())?;
        Ok(Outputs::from([(
            "power".into(),
            Artifact::TimeFrequency(map),
        )]))
    }
}
