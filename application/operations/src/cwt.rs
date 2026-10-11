use modewise_cwt::Cwt;
use modewise_pipeline::{
    Artifact, ArtifactKind, Inputs, Operation, OperationDescriptor, Outputs, Port, resolve_params,
};
use modewise_spectral::{FrequencyGrid, GridScale, TimeFrequencyTransform};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::input;

#[derive(Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct Params {
    /// Lowest frequency in Hz, above 0. The transform analyses the geometric centres of log-spaced bins.
    f_min: f64,
    /// Highest frequency in Hz, at most Nyquist.
    f_max: f64,
    /// Number of log-spaced frequencies.
    n_bins: usize,
    /// Morlet wavenumber (Torrence and Compo use 6).
    #[serde(default = "six")]
    omega0: f64,
    /// Divide power by scale (Liu et al. 2007; single source, so off by default).
    #[serde(default)]
    rectify: bool,
}

fn six() -> f64 {
    6.0
}

impl Params {
    /// Log-spaced bins; the transform analyses their geometric centres.
    fn method(&self) -> Result<Cwt, String> {
        let grid = FrequencyGrid::new(self.f_min, self.f_max, self.n_bins, GridScale::Log)
            .map_err(|e| e.to_string())?;
        Cwt::new(grid, self.omega0, self.rectify).map_err(|e| e.to_string())
    }
}

/// Morlet continuous wavelet transform with a cone-of-influence mask (docs/time-frequency.md).
pub struct CwtOperation;

impl Operation for CwtOperation {
    fn summary(&self) -> &'static str {
        "Morlet continuous wavelet transform power on log-spaced frequencies, with a cone of influence (docs/time-frequency.md)."
    }

    fn params_schema(&self) -> serde_json::Value {
        modewise_pipeline::schema_of::<Params>()
    }

    fn descriptor(&self) -> OperationDescriptor {
        OperationDescriptor {
            name: "cwt",
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
        typed.method()?;
        Ok(resolved)
    }

    fn run(&self, params: &Value, inputs: &Inputs) -> Result<Outputs, String> {
        let p: Params = serde_json::from_value(params.clone()).map_err(|e| e.to_string())?;
        let Artifact::Series(signal) = input(inputs, "signal")? else {
            return Err("input \"signal\" must be a series".into());
        };
        let map = p.method()?.transform(signal).map_err(|e| e.to_string())?;
        Ok(Outputs::from([(
            "power".into(),
            Artifact::TimeFrequency(map),
        )]))
    }
}
