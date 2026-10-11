use modewise_hilbert::{Padding, Weight, hilbert_spectrum, instantaneous};
use modewise_pipeline::{
    Artifact, ArtifactKind, Inputs, Operation, OperationDescriptor, Outputs, Port, resolve_params,
};
use modewise_spectral::{FrequencyGrid, GridScale};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::input;

#[derive(Serialize, Deserialize, JsonSchema, Clone, Copy, Default)]
#[serde(rename_all = "snake_case")]
enum PaddingName {
    #[default]
    Mirror,
    None,
}

#[derive(Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct HilbertParams {
    /// Extend the signal before the FFT (`mirror`); `none` matches scipy.signal.hilbert.
    #[serde(default)]
    padding: PaddingName,
}

/// Instantaneous amplitude and frequency of every mode (docs/time-frequency.md).
pub struct HilbertOperation;

impl Operation for HilbertOperation {
    fn summary(&self) -> &'static str {
        "Instantaneous amplitude and frequency of every mode from its analytic signal (docs/time-frequency.md)."
    }

    fn params_schema(&self) -> serde_json::Value {
        modewise_pipeline::schema_of::<HilbertParams>()
    }

    fn descriptor(&self) -> OperationDescriptor {
        OperationDescriptor {
            name: "hilbert",
            version: env!("CARGO_PKG_VERSION"),
            inputs: &[Port {
                name: "modes",
                kinds: &[ArtifactKind::Decomposition],
            }],
            outputs: &[Port {
                name: "attributes",
                kinds: &[ArtifactKind::Instantaneous],
            }],
        }
    }

    fn resolve(&self, params: Value) -> Result<Value, String> {
        resolve_params::<HilbertParams>(params).map(|(_, v)| v)
    }

    fn run(&self, params: &Value, inputs: &Inputs) -> Result<Outputs, String> {
        let p: HilbertParams = serde_json::from_value(params.clone()).map_err(|e| e.to_string())?;
        let Artifact::Decomposition(modes) = input(inputs, "modes")? else {
            return Err("input \"modes\" must be a decomposition".into());
        };
        let padding = match p.padding {
            PaddingName::Mirror => Padding::Mirror,
            PaddingName::None => Padding::None,
        };
        let attributes = instantaneous(modes, padding).map_err(|e| e.to_string())?;
        Ok(Outputs::from([(
            "attributes".into(),
            Artifact::Instantaneous(attributes),
        )]))
    }
}

#[derive(Serialize, Deserialize, JsonSchema, Clone, Copy, Default)]
#[serde(rename_all = "snake_case")]
enum ScaleName {
    #[default]
    Linear,
    Log,
}

#[derive(Serialize, Deserialize, JsonSchema, Clone, Copy, Default)]
#[serde(rename_all = "snake_case")]
enum WeightName {
    #[default]
    Power,
    Amplitude,
}

#[derive(Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct SpectrumParams {
    /// Lowest edge of the frequency grid in Hz (above 0 for a log grid).
    f_min: f64,
    /// Highest edge of the frequency grid in Hz.
    f_max: f64,
    /// Number of bins, each [low, high).
    n_bins: usize,
    /// `linear` or `log` bins (log bins report geometric centres).
    #[serde(default)]
    scale: ScaleName,
    /// `power` (amplitude²) or `amplitude`.
    #[serde(default)]
    weight: WeightName,
}

impl SpectrumParams {
    fn grid(&self) -> Result<FrequencyGrid, String> {
        let scale = match self.scale {
            ScaleName::Linear => GridScale::Linear,
            ScaleName::Log => GridScale::Log,
        };
        FrequencyGrid::new(self.f_min, self.f_max, self.n_bins, scale).map_err(|e| e.to_string())
    }
}

/// Hilbert spectrum: power (or amplitude) binned by instantaneous frequency, summed over modes.
pub struct HilbertSpectrumOperation;

impl Operation for HilbertSpectrumOperation {
    fn summary(&self) -> &'static str {
        "Hilbert spectrum: instantaneous amplitude or power of the modes binned on a frequency grid over time (docs/time-frequency.md)."
    }

    fn params_schema(&self) -> serde_json::Value {
        modewise_pipeline::schema_of::<SpectrumParams>()
    }

    fn descriptor(&self) -> OperationDescriptor {
        OperationDescriptor {
            name: "hilbert_spectrum",
            version: env!("CARGO_PKG_VERSION"),
            inputs: &[Port {
                name: "attributes",
                kinds: &[ArtifactKind::Instantaneous],
            }],
            outputs: &[Port {
                name: "power",
                kinds: &[ArtifactKind::TimeFrequency],
            }],
        }
    }

    fn resolve(&self, params: Value) -> Result<Value, String> {
        let (typed, resolved) = resolve_params::<SpectrumParams>(params)?;
        typed.grid()?;
        Ok(resolved)
    }

    fn run(&self, params: &Value, inputs: &Inputs) -> Result<Outputs, String> {
        let p: SpectrumParams =
            serde_json::from_value(params.clone()).map_err(|e| e.to_string())?;
        let Artifact::Instantaneous(attributes) = input(inputs, "attributes")? else {
            return Err("input \"attributes\" must be instantaneous attributes".into());
        };
        let weight = match p.weight {
            WeightName::Power => Weight::Power,
            WeightName::Amplitude => Weight::Amplitude,
        };
        let spectrum =
            hilbert_spectrum(attributes, &p.grid()?, weight).map_err(|e| e.to_string())?;
        Ok(Outputs::from([(
            "power".into(),
            Artifact::TimeFrequency(spectrum.map),
        )]))
    }
}
