use modewise_pipeline::{
    Artifact, ArtifactKind, Inputs, Operation, OperationDescriptor, Outputs, Port, resolve_params,
};
use modewise_spectral::TimeFrequencyTransform;
use modewise_stft::{Detrend, Welch, Window};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::input;

#[derive(Serialize, Deserialize, JsonSchema, Clone, Copy)]
#[serde(rename_all = "snake_case")]
enum WindowName {
    Boxcar,
    Hann,
}

#[derive(Serialize, Deserialize, JsonSchema, Clone, Copy)]
#[serde(rename_all = "snake_case")]
enum DetrendName {
    None,
    Constant,
    Linear,
}

/// Defaults are scipy.signal.periodogram's: the whole run, boxcar window, mean removed.
#[derive(Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct Params {
    /// Welch segment length in seconds; the whole run when absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    segment_s: Option<f64>,
    /// Overlap between segments in seconds; half a segment when absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    overlap_s: Option<f64>,
    /// `boxcar` or `hann` (periodic).
    #[serde(default = "boxcar")]
    window: WindowName,
    /// Per segment: `none`, `constant` (mean) or `linear`.
    #[serde(default = "constant")]
    detrend: DetrendName,
}

fn boxcar() -> WindowName {
    WindowName::Boxcar
}

fn constant() -> DetrendName {
    DetrendName::Constant
}

impl Params {
    fn check(&self) -> Result<(), String> {
        for (name, value) in [("segment_s", self.segment_s), ("overlap_s", self.overlap_s)] {
            if value.is_some_and(|v| !(v.is_finite() && v >= 0.0)) {
                return Err(format!("`{name}` must be finite and non-negative"));
            }
        }
        if self.overlap_s.is_some() && self.segment_s.is_none() {
            return Err("`overlap_s` needs `segment_s`".into());
        }
        Ok(())
    }

    fn method(&self, fs_hz: f64) -> Welch {
        let samples = |s: f64| (s * fs_hz).round() as usize;
        Welch {
            segment_len: self.segment_s.map(samples),
            overlap: self.overlap_s.map(samples),
            window: match self.window {
                WindowName::Boxcar => Window::Boxcar,
                WindowName::Hann => Window::Hann,
            },
            detrend: match self.detrend {
                DetrendName::None => Detrend::None,
                DetrendName::Constant => Detrend::Constant,
                DetrendName::Linear => Detrend::Linear,
            },
        }
    }
}

/// `spectrum`: power spectral density of each channel, one frame (periodogram or Welch).
pub struct SpectrumOperation;

impl Operation for SpectrumOperation {
    fn summary(&self) -> &'static str {
        "Power spectral density per channel as one frame: a periodogram, or Welch's method with segments (as scipy.signal)."
    }

    fn params_schema(&self) -> serde_json::Value {
        modewise_pipeline::schema_of::<Params>()
    }

    fn descriptor(&self) -> OperationDescriptor {
        OperationDescriptor {
            name: "spectrum",
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
        typed.check()?;
        Ok(resolved)
    }

    fn run(&self, params: &Value, inputs: &Inputs) -> Result<Outputs, String> {
        let typed = serde_json::from_value::<Params>(params.clone()).map_err(|e| e.to_string())?;
        let Artifact::Series(signal) = input(inputs, "signal")? else {
            return Err("input \"signal\" must be a series".into());
        };
        let map = typed
            .method(signal.sampling_rate().hz())
            .transform(signal)
            .map_err(|e| e.to_string())?;
        Ok(Outputs::from([(
            "power".into(),
            Artifact::TimeFrequency(map),
        )]))
    }
}
