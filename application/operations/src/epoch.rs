use modewise_epochs::{EpochParams, EpochSource, Epochs, Window};
use modewise_pipeline::{
    Artifact, ArtifactKind, Inputs, Operation, OperationDescriptor, Outputs, Port, resolve_params,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::input;

#[derive(Serialize, Deserialize, JsonSchema, Clone, Copy)]
#[serde(untagged)]
enum WindowParam {
    Named(WindowName),
    Fixed { tmin_s: f64, tmax_s: f64 },
}

#[derive(Serialize, Deserialize, JsonSchema, Clone, Copy)]
#[serde(rename_all = "snake_case")]
enum WindowName {
    Duration,
}

#[derive(Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct Params {
    /// Required: shift added to event times (haemodynamic delay in fMRI; 0 for EEG).
    lag_s: f64,
    /// `"duration"` uses each event's own length (blocks); `{ tmin_s = -0.2, tmax_s = 0.5 }` uses a fixed window around each onset (trials).
    #[serde(default = "duration")]
    window: WindowParam,
    /// Keep only these event labels; all by default.
    #[serde(default)]
    conditions: Option<Vec<String>>,
    /// A mode or frequency is resolvable in an epoch when frequency × epoch length ≥ this, and, for maps with a cone of influence, the whole epoch is outside it (docs/epochs.md).
    #[serde(default = "one")]
    min_cycles: f64,
    /// Merge consecutive same-label events (trials) into blocks when gaps are at most this long.
    #[serde(default)]
    merge_gap_s: Option<f64>,
}

fn duration() -> WindowParam {
    WindowParam::Named(WindowName::Duration)
}

fn one() -> f64 {
    1.0
}

impl Params {
    fn epoch_params(&self) -> Result<EpochParams, String> {
        if let Some(gap) = self.merge_gap_s
            && !(gap.is_finite() && gap >= 0.0)
        {
            return Err(format!("merge_gap_s must be finite and >= 0, got {gap}"));
        }
        let window = match self.window {
            WindowParam::Named(WindowName::Duration) => Window::Duration,
            WindowParam::Fixed { tmin_s, tmax_s } => Window::Fixed { tmin_s, tmax_s },
        };
        let params = EpochParams {
            lag_s: self.lag_s,
            window,
            conditions: self.conditions.clone(),
            min_cycles: self.min_cycles,
        };
        params.validate().map_err(|e| e.to_string())?;
        Ok(params)
    }
}

/// Cuts a full-run series or decomposition into epochs around events (rules: docs/epochs.md).
pub struct EpochOperation;

impl Operation for EpochOperation {
    fn summary(&self) -> &'static str {
        "Cuts windows around events out of a full-run series, decomposition, instantaneous attributes or time–frequency map, with an explicit haemodynamic lag (docs/epochs.md)."
    }

    fn params_schema(&self) -> serde_json::Value {
        modewise_pipeline::schema_of::<Params>()
    }

    fn descriptor(&self) -> OperationDescriptor {
        OperationDescriptor {
            name: "epoch",
            version: env!("CARGO_PKG_VERSION"),
            inputs: &[
                Port {
                    name: "data",
                    kinds: &[
                        ArtifactKind::Series,
                        ArtifactKind::Decomposition,
                        ArtifactKind::Instantaneous,
                        ArtifactKind::TimeFrequency,
                    ],
                },
                Port {
                    name: "events",
                    kinds: &[ArtifactKind::Events],
                },
            ],
            outputs: &[Port {
                name: "epochs",
                kinds: &[ArtifactKind::Epochs],
            }],
        }
    }

    fn resolve(&self, params: Value) -> Result<Value, String> {
        let (typed, resolved) = resolve_params::<Params>(params)?;
        typed.epoch_params()?;
        Ok(resolved)
    }

    fn run(&self, params: &Value, inputs: &Inputs) -> Result<Outputs, String> {
        let p: Params = serde_json::from_value(params.clone()).map_err(|e| e.to_string())?;
        let source = match input(inputs, "data")? {
            Artifact::Series(s) => EpochSource::Series(s.clone()),
            Artifact::Decomposition(d) => EpochSource::Decomposition(d.clone()),
            Artifact::Instantaneous(m) => EpochSource::Instantaneous(m.clone()),
            Artifact::TimeFrequency(m) => EpochSource::TimeFrequency(m.clone()),
            other => {
                return Err(format!(
                    "input \"data\" cannot be epoched: it is {}",
                    other.kind()
                ));
            }
        };
        let Artifact::Events(events) = input(inputs, "events")? else {
            return Err("input \"events\" must be events".into());
        };
        let events = match p.merge_gap_s {
            Some(gap) => events.merge_consecutive(gap),
            None => events.clone(),
        };
        let epochs = Epochs::cut(source, &events, p.epoch_params()?).map_err(|e| e.to_string())?;
        Ok(Outputs::from([("epochs".into(), Artifact::Epochs(epochs))]))
    }
}
