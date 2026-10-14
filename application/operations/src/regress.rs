use modewise_pipeline::{
    Artifact, ArtifactKind, Inputs, Operation, OperationDescriptor, Outputs, Port, resolve_params,
};
use modewise_regress::{Nuisance, design, residuals};
use modewise_signals::MultichannelSeries;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::input;

#[derive(Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct Params {
    /// Also remove a linear trend. The mean is always removed.
    #[serde(default = "yes")]
    detrend: bool,
    /// Also remove slow drifts below this frequency (Hz) with discrete cosine regressors in the same fit; none by default.
    #[serde(default)]
    high_pass_hz: Option<f64>,
}

fn yes() -> bool {
    true
}

impl Params {
    fn nuisance(&self) -> Nuisance {
        Nuisance {
            detrend: self.detrend,
            high_pass_hz: self.high_pass_hz,
        }
    }
}

/// Nuisance regression: confounds, trend and slow drifts out of every channel (docs/regression.md).
pub struct RegressOperation;

impl Operation for RegressOperation {
    fn summary(&self) -> &'static str {
        "Removes confound regressors, the mean, a linear trend and slow cosine drifts from every channel in one least-squares fit (docs/regression.md)."
    }

    fn params_schema(&self) -> serde_json::Value {
        modewise_pipeline::schema_of::<Params>()
    }

    fn descriptor(&self) -> OperationDescriptor {
        OperationDescriptor {
            name: "regress",
            version: env!("CARGO_PKG_VERSION"),
            inputs: &[
                Port {
                    name: "data",
                    kinds: &[ArtifactKind::Series],
                },
                Port {
                    name: "regressors",
                    kinds: &[ArtifactKind::Series],
                },
            ],
            outputs: &[Port {
                name: "signal",
                kinds: &[ArtifactKind::Series],
            }],
        }
    }

    fn resolve(&self, params: Value) -> Result<Value, String> {
        let (typed, resolved) = resolve_params::<Params>(params)?;
        if let Some(hz) = typed.high_pass_hz
            && !(hz.is_finite() && hz > 0.0)
        {
            return Err(format!("high_pass_hz must be above 0, got {hz}"));
        }
        Ok(resolved)
    }

    fn run(&self, params: &Value, inputs: &Inputs) -> Result<Outputs, String> {
        let p: Params = serde_json::from_value(params.clone()).map_err(|e| e.to_string())?;
        let Artifact::Series(data) = input(inputs, "data")? else {
            return Err("input \"data\" must be a series".into());
        };
        let Artifact::Series(regressors) = input(inputs, "regressors")? else {
            return Err("input \"regressors\" must be a series".into());
        };
        let fs = data.sampling_rate();
        if regressors.sampling_rate() != fs {
            return Err(format!(
                "regressors are sampled at {} Hz, the data at {} Hz",
                regressors.sampling_rate().hz(),
                fs.hz()
            ));
        }
        let x = design(p.nuisance(), data.n_samples(), fs.hz(), regressors.data())
            .map_err(|e| e.to_string())?;
        let clean = residuals(data.data(), x.view()).map_err(|e| e.to_string())?;
        let signal = MultichannelSeries::new(clean, fs, data.labels().to_vec())
            .map_err(|e| e.to_string())?;
        Ok(Outputs::from([("signal".into(), Artifact::Series(signal))]))
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use std::sync::Arc;

    use modewise_signals::SamplingRate;
    use ndarray::{Array2, array};
    use serde_json::json;

    use super::*;

    fn series(data: Array2<f64>, hz: f64) -> Arc<Artifact> {
        let labels = (0..data.nrows()).map(|c| format!("c{c}")).collect();
        Arc::new(Artifact::Series(
            MultichannelSeries::new(data, SamplingRate::new(hz).unwrap(), labels).unwrap(),
        ))
    }

    fn inputs(data_hz: f64, regressors_hz: f64) -> Inputs {
        Inputs::from([
            (
                "data".into(),
                series(array![[1.0, 3.0, 2.0, 6.0, 4.0, 5.0]], data_hz),
            ),
            (
                "regressors".into(),
                series(array![[0.0, 1.0, 0.0, 2.0, 0.0, 1.0]], regressors_hz),
            ),
        ])
    }

    #[test]
    fn keeps_labels_and_rate_and_removes_the_mean() {
        let params = RegressOperation.resolve(json!({})).unwrap();
        assert_eq!(params, json!({"detrend": true, "high_pass_hz": null}));
        let out = RegressOperation.run(&params, &inputs(1.0, 1.0)).unwrap();
        let Artifact::Series(s) = &out["signal"] else {
            panic!("not a series")
        };
        assert_eq!(s.labels(), ["c0"]);
        assert_eq!(s.sampling_rate().hz(), 1.0);
        assert!(s.data().sum().abs() < 1e-12);
    }

    #[test]
    fn rejects_other_rates_and_bad_cutoffs() {
        let params = RegressOperation.resolve(json!({})).unwrap();
        let err = RegressOperation
            .run(&params, &inputs(1.0, 2.0))
            .unwrap_err();
        assert!(err.contains("sampled at 2 Hz"), "{err}");
        for hz in [0.0, -0.1] {
            assert!(
                RegressOperation
                    .resolve(json!({"high_pass_hz": hz}))
                    .is_err()
            );
        }
        let params = RegressOperation
            .resolve(json!({"high_pass_hz": 0.6}))
            .unwrap();
        let err = RegressOperation
            .run(&params, &inputs(1.0, 1.0))
            .unwrap_err();
        assert!(err.contains("Nyquist"), "{err}");
    }

    #[test]
    fn rejects_other_kinds() {
        let params = RegressOperation.resolve(json!({})).unwrap();
        let mut bad = inputs(1.0, 1.0);
        bad.insert(
            "regressors".into(),
            Arc::new(Artifact::Events(
                modewise_epochs::Events::new(vec![]).unwrap(),
            )),
        );
        assert!(RegressOperation.run(&params, &bad).is_err());
        let mut bad = inputs(1.0, 1.0);
        bad.insert(
            "data".into(),
            Arc::new(Artifact::Events(
                modewise_epochs::Events::new(vec![]).unwrap(),
            )),
        );
        assert!(RegressOperation.run(&params, &bad).is_err());
    }
}
