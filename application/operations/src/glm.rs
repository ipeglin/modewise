use modewise_decomposition::Loadings;
use modewise_glm::{NoiseModel, fit, task_regressors};
use modewise_pipeline::{
    Artifact, ArtifactKind, Inputs, Operation, OperationDescriptor, Outputs, Port, resolve_params,
};
use modewise_regress::{Nuisance, design};
use ndarray::Array2;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::input;

#[derive(Serialize, Deserialize, JsonSchema, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum Noise {
    Ar1,
    Ols,
}

#[derive(Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct Params {
    /// Noise model: "ar1" prewhitens each channel with its own AR(1) coefficient; "ols" assumes independent samples.
    #[serde(default = "ar1")]
    noise: Noise,
    /// Also model a linear trend. The mean is always modelled.
    #[serde(default = "yes")]
    detrend: bool,
    /// Also model slow drifts below this frequency (Hz) with discrete cosine regressors; none by default.
    #[serde(default)]
    high_pass_hz: Option<f64>,
}

fn ar1() -> Noise {
    Noise::Ar1
}

fn yes() -> bool {
    true
}

/// Task GLM: one regressor per event condition, fitted with drift and confound regressors
/// (docs/glm.md).
pub struct GlmOperation;

impl Operation for GlmOperation {
    fn summary(&self) -> &'static str {
        "Fits a task GLM to every channel: one regressor per event condition (the events convolved with the canonical haemodynamic response), with the mean, a trend, slow drifts and optional confounds; gives betas and t per condition and channel (docs/glm.md)."
    }

    fn params_schema(&self) -> Value {
        modewise_pipeline::schema_of::<Params>()
    }

    fn descriptor(&self) -> OperationDescriptor {
        OperationDescriptor {
            name: "glm",
            version: env!("CARGO_PKG_VERSION"),
            inputs: &[
                Port {
                    name: "data",
                    kinds: &[ArtifactKind::Series],
                },
                Port {
                    name: "events",
                    kinds: &[ArtifactKind::Events],
                },
                Port {
                    name: "confounds",
                    kinds: &[ArtifactKind::Series],
                },
            ],
            outputs: &[
                Port {
                    name: "betas",
                    kinds: &[ArtifactKind::Loadings],
                },
                Port {
                    name: "t",
                    kinds: &[ArtifactKind::Loadings],
                },
            ],
        }
    }

    fn optional_inputs(&self) -> &'static [&'static str] {
        &["confounds"]
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
        let Artifact::Events(events) = input(inputs, "events")? else {
            return Err("input \"events\" must be events".into());
        };
        if events.is_empty() {
            return Err("there are no events to model".into());
        }
        let (n, fs) = (data.n_samples(), data.sampling_rate());
        let confounds = match inputs.get("confounds").map(|a| a.as_ref()) {
            None => Array2::zeros((0, n)),
            Some(Artifact::Series(c)) if c.sampling_rate() == fs => c.data().to_owned(),
            Some(Artifact::Series(c)) => {
                return Err(format!(
                    "confounds are sampled at {} Hz, the data at {} Hz",
                    c.sampling_rate().hz(),
                    fs.hz()
                ));
            }
            Some(_) => return Err("input \"confounds\" must be a series".into()),
        };
        let nuisance = Nuisance {
            detrend: p.detrend,
            high_pass_hz: p.high_pass_hz,
        };
        let x = design(nuisance, n, fs.hz(), confounds.view()).map_err(|e| e.to_string())?;
        let (conditions, task) = task_regressors(events, n, fs.hz());
        let noise = match p.noise {
            Noise::Ar1 => NoiseModel::Ar1,
            Noise::Ols => NoiseModel::Ols,
        };
        let f = fit(data.data(), task.view(), &conditions, x.view(), noise)
            .map_err(|e| e.to_string())?;
        let table = |values| {
            Loadings::new(values, conditions.clone(), data.labels().to_vec())
                .map(Artifact::Loadings)
                .map_err(|e| e.to_string())
        };
        Ok(Outputs::from([
            ("betas".into(), table(f.betas)?),
            ("t".into(), table(f.t)?),
        ]))
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use std::sync::Arc;

    use modewise_epochs::{Event, Events};
    use modewise_signals::{MultichannelSeries, SamplingRate};
    use serde_json::json;

    use super::*;

    fn series(data: Array2<f64>, hz: f64) -> Arc<Artifact> {
        let labels = (0..data.nrows()).map(|c| format!("c{c}")).collect();
        Arc::new(Artifact::Series(
            MultichannelSeries::new(data, SamplingRate::new(hz).unwrap(), labels).unwrap(),
        ))
    }

    fn events() -> Arc<Artifact> {
        let block = |onset_s, label: &str| Event {
            onset_s,
            duration_s: 10.0,
            label: label.into(),
        };
        Arc::new(Artifact::Events(
            Events::new(vec![
                block(0.0, "face"),
                block(20.0, "shape"),
                block(40.0, "face"),
                block(60.0, "shape"),
            ])
            .unwrap(),
        ))
    }

    fn inputs() -> Inputs {
        let n = 100;
        let signal = Array2::from_shape_fn((2, n), |(c, i)| ((i * (c + 3)) as f64 * 0.7).sin());
        Inputs::from([
            ("data".into(), series(signal, 1.0)),
            ("events".into(), events()),
        ])
    }

    #[test]
    fn gives_betas_and_t_per_condition_and_channel() {
        let params = GlmOperation.resolve(json!({})).unwrap();
        assert_eq!(
            params,
            json!({"noise": "ar1", "detrend": true, "high_pass_hz": null})
        );
        let out = GlmOperation.run(&params, &inputs()).unwrap();
        for port in ["betas", "t"] {
            let Artifact::Loadings(l) = &out[port] else {
                panic!("not loadings")
            };
            assert_eq!(l.latent_labels(), ["face", "shape"]);
            assert_eq!(l.channel_labels(), ["c0", "c1"]);
        }
    }

    #[test]
    fn confounds_join_the_fit() {
        let params = GlmOperation.resolve(json!({"noise": "ols"})).unwrap();
        let mut with = inputs();
        let Artifact::Series(data) = with["data"].as_ref() else {
            panic!()
        };
        with.insert(
            "confounds".into(),
            series(data.data().slice(ndarray::s![0..1, ..]).to_owned(), 1.0),
        );
        let out = GlmOperation.run(&params, &with).unwrap();
        let Artifact::Loadings(b) = &out["betas"] else {
            panic!()
        };
        assert!(b.values().column(0).iter().all(|v| v.abs() < 1e-10));

        with.insert("confounds".into(), series(Array2::zeros((1, 100)), 2.0));
        let err = GlmOperation.run(&params, &with).unwrap_err();
        assert!(err.contains("sampled at 2 Hz"), "{err}");
        with.insert("confounds".into(), events());
        assert!(GlmOperation.run(&params, &with).is_err());
    }

    #[test]
    fn rejects_bad_parameters_and_inputs() {
        for params in [
            json!({"high_pass_hz": 0.0}),
            json!({"high_pass_hz": -0.1}),
            json!({"noise": "ar2"}),
        ] {
            assert!(GlmOperation.resolve(params).is_err());
        }
        let params = GlmOperation.resolve(json!({})).unwrap();
        let mut bad = inputs();
        bad.insert(
            "events".into(),
            Arc::new(Artifact::Events(Events::new(vec![]).unwrap())),
        );
        let err = GlmOperation.run(&params, &bad).unwrap_err();
        assert!(err.contains("no events"), "{err}");
        let mut bad = inputs();
        bad.insert("events".into(), series(Array2::zeros((1, 100)), 1.0));
        assert!(GlmOperation.run(&params, &bad).is_err());
        let mut bad = inputs();
        bad.insert("data".into(), events());
        assert!(GlmOperation.run(&params, &bad).is_err());
    }
}
