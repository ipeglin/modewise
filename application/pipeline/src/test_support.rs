//! Fake operations and an in-memory store for engine tests.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use modewise_signals::{MultichannelSeries, SamplingRate};
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    Artifact, ArtifactKey, ArtifactKind, ArtifactStore, Inputs, Operation, OperationDescriptor,
    Outputs, Port, Registry, StoreError, resolve_params,
};

const SERIES_IN: &[Port] = &[Port {
    name: "signal",
    kinds: &[ArtifactKind::Series],
}];
const SERIES_OUT: &[Port] = &[Port {
    name: "signal",
    kinds: &[ArtifactKind::Series],
}];
const PAIR_IN: &[Port] = &[
    Port {
        name: "left",
        kinds: &[ArtifactKind::Series],
    },
    Port {
        name: "right",
        kinds: &[ArtifactKind::Series],
    },
];
const TWO_OUT: &[Port] = &[
    Port {
        name: "low",
        kinds: &[ArtifactKind::Series],
    },
    Port {
        name: "high",
        kinds: &[ArtifactKind::Series],
    },
];
const DECOMPOSITION_IN: &[Port] = &[Port {
    name: "modes",
    kinds: &[ArtifactKind::Decomposition],
}];

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ScaleParams {
    #[serde(default = "one")]
    factor: f64,
}

fn one() -> f64 {
    1.0
}

/// Multiplies its input(s) by `factor`. Counts how often it runs.
pub struct Scale {
    name: &'static str,
    inputs: &'static [Port],
    outputs: &'static [Port],
    optional: &'static [&'static str],
    pub runs: AtomicUsize,
}

impl Scale {
    fn new(name: &'static str, inputs: &'static [Port], outputs: &'static [Port]) -> Arc<Self> {
        Arc::new(Self {
            name,
            inputs,
            outputs,
            optional: &[],
            runs: AtomicUsize::new(0),
        })
    }
}

impl Operation for Scale {
    fn descriptor(&self) -> OperationDescriptor {
        OperationDescriptor {
            name: self.name,
            version: "1",
            inputs: self.inputs,
            outputs: self.outputs,
        }
    }

    fn optional_inputs(&self) -> &'static [&'static str] {
        self.optional
    }

    fn resolve(&self, params: Value) -> Result<Value, String> {
        resolve_params::<ScaleParams>(params).map(|(_, v)| v)
    }

    fn run(&self, params: &Value, inputs: &Inputs) -> Result<Outputs, String> {
        self.runs.fetch_add(1, Ordering::SeqCst);
        let factor = params["factor"].as_f64().unwrap_or(1.0);
        let Some(Artifact::Series(first)) = inputs.values().next().map(|a| a.as_ref()) else {
            return Err("expected a series".into());
        };
        let scaled = MultichannelSeries::new(
            first.data().mapv(|v| v * factor),
            first.sampling_rate(),
            first.labels().to_vec(),
        )
        .map_err(|e| e.to_string())?;
        Ok(self
            .outputs
            .iter()
            .map(|p| (p.name.to_string(), Artifact::Series(scaled.clone())))
            .collect())
    }
}

pub struct Ops {
    pub registry: Registry,
    pub scale: Arc<Scale>,
}

pub fn ops() -> Ops {
    let scale = Scale::new("scale", SERIES_IN, SERIES_OUT);
    let mut registry = Registry::new();
    registry.register(scale.clone());
    registry.register(Scale::new("pair", PAIR_IN, SERIES_OUT));
    registry.register(Arc::new(Scale {
        name: "maybe_pair",
        inputs: PAIR_IN,
        outputs: SERIES_OUT,
        optional: &["right"],
        runs: AtomicUsize::new(0),
    }));
    registry.register(Scale::new("split", SERIES_IN, TWO_OUT));
    registry.register(Scale::new("needs_modes", DECOMPOSITION_IN, SERIES_OUT));
    registry.register(Arc::new(Liar));
    Ops { registry, scale }
}

pub fn series(values: &[f64]) -> MultichannelSeries {
    let data = Array2::from_shape_vec((1, values.len()), values.to_vec()).unwrap();
    MultichannelSeries::new(data, SamplingRate::new(1.0).unwrap(), vec!["a".into()]).unwrap()
}

#[derive(Default)]
pub struct TestStore(Mutex<HashMap<ArtifactKey, Arc<Artifact>>>);

impl ArtifactStore for TestStore {
    fn get(&self, key: &ArtifactKey) -> Result<Option<Arc<Artifact>>, StoreError> {
        Ok(self.0.lock().unwrap().get(key).cloned())
    }

    fn put(&self, key: &ArtifactKey, artifact: Arc<Artifact>) -> Result<(), StoreError> {
        self.0.lock().unwrap().insert(key.clone(), artifact);
        Ok(())
    }

    fn contains(&self, key: &ArtifactKey) -> Result<bool, StoreError> {
        Ok(self.0.lock().unwrap().contains_key(key))
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct LiarParams {
    lie: String,
}

/// Breaks its own contract in the way `lie` names, to test the executor's checks.
pub struct Liar;

impl Operation for Liar {
    fn descriptor(&self) -> OperationDescriptor {
        OperationDescriptor {
            name: "liar",
            version: "1",
            inputs: SERIES_IN,
            outputs: SERIES_OUT,
        }
    }

    fn resolve(&self, params: Value) -> Result<Value, String> {
        resolve_params::<LiarParams>(params).map(|(_, v)| v)
    }

    fn run(&self, params: &Value, inputs: &Inputs) -> Result<Outputs, String> {
        let input = inputs
            .values()
            .next()
            .map(|a| a.as_ref().clone())
            .ok_or("no input")?;
        let mut outputs = Outputs::new();
        match params["lie"].as_str() {
            Some("missing") => {}
            Some("wrong_kind") => {
                let Artifact::Series(s) = &input else {
                    return Err("expected series".into());
                };
                let modes = ndarray::Array3::zeros((1, s.n_channels(), s.n_samples()));
                let d = modewise_decomposition::Decomposition::new(
                    modes,
                    vec![0.0],
                    s.sampling_rate(),
                    s.labels().to_vec(),
                    true,
                )
                .map_err(|e| e.to_string())?;
                outputs.insert("signal".into(), Artifact::Decomposition(d));
            }
            Some("extra") => {
                outputs.insert("signal".into(), input.clone());
                outputs.insert("bonus".into(), input);
            }
            _ => return Err("asked to fail".into()),
        }
        Ok(outputs)
    }
}

/// Accepts writes and then forgets them.
#[derive(Default)]
pub struct ForgetfulStore;

impl ArtifactStore for ForgetfulStore {
    fn get(&self, _: &ArtifactKey) -> Result<Option<Arc<Artifact>>, StoreError> {
        Ok(None)
    }

    fn put(&self, _: &ArtifactKey, _: Arc<Artifact>) -> Result<(), StoreError> {
        Ok(())
    }

    fn contains(&self, _: &ArtifactKey) -> Result<bool, StoreError> {
        Ok(false)
    }
}
