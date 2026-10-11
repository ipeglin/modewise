#![allow(clippy::unwrap_used)]

use std::collections::BTreeMap;
use std::f64::consts::PI;

use modewise_operations::builtin_registry;
use modewise_pipeline::{
    Artifact, ArtifactStore, PipelineError, PipelineSpec, Plan, RunResult, run,
};
use modewise_signals::{MultichannelSeries, SamplingRate};
use modewise_store_memory::MemoryStore;
use ndarray::Array2;

const SPEC: &str = r#"
[pipeline]
name = "band-power"
inputs = ["bold"]

[[step]]
id = "modes"
op = "mvmd"
input = "bold"
params = { num_modes = 3, alpha = 2000.0, init = "linear", tau = 0.0, tolerance = 1e-7 }

[[step]]
id = "slow"
op = "reconstruct"
input = "modes"
params = { modes = MODES }

[[step]]
id = "spectrum"
op = "stft"
input = "slow"
params = { window_len = 128, hop = 32 }
"#;

fn spec(modes: &str) -> PipelineSpec {
    PipelineSpec::from_toml(&SPEC.replace("MODES", modes)).unwrap()
}

fn bold() -> BTreeMap<String, Artifact> {
    let fs = 100.0;
    let data = Array2::from_shape_fn((2, 1000), |(c, i)| {
        let t = i as f64 / fs;
        let w = [1.0, 0.6][c];
        w * (2.0 * PI * 2.0 * t).sin()
            + (2.0 * PI * 12.0 * t).sin()
            + 0.3 * (2.0 * PI * 30.0 * t).sin()
    });
    let labels = vec!["left".into(), "right".into()];
    let series = MultichannelSeries::new(data, SamplingRate::new(fs).unwrap(), labels).unwrap();
    BTreeMap::from([("bold".to_string(), Artifact::Series(series))])
}

fn cached(result: &RunResult) -> Vec<(&str, bool)> {
    result
        .manifest
        .steps
        .iter()
        .map(|s| (s.id.as_str(), s.cached))
        .collect()
}

#[test]
fn runs_caches_and_reruns_only_changed_branch() {
    let registry = builtin_registry();
    let store = MemoryStore::new();

    let first = run(
        &Plan::new(&spec("[0]"), &registry).unwrap(),
        &bold(),
        &store,
    )
    .unwrap();
    assert_eq!(
        cached(&first),
        [("modes", false), ("slow", false), ("spectrum", false)]
    );

    let Artifact::TimeFrequency(map) = store
        .get(first.output_key("spectrum").unwrap())
        .unwrap()
        .unwrap()
        .as_ref()
        .clone()
    else {
        panic!("spectrum should be a time-frequency map");
    };
    let power = map.power();
    let peak_bin = (0..map.frequencies_hz().len())
        .max_by(|&a, &b| power[[0, a, 0]].total_cmp(&power[[0, b, 0]]))
        .unwrap();
    let peak_hz = map.frequencies_hz()[peak_bin];
    assert!(
        (peak_hz - 2.0).abs() <= 1.0,
        "slow band should peak near 2 Hz, got {peak_hz}"
    );

    let again = run(
        &Plan::new(&spec("[0]"), &registry).unwrap(),
        &bold(),
        &store,
    )
    .unwrap();
    assert_eq!(
        cached(&again),
        [("modes", true), ("slow", true), ("spectrum", true)]
    );

    let changed = run(
        &Plan::new(&spec("[0, 1]"), &registry).unwrap(),
        &bold(),
        &store,
    )
    .unwrap();
    assert_eq!(
        cached(&changed),
        [("modes", true), ("slow", false), ("spectrum", false)]
    );
}

#[test]
fn invalid_method_params_fail_at_planning() {
    let text = SPEC
        .replace("MODES", "[0]")
        .replace("num_modes = 3", "num_modes = 0");
    let err = Plan::new(
        &PipelineSpec::from_toml(&text).unwrap(),
        &builtin_registry(),
    )
    .err()
    .unwrap();
    assert!(
        matches!(&err, PipelineError::InvalidParams { step, .. } if step == "modes"),
        "{err}"
    );
}

#[test]
fn reconstruct_out_of_range_fails_at_run() {
    let registry = builtin_registry();
    let err = run(
        &Plan::new(&spec("[5]"), &registry).unwrap(),
        &bold(),
        &MemoryStore::new(),
    )
    .err()
    .unwrap();
    assert!(
        matches!(&err, PipelineError::StepFailed { step, .. } if step == "slow"),
        "{err}"
    );
}
