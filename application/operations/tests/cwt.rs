#![allow(clippy::unwrap_used)]

use std::collections::BTreeMap;
use std::f64::consts::PI;

use modewise_epochs::{Event, Events};
use modewise_operations::builtin_registry;
use modewise_pipeline::{Artifact, ArtifactStore, PipelineError, PipelineSpec, Plan, run};
use modewise_signals::{MultichannelSeries, SamplingRate};
use modewise_store_memory::MemoryStore;
use ndarray::{Array2, s};

const SPEC: &str = r#"
[pipeline]
name = "wavelets"
inputs = { bold = "series", trials = "events" }

[[step]]
id = "scalogram"
op = "cwt"
input = "bold"
params = { f_min = 0.01, f_max = 0.6, n_bins = 30 }

[[step]]
id = "blocks"
op = "epoch"
input = { data = "scalogram", events = "trials" }
params = { lag_s = 4.0 }
"#;

fn inputs() -> BTreeMap<String, Artifact> {
    let fs = 1.25;
    let data = Array2::from_shape_fn((1, 600), |(_, i)| (2.0 * PI * 0.2 * i as f64 / fs).sin());
    let bold =
        MultichannelSeries::new(data, SamplingRate::new(fs).unwrap(), vec!["a".into()]).unwrap();
    let blocks = [0.0, 200.0, 470.0]
        .iter()
        .map(|&onset_s| Event {
            onset_s,
            duration_s: 18.0,
            label: "x".into(),
        })
        .collect();
    BTreeMap::from([
        ("bold".into(), Artifact::Series(bold)),
        (
            "trials".into(),
            Artifact::Events(Events::new(blocks).unwrap()),
        ),
    ])
}

#[test]
fn scalogram_epochs_respect_cycles_and_the_cone_of_influence() {
    let store = MemoryStore::new();
    let result = run(
        &Plan::new(&PipelineSpec::from_toml(SPEC).unwrap(), &builtin_registry()).unwrap(),
        &inputs(),
        &store,
    )
    .unwrap();
    let get = |r: &str| {
        store
            .get(result.output_key(r).unwrap())
            .unwrap()
            .unwrap()
            .as_ref()
            .clone()
    };
    let Artifact::TimeFrequency(map) = get("scalogram") else {
        panic!()
    };
    let Artifact::Epochs(blocks) = get("blocks") else {
        panic!()
    };

    let freqs = map.frequencies_hz().to_vec();
    let row = |f: f64| {
        freqs
            .iter()
            .position(|&x| (x - f).abs() < f * 0.07)
            .unwrap()
    };
    let r = blocks.resolvable().unwrap();
    let (fast, slow) = (row(0.2), row(0.02));
    assert!(
        !r[[0, fast]],
        "the first block starts at sample 5 (0 s + 4 s lag); at 0.2 Hz the cone covers samples 0..=8"
    );
    assert!(
        r[[1, fast]],
        "a mid-run block fits 3.6 cycles of 0.2 Hz, clear of edges"
    );
    assert!(!r[[1, slow]], "0.02 Hz cannot fit in 18 s");
    let w = &blocks.windows()[1];
    assert_eq!(
        blocks.time_frequency(1).unwrap(),
        map.power().slice(s![.., .., w.start..w.end])
    );
}

#[test]
fn stft_frames_cannot_be_epoched_and_cwt_settings_are_checked() {
    let stft = SPEC.replace(
        "op = \"cwt\"\ninput = \"bold\"\nparams = { f_min = 0.01, f_max = 0.6, n_bins = 30 }",
        "op = \"stft\"\ninput = \"bold\"\nparams = { window_len = 32, hop = 16 }",
    );
    let err = run(
        &Plan::new(
            &PipelineSpec::from_toml(&stft).unwrap(),
            &builtin_registry(),
        )
        .unwrap(),
        &inputs(),
        &MemoryStore::new(),
    )
    .err()
    .unwrap();
    assert!(
        matches!(&err, PipelineError::StepFailed { step, message } if step == "blocks" && message.contains("one time column per sample")),
        "{err}"
    );

    for bad in [
        SPEC.replace("f_min = 0.01", "f_min = 0.0"),
        SPEC.replace("n_bins = 30", "n_bins = 30, omega0 = 2.0"),
        SPEC.replace("n_bins = 30", "n_bins = 30, morlet = true"),
    ] {
        assert!(matches!(
            Plan::new(&PipelineSpec::from_toml(&bad).unwrap(), &builtin_registry()),
            Err(PipelineError::InvalidParams { .. })
        ));
    }
    let above_nyquist = SPEC.replace("f_max = 0.6", "f_max = 2.0");
    let err = run(
        &Plan::new(
            &PipelineSpec::from_toml(&above_nyquist).unwrap(),
            &builtin_registry(),
        )
        .unwrap(),
        &inputs(),
        &MemoryStore::new(),
    )
    .err()
    .unwrap();
    assert!(
        matches!(&err, PipelineError::StepFailed { step, .. } if step == "scalogram"),
        "{err}"
    );
}

#[test]
fn cwt_operation_rejects_non_series_input() {
    use std::sync::Arc;

    use modewise_operations::CwtOperation;
    use modewise_pipeline::{Inputs, Operation};

    let params = CwtOperation
        .resolve(serde_json::json!({ "f_min": 0.05, "f_max": 0.5, "n_bins": 4, "rectify": true }))
        .unwrap();
    let events = Arc::new(inputs()["trials"].clone());
    let err = CwtOperation
        .run(&params, &Inputs::from([("signal".to_string(), events)]))
        .unwrap_err();
    assert!(err.contains("must be a series"));
}
