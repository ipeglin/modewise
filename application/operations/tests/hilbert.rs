#![allow(clippy::unwrap_used)]

use std::collections::BTreeMap;
use std::f64::consts::PI;

use modewise_epochs::{Event, Events};
use modewise_hilbert::{Padding, instantaneous};
use modewise_operations::builtin_registry;
use modewise_pipeline::{
    Artifact, ArtifactStore, PipelineError, PipelineSpec, Plan, RunResult, run,
};
use modewise_signals::{MultichannelSeries, SamplingRate};
use modewise_store_memory::MemoryStore;
use ndarray::{Array2, s};

const FS: f64 = 1.25;

const SPEC: &str = r#"
[pipeline]
name = "hht"
inputs = { bold = "series", trials = "events" }

[[step]]
id = "modes"
op = "mvmd"
input = "bold"
params = { num_modes = 2, alpha = 2000.0, init = "linear", tau = 0.0, tolerance = 1e-7 }

[[step]]
id = "attrs"
op = "hilbert"
input = "modes"

[[step]]
id = "spectrum"
op = "hilbert_spectrum"
input = "attrs"
params = { f_min = 0.005, f_max = 0.6, n_bins = 40, scale = "log" }

[[step]]
id = "blocks"
op = "epoch"
input = { data = "attrs", events = "trials" }
params = { lag_s = 4.0 }
"#;

fn inputs() -> BTreeMap<String, Artifact> {
    let data = Array2::from_shape_fn((2, 600), |(c, i)| {
        let t = i as f64 / FS;
        (1.0 + c as f64) * (2.0 * PI * 0.02 * t).sin() + (2.0 * PI * 0.2 * t).sin()
    });
    let bold = MultichannelSeries::new(
        data,
        SamplingRate::new(FS).unwrap(),
        vec!["l".into(), "r".into()],
    )
    .unwrap();
    let blocks = (0..10)
        .map(|b| Event {
            onset_s: 20.0 + 40.0 * b as f64,
            duration_s: 18.0,
            label: "face".into(),
        })
        .collect();
    BTreeMap::from([
        ("bold".to_string(), Artifact::Series(bold)),
        (
            "trials".to_string(),
            Artifact::Events(Events::new(blocks).unwrap()),
        ),
    ])
}

fn output(store: &MemoryStore, result: &RunResult, reference: &str) -> Artifact {
    store
        .get(result.output_key(reference).unwrap())
        .unwrap()
        .unwrap()
        .as_ref()
        .clone()
}

#[test]
fn full_run_hilbert_feeds_spectrum_and_epochs() {
    let store = MemoryStore::new();
    let result = run(
        &Plan::new(&PipelineSpec::from_toml(SPEC).unwrap(), &builtin_registry()).unwrap(),
        &inputs(),
        &store,
    )
    .unwrap();
    let Artifact::Decomposition(modes) = output(&store, &result, "modes") else {
        panic!()
    };
    let Artifact::Instantaneous(attrs) = output(&store, &result, "attrs") else {
        panic!()
    };
    let Artifact::TimeFrequency(spectrum) = output(&store, &result, "spectrum") else {
        panic!()
    };
    let Artifact::Epochs(blocks) = output(&store, &result, "blocks") else {
        panic!()
    };

    assert_eq!(
        attrs,
        instantaneous(&modes, Padding::Mirror).unwrap(),
        "the step equals the method"
    );
    let middle = attrs.frequency_hz().slice(s![.., 0, 250]).to_vec();
    assert!(
        (middle[0] - 0.02).abs() < 0.002 && (middle[1] - 0.2).abs() < 0.01,
        "{middle:?}"
    );
    assert_eq!(spectrum.power().dim(), (2, 40, 600));

    let resolvable = blocks.resolvable().unwrap();
    assert!(
        !resolvable[[0, 0]] && resolvable[[0, 1]],
        "same rule as for modes"
    );
    let (amplitude, frequency) = blocks.instantaneous(0).unwrap();
    let w = &blocks.windows()[0];
    assert_eq!(
        amplitude,
        attrs.amplitude().slice(s![.., .., w.start..w.end])
    );
    assert_eq!(
        frequency,
        attrs.frequency_hz().slice(s![.., .., w.start..w.end])
    );
    assert!(blocks.modes(0).is_none() && blocks.series(0).is_none());
}

#[test]
fn spectrum_settings_rerun_only_the_spectrum() {
    let store = MemoryStore::new();
    let registry = builtin_registry();
    run(
        &Plan::new(&PipelineSpec::from_toml(SPEC).unwrap(), &registry).unwrap(),
        &inputs(),
        &store,
    )
    .unwrap();
    let finer = SPEC.replace("n_bins = 40", "n_bins = 80");
    let again = run(
        &Plan::new(&PipelineSpec::from_toml(&finer).unwrap(), &registry).unwrap(),
        &inputs(),
        &store,
    )
    .unwrap();
    let cached: Vec<(&str, bool)> = again
        .manifest
        .steps
        .iter()
        .map(|s| (s.id.as_str(), s.cached))
        .collect();
    assert_eq!(
        cached,
        [
            ("modes", true),
            ("attrs", true),
            ("spectrum", false),
            ("blocks", true)
        ]
    );
}

#[test]
fn invalid_hilbert_settings_fail_at_planning() {
    let plan = |text: String| {
        Plan::new(
            &PipelineSpec::from_toml(&text).unwrap(),
            &builtin_registry(),
        )
    };
    for bad in [
        SPEC.replace("scale = \"log\"", "scale = \"mel\""),
        SPEC.replace("f_min = 0.005", "f_min = 0.0"),
        SPEC.replace("n_bins = 40", "n_bins = 0"),
        SPEC.replace(
            "input = \"modes\"\n",
            "input = \"modes\"\nparams = { padding = \"zero\" }\n",
        ),
    ] {
        assert!(matches!(
            plan(bad),
            Err(PipelineError::InvalidParams { .. })
        ));
    }
    let raw_series = SPEC.replace(
        "id = \"attrs\"\nop = \"hilbert\"\ninput = \"modes\"",
        "id = \"attrs\"\nop = \"hilbert\"\ninput = \"bold\"",
    );
    assert!(
        matches!(plan(raw_series), Err(PipelineError::KindMismatch { .. })),
        "hilbert needs modes, not a raw series"
    );
}

#[test]
fn operations_run_directly_with_every_option() {
    use std::sync::Arc;

    use modewise_operations::{HilbertOperation, HilbertSpectrumOperation};
    use modewise_pipeline::{Inputs, Operation};
    use serde_json::json;

    let store = MemoryStore::new();
    let result = run(
        &Plan::new(&PipelineSpec::from_toml(SPEC).unwrap(), &builtin_registry()).unwrap(),
        &inputs(),
        &store,
    )
    .unwrap();
    let modes = Arc::new(output(&store, &result, "modes"));
    let attrs = Arc::new(output(&store, &result, "attrs"));

    let plain = HilbertOperation
        .resolve(json!({ "padding": "none" }))
        .unwrap();
    let out = HilbertOperation
        .run(
            &plain,
            &Inputs::from([("modes".to_string(), modes.clone())]),
        )
        .unwrap();
    assert!(matches!(out["attributes"], Artifact::Instantaneous(_)));
    let wrong = Inputs::from([("modes".to_string(), attrs.clone())]);
    assert!(
        HilbertOperation
            .run(&plain, &wrong)
            .unwrap_err()
            .contains("must be a decomposition")
    );

    let linear = HilbertSpectrumOperation
        .resolve(json!({ "f_min": 0.0, "f_max": 0.6, "n_bins": 6, "weight": "amplitude" }))
        .unwrap();
    let out = HilbertSpectrumOperation
        .run(&linear, &Inputs::from([("attributes".to_string(), attrs)]))
        .unwrap();
    let Artifact::TimeFrequency(map) = &out["power"] else {
        panic!()
    };
    let expected = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55];
    assert!(
        map.frequencies_hz()
            .iter()
            .zip(expected)
            .all(|(a, b)| (a - b).abs() < 1e-12),
        "{:?}",
        map.frequencies_hz()
    );
    let wrong = Inputs::from([("attributes".to_string(), modes)]);
    assert!(
        HilbertSpectrumOperation
            .run(&linear, &wrong)
            .unwrap_err()
            .contains("must be instantaneous")
    );
}
