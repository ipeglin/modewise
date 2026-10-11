#![allow(clippy::unwrap_used)]

use std::collections::BTreeMap;
use std::f64::consts::PI;

use modewise_epochs::{Event, Events};
use modewise_operations::builtin_registry;
use modewise_pipeline::{
    Artifact, ArtifactStore, PipelineError, PipelineSpec, Plan, RunResult, run,
};
use modewise_signals::{MultichannelSeries, SamplingRate};
use modewise_store_memory::MemoryStore;
use ndarray::{Array2, s};

const FS: f64 = 1.25;
const SLOW_HZ: f64 = 0.02;
const FAST_HZ: f64 = 0.2;

fn spec(epoch_params: &str) -> PipelineSpec {
    PipelineSpec::from_toml(&format!(
        r#"
[pipeline]
name = "blocks"
inputs = {{ bold = "series", trials = "events" }}

[[step]]
id = "modes"
op = "mvmd"
input = "bold"
params = {{ num_modes = 2, alpha = 2000.0, init = "linear", tau = 0.0, tolerance = 1e-7 }}

[[step]]
id = "blocks"
op = "epoch"
input = {{ data = "modes", events = "trials" }}
params = {epoch_params}
"#
    ))
    .unwrap()
}

fn inputs() -> BTreeMap<String, Artifact> {
    let n = 600;
    let data = Array2::from_shape_fn((2, n), |(c, i)| {
        let t = i as f64 / FS;
        (1.0 + c as f64) * (2.0 * PI * SLOW_HZ * t).sin() + (2.0 * PI * FAST_HZ * t).sin()
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
            label: if b % 2 == 0 { "face" } else { "shape" }.into(),
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
fn full_run_decomposition_is_cut_with_lag_and_resolvability() {
    let store = MemoryStore::new();
    let result = run(
        &Plan::new(&spec("{ lag_s = 4.0 }"), &builtin_registry()).unwrap(),
        &inputs(),
        &store,
    )
    .unwrap();
    let Artifact::Decomposition(full) = output(&store, &result, "modes") else {
        panic!()
    };
    let Artifact::Epochs(epochs) = output(&store, &result, "blocks") else {
        panic!()
    };

    let freqs = full.centre_frequencies_hz();
    assert!(
        (freqs[0] - SLOW_HZ).abs() < 0.005 && (freqs[1] - FAST_HZ).abs() < 0.01,
        "{freqs:?}"
    );
    assert_eq!(epochs.windows().len(), 10);

    let resolvable = epochs.resolvable().unwrap();
    for i in 0..epochs.windows().len() {
        assert!(
            !resolvable[[i, 0]],
            "the 50 s slow mode cannot fit in an 18 s block"
        );
        assert!(resolvable[[i, 1]], "the 5 s fast mode fits 3.6 cycles");
        let w = &epochs.windows()[i];
        assert_eq!(
            epochs.modes(i).unwrap(),
            full.modes().slice(s![.., .., w.start..w.end]),
            "epochs are slices (E5)"
        );
    }
    let first = &epochs.windows()[0];
    assert_eq!(first.start, 30, "onset 20 s + lag 4 s = 24 s = sample 30");
}

#[test]
fn changing_the_lag_reruns_only_the_epoch_step() {
    let store = MemoryStore::new();
    let registry = builtin_registry();
    run(
        &Plan::new(&spec("{ lag_s = 4.0 }"), &registry).unwrap(),
        &inputs(),
        &store,
    )
    .unwrap();
    let swept = run(
        &Plan::new(&spec("{ lag_s = 6.0 }"), &registry).unwrap(),
        &inputs(),
        &store,
    )
    .unwrap();
    let cached: Vec<bool> = swept.manifest.steps.iter().map(|s| s.cached).collect();
    assert_eq!(cached, [true, false]);
}

#[test]
fn epochs_cannot_be_decomposed() {
    let mut text = String::from(
        "[pipeline]\nname = \"bad\"\ninputs = { bold = \"series\", trials = \"events\" }\n\
         [[step]]\nid = \"blocks\"\nop = \"epoch\"\ninput = { data = \"bold\", events = \"trials\" }\nparams = { lag_s = 0.0 }\n",
    );
    text.push_str("[[step]]\nid = \"modes\"\nop = \"mvmd\"\ninput = \"blocks\"\nparams = { num_modes = 2, alpha = 1000.0 }\n");
    let err = Plan::new(
        &PipelineSpec::from_toml(&text).unwrap(),
        &builtin_registry(),
    )
    .err()
    .unwrap();
    assert!(
        matches!(&err, PipelineError::KindMismatch { step, .. } if step == "modes"),
        "{err}"
    );
}

#[test]
fn lag_is_required_and_inputs_must_have_the_right_kind() {
    let err = Plan::new(&spec("{ window = \"duration\" }"), &builtin_registry())
        .err()
        .unwrap();
    assert!(
        matches!(&err, PipelineError::InvalidParams { step, message } if step == "blocks" && message.contains("lag_s")),
        "{err}"
    );
    assert!(
        Plan::new(
            &spec("{ lag_s = 1.0, min_cycles = 0.0 }"),
            &builtin_registry()
        )
        .is_err()
    );
    assert!(
        Plan::new(
            &spec("{ lag_s = 1.0, merge_gap_s = -1.0 }"),
            &builtin_registry()
        )
        .is_err()
    );
    assert!(
        Plan::new(
            &spec("{ lag_s = 1.0, window = { tmin_s = 1.0, tmax_s = 0.5 } }"),
            &builtin_registry()
        )
        .is_err()
    );

    let mut wrong = inputs();
    let bold = wrong["bold"].clone();
    wrong.insert("trials".into(), bold);
    let err = run(
        &Plan::new(&spec("{ lag_s = 0.0 }"), &builtin_registry()).unwrap(),
        &wrong,
        &MemoryStore::new(),
    )
    .err()
    .unwrap();
    assert!(
        matches!(&err, PipelineError::InputKind { name, .. } if name == "trials"),
        "{err}"
    );
}

#[test]
fn trials_merge_into_blocks_and_fixed_windows_work_on_series() {
    let trials = (0..6)
        .map(|i| Event {
            onset_s: 20.0 + 3.0 * i as f64,
            duration_s: 2.0,
            label: "face".into(),
        })
        .collect();
    let mut data = inputs();
    data.insert(
        "trials".into(),
        Artifact::Events(Events::new(trials).unwrap()),
    );
    let text = "[pipeline]\nname = \"m\"\ninputs = { bold = \"series\", trials = \"events\" }\n\
        [[step]]\nid = \"blocks\"\nop = \"epoch\"\ninput = { data = \"bold\", events = \"trials\" }\nparams = { lag_s = 0.0, merge_gap_s = 1.5 }\n\
        [[step]]\nid = \"trials\"\nop = \"epoch\"\ninput = { data = \"bold\", events = \"trials\" }\nparams = { lag_s = 0.0, window = { tmin_s = 0.0, tmax_s = 1.6 } }\n";
    let err = Plan::new(&PipelineSpec::from_toml(text).unwrap(), &builtin_registry());
    assert!(
        matches!(err, Err(PipelineError::DuplicateName(_))),
        "step id clashes with input name"
    );

    let text = text.replace("id = \"trials\"\nop", "id = \"per_trial\"\nop");
    let store = MemoryStore::new();
    let result = run(
        &Plan::new(
            &PipelineSpec::from_toml(&text).unwrap(),
            &builtin_registry(),
        )
        .unwrap(),
        &data,
        &store,
    )
    .unwrap();
    let Artifact::Epochs(blocks) = output(&store, &result, "blocks") else {
        panic!()
    };
    let Artifact::Epochs(per_trial) = output(&store, &result, "per_trial") else {
        panic!()
    };
    assert_eq!(
        blocks.windows().len(),
        1,
        "six trials 1 s apart merge into one block"
    );
    assert!(
        (blocks.duration_s(0) - 17.6).abs() < 1e-9,
        "20 s to 37 s, sampled at 1.25 Hz"
    );
    assert_eq!(per_trial.windows().len(), 6);
    assert!(
        per_trial.resolvable().is_none(),
        "a series has no frequencies to judge"
    );
}

#[test]
fn epoch_operation_rejects_wrong_input_kinds() {
    use std::sync::Arc;

    use modewise_operations::EpochOperation;
    use modewise_pipeline::{Inputs, Operation};

    let params = EpochOperation
        .resolve(serde_json::json!({ "lag_s": 0.0 }))
        .unwrap();
    let data = inputs();
    let events = Arc::new(data["trials"].clone());
    let bold = Arc::new(data["bold"].clone());

    let events_as_data = Inputs::from([
        ("data".to_string(), events.clone()),
        ("events".to_string(), events.clone()),
    ]);
    assert!(
        EpochOperation
            .run(&params, &events_as_data)
            .unwrap_err()
            .contains("cannot be epoched")
    );
    let series_as_events = Inputs::from([
        ("data".to_string(), bold.clone()),
        ("events".to_string(), bold),
    ]);
    assert!(
        EpochOperation
            .run(&params, &series_as_events)
            .unwrap_err()
            .contains("must be events")
    );
}
