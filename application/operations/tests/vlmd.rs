#![allow(clippy::unwrap_used)]

use std::collections::BTreeMap;
use std::f64::consts::PI;
use std::sync::Arc;

use modewise_epochs::{Event, Events};
use modewise_operations::{VlmdOperation, builtin_registry};
use modewise_pipeline::{
    Artifact, ArtifactStore, Inputs, Operation, PipelineError, PipelineSpec, Plan, run,
};
use modewise_signals::{MultichannelSeries, SamplingRate};
use modewise_store_memory::MemoryStore;
use ndarray::Array2;

const SPEC: &str = r#"
[pipeline]
name = "latent"
inputs = { bold = "series", trials = "events" }

[[step]]
id = "split"
op = "vlmd"
input = "bold"
params = { num_modes = 2, num_latents = 2, alpha = 2000.0, tolerance = 1e-6 }

[[step]]
id = "blocks"
op = "epoch"
input = { data = "split.modes", events = "trials" }
params = { lag_s = 4.0 }
"#;

fn inputs() -> BTreeMap<String, Artifact> {
    let fs = 1.25;
    let data = Array2::from_shape_fn((3, 500), |(c, i)| {
        let t = i as f64 / fs;
        (2.0 * PI * 0.05 * t).cos() * [1.0, 0.5, 0.0][c]
            + (2.0 * PI * 0.25 * t).sin() * [0.0, 0.5, 1.0][c]
    });
    let bold = MultichannelSeries::new(
        data,
        SamplingRate::new(fs).unwrap(),
        vec!["a".into(), "b".into(), "c".into()],
    )
    .unwrap();
    let blocks = [40.0, 200.0]
        .iter()
        .map(|&onset_s| Event {
            onset_s,
            duration_s: 40.0,
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
fn modes_epoch_and_every_output_is_stored() {
    let store = MemoryStore::new();
    let plan = Plan::new(&PipelineSpec::from_toml(SPEC).unwrap(), &builtin_registry()).unwrap();
    let result = run(&plan, &inputs(), &store).unwrap();
    let get = |r: &str| {
        store
            .get(result.output_key(r).unwrap())
            .unwrap()
            .unwrap()
            .as_ref()
            .clone()
    };
    let Artifact::Decomposition(modes) = get("split.modes") else {
        panic!()
    };
    let Artifact::Decomposition(latent) = get("split.latent_modes") else {
        panic!()
    };
    let Artifact::Loadings(loadings) = get("split.loadings") else {
        panic!()
    };
    let Artifact::Epochs(blocks) = get("blocks") else {
        panic!()
    };

    let centres = modes.centre_frequencies_hz();
    assert!(
        (centres[0] - 0.05).abs() < 0.005 && (centres[1] - 0.25).abs() < 0.005,
        "{centres:?}"
    );
    assert_eq!(latent.labels(), ["latent_0", "latent_1"]);
    assert_eq!(loadings.values().dim(), (2, 3));
    assert_eq!(loadings.channel_labels(), ["a", "b", "c"]);
    assert_eq!(blocks.modes(0).unwrap().dim().0, 2);
}

#[test]
fn bad_settings_and_inputs_are_rejected() {
    for bad in [
        SPEC.replace("num_modes = 2,", "num_modes = 0,"),
        SPEC.replace("alpha = 2000.0", "alpha = -1.0"),
        SPEC.replace("tolerance = 1e-6", "tolerance = 1e-6, lambda = 0.1"),
    ] {
        assert!(matches!(
            Plan::new(&PipelineSpec::from_toml(&bad).unwrap(), &builtin_registry()),
            Err(PipelineError::InvalidParams { .. })
        ));
    }
    let too_many = SPEC.replace("num_latents = 2", "num_latents = 4");
    let plan = Plan::new(
        &PipelineSpec::from_toml(&too_many).unwrap(),
        &builtin_registry(),
    )
    .unwrap();
    let err = run(&plan, &inputs(), &MemoryStore::new()).err().unwrap();
    assert!(
        matches!(&err, PipelineError::StepFailed { step, message } if step == "split" && message.contains("num_latents")),
        "{err}"
    );

    let params = VlmdOperation
        .resolve(serde_json::json!({ "num_modes": 1 }))
        .unwrap();
    let defaults =
        ["alpha", "reg_lambda", "reg_rho", "tau", "tolerance"].map(|k| params[k].as_f64().unwrap());
    assert_eq!(defaults, [100.0, 0.01, 1.0, 0.9, 1e-3]);
    assert_eq!(params["max_iterations"], 1000);
    assert_eq!(params["num_latents"], serde_json::Value::Null);
    let events = Arc::new(inputs()["trials"].clone());
    let err = VlmdOperation
        .run(&params, &Inputs::from([("signal".to_string(), events)]))
        .unwrap_err();
    assert!(err.contains("must be a series"));
}
