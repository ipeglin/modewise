#![allow(clippy::unwrap_used)]

use std::collections::BTreeMap;
use std::f64::consts::PI;
use std::sync::Arc;

use modewise_brain::{ChannelInfo, ChannelTable, Hemisphere, Region};
use modewise_connectivity::Measure;
use modewise_operations::{RegionAverageOperation, SelectOperation, builtin_registry};
use modewise_pipeline::{
    Artifact, ArtifactStore, Inputs, Operation, PipelineError, PipelineSpec, Plan, run,
};
use modewise_signals::{MultichannelSeries, SamplingRate};
use modewise_store_memory::MemoryStore;
use ndarray::Array2;

/// Left and right PFCm, left and right amygdala, and a midline channel without a region.
fn table() -> ChannelTable {
    let ch = |name: &str, region: &str, side, net: &str| {
        ChannelInfo::new(name)
            .with_region(Region::new(region))
            .with_hemisphere(side)
            .with_tag("yeo17", net)
    };
    ChannelTable::new(vec![
        ch("L_PFCm", "PFCm", Hemisphere::Left, "DefaultB"),
        ch("R_PFCm", "PFCm", Hemisphere::Right, "DefaultB"),
        ch("lAMY-lh", "lAMY", Hemisphere::Left, "Limbic"),
        ch("lAMY-rh", "lAMY", Hemisphere::Right, "Limbic"),
        ChannelInfo::new("vermis"),
    ])
    .unwrap()
}

fn bold() -> MultichannelSeries {
    let data = Array2::from_shape_fn((5, 300), |(c, t)| {
        let t = t as f64 / 1.25;
        let slow = (2.0 * PI * 0.03 * t).sin();
        let fast = (2.0 * PI * 0.2 * t + c as f64).sin();
        [1.0, 0.9, -0.5, -0.4, 0.0][c] * slow + 0.5 * fast
    });
    MultichannelSeries::new(data, SamplingRate::new(1.25).unwrap(), table().names()).unwrap()
}

fn inputs() -> BTreeMap<String, Artifact> {
    BTreeMap::from([
        ("bold".into(), Artifact::Series(bold())),
        ("channels".into(), Artifact::Channels(table())),
    ])
}

const SPEC: &str = r#"
[pipeline]
name = "regions"
inputs = { bold = "series", channels = "channels" }

[[step]]
id = "rois_first"
op = "select"
input = { data = "bold", channels = "channels" }
params = { rules = [{ region = "PFCm" }, { region = "*AMY*", hemisphere = "left" }] }

[[step]]
id = "modes"
op = "mvmd"
input = "bold"
params = { num_modes = 2, alpha = 2000.0 }

[[step]]
id = "rois_after"
op = "select"
input = { data = "modes", channels = "channels" }
params = { rules = [{ region = "PFCm" }, { region = "*AMY*", hemisphere = "left" }] }

[[step]]
id = "fc"
op = "functional_connectivity"
input = "modes"
params = { measure = "pearson" }

[[step]]
id = "fc_rois"
op = "select"
input = { data = "fc", channels = "channels" }
params = { rules = [{ tags = { yeo17 = "Limbic" } }] }

[[step]]
id = "networks"
op = "region_average"
input = { data = "fc", channels = "channels" }
params = { by = "tag", tag = "yeo17" }

[[step]]
id = "regions"
op = "region_average"
input = { data = "bold", channels = "channels" }
params = { by = "region", split_hemispheres = true }
"#;

fn outputs() -> (MemoryStore, modewise_pipeline::RunResult) {
    let store = MemoryStore::new();
    let plan = Plan::new(&PipelineSpec::from_toml(SPEC).unwrap(), &builtin_registry()).unwrap();
    let result = run(&plan, &inputs(), &store).unwrap();
    (store, result)
}

fn get(store: &MemoryStore, result: &modewise_pipeline::RunResult, r: &str) -> Artifact {
    store
        .get(result.output_key(r).unwrap())
        .unwrap()
        .unwrap()
        .as_ref()
        .clone()
}

#[test]
fn selection_works_before_and_after_decomposition_and_on_connectivity() {
    let (store, result) = outputs();
    let Artifact::Series(first) = get(&store, &result, "rois_first") else {
        panic!()
    };
    assert_eq!(first.labels(), ["L_PFCm", "R_PFCm", "lAMY-lh"]);
    assert_eq!(first.data().row(2), bold().data().row(2));
    let Artifact::Decomposition(after) = get(&store, &result, "rois_after") else {
        panic!()
    };
    let Artifact::Decomposition(modes) = get(&store, &result, "modes") else {
        panic!()
    };
    assert_eq!(after.labels(), first.labels());
    assert_eq!(after.mode(1).row(2), modes.mode(1).row(2));
    let Artifact::FunctionalConnectivity(fc) = get(&store, &result, "fc") else {
        panic!()
    };
    let Artifact::FunctionalConnectivity(limbic) = get(&store, &result, "fc_rois") else {
        panic!()
    };
    assert_eq!(limbic.labels(), ["lAMY-lh", "lAMY-rh"]);
    assert_eq!(limbic.matrix(0, 1)[[0, 1]], fc.matrix(0, 1)[[2, 3]]);
}

#[test]
fn region_averages_match_direct_means() {
    let (store, result) = outputs();
    let Artifact::Series(regions) = get(&store, &result, "regions") else {
        panic!()
    };
    assert_eq!(
        regions.labels(),
        ["PFCm_left", "PFCm_right", "lAMY_left", "lAMY_right"],
        "vermis has no region"
    );
    assert_eq!(regions.data().row(1), bold().data().row(1));

    let Artifact::FunctionalConnectivity(fc) = get(&store, &result, "fc") else {
        panic!()
    };
    let Artifact::FunctionalConnectivity(networks) = get(&store, &result, "networks") else {
        panic!()
    };
    assert_eq!(networks.labels(), ["DefaultB", "Limbic"]);
    assert_eq!(
        networks.measure(),
        Measure::Pearson { fisher_z: true },
        "r is averaged as Fisher z"
    );
    let m = fc.matrix(0, 0);
    let within = m[[0, 1]].atanh();
    let between = [m[[0, 2]], m[[0, 3]], m[[1, 2]], m[[1, 3]]]
        .iter()
        .map(|r| r.atanh())
        .sum::<f64>()
        / 4.0;
    assert!((networks.matrix(0, 0)[[0, 0]] - within).abs() < 1e-12);
    assert!((networks.matrix(0, 0)[[0, 1]] - between).abs() < 1e-12);
    assert!((networks.matrix(0, 0)[[1, 0]] - between).abs() < 1e-12);
}

#[test]
fn averaging_groups_by_hemisphere_and_decompositions() {
    let average = |data: Artifact, params: serde_json::Value| {
        let resolved = RegionAverageOperation.resolve(params).unwrap();
        let inputs = Inputs::from([
            ("data".to_string(), Arc::new(data)),
            (
                "channels".to_string(),
                Arc::new(Artifact::Channels(table())),
            ),
        ]);
        RegionAverageOperation
            .run(&resolved, &inputs)
            .map(|mut o| o.remove("averaged").unwrap())
    };
    let Artifact::Series(sides) = average(
        Artifact::Series(bold()),
        serde_json::json!({ "by": "hemisphere" }),
    )
    .unwrap() else {
        panic!()
    };
    assert_eq!(sides.labels(), ["left", "right"]);
    let x = bold();
    for t in [0, 17, 299] {
        assert!((sides.data()[[0, t]] - (x.data()[[0, t]] + x.data()[[2, t]]) / 2.0).abs() < 1e-12);
    }
    let modes = modewise_decomposition::Decomposition::new(
        ndarray::Array3::from_shape_fn((2, 5, 10), |(k, c, t)| (k * 100 + c * 10 + t) as f64),
        vec![0.03, 0.2],
        SamplingRate::new(1.25).unwrap(),
        table().names(),
        true,
    )
    .unwrap();
    let Artifact::Decomposition(by_net) = average(
        Artifact::Decomposition(modes),
        serde_json::json!({ "by": "tag", "tag": "yeo17" }),
    )
    .unwrap() else {
        panic!()
    };
    assert_eq!(by_net.mode(1)[[1, 4]], (120.0 + 130.0) / 2.0 + 4.0);
    let per_channel = modewise_decomposition::Decomposition::per_channel(
        ndarray::Array3::zeros((1, 5, 10)),
        ndarray::Array2::from_shape_fn((1, 5), |(_, c)| 0.1 * (c + 1) as f64),
        SamplingRate::new(1.25).unwrap(),
        table().names(),
    )
    .unwrap();
    let Artifact::Decomposition(sides) = average(
        Artifact::Decomposition(per_channel),
        serde_json::json!({ "by": "hemisphere" }),
    )
    .unwrap() else {
        panic!()
    };
    let averaged = sides.channel_centre_frequencies_hz().unwrap();
    assert!(
        (averaged[[0, 0]] - (0.1 + 0.3) / 2.0).abs() < 1e-12,
        "per-channel frequencies are averaged too"
    );
    assert!(!sides.aligned_across_channels());
    let none = average(
        Artifact::Series(bold()),
        serde_json::json!({ "by": "tag", "tag": "lobe" }),
    );
    assert!(none.unwrap_err().contains("no channel"));
}

#[test]
fn bad_rules_inputs_and_settings_are_rejected() {
    for bad in [
        serde_json::json!({ "rules": [] }),
        serde_json::json!({ "rules": [{ "lobe": "frontal" }] }),
        serde_json::json!({ "rules": [{ "hemisphere": "middle" }] }),
    ] {
        assert!(SelectOperation.resolve(bad).is_err());
    }
    for bad in [
        serde_json::json!({ "by": "tag" }),
        serde_json::json!({ "by": "region", "tag": "yeo17" }),
        serde_json::json!({ "by": "hemisphere", "split_hemispheres": true }),
        serde_json::json!({ "by": "lobe" }),
    ] {
        assert!(RegionAverageOperation.resolve(bad).is_err());
    }
    let select = |data: Artifact, rules: serde_json::Value| {
        let resolved = SelectOperation
            .resolve(serde_json::json!({ "rules": rules }))
            .unwrap();
        let inputs = Inputs::from([
            ("data".to_string(), Arc::new(data)),
            (
                "channels".to_string(),
                Arc::new(Artifact::Channels(table())),
            ),
        ]);
        SelectOperation
            .run(&resolved, &inputs)
            .map(|mut o| o.remove("selected").unwrap())
    };
    assert!(
        select(
            Artifact::Series(bold()),
            serde_json::json!([{ "region": "Nowhere" }])
        )
        .unwrap_err()
        .contains("matches no channel")
    );
    let unknown = MultichannelSeries::new(
        Array2::zeros((1, 4)),
        SamplingRate::new(1.0).unwrap(),
        vec!["x".into()],
    )
    .unwrap();
    assert!(
        select(Artifact::Series(unknown), serde_json::json!([{}]))
            .unwrap_err()
            .contains("not in the channel table")
    );
    let Artifact::Channels(t) = select(
        Artifact::Channels(table()),
        serde_json::json!([{ "hemisphere": "right" }]),
    )
    .unwrap() else {
        panic!()
    };
    assert_eq!(t.names(), ["R_PFCm", "lAMY-rh"]);
    let loadings = modewise_decomposition::Loadings::new(
        Array2::from_shape_fn((2, 5), |(l, c)| (l * 5 + c) as f64),
        vec!["a".into(), "b".into()],
        table().names(),
    )
    .unwrap();
    let Artifact::Loadings(l) = select(
        Artifact::Loadings(loadings),
        serde_json::json!([{ "region": "PFCm" }]),
    )
    .unwrap() else {
        panic!()
    };
    assert_eq!(l.values(), ndarray::array![[0.0, 1.0], [5.0, 6.0]]);
    let events = modewise_epochs::Events::new(vec![]).unwrap();
    assert!(
        select(Artifact::Events(events), serde_json::json!([{}]))
            .unwrap_err()
            .contains("has no channels")
    );
    let wrong = Inputs::from([
        ("data".to_string(), Arc::new(Artifact::Series(bold()))),
        ("channels".to_string(), Arc::new(Artifact::Series(bold()))),
    ]);
    let resolved = SelectOperation
        .resolve(serde_json::json!({ "rules": [{}] }))
        .unwrap();
    assert!(
        SelectOperation
            .run(&resolved, &wrong)
            .unwrap_err()
            .contains("channel table")
    );
}

#[test]
fn a_plan_rejects_a_selection_without_channels() {
    let no_channels = SPEC.replace(
        "inputs = { bold = \"series\", channels = \"channels\" }",
        "inputs = { bold = \"series\" }",
    );
    let err = Plan::new(
        &PipelineSpec::from_toml(&no_channels).unwrap(),
        &builtin_registry(),
    )
    .err()
    .unwrap();
    assert!(err.to_string().contains("channels"), "{err}");
}

#[test]
fn the_plan_knows_a_selection_keeps_the_kind_it_was_given() {
    let wrong = SPEC.replace(
        "[[step]]\nid = \"fc\"",
        "[[step]]\nid = \"bad\"\nop = \"reconstruct\"\ninput = \"rois_first\"\nparams = { modes = [0] }\n\n[[step]]\nid = \"fc\"",
    );
    let err = Plan::new(
        &PipelineSpec::from_toml(&wrong).unwrap(),
        &builtin_registry(),
    )
    .err()
    .unwrap();
    assert!(
        matches!(&err, PipelineError::KindMismatch { step, found, .. } if step == "bad" && found.to_string() == "series"),
        "{err}"
    );
}

#[test]
fn epochs_are_selected_and_averaged_through_their_source() {
    use modewise_epochs::{EpochParams, EpochSource, Epochs, Event, Events, Window};

    let events = Events::new(vec![Event {
        onset_s: 40.0,
        duration_s: 40.0,
        label: "x".into(),
    }])
    .unwrap();
    let params = EpochParams {
        lag_s: 0.0,
        window: Window::Duration,
        conditions: None,
        min_cycles: 1.0,
    };
    let epochs = Epochs::cut(EpochSource::Series(bold()), &events, params).unwrap();
    let with_channels = |data: Artifact| {
        Inputs::from([
            ("data".to_string(), Arc::new(data)),
            (
                "channels".to_string(),
                Arc::new(Artifact::Channels(table())),
            ),
        ])
    };

    let select = SelectOperation
        .resolve(serde_json::json!({ "rules": [{ "region": "*AMY*" }] }))
        .unwrap();
    let Artifact::Epochs(amy) = SelectOperation
        .run(&select, &with_channels(Artifact::Epochs(epochs.clone())))
        .unwrap()
        .remove("selected")
        .unwrap()
    else {
        panic!()
    };
    assert_eq!(amy.series(0).unwrap().dim().0, 2);
    assert_eq!(amy.windows(), epochs.windows());
    assert_eq!(
        amy.series(0).unwrap().row(0),
        epochs.series(0).unwrap().row(2)
    );

    let average = RegionAverageOperation
        .resolve(serde_json::json!({ "by": "region" }))
        .unwrap();
    let Artifact::Epochs(regions) = RegionAverageOperation
        .run(&average, &with_channels(Artifact::Epochs(epochs.clone())))
        .unwrap()
        .remove("averaged")
        .unwrap()
    else {
        panic!()
    };
    let pfcm = (&epochs.series(0).unwrap().row(0) + &epochs.series(0).unwrap().row(1)) / 2.0;
    assert_eq!(regions.series(0).unwrap().row(0), pfcm);
}
