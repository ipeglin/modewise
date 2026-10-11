#![allow(clippy::unwrap_used)]

use std::collections::BTreeMap;
use std::f64::consts::PI;
use std::sync::Arc;

use modewise_connectivity::Measure;
use modewise_operations::{FunctionalConnectivityOperation, builtin_registry};
use modewise_pipeline::{
    Artifact, ArtifactStore, Inputs, Operation, PipelineError, PipelineSpec, Plan, run,
};
use modewise_signals::{MultichannelSeries, SamplingRate};
use modewise_store_memory::MemoryStore;
use ndarray::Array2;

/// Slow-band connectivity the way band-pass studies do it: sum the modes in the band, then correlate.
const SPEC: &str = r#"
[pipeline]
name = "band-fc"
inputs = { bold = "series" }

[[step]]
id = "modes"
op = "mvmd"
input = "bold"
params = { num_modes = 2, alpha = 2000.0 }

[[step]]
id = "slow"
op = "reconstruct"
input = "modes"
params = { modes = [0] }

[[step]]
id = "fc"
op = "functional_connectivity"
input = "slow"
params = { measure = "pearson", fisher_z = true }

[[step]]
id = "per_mode"
op = "functional_connectivity"
input = "modes"
params = { measure = "plv" }
"#;

fn bold() -> BTreeMap<String, Artifact> {
    // Channels a and b share a slow rhythm in anti-phase; c shares only the fast one.
    let data = Array2::from_shape_fn((3, 400), |(c, t)| {
        let t = t as f64 / 1.25;
        let slow = (2.0 * PI * 0.03 * t).sin() * [1.0, -1.0, 0.0][c];
        slow + (2.0 * PI * 0.2 * t + c as f64).sin()
    });
    let bold = MultichannelSeries::new(
        data,
        SamplingRate::new(1.25).unwrap(),
        vec!["a".into(), "b".into(), "c".into()],
    )
    .unwrap();
    BTreeMap::from([("bold".into(), Artifact::Series(bold))])
}

#[test]
fn band_and_per_mode_connectivity_run_in_a_pipeline() {
    let store = MemoryStore::new();
    let plan = Plan::new(&PipelineSpec::from_toml(SPEC).unwrap(), &builtin_registry()).unwrap();
    let result = run(&plan, &bold(), &store).unwrap();
    let get = |r: &str| {
        store
            .get(result.output_key(r).unwrap())
            .unwrap()
            .unwrap()
            .as_ref()
            .clone()
    };
    let Artifact::FunctionalConnectivity(band) = get("fc") else {
        panic!()
    };
    let Artifact::FunctionalConnectivity(per_mode) = get("per_mode") else {
        panic!()
    };

    assert_eq!(band.measure(), Measure::Pearson { fisher_z: true });
    assert!(
        band.matrix(0, 0)[[0, 1]] < -2.0,
        "a and b are anti-correlated in the slow band: z = {}",
        band.matrix(0, 0)[[0, 1]]
    );
    assert_eq!(per_mode.values().dim(), (1, 2, 3, 3));
    assert!(per_mode.centre_frequencies_hz().unwrap()[0] < 0.1);
    assert!(
        per_mode.matrix(0, 1)[[0, 2]] > 0.9,
        "every channel carries the fast rhythm"
    );
}

#[test]
fn settings_and_inputs_are_checked() {
    let plv_z = SPEC.replace("measure = \"plv\"", "measure = \"plv\", fisher_z = true");
    let unknown = SPEC.replace("measure = \"plv\"", "measure = \"coherence\"");
    for bad in [plv_z, unknown] {
        assert!(matches!(
            Plan::new(&PipelineSpec::from_toml(&bad).unwrap(), &builtin_registry()),
            Err(PipelineError::InvalidParams { .. })
        ));
    }
    let params = FunctionalConnectivityOperation
        .resolve(serde_json::json!({ "measure": "pearson" }))
        .unwrap();
    assert_eq!(params["fisher_z"], false);
    let loadings = modewise_decomposition::Loadings::new(
        Array2::zeros((1, 1)),
        vec!["l".into()],
        vec!["a".into()],
    )
    .unwrap();
    let inputs = Inputs::from([("data".to_string(), Arc::new(Artifact::Loadings(loadings)))]);
    assert!(
        FunctionalConnectivityOperation
            .run(&params, &inputs)
            .unwrap_err()
            .contains("cannot be a loadings")
    );
}
