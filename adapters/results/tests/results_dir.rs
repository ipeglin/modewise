#![allow(clippy::unwrap_used)]

use std::collections::BTreeMap;
use std::f64::consts::PI;
use std::path::{Path, PathBuf};

use modewise_operations::builtin_registry;
use modewise_pipeline::{Artifact, InputSource, PipelineSpec, Plan};
use modewise_results::{ResultsDir, ResultsError};
use modewise_signals::{MultichannelSeries, SamplingRate};
use ndarray::Array2;

const SPEC: &str = r#"
[pipeline]
name = "band power"
inputs = ["bold"]

[[step]]
id = "modes"
op = "mvmd"
input = "bold"
params = { num_modes = 2, alpha = 2000.0, init = "linear" }

[[step]]
id = "power"
op = "stft"
input = "bold"
params = { window_len = 32, hop = 16 }
"#;

struct Tones;

impl InputSource for Tones {
    fn input_names(&self) -> Vec<String> {
        vec!["bold".into()]
    }

    fn recording_ids(&self) -> Vec<String> {
        vec!["sub-01".into(), "sub-02".into()]
    }

    fn load(&self, id: &str) -> Result<BTreeMap<String, Artifact>, String> {
        let f = if id == "sub-01" { 3.0 } else { 5.0 };
        let data = Array2::from_shape_fn((2, 128), |(c, t)| {
            (c + 1) as f64 * (2.0 * PI * f * t as f64 / 32.0).sin()
        });
        let series = MultichannelSeries::new(
            data,
            SamplingRate::new(32.0).unwrap(),
            vec!["l".into(), "r".into()],
        )
        .unwrap();
        Ok(BTreeMap::from([("bold".into(), Artifact::Series(series))]))
    }
}

fn temp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("modewise-results-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

fn run_into(dir: &Path) {
    let results = ResultsDir::open(dir).unwrap();
    let plan = Plan::new(&PipelineSpec::from_toml(SPEC).unwrap(), &builtin_registry()).unwrap();
    let summary = results.run_batch(&plan, &Tones, None, 2, &|_| {}).unwrap();
    assert_eq!(summary.succeeded, ["sub-01", "sub-02"]);
}

#[test]
fn saved_runs_reopen_with_their_outputs() {
    let dir = temp_dir("reopen");
    run_into(&dir);

    let results = ResultsDir::open(&dir).unwrap();
    assert_eq!(results.pipelines(), ["band_power"]);
    assert_eq!(results.recordings("band power"), ["sub-01", "sub-02"]);
    let Artifact::Decomposition(modes) = results
        .output("band power", "sub-02", "modes")
        .unwrap()
        .as_ref()
        .clone()
    else {
        panic!("modes should be a decomposition");
    };
    assert!(
        (modes.centre_frequencies_hz()[1] - 5.0).abs() < 0.5,
        "{:?}",
        modes.centre_frequencies_hz()
    );
    assert!(matches!(
        results.output("band power", "sub-02", "nope"),
        Err(ResultsError::MissingOutput { .. })
    ));
    assert!(matches!(
        results.output("band power", "sub-09", "modes"),
        Err(ResultsError::MissingRun { .. })
    ));
    assert!(dir.join("runs/band_power/summaries/all.json").exists());
}

#[test]
fn exports_one_hdf5_file_per_recording() {
    let dir = temp_dir("export");
    run_into(&dir);
    let results = ResultsDir::open(&dir).unwrap();
    let files = results
        .export_hdf5("band power", &dir.join("export"))
        .unwrap();
    assert_eq!(files.len(), 2);

    let file = hdf5_pure::File::open(&files[0]).unwrap();
    assert_eq!(
        file.dataset("modes/modes/modes").unwrap().shape().unwrap(),
        [2, 2, 128]
    );
    assert_eq!(
        file.dataset("power/power/power").unwrap().shape().unwrap(),
        [2, 17, 7]
    );
    assert_eq!(
        file.dataset("modes/modes/centre_frequencies_hz")
            .unwrap()
            .read_f64()
            .unwrap()
            .len(),
        2
    );
    assert!(file.root().attrs().unwrap().contains_key("manifest"));
}

const SERIES_SPEC: &str = r#"
[pipeline]
name = "band"
inputs = ["bold"]

[[step]]
id = "modes"
op = "mvmd"
input = "bold"
params = { num_modes = 2, alpha = 2000.0, init = "linear" }

[[step]]
id = "slow"
op = "reconstruct"
input = "modes"
params = { modes = [0] }
"#;

#[test]
fn exports_series_outputs_too() {
    let dir = temp_dir("series");
    let results = ResultsDir::open(&dir).unwrap();
    let plan = Plan::new(
        &PipelineSpec::from_toml(SERIES_SPEC).unwrap(),
        &builtin_registry(),
    )
    .unwrap();
    results.run_batch(&plan, &Tones, None, 1, &|_| {}).unwrap();
    let files = results.export_hdf5("band", &dir.join("export")).unwrap();
    let file = hdf5_pure::File::open(&files[0]).unwrap();
    assert_eq!(
        file.dataset("slow/signal/data").unwrap().shape().unwrap(),
        [2, 128]
    );
    assert_eq!(results.root(), dir.as_path());
    assert_eq!(results.store().root(), dir.join("store").as_path());
}

#[test]
fn failed_save_marks_only_that_recording_failed() {
    let dir = temp_dir("unsaveable");
    let results = ResultsDir::open(&dir).unwrap();
    std::fs::create_dir_all(dir.join("runs/band_power/sub-01.json")).unwrap();
    let plan = Plan::new(&PipelineSpec::from_toml(SPEC).unwrap(), &builtin_registry()).unwrap();
    let summary = results.run_batch(&plan, &Tones, None, 1, &|_| {}).unwrap();
    assert_eq!(summary.succeeded, ["sub-02"]);
    assert!(
        summary.failed["sub-01"].contains("saving run record failed"),
        "{:?}",
        summary.failed
    );
}

#[test]
fn unwritable_runs_folder_fails_the_batch() {
    let dir = temp_dir("blocked");
    let results = ResultsDir::open(&dir).unwrap();
    std::fs::create_dir_all(dir.join("runs")).unwrap();
    std::fs::write(dir.join("runs/band_power"), b"blocks the folder").unwrap();
    let plan = Plan::new(&PipelineSpec::from_toml(SPEC).unwrap(), &builtin_registry()).unwrap();
    assert!(results.run_batch(&plan, &Tones, None, 1, &|_| {}).is_err());
}

#[test]
fn missing_artifacts_and_shards_are_reported() {
    let dir = temp_dir("missing");
    run_into(&dir);
    std::fs::remove_dir_all(dir.join("store")).unwrap();
    let results = ResultsDir::open(&dir).unwrap();
    assert!(matches!(
        results.output("band power", "sub-01", "modes"),
        Err(ResultsError::MissingOutput { .. })
    ));
    assert!(
        results
            .export_hdf5("band power", &dir.join("export"))
            .is_err()
    );

    let plan = Plan::new(&PipelineSpec::from_toml(SPEC).unwrap(), &builtin_registry()).unwrap();
    let summary = results
        .run_batch(&plan, &Tones, Some((2, 2)), 1, &|_| {})
        .unwrap();
    assert_eq!(summary.succeeded, ["sub-02"]);
    assert!(
        dir.join("runs/band_power/summaries/shard-2-of-2.json")
            .exists()
    );
}

#[test]
fn input_mismatch_is_rejected_before_running() {
    let dir = temp_dir("mismatch");
    let results = ResultsDir::open(&dir).unwrap();
    let spec = SPEC
        .replace("inputs = [\"bold\"]", "inputs = [\"eeg\"]")
        .replace("input = \"bold\"", "input = \"eeg\"");
    let plan = Plan::new(
        &PipelineSpec::from_toml(&spec).unwrap(),
        &builtin_registry(),
    )
    .unwrap();
    assert!(matches!(
        results.run_batch(&plan, &Tones, None, 1, &|_| {}),
        Err(ResultsError::InputMismatch { .. })
    ));
}

struct TonesWithEvents;

impl InputSource for TonesWithEvents {
    fn input_names(&self) -> Vec<String> {
        vec!["bold".into(), "trials".into()]
    }

    fn recording_ids(&self) -> Vec<String> {
        vec!["sub-01".into()]
    }

    fn load(&self, id: &str) -> Result<BTreeMap<String, Artifact>, String> {
        let mut inputs = Tones.load(id)?;
        let events = modewise_epochs::Events::new(vec![
            modewise_epochs::Event {
                onset_s: 0.5,
                duration_s: 1.0,
                label: "a".into(),
            },
            modewise_epochs::Event {
                onset_s: 1.5,
                duration_s: 1.0,
                label: "b".into(),
            },
            modewise_epochs::Event {
                onset_s: 3.9,
                duration_s: 1.0,
                label: "late".into(),
            },
        ])
        .unwrap();
        inputs.insert("trials".into(), Artifact::Events(events));
        Ok(inputs)
    }
}

const EPOCH_SPEC: &str = r#"
[pipeline]
name = "cut"
inputs = { bold = "series", trials = "events" }

[[step]]
id = "modes"
op = "mvmd"
input = "bold"
params = { num_modes = 2, alpha = 2000.0, init = "linear" }

[[step]]
id = "blocks"
op = "epoch"
input = { data = "modes", events = "trials" }
params = { lag_s = 0.0 }

[[step]]
id = "trials"
op = "epoch"
input = { data = "bold", events = "trials" }
params = { lag_s = 0.0, window = { tmin_s = 0.0, tmax_s = 0.5 } }
"#;

#[test]
fn exports_events_and_epochs() {
    let dir = temp_dir("epochs");
    let results = ResultsDir::open(&dir).unwrap();
    let spec = EPOCH_SPEC.replace("id = \"trials\"\nop", "id = \"per_trial\"\nop");
    let plan = Plan::new(
        &PipelineSpec::from_toml(&spec).unwrap(),
        &builtin_registry(),
    )
    .unwrap();
    let summary = results
        .run_batch(&plan, &TonesWithEvents, None, 1, &|_| {})
        .unwrap();
    assert_eq!(summary.succeeded, ["sub-01"], "{:?}", summary.failed);

    let file = results
        .export_hdf5("cut", &dir.join("export"))
        .unwrap()
        .remove(0);
    let h5 = hdf5_pure::File::open(&file).unwrap();
    assert_eq!(
        h5.dataset("blocks/epochs/start_sample")
            .unwrap()
            .read_f64()
            .unwrap(),
        [16.0, 48.0]
    );
    assert_eq!(
        h5.dataset("blocks/epochs/resolvable")
            .unwrap()
            .shape()
            .unwrap(),
        [2, 2]
    );
    assert_eq!(
        h5.dataset("blocks/epochs/source/modes")
            .unwrap()
            .shape()
            .unwrap(),
        [2, 2, 128]
    );
    assert_eq!(
        h5.dataset("blocks/epochs/dropped_onset_s")
            .unwrap()
            .read_f64()
            .unwrap(),
        [3.9]
    );
    for strings in [
        "blocks/epochs/epoch_labels",
        "blocks/epochs/dropped_labels",
        "blocks/epochs/dropped_reasons",
        "modes/modes/labels",
    ] {
        assert!(h5.dataset(strings).is_ok(), "{strings} missing");
    }
    assert!(
        h5.dataset("per_trial/epochs/resolvable").is_err(),
        "series epochs carry no frequencies"
    );
    assert_eq!(
        h5.dataset("per_trial/epochs/source/data")
            .unwrap()
            .shape()
            .unwrap(),
        [2, 128]
    );
    let attrs = h5.group("per_trial/epochs").unwrap().attrs().unwrap();
    assert!(format!("{:?}", attrs["window"]).contains("fixed [0, 0.5) s"));
}

#[test]
fn unkept_steps_feed_later_steps_without_being_saved() {
    let dir = temp_dir("unkept");
    let spec = r#"
[pipeline]
name = "fc"
inputs = ["bold"]

[[step]]
id = "modes"
op = "mvmd"
input = "bold"
params = { num_modes = 2, alpha = 2000.0, init = "linear" }
keep = false

[[step]]
id = "fc"
op = "functional_connectivity"
input = "modes"
params = { measure = "pearson" }
"#;
    let results = ResultsDir::open(&dir).unwrap();
    let plan = Plan::new(&PipelineSpec::from_toml(spec).unwrap(), &builtin_registry()).unwrap();
    let summary = results.run_batch(&plan, &Tones, None, 2, &|_| {}).unwrap();
    assert_eq!(summary.succeeded, ["sub-01", "sub-02"]);
    assert!(matches!(
        results.output("fc", "sub-01", "fc").unwrap().as_ref(),
        Artifact::FunctionalConnectivity(_)
    ));
    let err = results.output("fc", "sub-01", "modes").unwrap_err();
    assert!(matches!(err, ResultsError::NotStored { .. }), "{err}");
    assert!(err.to_string().contains("keep = false"));
    let manifest = results.manifest("fc", "sub-01").unwrap();
    assert_eq!(
        manifest.steps.iter().map(|s| s.stored).collect::<Vec<_>>(),
        [false, true]
    );
}
