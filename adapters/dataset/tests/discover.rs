#![allow(clippy::unwrap_used)]

use std::path::{Path, PathBuf};

use hdf5_pure::FileBuilder;
use modewise_dataset::{Dataset, DatasetError, DatasetSpec};
use modewise_pipeline::{Artifact, InputSource};
use modewise_signals::MultichannelSeries;

fn series(artifact: &Artifact) -> &MultichannelSeries {
    match artifact {
        Artifact::Series(s) => s,
        other => panic!("expected a series, got {}", other.kind()),
    }
}

fn temp_root(name: &str) -> PathBuf {
    let root = std::env::temp_dir().join(format!("modewise-dataset-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    root
}

fn write_h5(path: &Path, rows: usize, cols: usize) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let data: Vec<f64> = (0..rows * cols).map(|i| i as f64).collect();
    let mut b = FileBuilder::new();
    b.create_dataset("signal")
        .with_f64_data(&data)
        .with_shape(&[rows as u64, cols as u64]);
    b.write(path).unwrap();
}

fn write(path: &Path, text: &str) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, text).unwrap();
}

fn eeg_like(root: &Path) {
    for subject in ["01", "02"] {
        for task in ["rest", "oddball"] {
            let stem = format!("sub-{subject}/eeg/sub-{subject}_task-{task}_eeg");
            write_h5(&root.join(format!("{stem}.h5")), 3, 10);
            write(
                &root.join(format!("{stem}.json")),
                r#"{"SamplingFrequency": 250}"#,
            );
        }
    }
    write(&root.join("sub-01/eeg/notes.txt"), "not a recording");
    write(&root.join("channels.txt"), "Fz\nCz\n\nPz\n");
}

const EEG_SPEC: &str = r#"
root = "."
filter = { task = ["rest"] }

[[input]]
name = "eeg"
pattern = "sub-{subject}/eeg/sub-{subject}_task-{task}_eeg.h5"
format = "hdf5"
dataset = "signal"
sidecar = "sub-{subject}/eeg/sub-{subject}_task-{task}_eeg.json"
labels_file = "channels.txt"
"#;

#[test]
fn discovers_filters_and_loads_bids_like_eeg() {
    let root = temp_root("eeg");
    eeg_like(&root);
    let dataset = Dataset::discover(DatasetSpec::from_toml(EEG_SPEC).unwrap(), &root).unwrap();

    assert_eq!(dataset.input_names(), ["eeg"]);
    assert_eq!(dataset.recordings().len(), 2);
    assert_eq!(
        dataset.recordings()[1].entities,
        [
            ("subject".to_string(), "02".to_string()),
            ("task".to_string(), "rest".to_string())
        ]
    );
    assert_eq!(
        dataset.recording_ids(),
        ["subject-01_task-rest", "subject-02_task-rest"]
    );
    let inputs = dataset.load("subject-02_task-rest").unwrap();
    let eeg = series(&inputs["eeg"]);
    assert_eq!(eeg.sampling_rate().hz(), 250.0);
    assert_eq!(eeg.labels(), ["Fz", "Cz", "Pz"]);
    assert_eq!(eeg.n_samples(), 10);
}

#[test]
fn reads_repetition_time_from_a_separate_tree() {
    let root = temp_root("tcp");
    write_h5(
        &root.join("parcellated/NDAR_INV1/task-hammer_run-01_bold.h5"),
        4,
        20,
    );
    write(
        &root.join("sub-NDARINV1/func/sub-NDARINV1_task-hammer_run-01_bold.json"),
        r#"{"RepetitionTime": 0.8}"#,
    );
    let spec = r#"
        root = "."
        [[input]]
        name = "bold"
        pattern = "parcellated/NDAR_{subject}/task-{task}_run-{run}_bold.h5"
        format = "hdf5"
        dataset = "signal"
        sidecar = "sub-NDAR{subject}/func/sub-NDAR{subject}_task-{task}_run-{run}_bold.json"
    "#;
    let dataset = Dataset::discover(DatasetSpec::from_toml(spec).unwrap(), &root).unwrap();
    assert_eq!(dataset.recording_ids(), ["subject-INV1_task-hammer_run-01"]);
    let loaded = dataset.load("subject-INV1_task-hammer_run-01").unwrap();
    let bold = series(&loaded["bold"]);
    assert_eq!(bold.sampling_rate().hz(), 1.25);
}

#[test]
fn second_input_missing_fails_only_that_recording() {
    let root = temp_root("pair");
    eeg_like(&root);
    write_h5(&root.join("derived/sub-01_task-rest_ref.h5"), 3, 10);
    let spec = format!(
        "{EEG_SPEC}\n[[input]]\nname = \"reference\"\npattern = \"derived/sub-{{subject}}_task-{{task}}_ref.h5\"\nformat = \"hdf5\"\ndataset = \"signal\"\nsampling_rate_hz = 250.0\n"
    );
    let dataset = Dataset::discover(DatasetSpec::from_toml(&spec).unwrap(), &root).unwrap();
    assert!(dataset.load("subject-01_task-rest").is_ok());
    assert!(dataset.load("subject-02_task-rest").is_err());
}

#[test]
fn rejects_invalid_specs() {
    let both = EEG_SPEC.replace("labels_file", "sampling_rate_hz = 1.0\nlabels_file");
    let no_dataset = EEG_SPEC.replace("dataset = \"signal\"\n", "");
    let typo = EEG_SPEC.replace("labels_file", "lables_file");
    for text in [
        both.as_str(),
        no_dataset.as_str(),
        typo.as_str(),
        "root = \".\"\ninput = []\n",
    ] {
        assert!(
            matches!(DatasetSpec::from_toml(text), Err(DatasetError::Spec(_))),
            "accepted {text}"
        );
    }
    let unknown_filter = EEG_SPEC.replace("task = [\"rest\"]", "session = [\"1\"]");
    let err = Dataset::discover(
        DatasetSpec::from_toml(&unknown_filter).unwrap(),
        &temp_root("f"),
    );
    assert!(matches!(err, Err(DatasetError::UnknownFilterEntity(e)) if e == "session"));
}

#[test]
fn reports_bad_files_sidecars_and_labels() {
    let missing = Dataset::from_file(Path::new("/nonexistent/dataset.toml"));
    assert!(matches!(missing, Err(DatasetError::Io { .. })));

    let root = temp_root("bad-sidecar");
    eeg_like(&root);
    write(
        &root.join("sub-01/eeg/sub-01_task-rest_eeg.json"),
        r#"{"Manufacturer": "x"}"#,
    );
    let dataset = Dataset::discover(DatasetSpec::from_toml(EEG_SPEC).unwrap(), &root).unwrap();
    let err = dataset.load("subject-01_task-rest").err().unwrap();
    assert!(
        err.contains("neither SamplingFrequency nor RepetitionTime"),
        "{err}"
    );

    std::fs::remove_file(root.join("channels.txt")).unwrap();
    assert!(
        dataset
            .load("subject-02_task-rest")
            .err()
            .unwrap()
            .contains("labels file")
    );
    assert!(dataset.load("no-such-recording").is_err());
}

#[test]
fn repeated_entities_must_agree_and_layout_is_applied() {
    let root = temp_root("repeat");
    write_h5(&root.join("sub-01/sub-01_bold.h5"), 3, 4);
    write_h5(&root.join("sub-01/sub-02_bold.h5"), 3, 4);
    let spec = r#"
        root = "."
        [[input]]
        name = "bold"
        pattern = "sub-{subject}/sub-{subject}_bold.h5"
        format = "hdf5"
        dataset = "signal"
        layout = "time_by_channels"
        sampling_rate_hz = 2.0
    "#;
    let dataset = Dataset::from_file(&{
        let path = root.join("dataset.toml");
        std::fs::write(&path, spec).unwrap();
        path
    })
    .unwrap();
    assert_eq!(
        dataset.recording_ids(),
        ["subject-01"],
        "sub-01/sub-02 must not match"
    );
    let loaded = dataset.load("subject-01").unwrap();
    let bold = series(&loaded["bold"]);
    assert_eq!((bold.n_channels(), bold.n_samples()), (4, 3));
}

#[test]
fn spec_without_inputs_is_rejected_by_discover_too() {
    let spec = DatasetSpec {
        root: ".".into(),
        filter: Default::default(),
        inputs: vec![],
        channels: None,
        participants: None,
    };
    assert!(matches!(
        Dataset::discover(spec, Path::new(".")),
        Err(DatasetError::Spec(_))
    ));
}

#[test]
fn missing_root_means_no_recordings_not_an_error() {
    let dataset = Dataset::discover(
        DatasetSpec::from_toml(EEG_SPEC).unwrap(),
        Path::new("/nonexistent/modewise"),
    )
    .unwrap();
    assert!(dataset.recordings().is_empty());
}

const EVENTS_SPEC: &str = r#"
root = "."

[[input]]
name = "eeg"
pattern = "sub-{subject}/eeg/sub-{subject}_task-{task}_eeg.h5"
format = "hdf5"
dataset = "signal"
sampling_rate_hz = 250.0

[[input]]
name = "trials"
pattern = "sub-{subject}/eeg/sub-{subject}_task-{task}_events.tsv"
format = "events_tsv"
"#;

#[test]
fn reads_bids_events_tsv_as_a_second_input() {
    let root = temp_root("events");
    eeg_like(&root);
    for subject in ["01", "02"] {
        for task in ["rest", "oddball"] {
            write(
                &root.join(format!(
                    "sub-{subject}/eeg/sub-{subject}_task-{task}_events.tsv"
                )),
                "onset\tduration\ttrial_type\tresponse_time\n2.5\t0.5\tstandard\t0.4\n0.5\t0.5\ttarget\tn/a\n",
            );
        }
    }
    let dataset = Dataset::discover(DatasetSpec::from_toml(EVENTS_SPEC).unwrap(), &root).unwrap();
    assert_eq!(dataset.input_names(), ["eeg", "trials"]);
    let loaded = dataset.load("subject-01_task-rest").unwrap();
    let Artifact::Events(events) = &loaded["trials"] else {
        panic!("expected events")
    };
    let rows: Vec<(f64, &str)> = events
        .as_slice()
        .iter()
        .map(|e| (e.onset_s, e.label.as_str()))
        .collect();
    assert_eq!(
        rows,
        [(0.5, "target"), (2.5, "standard")],
        "sorted by onset; n/a in other columns is fine"
    );

    let custom = EVENTS_SPEC.replace(
        "format = \"events_tsv\"",
        "format = \"events_tsv\"\nlabel_column = \"response_time\"",
    );
    let dataset = Dataset::discover(DatasetSpec::from_toml(&custom).unwrap(), &root).unwrap();
    let Artifact::Events(events) = &dataset.load("subject-01_task-rest").unwrap()["trials"] else {
        panic!()
    };
    assert_eq!(
        events.as_slice()[0].label,
        "n/a",
        "labels are taken verbatim from the chosen column"
    );
}

#[test]
fn bad_events_files_name_the_problem() {
    let root = temp_root("bad-events");
    eeg_like(&root);
    let file = root.join("sub-01/eeg/sub-01_task-rest_events.tsv");
    let dataset = |text: &str| {
        write(&file, text);
        Dataset::discover(DatasetSpec::from_toml(EVENTS_SPEC).unwrap(), &root)
            .unwrap()
            .load("subject-01_task-rest")
    };
    assert!(
        dataset("onset\tduration\n1\t2\n")
            .unwrap_err()
            .contains("no \"trial_type\" column")
    );
    assert!(
        dataset("onset\tduration\ttrial_type\nn/a\t1\tx\n")
            .unwrap_err()
            .contains("onset is n/a")
    );
    assert!(
        dataset("onset\tduration\ttrial_type\n1\tlong\tx\n")
            .unwrap_err()
            .contains("not a number")
    );
    assert!(
        dataset("onset\tduration\ttrial_type\n1\t-2\tx\n")
            .unwrap_err()
            .contains("duration")
    );
    assert!(
        dataset("onset\tduration\ttrial_type\n1\t2\n")
            .unwrap_err()
            .contains("missing trial_type")
    );
    assert!(dataset("").unwrap_err().contains("is empty"));
}

#[test]
fn events_inputs_reject_signal_fields_and_vice_versa() {
    let with = |extra: &str| {
        EVENTS_SPEC.replace(
            "format = \"events_tsv\"",
            &format!("format = \"events_tsv\"\n{extra}"),
        )
    };
    for extra in [
        "dataset = \"x\"",
        "sampling_rate_hz = 1.0",
        "sidecar = \"x.json\"",
        "labels_file = \"x.txt\"",
    ] {
        assert!(
            DatasetSpec::from_toml(&with(extra)).is_err(),
            "accepted {extra}"
        );
    }
    let hdf5_with_label_column = EVENTS_SPEC.replace(
        "sampling_rate_hz = 250.0",
        "sampling_rate_hz = 250.0\nlabel_column = \"x\"",
    );
    assert!(DatasetSpec::from_toml(&hdf5_with_label_column).is_err());
}
