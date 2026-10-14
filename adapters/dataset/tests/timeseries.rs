#![allow(clippy::unwrap_used)]

use std::path::{Path, PathBuf};

use modewise_dataset::{Dataset, DatasetSpec};
use modewise_pipeline::{Artifact, InputSource};

fn write(path: &Path, text: &str) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, text).unwrap();
}

fn project(name: &str) -> PathBuf {
    let root =
        std::env::temp_dir().join(format!("modewise-timeseries-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let func = root.join("sub-01/func");
    write(
        &func.join("sub-01_task-rest_desc-confounds_timeseries.tsv"),
        "trans_x\trot_x\tframewise_displacement\n0.1\t0.01\tn/a\n0.2\t0.02\t0.3\n0.1\t0.0\t0.2\n",
    );
    write(
        &func.join("sub-01_task-rest_bold.json"),
        r#"{"RepetitionTime": 0.8}"#,
    );
    root
}

const SPEC: &str = r#"
root = "."

[[input]]
name = "confounds"
pattern = "sub-{subject}/func/sub-{subject}_task-{task}_desc-confounds_timeseries.tsv"
format = "timeseries_tsv"
columns = ["trans_*", "rot_*", "framewise_displacement"]
sidecar = "sub-{subject}/func/sub-{subject}_task-{task}_bold.json"
"#;

#[test]
fn loads_chosen_columns_with_the_sidecar_rate() {
    let root = project("load");
    let ds = Dataset::discover(DatasetSpec::from_toml(SPEC).unwrap(), &root).unwrap();
    let loaded = ds.load(&ds.recording_ids()[0]).unwrap();
    let Artifact::Series(s) = &loaded["confounds"] else {
        panic!("not a series")
    };
    assert_eq!(s.labels(), ["trans_x", "rot_x", "framewise_displacement"]);
    assert_eq!(s.sampling_rate().hz(), 1.25);
    assert_eq!(s.data().row(2).to_vec(), vec![0.3, 0.3, 0.2]);

    let missing = SPEC.replace("\"rot_*\"", "\"global_signal\"");
    let ds = Dataset::discover(DatasetSpec::from_toml(&missing).unwrap(), &root).unwrap();
    let err = ds.load(&ds.recording_ids()[0]).unwrap_err();
    assert!(err.contains("no column matches \"global_signal\""), "{err}");
}

#[test]
fn timeseries_inputs_are_checked() {
    let rate = "sidecar = \"sub-{subject}/func/sub-{subject}_task-{task}_bold.json\"";
    let cases = [
        (SPEC.replace(rate, ""), "exactly one"),
        (
            SPEC.replace(rate, &format!("{rate}\nsampling_rate_hz = 1.25")),
            "exactly one",
        ),
        (
            SPEC.replace(
                "columns = [\"trans_*\", \"rot_*\", \"framewise_displacement\"]",
                "",
            ),
            "need `columns`",
        ),
        (
            SPEC.replace(rate, &format!("{rate}\nlabel_column = \"x\"")),
            "`label_column` does not apply to timeseries_tsv",
        ),
        (
            SPEC.replace("format = \"timeseries_tsv\"", "format = \"events_tsv\"")
                .replace(rate, ""),
            "`columns` only applies to timeseries_tsv",
        ),
        (
            format!(
                "{SPEC}\n[channels]\ntables = [{{ path = \"t.tsv\", format = \"tsv\" }}]\nnames_for = [\"confounds\"]\n"
            ),
            "takes its names from its columns",
        ),
    ];
    for (text, message) in cases {
        let err = DatasetSpec::from_toml(&text).err().unwrap().to_string();
        assert!(err.contains(message), "{message}: {err}");
    }
}
