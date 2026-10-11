#![allow(clippy::unwrap_used)]

use std::path::{Path, PathBuf};

use modewise_dataset::{Dataset, DatasetSpec};

fn temp_root(name: &str) -> PathBuf {
    let root = std::env::temp_dir().join(format!(
        "modewise-participants-{name}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    root
}

fn write(path: &Path, text: &str) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, text).unwrap();
}

/// Laid out like ds005237: participants.tsv keyed `sub-NDAR…`, phenotype keyed `NDAR_…`,
/// recordings keyed by the part after `NDAR_`.
fn project(name: &str) -> PathBuf {
    let root = temp_root(name);
    write(
        &root.join("participants.tsv"),
        "participant_id\tage\tsex\tGroup\nsub-NDARINVAA1\t20.5\tF\tPatient\nsub-NDARINVBB2\tn/a\tM\tGenPop\n",
    );
    write(
        &root.join("phenotype/shaps01.tsv"),
        "subjectkey\tsrc\tshaps_total\nNDAR_INVAA1\tX\t6\nNDAR_INVBB2\tY\t999\nNDAR_INVCC3\tZ\t1\n",
    );
    write(&root.join("NDAR_INVAA1/bold.txt"), "");
    root
}

const SPEC: &str = r#"
root = "."

[[input]]
name = "bold"
pattern = "NDAR_{subject}/bold.txt"
format = "events_tsv"

[participants]
tables = [
  { path = "participants.tsv", id_pattern = "sub-NDAR{subject}" },
  { path = "phenotype/shaps01.tsv", id_column = "subjectkey", id_pattern = "NDAR_{subject}", columns = ["shaps_total"], missing = ["999"] },
]
"#;

#[test]
fn tables_join_on_the_recording_subject_with_missing_codes() {
    let root = project("join");
    let ds = Dataset::discover(DatasetSpec::from_toml(SPEC).unwrap(), &root).unwrap();
    let p = ds.participants().unwrap();
    assert_eq!(p.columns(), ["age", "sex", "Group", "shaps_total"]);
    assert_eq!(p.number("INVAA1", "age").unwrap(), Some(20.5));
    assert_eq!(p.number("INVBB2", "age").unwrap(), None, "n/a");
    assert_eq!(
        p.number("INVBB2", "shaps_total").unwrap(),
        None,
        "999 is missing"
    );
    assert_eq!(
        p.number("INVCC3", "shaps_total").unwrap(),
        Some(1.0),
        "phenotype-only subjects are kept"
    );
    assert_eq!(p.text("INVAA1", "Group").unwrap(), Some("Patient"));
    assert!(ds.participants().is_some() && ds.recordings().len() == 1);
}

#[test]
fn bad_participant_tables_are_named() {
    let root = project("bad");
    let cases = [
        (
            SPEC.replace("columns = [\"shaps_total\"]", "columns = [\"shaps\"]"),
            "no column \"shaps\"",
        ),
        (
            SPEC.replace("id_column = \"subjectkey\"", "id_column = \"guid\""),
            "no column \"guid\"",
        ),
        (
            SPEC.replace(
                "id_pattern = \"sub-NDAR{subject}\"",
                "id_pattern = \"sub-{run}\"",
            ),
            "must contain exactly {subject}",
        ),
        (
            SPEC.replace(
                "id_pattern = \"sub-NDAR{subject}\"",
                "id_pattern = \"x-{subject}\"",
            ),
            "no id",
        ),
        (
            SPEC.replace("phenotype/shaps01.tsv", "phenotype/none.tsv"),
            "phenotype/none.tsv",
        ),
    ];
    for (text, message) in cases {
        let err = Dataset::discover(DatasetSpec::from_toml(&text).unwrap(), &root)
            .err()
            .unwrap()
            .to_string();
        assert!(err.contains(message), "{message}: {err}");
    }
    let prefixed = SPEC.replace("columns = [\"shaps_total\"], ", "prefix = \"shaps.\", ");
    let ds = Dataset::discover(DatasetSpec::from_toml(&prefixed).unwrap(), &root).unwrap();
    assert!(
        ds.participants()
            .unwrap()
            .columns()
            .contains(&"shaps.src".to_string())
    );
    write(
        &root.join("phenotype/shaps01.tsv"),
        "subjectkey\tshaps_total\nNDAR_INVAA1\t6\nNDAR_INVAA1\t7\n",
    );
    let err = Dataset::discover(DatasetSpec::from_toml(SPEC).unwrap(), &root)
        .err()
        .unwrap()
        .to_string();
    assert!(err.contains("more than one row"), "{err}");
    write(&root.join("phenotype/shaps01.tsv"), "");
    assert!(
        Dataset::discover(DatasetSpec::from_toml(SPEC).unwrap(), &root)
            .err()
            .unwrap()
            .to_string()
            .contains("empty")
    );
}

#[test]
fn clashing_columns_need_a_prefix() {
    let root = project("clash");
    write(
        &root.join("phenotype/other.tsv"),
        "subjectkey\tage\nNDAR_INVAA1\t21\n",
    );
    let clash = SPEC.replace("\n]\n", "\n  { path = \"phenotype/other.tsv\", id_column = \"subjectkey\", id_pattern = \"NDAR_{subject}\" },\n]\n");
    let err = Dataset::discover(DatasetSpec::from_toml(&clash).unwrap(), &root)
        .err()
        .unwrap()
        .to_string();
    assert!(err.contains("\"age\"") && err.contains("prefix"), "{err}");
}
