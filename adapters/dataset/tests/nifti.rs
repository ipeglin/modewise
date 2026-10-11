#![allow(clippy::unwrap_used)]

use std::path::{Path, PathBuf};

use modewise_dataset::{Dataset, DatasetSpec};
use modewise_pipeline::{Artifact, InputSource};

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../methods/parcellate/tests/fixtures")
}

fn project(name: &str) -> PathBuf {
    let root = std::env::temp_dir().join(format!("modewise-nifti-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(root.join("data/sub-01/func")).unwrap();
    std::fs::create_dir_all(root.join("atlas")).unwrap();
    std::fs::copy(
        fixtures().join("bold.nii.gz"),
        root.join("data/sub-01/func/sub-01_task-rest_bold.nii.gz"),
    )
    .unwrap();
    for atlas in ["atlas.nii.gz", "atlas_qform.nii.gz"] {
        std::fs::copy(fixtures().join(atlas), root.join("atlas").join(atlas)).unwrap();
    }
    root
}

const SPEC: &str = r#"
root = "data"

[[input]]
name = "bold"
pattern = "sub-{subject}/func/sub-{subject}_task-{task}_bold.nii.gz"
format = "nifti"
atlases = ["atlas/atlas.nii.gz", "atlas/atlas_qform.nii.gz"]
"#;

fn series(spec: &str, root: &Path) -> modewise_signals::MultichannelSeries {
    let ds = Dataset::discover(DatasetSpec::from_toml(spec).unwrap(), root).unwrap();
    let Artifact::Series(s) = ds
        .load(&ds.recording_ids()[0])
        .unwrap()
        .remove("bold")
        .unwrap()
    else {
        panic!()
    };
    s
}

#[test]
fn volumes_are_parcellated_at_load_time_with_atlases_in_order() {
    let root = project("load");
    let s = series(SPEC, &root);
    assert_eq!(
        (s.n_channels(), s.n_samples()),
        (10, 20),
        "5 labels from each atlas"
    );
    assert!(
        (s.sampling_rate().hz() - 1.0 / 0.8).abs() < 1e-6,
        "TR from the header"
    );
    assert_eq!(
        s.data().row(0),
        s.data().row(5),
        "sform and qform atlases agree"
    );
    assert_eq!(s.labels()[9], "ch9");

    let fixed = series(
        &SPEC.replace(
            "format = \"nifti\"",
            "format = \"nifti\"\nsampling_rate_hz = 2.0",
        ),
        &root,
    );
    assert_eq!(fixed.sampling_rate().hz(), 2.0);
    let z = series(
        &SPEC.replace(
            "format = \"nifti\"",
            "format = \"nifti\"\nstandardize = \"zscore_sample\"",
        ),
        &root,
    );
    let data = z.data();
    let row = data.row(2);
    assert!(row.mean().unwrap().abs() < 1e-12 && (row.var(1.0) - 1.0).abs() < 1e-12);

    std::fs::write(
        root.join("names.tsv"),
        format!(
            "name\n{}",
            (0..10).map(|i| format!("r{i}\n")).collect::<String>()
        ),
    )
    .unwrap();
    let named = SPEC.to_string()
        + "\n[channels]\ntables = [{ path = \"names.tsv\", format = \"tsv\" }]\nnames_for = [\"bold\"]\n";
    assert_eq!(series(&named, &root).labels()[3], "r3");
}

#[test]
fn nifti_inputs_are_checked() {
    let cases = [
        (
            SPEC.replace(
                "atlases = [\"atlas/atlas.nii.gz\", \"atlas/atlas_qform.nii.gz\"]",
                "",
            ),
            "need `atlases`",
        ),
        (
            SPEC.replace("format = \"nifti\"", "format = \"nifti\"\ndataset = \"x\""),
            "do not apply to nifti",
        ),
        (
            SPEC.replace(
                "format = \"nifti\"",
                "format = \"nifti\"\nsampling_rate_hz = 1.0\nsidecar = \"x.json\"",
            ),
            "at most one",
        ),
        (
            SPEC.replace(
                "format = \"nifti\"",
                "format = \"hdf5\"\ndataset = \"d\"\nsampling_rate_hz = 1.0",
            ),
            "only apply to nifti",
        ),
        (
            SPEC.replace("format = \"nifti\"", "format = \"events_tsv\"")
                .replace(
                    "atlases = [\"atlas/atlas.nii.gz\", \"atlas/atlas_qform.nii.gz\"]",
                    "standardize = \"zscore_sample\"",
                ),
            "does not apply to events_tsv",
        ),
    ];
    for (text, message) in cases {
        let err = DatasetSpec::from_toml(&text).err().unwrap().to_string();
        assert!(err.contains(message), "{message}: {err}");
    }
    let root = project("bad");
    let missing = SPEC.replace("atlas/atlas.nii.gz", "atlas/none.nii.gz");
    let ds = Dataset::discover(DatasetSpec::from_toml(&missing).unwrap(), &root).unwrap();
    assert!(
        ds.load(&ds.recording_ids()[0])
            .unwrap_err()
            .contains("none.nii.gz")
    );
}
