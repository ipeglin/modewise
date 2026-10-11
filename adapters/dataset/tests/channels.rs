#![allow(clippy::unwrap_used)]

use std::path::{Path, PathBuf};

use hdf5_pure::FileBuilder;
use modewise_brain::Hemisphere;
use modewise_dataset::{Dataset, DatasetSpec, TableFormat, read_table};
use modewise_pipeline::{Artifact, InputSource};

fn temp_root(name: &str) -> PathBuf {
    let root =
        std::env::temp_dir().join(format!("modewise-channels-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    root
}

fn write(path: &Path, text: &str) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, text).unwrap();
}

/// Excerpts of the real label files: a Schaefer `_info.txt` (name lines alternate with colour
/// lines) and a Tian scale II label list.
const SCHAEFER_INFO: &str = "17networks_LH_TempPar_IPL_1\n1 12 48 255 255\n17networks_LH_DefaultB_PFCm_1\n2 255 255 0 255\n17networks_RH_LimbicA_TempPole_1\n3 0 0 0 255\n";
const TIAN: &str = "aHIP-rh\nlAMY-rh\nlAMY-lh\n";

const SPEC: &str = r#"
root = "data"

[[input]]
name = "bold"
pattern = "sub-{subject}_bold.h5"
format = "hdf5"
dataset = "signal"
sampling_rate_hz = 1.25

[channels]
tables = [{ path = "atlas/schaefer_info.txt", format = "schaefer" }, { path = "atlas/tian.txt", format = "tian" }]
names_for = ["bold"]
"#;

fn project(name: &str) -> PathBuf {
    let base = temp_root(name);
    write(&base.join("atlas/schaefer_info.txt"), SCHAEFER_INFO);
    write(&base.join("atlas/tian.txt"), TIAN);
    let data: Vec<f64> = (0..6 * 8).map(|i| i as f64).collect();
    let mut b = FileBuilder::new();
    b.create_dataset("signal")
        .with_f64_data(&data)
        .with_shape(&[6, 8]);
    std::fs::create_dir_all(base.join("data")).unwrap();
    b.write(base.join("data/sub-01_bold.h5")).unwrap();
    base
}

#[test]
fn tables_concatenate_name_the_channels_and_become_an_input() {
    let base = project("load");
    write(&base.join("dataset.toml"), SPEC);
    let ds = Dataset::from_file(&base.join("dataset.toml")).unwrap();
    assert_eq!(ds.input_names(), ["bold", "channels"]);
    let table = ds.channels().unwrap();
    assert_eq!(table.len(), 6);
    assert_eq!(table.channels()[1].tag("yeo17"), Some("DefaultB"));
    assert_eq!(table.channels()[5].hemisphere(), Some(Hemisphere::Left));

    let loaded = ds.load(&ds.recording_ids()[0]).unwrap();
    let Artifact::Series(bold) = &loaded["bold"] else {
        panic!()
    };
    assert_eq!(bold.labels()[1], "17networks_LH_DefaultB_PFCm_1");
    assert!(matches!(&loaded["channels"], Artifact::Channels(t) if t.len() == 6));
}

#[test]
fn channel_specs_are_checked() {
    let cases = [
        (SPEC.replace("names_for = [\"bold\"]", "names_for = [\"eeg\"]"), "no input"),
        (SPEC.replace("tables = [{ path = \"atlas/schaefer_info.txt\", format = \"schaefer\" }, { path = \"atlas/tian.txt\", format = \"tian\" }]", "tables = []"), "at least one table"),
        (SPEC.replace("[channels]", "[channels]\nname = \"bold\""), "already an input"),
        (SPEC.replace("sampling_rate_hz = 1.25", "sampling_rate_hz = 1.25\nlabels_file = \"x.txt\""), "labels_file"),
        (SPEC.replace("format = \"tian\"", "format = \"aal\""), "unknown variant"),
    ];
    for (text, message) in cases {
        let err = DatasetSpec::from_toml(&text).err().unwrap().to_string();
        assert!(err.contains(message), "{message}: {err}");
    }
    let mut events = SPEC.to_string();
    events.push_str("\n[[input]]\nname = \"trials\"\npattern = \"sub-{subject}_events.tsv\"\nformat = \"events_tsv\"\n");
    let err = DatasetSpec::from_toml(
        &events.replace("names_for = [\"bold\"]", "names_for = [\"trials\"]"),
    )
    .err()
    .unwrap();
    assert!(err.to_string().contains("not a signal input"));
}

#[test]
fn a_table_that_does_not_match_the_data_fails_the_recording() {
    let base = project("mismatch");
    write(&base.join("atlas/tian.txt"), "aHIP-rh\n");
    write(&base.join("dataset.toml"), SPEC);
    let ds = Dataset::from_file(&base.join("dataset.toml")).unwrap();
    assert!(
        ds.load(&ds.recording_ids()[0]).is_err(),
        "5 names for 6 channels"
    );

    write(
        &base.join("atlas/tian.txt"),
        "aHIP-rh\n17networks_LH_TempPar_IPL_1\n",
    );
    let err = Dataset::from_file(&base.join("dataset.toml"))
        .err()
        .unwrap();
    assert!(err.to_string().contains("more than once"), "{err}");
    write(&base.join("atlas/tian.txt"), "");
    assert!(
        Dataset::from_file(&base.join("dataset.toml"))
            .err()
            .unwrap()
            .to_string()
            .contains("no channel labels")
    );
}

#[test]
fn every_format_reads_from_disk() {
    let base = temp_root("formats");
    let cases = [
        (
            "a.tsv",
            "name\tregion\themisphere\tlobe\nFp1\tfrontopolar\tleft\tfrontal\n",
            TableFormat::Tsv,
            Some("frontopolar"),
        ),
        ("b.txt", "Fz\nCz\n", TableFormat::Names, None),
        ("c.txt", TIAN, TableFormat::Tian, Some("aHIP")),
        (
            "d.txt",
            "1\t17Networks_LH_DefaultB_PFCm_1\t255 0 0 0\n",
            TableFormat::Schaefer,
            Some("PFCm"),
        ),
    ];
    for (file, text, format, region) in cases {
        write(&base.join(file), text);
        let table = read_table(&base.join(file), format).unwrap();
        assert_eq!(
            table.channels()[0].region().map(|r| r.as_str()),
            region,
            "{file}"
        );
    }
    assert!(read_table(&base.join("missing.tsv"), TableFormat::Tsv).is_err());
    write(&base.join("bad.tsv"), "label\nx\n");
    assert!(
        read_table(&base.join("bad.tsv"), TableFormat::Tsv)
            .unwrap_err()
            .contains("`name`")
    );
}

#[test]
fn the_tcp_example_table_describes_all_434_channels() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/tcp/channels.tsv");
    let table = read_table(&path, TableFormat::Tsv).unwrap();
    assert_eq!(table.len(), 434);
    let c = table.channels();
    assert_eq!(
        (c[0].hemisphere(), c[199].hemisphere(), c[200].hemisphere()),
        (
            Some(Hemisphere::Left),
            Some(Hemisphere::Left),
            Some(Hemisphere::Right)
        )
    );
    assert_eq!(c[0].tag("yeo17"), Some("TempPar"));
    assert_eq!(
        (c[400].name(), c[432].name(), c[433].name()),
        ("aHIP-rh", "cerebellum_left", "cerebellum_right")
    );
    assert_eq!(c[402].tag("tcp_network"), Some("medial_temporal_lobe"));
}
