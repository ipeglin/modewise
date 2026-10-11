#![allow(clippy::unwrap_used)]

use std::path::{Path, PathBuf};

use hdf5_pure::FileBuilder;
use modewise_cli::{ERROR, OK, PARTIAL, main};

const PIPELINE: &str = r#"
[pipeline]
name = "demo"
inputs = ["bold"]

[[step]]
id = "modes"
op = "mvmd"
input = "bold"
params = { num_modes = 2, alpha = 1000.0, init = "linear" }
"#;

const DATASET: &str = r#"
root = "data"

[[input]]
name = "bold"
pattern = "sub-{subject}/sub-{subject}_task-{task}_bold.h5"
format = "hdf5"
dataset = "signal"
sampling_rate_hz = 1.25
"#;

fn setup(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("modewise-cli-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    for subject in ["01", "02", "03"] {
        let path = dir.join(format!(
            "data/sub-{subject}/sub-{subject}_task-rest_bold.h5"
        ));
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let data: Vec<f64> = (0..2 * 64)
            .map(|i| ((i % 64) as f64 * 0.4).sin() * (1 + i / 64) as f64)
            .collect();
        let mut b = FileBuilder::new();
        b.create_dataset("signal")
            .with_f64_data(&data)
            .with_shape(&[2, 64]);
        b.write(&path).unwrap();
    }
    std::fs::write(dir.join("pipeline.toml"), PIPELINE).unwrap();
    std::fs::write(dir.join("dataset.toml"), DATASET).unwrap();
    dir
}

fn cli(dir: &Path, args: &[&str]) -> u8 {
    let mut argv = vec!["modewise".to_string()];
    argv.extend(
        args.iter()
            .map(|a| a.replace("{dir}", dir.to_str().unwrap())),
    );
    main(argv)
}

fn cached_steps(dir: &Path, recording: &str) -> Vec<bool> {
    let text =
        std::fs::read_to_string(dir.join(format!("out/runs/demo/{recording}.json"))).unwrap();
    let manifest: serde_json::Value = serde_json::from_str(&text).unwrap();
    manifest["steps"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s["cached"].as_bool().unwrap())
        .collect()
}

const RUN: &[&str] = &[
    "run",
    "{dir}/pipeline.toml",
    "--dataset",
    "{dir}/dataset.toml",
    "--out",
    "{dir}/out",
];

#[test]
fn runs_reruns_from_cache_and_exports() {
    let dir = setup("run");
    assert_eq!(cli(&dir, RUN), OK);
    assert_eq!(cached_steps(&dir, "subject-02_task-rest"), [false]);
    assert_eq!(cli(&dir, RUN), OK);
    assert_eq!(cached_steps(&dir, "subject-02_task-rest"), [true]);

    assert_eq!(
        cli(&dir, &["export", "{dir}/out", "--to", "{dir}/export"]),
        OK
    );
    assert_eq!(
        std::fs::read_dir(dir.join("export/demo")).unwrap().count(),
        3
    );
    assert_eq!(cli(&dir, &["inspect", "{dir}/dataset.toml", "--load"]), OK);
    assert_eq!(cli(&dir, &["ops"]), OK);
}

#[test]
fn shards_split_the_recordings() {
    let dir = setup("shard");
    let mut args = RUN.to_vec();
    args.extend(["--shard", "1/2"]);
    assert_eq!(cli(&dir, &args), OK);
    *args.last_mut().unwrap() = "2/2";
    assert_eq!(cli(&dir, &args), OK);

    let summary = |name: &str| -> serde_json::Value {
        let text =
            std::fs::read_to_string(dir.join(format!("out/runs/demo/summaries/{name}"))).unwrap();
        serde_json::from_str(&text).unwrap()
    };
    assert_eq!(
        summary("shard-1-of-2.json")["succeeded"]
            .as_array()
            .unwrap()
            .len(),
        2
    );
    assert_eq!(
        summary("shard-2-of-2.json")["succeeded"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
}

#[test]
fn reports_partial_failure_and_bad_arguments() {
    let dir = setup("errors");
    std::fs::write(
        dir.join("data/sub-03/sub-03_task-rest_bold.h5"),
        b"not hdf5",
    )
    .unwrap();
    assert_eq!(cli(&dir, RUN), PARTIAL);

    std::fs::write(
        dir.join("dataset.toml"),
        DATASET.replace("name = \"bold\"", "name = \"eeg\""),
    )
    .unwrap();
    assert_eq!(cli(&dir, RUN), ERROR);
    let mut bad_shard = RUN.to_vec();
    bad_shard.extend(["--shard", "3/2"]);
    assert_eq!(cli(&dir, &bad_shard), ERROR);
    assert_eq!(
        cli(&dir, &["export", "{dir}/nowhere", "--to", "{dir}/x"]),
        ERROR
    );
}

#[test]
fn inspect_summarises_many_values_and_empty_datasets() {
    let dir = setup("inspect");
    for i in 4..12 {
        let path = dir.join(format!("data/sub-{i:02}/sub-{i:02}_task-rest_bold.h5"));
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let mut b = FileBuilder::new();
        b.create_dataset("signal")
            .with_f64_data(&[0.0, 1.0, 0.0, 1.0])
            .with_shape(&[1, 4]);
        b.write(&path).unwrap();
    }
    assert_eq!(cli(&dir, &["inspect", "{dir}/dataset.toml"]), OK);
    std::fs::write(
        dir.join("empty.toml"),
        DATASET.replace("root = \"data\"", "root = \"nowhere\""),
    )
    .unwrap();
    assert_eq!(cli(&dir, &["inspect", "{dir}/empty.toml", "--load"]), OK);
    assert_eq!(cli(&dir, &["inspect", "{dir}/missing.toml"]), ERROR);
}

#[test]
fn exports_one_named_pipeline_and_prints_help() {
    let dir = setup("export-one");
    assert_eq!(cli(&dir, RUN), OK);
    assert_eq!(
        cli(
            &dir,
            &[
                "export",
                "{dir}/out",
                "--pipeline",
                "demo",
                "--to",
                "{dir}/x"
            ]
        ),
        OK
    );
    assert!(dir.join("x/demo").is_dir());
    assert_eq!(cli(&dir, &["--help"]), OK);
    assert_eq!(cli(&dir, &["frobnicate"]), ERROR);
}

fn binary(dir: &Path, args: &[&str]) -> (i32, String) {
    let args: Vec<String> = args
        .iter()
        .map(|a| a.replace("{dir}", dir.to_str().unwrap()))
        .collect();
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_modewise"))
        .args(&args)
        .output()
        .unwrap();
    (
        out.status.code().unwrap(),
        String::from_utf8(out.stdout).unwrap(),
    )
}

#[test]
fn binary_prints_version_ops_and_inspect_summary() {
    let dir = setup("binary");
    let (code, out) = binary(&dir, &["--version"]);
    assert_eq!(
        (code, out.trim()),
        (0, concat!("modewise ", env!("CARGO_PKG_VERSION")))
    );

    let (_, ops) = binary(&dir, &["ops"]);
    assert!(
        ops.contains(&format!(
            "mvmd {}\n  Multivariate variational mode decomposition",
            env!("CARGO_PKG_VERSION")
        )) && ops.contains("  inputs:  signal (series)\n  outputs: modes (decomposition)"),
        "{ops}"
    );
    assert!(ops.contains("stft") && ops.contains("reconstruct"));
    let (_, json) = binary(&dir, &["ops", "--json"]);
    let catalogue: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert!(
        catalogue["operations"]
            .as_array()
            .unwrap()
            .iter()
            .any(|o| o["name"] == "spectrum")
    );
    let (_, markdown) = binary(&dir, &["ops", "--markdown"]);
    assert!(markdown.starts_with("# Operation reference"));
    assert_ne!(
        binary(&dir, &["ops", "--json", "--markdown"]).0,
        0,
        "one format at a time"
    );

    for i in 4..12 {
        let path = dir.join(format!("data/sub-{i:02}/sub-{i:02}_task-rest_bold.h5"));
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let mut b = FileBuilder::new();
        b.create_dataset("signal")
            .with_f64_data(&[0.0, 1.0, 0.0, 1.0])
            .with_shape(&[1, 4]);
        b.write(&path).unwrap();
    }
    let (code, inspect) = binary(&dir, &["inspect", "{dir}/dataset.toml"]);
    assert_eq!(code, 0);
    assert!(
        inspect.starts_with("11 recordings, inputs [\"bold\"]"),
        "{inspect}"
    );
    assert!(
        inspect.contains("subject: 11 values: 01, 02, 03, 04, 05, 06, 07, 08 and 3 more"),
        "{inspect}"
    );
    assert!(inspect.contains("task: 1 values: rest\n"), "{inspect}");

    let (_, three) = binary(&setup("binary-few"), &["inspect", "{dir}/dataset.toml"]);
    assert!(three.contains("subject: 3 values: 01, 02, 03\n"), "{three}");
}
