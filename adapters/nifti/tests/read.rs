#![allow(clippy::unwrap_used)]

use std::path::PathBuf;

use modewise_nifti::{read_bold, read_labels};
use serde_json::Value;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../methods/parcellate/tests/fixtures")
        .join(name)
}

fn nibabel_affine(key: &str) -> Vec<Vec<f64>> {
    let f: Value =
        serde_json::from_str(&std::fs::read_to_string(fixture("nilearn.json")).unwrap()).unwrap();
    serde_json::from_value(f["headers"][key].clone()).unwrap()
}

fn assert_affine(got: &[[f64; 4]; 4], want: &[Vec<f64>]) {
    for r in 0..4 {
        for c in 0..4 {
            assert!(
                (got[r][c] - want[r][c]).abs() < 1e-5,
                "[{r}][{c}]: {} vs {}",
                got[r][c],
                want[r][c]
            );
        }
    }
}

#[test]
fn affines_match_nibabel_for_sform_and_rotated_qform() {
    assert_affine(
        &read_labels(&fixture("atlas.nii.gz")).unwrap().affine,
        &nibabel_affine("atlas"),
    );
    assert_affine(
        &read_labels(&fixture("rotated_qform.nii.gz"))
            .unwrap()
            .affine,
        &nibabel_affine("rotated_qform"),
    );
}

#[test]
fn repetition_time_units_and_volume_shapes() {
    assert!(
        (read_bold(&fixture("tr_ms.nii.gz"))
            .unwrap()
            .repetition_time_s
            .unwrap()
            - 0.8)
            .abs()
            < 1e-9,
        "800 ms"
    );
    assert!(
        (read_bold(&fixture("bold.nii.gz"))
            .unwrap()
            .repetition_time_s
            .unwrap()
            - 0.8)
            .abs()
            < 1e-6,
        "0.8 s"
    );
    assert_eq!(
        read_bold(&fixture("rotated_qform.nii.gz"))
            .err()
            .map(|e| e.to_string().contains("4D")),
        Some(true)
    );
    let flat = read_labels(&fixture("labels_4d.nii.gz")).unwrap();
    let plain = read_labels(&fixture("atlas.nii.gz")).unwrap();
    assert_eq!(flat.data, plain.data, "a one-volume 4D atlas reads as 3D");
    assert!(
        read_labels(&fixture("tr_ms.nii.gz")).is_err(),
        "three volumes are not an atlas"
    );
}

#[test]
fn the_sform_wins_when_both_are_set() {
    let both = read_labels(&fixture("sform_wins.nii.gz")).unwrap();
    assert_affine(&both.affine, &nibabel_affine("sform_wins"));
    assert_affine(&both.affine, &nibabel_affine("atlas"));
}

#[test]
fn microsecond_and_zero_repetition_times() {
    let us = read_bold(&fixture("tr_us.nii.gz"))
        .unwrap()
        .repetition_time_s
        .unwrap();
    assert!((us - 0.8).abs() < 1e-9, "800000 µs: {us}");
    assert_eq!(
        read_bold(&fixture("tr_zero.nii.gz"))
            .unwrap()
            .repetition_time_s,
        None,
        "a zero TR is no TR"
    );
}
