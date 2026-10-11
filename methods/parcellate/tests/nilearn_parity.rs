//! Accuracy against nilearn's NiftiLabelsMasker (fixture from `fixtures/generate.py`): an atlas on
//! a coarser, shifted grid, resampled onto the data by nearest neighbour.

#![allow(clippy::unwrap_used)]

use std::path::PathBuf;

use modewise_nifti::{read_bold, read_labels};
use modewise_parcellate::{Standardize, parcellate};
use serde_json::Value;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn expected(key: &str) -> Vec<Vec<f64>> {
    let f: Value =
        serde_json::from_str(&std::fs::read_to_string(fixture("nilearn.json")).unwrap()).unwrap();
    serde_json::from_value(f[key].clone()).unwrap()
}

#[test]
fn region_means_match_nilearn() {
    let bold = read_bold(&fixture("bold.nii.gz")).unwrap();
    assert_eq!(bold.repetition_time_s, Some(0.8f32 as f64));
    for atlas in ["atlas.nii.gz", "atlas_qform.nii.gz"] {
        let labels = read_labels(&fixture(atlas)).unwrap();
        for (standardize, key) in [
            (Standardize::None, "signals"),
            (Standardize::ZscoreSample, "zscored"),
        ] {
            let (signals, values) = parcellate(
                bold.data.view(),
                &bold.affine,
                labels.data.view(),
                &labels.affine,
                standardize,
            )
            .unwrap();
            assert_eq!(values, [1, 2, 3, 4, 5]);
            for (r, row) in expected(key).iter().enumerate() {
                for (t, want) in row.iter().enumerate() {
                    assert!(
                        (signals[[r, t]] - want).abs() < 1e-5,
                        "{atlas} {key} label {} t {t}: {} vs {want}",
                        values[r],
                        signals[[r, t]]
                    );
                }
            }
        }
    }
}

#[test]
fn labels_off_the_data_grid_give_nan_and_empty_atlases_fail() {
    let bold = read_bold(&fixture("bold.nii.gz")).unwrap();
    let mut labels = read_labels(&fixture("atlas.nii.gz")).unwrap();
    // Move the atlas far away: every label misses the data.
    labels.affine[0][3] += 1000.0;
    let mut atlas = labels.data.clone();
    atlas[[0, 0, 0]] = 9;
    let (signals, values) = parcellate(
        bold.data.view(),
        &bold.affine,
        atlas.view(),
        &labels.affine,
        Standardize::None,
    )
    .unwrap();
    assert_eq!(values, [1, 2, 3, 4, 5, 9]);
    assert!(signals.iter().all(|v| v.is_nan()));
    let empty = ndarray::Array3::<i32>::zeros((2, 2, 2));
    assert!(
        parcellate(
            bold.data.view(),
            &bold.affine,
            empty.view(),
            &labels.affine,
            Standardize::None
        )
        .is_err()
    );
    let flat = [[0.0; 4]; 4];
    assert!(
        parcellate(
            bold.data.view(),
            &bold.affine,
            labels.data.view(),
            &flat,
            Standardize::None
        )
        .is_err()
    );
}

#[test]
fn reading_rejects_wrong_dimensions() {
    assert!(
        read_bold(&fixture("atlas.nii.gz")).is_err(),
        "3D is not a run"
    );
    assert!(
        read_labels(&fixture("bold.nii.gz")).is_err(),
        "4D with many volumes is not an atlas"
    );
    assert!(read_bold(&fixture("missing.nii.gz")).is_err());
}
