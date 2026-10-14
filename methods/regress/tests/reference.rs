//! Accuracy against nilearn's `signal.clean` and numpy least squares (fixture from
//! `fixtures/generate.py`), and synthetic signals with a known answer.

#![allow(clippy::unwrap_used)]

use std::f64::consts::PI;

use modewise_regress::{Nuisance, design, residuals};
use ndarray::{Array1, Array2, ArrayView2, Axis, array, stack};
use serde_json::Value;

fn matrix(v: &Value) -> Array2<f64> {
    let rows: Vec<Vec<f64>> = serde_json::from_value(v.clone()).unwrap();
    let flat: Vec<f64> = rows.iter().flatten().copied().collect();
    Array2::from_shape_vec((rows.len(), rows[0].len()), flat).unwrap()
}

fn max_difference(a: ArrayView2<f64>, b: ArrayView2<f64>) -> f64 {
    (&a - &b).iter().fold(0.0, |m, v| m.max(v.abs()))
}

#[test]
fn matches_nilearn_and_least_squares() {
    let fixture: Value = serde_json::from_str(include_str!("fixtures/reference.json")).unwrap();
    let signals = matrix(&fixture["signals"]);
    let confounds = matrix(&fixture["confounds"]);
    let fs = fixture["sampling_rate_hz"].as_f64().unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let nuisance = Nuisance {
            detrend: case["detrend"].as_bool().unwrap(),
            high_pass_hz: case["high_pass_hz"].as_f64(),
        };
        let x = design(nuisance, signals.ncols(), fs, confounds.view()).unwrap();
        let ours = residuals(signals.view(), x.view()).unwrap();
        for source in ["nilearn", "lstsq"] {
            let error = max_difference(ours.view(), matrix(&case[source]).view());
            assert!(error < 1e-9, "{nuisance:?} vs {source}: {error}");
        }
    }
}

#[test]
fn removes_a_known_confound_and_keeps_the_rest() {
    let n = 300;
    let t = Array1::from_shape_fn(n, |i| i as f64);
    let wanted = t.mapv(|i| (2.0 * PI * 0.07 * i).sin());
    let confound = t.mapv(|i| (2.0 * PI * 0.011 * i).cos() + 0.01 * i);
    let signal = stack(Axis(0), &[(&wanted + &(&confound * 3.0) + 4.0).view()]).unwrap();
    let nuisance = Nuisance {
        detrend: false,
        high_pass_hz: None,
    };
    let x = design(nuisance, n, 1.0, confound.view().insert_axis(Axis(0))).unwrap();
    let clean = residuals(signal.view(), x.view()).unwrap();
    // The sine is nearly orthogonal to the confound, so what remains is close to it.
    let error = (&clean.row(0) - &wanted)
        .iter()
        .fold(0.0f64, |m, v| m.max(v.abs()));
    assert!(error < 0.05, "{error}");
}

#[test]
fn residuals_are_orthogonal_to_every_column() {
    let signal = array![
        [1.0, 4.0, 2.0, 8.0, 5.0, 7.0],
        [0.0, -1.0, 3.0, 2.0, 2.0, 9.0]
    ];
    let regressors = array![[0.5, 0.1, 0.9, 0.3, 0.2, 0.4]];
    let nuisance = Nuisance {
        detrend: true,
        high_pass_hz: Some(0.2),
    };
    let x = design(nuisance, 6, 1.0, regressors.view()).unwrap();
    let r = residuals(signal.view(), x.view()).unwrap();
    assert!(r.dot(&x).iter().all(|v| v.abs() < 1e-10));
}

#[test]
fn high_pass_removes_slow_drift_only() {
    let n = 400;
    let fs = 1.0;
    // DCT-II basis vectors: k = 2 is 0.0025 Hz, k = 80 is 0.1 Hz; the cutoff 0.01 Hz keeps k ≤ 8.
    let basis = |k: f64| Array1::from_shape_fn(n, |i| (PI / n as f64 * (i as f64 + 0.5) * k).cos());
    let (fast, slow) = (basis(80.0), basis(2.0));
    let signal = stack(Axis(0), &[(&fast + &slow).view()]).unwrap();
    let nuisance = Nuisance {
        detrend: false,
        high_pass_hz: Some(0.01),
    };
    let x = design(nuisance, n, fs, Array2::zeros((0, n)).view()).unwrap();
    let r = residuals(signal.view(), x.view()).unwrap();
    let error = (&r.row(0) - &fast)
        .iter()
        .fold(0.0f64, |m, v| m.max(v.abs()));
    assert!(error < 1e-10, "{error}");
}
