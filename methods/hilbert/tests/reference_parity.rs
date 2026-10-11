//! Accuracy against scipy.signal.hilbert (analytic signal) and the `emd` package
//! (amplitude, frequency, Hilbert-Huang spectrum). Fixtures from `fixtures/generate.py`.

#![allow(clippy::unwrap_used)]

use modewise_decomposition::Decomposition;
use modewise_hilbert::{Padding, Weight, analytic_signal, hilbert_spectrum, instantaneous};
use modewise_signals::SamplingRate;
use modewise_spectral::{FrequencyGrid, GridScale};
use ndarray::{Array2, Array3, Axis};
use serde_json::Value;

fn fixtures() -> Value {
    let path = format!(
        "{}/tests/fixtures/reference.json",
        env!("CARGO_MANIFEST_DIR")
    );
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn matrix(v: &Value) -> Array2<f64> {
    let rows: Vec<Vec<f64>> = v
        .as_array()
        .unwrap()
        .iter()
        .map(|r| {
            r.as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap())
                .collect()
        })
        .collect();
    Array2::from_shape_vec((rows.len(), rows[0].len()), rows.concat()).unwrap()
}

fn as_decomposition(modes: &Array2<f64>, fs: f64) -> Decomposition {
    let (k, n) = modes.dim();
    let cube = modes.clone().into_shape_with_order((k, 1, n)).unwrap();
    Decomposition::new(
        cube,
        vec![0.0; k],
        SamplingRate::new(fs).unwrap(),
        vec!["a".into()],
        true,
    )
    .unwrap()
}

#[test]
fn analytic_signal_matches_scipy() {
    for (n, case) in fixtures().as_object().unwrap() {
        let modes = matrix(&case["modes"]);
        let (re, im) = (
            matrix(&case["analytic_real"]),
            matrix(&case["analytic_imag"]),
        );
        for (k, row) in modes.axis_iter(Axis(0)).enumerate() {
            for (t, z) in analytic_signal(row, Padding::None).iter().enumerate() {
                assert!(
                    (z.re - re[[k, t]]).abs() < 1e-12 && (z.im - im[[k, t]]).abs() < 1e-12,
                    "n={n} k={k} t={t}"
                );
            }
        }
    }
}

#[test]
fn amplitude_and_frequency_match_emd() {
    for (n, case) in fixtures().as_object().unwrap() {
        let fs = case["fs"].as_f64().unwrap();
        let ours = instantaneous(
            &as_decomposition(&matrix(&case["modes"]), fs),
            Padding::None,
        )
        .unwrap();
        let (amplitude, frequency) = (matrix(&case["amplitude"]), matrix(&case["frequency_hz"]));
        for k in 0..amplitude.nrows() {
            for t in 0..amplitude.ncols() {
                assert!(
                    (ours.amplitude()[[k, 0, t]] - amplitude[[k, t]]).abs() < 1e-10,
                    "amplitude n={n} k={k} t={t}"
                );
                assert!(
                    (ours.frequency_hz()[[k, 0, t]] - frequency[[k, t]]).abs() < 1e-9,
                    "frequency n={n} k={k} t={t}"
                );
            }
        }
    }
}

#[test]
fn hilbert_spectrum_matches_emd() {
    for (n, case) in fixtures().as_object().unwrap() {
        let fs = case["fs"].as_f64().unwrap();
        let ours = instantaneous(
            &as_decomposition(&matrix(&case["modes"]), fs),
            Padding::None,
        )
        .unwrap();
        let edges: Vec<f64> = case["edges"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect();
        let grid = FrequencyGrid::new(
            edges[0],
            *edges.last().unwrap(),
            edges.len() - 1,
            GridScale::Linear,
        )
        .unwrap();
        for (a, b) in grid.edges().iter().zip(&edges) {
            assert!((a - b).abs() < 1e-12);
        }
        let spectrum = hilbert_spectrum(&ours, &grid, Weight::Power).unwrap();
        let expected = matrix(&case["spectrum"]);
        let peak = expected.iter().cloned().fold(0.0, f64::max);
        let map: Array3<f64> = spectrum.map.power().to_owned();
        for f in 0..expected.nrows() {
            for t in 0..expected.ncols() {
                assert!(
                    (map[[0, f, t]] - expected[[f, t]]).abs() <= 1e-9 * peak,
                    "n={n} f={f} t={t}"
                );
            }
        }
        assert_eq!(spectrum.total, 2 * expected.ncols());
    }
}
