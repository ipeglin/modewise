//! Accuracy against pycwt, an independent implementation of Torrence & Compo (1998).
//! Fixture from `fixtures/generate.py`.

#![allow(clippy::unwrap_used)]

use std::f64::consts::PI;

use modewise_cwt::Cwt;
use modewise_signals::{MultichannelSeries, SamplingRate};
use modewise_spectral::{FrequencyGrid, GridScale, TimeFrequencyTransform};
use ndarray::Array2;
use serde_json::Value;

fn floats(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

fn grid() -> FrequencyGrid {
    FrequencyGrid::new(0.2, 4.0, 24, GridScale::Log).unwrap()
}

#[test]
fn matches_pycwt_power_frequencies_and_cone_of_influence() {
    let path = format!("{}/tests/fixtures/pycwt.json", env!("CARGO_MANIFEST_DIR"));
    let all: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    for (n, case) in all.as_object().unwrap() {
        let x = floats(&case["signal"]);
        let fs = case["fs"].as_f64().unwrap();
        let series = MultichannelSeries::new(
            Array2::from_shape_vec((1, x.len()), x.clone()).unwrap(),
            SamplingRate::new(fs).unwrap(),
            vec!["a".into()],
        )
        .unwrap();
        let map = Cwt::new(grid(), 6.0, false)
            .unwrap()
            .transform(&series)
            .unwrap();

        for (a, b) in map.frequencies_hz().iter().zip(floats(&case["centres"])) {
            assert!((a - b).abs() < 1e-12, "n={n} frequency {a} vs {b}");
        }
        let expected: Vec<Vec<f64>> = case["power"]
            .as_array()
            .unwrap()
            .iter()
            .map(floats)
            .collect();
        let peak = expected.iter().flatten().cloned().fold(0.0, f64::max);
        for (j, row) in expected.iter().enumerate() {
            for (t, &value) in row.iter().enumerate() {
                assert!(
                    (map.power()[[0, j, t]] - value).abs() <= 1e-9 * peak,
                    "n={n} j={j} t={t}"
                );
            }
        }
        let reliable: Vec<Vec<f64>> = case["reliable"]
            .as_array()
            .unwrap()
            .iter()
            .map(floats)
            .collect();
        let mask = map.reliable().unwrap();
        for (j, row) in reliable.iter().enumerate() {
            for (t, &value) in row.iter().enumerate() {
                assert_eq!(
                    mask[[j, t]],
                    value == 1.0,
                    "n={n} cone of influence j={j} t={t}"
                );
            }
        }
    }
}

#[test]
fn tones_peak_at_their_frequency_and_rectification_divides_by_scale() {
    let fs = 20.0;
    let n = 400;
    let data = Array2::from_shape_fn((1, n), |(_, t)| (2.0 * PI * 2.0 * t as f64 / fs).sin());
    let series =
        MultichannelSeries::new(data, SamplingRate::new(fs).unwrap(), vec!["a".into()]).unwrap();
    let grid = FrequencyGrid::new(0.5, 8.0, 40, GridScale::Log).unwrap();
    let plain = Cwt::new(grid.clone(), 6.0, false).unwrap();
    let map = plain.transform(&series).unwrap();
    let column: Vec<f64> = (0..40).map(|j| map.power()[[0, j, n / 2]]).collect();
    let peak = column
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0;
    assert!(
        (map.frequencies_hz()[peak] - 2.0).abs() < 0.1,
        "peak at {} Hz",
        map.frequencies_hz()[peak]
    );
    assert_eq!(map.sampling_rate(), Some(SamplingRate::new(fs).unwrap()));

    let rectified = Cwt::new(grid, 6.0, true)
        .unwrap()
        .transform(&series)
        .unwrap();
    for j in [0, 20, 39] {
        let scale = 1.0 / (plain.fourier_factor() * map.frequencies_hz()[j]);
        assert!((rectified.power()[[0, j, 100]] - map.power()[[0, j, 100]] / scale).abs() < 1e-9);
    }
}

#[test]
fn rejects_bad_settings() {
    let series = MultichannelSeries::new(
        Array2::zeros((1, 64)),
        SamplingRate::new(10.0).unwrap(),
        vec!["a".into()],
    )
    .unwrap();
    assert!(Cwt::new(grid(), 4.0, false).is_err());
    assert!(Cwt::new(grid(), f64::NAN, false).is_err());
    let above_nyquist = FrequencyGrid::new(1.0, 8.0, 4, GridScale::Log).unwrap();
    assert!(
        Cwt::new(above_nyquist, 6.0, false)
            .unwrap()
            .transform(&series)
            .is_err()
    );
    let d = Cwt::new(grid(), 6.0, false).unwrap().descriptor();
    assert_eq!((d.name, d.multivariate), ("cwt", false));
}

#[test]
fn nyquist_itself_is_allowed() {
    // A single log bin [4, 6.25) has geometric centre exactly 5 Hz, Nyquist at 10 Hz.
    let grid = FrequencyGrid::new(4.0, 6.25, 1, GridScale::Log).unwrap();
    assert_eq!(grid.centres(), [5.0]);
    let series = MultichannelSeries::new(
        Array2::zeros((1, 32)),
        SamplingRate::new(10.0).unwrap(),
        vec!["a".into()],
    )
    .unwrap();
    assert!(
        Cwt::new(grid, 6.0, false)
            .unwrap()
            .transform(&series)
            .is_ok()
    );
}
