//! Accuracy check against the reference Python MVMD (fixtures from `fixtures/generate.py`).

#![allow(clippy::unwrap_used)]

use modewise_decomposition::ModeDecomposer;
use modewise_mvmd::{FrequencyInit, Mvmd, MvmdParams};
use modewise_signals::{MultichannelSeries, SamplingRate};
use ndarray::{Array2, Array3, ArrayView2};
use serde_json::Value;

const MAX_FREQUENCY_ERROR_HZ: f64 = 1e-4;
const MAX_MODE_RELATIVE_ERROR: f64 = 1e-6;

fn load(name: &str) -> Value {
    let path = format!("{}/tests/fixtures/{name}.json", env!("CARGO_MANIFEST_DIR"));
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

fn matrix(v: &Value) -> Array2<f64> {
    let rows: Vec<Vec<f64>> = v.as_array().unwrap().iter().map(floats).collect();
    Array2::from_shape_vec((rows.len(), rows[0].len()), rows.concat()).unwrap()
}

fn cube(v: &Value) -> Array3<f64> {
    let slabs: Vec<Array2<f64>> = v.as_array().unwrap().iter().map(matrix).collect();
    let (c, t) = slabs[0].dim();
    let flat: Vec<f64> = slabs.iter().flat_map(|s| s.iter().copied()).collect();
    Array3::from_shape_vec((slabs.len(), c, t), flat).unwrap()
}

fn params(v: &Value) -> MvmdParams {
    let init = match v["init"].as_u64().unwrap() {
        1 => FrequencyInit::Linear,
        2 => FrequencyInit::Exponential,
        _ => FrequencyInit::Zero,
    };
    MvmdParams {
        init,
        tolerance: v["tolerance"].as_f64().unwrap(),
        tau: v["tau"].as_f64().unwrap(),
        ..MvmdParams::new(
            v["num_modes"].as_u64().unwrap() as usize,
            v["alpha"].as_f64().unwrap(),
        )
    }
}

fn relative_error(ours: ArrayView2<f64>, reference: ArrayView2<f64>) -> f64 {
    let diff: f64 = (&ours - &reference)
        .iter()
        .map(|d| d * d)
        .sum::<f64>()
        .sqrt();
    let norm: f64 = reference.iter().map(|r| r * r).sum::<f64>().sqrt();
    diff / norm
}

fn check(name: &str) {
    let fixture = load(name);
    let signal = matrix(&fixture["signal"]);
    let labels = (0..signal.nrows()).map(|i| format!("ch{i}")).collect();
    let fs = SamplingRate::new(fixture["sampling_rate_hz"].as_f64().unwrap()).unwrap();
    let series = MultichannelSeries::new(signal, fs, labels).unwrap();

    let result = Mvmd::new(params(&fixture["params"]))
        .unwrap()
        .decompose(&series)
        .unwrap();

    let expected_freqs = floats(&fixture["centre_frequencies_hz"]);
    for (ours, reference) in result.centre_frequencies_hz().iter().zip(&expected_freqs) {
        let error = (ours - reference).abs();
        assert!(
            error <= MAX_FREQUENCY_ERROR_HZ,
            "{name}: {ours} vs {reference} Hz"
        );
    }

    let expected_modes = cube(&fixture["modes"]);
    assert_eq!(result.modes().dim(), expected_modes.dim());
    for k in 0..result.n_modes() {
        let error = relative_error(
            result.mode(k),
            expected_modes.index_axis(ndarray::Axis(0), k),
        );
        assert!(
            error <= MAX_MODE_RELATIVE_ERROR,
            "{name}: mode {k} relative error {error:e}"
        );
    }
}

#[test]
fn matches_reference_on_two_tones() {
    check("two_tone");
}

#[test]
fn matches_reference_on_bold_like_signal_with_odd_length() {
    check("bold_like");
}
