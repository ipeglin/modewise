//! MODWT MRA against PyWavelets `mra(transform="swt")` (fixture from `fixtures/generate.py`).

#![allow(clippy::unwrap_used)]

use modewise_decomposition::ModeDecomposer;
use modewise_signals::{MultichannelSeries, SamplingRate};
use modewise_wavelet::{Boundary, ModwtMra};
use ndarray::{Array2, Array3};
use serde_json::Value;

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

#[test]
fn modes_match_pywavelets() {
    let path = format!("{}/tests/fixtures/mra.json", env!("CARGO_MANIFEST_DIR"));
    let f: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let signal = MultichannelSeries::new(
        matrix(&f["signal"]),
        SamplingRate::new(1.0).unwrap(),
        vec!["a".into(), "b".into()],
    )
    .unwrap();
    for case in f["cases"].as_array().unwrap() {
        let wavelet = case["wavelet"].as_str().unwrap();
        let level = case["level"].as_u64().unwrap() as usize;
        let d = ModwtMra::new(wavelet, Some(level), Boundary::Periodic)
            .unwrap()
            .decompose(&signal)
            .unwrap();
        let slabs: Vec<Array2<f64>> = case["modes"]
            .as_array()
            .unwrap()
            .iter()
            .map(matrix)
            .collect();
        let want = Array3::from_shape_vec(
            (slabs.len(), 2, 256),
            slabs.iter().flat_map(|s| s.iter().copied()).collect(),
        )
        .unwrap();
        let error = (&d.modes() - &want)
            .iter()
            .fold(0.0f64, |m, v| m.max(v.abs()));
        assert!(error < 1e-10, "{wavelet} level {level}: {error:e}");
    }
}
