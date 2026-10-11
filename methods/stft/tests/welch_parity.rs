//! Welch spectra and periodograms against scipy.signal (fixture from `fixtures/generate_welch.py`).

#![allow(clippy::unwrap_used)]

use modewise_signals::{MultichannelSeries, SamplingRate};
use modewise_spectral::TimeFrequencyTransform;
use modewise_stft::{Detrend, Welch, Window};
use ndarray::Array2;
use serde_json::Value;

fn floats(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

#[test]
fn spectra_match_scipy() {
    let path = format!("{}/tests/fixtures/welch.json", env!("CARGO_MANIFEST_DIR"));
    let f: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let rows: Vec<Vec<f64>> = f["signal"].as_array().unwrap().iter().map(floats).collect();
    let data = Array2::from_shape_vec((rows.len(), rows[0].len()), rows.concat()).unwrap();
    let fs = SamplingRate::new(f["sampling_rate_hz"].as_f64().unwrap()).unwrap();
    let signal = MultichannelSeries::new(data, fs, vec!["a".into(), "b".into()]).unwrap();
    for case in f["cases"].as_array().unwrap() {
        let welch = Welch {
            segment_len: case["segment_len"].as_u64().map(|v| v as usize),
            overlap: case["overlap"].as_u64().map(|v| v as usize),
            window: if case["window"] == "hann" {
                Window::Hann
            } else {
                Window::Boxcar
            },
            detrend: match case["detrend"].as_str().unwrap() {
                "linear" => Detrend::Linear,
                "constant" => Detrend::Constant,
                _ => Detrend::None,
            },
        };
        let map = welch.transform(&signal).unwrap();
        let name = case["name"].as_str().unwrap();
        let freqs = floats(&case["frequencies"]);
        assert_eq!(map.frequencies_hz().len(), freqs.len(), "{name}");
        for (got, want) in map.frequencies_hz().iter().zip(&freqs) {
            assert!((got - want).abs() < 1e-12, "{name}: {got} vs {want}");
        }
        for (c, want) in case["power"].as_array().unwrap().iter().enumerate() {
            for (k, w) in floats(want).iter().enumerate() {
                let got = map.power()[[c, k, 0]];
                assert!(
                    (got - w).abs() <= 1e-10 * w.abs().max(1e-12),
                    "{name} channel {c} bin {k}: {got} vs {w}"
                );
            }
        }
    }
}
