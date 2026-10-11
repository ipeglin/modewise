//! EEMD and CEEMDAN against references built on PyEMD with the same noise realisations
//! (fixture from `fixtures/generate_ensemble.py`).

#![allow(clippy::unwrap_used)]

use modewise_emd::{Stop, testing};
use serde_json::Value;

fn fixture() -> Value {
    let path = format!(
        "{}/tests/fixtures/ensemble.json",
        env!("CARGO_MANIFEST_DIR")
    );
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

fn rows(v: &Value) -> Vec<Vec<f64>> {
    v.as_array().unwrap().iter().map(floats).collect()
}

fn max_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()))
}

fn sd(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mean = x.iter().sum::<f64>() / n;
    (x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt()
}

const FIXED: Stop = Stop::Fixed { siftings: 10 };

#[test]
fn eemd_matches_the_reference_trial_by_trial() {
    let f = fixture();
    let x = floats(&f["signal"]);
    let e = &f["eemd"];
    let scale = e["noise_sd"].as_f64().unwrap() * sd(&x);
    let noises: Vec<Vec<f64>> = rows(&f["noises"])
        .into_iter()
        .map(|w| w.iter().map(|v| v * scale).collect())
        .collect();
    let (imfs, residual) =
        testing::eemd(&x, &noises, FIXED, e["max_imfs"].as_u64().unwrap() as usize);
    let want = rows(&e["imfs"]);
    assert_eq!(imfs.len(), want.len());
    for (j, (got, want)) in imfs.iter().zip(&want).enumerate() {
        assert!(
            max_diff(got, want) < 1e-9,
            "IMF {}: {:e}",
            j + 1,
            max_diff(got, want)
        );
    }
    assert!(max_diff(&residual, &floats(&e["residual"])) < 1e-9);
}

#[test]
fn ceemdan_matches_a_transcription_of_iceemdan() {
    let f = fixture();
    let x = floats(&f["signal"]);
    let c = &f["ceemdan"];
    let (imfs, residual) = testing::ceemdan(
        &x,
        &rows(&f["noises"]),
        FIXED,
        c["epsilon"].as_f64().unwrap(),
        None,
    );
    let want = rows(&c["imfs"]);
    assert_eq!(imfs.len(), want.len());
    for (j, (got, want)) in imfs.iter().zip(&want).enumerate() {
        assert!(
            max_diff(got, want) < 1e-9,
            "IMF {}: {:e}",
            j + 1,
            max_diff(got, want)
        );
    }
    assert!(max_diff(&residual, &floats(&c["residual"])) < 1e-9);
    let sum: Vec<f64> = (0..x.len())
        .map(|t| imfs.iter().map(|m| m[t]).sum::<f64>() + residual[t])
        .collect();
    assert!(
        max_diff(&sum, &x) < 1e-12,
        "CEEMDAN modes sum to the signal"
    );
}
