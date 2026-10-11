//! VMD against vmdpy (fixture from `fixtures/generate_vmd.py`), per channel, at a fixed
//! iteration count. vmdpy returns the modes after its 498th update.

#![allow(clippy::unwrap_used)]

use modewise_decomposition::ModeDecomposer;
use modewise_mvmd::{FrequencyInit, MvmdParams, Vmd};
use modewise_signals::{MultichannelSeries, SamplingRate};
use ndarray::{Array2, Array3};
use serde_json::Value;

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

#[test]
fn per_channel_modes_match_vmdpy() {
    let path = format!("{}/tests/fixtures/vmd.json", env!("CARGO_MANIFEST_DIR"));
    let f: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let fs = SamplingRate::new(f["sampling_rate_hz"].as_f64().unwrap()).unwrap();
    let signal =
        MultichannelSeries::new(matrix(&f["signal"]), fs, vec!["a".into(), "b".into()]).unwrap();
    for case in f["cases"].as_array().unwrap() {
        let p = &case["params"];
        let params = MvmdParams {
            init: if p["init"].as_u64() == Some(1) {
                FrequencyInit::Uniform
            } else {
                FrequencyInit::Zero
            },
            tau: p["tau"].as_f64().unwrap(),
            tolerance: 1e-300,
            max_iterations: p["iterations"].as_u64().unwrap() as usize - 1,
            ..MvmdParams::new(
                p["num_modes"].as_u64().unwrap() as usize,
                p["alpha"].as_f64().unwrap(),
            )
        };
        let d = Vmd::new(params).unwrap().decompose(&signal).unwrap();
        let slabs: Vec<Array2<f64>> = case["modes"]
            .as_array()
            .unwrap()
            .iter()
            .map(matrix)
            .collect();
        let (c, t) = slabs[0].dim();
        let want = Array3::from_shape_vec(
            (slabs.len(), c, t),
            slabs.iter().flat_map(|s| s.iter().copied()).collect(),
        )
        .unwrap();
        let scale = want.iter().map(|v| v * v).sum::<f64>().sqrt();
        let error = (&d.modes() - &want)
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt()
            / scale;
        assert!(
            error < 1e-6,
            "init {} tau {}: relative mode error {error:e}",
            p["init"],
            p["tau"]
        );
        let freqs = matrix(&case["channel_centre_frequencies_hz"]);
        let ours = d.channel_centre_frequencies_hz().unwrap();
        for (a, b) in ours.iter().zip(&freqs) {
            assert!((a - b).abs() < 1e-4, "centre frequency {a} vs {b}");
        }
        assert!(!d.aligned_across_channels());
    }
}
