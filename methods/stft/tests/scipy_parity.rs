//! Accuracy check against scipy.signal.stft (fixture from `fixtures/generate.py`).

#![allow(clippy::unwrap_used)]

use modewise_signals::{MultichannelSeries, SamplingRate};
use modewise_spectral::TimeFrequencyTransform;
use modewise_stft::Stft;
use ndarray::Array2;
use serde_json::Value;

fn floats(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

fn close(a: f64, b: f64, scale: f64) -> bool {
    (a - b).abs() <= 1e-9 * scale.max(1.0)
}

#[test]
fn matches_scipy() {
    let path = format!(
        "{}/tests/fixtures/scipy_two_channel.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let fixture: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();

    let rows: Vec<Vec<f64>> = fixture["signal"]
        .as_array()
        .unwrap()
        .iter()
        .map(floats)
        .collect();
    let data = Array2::from_shape_vec((rows.len(), rows[0].len()), rows.concat()).unwrap();
    let fs = SamplingRate::new(fixture["sampling_rate_hz"].as_f64().unwrap()).unwrap();
    let series = MultichannelSeries::new(data, fs, vec!["a".into(), "b".into()]).unwrap();

    let stft = Stft::new(
        fixture["window_len"].as_u64().unwrap() as usize,
        fixture["hop"].as_u64().unwrap() as usize,
    )
    .unwrap();
    let map = stft.transform(&series).unwrap();

    for (a, b) in map
        .frequencies_hz()
        .iter()
        .zip(floats(&fixture["frequencies_hz"]))
    {
        assert!(close(*a, b, 1.0), "frequency {a} vs {b}");
    }
    for (a, b) in map.times_s().iter().zip(floats(&fixture["times_s"])) {
        assert!(close(*a, b, 1.0), "time {a} vs {b}");
    }
    let expected: Vec<f64> = fixture["power"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|ch| ch.as_array().unwrap().iter().flat_map(floats))
        .collect();
    let peak = expected.iter().cloned().fold(0.0, f64::max);
    assert_eq!(map.power().len(), expected.len());
    for (a, b) in map.power().iter().zip(&expected) {
        assert!(close(*a, *b, peak), "power {a} vs {b}");
    }
}

#[test]
fn tone_peaks_at_its_bin() {
    let fs = 64.0;
    let data = Array2::from_shape_fn((1, 256), |(_, i)| {
        (2.0 * std::f64::consts::PI * 8.0 * i as f64 / fs).sin()
    });
    let series =
        MultichannelSeries::new(data, SamplingRate::new(fs).unwrap(), vec!["a".into()]).unwrap();
    let map = Stft::new(64, 16).unwrap().transform(&series).unwrap();
    let power = map.power();
    for frame in 0..map.times_s().len() {
        let column = power.slice(ndarray::s![0, .., frame]);
        let peak = column
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0;
        assert_eq!(map.frequencies_hz()[peak], 8.0);
    }
}

#[test]
fn signal_of_exactly_one_window_gives_one_frame() {
    let series = MultichannelSeries::new(
        Array2::ones((1, 16)),
        SamplingRate::new(1.0).unwrap(),
        vec!["a".into()],
    )
    .unwrap();
    let map = Stft::new(16, 4).unwrap().transform(&series).unwrap();
    assert_eq!(map.times_s(), [8.0]);
}

#[test]
fn rejects_short_signal_and_bad_parameters() {
    let data = Array2::zeros((1, 10));
    let series =
        MultichannelSeries::new(data, SamplingRate::new(1.0).unwrap(), vec!["a".into()]).unwrap();
    assert!(Stft::new(16, 4).unwrap().transform(&series).is_err());
    assert!(Stft::new(1, 4).is_err());
    assert!(Stft::new(16, 0).is_err());
}

#[test]
fn describes_itself_as_univariate() {
    let d = Stft::new(8, 4).unwrap().descriptor();
    assert_eq!((d.name, d.multivariate), ("stft", false));
}
