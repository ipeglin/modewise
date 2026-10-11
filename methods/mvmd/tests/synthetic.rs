#![allow(clippy::unwrap_used)]

use std::f64::consts::PI;

use modewise_decomposition::ModeDecomposer;
use modewise_mvmd::{FrequencyInit, Mvmd, MvmdParams};
use modewise_signals::{MultichannelSeries, SamplingRate};
use ndarray::Array2;

fn tones(fs: f64, n: usize, freqs: &[f64], weights: &[&[f64]]) -> MultichannelSeries {
    let data = Array2::from_shape_fn((weights.len(), n), |(c, i)| {
        let t = i as f64 / fs;
        freqs
            .iter()
            .zip(weights[c])
            .map(|(f, w)| w * (2.0 * PI * f * t).sin())
            .sum()
    });
    let labels = (0..weights.len()).map(|i| format!("ch{i}")).collect();
    MultichannelSeries::new(data, SamplingRate::new(fs).unwrap(), labels).unwrap()
}

#[test]
fn recovers_tone_frequencies_shared_across_channels() {
    let signal = tones(
        100.0,
        1000,
        &[2.0, 12.0, 30.0],
        &[&[1.0, 0.5, 0.3], &[0.4, 1.0, 0.6]],
    );
    let params = MvmdParams {
        init: FrequencyInit::Linear,
        tolerance: 1e-7,
        tau: 0.0,
        ..MvmdParams::new(3, 2000.0)
    };
    let result = Mvmd::new(params).unwrap().decompose(&signal).unwrap();

    for (found, expected) in result.centre_frequencies_hz().iter().zip([2.0, 12.0, 30.0]) {
        assert!(
            (found - expected).abs() < 0.5,
            "found {found} Hz, expected {expected} Hz"
        );
    }
    assert!(result.aligned_across_channels());
    assert_eq!(result.labels(), signal.labels());
}

#[test]
fn modes_are_sorted_by_frequency() {
    let signal = tones(50.0, 500, &[3.0, 15.0], &[&[1.0, 1.0]]);
    let result = Mvmd::new(MvmdParams::new(2, 1000.0))
        .unwrap()
        .decompose(&signal)
        .unwrap();
    let f = result.centre_frequencies_hz();
    assert!(f[0] <= f[1], "{f:?}");
}

#[test]
fn rejects_invalid_parameters() {
    assert!(Mvmd::new(MvmdParams::new(0, 1000.0)).is_err());
}

#[test]
fn describes_itself_and_keeps_params() {
    use modewise_decomposition::ModeDecomposer;
    let mvmd = Mvmd::new(MvmdParams::new(3, 500.0)).unwrap();
    assert_eq!(mvmd.params().num_modes, 3);
    let d = mvmd.descriptor();
    assert_eq!((d.name, d.multivariate), ("mvmd", true));
}

#[test]
fn silent_signal_keeps_finite_frequencies() {
    let silent = MultichannelSeries::new(
        Array2::zeros((2, 64)),
        SamplingRate::new(10.0).unwrap(),
        vec!["a".into(), "b".into()],
    )
    .unwrap();
    let params = MvmdParams {
        init: FrequencyInit::Linear,
        ..MvmdParams::new(3, 1000.0)
    };
    let result = Mvmd::new(params).unwrap().decompose(&silent).unwrap();
    assert_eq!(
        result.centre_frequencies_hz(),
        [0.0, 2.5, 5.0],
        "zero-power modes keep their starting frequency"
    );
    assert!(result.modes().iter().all(|v| *v == 0.0));
}
