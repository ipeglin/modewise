//! Synthetic checks for EMD and MEMD.

#![allow(clippy::unwrap_used)]

use modewise_decomposition::ModeDecomposer;
use modewise_emd::{Emd, Memd, Stop};
use modewise_signals::{MultichannelSeries, SamplingRate};
use ndarray::{Array2, Axis};

fn tones(n: usize, fs: f64, per_channel: &[&[(f64, f64)]]) -> MultichannelSeries {
    let data = Array2::from_shape_fn((per_channel.len(), n), |(c, i)| {
        let t = i as f64 / fs;
        per_channel[c]
            .iter()
            .map(|(a, f)| a * (2.0 * std::f64::consts::PI * f * t).sin())
            .sum()
    });
    let labels = (0..per_channel.len()).map(|c| format!("ch{c}")).collect();
    MultichannelSeries::new(data, SamplingRate::new(fs).unwrap(), labels).unwrap()
}

fn sums_back(d: &modewise_decomposition::Decomposition, s: &MultichannelSeries) -> f64 {
    (&d.modes().sum_axis(Axis(0)) - &s.data())
        .iter()
        .fold(0.0f64, |m, v| m.max(v.abs()))
}

fn extrema_and_zeros(x: &[f64]) -> (usize, usize) {
    let ext = x
        .windows(3)
        .filter(|w| (w[1] - w[0]) * (w[2] - w[1]) < 0.0)
        .count();
    let zeros = x.windows(2).filter(|w| w[0] * w[1] < 0.0).count();
    (ext, zeros)
}

#[test]
fn emd_separates_two_tones_into_imfs_that_sum_to_the_signal() {
    let s = tones(1000, 10.0, &[&[(1.0, 1.5), (1.0, 0.2)]]);
    let d = Emd::new(Stop::RILLING_EMD, None)
        .unwrap()
        .decompose(&s)
        .unwrap();
    assert!(
        sums_back(&d, &s) < 1e-12,
        "the residual makes the sum exact"
    );
    let k = d.n_modes();
    let f = d.channel_centre_frequencies_hz().unwrap();
    assert!(
        (f[[k - 1, 0]] - 1.5).abs() < 0.05,
        "IMF 1 at 1.5 Hz: {}",
        f[[k - 1, 0]]
    );
    assert!(
        (f[[k - 2, 0]] - 0.2).abs() < 0.05,
        "IMF 2 at 0.2 Hz: {}",
        f[[k - 2, 0]]
    );
    for j in 1..k {
        let imf = d.modes().slice(ndarray::s![j, 0, ..]).to_vec();
        let (ext, zeros) = extrema_and_zeros(&imf[50..950]);
        assert!(
            ext.abs_diff(zeros) <= 2,
            "IMF slot {j}: {ext} extrema, {zeros} zero crossings"
        );
    }
    assert!(!d.aligned_across_channels());
}

#[test]
fn channels_with_fewer_imfs_get_zero_modes_with_nan_frequency() {
    let s = tones(600, 10.0, &[&[(1.0, 1.5), (1.0, 0.2)], &[(1.0, 1.5)]]);
    let d = Emd::new(Stop::RILLING_EMD, Some(2))
        .unwrap()
        .decompose(&s)
        .unwrap();
    assert_eq!(d.n_modes(), 3, "two IMFs and the residual");
    let f = d.channel_centre_frequencies_hz().unwrap();
    assert!(
        f[[2, 0]].is_finite() && f[[2, 1]].is_finite(),
        "both have IMF 1"
    );
    assert!(sums_back(&d, &s) < 1e-12);
    for stop in [Stop::SNumber { count: 3 }, Stop::Fixed { siftings: 8 }] {
        let d = Emd::new(stop, None).unwrap().decompose(&s).unwrap();
        assert!(sums_back(&d, &s) < 1e-12);
        assert!(
            (d.channel_centre_frequencies_hz().unwrap()[[d.n_modes() - 1, 1]] - 1.5).abs() < 0.05
        );
    }
}

#[test]
fn a_signal_without_oscillation_is_all_residual() {
    let ramp = MultichannelSeries::new(
        Array2::from_shape_fn((1, 50), |(_, t)| t as f64),
        SamplingRate::new(1.0).unwrap(),
        vec!["a".into()],
    )
    .unwrap();
    let d = Emd::new(Stop::RILLING_EMD, None)
        .unwrap()
        .decompose(&ramp)
        .unwrap();
    assert_eq!(d.n_modes(), 1);
    assert_eq!(d.mode(0), ramp.data());
}

#[test]
fn memd_puts_a_shared_tone_in_the_same_imf_of_every_channel() {
    let s = tones(
        800,
        10.0,
        &[
            &[(1.0, 1.2), (0.6, 0.15)],
            &[(0.4, 1.2), (1.0, 0.15)],
            &[(0.8, 1.2)],
            &[(1.0, 0.15)],
        ],
    );
    let d = Memd::new(Stop::RILLING_MEMD, None, 64)
        .unwrap()
        .decompose(&s)
        .unwrap();
    assert!(d.aligned_across_channels());
    assert!(sums_back(&d, &s) < 1e-12);
    let k = d.n_modes();
    let fast = d.mode(k - 1);
    let rms = |c: usize| (fast.row(c).iter().map(|v| v * v).sum::<f64>() / 800.0).sqrt();
    assert!((rms(0) - 1.0 / 2f64.sqrt()).abs() < 0.1 && (rms(1) - 0.4 / 2f64.sqrt()).abs() < 0.1);
    assert!(
        rms(3) < 0.1,
        "a channel without the fast tone has little of it"
    );
    assert!((d.centre_frequencies_hz()[k - 1] - 1.2).abs() < 0.1);
}

#[test]
fn bad_settings_are_rejected() {
    let bad = [
        Stop::Rilling {
            sd: 0.0,
            sd2: 0.5,
            tol: 0.05,
        },
        Stop::Rilling {
            sd: 0.05,
            sd2: f64::NAN,
            tol: 0.05,
        },
        Stop::SNumber { count: 0 },
        Stop::Fixed { siftings: 0 },
    ];
    for stop in bad {
        assert!(Emd::new(stop, None).is_err());
        assert!(Memd::new(stop, None, 64).is_err());
    }
    assert!(Emd::new(Stop::RILLING_EMD, Some(0)).is_err());
    assert!(Memd::new(Stop::RILLING_MEMD, None, 5).is_err());
    let two = tones(100, 1.0, &[&[(1.0, 0.1)], &[(1.0, 0.2)]]);
    assert!(
        Memd::new(Stop::RILLING_MEMD, None, 64)
            .unwrap()
            .decompose(&two)
            .unwrap_err()
            .to_string()
            .contains("emd")
    );
    let emd = Emd::new(Stop::RILLING_EMD, None).unwrap().descriptor();
    let memd = Memd::new(Stop::RILLING_MEMD, None, 64)
        .unwrap()
        .descriptor();
    assert_eq!(
        (emd.name, emd.multivariate, memd.name, memd.multivariate),
        ("emd", false, "memd", true)
    );
}

/// After a pure tone is sifted out, only rounding noise is left; it must not become an IMF.
#[test]
fn rounding_noise_left_after_the_last_imf_stays_in_the_residual() {
    for amplitude in [1.0, 1000.0] {
        let data = Array2::from_shape_fn((1, 400), |(_, t)| {
            amplitude * (2.0 * std::f64::consts::PI * 0.1 * t as f64).sin()
        });
        let s = MultichannelSeries::new(data, SamplingRate::new(1.0).unwrap(), vec!["a".into()])
            .unwrap();
        let d = Emd::new(Stop::Fixed { siftings: 50 }, None)
            .unwrap()
            .decompose(&s)
            .unwrap();
        assert_eq!(
            d.n_modes(),
            2,
            "amplitude {amplitude}: one IMF and the residual"
        );
    }
}

#[test]
fn centroids_weight_the_one_sided_spectrum_including_nyquist() {
    let nyquist = Array2::from_shape_fn((1, 8), |(_, t)| if t % 2 == 0 { 1.0 } else { -1.0 });
    assert_eq!(modewise_emd::testing::centroid_hz(nyquist.view(), 2.0), 1.0);
    let dc = Array2::from_elem((1, 8), 3.0);
    assert_eq!(modewise_emd::testing::centroid_hz(dc.view(), 2.0), 0.0);
    assert!(modewise_emd::testing::centroid_hz(Array2::zeros((1, 8)).view(), 2.0).is_nan());
}
