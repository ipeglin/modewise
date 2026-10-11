#![allow(clippy::unwrap_used)]

use std::f64::consts::PI;

use modewise_decomposition::Decomposition;
use modewise_hilbert::{
    Padding, Weight, analytic_signal, hilbert_spectrum, instantaneous, instantaneous_frequency,
};
use modewise_signals::SamplingRate;
use modewise_spectral::{FrequencyGrid, GridScale};
use ndarray::{Array1, Array3};
use rustfft::num_complex::Complex64;

#[test]
fn frequencies_above_a_quarter_of_the_sampling_rate_do_not_wrap() {
    let fs = 10.0;
    for f in [1.0, 2.6, 4.0, 4.9] {
        let z: Vec<Complex64> = (0..50)
            .map(|t| Complex64::from_polar(1.0, 2.0 * PI * f * t as f64 / fs))
            .collect();
        for (t, estimate) in instantaneous_frequency(&z, fs).iter().enumerate() {
            assert!((estimate - f).abs() < 1e-9, "f={f} t={t} got {estimate}");
        }
    }
}

#[test]
fn chirp_frequency_is_tracked_without_lag() {
    // f(t) = 1 + 0.5 t Hz; the centred estimate is exact for a linear chirp away from the ends.
    let fs = 100.0;
    let z: Vec<Complex64> = (0..800)
        .map(|i| {
            let t = i as f64 / fs;
            Complex64::from_polar(1.0, 2.0 * PI * (t + 0.25 * t * t))
        })
        .collect();
    let estimate = instantaneous_frequency(&z, fs);
    for (i, value) in estimate.iter().enumerate().take(799).skip(1) {
        let t = i as f64 / fs;
        assert!((value - (1.0 + 0.5 * t)).abs() < 1e-6, "t={t}: {value}");
    }
}

#[test]
fn mirror_padding_reduces_edge_error_for_non_periodic_signals() {
    // 3.3 cycles: not periodic in the window, so the plain FFT wraps a jump.
    let fs = 50.0;
    let n = 200;
    let x = Array1::from_shape_fn(n, |t| (2.0 * PI * 0.825 * t as f64 / fs).cos());
    let edge_error = |padding| {
        let z = analytic_signal(x.view(), padding);
        z[..10]
            .iter()
            .map(|v| (v.norm() - 1.0).abs())
            .fold(0.0, f64::max)
    };
    assert!(
        edge_error(Padding::Mirror) < edge_error(Padding::None) / 2.0,
        "{} vs {}",
        edge_error(Padding::Mirror),
        edge_error(Padding::None)
    );
}

#[test]
fn spectrum_puts_a_tone_in_its_bin_and_counts_drops() {
    let fs = 100.0;
    let n = 400;
    let tone = Array3::from_shape_fn((1, 1, n), |(_, _, t)| {
        2.0 * (2.0 * PI * 10.0 * t as f64 / fs).cos()
    });
    let modes = Decomposition::new(
        tone,
        vec![10.0],
        SamplingRate::new(fs).unwrap(),
        vec!["a".into()],
        true,
    )
    .unwrap();
    let attrs = instantaneous(&modes, Padding::Mirror).unwrap();
    let grid = FrequencyGrid::new(5.5, 15.5, 10, GridScale::Linear).unwrap(); // 10 Hz at a bin centre

    let power = hilbert_spectrum(&attrs, &grid, Weight::Power).unwrap();
    let bin_10 = grid.bin(10.0).unwrap();
    let middle = n / 2;
    let column: Vec<f64> = (0..grid.n_bins())
        .map(|b| power.map.power()[[0, b, middle]])
        .collect();
    assert!(
        (power.map.power()[[0, bin_10, middle]] - 4.0).abs() < 1e-4,
        "amplitude 2 gives power 4: bin {bin_10}, column {column:?}, IF {}, amplitude {}",
        attrs.frequency_hz()[[0, 0, middle]],
        attrs.amplitude()[[0, 0, middle]]
    );
    assert_eq!(power.dropped, 0);
    let amplitude = hilbert_spectrum(&attrs, &grid, Weight::Amplitude).unwrap();
    assert!((amplitude.map.power()[[0, bin_10, middle]] - 2.0).abs() < 1e-4);

    let narrow = FrequencyGrid::new(20.0, 30.0, 2, GridScale::Linear).unwrap();
    let outside = hilbert_spectrum(&attrs, &narrow, Weight::Power).unwrap();
    assert_eq!((outside.dropped, outside.total), (n, n));
    assert!(outside.map.power().iter().all(|v| *v == 0.0));
    assert!((power.map.times_s()[1] - 0.01).abs() < 1e-15);
}

#[test]
fn mirror_padding_matches_a_split_symmetric_extension() {
    for n in [10usize, 11] {
        let x = Array1::from_shape_fn(n, |t| ((t * t) as f64 * 0.37).sin() + t as f64 * 0.1);
        let left = n / 2;
        let right = n - left;
        let mut extended: Vec<f64> = (0..left).rev().map(|i| x[i]).collect();
        extended.extend(x.iter().copied());
        extended.extend((0..right).map(|i| x[n - 1 - i]));
        let by_hand = analytic_signal(Array1::from(extended).view(), Padding::None);
        let mirrored = analytic_signal(x.view(), Padding::Mirror);
        for (t, z) in mirrored.iter().enumerate() {
            assert!((z - by_hand[left + t]).norm() < 1e-12, "n={n} t={t}");
        }
    }
}

#[test]
fn spectrum_counts_every_mode_and_channel() {
    let fs = 20.0;
    let cube = Array3::from_shape_fn((2, 3, 40), |(k, c, t)| {
        (1.0 + c as f64) * (2.0 * PI * (2.0 + 3.0 * k as f64) * t as f64 / fs).cos()
    });
    let modes = Decomposition::new(
        cube,
        vec![2.0, 5.0],
        SamplingRate::new(fs).unwrap(),
        vec!["a".into(), "b".into(), "c".into()],
        true,
    )
    .unwrap();
    let attrs = instantaneous(&modes, Padding::None).unwrap();
    let grid = FrequencyGrid::new(0.5, 9.5, 9, GridScale::Linear).unwrap();
    let spectrum = hilbert_spectrum(&attrs, &grid, Weight::Power).unwrap();
    assert_eq!(spectrum.total, 2 * 3 * 40);
    assert_eq!(spectrum.dropped, 0);
    let bin_5 = grid.bin(5.0).unwrap();
    assert!(
        (spectrum.map.power()[[2, bin_5, 20]] - 9.0).abs() < 1e-6,
        "channel c has amplitude 3"
    );
}
