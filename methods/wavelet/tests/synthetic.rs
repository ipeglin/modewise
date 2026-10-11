//! Synthetic checks for the MODWT MRA.

#![allow(clippy::unwrap_used)]

use modewise_decomposition::ModeDecomposer;
use modewise_signals::{MultichannelSeries, SamplingRate};
use modewise_wavelet::{Boundary, ModwtMra, wavelet_names};
use ndarray::{Array2, Axis, s};

fn series(n: usize, f: impl Fn(f64) -> f64) -> MultichannelSeries {
    let data = Array2::from_shape_fn((1, n), |(_, t)| f(t as f64));
    MultichannelSeries::new(data, SamplingRate::new(1.0).unwrap(), vec!["a".into()]).unwrap()
}

fn max_abs<'a>(a: impl IntoIterator<Item = &'a f64>) -> f64 {
    a.into_iter().fold(0.0f64, |m, v| m.max(v.abs()))
}

#[test]
fn modes_sum_to_the_signal_at_any_length_and_boundary() {
    let x = series(301, |t| (0.2 * t).sin() + 0.003 * t);
    for boundary in [Boundary::Periodic, Boundary::Reflection] {
        let d = ModwtMra::new("sym4", None, boundary)
            .unwrap()
            .decompose(&x)
            .unwrap();
        assert!(max_abs((&d.modes().sum_axis(Axis(0)) - &x.data()).iter()) < 1e-10);
        assert!(d.aligned_across_channels());
    }
}

#[test]
fn shifting_the_signal_shifts_every_mode() {
    let n = 128;
    let x = series(n, |t| (0.3 * t).sin() + (0.05 * t).cos());
    let shifted = series(n, |t| {
        (0.3 * ((t + 128.0 - 7.0) % 128.0)).sin() + (0.05 * ((t + 128.0 - 7.0) % 128.0)).cos()
    });
    let m = ModwtMra::new("db4", Some(3), Boundary::Periodic).unwrap();
    let (a, b) = (m.decompose(&x).unwrap(), m.decompose(&shifted).unwrap());
    for k in 0..a.n_modes() {
        for t in 0..n {
            assert!((a.modes()[[k, 0, t]] - b.modes()[[k, 0, (t + 7) % n]]).abs() < 1e-10);
        }
    }
}

#[test]
fn a_tone_lands_in_its_octave() {
    // D_j covers [fs/2^{j+1}, fs/2^j]: 0.35 Hz is in D_1 and 0.09 Hz in D_3.
    let x = series(1024, |t| {
        (2.0 * std::f64::consts::PI * 0.35 * t).sin()
            + (2.0 * std::f64::consts::PI * 0.09 * t).sin()
    });
    let d = ModwtMra::new("sym8", Some(4), Boundary::Periodic)
        .unwrap()
        .decompose(&x)
        .unwrap();
    let energy = |k: usize| d.mode(k).iter().map(|v| v * v).sum::<f64>();
    let total: f64 = (0..d.n_modes()).map(energy).sum();
    let (d1, d3) = (energy(4), energy(2));
    assert!(
        d1 / total > 0.4 && d3 / total > 0.35,
        "D1 {d1}, D3 {d3} of {total}"
    );
    assert!(
        (d.centre_frequencies_hz()[4] - 0.35).abs() < 0.05
            && (d.centre_frequencies_hz()[2] - 0.09).abs() < 0.03
    );
}

#[test]
fn reflection_avoids_the_jump_a_trend_makes_at_the_ends() {
    let x = series(200, |t| 0.05 * t);
    let edge = |boundary| {
        let d = ModwtMra::new("db2", Some(2), boundary)
            .unwrap()
            .decompose(&x)
            .unwrap();
        max_abs(d.modes().slice(s![2, 0, ..]).iter())
    };
    assert!(edge(Boundary::Reflection) < 0.1 * edge(Boundary::Periodic));
}

#[test]
fn settings_are_checked_and_levels_default_to_what_fits() {
    assert_eq!(wavelet_names().count(), 25);
    assert!(
        ModwtMra::new("morlet", None, Boundary::Periodic)
            .unwrap_err()
            .to_string()
            .contains("db4")
    );
    assert!(ModwtMra::new("db4", Some(0), Boundary::Periodic).is_err());
    let m = ModwtMra::new("db4", None, Boundary::Periodic).unwrap();
    // db4 has 8 taps: (2^J − 1)·7 + 1 ≤ n.
    assert_eq!(
        (
            m.default_levels(100),
            m.default_levels(8),
            m.default_levels(2)
        ),
        (3, 1, 1)
    );
    // Level 2 needs (2² − 1)·7 + 1 = 22 samples, level 3 needs 50.
    assert_eq!(
        (
            m.default_levels(21),
            m.default_levels(22),
            m.default_levels(49),
            m.default_levels(50)
        ),
        (1, 2, 2, 3)
    );
    // Filters longer than the series wrap around it, and the MRA stays additive.
    let tiny = series(16, |t| (0.9 * t).sin());
    let wrapped = ModwtMra::new("db4", Some(3), Boundary::Periodic)
        .unwrap()
        .decompose(&tiny)
        .unwrap();
    assert!(max_abs((&wrapped.modes().sum_axis(Axis(0)) - &tiny.data()).iter()) < 1e-10);
    assert_eq!(m.wavelet(), "db4");
    let short = series(16, |t| t.sin());
    assert!(
        ModwtMra::new("haar", Some(5), Boundary::Periodic)
            .unwrap()
            .decompose(&short)
            .is_err()
    );
    assert_eq!(
        ModwtMra::new("haar", Some(4), Boundary::Periodic)
            .unwrap()
            .decompose(&short)
            .unwrap()
            .n_modes(),
        5
    );
    let d = m.descriptor();
    assert_eq!((d.name, d.multivariate), ("modwt_mra", false));
}
