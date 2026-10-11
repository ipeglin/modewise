//! Synthetic checks for EEMD and CEEMDAN.

#![allow(clippy::unwrap_used)]

use modewise_decomposition::{Decomposition, ModeDecomposer};
use modewise_emd::{Ceemdan, Eemd, Emd, Stop};
use modewise_signals::{MultichannelSeries, SamplingRate};
use ndarray::{Array1, Array2, Axis};

const TAU: f64 = 2.0 * std::f64::consts::PI;

fn series(rows: Vec<Array1<f64>>) -> MultichannelSeries {
    let n = rows[0].len();
    let labels = (0..rows.len()).map(|c| format!("ch{c}")).collect();
    let data = Array2::from_shape_fn((rows.len(), n), |(c, t)| rows[c][t]);
    MultichannelSeries::new(data, SamplingRate::new(1.0).unwrap(), labels).unwrap()
}

/// Huang's intermittency test: a slow tone with bursts of a fast one.
fn intermittent(n: usize) -> (Array1<f64>, Array1<f64>) {
    let slow = Array1::from_shape_fn(n, |t| (TAU * 0.01 * t as f64).sin());
    let bursts = Array1::from_shape_fn(n, |t| {
        if (t / 100) % 3 == 1 {
            0.3 * (TAU * 0.2 * t as f64).sin()
        } else {
            0.0
        }
    });
    (slow.clone() + bursts, slow)
}

/// Best correlation between the slow tone and any single mode of channel 0.
fn best_match(d: &Decomposition, slow: &Array1<f64>) -> f64 {
    (0..d.n_modes())
        .map(|k| {
            let m = d.modes().index_axis(Axis(0), k).row(0).to_owned();
            let (a, b) = (&m - m.mean().unwrap(), slow - slow.mean().unwrap());
            a.dot(&b) / (a.dot(&a).sqrt() * b.dot(&b).sqrt()).max(1e-300)
        })
        .fold(f64::MIN, f64::max)
}

#[test]
fn noise_assistance_reduces_mode_mixing() {
    let (x, slow) = intermittent(1200);
    let s = series(vec![x]);
    let emd = best_match(
        &Emd::new(Stop::RILLING_EMD, None)
            .unwrap()
            .decompose(&s)
            .unwrap(),
        &slow,
    );
    let eemd = best_match(
        &Eemd::new(50, 0.2, 1).unwrap().decompose(&s).unwrap(),
        &slow,
    );
    let ceemdan = best_match(
        &Ceemdan::new(50, 0.2, 1).unwrap().decompose(&s).unwrap(),
        &slow,
    );
    println!("slow-tone match: emd {emd:.3}, eemd {eemd:.3}, ceemdan {ceemdan:.3}");
    assert!(
        eemd > emd && ceemdan > emd,
        "emd {emd}, eemd {eemd}, ceemdan {ceemdan}"
    );
    assert!(eemd > 0.95 && ceemdan > 0.95);
}

#[test]
fn ceemdan_sums_to_the_signal_and_eemd_to_it_plus_mean_noise() {
    let (x, _) = intermittent(400);
    let s = series(vec![x.clone(), Array1::from_elem(400, 2.0)]);
    let c = Ceemdan::new(20, 0.2, 3).unwrap().decompose(&s).unwrap();
    assert!(
        c.modes().iter().all(|v| v.is_finite()),
        "a constant channel is all residual, not NaN"
    );
    assert_eq!(
        c.modes().index_axis(Axis(1), 1).index_axis(Axis(0), 0),
        Array1::from_elem(400, 2.0)
    );
    let error = (&c.modes().sum_axis(Axis(0)) - &s.data())
        .iter()
        .fold(0.0f64, |m, v| m.max(v.abs()));
    assert!(error < 1e-12, "{error}");
    let e = Eemd::new(20, 0.2, 3).unwrap().decompose(&s).unwrap();
    let rms = |v: ndarray::ArrayView1<f64>| (v.dot(&v) / v.len() as f64).sqrt();
    let diff = &e.modes().sum_axis(Axis(0)).row(0) - &x;
    let noise_rms = 0.2 * x.std(1.0) / 20f64.sqrt();
    assert!(
        (rms(diff.view()) - noise_rms).abs() < 0.3 * noise_rms,
        "{} vs {noise_rms}",
        rms(diff.view())
    );
    // At most ⌊log₂ n⌋ − 1 IMFs; fewer once the residual stops oscillating.
    assert!(e.n_modes() <= (400f64.log2() as usize - 1) + 1 && e.n_modes() >= 4);
}

#[test]
fn seeds_reproduce_and_differ_and_channels_get_their_own_noise() {
    let (x, _) = intermittent(300);
    let s = series(vec![x.clone(), x]);
    for method in [
        &Eemd::new(10, 0.2, 7).unwrap() as &dyn ModeDecomposer,
        &Ceemdan::new(10, 0.2, 7).unwrap(),
    ] {
        let a = method.decompose(&s).unwrap();
        assert_eq!(
            a.modes(),
            method.decompose(&s).unwrap().modes(),
            "same seed, same modes"
        );
        assert_ne!(
            a.modes().index_axis(Axis(1), 0),
            a.modes().index_axis(Axis(1), 1),
            "independent noise per channel"
        );
    }
    let other = Eemd::new(10, 0.2, 8).unwrap().decompose(&s).unwrap();
    assert_ne!(
        other.modes(),
        Eemd::new(10, 0.2, 7)
            .unwrap()
            .decompose(&s)
            .unwrap()
            .modes()
    );
}

#[test]
fn bad_settings_are_rejected_and_methods_are_named() {
    assert!(Eemd::new(0, 0.2, 0).is_err());
    assert!(Eemd::new(10, 0.0, 0).is_err());
    assert!(Ceemdan::new(10, f64::NAN, 0).is_err());
    assert!(
        Ceemdan {
            max_imfs: Some(0),
            ..Ceemdan::new(10, 0.2, 0).unwrap()
        }
        .validated()
        .is_err()
    );
    assert!(
        Eemd {
            stop: Stop::Fixed { siftings: 0 },
            ..Eemd::new(10, 0.2, 0).unwrap()
        }
        .validated()
        .is_err()
    );
    let (e, c) = (
        Eemd::new(1, 0.2, 0).unwrap().descriptor(),
        Ceemdan::new(1, 0.2, 0).unwrap().descriptor(),
    );
    assert_eq!(
        (e.name, e.multivariate, c.name, c.multivariate),
        ("eemd", false, "ceemdan", false)
    );
    let limited = Ceemdan {
        max_imfs: Some(2),
        ..Ceemdan::new(10, 0.2, 0).unwrap()
    };
    assert_eq!(
        limited
            .decompose(&series(vec![intermittent(300).0]))
            .unwrap()
            .n_modes(),
        3
    );
}
