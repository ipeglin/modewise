#![allow(clippy::unwrap_used)]

use std::f64::consts::PI;

use modewise_decomposition::ModeDecomposer;
use modewise_signals::{MultichannelSeries, SamplingRate};
use modewise_vlmd::{Vlmd, VlmdParams};
use ndarray::{Array2, Axis, array};
use proptest::prelude::*;

/// Two latent tones mixed into four channels by a sparse A; channel 3 only hears latent 1.
fn mixed(fs: f64) -> (MultichannelSeries, Array2<f64>) {
    let n = 400;
    let a = array![[1.0, 0.0, 0.6, 0.0], [0.0, 1.0, -0.5, 0.8]];
    let z = Array2::from_shape_fn((2, n), |(l, t)| {
        let t = t as f64;
        (2.0 * PI * 0.04 * t).cos() * [1.0, 0.3][l]
            + (2.0 * PI * 0.21 * t + l as f64).cos() * [0.4, 1.0][l]
    });
    let x = a.t().dot(&z);
    let labels = (0..4).map(|c| format!("c{c}")).collect();
    (
        MultichannelSeries::new(x, SamplingRate::new(fs).unwrap(), labels).unwrap(),
        a,
    )
}

fn params() -> VlmdParams {
    VlmdParams {
        num_latents: Some(2),
        alpha: 2000.0,
        tolerance: 1e-7,
        ..VlmdParams::new(2)
    }
}

#[test]
fn recovers_tone_frequencies_and_the_zero_loading() {
    let (series, truth) = mixed(1.0);
    let out = Vlmd::new(params())
        .unwrap()
        .decompose_latent(&series)
        .unwrap();
    let centres = out.modes.centre_frequencies_hz();
    assert!(
        (centres[0] - 0.04).abs() < 0.002 && (centres[1] - 0.21).abs() < 0.002,
        "{centres:?}"
    );

    // The channel modes are band-limited copies of the channels: each matches its tone closely.
    for (k, f) in [(0, 0.04), (1, 0.21)] {
        let mode = out.modes.mode(k);
        let spectrum_peak = (0..4)
            .map(|c| dominant_frequency(mode.row(c).to_vec()))
            .collect::<Vec<_>>();
        assert!(
            spectrum_peak.iter().all(|p| (p - f).abs() < 0.005),
            "mode {k}: {spectrum_peak:?}"
        );
    }
    // Channel 3 carries no 0.04 tone from latent 0 in truth; its slow mode stays small.
    let slow = out.modes.mode(0);
    let energy = |c: usize| slow.row(c).iter().map(|v| v * v).sum::<f64>();
    assert!(
        energy(3) < 0.2 * energy(0),
        "{} vs {}",
        energy(3),
        energy(0)
    );
    assert_eq!(truth.dim(), out.loadings.values().dim());
}

#[test]
fn outputs_are_consistent_and_scale_with_the_sampling_rate() {
    let (series, _) = mixed(1.0);
    let (fast, _) = mixed(10.0);
    let vlmd = Vlmd::new(params()).unwrap();
    let out = vlmd.decompose_latent(&series).unwrap();
    let out10 = vlmd.decompose_latent(&fast).unwrap();
    for (a, b) in out
        .modes
        .centre_frequencies_hz()
        .iter()
        .zip(out10.modes.centre_frequencies_hz())
    {
        assert!((a * 10.0 - b).abs() < 1e-12);
    }
    assert_eq!(out.modes.modes(), out10.modes.modes());

    let a = out.loadings.values();
    assert!(a.iter().all(|v| v.abs() <= 1.0 + 1e-15));
    for k in 0..2 {
        let rebuilt = a.t().dot(&out.latent_modes.mode(k));
        for (x, y) in rebuilt.iter().zip(out.modes.mode(k)) {
            assert!((x - y).abs() < 1e-12);
        }
    }
    assert_eq!(out.latent_modes.labels(), ["latent_0", "latent_1"]);
    assert_eq!(out.loadings.channel_labels(), series.labels());
    assert_eq!(vlmd.decompose(&series).unwrap(), out.modes);
    let d = vlmd.descriptor();
    assert_eq!((d.name, d.multivariate), ("vlmd", true));
    assert!(out.iterations > 1 && out.iterations < 1000);
}

#[test]
fn the_iteration_cap_is_respected() {
    let (series, _) = mixed(1.0);
    let p = VlmdParams {
        max_iterations: 2,
        ..params()
    };
    let vlmd = Vlmd::new(p.clone()).unwrap();
    assert_eq!(vlmd.params(), &p);
    assert_eq!(vlmd.decompose_latent(&series).unwrap().iterations, 2);
}

#[test]
fn too_many_latents_are_refused() {
    let (series, _) = mixed(1.0);
    let p = VlmdParams {
        num_latents: Some(5),
        ..params()
    };
    assert!(Vlmd::new(p).unwrap().decompose_latent(&series).is_err());
    assert!(Vlmd::new(VlmdParams::new(0)).is_err());
}

#[test]
fn a_silent_signal_gives_empty_modes_at_zero_frequency() {
    let series = MultichannelSeries::new(
        Array2::zeros((2, 64)),
        SamplingRate::new(1.0).unwrap(),
        vec!["a".into(), "b".into()],
    )
    .unwrap();
    let out = Vlmd::new(VlmdParams::new(2))
        .unwrap()
        .decompose_latent(&series)
        .unwrap();
    assert_eq!(out.modes.centre_frequencies_hz(), [0.0, 0.0]);
    assert!(out.modes.modes().iter().all(|&v| v == 0.0));
}

fn dominant_frequency(x: Vec<f64>) -> f64 {
    let n = x.len();
    (1..n / 2)
        .map(|k| {
            let (re, im) = x.iter().enumerate().fold((0.0, 0.0), |(re, im), (t, v)| {
                let phase = 2.0 * PI * (k * t) as f64 / n as f64;
                (re + v * phase.cos(), im - v * phase.sin())
            });
            (k as f64 / n as f64, re * re + im * im)
        })
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .unwrap()
        .0
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(24))]

    #[test]
    fn any_signal_gives_finite_bounded_sorted_results(
        data in prop::collection::vec(-5.0f64..5.0, 3 * 40),
        modes in 1usize..4,
        latents in 1usize..4,
        lambda in 0.0f64..1.0,
    ) {
        let x = Array2::from_shape_vec((3, 40), data).unwrap();
        let series = MultichannelSeries::new(x, SamplingRate::new(2.0).unwrap(), vec!["a".into(), "b".into(), "c".into()]).unwrap();
        let p = VlmdParams { num_latents: Some(latents), reg_lambda: lambda, max_iterations: 50, ..VlmdParams::new(modes) };
        let out = Vlmd::new(p).unwrap().decompose_latent(&series).unwrap();
        let centres = out.modes.centre_frequencies_hz();
        prop_assert!(centres.windows(2).all(|w| w[0] <= w[1]));
        prop_assert!(centres.iter().all(|&f| (0.0..=1.0).contains(&f)));
        prop_assert!(out.loadings.values().iter().all(|v| v.abs() <= 1.0 + 1e-15));
        prop_assert!(out.modes.modes().iter().all(|v| v.is_finite()));
        prop_assert_eq!(out.latent_modes.modes().len_of(Axis(1)), latents);
    }
}
