//! Synthetic checks for VMD and NA-MVMD.

#![allow(clippy::unwrap_used)]

use modewise_decomposition::ModeDecomposer;
use modewise_mvmd::{FrequencyInit, Mvmd, MvmdParams, NaMvmd, NoiseParams, Vmd};
use modewise_signals::{MultichannelSeries, SamplingRate};
use ndarray::{Array2, Axis, s};

fn tones(fs: f64, n: usize, per_channel: &[&[(f64, f64)]]) -> MultichannelSeries {
    let data = Array2::from_shape_fn((per_channel.len(), n), |(c, i)| {
        let t = i as f64 / fs;
        per_channel[c]
            .iter()
            .map(|(a, f)| a * (2.0 * std::f64::consts::PI * f * t).cos())
            .sum()
    });
    let labels = (0..per_channel.len()).map(|c| format!("ch{c}")).collect();
    MultichannelSeries::new(data, SamplingRate::new(fs).unwrap(), labels).unwrap()
}

fn params(k: usize) -> MvmdParams {
    MvmdParams {
        init: FrequencyInit::Uniform,
        tau: 0.0,
        tolerance: 1e-9,
        ..MvmdParams::new(k, 2000.0)
    }
}

#[test]
fn vmd_finds_each_channels_own_tones_and_keeps_the_nyquist_bin() {
    let signal = tones(
        1.0,
        400,
        &[&[(1.0, 0.05), (0.5, 0.5)], &[(1.0, 0.12), (0.5, 0.3)]],
    );
    let d = Vmd::new(params(2)).unwrap().decompose(&signal).unwrap();
    let f = d.channel_centre_frequencies_hz().unwrap();
    // A tone at Nyquist leaks below it, so that centroid sits slightly under 0.5.
    for (got, want) in f.iter().zip([0.05, 0.12, 0.5, 0.3]) {
        assert!((got - want).abs() < 5e-3, "{got} vs {want}");
    }
    // vmdpy drops the Nyquist bin and would return nothing here.
    let nyquist = d.modes().slice(s![1, 0, ..]).to_owned();
    let rms = (nyquist.mapv(|v| v * v).sum() / nyquist.len() as f64).sqrt();
    // cos(πn) is ±1 at every sample, so its RMS equals its amplitude.
    assert!((rms - 0.5).abs() < 0.02, "Nyquist tone RMS {rms}");
    assert_eq!(
        d.centre_frequencies_hz()[0],
        (f[[0, 0]] + f[[0, 1]]) / 2.0,
        "median of two channels"
    );
    assert!(Vmd::new(MvmdParams::new(0, 1.0)).is_err());
}

#[test]
fn na_mvmd_is_mvmd_on_the_unit_sd_signal_with_noise_channels() {
    let signal = tones(
        1.0,
        300,
        &[&[(3.0, 0.05), (1.0, 0.2)], &[(0.2, 0.05), (0.4, 0.2)]],
    );
    let noise = NoiseParams {
        channels: 2,
        sd: 0.8,
        seed: 5,
    };
    let na = NaMvmd::new(params(3), noise.clone()).unwrap();
    let (augmented, sds) = na.augment(&signal).unwrap();
    assert_eq!(augmented.n_channels(), 4);
    assert_eq!(&augmented.labels()[2..], ["noise0", "noise1"]);
    for row in augmented.data().slice(s![..2, ..]).rows() {
        assert!((row.std(1.0) - 1.0).abs() < 1e-12);
    }
    for row in augmented.data().slice(s![2.., ..]).rows() {
        assert!((row.std(1.0) - 0.8).abs() < 0.1, "noise SD near 0.8");
    }
    let manual = Mvmd::new(params(3)).unwrap().decompose(&augmented).unwrap();
    let d = na.decompose(&signal).unwrap();
    assert_eq!(d.centre_frequencies_hz(), manual.centre_frequencies_hz());
    for (c, sd) in sds.iter().enumerate() {
        let want = manual.modes().index_axis(Axis(1), c).mapv(|v| v * sd);
        assert_eq!(d.modes().index_axis(Axis(1), c), want);
    }
    assert!(d.aligned_across_channels());
    assert_eq!(
        na.augment(&signal).unwrap().0,
        augmented,
        "seeded noise is reproducible"
    );
    let other = NaMvmd::new(params(3), NoiseParams { seed: 6, ..noise }).unwrap();
    assert_ne!(other.augment(&signal).unwrap().0, augmented);
}

#[test]
fn na_mvmd_rejects_bad_settings_and_constant_channels() {
    assert!(
        NaMvmd::new(
            params(2),
            NoiseParams {
                channels: 0,
                ..NoiseParams::default()
            }
        )
        .is_err()
    );
    for sd in [0.0, f64::NAN] {
        assert!(
            NaMvmd::new(
                params(2),
                NoiseParams {
                    sd,
                    ..NoiseParams::default()
                }
            )
            .is_err()
        );
    }
    let na = NaMvmd::new(params(2), NoiseParams::default()).unwrap();
    let flat = tones(1.0, 50, &[&[(1.0, 0.1)], &[(0.0, 0.1)]]);
    assert!(na.decompose(&flat).unwrap_err().to_string().contains("ch1"));
}

/// The paper's claim: with K larger than the number of components, the extra modes go to the
/// noise instead of splitting a component.
#[test]
fn surplus_modes_go_to_the_noise() {
    let signal = tones(
        1.0,
        400,
        &[&[(1.0, 0.04), (0.6, 0.15)], &[(0.5, 0.04), (1.0, 0.15)]],
    );
    let k = 5;
    let split = |freqs: &[f64]| {
        freqs
            .iter()
            .filter(|f| (**f - 0.04).abs() < 0.02 || (**f - 0.15).abs() < 0.02)
            .count()
    };
    let plain = Mvmd::new(params(k)).unwrap().decompose(&signal).unwrap();
    let na = NaMvmd::new(params(k), NoiseParams::default())
        .unwrap()
        .decompose(&signal)
        .unwrap();
    assert_eq!(
        split(plain.centre_frequencies_hz()),
        k,
        "MVMD splits the tones across all modes"
    );
    assert!(
        split(na.centre_frequencies_hz()) < k,
        "NA-MVMD sends some modes to the noise"
    );
}

#[test]
fn descriptors_name_each_method() {
    let vmd = Vmd::new(params(2)).unwrap().descriptor();
    let na = NaMvmd::new(params(2), NoiseParams::default())
        .unwrap()
        .descriptor();
    assert_eq!((vmd.name, vmd.multivariate), ("vmd", false));
    assert_eq!((na.name, na.multivariate), ("na_mvmd", true));
}
