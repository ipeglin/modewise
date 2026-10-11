//! Opt-in checks on local data that is not in the repo. Run with
//! `cargo test --release -p modewise_hdf5 -- --ignored --nocapture` and:
//! - `MODEWISE_TCP_H5`: an official ds005237 `*_parcellated.h5` file
//! - `MODEWISE_LEGACY_H5`: a legacy thesis consolidated `*_bold.h5` file

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::PathBuf;

use modewise_decomposition::ModeDecomposer;
use modewise_hdf5::{Layout, read_series};
use modewise_mvmd::{FrequencyInit, Mvmd, MvmdParams};
use modewise_signals::SamplingRate;

/// TR = 0.8 s in the ds005237 BOLD sidecars.
const TCP_HZ: f64 = 1.25;

fn env_path(var: &str) -> PathBuf {
    PathBuf::from(std::env::var(var).unwrap_or_else(|_| panic!("set {var} to run this test")))
}

#[test]
#[ignore = "needs MODEWISE_TCP_H5"]
fn reads_official_tcp_parcellated_file() {
    let fs = SamplingRate::new(TCP_HZ).unwrap();
    let series = read_series(
        &env_path("MODEWISE_TCP_H5"),
        "dataset",
        Layout::ChannelsByTime,
        fs,
        None,
    )
    .unwrap();
    println!(
        "channels {} samples {}",
        series.n_channels(),
        series.n_samples()
    );
    assert!(series.n_channels() > 400);
    assert!(series.n_samples() > 400);
}

#[test]
#[ignore = "needs MODEWISE_LEGACY_H5"]
fn mvmd_matches_legacy_thesis_output() {
    let path = env_path("MODEWISE_LEGACY_H5");
    let fs = SamplingRate::new(TCP_HZ).unwrap();
    let input = read_series(
        &path,
        "01fmri_parcellation/full_run_std",
        Layout::ChannelsByTime,
        fs,
        None,
    )
    .unwrap();
    let legacy = hdf5_pure::File::open(&path).unwrap();
    let legacy_modes = legacy
        .dataset("04hht/full_run_std/modes")
        .unwrap()
        .read_f64()
        .unwrap();
    let legacy_freqs = legacy
        .dataset("04hht/full_run_std/center_frequencies")
        .unwrap()
        .read_f64()
        .unwrap();

    for init in [
        FrequencyInit::Exponential,
        FrequencyInit::Zero,
        FrequencyInit::Linear,
    ] {
        let params = MvmdParams {
            init,
            tolerance: 1e-8,
            tau: 1e-3,
            max_iterations: 500,
            ..MvmdParams::new(10, 1000.0)
        };
        let started = std::time::Instant::now();
        let result = Mvmd::new(params).unwrap().decompose(&input).unwrap();
        let ours = result.modes();
        let per_mode = ours.len() / result.n_modes();
        let errors: Vec<f64> = (0..result.n_modes())
            .map(|k| {
                let ours = ours.index_axis(ndarray::Axis(0), k);
                let theirs = &legacy_modes[k * per_mode..(k + 1) * per_mode];
                let diff: f64 = ours.iter().zip(theirs).map(|(a, b)| (a - b).powi(2)).sum();
                let norm: f64 = theirs.iter().map(|b| b * b).sum();
                (diff / norm).sqrt()
            })
            .collect();
        let max_freq_error = result
            .centre_frequencies_hz()
            .iter()
            .zip(&legacy_freqs)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        let max_error = errors.iter().cloned().fold(0.0, f64::max);
        println!(
            "{init:?}: {:.1?}, max mode relative error {max_error:.2e}, max centre frequency error {max_freq_error:.2e} Hz",
            started.elapsed()
        );
        if max_error <= 1e-6 {
            assert!(
                max_freq_error <= 1e-3,
                "frequencies differ by {max_freq_error} Hz"
            );
            return;
        }
    }
    panic!("no frequency initialisation reproduced the legacy modes");
}
