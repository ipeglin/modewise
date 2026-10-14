#![allow(clippy::unwrap_used)]

//! Synthetic check: with AR(1) noise and no task effect, AR(1) prewhitening keeps the false
//! positive rate near 5%, while OLS, which assumes independent samples, inflates it
//! (Purdon & Weisskoff 1998; Woolrich et al. 2001).

use modewise_epochs::{Event, Events};
use modewise_glm::{NoiseModel, fit, task_regressors};
use modewise_regress::{Nuisance, design};
use ndarray::Array2;

/// Standard normal draws (xorshift and Box–Muller), fixed seed.
fn normals(count: usize) -> Vec<f64> {
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut uniform = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    (0..count)
        .map(|_| {
            let (u, v) = (uniform().max(f64::MIN_POSITIVE), uniform());
            (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
        })
        .collect()
}

#[test]
fn prewhitening_controls_false_positives() {
    let (n, channels, rho) = (200, 2000, 0.5);
    let events = Events::new(
        (0..5)
            .map(|b| Event {
                onset_s: 40.0 * b as f64,
                duration_s: 20.0,
                label: "task".into(),
            })
            .collect(),
    )
    .unwrap();
    let (labels, task) = task_regressors(&events, n, 1.0);
    let nuisance = Nuisance {
        detrend: true,
        high_pass_hz: None,
    };
    let x = design(nuisance, n, 1.0, Array2::zeros((0, n)).view()).unwrap();
    let e = normals(n * channels);
    let mut signal = Array2::zeros((channels, n));
    for c in 0..channels {
        signal[[c, 0]] = e[c * n] / (1.0f64 - rho * rho).sqrt();
        for i in 1..n {
            signal[[c, i]] = rho * signal[[c, i - 1]] + e[c * n + i];
        }
    }
    let rate = |noise| {
        let f = fit(signal.view(), task.view(), &labels, x.view(), noise).unwrap();
        f.t.iter().filter(|t| t.abs() > 1.972).count() as f64 / channels as f64
    };
    let (ols, ar1) = (rate(NoiseModel::Ols), rate(NoiseModel::Ar1));
    // AR(1) lands a little above 5% (6.7% here): the lag-1 estimate from OLS residuals is biased
    // towards 0 (Worsley et al. 2002), as in nilearn, which uses the same estimate.
    assert!((0.035..0.08).contains(&ar1), "AR(1) {ar1}");
    assert!(ols > 0.1, "OLS {ols}");
}
