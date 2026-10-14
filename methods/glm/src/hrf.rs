use modewise_epochs::Events;
use ndarray::Array2;

/// Support of the canonical response, as in SPM's `spm_hrf` and nilearn's `spm_hrf`.
pub const HRF_LENGTH_S: f64 = 32.0;
const PEAK_SHAPE: u32 = 6;
const UNDERSHOOT_SHAPE: u32 = 16;
const UNDERSHOOT_RATIO: f64 = 1.0 / 6.0;

/// Regularised lower incomplete gamma P(k, x) for integer k: 1 − e^−x · Σ_{j<k} x^j / j!.
fn gamma_cdf(k: u32, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let (mut term, mut sum) = (1.0, 1.0);
    for j in 1..k {
        term *= x / f64::from(j);
        sum += term;
    }
    1.0 - (-x).exp() * sum
}

fn gamma_pdf(k: u32, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let factorial: f64 = (1..k).map(f64::from).product();
    x.powi(k as i32 - 1) * (-x).exp() / factorial
}

fn unscaled_integral(t: f64) -> f64 {
    let t = t.clamp(0.0, HRF_LENGTH_S);
    gamma_cdf(PEAK_SHAPE, t) - UNDERSHOOT_RATIO * gamma_cdf(UNDERSHOOT_SHAPE, t)
}

/// The canonical response at `t` seconds after an impulse: a gamma density peaking at 5 s minus
/// one sixth of one peaking at 15 s (SPM's defaults: shapes 6 and 16, scale 1 s), cut at 32 s
/// and scaled to unit integral.
pub fn canonical_hrf(t: f64) -> f64 {
    if !(0.0..=HRF_LENGTH_S).contains(&t) {
        return 0.0;
    }
    (gamma_pdf(PEAK_SHAPE, t) - UNDERSHOOT_RATIO * gamma_pdf(UNDERSHOOT_SHAPE, t))
        / unscaled_integral(HRF_LENGTH_S)
}

/// ∫₀ᵗ of [`canonical_hrf`]: 0 before the impulse and 1 from 32 s on.
pub fn canonical_hrf_integral(t: f64) -> f64 {
    unscaled_integral(t) / unscaled_integral(HRF_LENGTH_S)
}

/// One regressor per condition (event label, sorted), condition × sample: every event of the
/// condition as a unit box from onset to onset + duration, convolved with the canonical
/// response in closed form and read at sample times i / fs. An event with duration 0 is a
/// unit-area impulse, as SPM models events when every duration is 0.
pub fn task_regressors(
    events: &Events,
    n_samples: usize,
    sampling_rate_hz: f64,
) -> (Vec<String>, Array2<f64>) {
    let mut labels: Vec<String> = events.as_slice().iter().map(|e| e.label.clone()).collect();
    labels.sort();
    labels.dedup();
    let mut regressors = Array2::zeros((labels.len(), n_samples));
    for event in events.as_slice() {
        let row = labels.partition_point(|l| *l < event.label);
        for (i, value) in regressors.row_mut(row).iter_mut().enumerate() {
            let since = i as f64 / sampling_rate_hz - event.onset_s;
            *value += if event.duration_s == 0.0 {
                canonical_hrf(since)
            } else {
                canonical_hrf_integral(since) - canonical_hrf_integral(since - event.duration_s)
            };
        }
    }
    (labels, regressors)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use modewise_epochs::Event;

    use super::*;

    fn event(onset_s: f64, duration_s: f64, label: &str) -> Event {
        Event {
            onset_s,
            duration_s,
            label: label.into(),
        }
    }

    #[test]
    fn the_response_peaks_near_five_seconds_and_dips_after() {
        let peak = (0..3200)
            .map(|i| i as f64 / 100.0)
            .max_by(|a, b| canonical_hrf(*a).total_cmp(&canonical_hrf(*b)))
            .unwrap();
        assert!((4.5..5.5).contains(&peak), "{peak}");
        assert!(canonical_hrf(15.0) < 0.0);
        assert_eq!(canonical_hrf(-0.1), 0.0);
        assert_eq!(canonical_hrf(32.1), 0.0);
    }

    #[test]
    fn the_integral_matches_numerical_integration_of_the_density() {
        let dt = 1e-4;
        let mut sum = 0.0;
        for i in 0..=(20.0 / dt) as usize {
            let t = i as f64 * dt;
            if i % 25_000 == 0 {
                assert!((sum - canonical_hrf_integral(t)).abs() < 1e-7, "{t}");
            }
            sum += dt * (canonical_hrf(t) + canonical_hrf(t + dt)) / 2.0;
        }
        assert_eq!(canonical_hrf_integral(0.0), 0.0);
        assert_eq!(canonical_hrf_integral(40.0), 1.0);
    }

    #[test]
    fn a_long_block_plateaus_at_one() {
        let events = Events::new(vec![event(0.0, 100.0, "task")]).unwrap();
        let (labels, x) = task_regressors(&events, 200, 1.0);
        assert_eq!(labels, ["task"]);
        assert!((x[[0, 50]] - 1.0).abs() < 1e-12);
        assert!(x[[0, 0]].abs() < 1e-12);
        assert!(x[[0, 140]].abs() < 1e-12);
    }

    #[test]
    fn conditions_are_sorted_and_events_add_up() {
        let events = Events::new(vec![
            event(10.0, 0.0, "b"),
            event(0.0, 0.0, "a"),
            event(40.0, 0.0, "b"),
        ])
        .unwrap();
        let (labels, x) = task_regressors(&events, 200, 2.0);
        assert_eq!(labels, ["a", "b"]);
        assert!((x[[0, 10]] - canonical_hrf(5.0)).abs() < 1e-15);
        assert!((x[[1, 30]] - canonical_hrf(5.0)).abs() < 1e-15);
        assert!((x[[1, 45]] - canonical_hrf(12.5)).abs() < 1e-15);
        assert!((x[[1, 90]] - canonical_hrf(5.0)).abs() < 1e-15);
    }

    #[test]
    fn a_short_box_approaches_a_scaled_impulse() {
        let short = Events::new(vec![event(0.0, 1e-3, "x")]).unwrap();
        let impulse = Events::new(vec![event(0.0, 0.0, "x")]).unwrap();
        let (_, a) = task_regressors(&short, 30, 1.0);
        let (_, b) = task_regressors(&impulse, 30, 1.0);
        for i in 1..30 {
            assert!((a[[0, i]] / 1e-3 - b[[0, i]]).abs() < 1e-3, "{i}");
        }
    }
}
