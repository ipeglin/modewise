//! Accuracy against numpy and scipy (fixture from `fixtures/generate.py`), and the behaviour
//! that defines each measure.

#![allow(clippy::unwrap_used)]

use std::f64::consts::PI;

use modewise_connectivity::Measure;
use modewise_decomposition::Decomposition;
use modewise_epochs::{EpochParams, EpochSource, Epochs, Event, Events, Window};
use modewise_fc::{Source, functional_connectivity};
use modewise_signals::{MultichannelSeries, SamplingRate};
use ndarray::{Array2, Array3, ArrayView2};
use serde_json::Value;

const PEARSON: Measure = Measure::Pearson { fisher_z: false };

fn matrix(v: &Value) -> Array2<f64> {
    let rows: Vec<Vec<f64>> = serde_json::from_value(v.clone()).unwrap();
    Array2::from_shape_fn((rows.len(), rows[0].len()), |(i, j)| rows[i][j])
}

fn assert_close(got: ArrayView2<f64>, want: &Array2<f64>, what: &str) {
    for ((i, j), w) in want.indexed_iter() {
        assert!(
            (got[[i, j]] - w).abs() < 1e-12,
            "{what} [{i},{j}]: {} vs {w}",
            got[[i, j]]
        );
    }
}

fn labels(n: usize) -> Vec<String> {
    (0..n).map(|c| format!("c{c}")).collect()
}

fn fixture() -> (Value, MultichannelSeries, Epochs) {
    let path = format!(
        "{}/tests/fixtures/reference.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let f: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let x = matrix(&f["signal"]);
    let series = MultichannelSeries::new(
        x.clone(),
        SamplingRate::new(1.0).unwrap(),
        labels(x.nrows()),
    )
    .unwrap();
    let windows: Vec<(f64, f64)> = serde_json::from_value(f["windows"].clone()).unwrap();
    let events = Events::new(
        windows
            .iter()
            .map(|&(a, b)| Event {
                onset_s: a,
                duration_s: b - a,
                label: "x".into(),
            })
            .collect(),
    )
    .unwrap();
    let params = EpochParams {
        lag_s: 0.0,
        window: Window::Duration,
        conditions: None,
        min_cycles: 1.0,
    };
    let epochs = Epochs::cut(EpochSource::Series(series.clone()), &events, params).unwrap();
    (f, series, epochs)
}

#[test]
fn pearson_and_plv_match_numpy_and_scipy() {
    let (f, series, epochs) = fixture();
    for (measure, key) in [(PEARSON, "pearson"), (Measure::PhaseLockingValue, "plv")] {
        let whole = functional_connectivity(Source::Series(&series), measure).unwrap();
        assert_close(whole.matrix(0, 0), &matrix(&f[key]), key);
        let per_epoch = functional_connectivity(Source::Epochs(&epochs), measure).unwrap();
        for (w, want) in f[format!("{key}_windows")]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
        {
            assert_close(
                per_epoch.matrix(w, 0),
                &matrix(want),
                &format!("{key} window {w}"),
            );
        }
        assert_eq!(per_epoch.windows().unwrap().len(), 2);
        assert!(
            per_epoch.resolvable().is_none(),
            "series epochs carry no cycle flags"
        );
    }
}

#[test]
fn fisher_z_is_atanh_with_an_undefined_diagonal() {
    let (_, series, _) = fixture();
    let r = functional_connectivity(Source::Series(&series), PEARSON).unwrap();
    let z = functional_connectivity(Source::Series(&series), Measure::Pearson { fisher_z: true })
        .unwrap();
    for ((i, j), &zij) in z.matrix(0, 0).indexed_iter() {
        if i == j {
            assert!(zij.is_nan());
        } else {
            assert!((zij - r.matrix(0, 0)[[i, j]].atanh()).abs() < 1e-12);
        }
    }
}

#[test]
fn phase_locking_ignores_amplitude_and_lag_but_pearson_does_not() {
    // Same 0.05 cycles/sample oscillation, one shifted a quarter cycle and amplitude-modulated.
    let n = 400;
    let x = Array2::from_shape_fn((2, n), |(c, t)| {
        let t = t as f64;
        let phase = 2.0 * PI * 0.05 * t;
        if c == 0 {
            phase.cos()
        } else {
            (1.0 + 0.5 * (0.01 * t).sin()) * (phase - PI / 2.0).cos()
        }
    });
    let series = MultichannelSeries::new(x, SamplingRate::new(1.0).unwrap(), labels(2)).unwrap();
    let plv = functional_connectivity(Source::Series(&series), Measure::PhaseLockingValue).unwrap();
    let r = functional_connectivity(Source::Series(&series), PEARSON).unwrap();
    assert!(
        plv.matrix(0, 0)[[0, 1]] > 0.99,
        "{}",
        plv.matrix(0, 0)[[0, 1]]
    );
    assert!(
        r.matrix(0, 0)[[0, 1]].abs() < 0.1,
        "{}",
        r.matrix(0, 0)[[0, 1]]
    );
}

#[test]
fn decompositions_give_one_matrix_per_mode_with_cycle_flags_per_epoch() {
    let n = 300;
    let modes = Array3::from_shape_fn((2, 3, n), |(k, c, t)| {
        let f = [0.02, 0.2][k];
        (2.0 * PI * f * t as f64 + c as f64 * 0.3 * (k as f64 + 1.0)).sin()
    });
    let d = Decomposition::new(
        modes,
        vec![0.02, 0.2],
        SamplingRate::new(1.0).unwrap(),
        labels(3),
        true,
    )
    .unwrap();
    let whole = functional_connectivity(Source::Decomposition(&d), PEARSON).unwrap();
    assert_eq!(whole.values().dim(), (1, 2, 3, 3));
    assert_eq!(whole.centre_frequencies_hz(), Some(&[0.02, 0.2][..]));
    assert!(whole.windows().is_none());

    // A 20 s epoch holds 0.4 cycles at 0.02 Hz and 4 at 0.2 Hz.
    let events = Events::new(vec![Event {
        onset_s: 100.0,
        duration_s: 20.0,
        label: "x".into(),
    }])
    .unwrap();
    let params = EpochParams {
        lag_s: 0.0,
        window: Window::Duration,
        conditions: None,
        min_cycles: 1.0,
    };
    let epochs = Epochs::cut(EpochSource::Decomposition(d.clone()), &events, params).unwrap();
    let per_epoch = functional_connectivity(Source::Epochs(&epochs), PEARSON).unwrap();
    assert_eq!(per_epoch.values().dim(), (1, 2, 3, 3));
    assert_eq!(
        per_epoch.resolvable().unwrap().row(0).to_vec(),
        [false, true]
    );
    let direct = modewise_fc::correlation(d.mode(1).slice(ndarray::s![.., 100..120]));
    assert_close(per_epoch.matrix(0, 1), &direct, "epoch of mode 1");
}

#[test]
fn flat_and_silent_channels_are_nan_and_other_sources_are_refused() {
    let mut x = Array2::from_shape_fn((3, 50), |(c, t)| ((t * (c + 1)) as f64 * 0.3).sin());
    x.row_mut(1).fill(0.0);
    let series = MultichannelSeries::new(x, SamplingRate::new(1.0).unwrap(), labels(3)).unwrap();
    for measure in [PEARSON, Measure::PhaseLockingValue] {
        let m = functional_connectivity(Source::Series(&series), measure).unwrap();
        let m = m.matrix(0, 0);
        assert!(m.row(1).iter().all(|v| v.is_nan()) && m.column(1).iter().all(|v| v.is_nan()));
        assert!((m[[0, 0]] - 1.0).abs() < 1e-12 && m[[0, 2]].is_finite());
    }

    let attrs = modewise_decomposition::InstantaneousModes::new(
        Array3::ones((1, 1, 40)),
        Array3::from_elem((1, 1, 40), 0.1),
        vec![0.1],
        SamplingRate::new(1.0).unwrap(),
        labels(1),
    )
    .unwrap();
    let events = Events::new(vec![Event {
        onset_s: 0.0,
        duration_s: 20.0,
        label: "x".into(),
    }])
    .unwrap();
    let params = EpochParams {
        lag_s: 0.0,
        window: Window::Duration,
        conditions: None,
        min_cycles: 1.0,
    };
    let epochs = Epochs::cut(EpochSource::Instantaneous(attrs), &events, params).unwrap();
    assert!(functional_connectivity(Source::Epochs(&epochs), PEARSON).is_err());
}

#[test]
fn silent_rows_have_no_phase_rather_than_nan() {
    let x = ndarray::array![[0.0, 0.0, 0.0, 0.0], [1.0, -1.0, 1.0, -1.0]];
    let (re, im, silent) = modewise_fc::unit_phases(x.view(), modewise_hilbert::Padding::Mirror);
    assert_eq!(silent, [true, false]);
    assert!(re.row(0).iter().chain(im.row(0)).all(|&v| v == 0.0));
    assert!(
        re.row(1)
            .iter()
            .zip(im.row(1))
            .all(|(a, b)| (a.hypot(*b) - 1.0).abs() < 1e-12)
    );
}
