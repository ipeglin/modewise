#![allow(clippy::unwrap_used)]

use modewise_decomposition::Decomposition;
use modewise_epochs::{EpochError, EpochParams, EpochSource, Epochs, Event, Events, Window};
use modewise_signals::{MultichannelSeries, SamplingRate};
use ndarray::{Array2, Array3, s};

const FS: f64 = 1.25;

fn ev(onset_s: f64, duration_s: f64, label: &str) -> Event {
    Event {
        onset_s,
        duration_s,
        label: label.into(),
    }
}

fn params(lag_s: f64) -> EpochParams {
    EpochParams {
        lag_s,
        window: Window::Duration,
        conditions: None,
        min_cycles: 1.0,
    }
}

fn series(n: usize) -> MultichannelSeries {
    let data = Array2::from_shape_fn((2, n), |(c, t)| (c * 1000 + t) as f64);
    MultichannelSeries::new(
        data,
        SamplingRate::new(FS).unwrap(),
        vec!["a".into(), "b".into()],
    )
    .unwrap()
}

fn decomposition(n: usize, freqs: Vec<f64>) -> Decomposition {
    let modes = Array3::from_shape_fn((freqs.len(), 1, n), |(k, _, t)| (k * 1000 + t) as f64);
    Decomposition::new(
        modes,
        freqs,
        SamplingRate::new(FS).unwrap(),
        vec!["a".into()],
        true,
    )
    .unwrap()
}

#[test]
fn slow_components_are_flagged_in_short_blocks() {
    let events = Events::new(vec![ev(10.0, 18.0, "face"), ev(60.0, 18.0, "face")]).unwrap();
    let epochs = Epochs::cut(
        EpochSource::Decomposition(decomposition(200, vec![0.02, 0.2])),
        &events,
        params(0.0),
    )
    .unwrap();
    let resolvable = epochs.resolvable().unwrap();
    assert!(
        epochs.series(0).is_none(),
        "a decomposition source has modes, not a series"
    );
    assert_eq!(resolvable.dim(), (2, 2));
    assert!(
        !resolvable[[0, 0]],
        "0.02 Hz has a 50 s period; 18 s cannot hold one cycle"
    );
    assert!(resolvable[[0, 1]], "0.2 Hz fits 3.6 cycles");

    let strict = EpochParams {
        min_cycles: 4.0,
        ..params(0.0)
    };
    let epochs = Epochs::cut(
        EpochSource::Decomposition(decomposition(200, vec![0.02, 0.2])),
        &events,
        strict,
    )
    .unwrap();
    assert!(!epochs.resolvable().unwrap()[[0, 1]], "3.6 cycles < 4");
}

#[test]
fn epochs_are_exact_slices_of_the_full_run() {
    let full = series(100);
    let events = Events::new(vec![ev(8.0, 4.0, "x"), ev(11.0602769852, 1.968626976, "y")]).unwrap();
    let epochs = Epochs::cut(EpochSource::Series(full.clone()), &events, params(0.0)).unwrap();
    let windows = epochs.windows();
    assert_eq!(
        (windows[0].start, windows[0].end),
        (10, 15),
        "8 s = sample 10; [8, 12) s covers samples 10..15"
    );
    let first_after = |t: f64| (t * FS).ceil() as usize;
    assert_eq!(
        (windows[1].start, windows[1].end),
        (
            first_after(11.0602769852),
            first_after(11.0602769852 + 1.968626976)
        )
    );
    for (i, w) in windows.iter().enumerate() {
        assert_eq!(
            epochs.series(i).unwrap(),
            full.data().slice(s![.., w.start..w.end])
        );
    }
    assert!(epochs.modes(0).is_none() && epochs.resolvable().is_none());
}

#[test]
fn lag_shifts_every_window() {
    let events = Events::new(vec![ev(8.0, 4.0, "x"), ev(20.0, 4.0, "x")]).unwrap();
    let plain = Epochs::cut(EpochSource::Series(series(100)), &events, params(0.0)).unwrap();
    let lagged = Epochs::cut(EpochSource::Series(series(100)), &events, params(4.0)).unwrap();
    for (a, b) in plain.windows().iter().zip(lagged.windows()) {
        assert_eq!(
            (b.start, b.end),
            (a.start + 5, a.end + 5),
            "4 s at 1.25 Hz is exactly 5 samples"
        );
        assert_eq!(
            a.event_onset_s, b.event_onset_s,
            "the event keeps its own onset"
        );
    }
}

#[test]
fn windows_outside_the_recording_are_dropped_not_padded() {
    let events = Events::new(vec![
        ev(-5.0, 4.0, "early"),
        ev(10.0, 4.0, "ok"),
        ev(78.0, 4.0, "late"),
        ev(30.0, 0.0, "impulse"),
    ])
    .unwrap();
    let epochs = Epochs::cut(EpochSource::Series(series(100)), &events, params(0.0)).unwrap();
    assert_eq!(
        epochs
            .windows()
            .iter()
            .map(|w| w.label.as_str())
            .collect::<Vec<_>>(),
        ["ok"]
    );
    let reasons: Vec<(&str, &str)> = epochs
        .dropped()
        .iter()
        .map(|d| (d.label.as_str(), d.reason.as_str()))
        .collect();
    assert!(reasons[0].0 == "early" && reasons[0].1.contains("before the recording"));
    assert!(
        reasons
            .iter()
            .any(|(l, r)| *l == "late" && r.contains("after the recording"))
    );
    assert!(
        reasons
            .iter()
            .any(|(l, r)| *l == "impulse" && r.contains("at least 2"))
    );
}

#[test]
fn fixed_windows_cut_trials_like_eeg_epochs() {
    let fs = 100.0;
    let data = Array2::from_shape_fn((1, 500), |(_, t)| t as f64);
    let eeg =
        MultichannelSeries::new(data, SamplingRate::new(fs).unwrap(), vec!["Cz".into()]).unwrap();
    let events = Events::new(vec![ev(1.0, 0.0, "tone"), ev(2.5, 0.0, "tone")]).unwrap();
    let p = EpochParams {
        window: Window::Fixed {
            tmin_s: -0.2,
            tmax_s: 0.5,
        },
        ..params(0.0)
    };
    let epochs = Epochs::cut(EpochSource::Series(eeg), &events, p).unwrap();
    let w = &epochs.windows()[0];
    assert_eq!((w.start, w.end), (80, 150));
    assert!((epochs.duration_s(0) - 0.7).abs() < 1e-12);
}

#[test]
fn back_to_back_blocks_share_no_sample_and_conditions_filter() {
    let events = Events::new(vec![
        ev(8.0, 4.0, "face"),
        ev(12.0, 4.0, "shape"),
        ev(16.0, 4.0, "face"),
    ])
    .unwrap();
    let epochs = Epochs::cut(EpochSource::Series(series(100)), &events, params(0.0)).unwrap();
    let w = epochs.windows();
    assert_eq!(w[0].end, w[1].start);
    assert_eq!(w[1].end, w[2].start);

    let faces = EpochParams {
        conditions: Some(vec!["face".into()]),
        ..params(0.0)
    };
    let only = Epochs::cut(EpochSource::Series(series(100)), &events, faces).unwrap();
    assert_eq!(only.windows().len(), 2);
    assert!(
        only.dropped().is_empty(),
        "filtered events are not 'dropped'"
    );
}

#[test]
fn rejects_invalid_parameters() {
    let events = Events::new(vec![ev(8.0, 4.0, "x")]).unwrap();
    let cut = |p: EpochParams| Epochs::cut(EpochSource::Series(series(100)), &events, p);
    assert!(matches!(
        cut(params(f64::NAN)),
        Err(EpochError::InvalidParam { name: "lag_s", .. })
    ));
    assert!(matches!(
        cut(EpochParams {
            min_cycles: 0.0,
            ..params(0.0)
        }),
        Err(EpochError::InvalidParam {
            name: "min_cycles",
            ..
        })
    ));
    let bad_window = EpochParams {
        window: Window::Fixed {
            tmin_s: 0.5,
            tmax_s: 0.5,
        },
        ..params(0.0)
    };
    assert!(matches!(
        cut(bad_window),
        Err(EpochError::EmptyWindow { .. })
    ));
    let nan_window = EpochParams {
        window: Window::Fixed {
            tmin_s: f64::NAN,
            tmax_s: 0.5,
        },
        ..params(0.0)
    };
    assert!(matches!(
        cut(nan_window),
        Err(EpochError::InvalidParam { name: "tmin_s", .. })
    ));
}

#[test]
fn from_parts_round_trips_and_checks_windows() {
    let events = Events::new(vec![ev(8.0, 4.0, "x"), ev(90.0, 4.0, "late")]).unwrap();
    let epochs = Epochs::cut(EpochSource::Series(series(100)), &events, params(1.0)).unwrap();
    let rebuilt = Epochs::from_parts(
        epochs.source().clone(),
        epochs.windows().to_vec(),
        epochs.dropped().to_vec(),
        epochs.params().clone(),
    )
    .unwrap();
    assert_eq!(rebuilt, epochs);

    let mut bad = epochs.windows().to_vec();
    bad[0].end = 101;
    let err = Epochs::from_parts(
        epochs.source().clone(),
        bad,
        vec![],
        epochs.params().clone(),
    );
    assert!(matches!(
        err,
        Err(EpochError::WindowOutOfRange { index: 0, .. })
    ));
}

#[test]
fn boundary_windows_are_kept() {
    // 100 samples at 1.25 Hz = 80 s.
    let events = Events::new(vec![
        ev(0.0, 4.0, "first"),
        ev(76.0, 4.0, "last"),
        ev(40.0, 1.6, "two"),
    ])
    .unwrap();
    let epochs = Epochs::cut(EpochSource::Series(series(100)), &events, params(0.0)).unwrap();
    let kept: Vec<(&str, usize, usize)> = epochs
        .windows()
        .iter()
        .map(|w| (w.label.as_str(), w.start, w.end))
        .collect();
    assert_eq!(
        kept,
        [("first", 0, 5), ("two", 50, 52), ("last", 95, 100)],
        "{:?}",
        epochs.dropped()
    );
}

#[test]
fn exactly_min_cycles_counts_as_resolvable() {
    // 4 s epoch (5 samples at 1.25 Hz) and a 0.25 Hz mode: exactly one cycle.
    let events = Events::new(vec![ev(8.0, 4.0, "x")]).unwrap();
    let epochs = Epochs::cut(
        EpochSource::Decomposition(decomposition(100, vec![0.25, 0.2])),
        &events,
        params(0.0),
    )
    .unwrap();
    let r = epochs.resolvable().unwrap();
    assert!(r[[0, 0]], "1.0 cycle >= 1.0");
    assert!(!r[[0, 1]], "0.8 cycle < 1.0");
    let modes = epochs.modes(0).unwrap();
    assert_eq!(modes.dim(), (2, 1, 5));
    assert_eq!(
        modes[[1, 0, 0]],
        1010.0,
        "mode 1, first sample of the window (sample 10)"
    );
}

#[test]
fn from_parts_accepts_windows_touching_the_edges() {
    let source = EpochSource::Series(series(10));
    let window = |start: usize, end: usize| modewise_epochs::EpochWindow {
        label: "x".into(),
        event_onset_s: 0.0,
        start,
        end,
    };
    assert!(
        Epochs::from_parts(
            source.clone(),
            vec![window(8, 10), window(0, 2)],
            vec![],
            params(0.0)
        )
        .is_ok()
    );
    for bad in [window(9, 11), window(4, 5), window(5, 5)] {
        assert!(Epochs::from_parts(source.clone(), vec![bad], vec![], params(0.0)).is_err());
    }
}

#[test]
fn time_frequency_epochs_need_per_sample_times_and_respect_the_cone_of_influence() {
    use modewise_spectral::TimeFrequencyMap;

    // 100 samples at 1.25 Hz; two frequency rows: 0.5 Hz (2 s period) and 0.05 Hz (20 s period).
    let n = 100;
    let times: Vec<f64> = (0..n).map(|t| t as f64 / FS).collect();
    let power = ndarray::Array3::from_shape_fn((1, 2, n), |(_, f, t)| (f * 1000 + t) as f64);
    let mut reliable = ndarray::Array2::from_elem((2, n), true);
    for t in 0..10 {
        reliable[[0, t]] = false; // the fast row is unreliable in the first 10 samples
    }
    let frames = TimeFrequencyMap::new(
        power.clone(),
        vec![0.5, 0.05],
        times.clone(),
        vec!["a".into()],
    )
    .unwrap();
    let events = Events::new(vec![ev(4.0, 8.0, "early"), ev(40.0, 8.0, "late")]).unwrap();
    assert_eq!(
        Epochs::cut(
            EpochSource::TimeFrequency(frames.clone()),
            &events,
            params(0.0)
        )
        .err(),
        Some(EpochError::NotPerSample)
    );

    let map = frames
        .with_sampling_rate(SamplingRate::new(FS).unwrap())
        .unwrap()
        .with_reliable(reliable)
        .unwrap();
    let epochs = Epochs::cut(
        EpochSource::TimeFrequency(map.clone()),
        &events,
        params(0.0),
    )
    .unwrap();
    let r = epochs.resolvable().unwrap();
    assert!(
        !r[[0, 0]],
        "the early epoch (samples 5..15) overlaps the unreliable edge"
    );
    assert!(
        r[[1, 0]],
        "the late epoch is clear of the edge and holds 4 cycles of 0.5 Hz"
    );
    assert!(
        !r[[1, 1]],
        "0.05 Hz has a 20 s period; an 8 s epoch cannot hold it"
    );
    let w = &epochs.windows()[1];
    assert_eq!(
        epochs.time_frequency(1).unwrap(),
        map.power().slice(s![.., .., w.start..w.end])
    );
    assert!(epochs.time_frequency(1).is_some() && epochs.modes(1).is_none());
}

#[test]
fn instantaneous_epochs_slice_amplitude_and_frequency() {
    use modewise_decomposition::InstantaneousModes;

    let amplitude = Array3::from_shape_fn((2, 1, 100), |(k, _, t)| (k * 1000 + t) as f64);
    let frequency = Array3::from_elem((2, 1, 100), 0.2);
    let attrs = InstantaneousModes::new(
        amplitude,
        frequency,
        vec![0.02, 0.2],
        SamplingRate::new(FS).unwrap(),
        vec!["a".into()],
    )
    .unwrap();
    let events = Events::new(vec![ev(8.0, 18.0, "x")]).unwrap();
    let epochs = Epochs::cut(EpochSource::Instantaneous(attrs), &events, params(0.0)).unwrap();
    let (a, f) = epochs.instantaneous(0).unwrap();
    assert_eq!(
        (a.dim(), a[[1, 0, 0]], f[[0, 0, 0]]),
        ((2, 1, 23), 1010.0, 0.2)
    );
    let r = epochs.resolvable().unwrap();
    assert!(!r[[0, 0]] && r[[0, 1]]);
    assert!(epochs.time_frequency(0).is_none());
}
