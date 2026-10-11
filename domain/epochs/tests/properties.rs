#![allow(clippy::unwrap_used)]

use modewise_epochs::{EpochParams, EpochSource, Epochs, Event, Events, Window};
use modewise_signals::{MultichannelSeries, SamplingRate};
use ndarray::Array2;
use proptest::prelude::*;

proptest! {
    /// Every kept window lies inside the recording, its samples' times lie in
    /// [onset + lag, onset + duration + lag), and every event is kept or reported.
    #[test]
    fn windows_respect_their_time_interval(
        fs in 0.5f64..500.0,
        n in 10usize..400,
        lag in -5.0f64..10.0,
        raw in prop::collection::vec((-10.0f64..200.0, 0.0f64..30.0), 1..12),
    ) {
        let series = MultichannelSeries::new(Array2::zeros((1, n)), SamplingRate::new(fs).unwrap(), vec!["a".into()]).unwrap();
        let events = Events::new(raw.iter().map(|&(onset_s, duration_s)| Event { onset_s, duration_s, label: "e".into() }).collect()).unwrap();
        let params = EpochParams { lag_s: lag, window: Window::Duration, conditions: None, min_cycles: 1.0 };
        let epochs = Epochs::cut(EpochSource::Series(series), &events, params).unwrap();

        prop_assert_eq!(epochs.windows().len() + epochs.dropped().len(), events.len());
        for w in epochs.windows() {
            let event = events.as_slice().iter().find(|e| e.onset_s == w.event_onset_s).unwrap();
            let (t0, t1) = (event.onset_s + lag, event.onset_s + event.duration_s + lag);
            prop_assert!(w.end <= n && w.end - w.start >= 2);
            prop_assert!(w.start as f64 / fs >= t0 - 1e-6);
            prop_assert!((w.end - 1) as f64 / fs < t1 + 1e-6);
            if w.start > 0 {
                prop_assert!((w.start - 1) as f64 / fs < t0 + 1e-6, "the window starts at the first sample in range");
            }
        }
    }
}
