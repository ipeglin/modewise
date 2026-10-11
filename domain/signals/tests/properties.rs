#![allow(clippy::unwrap_used)]

use modewise_signals::{MultichannelSeries, SamplingRate};
use ndarray::Array2;
use proptest::prelude::*;

proptest! {
    #[test]
    fn any_finite_data_with_unique_labels_is_accepted(
        channels in 1usize..6,
        samples in 2usize..64,
        hz in 0.01f64..10_000.0,
        seed in any::<u32>(),
    ) {
        let data = Array2::from_shape_fn((channels, samples), |(c, t)| ((seed as f64 + (c * samples + t) as f64) * 1e-3).sin());
        let labels: Vec<String> = (0..channels).map(|i| format!("c{i}")).collect();
        let series = MultichannelSeries::new(data.clone(), SamplingRate::new(hz).unwrap(), labels.clone()).unwrap();
        prop_assert_eq!(series.data(), data.view());
        prop_assert_eq!(series.labels(), labels.as_slice());
        prop_assert!((series.duration_s() - samples as f64 / hz).abs() <= 1e-9 * series.duration_s());
    }

    #[test]
    fn any_non_finite_value_is_rejected(
        channels in 1usize..4,
        samples in 2usize..16,
        at in any::<prop::sample::Index>(),
        bad in prop_oneof![Just(f64::NAN), Just(f64::INFINITY), Just(f64::NEG_INFINITY)],
    ) {
        let mut data = Array2::<f64>::zeros((channels, samples));
        let i = at.index(channels * samples);
        data[[i / samples, i % samples]] = bad;
        let labels = (0..channels).map(|i| format!("c{i}")).collect();
        prop_assert!(MultichannelSeries::new(data, SamplingRate::new(1.0).unwrap(), labels).is_err());
    }
}
