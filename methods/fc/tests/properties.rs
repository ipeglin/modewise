#![allow(clippy::unwrap_used)]

use modewise_connectivity::Measure;
use modewise_fc::{Source, functional_connectivity};
use modewise_signals::{MultichannelSeries, SamplingRate};
use ndarray::Array2;
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn matrices_are_symmetric_bounded_with_unit_diagonal(
        data in prop::collection::vec(-10.0f64..10.0, 4 * 30),
        plv in any::<bool>(),
    ) {
        let x = Array2::from_shape_vec((4, 30), data).unwrap();
        let series = MultichannelSeries::new(x, SamplingRate::new(1.0).unwrap(), (0..4).map(|c| c.to_string()).collect()).unwrap();
        let measure = if plv { Measure::PhaseLockingValue } else { Measure::Pearson { fisher_z: false } };
        let c = functional_connectivity(Source::Series(&series), measure).unwrap();
        let m = c.matrix(0, 0);
        let low = if plv { 0.0 } else { -1.0 };
        for ((i, j), &v) in m.indexed_iter() {
            prop_assert!(v.is_nan() || (low..=1.0).contains(&v), "{v}");
            prop_assert!(v.to_bits() == m[[j, i]].to_bits() || (v - m[[j, i]]).abs() < 1e-12);
            if i == j {
                prop_assert!(v.is_nan() || (v - 1.0).abs() < 1e-12);
            }
        }
    }
}
