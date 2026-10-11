#![allow(clippy::unwrap_used)]

use hdf5_pure::FileBuilder;
use modewise_hdf5::{Layout, read_series};
use modewise_signals::SamplingRate;
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    #[test]
    fn both_layouts_read_back_the_written_values(rows in 1usize..6, cols in 2usize..30, seed in any::<u32>(), transposed in any::<bool>()) {
        let values: Vec<f64> = (0..rows * cols).map(|i| ((seed as f64 + i as f64) * 0.11).cos()).collect();
        let path = std::env::temp_dir().join(format!("modewise-prop-h5-{}-{seed}.h5", std::process::id()));
        let mut b = FileBuilder::new();
        b.create_dataset("x").with_f64_data(&values).with_shape(&[rows as u64, cols as u64]);
        b.write(&path).unwrap();

        let fs = SamplingRate::new(1.0).unwrap();
        let (layout, channels, samples) = if transposed { (Layout::TimeByChannels, cols, rows) } else { (Layout::ChannelsByTime, rows, cols) };
        if samples < 2 {
            prop_assert!(read_series(&path, "x", layout, fs, None).is_err());
        } else {
            let series = read_series(&path, "x", layout, fs, None).unwrap();
            prop_assert_eq!((series.n_channels(), series.n_samples()), (channels, samples));
            for c in 0..channels {
                for t in 0..samples {
                    let stored = if transposed { values[t * cols + c] } else { values[c * cols + t] };
                    prop_assert_eq!(series.data()[[c, t]], stored);
                }
            }
        }
        std::fs::remove_file(&path).unwrap();
    }
}
