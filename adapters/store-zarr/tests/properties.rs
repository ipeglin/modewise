#![allow(clippy::unwrap_used)]

use std::sync::Arc;

use modewise_decomposition::Decomposition;
use modewise_pipeline::{Artifact, ArtifactKey, ArtifactStore};
use modewise_signals::{MultichannelSeries, SamplingRate};
use modewise_store_zarr::ZarrStore;
use ndarray::{Array2, Array3};
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    #[test]
    fn random_shapes_round_trip_exactly(
        modes in 1usize..5, channels in 1usize..5, samples in 2usize..40,
        hz in 0.1f64..1000.0, seed in any::<u32>(),
    ) {
        let value = |i: usize| ((seed as f64 + i as f64) * 0.37).sin() * 1e3;
        let labels: Vec<String> = (0..channels).map(|c| format!("ch{c}")).collect();
        let fs = SamplingRate::new(hz).unwrap();
        let series = MultichannelSeries::new(Array2::from_shape_fn((channels, samples), |(c, t)| value(c * samples + t)), fs, labels.clone()).unwrap();
        let cube = Array3::from_shape_fn((modes, channels, samples), |(k, c, t)| value(k * 1000 + c * samples + t));
        let freqs: Vec<f64> = (0..modes).map(|k| k as f64 * hz / 10.0).collect();
        let decomposition = Decomposition::new(cube, freqs, fs, labels, seed % 2 == 0).unwrap();

        let root = std::env::temp_dir().join(format!("modewise-prop-zarr-{}-{seed}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let store = ZarrStore::open(&root).unwrap();
        for (name, artifact) in [("a1", Artifact::Series(series)), ("b2", Artifact::Decomposition(decomposition))] {
            let key: ArtifactKey = serde_json::from_value(serde_json::json!(name)).unwrap();
            store.put(&key, Arc::new(artifact.clone())).unwrap();
            let back = store.get(&key).unwrap();
            prop_assert_eq!(back.as_deref(), Some(&artifact));
        }
        std::fs::remove_dir_all(&root).unwrap();
    }
}
