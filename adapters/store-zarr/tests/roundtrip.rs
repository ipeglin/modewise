#![allow(clippy::unwrap_used)]

use std::path::PathBuf;
use std::sync::Arc;

use modewise_decomposition::{Decomposition, InstantaneousModes};
use modewise_epochs::{EpochParams, EpochSource, Epochs, Event, Events, Window};
use modewise_pipeline::{Artifact, ArtifactKey, ArtifactStore};
use modewise_signals::{MultichannelSeries, SamplingRate};
use modewise_spectral::TimeFrequencyMap;
use modewise_store_zarr::ZarrStore;
use ndarray::{Array2, Array3};

fn temp_store(name: &str) -> (PathBuf, ZarrStore) {
    let root = std::env::temp_dir().join(format!("modewise-zarr-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let store = ZarrStore::open(&root).unwrap();
    (root, store)
}

fn key(text: &str) -> ArtifactKey {
    serde_json::from_value(serde_json::json!(text)).unwrap()
}

fn labels() -> Vec<String> {
    vec!["a".into(), "b".into()]
}

fn fs() -> SamplingRate {
    SamplingRate::new(1.25).unwrap()
}

fn artifacts() -> Vec<Artifact> {
    let series = MultichannelSeries::new(
        Array2::from_shape_fn((2, 5), |(c, t)| c as f64 * 10.0 + t as f64),
        fs(),
        labels(),
    )
    .unwrap();
    let modes = Array3::from_shape_fn((3, 2, 5), |(k, c, t)| (k * 100 + c * 10 + t) as f64 * 0.5);
    let decomposition =
        Decomposition::new(modes, vec![0.01, 0.1, 0.3], fs(), labels(), true).unwrap();
    let power = Array3::from_shape_fn((2, 4, 3), |(c, f, t)| (c + f * t) as f64);
    let map = TimeFrequencyMap::new(
        power,
        vec![0.0, 0.1, 0.2, 0.3],
        vec![1.0, 2.0, 3.0],
        labels(),
    )
    .unwrap();
    vec![
        Artifact::Series(series),
        Artifact::Decomposition(decomposition),
        Artifact::TimeFrequency(map),
    ]
}

#[test]
fn every_artifact_kind_round_trips_and_survives_reopening() {
    let (root, store) = temp_store("roundtrip");
    let keys = ["aa11", "bb22", "cc33"].map(key);
    for (k, artifact) in keys.iter().zip(artifacts()) {
        assert!(!store.contains(k).unwrap());
        store.put(k, Arc::new(artifact)).unwrap();
        assert!(store.contains(k).unwrap());
    }
    drop(store);

    let reopened = ZarrStore::open(&root).unwrap();
    for (k, expected) in keys.iter().zip(artifacts()) {
        let back = reopened.get(k).unwrap().unwrap();
        assert_eq!(back.as_ref(), &expected);
    }
    assert!(reopened.get(&key("dd44")).unwrap().is_none());
}

#[test]
fn new_store_writes_root_metadata_and_reopening_keeps_it() {
    let (root, store) = temp_store("root");
    let meta = root.join("zarr.json");
    assert!(meta.exists());
    let first = std::fs::metadata(&meta).unwrap().modified().unwrap();
    drop(store);
    std::thread::sleep(std::time::Duration::from_millis(20));
    ZarrStore::open(&root).unwrap();
    assert_eq!(
        std::fs::metadata(&meta).unwrap().modified().unwrap(),
        first,
        "reopening must not rewrite it"
    );
}

#[test]
fn half_written_artifact_is_not_stored() {
    let (root, store) = temp_store("partial");
    let k = key("ee55");
    store.put(&k, Arc::new(artifacts().remove(0))).unwrap();
    std::fs::remove_file(root.join("ee/ee55/zarr.json")).unwrap();
    assert!(!store.contains(&k).unwrap());
    assert!(store.get(&k).unwrap().is_none());
}

#[test]
fn reports_unusable_folders_and_foreign_groups() {
    let root = std::env::temp_dir().join(format!("modewise-zarr-file-{}", std::process::id()));
    std::fs::write(&root, b"a file, not a folder").unwrap();
    assert!(ZarrStore::open(&root).is_err());

    let (root, store) = temp_store("foreign");
    let k = key("ff66");
    store.put(&k, Arc::new(artifacts().remove(0))).unwrap();
    let meta = root.join("ff/ff66/zarr.json");
    let text = std::fs::read_to_string(&meta)
        .unwrap()
        .replace("\"series\"", "\"mystery\"");
    std::fs::write(&meta, text).unwrap();
    let err = store.get(&k).err().unwrap();
    assert!(err.0.contains("unknown artifact kind"), "{}", err.0);

    std::fs::write(
        &meta,
        r#"{"zarr_format": 3, "node_type": "group", "attributes": {}}"#,
    )
    .unwrap();
    assert!(
        store
            .get(&k)
            .err()
            .unwrap()
            .0
            .contains("not a modewise artifact")
    );
}

fn events() -> Events {
    Events::new(vec![
        Event {
            onset_s: 0.8,
            duration_s: 1.6,
            label: "face".into(),
        },
        Event {
            onset_s: 2.4,
            duration_s: 0.0,
            label: "shape".into(),
        },
        Event {
            onset_s: 9.0,
            duration_s: 3.0,
            label: "late".into(),
        },
    ])
    .unwrap()
}

#[test]
fn events_and_epochs_round_trip() {
    let (root, store) = temp_store("epochs");
    let mut all = artifacts();
    let Artifact::Series(series) = all.remove(0) else {
        unreachable!()
    };
    let Artifact::Decomposition(decomposition) = all.remove(0) else {
        unreachable!()
    };
    let fixed = EpochParams {
        lag_s: 0.0,
        window: Window::Fixed {
            tmin_s: 0.0,
            tmax_s: 1.6,
        },
        conditions: Some(vec!["face".into(), "shape".into()]),
        min_cycles: 1.0,
    };
    let blocks = EpochParams {
        lag_s: 0.8,
        window: Window::Duration,
        conditions: None,
        min_cycles: 2.0,
    };
    let cases = [
        ("e1", Artifact::Events(events())),
        (
            "e2",
            Artifact::Epochs(Epochs::cut(EpochSource::Series(series), &events(), fixed).unwrap()),
        ),
        (
            "e3",
            Artifact::Epochs(
                Epochs::cut(EpochSource::Decomposition(decomposition), &events(), blocks).unwrap(),
            ),
        ),
    ];
    for (name, artifact) in &cases {
        store.put(&key(name), Arc::new(artifact.clone())).unwrap();
    }
    let reopened = ZarrStore::open(&root).unwrap();
    for (name, artifact) in &cases {
        assert_eq!(
            reopened.get(&key(name)).unwrap().as_deref(),
            Some(artifact),
            "{name}"
        );
    }
    let Artifact::Epochs(e3) = &cases[2].1 else {
        unreachable!()
    };
    assert!(
        !e3.dropped().is_empty(),
        "the test should cover dropped events too"
    );
}

#[test]
fn corrupted_epoch_metadata_is_reported() {
    let (root, store) = temp_store("bad-epochs");
    let Artifact::Series(series) = artifacts().remove(0) else {
        unreachable!()
    };
    let params = EpochParams {
        lag_s: 0.0,
        window: Window::Duration,
        conditions: None,
        min_cycles: 1.0,
    };
    let epochs = Epochs::cut(EpochSource::Series(series), &events(), params).unwrap();
    store
        .put(&key("ab12"), Arc::new(Artifact::Epochs(epochs)))
        .unwrap();
    let meta = root.join("ab/ab12/zarr.json");
    let original = std::fs::read_to_string(&meta).unwrap();

    std::fs::write(&meta, original.replace("\"duration\"", "\"sideways\"")).unwrap();
    assert!(
        store
            .get(&key("ab12"))
            .err()
            .unwrap()
            .0
            .contains("unknown epoch window")
    );

    std::fs::write(&meta, &original).unwrap();
    store
        .put(&key("cd34"), Arc::new(Artifact::Events(events())))
        .unwrap();
    std::fs::remove_dir_all(root.join("ab/ab12/source")).unwrap();
    std::fs::create_dir_all(root.join("ab/ab12/source")).unwrap();
    std::fs::copy(
        root.join("cd/cd34/zarr.json"),
        root.join("ab/ab12/source/zarr.json"),
    )
    .unwrap();
    assert!(
        store
            .get(&key("ab12"))
            .err()
            .unwrap()
            .0
            .contains("cannot hold a events source")
    );
}

#[test]
fn instantaneous_attributes_and_their_epochs_round_trip() {
    let (root, store) = temp_store("instantaneous");
    let amplitude = Array3::from_shape_fn((2, 2, 5), |(k, c, t)| (k * 10 + c + t) as f64);
    let frequency = Array3::from_shape_fn((2, 2, 5), |(k, _, t)| {
        0.1 + k as f64 * 0.2 + t as f64 * 1e-3
    });
    let attrs =
        InstantaneousModes::new(amplitude, frequency, vec![0.1, 0.3], fs(), labels()).unwrap();
    let params = EpochParams {
        lag_s: 0.0,
        window: Window::Duration,
        conditions: None,
        min_cycles: 1.0,
    };
    let epochs = Epochs::cut(EpochSource::Instantaneous(attrs.clone()), &events(), params).unwrap();
    let cases = [
        ("i1", Artifact::Instantaneous(attrs)),
        ("i2", Artifact::Epochs(epochs)),
    ];
    for (name, artifact) in &cases {
        store.put(&key(name), Arc::new(artifact.clone())).unwrap();
    }
    let reopened = ZarrStore::open(&root).unwrap();
    for (name, artifact) in &cases {
        assert_eq!(
            reopened.get(&key(name)).unwrap().as_deref(),
            Some(artifact),
            "{name}"
        );
    }
}

#[test]
fn time_frequency_mask_sampling_rate_and_epochs_round_trip() {
    let (root, store) = temp_store("tf-mask");
    let power = Array3::from_shape_fn((2, 4, 3), |(c, f, t)| (c + f * t) as f64);
    let times = vec![0.0, 0.8, 1.6];
    let mut mask = ndarray::Array2::from_elem((4, 3), true);
    mask[[3, 0]] = false;
    let map = TimeFrequencyMap::new(power, vec![0.1, 0.2, 0.3, 0.4], times, labels())
        .unwrap()
        .with_sampling_rate(fs())
        .unwrap()
        .with_reliable(mask)
        .unwrap();
    let params = EpochParams {
        lag_s: 0.0,
        window: Window::Duration,
        conditions: None,
        min_cycles: 1.0,
    };
    let one = Events::new(vec![Event {
        onset_s: 0.0,
        duration_s: 1.6,
        label: "x".into(),
    }])
    .unwrap();
    let epochs = Epochs::cut(EpochSource::TimeFrequency(map.clone()), &one, params).unwrap();
    let cases = [
        ("t1", Artifact::TimeFrequency(map)),
        ("t2", Artifact::Epochs(epochs)),
    ];
    for (name, artifact) in &cases {
        store.put(&key(name), Arc::new(artifact.clone())).unwrap();
    }
    let reopened = ZarrStore::open(&root).unwrap();
    for (name, artifact) in &cases {
        assert_eq!(
            reopened.get(&key(name)).unwrap().as_deref(),
            Some(artifact),
            "{name}"
        );
    }
}

#[test]
fn loadings_round_trip() {
    let (root, store) = temp_store("loadings");
    let loadings = modewise_decomposition::Loadings::new(
        ndarray::Array2::from_shape_fn((2, 2), |(l, c)| l as f64 - 0.5 * c as f64),
        vec!["latent_0".into(), "latent_1".into()],
        labels(),
    )
    .unwrap();
    let artifact = Artifact::Loadings(loadings);
    store.put(&key("l1"), Arc::new(artifact.clone())).unwrap();
    let reopened = ZarrStore::open(&root).unwrap();
    assert_eq!(
        reopened.get(&key("l1")).unwrap().as_deref(),
        Some(&artifact)
    );
}

#[test]
fn connectivity_round_trips_with_and_without_windows() {
    use modewise_connectivity::{FunctionalConnectivity, Measure, WindowInfo};

    let (root, store) = temp_store("functional_connectivity");
    let values =
        ndarray::Array4::from_shape_fn((2, 1, 2, 2), |(w, _, i, j)| (w + i + j) as f64 * 0.1);
    let whole = FunctionalConnectivity::new(
        values.slice(ndarray::s![0..1, .., .., ..]).to_owned(),
        Measure::Pearson { fisher_z: true },
        labels(),
    )
    .unwrap()
    .with_centre_frequencies(vec![0.05])
    .unwrap();
    let windows = (0..2)
        .map(|i| WindowInfo {
            label: format!("w{i}"),
            event_onset_s: 3.5 * i as f64,
        })
        .collect();
    let epochs = FunctionalConnectivity::new(values, Measure::PhaseLockingValue, labels())
        .unwrap()
        .with_windows(windows, Some(ndarray::array![[true], [false]]))
        .unwrap();
    let cases = [
        ("c1", Artifact::FunctionalConnectivity(whole)),
        ("c2", Artifact::FunctionalConnectivity(epochs)),
    ];
    for (name, artifact) in &cases {
        store.put(&key(name), Arc::new(artifact.clone())).unwrap();
    }
    let reopened = ZarrStore::open(&root).unwrap();
    for (name, artifact) in &cases {
        assert_eq!(
            reopened.get(&key(name)).unwrap().as_deref(),
            Some(artifact),
            "{name}"
        );
    }
}

#[test]
fn channel_tables_round_trip_with_every_description() {
    use modewise_brain::{ChannelInfo, ChannelTable, Hemisphere, Region};

    let (root, store) = temp_store("channels");
    let table = ChannelTable::new(vec![
        ChannelInfo::new("17networks_LH_DefaultB_PFCm_1")
            .with_region(Region::new("PFCm"))
            .with_hemisphere(Hemisphere::Left)
            .with_tag("yeo17", "DefaultB")
            .with_tag("lobe", "frontal"),
        ChannelInfo::new("vermis"),
    ])
    .unwrap();
    let artifact = Artifact::Channels(table);
    store.put(&key("ch"), Arc::new(artifact.clone())).unwrap();
    let reopened = ZarrStore::open(&root).unwrap();
    assert_eq!(
        reopened.get(&key("ch")).unwrap().as_deref(),
        Some(&artifact)
    );
}

#[test]
fn per_channel_centre_frequencies_round_trip_with_missing_modes() {
    let (root, store) = temp_store("per-channel");
    let per_channel = ndarray::array![[0.1, f64::NAN], [f64::NAN, f64::NAN]];
    let d =
        Decomposition::per_channel(Array3::ones((2, 2, 5)), per_channel.clone(), fs(), labels())
            .unwrap();
    let k = key("per-channel");
    store.put(&k, Arc::new(Artifact::Decomposition(d))).unwrap();
    let Artifact::Decomposition(back) = &*store.get(&k).unwrap().unwrap() else {
        panic!()
    };
    let stored = back.channel_centre_frequencies_hz().unwrap();
    assert!(
        stored
            .iter()
            .zip(&per_channel)
            .all(|(a, b)| a.to_bits() == b.to_bits())
    );
    assert_eq!(back.centre_frequencies_hz()[0], 0.1);
    assert!(
        back.centre_frequencies_hz()[1].is_nan(),
        "a mode no channel has stays NaN"
    );
    assert!(!back.aligned_across_channels());
    let _ = std::fs::remove_dir_all(root);
}
