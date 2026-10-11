#![allow(clippy::unwrap_used)]

use modewise_cohort::{CohortError, Features, RecordingOutput, entities, feature_table};
use modewise_connectivity::{FunctionalConnectivity, Measure, WindowInfo};
use modewise_decomposition::Decomposition;
use modewise_epochs::{EpochParams, EpochSource, Epochs, Event, Events, Window};
use modewise_pipeline::Artifact;
use modewise_signals::SamplingRate;
use ndarray::{Array2, Array3, Array4};

const EDGES: Features = Features::Edges {
    resolvable_only: true,
};

fn labels() -> Vec<String> {
    vec!["a".into(), "b".into(), "c".into()]
}

/// Three windows (face, shape, face), two components; component 1 unresolvable in window 2.
fn connectivity(scale: f64) -> Artifact {
    let values = Array4::from_shape_fn((3, 2, 3, 3), |(w, k, i, j)| {
        if i == j {
            1.0
        } else {
            scale * 0.1 * (1 + w + k + i + j) as f64
        }
    });
    let windows = ["face", "shape", "face"]
        .iter()
        .enumerate()
        .map(|(i, l)| WindowInfo {
            label: l.to_string(),
            event_onset_s: i as f64,
        })
        .collect();
    let resolvable = ndarray::array![[true, true], [true, true], [true, false]];
    let c = FunctionalConnectivity::new(values, Measure::Pearson { fisher_z: false }, labels())
        .unwrap()
        .with_windows(windows, Some(resolvable))
        .unwrap();
    Artifact::FunctionalConnectivity(c)
}

#[test]
fn edges_average_conditions_as_fisher_z_and_skip_unresolvable_windows() {
    let fc = connectivity(1.0);
    let outputs = [RecordingOutput {
        recording: "subject-01_task-hammer",
        artifact: &fc,
    }];
    let t = feature_table(&outputs, &EDGES, "subject").unwrap();
    assert_eq!(t.feature_names()[..3], ["m0:a~b", "m0:a~c", "m0:b~c"]);
    assert_eq!(t.feature_names().len(), 6);
    let conditions: Vec<_> = t
        .observations()
        .iter()
        .map(|o| o.condition.clone().unwrap())
        .collect();
    assert_eq!(conditions, ["face", "shape"]);
    assert_eq!(t.observations()[0].subject.as_deref(), Some("01"));
    // m0:a~b in face: windows 0 and 2, r = 0.2 and 0.4 (1 + w + k + i + j with i=0, j=1).
    let face = t.feature("m0:a~b").unwrap()[0];
    assert!((face - (0.2f64.atanh() + 0.4f64.atanh()) / 2.0).abs() < 1e-12);
    // m1 in face: window 2 is unresolvable, so only window 0 (r = 0.3) counts.
    assert!((t.feature("m1:a~b").unwrap()[0] - 0.3f64.atanh()).abs() < 1e-12);

    let all = feature_table(
        &outputs,
        &Features::Edges {
            resolvable_only: false,
        },
        "subject",
    )
    .unwrap();
    assert!(
        (all.feature("m1:a~b").unwrap()[0] - (0.3f64.atanh() + 0.5f64.atanh()) / 2.0).abs() < 1e-12
    );
}

#[test]
fn recordings_stack_and_must_agree() {
    let (one, two) = (connectivity(1.0), connectivity(0.5));
    let outputs = [
        RecordingOutput {
            recording: "subject-01_run-1",
            artifact: &one,
        },
        RecordingOutput {
            recording: "subject-02_run-1",
            artifact: &two,
        },
    ];
    let t = feature_table(&outputs, &EDGES, "subject").unwrap();
    assert_eq!(t.values().nrows(), 4);
    assert_eq!(t.observations()[2].subject.as_deref(), Some("02"));

    let plv = FunctionalConnectivity::new(
        Array4::from_elem((1, 1, 2, 2), 0.5),
        Measure::PhaseLockingValue,
        vec!["a".into(), "b".into()],
    )
    .unwrap();
    let small = Artifact::FunctionalConnectivity(plv);
    let mixed = [
        RecordingOutput {
            recording: "subject-01",
            artifact: &one,
        },
        RecordingOutput {
            recording: "subject-02",
            artifact: &small,
        },
    ];
    assert!(matches!(
        feature_table(&mixed, &EDGES, "subject"),
        Err(CohortError::Mismatch { .. })
    ));
    let only_plv = feature_table(&mixed[1..], &EDGES, "subject").unwrap();
    assert_eq!(
        (
            only_plv.feature("a~b").is_none(),
            only_plv.feature("m0:a~b").unwrap()[0]
        ),
        (true, 0.5),
        "PLV is not transformed"
    );
    assert_eq!(
        feature_table(&[], &EDGES, "subject").unwrap_err(),
        CohortError::Empty
    );
    let wrong = [RecordingOutput {
        recording: "subject-01",
        artifact: &small,
    }];
    assert!(matches!(
        feature_table(
            &wrong,
            &Features::ModePower {
                resolvable_only: true
            },
            "subject"
        ),
        Err(CohortError::Recording { .. })
    ));
}

fn modes() -> Decomposition {
    let data = Array3::from_shape_fn((2, 3, 40), |(k, c, t)| {
        ((k + 1) * (c + 1)) as f64 * if t % 2 == 0 { 1.0 } else { -1.0 }
    });
    Decomposition::new(
        data,
        vec![0.02, 0.2],
        SamplingRate::new(1.0).unwrap(),
        labels(),
        true,
    )
    .unwrap()
}

#[test]
fn mode_power_for_whole_runs_and_conditions() {
    let whole = Artifact::Decomposition(modes());
    let t = feature_table(
        &[RecordingOutput {
            recording: "subject-01",
            artifact: &whole,
        }],
        &Features::ModePower {
            resolvable_only: true,
        },
        "subject",
    )
    .unwrap();
    assert_eq!(
        t.feature_names()[..4],
        ["m0:centre_hz", "m0:a", "m0:b", "m0:c"]
    );
    assert_eq!(t.feature("m1:c").unwrap()[0], 36.0, "variance of ±6");
    assert_eq!(t.feature("m1:centre_hz").unwrap()[0], 0.2);

    let events = Events::new(vec![
        Event {
            onset_s: 0.0,
            duration_s: 10.0,
            label: "face".into(),
        },
        Event {
            onset_s: 20.0,
            duration_s: 10.0,
            label: "shape".into(),
        },
    ])
    .unwrap();
    let params = EpochParams {
        lag_s: 0.0,
        window: Window::Duration,
        conditions: None,
        min_cycles: 1.0,
    };
    let epochs = Artifact::Epochs(
        Epochs::cut(EpochSource::Decomposition(modes()), &events, params).unwrap(),
    );
    let per_condition = feature_table(
        &[RecordingOutput {
            recording: "subject-01",
            artifact: &epochs,
        }],
        &Features::ModePower {
            resolvable_only: true,
        },
        "subject",
    )
    .unwrap();
    assert_eq!(per_condition.values().nrows(), 2);
    assert!(
        per_condition.feature("m0:a").unwrap()[0].is_nan(),
        "10 s holds 0.2 cycles of 0.02 Hz"
    );
    assert_eq!(per_condition.feature("m1:a").unwrap()[1], 4.0);
    let kept = feature_table(
        &[RecordingOutput {
            recording: "subject-01",
            artifact: &epochs,
        }],
        &Features::ModePower {
            resolvable_only: false,
        },
        "subject",
    )
    .unwrap();
    assert_eq!(kept.feature("m0:a").unwrap()[0], 1.0);

    let series_epochs = Epochs::cut(
        EpochSource::Series(
            modewise_signals::MultichannelSeries::new(
                Array2::zeros((3, 40)),
                SamplingRate::new(1.0).unwrap(),
                labels(),
            )
            .unwrap(),
        ),
        &events,
        EpochParams {
            lag_s: 0.0,
            window: Window::Duration,
            conditions: None,
            min_cycles: 1.0,
        },
    )
    .unwrap();
    let bad = Artifact::Epochs(series_epochs);
    assert!(
        feature_table(
            &[RecordingOutput {
                recording: "subject-01",
                artifact: &bad
            }],
            &Features::ModePower {
                resolvable_only: true
            },
            "subject"
        )
        .is_err()
    );
}

#[test]
fn recording_ids_give_entities() {
    let e = entities("subject-INVAG388HJL_task-hammerAP_run-01");
    assert_eq!(
        (e["subject"].as_str(), e["task"].as_str(), e["run"].as_str()),
        ("INVAG388HJL", "hammerAP", "01")
    );
    assert!(entities("plain").is_empty());
}

fn band(name: &str, low_hz: f64, high_hz: f64) -> modewise_cohort::Band {
    modewise_cohort::Band {
        name: name.into(),
        low_hz,
        high_hz,
    }
}

/// Power at 0, 0.01, …, 0.05 Hz with amplitude 1 + bin index in channel a and 2 in channel b,
/// two frames.
fn spectrum() -> Artifact {
    let power = Array3::from_shape_fn((2, 6, 2), |(c, k, _)| {
        if c == 0 {
            ((1 + k) as f64).powi(2)
        } else {
            4.0
        }
    });
    let freqs = (0..6).map(|k| k as f64 * 0.01).collect();
    let map = modewise_spectral::TimeFrequencyMap::new(
        power,
        freqs,
        vec![0.0, 1.0],
        vec!["a".into(), "b".into()],
    )
    .unwrap();
    Artifact::TimeFrequency(map)
}

#[test]
fn band_amplitude_is_alff_per_band_with_bands_as_conditions() {
    let output = spectrum();
    let outputs = [RecordingOutput {
        recording: "subject-01_task-rest",
        artifact: &output,
    }];
    let bands = vec![band("low", 0.01, 0.03), band("high", 0.03, 0.06)];
    let alff = feature_table(
        &outputs,
        &Features::BandAmplitude {
            bands: bands.clone(),
            relative: false,
        },
        "subject",
    )
    .unwrap();
    assert_eq!(alff.feature_names(), ["a", "b"]);
    let conditions: Vec<_> = alff
        .observations()
        .iter()
        .map(|o| o.condition.clone().unwrap())
        .collect();
    assert_eq!(conditions, ["low", "high"]);
    // Channel a: bins 1 and 2 have amplitudes 2 and 3; bins 3–5 have 4, 5 and 6.
    assert_eq!(alff.values().row(0).to_vec(), [2.5, 2.0]);
    assert_eq!(alff.values().row(1).to_vec(), [5.0, 2.0]);
    let falff = feature_table(
        &outputs,
        &Features::BandAmplitude {
            bands,
            relative: true,
        },
        "subject",
    )
    .unwrap();
    // Above 0 Hz, channel a sums to 2 + 3 + 4 + 5 + 6 = 20 per frame; channel b to 10.
    assert_eq!(falff.values().row(0).to_vec(), [5.0 / 20.0, 4.0 / 10.0]);
}

#[test]
fn bad_bands_and_outputs_are_refused() {
    let output = spectrum();
    let outputs = [RecordingOutput {
        recording: "subject-01",
        artifact: &output,
    }];
    let refused = |bands: Vec<modewise_cohort::Band>| {
        feature_table(
            &outputs,
            &Features::BandAmplitude {
                bands,
                relative: false,
            },
            "subject",
        )
        .unwrap_err()
        .to_string()
    };
    for bands in [
        vec![band("x", 0.03, 0.01)],
        vec![band("x", 0.02, 0.02)],
        vec![band("x", f64::NAN, 0.01)],
        vec![band("x", -0.01, 0.01)],
        vec![band("x", 0.0, f64::INFINITY)],
    ] {
        assert!(refused(bands).contains("needs 0 ≤ low < high"));
    }
    assert!(refused(vec![band("x", 0.01, 0.02), band("x", 0.02, 0.03)]).contains("twice"));
    assert!(refused(vec![band("gap", 0.011, 0.019)]).contains("no frequency bins"));
    let fc = connectivity(1.0);
    let wrong = [RecordingOutput {
        recording: "subject-01",
        artifact: &fc,
    }];
    let err = feature_table(
        &wrong,
        &Features::BandAmplitude {
            bands: vec![band("x", 0.0, 0.1)],
            relative: false,
        },
        "subject",
    )
    .unwrap_err();
    assert!(err.to_string().contains("spectrum"));
}

#[test]
fn mean_averages_each_conditions_epochs_of_a_series() {
    // Channel a is 1 in the face blocks and 3 in the shape block, b is the sample index.
    let data = Array2::from_shape_fn((3, 60), |(c, t)| match c {
        0 if (20..30).contains(&t) => 3.0,
        0 => 1.0,
        1 => t as f64,
        _ => 0.0,
    });
    let series =
        modewise_signals::MultichannelSeries::new(data, SamplingRate::new(1.0).unwrap(), labels())
            .unwrap();
    let event = |onset_s: f64, duration_s: f64, label: &str| Event {
        onset_s,
        duration_s,
        label: label.into(),
    };
    let events = Events::new(vec![
        event(0.0, 10.0, "face"),
        event(20.0, 10.0, "shape"),
        event(40.0, 4.0, "face"),
    ])
    .unwrap();
    let params = EpochParams {
        lag_s: 0.0,
        window: Window::Duration,
        conditions: None,
        min_cycles: 1.0,
    };
    let epochs =
        Artifact::Epochs(Epochs::cut(EpochSource::Series(series), &events, params).unwrap());
    let outputs = [RecordingOutput {
        recording: "subject-01",
        artifact: &epochs,
    }];
    let t = feature_table(&outputs, &Features::Mean, "subject").unwrap();
    assert_eq!(t.feature_names(), ["a", "b", "c"]);
    let conditions: Vec<_> = t
        .observations()
        .iter()
        .map(|o| o.condition.clone().unwrap())
        .collect();
    assert_eq!(conditions, ["face", "shape"]);
    assert_eq!(
        t.values().row(0).to_vec(),
        [1.0, (4.5 + 41.5) / 2.0, 0.0],
        "each face block counts once"
    );
    assert_eq!(t.values().row(1).to_vec(), [3.0, 24.5, 0.0]);

    let fc = connectivity(1.0);
    let wrong = [RecordingOutput {
        recording: "subject-01",
        artifact: &fc,
    }];
    assert!(
        feature_table(&wrong, &Features::Mean, "subject")
            .unwrap_err()
            .to_string()
            .contains("epochs of a series")
    );
    let mode_epochs = Artifact::Epochs(
        Epochs::cut(
            EpochSource::Decomposition(modes()),
            &events,
            EpochParams {
                lag_s: 0.0,
                window: Window::Duration,
                conditions: None,
                min_cycles: 1.0,
            },
        )
        .unwrap(),
    );
    let modal = [RecordingOutput {
        recording: "subject-01",
        artifact: &mode_epochs,
    }];
    assert!(
        feature_table(&modal, &Features::Mean, "subject")
            .unwrap_err()
            .to_string()
            .contains("mode_power")
    );
}
