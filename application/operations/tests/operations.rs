#![allow(clippy::unwrap_used)]

use std::sync::Arc;

use modewise_operations::{MvmdOperation, ReconstructOperation, StftOperation, builtin_registry};
use modewise_pipeline::{Artifact, Inputs, Operation};
use modewise_signals::{MultichannelSeries, SamplingRate};
use ndarray::Array2;
use serde_json::json;

fn series_input(port: &str) -> Inputs {
    let data = Array2::from_shape_fn((1, 64), |(_, t)| (t as f64 * 0.3).sin());
    let s =
        MultichannelSeries::new(data, SamplingRate::new(8.0).unwrap(), vec!["a".into()]).unwrap();
    Inputs::from([(port.to_string(), Arc::new(Artifact::Series(s)))])
}

#[test]
fn registry_lists_builtin_operations() {
    let names: Vec<_> = builtin_registry().names().collect();
    assert_eq!(
        names,
        [
            "ceemdan",
            "cwt",
            "eemd",
            "emd",
            "epoch",
            "functional_connectivity",
            "glm",
            "hilbert",
            "hilbert_spectrum",
            "memd",
            "modwt_mra",
            "mvmd",
            "na_mvmd",
            "reconstruct",
            "region_average",
            "regress",
            "select",
            "spectrum",
            "stft",
            "vlmd",
            "vmd"
        ]
    );
}

#[test]
fn mvmd_resolve_fills_defaults_for_every_init() {
    for init in ["zero", "linear", "exponential"] {
        let resolved = MvmdOperation
            .resolve(json!({"num_modes": 2, "alpha": 100.0, "init": init}))
            .unwrap();
        assert_eq!(resolved["init"], init);
        assert_eq!(resolved["max_iterations"], 500);
        assert_eq!(resolved["tolerance"], 1e-3);
        assert_eq!(resolved["tau"], 1e-2);
        let out = MvmdOperation
            .run(&resolved, &series_input("signal"))
            .unwrap();
        assert!(matches!(out["modes"], Artifact::Decomposition(_)));
    }
    let defaulted = MvmdOperation
        .resolve(json!({"num_modes": 2, "alpha": 100.0}))
        .unwrap();
    assert_eq!(defaulted["init"], "zero");
}

#[test]
fn operations_reject_bad_params_and_wrong_inputs() {
    assert!(MvmdOperation.resolve(json!({"num_modes": 2})).is_err());
    assert!(
        MvmdOperation
            .resolve(json!({"num_modes": 2, "alpha": 1.0, "init": "random"}))
            .is_err()
    );
    assert!(
        StftOperation
            .resolve(json!({"window_len": 1, "hop": 1}))
            .is_err()
    );
    assert!(
        ReconstructOperation
            .resolve(json!({"modes": "all"}))
            .is_err()
    );

    let mvmd = MvmdOperation
        .resolve(json!({"num_modes": 2, "alpha": 100.0}))
        .unwrap();
    let modes = MvmdOperation
        .run(&mvmd, &series_input("signal"))
        .unwrap()
        .remove("modes")
        .unwrap();
    let as_signal = Inputs::from([("signal".to_string(), Arc::new(modes.clone()))]);
    assert!(
        MvmdOperation
            .run(&mvmd, &as_signal)
            .unwrap_err()
            .contains("must be a series")
    );

    let stft = StftOperation
        .resolve(json!({"window_len": 16, "hop": 8}))
        .unwrap();
    assert!(
        StftOperation
            .run(&stft, &as_signal)
            .unwrap_err()
            .contains("must be a series")
    );
    assert!(
        StftOperation
            .run(&stft, &Inputs::new())
            .unwrap_err()
            .contains("missing")
    );

    let reconstruct = ReconstructOperation.resolve(json!({"modes": [0]})).unwrap();
    assert!(
        ReconstructOperation
            .run(&reconstruct, &series_input("modes"))
            .unwrap_err()
            .contains("must be a decomposition")
    );
}

#[test]
fn vmd_and_na_mvmd_share_mvmd_parameters() {
    let registry = builtin_registry();
    let vmd = registry.get("vmd").unwrap();
    let na = registry.get("na_mvmd").unwrap();
    let base = json!({"num_modes": 2, "alpha": 100.0, "init": "uniform"});
    let resolved = na.resolve(base.clone()).unwrap();
    assert_eq!(
        (
            resolved["noise_channels"].clone(),
            resolved["noise_sd"].clone(),
            resolved["seed"].clone()
        ),
        (json!(1), json!(0.8), json!(0))
    );
    assert!(vmd.resolve(base.clone()).unwrap().get("noise_sd").is_none());
    let noisy = json!({"num_modes": 2, "alpha": 100.0, "noise_sd": 0.5});
    assert!(
        vmd.resolve(noisy.clone())
            .unwrap_err()
            .contains("only apply to na_mvmd")
    );
    assert!(
        registry
            .get("mvmd")
            .unwrap()
            .resolve(noisy.clone())
            .is_err()
    );
    assert_eq!(na.resolve(noisy).unwrap()["noise_sd"], json!(0.5));
    assert!(
        na.resolve(json!({"num_modes": 2, "alpha": 100.0, "noise_channels": 0}))
            .is_err()
    );

    for op in [vmd, na] {
        let params = op.resolve(base.clone()).unwrap();
        let out = op.run(&params, &series_input("signal")).unwrap();
        let Artifact::Decomposition(d) = &out["modes"] else {
            panic!()
        };
        assert_eq!(d.n_modes(), 2);
    }
}

#[test]
fn emd_and_memd_fill_their_stop_defaults_and_reject_foreign_settings() {
    let registry = builtin_registry();
    let emd = registry.get("emd").unwrap();
    let memd = registry.get("memd").unwrap();
    let r = emd.resolve(json!({})).unwrap();
    assert_eq!(
        (
            r["stop"].clone(),
            r["sd"].clone(),
            r["sd2"].clone(),
            r["tol"].clone()
        ),
        (json!("rilling"), json!(0.05), json!(0.5), json!(0.05))
    );
    assert!(r.get("num_directions").is_none());
    let r = memd.resolve(json!({})).unwrap();
    assert_eq!(
        (r["sd"].clone(), r["num_directions"].clone()),
        (json!(0.075), json!(64))
    );
    assert_eq!(
        emd.resolve(json!({"stop": "s_number"})).unwrap()["count"],
        json!(2)
    );
    assert_eq!(
        emd.resolve(json!({"stop": "fixed", "max_imfs": 3}))
            .unwrap()["siftings"],
        json!(10)
    );
    for bad in [
        json!({"stop": "fixed", "sd": 0.1}),
        json!({"stop": "rilling", "count": 2}),
        json!({"stop": "s_number", "siftings": 2}),
        json!({"num_directions": 32}),
        json!({"stop": "fixed", "siftings": 0}),
        json!({"max_imfs": 0}),
    ] {
        assert!(emd.resolve(bad.clone()).is_err(), "{bad}");
    }
    assert!(memd.resolve(json!({"num_directions": 4})).is_err());

    let params = emd.resolve(json!({"max_imfs": 2})).unwrap();
    let out = emd.run(&params, &series_input("signal")).unwrap();
    let Artifact::Decomposition(d) = &out["modes"] else {
        panic!()
    };
    assert!(d.n_modes() <= 3 && !d.aligned_across_channels());
    let three = {
        let data = Array2::from_shape_fn((3, 64), |(c, t)| {
            ((t as f64) * (0.3 + 0.1 * c as f64)).sin()
        });
        let s = MultichannelSeries::new(
            data,
            SamplingRate::new(8.0).unwrap(),
            vec!["a".into(), "b".into(), "c".into()],
        )
        .unwrap();
        Inputs::from([("signal".to_string(), Arc::new(Artifact::Series(s)))])
    };
    let out = memd.run(&memd.resolve(json!({})).unwrap(), &three).unwrap();
    let Artifact::Decomposition(d) = &out["modes"] else {
        panic!()
    };
    assert!(d.aligned_across_channels());
}

#[test]
fn eemd_and_ceemdan_fill_noise_defaults_and_keep_noise_settings_to_themselves() {
    let registry = builtin_registry();
    let eemd = registry.get("eemd").unwrap();
    let ceemdan = registry.get("ceemdan").unwrap();
    let r = eemd.resolve(json!({})).unwrap();
    assert_eq!(
        (
            r["stop"].clone(),
            r["siftings"].clone(),
            r["trials"].clone(),
            r["noise_sd"].clone(),
            r["seed"].clone()
        ),
        (json!("fixed"), json!(10), json!(100), json!(0.2), json!(0))
    );
    let r = ceemdan.resolve(json!({})).unwrap();
    assert_eq!(
        (r["stop"].clone(), r["sd"].clone(), r["epsilon"].clone()),
        (json!("rilling"), json!(0.05), json!(0.2))
    );
    assert!(r.get("noise_sd").is_none());
    for (op, bad) in [
        (&eemd, json!({"epsilon": 0.1})),
        (&ceemdan, json!({"noise_sd": 0.1})),
        (&eemd, json!({"num_directions": 8})),
        (&ceemdan, json!({"num_directions": 8})),
        (&eemd, json!({"trials": 0})),
        (&ceemdan, json!({"epsilon": -1.0})),
    ] {
        assert!(op.resolve(bad.clone()).is_err(), "{bad}");
    }
    let emd = registry.get("emd").unwrap();
    assert!(
        emd.resolve(json!({"seed": 1}))
            .unwrap_err()
            .contains("eemd and ceemdan")
    );
    assert!(
        ceemdan
            .resolve(json!({"noise_sd": 0.1}))
            .unwrap_err()
            .contains("`noise_sd` only applies to eemd")
    );
    for op in [eemd, ceemdan] {
        let params = op.resolve(json!({"trials": 4, "max_imfs": 2})).unwrap();
        let out = op.run(&params, &series_input("signal")).unwrap();
        let Artifact::Decomposition(d) = &out["modes"] else {
            panic!()
        };
        assert!(d.n_modes() <= 3 && !d.aligned_across_channels());
    }
}

#[test]
fn modwt_mra_defaults_to_sym4_and_checks_its_settings() {
    let op = builtin_registry().get("modwt_mra").unwrap();
    let r = op.resolve(json!({})).unwrap();
    assert_eq!(
        (r["wavelet"].clone(), r["boundary"].clone()),
        (json!("sym4"), json!("periodic"))
    );
    assert!(r.get("levels").is_none());
    assert!(op.resolve(json!({"wavelet": "mexh"})).is_err());
    assert!(op.resolve(json!({"levels": 0})).is_err());
    let params = op
        .resolve(json!({"wavelet": "db2", "levels": 3, "boundary": "reflection"}))
        .unwrap();
    let out = op.run(&params, &series_input("signal")).unwrap();
    let Artifact::Decomposition(d) = &out["modes"] else {
        panic!()
    };
    assert_eq!(d.n_modes(), 4);
    assert!(d.aligned_across_channels());
}

#[test]
fn spectrum_defaults_to_a_periodogram_and_takes_welch_segments_in_seconds() {
    let op = builtin_registry().get("spectrum").unwrap();
    let r = op.resolve(json!({})).unwrap();
    assert_eq!(
        (r["window"].clone(), r["detrend"].clone()),
        (json!("boxcar"), json!("constant"))
    );
    for bad in [
        json!({"segment_s": -1.0}),
        json!({"overlap_s": 1.0}),
        json!({"window": "hamming"}),
    ] {
        assert!(op.resolve(bad.clone()).is_err(), "{bad}");
    }
    let whole = op.run(&r, &series_input("signal")).unwrap();
    let Artifact::TimeFrequency(m) = &whole["power"] else {
        panic!()
    };
    assert_eq!(
        (m.frequencies_hz().len(), m.power().dim().2),
        (33, 1),
        "64 samples: 33 bins, one frame"
    );
    let welch = op
        .resolve(json!({"segment_s": 4.0, "overlap_s": 2.0, "window": "hann", "detrend": "linear"}))
        .unwrap();
    let out = op.run(&welch, &series_input("signal")).unwrap();
    let Artifact::TimeFrequency(m) = &out["power"] else {
        panic!()
    };
    assert_eq!(
        m.frequencies_hz().len(),
        17,
        "4 s at 8 Hz: 32-sample segments"
    );
    let none = op
        .resolve(json!({"segment_s": 100.0, "detrend": "none"}))
        .unwrap();
    assert!(
        op.run(&none, &series_input("signal")).is_err(),
        "a segment longer than the run"
    );
}
