//! Accuracy against the reference VLMD (Dmocrito/vlmd) with an exactly solved LASSO step,
//! and of the LASSO alone against scikit-learn. Fixtures from `fixtures/generate.py`.

#![allow(clippy::unwrap_used)]

use modewise_signals::{MultichannelSeries, SamplingRate};
use modewise_vlmd::{Vlmd, VlmdParams, lasso};
use ndarray::{Array2, Array3};
use serde_json::Value;

fn fixture(name: &str) -> Value {
    let path = format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn matrix(v: &Value) -> Array2<f64> {
    let rows: Vec<Vec<f64>> = serde_json::from_value(v.clone()).unwrap();
    Array2::from_shape_fn((rows.len(), rows[0].len()), |(i, j)| rows[i][j])
}

fn cube(v: &Value) -> Array3<f64> {
    let k: Vec<Vec<Vec<f64>>> = serde_json::from_value(v.clone()).unwrap();
    Array3::from_shape_fn((k.len(), k[0].len(), k[0][0].len()), |(a, b, c)| k[a][b][c])
}

fn assert_close(
    name: &str,
    got: impl IntoIterator<Item = f64>,
    want: impl IntoIterator<Item = f64>,
    scale: f64,
    tol: f64,
) {
    for (i, (g, w)) in got.into_iter().zip(want).enumerate() {
        assert!((g - w).abs() <= tol * scale, "{name}[{i}]: {g} vs {w}");
    }
}

#[test]
fn lasso_matches_scikit_learn() {
    let f = fixture("lasso.json");
    let z = matrix(&f["latents"]);
    let x = matrix(&f["signal"]);
    for case in f["cases"].as_array().unwrap() {
        let mut a = Array2::zeros((z.nrows(), x.nrows()));
        lasso(
            z.view(),
            x.view(),
            case["penalty"].as_f64().unwrap(),
            &mut a,
        );
        let want = matrix(&case["loadings"]);
        assert_close(
            "loadings",
            a.iter().copied(),
            want.iter().copied(),
            1.0,
            1e-9,
        );
    }
}

#[test]
fn vlmd_matches_reference_with_exact_lasso() {
    let f = fixture("reference.json");
    for (name, case) in f["cases"].as_object().unwrap() {
        let x = matrix(&case["signal"]);
        let p = &case["params"];
        let get = |key: &str, default: f64| p.get(key).and_then(Value::as_f64).unwrap_or(default);
        let params = VlmdParams {
            num_modes: p["num_modes"].as_u64().unwrap() as usize,
            num_latents: case["num_latents"].as_u64().map(|l| l as usize),
            alpha: get("alpha", 100.0),
            reg_lambda: get("reg_lambda", 0.01),
            reg_rho: get("reg_rho", 1.0),
            tau: get("tau", 0.9),
            tolerance: get("tolerance", 1e-3),
            max_iterations: get("max_iter", 1000.0) as usize,
        };
        let labels = (0..x.nrows()).map(|c| format!("c{c}")).collect();
        let series = MultichannelSeries::new(x, SamplingRate::new(1.0).unwrap(), labels).unwrap();
        let out = Vlmd::new(params)
            .unwrap()
            .decompose_latent(&series)
            .unwrap();

        assert_eq!(
            out.iterations as u64,
            case["iterations"].as_u64().unwrap(),
            "{name} iterations"
        );
        let centres: Vec<f64> = serde_json::from_value(case["centres"].clone()).unwrap();
        assert_close(
            name,
            out.modes.centre_frequencies_hz().iter().copied(),
            centres,
            1.0,
            1e-8,
        );
        assert_close(
            name,
            out.loadings.values().iter().copied(),
            matrix(&case["loadings"]).iter().copied(),
            1.0,
            1e-7,
        );
        for (label, ours, theirs) in [
            (
                "latent modes",
                out.latent_modes.modes(),
                cube(&case["latent_modes"]),
            ),
            ("modes", out.modes.modes(), cube(&case["modes"])),
        ] {
            let peak = theirs.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            assert_eq!(ours.dim(), theirs.dim(), "{name} {label}");
            assert_close(
                &format!("{name} {label}"),
                ours.iter().copied(),
                theirs.iter().copied(),
                peak,
                1e-6,
            );
        }
    }
}
