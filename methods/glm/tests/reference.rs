#![allow(clippy::unwrap_used)]

//! Accuracy against nilearn and statsmodels (fixtures/generate.py).

use modewise_epochs::{Event, Events};
use modewise_glm::{NoiseModel, fit, task_regressors};
use ndarray::Array2;
use serde_json::Value;

fn reference() -> Value {
    serde_json::from_str(include_str!("fixtures/reference.json")).unwrap()
}

fn vector(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

fn matrix(v: &Value) -> Array2<f64> {
    let rows: Vec<Vec<f64>> = v.as_array().unwrap().iter().map(vector).collect();
    Array2::from_shape_fn((rows.len(), rows[0].len()), |(i, j)| rows[i][j])
}

fn max_difference(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    assert_eq!(a.dim(), b.dim());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

#[test]
fn regressors_match_nilearn_on_a_fine_grid() {
    let r = reference();
    let n = r["n"].as_u64().unwrap() as usize;
    let fs = 1.0 / r["t_r"].as_f64().unwrap();
    for case in r["regressors"].as_array().unwrap() {
        let events = vector(&case["onsets"])
            .into_iter()
            .zip(vector(&case["durations"]))
            .map(|(onset_s, duration_s)| Event {
                onset_s,
                duration_s,
                label: "x".into(),
            })
            .collect();
        let (_, ours) = task_regressors(&Events::new(events).unwrap(), n, fs);
        let theirs = Array2::from_shape_vec((1, n), vector(&case["values"])).unwrap();
        let peak = theirs.iter().fold(0.0, |m: f64, v| m.max(v.abs()));
        // nilearn convolves on a grid of T_R / 2000 and shifts its response by one grid step.
        assert!(
            max_difference(&ours, &theirs) < 1e-3 * peak,
            "{}: {} of peak {peak}",
            case["name"],
            max_difference(&ours, &theirs)
        );
    }
}

#[test]
fn fits_match_statsmodels() {
    let f = &reference()["fit"];
    let (signal, task, nuisance) = (
        matrix(&f["signal"]),
        matrix(&f["task"]),
        matrix(&f["nuisance"]),
    );
    let labels = ["face".to_string(), "shape".to_string()];
    let ols = fit(
        signal.view(),
        task.view(),
        &labels,
        nuisance.view(),
        NoiseModel::Ols,
    )
    .unwrap();
    assert!(max_difference(&ols.betas, &matrix(&f["ols_betas"])) < 1e-9);
    assert!(max_difference(&ols.t, &matrix(&f["ols_t"])) < 1e-9);
    assert_eq!(ols.degrees_of_freedom, f["df"].as_f64().unwrap());

    let ar1 = fit(
        signal.view(),
        task.view(),
        &labels,
        nuisance.view(),
        NoiseModel::Ar1,
    )
    .unwrap();
    let rho = ar1.ar1.unwrap();
    for (ours, theirs) in rho.iter().zip(vector(&f["rho"])) {
        assert!((ours - theirs).abs() < 1e-12, "{ours} {theirs}");
    }
    assert!(max_difference(&ar1.betas, &matrix(&f["ar1_betas"])) < 1e-9);
    assert!(max_difference(&ar1.t, &matrix(&f["ar1_t"])) < 1e-9);
    // nilearn leaves the first sample unwhitened: close, not equal.
    let nilearn = matrix(&f["nilearn_ar1_betas"]);
    assert!(max_difference(&ar1.betas, &nilearn) < 0.05);
    assert!(max_difference(&ar1.betas, &nilearn) > 1e-6);
}
