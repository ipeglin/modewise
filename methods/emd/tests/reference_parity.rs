//! Accuracy against scipy (splines), PyEMD (EMD) and MEMD-Python (MEMD); fixtures from
//! `fixtures/generate.py`.

#![allow(clippy::unwrap_used)]

use modewise_decomposition::ModeDecomposer;
use modewise_emd::{Emd, Memd, Stop};
use modewise_signals::{MultichannelSeries, SamplingRate};
use ndarray::{Array2, Array3, Axis};
use serde_json::Value;

fn fixture() -> Value {
    let path = format!("{}/tests/fixtures/emd.json", env!("CARGO_MANIFEST_DIR"));
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

fn matrix(v: &Value) -> Array2<f64> {
    let rows: Vec<Vec<f64>> = v.as_array().unwrap().iter().map(floats).collect();
    Array2::from_shape_vec((rows.len(), rows[0].len()), rows.concat()).unwrap()
}

fn cube(v: &Value) -> Array3<f64> {
    let slabs: Vec<Array2<f64>> = v.as_array().unwrap().iter().map(matrix).collect();
    let (r, c) = slabs[0].dim();
    Array3::from_shape_vec(
        (slabs.len(), r, c),
        slabs.iter().flat_map(|s| s.iter().copied()).collect(),
    )
    .unwrap()
}

fn series(data: Array2<f64>) -> MultichannelSeries {
    let labels = (0..data.nrows()).map(|c| format!("ch{c}")).collect();
    MultichannelSeries::new(data, SamplingRate::new(1.0).unwrap(), labels).unwrap()
}

fn max_abs(a: impl IntoIterator<Item = f64>) -> f64 {
    a.into_iter().fold(0.0f64, |m, v| m.max(v.abs()))
}

#[test]
fn splines_match_scipy_not_a_knot() {
    for case in fixture()["splines"].as_array().unwrap() {
        let knots = floats(&case["knots"]);
        let got = modewise_emd::testing::spline(
            &knots,
            matrix(&case["values"]).view(),
            case["n"].as_u64().unwrap() as usize,
        );
        let error = max_abs((&got - &matrix(&case["expected"])).into_iter());
        assert!(error < 1e-9, "{} knots: {error:e}", knots.len());
    }
}

#[test]
fn emd_matches_pyemd_with_fixed_siftings() {
    let emd = Emd::new(Stop::Fixed { siftings: 10 }, Some(3)).unwrap();
    for case in fixture()["emd"].as_array().unwrap() {
        let signal = floats(&case["signal"]);
        let n = signal.len();
        let d = emd
            .decompose(&series(Array2::from_shape_vec((1, n), signal).unwrap()))
            .unwrap();
        let imfs = matrix(&case["imfs"]);
        assert_eq!(d.n_modes(), imfs.nrows() + 1);
        for (j, want) in imfs.rows().into_iter().enumerate() {
            let got = d
                .modes()
                .index_axis(Axis(0), imfs.nrows() - j)
                .row(0)
                .to_owned();
            let error = max_abs((&got - &want).into_iter());
            assert!(error < 1e-9, "IMF {}: {error:e}", j + 1);
        }
        let residual = d.modes().index_axis(Axis(0), 0).row(0).to_owned();
        assert!(
            max_abs(
                (&residual
                    - &floats(&case["residual"])
                        .into_iter()
                        .collect::<ndarray::Array1<f64>>())
                    .into_iter()
            ) < 1e-9
        );
    }
}

#[test]
fn memd_matches_memd_python() {
    for case in fixture()["memd"].as_array().unwrap() {
        let stop = match case["stop"].as_str().unwrap() {
            "fix_h" => Stop::SNumber {
                count: case["count"].as_u64().unwrap() as usize,
            },
            _ => Stop::RILLING_MEMD,
        };
        let signal = series(matrix(&case["signal"]));
        let d = Memd::new(stop, None, 64)
            .unwrap()
            .with_memd_flat_rule()
            .decompose(&signal)
            .unwrap();
        let want = cube(&case["modes"]);
        let name = case["name"].as_str().unwrap();
        assert_eq!(d.n_modes(), want.dim().0, "{name}: mode count");
        let k = want.dim().0;
        for j in 0..k {
            let got = d.modes().index_axis(Axis(0), k - 1 - j).to_owned();
            let error = max_abs((&got - &want.index_axis(Axis(0), j)).into_iter());
            assert!(error < 1e-8, "{name} IMF {}: {error:e}", j + 1);
        }
        assert!(d.aligned_across_channels());
    }
}

/// Without memd.m's flat-projection rule (the default), MEMD agrees with the reference until the
/// first slow mode where a projection is wholly negative.
#[test]
fn memd_without_the_flat_rule_agrees_on_the_fast_modes() {
    let case = &fixture()["memd"][0];
    let d = Memd::new(Stop::RILLING_MEMD, None, 64)
        .unwrap()
        .decompose(&series(matrix(&case["signal"])))
        .unwrap();
    let want = cube(&case["modes"]);
    let k = d.n_modes();
    for j in 0..5 {
        let got = d.modes().index_axis(Axis(0), k - 1 - j).to_owned();
        assert!(
            max_abs((&got - &want.index_axis(Axis(0), j)).into_iter()) < 1e-8,
            "IMF {}",
            j + 1
        );
    }
}

fn indices(v: &Value) -> Vec<usize> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as usize)
        .collect()
}

#[test]
fn boundary_mirroring_matches_pyemd_including_ties() {
    let (mut compared, mut refused) = (0, 0);
    for case in fixture()["boundary"].as_array().unwrap() {
        let x = floats(&case["signal"]);
        let got =
            modewise_emd::testing::mirror(&indices(&case["indmin"]), &indices(&case["indmax"]), &x);
        let want = &case["knots"];
        let Some(((min_t, min_s), (max_t, max_s))) = got else {
            refused += 1;
            continue;
        };
        let values = |s: &[usize]| s.iter().map(|&i| x[i]).collect::<Vec<_>>();
        assert_eq!(min_t, floats(&want["min_t"]), "{x:?}");
        assert_eq!(max_t, floats(&want["max_t"]), "{x:?}");
        assert_eq!(values(&min_s), floats(&want["min_z"]));
        assert_eq!(values(&max_s), floats(&want["max_z"]));
        compared += 1;
    }
    assert_eq!(refused, 0, "PyEMD mirrored every case; {compared} compared");
}
