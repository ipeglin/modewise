//! Accuracy against FSL PALM (fixture from `fixtures/palm.py`): contrasts, F, v and G, and
//! exhaustive permutations free, within blocks and of whole blocks.

#![allow(clippy::unwrap_used)]

use modewise_stats::{Blocks, Exchange, Glm, PermutationTest, Tail, TestResult};
use ndarray::Array2;
use serde_json::Value;

fn fixture() -> Value {
    let path = format!("{}/tests/fixtures/palm.json", env!("CARGO_MANIFEST_DIR"));
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn matrix(v: &Value) -> Array2<f64> {
    let rows: Vec<Vec<f64>> = serde_json::from_value(v.clone()).unwrap();
    Array2::from_shape_fn((rows.len(), rows[0].len()), |(i, j)| rows[i][j])
}

fn floats(v: &Value) -> Vec<f64> {
    serde_json::from_value(v.clone()).unwrap()
}

fn labels(v: &Value) -> Vec<usize> {
    serde_json::from_value(v.clone()).unwrap()
}

fn close(got: &[f64], want: &[f64], tol: f64, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}");
    for (a, b) in got.iter().zip(want) {
        assert!(
            (a - b).abs() <= tol * b.abs().max(1.0),
            "{what}: {a} vs {b}"
        );
    }
}

fn run(case: &Value, exchange: Exchange, blocks: Option<Blocks>) -> TestResult {
    let mut glm = Glm::new(matrix(&case["design"]), matrix(&case["contrast"])).unwrap();
    if !case["vg"].is_null() {
        let groups = labels(&case["vg"]).iter().map(|g| g - 1).collect();
        glm = glm.with_variance_groups(groups).unwrap();
    }
    PermutationTest {
        glm,
        exchange,
        blocks,
        tail: Tail::Both,
        n_permutations: 100_000,
        seed: 1,
        nbs: None,
    }
    .run(matrix(&case["y"]).view())
    .unwrap()
}

fn statistic_matches(name: &str, kind: &str) -> TestResult {
    let case = &fixture()[name];
    let r = run(case, Exchange::Permute, None);
    assert_eq!(r.statistic, kind);
    assert_eq!(case["statistic"], format!("{}stat", kind.to_lowercase()));
    close(
        r.t.as_slice().unwrap(),
        &floats(&case["value"]),
        1e-10,
        name,
    );
    r
}

#[test]
fn t_and_f_contrasts_match_palm() {
    for (name, kind) in [("contrast_t", "t"), ("f", "F")] {
        let case = &fixture()[name];
        let r = statistic_matches(name, kind);
        assert!(
            r.df.iter()
                .all(|&d| d == case["dof"]["df2"].as_f64().unwrap())
        );
        let p = floats(&case["p_parametric"]);
        close(r.p_parametric.as_slice().unwrap(), &p, 1e-9, name);
    }
}

#[test]
fn g_matches_palm() {
    let case = &fixture()["g"];
    let r = statistic_matches("g", "G");
    close(
        r.df.as_slice().unwrap(),
        &floats(&case["dof"]["df2"]),
        1e-10,
        "df",
    );
    let p = floats(&case["p_parametric"]);
    close(r.p_parametric.as_slice().unwrap(), &p, 1e-9, "p");
}

#[test]
fn v_matches_palm_and_its_degrees_of_freedom_follow_welch() {
    for name in ["v", "welch"] {
        let case = &fixture()[name];
        let r = statistic_matches(name, "v");
        // PALM gives v one third of the Welch–Satterthwaite degrees of freedom (Welch 1947),
        // which scipy, Welch's ANOVA and PALM's own G at one contrast row use; modewise follows
        // Welch, so the parametric p-values differ from PALM's.
        let palm = floats(&case["dof"]["df2"]);
        let tripled: Vec<f64> = palm.iter().map(|d| 3.0 * d).collect();
        close(r.df.as_slice().unwrap(), &tripled, 1e-10, name);
    }
}

#[test]
fn exhaustive_permutations_match_an_enumeration_and_palm() {
    let f = fixture();
    let block = |name: &str, whole| {
        Some(Blocks {
            block: labels(&f[name]["eb"]),
            whole,
        })
    };
    // PALM enumerates the distinct arrangements of the tested regressor (70 and 6 here). It
    // counts the mirrored arrangement, whose |t| equals the observed one in exact arithmetic,
    // as smaller by rounding; modewise counts it as a tie, as the numpy enumeration does.
    for (name, exchange, blocks, count, palm_mirror) in [
        ("exhaustive", Exchange::Permute, None, 40_320, Some(70.0)),
        (
            "within",
            Exchange::Permute,
            block("within", false),
            216,
            None,
        ),
        (
            "whole",
            Exchange::Permute,
            block("whole", true),
            24,
            Some(6.0),
        ),
        (
            "whole_flip",
            Exchange::SignFlip,
            block("whole_flip", true),
            32,
            None,
        ),
    ] {
        let case = &f[name];
        let r = run(case, exchange, blocks);
        assert!(r.exhaustive && r.n_rearrangements == count - 1, "{name}");
        close(
            r.t.as_slice().unwrap(),
            &floats(&case["value"]),
            1e-10,
            name,
        );
        let p = r.p_permutation.as_slice().unwrap();
        close(p, &floats(&case["p_enumerated"]), 1e-12, name);
        let fwer = r.p_fwer.as_slice().unwrap();
        close(fwer, &floats(&case["p_fwer_enumerated"]), 1e-12, name);
        let palm: Vec<f64> = match palm_mirror {
            None => floats(&case["p_permutation"]),
            Some(distinct) => floats(&case["p_permutation"])
                .iter()
                .map(|v| v + 1.0 / distinct)
                .collect(),
        };
        close(p, &palm, 1e-12, name);
    }
}
