#![allow(clippy::unwrap_used)]

use std::collections::BTreeMap;

use modewise_cohort::{Design, Settings, test};
use modewise_stats::{Exchange, Fdr, Glm, PermutationTest, Tail};
use modewise_study::{FeatureTable, Observation, Participants};
use ndarray::Array2;

/// t of column `j` of `design` for each feature of `y`.
fn glm_t(design: Array2<f64>, j: usize, y: ndarray::ArrayView2<f64>) -> ndarray::Array1<f64> {
    let p = design.ncols();
    let contrast = Array2::from_shape_fn((1, p), |(_, k)| f64::from(k == j));
    Glm::new(design, contrast).unwrap().statistic(y)
}

fn settings() -> Settings {
    Settings {
        covariates: vec![],
        condition: None,
        tail: Tail::Both,
        n_permutations: 200,
        seed: 3,
        fdr: Fdr::BenjaminiHochberg,
        nbs_threshold: None,
    }
}

/// 10 subjects: group, age (missing for s09), sex; two features, the first with a group effect.
fn table(conditions: &[&str]) -> FeatureTable {
    let mut rows = Vec::new();
    let mut values = Vec::new();
    for s in 0..10 {
        for (c, condition) in conditions.iter().enumerate() {
            rows.push(Observation {
                recording: format!("subject-s{s:02}"),
                subject: Some(format!("s{s:02}")),
                condition: (!condition.is_empty()).then(|| condition.to_string()),
            });
            let group = if s < 5 { 1.0 } else { 0.0 };
            values.extend([
                group * 2.0 + (s as f64 * 0.37).sin() + c as f64,
                (s as f64 * 1.3).cos() + 0.1 * c as f64,
            ]);
        }
    }
    let p_rows: BTreeMap<String, BTreeMap<String, String>> = (0..10)
        .map(|s| {
            let mut v = BTreeMap::from([
                (
                    "group".to_string(),
                    if s < 5 { "Patient" } else { "GenPop" }.to_string(),
                ),
                (
                    "sex".to_string(),
                    if s % 2 == 0 { "F" } else { "M" }.to_string(),
                ),
                ("shaps".to_string(), (s % 4).to_string()),
            ]);
            if s != 9 {
                v.insert("age".to_string(), (20 + 3 * s).to_string());
            }
            (format!("s{s:02}"), v)
        })
        .collect();
    let p = Participants::new(
        vec!["group".into(), "sex".into(), "shaps".into(), "age".into()],
        p_rows,
    )
    .unwrap();
    let n = rows.len();
    FeatureTable::new(
        rows,
        vec!["m0:a~b".into(), "m0:a~c".into()],
        Array2::from_shape_vec((n, 2), values).unwrap(),
    )
    .unwrap()
    .with_participants(&p)
    .unwrap()
}

#[test]
fn group_glm_with_covariates_matches_a_hand_built_design() {
    let t = table(&[""]);
    let s = Settings {
        covariates: vec!["age".into(), "sex".into()],
        ..settings()
    };
    let r = test(
        &t,
        &Design::Groups {
            column: "group".into(),
            levels: vec!["Patient".into(), "GenPop".into()],
            welch: false,
        },
        &s,
    )
    .unwrap();
    assert_eq!((r.n_observations, r.dropped), (9, 1), "s09 has no age");
    assert_eq!(r.regressors, ["intercept", "group=Patient", "age", "sex=M"]);
    let kept: Vec<usize> = (0..9).collect();
    let centre = |x: Vec<f64>| {
        let m = x.iter().sum::<f64>() / x.len() as f64;
        x.into_iter().map(|v| v - m).collect::<Vec<_>>()
    };
    let age = centre(kept.iter().map(|&s| (20 + 3 * s) as f64).collect());
    let male = centre(
        kept.iter()
            .map(|&s| if s % 2 == 1 { 1.0 } else { 0.0 })
            .collect(),
    );
    let design = Array2::from_shape_fn((9, 4), |(i, c)| {
        [1.0, if i < 5 { 1.0 } else { 0.0 }, age[i], male[i]][c]
    });
    let y = t.values().slice(ndarray::s![..9, ..]).to_owned();
    let expected = glm_t(design, 1, y.view());
    for f in 0..2 {
        assert!((r.t[f] - expected[f]).abs() < 1e-12);
    }
    assert!(r.p_permutation[0] < 0.05 && r.q_fdr[0] >= r.p_permutation[0]);
    assert!(
        r.p_fwer
            .iter()
            .zip(r.p_permutation.iter())
            .all(|(f, p)| f >= p)
    );
}

#[test]
fn welch_paired_and_correlation_designs() {
    let t = table(&[""]);
    let welch = test(
        &t,
        &Design::Groups {
            column: "group".into(),
            levels: vec!["Patient".into(), "GenPop".into()],
            welch: true,
        },
        &settings(),
    )
    .unwrap();
    let design = Array2::from_shape_fn((10, 2), |(i, c)| f64::from(c == 0 || i < 5));
    let contrast = ndarray::array![[0.0, 1.0]];
    let v = Glm::new(design, contrast).unwrap();
    let direct = PermutationTest {
        glm: v
            .with_variance_groups((0..10).map(|s| usize::from(s >= 5)).collect())
            .unwrap(),
        exchange: Exchange::Permute,
        blocks: None,
        tail: Tail::Both,
        n_permutations: 200,
        seed: 3,
        nbs: None,
    }
    .run(t.values())
    .unwrap();
    assert_eq!((welch.statistic, direct.statistic), ("v", "v"));
    assert_eq!(welch.t, direct.t);
    assert_eq!(welch.p_permutation, direct.p_permutation);

    let both = table(&["face", "shape"]);
    let paired = test(
        &both,
        &Design::Paired {
            a: "face".into(),
            b: "shape".into(),
        },
        &Settings {
            n_permutations: 2000,
            ..settings()
        },
    )
    .unwrap();
    assert!(
        paired.exhaustive && paired.n_rearrangements == 1023,
        "2^10 − 1 flips enumerated"
    );
    // face − shape is −1 (up to rounding) in every recording: a huge negative t, and only the
    // all-negative and all-positive flips are as extreme.
    assert!(paired.t[0] < -1e6, "{}", paired.t[0]);
    assert!((paired.p_permutation[0] - 2.0 / 1024.0).abs() < 1e-12);
    assert_eq!(paired.regressors, ["intercept"]);

    let score = Settings {
        condition: Some("face".into()),
        ..settings()
    };
    let correlation = test(
        &both,
        &Design::Correlation {
            score: "shaps".into(),
        },
        &score,
    )
    .unwrap();
    assert_eq!(
        (
            correlation.regressors[1].as_str(),
            correlation.n_observations
        ),
        ("shaps", 10)
    );
}

#[test]
fn designs_refuse_what_breaks_their_assumptions() {
    let both = table(&["face", "shape"]);
    let groups = Design::Groups {
        column: "group".into(),
        levels: vec!["Patient".into(), "GenPop".into()],
        welch: false,
    };
    let err = |t: &FeatureTable, d: &Design, s: &Settings| test(t, d, s).unwrap_err();
    assert!(err(&both, &groups, &settings()).contains("choose one with `condition`"));
    assert!(
        err(
            &both,
            &groups,
            &Settings {
                condition: Some("rest".into()),
                ..settings()
            }
        )
        .contains("no condition")
    );
    let welch_cov = Design::Groups {
        column: "group".into(),
        levels: vec!["Patient".into(), "GenPop".into()],
        welch: true,
    };
    let with_age = Settings {
        covariates: vec!["age".into()],
        ..settings()
    };
    let v = test(&table(&[""]), &welch_cov, &with_age).unwrap();
    assert_eq!(v.statistic, "v", "Welch with covariates is Aspin–Welch v");
    assert_eq!(v.regressors, ["intercept", "group=Patient", "age"]);
    assert!(
        err(
            &both,
            &Design::Paired {
                a: "face".into(),
                b: "rest".into()
            },
            &settings()
        )
        .contains("no recording has both")
    );
    assert!(
        err(
            &table(&[""]),
            &Design::Correlation {
                score: "group".into()
            },
            &settings()
        )
        .contains("numeric")
    );
    assert!(
        err(
            &table(&[""]),
            &groups,
            &Settings {
                covariates: vec!["height".into()],
                ..settings()
            }
        )
        .contains("height")
    );

    let mut twice = table(&[""]).observations().to_vec();
    twice[1].subject = Some("s00".into());
    let dup =
        FeatureTable::new(twice, vec!["x".into(), "y".into()], Array2::zeros((10, 2))).unwrap();
    let corr = Design::Correlation {
        score: "shaps".into(),
    };
    assert!(
        err(&dup, &corr, &settings()).contains("no column")
            || err(&dup, &corr, &settings()).contains("more than once")
    );
    let p = Participants::new(
        vec!["shaps".into()],
        (0..10)
            .map(|s| {
                (
                    format!("s{s:02}"),
                    BTreeMap::from([("shaps".to_string(), "1".to_string())]),
                )
            })
            .collect(),
    )
    .unwrap();
    assert!(
        err(&dup.with_participants(&p).unwrap(), &corr, &settings()).contains("more than once")
    );
    let one_group = Design::Groups {
        column: "group".into(),
        levels: vec!["Patient".into(), "Nobody".into()],
        welch: true,
    };
    assert!(err(&table(&[""]), &one_group, &settings()).contains("at least two"));
}

#[test]
fn nbs_maps_edge_features_to_networks() {
    let t = table(&[""]);
    let s = Settings {
        nbs_threshold: Some(2.0),
        tail: Tail::Greater,
        ..settings()
    };
    let r = test(
        &t,
        &Design::Groups {
            column: "group".into(),
            levels: vec!["Patient".into(), "GenPop".into()],
            welch: false,
        },
        &s,
    )
    .unwrap();
    // Only the edge with the group effect passes. A one-edge component's p is the chance that any
    // edge passes under the null, so it is at least that edge's FWER p.
    assert_eq!(r.components.len(), 1);
    assert_eq!(
        (r.components[0].network, r.components[0].edges.clone()),
        (0, vec!["m0:a~b".to_string()])
    );
    assert!(
        r.components[0].p >= r.p_fwer[0] - 1e-12 && r.components[0].p < 0.2,
        "{}",
        r.components[0].p
    );
}

#[test]
fn boundaries_filters_and_covariates_in_paired_designs() {
    let t = table(&[""]);
    // Groups of exactly two are allowed; a group of one is not.
    let mut p_rows = BTreeMap::new();
    for s in 0..10 {
        let group = match s {
            0 | 1 => "A",
            2 | 3 => "B",
            4 => "C",
            _ => "D",
        };
        p_rows.insert(
            format!("s{s:02}"),
            BTreeMap::from([("g".to_string(), group.to_string())]),
        );
    }
    let p = Participants::new(vec!["g".into()], p_rows).unwrap();
    let small = FeatureTable::new(
        t.observations().to_vec(),
        t.feature_names().to_vec(),
        t.values().to_owned(),
    )
    .unwrap()
    .with_participants(&p)
    .unwrap();
    let pair = |a: &str, b: &str| {
        test(
            &small,
            &Design::Groups {
                column: "g".into(),
                levels: vec![a.into(), b.into()],
                welch: true,
            },
            &settings(),
        )
    };
    assert_eq!(pair("A", "B").unwrap().n_observations, 4);
    assert!(pair("C", "D").unwrap_err().contains("at least two"));
    let three = Design::Groups {
        column: "g".into(),
        levels: vec!["A".into(), "B".into(), "C".into()],
        welch: false,
    };
    let err = test(&small, &three, &settings()).unwrap_err();
    assert!(err.contains("at least two"), "C has one subject: {err}");

    // A missing group label and a missing covariate are both counted.
    let mut partial = BTreeMap::new();
    for s in 0..10 {
        let mut v = BTreeMap::new();
        if s != 0 {
            v.insert("g".to_string(), if s < 5 { "A" } else { "B" }.to_string());
        }
        if s != 8 && s != 9 {
            v.insert("age".to_string(), (30 + s).to_string());
        }
        partial.insert(format!("s{s:02}"), v);
    }
    let p = Participants::new(vec!["g".into(), "age".into()], partial).unwrap();
    let gappy = FeatureTable::new(
        t.observations().to_vec(),
        t.feature_names().to_vec(),
        t.values().to_owned(),
    )
    .unwrap()
    .with_participants(&p)
    .unwrap();
    let r = test(
        &gappy,
        &Design::Groups {
            column: "g".into(),
            levels: vec!["A".into(), "B".into()],
            welch: false,
        },
        &Settings {
            covariates: vec!["age".into()],
            ..settings()
        },
    )
    .unwrap();
    assert_eq!(
        (r.n_observations, r.dropped),
        (7, 3),
        "one missing label, two missing ages"
    );

    // The condition filter picks the right rows: compare with a GLM on the face rows by hand.
    let both = table(&["face", "shape"]);
    let s = Settings {
        condition: Some("shape".into()),
        ..settings()
    };
    let r = test(
        &both,
        &Design::Correlation {
            score: "shaps".into(),
        },
        &s,
    )
    .unwrap();
    let shaps: Vec<f64> = (0..10).map(|s| (s % 4) as f64).collect();
    let mean = shaps.iter().sum::<f64>() / 10.0;
    let design =
        Array2::from_shape_fn((10, 2), |(i, c)| if c == 0 { 1.0 } else { shaps[i] - mean });
    let shape_rows: Vec<usize> = (0..10).map(|s| 2 * s + 1).collect();
    let y = both.values().select(ndarray::Axis(0), &shape_rows);
    assert_eq!(r.t, glm_t(design, 1, y.view()));

    // Paired with a covariate: the intercept of differences on a centred covariate.
    let r = test(
        &both,
        &Design::Paired {
            a: "face".into(),
            b: "shape".into(),
        },
        &Settings {
            covariates: vec!["shaps".into()],
            ..settings()
        },
    )
    .unwrap();
    assert_eq!(r.regressors, ["intercept", "shaps"]);
    let diffs = Array2::from_shape_fn((10, 2), |(s, f)| {
        both.values()[[2 * s, f]] - both.values()[[2 * s + 1, f]]
    });
    let expected = glm_t(
        Array2::from_shape_fn((10, 2), |(i, c)| if c == 0 { 1.0 } else { shaps[i] - mean }),
        0,
        diffs.view(),
    );
    for f in 0..2 {
        assert!(
            (r.t[f] - expected[f]).abs() < 1e-9 * expected[f].abs().max(1.0),
            "{} vs {}",
            r.t[f],
            expected[f]
        );
    }
}

#[test]
fn three_groups_give_f_or_g() {
    let t = table(&[""]);
    let p = Participants::new(
        vec!["site".into()],
        (0..10)
            .map(|s| {
                let site = ["A", "B", "C"][s % 3].to_string();
                (format!("s{s:02}"), BTreeMap::from([("site".into(), site)]))
            })
            .collect(),
    )
    .unwrap();
    let t = t.with_participants(&p).unwrap();
    let design = |welch| Design::Groups {
        column: "site".into(),
        levels: vec!["A".into(), "B".into(), "C".into()],
        welch,
    };
    let r = test(&t, &design(false), &settings()).unwrap();
    assert_eq!((r.statistic, r.rank), ("F", 2));
    assert_eq!(r.regressors, ["intercept", "site=A", "site=B"]);
    let x = Array2::from_shape_fn((10, 3), |(i, c)| match c {
        0 => 1.0,
        c => f64::from(i % 3 == c - 1),
    });
    let f = Glm::new(x, ndarray::array![[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]).unwrap();
    assert_eq!(r.t, f.statistic(t.values()));
    assert_eq!(test(&t, &design(true), &settings()).unwrap().statistic, "G");
    let repeated = Design::Groups {
        column: "site".into(),
        levels: vec!["A".into(), "A".into()],
        welch: false,
    };
    assert!(
        test(&t, &repeated, &settings())
            .unwrap_err()
            .contains("different levels")
    );
}

/// The design matrix of a within-subject effect: condition, plus one column per subject.
fn within_subject(table: &FeatureTable) -> (Array2<f64>, Vec<String>) {
    let rows = table.observations();
    let design = Array2::from_shape_fn((rows.len(), 11), |(i, c)| match c {
        0 => f64::from(rows[i].condition.as_deref() == Some("face")),
        c => f64::from(i / 2 == c - 1),
    });
    let mut names = vec!["face".to_string()];
    names.extend((0..10).map(|s| format!("s{s:02}")));
    (design, names)
}

#[test]
fn a_design_matrix_takes_blocks_and_within_subject_permutation_is_paired_sign_flipping() {
    let template = table(&["face", "shape"]);
    let values = Array2::from_shape_fn((20, 2), |(i, f)| {
        (i as f64 * 0.7 + f as f64).sin() + 0.4 * f64::from(i % 2 == 0)
    });
    let both = FeatureTable::new(
        template.observations().to_vec(),
        template.feature_names().to_vec(),
        values,
    )
    .unwrap();
    let (design, regressors) = within_subject(&both);
    let matrix = |blocks: Option<&str>, whole| Design::Matrix {
        design: design.clone(),
        regressors: regressors.clone(),
        contrast: tested_first(11),
        exchange: Exchange::Permute,
        blocks: blocks.map(Into::into),
        whole_blocks: whole,
        variance_groups: None,
    };
    let s = Settings {
        n_permutations: 2000,
        ..settings()
    };
    let within = test(&both, &matrix(Some("subject"), false), &s).unwrap();
    let paired = test(
        &both,
        &Design::Paired {
            a: "face".into(),
            b: "shape".into(),
        },
        &s,
    )
    .unwrap();
    assert!(within.exhaustive && within.n_rearrangements == 1023);
    assert_eq!(within.regressors, regressors);
    for f in 0..2 {
        assert!((within.t[f] - paired.t[f]).abs() < 1e-9 * paired.t[f].abs());
        assert!((within.p_permutation[f] - paired.p_permutation[f]).abs() < 1e-12);
    }
    let unblocked = test(&both, &matrix(None, false), &s).unwrap_err();
    assert!(
        unblocked.contains("use test_glm with blocks"),
        "{unblocked}"
    );

    // A between-subject group with two rows per subject: whole subjects move together.
    let group = Array2::from_shape_fn((20, 2), |(i, c)| f64::from(c == 0 || i < 10));
    let whole = Design::Matrix {
        design: group.clone(),
        regressors: vec!["intercept".into(), "patient".into()],
        contrast: ndarray::array![[0.0, 1.0]],
        exchange: Exchange::Permute,
        blocks: Some("subject".into()),
        whole_blocks: true,
        variance_groups: Some("group".into()),
    };
    let r = test(&template, &whole, &settings()).unwrap();
    assert_eq!(
        (r.statistic, r.n_observations, r.exhaustive),
        ("v", 20, false)
    );
}

fn tested_first(p: usize) -> Array2<f64> {
    Array2::from_shape_fn((1, p), |(_, k)| f64::from(k == 0))
}

#[test]
fn a_design_matrix_is_checked() {
    let one = table(&[""]);
    let ok = Design::Matrix {
        design: Array2::ones((10, 1)),
        regressors: vec!["intercept".into()],
        contrast: ndarray::array![[1.0]],
        exchange: Exchange::SignFlip,
        blocks: None,
        whole_blocks: false,
        variance_groups: None,
    };
    assert_eq!(
        test(&one, &ok, &settings()).unwrap().regressors,
        ["intercept"]
    );
    let with = |f: &dyn Fn(&mut Design)| {
        let mut d = ok.clone();
        f(&mut d);
        test(&one, &d, &settings()).unwrap_err()
    };
    let set_design = |x: Array2<f64>| {
        move |d: &mut Design| {
            if let Design::Matrix { design, .. } = d {
                *design = x.clone();
            }
        }
    };
    assert!(with(&set_design(Array2::ones((9, 1)))).contains("9 rows"));
    let mut missing = Array2::ones((10, 1));
    missing[[3, 0]] = f64::NAN;
    assert!(with(&set_design(missing)).contains("missing"));
    assert!(
        with(&|d| if let Design::Matrix { regressors, .. } = d {
            regressors.push("extra".into());
        })
        .contains("2 regressor names")
    );
    assert!(
        with(&|d| if let Design::Matrix { blocks, .. } = d {
            *blocks = Some("age".into());
        })
        .contains("has no age")
    );
    assert!(
        with(&|d| if let Design::Matrix {
            variance_groups, ..
        } = d
        {
            *variance_groups = Some("height".into());
        })
        .contains("height")
    );
    let covariates = Settings {
        covariates: vec!["age".into()],
        ..settings()
    };
    assert!(
        test(&one, &ok, &covariates)
            .unwrap_err()
            .contains("no covariates")
    );
}
