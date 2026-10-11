#![allow(clippy::unwrap_used)]

use std::collections::BTreeMap;

use modewise_cohort::{Design, Settings, test};
use modewise_stats::{Exchange, Fdr, Glm, PermutationTest, Statistic, Tail};
use modewise_study::{FeatureTable, Observation, Participants};
use ndarray::Array2;

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
            a: "Patient".into(),
            b: "GenPop".into(),
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
    let expected = Glm::new(design, 1).unwrap().t(y.view());
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
            a: "Patient".into(),
            b: "GenPop".into(),
            welch: true,
        },
        &settings(),
    )
    .unwrap();
    let in_a: Vec<bool> = (0..10).map(|s| s < 5).collect();
    let direct = PermutationTest {
        statistic: Statistic::Welch { in_a },
        exchange: Exchange::Permute,
        tail: Tail::Both,
        n_permutations: 200,
        seed: 3,
        nbs: None,
    }
    .run(t.values())
    .unwrap();
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
        a: "Patient".into(),
        b: "GenPop".into(),
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
        a: "Patient".into(),
        b: "GenPop".into(),
        welch: true,
    };
    assert!(
        err(
            &table(&[""]),
            &welch_cov,
            &Settings {
                covariates: vec!["age".into()],
                ..settings()
            }
        )
        .contains("no covariates")
    );
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
        a: "Patient".into(),
        b: "Nobody".into(),
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
            a: "Patient".into(),
            b: "GenPop".into(),
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
                a: a.into(),
                b: b.into(),
                welch: true,
            },
            &settings(),
        )
    };
    assert_eq!(pair("A", "B").unwrap().n_observations, 4);
    assert!(pair("C", "D").unwrap_err().contains("at least two"));

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
            a: "A".into(),
            b: "B".into(),
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
    assert_eq!(r.t, Glm::new(design, 1).unwrap().t(y.view()));

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
    let expected = Glm::new(
        Array2::from_shape_fn((10, 2), |(i, c)| if c == 0 { 1.0 } else { shaps[i] - mean }),
        0,
    )
    .unwrap()
    .t(diffs.view());
    for f in 0..2 {
        assert!(
            (r.t[f] - expected[f]).abs() < 1e-9 * expected[f].abs().max(1.0),
            "{} vs {}",
            r.t[f],
            expected[f]
        );
    }
}
