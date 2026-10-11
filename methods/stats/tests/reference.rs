//! Accuracy against statsmodels, scipy and bctpy (fixture from `fixtures/generate.py`), and the
//! behaviour of permutation inference.

#![allow(clippy::unwrap_used)]

use modewise_stats::{
    Edge, Exchange, Fdr, Glm, PermutationTest, Statistic, Tail, adjust, incomplete_beta, t_upper,
    welch_df, welch_t,
};
use ndarray::{Array1, Array2};
use serde_json::Value;

fn fixture() -> Value {
    let path = format!(
        "{}/tests/fixtures/reference.json",
        env!("CARGO_MANIFEST_DIR")
    );
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn matrix(v: &Value) -> Array2<f64> {
    let rows: Vec<Vec<f64>> = serde_json::from_value(v.clone()).unwrap();
    Array2::from_shape_fn((rows.len(), rows[0].len()), |(i, j)| rows[i][j])
}

fn floats(v: &Value) -> Vec<f64> {
    serde_json::from_value(v.clone()).unwrap()
}

fn close(a: f64, b: f64, tol: f64, what: &str) {
    assert!(
        (a - b).abs() <= tol * b.abs().max(1.0),
        "{what}: {a} vs {b}"
    );
}

fn test(
    statistic: Statistic,
    exchange: Exchange,
    tail: Tail,
    n_permutations: usize,
) -> PermutationTest {
    PermutationTest {
        statistic,
        exchange,
        tail,
        n_permutations,
        seed: 7,
        nbs: None,
    }
}

#[test]
fn glm_t_and_p_match_statsmodels() {
    let f = &fixture()["glm"];
    let (design, y) = (matrix(&f["design"]), matrix(&f["y"]));
    let glm = Glm::new(design.clone(), 1).unwrap();
    let r = test(Statistic::Glm(glm), Exchange::Permute, Tail::Both, 999)
        .run(y.view())
        .unwrap();
    for (i, (t, p)) in floats(&f["t"])
        .iter()
        .zip(floats(&f["p_two_sided"]))
        .enumerate()
    {
        close(r.t[i], *t, 1e-10, "t");
        close(r.p_parametric[i], p, 1e-9, "p");
    }
    let age = Glm::new(design, 2).unwrap();
    for (got, want) in age.t(y.view()).iter().zip(floats(&f["t_age"])) {
        close(*got, want, 1e-10, "t of age");
    }
}

#[test]
fn welch_matches_scipy() {
    let f = &fixture()["welch"];
    let y = Array2::from_shape_vec((24, 1), floats(&f["y"])).unwrap();
    let in_a: Vec<bool> = serde_json::from_value(f["in_a"].clone()).unwrap();
    close(
        welch_t(y.view(), &in_a)[0],
        f["t"].as_f64().unwrap(),
        1e-12,
        "t",
    );
    close(
        welch_df(y.view(), &in_a)[0],
        f["df"].as_f64().unwrap(),
        1e-10,
        "df",
    );
    let r = test(Statistic::Welch { in_a }, Exchange::Permute, Tail::Both, 99)
        .run(y.view())
        .unwrap();
    close(r.p_parametric[0], f["p"].as_f64().unwrap(), 1e-9, "p");
}

#[test]
fn distribution_matches_scipy() {
    let f = fixture();
    for case in f["betainc"].as_array().unwrap() {
        let g = |k: &str| case[k].as_f64().unwrap();
        close(
            incomplete_beta(g("a"), g("b"), g("x")),
            g("value"),
            1e-12,
            "betainc",
        );
    }
    for case in f["t_upper"].as_array().unwrap() {
        let g = |k: &str| case[k].as_f64().unwrap();
        close(t_upper(g("t"), g("df")), g("value"), 1e-12, "t sf");
    }
}

#[test]
fn fdr_matches_statsmodels() {
    let f = &fixture()["fdr"];
    let p = floats(&f["p"]);
    for (method, key) in [
        (Fdr::BenjaminiHochberg, "bh"),
        (Fdr::BenjaminiYekutieli, "by"),
    ] {
        for (got, want) in adjust(&p, method).iter().zip(floats(&f[key])) {
            close(*got, want, 1e-14, key);
        }
    }
}

#[test]
fn small_sign_flip_designs_are_enumerated_exactly() {
    let f = &fixture()["sign_flip"];
    let y = matrix(&f["y"]);
    let glm = Glm::new(Array2::ones((8, 1)), 0).unwrap();
    let r = test(
        Statistic::Glm(glm.clone()),
        Exchange::SignFlip,
        Tail::Both,
        1000,
    )
    .run(y.view())
    .unwrap();
    assert!(r.exhaustive && r.n_rearrangements == 255);
    for (i, (p, fwer)) in floats(&f["p"]).iter().zip(floats(&f["p_fwer"])).enumerate() {
        close(r.p_permutation[i], *p, 1e-12, "exact p");
        close(r.p_fwer[i], fwer, 1e-12, "exact FWER");
    }
    let sampled = test(Statistic::Glm(glm), Exchange::SignFlip, Tail::Both, 100)
        .run(y.view())
        .unwrap();
    assert!(!sampled.exhaustive && sampled.n_rearrangements == 100);
}

#[test]
fn nbs_components_match_bctpy_and_get_permutation_p() {
    let f = &fixture()["nbs"];
    let tmap = matrix(&f["t"]);
    let nodes = tmap.nrows();
    let pairs: Vec<(usize, usize)> = (0..nodes)
        .flat_map(|i| (i + 1..nodes).map(move |j| (i, j)))
        .collect();
    let edges: Vec<Option<Edge>> = pairs
        .iter()
        .map(|&(a, b)| Some(Edge { network: 0, a, b }))
        .collect();
    let observed: Vec<f64> = pairs.iter().map(|&(a, b)| tmap[[a, b]]).collect();
    let mut found: Vec<Vec<[usize; 2]>> =
        modewise_stats::components_for_test(&observed, &edges, 3.0)
            .into_iter()
            .map(|(_, e)| {
                e.iter()
                    .map(|&f| [pairs[f].0, pairs[f].1])
                    .collect::<Vec<_>>()
            })
            .map(|mut e: Vec<[usize; 2]>| {
                e.sort();
                e
            })
            .collect();
    found.sort();
    let want: Vec<Vec<[usize; 2]>> = serde_json::from_value(f["components"].clone()).unwrap();
    assert_eq!(found, want);
}

#[test]
fn permutation_p_approaches_parametric_p_for_gaussian_data_and_is_reproducible() {
    let n = 40;
    let mut state = 12345u64;
    let mut normal = || {
        // Box–Muller on a simple LCG, enough for a test.
        let mut uniform = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        };
        let (u, v) = (uniform(), uniform());
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    };
    let group = Array1::from_shape_fn(n, |i| if i < n / 2 { 1.0 } else { 0.0 });
    let y = Array2::from_shape_fn((n, 5), |(i, f)| {
        normal() + if f == 0 { 0.6 * group[i] } else { 0.0 }
    });
    let design = Array2::from_shape_fn((n, 2), |(i, c)| if c == 0 { 1.0 } else { group[i] });
    let run = |perms| {
        test(
            Statistic::Glm(Glm::new(design.clone(), 1).unwrap()),
            Exchange::Permute,
            Tail::Both,
            perms,
        )
        .run(y.view())
        .unwrap()
    };
    let r = run(4000);
    for f in 0..5 {
        let (p, q) = (r.p_permutation[f], r.p_parametric[f]);
        let se = (q * (1.0 - q) / 4000.0).sqrt();
        assert!(
            (p - q).abs() < 4.0 * se + 0.01,
            "feature {f}: permutation {p} vs parametric {q}"
        );
        assert!(
            r.p_fwer[f] >= r.p_permutation[f] - 1e-12,
            "FWER is never smaller"
        );
    }
    assert_eq!(run(300), run(300), "same seed, same result");
    let other_threads = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap()
        .install(|| run(300));
    assert_eq!(other_threads, run(300), "independent of the thread count");
}

#[test]
fn tails_nan_features_and_invalid_tests() {
    let y = ndarray::array![[1.0, f64::NAN], [2.0, 1.0], [3.0, 2.0], [4.0, 1.0]];
    let glm = Glm::new(Array2::ones((4, 1)), 0).unwrap();
    let greater = test(
        Statistic::Glm(glm.clone()),
        Exchange::SignFlip,
        Tail::Greater,
        100,
    )
    .run(y.view())
    .unwrap();
    let less = test(
        Statistic::Glm(glm.clone()),
        Exchange::SignFlip,
        Tail::Less,
        100,
    )
    .run(y.view())
    .unwrap();
    assert!(greater.p_parametric[0] < 0.05 && less.p_parametric[0] > 0.95);
    assert!(greater.p_permutation[0] < less.p_permutation[0]);
    assert!(
        greater.t[1].is_nan() && greater.p_permutation[1].is_nan() && greater.p_fwer[1].is_nan()
    );
    assert!(
        test(
            Statistic::Glm(glm.clone()),
            Exchange::SignFlip,
            Tail::Both,
            0
        )
        .run(y.view())
        .is_err()
    );
    assert!(
        test(Statistic::Glm(glm), Exchange::SignFlip, Tail::Both, 10)
            .run(y.slice(ndarray::s![..3, ..]))
            .is_err()
    );
    let welch = Statistic::Welch {
        in_a: vec![true, true, false, false],
    };
    assert!(
        test(welch, Exchange::SignFlip, Tail::Both, 10)
            .run(y.view())
            .is_err()
    );
}

#[test]
fn nbs_finds_a_planted_subnetwork_and_not_noise() {
    // 6 nodes, 15 edges; edges among nodes 0-2 carry a group effect, the rest are noise.
    let (n, nodes) = (30, 6);
    let pairs: Vec<(usize, usize)> = (0..nodes)
        .flat_map(|i| (i + 1..nodes).map(move |j| (i, j)))
        .collect();
    let edges: Vec<Option<Edge>> = pairs
        .iter()
        .map(|&(a, b)| Some(Edge { network: 0, a, b }))
        .collect();
    let mut state = 99u64;
    let mut noise = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64 / (1u64 << 53) as f64) - 0.5
    };
    let in_a: Vec<bool> = (0..n).map(|i| i < n / 2).collect();
    let y = Array2::from_shape_fn((n, pairs.len()), |(i, f)| {
        let (a, b) = pairs[f];
        noise() + if in_a[i] && a < 3 && b < 3 { 0.8 } else { 0.0 }
    });
    let design = Array2::from_shape_fn((n, 2), |(i, c)| if c == 0 || in_a[i] { 1.0 } else { 0.0 });
    let mut t = test(
        Statistic::Glm(Glm::new(design, 1).unwrap()),
        Exchange::Permute,
        Tail::Greater,
        500,
    );
    t.nbs = Some((edges, 3.0));
    let r = t.run(y.view()).unwrap();
    let planted: Vec<usize> = pairs
        .iter()
        .enumerate()
        .filter(|(_, (a, b))| *a < 3 && *b < 3)
        .map(|(f, _)| f)
        .collect();
    let main = r.components.iter().max_by_key(|c| c.edges.len()).unwrap();
    assert_eq!(main.edges, planted);
    assert!(main.p < 0.01, "planted subnetwork p = {}", main.p);
    assert_eq!(main.network, 0);
}

#[test]
fn freedman_lane_sign_flips_match_an_independent_enumeration() {
    let f = &fixture()["freedman_lane"];
    let (design, y) = (matrix(&f["design"]), matrix(&f["y"]));
    let glm = Glm::new(design, 0).unwrap();
    let both = test(
        Statistic::Glm(glm.clone()),
        Exchange::SignFlip,
        Tail::Both,
        1000,
    )
    .run(y.view())
    .unwrap();
    let greater = test(Statistic::Glm(glm), Exchange::SignFlip, Tail::Greater, 1000)
        .run(y.view())
        .unwrap();
    for i in 0..2 {
        close(both.t[i], floats(&f["t"])[i], 1e-10, "t");
        close(
            both.p_permutation[i],
            floats(&f["p_both"])[i],
            1e-12,
            "two-sided",
        );
        close(
            greater.p_permutation[i],
            floats(&f["p_greater"])[i],
            1e-12,
            "one-sided",
        );
    }
}

#[test]
fn enumeration_starts_at_exactly_two_to_the_n_minus_one_and_random_flips_find_effects() {
    let y = Array2::from_shape_fn((8, 1), |(i, _)| 1.0 + 0.1 * i as f64);
    let glm = Glm::new(Array2::ones((8, 1)), 0).unwrap();
    let at = |perms| {
        test(
            Statistic::Glm(glm.clone()),
            Exchange::SignFlip,
            Tail::Greater,
            perms,
        )
        .run(y.view())
        .unwrap()
    };
    assert!(at(255).exhaustive && !at(254).exhaustive);
    assert!(
        (at(255).p_permutation[0] - 1.0 / 256.0).abs() < 1e-15,
        "only the observed signs are as large"
    );

    let n = 20;
    let effect =
        Array2::from_shape_fn((n, 1), |(i, _)| 0.8 + ((i * 7919) % 13) as f64 / 13.0 - 0.5);
    let sampled = test(
        Statistic::Glm(Glm::new(Array2::ones((n, 1)), 0).unwrap()),
        Exchange::SignFlip,
        Tail::Both,
        500,
    )
    .run(effect.view())
    .unwrap();
    assert!(
        !sampled.exhaustive && sampled.p_permutation[0] < 0.01,
        "{}",
        sampled.p_permutation[0]
    );
}

#[test]
fn components_match_a_brute_force_search_on_random_graphs() {
    let mut state = 7u64;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as f64 / (1u64 << 31) as f64
    };
    for _ in 0..40 {
        let nodes = 6;
        let edges: Vec<Option<Edge>> = (0..2)
            .flat_map(|network| {
                (0..nodes)
                    .flat_map(move |a| (a + 1..nodes).map(move |b| Some(Edge { network, a, b })))
            })
            .collect();
        let stat: Vec<f64> = edges.iter().map(|_| next() * 4.0).collect();
        let mut found: Vec<Vec<usize>> = modewise_stats::components_for_test(&stat, &edges, 3.0)
            .into_iter()
            .map(|(_, e)| e)
            .collect();
        found.sort();
        // Brute force: grow each component from an unvisited suprathreshold edge.
        let supra: Vec<usize> = (0..edges.len()).filter(|&f| stat[f] >= 3.0).collect();
        let mut seen = vec![false; edges.len()];
        let mut expected = Vec::new();
        for &start in &supra {
            if seen[start] {
                continue;
            }
            let mut component = vec![start];
            seen[start] = true;
            let mut grew = true;
            while grew {
                grew = false;
                for &f in &supra {
                    let touches = |g: usize| {
                        let (x, y) = (edges[g].unwrap(), edges[f].unwrap());
                        x.network == y.network
                            && (x.a == y.a || x.a == y.b || x.b == y.a || x.b == y.b)
                    };
                    if !seen[f] && component.iter().any(|&g| touches(g)) {
                        seen[f] = true;
                        component.push(f);
                        grew = true;
                    }
                }
            }
            component.sort();
            expected.push(component);
        }
        expected.sort();
        assert_eq!(found, expected);
    }
}

#[test]
fn near_singular_designs_are_refused_relative_to_their_scale() {
    let n = 6;
    let design = Array2::from_shape_fn((n, 2), |(i, c)| {
        if c == 0 {
            1e3
        } else {
            1e3 + if i == 0 { 1e-10 } else { 0.0 }
        }
    });
    assert!(Glm::new(design, 1).is_err());
}

#[test]
fn rearrangements_equal_up_to_rounding_count_as_ties() {
    // Flipping every sign of one-sample data gives exactly −t, so |t| ties with the observed.
    let y = Array2::from_shape_fn((6, 1), |(i, _)| 0.3 + 0.17 * i as f64);
    let glm = Glm::new(Array2::ones((6, 1)), 0).unwrap();
    let r = test(Statistic::Glm(glm), Exchange::SignFlip, Tail::Both, 100)
        .run(y.view())
        .unwrap();
    assert!(r.exhaustive);
    assert!(
        (r.p_permutation[0] - 2.0 / 64.0).abs() < 1e-15 && (r.p_fwer[0] - 2.0 / 64.0).abs() < 1e-15
    );
}
