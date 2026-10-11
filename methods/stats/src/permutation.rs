//! Permutation inference with family-wise error by the maximum statistic.

use modewise_method::MethodError;
use ndarray::{Array1, Array2, ArrayView2, Axis};
use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

use crate::dist::t_upper;
use crate::glm::{Glm, welch_df, welch_t};
use crate::nbs::{Component, Edge, components, largest};

/// Which differences count as evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tail {
    /// The tested effect is positive.
    Greater,
    /// The tested effect is negative.
    Less,
    /// Either sign.
    Both,
}

/// Rearranged statistics within this relative distance of the observed one count as ties (as
/// large). Rearrangements that are equal in exact arithmetic, such as flipping every sign in a
/// two-sided test, must not be split by rounding.
const TIE: f64 = 1e-12;

fn at_least(value: f64, observed: f64) -> bool {
    value >= observed - TIE * observed.abs().max(1.0)
}

impl Tail {
    fn stat(self, t: f64) -> f64 {
        match self {
            Self::Greater => t,
            Self::Less => -t,
            Self::Both => t.abs(),
        }
    }
}

/// What the observations may be rearranged by under the null hypothesis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Exchange {
    /// Observations are exchangeable: shuffle them.
    Permute,
    /// Errors are symmetric around zero: flip signs (paired differences).
    SignFlip,
}

/// The statistic tested.
#[derive(Debug, Clone)]
pub enum Statistic {
    /// t of one regressor of a GLM; nuisance regressors are handled by Freedman–Lane.
    Glm(Glm),
    /// Welch's t of group a against the rest, for two groups without covariates.
    Welch { in_a: Vec<bool> },
}

#[derive(Debug, Clone)]
pub struct PermutationTest {
    pub statistic: Statistic,
    pub exchange: Exchange,
    pub tail: Tail,
    pub n_permutations: usize,
    pub seed: u64,
    /// Network-based statistic over edge features, at this primary threshold of the tailed t.
    pub nbs: Option<(Vec<Option<Edge>>, f64)>,
}

/// Per-feature results.
#[derive(Debug, Clone, PartialEq)]
pub struct TestResult {
    pub t: Array1<f64>,
    pub df: Array1<f64>,
    /// From Student's t (Welch–Satterthwaite df for Welch).
    pub p_parametric: Array1<f64>,
    /// (b + 1) / (m + 1) over m rearrangements (Phipson & Smyth 2010); exact when enumerated.
    pub p_permutation: Array1<f64>,
    /// Family-wise error over all features, by the null distribution of the maximum statistic.
    pub p_fwer: Array1<f64>,
    pub components: Vec<Component>,
    pub n_rearrangements: usize,
    pub exhaustive: bool,
}

impl PermutationTest {
    fn t(&self, y: ArrayView2<f64>) -> Array1<f64> {
        match &self.statistic {
            Statistic::Glm(glm) => glm.t(y),
            Statistic::Welch { in_a } => welch_t(y, in_a),
        }
    }

    /// Runs the test on `y` (observations × features). Results do not depend on the number of
    /// threads: each rearrangement draws from its own random stream.
    pub fn run(&self, y: ArrayView2<f64>) -> Result<TestResult, MethodError> {
        let n = y.nrows();
        let expected = match &self.statistic {
            Statistic::Glm(glm) => glm.n_observations(),
            Statistic::Welch { in_a } => in_a.len(),
        };
        if n != expected {
            return Err(MethodError::ShapeMismatch(format!(
                "{n} observations but the design has {expected}"
            )));
        }
        if self.n_permutations == 0 {
            return Err(MethodError::invalid("n_permutations", "must be at least 1"));
        }
        if let (Statistic::Welch { .. }, Exchange::SignFlip) = (&self.statistic, self.exchange) {
            return Err(MethodError::invalid(
                "exchange",
                "Welch's t compares groups; use permutation",
            ));
        }
        let base = match &self.statistic {
            Statistic::Glm(glm) => glm.nuisance_residuals(y),
            Statistic::Welch { .. } => y.to_owned(),
        };
        let t = self.t(y);
        let observed: Vec<f64> = t.iter().map(|&v| self.tail.stat(v)).collect();
        let exhaustive = self.exchange == Exchange::SignFlip
            && n < 63
            && (1u64 << n) - 1 <= self.n_permutations as u64;
        let m = if exhaustive {
            (1usize << n) - 1
        } else {
            self.n_permutations
        };

        let rearranged = |r: usize| -> Array2<f64> {
            match (self.exchange, exhaustive) {
                (Exchange::SignFlip, true) => {
                    let flips = r + 1; // 1..2^n − 1; 0 would be the observed data
                    let mut out = base.clone();
                    for (i, mut row) in out.axis_iter_mut(Axis(0)).enumerate() {
                        if flips >> i & 1 == 1 {
                            row.mapv_inplace(|v| -v);
                        }
                    }
                    out
                }
                (Exchange::SignFlip, false) => {
                    let mut rng = ChaCha8Rng::seed_from_u64(self.seed);
                    rng.set_stream(r as u64);
                    let mut out = base.clone();
                    for mut row in out.axis_iter_mut(Axis(0)) {
                        if rand::Rng::random::<bool>(&mut rng) {
                            row.mapv_inplace(|v| -v);
                        }
                    }
                    out
                }
                (Exchange::Permute, _) => {
                    let mut rng = ChaCha8Rng::seed_from_u64(self.seed);
                    rng.set_stream(r as u64);
                    let mut order: Vec<usize> = (0..n).collect();
                    order.shuffle(&mut rng);
                    base.select(Axis(0), &order)
                }
            }
        };

        let features = observed.len();
        let null: Vec<(Vec<bool>, f64, usize)> = (0..m)
            .into_par_iter()
            .map(|r| {
                let stats: Vec<f64> = self
                    .t(rearranged(r).view())
                    .iter()
                    .map(|&v| self.tail.stat(v))
                    .collect();
                let exceeds = stats
                    .iter()
                    .zip(&observed)
                    .map(|(s, o)| at_least(*s, *o))
                    .collect();
                let max = stats
                    .iter()
                    .copied()
                    .filter(|v| !v.is_nan())
                    .fold(f64::NEG_INFINITY, f64::max);
                let biggest = self
                    .nbs
                    .as_ref()
                    .map_or(0, |(edges, threshold)| largest(&stats, edges, *threshold));
                (exceeds, max, biggest)
            })
            .collect();

        let ratio = |count: usize| (count + 1) as f64 / (m + 1) as f64;
        let nan_or = |o: f64, value: f64| if o.is_nan() { f64::NAN } else { value };
        let p_permutation = Array1::from_shape_fn(features, |f| {
            nan_or(
                observed[f],
                ratio(null.iter().filter(|(e, _, _)| e[f]).count()),
            )
        });
        let p_fwer = Array1::from_shape_fn(features, |f| {
            nan_or(
                observed[f],
                ratio(
                    null.iter()
                        .filter(|(_, max, _)| at_least(*max, observed[f]))
                        .count(),
                ),
            )
        });
        let found = match &self.nbs {
            Some((edges, threshold)) => components(&observed, edges, *threshold)
                .into_iter()
                .map(|(network, edges)| {
                    let size = edges.len();
                    Component {
                        network,
                        edges,
                        p: ratio(null.iter().filter(|(_, _, b)| *b >= size).count()),
                    }
                })
                .collect(),
            None => Vec::new(),
        };

        let df = match &self.statistic {
            Statistic::Glm(glm) => Array1::from_elem(features, glm.degrees_of_freedom()),
            Statistic::Welch { in_a } => welch_df(y, in_a),
        };
        let p_parametric = Array1::from_shape_fn(features, |f| {
            let (tf, d) = (t[f], df[f]);
            match self.tail {
                Tail::Greater => t_upper(tf, d),
                Tail::Less => t_upper(-tf, d),
                Tail::Both => (2.0 * t_upper(tf.abs(), d)).min(1.0),
            }
        });
        Ok(TestResult {
            t,
            df,
            p_parametric,
            p_permutation,
            p_fwer,
            components: found,
            n_rearrangements: m,
            exhaustive,
        })
    }
}
