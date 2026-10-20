//! Permutation inference with family-wise error by the maximum statistic.

use modewise_method::MethodError;
use ndarray::{Array1, Array2, ArrayView2, Axis};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

use crate::dist::{f_upper, t_upper};
use crate::glm::Glm;
use crate::nbs::{Component, Edge, components, largest};
use crate::rearrange::{Blocks, Rearrangements};

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

#[derive(Debug, Clone)]
pub struct PermutationTest {
    /// The model and contrast; nuisance regressors are handled by Freedman–Lane.
    pub glm: Glm,
    pub exchange: Exchange,
    pub blocks: Option<Blocks>,
    pub tail: Tail,
    pub n_permutations: usize,
    pub seed: u64,
    /// Network-based statistic over edge features, at this primary threshold of the tailed
    /// statistic.
    pub nbs: Option<(Vec<Option<Edge>>, f64)>,
}

/// Per-feature results.
#[derive(Debug, Clone, PartialEq)]
pub struct TestResult {
    /// "t", "F", "v" or "G".
    pub statistic: &'static str,
    /// Rows of the contrast: the numerator degrees of freedom of F and G.
    pub rank: usize,
    /// The statistic per feature.
    pub t: Array1<f64>,
    /// Denominator degrees of freedom.
    pub df: Array1<f64>,
    /// From Student's t for t and v, from F for F and G.
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
    /// Runs the test on `y` (observations × features). Results do not depend on the number of
    /// threads: each rearrangement draws from its own random stream.
    pub fn run(&self, y: ArrayView2<f64>) -> Result<TestResult, MethodError> {
        let n = y.nrows();
        let expected = self.glm.n_observations();
        if n != expected {
            return Err(MethodError::ShapeMismatch(format!(
                "{n} observations but the design has {expected}"
            )));
        }
        if self.n_permutations == 0 {
            return Err(MethodError::invalid("n_permutations", "must be at least 1"));
        }
        let rank = self.glm.rank();
        if rank > 1 && self.tail != Tail::Both {
            return Err(MethodError::invalid(
                "tail",
                "an F test has no direction; use both",
            ));
        }
        let rearrangements = Rearrangements::new(n, self.exchange, self.blocks.as_ref())?;
        let base = self.glm.nuisance_residuals(y);
        let t = self.glm.statistic(y);
        let observed: Vec<f64> = t.iter().map(|&v| self.tail.stat(v)).collect();
        let exhaustive = rearrangements
            .count()
            .is_some_and(|c| c - 1 <= self.n_permutations as u128);
        let m = if exhaustive {
            rearrangements.count().map_or(0, |c| c as usize - 1)
        } else {
            self.n_permutations
        };

        let rearranged = |r: usize| -> Array2<f64> {
            let order = if exhaustive {
                rearrangements.enumerated(r as u128 + 1) // 0 would be the observed data
            } else {
                let mut rng = ChaCha8Rng::seed_from_u64(self.seed);
                rng.set_stream(r as u64);
                rearrangements.random(&mut rng)
            };
            let mut out = base.select(Axis(0), &order.source);
            for (mut row, flip) in out.axis_iter_mut(Axis(0)).zip(order.flip) {
                if flip {
                    row.mapv_inplace(|v| -v);
                }
            }
            out
        };

        let features = observed.len();
        let null: Vec<(Vec<bool>, f64, usize)> = (0..m)
            .into_par_iter()
            .map(|r| {
                let stats: Vec<f64> = self
                    .glm
                    .statistic(rearranged(r).view())
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

        let df = self.glm.degrees_of_freedom(y);
        let p_parametric = Array1::from_shape_fn(features, |f| {
            let (tf, d) = (t[f], df[f]);
            match self.tail {
                _ if rank > 1 => f_upper(tf, rank as f64, d),
                Tail::Greater => t_upper(tf, d),
                Tail::Less => t_upper(-tf, d),
                Tail::Both => (2.0 * t_upper(tf.abs(), d)).min(1.0),
            }
        });
        Ok(TestResult {
            statistic: self.glm.name(),
            rank,
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
