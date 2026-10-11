//! Group statistics on feature tables: GLM t with Freedman–Lane permutation, Welch's t, sign
//! flipping for paired designs, FWER by the maximum statistic, FDR, and the network-based
//! statistic (docs/statistics.md).

mod dist;
mod fdr;
mod glm;
mod nbs;
mod permutation;

pub use dist::{incomplete_beta, t_upper};
pub use fdr::{Fdr, adjust};
pub use glm::{Glm, welch_df, welch_t};
pub use nbs::{Component, Edge};
pub use permutation::{Exchange, PermutationTest, Statistic, Tail, TestResult};

/// Components of suprathreshold edges, exposed for reference tests.
#[doc(hidden)]
pub fn components_for_test(
    stat: &[f64],
    edges: &[Option<Edge>],
    threshold: f64,
) -> Vec<(usize, Vec<usize>)> {
    nbs::components(stat, edges, threshold)
}
