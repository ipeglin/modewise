//! Group statistics on feature tables: GLM t and F contrasts (v and G with variance groups) with
//! Freedman–Lane permutation or sign flipping, exchangeability blocks, FWER by the maximum
//! statistic, FDR, and the network-based statistic (docs/statistics.md).

mod dist;
mod fdr;
mod glm;
mod nbs;
mod permutation;
mod rearrange;

pub use dist::{f_upper, incomplete_beta, t_upper};
pub use fdr::{Fdr, adjust};
pub use glm::Glm;
pub use nbs::{Component, Edge};
pub use permutation::{Exchange, PermutationTest, Tail, TestResult};
pub use rearrange::Blocks;

/// Components of suprathreshold edges, exposed for reference tests.
#[doc(hidden)]
pub fn components_for_test(
    stat: &[f64],
    edges: &[Option<Edge>],
    threshold: f64,
) -> Vec<(usize, Vec<usize>)> {
    nbs::components(stat, edges, threshold)
}
