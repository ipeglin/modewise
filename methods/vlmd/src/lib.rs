//! Variational Latent Mode Decomposition.
//!
//! Morante and Rehman (2025), "Latent Mode Decomposition", arXiv:2505.17797.
//! Built from the paper's equations; where the paper and its reference code
//! (github.com/Dmocrito/vlmd) disagree, `docs/vlmd.md` records which one is followed and why.

mod decomposer;
mod lasso;
mod params;

pub use decomposer::{Vlmd, VlmdOutput};
pub use lasso::lasso;
pub use params::VlmdParams;
