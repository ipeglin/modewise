//! Task GLM: event regressors convolved with the canonical haemodynamic response, fitted to
//! every channel with drift and confound regressors, by OLS or with AR(1) prewhitening
//! (docs/glm.md).

mod fit;
mod hrf;

pub use fit::{Fit, NoiseModel, fit};
pub use hrf::{HRF_LENGTH_S, canonical_hrf, canonical_hrf_integral, task_regressors};
