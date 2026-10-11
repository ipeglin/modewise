use modewise_decomposition::ModeDecomposer;
use modewise_vlmd::{Vlmd, VlmdParams};
use numpy::{AllowTypeChange, PyArrayLike2};
use pyo3::prelude::*;

use crate::convert::{to_series, value_error};
use crate::results::{Decomposition, Loadings};

/// Variational Latent Mode Decomposition of a channels × samples array.
///
/// Returns (modes, latent_modes, loadings): channel modes (modes × channels × samples),
/// latent modes (modes × latents × samples) and the latent × channel loadings.
#[pyfunction]
#[pyo3(signature = (
    signal, sampling_rate, num_modes, *,
    num_latents = None, alpha = 100.0, reg_lambda = 0.01, reg_rho = 1.0, tau = 0.9,
    tolerance = 1e-3, max_iterations = 1000, labels = None
))]
#[allow(clippy::too_many_arguments)]
pub fn vlmd(
    py: Python<'_>,
    signal: PyArrayLike2<'_, f64, AllowTypeChange>,
    sampling_rate: f64,
    num_modes: usize,
    num_latents: Option<usize>,
    alpha: f64,
    reg_lambda: f64,
    reg_rho: f64,
    tau: f64,
    tolerance: f64,
    max_iterations: usize,
    labels: Option<Vec<String>>,
) -> PyResult<(Decomposition, Decomposition, Loadings)> {
    let series = to_series(signal, sampling_rate, labels)?;
    let params = VlmdParams {
        num_modes,
        num_latents,
        alpha,
        reg_lambda,
        reg_rho,
        tau,
        tolerance,
        max_iterations,
    };
    let method = Vlmd::new(params).map_err(value_error)?;
    let out = py
        .detach(|| method.decompose_latent(&series))
        .map_err(value_error)?;
    let name = method.descriptor().name;
    Ok((
        Decomposition::from_domain(py, &out.modes, name),
        Decomposition::from_domain(py, &out.latent_modes, name),
        Loadings::from_domain(py, &out.loadings),
    ))
}
