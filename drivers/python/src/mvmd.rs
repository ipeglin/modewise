use modewise_decomposition::ModeDecomposer;
use modewise_mvmd::{FrequencyInit, Mvmd, MvmdParams, NaMvmd, NoiseParams, Vmd};
use numpy::{AllowTypeChange, PyArrayLike2};
use pyo3::prelude::*;

use crate::convert::{to_series, value_error};
use crate::results::Decomposition;

fn parse_init(init: &str) -> PyResult<FrequencyInit> {
    match init {
        "zero" => Ok(FrequencyInit::Zero),
        "linear" => Ok(FrequencyInit::Linear),
        "exponential" => Ok(FrequencyInit::Exponential),
        "uniform" => Ok(FrequencyInit::Uniform),
        other => Err(value_error(format!(
            "init must be 'zero', 'linear', 'exponential' or 'uniform', got {other:?}"
        ))),
    }
}

/// Multivariate Variational Mode Decomposition of a channels × samples array.
#[pyfunction]
#[pyo3(signature = (
    signal, sampling_rate, num_modes, alpha, *,
    init = "zero", tolerance = 1e-3, tau = 1e-2, max_iterations = 500, labels = None
))]
#[allow(clippy::too_many_arguments)]
pub fn mvmd(
    py: Python<'_>,
    signal: PyArrayLike2<'_, f64, AllowTypeChange>,
    sampling_rate: f64,
    num_modes: usize,
    alpha: f64,
    init: &str,
    tolerance: f64,
    tau: f64,
    max_iterations: usize,
    labels: Option<Vec<String>>,
) -> PyResult<Decomposition> {
    let series = to_series(signal, sampling_rate, labels)?;
    let params = MvmdParams {
        num_modes,
        alpha,
        init: parse_init(init)?,
        tolerance,
        tau,
        max_iterations,
    };
    let method = Mvmd::new(params).map_err(value_error)?;
    let result = py
        .detach(|| method.decompose(&series))
        .map_err(value_error)?;

    Ok(Decomposition::from_domain(
        py,
        &result,
        method.descriptor().name,
    ))
}

fn run(
    py: Python<'_>,
    method: &(dyn ModeDecomposer + Sync),
    series: &modewise_signals::MultichannelSeries,
) -> PyResult<Decomposition> {
    let result = py
        .detach(|| method.decompose(series))
        .map_err(value_error)?;
    Ok(Decomposition::from_domain(
        py,
        &result,
        method.descriptor().name,
    ))
}

/// Variational Mode Decomposition of each channel on its own.
#[pyfunction]
#[pyo3(signature = (
    signal, sampling_rate, num_modes, alpha, *,
    init = "zero", tolerance = 1e-3, tau = 1e-2, max_iterations = 500, labels = None
))]
#[allow(clippy::too_many_arguments)]
pub fn vmd(
    py: Python<'_>,
    signal: PyArrayLike2<'_, f64, AllowTypeChange>,
    sampling_rate: f64,
    num_modes: usize,
    alpha: f64,
    init: &str,
    tolerance: f64,
    tau: f64,
    max_iterations: usize,
    labels: Option<Vec<String>>,
) -> PyResult<Decomposition> {
    let series = to_series(signal, sampling_rate, labels)?;
    let params = MvmdParams {
        num_modes,
        alpha,
        init: parse_init(init)?,
        tolerance,
        tau,
        max_iterations,
    };
    run(py, &Vmd::new(params).map_err(value_error)?, &series)
}

/// Noise-assisted MVMD: white-noise channels appended to the unit-SD signal.
#[pyfunction]
#[pyo3(signature = (
    signal, sampling_rate, num_modes, alpha, *,
    noise_channels = 1, noise_sd = 0.8, seed = 0,
    init = "zero", tolerance = 1e-3, tau = 1e-2, max_iterations = 500, labels = None
))]
#[allow(clippy::too_many_arguments)]
pub fn na_mvmd(
    py: Python<'_>,
    signal: PyArrayLike2<'_, f64, AllowTypeChange>,
    sampling_rate: f64,
    num_modes: usize,
    alpha: f64,
    noise_channels: usize,
    noise_sd: f64,
    seed: u64,
    init: &str,
    tolerance: f64,
    tau: f64,
    max_iterations: usize,
    labels: Option<Vec<String>>,
) -> PyResult<Decomposition> {
    let series = to_series(signal, sampling_rate, labels)?;
    let params = MvmdParams {
        num_modes,
        alpha,
        init: parse_init(init)?,
        tolerance,
        tau,
        max_iterations,
    };
    let noise = NoiseParams {
        channels: noise_channels,
        sd: noise_sd,
        seed,
    };
    run(
        py,
        &NaMvmd::new(params, noise).map_err(value_error)?,
        &series,
    )
}
