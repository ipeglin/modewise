use modewise_decomposition::ModeDecomposer;
use modewise_emd::{Ceemdan, Eemd, Emd, Memd, Stop};
use numpy::{AllowTypeChange, PyArrayLike2};
use pyo3::prelude::*;

use crate::convert::{to_series, value_error};
use crate::results::Decomposition;

fn parse_stop(
    stop: &str,
    thresholds: Option<(f64, f64, f64)>,
    default: Stop,
    count: usize,
    siftings: usize,
) -> PyResult<Stop> {
    match stop {
        "rilling" => {
            Ok(thresholds.map_or(default, |(sd, sd2, tol)| Stop::Rilling { sd, sd2, tol }))
        }
        "s_number" => Ok(Stop::SNumber { count }),
        "fixed" => Ok(Stop::Fixed { siftings }),
        other => Err(value_error(format!(
            "stop must be 'rilling', 's_number' or 'fixed', got {other:?}"
        ))),
    }
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

/// Empirical mode decomposition of each channel on its own.
#[pyfunction]
#[pyo3(signature = (signal, sampling_rate, *, stop = "rilling", thresholds = None, count = 2, siftings = 10, max_imfs = None, labels = None))]
#[allow(clippy::too_many_arguments)]
pub fn emd(
    py: Python<'_>,
    signal: PyArrayLike2<'_, f64, AllowTypeChange>,
    sampling_rate: f64,
    stop: &str,
    thresholds: Option<(f64, f64, f64)>,
    count: usize,
    siftings: usize,
    max_imfs: Option<usize>,
    labels: Option<Vec<String>>,
) -> PyResult<Decomposition> {
    let series = to_series(signal, sampling_rate, labels)?;
    let stop = parse_stop(stop, thresholds, Stop::RILLING_EMD, count, siftings)?;
    run(py, &Emd::new(stop, max_imfs).map_err(value_error)?, &series)
}

/// Multivariate empirical mode decomposition.
#[pyfunction]
#[pyo3(signature = (signal, sampling_rate, *, num_directions = 64, stop = "rilling", thresholds = None, count = 2, siftings = 10, max_imfs = None, labels = None))]
#[allow(clippy::too_many_arguments)]
pub fn memd(
    py: Python<'_>,
    signal: PyArrayLike2<'_, f64, AllowTypeChange>,
    sampling_rate: f64,
    num_directions: usize,
    stop: &str,
    thresholds: Option<(f64, f64, f64)>,
    count: usize,
    siftings: usize,
    max_imfs: Option<usize>,
    labels: Option<Vec<String>>,
) -> PyResult<Decomposition> {
    let series = to_series(signal, sampling_rate, labels)?;
    let stop = parse_stop(stop, thresholds, Stop::RILLING_MEMD, count, siftings)?;
    run(
        py,
        &Memd::new(stop, max_imfs, num_directions).map_err(value_error)?,
        &series,
    )
}

/// Ensemble EMD (Wu and Huang) of each channel on its own.
#[pyfunction]
#[pyo3(signature = (signal, sampling_rate, *, trials = 100, noise_sd = 0.2, seed = 0, stop = "fixed", thresholds = None, count = 2, siftings = 10, max_imfs = None, labels = None))]
#[allow(clippy::too_many_arguments)]
pub fn eemd(
    py: Python<'_>,
    signal: PyArrayLike2<'_, f64, AllowTypeChange>,
    sampling_rate: f64,
    trials: usize,
    noise_sd: f64,
    seed: u64,
    stop: &str,
    thresholds: Option<(f64, f64, f64)>,
    count: usize,
    siftings: usize,
    max_imfs: Option<usize>,
    labels: Option<Vec<String>>,
) -> PyResult<Decomposition> {
    let series = to_series(signal, sampling_rate, labels)?;
    let stop = parse_stop(stop, thresholds, Stop::RILLING_EMD, count, siftings)?;
    let method = Eemd {
        stop,
        max_imfs,
        trials,
        noise_sd,
        seed,
    }
    .validated()
    .map_err(value_error)?;
    run(py, &method, &series)
}

/// CEEMDAN, improved form (Colominas et al.), of each channel on its own.
#[pyfunction]
#[pyo3(signature = (signal, sampling_rate, *, trials = 100, epsilon = 0.2, seed = 0, stop = "rilling", thresholds = None, count = 2, siftings = 10, max_imfs = None, labels = None))]
#[allow(clippy::too_many_arguments)]
pub fn ceemdan(
    py: Python<'_>,
    signal: PyArrayLike2<'_, f64, AllowTypeChange>,
    sampling_rate: f64,
    trials: usize,
    epsilon: f64,
    seed: u64,
    stop: &str,
    thresholds: Option<(f64, f64, f64)>,
    count: usize,
    siftings: usize,
    max_imfs: Option<usize>,
    labels: Option<Vec<String>>,
) -> PyResult<Decomposition> {
    let series = to_series(signal, sampling_rate, labels)?;
    let stop = parse_stop(stop, thresholds, Stop::RILLING_EMD, count, siftings)?;
    let method = Ceemdan {
        stop,
        max_imfs,
        trials,
        epsilon,
        seed,
    }
    .validated()
    .map_err(value_error)?;
    run(py, &method, &series)
}
