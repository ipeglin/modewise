use std::fmt::Display;

use modewise_signals::{MultichannelSeries, SamplingRate};
use ndarray::Array2;
use numpy::{AllowTypeChange, PyArrayLike2};
use pyo3::PyErr;
use pyo3::exceptions::PyValueError;

pub(crate) fn value_error(err: impl Display) -> PyErr {
    PyValueError::new_err(err.to_string())
}

/// Copies the array so the computation can run without holding the GIL.
pub(crate) fn to_series(
    signal: PyArrayLike2<'_, f64, AllowTypeChange>,
    sampling_rate: f64,
    labels: Option<Vec<String>>,
) -> Result<MultichannelSeries, PyErr> {
    series_from(signal.as_array().to_owned(), sampling_rate, labels)
}

pub(crate) fn series_from(
    data: Array2<f64>,
    sampling_rate: f64,
    labels: Option<Vec<String>>,
) -> Result<MultichannelSeries, PyErr> {
    let labels = labels.unwrap_or_else(|| (0..data.nrows()).map(|i| format!("ch{i}")).collect());
    let fs = SamplingRate::new(sampling_rate).map_err(value_error)?;
    MultichannelSeries::new(data, fs, labels).map_err(value_error)
}
