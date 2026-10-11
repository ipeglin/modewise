use modewise_decomposition::ModeDecomposer;
use modewise_wavelet::{Boundary, ModwtMra};
use numpy::{AllowTypeChange, PyArrayLike2};
use pyo3::prelude::*;

use crate::convert::{to_series, value_error};
use crate::results::Decomposition;

/// MODWT multiresolution analysis of each channel.
#[pyfunction]
#[pyo3(signature = (signal, sampling_rate, *, wavelet = "sym4", levels = None, boundary = "periodic", labels = None))]
pub fn modwt_mra(
    py: Python<'_>,
    signal: PyArrayLike2<'_, f64, AllowTypeChange>,
    sampling_rate: f64,
    wavelet: &str,
    levels: Option<usize>,
    boundary: &str,
    labels: Option<Vec<String>>,
) -> PyResult<Decomposition> {
    let boundary = match boundary {
        "periodic" => Boundary::Periodic,
        "reflection" => Boundary::Reflection,
        other => {
            return Err(value_error(format!(
                "boundary must be 'periodic' or 'reflection', got {other:?}"
            )));
        }
    };
    let series = to_series(signal, sampling_rate, labels)?;
    let method = ModwtMra::new(wavelet, levels, boundary).map_err(value_error)?;
    let result = py
        .detach(|| method.decompose(&series))
        .map_err(value_error)?;
    Ok(Decomposition::from_domain(
        py,
        &result,
        method.descriptor().name,
    ))
}
