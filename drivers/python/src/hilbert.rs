use modewise_hilbert::{Padding, Weight, hilbert_spectrum as spectrum, instantaneous};
use modewise_spectral::{FrequencyGrid, GridScale};
use pyo3::prelude::*;

use crate::convert::value_error;
use crate::results::{Decomposition, Instantaneous, TimeFrequencyMap};

/// Instantaneous amplitude and frequency of every mode (docs/time-frequency.md).
#[pyfunction]
#[pyo3(signature = (modes, *, padding = "mirror"))]
pub fn hilbert(py: Python<'_>, modes: &Decomposition, padding: &str) -> PyResult<Instantaneous> {
    let padding = match padding {
        "mirror" => Padding::Mirror,
        "none" => Padding::None,
        other => {
            return Err(value_error(format!(
                "padding must be 'mirror' or 'none', got {other:?}"
            )));
        }
    };
    let domain = modes.to_domain(py)?;
    let attributes = py
        .detach(|| instantaneous(&domain, padding))
        .map_err(value_error)?;
    Ok(Instantaneous::from_domain(py, &attributes))
}

/// Hilbert spectrum: power (or amplitude) binned by instantaneous frequency, summed over modes.
#[pyfunction]
#[pyo3(signature = (attributes, f_min, f_max, n_bins, *, scale = "linear", weight = "power"))]
pub fn hilbert_spectrum(
    py: Python<'_>,
    attributes: &Instantaneous,
    f_min: f64,
    f_max: f64,
    n_bins: usize,
    scale: &str,
    weight: &str,
) -> PyResult<TimeFrequencyMap> {
    let scale = match scale {
        "linear" => GridScale::Linear,
        "log" => GridScale::Log,
        other => {
            return Err(value_error(format!(
                "scale must be 'linear' or 'log', got {other:?}"
            )));
        }
    };
    let weight = match weight {
        "power" => Weight::Power,
        "amplitude" => Weight::Amplitude,
        other => {
            return Err(value_error(format!(
                "weight must be 'power' or 'amplitude', got {other:?}"
            )));
        }
    };
    let grid = FrequencyGrid::new(f_min, f_max, n_bins, scale).map_err(value_error)?;
    let domain = attributes.to_domain(py)?;
    let result = py
        .detach(|| spectrum(&domain, &grid, weight))
        .map_err(value_error)?;
    Ok(TimeFrequencyMap::from_domain(
        py,
        &result.map,
        "hilbert_spectrum",
    ))
}

/// Morlet continuous wavelet transform (Torrence & Compo 1998) of a channels × samples array.
/// Frequencies are the geometric centres of `n_bins` log-spaced bins over [f_min, f_max].
#[pyfunction]
#[pyo3(signature = (signal, sampling_rate, f_min, f_max, n_bins, *, omega0 = 6.0, rectify = false, labels = None))]
#[allow(clippy::too_many_arguments)]
pub fn cwt(
    py: Python<'_>,
    signal: numpy::PyArrayLike2<'_, f64, numpy::AllowTypeChange>,
    sampling_rate: f64,
    f_min: f64,
    f_max: f64,
    n_bins: usize,
    omega0: f64,
    rectify: bool,
    labels: Option<Vec<String>>,
) -> PyResult<TimeFrequencyMap> {
    use modewise_spectral::TimeFrequencyTransform;
    let series = crate::convert::to_series(signal, sampling_rate, labels)?;
    let grid = FrequencyGrid::new(f_min, f_max, n_bins, GridScale::Log).map_err(value_error)?;
    let method = modewise_cwt::Cwt::new(grid, omega0, rectify).map_err(value_error)?;
    let map = py
        .detach(|| method.transform(&series))
        .map_err(value_error)?;
    Ok(TimeFrequencyMap::from_domain(py, &map, "cwt"))
}
