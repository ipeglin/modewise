use modewise_spectral::TimeFrequencyTransform;
use modewise_stft::Stft;
use numpy::{AllowTypeChange, PyArrayLike2};
use pyo3::prelude::*;

use crate::convert::{to_series, value_error};
use crate::results::TimeFrequencyMap;

/// Short-time Fourier transform with a periodic Hann window. Power is |X|².
#[pyfunction]
#[pyo3(signature = (signal, sampling_rate, window_len, hop, *, labels = None))]
pub fn stft(
    py: Python<'_>,
    signal: PyArrayLike2<'_, f64, AllowTypeChange>,
    sampling_rate: f64,
    window_len: usize,
    hop: usize,
    labels: Option<Vec<String>>,
) -> PyResult<TimeFrequencyMap> {
    let series = to_series(signal, sampling_rate, labels)?;
    let method = Stft::new(window_len, hop).map_err(value_error)?;
    let map = py
        .detach(|| method.transform(&series))
        .map_err(value_error)?;

    Ok(TimeFrequencyMap::from_domain(
        py,
        &map,
        method.descriptor().name,
    ))
}

/// Power spectral density, one frame per channel: a periodogram, or Welch's method with
/// `segment_s` (docs/pipelines.md, `spectrum`).
#[pyfunction]
#[pyo3(signature = (signal, sampling_rate, *, segment_s = None, overlap_s = None, window = "boxcar", detrend = "constant", labels = None))]
#[allow(clippy::too_many_arguments)]
pub fn spectrum(
    py: Python<'_>,
    signal: numpy::PyArrayLike2<'_, f64, numpy::AllowTypeChange>,
    sampling_rate: f64,
    segment_s: Option<f64>,
    overlap_s: Option<f64>,
    window: &str,
    detrend: &str,
    labels: Option<Vec<String>>,
) -> PyResult<crate::results::TimeFrequencyMap> {
    use modewise_stft::{Detrend, Welch, Window};
    let series = crate::convert::to_series(signal, sampling_rate, labels)?;
    let window = match window {
        "boxcar" => Window::Boxcar,
        "hann" => Window::Hann,
        other => {
            return Err(crate::convert::value_error(format!(
                "window must be 'boxcar' or 'hann', got {other:?}"
            )));
        }
    };
    let detrend = match detrend {
        "none" => Detrend::None,
        "constant" => Detrend::Constant,
        "linear" => Detrend::Linear,
        other => {
            return Err(crate::convert::value_error(format!(
                "detrend must be 'none', 'constant' or 'linear', got {other:?}"
            )));
        }
    };
    let samples = |s: f64| (s * sampling_rate).round() as usize;
    let welch = Welch {
        segment_len: segment_s.map(samples),
        overlap: overlap_s.map(samples),
        window,
        detrend,
    };
    let map = py
        .detach(|| welch.transform(&series))
        .map_err(crate::convert::value_error)?;
    Ok(crate::results::TimeFrequencyMap::from_domain(
        py, &map, "spectrum",
    ))
}
