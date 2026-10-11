use modewise_signals::MultichannelSeries;
use numpy::{
    AllowTypeChange, IntoPyArray, PyArray1, PyArray2, PyArray3, PyArrayLike2, PyArrayMethods,
};
use pyo3::prelude::*;

use crate::convert::{series_from, to_series};

/// Signals laid out as channels × samples.
#[pyclass(frozen, module = "modewise")]
pub struct Series {
    #[pyo3(get)]
    data: Py<PyArray2<f64>>,
    #[pyo3(get)]
    sampling_rate: f64,
    #[pyo3(get)]
    labels: Vec<String>,
}

#[pymethods]
impl Series {
    #[new]
    #[pyo3(signature = (data, sampling_rate, labels = None))]
    fn new(
        py: Python<'_>,
        data: PyArrayLike2<'_, f64, AllowTypeChange>,
        sampling_rate: f64,
        labels: Option<Vec<String>>,
    ) -> PyResult<Self> {
        Ok(Self::from_domain(
            py,
            &to_series(data, sampling_rate, labels)?,
        ))
    }

    fn __repr__(&self) -> String {
        format!(
            "Series(channels={}, sampling_rate={})",
            self.labels.len(),
            self.sampling_rate
        )
    }
}

impl Series {
    pub(crate) fn to_domain(&self, py: Python<'_>) -> PyResult<MultichannelSeries> {
        let data = self.data.bind(py).readonly().as_array().to_owned();
        series_from(data, self.sampling_rate, Some(self.labels.clone()))
    }

    pub(crate) fn from_domain(py: Python<'_>, s: &MultichannelSeries) -> Self {
        Self {
            data: s.data().to_owned().into_pyarray(py).unbind(),
            sampling_rate: s.sampling_rate().hz(),
            labels: s.labels().to_vec(),
        }
    }
}

/// Result of a mode decomposition.
#[pyclass(frozen, module = "modewise")]
pub struct Decomposition {
    /// Modes as mode × channel × sample, sorted by centre frequency.
    #[pyo3(get)]
    modes: Py<PyArray3<f64>>,
    #[pyo3(get)]
    centre_frequencies_hz: Py<PyArray1<f64>>,
    /// Mode × channel, for methods that decompose each channel on its own (NaN: no such mode).
    #[pyo3(get)]
    channel_centre_frequencies_hz: Option<Py<PyArray2<f64>>>,
    /// True when mode k means the same oscillation in every channel.
    #[pyo3(get)]
    aligned_across_channels: bool,
    #[pyo3(get)]
    labels: Vec<String>,
    #[pyo3(get)]
    sampling_rate: f64,
    #[pyo3(get)]
    method: String,
    #[pyo3(get)]
    n_modes: usize,
}

#[pymethods]
impl Decomposition {
    /// Modes computed elsewhere (mode × channel × sample), so modewise can work on them. Give
    /// `channel_centre_frequencies_hz` (mode × channel) instead of `centre_frequencies_hz` for
    /// modes found per channel; each mode's frequency is then the channel median.
    #[new]
    #[pyo3(signature = (modes, centre_frequencies_hz, sampling_rate, labels = None, method = "external", *, channel_centre_frequencies_hz = None))]
    fn new(
        py: Python<'_>,
        modes: numpy::PyArrayLike3<'_, f64, AllowTypeChange>,
        centre_frequencies_hz: Option<Vec<f64>>,
        sampling_rate: f64,
        labels: Option<Vec<String>>,
        method: &str,
        channel_centre_frequencies_hz: Option<numpy::PyArrayLike2<'_, f64, AllowTypeChange>>,
    ) -> PyResult<Self> {
        use modewise_decomposition::Decomposition as D;
        let modes = modes.as_array().to_owned();
        let labels =
            labels.unwrap_or_else(|| (0..modes.dim().1).map(|c| format!("ch{c}")).collect());
        let fs = modewise_signals::SamplingRate::new(sampling_rate)
            .map_err(crate::convert::value_error)?;
        let d = match (centre_frequencies_hz, channel_centre_frequencies_hz) {
            (Some(centres), None) => D::new(modes, centres, fs, labels, true),
            (None, Some(per_channel)) => {
                D::per_channel(modes, per_channel.as_array().to_owned(), fs, labels)
            }
            _ => {
                return Err(crate::convert::value_error(
                    "give exactly one of centre_frequencies_hz or channel_centre_frequencies_hz",
                ));
            }
        }
        .map_err(crate::convert::value_error)?;
        Ok(Self::from_domain(py, &d, method))
    }

    fn __repr__(&self) -> String {
        format!(
            "Decomposition(method={:?}, modes={}, channels={})",
            self.method,
            self.n_modes,
            self.labels.len()
        )
    }
}

impl Decomposition {
    pub(crate) fn to_domain(
        &self,
        py: Python<'_>,
    ) -> PyResult<modewise_decomposition::Decomposition> {
        let fs = modewise_signals::SamplingRate::new(self.sampling_rate)
            .map_err(crate::convert::value_error)?;
        let modes = self.modes.bind(py).readonly().as_array().to_owned();
        match &self.channel_centre_frequencies_hz {
            Some(per_channel) => modewise_decomposition::Decomposition::per_channel(
                modes,
                per_channel.bind(py).readonly().as_array().to_owned(),
                fs,
                self.labels.clone(),
            ),
            None => modewise_decomposition::Decomposition::new(
                modes,
                self.centre_frequencies_hz
                    .bind(py)
                    .readonly()
                    .as_array()
                    .to_vec(),
                fs,
                self.labels.clone(),
                self.aligned_across_channels,
            ),
        }
        .map_err(crate::convert::value_error)
    }

    pub(crate) fn from_domain(
        py: Python<'_>,
        d: &modewise_decomposition::Decomposition,
        method: &str,
    ) -> Self {
        Self {
            modes: d.modes().to_owned().into_pyarray(py).unbind(),
            centre_frequencies_hz: d.centre_frequencies_hz().to_vec().into_pyarray(py).unbind(),
            channel_centre_frequencies_hz: d
                .channel_centre_frequencies_hz()
                .map(|f| f.to_owned().into_pyarray(py).unbind()),
            aligned_across_channels: d.aligned_across_channels(),
            labels: d.labels().to_vec(),
            sampling_rate: d.sampling_rate().hz(),
            method: method.to_string(),
            n_modes: d.n_modes(),
        }
    }
}

/// Time-frequency power of a signal.
#[pyclass(frozen, module = "modewise")]
pub struct TimeFrequencyMap {
    /// Power as channel × frequency × time.
    #[pyo3(get)]
    power: Py<PyArray3<f64>>,
    #[pyo3(get)]
    frequencies_hz: Py<PyArray1<f64>>,
    #[pyo3(get)]
    times_s: Py<PyArray1<f64>>,
    #[pyo3(get)]
    labels: Vec<String>,
    #[pyo3(get)]
    method: String,
    /// Samples per second when there is one time column per sample (CWT, Hilbert spectrum); None for STFT frames.
    #[pyo3(get)]
    sampling_rate: Option<f64>,
    /// Frequency × time: False where edge effects make values unreliable (cone of influence); None if not computed.
    #[pyo3(get)]
    reliable: Option<Py<PyArray2<bool>>>,
}

#[pymethods]
impl TimeFrequencyMap {
    fn __repr__(&self) -> String {
        format!(
            "TimeFrequencyMap(method={:?}, channels={})",
            self.method,
            self.labels.len()
        )
    }
}

impl TimeFrequencyMap {
    pub(crate) fn from_domain(
        py: Python<'_>,
        m: &modewise_spectral::TimeFrequencyMap,
        method: &str,
    ) -> Self {
        Self {
            power: m.power().to_owned().into_pyarray(py).unbind(),
            frequencies_hz: m.frequencies_hz().to_vec().into_pyarray(py).unbind(),
            times_s: m.times_s().to_vec().into_pyarray(py).unbind(),
            labels: m.labels().to_vec(),
            method: method.to_string(),
            sampling_rate: m.sampling_rate().map(|fs| fs.hz()),
            reliable: m.reliable().map(|r| r.to_owned().into_pyarray(py).unbind()),
        }
    }
}

/// Functional connectivity (undirected, symmetric), window × component × channel × channel.
#[pyclass(frozen, module = "modewise")]
pub struct FunctionalConnectivity {
    #[pyo3(get)]
    values: Py<numpy::PyArray4<f64>>,
    /// "pearson", "pearson_fisher_z" or "plv".
    #[pyo3(get)]
    measure: String,
    #[pyo3(get)]
    labels: Vec<String>,
    /// One per component when computed on a decomposition.
    #[pyo3(get)]
    centre_frequencies_hz: Option<Py<PyArray1<f64>>>,
    /// One per window when computed on epochs.
    #[pyo3(get)]
    window_labels: Option<Vec<String>>,
    #[pyo3(get)]
    window_onsets_s: Option<Py<PyArray1<f64>>>,
    /// Window × component: True where the window holds enough cycles of the component.
    #[pyo3(get)]
    resolvable: Option<Py<PyArray2<bool>>>,
}

#[pymethods]
impl FunctionalConnectivity {
    fn __repr__(&self) -> String {
        format!(
            "FunctionalConnectivity(measure={:?}, channels={})",
            self.measure,
            self.labels.len()
        )
    }
}

impl FunctionalConnectivity {
    pub(crate) fn from_domain(
        py: Python<'_>,
        c: &modewise_connectivity::FunctionalConnectivity,
    ) -> Self {
        let windows = c.windows();
        Self {
            values: c.values().to_owned().into_pyarray(py).unbind(),
            measure: c.measure().name().to_string(),
            labels: c.labels().to_vec(),
            centre_frequencies_hz: c
                .centre_frequencies_hz()
                .map(|hz| hz.to_vec().into_pyarray(py).unbind()),
            window_labels: windows.map(|w| w.iter().map(|w| w.label.clone()).collect()),
            window_onsets_s: windows.map(|w| {
                w.iter()
                    .map(|w| w.event_onset_s)
                    .collect::<Vec<_>>()
                    .into_pyarray(py)
                    .unbind()
            }),
            resolvable: c
                .resolvable()
                .map(|r| r.to_owned().into_pyarray(py).unbind()),
        }
    }
}

/// How strongly each latent component contributes to each channel (latent × channel).
#[pyclass(frozen, module = "modewise")]
pub struct Loadings {
    #[pyo3(get)]
    values: Py<PyArray2<f64>>,
    #[pyo3(get)]
    latent_labels: Vec<String>,
    #[pyo3(get)]
    channel_labels: Vec<String>,
}

#[pymethods]
impl Loadings {
    fn __repr__(&self) -> String {
        format!(
            "Loadings(latents={}, channels={})",
            self.latent_labels.len(),
            self.channel_labels.len()
        )
    }
}

impl Loadings {
    pub(crate) fn from_domain(py: Python<'_>, l: &modewise_decomposition::Loadings) -> Self {
        Self {
            values: l.values().to_owned().into_pyarray(py).unbind(),
            latent_labels: l.latent_labels().to_vec(),
            channel_labels: l.channel_labels().to_vec(),
        }
    }
}

/// Instantaneous amplitude and frequency of modes, each mode × channel × sample.
#[pyclass(frozen, module = "modewise")]
pub struct Instantaneous {
    #[pyo3(get)]
    amplitude: Py<PyArray3<f64>>,
    #[pyo3(get)]
    frequency_hz: Py<PyArray3<f64>>,
    #[pyo3(get)]
    centre_frequencies_hz: Py<PyArray1<f64>>,
    #[pyo3(get)]
    labels: Vec<String>,
    #[pyo3(get)]
    sampling_rate: f64,
}

#[pymethods]
impl Instantaneous {
    fn __repr__(&self) -> String {
        format!(
            "Instantaneous(channels={}, sampling_rate={})",
            self.labels.len(),
            self.sampling_rate
        )
    }
}

impl Instantaneous {
    pub(crate) fn from_domain(
        py: Python<'_>,
        m: &modewise_decomposition::InstantaneousModes,
    ) -> Self {
        Self {
            amplitude: m.amplitude().to_owned().into_pyarray(py).unbind(),
            frequency_hz: m.frequency_hz().to_owned().into_pyarray(py).unbind(),
            centre_frequencies_hz: m.centre_frequencies_hz().to_vec().into_pyarray(py).unbind(),
            labels: m.labels().to_vec(),
            sampling_rate: m.sampling_rate().hz(),
        }
    }

    pub(crate) fn to_domain(
        &self,
        py: Python<'_>,
    ) -> PyResult<modewise_decomposition::InstantaneousModes> {
        let fs = modewise_signals::SamplingRate::new(self.sampling_rate)
            .map_err(crate::convert::value_error)?;
        modewise_decomposition::InstantaneousModes::new(
            self.amplitude.bind(py).readonly().as_array().to_owned(),
            self.frequency_hz.bind(py).readonly().as_array().to_owned(),
            self.centre_frequencies_hz
                .bind(py)
                .readonly()
                .as_array()
                .to_vec(),
            fs,
            self.labels.clone(),
        )
        .map_err(crate::convert::value_error)
    }
}

/// Converts a stored artifact into the matching Python result type.
pub(crate) fn artifact_to_py(
    py: Python<'_>,
    artifact: &modewise_pipeline::Artifact,
    method: &str,
) -> PyResult<Py<PyAny>> {
    use modewise_pipeline::Artifact;
    let object = match artifact {
        Artifact::Events(e) => {
            Bound::new(py, crate::events::Events { inner: e.clone() })?.into_any()
        }
        Artifact::Epochs(e) => {
            Bound::new(py, crate::events::Epochs { inner: e.clone() })?.into_any()
        }
        Artifact::Instantaneous(m) => Bound::new(py, Instantaneous::from_domain(py, m))?.into_any(),
        Artifact::Series(s) => Bound::new(py, Series::from_domain(py, s))?.into_any(),
        Artifact::Loadings(l) => Bound::new(py, Loadings::from_domain(py, l))?.into_any(),
        Artifact::Channels(t) => {
            Bound::new(py, crate::channels::ChannelTable { inner: t.clone() })?.into_any()
        }
        Artifact::FunctionalConnectivity(c) => {
            Bound::new(py, FunctionalConnectivity::from_domain(py, c))?.into_any()
        }
        Artifact::Decomposition(d) => {
            Bound::new(py, Decomposition::from_domain(py, d, method))?.into_any()
        }
        Artifact::TimeFrequency(m) => {
            Bound::new(py, TimeFrequencyMap::from_domain(py, m, method))?.into_any()
        }
    };
    Ok(object.unbind())
}
