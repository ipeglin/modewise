use std::path::PathBuf;

use modewise_epochs::{EpochSource, Event};
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::exceptions::{PyIndexError, PyOSError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::convert::value_error;
use crate::results::{Decomposition, Series};

/// Events with onsets and durations in seconds from the first sample (as in BIDS events.tsv).
#[pyclass(frozen, module = "modewise")]
pub struct Events {
    pub(crate) inner: modewise_epochs::Events,
}

#[pymethods]
impl Events {
    #[new]
    fn new(onsets_s: Vec<f64>, durations_s: Vec<f64>, labels: Vec<String>) -> PyResult<Self> {
        if onsets_s.len() != durations_s.len() || onsets_s.len() != labels.len() {
            return Err(PyValueError::new_err(format!(
                "onsets_s, durations_s and labels must have the same length, got {}, {} and {}",
                onsets_s.len(),
                durations_s.len(),
                labels.len()
            )));
        }
        let events = onsets_s
            .into_iter()
            .zip(durations_s)
            .zip(labels)
            .map(|((onset_s, duration_s), label)| Event {
                onset_s,
                duration_s,
                label,
            })
            .collect();
        Ok(Self {
            inner: modewise_epochs::Events::new(events).map_err(value_error)?,
        })
    }

    #[getter]
    fn onsets_s<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner
            .as_slice()
            .iter()
            .map(|e| e.onset_s)
            .collect::<Vec<_>>()
            .into_pyarray(py)
    }

    #[getter]
    fn durations_s<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner
            .as_slice()
            .iter()
            .map(|e| e.duration_s)
            .collect::<Vec<_>>()
            .into_pyarray(py)
    }

    #[getter]
    fn labels(&self) -> Vec<String> {
        self.inner
            .as_slice()
            .iter()
            .map(|e| e.label.clone())
            .collect()
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        format!("Events({} events)", self.inner.len())
    }
}

/// Reads a BIDS events.tsv file (columns onset, duration and a label column).
#[pyfunction]
#[pyo3(signature = (path, *, label_column = "trial_type"))]
pub fn read_events_tsv(path: PathBuf, label_column: &str) -> PyResult<Events> {
    if !path.is_file() {
        return Err(PyOSError::new_err(format!(
            "no events file at {}",
            path.display()
        )));
    }
    let inner =
        modewise_dataset::read_events_tsv(&path, label_column).map_err(PyValueError::new_err)?;
    Ok(Events { inner })
}

/// Epochs cut from a full-run series or decomposition. Each epoch is a view into the source.
#[pyclass(frozen, module = "modewise")]
pub struct Epochs {
    pub(crate) inner: modewise_epochs::Epochs,
}

#[pymethods]
impl Epochs {
    /// Sample index where each epoch starts (0-based, inclusive).
    #[getter]
    fn start_sample<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u64>> {
        self.inner
            .windows()
            .iter()
            .map(|w| w.start as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py)
    }

    /// Sample index where each epoch ends (exclusive).
    #[getter]
    fn end_sample<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u64>> {
        self.inner
            .windows()
            .iter()
            .map(|w| w.end as u64)
            .collect::<Vec<_>>()
            .into_pyarray(py)
    }

    #[getter]
    fn event_onsets_s<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner
            .windows()
            .iter()
            .map(|w| w.event_onset_s)
            .collect::<Vec<_>>()
            .into_pyarray(py)
    }

    #[getter]
    fn labels(&self) -> Vec<String> {
        self.inner
            .windows()
            .iter()
            .map(|w| w.label.clone())
            .collect()
    }

    #[getter]
    fn durations_s<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        (0..self.inner.windows().len())
            .map(|i| self.inner.duration_s(i))
            .collect::<Vec<_>>()
            .into_pyarray(py)
    }

    /// Epochs × modes: True where the mode's period fits `min_cycles` times. None for series sources.
    #[getter]
    fn resolvable<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<bool>>> {
        self.inner.resolvable().map(|r| r.into_pyarray(py))
    }

    /// Events that produced no epoch, with the reason.
    #[getter]
    fn dropped<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyDict>>> {
        self.inner
            .dropped()
            .iter()
            .map(|d| {
                let row = PyDict::new(py);
                row.set_item("label", &d.label)?;
                row.set_item("event_onset_s", d.event_onset_s)?;
                row.set_item("reason", &d.reason)?;
                Ok(row)
            })
            .collect()
    }

    #[getter]
    fn lag_s(&self) -> f64 {
        self.inner.params().lag_s
    }

    #[getter]
    fn min_cycles(&self) -> f64 {
        self.inner.params().min_cycles
    }

    /// The full-run result the epochs were cut from.
    #[getter]
    fn source(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let object = match self.inner.source() {
            EpochSource::Series(s) => Bound::new(py, Series::from_domain(py, s))?.into_any(),
            EpochSource::Decomposition(d) => {
                Bound::new(py, Decomposition::from_domain(py, d, ""))?.into_any()
            }
            EpochSource::Instantaneous(m) => {
                Bound::new(py, crate::results::Instantaneous::from_domain(py, m))?.into_any()
            }
            EpochSource::TimeFrequency(m) => {
                Bound::new(py, crate::results::TimeFrequencyMap::from_domain(py, m, ""))?.into_any()
            }
        };
        Ok(object.unbind())
    }

    /// Data of epoch `i`: channels × samples for a series source, modes × channels × samples for a decomposition.
    fn data<'py>(&self, py: Python<'py>, i: usize) -> PyResult<Bound<'py, PyAny>> {
        if i >= self.inner.windows().len() {
            return Err(PyIndexError::new_err(format!(
                "epoch {i} does not exist; there are {}",
                self.inner.windows().len()
            )));
        }
        if let Some(view) = self.inner.series(i) {
            return Ok(view.to_owned().into_pyarray(py).into_any());
        }
        if let Some(view) = self.inner.modes(i) {
            return Ok(view.to_owned().into_pyarray(py).into_any());
        }
        if let Some(view) = self.inner.time_frequency(i) {
            return Ok(view.to_owned().into_pyarray(py).into_any());
        }
        if let Some((amplitude, frequency)) = self.inner.instantaneous(i) {
            let both = PyDict::new(py);
            both.set_item("amplitude", amplitude.to_owned().into_pyarray(py))?;
            both.set_item("frequency_hz", frequency.to_owned().into_pyarray(py))?;
            return Ok(both.into_any());
        }
        Err(PyValueError::new_err("epoch source holds no data"))
    }

    fn __len__(&self) -> usize {
        self.inner.windows().len()
    }

    fn __repr__(&self) -> String {
        format!(
            "Epochs({} kept, {} dropped, lag_s={})",
            self.inner.windows().len(),
            self.inner.dropped().len(),
            self.inner.params().lag_s
        )
    }
}
