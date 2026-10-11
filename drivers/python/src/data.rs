use std::collections::BTreeMap;
use std::path::PathBuf;

use modewise_dataset::{DatasetError, DatasetSpec};
use modewise_pipeline::InputSource;
use pyo3::exceptions::{PyKeyError, PyOSError, PyValueError};
use pyo3::prelude::*;

fn dataset_error(e: DatasetError) -> PyErr {
    match e {
        DatasetError::Io { .. } => PyOSError::new_err(e.to_string()),
        _ => PyValueError::new_err(e.to_string()),
    }
}

/// Recordings found on disk by file patterns, ready to load or run pipelines over.
#[pyclass(frozen, module = "modewise")]
pub struct Dataset {
    inner: modewise_dataset::Dataset,
}

impl Dataset {
    pub(crate) fn inner(&self) -> &modewise_dataset::Dataset {
        &self.inner
    }
}

#[pymethods]
impl Dataset {
    /// Reads a dataset spec. A relative `root` is resolved against the file's folder.
    #[staticmethod]
    fn from_file(path: PathBuf) -> PyResult<Self> {
        let inner = modewise_dataset::Dataset::from_file(&path).map_err(dataset_error)?;
        Ok(Self { inner })
    }

    /// Same as `from_file`, from text. A relative `root` is resolved against `base` (default: the current folder).
    #[staticmethod]
    #[pyo3(signature = (text, base = None))]
    fn from_toml(text: &str, base: Option<PathBuf>) -> PyResult<Self> {
        let spec = DatasetSpec::from_toml(text).map_err(dataset_error)?;
        let base = base.unwrap_or_else(|| PathBuf::from("."));
        let inner = modewise_dataset::Dataset::discover(spec, &base).map_err(dataset_error)?;
        Ok(Self { inner })
    }

    #[getter]
    fn recordings(&self) -> Vec<String> {
        self.inner.recording_ids()
    }

    #[getter]
    fn inputs(&self) -> Vec<String> {
        self.inner.input_names()
    }

    /// Entity values of one recording, for example {"subject": "01", "task": "rest"}.
    fn entities(&self, recording: &str) -> PyResult<BTreeMap<String, String>> {
        self.inner
            .recording(recording)
            .map(|r| r.entities.iter().cloned().collect())
            .ok_or_else(|| PyKeyError::new_err(recording.to_string()))
    }

    /// Reads every input of one recording: Series for signals, Events for events files.
    fn load(&self, py: Python<'_>, recording: &str) -> PyResult<BTreeMap<String, Py<PyAny>>> {
        if self.inner.recording(recording).is_none() {
            return Err(PyKeyError::new_err(recording.to_string()));
        }
        let inputs = py
            .detach(|| self.inner.load(recording))
            .map_err(PyValueError::new_err)?;
        inputs
            .iter()
            .map(|(k, v)| Ok((k.clone(), crate::results::artifact_to_py(py, v, "")?)))
            .collect()
    }

    /// Participant information from the spec's `[participants]` section, or None.
    #[getter]
    fn participants(&self) -> Option<crate::cohort::Participants> {
        self.inner()
            .participants()
            .map(|p| crate::cohort::Participants { inner: p.clone() })
    }

    fn __len__(&self) -> usize {
        self.inner.recordings().len()
    }

    fn __repr__(&self) -> String {
        format!(
            "Dataset(recordings={}, inputs={:?})",
            self.inner.recordings().len(),
            self.inner.input_names()
        )
    }
}
