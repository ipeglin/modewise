use std::path::{Path, PathBuf};

use modewise_results::{ResultsDir, ResultsError};
use pyo3::exceptions::{PyKeyError, PyOSError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate::results::artifact_to_py;

pub(crate) fn results_error(e: ResultsError) -> PyErr {
    match e {
        ResultsError::Io { .. } => PyOSError::new_err(e.to_string()),
        ResultsError::MissingRun { .. } | ResultsError::MissingOutput { .. } => {
            PyKeyError::new_err(e.to_string())
        }
        ResultsError::InputMismatch { .. }
        | ResultsError::NotStored { .. }
        | ResultsError::Pipeline(_) => PyValueError::new_err(e.to_string()),
        ResultsError::Store(_) => PyRuntimeError::new_err(e.to_string()),
    }
}

/// A results folder written by `Pipeline.run_dataset` or `modewise run`.
#[pyclass(frozen, module = "modewise")]
pub struct Results {
    inner: ResultsDir,
}

impl Results {
    pub(crate) fn open_dir(path: &Path) -> PyResult<Self> {
        Ok(Self {
            inner: ResultsDir::open(path).map_err(results_error)?,
        })
    }

    pub(crate) fn inner(&self) -> &ResultsDir {
        &self.inner
    }
}

#[pymethods]
impl Results {
    /// Opens a results folder, creating it if needed.
    #[new]
    fn new(path: PathBuf) -> PyResult<Self> {
        Self::open_dir(&path)
    }

    #[getter]
    fn path(&self) -> PathBuf {
        self.inner.root().to_path_buf()
    }

    #[getter]
    fn pipelines(&self) -> Vec<String> {
        self.inner.pipelines()
    }

    fn recordings(&self, pipeline: &str) -> Vec<String> {
        self.inner.recordings(pipeline)
    }

    /// Provenance record of one recording's run, as a dict.
    fn manifest<'py>(
        &self,
        py: Python<'py>,
        pipeline: &str,
        recording: &str,
    ) -> PyResult<Bound<'py, PyAny>> {
        let manifest = self
            .inner
            .manifest(pipeline, recording)
            .map_err(results_error)?;
        py.import("json")?
            .call_method1("loads", (manifest.to_json(),))
    }

    /// A step output of one recording, by `step.output`, or by `step` when it has one output.
    pub(crate) fn output(
        &self,
        py: Python<'_>,
        pipeline: &str,
        recording: &str,
        reference: &str,
    ) -> PyResult<Py<PyAny>> {
        let artifact = py
            .detach(|| self.inner.output(pipeline, recording, reference))
            .map_err(results_error)?;
        let step = reference.split('.').next().unwrap_or(reference);
        let manifest = self
            .inner
            .manifest(pipeline, recording)
            .map_err(results_error)?;
        let op = manifest
            .steps
            .iter()
            .find(|s| s.id == step)
            .map_or("", |s| s.op.as_str());
        artifact_to_py(py, artifact.as_ref(), op)
    }

    /// One table over every recording of a pipeline (docs/features.md). `features` is "edges"
    /// (functional connectivity), "mode_power" (a decomposition or its epochs) or
    /// "band_amplitude" (a spectrum: ALFF per band, or fALFF with `relative`), "mean" (epochs of
    /// a series: each condition's block average per channel) or "coefficients" (loadings such as
    /// GLM betas: one row per condition).
    #[pyo3(signature = (pipeline, output, features = "edges", *, resolvable_only = true, participants = None, subject_entity = "subject", bands = None, relative = false))]
    #[allow(clippy::too_many_arguments)]
    fn feature_table(
        &self,
        py: Python<'_>,
        pipeline: &str,
        output: &str,
        features: &str,
        resolvable_only: bool,
        participants: Option<PyRef<'_, crate::cohort::Participants>>,
        subject_entity: &str,
        bands: Option<Bound<'_, pyo3::types::PyDict>>,
        relative: bool,
    ) -> PyResult<crate::cohort::FeatureTable> {
        let features = match (features, bands) {
            ("edges", None) => modewise_cohort::Features::Edges { resolvable_only },
            ("mode_power", None) => modewise_cohort::Features::ModePower { resolvable_only },
            ("mean", None) => modewise_cohort::Features::Mean,
            ("coefficients", None) => modewise_cohort::Features::Coefficients,
            ("band_amplitude", Some(bands)) => modewise_cohort::Features::BandAmplitude {
                bands: bands
                    .iter()
                    .map(|(name, range)| {
                        let (low_hz, high_hz) = range.extract::<(f64, f64)>()?;
                        Ok(modewise_cohort::Band {
                            name: name.extract()?,
                            low_hz,
                            high_hz,
                        })
                    })
                    .collect::<PyResult<_>>()?,
                relative,
            },
            ("band_amplitude", None) => {
                return Err(crate::convert::value_error(
                    "band_amplitude needs bands, for example {'slow5': (0.01, 0.027)}",
                ));
            }
            (_, Some(_)) => {
                return Err(crate::convert::value_error(
                    "bands only apply to band_amplitude",
                ));
            }
            (other, None) => {
                return Err(crate::convert::value_error(format!(
                    "features must be 'edges', 'mode_power', 'band_amplitude', 'mean' or 'coefficients', got {other:?}"
                )));
            }
        };
        self.feature_table_impl(
            py,
            pipeline,
            output,
            features,
            participants.as_deref(),
            subject_entity,
        )
    }

    /// Stacks one output across recordings into a single array with the recording first, for
    /// deep learning. Shapes must match. Returns (recordings, array).
    fn stack<'py>(
        &self,
        py: Python<'py>,
        pipeline: &str,
        output: &str,
    ) -> PyResult<(Vec<String>, Bound<'py, numpy::PyArrayDyn<f64>>)> {
        use numpy::IntoPyArray;
        let (recordings, stacked) = self.stack_impl(pipeline, output)?;
        Ok((recordings, stacked.into_pyarray(py)))
    }

    /// Writes one HDF5 file per recording (for MATLAB and other tools). Returns the file paths.
    fn export_hdf5(&self, py: Python<'_>, pipeline: &str, to: PathBuf) -> PyResult<Vec<PathBuf>> {
        py.detach(|| self.inner.export_hdf5(pipeline, &to))
            .map_err(results_error)
    }

    fn __repr__(&self) -> String {
        format!(
            "Results({:?}, pipelines={:?})",
            self.inner.root(),
            self.inner.pipelines()
        )
    }
}
