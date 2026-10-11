use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use modewise_cohort::{Features, RecordingOutput, feature_table as build};
use modewise_pipeline::Artifact;
use numpy::{IntoPyArray, PyArray2};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::convert::value_error;
use crate::saved::{Results, results_error};

/// Information about each subject, joined from participants.tsv and phenotype tables.
#[pyclass(frozen, module = "modewise")]
pub struct Participants {
    pub(crate) inner: modewise_study::Participants,
}

#[pymethods]
impl Participants {
    #[getter]
    fn columns(&self) -> Vec<String> {
        self.inner.columns().to_vec()
    }

    #[getter]
    fn subjects(&self) -> Vec<String> {
        self.inner.subjects().map(str::to_string).collect()
    }

    /// One subject's value as text, or None when missing.
    fn value(&self, subject: &str, column: &str) -> PyResult<Option<String>> {
        Ok(self
            .inner
            .text(subject, column)
            .map_err(value_error)?
            .map(str::to_string))
    }

    /// A pandas DataFrame indexed by subject (needs pandas).
    fn to_pandas<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let columns = PyDict::new(py);
        let subjects = self.subjects();
        for column in self.inner.columns() {
            let values: Vec<Option<String>> = subjects
                .iter()
                .map(|s| self.inner.text(s, column).map(|v| v.map(str::to_string)))
                .collect::<Result<_, _>>()
                .map_err(value_error)?;
            columns.set_item(column, values)?;
        }
        let pandas = py.import("pandas")?;
        let index = pandas.call_method1("Index", (subjects,))?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("index", index)?;
        let frame = pandas.call_method("DataFrame", (columns,), Some(&kwargs))?;
        frame.getattr("index")?.setattr("name", "subject")?;
        Ok(frame)
    }

    fn __len__(&self) -> usize {
        self.inner.subjects().count()
    }

    fn __repr__(&self) -> String {
        format!(
            "Participants({} subjects, columns={:?})",
            self.inner.subjects().count(),
            self.inner.columns()
        )
    }
}

/// Observations × named features with participant columns: the handover to statistics and
/// machine learning (docs/features.md).
#[pyclass(frozen, module = "modewise")]
pub struct FeatureTable {
    inner: modewise_study::FeatureTable,
}

impl FeatureTable {
    pub(crate) fn inner(&self) -> &modewise_study::FeatureTable {
        &self.inner
    }
}

#[pymethods]
impl FeatureTable {
    /// A table from your own values (observations × features), for example ratios or contrasts
    /// computed in numpy, so they can go to `test_groups`, `test_paired` or `test_correlation`.
    /// `conditions` labels rows for paired tests; `metadata` and `participants` add the columns
    /// designs use for groups and covariates.
    #[new]
    #[pyo3(signature = (values, feature_names, recordings, *, subjects = None, conditions = None, metadata = None, participants = None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        values: numpy::PyArrayLike2<'_, f64, numpy::AllowTypeChange>,
        feature_names: Vec<String>,
        recordings: Vec<String>,
        subjects: Option<Vec<Option<String>>>,
        conditions: Option<Vec<Option<String>>>,
        metadata: Option<BTreeMap<String, Vec<Option<String>>>>,
        participants: Option<PyRef<'_, Participants>>,
    ) -> PyResult<Self> {
        let n = recordings.len();
        let column = |given: Option<Vec<Option<String>>>, name: &str| match given {
            Some(v) if v.len() != n => Err(value_error(format!(
                "{name} has {} values for {n} recordings",
                v.len()
            ))),
            Some(v) => Ok(v),
            None => Ok(vec![None; n]),
        };
        let subjects = column(subjects, "subjects")?;
        let conditions = column(conditions, "conditions")?;
        let observations = recordings
            .into_iter()
            .zip(subjects)
            .zip(conditions)
            .map(
                |((recording, subject), condition)| modewise_study::Observation {
                    recording,
                    subject,
                    condition,
                },
            )
            .collect();
        let mut table = modewise_study::FeatureTable::new(
            observations,
            feature_names,
            values.as_array().to_owned(),
        )
        .map_err(value_error)?;
        if let Some(p) = participants {
            table = table.with_participants(&p.inner).map_err(value_error)?;
        }
        for (name, values) in metadata.unwrap_or_default() {
            table = table.with_column(&name, values).map_err(value_error)?;
        }
        Ok(Self { inner: table })
    }

    /// Observations × features.
    #[getter]
    fn values<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.inner.values().to_owned().into_pyarray(py)
    }

    #[getter]
    fn feature_names(&self) -> Vec<String> {
        self.inner.feature_names().to_vec()
    }

    #[getter]
    fn recordings(&self) -> Vec<String> {
        self.inner
            .observations()
            .iter()
            .map(|o| o.recording.clone())
            .collect()
    }

    #[getter]
    fn subjects(&self) -> Vec<Option<String>> {
        self.inner
            .observations()
            .iter()
            .map(|o| o.subject.clone())
            .collect()
    }

    #[getter]
    fn conditions(&self) -> Vec<Option<String>> {
        self.inner
            .observations()
            .iter()
            .map(|o| o.condition.clone())
            .collect()
    }

    /// Participant columns, each a list with one value (or None) per observation.
    #[getter]
    fn metadata(&self) -> PyResult<BTreeMap<String, Vec<Option<String>>>> {
        self.inner
            .metadata_columns()
            .map(|c| {
                Ok((
                    c.to_string(),
                    self.inner.metadata(c).map_err(value_error)?.to_vec(),
                ))
            })
            .collect()
    }

    /// A pandas DataFrame: recording, subject, condition, participant columns, then features.
    fn to_pandas<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let pandas = py.import("pandas")?;
        let ids = PyDict::new(py);
        ids.set_item("recording", self.recordings())?;
        ids.set_item("subject", self.subjects())?;
        ids.set_item("condition", self.conditions())?;
        for (column, values) in self.metadata()? {
            ids.set_item(column, values)?;
        }
        let left = pandas.call_method1("DataFrame", (ids,))?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("columns", self.feature_names())?;
        let right = pandas.call_method("DataFrame", (self.values(py),), Some(&kwargs))?;
        let concat_kwargs = PyDict::new(py);
        concat_kwargs.set_item("axis", 1)?;
        pandas.call_method("concat", ((left, right),), Some(&concat_kwargs))
    }

    /// Writes a tab-separated file (missing values as n/a), for MATLAB, R or spreadsheets.
    fn to_tsv(&self, path: PathBuf) -> PyResult<()> {
        modewise_results::write_feature_table(&path, &self.inner).map_err(results_error)
    }

    fn __repr__(&self) -> String {
        format!(
            "FeatureTable({} observations × {} features)",
            self.inner.observations().len(),
            self.inner.feature_names().len()
        )
    }
}

impl Results {
    pub(crate) fn feature_table_impl(
        &self,
        py: Python<'_>,
        pipeline: &str,
        output: &str,
        features: Features,
        participants: Option<&Participants>,
        subject_entity: &str,
    ) -> PyResult<FeatureTable> {
        let recordings = self.inner().recordings(pipeline);
        let artifacts: Vec<Arc<Artifact>> = recordings
            .iter()
            .map(|r| self.inner().output(pipeline, r, output))
            .collect::<Result<_, _>>()
            .map_err(results_error)?;
        let outputs: Vec<RecordingOutput<'_>> = recordings
            .iter()
            .zip(&artifacts)
            .map(|(recording, a)| RecordingOutput {
                recording,
                artifact: a.as_ref(),
            })
            .collect();
        let table = py
            .detach(|| build(&outputs, &features, subject_entity))
            .map_err(value_error)?;
        let table = match participants {
            Some(p) => table.with_participants(&p.inner).map_err(value_error)?,
            None => table,
        };
        Ok(FeatureTable { inner: table })
    }
}

/// The main array of an output: data, modes, power, amplitude, connectivity or loadings.
fn main_array(artifact: &Artifact) -> Result<ndarray::ArrayD<f64>, String> {
    Ok(match artifact {
        Artifact::Series(s) => s.data().to_owned().into_dyn(),
        Artifact::Decomposition(d) => d.modes().to_owned().into_dyn(),
        Artifact::TimeFrequency(m) => m.power().to_owned().into_dyn(),
        Artifact::Instantaneous(m) => m.amplitude().to_owned().into_dyn(),
        Artifact::FunctionalConnectivity(c) => c.values().to_owned().into_dyn(),
        Artifact::Loadings(l) => l.values().to_owned().into_dyn(),
        other => return Err(format!("a {} has no single array to stack", other.kind())),
    })
}

impl Results {
    pub(crate) fn stack_impl(
        &self,
        pipeline: &str,
        output: &str,
    ) -> PyResult<(Vec<String>, ndarray::ArrayD<f64>)> {
        let recordings = self.inner().recordings(pipeline);
        let mut arrays = Vec::with_capacity(recordings.len());
        for r in &recordings {
            let artifact = self
                .inner()
                .output(pipeline, r, output)
                .map_err(results_error)?;
            let array = main_array(&artifact).map_err(value_error)?;
            if let Some(first) = arrays
                .first()
                .map(|a: &ndarray::ArrayD<f64>| a.shape().to_vec())
                && first != array.shape()
            {
                return Err(value_error(format!(
                    "recording {r:?} has shape {:?}, the first had {first:?}; stacking needs equal shapes",
                    array.shape()
                )));
            }
            arrays.push(array);
        }
        let views: Vec<_> = arrays.iter().map(|a| a.view()).collect();
        let stacked =
            ndarray::stack(ndarray::Axis(0), &views).map_err(|e| value_error(e.to_string()))?;
        Ok((recordings, stacked))
    }
}
