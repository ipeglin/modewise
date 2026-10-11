use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use modewise_operations::builtin_registry;
use modewise_pipeline::{
    Artifact, ArtifactStore, PipelineError, PipelineSpec, Plan, RunResult, run,
};
use modewise_store_memory::MemoryStore as Store;
use numpy::{AllowTypeChange, PyArrayLike2};
use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::convert::series_from;
use crate::results::{Series, artifact_to_py};

/// Spec and planning problems are the caller's to fix (ValueError); failures while running are not.
fn pipeline_error(err: PipelineError) -> PyErr {
    match err {
        PipelineError::StepFailed { .. } | PipelineError::Store(_) => {
            PyRuntimeError::new_err(err.to_string())
        }
        _ => PyValueError::new_err(err.to_string()),
    }
}

/// Keeps step results in memory so reruns skip unchanged steps.
#[pyclass(frozen, module = "modewise")]
pub struct MemoryStore {
    inner: Arc<Store>,
}

#[pymethods]
impl MemoryStore {
    #[new]
    fn new() -> Self {
        Self {
            inner: Arc::new(Store::new()),
        }
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }
}

/// A checked pipeline of named steps.
#[pyclass(frozen, module = "modewise")]
pub struct Pipeline {
    plan: Plan,
}

impl Pipeline {
    fn build(spec: Result<PipelineSpec, PipelineError>) -> PyResult<Self> {
        let plan = Plan::new(&spec.map_err(pipeline_error)?, &builtin_registry())
            .map_err(pipeline_error)?;
        Ok(Self { plan })
    }
}

#[pymethods]
impl Pipeline {
    #[staticmethod]
    fn from_toml(text: &str) -> PyResult<Self> {
        Self::build(PipelineSpec::from_toml(text))
    }

    #[staticmethod]
    fn from_file(path: PathBuf) -> PyResult<Self> {
        Self::build(PipelineSpec::from_toml(&std::fs::read_to_string(path)?))
    }

    /// Same layout as the TOML file: {"pipeline": {...}, "step": [{...}, ...]}.
    #[staticmethod]
    fn from_dict(py: Python<'_>, spec: &Bound<'_, PyDict>) -> PyResult<Self> {
        let json: String = py
            .import("json")?
            .call_method1("dumps", (spec,))?
            .extract()?;
        Self::build(PipelineSpec::from_json(&json))
    }

    #[getter]
    fn name(&self) -> String {
        self.plan.name().to_string()
    }

    #[getter]
    fn inputs(&self) -> Vec<String> {
        self.plan.inputs().to_vec()
    }

    /// Step ids in the order they run.
    #[getter]
    fn steps(&self) -> Vec<String> {
        self.plan.step_ids().map(str::to_string).collect()
    }

    /// Runs over every recording of `dataset`, saving results to the folder `out`.
    /// Unchanged steps from earlier runs into the same folder are reused.
    #[pyo3(signature = (dataset, out, *, shard = None, workers = 0, progress = true))]
    fn run_dataset(
        &self,
        py: Python<'_>,
        dataset: &crate::data::Dataset,
        out: PathBuf,
        shard: Option<(usize, usize)>,
        workers: usize,
        progress: bool,
    ) -> PyResult<BatchRun> {
        if let Some((i, n)) = shard
            && (n == 0 || i == 0 || i > n)
        {
            return Err(PyValueError::new_err(format!(
                "shard must be (i, n) with 1 <= i <= n, got ({i}, {n})"
            )));
        }
        let results = crate::saved::Results::open_dir(&out)?;
        let report = |run: &modewise_pipeline::RecordingRun| {
            if !progress {
                return;
            }
            let line = match &run.result {
                Ok(r) => {
                    let cached = r.manifest.steps.iter().filter(|s| s.cached).count();
                    format!(
                        "ok      {} ({cached}/{} steps cached)",
                        run.id,
                        r.manifest.steps.len()
                    )
                }
                Err(e) => format!("FAILED  {}: {e}", run.id),
            };
            Python::attach(|py| {
                let _ = py
                    .import("builtins")
                    .and_then(|b| b.getattr("print"))
                    .and_then(|p| p.call1((line,)));
            });
        };
        let summary = py
            .detach(|| {
                results
                    .inner()
                    .run_batch(&self.plan, dataset.inner(), shard, workers, &report)
            })
            .map_err(crate::saved::results_error)?;
        Ok(BatchRun {
            succeeded: summary.succeeded,
            failed: summary.failed,
            results: Py::new(py, results)?,
            pipeline: self.plan.name().to_string(),
        })
    }

    #[pyo3(signature = (inputs, sampling_rate = None, *, labels = None, store = None))]
    fn run(
        &self,
        py: Python<'_>,
        inputs: &Bound<'_, PyDict>,
        sampling_rate: Option<f64>,
        labels: Option<BTreeMap<String, Vec<String>>>,
        store: Option<Bound<'_, MemoryStore>>,
    ) -> PyResult<PipelineRun> {
        let mut labels = labels.unwrap_or_default();
        let mut series = BTreeMap::new();
        for (name, value) in inputs.iter() {
            let name: String = name.extract()?;
            if let Ok(events) = value.cast::<crate::events::Events>() {
                series.insert(name, Artifact::Events(events.get().inner.clone()));
                continue;
            }
            if let Ok(table) = value.cast::<crate::channels::ChannelTable>() {
                series.insert(name, Artifact::Channels(table.get().inner.clone()));
                continue;
            }
            let input = match value.cast::<Series>() {
                Ok(s) => {
                    if labels.contains_key(&name) {
                        return Err(PyValueError::new_err(format!(
                            "input {name:?} is a Series and already has labels"
                        )));
                    }
                    s.get().to_domain(py)?
                }
                Err(_) => {
                    let fs = sampling_rate.ok_or_else(|| {
                        PyValueError::new_err(format!(
                            "input {name:?} is a plain array; pass sampling_rate or wrap it in modewise.Series"
                        ))
                    })?;
                    let array: PyArrayLike2<'_, f64, AllowTypeChange> = value.extract()?;
                    series_from(array.as_array().to_owned(), fs, labels.remove(&name))?
                }
            };
            series.insert(name, Artifact::Series(input));
        }
        if let Some(unused) = labels.keys().next() {
            return Err(PyValueError::new_err(format!(
                "labels given for unknown input {unused:?}"
            )));
        }
        let store = store.map_or_else(|| Arc::new(Store::new()), |s| Arc::clone(&s.get().inner));
        let result = py
            .detach(|| run(&self.plan, &series, store.as_ref()))
            .map_err(pipeline_error)?;
        Ok(PipelineRun { result, store })
    }
}

/// Outcome of running a pipeline over a dataset.
#[pyclass(frozen, module = "modewise")]
pub struct BatchRun {
    #[pyo3(get)]
    succeeded: Vec<String>,
    /// Recording id -> error message.
    #[pyo3(get)]
    failed: BTreeMap<String, String>,
    #[pyo3(get)]
    results: Py<crate::saved::Results>,
    pipeline: String,
}

#[pymethods]
impl BatchRun {
    /// A step output of one recording, by `step.output`, or by `step` when it has one output.
    fn output(&self, py: Python<'_>, recording: &str, reference: &str) -> PyResult<Py<PyAny>> {
        self.results
            .get()
            .output(py, &self.pipeline, recording, reference)
    }

    fn __repr__(&self) -> String {
        format!(
            "BatchRun(succeeded={}, failed={})",
            self.succeeded.len(),
            self.failed.len()
        )
    }
}

/// Outcome of one pipeline run.
#[pyclass(frozen, module = "modewise")]
pub struct PipelineRun {
    result: RunResult,
    store: Arc<Store>,
}

#[pymethods]
impl PipelineRun {
    /// Output names, as `step.output`.
    #[getter]
    fn outputs(&self) -> Vec<String> {
        self.result.outputs().keys().cloned().collect()
    }

    /// Which steps were reused from the store instead of computed.
    #[getter]
    fn cached(&self) -> BTreeMap<String, bool> {
        let steps = &self.result.manifest.steps;
        steps.iter().map(|s| (s.id.clone(), s.cached)).collect()
    }

    /// Provenance record: operations, parameters, data keys and timings.
    #[getter]
    fn manifest<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        py.import("json")?
            .call_method1("loads", (self.result.manifest.to_json(),))
    }

    /// A step output, by `step.output`, or by `step` when it has one output.
    fn output(&self, py: Python<'_>, reference: &str) -> PyResult<Py<PyAny>> {
        let key = self.result.output_key(reference).ok_or_else(|| {
            PyKeyError::new_err(format!(
                "no output {reference:?}; outputs are {:?}",
                self.outputs()
            ))
        })?;
        let artifact = self
            .store
            .get(key)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            .ok_or_else(|| {
                let step = reference.split('.').next().unwrap_or(reference);
                let unkept = self
                    .result
                    .manifest
                    .steps
                    .iter()
                    .any(|s| s.id == step && !s.stored);
                if unkept {
                    PyValueError::new_err(format!(
                        "step {step:?} has keep = false, so its output {reference:?} was not stored"
                    ))
                } else {
                    PyRuntimeError::new_err(format!("output {reference:?} missing from store"))
                }
            })?;
        let step = reference.split('.').next().unwrap_or(reference);
        let op = self
            .result
            .manifest
            .steps
            .iter()
            .find(|s| s.id == step)
            .map_or("", |s| s.op.as_str());
        artifact_to_py(py, artifact.as_ref(), op)
    }
}

/// Every built-in operation with its summary, ports and parameter JSON Schema (as
/// `modewise ops --json`), for tools and agents.
#[pyfunction]
pub fn operations(py: Python<'_>) -> PyResult<Py<PyAny>> {
    let json = serde_json::to_string(&modewise_operations::catalogue()["operations"])
        .map_err(crate::convert::value_error)?;
    let loads = py.import("json")?.getattr("loads")?;
    Ok(loads.call1((json,))?.unbind())
}
