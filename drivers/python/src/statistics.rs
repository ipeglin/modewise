use modewise_cohort::{Design, Report, Settings, test};
use modewise_stats::{Fdr, Tail};
use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::cohort::FeatureTable;
use crate::convert::value_error;

/// Per-feature results of a permutation test (docs/statistics.md).
#[pyclass(frozen, module = "modewise")]
pub struct StatsReport {
    inner: Report,
}

fn array<'py>(py: Python<'py>, a: &ndarray::Array1<f64>) -> Bound<'py, PyArray1<f64>> {
    a.to_owned().into_pyarray(py)
}

#[pymethods]
impl StatsReport {
    #[getter]
    fn feature_names(&self) -> Vec<String> {
        self.inner.feature_names.clone()
    }
    #[getter]
    fn t<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        array(py, &self.inner.t)
    }
    #[getter]
    fn df<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        array(py, &self.inner.df)
    }
    /// From Student's t (Welch–Satterthwaite df for Welch); for reference.
    #[getter]
    fn p_parametric<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        array(py, &self.inner.p_parametric)
    }
    /// (b + 1) / (m + 1) over m rearrangements; exact when every sign flip was enumerated.
    #[getter]
    fn p_permutation<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        array(py, &self.inner.p_permutation)
    }
    /// Family-wise error over all features (maximum statistic).
    #[getter]
    fn p_fwer<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        array(py, &self.inner.p_fwer)
    }
    /// False discovery rate q-values of the permutation p-values.
    #[getter]
    fn q_fdr<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        array(py, &self.inner.q_fdr)
    }
    /// Network-based statistic components: dicts with "network", "edges" and "p".
    #[getter]
    fn components<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyDict>>> {
        self.inner
            .components
            .iter()
            .map(|c| {
                let d = PyDict::new(py);
                d.set_item("network", c.network)?;
                d.set_item("edges", c.edges.clone())?;
                d.set_item("p", c.p)?;
                Ok(d)
            })
            .collect()
    }
    #[getter]
    fn n_observations(&self) -> usize {
        self.inner.n_observations
    }
    /// Rows left out because a value the design needs is missing.
    #[getter]
    fn dropped(&self) -> usize {
        self.inner.dropped
    }
    #[getter]
    fn n_rearrangements(&self) -> usize {
        self.inner.n_rearrangements
    }
    #[getter]
    fn exhaustive(&self) -> bool {
        self.inner.exhaustive
    }
    /// Design columns: intercept, the tested regressor, then the (centred) covariates.
    #[getter]
    fn regressors(&self) -> Vec<String> {
        self.inner.regressors.clone()
    }

    /// A pandas DataFrame with one row per feature (needs pandas).
    fn to_pandas<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let columns = PyDict::new(py);
        columns.set_item("feature", self.feature_names())?;
        for (name, values) in [
            ("t", &self.inner.t),
            ("df", &self.inner.df),
            ("p_parametric", &self.inner.p_parametric),
            ("p_permutation", &self.inner.p_permutation),
            ("p_fwer", &self.inner.p_fwer),
            ("q_fdr", &self.inner.q_fdr),
        ] {
            columns.set_item(name, array(py, values))?;
        }
        py.import("pandas")?.call_method1("DataFrame", (columns,))
    }

    fn __repr__(&self) -> String {
        format!(
            "StatsReport({} features, n = {}, {} rearrangements{})",
            self.inner.feature_names.len(),
            self.inner.n_observations,
            self.inner.n_rearrangements,
            if self.inner.exhaustive { ", exact" } else { "" }
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn settings(
    covariates: Vec<String>,
    condition: Option<String>,
    tail: &str,
    n_permutations: usize,
    seed: u64,
    fdr: &str,
    nbs_threshold: Option<f64>,
) -> PyResult<Settings> {
    let tail = match tail {
        "both" => Tail::Both,
        "greater" => Tail::Greater,
        "less" => Tail::Less,
        other => {
            return Err(value_error(format!(
                "tail must be 'both', 'greater' or 'less', got {other:?}"
            )));
        }
    };
    let fdr = match fdr {
        "bh" => Fdr::BenjaminiHochberg,
        "by" => Fdr::BenjaminiYekutieli,
        other => {
            return Err(value_error(format!(
                "fdr must be 'bh' or 'by', got {other:?}"
            )));
        }
    };
    Ok(Settings {
        covariates,
        condition,
        tail,
        n_permutations,
        seed,
        fdr,
        nbs_threshold,
    })
}

fn run(
    py: Python<'_>,
    table: &FeatureTable,
    design: Design,
    settings: Settings,
) -> PyResult<StatsReport> {
    let inner = py
        .detach(|| test(table.inner(), &design, &settings))
        .map_err(value_error)?;
    Ok(StatsReport { inner })
}

/// Group `groups[0]` minus `groups[1]` of a participant column, per feature, adjusting for
/// covariates (GLM, Freedman–Lane permutation). `welch=True` (no covariates) uses Welch's t.
#[pyfunction]
#[pyo3(signature = (table, column, groups, *, covariates = vec![], condition = None, welch = false, tail = "both", n_permutations = 5000, seed = 0, fdr = "bh", nbs_threshold = None))]
#[allow(clippy::too_many_arguments)]
pub fn test_groups(
    py: Python<'_>,
    table: &FeatureTable,
    column: String,
    groups: (String, String),
    covariates: Vec<String>,
    condition: Option<String>,
    welch: bool,
    tail: &str,
    n_permutations: usize,
    seed: u64,
    fdr: &str,
    nbs_threshold: Option<f64>,
) -> PyResult<StatsReport> {
    let s = settings(
        covariates,
        condition,
        tail,
        n_permutations,
        seed,
        fdr,
        nbs_threshold,
    )?;
    run(
        py,
        table,
        Design::Groups {
            column,
            a: groups.0,
            b: groups.1,
            welch,
        },
        s,
    )
}

/// Condition `conditions[0]` minus `conditions[1]` within each recording, by sign flipping.
#[pyfunction]
#[pyo3(signature = (table, conditions, *, covariates = vec![], tail = "both", n_permutations = 5000, seed = 0, fdr = "bh", nbs_threshold = None))]
#[allow(clippy::too_many_arguments)]
pub fn test_paired(
    py: Python<'_>,
    table: &FeatureTable,
    conditions: (String, String),
    covariates: Vec<String>,
    tail: &str,
    n_permutations: usize,
    seed: u64,
    fdr: &str,
    nbs_threshold: Option<f64>,
) -> PyResult<StatsReport> {
    let s = settings(
        covariates,
        None,
        tail,
        n_permutations,
        seed,
        fdr,
        nbs_threshold,
    )?;
    run(
        py,
        table,
        Design::Paired {
            a: conditions.0,
            b: conditions.1,
        },
        s,
    )
}

/// The slope of a numeric participant column per feature, adjusting for covariates.
#[pyfunction]
#[pyo3(signature = (table, score, *, covariates = vec![], condition = None, tail = "both", n_permutations = 5000, seed = 0, fdr = "bh", nbs_threshold = None))]
#[allow(clippy::too_many_arguments)]
pub fn test_correlation(
    py: Python<'_>,
    table: &FeatureTable,
    score: String,
    covariates: Vec<String>,
    condition: Option<String>,
    tail: &str,
    n_permutations: usize,
    seed: u64,
    fdr: &str,
    nbs_threshold: Option<f64>,
) -> PyResult<StatsReport> {
    let s = settings(
        covariates,
        condition,
        tail,
        n_permutations,
        seed,
        fdr,
        nbs_threshold,
    )?;
    run(py, table, Design::Correlation { score }, s)
}
