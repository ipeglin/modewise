use modewise_cohort::{Design, Report, Settings, test};
use modewise_stats::{Exchange, Fdr, Tail};
use ndarray::{Array2, Ix2};
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
    /// "t", "F", "v" (t with variance groups) or "G" (F with variance groups).
    #[getter]
    fn statistic(&self) -> &'static str {
        self.inner.statistic
    }
    /// Rows of the contrast: 1 for t and v, the numerator degrees of freedom for F and G.
    #[getter]
    fn rank(&self) -> usize {
        self.inner.rank
    }
    /// The statistic per feature (t, F, v or G, as named by `statistic`).
    #[getter]
    fn t<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        array(py, &self.inner.t)
    }
    /// Denominator degrees of freedom (Welch's approximation for v and G).
    #[getter]
    fn df<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        array(py, &self.inner.df)
    }
    /// From Student's t (t, v) or F (F, G); for reference.
    #[getter]
    fn p_parametric<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        array(py, &self.inner.p_parametric)
    }
    /// (b + 1) / (m + 1) over m rearrangements; exact when every rearrangement was enumerated.
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
    /// Design columns: intercept, the tested regressors, then the (centred) covariates.
    #[getter]
    fn regressors(&self) -> Vec<String> {
        self.inner.regressors.clone()
    }

    /// A pandas DataFrame with one row per feature, the statistic's column named by
    /// `statistic` (needs pandas).
    fn to_pandas<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let columns = PyDict::new(py);
        columns.set_item("feature", self.feature_names())?;
        for (name, values) in [
            (self.inner.statistic, &self.inner.t),
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
            "StatsReport({} of {} features, n = {}, {} rearrangements{})",
            self.inner.statistic,
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

/// Groups of a participant column, per feature, adjusting for covariates (GLM, Freedman–Lane
/// permutation): `groups[0]` minus `groups[1]` (t) for two, all equal (F) for more.
/// `welch=True` gives each group its own variance (v or G).
#[pyfunction]
#[pyo3(signature = (table, column, groups, *, covariates = vec![], condition = None, welch = false, tail = "both", n_permutations = 5000, seed = 0, fdr = "bh", nbs_threshold = None))]
#[allow(clippy::too_many_arguments)]
pub fn test_groups(
    py: Python<'_>,
    table: &FeatureTable,
    column: String,
    groups: Vec<String>,
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
            levels: groups,
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

/// A design matrix (one row per table row) tested by a contrast: one row gives t, more give F.
/// Blocks and variance groups name metadata columns, or "subject".
#[pyfunction]
#[pyo3(signature = (table, design, contrast, *, regressors = None, exchange = "permute", blocks = None, whole_blocks = false, variance_groups = None, tail = "both", n_permutations = 5000, seed = 0, fdr = "bh", nbs_threshold = None))]
#[allow(clippy::too_many_arguments)]
pub fn test_glm(
    py: Python<'_>,
    table: &FeatureTable,
    design: numpy::PyArrayLike2<'_, f64, numpy::AllowTypeChange>,
    contrast: numpy::PyArrayLikeDyn<'_, f64, numpy::AllowTypeChange>,
    regressors: Option<Vec<String>>,
    exchange: &str,
    blocks: Option<String>,
    whole_blocks: bool,
    variance_groups: Option<String>,
    tail: &str,
    n_permutations: usize,
    seed: u64,
    fdr: &str,
    nbs_threshold: Option<f64>,
) -> PyResult<StatsReport> {
    let design = design.as_array().to_owned();
    let contrast = contrast.as_array();
    let contrast: Array2<f64> = match contrast.ndim() {
        1 => contrast
            .to_owned()
            .into_shape_with_order((1, contrast.len())),
        2 => contrast.to_owned().into_dimensionality::<Ix2>(),
        n => {
            return Err(value_error(format!(
                "contrast must be 1-D or 2-D, got {n}-D"
            )));
        }
    }
    .map_err(|e| value_error(e.to_string()))?;
    let exchange = match exchange {
        "permute" => Exchange::Permute,
        "signflip" => Exchange::SignFlip,
        other => {
            return Err(value_error(format!(
                "exchange must be 'permute' or 'signflip', got {other:?}"
            )));
        }
    };
    let regressors =
        regressors.unwrap_or_else(|| (0..design.ncols()).map(|c| format!("x{c}")).collect());
    let s = settings(vec![], None, tail, n_permutations, seed, fdr, nbs_threshold)?;
    run(
        py,
        table,
        Design::Matrix {
            design,
            regressors,
            contrast,
            exchange,
            blocks,
            whole_blocks,
            variance_groups,
        },
        s,
    )
}
