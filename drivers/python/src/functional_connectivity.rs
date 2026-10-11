use modewise_connectivity::Measure;
use modewise_fc::{Source, functional_connectivity as compute};
use pyo3::prelude::*;

use crate::convert::value_error;
use crate::events::Epochs;
use crate::results::{Decomposition, FunctionalConnectivity, Series};

/// Functional connectivity of a Series (one matrix), a Decomposition (one per mode) or
/// Epochs (one per epoch, and per mode for a decomposition source); see docs/connectivity.md.
///
/// measure: "pearson" or "plv" (phase locking value). fisher_z applies to pearson only.
#[pyfunction]
#[pyo3(signature = (data, measure = "pearson", *, fisher_z = false))]
pub fn functional_connectivity(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    measure: &str,
    fisher_z: bool,
) -> PyResult<FunctionalConnectivity> {
    let measure = match (measure, fisher_z) {
        ("pearson", fisher_z) => Measure::Pearson { fisher_z },
        ("plv", false) => Measure::PhaseLockingValue,
        ("plv", true) => return Err(value_error("fisher_z applies to pearson only")),
        (other, _) => {
            return Err(value_error(format!(
                "measure must be 'pearson' or 'plv', got {other:?}"
            )));
        }
    };
    let result = if let Ok(s) = data.cast::<Series>() {
        let series = s.get().to_domain(py)?;
        py.detach(|| compute(Source::Series(&series), measure))
    } else if let Ok(d) = data.cast::<Decomposition>() {
        let modes = d.get().to_domain(py)?;
        py.detach(|| compute(Source::Decomposition(&modes), measure))
    } else if let Ok(e) = data.cast::<Epochs>() {
        let epochs = &e.get().inner;
        py.detach(|| compute(Source::Epochs(epochs), measure))
    } else {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "data must be a Series, Decomposition or Epochs",
        ));
    };
    Ok(FunctionalConnectivity::from_domain(
        py,
        &result.map_err(value_error)?,
    ))
}
