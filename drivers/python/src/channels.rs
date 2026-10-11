use std::collections::BTreeMap;
use std::path::PathBuf;

use modewise_brain::{ChannelInfo, Hemisphere, Pattern, Region, Rule, Selection};
use modewise_dataset::TableFormat;
use pyo3::exceptions::{PyOSError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::convert::value_error;

/// What each channel is: name, region, hemisphere ("left", "right" or None) and open tags.
#[pyclass(frozen, module = "modewise")]
pub struct ChannelTable {
    pub(crate) inner: modewise_brain::ChannelTable,
}

fn hemisphere(text: Option<&str>) -> PyResult<Option<Hemisphere>> {
    text.map(|h| {
        Hemisphere::parse(h)
            .ok_or_else(|| value_error(format!("hemisphere {h:?} is not left or right")))
    })
    .transpose()
}

#[pymethods]
impl ChannelTable {
    /// Builds a table from dicts with "name" and optionally "region", "hemisphere" and "tags".
    #[staticmethod]
    fn from_records(records: Vec<Bound<'_, PyDict>>) -> PyResult<Self> {
        let channels = records
            .iter()
            .map(|r| {
                let get = |key: &str| -> PyResult<Option<String>> {
                    r.get_item(key)?
                        .filter(|v| !v.is_none())
                        .map(|v| v.extract())
                        .transpose()
                };
                let name =
                    get("name")?.ok_or_else(|| value_error("every record needs a \"name\""))?;
                let mut c = ChannelInfo::new(name);
                if let Some(region) = get("region")? {
                    c = c.with_region(Region::new(region));
                }
                if let Some(side) = hemisphere(get("hemisphere")?.as_deref())? {
                    c = c.with_hemisphere(side);
                }
                let tags: BTreeMap<String, String> = r
                    .get_item("tags")?
                    .filter(|v| !v.is_none())
                    .map(|v| v.extract())
                    .transpose()?
                    .unwrap_or_default();
                for (k, v) in tags {
                    c = c.with_tag(k, v);
                }
                Ok(c)
            })
            .collect::<PyResult<Vec<_>>>()?;
        Ok(Self {
            inner: modewise_brain::ChannelTable::new(channels).map_err(value_error)?,
        })
    }

    #[getter]
    fn names(&self) -> Vec<String> {
        self.inner.names()
    }

    #[getter]
    fn regions(&self) -> Vec<Option<String>> {
        self.inner
            .channels()
            .iter()
            .map(|c| c.region().map(|r| r.as_str().to_string()))
            .collect()
    }

    #[getter]
    fn hemispheres(&self) -> Vec<Option<&'static str>> {
        self.inner
            .channels()
            .iter()
            .map(|c| c.hemisphere().map(Hemisphere::as_str))
            .collect()
    }

    #[getter]
    fn tags(&self) -> Vec<BTreeMap<String, String>> {
        self.inner
            .channels()
            .iter()
            .map(|c| c.tags().clone())
            .collect()
    }

    /// Names of the channels matching any rule. A rule is a dict with any of "name", "region",
    /// "hemisphere" and "tags"; text matches exactly or with `*` wildcards.
    fn select(&self, rules: Vec<Bound<'_, PyDict>>) -> PyResult<Vec<String>> {
        let rules = rules
            .iter()
            .map(|r| {
                let get = |key: &str| -> PyResult<Option<String>> {
                    r.get_item(key)?
                        .filter(|v| !v.is_none())
                        .map(|v| v.extract())
                        .transpose()
                };
                let tags: BTreeMap<String, String> = r
                    .get_item("tags")?
                    .filter(|v| !v.is_none())
                    .map(|v| v.extract())
                    .transpose()?
                    .unwrap_or_default();
                Ok(Rule {
                    name: get("name")?.map(Pattern::new),
                    region: get("region")?.map(Pattern::new),
                    hemisphere: hemisphere(get("hemisphere")?.as_deref())?,
                    tags: tags
                        .into_iter()
                        .map(|(k, v)| (k, Pattern::new(v)))
                        .collect(),
                })
            })
            .collect::<PyResult<Vec<_>>>()?;
        let selection = Selection::new(rules).map_err(value_error)?;
        Ok(self
            .inner
            .channels()
            .iter()
            .filter(|c| selection.matches(c))
            .map(|c| c.name().to_string())
            .collect())
    }

    fn __add__(&self, other: &ChannelTable) -> PyResult<Self> {
        let inner = modewise_brain::ChannelTable::concat([self.inner.clone(), other.inner.clone()])
            .map_err(value_error)?;
        Ok(Self { inner })
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        format!("ChannelTable({} channels)", self.inner.len())
    }
}

/// Reads a channel table from an atlas label file: "tsv" (name, region, hemisphere and tag
/// columns; covers BIDS dseg.tsv), "schaefer", "tian" or "names" (one per line).
#[pyfunction]
#[pyo3(signature = (path, format = "tsv"))]
pub fn read_channels(path: PathBuf, format: &str) -> PyResult<ChannelTable> {
    let format = match format {
        "tsv" => TableFormat::Tsv,
        "schaefer" => TableFormat::Schaefer,
        "tian" => TableFormat::Tian,
        "names" => TableFormat::Names,
        other => {
            return Err(PyValueError::new_err(format!(
                "format must be tsv, schaefer, tian or names, got {other:?}"
            )));
        }
    };
    if !path.is_file() {
        return Err(PyOSError::new_err(format!(
            "no channel table at {}",
            path.display()
        )));
    }
    let inner = modewise_dataset::read_table(&path, format).map_err(PyValueError::new_err)?;
    Ok(ChannelTable { inner })
}
