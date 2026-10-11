use std::path::PathBuf;

use modewise_hdf5::{Hdf5Error, Layout, read_series};
use modewise_signals::SamplingRate;
use pyo3::exceptions::PyOSError;
use pyo3::prelude::*;

use crate::convert::value_error;
use crate::results::Series;

/// Reads one 2D numeric dataset from an HDF5 file as a Series.
#[pyfunction]
#[pyo3(signature = (path, dataset, sampling_rate, *, layout = "channels_by_time", labels = None))]
pub fn read_hdf5(
    py: Python<'_>,
    path: PathBuf,
    dataset: &str,
    sampling_rate: f64,
    layout: &str,
    labels: Option<Vec<String>>,
) -> PyResult<Series> {
    let layout = match layout {
        "channels_by_time" => Layout::ChannelsByTime,
        "time_by_channels" => Layout::TimeByChannels,
        other => {
            return Err(value_error(format!(
                "layout must be 'channels_by_time' or 'time_by_channels', got {other:?}"
            )));
        }
    };
    let fs = SamplingRate::new(sampling_rate).map_err(value_error)?;
    let series = py
        .detach(|| read_series(&path, dataset, layout, fs, labels))
        .map_err(|e| match e {
            Hdf5Error::Open { .. } => PyOSError::new_err(e.to_string()),
            _ => value_error(e),
        })?;
    Ok(Series::from_domain(py, &series))
}

/// Region signals from a 4D NIfTI run: the mean over each atlas label's voxels, as nilearn's
/// NiftiLabelsMasker (docs/volumes.md). Atlases are 3D label volumes, concatenated in order.
#[pyfunction]
#[pyo3(signature = (path, atlases, *, sampling_rate = None, standardize = "none", labels = None))]
pub fn read_nifti(
    path: PathBuf,
    atlases: Vec<PathBuf>,
    sampling_rate: Option<f64>,
    standardize: &str,
    labels: Option<Vec<String>>,
) -> PyResult<crate::results::Series> {
    let standardize = match standardize {
        "none" => modewise_parcellate::Standardize::None,
        "zscore_sample" => modewise_parcellate::Standardize::ZscoreSample,
        other => {
            return Err(crate::convert::value_error(format!(
                "standardize must be 'none' or 'zscore_sample', got {other:?}"
            )));
        }
    };
    if atlases.is_empty() {
        return Err(crate::convert::value_error("give at least one atlas"));
    }
    let bold = modewise_nifti::read_bold(&path).map_err(crate::convert::value_error)?;
    let hz = match sampling_rate {
        Some(hz) => hz,
        None => {
            1.0 / bold.repetition_time_s.ok_or_else(|| {
                crate::convert::value_error(
                    "the NIfTI header has no repetition time; pass sampling_rate",
                )
            })?
        }
    };
    let mut rows = Vec::new();
    for atlas in &atlases {
        let atlas_labels =
            modewise_nifti::read_labels(atlas).map_err(crate::convert::value_error)?;
        let (signals, _) = modewise_parcellate::parcellate(
            bold.data.view(),
            &bold.affine,
            atlas_labels.data.view(),
            &atlas_labels.affine,
            standardize,
        )
        .map_err(crate::convert::value_error)?;
        rows.push(signals);
    }
    let views: Vec<_> = rows.iter().map(|r| r.view()).collect();
    let data =
        ndarray::concatenate(ndarray::Axis(0), &views).map_err(crate::convert::value_error)?;
    let series = crate::convert::series_from(data, hz, labels)?;
    Python::attach(|py| Ok(crate::results::Series::from_domain(py, &series)))
}
