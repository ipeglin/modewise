//! Reads multichannel signals from HDF5 files. Pure Rust, no HDF5 C library.

use std::path::Path;

use hdf5_pure::{DType, File};
use modewise_signals::{MultichannelSeries, SamplingRate, SignalError};
use ndarray::Array2;
use thiserror::Error;

/// How the 2D dataset is oriented on disk.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Layout {
    #[default]
    ChannelsByTime,
    TimeByChannels,
}

#[derive(Debug, Error)]
pub enum Hdf5Error {
    #[error("cannot open {path}: {message}")]
    Open { path: String, message: String },
    #[error("cannot read dataset {dataset:?}: {message}")]
    Dataset { dataset: String, message: String },
    #[error("dataset {dataset:?} must be 2D (channels and time), got shape {shape:?}")]
    NotTwoDimensional { dataset: String, shape: Vec<u64> },
    #[error("dataset {dataset:?} is not numeric ({dtype})")]
    NotNumeric { dataset: String, dtype: String },
    #[error(transparent)]
    Signal(#[from] SignalError),
}

/// Reads one 2D numeric dataset as a signal. Any integer or float type is converted to f64.
/// Without `labels`, channels are named `ch0`, `ch1`, ...
pub fn read_series(
    path: &Path,
    dataset: &str,
    layout: Layout,
    sampling_rate: SamplingRate,
    labels: Option<Vec<String>>,
) -> Result<MultichannelSeries, Hdf5Error> {
    let file = File::open(path).map_err(|e| Hdf5Error::Open {
        path: path.display().to_string(),
        message: e.to_string(),
    })?;
    let dataset_error = |message: String| Hdf5Error::Dataset {
        dataset: dataset.to_string(),
        message,
    };
    let ds = file
        .dataset(dataset)
        .map_err(|e| dataset_error(e.to_string()))?;

    let dtype = ds.dtype().map_err(|e| dataset_error(e.to_string()))?;
    if !is_numeric(&dtype) {
        return Err(Hdf5Error::NotNumeric {
            dataset: dataset.to_string(),
            dtype: format!("{dtype:?}"),
        });
    }
    let shape = ds.shape().map_err(|e| dataset_error(e.to_string()))?;
    let [rows, cols] = shape[..] else {
        return Err(Hdf5Error::NotTwoDimensional {
            dataset: dataset.to_string(),
            shape,
        });
    };
    let values = ds.read_f64().map_err(|e| dataset_error(e.to_string()))?;
    let stored = Array2::from_shape_vec((rows as usize, cols as usize), values)
        .map_err(|e| dataset_error(e.to_string()))?;
    let data = match layout {
        Layout::ChannelsByTime => stored,
        Layout::TimeByChannels => stored.reversed_axes().as_standard_layout().into_owned(),
    };

    let labels = labels.unwrap_or_else(|| (0..data.nrows()).map(|i| format!("ch{i}")).collect());
    Ok(MultichannelSeries::new(data, sampling_rate, labels)?)
}

fn is_numeric(dtype: &DType) -> bool {
    use DType::*;
    matches!(
        dtype,
        F32 | F64 | I8 | I16 | I32 | I64 | U8 | U16 | U32 | U64
    )
}
