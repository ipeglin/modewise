//! NIfTI-1 volumes (BOLD runs and label atlases) with their voxel-to-world affines.

use std::path::Path;

use ndarray::{Array3, Array4, Ix3, Ix4};
use nifti::{IntoNdArray, NiftiHeader, NiftiObject, ReaderOptions};
use thiserror::Error;

pub type Affine = [[f64; 4]; 4];

#[derive(Debug, Error)]
pub enum NiftiError {
    #[error("{path}: {message}")]
    Read { path: String, message: String },
}

/// A 4D run: voxels × volumes, its affine, and the repetition time if the header gives one.
pub struct Bold {
    pub data: Array4<f32>,
    pub affine: Affine,
    pub repetition_time_s: Option<f64>,
}

/// A 3D label volume and its affine.
pub struct Labels {
    pub data: Array3<i32>,
    pub affine: Affine,
}

fn error(path: &Path, message: impl ToString) -> NiftiError {
    NiftiError::Read {
        path: path.display().to_string(),
        message: message.to_string(),
    }
}

pub fn read_bold(path: &Path) -> Result<Bold, NiftiError> {
    let object = ReaderOptions::new()
        .read_file(path)
        .map_err(|e| error(path, e))?;
    let header = object.header().clone();
    let data = object
        .into_volume()
        .into_ndarray::<f32>()
        .map_err(|e| error(path, e))?
        .into_dimensionality::<Ix4>()
        .map_err(|_| error(path, "a BOLD run must be 4D (x, y, z, time)"))?;
    Ok(Bold {
        data,
        affine: affine(&header),
        repetition_time_s: repetition_time(&header),
    })
}

pub fn read_labels(path: &Path) -> Result<Labels, NiftiError> {
    let object = ReaderOptions::new()
        .read_file(path)
        .map_err(|e| error(path, e))?;
    let header = object.header().clone();
    let volume = object
        .into_volume()
        .into_ndarray::<f64>()
        .map_err(|e| error(path, e))?;
    let volume = match volume.ndim() {
        3 => volume,
        4 if volume.shape()[3] == 1 => volume.index_axis_move(ndarray::Axis(3), 0),
        _ => return Err(error(path, "a label atlas must be 3D")),
    };
    let data = volume
        .into_dimensionality::<Ix3>()
        .map_err(|_| error(path, "a label atlas must be 3D"))?;
    if data.iter().any(|v| v.fract() != 0.0) {
        return Err(error(path, "label values must be whole numbers"));
    }
    Ok(Labels {
        data: data.mapv(|v| v as i32),
        affine: affine(&header),
    })
}

/// The sform when its code is set, otherwise the qform (as nibabel and nilearn choose).
#[allow(clippy::needless_range_loop)]
pub fn affine(h: &NiftiHeader) -> Affine {
    if h.sform_code > 0 {
        let row = |r: [f32; 4]| r.map(f64::from);
        return [
            row(h.srow_x),
            row(h.srow_y),
            row(h.srow_z),
            [0.0, 0.0, 0.0, 1.0],
        ];
    }
    let (b, c, d) = (
        f64::from(h.quatern_b),
        f64::from(h.quatern_c),
        f64::from(h.quatern_d),
    );
    let a = (1.0 - (b * b + c * c + d * d)).max(0.0).sqrt();
    let rotation = [
        [
            a * a + b * b - c * c - d * d,
            2.0 * (b * c - a * d),
            2.0 * (b * d + a * c),
        ],
        [
            2.0 * (b * c + a * d),
            a * a + c * c - b * b - d * d,
            2.0 * (c * d - a * b),
        ],
        [
            2.0 * (b * d - a * c),
            2.0 * (c * d + a * b),
            a * a + d * d - b * b - c * c,
        ],
    ];
    let qfac = if h.pixdim[0] < 0.0 { -1.0 } else { 1.0 };
    let scale = [
        f64::from(h.pixdim[1]),
        f64::from(h.pixdim[2]),
        f64::from(h.pixdim[3]) * qfac,
    ];
    let offset = [
        f64::from(h.quatern_x),
        f64::from(h.quatern_y),
        f64::from(h.quatern_z),
    ];
    let mut m = [[0.0; 4]; 4];
    for r in 0..3 {
        for col in 0..3 {
            m[r][col] = rotation[r][col] * scale[col];
        }
        m[r][3] = offset[r];
    }
    m[3][3] = 1.0;
    m
}

/// pixdim[4] in seconds, from the header's time units (seconds, milliseconds or microseconds).
fn repetition_time(h: &NiftiHeader) -> Option<f64> {
    let tr = f64::from(h.pixdim[4]);
    let factor = match h.xyzt_units & 0x38 {
        8 => 1.0,
        16 => 1e-3,
        24 => 1e-6,
        _ => return None,
    };
    (tr > 0.0).then_some(tr * factor)
}
