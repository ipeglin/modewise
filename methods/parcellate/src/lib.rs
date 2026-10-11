//! Region signals from volumes: the mean over each atlas label's voxels (docs/volumes.md).
//!
//! As nilearn's `NiftiLabelsMasker` (resampling_target="data", strategy="mean"): the atlas is
//! resampled onto the data grid by nearest neighbour through both affines, label 0 is
//! background, and labels come out in ascending order.

use modewise_method::MethodError;
use ndarray::{Array2, ArrayView3, ArrayView4, Axis};
use tracing::warn;

/// Slack for rounding error when deciding whether a point lies inside the atlas grid.
const EDGE: f64 = 1e-9;

/// A voxel-to-world transform: world = affine · [i, j, k, 1].
pub type Affine = [[f64; 4]; 4];

/// How region signals are scaled after averaging.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Standardize {
    #[default]
    None,
    /// Each region's signal minus its mean, divided by its sample SD (ddof = 1).
    ZscoreSample,
}

/// Region signals (labels × volumes) and the label values, ascending. A label with no voxel on
/// the data grid gives a NaN row, so names from a label table stay aligned.
pub fn parcellate(
    data: ArrayView4<f32>,
    data_affine: &Affine,
    labels: ArrayView3<i32>,
    labels_affine: &Affine,
    standardize: Standardize,
) -> Result<(Array2<f64>, Vec<i32>), MethodError> {
    let to_atlas = compose(
        &invert(labels_affine)
            .ok_or_else(|| MethodError::invalid("atlas", "affine is singular"))?,
        data_affine,
    );
    let mut values: Vec<i32> = labels.iter().copied().filter(|&v| v != 0).collect();
    values.sort_unstable();
    values.dedup();
    if values.is_empty() {
        return Err(MethodError::invalid(
            "atlas",
            "has no labels besides background (0)",
        ));
    }
    let (nx, ny, nz, nt) = data.dim();
    let (ax, ay, az) = labels.dim();
    let mut sums = Array2::<f64>::zeros((values.len(), nt));
    let mut counts = vec![0usize; values.len()];
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let p = apply(&to_atlas, [i as f64, j as f64, k as f64]);
                // As scipy's order-0 affine_transform (used by nilearn and nibabel): a point
                // outside [0, n − 1] is background, and the nearest voxel is ⌊p + ½⌋.
                let shape = [ax, ay, az];
                if p.iter()
                    .zip(shape)
                    .any(|(&v, n)| v < -EDGE || v > (n - 1) as f64 + EDGE)
                {
                    continue;
                }
                let at = |d: usize| (((p[d] + 0.5).floor().max(0.0)) as usize).min(shape[d] - 1);
                let label = labels[[at(0), at(1), at(2)]];
                let Ok(row) = values.binary_search(&label) else {
                    continue;
                };
                counts[row] += 1;
                let mut sum = sums.row_mut(row);
                for (s, v) in sum.iter_mut().zip(data.slice(ndarray::s![i, j, k, ..])) {
                    *s += f64::from(*v);
                }
            }
        }
    }
    for (row, (&count, &label)) in counts.iter().zip(&values).enumerate() {
        if count == 0 {
            warn!(
                label,
                "atlas label has no voxel on the data grid; its signal is NaN"
            );
            sums.row_mut(row).fill(f64::NAN);
        } else {
            sums.row_mut(row).mapv_inplace(|v| v / count as f64);
        }
    }
    if standardize == Standardize::ZscoreSample {
        for mut row in sums.axis_iter_mut(Axis(0)) {
            let n = row.len() as f64;
            let mean = row.sum() / n;
            let sd = (row.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
            row.mapv_inplace(|v| if sd > 0.0 { (v - mean) / sd } else { 0.0 });
        }
    }
    Ok((sums, values))
}

fn apply(m: &Affine, p: [f64; 3]) -> [f64; 3] {
    let mut out = [0.0; 3];
    for (r, o) in out.iter_mut().enumerate() {
        *o = m[r][0] * p[0] + m[r][1] * p[1] + m[r][2] * p[2] + m[r][3];
    }
    out
}

fn compose(a: &Affine, b: &Affine) -> Affine {
    let mut out = [[0.0; 4]; 4];
    for (r, row) in out.iter_mut().enumerate() {
        for (c, v) in row.iter_mut().enumerate() {
            *v = (0..4).map(|k| a[r][k] * b[k][c]).sum();
        }
    }
    out
}

/// Inverse of an affine (rotation-scaling block plus translation).
#[allow(clippy::needless_range_loop)]
fn invert(m: &Affine) -> Option<Affine> {
    let a = [
        [m[0][0], m[0][1], m[0][2]],
        [m[1][0], m[1][1], m[1][2]],
        [m[2][0], m[2][1], m[2][2]],
    ];
    let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    if det.abs() < 1e-12 {
        return None;
    }
    let cof = |r: usize, c: usize| {
        let (r1, r2) = ((r + 1) % 3, (r + 2) % 3);
        let (c1, c2) = ((c + 1) % 3, (c + 2) % 3);
        a[r1][c1] * a[r2][c2] - a[r1][c2] * a[r2][c1]
    };
    let mut inv = [[0.0; 4]; 4];
    for r in 0..3 {
        for c in 0..3 {
            inv[r][c] = cof(c, r) / det;
        }
    }
    for r in 0..3 {
        inv[r][3] = -(0..3).map(|c| inv[r][c] * m[c][3]).sum::<f64>();
    }
    inv[3][3] = 1.0;
    Some(inv)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn affine_inverse_round_trips() {
        let m = [
            [2.0, 0.7, -0.4, -6.0],
            [0.5, 2.5, 0.3, -7.0],
            [0.2, -0.9, 3.0, 4.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let id = compose(&invert(&m).unwrap(), &m);
        for (r, row) in id.iter().enumerate() {
            for (c, v) in row.iter().enumerate() {
                assert!((v - if r == c { 1.0 } else { 0.0 }).abs() < 1e-12);
            }
        }
        assert!(invert(&[[0.0; 4]; 4]).is_none());
        assert_eq!(apply(&m, [1.0, 0.0, 0.0]), [-4.0, -6.5, 4.2]);
    }
}

#[cfg(test)]
mod edges {
    use ndarray::{Array3, Array4};

    use super::*;

    const IDENTITY: Affine = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];

    /// The label a single data voxel gets when it sits at atlas coordinate `x` of a 3-voxel atlas
    /// labelled 1, 2, 3; `None` for background.
    fn label_at(x: f64) -> Option<i32> {
        let data = Array4::<f32>::ones((1, 1, 1, 2));
        let atlas = Array3::from_shape_vec((3, 1, 1), vec![1, 2, 3]).unwrap();
        let mut shifted = IDENTITY;
        shifted[0][3] = x;
        let (signals, values) = parcellate(
            data.view(),
            &shifted,
            atlas.view(),
            &IDENTITY,
            Standardize::None,
        )
        .unwrap();
        values
            .iter()
            .zip(signals.rows())
            .find(|(_, r)| r[0].is_finite())
            .map(|(v, _)| *v)
    }

    #[test]
    fn bounds_are_checked_before_rounding_as_scipy_does() {
        assert_eq!(label_at(-0.3), None, "outside [0, n − 1] is background");
        assert_eq!(label_at(0.0), Some(1));
        assert_eq!(label_at(0.49), Some(1));
        assert_eq!(label_at(0.5), Some(2), "halves round up");
        assert_eq!(label_at(2.0), Some(3));
        assert_eq!(label_at(2.3), None);
        assert_eq!(
            label_at(2.0 + 1e-12),
            Some(3),
            "rounding error at the edge is tolerated"
        );
        assert_eq!(label_at(-1e-12), Some(1));
    }

    #[test]
    fn constant_signals_standardise_to_zero() {
        let data = Array4::<f32>::from_elem((1, 1, 1, 4), 3.0);
        let atlas = Array3::from_elem((1, 1, 1), 1);
        let (signals, _) = parcellate(
            data.view(),
            &IDENTITY,
            atlas.view(),
            &IDENTITY,
            Standardize::ZscoreSample,
        )
        .unwrap();
        assert!(signals.iter().all(|&v| v == 0.0));
    }
}
