use modewise_hilbert::{Padding, analytic_signal};
use ndarray::{Array2, ArrayView2, Axis};

/// Pearson correlation between the rows (channels) of a channel × sample array.
/// A channel with zero variance gets NaN in its row and column.
pub fn correlation(x: ArrayView2<f64>) -> Array2<f64> {
    let n = x.ncols() as f64;
    let mut z = x.to_owned();
    let mut flat = vec![false; x.nrows()];
    for (mut row, flat) in z.axis_iter_mut(Axis(0)).zip(&mut flat) {
        let mean = row.sum() / n;
        row.mapv_inplace(|v| v - mean);
        let norm = row.dot(&row).sqrt();
        if norm > 0.0 {
            row /= norm;
        } else {
            *flat = true;
        }
    }
    let mut r = z.dot(&z.t()).mapv(|v| v.clamp(-1.0, 1.0));
    for i in 0..r.nrows() {
        r[[i, i]] = 1.0;
    }
    mask_rows(&mut r, &flat);
    r
}

/// Fisher z = atanh(r). The diagonal is undefined (r = 1) and becomes NaN.
pub(crate) fn fisher_z(r: Array2<f64>) -> Array2<f64> {
    let mut z = r.mapv(f64::atanh);
    z.diag_mut().fill(f64::NAN);
    z
}

/// Unit phase vectors `e^{iφ}` of each row's analytic signal, as real and imaginary parts,
/// and which rows are all zero (no phase).
pub fn unit_phases(x: ArrayView2<f64>, padding: Padding) -> (Array2<f64>, Array2<f64>, Vec<bool>) {
    let (rows, n) = x.dim();
    let (mut re, mut im) = (Array2::zeros((rows, n)), Array2::zeros((rows, n)));
    let mut silent = vec![false; rows];
    for (c, row) in x.axis_iter(Axis(0)).enumerate() {
        let z = analytic_signal(row, padding);
        silent[c] = z.iter().all(|v| v.norm() == 0.0);
        for (t, v) in z.iter().enumerate() {
            let a = v.norm();
            if a > 0.0 {
                re[[c, t]] = v.re / a;
                im[[c, t]] = v.im / a;
            }
        }
    }
    (re, im, silent)
}

/// Phase locking value `|mean_t e^{i(φ_a − φ_b)}|` for every pair of rows of unit phase vectors.
pub fn phase_locking(re: &ArrayView2<f64>, im: &ArrayView2<f64>, silent: &[bool]) -> Array2<f64> {
    let n = re.ncols() as f64;
    let real = re.dot(&re.t()) + im.dot(&im.t());
    let imaginary = im.dot(&re.t()) - re.dot(&im.t());
    let mut plv = Array2::from_shape_fn(real.dim(), |(i, j)| {
        (real[[i, j]].hypot(imaginary[[i, j]]) / n).min(1.0)
    });
    mask_rows(&mut plv, silent);
    plv
}

fn mask_rows(m: &mut Array2<f64>, masked: &[bool]) {
    for (i, &bad) in masked.iter().enumerate() {
        if bad {
            m.row_mut(i).fill(f64::NAN);
            m.column_mut(i).fill(f64::NAN);
        }
    }
}
