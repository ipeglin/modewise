//! Nuisance regression: removes confound regressors, a trend and slow cosine drifts from every
//! channel in one least-squares fit (docs/regression.md).

use modewise_method::MethodError;
use ndarray::{Array1, Array2, ArrayView2, Axis};

/// What to regress out besides the given regressors. The mean is always removed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Nuisance {
    /// Add a linear trend.
    pub detrend: bool,
    /// Add discrete cosine regressors up to this frequency, a high-pass filter in the same fit.
    pub high_pass_hz: Option<f64>,
}

/// Discrete cosine (DCT-II) drift regressors, samples × order, with every frequency up to
/// `high_pass_hz`: order = ⌊2·n·high_pass/fs⌋, as in SPM's `spm_filter` and nilearn's
/// `create_cosine_drift`. Each column has unit norm.
pub fn cosine_drift(
    n_samples: usize,
    sampling_rate_hz: f64,
    high_pass_hz: f64,
) -> Result<Array2<f64>, MethodError> {
    if !(high_pass_hz.is_finite() && high_pass_hz > 0.0 && high_pass_hz < sampling_rate_hz / 2.0) {
        return Err(MethodError::invalid(
            "high_pass_hz",
            format!(
                "must be above 0 and below the Nyquist frequency {} Hz, got {high_pass_hz}",
                sampling_rate_hz / 2.0
            ),
        ));
    }
    let n = n_samples as f64;
    let order = ((2.0 * n * high_pass_hz / sampling_rate_hz).floor() as usize)
        .min(n_samples.saturating_sub(1));
    let norm = (2.0 / n).sqrt();
    Ok(Array2::from_shape_fn((n_samples, order), |(t, k)| {
        norm * (std::f64::consts::PI / n * (t as f64 + 0.5) * (k + 1) as f64).cos()
    }))
}

/// The design (samples × columns): constant, then the linear trend, cosine drifts, and the
/// regressors (regressor × sample) in their given order.
pub fn design(
    nuisance: Nuisance,
    n_samples: usize,
    sampling_rate_hz: f64,
    regressors: ArrayView2<f64>,
) -> Result<Array2<f64>, MethodError> {
    if regressors.ncols() != n_samples {
        return Err(MethodError::ShapeMismatch(format!(
            "regressors have {} samples, the signal {n_samples}",
            regressors.ncols()
        )));
    }
    let mut columns = vec![Array2::ones((n_samples, 1))];
    if nuisance.detrend {
        let centre = (n_samples as f64 - 1.0) / 2.0;
        columns.push(Array2::from_shape_fn((n_samples, 1), |(t, _)| {
            t as f64 - centre
        }));
    }
    if let Some(hz) = nuisance.high_pass_hz {
        columns.push(cosine_drift(n_samples, sampling_rate_hz, hz)?);
    }
    columns.push(regressors.t().to_owned());
    let views: Vec<_> = columns.iter().map(|c| c.view()).collect();
    ndarray::concatenate(Axis(1), &views).map_err(|e| MethodError::ShapeMismatch(e.to_string()))
}

/// Each channel (row of `signal`) minus its least-squares fit on the design columns.
/// Columns that add nothing to the span of the others (collinear or all zero) are dropped, as
/// nilearn's `signal.clean` does with a pivoted QR.
pub fn residuals(
    signal: ArrayView2<f64>,
    design: ArrayView2<f64>,
) -> Result<Array2<f64>, MethodError> {
    if design.nrows() != signal.ncols() {
        return Err(MethodError::ShapeMismatch(format!(
            "design has {} samples, the signal {}",
            design.nrows(),
            signal.ncols()
        )));
    }
    if design.iter().chain(signal.iter()).any(|v| !v.is_finite()) {
        return Err(MethodError::invalid(
            "regressors",
            "signal and regressors must be finite",
        ));
    }
    let q = orthonormal_basis(design);
    if q.ncols() >= signal.ncols() {
        return Err(MethodError::invalid(
            "regressors",
            format!(
                "{} independent regressors leave nothing of {} samples",
                q.ncols(),
                signal.ncols()
            ),
        ));
    }
    Ok(&signal - &signal.dot(&q).dot(&q.t()))
}

/// Orthonormal basis (samples × rank) of the columns' span by Gram–Schmidt with column
/// pivoting, orthogonalised twice per column. Columns are scaled to unit norm first, so the
/// rank tolerance, max(n, p)·ε as in numpy's `matrix_rank`, does not depend on their units.
fn orthonormal_basis(design: ArrayView2<f64>) -> Array2<f64> {
    let (n, p) = design.dim();
    let tolerance = n.max(p) as f64 * f64::EPSILON;
    let mut remaining: Vec<Array1<f64>> = design
        .columns()
        .into_iter()
        .filter_map(|c| {
            let norm = c.dot(&c).sqrt();
            (norm > 0.0).then(|| &c / norm)
        })
        .collect();
    let mut basis: Vec<Array1<f64>> = Vec::new();
    while !remaining.is_empty() {
        let norms: Vec<f64> = remaining.iter().map(|c| c.dot(c).sqrt()).collect();
        let (best, &norm) = norms
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap_or((0, &0.0));
        if norm <= tolerance {
            break;
        }
        let mut q = remaining.swap_remove(best);
        for b in &basis {
            q = &q - &(b * q.dot(b));
        }
        q /= q.dot(&q).sqrt();
        for c in &mut remaining {
            *c = &*c - &(&q * c.dot(&q));
        }
        basis.push(q);
    }
    let mut q = Array2::zeros((n, basis.len()));
    for (k, b) in basis.into_iter().enumerate() {
        q.column_mut(k).assign(&b);
    }
    q
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use ndarray::{Array1, Array2, array, stack};

    #[test]
    fn cosine_order_follows_the_cutoff() {
        // 2 · 100 · 0.01 / 1.0 = 2 cosines.
        assert_eq!(cosine_drift(100, 1.0, 0.01).unwrap().ncols(), 2);
        assert_eq!(cosine_drift(100, 1.0, 0.0099).unwrap().ncols(), 1);
        let c = cosine_drift(100, 1.0, 0.01).unwrap();
        for col in c.columns() {
            assert!((col.dot(&col) - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn cosine_cutoff_must_be_below_nyquist() {
        for hz in [0.0, -1.0, 0.5, f64::NAN] {
            assert!(matches!(
                cosine_drift(100, 1.0, hz),
                Err(MethodError::InvalidParameter {
                    name: "high_pass_hz",
                    ..
                })
            ));
        }
    }

    #[test]
    fn design_orders_columns() {
        let regressors = array![[1.0, 2.0, 3.0, 4.0]];
        let nuisance = Nuisance {
            detrend: true,
            high_pass_hz: None,
        };
        let d = design(nuisance, 4, 1.0, regressors.view()).unwrap();
        assert_eq!(d.column(0).to_vec(), vec![1.0; 4]);
        assert_eq!(d.column(1).to_vec(), vec![-1.5, -0.5, 0.5, 1.5]);
        assert_eq!(d.column(2).to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn design_rejects_other_lengths() {
        let nuisance = Nuisance {
            detrend: false,
            high_pass_hz: None,
        };
        let err = design(nuisance, 5, 1.0, Array2::zeros((1, 4)).view()).unwrap_err();
        assert!(matches!(err, MethodError::ShapeMismatch(_)));
        let err = residuals(Array2::zeros((1, 5)).view(), Array2::ones((4, 1)).view());
        assert!(matches!(err, Err(MethodError::ShapeMismatch(_))));
    }

    #[test]
    fn collinear_and_zero_columns_are_dropped() {
        let x = array![
            [1.0, 0.0, 2.0, 0.0],
            [1.0, 0.0, 2.0, 1.0],
            [1.0, 0.0, 2.0, 0.0],
            [1.0, 0.0, 2.0, 3.0]
        ];
        let q = orthonormal_basis(x.view());
        assert_eq!(q.ncols(), 2);
        let gram = q.t().dot(&q);
        assert!(
            (&gram - &Array2::<f64>::eye(2))
                .iter()
                .all(|v| v.abs() < 1e-12)
        );
    }

    #[test]
    fn units_of_the_regressors_do_not_matter() {
        let signal = array![[1.0, 4.0, 2.0, 8.0, 5.0, 7.0]];
        let v = array![0.5, -0.1, 0.9, 0.3, -0.2, 0.4];
        let fit = |scale: f64| {
            let x = stack![Axis(1), Array1::ones(6), &v * scale];
            residuals(signal.view(), x.view()).unwrap()
        };
        let reference = fit(1.0);
        for scale in [1e-14, 1e14] {
            assert!((&fit(scale) - &reference).iter().all(|d| d.abs() < 1e-10));
        }
    }

    #[test]
    fn nearly_collinear_columns_stay_orthonormal() {
        let a = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = array![0.3, -0.7, 0.1, 0.9, -0.4, 0.2, -0.8, 0.5];
        let x = stack![Axis(1), a, &a + &(&b * 1e-7)];
        let q = orthonormal_basis(x.view());
        let gram = q.t().dot(&q);
        let error = (&gram - &Array2::<f64>::eye(q.ncols()))
            .iter()
            .fold(0.0f64, |m, v| m.max(v.abs()));
        assert!(error < 1e-13, "{error}");
    }

    #[test]
    fn rejects_non_finite_values_and_full_rank() {
        let mut y = Array2::zeros((1, 3));
        y[[0, 1]] = f64::NAN;
        assert!(residuals(y.view(), Array2::ones((3, 1)).view()).is_err());
        let full = Array2::eye(3);
        assert!(residuals(Array2::ones((1, 3)).view(), full.view()).is_err());
    }
}
