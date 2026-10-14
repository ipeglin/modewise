use modewise_method::MethodError;
use modewise_regress::orthonormal_basis;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, s};

/// How the noise in each channel is modelled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoiseModel {
    /// Independent samples: ordinary least squares.
    Ols,
    /// First-order autoregressive noise: AR(1) estimated from the OLS residuals, then the data
    /// and design are prewhitened and refitted.
    Ar1,
}

/// Estimates per task regressor and channel (regressor × channel).
#[derive(Debug, Clone, PartialEq)]
pub struct Fit {
    pub betas: Array2<f64>,
    pub t: Array2<f64>,
    /// Residual degrees of freedom: samples minus the rank of the design.
    pub degrees_of_freedom: f64,
    /// The AR(1) coefficient per channel, for [`NoiseModel::Ar1`].
    pub ar1: Option<Array1<f64>>,
}

/// Fits every channel (row of `signal`) on the task regressors (regressor × sample, named by
/// `labels`) and the nuisance design (sample × column, for example from
/// `modewise_regress::design`). Nuisance columns that add nothing are dropped; a task regressor
/// that adds nothing is an error.
pub fn fit(
    signal: ArrayView2<f64>,
    task: ArrayView2<f64>,
    labels: &[String],
    nuisance: ArrayView2<f64>,
    noise: NoiseModel,
) -> Result<Fit, MethodError> {
    let n = signal.ncols();
    if task.ncols() != n || nuisance.nrows() != n || labels.len() != task.nrows() {
        return Err(MethodError::ShapeMismatch(format!(
            "{} task regressors ({} labels) of {} samples and a nuisance design of {} for a \
             signal of {n} samples",
            task.nrows(),
            labels.len(),
            task.ncols(),
            nuisance.nrows()
        )));
    }
    if [signal, task, nuisance.t()]
        .iter()
        .any(|a| a.iter().any(|v| !v.is_finite()))
    {
        return Err(MethodError::invalid(
            "signal",
            "signal and regressors must be finite",
        ));
    }
    let ols = Model::new(task.t(), nuisance, labels)?;
    let (betas, t, residuals) = ols.estimate(signal);
    if noise == NoiseModel::Ols {
        return Ok(Fit {
            betas,
            t,
            degrees_of_freedom: ols.df,
            ar1: None,
        });
    }
    let rho = residuals
        .rows()
        .into_iter()
        .map(lag_one_autocorrelation)
        .collect::<Array1<f64>>();
    if let Some(c) = rho.iter().position(|r| r.is_nan() || r.abs() >= 1.0) {
        return Err(MethodError::invalid(
            "signal",
            format!(
                "channel {c}: AR(1) coefficient {} is not below 1 in size",
                rho[c]
            ),
        ));
    }
    let (mut betas, mut t) = (betas, t);
    for (c, &r) in rho.iter().enumerate() {
        let model = Model::new(
            whiten(task.t(), r).view(),
            whiten(nuisance, r).view(),
            labels,
        )?;
        let y = whiten(signal.slice(s![c..=c, ..]).t(), r);
        let (b, tc, _) = model.estimate(y.t());
        betas.column_mut(c).assign(&b.column(0));
        t.column_mut(c).assign(&tc.column(0));
    }
    Ok(Fit {
        betas,
        t,
        degrees_of_freedom: ols.df,
        ar1: Some(rho),
    })
}

/// Yule–Walker lag-1 estimate with unbiased autocovariances, (Σ eₜeₜ₊₁ / (n−1)) / (Σ eₜ² / n),
/// as nilearn's `_yule_walker` and statsmodels' `yule_walker(method="adjusted")`. Residuals of a
/// design with a constant have mean 0, so no mean is removed. A channel fitted exactly has no
/// noise to model and gets 0.
fn lag_one_autocorrelation(e: ArrayView1<f64>) -> f64 {
    let n = e.len() as f64;
    if e.dot(&e) == 0.0 {
        return 0.0;
    }
    let lagged = e.slice(s![1..]).dot(&e.slice(s![..-1]));
    (lagged / (n - 1.0)) / (e.dot(&e) / n)
}

/// Prais–Winsten transform (rows are samples): the first row scaled by √(1−ρ²), every later
/// row minus ρ times the one before. It whitens AR(1) noise exactly.
fn whiten(x: ArrayView2<f64>, rho: f64) -> Array2<f64> {
    let mut w = x.to_owned();
    for i in (1..x.nrows()).rev() {
        let previous = x.row(i - 1);
        w.row_mut(i).scaled_add(-rho, &previous);
    }
    w.row_mut(0).mapv_inplace(|v| v * (1.0 - rho * rho).sqrt());
    w
}

/// The task regressors after removing the nuisance span, factored as Q·R (Gram–Schmidt,
/// twice), so β = R⁻¹Qᵀy and Var(β) = σ²·R⁻¹R⁻ᵀ.
struct Model {
    nuisance: Array2<f64>,
    q: Array2<f64>,
    r_inv: Array2<f64>,
    df: f64,
}

impl Model {
    fn new(
        task: ArrayView2<f64>,
        nuisance: ArrayView2<f64>,
        labels: &[String],
    ) -> Result<Self, MethodError> {
        let (n, k) = task.dim();
        let basis = orthonormal_basis(nuisance);
        let tolerance = n.max(k + basis.ncols()) as f64 * f64::EPSILON;
        let mut q = &task - &basis.dot(&basis.t().dot(&task));
        let mut r = Array2::<f64>::zeros((k, k));
        for j in 0..k {
            let mut v = q.column(j).to_owned();
            for _ in 0..2 {
                for i in 0..j {
                    let coefficient = q.column(i).dot(&v);
                    r[[i, j]] += coefficient;
                    v.scaled_add(-coefficient, &q.column(i));
                }
            }
            let norm = v.dot(&v).sqrt();
            let scale = task.column(j).dot(&task.column(j)).sqrt();
            if norm <= tolerance * scale {
                return Err(MethodError::invalid(
                    "events",
                    format!(
                        "condition {:?} cannot be estimated: its regressor is zero or a \
                         combination of the other regressors",
                        labels[j]
                    ),
                ));
            }
            r[[j, j]] = norm;
            q.column_mut(j).assign(&(v / norm));
        }
        let df = n as f64 - (basis.ncols() + k) as f64;
        if df < 1.0 {
            return Err(MethodError::invalid(
                "signal",
                format!(
                    "{} independent regressors leave no degrees of freedom in {n} samples",
                    basis.ncols() + k
                ),
            ));
        }
        Ok(Self {
            nuisance: basis,
            q,
            r_inv: upper_triangular_inverse(&r),
            df,
        })
    }

    /// Betas, t (regressor × channel) and residuals (channel × sample) for channel × sample `y`.
    fn estimate(&self, y: ArrayView2<f64>) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
        let y = &y - &y.dot(&self.nuisance).dot(&self.nuisance.t());
        let projections = y.dot(&self.q);
        let residuals = &y - &projections.dot(&self.q.t());
        let betas = self.r_inv.dot(&projections.t());
        let variance = residuals.map_axis(Axis(1), |e| e.dot(&e) / self.df);
        let spread = self.r_inv.map_axis(Axis(1), |row| row.dot(&row));
        let t = Array2::from_shape_fn(betas.dim(), |(j, c)| {
            betas[[j, c]] / (variance[c] * spread[j]).sqrt()
        });
        (betas, t, residuals)
    }
}

fn upper_triangular_inverse(r: &Array2<f64>) -> Array2<f64> {
    let k = r.nrows();
    let mut inverse = Array2::zeros((k, k));
    for j in 0..k {
        inverse[[j, j]] = 1.0 / r[[j, j]];
        for i in (0..j).rev() {
            let sum: f64 = (i + 1..=j).map(|l| r[[i, l]] * inverse[[l, j]]).sum();
            inverse[[i, j]] = -sum / r[[i, i]];
        }
    }
    inverse
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use ndarray::array;

    use super::*;

    fn labels(k: usize) -> Vec<String> {
        (0..k).map(|i| format!("c{i}")).collect()
    }

    #[test]
    fn recovers_betas_of_a_noiseless_signal() {
        let n = 50;
        let task = Array2::from_shape_fn((2, n), |(j, i)| ((i * (j + 2)) as f64 * 0.37).sin());
        let nuisance = Array2::from_shape_fn((n, 2), |(i, j)| (i as f64).powi(j as i32));
        let signal = task
            .t()
            .dot(&array![[2.0, -1.0], [0.5, 3.0]])
            .t()
            .to_owned()
            + nuisance.column(1).insert_axis(Axis(0))
            + 7.0;
        let f = fit(
            signal.view(),
            task.view(),
            &labels(2),
            nuisance.view(),
            NoiseModel::Ols,
        )
        .unwrap();
        assert!(
            (&f.betas - &array![[2.0, -1.0], [0.5, 3.0]])
                .iter()
                .all(|d| d.abs() < 1e-10)
        );
        assert_eq!(f.degrees_of_freedom, 46.0);
        assert_eq!(f.ar1, None);
    }

    #[test]
    fn whitening_inverts_an_ar1_process() {
        let e = array![0.3, -1.2, 0.8, 0.1, 2.0];
        let mut x = Array2::zeros((5, 1));
        x[[0, 0]] = e[0] / (1.0f64 - 0.25).sqrt();
        for i in 1..5 {
            x[[i, 0]] = 0.5 * x[[i - 1, 0]] + e[i];
        }
        let w = whiten(x.view(), 0.5);
        assert!((&w.column(0) - &e).iter().all(|d| d.abs() < 1e-14));
    }

    #[test]
    fn upper_triangular_inverse_is_an_inverse() {
        let r = array![[2.0, 1.0, -3.0], [0.0, 0.5, 4.0], [0.0, 0.0, -1.5]];
        let product = r.dot(&upper_triangular_inverse(&r));
        assert!(
            (&product - &Array2::<f64>::eye(3))
                .iter()
                .all(|d| d.abs() < 1e-14)
        );
    }

    #[test]
    fn lag_one_autocorrelation_matches_the_adjusted_estimator() {
        let e = array![1.0, -1.0, 2.0, 0.0];
        let expected = ((-1.0 - 2.0 + 0.0) / 3.0) / (6.0 / 4.0);
        assert!((lag_one_autocorrelation(e.view()) - expected).abs() < 1e-15);
    }

    #[test]
    fn ar1_needs_a_coefficient_below_one() {
        let ones = Array2::ones((4, 1));
        let fit_ar1 = |signal: Array2<f64>| {
            fit(
                signal.view(),
                array![[1.0, 1.0, -1.0, -1.0]].view(),
                &labels(1),
                ones.view(),
                NoiseModel::Ar1,
            )
        };
        let err = fit_ar1(array![[1.0, -1.0, 1.0, -1.0]])
            .unwrap_err()
            .to_string();
        assert!(err.contains("channel 0: AR(1) coefficient -1"), "{err}");
        let exact = fit_ar1(array![[3.0, 3.0, 1.0, 1.0]]).unwrap();
        assert_eq!(exact.ar1.unwrap()[0], 0.0);
        assert!((exact.betas[[0, 0]] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn a_regressor_in_the_nuisance_span_is_named() {
        let n = 20;
        let task =
            Array2::from_shape_fn((2, n), |(j, i)| if j == 0 { (i as f64).sin() } else { 1.0 });
        let nuisance = Array2::ones((n, 1));
        let signal = Array2::from_shape_fn((1, n), |(_, i)| (i as f64).cos());
        let err = fit(
            signal.view(),
            task.view(),
            &labels(2),
            nuisance.view(),
            NoiseModel::Ols,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("\"c1\" cannot be estimated"), "{err}");
    }

    #[test]
    fn shapes_and_values_are_checked() {
        let task = Array2::zeros((1, 4));
        let nuisance = Array2::ones((4, 1));
        for (short_signal, short_nuisance, names) in [(3, 4, 1), (4, 3, 1), (4, 4, 2)] {
            let err = fit(
                Array2::zeros((1, short_signal)).view(),
                task.view(),
                &labels(names),
                Array2::ones((short_nuisance, 1)).view(),
                NoiseModel::Ols,
            );
            assert!(matches!(err, Err(MethodError::ShapeMismatch(_))));
        }
        let mut signal = Array2::zeros((1, 4));
        signal[[0, 1]] = f64::NAN;
        let err = fit(
            signal.view(),
            task.view(),
            &labels(1),
            nuisance.view(),
            NoiseModel::Ols,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("finite"), "{err}");
        let task = array![[1.0, 0.0, 0.0]];
        let collinear_nuisance = fit(
            array![[1.0, 2.0, 4.0]].view(),
            task.view(),
            &labels(1),
            Array2::ones((3, 2)).view(),
            NoiseModel::Ols,
        );
        assert_eq!(collinear_nuisance.unwrap().degrees_of_freedom, 1.0);
        let err = fit(
            array![[1.0, 2.0]].view(),
            array![[1.0, 0.0]].view(),
            &labels(1),
            Array2::ones((2, 1)).view(),
            NoiseModel::Ols,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("no degrees of freedom"), "{err}");
    }
}
