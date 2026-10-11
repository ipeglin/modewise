//! Ordinary least squares for many features sharing one design.

use modewise_method::MethodError;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};

/// A design matrix (observations × regressors) and the regressor being tested.
#[derive(Debug, Clone)]
pub struct Glm {
    design: Array2<f64>,
    tested: usize,
    /// (MᵀM)⁻¹Mᵀ, so coefficients are one product per permutation.
    pinv: Array2<f64>,
    /// [(MᵀM)⁻¹]_jj for the tested regressor.
    scale: f64,
    /// Projection onto the nuisance regressors (all but the tested one); `None` if there are none.
    nuisance: Option<Array2<f64>>,
}

impl Glm {
    pub fn new(design: Array2<f64>, tested: usize) -> Result<Self, MethodError> {
        let (n, p) = design.dim();
        if tested >= p {
            return Err(MethodError::invalid(
                "tested",
                format!("regressor {tested} does not exist; there are {p}"),
            ));
        }
        if n <= p {
            return Err(MethodError::invalid(
                "design",
                format!("{n} observations cannot fit {p} regressors with error left over"),
            ));
        }
        let inverse = invert(&design.t().dot(&design))
            .ok_or_else(|| MethodError::invalid("design", "regressors are collinear"))?;
        let nuisance_columns: Vec<usize> = (0..p).filter(|&c| c != tested).collect();
        let nuisance = if nuisance_columns.is_empty() {
            None
        } else {
            let z = design.select(Axis(1), &nuisance_columns);
            let zz = invert(&z.t().dot(&z)).ok_or_else(|| {
                MethodError::invalid("design", "nuisance regressors are collinear")
            })?;
            Some(z.dot(&zz).dot(&z.t()))
        };
        Ok(Self {
            pinv: inverse.dot(&design.t()),
            scale: inverse[[tested, tested]],
            design,
            tested,
            nuisance,
        })
    }

    pub fn n_observations(&self) -> usize {
        self.design.nrows()
    }

    pub fn degrees_of_freedom(&self) -> f64 {
        (self.design.nrows() - self.design.ncols()) as f64
    }

    /// t of the tested coefficient for each feature (column of `y`).
    pub fn t(&self, y: ArrayView2<f64>) -> Array1<f64> {
        let beta = self.pinv.dot(&y);
        let residuals = &y - &self.design.dot(&beta);
        let df = self.degrees_of_freedom();
        let rss = residuals.map_axis(Axis(0), |r| r.dot(&r));
        Array1::from_shape_fn(y.ncols(), |f| {
            beta[[self.tested, f]] / (rss[f] / df * self.scale).sqrt()
        })
    }

    /// Residuals of each feature after the nuisance regressors (Freedman & Lane 1983).
    pub fn nuisance_residuals(&self, y: ArrayView2<f64>) -> Array2<f64> {
        match &self.nuisance {
            Some(h) => &y - &h.dot(&y),
            None => y.to_owned(),
        }
    }
}

/// Mean, variance of the mean, and size of one group in a column.
fn group(column: ArrayView1<f64>, in_a: &[bool], member: bool) -> (f64, f64, f64) {
    let values: Vec<f64> = column
        .iter()
        .zip(in_a)
        .filter(|(_, g)| **g == member)
        .map(|(v, _)| *v)
        .collect();
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
    (mean, var / n, n)
}

/// Welch's t of group `a` minus the rest, for each feature (column of `y`).
pub fn welch_t(y: ArrayView2<f64>, in_a: &[bool]) -> Array1<f64> {
    Array1::from_iter(y.columns().into_iter().map(|c| {
        let ((ma, va, _), (mb, vb, _)) = (group(c, in_a, true), group(c, in_a, false));
        (ma - mb) / (va + vb).sqrt()
    }))
}

/// Welch–Satterthwaite degrees of freedom per feature.
pub fn welch_df(y: ArrayView2<f64>, in_a: &[bool]) -> Array1<f64> {
    Array1::from_iter(y.columns().into_iter().map(|c| {
        let ((_, a, na), (_, b, nb)) = (group(c, in_a, true), group(c, in_a, false));
        (a + b).powi(2) / (a * a / (na - 1.0) + b * b / (nb - 1.0))
    }))
}

/// Inverse of a small symmetric positive definite matrix by Gauss–Jordan elimination with
/// partial pivoting; `None` when (nearly) singular.
fn invert(m: &Array2<f64>) -> Option<Array2<f64>> {
    let n = m.nrows();
    let mut a = m.clone();
    let mut inv = Array2::eye(n);
    let scale = m.iter().fold(0.0f64, |s, v| s.max(v.abs()));
    for col in 0..n {
        let pivot = (col..n).max_by(|&i, &j| a[[i, col]].abs().total_cmp(&a[[j, col]].abs()))?;
        if a[[pivot, col]].abs() <= 1e-12 * scale {
            return None;
        }
        for k in 0..n {
            a.swap([pivot, k], [col, k]);
            inv.swap([pivot, k], [col, k]);
        }
        let d = a[[col, col]];
        for k in 0..n {
            a[[col, k]] /= d;
            inv[[col, k]] /= d;
        }
        for row in 0..n {
            if row != col {
                let f = a[[row, col]];
                for k in 0..n {
                    a[[row, k]] -= f * a[[col, k]];
                    inv[[row, k]] -= f * inv[[col, k]];
                }
            }
        }
    }
    Some(inv)
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    #[test]
    fn inversion_and_rejections() {
        let inv = invert(&array![[4.0, 2.0], [2.0, 3.0]]).unwrap();
        assert!(
            (&inv.dot(&array![[4.0, 2.0], [2.0, 3.0]]) - &Array2::<f64>::eye(2))
                .iter()
                .all(|v| v.abs() < 1e-14)
        );
        assert!(invert(&array![[1.0, 2.0], [2.0, 4.0]]).is_none());
        let collinear = array![[1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0]];
        assert!(Glm::new(collinear, 1).is_err());
        assert!(
            Glm::new(Array2::ones((2, 2)), 0).is_err(),
            "no error degrees of freedom"
        );
        assert!(Glm::new(Array2::ones((4, 1)), 1).is_err());
        let with_nuisance = array![
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 0.5, 1.0]
        ];
        assert!(
            Glm::new(with_nuisance, 1).is_err(),
            "intercept duplicated in the nuisance"
        );
    }

    #[test]
    fn one_sample_t_by_hand() {
        let y = array![[1.0], [2.0], [3.0], [6.0]];
        let glm = Glm::new(Array2::ones((4, 1)), 0).unwrap();
        // mean 3, sd sqrt(14/3), t = 3 / (sd / 2).
        let expected = 3.0 / ((14.0f64 / 3.0).sqrt() / 2.0);
        assert!((glm.t(y.view())[0] - expected).abs() < 1e-12);
        assert_eq!(glm.nuisance_residuals(y.view()), y);
        assert_eq!((glm.n_observations(), glm.degrees_of_freedom()), (4, 3.0));
    }
}
