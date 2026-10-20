//! Least squares for many features sharing one design, tested by a contrast (Winkler et al. 2014).

use modewise_method::MethodError;
use ndarray::{Array1, Array2, ArrayView2, Axis};

/// A design matrix (observations × regressors) and a contrast (rows × regressors). One row gives
/// t, several give F. With variance groups, t becomes Aspin–Welch v and F becomes G.
#[derive(Debug, Clone)]
pub struct Glm {
    design: Array2<f64>,
    contrast: Array2<f64>,
    /// (MᵀM)⁻¹Mᵀ, so coefficients are one product per rearrangement.
    pinv: Array2<f64>,
    /// (C(MᵀM)⁻¹Cᵀ)⁻¹.
    precision: Array2<f64>,
    /// Projection onto the nuisance space {Mb : Cb = 0}; `None` if it is empty.
    nuisance: Option<Array2<f64>>,
    variance: Option<VarianceGroups>,
}

/// Groups of observations with their own error variance.
#[derive(Debug, Clone)]
struct VarianceGroups {
    group: Vec<usize>,
    /// Sum of the residual-forming matrix's diagonal over each group.
    trace: Vec<f64>,
    /// Size of each group.
    size: Vec<f64>,
    /// MᵀM over each group's rows.
    gram: Vec<Array2<f64>>,
}

impl Glm {
    pub fn new(design: Array2<f64>, contrast: Array2<f64>) -> Result<Self, MethodError> {
        let (n, p) = design.dim();
        if contrast.ncols() != p || contrast.nrows() == 0 {
            return Err(MethodError::invalid(
                "contrast",
                format!(
                    "needs at least one row of {p} weights, one per regressor; got {} × {}",
                    contrast.nrows(),
                    contrast.ncols()
                ),
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
        let null = null_space(&contrast)
            .ok_or_else(|| MethodError::invalid("contrast", "rows must be linearly independent"))?;
        let precision = invert(&contrast.dot(&inverse).dot(&contrast.t()))
            .ok_or_else(|| MethodError::invalid("contrast", "rows must be linearly independent"))?;
        let nuisance = if null.ncols() == 0 {
            None
        } else {
            let z = design.dot(&null);
            let zz = invert(&z.t().dot(&z)).ok_or_else(|| {
                MethodError::invalid("design", "nuisance regressors are collinear")
            })?;
            Some(z.dot(&zz).dot(&z.t()))
        };
        Ok(Self {
            pinv: inverse.dot(&design.t()),
            precision,
            design,
            contrast,
            nuisance,
            variance: None,
        })
    }

    /// Gives each group (0, 1, … per observation) its own error variance (Winkler et al. 2014).
    pub fn with_variance_groups(mut self, group: Vec<usize>) -> Result<Self, MethodError> {
        let n = self.n_observations();
        if group.len() != n {
            return Err(MethodError::ShapeMismatch(format!(
                "{} variance groups for {n} observations",
                group.len()
            )));
        }
        let count = group.iter().max().map_or(0, |g| g + 1);
        let leverage = (&self.design * &self.pinv.t()).sum_axis(Axis(1));
        let mut trace = vec![0.0; count];
        let mut size = vec![0.0; count];
        let mut gram = vec![Array2::zeros((self.design.ncols(), self.design.ncols())); count];
        for (i, &g) in group.iter().enumerate() {
            trace[g] += 1.0 - leverage[i];
            size[g] += 1.0;
            let row = self.design.row(i);
            gram[g] += &(row.insert_axis(Axis(1)).dot(&row.insert_axis(Axis(0))));
        }
        if count < 2 {
            return Err(MethodError::invalid(
                "variance groups",
                "need at least two; with one, use the GLM without them",
            ));
        }
        if let Some(g) = (0..count).find(|&g| trace[g] <= 1e-9) {
            return Err(MethodError::invalid(
                "variance groups",
                format!(
                    "group {g} has no residual degrees of freedom (empty, or fitted exactly by the design)"
                ),
            ));
        }
        self.variance = Some(VarianceGroups {
            group,
            trace,
            size,
            gram,
        });
        Ok(self)
    }

    pub fn n_observations(&self) -> usize {
        self.design.nrows()
    }

    /// Rows of the contrast: 1 for t or v, more for F or G.
    pub fn rank(&self) -> usize {
        self.contrast.nrows()
    }

    /// "t", "F", "v" or "G".
    pub fn name(&self) -> &'static str {
        match (self.variance.is_some(), self.rank() == 1) {
            (false, true) => "t",
            (false, false) => "F",
            (true, true) => "v",
            (true, false) => "G",
        }
    }

    /// The statistic for each feature (column of `y`).
    pub fn statistic(&self, y: ArrayView2<f64>) -> Array1<f64> {
        let beta = self.pinv.dot(&y);
        let residuals = &y - &self.design.dot(&beta);
        let effect = self.contrast.dot(&beta);
        match &self.variance {
            None => {
                let df = self.residual_df();
                let rss = residuals.map_axis(Axis(0), |r| r.dot(&r));
                Array1::from_shape_fn(y.ncols(), |f| {
                    let e = effect.column(f);
                    let variance = rss[f] / df;
                    if self.rank() == 1 {
                        e[0] / (variance / self.precision[[0, 0]]).sqrt()
                    } else {
                        e.dot(&self.precision.dot(&e)) / self.rank() as f64 / variance
                    }
                })
            }
            Some(vg) => Array1::from_shape_fn(y.ncols(), |f| {
                let weights = vg.weights(residuals.column(f).iter().copied());
                let mut weighted = Array2::zeros(vg.gram[0].raw_dim());
                for (gram, w) in vg.gram.iter().zip(&weights) {
                    weighted.scaled_add(*w, gram);
                }
                let Some(covariance) = invert(&weighted)
                    .and_then(|c| invert(&self.contrast.dot(&c).dot(&self.contrast.t())))
                else {
                    return f64::NAN;
                };
                let e = effect.column(f);
                let s = self.rank() as f64;
                if self.rank() == 1 {
                    e[0] * covariance[[0, 0]].sqrt()
                } else {
                    let spread = vg.spread(&weights);
                    let lambda = 1.0 + 2.0 * (s - 1.0) / (s * (s + 2.0)) * spread;
                    e.dot(&covariance.dot(&e)) / (s * lambda)
                }
            }),
        }
    }

    /// Denominator degrees of freedom per feature: n − rank(M) without variance groups; with
    /// them, Welch's approximation s(s + 2) / (3 Σ), which for one row is Welch–Satterthwaite.
    pub fn degrees_of_freedom(&self, y: ArrayView2<f64>) -> Array1<f64> {
        match &self.variance {
            None => Array1::from_elem(y.ncols(), self.residual_df()),
            Some(vg) => {
                let residuals = &y - &self.design.dot(&self.pinv.dot(&y));
                let s = self.rank() as f64;
                residuals.map_axis(Axis(0), |r| {
                    s * (s + 2.0) / (3.0 * vg.spread(&vg.weights(r.iter().copied())))
                })
            }
        }
    }

    /// Residuals of each feature after the nuisance regressors (Freedman & Lane 1983).
    pub fn nuisance_residuals(&self, y: ArrayView2<f64>) -> Array2<f64> {
        match &self.nuisance {
            Some(h) => &y - &h.dot(&y),
            None => y.to_owned(),
        }
    }

    fn residual_df(&self) -> f64 {
        (self.design.nrows() - self.design.ncols()) as f64
    }
}

impl VarianceGroups {
    /// Per group, its residual degrees of freedom over its residual sum of squares: one over
    /// its variance (Horn et al. 1975).
    fn weights(&self, residuals: impl Iterator<Item = f64>) -> Vec<f64> {
        let mut rss = vec![0.0; self.trace.len()];
        for (r, &g) in residuals.zip(&self.group) {
            rss[g] += r * r;
        }
        self.trace.iter().zip(&rss).map(|(t, r)| t / r).collect()
    }

    /// Σ over groups of (1 − share of the total weight)² / residual degrees of freedom.
    fn spread(&self, weights: &[f64]) -> f64 {
        let total: f64 = weights.iter().zip(&self.size).map(|(w, n)| w * n).sum();
        (0..weights.len())
            .map(|g| (1.0 - weights[g] * self.size[g] / total).powi(2) / self.trace[g])
            .sum()
    }
}

/// Orthonormal basis (as columns) of the vectors b with Cb = 0; `None` when the rows of C are
/// linearly dependent.
fn null_space(contrast: &Array2<f64>) -> Option<Array2<f64>> {
    let p = contrast.ncols();
    let mut basis: Vec<Array1<f64>> = Vec::new();
    let reduce = |v: Array1<f64>, basis: &mut Vec<Array1<f64>>| {
        let scale = v.dot(&v).sqrt();
        let mut v = v;
        for _ in 0..2 {
            for b in basis.iter() {
                let along = v.dot(b);
                v.scaled_add(-along, b);
            }
        }
        let norm = v.dot(&v).sqrt();
        (norm > 1e-10 * scale).then(|| v / norm)
    };
    for row in contrast.rows() {
        let v = reduce(row.to_owned(), &mut basis)?;
        basis.push(v);
    }
    let rows = basis.len();
    for j in 0..p {
        if basis.len() == p {
            break;
        }
        if let Some(v) = reduce(Array1::from_shape_fn(p, |k| f64::from(k == j)), &mut basis) {
            basis.push(v);
        }
    }
    Some(Array2::from_shape_fn((p, p - rows), |(k, c)| {
        basis[rows + c][k]
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

    fn tested(j: usize, p: usize) -> Array2<f64> {
        Array2::from_shape_fn((1, p), |(_, k)| f64::from(k == j))
    }

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
        assert!(Glm::new(collinear, tested(1, 2)).is_err());
        assert!(
            Glm::new(Array2::ones((2, 2)), tested(0, 2)).is_err(),
            "no error degrees of freedom"
        );
        assert!(Glm::new(Array2::ones((4, 1)), tested(1, 2)).is_err());
        assert!(Glm::new(Array2::ones((4, 1)), Array2::zeros((0, 1))).is_err());
        let design = array![[1.0, 0.0], [1.0, 1.0], [1.0, 0.0], [1.0, 1.0], [1.0, 0.5]];
        assert!(
            Glm::new(design.clone(), array![[0.0, 1.0], [0.0, 2.0]]).is_err(),
            "dependent contrast rows"
        );
        assert!(Glm::new(design.clone(), array![[0.0, 0.0]]).is_err());
        let glm = Glm::new(design, tested(1, 2)).unwrap();
        assert!(glm.clone().with_variance_groups(vec![0; 4]).is_err());
        assert!(
            glm.clone().with_variance_groups(vec![0; 5]).is_err(),
            "one group"
        );
        assert!(
            glm.with_variance_groups(vec![0, 0, 0, 0, 2]).is_err(),
            "group 1 is empty"
        );
    }

    #[test]
    fn one_sample_t_by_hand() {
        let y = array![[1.0], [2.0], [3.0], [6.0]];
        let glm = Glm::new(Array2::ones((4, 1)), array![[1.0]]).unwrap();
        // mean 3, sd sqrt(14/3), t = 3 / (sd / 2).
        let expected = 3.0 / ((14.0f64 / 3.0).sqrt() / 2.0);
        assert!((glm.statistic(y.view())[0] - expected).abs() < 1e-12);
        assert_eq!(glm.nuisance_residuals(y.view()), y);
        assert_eq!(glm.degrees_of_freedom(y.view())[0], 3.0);
        assert_eq!((glm.n_observations(), glm.rank(), glm.name()), (4, 1, "t"));
    }

    #[test]
    fn null_space_is_orthogonal_to_the_contrast() {
        let c = array![[0.0, 1.0, -1.0, 0.0], [1.0, 1.0, 0.0, 0.0]];
        let n = null_space(&c).unwrap();
        assert_eq!(n.dim(), (4, 2));
        assert!(c.dot(&n).iter().all(|v| v.abs() < 1e-14));
        assert!(
            (&n.t().dot(&n) - &Array2::<f64>::eye(2))
                .iter()
                .all(|v| v.abs() < 1e-14)
        );
    }

    #[test]
    fn f_of_one_row_is_t_squared_and_names_follow_the_test() {
        let design = array![
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 0.5],
            [1.0, 0.2]
        ];
        let y = array![[1.0], [2.5], [0.7], [3.0], [1.9], [1.1]];
        let t = Glm::new(design.clone(), tested(1, 2)).unwrap();
        let f = Glm::new(design.clone(), array![[0.0, 1.0], [1.0, 0.0]]).unwrap();
        assert_eq!((f.name(), f.rank()), ("F", 2));
        let groups = vec![0, 0, 0, 1, 1, 1];
        let v = t.clone().with_variance_groups(groups.clone()).unwrap();
        let g = f.with_variance_groups(groups).unwrap();
        assert_eq!((v.name(), g.name()), ("v", "G"));
        assert!(v.degrees_of_freedom(y.view())[0] > 0.0);
        let scaled = Glm::new(design, array![[0.0, 2.0]]).unwrap();
        assert!(
            (scaled.statistic(y.view())[0].powi(2) - t.statistic(y.view())[0].powi(2)).abs()
                < 1e-12,
            "scaling a contrast does not change t²"
        );
    }
}
