use std::sync::atomic::{AtomicUsize, Ordering};

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut1, Zip};
use rayon::prelude::*;
use tracing::warn;

/// The reference's per-call budget (scikit-learn `max_iter=1000`).
const MAX_SWEEPS: usize = 1000;
const RELATIVE_STEP: f64 = 1e-13;
const KKT_SLACK: f64 = 1e-10;
const GAP_EVERY: usize = 32;
const GAP: f64 = 1e-14;

/// Sparse loadings for every channel: column `c` of `loadings` (latent × channel) minimises
/// `(1/2T)‖x_c − Zᵀa‖² + penalty·‖a‖₁`, where `latents` is Z (latent × sample) and `signal`
/// is X (channel × sample). This is scikit-learn's `Lasso(alpha=penalty, fit_intercept=False)`.
///
/// Cyclic coordinate descent on the Gram matrix, warm-started from `loadings`, one channel per
/// thread. A channel stops at the first of:
/// - after sweeps 1, 2, 4, 8, …: the optimality conditions, solved exactly for the current signs, hold
/// - no coefficient moves by more than 1e-13 of the largest one
/// - every 32 sweeps: the duality gap is at most 1e-14 of the channel's mean power
/// - 1000 sweeps. Nearly collinear latents can need more; a warning then says how many channels stopped early.
pub fn lasso(
    latents: ArrayView2<f64>,
    signal: ArrayView2<f64>,
    penalty: f64,
    loadings: &mut Array2<f64>,
) {
    let n_samples = latents.ncols() as f64;
    let gram = latents.dot(&latents.t()) / n_samples;
    let correlation = latents.dot(&signal.t()) / n_samples;
    let full_factor = cholesky(gram.clone());
    let capped = AtomicUsize::new(0);
    let channels: Vec<_> = loadings
        .columns_mut()
        .into_iter()
        .zip(correlation.columns())
        .zip(signal.rows())
        .collect();
    let n_channels = channels.len();
    channels.into_par_iter().for_each(|((mut a, b), x)| {
        let problem = Problem {
            latents,
            gram: &gram,
            full_factor: full_factor.as_ref(),
            b,
            x,
            penalty,
        };
        if !problem.solve(&mut a, MAX_SWEEPS) {
            capped.fetch_add(1, Ordering::Relaxed);
        }
    });
    let capped = capped.into_inner();
    if capped > 0 {
        warn!(
            capped,
            n_channels,
            "lasso hit its sweep limit on some channels, so their loadings are approximate; nearly collinear latents cause this, and fewer latents help"
        );
    }
}

/// One channel's lasso: minimise `(1/2T)‖x − Zᵀa‖² + penalty·‖a‖₁`, with `G = ZZᵀ/T` and `b = Zx/T`.
struct Problem<'a> {
    latents: ArrayView2<'a, f64>,
    gram: &'a Array2<f64>,
    /// Cholesky factor of the whole of G, shared by every channel whose loadings are all nonzero.
    full_factor: Option<&'a Array2<f64>>,
    b: ArrayView1<'a, f64>,
    x: ArrayView1<'a, f64>,
    penalty: f64,
}

impl Problem<'_> {
    /// Returns false when the sweep limit stopped it.
    fn solve(&self, a: &mut ArrayViewMut1<f64>, max_sweeps: usize) -> bool {
        let energy = self.x.dot(&self.x) / self.x.len() as f64;
        for sweep in 1..=max_sweeps {
            let (step, largest) = sweep_once(self.gram, self.b, self.penalty, a);
            if step <= RELATIVE_STEP * largest {
                return true;
            }
            if sweep.is_power_of_two()
                && let Some(exact) = self.solve_for_signs(a.view())
            {
                a.assign(&exact);
                return true;
            }
            if sweep % GAP_EVERY == 0 && self.duality_gap(a.view()) <= GAP * energy {
                return true;
            }
        }
        false
    }

    /// The lasso solution with the nonzero pattern and signs of `a`, if one exists: it solves
    /// `G_SS x = b_S − penalty·sign(a_S)`, keeps every sign, and leaves `|b_j − G_j x| ≤ penalty`
    /// for every zero coefficient. These are the optimality conditions, so the result is the minimum.
    fn solve_for_signs(&self, a: ArrayView1<f64>) -> Option<Array1<f64>> {
        let support: Vec<usize> = (0..a.len()).filter(|&j| a[j] != 0.0).collect();
        let mut exact = Array1::zeros(a.len());
        if !support.is_empty() {
            let rhs: Vec<f64> = support
                .iter()
                .map(|&j| self.b[j] - self.penalty * a[j].signum())
                .collect();
            let x = if support.len() == a.len() {
                cholesky_solve(self.full_factor?, rhs)
            } else {
                let g = Array2::from_shape_fn((support.len(), support.len()), |(i, k)| {
                    self.gram[[support[i], support[k]]]
                });
                cholesky_solve(&cholesky(g)?, rhs)
            };
            for (&j, &v) in support.iter().zip(&x) {
                if v * a[j] <= 0.0 {
                    return None;
                }
                exact[j] = v;
            }
        }
        let scale = self.b.iter().fold(self.penalty, |m, v| m.max(v.abs()));
        let optimal = (0..a.len()).filter(|&j| a[j] == 0.0).all(|j| {
            (self.b[j] - self.gram.row(j).dot(&exact)).abs() <= self.penalty + KKT_SLACK * scale
        });
        optimal.then_some(exact)
    }

    /// scikit-learn's duality gap for the lasso, scaled by 1/T; zero at the minimum. With
    /// near-collinear latents the minimiser is poorly determined but the gap still closes.
    fn duality_gap(&self, a: ArrayView1<f64>) -> f64 {
        let n = self.x.len() as f64;
        let residual = &self.x - &self.latents.t().dot(&a);
        let fit = residual.dot(&residual) / n;
        let largest_correlation = (self.b.to_owned() - self.gram.dot(&a))
            .iter()
            .fold(0.0f64, |m, v| m.max(v.abs()));
        let scale = if largest_correlation > self.penalty {
            self.penalty / largest_correlation
        } else {
            1.0
        };
        let l1: f64 = a.iter().map(|v| v.abs()).sum();
        0.5 * fit * (1.0 + scale * scale) + self.penalty * l1 - scale * residual.dot(&self.x) / n
    }
}

/// One cyclic pass. Returns the largest coefficient change and the largest coefficient.
fn sweep_once(
    gram: &Array2<f64>,
    b: ArrayView1<f64>,
    penalty: f64,
    a: &mut ArrayViewMut1<f64>,
) -> (f64, f64) {
    let (mut largest, mut step) = (0.0f64, 0.0f64);
    for j in 0..a.len() {
        let g = gram[[j, j]];
        let old = a[j];
        let new = if g > 0.0 {
            let mut others = 0.0;
            Zip::from(gram.row(j))
                .and(&*a)
                .for_each(|&gji, &ai| others += gji * ai);
            soft_threshold(b[j] - others + g * old, penalty) / g
        } else {
            0.0
        };
        a[j] = new;
        step = step.max((new - old).abs());
        largest = largest.max(new.abs());
    }
    (step, largest)
}

/// Lower Cholesky factor of a symmetric positive definite `g`; `None` when `g` is (nearly) singular.
fn cholesky(mut g: Array2<f64>) -> Option<Array2<f64>> {
    let n = g.nrows();
    let floor = 1e-12 * g.diag().iter().fold(0.0f64, |m, v| m.max(*v));
    for j in 0..n {
        let pivot = g[[j, j]] - (0..j).map(|k| g[[j, k]] * g[[j, k]]).sum::<f64>();
        if pivot <= floor {
            return None;
        }
        g[[j, j]] = pivot.sqrt();
        for i in j + 1..n {
            let dot: f64 = (0..j).map(|k| g[[i, k]] * g[[j, k]]).sum();
            g[[i, j]] = (g[[i, j]] - dot) / g[[j, j]];
        }
    }
    Some(g)
}

/// Solves `L Lᵀ x = rhs` for a lower Cholesky factor `L`.
fn cholesky_solve(factor: &Array2<f64>, mut rhs: Vec<f64>) -> Vec<f64> {
    let n = rhs.len();
    for i in 0..n {
        rhs[i] = (rhs[i] - (0..i).map(|k| factor[[i, k]] * rhs[k]).sum::<f64>()) / factor[[i, i]];
    }
    for i in (0..n).rev() {
        rhs[i] =
            (rhs[i] - (i + 1..n).map(|k| factor[[k, i]] * rhs[k]).sum::<f64>()) / factor[[i, i]];
    }
    rhs
}

fn soft_threshold(value: f64, threshold: f64) -> f64 {
    value.signum() * (value.abs() - threshold).max(0.0)
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    #[test]
    fn soft_threshold_shrinks_towards_zero() {
        assert_eq!(soft_threshold(3.0, 1.0), 2.0);
        assert_eq!(soft_threshold(-3.0, 1.0), -2.0);
        assert_eq!(soft_threshold(0.5, 1.0), 0.0);
        assert_eq!(soft_threshold(-0.5, 1.0), 0.0);
    }

    #[test]
    fn zero_penalty_is_least_squares_and_empty_latents_get_zero() {
        // x = 2 z0 − z1, exactly; z2 is all zeros.
        let z = array![
            [1.0, 0.0, 1.0, 2.0],
            [0.0, 1.0, 1.0, -1.0],
            [0.0, 0.0, 0.0, 0.0]
        ];
        let x = array![[2.0, -1.0, 1.0, 5.0]];
        let mut a = Array2::from_elem((3, 1), 0.3);
        lasso(z.view(), x.view(), 0.0, &mut a);
        for (got, want) in a.iter().zip([2.0, -1.0, 0.0]) {
            assert!((got - want).abs() < 1e-10, "{a:?}");
        }
    }

    #[test]
    fn cholesky_solves_and_refuses_singular_systems() {
        let factor = cholesky(array![[4.0, 2.0], [2.0, 3.0]]).unwrap();
        let x = cholesky_solve(&factor, vec![2.0, 5.0]);
        assert!(
            (x[0] + 0.5).abs() < 1e-14 && (x[1] - 2.0).abs() < 1e-14,
            "{x:?}"
        );
        assert!(cholesky(array![[1.0, 1.0], [1.0, 1.0]]).is_none());
        assert!(cholesky(array![[0.0]]).is_none());
    }

    #[test]
    fn exact_step_rejects_wrong_signs_and_violated_conditions() {
        // Orthonormal design: the solution is the soft-thresholded correlation [1.5, 0, −0.2].
        let gram = Array2::eye(3);
        let factor = cholesky(gram.clone());
        let latents = Array2::zeros((3, 1));
        let x = ndarray::arr1(&[0.0]);
        let solve = |b: [f64; 3], a: [f64; 3]| {
            let b = ndarray::arr1(&b);
            let problem = Problem {
                latents: latents.view(),
                gram: &gram,
                full_factor: factor.as_ref(),
                b: b.view(),
                x: x.view(),
                penalty: 0.5,
            };
            problem.solve_for_signs(ndarray::arr1(&a).view())
        };
        let b = [2.0, 0.3, -0.7];
        let exact = solve(b, [1.0, 0.0, -1.0]).unwrap();
        assert!(
            (exact[0] - 1.5).abs() < 1e-15 && exact[1] == 0.0 && (exact[2] + 0.2).abs() < 1e-15
        );
        assert!(
            solve(b, [1.0, 0.0, 1.0]).is_none(),
            "wrong sign on the third"
        );
        assert!(
            solve(b, [1.0, 0.0, 0.0]).is_none(),
            "third must be nonzero: |−0.7| > 0.5"
        );
        assert_eq!(
            solve([0.1, -0.2, 0.0], [0.0; 3]).unwrap(),
            ndarray::arr1(&[0.0; 3])
        );
        let all = solve([2.0, 0.7, -0.7], [1.0, 1.0, -1.0]).unwrap();
        assert!(
            (all[1] - 0.2).abs() < 1e-15,
            "full support uses the shared factor: {all:?}"
        );
    }

    /// A problem built from latents and one channel, owning what `Problem` borrows.
    struct Data {
        latents: Array2<f64>,
        gram: Array2<f64>,
        factor: Option<Array2<f64>>,
        b: Array1<f64>,
        x: Array1<f64>,
    }

    impl Data {
        fn new(latents: Array2<f64>, x: Array1<f64>) -> Self {
            let n = latents.ncols() as f64;
            let gram = latents.dot(&latents.t()) / n;
            let b = latents.dot(&x) / n;
            Self {
                factor: cholesky(gram.clone()),
                latents,
                gram,
                b,
                x,
            }
        }

        fn problem(&self, penalty: f64) -> Problem<'_> {
            Problem {
                latents: self.latents.view(),
                gram: &self.gram,
                full_factor: self.factor.as_ref(),
                b: self.b.view(),
                x: self.x.view(),
                penalty,
            }
        }

        fn objective(&self, a: &Array1<f64>, penalty: f64) -> f64 {
            let r = &self.x - &self.latents.t().dot(a);
            r.dot(&r) / (2.0 * self.x.len() as f64)
                + penalty * a.iter().map(|v| v.abs()).sum::<f64>()
        }
    }

    fn correlated() -> Data {
        let latents = Array2::from_shape_fn((3, 50), |(l, t)| {
            let t = t as f64;
            [
                (0.3 * t).sin(),
                (0.3 * t).sin() + 0.05 * (1.1 * t).cos(),
                (0.07 * t).cos(),
            ][l]
        });
        let x = ndarray::Array1::from_shape_fn(50, |t| {
            2.0 * (0.3 * t as f64).sin() - 0.4 * (0.07 * t as f64).cos()
        });
        Data::new(latents, x)
    }

    #[test]
    fn a_sweep_updates_each_coordinate_against_the_others() {
        let gram = array![[2.0, 1.0], [1.0, 2.0]];
        let b = array![3.0, 0.0];
        let mut a = array![0.0, 0.0];
        let (step, largest) = sweep_once(&gram, b.view(), 0.5, &mut a.view_mut());
        // a0 = S(3, 0.5)/2 = 1.25; a1 = S(0 − 1·1.25, 0.5)/2 = −0.375.
        assert_eq!(a, array![1.25, -0.375]);
        assert_eq!((step, largest), (1.25, 1.25));

        // From a warm start the step is the change, not the new value:
        // a0 = S(3 − (2·1 − 0.375) + 2·1, 0.5)/2 = 1.4375, a change of 0.4375;
        // a1 = S(0 − (1.4375 − 0.75) − 0.75, 0.5)/2 = −0.46875.
        let mut warm = array![1.0, -0.375];
        let (step, _) = sweep_once(&gram, b.view(), 0.5, &mut warm.view_mut());
        assert_eq!((warm, step), (array![1.4375, -0.46875], 0.4375));
    }

    #[test]
    fn duality_gap_is_primal_minus_dual_and_closes_at_the_minimum() {
        let data = correlated();
        for (penalty, a) in [(0.01, array![0.5, -0.2, 0.1]), (5.0, array![0.0, 0.0, 0.0])] {
            let problem = data.problem(penalty);
            let n = data.x.len() as f64;
            let r = &data.x - &data.latents.t().dot(&a);
            let correlation = (&data.b - &data.gram.dot(&a))
                .iter()
                .fold(0.0f64, |m, v| m.max(v.abs()));
            let s = if correlation > penalty {
                penalty / correlation
            } else {
                1.0
            };
            let dual =
                (data.x.dot(&data.x) - (&data.x - &(&r * s)).mapv(|v| v * v).sum()) / (2.0 * n);
            let expected = data.objective(&a, penalty) - dual;
            assert!(
                (problem.duality_gap(a.view()) - expected).abs() < 1e-12,
                "penalty {penalty}"
            );
        }
        let problem = data.problem(0.01);
        let mut a = Array1::zeros(3);
        assert!(problem.solve(&mut a.view_mut(), MAX_SWEEPS));
        assert!(problem.duality_gap(a.view()).abs() < 1e-12);
    }

    #[test]
    fn optimality_check_uses_the_full_gradient_within_a_small_slack() {
        let gram = array![[1.0, 0.5], [0.5, 1.0]];
        let factor = cholesky(gram.clone());
        let (latents, x) = (Array2::zeros((2, 1)), ndarray::arr1(&[0.0]));
        let check = |b1: f64| {
            let b = array![2.5, b1];
            let problem = Problem {
                latents: latents.view(),
                gram: &gram,
                full_factor: factor.as_ref(),
                b: b.view(),
                x: x.view(),
                penalty: 0.5,
            };
            problem.solve_for_signs(array![1.0, 0.0].view())
        };
        // a0 = 2.5 − 0.5 = 2; the zero coefficient's gradient is b1 − 0.5·2 = b1 − 1.
        assert_eq!(
            check(1.5).unwrap(),
            array![2.0, 0.0],
            "gradient exactly at the penalty"
        );
        assert!(
            check(1.5 + 1e-10).is_some(),
            "within 1e-10 of the largest |b|"
        );
        assert!(check(1.5 + 1e-9).is_none());
        assert!(check(0.5 - 1e-9).is_none(), "gradient −0.5 − 1e-9");

        // A support that is not the leading block needs its own factor.
        let gram = array![[4.0, 1.0], [1.0, 1.0]];
        let factor = cholesky(gram.clone());
        let b = array![2.3, 2.5];
        let problem = Problem {
            latents: latents.view(),
            gram: &gram,
            full_factor: factor.as_ref(),
            b: b.view(),
            x: x.view(),
            penalty: 0.5,
        };
        assert_eq!(
            problem.solve_for_signs(array![0.0, 1.0].view()).unwrap(),
            array![0.0, 2.0]
        );
    }

    #[test]
    fn the_sweep_budget_is_reported_and_a_larger_one_reaches_the_minimum() {
        // Nearly collinear latents and a start with the wrong signs: coordinate descent crawls.
        let data = correlated();
        let problem = data.problem(0.01);
        let mut a = array![-5.0, 5.0, 0.0];
        assert!(!problem.solve(&mut a.view_mut(), 1));
        assert!(
            !problem.solve(&mut a.view_mut(), MAX_SWEEPS),
            "still crawling after 1000 more"
        );
        assert!(problem.solve(&mut a.view_mut(), 1_000_000));
        let mut from_zero = Array1::zeros(3);
        assert!(problem.solve(&mut from_zero.view_mut(), MAX_SWEEPS));
        assert!((data.objective(&a, 0.01) - data.objective(&from_zero, 0.01)).abs() < 1e-12);
    }

    #[test]
    fn cholesky_floor_is_relative_to_the_diagonal() {
        assert!(cholesky(array![[1e6, 1e6], [1e6, 1e6 + 1e-7]]).is_none());
        assert!(cholesky(array![[1e-6, 0.0], [0.0, 1e-6]]).is_some());
    }

    #[test]
    fn large_penalty_gives_all_zeros() {
        let z = array![[1.0, 2.0, 3.0], [3.0, 1.0, 0.0]];
        let x = array![[1.0, 1.0, 1.0], [0.0, 2.0, 1.0]];
        let mut a = Array2::from_elem((2, 2), 1.0);
        lasso(z.view(), x.view(), 1e6, &mut a);
        assert!(a.iter().all(|&v| v == 0.0));
    }
}
