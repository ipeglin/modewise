use modewise_decomposition::{Decomposition, Loadings, ModeDecomposer};
use modewise_fourier::{from_half_spectrum, to_half_spectrum};
use modewise_method::{MethodDescriptor, MethodError};
use modewise_signals::MultichannelSeries;
use ndarray::{Array1, Array2, Array3, Axis, Zip, s};
use rustfft::num_complex::Complex64;
use tracing::{debug, warn};

use crate::lasso::lasso;
use crate::params::VlmdParams;

/// Below this total spectral energy a mode is treated as empty, as in the reference.
const EMPTY_MODE: f64 = 1e-10;

#[derive(Debug, Clone)]
pub struct Vlmd {
    params: VlmdParams,
}

/// Everything VLMD estimates. `modes` are the channel modes `U⁽ᵏ⁾ = Θ⁽ᵏ⁾A` (Eq. 7);
/// they sum to the latent fit `ZA`, not exactly to the signal.
#[derive(Debug, Clone, PartialEq)]
pub struct VlmdOutput {
    pub modes: Decomposition,
    pub latent_modes: Decomposition,
    pub loadings: Loadings,
    pub iterations: usize,
}

struct State {
    loadings: Array2<f64>,
    latents: Array2<f64>,
    latents_hat: Array2<Complex64>,
    modes_hat: Array3<Complex64>,
    modes_sum: Array2<Complex64>,
    dual: Array2<Complex64>,
}

impl Vlmd {
    pub fn new(params: VlmdParams) -> Result<Self, MethodError> {
        params.validate()?;
        Ok(Self { params })
    }

    pub fn params(&self) -> &VlmdParams {
        &self.params
    }

    /// Channel modes, latent modes and loadings, with modes ordered by centre frequency.
    pub fn decompose_latent(&self, signal: &MultichannelSeries) -> Result<VlmdOutput, MethodError> {
        let p = &self.params;
        let n_channels = signal.n_channels();
        let n_latents = p.latents_for(n_channels)?;
        let x = signal.data();
        let x_hat = to_half_spectrum(x);
        let n_bins = x_hat.ncols();
        let freqs = Array1::linspace(0.0, 0.5, n_bins);
        let penalty = p.reg_lambda * 1e-3;
        let fit = 2.0 / p.reg_rho;
        let narrowness = 4.0 * p.alpha / p.reg_rho;

        let latents = x.slice(s![..n_latents, ..]).to_owned();
        let mut state = State {
            loadings: Array2::eye(n_latents.max(n_channels))
                .slice_move(s![..n_latents, ..n_channels]),
            latents_hat: to_half_spectrum(latents.view()),
            latents,
            modes_hat: Array3::zeros((p.num_modes, n_latents, n_bins)),
            modes_sum: Array2::zeros((n_latents, n_bins)),
            dual: Array2::zeros((n_latents, n_bins)),
        };
        let mut omegas = vec![0.0; p.num_modes];
        let mut used = omegas.clone();
        let mut change = p.tolerance + f64::EPSILON;
        let mut iterations = 0;
        while iterations < p.max_iterations && change > p.tolerance {
            lasso(state.latents.view(), x, penalty, &mut state.loadings);
            bound_loadings(&mut state.loadings);
            update_latents(&mut state, &x_hat, fit);
            let next = update_modes(&mut state, &freqs, &omegas, narrowness);
            Zip::from(&mut state.dual)
                .and(&state.latents_hat)
                .and(&state.modes_sum)
                .for_each(|g, z, m| *g += (z - m) * p.tau);
            state.latents = from_half_spectrum(state.latents_hat.view());
            change = omegas
                .iter()
                .zip(&next)
                .map(|(old, new)| ((new - old) / (old + 1e-10)).powi(2))
                .sum();
            used = std::mem::replace(&mut omegas, next);
            iterations += 1;
        }
        debug!(
            iterations,
            change,
            converged = change <= p.tolerance,
            "vlmd finished"
        );
        self.output(signal, state, &used, iterations)
    }

    fn output(
        &self,
        signal: &MultichannelSeries,
        state: State,
        omegas: &[f64],
        iterations: usize,
    ) -> Result<VlmdOutput, MethodError> {
        let fs = signal.sampling_rate();
        let (n_latents, n_channels) = state.loadings.dim();
        let n_samples = signal.n_samples();
        let mut order: Vec<usize> = (0..omegas.len()).collect();
        order.sort_by(|&a, &b| omegas[a].total_cmp(&omegas[b]));

        let mut latent_modes = Array3::zeros((order.len(), n_latents, n_samples));
        let mut modes = Array3::zeros((order.len(), n_channels, n_samples));
        for (i, &k) in order.iter().enumerate() {
            let theta = from_half_spectrum(state.modes_hat.index_axis(Axis(0), k));
            modes
                .index_axis_mut(Axis(0), i)
                .assign(&state.loadings.t().dot(&theta));
            latent_modes.index_axis_mut(Axis(0), i).assign(&theta);
        }
        let centres: Vec<f64> = order.iter().map(|&k| omegas[k] * fs.hz()).collect();
        let latent_labels: Vec<String> = (0..n_latents).map(|l| format!("latent_{l}")).collect();
        Ok(VlmdOutput {
            modes: Decomposition::new(modes, centres.clone(), fs, signal.labels().to_vec(), true)?,
            latent_modes: Decomposition::new(
                latent_modes,
                centres,
                fs,
                latent_labels.clone(),
                true,
            )?,
            loadings: Loadings::new(state.loadings, latent_labels, signal.labels().to_vec())?,
            iterations,
        })
    }
}

/// Keeps |a| ≤ 1 by rescaling each latent's row, which the latent signal can absorb.
fn bound_loadings(loadings: &mut Array2<f64>) {
    for mut row in loadings.rows_mut() {
        let largest = row.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        if largest > 1.0 {
            row /= largest;
        }
    }
}

/// Eq. (16), one latent at a time against the residual left by the others.
fn update_latents(state: &mut State, x_hat: &Array2<Complex64>, fit: f64) {
    let mut residual = x_hat
        - &state
            .loadings
            .t()
            .mapv(|v| Complex64::new(v, 0.0))
            .dot(&state.latents_hat);
    for l in 0..state.loadings.nrows() {
        let a = state.loadings.row(l);
        let mut z = state.latents_hat.row_mut(l);
        for (c, &acl) in a.iter().enumerate() {
            if acl != 0.0 {
                Zip::from(residual.row_mut(c))
                    .and(&z)
                    .for_each(|r, &zl| *r += zl * acl);
            }
        }
        let mut projection = Array1::<Complex64>::zeros(z.len());
        for (c, &acl) in a.iter().enumerate() {
            if acl != 0.0 {
                projection.scaled_add(Complex64::new(acl, 0.0), &residual.row(c));
            }
        }
        let denominator = 1.0 + fit * a.dot(&a);
        Zip::from(&mut z)
            .and(&projection)
            .and(state.modes_sum.row(l))
            .and(state.dual.row(l))
            .for_each(|z, &proj, &modes, &dual| *z = (proj * fit + modes - dual) / denominator);
        for (c, &acl) in a.iter().enumerate() {
            if acl != 0.0 {
                Zip::from(residual.row_mut(c))
                    .and(&z)
                    .for_each(|r, &zl| *r -= zl * acl);
            }
        }
    }
}

/// Eqs. (18) and (21): each mode against the others, then its centre frequency.
/// Returns the next frequencies; the filters use the current ones.
fn update_modes(
    state: &mut State,
    freqs: &Array1<f64>,
    omegas: &[f64],
    narrowness: f64,
) -> Vec<f64> {
    let mut next = omegas.to_vec();
    for (k, mut mode) in state.modes_hat.axis_iter_mut(Axis(0)).enumerate() {
        let (mut power, mut weighted) = (0.0, 0.0);
        Zip::indexed(&mut mode)
            .and(&mut state.modes_sum)
            .and(&state.latents_hat)
            .and(&state.dual)
            .for_each(|(_, j), m, sum, &z, &g| {
                let others = *sum - *m;
                let updated =
                    (z - others + g) / (1.0 + narrowness * (freqs[j] - omegas[k]).powi(2));
                *m = updated;
                *sum = others + updated;
                power += updated.norm_sqr();
                weighted += updated.norm_sqr() * freqs[j];
            });
        next[k] = if power < EMPTY_MODE {
            warn!(
                mode = k,
                "vlmd mode has no energy; its frequency is set to 0"
            );
            0.0
        } else {
            weighted / power
        };
    }
    next
}

impl ModeDecomposer for Vlmd {
    fn descriptor(&self) -> MethodDescriptor {
        MethodDescriptor {
            name: "vlmd",
            version: env!("CARGO_PKG_VERSION"),
            multivariate: true,
        }
    }

    fn decompose(&self, signal: &MultichannelSeries) -> Result<Decomposition, MethodError> {
        Ok(self.decompose_latent(signal)?.modes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_empty_mode_gets_frequency_zero_not_nan() {
        let mut state = State {
            loadings: Array2::eye(1),
            latents: Array2::zeros((1, 4)),
            latents_hat: Array2::zeros((1, 5)),
            modes_hat: Array3::zeros((2, 1, 5)),
            modes_sum: Array2::zeros((1, 5)),
            dual: Array2::zeros((1, 5)),
        };
        let freqs = Array1::linspace(0.0, 0.5, 5);
        assert_eq!(
            update_modes(&mut state, &freqs, &[0.1, 0.3], 10.0),
            [0.0, 0.0]
        );
        state.latents_hat[[0, 2]] = Complex64::new(1e-6, 0.0);
        assert_eq!(
            update_modes(&mut state, &freqs, &[0.1, 0.3], 10.0),
            [0.0, 0.0],
            "energy below 1e-10"
        );
    }

    #[test]
    fn rows_above_one_are_rescaled_and_others_left_alone() {
        let mut a = ndarray::array![[2.0, -4.0], [0.5, -1.0]];
        bound_loadings(&mut a);
        assert_eq!(a, ndarray::array![[0.5, -1.0], [0.5, -1.0]]);
    }
}
