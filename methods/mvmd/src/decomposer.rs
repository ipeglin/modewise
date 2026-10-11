use modewise_decomposition::{Decomposition, ModeDecomposer};
use modewise_fourier::{from_half_spectrum, to_half_spectrum};
use modewise_method::{MethodDescriptor, MethodError};
use modewise_signals::MultichannelSeries;
use ndarray::{Array1, Array2, Array3, Axis, Zip};
use rustfft::num_complex::Complex64;
use tracing::debug;

use crate::params::MvmdParams;

#[derive(Debug, Clone)]
pub struct Mvmd {
    params: MvmdParams,
}

struct Solution {
    modes_hat: Array3<Complex64>,
    omegas: Vec<f64>,
}

impl Mvmd {
    pub fn new(params: MvmdParams) -> Result<Self, MethodError> {
        params.validate()?;
        Ok(Self { params })
    }

    pub fn params(&self) -> &MvmdParams {
        &self.params
    }

    fn solve(&self, signal_hat: &Array2<Complex64>, n_samples: usize) -> Solution {
        let p = &self.params;
        let (n_channels, n_bins) = signal_hat.dim();
        let freqs = Array1::linspace(0.0, 0.5, n_bins);
        let mut modes_hat = Array3::<Complex64>::zeros((p.num_modes, n_channels, n_bins));
        let mut modes_sum = Array2::<Complex64>::zeros((n_channels, n_bins));
        let mut lambda = Array2::<Complex64>::zeros((n_channels, n_bins));
        let mut omegas = p.init.omegas(p.num_modes);
        // Report the frequencies the final modes were filtered with, as the reference does.
        let mut used_omegas = omegas.clone();

        let mut change = p.tolerance + f64::EPSILON;
        let mut iteration = 0;
        while iteration < p.max_iterations && change > p.tolerance {
            change = 0.0;
            let mut next_omegas = omegas.clone();
            for (k, mut mode) in modes_hat.axis_iter_mut(Axis(0)).enumerate() {
                let (mut power, mut weighted) = (0.0, 0.0);
                Zip::indexed(&mut mode)
                    .and(&mut modes_sum)
                    .and(signal_hat)
                    .and(&lambda)
                    .for_each(|(_, j), m, sum, s, l| {
                        let others = *sum - *m;
                        let filter = 1.0 + p.alpha * (freqs[j] - omegas[k]).powi(2);
                        let updated = (*s - others - *l * 0.5) / filter;
                        change += (updated - *m).norm_sqr();
                        *m = updated;
                        *sum = others + updated;
                        power += updated.norm_sqr();
                        weighted += updated.norm_sqr() * freqs[j];
                    });
                if power > 0.0 {
                    next_omegas[k] = weighted / power;
                }
            }
            Zip::from(&mut lambda)
                .and(&modes_sum)
                .and(signal_hat)
                .for_each(|l, sum, s| *l += (*sum - *s) * p.tau);
            used_omegas = std::mem::replace(&mut omegas, next_omegas);
            iteration += 1;
            change /= n_samples as f64;
        }
        debug!(
            iteration,
            change,
            converged = change <= p.tolerance,
            "mvmd finished"
        );
        Solution {
            modes_hat,
            omegas: used_omegas,
        }
    }
}

impl ModeDecomposer for Mvmd {
    fn descriptor(&self) -> MethodDescriptor {
        MethodDescriptor {
            name: "mvmd",
            version: env!("CARGO_PKG_VERSION"),
            multivariate: true,
        }
    }

    fn decompose(&self, signal: &MultichannelSeries) -> Result<Decomposition, MethodError> {
        let n_samples = signal.n_samples();
        let signal_hat = to_half_spectrum(signal.data());
        let Solution { modes_hat, omegas } = self.solve(&signal_hat, n_samples);

        let mut order: Vec<usize> = (0..omegas.len()).collect();
        order.sort_by(|&a, &b| omegas[a].total_cmp(&omegas[b]));

        let mut modes = Array3::zeros((order.len(), signal.n_channels(), n_samples));
        for (mut out, &k) in modes.axis_iter_mut(Axis(0)).zip(&order) {
            out.assign(&from_half_spectrum(modes_hat.index_axis(Axis(0), k)));
        }
        let fs = signal.sampling_rate();
        let centre_frequencies_hz = order.iter().map(|&k| omegas[k] * fs.hz()).collect();

        Decomposition::new(
            modes,
            centre_frequencies_hz,
            fs,
            signal.labels().to_vec(),
            true,
        )
    }
}
