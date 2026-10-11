//! Continuous wavelet transform with a complex Morlet wavelet, after Torrence & Compo (1998).
//!
//! Conventions (docs/time-frequency.md), matching the pycwt implementation of the same paper:
//! - wavelet in frequency space `π^-1/4 · exp(-(sω − ω0)² / 2)`, normalised by `√(2πs / δt)`
//! - Fourier period of scale `s`: `λ·s` with `λ = 4π / (ω0 + √(2 + ω0²))`
//! - signal zero-padded to the next power of two before the FFT
//! - cone of influence: e-folding time `√2·s`, i.e. a period `P` is reliable at sample `i`
//!   when `P ≤ λ/√2 · δt · (min(i, n − 1 − i) + 0.5)`, the distance to the record boundary
//!   half a sample beyond the first and last sample (as pycwt).

use std::f64::consts::PI;

use modewise_method::{MethodDescriptor, MethodError};
use modewise_signals::MultichannelSeries;
use modewise_spectral::{FrequencyGrid, TimeFrequencyMap, TimeFrequencyTransform};
use ndarray::{Array2, Array3};
use rustfft::FftPlanner;
use rustfft::num_complex::Complex64;

#[derive(Debug, Clone)]
pub struct Cwt {
    grid: FrequencyGrid,
    omega0: f64,
    rectify: bool,
}

impl Cwt {
    /// `grid` gives the frequencies to analyse (its bin centres); `omega0` is the Morlet
    /// wavenumber (6 in Torrence & Compo). With `rectify`, power is divided by scale
    /// (Liu, Liang & Weisberg 2007).
    pub fn new(grid: FrequencyGrid, omega0: f64, rectify: bool) -> Result<Self, MethodError> {
        if !(omega0.is_finite() && omega0 >= 5.0) {
            return Err(MethodError::invalid(
                "omega0",
                "must be finite and >= 5 for the Morlet to be admissible in practice",
            ));
        }
        Ok(Self {
            grid,
            omega0,
            rectify,
        })
    }

    /// Fourier period of scale 1, `λ`.
    pub fn fourier_factor(&self) -> f64 {
        4.0 * PI / (self.omega0 + (2.0 + self.omega0 * self.omega0).sqrt())
    }
}

impl TimeFrequencyTransform for Cwt {
    fn descriptor(&self) -> MethodDescriptor {
        MethodDescriptor {
            name: "cwt",
            version: env!("CARGO_PKG_VERSION"),
            multivariate: false,
        }
    }

    fn transform(&self, signal: &MultichannelSeries) -> Result<TimeFrequencyMap, MethodError> {
        let fs = signal.sampling_rate();
        let dt = fs.period_s();
        let n = signal.n_samples();
        let padded = n.next_power_of_two();
        let lambda = self.fourier_factor();
        let freqs = self.grid.centres();
        let nyquist = fs.hz() / 2.0;
        if let Some(f) = freqs.iter().find(|&&f| f <= 0.0 || f > nyquist) {
            return Err(MethodError::invalid(
                "frequencies",
                format!("{f} Hz is outside (0, {nyquist}] Hz"),
            ));
        }
        let scales: Vec<f64> = freqs.iter().map(|f| 1.0 / (lambda * f)).collect();
        let omegas: Vec<f64> = (0..padded)
            .map(|k| {
                // The Nyquist bin counts as negative, as numpy's fftfreq (and pycwt) do.
                let k = if k < padded / 2 {
                    k as f64
                } else {
                    k as f64 - padded as f64
                };
                2.0 * PI * k / (padded as f64 * dt)
            })
            .collect();

        let mut planner = FftPlanner::new();
        let forward = planner.plan_fft_forward(padded);
        let inverse = planner.plan_fft_inverse(padded);
        let mut power = Array3::zeros((signal.n_channels(), scales.len(), n));
        for (c, channel) in signal.data().rows().into_iter().enumerate() {
            let mut spectrum: Vec<Complex64> =
                channel.iter().map(|&v| Complex64::new(v, 0.0)).collect();
            spectrum.resize(padded, Complex64::new(0.0, 0.0));
            forward.process(&mut spectrum);
            for (j, &s) in scales.iter().enumerate() {
                let norm = (2.0 * PI * s / dt).sqrt() * PI.powf(-0.25);
                let mut w: Vec<Complex64> = spectrum
                    .iter()
                    .zip(&omegas)
                    .map(|(x, &om)| x * norm * (-0.5 * (s * om - self.omega0).powi(2)).exp())
                    .collect();
                inverse.process(&mut w);
                let rectify = if self.rectify { s } else { 1.0 };
                for t in 0..n {
                    power[[c, j, t]] = (w[t] / padded as f64).norm_sqr() / rectify;
                }
            }
        }

        let coi_factor = lambda / 2f64.sqrt() * dt;
        let reliable = Array2::from_shape_fn((freqs.len(), n), |(j, t)| {
            1.0 / freqs[j] <= coi_factor * (t.min(n - 1 - t) as f64 + 0.5)
        });
        let times = (0..n).map(|t| t as f64 * dt).collect();
        TimeFrequencyMap::new(power, freqs, times, signal.labels().to_vec())?
            .with_sampling_rate(fs)?
            .with_reliable(reliable)
    }
}
