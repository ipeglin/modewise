//! Short-time Fourier transform with a periodic Hann window, and Welch power spectra.

mod welch;

pub use welch::{Detrend, Welch, Window};

use std::f64::consts::PI;

use modewise_method::{MethodDescriptor, MethodError};
use modewise_signals::MultichannelSeries;
use modewise_spectral::{TimeFrequencyMap, TimeFrequencyTransform};
use ndarray::Array3;
use rustfft::FftPlanner;
use rustfft::num_complex::Complex64;

/// Frames start at sample 0 and step by `hop`. Frames that would run past the
/// end are dropped. Power is the unscaled squared magnitude, |X|².
#[derive(Debug, Clone)]
pub struct Stft {
    window_len: usize,
    hop: usize,
}

impl Stft {
    pub fn new(window_len: usize, hop: usize) -> Result<Self, MethodError> {
        if window_len < 2 {
            return Err(MethodError::invalid("window_len", "must be at least 2"));
        }
        if hop == 0 {
            return Err(MethodError::invalid("hop", "must be at least 1"));
        }
        Ok(Self { window_len, hop })
    }

    fn window(&self) -> Vec<f64> {
        let n = self.window_len as f64;
        (0..self.window_len)
            .map(|i| 0.5 - 0.5 * (2.0 * PI * i as f64 / n).cos())
            .collect()
    }
}

impl TimeFrequencyTransform for Stft {
    fn descriptor(&self) -> MethodDescriptor {
        MethodDescriptor {
            name: "stft",
            version: env!("CARGO_PKG_VERSION"),
            multivariate: false,
        }
    }

    fn transform(&self, signal: &MultichannelSeries) -> Result<TimeFrequencyMap, MethodError> {
        let (n, hop) = (self.window_len, self.hop);
        if signal.n_samples() < n {
            return Err(MethodError::SignalTooShort {
                method: "stft",
                needed: n,
                actual: signal.n_samples(),
            });
        }
        let n_frames = (signal.n_samples() - n) / hop + 1;
        let n_bins = n / 2 + 1;
        let window = self.window();
        let fft = FftPlanner::new().plan_fft_forward(n);

        let mut power = Array3::zeros((signal.n_channels(), n_bins, n_frames));
        for (c, channel) in signal.data().rows().into_iter().enumerate() {
            for frame in 0..n_frames {
                let start = frame * hop;
                let mut buffer: Vec<Complex64> = (0..n)
                    .map(|i| Complex64::new(channel[start + i] * window[i], 0.0))
                    .collect();
                fft.process(&mut buffer);
                for (bin, value) in buffer.iter().take(n_bins).enumerate() {
                    power[[c, bin, frame]] = value.norm_sqr();
                }
            }
        }

        let fs = signal.sampling_rate().hz();
        let frequencies_hz = (0..n_bins).map(|k| k as f64 * fs / n as f64).collect();
        let times_s = (0..n_frames)
            .map(|f| (f * hop) as f64 / fs + n as f64 / (2.0 * fs))
            .collect();
        TimeFrequencyMap::new(power, frequencies_hz, times_s, signal.labels().to_vec())
    }
}
