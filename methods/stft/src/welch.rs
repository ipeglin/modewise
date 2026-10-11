//! Power spectral density by Welch's method, as scipy.signal.welch (density scaling, one-sided,
//! mean over segments). One boxcar segment over the whole run is scipy.signal.periodogram.

use std::f64::consts::PI;

use modewise_method::{MethodDescriptor, MethodError};
use modewise_signals::MultichannelSeries;
use modewise_spectral::{TimeFrequencyMap, TimeFrequencyTransform};
use ndarray::Array3;
use rustfft::FftPlanner;
use rustfft::num_complex::Complex64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Window {
    Boxcar,
    /// Periodic Hann, as scipy's get_window("hann", n).
    Hann,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Detrend {
    None,
    /// Subtract each segment's mean.
    Constant,
    /// Subtract each segment's least-squares line.
    Linear,
}

/// One frame of power spectral density per channel (power units per Hz).
#[derive(Debug, Clone)]
pub struct Welch {
    /// Samples per segment; `None` is the whole signal.
    pub segment_len: Option<usize>,
    /// Samples shared by neighbouring segments; `None` is half a segment, as scipy.
    pub overlap: Option<usize>,
    pub window: Window,
    pub detrend: Detrend,
}

impl Welch {
    fn window(&self, n: usize) -> Vec<f64> {
        match self.window {
            Window::Boxcar => vec![1.0; n],
            Window::Hann => (0..n)
                .map(|i| 0.5 - 0.5 * (2.0 * PI * i as f64 / n as f64).cos())
                .collect(),
        }
    }
}

fn detrend(x: &mut [f64], how: Detrend) {
    let n = x.len() as f64;
    match how {
        Detrend::None => {}
        Detrend::Constant => {
            let mean = x.iter().sum::<f64>() / n;
            x.iter_mut().for_each(|v| *v -= mean);
        }
        Detrend::Linear => {
            let t_mean = (n - 1.0) / 2.0;
            let y_mean = x.iter().sum::<f64>() / n;
            let (mut sxy, mut sxx) = (0.0, 0.0);
            for (i, v) in x.iter().enumerate() {
                let dt = i as f64 - t_mean;
                // Σ(t − t̄) = 0, so the mean of y drops out of the covariance.
                sxy += dt * v;
                sxx += dt * dt;
            }
            let slope = if sxx > 0.0 { sxy / sxx } else { 0.0 };
            for (i, v) in x.iter_mut().enumerate() {
                *v -= y_mean + slope * (i as f64 - t_mean);
            }
        }
    }
}

impl TimeFrequencyTransform for Welch {
    fn descriptor(&self) -> MethodDescriptor {
        MethodDescriptor {
            name: "spectrum",
            version: env!("CARGO_PKG_VERSION"),
            multivariate: false,
        }
    }

    fn transform(&self, signal: &MultichannelSeries) -> Result<TimeFrequencyMap, MethodError> {
        let total = signal.n_samples();
        let n = self.segment_len.unwrap_or(total);
        if n < 2 {
            return Err(MethodError::invalid(
                "segment_len",
                "must be at least 2 samples",
            ));
        }
        if n > total {
            return Err(MethodError::SignalTooShort {
                method: "spectrum",
                needed: n,
                actual: total,
            });
        }
        let overlap = self.overlap.unwrap_or(n / 2);
        if overlap >= n {
            return Err(MethodError::invalid(
                "overlap",
                "must be shorter than a segment",
            ));
        }
        let step = n - overlap;
        let starts: Vec<usize> = (0..=(total - n) / step).map(|k| k * step).collect();
        let window = self.window(n);
        let fs = signal.sampling_rate().hz();
        let scale = 1.0 / (fs * window.iter().map(|w| w * w).sum::<f64>());
        let n_bins = n / 2 + 1;
        let fft = FftPlanner::new().plan_fft_forward(n);
        let mut power = Array3::zeros((signal.n_channels(), n_bins, 1));
        for (c, row) in signal.data().rows().into_iter().enumerate() {
            for &start in &starts {
                let mut seg: Vec<f64> = row.iter().skip(start).take(n).copied().collect();
                detrend(&mut seg, self.detrend);
                let mut buf: Vec<Complex64> = seg
                    .iter()
                    .zip(&window)
                    .map(|(x, w)| Complex64::new(x * w, 0.0))
                    .collect();
                fft.process(&mut buf);
                for (k, z) in buf.iter().take(n_bins).enumerate() {
                    // One-sided: every bin but DC and an even-length Nyquist counts twice.
                    let twice = k != 0 && !(n.is_multiple_of(2) && k == n / 2);
                    power[[c, k, 0]] += z.norm_sqr() * scale * if twice { 2.0 } else { 1.0 };
                }
            }
        }
        power /= starts.len() as f64;
        let frequencies = (0..n_bins).map(|k| k as f64 * fs / n as f64).collect();
        let span = (starts.last().copied().unwrap_or(0) + n) as f64;
        TimeFrequencyMap::new(
            power,
            frequencies,
            vec![span / (2.0 * fs)],
            signal.labels().to_vec(),
        )
    }
}

#[cfg(test)]
mod tests {
    use modewise_signals::SamplingRate;
    use ndarray::Array2;

    use super::*;

    fn signal(n: usize) -> MultichannelSeries {
        let data = Array2::from_shape_fn((1, n), |(_, t)| (t as f64 * 0.7).sin() + 0.1 * t as f64);
        MultichannelSeries::new(data, SamplingRate::new(2.0).unwrap(), vec!["a".into()]).unwrap()
    }

    fn welch(segment_len: Option<usize>, overlap: Option<usize>) -> Welch {
        Welch {
            segment_len,
            overlap,
            window: Window::Hann,
            detrend: Detrend::Linear,
        }
    }

    #[test]
    fn segments_must_fit_and_the_frame_sits_at_the_centre_of_the_span() {
        for bad in [Some(0), Some(1)] {
            assert!(welch(bad, None).transform(&signal(20)).is_err());
        }
        assert!(welch(Some(21), None).transform(&signal(20)).is_err());
        assert!(welch(Some(8), Some(8)).transform(&signal(20)).is_err());
        assert!(welch(Some(2), None).transform(&signal(20)).is_ok());
        // Segments of 8 every 4 samples cover 0..20 when 20 samples are given, 10 s at 2 Hz.
        let map = welch(Some(8), Some(4)).transform(&signal(21)).unwrap();
        assert_eq!(map.times_s(), [5.0]);
        let whole = welch(None, None).transform(&signal(21)).unwrap();
        assert_eq!(whole.times_s(), [5.25]);
        let d = welch(None, None).descriptor();
        assert_eq!((d.name, d.multivariate), ("spectrum", false));
    }

    #[test]
    fn a_single_sample_segment_has_no_slope() {
        let mut x = [3.0];
        detrend(&mut x, Detrend::Linear);
        assert_eq!(x, [0.0]);
    }
}
