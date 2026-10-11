use std::f64::consts::PI;

use modewise_decomposition::{Decomposition, InstantaneousModes};
use modewise_method::MethodError;
use ndarray::{Array3, ArrayView1, Axis};
use rustfft::FftPlanner;
use rustfft::num_complex::Complex64;

/// How the signal is extended before the FFT, which treats it as periodic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Padding {
    /// Mirror-extend to twice the length, which avoids a wrap-around jump.
    #[default]
    Mirror,
    /// Use the signal as it is (matches `scipy.signal.hilbert`).
    None,
}

/// Analytic signal `x + i·H{x}` by the FFT method: keep DC (and Nyquist for even lengths),
/// double positive frequencies, zero negative ones.
pub fn analytic_signal(x: ArrayView1<f64>, padding: Padding) -> Vec<Complex64> {
    let n = x.len();
    if n == 0 {
        return Vec::new();
    }
    let (extended, offset): (Vec<f64>, usize) = match padding {
        Padding::None => (x.to_vec(), 0),
        // The signal followed by its reverse is periodic without a jump. Where the reversed
        // part goes does not matter: the FFT method is unaffected by rotating the sequence.
        Padding::Mirror => (x.iter().chain(x.iter().rev()).copied().collect(), 0),
    };
    let m = extended.len();
    let mut planner = FftPlanner::new();
    let mut buffer: Vec<Complex64> = extended.iter().map(|&v| Complex64::new(v, 0.0)).collect();
    planner.plan_fft_forward(m).process(&mut buffer);
    for (k, value) in buffer.iter_mut().enumerate() {
        let gain = if k == 0 || (m % 2 == 0 && k == m / 2) {
            1.0
        } else if k < m.div_ceil(2) {
            2.0
        } else {
            0.0
        };
        *value *= gain;
    }
    planner.plan_fft_inverse(m).process(&mut buffer);
    let scale = 1.0 / m as f64;
    buffer[offset..offset + n]
        .iter()
        .map(|z| z * scale)
        .collect()
}

/// Instantaneous frequency in Hz from an analytic signal: centred difference of the unwrapped
/// phase, `(φ[n+1] − φ[n−1]) · fs / 4π`; at each end, the slope of a line through 3 points (the
/// same estimator as a 3-point Savitzky-Golay derivative). Phase steps are taken one sample at
/// a time, so frequencies up to Nyquist are estimated without wrapping.
pub fn instantaneous_frequency(z: &[Complex64], fs: f64) -> Vec<f64> {
    let n = z.len();
    let to_hz = |radians: f64| radians * fs / (2.0 * PI);
    let step: Vec<f64> = z.windows(2).map(|w| (w[1] * w[0].conj()).arg()).collect();
    match n {
        1 => vec![0.0],
        2 => vec![to_hz(step[0]); 2],
        _ => (0..n)
            .map(|i| {
                let first_step = i.clamp(1, n - 2) - 1;
                to_hz(step[first_step] + step[first_step + 1]) / 2.0
            })
            .collect(),
    }
}

/// Instantaneous amplitude and frequency of every mode and channel.
pub fn instantaneous(
    modes: &Decomposition,
    padding: Padding,
) -> Result<InstantaneousModes, MethodError> {
    let fs = modes.sampling_rate().hz();
    let dim = modes.modes().dim();
    let mut amplitude = Array3::zeros(dim);
    let mut frequency = Array3::zeros(dim);
    for (k, mode) in modes.modes().axis_iter(Axis(0)).enumerate() {
        for (c, signal) in mode.axis_iter(Axis(0)).enumerate() {
            let z = analytic_signal(signal, padding);
            for (t, (value, f)) in z.iter().zip(instantaneous_frequency(&z, fs)).enumerate() {
                amplitude[[k, c, t]] = value.norm();
                frequency[[k, c, t]] = f;
            }
        }
    }
    InstantaneousModes::new(
        amplitude,
        frequency,
        modes.centre_frequencies_hz().to_vec(),
        modes.sampling_rate(),
        modes.labels().to_vec(),
    )
}

#[cfg(test)]
mod tests {
    use ndarray::Array1;

    use super::*;

    #[test]
    fn analytic_signal_of_cosine_is_complex_exponential() {
        for n in [64usize, 65] {
            let k = 5.0;
            let x = Array1::from_shape_fn(n, |t| (2.0 * PI * k * t as f64 / n as f64).cos());
            let z = analytic_signal(x.view(), Padding::None);
            for (t, value) in z.iter().enumerate() {
                let expected = Complex64::from_polar(1.0, 2.0 * PI * k * t as f64 / n as f64);
                assert!((value - expected).norm() < 1e-12, "n={n} t={t}");
            }
        }
    }

    #[test]
    fn edge_cases() {
        assert!(analytic_signal(Array1::<f64>::zeros(0).view(), Padding::Mirror).is_empty());
        assert_eq!(instantaneous_frequency(&[], 1.0), Vec::<f64>::new());
        assert_eq!(
            instantaneous_frequency(&[Complex64::new(1.0, 0.0)], 1.0),
            [0.0]
        );
        let quarter = [Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0)];
        assert_eq!(
            instantaneous_frequency(&quarter, 4.0),
            [1.0, 1.0],
            "a quarter turn per sample at 4 Hz is 1 Hz"
        );
    }
}
