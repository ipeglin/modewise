//! Spectra of mirror-extended signals, as in the MVMD and VLMD reference implementations.
//! Ported from Dmocrito/mvmd (MIT; notice in THIRD_PARTY_NOTICES.md).
//!
//! A signal of length T is padded symmetrically to 2T. Only the non-negative
//! half of its spectrum (T + 1 bins over [0, 0.5] cycles per sample) is kept.
//!
//! Also the spectral centroid that EMD-family and wavelet modes report as centre frequency.

mod centroid;

pub use centroid::centroid_hz;

use ndarray::{Array2, ArrayView1, ArrayView2};
use rustfft::FftPlanner;
use rustfft::num_complex::Complex64;

pub fn to_half_spectrum(signal: ArrayView2<f64>) -> Array2<Complex64> {
    let (n_channels, n) = signal.dim();
    let fft = FftPlanner::new().plan_fft_forward(2 * n);
    let mut out = Array2::zeros((n_channels, n + 1));
    for (channel, mut row) in signal.rows().into_iter().zip(out.rows_mut()) {
        let mut buffer = mirror_extend(channel);
        fft.process(&mut buffer);
        row.iter_mut().zip(&buffer).for_each(|(o, b)| *o = *b);
    }
    out
}

pub fn from_half_spectrum(half: ArrayView2<Complex64>) -> Array2<f64> {
    let (n_channels, n_bins) = half.dim();
    let n = n_bins - 1;
    let ifft = FftPlanner::new().plan_fft_inverse(2 * n);
    let scale = 1.0 / (2 * n) as f64;
    let mut out = Array2::zeros((n_channels, n));
    for (spectrum, mut row) in half.rows().into_iter().zip(out.rows_mut()) {
        let mut buffer: Vec<Complex64> = (0..2 * n)
            .map(|j| {
                if j < n {
                    spectrum[j]
                } else {
                    spectrum[2 * n - j].conj()
                }
            })
            .collect();
        ifft.process(&mut buffer);
        let start = n / 2;
        row.iter_mut()
            .zip(&buffer[start..start + n])
            .for_each(|(o, b)| *o = b.re * scale);
    }
    out
}

fn mirror_extend(x: ArrayView1<f64>) -> Vec<Complex64> {
    let n = x.len();
    let left = n / 2;
    let right = n - left;
    let head = (0..left).rev().map(|i| x[i]);
    let tail = (0..right).map(|i| x[n - 1 - i]);
    head.chain(x.iter().copied())
        .chain(tail)
        .map(|v| Complex64::new(v, 0.0))
        .collect()
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    #[test]
    fn mirror_matches_numpy_symmetric_pad() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let padded: Vec<f64> = mirror_extend(x.view()).iter().map(|c| c.re).collect();
        assert_eq!(padded, [2.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 4.0, 3.0]);
    }

    /// Direct inverse DFT of the Hermitian extension, as the reference builds it.
    #[test]
    fn matches_direct_inverse_dft() {
        let n = 4;
        let half = ndarray::Array2::from_shape_vec(
            (1, n + 1),
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(0.5, 0.25),
                Complex64::new(-0.3, 0.1),
                Complex64::new(0.2, -0.4),
                Complex64::new(0.7, 0.6),
            ],
        )
        .unwrap();
        let full: Vec<Complex64> = (0..2 * n)
            .map(|j| {
                if j < n {
                    half[[0, j]]
                } else {
                    half[[0, 2 * n - j]].conj()
                }
            })
            .collect();
        let expected: Vec<f64> = (n / 2..n / 2 + n)
            .map(|t| {
                let sum: Complex64 = full
                    .iter()
                    .enumerate()
                    .map(|(k, x)| {
                        x * Complex64::from_polar(
                            1.0,
                            2.0 * std::f64::consts::PI * (k * t) as f64 / (2 * n) as f64,
                        )
                    })
                    .sum();
                sum.re / (2 * n) as f64
            })
            .collect();
        let ours = from_half_spectrum(half.view());
        for (a, b) in ours.iter().zip(&expected) {
            assert!((a - b).abs() < 1e-12, "{a} != {b}");
        }
    }

    #[test]
    fn round_trip_recovers_signal() {
        for x in [
            array![[1.0, -2.0, 3.0, 0.5]],
            array![[0.3, 2.0, -1.0, 4.0, 7.0]],
            array![[0.3, -2.0, 1.5]],
        ] {
            let back = from_half_spectrum(to_half_spectrum(x.view()).view());
            for (a, b) in back.iter().zip(&x) {
                assert!((a - b).abs() < 1e-12, "{a} != {b}");
            }
        }
    }
}
