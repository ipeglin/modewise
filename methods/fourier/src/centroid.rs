//! A mode's centre frequency: the power-weighted mean frequency of its spectrum.

use ndarray::ArrayView2;
use rustfft::{FftPlanner, num_complex::Complex64};

/// Centroid over the one-sided spectrum, pooled over the rows (channels) given; NaN for a
/// mode with no power.
pub fn centroid_hz(rows: ArrayView2<f64>, fs_hz: f64) -> f64 {
    let n = rows.ncols();
    let fft = FftPlanner::new().plan_fft_forward(n);
    let (mut power, mut weighted) = (0.0, 0.0);
    for row in rows.rows() {
        let mut buf: Vec<Complex64> = row.iter().map(|&v| Complex64::new(v, 0.0)).collect();
        fft.process(&mut buf);
        for (k, c) in buf.iter().take(n / 2 + 1).enumerate() {
            let p = c.norm_sqr();
            power += p;
            weighted += p * k as f64 * fs_hz / n as f64;
        }
    }
    if power > 0.0 {
        weighted / power
    } else {
        f64::NAN
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Array2;

    use super::*;

    #[test]
    fn centroid_weights_the_one_sided_spectrum_pooled_over_rows() {
        let nyquist = Array2::from_shape_fn((1, 8), |(_, t)| if t % 2 == 0 { 1.0 } else { -1.0 });
        assert_eq!(centroid_hz(nyquist.view(), 2.0), 1.0);
        assert_eq!(centroid_hz(Array2::from_elem((1, 8), 3.0).view(), 2.0), 0.0);
        assert!(centroid_hz(Array2::zeros((1, 8)).view(), 2.0).is_nan());
        // A unit cosine at bin 1 of 8 and twice the power at bin 2: (1·0.25 + 4·0.5)/(1 + 4) Hz.
        let tone = |k: f64, a: f64| {
            (0..8).map(move |t| a * (2.0 * std::f64::consts::PI * k * t as f64 / 8.0).cos())
        };
        let rows =
            Array2::from_shape_vec((2, 8), tone(1.0, 1.0).chain(tone(2.0, 2.0)).collect()).unwrap();
        assert!((centroid_hz(rows.view(), 2.0) - (0.25 + 4.0 * 0.5) / 5.0).abs() < 1e-12);
    }
}
