//! Empirical mode decomposition, per channel (EMD) and multivariate (MEMD), on one sifting
//! engine. Sources and choices are in docs/emd.md.

mod boundary;
mod decomposers;
mod directions;
mod ensemble;
mod extrema;
mod layout;
mod sift;
mod spline;

pub use decomposers::{Emd, Memd};
pub use ensemble::{Ceemdan, Eemd};
pub use sift::Stop;

#[doc(hidden)]
pub mod testing {
    //! Internals exposed for accuracy tests against scipy and PyEMD.

    /// Knot positions and sample indices of the lower and upper envelopes, or `None`.
    pub type Mirrored = ((Vec<f64>, Vec<usize>), (Vec<f64>, Vec<usize>));

    pub fn mirror(indmin: &[usize], indmax: &[usize], x: &[f64]) -> Option<Mirrored> {
        let (lo, hi) = crate::boundary::mirror(indmin, indmax, x).ok()?;
        Some(((lo.positions, lo.samples), (hi.positions, hi.samples)))
    }

    /// EEMD and CEEMDAN of one channel with given noise realisations: (IMFs, residual).
    pub fn eemd(
        x: &[f64],
        noises: &[Vec<f64>],
        stop: crate::Stop,
        max_imfs: usize,
    ) -> (Vec<Vec<f64>>, Vec<f64>) {
        let noises: Vec<_> = noises
            .iter()
            .map(|w| ndarray::Array1::from(w.clone()))
            .collect();
        let (imfs, r) =
            crate::ensemble::eemd_channel(ndarray::ArrayView1::from(x), &noises, stop, max_imfs);
        (imfs.into_iter().map(|m| m.to_vec()).collect(), r.to_vec())
    }

    pub fn ceemdan(
        x: &[f64],
        noises: &[Vec<f64>],
        stop: crate::Stop,
        epsilon: f64,
        max_imfs: Option<usize>,
    ) -> (Vec<Vec<f64>>, Vec<f64>) {
        let noises: Vec<_> = noises
            .iter()
            .map(|w| ndarray::Array1::from(w.clone()))
            .collect();
        let (imfs, r) = crate::ensemble::ceemdan_channel(
            ndarray::ArrayView1::from(x),
            &noises,
            stop,
            epsilon,
            max_imfs,
        );
        (imfs.into_iter().map(|m| m.to_vec()).collect(), r.to_vec())
    }

    pub fn centroid_hz(rows: ndarray::ArrayView2<f64>, fs_hz: f64) -> f64 {
        modewise_fourier::centroid_hz(rows, fs_hz)
    }

    pub fn spline(
        knots: &[f64],
        values: ndarray::ArrayView2<f64>,
        n: usize,
    ) -> ndarray::Array2<f64> {
        crate::spline::interpolate(knots, values, n)
    }
}
