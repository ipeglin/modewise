//! Per-channel decompositions laid out as one `Decomposition`, aligned by IMF index from the top.

use modewise_decomposition::Decomposition;
use modewise_method::MethodError;
use modewise_signals::MultichannelSeries;
use ndarray::{Array1, Array2, Array3, ArrayView1, Axis, s};
use rayon::prelude::*;

use crate::directions::directions;
use crate::sift::{Stop, decompose};
use modewise_fourier::centroid_hz;

/// One channel's IMFs (highest frequency first) and residual.
pub(crate) type Channel = (Vec<Array1<f64>>, Array1<f64>);

/// EMD of one channel.
pub(crate) fn emd1(x: ArrayView1<f64>, stop: Stop, max_imfs: Option<usize>) -> Channel {
    let (imfs, residual) = decompose(
        x.insert_axis(Axis(1)),
        &directions(1, 1),
        stop,
        max_imfs,
        false,
    );
    (
        imfs.into_iter().map(|m| m.column(0).to_owned()).collect(),
        residual.column(0).to_owned(),
    )
}

/// Runs `channel` on every channel in parallel. Mode 0 is the residual and the last mode every
/// channel's IMF 1; a channel with fewer IMFs has zero modes (NaN frequency) above its residual.
pub(crate) fn per_channel(
    signal: &MultichannelSeries,
    channel: impl Fn(usize, ArrayView1<f64>) -> Channel + Sync,
) -> Result<Decomposition, MethodError> {
    let rows: Vec<Array1<f64>> = signal
        .data()
        .rows()
        .into_iter()
        .map(|r| r.to_owned())
        .collect();
    let results: Vec<Channel> = rows
        .par_iter()
        .enumerate()
        .map(|(c, row)| channel(c, row.view()))
        .collect();
    let top = results
        .iter()
        .map(|(imfs, _)| imfs.len())
        .max()
        .unwrap_or(0);
    let (n_channels, n) = signal.data().dim();
    let fs = signal.sampling_rate();
    let mut modes = Array3::zeros((top + 1, n_channels, n));
    let mut freqs = Array2::from_elem((top + 1, n_channels), f64::NAN);
    for (c, (imfs, residual)) in results.iter().enumerate() {
        let slots = std::iter::once((0, residual))
            .chain(imfs.iter().enumerate().map(|(j, imf)| (top - j, imf)));
        for (slot, mode) in slots {
            modes.slice_mut(s![slot, c, ..]).assign(mode);
            freqs[[slot, c]] = centroid_hz(mode.view().insert_axis(Axis(0)), fs.hz());
        }
    }
    Decomposition::per_channel(modes, freqs, fs, signal.labels().to_vec())
}
