use modewise_decomposition::{Decomposition, ModeDecomposer};
use modewise_method::{MethodDescriptor, MethodError};
use modewise_signals::MultichannelSeries;
use ndarray::{Array2, Array3, Axis};
use rayon::prelude::*;

use crate::{Mvmd, MvmdParams};

/// Variational Mode Decomposition of each channel on its own: MVMD with one channel, which is
/// VMD (Rehman and Aftab 2019, section III; same grid, mirror and update as vmdpy). Mode k is
/// the k-th lowest centre frequency of its channel, so modes are not aligned across channels.
#[derive(Debug, Clone)]
pub struct Vmd {
    inner: Mvmd,
}

impl Vmd {
    pub fn new(params: MvmdParams) -> Result<Self, MethodError> {
        Ok(Self {
            inner: Mvmd::new(params)?,
        })
    }
}

impl ModeDecomposer for Vmd {
    fn descriptor(&self) -> MethodDescriptor {
        MethodDescriptor {
            name: "vmd",
            version: env!("CARGO_PKG_VERSION"),
            multivariate: false,
        }
    }

    fn decompose(&self, signal: &MultichannelSeries) -> Result<Decomposition, MethodError> {
        let fs = signal.sampling_rate();
        let rows: Vec<_> = signal
            .data()
            .axis_iter(Axis(0))
            .map(|r| r.to_owned())
            .collect();
        let channels = rows
            .into_par_iter()
            .map(|row| {
                let single =
                    MultichannelSeries::new(row.insert_axis(Axis(0)), fs, vec!["x".into()])
                        .map_err(|e| MethodError::ShapeMismatch(e.to_string()))?;
                self.inner.decompose(&single)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let k = self.inner.params().num_modes;
        let mut modes = Array3::zeros((k, signal.n_channels(), signal.n_samples()));
        let mut centres = Array2::zeros((k, signal.n_channels()));
        for (c, d) in channels.iter().enumerate() {
            modes
                .index_axis_mut(Axis(1), c)
                .assign(&d.modes().index_axis(Axis(1), 0));
            centres
                .column_mut(c)
                .assign(&ndarray::ArrayView1::from(d.centre_frequencies_hz()));
        }
        Decomposition::per_channel(modes, centres, fs, signal.labels().to_vec())
    }
}
