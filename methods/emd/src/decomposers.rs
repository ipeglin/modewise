use modewise_decomposition::{Decomposition, ModeDecomposer};
use modewise_method::{MethodDescriptor, MethodError};
use modewise_signals::MultichannelSeries;
use ndarray::{Array3, Axis};
use tracing::warn;

use crate::directions::directions;
use crate::layout::{emd1, per_channel};
use crate::sift::{Stop, decompose};
use modewise_fourier::centroid_hz;

pub(crate) fn validate(stop: Stop, max_imfs: Option<usize>) -> Result<(), MethodError> {
    match stop {
        Stop::Rilling { sd, sd2, tol } => {
            if ![sd, sd2, tol].iter().all(|v| v.is_finite() && *v > 0.0) {
                return Err(MethodError::invalid(
                    "stop",
                    "rilling thresholds must be finite and positive",
                ));
            }
        }
        Stop::SNumber { count: 0 } | Stop::Fixed { siftings: 0 } => {
            return Err(MethodError::invalid("stop", "the count must be at least 1"));
        }
        _ => {}
    }
    if max_imfs == Some(0) {
        return Err(MethodError::invalid("max_imfs", "must be at least 1"));
    }
    Ok(())
}

/// Empirical mode decomposition (Huang et al. 1998) of each channel on its own.
///
/// Channels can give different numbers of IMFs. Modes are aligned by IMF index from the top:
/// the last mode is every channel's IMF 1 (highest frequency), mode 0 is the residual, and a
/// channel with fewer IMFs has zero modes (NaN frequency) just above its residual.
#[derive(Debug, Clone)]
pub struct Emd {
    stop: Stop,
    max_imfs: Option<usize>,
}

impl Emd {
    pub fn new(stop: Stop, max_imfs: Option<usize>) -> Result<Self, MethodError> {
        validate(stop, max_imfs)?;
        Ok(Self { stop, max_imfs })
    }
}

impl ModeDecomposer for Emd {
    fn descriptor(&self) -> MethodDescriptor {
        MethodDescriptor {
            name: "emd",
            version: env!("CARGO_PKG_VERSION"),
            multivariate: false,
        }
    }

    fn decompose(&self, signal: &MultichannelSeries) -> Result<Decomposition, MethodError> {
        per_channel(signal, |_, x| emd1(x, self.stop, self.max_imfs))
    }
}

/// Multivariate EMD (Rehman and Mandic 2010): envelopes from projections of all channels along
/// `num_directions` Hammersley directions, so every channel gets the same IMFs.
#[derive(Debug, Clone)]
pub struct Memd {
    stop: Stop,
    max_imfs: Option<usize>,
    num_directions: usize,
    memd_flat_rule: bool,
}

impl Memd {
    pub fn new(
        stop: Stop,
        max_imfs: Option<usize>,
        num_directions: usize,
    ) -> Result<Self, MethodError> {
        validate(stop, max_imfs)?;
        if num_directions < 6 {
            return Err(MethodError::invalid(
                "num_directions",
                "must be at least 6, as memd.m requires",
            ));
        }
        Ok(Self {
            stop,
            max_imfs,
            num_directions,
            memd_flat_rule: false,
        })
    }
}

impl Memd {
    /// Reproduces memd.m exactly, including its rule that a projection with every value below
    /// 1e-5 has no extrema. For comparisons with memd.m results only.
    #[doc(hidden)]
    pub fn with_memd_flat_rule(mut self) -> Self {
        self.memd_flat_rule = true;
        self
    }
}

impl ModeDecomposer for Memd {
    fn descriptor(&self) -> MethodDescriptor {
        MethodDescriptor {
            name: "memd",
            version: env!("CARGO_PKG_VERSION"),
            multivariate: true,
        }
    }

    fn decompose(&self, signal: &MultichannelSeries) -> Result<Decomposition, MethodError> {
        let (n_channels, n) = signal.data().dim();
        if n_channels < 3 {
            return Err(MethodError::invalid(
                "signal",
                "MEMD needs at least 3 channels, as memd.m; use emd for fewer",
            ));
        }
        if self.num_directions < 2 * n_channels {
            warn!(
                directions = self.num_directions,
                channels = n_channels,
                "fewer directions than twice the channels; envelopes may be poorly sampled"
            );
        }
        let data = signal.data();
        let x = data.t();
        let (imfs, residual) = decompose(
            x,
            &directions(self.num_directions, n_channels),
            self.stop,
            self.max_imfs,
            self.memd_flat_rule,
        );
        let fs = signal.sampling_rate();
        let mut modes = Array3::zeros((imfs.len() + 1, n_channels, n));
        modes.index_axis_mut(Axis(0), 0).assign(&residual.t());
        for (j, imf) in imfs.iter().enumerate() {
            modes
                .index_axis_mut(Axis(0), imfs.len() - j)
                .assign(&imf.t());
        }
        let centres = modes
            .axis_iter(Axis(0))
            .map(|m| centroid_hz(m, fs.hz()))
            .collect();
        Decomposition::new(modes, centres, fs, signal.labels().to_vec(), true)
    }
}
