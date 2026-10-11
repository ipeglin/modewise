use modewise_decomposition::{Decomposition, ModeDecomposer};
use modewise_method::{MethodDescriptor, MethodError};
use modewise_signals::MultichannelSeries;
use ndarray::{Array2, Axis, s};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};

use crate::{Mvmd, MvmdParams};

/// White-noise channels added before MVMD. Defaults follow the reference `na_mvmd.m`.
#[derive(Debug, Clone, PartialEq)]
pub struct NoiseParams {
    pub channels: usize,
    /// Noise SD, relative to the unit-SD signal channels.
    pub sd: f64,
    pub seed: u64,
}

impl Default for NoiseParams {
    fn default() -> Self {
        Self {
            channels: 1,
            sd: 0.8,
            seed: 0,
        }
    }
}

/// Noise-assisted MVMD (Zisou, Apostolidis and Hadjileontiadis, ICASSP 2021). Each channel is
/// scaled to unit SD (ddof = 1), white-noise channels are appended, MVMD runs on the whole set,
/// the noise channels are dropped and the modes scaled back. The paper's generalised
/// cross-spectrum centroid for one snapshot is Σ_c Σ_f f|û_c|² / Σ_c Σ_f |û_c|², the update
/// MVMD already uses.
#[derive(Debug, Clone)]
pub struct NaMvmd {
    inner: Mvmd,
    noise: NoiseParams,
}

impl NaMvmd {
    pub fn new(params: MvmdParams, noise: NoiseParams) -> Result<Self, MethodError> {
        if noise.channels == 0 {
            return Err(MethodError::invalid("noise_channels", "must be at least 1"));
        }
        if !(noise.sd.is_finite() && noise.sd > 0.0) {
            return Err(MethodError::invalid(
                "noise_sd",
                "must be finite and positive",
            ));
        }
        Ok(Self {
            inner: Mvmd::new(params)?,
            noise,
        })
    }

    /// The unit-SD signal with the noise channels appended, and each channel's original SD.
    pub fn augment(
        &self,
        signal: &MultichannelSeries,
    ) -> Result<(MultichannelSeries, Vec<f64>), MethodError> {
        let (n_channels, n_samples) = signal.data().dim();
        let sds: Vec<f64> = signal
            .data()
            .axis_iter(Axis(0))
            .map(|row| row.std(1.0))
            .collect();
        if let Some(c) = sds.iter().position(|&sd| !(sd > 0.0 && sd.is_finite())) {
            return Err(MethodError::invalid(
                "signal",
                format!("channel {:?} is constant", signal.labels()[c]),
            ));
        }
        let mut data = Array2::zeros((n_channels + self.noise.channels, n_samples));
        for (c, sd) in sds.iter().enumerate() {
            data.row_mut(c)
                .assign(&signal.data().row(c).mapv(|v| v / sd));
        }
        let mut rng = ChaCha8Rng::seed_from_u64(self.noise.seed);
        for v in data.slice_mut(s![n_channels.., ..]).iter_mut() {
            let z: f64 = StandardNormal.sample(&mut rng);
            *v = self.noise.sd * z;
        }
        let mut labels = signal.labels().to_vec();
        labels.extend((0..self.noise.channels).map(|i| format!("noise{i}")));
        Ok((
            MultichannelSeries::new(data, signal.sampling_rate(), labels)
                .map_err(|e| MethodError::ShapeMismatch(e.to_string()))?,
            sds,
        ))
    }
}

impl ModeDecomposer for NaMvmd {
    fn descriptor(&self) -> MethodDescriptor {
        MethodDescriptor {
            name: "na_mvmd",
            version: env!("CARGO_PKG_VERSION"),
            multivariate: true,
        }
    }

    fn decompose(&self, signal: &MultichannelSeries) -> Result<Decomposition, MethodError> {
        let (augmented, sds) = self.augment(signal)?;
        let full = self.inner.decompose(&augmented)?;
        let mut modes = full
            .modes()
            .slice(s![.., ..signal.n_channels(), ..])
            .to_owned();
        for (c, sd) in sds.iter().enumerate() {
            modes.index_axis_mut(Axis(1), c).mapv_inplace(|v| v * sd);
        }
        Decomposition::new(
            modes,
            full.centre_frequencies_hz().to_vec(),
            signal.sampling_rate(),
            signal.labels().to_vec(),
            true,
        )
    }
}
