use modewise_decomposition::InstantaneousModes;
use modewise_method::MethodError;
use modewise_spectral::{FrequencyGrid, TimeFrequencyMap};
use ndarray::Array3;

/// What each sample contributes to its frequency bin.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Weight {
    /// Amplitude squared (the `emd` package default).
    #[default]
    Power,
    Amplitude,
}

pub struct HilbertSpectrum {
    pub map: TimeFrequencyMap,
    /// Samples whose instantaneous frequency fell outside the grid (or was not finite).
    pub dropped: usize,
    pub total: usize,
}

/// Hilbert spectrum, channel × frequency × sample: each sample's weight is added to the bin
/// holding its instantaneous frequency, summed over modes (Huang et al. 1998).
pub fn hilbert_spectrum(
    modes: &InstantaneousModes,
    grid: &FrequencyGrid,
    weight: Weight,
) -> Result<HilbertSpectrum, MethodError> {
    let (n_modes, n_channels, n_samples) = modes.amplitude().dim();
    let mut power = Array3::zeros((n_channels, grid.n_bins(), n_samples));
    let amplitude = modes.amplitude();
    let frequency = modes.frequency_hz();
    let mut dropped = 0;
    for k in 0..n_modes {
        for c in 0..n_channels {
            for t in 0..n_samples {
                match grid.bin(frequency[[k, c, t]]) {
                    Some(bin) => {
                        let a = amplitude[[k, c, t]];
                        power[[c, bin, t]] += match weight {
                            Weight::Power => a * a,
                            Weight::Amplitude => a,
                        };
                    }
                    None => dropped += 1,
                }
            }
        }
    }
    let fs = modes.sampling_rate().hz();
    let times = (0..n_samples).map(|t| t as f64 / fs).collect();
    let map = TimeFrequencyMap::new(power, grid.centres(), times, modes.labels().to_vec())?
        .with_sampling_rate(modes.sampling_rate())?;
    Ok(HilbertSpectrum {
        map,
        dropped,
        total: n_modes * n_channels * n_samples,
    })
}
