use modewise_decomposition::{Decomposition, ModeDecomposer};
use modewise_fourier::centroid_hz;
use modewise_method::{MethodDescriptor, MethodError};
use modewise_signals::MultichannelSeries;
use ndarray::{Array3, Axis, s};
use rayon::prelude::*;

use crate::filters::FILTERS;
use crate::modwt::{Filters, mra};

/// How the series is extended past its ends.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Boundary {
    /// Circular, as the MODWT is defined (MATLAB and PyWavelets default).
    #[default]
    Periodic,
    /// The series followed by its reverse, then cut back to the original length (Percival and
    /// Walden 2000; MATLAB `modwt(..., 'reflection')`). Fewer edge effects for non-periodic series.
    Reflection,
}

/// Wavelets available, by PyWavelets name.
pub fn wavelet_names() -> impl Iterator<Item = &'static str> {
    FILTERS.iter().map(|(name, _)| *name)
}

/// MODWT multiresolution analysis (Percival and Walden 2000): details D_1..D_J and the smooth
/// S_J of each channel, additive and shift-invariant. Mode 0 is S_J and the last mode D_1. The
/// same filter bank is used for every channel, so modes are aligned across channels.
#[derive(Debug, Clone)]
pub struct ModwtMra {
    wavelet: &'static str,
    filter: &'static [f64],
    levels: Option<usize>,
    boundary: Boundary,
}

impl ModwtMra {
    pub fn new(
        wavelet: &str,
        levels: Option<usize>,
        boundary: Boundary,
    ) -> Result<Self, MethodError> {
        let Some(&(name, filter)) = FILTERS.iter().find(|(name, _)| *name == wavelet) else {
            let known: Vec<_> = wavelet_names().collect();
            return Err(MethodError::invalid(
                "wavelet",
                format!("unknown {wavelet:?}; choose one of {}", known.join(", ")),
            ));
        };
        if levels == Some(0) {
            return Err(MethodError::invalid("levels", "must be at least 1"));
        }
        Ok(Self {
            wavelet: name,
            filter,
            levels,
            boundary,
        })
    }

    /// The deepest level whose equivalent filter, (2^J − 1)(L − 1) + 1 long, fits in `n`
    /// samples (Percival and Walden 2000), and at least 1.
    pub fn default_levels(&self, n: usize) -> usize {
        let l = self.filter.len();
        (1..)
            .take_while(|&j: &usize| ((1usize << j) - 1) * (l - 1) < n)
            .last()
            .unwrap_or(1)
    }

    pub fn wavelet(&self) -> &str {
        self.wavelet
    }
}

impl ModeDecomposer for ModwtMra {
    fn descriptor(&self) -> MethodDescriptor {
        MethodDescriptor {
            name: "modwt_mra",
            version: env!("CARGO_PKG_VERSION"),
            multivariate: false,
        }
    }

    fn decompose(&self, signal: &MultichannelSeries) -> Result<Decomposition, MethodError> {
        let (n_channels, n) = signal.data().dim();
        let levels = self.levels.unwrap_or_else(|| self.default_levels(n));
        let longest = (n.ilog2() as usize).max(1);
        if levels > longest {
            return Err(MethodError::invalid(
                "levels",
                format!("at most ⌊log₂ n⌋ = {longest} for {n} samples"),
            ));
        }
        let filters = Filters::from_scaling(self.filter);
        let rows: Vec<Vec<f64>> = signal
            .data()
            .rows()
            .into_iter()
            .map(|r| r.to_vec())
            .collect();
        let channels: Vec<_> = rows
            .par_iter()
            .map(|x| match self.boundary {
                Boundary::Periodic => mra(x, &filters, levels),
                Boundary::Reflection => {
                    let mut extended = x.clone();
                    extended.extend(x.iter().rev());
                    let (details, smooth) = mra(&extended, &filters, levels);
                    (
                        details.into_iter().map(|d| d[..n].to_vec()).collect(),
                        smooth[..n].to_vec(),
                    )
                }
            })
            .collect();
        let mut modes = Array3::zeros((levels + 1, n_channels, n));
        for (c, (details, smooth)) in channels.iter().enumerate() {
            modes
                .slice_mut(s![0, c, ..])
                .assign(&ndarray::ArrayView1::from(smooth.as_slice()));
            for (j, d) in details.iter().enumerate() {
                modes
                    .slice_mut(s![levels - j, c, ..])
                    .assign(&ndarray::ArrayView1::from(d.as_slice()));
            }
        }
        let fs = signal.sampling_rate();
        let centres = modes
            .axis_iter(Axis(0))
            .map(|m| centroid_hz(m, fs.hz()))
            .collect();
        Decomposition::new(modes, centres, fs, signal.labels().to_vec(), true)
    }
}
