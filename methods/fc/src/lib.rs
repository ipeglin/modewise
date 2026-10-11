//! Functional connectivity: Pearson correlation and phase locking value, for a whole run or
//! per epoch (docs/connectivity.md).

mod measures;

use modewise_connectivity::{FunctionalConnectivity, Measure, WindowInfo};
use modewise_decomposition::Decomposition;
use modewise_epochs::{EpochSource, Epochs};
use modewise_hilbert::Padding;
use modewise_method::MethodError;
use modewise_signals::MultichannelSeries;
use ndarray::{Array4, ArrayView2};
use rayon::prelude::*;

pub use measures::{correlation, phase_locking, unit_phases};

/// What to compute functional connectivity on.
#[derive(Debug, Clone, Copy)]
pub enum Source<'a> {
    Series(&'a MultichannelSeries),
    /// One matrix per mode.
    Decomposition(&'a Decomposition),
    /// One matrix per epoch (and per mode for a decomposition source). Phases come from the
    /// full run before cutting, so epoch edges add no Hilbert edge effects.
    Epochs(&'a Epochs),
}

/// Functional connectivity of every component, laid out as window × component × channel × channel.
pub fn functional_connectivity(
    source: Source<'_>,
    measure: Measure,
) -> Result<FunctionalConnectivity, MethodError> {
    let (components, labels, centres, epochs) = match source {
        Source::Series(s) => (vec![s.data()], s.labels().to_vec(), None, None),
        Source::Decomposition(d) => (
            modes(d),
            d.labels().to_vec(),
            Some(d.centre_frequencies_hz().to_vec()),
            None,
        ),
        Source::Epochs(e) => match e.source() {
            EpochSource::Series(s) => (vec![s.data()], s.labels().to_vec(), None, Some(e)),
            EpochSource::Decomposition(d) => (
                modes(d),
                d.labels().to_vec(),
                Some(d.centre_frequencies_hz().to_vec()),
                Some(e),
            ),
            _ => {
                return Err(MethodError::invalid(
                    "data",
                    "functional connectivity needs epochs of a series or a decomposition",
                ));
            }
        },
    };
    let n_samples = components.first().map_or(0, |c| c.ncols());
    let windows: Vec<(usize, usize)> = match epochs {
        Some(e) => e.windows().iter().map(|w| (w.start, w.end)).collect(),
        None => vec![(0, n_samples)],
    };

    let n_channels = labels.len();
    let mut values = Array4::zeros((windows.len(), components.len(), n_channels, n_channels));
    let phases: Option<Vec<_>> = match measure {
        Measure::PhaseLockingValue => Some(
            components
                .par_iter()
                .map(|c| unit_phases(*c, Padding::Mirror))
                .collect(),
        ),
        Measure::Pearson { .. } => None,
    };
    let cells: Vec<(usize, usize)> = (0..windows.len())
        .flat_map(|w| (0..components.len()).map(move |k| (w, k)))
        .collect();
    let matrices: Vec<_> = cells
        .par_iter()
        .map(|&(w, k)| {
            let (start, end) = windows[w];
            let range = ndarray::s![.., start..end];
            match (&phases, measure) {
                (Some(p), _) => phase_locking(&p[k].0.slice(range), &p[k].1.slice(range), &p[k].2),
                (None, Measure::Pearson { fisher_z }) => {
                    let r = correlation(components[k].slice(range));
                    if fisher_z { measures::fisher_z(r) } else { r }
                }
                (None, Measure::PhaseLockingValue) => {
                    unreachable!("phases exist for the phase locking value")
                }
            }
        })
        .collect();
    for (&(w, k), matrix) in cells.iter().zip(matrices) {
        values.slice_mut(ndarray::s![w, k, .., ..]).assign(&matrix);
    }

    let mut result = FunctionalConnectivity::new(values, measure, labels)?;
    if let Some(hz) = centres {
        result = result.with_centre_frequencies(hz)?;
    }
    if let Some(e) = epochs {
        let info = e
            .windows()
            .iter()
            .map(|w| WindowInfo {
                label: w.label.clone(),
                event_onset_s: w.event_onset_s,
            })
            .collect();
        result = result.with_windows(info, e.resolvable())?;
    }
    Ok(result)
}

fn modes(d: &Decomposition) -> Vec<ArrayView2<'_, f64>> {
    (0..d.n_modes()).map(|k| d.mode(k)).collect()
}
