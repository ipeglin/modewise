use modewise_connectivity::{FunctionalConnectivity, Measure};
use modewise_decomposition::Decomposition;
use modewise_epochs::{EpochSource, Epochs};
use modewise_pipeline::Artifact;
use modewise_spectral::TimeFrequencyMap;
use ndarray::{ArrayView2, Axis};

/// A named frequency band, [low, high) in Hz.
#[derive(Debug, Clone, PartialEq)]
pub struct Band {
    pub name: String,
    pub low_hz: f64,
    pub high_hz: f64,
}

/// Which features to take from each recording's output.
#[derive(Debug, Clone, PartialEq)]
pub enum Features {
    /// Upper-triangle edges of functional connectivity, per component. Pearson r becomes
    /// Fisher z, so values can be averaged and tested. Named `m{k}:{a}~{b}`.
    Edges { resolvable_only: bool },
    /// Variance of each mode per channel, plus each mode's centre frequency. Named
    /// `m{k}:{channel}` and `m{k}:centre_hz`. Modes are matched across recordings by rank.
    ModePower { resolvable_only: bool },
    /// Mean amplitude (√power) per channel within each band, over all frames: ALFF (Zang et al.
    /// 2007). With `relative`, the band's summed amplitude over that of every frequency above 0:
    /// fALFF (Zou et al. 2008). One row per band, with the band as the condition; features are
    /// the channels.
    BandAmplitude { bands: Vec<Band>, relative: bool },
    /// The mean signal per channel in each condition's epochs (each epoch's mean counts once):
    /// block averages, for example task blocks against control blocks. One row per condition.
    Mean,
}

/// Feature names and, per condition (`None` for a whole run), one row of values.
pub(crate) struct Extracted {
    pub names: Vec<String>,
    pub rows: Vec<(Option<String>, Vec<f64>)>,
}

impl Features {
    pub(crate) fn extract(&self, artifact: &Artifact) -> Result<Extracted, String> {
        match (self, artifact) {
            (Self::Edges { resolvable_only }, Artifact::FunctionalConnectivity(c)) => {
                Ok(edges(c, *resolvable_only))
            }
            (Self::ModePower { .. }, Artifact::Decomposition(d)) => Ok(mode_power_whole(d)),
            (Self::ModePower { resolvable_only }, Artifact::Epochs(e)) => {
                mode_power_epochs(e, *resolvable_only)
            }
            (Self::BandAmplitude { bands, relative }, Artifact::TimeFrequency(m)) => {
                band_amplitude(m, bands, *relative)
            }
            (Self::Mean, Artifact::Epochs(e)) => epoch_means(e),
            (Self::Mean, other) => Err(format!(
                "mean needs epochs of a series, not {}",
                other.kind()
            )),
            (Self::BandAmplitude { .. }, other) => Err(format!(
                "band amplitude needs a spectrum or time-frequency map, not {}",
                other.kind()
            )),
            (Self::Edges { .. }, other) => Err(format!(
                "edges need functional connectivity, not {}",
                other.kind()
            )),
            (Self::ModePower { .. }, other) => Err(format!(
                "mode power needs a decomposition or its epochs, not {}",
                other.kind()
            )),
        }
    }
}

/// Window indices per condition label, in order of first appearance; one unlabelled group
/// when there are no windows.
fn conditions(labels: Option<Vec<String>>, n_windows: usize) -> Vec<(Option<String>, Vec<usize>)> {
    let Some(labels) = labels else {
        return vec![(None, (0..n_windows).collect())];
    };
    let mut groups: Vec<(Option<String>, Vec<usize>)> = Vec::new();
    for (w, label) in labels.into_iter().enumerate() {
        match groups
            .iter_mut()
            .find(|(l, _)| l.as_deref() == Some(label.as_str()))
        {
            Some((_, ws)) => ws.push(w),
            None => groups.push((Some(label), vec![w])),
        }
    }
    groups
}

fn mean(values: impl Iterator<Item = f64>) -> f64 {
    let (sum, n) = values.fold((0.0, 0usize), |(s, n), v| (s + v, n + 1));
    if n == 0 { f64::NAN } else { sum / n as f64 }
}

fn edges(c: &FunctionalConnectivity, resolvable_only: bool) -> Extracted {
    let labels = c.labels();
    let n = labels.len();
    let to_z = c.measure() == (Measure::Pearson { fisher_z: false });
    let pairs: Vec<(usize, usize)> = (0..n)
        .flat_map(|i| (i + 1..n).map(move |j| (i, j)))
        .collect();
    let names = (0..c.n_components())
        .flat_map(|k| {
            pairs
                .iter()
                .map(move |&(i, j)| format!("m{k}:{}~{}", labels[i], labels[j]))
        })
        .collect();
    let groups = conditions(
        c.windows()
            .map(|w| w.iter().map(|w| w.label.clone()).collect()),
        c.n_windows(),
    );
    let rows = groups
        .into_iter()
        .map(|(condition, windows)| {
            let row = (0..c.n_components())
                .flat_map(|k| {
                    let usable: Vec<usize> = windows
                        .iter()
                        .copied()
                        .filter(|&w| !resolvable_only || c.resolvable().is_none_or(|r| r[[w, k]]))
                        .collect();
                    pairs.iter().map(move |&(i, j)| {
                        mean(usable.iter().map(|&w| {
                            let v = c.matrix(w, k)[[i, j]];
                            if to_z { v.atanh() } else { v }
                        }))
                    })
                })
                .collect();
            (condition, row)
        })
        .collect();
    Extracted { names, rows }
}

fn power_names(labels: &[String], n_modes: usize) -> Vec<String> {
    (0..n_modes)
        .flat_map(|k| {
            std::iter::once(format!("m{k}:centre_hz"))
                .chain(labels.iter().map(move |l| format!("m{k}:{l}")))
        })
        .collect()
}

fn variance(x: ArrayView2<f64>, channel: usize) -> f64 {
    x.index_axis(Axis(0), channel).var(0.0)
}

fn mode_power_whole(d: &Decomposition) -> Extracted {
    let row = (0..d.n_modes())
        .flat_map(|k| {
            let mode = d.mode(k);
            std::iter::once(d.centre_frequencies_hz()[k])
                .chain((0..mode.nrows()).map(move |c| variance(mode, c)))
        })
        .collect();
    Extracted {
        names: power_names(d.labels(), d.n_modes()),
        rows: vec![(None, row)],
    }
}

fn mode_power_epochs(e: &Epochs, resolvable_only: bool) -> Result<Extracted, String> {
    let EpochSource::Decomposition(d) = e.source() else {
        return Err("mode power needs epochs of a decomposition".into());
    };
    let resolvable = e.resolvable();
    let groups = conditions(
        Some(e.windows().iter().map(|w| w.label.clone()).collect()),
        e.windows().len(),
    );
    let rows = groups
        .into_iter()
        .map(|(condition, windows)| {
            let row = (0..d.n_modes())
                .flat_map(|k| {
                    let usable: Vec<usize> = windows
                        .iter()
                        .copied()
                        .filter(|&w| {
                            !resolvable_only || resolvable.as_ref().is_none_or(|r| r[[w, k]])
                        })
                        .collect();
                    let channels = d.labels().len();
                    std::iter::once(d.centre_frequencies_hz()[k]).chain((0..channels).map(
                        move |c| {
                            mean(
                                usable
                                    .iter()
                                    .filter_map(|&w| e.modes(w))
                                    .map(|m| variance(m.index_axis(Axis(0), k), c)),
                            )
                        },
                    ))
                })
                .collect();
            (condition, row)
        })
        .collect();
    Ok(Extracted {
        names: power_names(d.labels(), d.n_modes()),
        rows,
    })
}

fn band_amplitude(
    m: &TimeFrequencyMap,
    bands: &[Band],
    relative: bool,
) -> Result<Extracted, String> {
    let freqs = m.frequencies_hz();
    let amplitude = m.power().mapv(f64::sqrt);
    let total: Vec<usize> = (0..freqs.len()).filter(|&k| freqs[k] > 0.0).collect();
    let mut rows = Vec::with_capacity(bands.len());
    for (i, band) in bands.iter().enumerate() {
        if !(band.low_hz.is_finite()
            && band.high_hz.is_finite()
            && 0.0 <= band.low_hz
            && band.low_hz < band.high_hz)
        {
            return Err(format!("band {:?} needs 0 ≤ low < high", band.name));
        }
        if bands[..i].iter().any(|b| b.name == band.name) {
            return Err(format!("band {:?} is listed twice", band.name));
        }
        let bins: Vec<usize> = (0..freqs.len())
            .filter(|&k| band.low_hz <= freqs[k] && freqs[k] < band.high_hz)
            .collect();
        if bins.is_empty() {
            return Err(format!(
                "band {:?} has no frequency bins at this resolution",
                band.name
            ));
        }
        let row = amplitude
            .axis_iter(Axis(0))
            .map(|channel| {
                let sum = |ks: &[usize]| ks.iter().map(|&k| channel.row(k).sum()).sum::<f64>();
                if relative {
                    sum(&bins) / sum(&total)
                } else {
                    sum(&bins) / (bins.len() * channel.ncols()) as f64
                }
            })
            .collect();
        rows.push((Some(band.name.clone()), row));
    }
    Ok(Extracted {
        names: m.labels().to_vec(),
        rows,
    })
}

fn epoch_means(e: &Epochs) -> Result<Extracted, String> {
    let EpochSource::Series(series) = e.source() else {
        return Err(
            "mean needs epochs of a series; decompose after epoching or use mode_power".into(),
        );
    };
    let groups = conditions(
        Some(e.windows().iter().map(|w| w.label.clone()).collect()),
        e.windows().len(),
    );
    let channels = series.labels().len();
    let rows = groups
        .into_iter()
        .map(|(condition, windows)| {
            let row = (0..channels)
                .map(|c| {
                    mean(
                        windows
                            .iter()
                            .filter_map(|&w| e.series(w))
                            .map(|x| mean(x.row(c).iter().copied())),
                    )
                })
                .collect();
            (condition, row)
        })
        .collect();
    Ok(Extracted {
        names: series.labels().to_vec(),
        rows,
    })
}
