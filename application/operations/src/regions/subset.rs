//! Keeping some channels, or averaging groups of them, for every artifact kind with channels.

use modewise_brain::ChannelTable;
use modewise_connectivity::{FunctionalConnectivity, Measure};
use modewise_decomposition::{Decomposition, InstantaneousModes, Loadings};
use modewise_epochs::{EpochSource, Epochs};
use modewise_pipeline::Artifact;
use modewise_signals::MultichannelSeries;
use modewise_spectral::TimeFrequencyMap;
use ndarray::{Array, Array2, Array4, ArrayView, Axis, Dimension, RemoveAxis};

/// Channel labels of an artifact, if it has channels.
pub(crate) fn labels(artifact: &Artifact) -> Option<&[String]> {
    Some(match artifact {
        Artifact::Series(s) => s.labels(),
        Artifact::Decomposition(d) => d.labels(),
        Artifact::Instantaneous(m) => m.labels(),
        Artifact::TimeFrequency(m) => m.labels(),
        Artifact::FunctionalConnectivity(c) => c.labels(),
        Artifact::Loadings(l) => l.channel_labels(),
        Artifact::Epochs(e) => return source_labels(e.source()),
        Artifact::Events(_) | Artifact::Channels(_) => return None,
    })
}

fn source_labels(source: &EpochSource) -> Option<&[String]> {
    Some(match source {
        EpochSource::Series(s) => s.labels(),
        EpochSource::Decomposition(d) => d.labels(),
        EpochSource::Instantaneous(m) => m.labels(),
        EpochSource::TimeFrequency(m) => m.labels(),
    })
}

/// How to turn a group of channels into one: keep a subset, or average the group.
pub(crate) trait Combine {
    /// Combines along `axis` of `values`, one output per group.
    fn along<D: Dimension + RemoveAxis>(
        &self,
        values: ArrayView<f64, D>,
        axis: usize,
    ) -> Array<f64, D>;
    fn labels(&self) -> Vec<String>;
}

/// Keeps the channels at these positions.
pub(crate) struct Keep<'a> {
    pub positions: &'a [usize],
    pub labels: &'a [String],
}

impl Combine for Keep<'_> {
    fn along<D: Dimension + RemoveAxis>(
        &self,
        values: ArrayView<f64, D>,
        axis: usize,
    ) -> Array<f64, D> {
        values.select(Axis(axis), self.positions)
    }

    fn labels(&self) -> Vec<String> {
        self.positions
            .iter()
            .map(|&i| self.labels[i].clone())
            .collect()
    }
}

/// Averages each named group of channel positions.
pub(crate) struct Average {
    pub groups: Vec<(String, Vec<usize>)>,
}

impl Combine for Average {
    fn along<D: Dimension + RemoveAxis>(
        &self,
        values: ArrayView<f64, D>,
        axis: usize,
    ) -> Array<f64, D> {
        let mut out = values.select(Axis(axis), &vec![0; self.groups.len()]);
        for (g, (_, members)) in self.groups.iter().enumerate() {
            let mut lane = out.index_axis_mut(Axis(axis), g);
            lane.fill(0.0);
            for &m in members {
                lane += &values.index_axis(Axis(axis), m);
            }
            lane /= members.len() as f64;
        }
        out
    }

    fn labels(&self) -> Vec<String> {
        self.groups.iter().map(|(name, _)| name.clone()).collect()
    }
}

/// Applies `combine` to the channel axis of every array in the artifact.
pub(crate) fn combine_channels(
    artifact: &Artifact,
    combine: &impl Combine,
) -> Result<Artifact, String> {
    let fail = |e: modewise_method::MethodError| e.to_string();
    Ok(match artifact {
        Artifact::Series(s) => Artifact::Series(series(s, combine)?),
        Artifact::Decomposition(d) => Artifact::Decomposition(decomposition(d, combine)?),
        Artifact::Instantaneous(m) => Artifact::Instantaneous(instantaneous(m, combine)?),
        Artifact::TimeFrequency(m) => Artifact::TimeFrequency(time_frequency(m, combine)?),
        Artifact::Loadings(l) => Artifact::Loadings(
            Loadings::new(
                combine.along(l.values(), 1),
                l.latent_labels().to_vec(),
                combine.labels(),
            )
            .map_err(fail)?,
        ),
        Artifact::Epochs(e) => {
            let source = match e.source() {
                EpochSource::Series(s) => EpochSource::Series(series(s, combine)?),
                EpochSource::Decomposition(d) => {
                    EpochSource::Decomposition(decomposition(d, combine)?)
                }
                EpochSource::Instantaneous(m) => {
                    EpochSource::Instantaneous(instantaneous(m, combine)?)
                }
                EpochSource::TimeFrequency(m) => {
                    EpochSource::TimeFrequency(time_frequency(m, combine)?)
                }
            };
            Artifact::Epochs(
                Epochs::from_parts(
                    source,
                    e.windows().to_vec(),
                    e.dropped().to_vec(),
                    e.params().clone(),
                )
                .map_err(|e| e.to_string())?,
            )
        }
        Artifact::FunctionalConnectivity(_) | Artifact::Events(_) | Artifact::Channels(_) => {
            return Err(format!(
                "{} has no channel axis to combine this way",
                artifact.kind()
            ));
        }
    })
}

fn series(s: &MultichannelSeries, c: &impl Combine) -> Result<MultichannelSeries, String> {
    MultichannelSeries::new(c.along(s.data(), 0), s.sampling_rate(), c.labels())
        .map_err(|e| e.to_string())
}

fn decomposition(d: &Decomposition, c: &impl Combine) -> Result<Decomposition, String> {
    let combined = match d.channel_centre_frequencies_hz() {
        Some(per_channel) => Decomposition::per_channel(
            c.along(d.modes(), 1),
            c.along(per_channel, 1),
            d.sampling_rate(),
            c.labels(),
        ),
        None => Decomposition::new(
            c.along(d.modes(), 1),
            d.centre_frequencies_hz().to_vec(),
            d.sampling_rate(),
            c.labels(),
            d.aligned_across_channels(),
        ),
    };
    combined.map_err(|e| e.to_string())
}

fn instantaneous(m: &InstantaneousModes, c: &impl Combine) -> Result<InstantaneousModes, String> {
    InstantaneousModes::new(
        c.along(m.amplitude(), 1),
        c.along(m.frequency_hz(), 1),
        m.centre_frequencies_hz().to_vec(),
        m.sampling_rate(),
        c.labels(),
    )
    .map_err(|e| e.to_string())
}

fn time_frequency(m: &TimeFrequencyMap, c: &impl Combine) -> Result<TimeFrequencyMap, String> {
    let mut map = TimeFrequencyMap::new(
        c.along(m.power(), 0),
        m.frequencies_hz().to_vec(),
        m.times_s().to_vec(),
        c.labels(),
    )
    .map_err(|e| e.to_string())?;
    if let Some(fs) = m.sampling_rate() {
        map = map.with_sampling_rate(fs).map_err(|e| e.to_string())?;
    }
    if let Some(r) = m.reliable() {
        map = map.with_reliable(r.to_owned()).map_err(|e| e.to_string())?;
    }
    Ok(map)
}

/// Keeps the rows and columns at `positions`.
pub(crate) fn keep_connectivity(
    c: &FunctionalConnectivity,
    positions: &[usize],
) -> Result<FunctionalConnectivity, String> {
    let values = c
        .values()
        .select(Axis(2), positions)
        .select(Axis(3), positions);
    let labels = positions.iter().map(|&i| c.labels()[i].clone()).collect();
    rebuild(c, values, c.measure(), labels)
}

/// Group × group matrices: the mean over every pair of distinct channels, one from each group.
/// Pearson r is averaged as Fisher z (and reported as z); z and PLV are averaged directly.
/// Pairs with NaN are skipped; a group pair with no valid pair gives NaN.
pub(crate) fn average_connectivity(
    c: &FunctionalConnectivity,
    groups: &[(String, Vec<usize>)],
) -> Result<FunctionalConnectivity, String> {
    let identity: fn(f64) -> f64 = |v| v;
    let (to_mean, measure): (fn(f64) -> f64, Measure) = match c.measure() {
        Measure::Pearson { fisher_z: false } => (f64::atanh, Measure::Pearson { fisher_z: true }),
        m => (identity, m),
    };
    let (n_windows, n_components, _, _) = c.values().dim();
    let mut values = Array4::from_elem(
        (n_windows, n_components, groups.len(), groups.len()),
        f64::NAN,
    );
    for w in 0..n_windows {
        for k in 0..n_components {
            let m = c.matrix(w, k);
            let block = |a: &[usize], b: &[usize]| {
                let pairs: Vec<f64> = a
                    .iter()
                    .flat_map(|&i| b.iter().filter(move |&&j| j != i).map(move |&j| m[[i, j]]))
                    .map(to_mean)
                    .filter(|v| !v.is_nan())
                    .collect();
                if pairs.is_empty() {
                    f64::NAN
                } else {
                    pairs.iter().sum::<f64>() / pairs.len() as f64
                }
            };
            let out: Array2<f64> = Array2::from_shape_fn((groups.len(), groups.len()), |(g, h)| {
                block(&groups[g].1, &groups[h].1)
            });
            values.slice_mut(ndarray::s![w, k, .., ..]).assign(&out);
        }
    }
    let labels = groups.iter().map(|(n, _)| n.clone()).collect();
    rebuild(c, values, measure, labels)
}

fn rebuild(
    c: &FunctionalConnectivity,
    values: Array4<f64>,
    measure: Measure,
    labels: Vec<String>,
) -> Result<FunctionalConnectivity, String> {
    let fail = |e: modewise_method::MethodError| e.to_string();
    let mut out = FunctionalConnectivity::new(values, measure, labels).map_err(fail)?;
    if let Some(hz) = c.centre_frequencies_hz() {
        out = out.with_centre_frequencies(hz.to_vec()).map_err(fail)?;
    }
    if let Some(windows) = c.windows() {
        out = out
            .with_windows(windows.to_vec(), c.resolvable().map(|r| r.to_owned()))
            .map_err(fail)?;
    }
    Ok(out)
}

/// Keeps the channels of a channel table at these positions.
pub(crate) fn keep_table(
    table: &ChannelTable,
    positions: &[usize],
) -> Result<ChannelTable, String> {
    ChannelTable::new(
        positions
            .iter()
            .map(|&i| table.channels()[i].clone())
            .collect(),
    )
    .map_err(|e| e.to_string())
}
