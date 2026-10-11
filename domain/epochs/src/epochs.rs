use modewise_decomposition::{Decomposition, InstantaneousModes};
use modewise_signals::{MultichannelSeries, SamplingRate};
use modewise_spectral::TimeFrequencyMap;
use ndarray::{Array2, ArrayView2, ArrayView3, s};
use thiserror::Error;

use crate::Events;

/// Tolerance for event times that land on a sample within floating-point error.
const ON_SAMPLE: f64 = 1e-9;

#[derive(Debug, Error, PartialEq)]
pub enum EpochError {
    #[error("{name} must be finite{extra}, got {value}")]
    InvalidParam {
        name: &'static str,
        value: f64,
        extra: &'static str,
    },
    #[error("fixed window needs tmax_s > tmin_s, got [{tmin_s}, {tmax_s})")]
    EmptyWindow { tmin_s: f64, tmax_s: f64 },
    #[error(
        "a time-frequency map can only be epoched when it has one time column per sample (CWT, Hilbert spectrum), not frames (STFT)"
    )]
    NotPerSample,
    #[error("epoch {index}: samples {start}..{end} do not fit a source of {n_samples} samples")]
    WindowOutOfRange {
        index: usize,
        start: usize,
        end: usize,
        n_samples: usize,
    },
}

/// Which part of the timeline around each event becomes an epoch.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Window {
    /// From the event onset for the event's own duration, for task blocks.
    Duration,
    /// A fixed window `[onset + tmin_s, onset + tmax_s)`, for trials (as in MNE).
    Fixed { tmin_s: f64, tmax_s: f64 },
}

#[derive(Debug, Clone, PartialEq)]
pub struct EpochParams {
    /// Shift added to every event time, for example the haemodynamic delay in fMRI. Use 0 for EEG.
    pub lag_s: f64,
    pub window: Window,
    /// Keep only events with these labels. `None` keeps all.
    pub conditions: Option<Vec<String>>,
    /// A component is resolvable in an epoch when frequency × epoch length ≥ this.
    pub min_cycles: f64,
}

impl EpochParams {
    pub fn validate(&self) -> Result<(), EpochError> {
        if !self.lag_s.is_finite() {
            return Err(EpochError::InvalidParam {
                name: "lag_s",
                value: self.lag_s,
                extra: "",
            });
        }
        if !(self.min_cycles.is_finite() && self.min_cycles > 0.0) {
            return Err(EpochError::InvalidParam {
                name: "min_cycles",
                value: self.min_cycles,
                extra: " and > 0",
            });
        }
        if let Window::Fixed { tmin_s, tmax_s } = self.window {
            for (name, value) in [("tmin_s", tmin_s), ("tmax_s", tmax_s)] {
                if !value.is_finite() {
                    return Err(EpochError::InvalidParam {
                        name,
                        value,
                        extra: "",
                    });
                }
            }
            if tmax_s <= tmin_s {
                return Err(EpochError::EmptyWindow { tmin_s, tmax_s });
            }
        }
        Ok(())
    }
}

/// One epoch: samples `start..end` of the full-run source.
#[derive(Debug, Clone, PartialEq)]
pub struct EpochWindow {
    pub label: String,
    pub event_onset_s: f64,
    pub start: usize,
    pub end: usize,
}

/// An event that produced no epoch, and why.
#[derive(Debug, Clone, PartialEq)]
pub struct DroppedEvent {
    pub label: String,
    pub event_onset_s: f64,
    pub reason: String,
}

/// The full-run result the epochs are cut from. Epochs are views into it, never recomputed.
#[derive(Debug, Clone, PartialEq)]
pub enum EpochSource {
    Series(MultichannelSeries),
    Decomposition(Decomposition),
    /// Instantaneous amplitude and frequency of modes (from a Hilbert transform).
    Instantaneous(InstantaneousModes),
    /// A time-frequency map with one time column per sample (CWT, Hilbert spectrum).
    TimeFrequency(TimeFrequencyMap),
}

impl EpochSource {
    fn n_samples(&self) -> usize {
        match self {
            Self::Series(s) => s.n_samples(),
            Self::Decomposition(d) => d.modes().dim().2,
            Self::Instantaneous(m) => m.n_samples(),
            Self::TimeFrequency(m) => m.times_s().len(),
        }
    }

    fn sampling_rate(&self) -> Result<SamplingRate, EpochError> {
        Ok(match self {
            Self::Series(s) => s.sampling_rate(),
            Self::Decomposition(d) => d.sampling_rate(),
            Self::Instantaneous(m) => m.sampling_rate(),
            Self::TimeFrequency(m) => m.sampling_rate().ok_or(EpochError::NotPerSample)?,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Epochs {
    source: EpochSource,
    windows: Vec<EpochWindow>,
    dropped: Vec<DroppedEvent>,
    params: EpochParams,
}

/// First sample whose time `i / fs` is at or after `t`.
fn first_sample_at_or_after(t: f64, fs: f64) -> f64 {
    (t * fs - ON_SAMPLE).ceil()
}

impl Epochs {
    /// Cuts `source` into one epoch per event. Samples at times in `[start, end)` belong to
    /// the epoch, so back-to-back windows never share a sample. Windows that reach outside the
    /// recording, or hold fewer than 2 samples, are dropped and reported, never padded.
    pub fn cut(
        source: EpochSource,
        events: &Events,
        params: EpochParams,
    ) -> Result<Self, EpochError> {
        params.validate()?;
        let fs = source.sampling_rate()?.hz();
        let n = source.n_samples();
        let mut windows = Vec::new();
        let mut dropped = Vec::new();
        for event in events.as_slice() {
            if let Some(conditions) = &params.conditions
                && !conditions.contains(&event.label)
            {
                continue;
            }
            let anchor = event.onset_s + params.lag_s;
            let (t_start, t_end) = match params.window {
                Window::Duration => (anchor, anchor + event.duration_s),
                Window::Fixed { tmin_s, tmax_s } => (anchor + tmin_s, anchor + tmax_s),
            };
            let (start, end) = (
                first_sample_at_or_after(t_start, fs),
                first_sample_at_or_after(t_end, fs),
            );
            let drop = |reason: String| DroppedEvent {
                label: event.label.clone(),
                event_onset_s: event.onset_s,
                reason,
            };
            if start < 0.0 {
                dropped.push(drop(format!(
                    "window starts at {t_start:.3} s, before the recording"
                )));
            } else if end > n as f64 {
                dropped.push(drop(format!(
                    "window ends at {t_end:.3} s, after the recording ({:.3} s)",
                    n as f64 / fs
                )));
            } else if end - start < 2.0 {
                dropped.push(drop(format!(
                    "window holds {} sample(s); at least 2 are needed",
                    (end - start).max(0.0)
                )));
            } else {
                windows.push(EpochWindow {
                    label: event.label.clone(),
                    event_onset_s: event.onset_s,
                    start: start as usize,
                    end: end as usize,
                });
            }
        }
        Ok(Self {
            source,
            windows,
            dropped,
            params,
        })
    }

    /// Rebuilds epochs from stored parts, checking every window against the source.
    pub fn from_parts(
        source: EpochSource,
        windows: Vec<EpochWindow>,
        dropped: Vec<DroppedEvent>,
        params: EpochParams,
    ) -> Result<Self, EpochError> {
        params.validate()?;
        source.sampling_rate()?;
        let n_samples = source.n_samples();
        for (index, w) in windows.iter().enumerate() {
            if w.end > n_samples || w.end < w.start + 2 {
                return Err(EpochError::WindowOutOfRange {
                    index,
                    start: w.start,
                    end: w.end,
                    n_samples,
                });
            }
        }
        Ok(Self {
            source,
            windows,
            dropped,
            params,
        })
    }

    pub fn source(&self) -> &EpochSource {
        &self.source
    }

    pub fn windows(&self) -> &[EpochWindow] {
        &self.windows
    }

    pub fn dropped(&self) -> &[DroppedEvent] {
        &self.dropped
    }

    pub fn params(&self) -> &EpochParams {
        &self.params
    }

    /// Length of epoch `i` in seconds, from its sample count.
    pub fn duration_s(&self, i: usize) -> f64 {
        let w = &self.windows[i];
        let fs = self.source.sampling_rate().map_or(1.0, |fs| fs.hz());
        (w.end - w.start) as f64 / fs
    }

    /// For a decomposition source: epochs × modes, true where the mode's centre frequency
    /// fits at least `min_cycles` cycles into the epoch. `None` for sources without frequencies.
    pub fn resolvable(&self) -> Option<Array2<bool>> {
        let freqs = match &self.source {
            EpochSource::Series(_) => return None,
            EpochSource::Decomposition(d) => d.centre_frequencies_hz(),
            EpochSource::Instantaneous(m) => m.centre_frequencies_hz(),
            EpochSource::TimeFrequency(m) => m.frequencies_hz(),
        };
        let reliable = match &self.source {
            EpochSource::TimeFrequency(m) => m.reliable(),
            _ => None,
        };
        Some(Array2::from_shape_fn(
            (self.windows.len(), freqs.len()),
            |(i, k)| {
                let w = &self.windows[i];
                let fits = freqs[k] * self.duration_s(i) >= self.params.min_cycles - ON_SAMPLE;
                let clear_of_edges =
                    reliable.is_none_or(|r| r.slice(s![k, w.start..w.end]).iter().all(|&ok| ok));
                fits && clear_of_edges
            },
        ))
    }

    /// Epoch `i` of a time-frequency source: channel × frequency × sample, a view into the full run.
    pub fn time_frequency(&self, i: usize) -> Option<ArrayView3<'_, f64>> {
        let EpochSource::TimeFrequency(m) = &self.source else {
            return None;
        };
        let w = &self.windows[i];
        Some(m.power().slice_move(s![.., .., w.start..w.end]))
    }

    /// Epoch `i` of a series source: channels × samples, a view into the full run.
    pub fn series(&self, i: usize) -> Option<ArrayView2<'_, f64>> {
        let EpochSource::Series(s) = &self.source else {
            return None;
        };
        let w = &self.windows[i];
        Some(s.data().slice_move(s![.., w.start..w.end]))
    }

    /// Epoch `i` of an instantaneous source: (amplitude, frequency in Hz), each modes × channels × samples.
    pub fn instantaneous(&self, i: usize) -> Option<(ArrayView3<'_, f64>, ArrayView3<'_, f64>)> {
        let EpochSource::Instantaneous(m) = &self.source else {
            return None;
        };
        let w = &self.windows[i];
        Some((
            m.amplitude().slice_move(s![.., .., w.start..w.end]),
            m.frequency_hz().slice_move(s![.., .., w.start..w.end]),
        ))
    }

    /// Epoch `i` of a decomposition source: modes × channels × samples, a view into the full run.
    pub fn modes(&self, i: usize) -> Option<ArrayView3<'_, f64>> {
        let EpochSource::Decomposition(d) = &self.source else {
            return None;
        };
        let w = &self.windows[i];
        Some(d.modes().slice_move(s![.., .., w.start..w.end]))
    }
}
