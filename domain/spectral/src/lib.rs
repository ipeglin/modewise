use modewise_method::{MethodDescriptor, MethodError};
use modewise_signals::{MultichannelSeries, SamplingRate};
use ndarray::{Array2, Array3, ArrayView2, ArrayView3};

/// Frequency bins `[edges[i], edges[i + 1])`, spaced linearly or logarithmically.
#[derive(Debug, Clone, PartialEq)]
pub struct FrequencyGrid {
    edges: Vec<f64>,
    scale: GridScale,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GridScale {
    Linear,
    Log,
}

impl FrequencyGrid {
    pub fn new(
        f_min: f64,
        f_max: f64,
        n_bins: usize,
        scale: GridScale,
    ) -> Result<Self, MethodError> {
        if n_bins == 0 {
            return Err(MethodError::invalid("n_bins", "must be at least 1"));
        }
        let lowest = if scale == GridScale::Log {
            0.0
        } else {
            f64::NEG_INFINITY
        };
        if !(f_min.is_finite() && f_max.is_finite() && f_min > lowest && f_max > f_min) {
            return Err(MethodError::invalid(
                "frequency range",
                format!(
                    "need finite f_min < f_max{}, got [{f_min}, {f_max}]",
                    if scale == GridScale::Log {
                        " and f_min > 0"
                    } else {
                        ""
                    }
                ),
            ));
        }
        let edges = (0..=n_bins)
            .map(|i| {
                let t = i as f64 / n_bins as f64;
                match scale {
                    GridScale::Linear => f_min + t * (f_max - f_min),
                    GridScale::Log => f_min * (f_max / f_min).powf(t),
                }
            })
            .collect();
        Ok(Self { edges, scale })
    }

    pub fn edges(&self) -> &[f64] {
        &self.edges
    }

    pub fn n_bins(&self) -> usize {
        self.edges.len() - 1
    }

    /// Bin centres: arithmetic for linear grids, geometric for log grids.
    pub fn centres(&self) -> Vec<f64> {
        self.edges
            .windows(2)
            .map(|w| match self.scale {
                GridScale::Linear => (w[0] + w[1]) / 2.0,
                GridScale::Log => (w[0] * w[1]).sqrt(),
            })
            .collect()
    }

    pub fn scale(&self) -> GridScale {
        self.scale
    }

    /// The bin holding `f`, or `None` when `f` is outside `[first edge, last edge)` or not finite.
    pub fn bin(&self, f: f64) -> Option<usize> {
        let last = *self.edges.last()?;
        if !(f >= self.edges[0] && f < last) {
            return None;
        }
        Some(self.edges.partition_point(|&e| e <= f) - 1)
    }
}

/// Maps a signal directly to time-frequency power.
pub trait TimeFrequencyTransform {
    fn descriptor(&self) -> MethodDescriptor;
    fn transform(&self, signal: &MultichannelSeries) -> Result<TimeFrequencyMap, MethodError>;
}

/// Power laid out as channel × frequency × time.
#[derive(Debug, Clone, PartialEq)]
pub struct TimeFrequencyMap {
    power: Array3<f64>,
    frequencies_hz: Vec<f64>,
    times_s: Vec<f64>,
    labels: Vec<String>,
    sampling_rate: Option<SamplingRate>,
    reliable: Option<Array2<bool>>,
}

impl TimeFrequencyMap {
    pub fn new(
        power: Array3<f64>,
        frequencies_hz: Vec<f64>,
        times_s: Vec<f64>,
        labels: Vec<String>,
    ) -> Result<Self, MethodError> {
        let expected = (labels.len(), frequencies_hz.len(), times_s.len());
        if power.dim() != expected {
            return Err(MethodError::ShapeMismatch(format!(
                "power has shape {:?}, axes imply {expected:?}",
                power.dim()
            )));
        }
        Ok(Self {
            power,
            frequencies_hz,
            times_s,
            labels,
            sampling_rate: None,
            reliable: None,
        })
    }

    pub fn power(&self) -> ArrayView3<'_, f64> {
        self.power.view()
    }

    pub fn frequencies_hz(&self) -> &[f64] {
        &self.frequencies_hz
    }

    pub fn times_s(&self) -> &[f64] {
        &self.times_s
    }

    pub fn labels(&self) -> &[String] {
        &self.labels
    }

    /// Set when there is one time column per sample at `i / fs` (CWT, Hilbert spectrum), which
    /// makes the map epochable. STFT frames are not per-sample and leave this unset.
    pub fn with_sampling_rate(mut self, sampling_rate: SamplingRate) -> Result<Self, MethodError> {
        let fs = sampling_rate.hz();
        let aligned = self
            .times_s
            .iter()
            .enumerate()
            .all(|(i, t)| (t - i as f64 / fs).abs() <= 1e-9 * t.abs().max(1.0));
        if !aligned {
            return Err(MethodError::ShapeMismatch(
                "times are not one per sample at i / fs".into(),
            ));
        }
        self.sampling_rate = Some(sampling_rate);
        Ok(self)
    }

    /// Frequency × time mask: false where edge effects make the value unreliable (for example
    /// inside a wavelet's cone of influence).
    pub fn with_reliable(mut self, reliable: Array2<bool>) -> Result<Self, MethodError> {
        let expected = (self.frequencies_hz.len(), self.times_s.len());
        if reliable.dim() != expected {
            return Err(MethodError::ShapeMismatch(format!(
                "reliable mask {:?}, expected {expected:?}",
                reliable.dim()
            )));
        }
        self.reliable = Some(reliable);
        Ok(self)
    }

    pub fn sampling_rate(&self) -> Option<SamplingRate> {
        self.sampling_rate
    }

    pub fn reliable(&self) -> Option<ArrayView2<'_, bool>> {
        self.reliable.as_ref().map(|r| r.view())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_bins_are_half_open() {
        let g = FrequencyGrid::new(0.0, 1.0, 4, GridScale::Linear).unwrap();
        assert_eq!(g.edges(), [0.0, 0.25, 0.5, 0.75, 1.0]);
        assert_eq!(g.centres(), [0.125, 0.375, 0.625, 0.875]);
        assert_eq!(
            (
                g.bin(0.0),
                g.bin(0.25),
                g.bin(0.999),
                g.bin(1.0),
                g.bin(-0.1),
                g.bin(f64::NAN)
            ),
            (Some(0), Some(1), Some(3), None, None, None)
        );
        assert_eq!(g.n_bins(), 4);
        assert_eq!(g.scale(), GridScale::Linear);
        let uneven = FrequencyGrid::new(0.1, 0.7, 3, GridScale::Linear).unwrap();
        assert!(
            uneven
                .centres()
                .iter()
                .zip(uneven.edges().windows(2))
                .all(|(c, w)| (c - (w[0] + w[1]) / 2.0).abs() < 1e-15)
        );

        let log = FrequencyGrid::new(0.01, 1.0, 2, GridScale::Log).unwrap();
        assert!((log.edges()[1] - 0.1).abs() < 1e-12);
        assert!((log.centres()[0] - (0.01f64 * 0.1).sqrt()).abs() < 1e-12);

        assert!(FrequencyGrid::new(0.0, 1.0, 0, GridScale::Linear).is_err());
        assert!(FrequencyGrid::new(1.0, 1.0, 3, GridScale::Linear).is_err());
        assert!(
            FrequencyGrid::new(0.0, 1.0, 3, GridScale::Log).is_err(),
            "log grids need f_min > 0"
        );
        assert!(FrequencyGrid::new(f64::NAN, 1.0, 3, GridScale::Linear).is_err());
    }

    #[test]
    fn rejects_axis_mismatch() {
        let power = Array3::zeros((1, 3, 2));
        let err = TimeFrequencyMap::new(power, vec![0.0, 1.0], vec![0.0, 1.0], vec!["a".into()]);
        assert!(matches!(err, Err(MethodError::ShapeMismatch(_))));
    }

    #[test]
    fn optional_sampling_rate_and_reliability_mask_are_checked() {
        let map = TimeFrequencyMap::new(
            Array3::zeros((1, 2, 3)),
            vec![1.0, 2.0],
            vec![0.0, 0.5, 1.0],
            vec!["a".into()],
        )
        .unwrap();
        assert!(map.sampling_rate().is_none() && map.reliable().is_none());
        let fs = SamplingRate::new(2.0).unwrap();
        let with = map
            .clone()
            .with_sampling_rate(fs)
            .unwrap()
            .with_reliable(Array2::from_elem((2, 3), true))
            .unwrap();
        assert_eq!(with.sampling_rate(), Some(fs));
        assert_eq!(with.reliable().unwrap().dim(), (2, 3));
        assert!(
            map.clone()
                .with_sampling_rate(SamplingRate::new(4.0).unwrap())
                .is_err(),
            "frames are not at i / 4 Hz"
        );
        assert!(map.with_reliable(Array2::from_elem((3, 2), true)).is_err());
    }

    #[test]
    fn accumulated_rounding_in_long_time_axes_is_accepted() {
        // Times built by repeated addition drift by ~1e-7 s after 1000 s; that is still per-sample.
        let n = 1250;
        let mut t = 0.0;
        let times: Vec<f64> = (0..n)
            .map(|_| {
                let now = t;
                t += 0.8;
                now
            })
            .collect();
        let mut times = times;
        times[n - 1] += 1e-7;
        let map =
            TimeFrequencyMap::new(Array3::zeros((1, 1, n)), vec![0.1], times, vec!["a".into()])
                .unwrap();
        assert!(
            map.with_sampling_rate(SamplingRate::new(1.25).unwrap())
                .is_ok()
        );
    }

    #[test]
    fn accepts_matching_axes() {
        let power = Array3::zeros((1, 2, 3));
        let map =
            TimeFrequencyMap::new(power, vec![0.0, 1.0], vec![0.0, 1.0, 2.0], vec!["a".into()])
                .unwrap();
        assert_eq!(map.frequencies_hz(), [0.0, 1.0]);
        assert_eq!(map.times_s(), [0.0, 1.0, 2.0]);
        assert_eq!(map.labels(), ["a"]);
        assert_eq!(map.power().dim(), (1, 2, 3));
    }
}
