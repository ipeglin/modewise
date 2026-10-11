use modewise_method::{MethodDescriptor, MethodError};
use modewise_signals::{MultichannelSeries, SamplingRate};
use ndarray::{Array2, Array3, ArrayView2, ArrayView3, Axis};

mod loadings;

pub use loadings::Loadings;

/// Instantaneous amplitude and frequency of every mode, laid out as mode × channel × sample,
/// with the centre frequencies of the modes they came from.
#[derive(Debug, Clone, PartialEq)]
pub struct InstantaneousModes {
    amplitude: Array3<f64>,
    frequency_hz: Array3<f64>,
    centre_frequencies_hz: Vec<f64>,
    sampling_rate: SamplingRate,
    labels: Vec<String>,
}

impl InstantaneousModes {
    pub fn new(
        amplitude: Array3<f64>,
        frequency_hz: Array3<f64>,
        centre_frequencies_hz: Vec<f64>,
        sampling_rate: SamplingRate,
        labels: Vec<String>,
    ) -> Result<Self, MethodError> {
        let (n_modes, n_channels, _) = amplitude.dim();
        if frequency_hz.dim() != amplitude.dim() {
            return Err(MethodError::ShapeMismatch(format!(
                "amplitude {:?} and frequency {:?} differ",
                amplitude.dim(),
                frequency_hz.dim()
            )));
        }
        if centre_frequencies_hz.len() != n_modes || labels.len() != n_channels {
            return Err(MethodError::ShapeMismatch(format!(
                "{n_modes} modes and {n_channels} channels, but {} centre frequencies and {} labels",
                centre_frequencies_hz.len(),
                labels.len()
            )));
        }
        Ok(Self {
            amplitude,
            frequency_hz,
            centre_frequencies_hz,
            sampling_rate,
            labels,
        })
    }

    pub fn amplitude(&self) -> ArrayView3<'_, f64> {
        self.amplitude.view()
    }

    pub fn frequency_hz(&self) -> ArrayView3<'_, f64> {
        self.frequency_hz.view()
    }

    pub fn centre_frequencies_hz(&self) -> &[f64] {
        &self.centre_frequencies_hz
    }

    pub fn sampling_rate(&self) -> SamplingRate {
        self.sampling_rate
    }

    pub fn labels(&self) -> &[String] {
        &self.labels
    }

    pub fn n_samples(&self) -> usize {
        self.amplitude.dim().2
    }
}

fn median(mut values: Vec<f64>) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.sort_by(f64::total_cmp);
    let mid = values.len() / 2;
    if values.len() % 2 == 1 {
        values[mid]
    } else {
        (values[mid - 1] + values[mid]) / 2.0
    }
}

/// Splits a signal into time-domain modes.
pub trait ModeDecomposer {
    fn descriptor(&self) -> MethodDescriptor;
    fn decompose(&self, signal: &MultichannelSeries) -> Result<Decomposition, MethodError>;
}

/// Modes laid out as mode × channel × sample, ordered by ascending centre frequency.
#[derive(Debug, Clone, PartialEq)]
pub struct Decomposition {
    modes: Array3<f64>,
    centre_frequencies_hz: Vec<f64>,
    sampling_rate: SamplingRate,
    labels: Vec<String>,
    aligned_across_channels: bool,
    /// Mode × channel, for methods that decompose each channel on its own.
    channel_centre_frequencies_hz: Option<Array2<f64>>,
}

impl Decomposition {
    pub fn new(
        modes: Array3<f64>,
        centre_frequencies_hz: Vec<f64>,
        sampling_rate: SamplingRate,
        labels: Vec<String>,
        aligned_across_channels: bool,
    ) -> Result<Self, MethodError> {
        let (n_modes, n_channels, _) = modes.dim();
        if centre_frequencies_hz.len() != n_modes {
            return Err(MethodError::ShapeMismatch(format!(
                "{n_modes} modes but {} centre frequencies",
                centre_frequencies_hz.len()
            )));
        }
        if labels.len() != n_channels {
            return Err(MethodError::ShapeMismatch(format!(
                "{n_channels} channels but {} labels",
                labels.len()
            )));
        }
        Ok(Self {
            modes,
            centre_frequencies_hz,
            sampling_rate,
            labels,
            aligned_across_channels,
            channel_centre_frequencies_hz: None,
        })
    }

    /// Modes from a method that decomposes each channel on its own (VMD, EMD). Mode k need not
    /// mean the same oscillation in every channel, so the decomposition is not aligned. Each
    /// mode's centre frequency is the median of its channels' finite values; NaN marks a channel
    /// without that mode.
    pub fn per_channel(
        modes: Array3<f64>,
        channel_centre_frequencies_hz: Array2<f64>,
        sampling_rate: SamplingRate,
        labels: Vec<String>,
    ) -> Result<Self, MethodError> {
        let centres = channel_centre_frequencies_hz
            .rows()
            .into_iter()
            .map(|row| median(row.iter().copied().filter(|v| v.is_finite()).collect()))
            .collect();
        Self::new(modes, centres, sampling_rate, labels, false)?
            .with_channel_centre_frequencies(channel_centre_frequencies_hz)
    }

    /// Attaches per-channel centre frequencies (mode × channel).
    pub fn with_channel_centre_frequencies(
        mut self,
        frequencies_hz: Array2<f64>,
    ) -> Result<Self, MethodError> {
        let (n_modes, n_channels, _) = self.modes.dim();
        if frequencies_hz.dim() != (n_modes, n_channels) {
            return Err(MethodError::ShapeMismatch(format!(
                "{n_modes} modes and {n_channels} channels, but channel centre frequencies are {:?}",
                frequencies_hz.dim()
            )));
        }
        self.channel_centre_frequencies_hz = Some(frequencies_hz);
        Ok(self)
    }

    /// Mode × channel centre frequencies, when the method gives them per channel.
    pub fn channel_centre_frequencies_hz(&self) -> Option<ArrayView2<'_, f64>> {
        self.channel_centre_frequencies_hz
            .as_ref()
            .map(|f| f.view())
    }

    pub fn modes(&self) -> ArrayView3<'_, f64> {
        self.modes.view()
    }

    pub fn mode(&self, index: usize) -> ArrayView2<'_, f64> {
        self.modes.index_axis(Axis(0), index)
    }

    pub fn centre_frequencies_hz(&self) -> &[f64] {
        &self.centre_frequencies_hz
    }

    pub fn sampling_rate(&self) -> SamplingRate {
        self.sampling_rate
    }

    pub fn labels(&self) -> &[String] {
        &self.labels
    }

    /// True when mode `k` means the same oscillation in every channel.
    pub fn aligned_across_channels(&self) -> bool {
        self.aligned_across_channels
    }

    pub fn n_modes(&self) -> usize {
        self.modes.len_of(Axis(0))
    }

    /// Sums the chosen modes back into a signal, for example to keep one frequency band.
    pub fn reconstruct(&self, modes: &[usize]) -> Result<MultichannelSeries, MethodError> {
        if modes.is_empty() {
            return Err(MethodError::invalid("modes", "choose at least one mode"));
        }
        let (_, n_channels, n_samples) = self.modes.dim();
        let mut sum = Array2::zeros((n_channels, n_samples));
        for (i, &k) in modes.iter().enumerate() {
            if k >= self.n_modes() {
                return Err(MethodError::invalid(
                    "modes",
                    format!(
                        "mode {k} does not exist; there are {} modes",
                        self.n_modes()
                    ),
                ));
            }
            if modes[..i].contains(&k) {
                return Err(MethodError::invalid(
                    "modes",
                    format!("mode {k} is listed twice"),
                ));
            }
            sum += &self.mode(k);
        }
        MultichannelSeries::new(sum, self.sampling_rate, self.labels.clone())
            .map_err(|e| MethodError::ShapeMismatch(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fs() -> SamplingRate {
        SamplingRate::new(1.0).unwrap()
    }

    #[test]
    fn rejects_frequency_count_mismatch() {
        let modes = Array3::zeros((2, 1, 4));
        let err = Decomposition::new(modes, vec![0.1], fs(), vec!["a".into()], true);
        assert!(matches!(err, Err(MethodError::ShapeMismatch(_))));
    }

    #[test]
    fn rejects_label_count_mismatch() {
        let modes = Array3::zeros((1, 2, 4));
        let err = Decomposition::new(modes, vec![0.1], fs(), vec!["a".into()], true);
        assert!(matches!(err, Err(MethodError::ShapeMismatch(_))));
    }

    #[test]
    fn reconstruct_sums_chosen_modes() {
        let mut modes = Array3::zeros((3, 1, 2));
        modes[[0, 0, 0]] = 1.0;
        modes[[1, 0, 0]] = 2.0;
        modes[[2, 0, 1]] = 4.0;
        let d =
            Decomposition::new(modes, vec![0.1, 0.2, 0.3], fs(), vec!["a".into()], true).unwrap();
        let series = d.reconstruct(&[0, 2]).unwrap();
        assert_eq!(series.data().row(0).to_vec(), [1.0, 4.0]);
        assert_eq!(series.labels(), ["a"]);
    }

    #[test]
    fn reconstruct_rejects_bad_selection() {
        let d = Decomposition::new(
            Array3::zeros((2, 1, 2)),
            vec![0.1, 0.2],
            fs(),
            vec!["a".into()],
            true,
        )
        .unwrap();
        for selection in [&[][..], &[2], &[1, 1]] {
            assert!(d.reconstruct(selection).is_err(), "accepted {selection:?}");
        }
    }

    #[test]
    fn getters_return_what_was_stored() {
        let d = Decomposition::new(
            Array3::zeros((2, 1, 3)),
            vec![0.1, 0.2],
            fs(),
            vec!["amy".into()],
            false,
        )
        .unwrap();
        assert_eq!(d.centre_frequencies_hz(), [0.1, 0.2]);
        assert_eq!(d.labels(), ["amy"]);
        assert!(!d.aligned_across_channels());
        assert_eq!(d.sampling_rate(), fs());
        let aligned = Decomposition::new(
            Array3::zeros((1, 1, 3)),
            vec![0.1],
            fs(),
            vec!["a".into()],
            true,
        )
        .unwrap();
        assert!(aligned.aligned_across_channels());
    }

    #[test]
    fn instantaneous_modes_check_shapes() {
        let ok = InstantaneousModes::new(
            Array3::zeros((2, 1, 4)),
            Array3::zeros((2, 1, 4)),
            vec![0.1, 0.2],
            fs(),
            vec!["a".into()],
        )
        .unwrap();
        assert_eq!(
            (ok.n_samples(), ok.centre_frequencies_hz(), ok.labels()),
            (4, [0.1, 0.2].as_slice(), ["a".to_string()].as_slice())
        );
        assert_eq!(ok.sampling_rate(), fs());
        assert_eq!(ok.frequency_hz().dim(), ok.amplitude().dim());
        assert!(
            InstantaneousModes::new(
                Array3::zeros((2, 1, 4)),
                Array3::zeros((2, 1, 5)),
                vec![0.1, 0.2],
                fs(),
                vec!["a".into()]
            )
            .is_err()
        );
        assert!(
            InstantaneousModes::new(
                Array3::zeros((2, 1, 4)),
                Array3::zeros((2, 1, 4)),
                vec![0.1],
                fs(),
                vec!["a".into()]
            )
            .is_err()
        );
        assert!(
            InstantaneousModes::new(
                Array3::zeros((2, 1, 4)),
                Array3::zeros((2, 1, 4)),
                vec![0.1, 0.2],
                fs(),
                vec![]
            )
            .is_err()
        );
    }

    #[test]
    fn exposes_single_mode() {
        let mut modes = Array3::zeros((2, 1, 3));
        modes[[1, 0, 2]] = 5.0;
        let d = Decomposition::new(modes, vec![0.1, 0.2], fs(), vec!["a".into()], true).unwrap();
        assert_eq!(d.n_modes(), 2);
        assert_eq!(d.mode(1)[[0, 2]], 5.0);
    }
}

#[cfg(test)]
mod per_channel {
    use super::*;

    #[test]
    fn mode_frequencies_are_channel_medians_ignoring_missing_modes() {
        let fs = SamplingRate::new(1.0).unwrap();
        let channel = ndarray::array![
            [0.1, 0.3, 0.2],
            [0.05, f64::NAN, 0.25],
            [f64::NAN, f64::NAN, f64::NAN]
        ];
        let labels = vec!["a".into(), "b".into(), "c".into()];
        let d = Decomposition::per_channel(
            Array3::zeros((3, 3, 4)),
            channel.clone(),
            fs,
            labels.clone(),
        )
        .unwrap();
        assert_eq!(d.centre_frequencies_hz()[..2], [0.2, 0.15]);
        assert!(d.centre_frequencies_hz()[2].is_nan());
        assert!(!d.aligned_across_channels());
        let stored = d.channel_centre_frequencies_hz().unwrap();
        assert!(
            stored
                .iter()
                .zip(&channel)
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
        assert!(
            Decomposition::per_channel(Array3::zeros((3, 2, 4)), channel, fs, labels[..2].to_vec())
                .is_err()
        );
        let plain = Decomposition::new(
            Array3::zeros((1, 1, 4)),
            vec![0.1],
            fs,
            vec!["a".into()],
            true,
        )
        .unwrap();
        assert!(plain.channel_centre_frequencies_hz().is_none());
    }
}
