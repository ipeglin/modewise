use std::collections::HashSet;

use ndarray::{Array2, ArrayView2};

use crate::{SamplingRate, SignalError};

/// Equally sampled signals, laid out as channels × samples.
#[derive(Debug, Clone, PartialEq)]
pub struct MultichannelSeries {
    data: Array2<f64>,
    sampling_rate: SamplingRate,
    labels: Vec<String>,
}

impl MultichannelSeries {
    pub fn new(
        data: Array2<f64>,
        sampling_rate: SamplingRate,
        labels: Vec<String>,
    ) -> Result<Self, SignalError> {
        let (n_channels, n_samples) = data.dim();
        if n_channels == 0 {
            return Err(SignalError::NoChannels);
        }
        if n_samples < 2 {
            return Err(SignalError::TooFewSamples(n_samples));
        }
        if labels.len() != n_channels {
            return Err(SignalError::LabelCountMismatch {
                expected: n_channels,
                actual: labels.len(),
            });
        }
        let mut seen = HashSet::new();
        if let Some(duplicate) = labels.iter().find(|label| !seen.insert(*label)) {
            return Err(SignalError::DuplicateLabel(duplicate.clone()));
        }
        if let Some(((channel, sample), _)) = data.indexed_iter().find(|(_, v)| !v.is_finite()) {
            return Err(SignalError::NonFiniteValue { channel, sample });
        }
        Ok(Self {
            data,
            sampling_rate,
            labels,
        })
    }

    pub fn data(&self) -> ArrayView2<'_, f64> {
        self.data.view()
    }

    pub fn sampling_rate(&self) -> SamplingRate {
        self.sampling_rate
    }

    pub fn labels(&self) -> &[String] {
        &self.labels
    }

    pub fn n_channels(&self) -> usize {
        self.data.nrows()
    }

    pub fn n_samples(&self) -> usize {
        self.data.ncols()
    }

    pub fn duration_s(&self) -> f64 {
        self.n_samples() as f64 * self.sampling_rate.period_s()
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    fn fs() -> SamplingRate {
        SamplingRate::new(2.0).unwrap()
    }

    fn labels(names: &[&str]) -> Vec<String> {
        names.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn valid_series_exposes_shape_and_duration() {
        let data = array![[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]];
        let series = MultichannelSeries::new(data, fs(), labels(&["a", "b"])).unwrap();
        assert_eq!(series.n_channels(), 2);
        assert_eq!(series.n_samples(), 4);
        assert_eq!(series.duration_s(), 2.0);
        assert_eq!(series.labels(), ["a", "b"]);
        assert_eq!(series.data()[[1, 2]], 7.0);
    }

    #[test]
    fn rejects_no_channels() {
        let data = Array2::<f64>::zeros((0, 4));
        let err = MultichannelSeries::new(data, fs(), vec![]).unwrap_err();
        assert_eq!(err, SignalError::NoChannels);
    }

    #[test]
    fn rejects_too_few_samples() {
        let err = MultichannelSeries::new(array![[1.0]], fs(), labels(&["a"])).unwrap_err();
        assert_eq!(err, SignalError::TooFewSamples(1));
    }

    #[test]
    fn rejects_label_count_mismatch() {
        let err =
            MultichannelSeries::new(array![[1.0, 2.0]], fs(), labels(&["a", "b"])).unwrap_err();
        assert_eq!(
            err,
            SignalError::LabelCountMismatch {
                expected: 1,
                actual: 2
            }
        );
    }

    #[test]
    fn rejects_duplicate_labels() {
        let data = array![[1.0, 2.0], [3.0, 4.0]];
        let err = MultichannelSeries::new(data, fs(), labels(&["a", "a"])).unwrap_err();
        assert_eq!(err, SignalError::DuplicateLabel("a".into()));
    }

    #[test]
    fn rejects_non_finite_values() {
        let data = array![[1.0, 2.0], [3.0, f64::NAN]];
        let err = MultichannelSeries::new(data, fs(), labels(&["a", "b"])).unwrap_err();
        assert_eq!(
            err,
            SignalError::NonFiniteValue {
                channel: 1,
                sample: 1
            }
        );
    }
}
