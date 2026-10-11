use crate::SignalError;

/// Samples per second, in Hz. Always finite and positive.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplingRate(f64);

impl SamplingRate {
    pub fn new(hz: f64) -> Result<Self, SignalError> {
        if hz.is_finite() && hz > 0.0 {
            Ok(Self(hz))
        } else {
            Err(SignalError::InvalidSamplingRate(hz))
        }
    }

    pub fn hz(self) -> f64 {
        self.0
    }

    pub fn period_s(self) -> f64 {
        1.0 / self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_positive_rate() {
        let fs = SamplingRate::new(1.25).unwrap();
        assert_eq!(fs.hz(), 1.25);
        assert_eq!(fs.period_s(), 0.8);
    }

    #[test]
    fn rejects_zero_negative_and_non_finite() {
        for hz in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(SamplingRate::new(hz).is_err(), "accepted {hz}");
        }
    }
}
