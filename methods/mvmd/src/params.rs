use modewise_method::MethodError;

/// Starting centre frequencies, in normalised frequency (cycles per sample).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FrequencyInit {
    #[default]
    Zero,
    /// Evenly spaced over [0, 0.5].
    Linear,
    /// Log-spaced over [0.0005, 0.5].
    Exponential,
    /// (0.5 / K)·k, as the original MATLAB VMD and MVMD codes and vmdpy (`init = 1`).
    Uniform,
}

impl FrequencyInit {
    pub(crate) fn omegas(self, num_modes: usize) -> Vec<f64> {
        let step = |i: usize| i as f64 / (num_modes - 1).max(1) as f64;
        (0..num_modes)
            .map(|i| match self {
                Self::Zero => 0.0,
                Self::Linear => 0.5 * step(i),
                Self::Exponential => 0.5 * 10f64.powf(-3.0 + 3.0 * step(i)),
                Self::Uniform => 0.5 * i as f64 / num_modes as f64,
            })
            .collect()
    }
}

/// Defaults follow the reference implementation.
#[derive(Debug, Clone, PartialEq)]
pub struct MvmdParams {
    pub num_modes: usize,
    /// Bandwidth penalty. Larger values give narrower modes.
    pub alpha: f64,
    pub init: FrequencyInit,
    /// Stop when the mean squared mode update falls below this.
    pub tolerance: f64,
    /// Dual ascent step. 0 allows noise to stay out of the modes.
    pub tau: f64,
    pub max_iterations: usize,
}

impl MvmdParams {
    pub fn new(num_modes: usize, alpha: f64) -> Self {
        Self {
            num_modes,
            alpha,
            init: FrequencyInit::Zero,
            tolerance: 1e-3,
            tau: 1e-2,
            max_iterations: 500,
        }
    }

    pub(crate) fn validate(&self) -> Result<(), MethodError> {
        if self.num_modes == 0 {
            return Err(MethodError::invalid("num_modes", "must be at least 1"));
        }
        if !(self.alpha.is_finite() && self.alpha > 0.0) {
            return Err(MethodError::invalid("alpha", "must be finite and positive"));
        }
        if !(self.tolerance.is_finite() && self.tolerance > 0.0) {
            return Err(MethodError::invalid(
                "tolerance",
                "must be finite and positive",
            ));
        }
        if !(self.tau.is_finite() && self.tau >= 0.0) {
            return Err(MethodError::invalid(
                "tau",
                "must be finite and non-negative",
            ));
        }
        if self.max_iterations == 0 {
            return Err(MethodError::invalid("max_iterations", "must be at least 1"));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn init_matches_numpy_spacing() {
        assert_eq!(FrequencyInit::Linear.omegas(3), [0.0, 0.25, 0.5]);
        let exp = FrequencyInit::Exponential.omegas(4);
        let expected = [0.0005, 0.005, 0.05, 0.5];
        for (a, b) in exp.iter().zip(expected) {
            assert!((a - b).abs() < 1e-15, "{a} != {b}");
        }
        assert_eq!(FrequencyInit::Exponential.omegas(1), [0.0005]);
    }

    #[test]
    fn rejects_invalid_values() {
        let base = MvmdParams::new(3, 1000.0);
        let cases = [
            MvmdParams {
                num_modes: 0,
                ..base.clone()
            },
            MvmdParams {
                alpha: 0.0,
                ..base.clone()
            },
            MvmdParams {
                alpha: f64::NAN,
                ..base.clone()
            },
            MvmdParams {
                tolerance: 0.0,
                ..base.clone()
            },
            MvmdParams {
                tau: -1.0,
                ..base.clone()
            },
            MvmdParams {
                max_iterations: 0,
                ..base.clone()
            },
        ];
        for params in cases {
            assert!(params.validate().is_err(), "accepted {params:?}");
        }
        assert!(base.validate().is_ok());
    }
}
