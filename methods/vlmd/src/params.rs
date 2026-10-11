use modewise_method::MethodError;

/// Defaults follow the reference implementation.
#[derive(Debug, Clone, PartialEq)]
pub struct VlmdParams {
    /// Number of modes K.
    pub num_modes: usize,
    /// Number of latent components L (at most the number of channels). `None` uses one per channel.
    pub num_latents: Option<usize>,
    /// Bandwidth penalty α: larger values give narrower modes.
    pub alpha: f64,
    /// Sparsity λ of the loadings, on the reference scale (see `lasso`).
    pub reg_lambda: f64,
    /// ADMM penalty ρ: how closely the latent components must equal the sum of their modes.
    pub reg_rho: f64,
    /// Dual step τ in [0, 1].
    pub tau: f64,
    /// Stop when Σ((ωₙ₊₁ − ωₙ)/ωₙ)² falls to this.
    pub tolerance: f64,
    pub max_iterations: usize,
}

impl VlmdParams {
    pub fn new(num_modes: usize) -> Self {
        Self {
            num_modes,
            num_latents: None,
            alpha: 100.0,
            reg_lambda: 0.01,
            reg_rho: 1.0,
            tau: 0.9,
            tolerance: 1e-3,
            max_iterations: 1000,
        }
    }

    pub(crate) fn validate(&self) -> Result<(), MethodError> {
        let positive = |v: f64| v.is_finite() && v > 0.0;
        if self.num_modes == 0 {
            return Err(MethodError::invalid("num_modes", "must be at least 1"));
        }
        if self.num_latents == Some(0) {
            return Err(MethodError::invalid("num_latents", "must be at least 1"));
        }
        if !positive(self.alpha) {
            return Err(MethodError::invalid("alpha", "must be finite and positive"));
        }
        if !(self.reg_lambda.is_finite() && self.reg_lambda >= 0.0) {
            return Err(MethodError::invalid(
                "reg_lambda",
                "must be finite and non-negative",
            ));
        }
        if !positive(self.reg_rho) {
            return Err(MethodError::invalid(
                "reg_rho",
                "must be finite and positive",
            ));
        }
        if !(0.0..=1.0).contains(&self.tau) {
            return Err(MethodError::invalid("tau", "must be between 0 and 1"));
        }
        if !positive(self.tolerance) {
            return Err(MethodError::invalid(
                "tolerance",
                "must be finite and positive",
            ));
        }
        if self.max_iterations == 0 {
            return Err(MethodError::invalid("max_iterations", "must be at least 1"));
        }
        Ok(())
    }

    pub(crate) fn latents_for(&self, n_channels: usize) -> Result<usize, MethodError> {
        let latents = self.num_latents.unwrap_or(n_channels);
        if latents > n_channels {
            return Err(MethodError::invalid(
                "num_latents",
                format!(
                    "must be at most the number of channels ({n_channels}); the model assumes L ≤ C"
                ),
            ));
        }
        Ok(latents)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_invalid_values() {
        let base = VlmdParams::new(3);
        assert!(base.validate().is_ok());
        let cases = [
            VlmdParams {
                num_modes: 0,
                ..base.clone()
            },
            VlmdParams {
                num_latents: Some(0),
                ..base.clone()
            },
            VlmdParams {
                alpha: 0.0,
                ..base.clone()
            },
            VlmdParams {
                alpha: f64::INFINITY,
                ..base.clone()
            },
            VlmdParams {
                reg_lambda: -1e-9,
                ..base.clone()
            },
            VlmdParams {
                reg_lambda: f64::NAN,
                ..base.clone()
            },
            VlmdParams {
                reg_rho: 0.0,
                ..base.clone()
            },
            VlmdParams {
                reg_rho: f64::NAN,
                ..base.clone()
            },
            VlmdParams {
                tau: 1.0 + 1e-9,
                ..base.clone()
            },
            VlmdParams {
                tau: -1e-9,
                ..base.clone()
            },
            VlmdParams {
                tolerance: 0.0,
                ..base.clone()
            },
            VlmdParams {
                max_iterations: 0,
                ..base.clone()
            },
        ];
        for case in cases {
            assert!(case.validate().is_err(), "{case:?}");
        }
        for ok in [
            VlmdParams {
                reg_lambda: 0.0,
                ..base.clone()
            },
            VlmdParams {
                tau: 0.0,
                ..base.clone()
            },
            VlmdParams {
                tau: 1.0,
                ..base.clone()
            },
        ] {
            assert!(ok.validate().is_ok(), "{ok:?}");
        }
    }

    #[test]
    fn latents_default_to_channels_and_cannot_exceed_them() {
        let p = VlmdParams::new(2);
        assert_eq!(p.latents_for(5).unwrap(), 5);
        let three = VlmdParams {
            num_latents: Some(3),
            ..p
        };
        assert_eq!(three.latents_for(3).unwrap(), 3);
        assert!(three.latents_for(2).is_err());
    }
}
