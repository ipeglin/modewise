use thiserror::Error;

/// Identity of a method, recorded with every result it produces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MethodDescriptor {
    pub name: &'static str,
    pub version: &'static str,
    pub multivariate: bool,
}

#[derive(Debug, Error, PartialEq)]
pub enum MethodError {
    #[error("invalid parameter `{name}`: {reason}")]
    InvalidParameter { name: &'static str, reason: String },
    #[error("signal too short: {method} needs at least {needed} samples, got {actual}")]
    SignalTooShort {
        method: &'static str,
        needed: usize,
        actual: usize,
    },
    #[error("inconsistent result shape: {0}")]
    ShapeMismatch(String),
}

impl MethodError {
    pub fn invalid(name: &'static str, reason: impl Into<String>) -> Self {
        Self::InvalidParameter {
            name,
            reason: reason.into(),
        }
    }
}
