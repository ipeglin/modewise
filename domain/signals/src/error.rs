use thiserror::Error;

#[derive(Debug, Error, PartialEq)]
pub enum SignalError {
    #[error("sampling rate must be finite and positive, got {0}")]
    InvalidSamplingRate(f64),
    #[error("series needs at least one channel")]
    NoChannels,
    #[error("series needs at least 2 samples, got {0}")]
    TooFewSamples(usize),
    #[error("non-finite value at channel {channel}, sample {sample}")]
    NonFiniteValue { channel: usize, sample: usize },
    #[error("expected {expected} channel labels, got {actual}")]
    LabelCountMismatch { expected: usize, actual: usize },
    #[error("duplicate channel label {0:?}")]
    DuplicateLabel(String),
}
