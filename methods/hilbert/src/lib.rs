//! Hilbert transform: analytic signal, instantaneous amplitude and frequency, Hilbert spectrum.
//!
//! Conventions and sources: `docs/time-frequency.md`.

mod analytic;
mod spectrum;

pub use analytic::{Padding, analytic_signal, instantaneous, instantaneous_frequency};
pub use spectrum::{HilbertSpectrum, Weight, hilbert_spectrum};
