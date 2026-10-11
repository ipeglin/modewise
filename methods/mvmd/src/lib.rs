//! Multivariate Variational Mode Decomposition and the methods built on its solver.
//!
//! - MVMD: Rehman and Aftab (2019), IEEE Transactions on Signal Processing. Ported from the
//!   reference at github.com/Dmocrito/mvmd (MIT; notice in THIRD_PARTY_NOTICES.md).
//! - VMD: Dragomiretskiy and Zosso (2014), the single-channel case, run on each channel.
//! - NA-MVMD: Zisou, Apostolidis and Hadjileontiadis (2021), MVMD with white-noise channels.

mod decomposer;
mod noise_assisted;
mod params;
mod vmd;

pub use decomposer::Mvmd;
pub use noise_assisted::{NaMvmd, NoiseParams};
pub use params::{FrequencyInit, MvmdParams};
pub use vmd::Vmd;
