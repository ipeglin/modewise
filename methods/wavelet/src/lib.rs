//! Wavelet multiresolution analysis: the MODWT MRA, with modes from the smooth (lowest) to the
//! first detail (highest). Sources and choices are in docs/wavelets.md.

mod decomposer;
mod filters;
mod modwt;

pub use decomposer::{Boundary, ModwtMra, wavelet_names};
