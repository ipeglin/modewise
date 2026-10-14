//! Built-in pipeline operations: adapters from methods to the pipeline engine.

mod catalogue;
mod cwt;
mod emd;
mod epoch;
mod functional_connectivity;
mod glm;
mod hilbert;
mod mvmd;
mod reconstruct;
mod regions;
mod regress;
mod spectrum;
mod stft;
mod vlmd;
mod wavelet;

use std::sync::Arc;

use modewise_pipeline::{Artifact, Inputs, Registry};

pub use catalogue::{catalogue, reference_markdown};
pub use cwt::CwtOperation;
pub use emd::{CeemdanOperation, EemdOperation, EmdOperation, MemdOperation};
pub use epoch::EpochOperation;
pub use functional_connectivity::FunctionalConnectivityOperation;
pub use glm::GlmOperation;
pub use hilbert::{HilbertOperation, HilbertSpectrumOperation};
pub use mvmd::{MvmdOperation, NaMvmdOperation, VmdOperation};
pub use reconstruct::ReconstructOperation;
pub use regions::{RegionAverageOperation, SelectOperation};
pub use regress::RegressOperation;
pub use spectrum::SpectrumOperation;
pub use stft::StftOperation;
pub use vlmd::VlmdOperation;
pub use wavelet::ModwtMraOperation;

pub fn register_builtin(registry: &mut Registry) {
    registry.register(Arc::new(FunctionalConnectivityOperation));
    registry.register(Arc::new(CwtOperation));
    registry.register(Arc::new(EpochOperation));
    registry.register(Arc::new(GlmOperation));
    registry.register(Arc::new(HilbertOperation));
    registry.register(Arc::new(HilbertSpectrumOperation));
    registry.register(Arc::new(CeemdanOperation));
    registry.register(Arc::new(EemdOperation));
    registry.register(Arc::new(EmdOperation));
    registry.register(Arc::new(MemdOperation));
    registry.register(Arc::new(ModwtMraOperation));
    registry.register(Arc::new(MvmdOperation));
    registry.register(Arc::new(NaMvmdOperation));
    registry.register(Arc::new(VmdOperation));
    registry.register(Arc::new(SpectrumOperation));
    registry.register(Arc::new(StftOperation));
    registry.register(Arc::new(VlmdOperation));
    registry.register(Arc::new(ReconstructOperation));
    registry.register(Arc::new(RegressOperation));
    registry.register(Arc::new(RegionAverageOperation));
    registry.register(Arc::new(SelectOperation));
}

pub fn builtin_registry() -> Registry {
    let mut registry = Registry::new();
    register_builtin(&mut registry);
    registry
}

fn input<'a>(inputs: &'a Inputs, port: &str) -> Result<&'a Artifact, String> {
    inputs
        .get(port)
        .map(|a| a.as_ref())
        .ok_or_else(|| format!("input {port:?} missing"))
}
