//! Python bindings. Composition root for the `modewise` Python package.

mod channels;
mod cohort;
mod convert;
mod data;
mod emd;
mod events;
mod functional_connectivity;
mod hilbert;
mod io;
mod mvmd;
mod pipeline;
mod results;
mod saved;
mod statistics;
mod stft;
mod vlmd;
mod wavelet;

use pyo3::prelude::*;

/// Runs the `modewise` command with the given arguments (the first is the program name).
#[pyfunction]
fn cli(py: Python<'_>, args: Vec<String>) -> u8 {
    py.detach(|| modewise_cli::main(args))
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<results::Series>()?;
    m.add_class::<results::Decomposition>()?;
    m.add_class::<results::TimeFrequencyMap>()?;
    m.add_class::<pipeline::MemoryStore>()?;
    m.add_class::<pipeline::Pipeline>()?;
    m.add_class::<pipeline::PipelineRun>()?;
    m.add_class::<pipeline::BatchRun>()?;
    m.add_class::<data::Dataset>()?;
    m.add_class::<events::Events>()?;
    m.add_class::<results::Instantaneous>()?;
    m.add_class::<results::Loadings>()?;
    m.add_class::<channels::ChannelTable>()?;
    m.add_class::<cohort::Participants>()?;
    m.add_class::<cohort::FeatureTable>()?;
    m.add_class::<statistics::StatsReport>()?;
    m.add_function(wrap_pyfunction!(statistics::test_groups, m)?)?;
    m.add_function(wrap_pyfunction!(pipeline::operations, m)?)?;
    m.add_function(wrap_pyfunction!(statistics::test_paired, m)?)?;
    m.add_function(wrap_pyfunction!(statistics::test_correlation, m)?)?;
    m.add_function(wrap_pyfunction!(channels::read_channels, m)?)?;
    m.add_class::<results::FunctionalConnectivity>()?;
    m.add_function(wrap_pyfunction!(
        functional_connectivity::functional_connectivity,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(vlmd::vlmd, m)?)?;
    m.add_function(wrap_pyfunction!(hilbert::cwt, m)?)?;
    m.add_function(wrap_pyfunction!(hilbert::hilbert, m)?)?;
    m.add_function(wrap_pyfunction!(hilbert::hilbert_spectrum, m)?)?;
    m.add_class::<events::Epochs>()?;
    m.add_function(wrap_pyfunction!(events::read_events_tsv, m)?)?;
    m.add_class::<saved::Results>()?;
    m.add_function(wrap_pyfunction!(cli, m)?)?;
    m.add_function(wrap_pyfunction!(io::read_hdf5, m)?)?;
    m.add_function(wrap_pyfunction!(io::read_nifti, m)?)?;
    m.add_function(wrap_pyfunction!(emd::emd, m)?)?;
    m.add_function(wrap_pyfunction!(emd::eemd, m)?)?;
    m.add_function(wrap_pyfunction!(emd::ceemdan, m)?)?;
    m.add_function(wrap_pyfunction!(emd::memd, m)?)?;
    m.add_function(wrap_pyfunction!(wavelet::modwt_mra, m)?)?;
    m.add_function(wrap_pyfunction!(mvmd::mvmd, m)?)?;
    m.add_function(wrap_pyfunction!(mvmd::vmd, m)?)?;
    m.add_function(wrap_pyfunction!(mvmd::na_mvmd, m)?)?;
    m.add_function(wrap_pyfunction!(stft::stft, m)?)?;
    m.add_function(wrap_pyfunction!(stft::spectrum, m)?)?;
    Ok(())
}
