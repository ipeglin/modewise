# Rust

modewise is a Cargo workspace of small crates: domain types (`modewise_signals`,
`modewise_decomposition`, …), one crate per method (`modewise_mvmd`, `modewise_emd`, …), and the
pipeline engine (`modewise_pipeline`, `modewise_operations`). The crates are not on crates.io yet;
depend on them from the repository:

```toml
[dependencies]
modewise_mvmd = { git = "https://github.com/ipeglin/modewise" }
modewise_signals = { git = "https://github.com/ipeglin/modewise" }
modewise_decomposition = { git = "https://github.com/ipeglin/modewise" }
```

The full API is in the [Rust reference](../reference/rust.md).

## Decompose a signal

A `MultichannelSeries` holds channels × samples with a sampling rate and channel labels. Every
decomposition implements `ModeDecomposer`.

```rust
use std::error::Error;
use std::f64::consts::PI;

use modewise_decomposition::ModeDecomposer;
use modewise_mvmd::{FrequencyInit, Mvmd, MvmdParams};
use modewise_signals::{MultichannelSeries, SamplingRate};
use ndarray::Array2;

fn main() -> Result<(), Box<dyn Error>> {
    let fs = 1.25; // Hz, a repetition time of 0.8 s
    let data = Array2::from_shape_fn((2, 600), |(c, i)| {
        let t = i as f64 / fs;
        (1.0 - 0.5 * c as f64) * (2.0 * PI * 0.02 * t).sin() + (2.0 * PI * 0.2 * t).sin()
    });
    let signal = MultichannelSeries::new(data, SamplingRate::new(fs)?, vec!["left".into(), "right".into()])?;

    let params = MvmdParams { init: FrequencyInit::Linear, tau: 0.0, tolerance: 1e-7, ..MvmdParams::new(2, 2000.0) };
    let modes = Mvmd::new(params)?.decompose(&signal)?;

    assert_eq!(modes.modes().dim(), (2, 2, 600)); // mode × channel × sample
    assert!((modes.centre_frequencies_hz()[0] - 0.02).abs() < 0.002);
    Ok(())
}
```

## Run a pipeline

The pipeline engine plans a TOML spec against the registry of operations and runs it. Results go
to an `ArtifactStore`: `MemoryStore` here, or the Zarr store that the CLI and Python use for
results folders.

```rust
use std::collections::BTreeMap;
use std::error::Error;

use modewise_operations::builtin_registry;
use modewise_pipeline::{Artifact, ArtifactStore, PipelineSpec, Plan, run};
use modewise_signals::{MultichannelSeries, SamplingRate};
use modewise_store_memory::MemoryStore;
use ndarray::Array2;

const PIPELINE: &str = r#"
[pipeline]
name = "spectra"
inputs = ["bold"]

[[step]]
id = "psd"
op = "spectrum"
input = "bold"
"#;

fn main() -> Result<(), Box<dyn Error>> {
    let data = Array2::from_shape_fn((1, 400), |(_, i)| (i as f64 * 0.3).sin());
    let bold = MultichannelSeries::new(data, SamplingRate::new(1.0)?, vec!["a".into()])?;

    let plan = Plan::new(&PipelineSpec::from_toml(PIPELINE)?, &builtin_registry())?;
    let store = MemoryStore::new();
    let result = run(&plan, &BTreeMap::from([("bold".to_string(), Artifact::Series(bold))]), &store)?;

    let key = result.output_key("psd").ok_or("no output psd")?;
    let Some(artifact) = store.get(key)? else { return Err("not stored".into()) };
    let Artifact::TimeFrequency(psd) = artifact.as_ref() else { return Err("not a spectrum".into()) };
    assert_eq!(psd.frequencies_hz().len(), 201);
    Ok(())
}
```

## Write your own method

A new decomposition implements `ModeDecomposer` (`descriptor` and `decompose`), and an operation
wraps it for pipelines: `Operation` with `descriptor`, `resolve` (check parameters and fill
defaults), `run`, and `summary` and `params_schema` for the generated reference. The operations in
`application/operations/src/` are worked examples, and `CLAUDE.md` lists the conventions every
method follows (sources, tests, no single trusted reference).
