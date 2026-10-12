# modewise

Data-adaptive signal decomposition for multichannel recordings: fMRI first, and EEG, MEG and fNIRS
by design. A Rust core with a Python package and a command line. Written for the Brain Kyb Lab at
NTNU, and open to anyone.

modewise splits signals into oscillatory modes (MVMD, EMD and their relatives, wavelets), turns
them into time–frequency maps, connectivity and band features, and tests them across a cohort.
Analyses are pipelines you write in TOML or Python. Steps are cached by content, every run is
recorded, and each method is checked against independent references.

Start in [Python](getting-started/python.md), [Rust](getting-started/rust.md) or on the
[command line](getting-started/cli.md). Operations and their parameters are listed in the
[operation reference](reference/operations.md).

```{toctree}
:caption: Getting started
:maxdepth: 1

getting-started/python
getting-started/rust
getting-started/cli
agents
```

```{toctree}
:caption: User guide
:maxdepth: 1

pipelines
datasets
epochs
regions
volumes
time-frequency
connectivity
features
statistics
```

```{toctree}
:caption: Examples
:maxdepth: 1

auto_examples/index
case-studies
```

```{toctree}
:caption: Methods
:maxdepth: 1

emd
wavelets
vlmd
thesis-acceptance
```

```{toctree}
:caption: Reference
:maxdepth: 1

reference/operations
reference/python
reference/rust
```
