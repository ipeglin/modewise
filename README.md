<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/modewise-monochrome-dark-transparent.svg">
    <img alt="modewise" src="docs/assets/modewise-monochrome-light-transparent.svg" width="440">
  </picture>
</h1>

Data-adaptive signal decomposition for multichannel recordings (fMRI, EEG, MEG, fNIRS), with a Rust core and a Python front end. Written for the Brain Kyb Lab at NTNU.

- **Methods:** mode decompositions (MVMD, noise-assisted MVMD, VMD, VLMD, EMD, EEMD, CEEMDAN, MEMD), wavelet multiresolution analysis (MODWT), spectra (periodogram, Welch) and band amplitude (ALFF, fALFF), time–frequency maps (STFT, Morlet CWT, Hilbert spectrum), functional connectivity, and group statistics.
- **Pipelines:** you compose them, in TOML or Python. Steps are cached by content, so rerunning with one changed parameter only recomputes what depends on it. Every run is recorded.
- **Datasets:** BIDS and other layouts, found by file patterns. Runs scale from one recording to a cohort, in parallel or split across Slurm jobs.
- **Epochs:** cut from the full-run result with an explicit haemodynamic lag. Components that do not fit in an epoch are flagged ([docs/epochs.md](docs/epochs.md)).
- **Results:** kept in a folder you can reopen from Python, and exportable to HDF5 for MATLAB.

## Install

modewise needs Python 3.10 or newer. Prebuilt wheels need no Rust.

```bash
pip install modewise                                  # add "modewise[xarray]" for to_xarray()

# Old Linux such as CentOS 7: only take prebuilt wheels, or pip may try to build numpy/pandas from source
pip install --only-binary=:all: modewise

# From source (needs Rust 1.91 or newer, https://rustup.rs)
pip install "git+https://github.com/ipeglin/modewise.git#subdirectory=drivers/python"
```

Installing also gives you the `modewise` command.

## Quick start (Python, Spyder, notebooks)

[`examples/quickstart.py`](examples/quickstart.py) runs as-is on synthetic data:

```python
import modewise

result = modewise.mvmd(x, sampling_rate=1.25, num_modes=10, alpha=1000.0, init="exponential")
result.modes                   # mode × channel × sample
result.centre_frequencies_hz

pipeline = modewise.Pipeline.from_file("examples/tcp/blocks.toml")
dataset = modewise.Dataset.from_file("examples/tcp/dataset.toml")
batch = pipeline.run_dataset(dataset, "results/")        # whole cohort, in parallel
epochs = batch.output(batch.succeeded[0], "blocks")
epochs.resolvable              # epoch × mode: does the mode's period fit in the epoch?

modewise.Results("results/").export_hdf5("tcp-blocks", "export/")    # for MATLAB
```

## Command line and HPC

```bash
modewise inspect examples/tcp/dataset.toml --load        # check what a dataset spec finds
modewise run examples/tcp/blocks.toml --dataset examples/tcp/dataset.toml --out results/
modewise run ... --shard 3/8                             # one Slurm array task: every 8th recording from #3
modewise export results/ --to export/                    # one .h5 per recording
modewise ops                                             # available operations
```

**Exit codes:** `0` means all recordings succeeded. `2` means some failed; the summary is in `results/runs/<pipeline>/summaries/`. `1` means invalid input.

## Results

A results folder holds a Zarr store, a manifest per recording, and a summary per run or shard. Open it with `modewise.Results(path)`.

- **Python 3.11 or newer:** zarr-python 3 can read the store directly.
- **MATLAB:** use the HDF5 export. `h5read` returns dimensions in reverse order, so modes appear as time × channel × mode.

## Documentation

Online: **https://ipeglin.github.io/modewise/**.

The documentation site is built from `docs/` (see [docs/README.md](docs/README.md)); start with [docs/index.md](docs/index.md) and the getting-started pages for [Python](docs/getting-started/python.md), [Rust](docs/getting-started/rust.md) and the [command line](docs/getting-started/cli.md). Coding agents should start with [docs/agents.md](docs/agents.md); the site also serves `llms.txt` and `llms-full.txt`.

- [docs/pipelines.md](docs/pipelines.md): pipeline files and each operation's parameters and evidence.
- [docs/reference/operations.md](docs/reference/operations.md): every operation with its parameters, types and defaults, generated from the code (`modewise ops --json` for tools and agents).
- [docs/datasets.md](docs/datasets.md): dataset specs, patterns, formats.
- [docs/epochs.md](docs/epochs.md): how epochs relate to decompositions, with sources.
- [docs/features.md](docs/features.md): participants, cohort feature tables, and handover to statistics and ML.
- [docs/thesis-acceptance.md](docs/thesis-acceptance.md): how far modewise reproduces the thesis pipeline, and why it differs where it does.
- [docs/statistics.md](docs/statistics.md): group, paired and correlation tests with permutation inference, FWER, FDR and NBS.
- [docs/volumes.md](docs/volumes.md): parcellating NIfTI runs with label atlases (nilearn-compatible).
- [docs/regions.md](docs/regions.md): describing channels with atlases, selecting regions, and region summaries.
- [docs/connectivity.md](docs/connectivity.md): functional connectivity (and how it differs from effective connectivity), with sources.
- [docs/wavelets.md](docs/wavelets.md): MODWT multiresolution analysis, with the octave bands at common sampling rates.
- [docs/emd.md](docs/emd.md): EMD, EEMD, CEEMDAN and MEMD, their sources, and where modewise differs from memd.m.
- [docs/vlmd.md](docs/vlmd.md): VLMD, and where the paper and its reference code disagree.
- [TESTING.md](TESTING.md): how modewise is tested; no single source is trusted on its own.

## Citing and credits

modewise is developed by Ian Philip Eglin (Department of Electronic Systems, NTNU). It is a general toolbox that grew out of the analysis pipeline from Ian Philip Eglin's master's thesis, which was written at the Brain Kyb Lab, Department of Engineering Cybernetics, NTNU, under the supervision of Marta Molinas.

If you use modewise in research, please cite it: GitHub's "Cite this repository" button uses [CITATION.cff](CITATION.cff).

## Licence

BSD-3-Clause: see [LICENSE](LICENSE). The MVMD code is ported from MIT-licensed work the EMD boundary handling follows Apache-2.0-licensed PyEMD, and the wavelet filters come from MIT-licensed PyWavelets, all credited in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The licences of compiled Rust dependencies are in [drivers/python/THIRD_PARTY_LICENSES.md](drivers/python/THIRD_PARTY_LICENSES.md) and ship in every wheel.

## Development

See [CLAUDE.md](CLAUDE.md) for commands. In short: `cargo test --workspace`. Python tests run from `drivers/python` after `maturin develop`. CI also tests the Linux wheel on CentOS 7.
