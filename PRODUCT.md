# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

This applies only to a future UI, which is parked: no UI design work for now (2026-10-10). Today the product's interface is the Python API and the CLI. `web` is a placeholder until the UI is taken up.

## Stack

The core is Rust with a Python package (PyO3/maturin) and a CLI. A UI stack is not chosen; it is decided if and when the UI is taken up.

## Users

Primary users: researchers in the NTNU Brain Kyb Lab who analyse brain and physiological signals. Later, the toolbox should also serve external researchers, so nothing may depend on lab-only data, infrastructure or knowledge. Most lab members work in Python (Spyder, notebooks). Many run on NTNU's HPC cluster (IDUN, CentOS) as well as on laptops. They are not expected to install or write Rust.

Confirmed on 2026-10-10:

- **How they work:** in three ways, all expected:
  - function calls (`modewise.mvmd(x, ...)`) for exploring
  - Python pipelines over many recordings
  - CLI batch jobs on HPC (Slurm) with no Python code
- **Typical scale:** a cohort batch of tens to hundreds of subjects with several runs each. Results are stored and revisited.
- **Where results go:**
  - Python objects in the same session (numpy, xarray)
  - files that colleagues and later sessions reopen
  - ML/DL models
  - MATLAB, since some lab members need it

## Product Purpose

modewise is a toolbox for data-adaptive signal analysis of multichannel signals. It covers:

- **mode decompositions:** MVMD, NA-MVMD, VMD, VLMD, EMD, EEMD, CEEMDAN and MEMD, plus the MODWT wavelet MRA
- **classic transforms:** STFT and wavelets

Users compose their own pipelines instead of following fixed stages. The toolbox rewrites a master's thesis pipeline (fMRI BOLD, MVMD) as a reusable lab tool. Success means lab members can:

- run analyses on their own data and modality
- trust the numbers
- hand results to their own models

The current focus is the experience of using the core: a clear Python API, a CLI, and results that are easy to reopen. A UI for viewing data and results may come later; it is parked.

## Positioning

- **Speed and familiarity:** a Rust core for speed and safety, used from Python.
- **Trustworthy numbers:** every method is checked by several independent lines of evidence. MVMD matches the reference implementation `Dmocrito/mvmd` to 1e-12, recovers known tone frequencies, and reproduces the legacy thesis outputs.
- **Reruns skip unchanged steps:** pipelines cache by content.
- **Every run is recorded:** each run writes a provenance manifest.
- **Not tied to fMRI:** fMRI comes first, while EEG, MEG and fNIRS come through adapters.

## Operating Context

- **Where it lives:** a private GitHub repo during development, which moves to the lab's GitHub organisation when finished. Making it public for external users comes later; the BSD-3-Clause licence already allows that.
- **How lab members install it:** prebuilt wheels (`manylinux2014`, Python 3.10+). On old Linux they install with `--only-binary=:all:`.
- **Data sources:** BIDS and BIDS-like datasets, HDF5 files, and later NIfTI, CIFTI, EDF and SNIRF.
- **ML handover:** machine learning lives in a separate Python package that consumes exported features.

## Capabilities and Constraints

- **Built today:**
  - MVMD, NA-MVMD, VMD (per channel), EMD, EEMD and CEEMDAN (per channel), MEMD, the MODWT MRA, VLMD (channel modes, latent modes and sparse loadings), functional connectivity (undirected: Pearson, phase locking value; per mode and per epoch), STFT, the Hilbert transform (instantaneous amplitude and frequency, Hilbert spectrum), and the Morlet CWT with a cone of influence
  - a pipeline engine (TOML or Python, cached, with a manifest)
  - dataset discovery by file pattern (any modality)
  - batch runs over cohorts, with shards for Slurm
  - results folders (a Zarr store plus manifests), with HDF5 export for MATLAB
  - events (BIDS `events.tsv`) and epochs: cut a full-run series or decomposition by event windows with an explicit lag, with resolvability flags (`docs/epochs.md`)
  - the `modewise` command (`run`, `inspect`, `ops`, `export`), also installed by pip
  - statistics: group, paired and correlation tests (GLM with covariates, Welch), permutation p, FWER, FDR, NBS
  - cohorts: participants from BIDS tables, feature tables (connectivity edges, mode power) with TSV, pandas and stacked-array handover
  - volumes: NIfTI runs parcellated with label atlases at load time (nilearn-compatible)
  - regions: channel tables from several atlases (TSV/BIDS dseg, Schaefer, Tian), selection before or after decomposition, and region averages
  - the Python API (`mvmd`, `na_mvmd`, `vmd`, `emd`, `eemd`, `ceemdan`, `memd`, `modwt_mra`, `vlmd`, `functional_connectivity`, `ChannelTable`, `read_channels`, `read_nifti`, `Participants`, `FeatureTable`, `Results.feature_table`, `Results.stack`, `test_groups`, `test_paired`, `test_correlation`, `spectrum`, `stft`, `cwt`, `hilbert`, `hilbert_spectrum`, `Pipeline`, `Dataset`, `Results`, `Series`, `Events`, `Epochs`, `read_hdf5`, `read_events_tsv`, `to_xarray`)
- **Planned:**
  - more methods and formats
  - group statistics and feature export
  - a server and UI (parked)
- **Constraints:**
  - Must stay general across storage sources, modalities and outputs, from plain signals to ML handover.
  - Real datasets (such as ds005237, a restricted TCP dataset) are never committed to the repo.
- **Required by usage:**
  - stored results that can be reopened from Python and MATLAB
  - cohort-scale batch runs that can be split across Slurm jobs
- **Undecided:**
  - everything about the UI (parked)

## Brand Commitments

The name is `modewise`. It must stay modest. It must not present itself as the lab's official or standard tool, because the lab has many people and many repositories.

## Evidence on Hand

- The thesis pipeline and its outputs (14 subjects of legacy results). This is one empirical source, and also the regression baseline.
- Reference-parity fixtures for MVMD (against `Dmocrito/mvmd`), VMD (against vmdpy), EMD (against PyEMD), MEMD (against MEMD-Python), the MODWT MRA (against PyWavelets), STFT and spectra (against scipy).
- The public dataset ds005237 (TCP), fetched locally for opt-in tests.
- There are no users, testimonials or published benchmarks yet, and none may be invented.

## Product Principles

1. **Trust over features:** every method is checked against a reference before it is offered.
2. **Python first, Rust inside:** lab members never need Rust, and results match across CLI and Python.
3. **Reproducible by default:** runs are cached and recorded, so any result can be traced to its parameters and inputs.
4. **General, not dataset-shaped:** no dataset, modality or storage layout is baked into the core.
5. **Modest:** a useful lab tool, not a declared standard.
6. **Consensus, not a single source:** methods and defaults rest on agreement across independent sources. No paper, implementation or earlier work (including the original thesis) is trusted alone, and disagreements are documented.
7. **Only claim what the data supports:** components the data cannot resolve, for example periods longer than an epoch, are flagged and never silently reported (see `docs/epochs.md`).
