# Changelog

All notable changes to modewise. Versions follow [semantic versioning](https://semver.org); while
the major version is 0, minor versions may change the API.

## 0.2.0

The first public release.

### Methods

- Decompositions: VMD and noise-assisted MVMD (with a `uniform` start), EMD, EEMD, CEEMDAN and
  MEMD on one sifting engine, and the MODWT multiresolution analysis (25 wavelets). Methods that
  decompose each channel on its own keep per-channel centre frequencies.
- Functional connectivity (Pearson, phase locking value), per mode and per epoch.
- Power spectra (periodogram and Welch).
- NIfTI runs parcellated with label atlases at load time (nilearn-compatible).

### Analysis

- Regions: channel tables from several atlases, `select` and `region_average`.
- Cohorts: participants tables and feature tables (edges, mode power, band amplitude (ALFF and
  fALFF), block averages, or your own values), with pandas and TSV handover.
- Statistics: GLM permutation tests (groups, paired, correlation) with family-wise error, FDR and
  the network-based statistic.

### Pipelines

- `keep = false` keeps a step's outputs in memory instead of the results folder.
- Results folders store arrays in chunks of a few megabytes: writes are about 30 times faster.
  Folders from 0.1.0 still read.

### Documentation

- A documentation site with getting-started pages for Python, Rust and the command line, a user
  guide, method pages with their sources and evidence, a generated operation reference
  (`modewise ops --json`, `modewise.operations()`), the Python and Rust APIs, an examples gallery,
  and pages and `llms.txt` for agents. All code in the docs runs in CI.
- Three case studies on public data reproduce published findings: frequency-specific amplitude
  (Zuo et al. 2010), faces against shapes in the amygdala (Hariri et al. 2002) and connectome
  fingerprinting (Finn et al. 2015).

## 0.1.0

MVMD and STFT, Hilbert and Morlet time–frequency maps, VLMD, epochs with an explicit
haemodynamic lag, the pipeline engine with content-addressed caching, datasets found by file
patterns, results folders with HDF5 export, the command line and the Python package.
