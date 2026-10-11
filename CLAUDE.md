# CLAUDE.md

`modewise`: toolbox for data-adaptive signal analysis (MVMD first) primarily for the NTNU Brain Kyb Lab, and later for external researchers too. Crates use the `modewise_` prefix.
It generalises the analysis pipeline of the author's master's thesis. The thesis is one source among many. Its code (not public) also serves as a regression baseline.
Machine-specific paths and private sources live in the untracked `CLAUDE.local.md`; never put local paths or private repository links in tracked files.

Roadmap: `plans/roadmap.md` (excluded via `.git/info/exclude`). Revise it in place and record rejected options in it; old versions are in `plans/archive/`.

## Commands
- Build: `cargo build --workspace`
- Test: `cargo test --workspace`
- Lint: `cargo clippy --workspace --all-targets -- -D warnings`
- Format: `cargo fmt --all -- --check`
- CLI: `cargo run --release -p modewise_cli -- run <pipeline.toml> --dataset <dataset.toml> --out <results>` (also `inspect`, `ops`, `export`)
- Real-data tests (opt-in, data never committed): `MODEWISE_TCP_H5=<ds005237 *_parcellated.h5> MODEWISE_LEGACY_H5=<legacy consolidated *_bold.h5> cargo test --release -p modewise_hdf5 -- --ignored --nocapture`
- Coverage: `cargo llvm-cov --workspace --exclude modewise_python --summary-only` (CI fails below 95% lines)
- Glass-to-glass and smoke tests against the installed package: `drivers/python/.venv/bin/python -m pytest tests/e2e`
- System test on public ds005237 data (downloads ~3.4 MB): `MODEWISE_SYSTEM_TESTS=1 drivers/python/.venv/bin/python -m pytest tests/system`
- Thesis acceptance test (opt-in, local data; paths in `CLAUDE.local.md`): `MODEWISE_LEGACY_DIR=… MODEWISE_TCP_ROOT=… MODEWISE_LEGACY_ATLASES=… drivers/python/.venv/bin/python -m pytest tests/acceptance` (about 3 minutes; see `docs/thesis-acceptance.md`)
- Operation reference: `docs/reference/operations.md` is generated from each operation's `summary()` and `params_schema()` (doc comments on its `Params` fields). A test fails when it is stale; regenerate with `MODEWISE_UPDATE_DOCS=1 cargo test -p modewise_operations --test reference`. Every parameter needs a doc comment.
- Docs site (Sphinx + MyST, `docs/README.md`): `uv venv -p 3.12 .docs-venv && VIRTUAL_ENV=$PWD/.docs-venv uv pip install -r docs/requirements.txt ./drivers/python && .docs-venv/bin/python docs/build.py` (warnings are errors; CI builds it; it runs the `examples/gallery` scripts). Case studies (`examples/case-studies/`, real data via `fetch_tcp.py`) run by hand; their `REPORT.md` must carry the current version stamp (a test checks). Every ```python block in `docs/` runs (`drivers/python/.venv/bin/python -m pytest tests/docs`; use ```{code-block} python for blocks that need data on disk), every TOML pipeline, step and dataset example is checked against the code, and the Rust code in `docs/getting-started/rust.md` runs as doctests (`cargo test -p modewise_docs`). `docs/_extra/llms.txt` must list every page in the toctrees (a test checks); `llms-full.txt` is generated at build time. The Python API docs come from `_core.pyi`; `tests/test_stubs.py` keeps it in step with the compiled module.
- Mutation testing: `cargo mutants --workspace --exclude 'drivers/python/**' -j 4` (surviving mutants need a test or a written reason)
- Fuzzing (nightly): `cd fuzz && cargo +nightly fuzz run <pipeline_spec|dataset_spec|pattern> corpus/<target> seeds/<target> -- -max_total_time=240`
- Python bindings (in `drivers/python`): `uv venv -p 3.10 .venv && VIRTUAL_ENV=$PWD/.venv uv pip install maturin numpy pytest`, then `.venv/bin/maturin develop --release` and `.venv/bin/python -m pytest tests`

## Architecture (hexagonal)
- `domain/`: pure types and traits for multichannel signals, brain space, decomposition, spectra, connectivity, study and stats. No IO. Keep it modality-generic; fMRI-specific terms belong in adapters.
- `methods/`: one crate per method (`mvmd` also holds VMD and NA-MVMD, `emd` holds EMD, EEMD, CEEMDAN and MEMD, `wavelet` holds the MODWT MRA, `vlmd`, `stft`, `hilbert`, `cwt`, `fc` for connectivity, `parcellate` for volumes), plus `fourier` (mirror half-spectrum helpers shared by MVMD and VLMD, and the spectral centroid). Each crate implements `ModeDecomposer` (EMD, EEMD, CEEMDAN, MEMD, VMD, NA-MVMD, MVMD, VLMD, MODWT MRA) or `TimeFrequencyTransform` (STFT, CWT) and registers as an operation. `hilbert_spectrum` turns modes into a time-frequency map.
- `application/`: ports (traits for IO) and the pipeline engine (spec, registry, DAG executor, content-addressed cache).
- `domain/epochs`: `Events` and `Epochs` (epochs are views into a full-run source).
- `domain/brain`: `ChannelInfo` (name, region, hemisphere, open tags), `ChannelTable` and `Selection`. Atlas label readers live in `adapters/dataset` (`channels.rs`).
- `domain/study`: `Participants` and `FeatureTable` (cohort level). `application/cohort` turns per-recording results into a `FeatureTable` and builds statistical designs; `methods/stats` holds the GLM, permutation inference, FDR and NBS.
- `domain/connectivity`: `FunctionalConnectivity` matrices (window × component × channel × channel, symmetric). Effective (directed) connectivity is a different claim and gets its own type.
- `adapters/`: implementations of the ports: `dataset` (file patterns, sidecars, HDF5, NIfTI and `events_tsv` inputs), `hdf5` (reader), `nifti` (NIfTI-1 volumes and affines), `store-memory`, `store-zarr`, `results` (results folder, HDF5 export). Planned: more formats by demand.
- `drivers/`: CLI and Python bindings. These are the only composition roots and handle dependency injection.

Dependencies point inward only. Never import an adapter from domain, methods or application.

## Conventions
- Rust edition 2024, minimum Rust 1.91 (set by `zarrs`). Release builds keep `strip = false`; rustc's strip corrupts dylibs on macOS 27.
- Use `thiserror` in library crates and `anyhow` only in drivers.
- Use `tracing` for logging, not `println!`.
- Do not call `unwrap` or `expect` outside tests.
- Epochs and frequency components follow `docs/epochs.md` (decompose the full run first, haemodynamic lag explicit, flag components that do not fit in an epoch).
- Each operation owns its typed parameters. Do not add a global config struct.
- Never hard-code dataset-specific rules (for example TCP questionnaires) in core crates. Adapters must stay general across storage layouts, modalities (fMRI, EEG, MEG, fNIRS) and outputs (signals, ML handover).
- No single source is trusted on its own: not the thesis, not one paper, not one reference implementation. Methods, defaults and rules need agreement across independent sources (papers, implementations, theory or synthetic checks), cited in code docs or `docs/`. When sources disagree, or only one exists, document that instead of picking silently.
- Follow `TESTING.md`: unit, property, reference accuracy, integration, system and glass-to-glass layers. Every numerical method needs synthetic-signal tests and an accuracy test against a reference implementation. Fixtures live in `methods/*/tests/fixtures/` with the `generate.py` that made them.
- UI work follows the global workflow: impeccable init, then shape (DESIGN.md), then build with `design-taste-frontend`, then critique/audit, then polish. Keep the impeccable hook enabled.

## Reference material
- Reference implementations: MVMD github.com/Dmocrito/mvmd (MIT). VLMD: arXiv:2505.17797, github.com/Dmocrito/vlmd (GPL-3: never copy its code; it is used only to generate fixtures, see `docs/vlmd.md`). MEMD: github.com/mariogrune/MEMD-Python- (no licence: run for fixtures, never copied). EMD: github.com/laszukdawid/PyEMD (Apache-2.0; boundary mirroring credited in THIRD_PARTY_NOTICES.md). VMD: github.com/vrcarva/vmdpy (MIT, fixtures only). Wavelets: PyWavelets (MIT; filter tables generated from it). See `docs/emd.md`.
- Test data: OpenNeuro ds005237 (TCP, parcellated `.h5`; see `docs/datasets.md` and `examples/tcp/`) and ds004920.
- Local copies and private sources: see `CLAUDE.local.md` if present.
