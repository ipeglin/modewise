# Contributing to modewise

Thank you for helping. Bug reports, method requests, documentation fixes and code are all
welcome. Please open an issue first for anything larger than a small fix, so we can agree on the
approach.

## Set up

You need Rust 1.91 or newer and Python 3.10 or newer.

```bash
cargo build --workspace
cargo test --workspace
cd drivers/python && uv venv -p 3.10 .venv && VIRTUAL_ENV=$PWD/.venv uv pip install maturin numpy pytest h5py pandas
.venv/bin/maturin develop --release && .venv/bin/python -m pytest tests ../../tests/e2e ../../tests/docs
```

`docs/README.md` explains how to build the documentation site.

## What a change needs

- **Architecture.** modewise is hexagonal: `domain/` has pure types, `methods/` one crate per
  method, `application/` the pipeline engine and operations, `adapters/` file formats and stores,
  `drivers/` the CLI and Python package. Dependencies point inward only.
- **Evidence for methods.** No single source is trusted on its own. A new method or default needs
  agreement across independent sources (papers, reference implementations, theory or synthetic
  checks), cited in its code docs or its page in `docs/`. Where sources disagree, document it.
- **Tests.** Every numerical method has synthetic-signal tests and an accuracy test against a
  reference implementation, with the fixture and the `generate.py` that made it. `TESTING.md`
  describes the layers. CI requires 95% line coverage, and surviving mutants (`cargo mutants`)
  need a test or a written reason.
- **Documentation.** Every operation parameter has a doc comment; regenerate the operation
  reference with `MODEWISE_UPDATE_DOCS=1 cargo test -p modewise_operations --test reference`.
  Code in `docs/` runs in CI, and public Python functions need a documented stub in `_core.pyi`.
- **Style.** `cargo fmt`, `cargo clippy --workspace --all-targets -- -D warnings`, `thiserror` in
  libraries, `tracing` for logging, no `unwrap` or `expect` outside tests.

## Licence

By contributing, you agree that your contributions are licensed under the BSD-3-Clause licence
of the project (`LICENSE`).
