# Building the documentation

The site is built with Sphinx and MyST from the markdown files in this folder.

```bash
uv venv -p 3.12 .docs-venv && VIRTUAL_ENV=$PWD/.docs-venv uv pip install -r docs/requirements.txt ./drivers/python
.docs-venv/bin/python docs/build.py            # HTML in docs/_build/html, warnings are errors
```

- **Generated pages:** `reference/operations.md` comes from the operations' own schemas
  (`MODEWISE_UPDATE_DOCS=1 cargo test -p modewise_operations --test reference`). The Python API
  pages come from `drivers/python/python/modewise/_core.pyi` at build time. The Rust API is
  rustdoc, built by `build.py` into `_build/html/rust/`.
- **Code in pages runs:** every ```python block runs under pytest (`tests/docs`), with a page's
  blocks in order; use ```{code-block} python for code that needs data on disk. Every TOML
  pipeline, step or dataset example must plan or validate against the code, and the Rust page's
  code runs as doctests (`cargo test -p modewise_docs`).
- **Examples:** `examples/gallery/plot_*.py` run at every build (sphinx-gallery), so their
  output and figures match the current code; keep them to seconds on synthetic data. Real-data
  case studies live in `examples/case-studies/`, run by hand, and write a `REPORT.md` stamped with
  the modewise version; a test fails when a report is older than the current version.
- **Where pages live:** the existing guides stay at `docs/<topic>.md`, because code and other
  docs refer to those paths; `index.md` arranges them.
