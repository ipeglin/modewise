# Testing

modewise is used for research results, so every claim it makes about a signal must be tested. Code is not done until the layers below that apply to it have tests.

## Layers

| Layer | What it proves | Where | Runs in CI |
|---|---|---|---|
| Unit | One type or function behaves as specified, including every error case | `#[cfg(test)]` next to the code | yes |
| Property | Invariants hold for many generated inputs, for example round trips, ordering and shape rules | `tests/properties.rs` per crate (`proptest`) | yes |
| Reference accuracy | Methods match a published reference implementation within agreed limits | `methods/*/tests/` with fixtures and the `generate.py` that made them | yes |
| Integration | Crates work together through their ports, with real files in temp folders | `tests/` per crate | yes |
| System | The toolbox works on real public data (ds005237 files downloaded in CI) | `tests/system/` | yes |
| Glass-to-glass | The user's path end to end: install the wheel, run the `modewise` command and the Python API on a dataset, open the results in independent tools (h5py, zarr-python), and compare numbers | `tests/e2e/` | yes, on CentOS 7 |
| Smoke | The installed package imports, `modewise --version` and `modewise ops` run | `tests/e2e/` | yes |
| Local real data | Legacy thesis outputs and restricted data still reproduce (one line of evidence, and a regression check) | `#[ignore]` tests, opt-in env vars (see `CLAUDE.md`) | no (local only) |
| Fuzz | Parsers never panic on hostile input (pipeline, dataset and pattern specs) | `fuzz/` (`cargo-fuzz`) | nightly / manual |

## Quality checks on the tests themselves

- **Coverage** (`cargo llvm-cov`): reported in CI. Lines that no test reaches need a test or a reason.
- **Mutation testing** (`cargo mutants`): run before each release and on new modules. A surviving mutant means a missing assertion, unless it is equivalent (cannot change any output). Equivalent mutants are listed with their reason in the PR that adds them.

## Rules

- **Every bug fix starts with a failing test** that reproduces it.
- **Every error variant** has a test that triggers it.
- **Correctness needs independent lines of evidence:** for example a reference implementation, a published description, a synthetic signal with a known answer, or theory. One source alone (a paper, an implementation, or the thesis) is not enough. Note it when only one exists.
- **Every numerical method** has synthetic-signal tests (known frequencies recovered) and a reference accuracy test.
- **Real data is never committed.** Tests either download public data or are opt-in.
- **Tests use temp folders** and never write into the repo.
- **No flaky tests.** Seed random generators; never depend on timing or thread order.
