# For agents

This page is for coding agents (and people) who use modewise to analyse data. It lists how to
discover what modewise can do, the workflow, and the conventions that are easy to get wrong.
[`llms.txt`](https://llmstxt.org) at the site root indexes the documentation in markdown, and
`llms-full.txt` holds all of it in one file.

## Discover operations instead of guessing

Every pipeline step names an operation. Do not invent operation or parameter names: list them.

- **Python:** `modewise.operations()` returns each operation's name, summary, inputs and outputs
  (port names and data kinds) and `params`, a JSON Schema with types, descriptions, defaults and
  the required parameters.
- **Shell:** `modewise ops --json` prints the same; `modewise ops` is a short human list.
- **Docs:** the [operation reference](reference/operations.md) is generated from the same data.

Unknown parameters are rejected, and missing ones take the defaults in the schema.

## Workflow

1. **Describe the data** in a dataset spec ([datasets](datasets.md)): file patterns with
   `{entity}` fields, the format, and the sampling rate (a sidecar or `sampling_rate_hz`).
   Check it with `modewise inspect dataset.toml --load` before running anything.
2. **Write the pipeline** in TOML or as a dict ([pipelines](pipelines.md)). Parsing it checks
   every operation, parameter, reference and data kind, so a pipeline that loads will run.
3. **Run** it with `Pipeline.run_dataset(dataset, "results/")` or `modewise run`. Unchanged steps
   are reused from the results folder. One failing recording does not stop the others; read
   `BatchRun.failed`.
4. **Collect** features across recordings with `Results.feature_table` ([features](features.md)),
   or arrays with `Results.stack`. Values you derive yourself go into `modewise.FeatureTable(...)`.
5. **Test** with `test_groups`, `test_paired`, `test_correlation` or `test_glm`
   ([statistics](statistics.md)); use `p_fwer` or `q_fdr`, not the uncorrected p-values, when
   many features are tested.

## A complete run

```python
import numpy as np
import h5py
import modewise

# A tiny dataset: eight subjects, two channels, 1.25 Hz.
rng = np.random.default_rng(0)
t = np.arange(400) / 1.25
for subject in ["01", "02", "03", "04", "05", "06", "07", "08"]:
    rhythm = np.sin(2 * np.pi * 0.05 * t + rng.uniform(0, 6))
    with h5py.File(f"sub-{subject}_task-rest_bold.h5", "w") as f:
        f["timeseries"] = np.vstack([rhythm, rng.standard_normal(t.size)]) + 0.3 * rng.standard_normal((2, t.size))

dataset = modewise.Dataset.from_toml("""
root = "."
[[input]]
name = "bold"
pattern = "sub-{subject}_task-{task}_bold.h5"
format = "hdf5"
dataset = "timeseries"
sampling_rate_hz = 1.25
""")
print(dataset.recordings)

# Look up an operation's parameters before using it.
spectrum = next(op for op in modewise.operations() if op["name"] == "spectrum")
print(spectrum["summary"], sorted(spectrum["params"]["properties"]))

pipeline = modewise.Pipeline.from_dict({
    "pipeline": {"name": "amplitude", "inputs": ["bold"]},
    "step": [{"id": "psd", "op": "spectrum", "input": "bold"}],
})
batch = pipeline.run_dataset(dataset, "results", progress=False)
assert not batch.failed, batch.failed

table = batch.results.feature_table("amplitude", "psd", "band_amplitude",
                                    bands={"slow4": (0.027, 0.073), "slow5": (0.01, 0.027)})
report = modewise.test_paired(table, ("slow4", "slow5"), n_permutations=1000, seed=0)
print(report.feature_names, report.t.round(1), report.p_permutation)
```

## Conventions

- **Arrays are channels × samples** in; modes are **mode × channel × sample**; time–frequency
  power is **channel × frequency × time**; connectivity is **window × component × channel ×
  channel**.
- **Modes go from low to high frequency.** For EMD-family results mode 0 is the residual, and for
  the MODWT MRA it is the smooth.
- **Aligned or not:** `Decomposition.aligned_across_channels` says whether mode k is the same band
  in every channel (MVMD, MEMD, MODWT MRA) or not (VMD, EMD, EEMD, CEEMDAN, whose per-channel
  frequencies are in `channel_centre_frequencies_hz`, NaN where a channel has fewer modes). Do
  not compare mode k across channels when it is not aligned.
- **Epochs need an explicit lag** (`lag_s`, the haemodynamic delay in fMRI, 0 for EEG), and are cut
  from a full-run result: decompose first, then epoch ([epochs](epochs.md)). Check `resolvable`
  before using a slow mode in a short epoch.
- **Large intermediates:** set `keep = false` on steps whose outputs only feed later steps (for
  example modes before connectivity); they are then not written to the results folder.
- **Seeds:** methods that use noise (NA-MVMD, EEMD, CEEMDAN) and permutation tests take a `seed`;
  set it for reproducible results.
- **Every method states its sources** and where references disagree; read the method page before
  explaining a result ([EMD](emd.md), [wavelets](wavelets.md), [VLMD](vlmd.md)).

## When something fails

Errors name the problem: the step, the parameter, the file or the recording. A `ValueError` from
`Pipeline.from_toml` or `Dataset.from_toml` means the spec is invalid and nothing ran.
`BatchRun.failed` maps each failed recording to its error. In the shell, exit code `2` means some
recordings failed and `1` means invalid input.
