# Python

## Install

modewise needs Python 3.10 or newer. Prebuilt wheels need no Rust; see the README for wheels and
building from source. Installing also gives you the `modewise` command.

## Decompose a signal

Every method takes a channels × samples array and a sampling rate.

```python
import numpy as np
import modewise

fs = 1.25                                  # Hz, a repetition time of 0.8 s
t = np.arange(600) / fs
slow, fast = np.sin(2 * np.pi * 0.02 * t), np.sin(2 * np.pi * 0.2 * t)
x = np.vstack([slow + fast, 0.5 * slow + fast])

result = modewise.mvmd(x, fs, num_modes=2, alpha=2000.0, init="linear", tau=0.0, tolerance=1e-7)
print(result.centre_frequencies_hz)        # about [0.02, 0.2]
print(result.modes.shape)                  # (mode, channel, sample) = (2, 2, 600)
```

The other decompositions work the same way: `vmd`, `na_mvmd`, `vlmd`, `emd`, `eemd`,
`ceemdan`, `memd` and `modwt_mra`. Time–frequency maps come from `stft`, `cwt`, `spectrum`,
`hilbert` and `hilbert_spectrum`.

## Build a pipeline

A pipeline names its inputs and steps; each step is an operation from the
[operation reference](../reference/operations.md). Steps are cached by content, so a rerun only
computes what changed.

```python
pipeline = modewise.Pipeline.from_dict({
    "pipeline": {"name": "bands", "inputs": ["bold"]},
    "step": [
        {"id": "modes", "op": "mvmd", "input": "bold",
         "params": {"num_modes": 2, "alpha": 2000.0, "init": "linear", "tau": 0.0}},
        {"id": "slow", "op": "reconstruct", "input": "modes", "params": {"modes": [0]}},
        {"id": "psd", "op": "spectrum", "input": "slow"},
    ],
})
run = pipeline.run({"bold": modewise.Series(x, fs, labels=["left", "right"])})
psd = run.output("psd")
print(psd.frequencies_hz[np.argmax(psd.power[0, :, 0])])   # the slow mode's peak, about 0.02 Hz
```

The same pipeline can be a TOML file (`Pipeline.from_file`); [pipelines](../pipelines.md)
describes the format.

## Run over a dataset

A dataset spec finds recordings by file patterns and says how to read them
([datasets](../datasets.md)). `run_dataset` processes every recording in parallel and saves the
results in a folder you can reopen.

```{code-block} python
dataset = modewise.Dataset.from_file("dataset.toml")
batch = pipeline.run_dataset(dataset, "results/")
results = modewise.Results("results/")
table = results.feature_table("bands", "psd", "band_amplitude",
                              bands={"slow5": (0.01, 0.027), "slow4": (0.027, 0.073)},
                              participants=dataset.participants)
report = modewise.test_paired(table, ("slow4", "slow5"))   # per channel, permutation p, FWER and FDR
```

[Features](../features.md) and [statistics](../statistics.md) cover the feature tables and the
tests. `modewise.operations()` lists every operation with its parameter schema, for tools and
agents.
