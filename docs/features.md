# Cohort features

How modewise turns many recordings' results into one table, the handover to statistics (S8b) and to machine learning outside modewise.

## Participants

A dataset spec's `[participants]` section reads `participants.tsv` and any phenotype tables, joined on the subject entity (`docs/datasets.md`). In Python, `Dataset.participants` has `.columns`, `.subjects`, `.value(subject, column)` and `.to_pandas()`.

- **Text first:** values stay text until a number is needed, so group labels and scores live side by side.
- **Missing codes:** empty cells, `n/a`, `NA`, and any code listed under `missing` (such as `-9` or `999`) mean "no value".
- **Derived scores stay out of core:** a total recomputed from items, or a group defined by thresholds, is a dataset rule. Compute it in Python or an example script (as `examples/tcp/make_participants.py` does for the thesis's anhedonia grouping), then join it as one more table.

## The feature table

`Results.feature_table(pipeline, output, features, participants=...)` builds one table over every recording of a pipeline:

- **Rows:** one per recording. When the output was computed on epochs, one per recording and condition (the epochs' event label, such as `face` and `shape`), in order of first appearance. For `band_amplitude`, one per recording and band.
- **Columns:** `recording`, `subject` (taken from the recording id), `condition`, the participant columns, then the features.

| `features` | From | Feature names | Values |
|---|---|---|---|
| `edges` | functional connectivity | `m{k}:{a}~{b}`: the upper triangle per component | Pearson r becomes **Fisher z**, so values can be averaged and tested; Fisher z and PLV stay as they are. A condition's epochs are averaged. |
| `mode_power` | a decomposition, or its epochs | `m{k}:{channel}` and `m{k}:centre_hz` | each mode's variance per channel, and its centre frequency; averaged over a condition's epochs |
| `mean` | epochs of a series | `{channel}`, with the epoch label as the condition | the mean signal in each condition's epochs (each epoch's mean counts once): block averages for task contrasts |
| `band_amplitude` | a spectrum or time-frequency map; `bands={"slow5": (0.01, 0.027), ...}` | `{channel}`, with the band as the condition | the mean amplitude (√power) over the band's bins [low, high) and all frames: ALFF. With `relative=True`, the band's summed amplitude over that of every frequency above 0: fALFF. |

**Band amplitude.** ALFF is the mean square-root power in a band (Zang et al. 2007); fALFF divides the band's amplitude by that of the whole spectrum (Zou et al. 2008), which reduces the influence of physiological noise in non-grey-matter regions. Zuo et al. (2010) split the classic 0.01–0.08 Hz range into slow-5 (0.01–0.027 Hz) and slow-4 (0.027–0.073 Hz) and found their balance differs between regions. Bands are half-open, so adjacent bands share no bin. Because bands are conditions, `test_paired(table, ("slow4", "slow5"))` compares them within each channel, as Zuo et al.'s paired test does. ALFF is in units of √(power per Hz); compare it within a study, or use fALFF across studies.

**Resolvable epochs only.** By default (`resolvable_only=True`), an epoch too short to hold a component (`docs/epochs.md`) is left out of that component's average. A condition with no usable epoch gives NaN. Pass `resolvable_only=False` to keep every epoch.

**Modes are matched by rank.** `m0` is each recording's lowest-frequency mode. Decompositions of different recordings need not share centre frequencies, so `m{k}:centre_hz` is included for checking alignment. Analyses that need identical bands across subjects should `reconstruct` fixed bands first (`docs/connectivity.md`).

## Your own features

Values you compute yourself (ratios between bands, contrasts between regions, anything from
numpy) can go to the same tests: build a `FeatureTable` from them.

```python
import numpy as np
import modewise

ratios = np.random.default_rng(0).standard_normal((8, 2))
table = modewise.FeatureTable(
    ratios, ["striatum", "cortex"], [f"sub-{i}" for i in range(8)],
    subjects=[str(i) for i in range(8)],
    metadata={"group": ["patient", "control"] * 4},
)
report = modewise.test_groups(table, "group", ("patient", "control"), n_permutations=1000)
print(report.feature_names, report.p_permutation.round(3))
```

Rows labelled with `conditions` work with `test_paired`; `participants=` joins a participants
table as `feature_table` does.

## Handover

| Target | How |
|---|---|
| Python statistics and scikit-learn | `table.values` (NumPy), `table.feature_names`, `table.metadata`, or `table.to_pandas()` |
| Parquet | `table.to_pandas().to_parquet(path)` (pandas writes it) |
| MATLAB, R, spreadsheets | `table.to_tsv(path)`, with missing values written as `n/a` (BIDS) |
| Deep learning on whole arrays (spectra, modes, matrices) | `recordings, array = results.stack(pipeline, output)` stacks one output across recordings, recording first. Shapes must match. |

Statistical tests run directly on the table (`docs/statistics.md`). Classifiers are not part of modewise: the table is the contract, and models live in the analysis code that uses it.
