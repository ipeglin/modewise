# Thesis acceptance

modewise generalises the analysis pipeline of the author's master's thesis. This document checks how far modewise reproduces that pipeline. Where modewise deliberately differs, it gives the sourced reason.

The test is `tests/acceptance/test_thesis.py`. It is opt-in and runs on local data: the thesis's own outputs, which are not public in this form, and a ds005237 checkout (see the test's docstring). The cohort is the 14 subjects for which thesis outputs exist.

## What was rebuilt

| Thesis stage | modewise | Where |
|---|---|---|
| Parcellation (Schaefer 400, 17 networks, plus Tian scale II; 432 channels, z-scored) | the thesis's parcellated series as input, named by the same atlas label files | `[channels]` with `schaefer` and `tian` tables |
| Subject selection (SHAPS and TEPS anhedonia rules) | `examples/tcp/make_participants.py` | `[participants]` |
| MVMD (K = 10, α = 1000, exponential init, tolerance 1e-8, τ = 1e-3) | `mvmd` with the same settings | `examples/tcp/thesis.toml` |
| HHT (Hilbert spectrum of the modes) | `hilbert` → `hilbert_spectrum`, 0.01–0.25 Hz, 224 log bins | same |
| CWT (Morlet, 224 frequencies, 0.01–0.25 Hz) | `cwt` on the same range and count | same |
| Face and shape blocks | `epoch` | same |
| FC per block (Pearson, per mode) | `functional_connectivity` per mode and block | same |
| ROI selection (PFCv, PFCm, amygdala) | `select`, after whole-brain MVMD (the thesis default) | same |
| Group FC statistics (planned in the thesis code, never finished) | `test_paired`, `test_groups` | `docs/statistics.md` |
| DenseNet features and kNN / random forest | not in modewise: `FeatureTable` and `stack` hand over to ML code | `docs/features.md` |

## Where modewise reproduces the thesis exactly

| Check | Result |
|---|---|
| Whole-run MVMD on resting-state runs, all 14 subjects | identical modes: correlation 1.0 for every mode of every subject |
| Centre-frequency labels of those modes | within 0.00036 Hz. The modes are identical; the labels differ by one update step (which iteration's estimate is reported), far below the 0.0026 Hz frequency resolution of a 488-sample run. |
| Per-block MVMD on the thesis's own blocks | identical modes and centre frequencies |
| Pearson FC of the thesis's own blocks | identical (largest difference 1e-15) |
| Fisher z of those blocks | identical once the thesis's ±0.9999 clip is applied |
| Block count and onsets (hammerAP) | 18 and 18, same onsets |
| Subject grouping of the cohort | identical (7 anhedonic, 7 control) |

## Where modewise differs, and why

| Topic | Thesis | modewise | Reason |
|---|---|---|---|
| Decomposition scope | the stored task outputs decompose each block separately; the thesis code was later changed to decompose the run and slice | decompose the full run, then cut | `docs/epochs.md`: a 23-sample block cannot resolve slow-5 or slow-4. The thesis code's own later fix agrees. |
| Haemodynamic lag | none | explicit (`lag_s = 5.0` in the example; required) | `docs/epochs.md`, rule E3 |
| Block length | from onset to the next event, about 17.9 s (23–24 samples) | the merged events' own duration, 16.8–17.6 s | events.tsv durations; the thesis blocks include the following cue |
| Fisher z | r clipped to ±0.9999 first | no clip; diagonal undefined | `docs/connectivity.md`: no source supports the clip |
| Instantaneous frequency | backward phase difference | centred difference | `docs/time-frequency.md`: no half-sample lag |
| Envelope features | `log1p`, then divided by each channel's maximum | raw amplitude | feature scaling for a network, not part of the method (`docs/time-frequency.md`) |
| CWT | `scirs2` Morlet with scale = ω0/(2πf·dt), no Fourier factor; scalograms divided by their maximum | Torrence & Compo with the Fourier factor (λ ≈ 1.033), checked against pycwt; unscaled power | `docs/time-frequency.md`. Patterns agree only loosely (median channel correlation ≈ 0.36), because the frequency labels shift by about 3% and the scaling differs. |
| TEPS anhedonia rule | the missing-value codes −300, −900 and 99 averaged as answers | same in `thesis_group`; `corrected_group` treats them as unanswered | the NDA definitions file marks them as codes. Cohort-wide, the correction moves 8 subjects between groups; none of the 14 here. |

## Stored thesis outputs that could not be reproduced

- **Task HHT envelope and instantaneous frequency** do not follow from the stored block modes by the thesis's own formula. Checked:
  - the scipy-convention analytic signal (the thesis used `scirs2`'s copy of it)
  - the backward difference
  - `log1p` with per-channel max-divide
  - mode reordering
  - a full-run decomposition sliced to the block

  Correlations stay at 0.4–0.8. These groups lack the `decomposition_scope` attribute, which the thesis code itself treats as stale, so they were most likely written by a different code version from the stored modes.
- **Resting-state HHT** follows from the stored whole-run modes only approximately: envelope r ≥ 0.94 and frequency r ≥ 0.83 across subjects.

Neither affects modewise: its Hilbert transform is checked independently against scipy and the `emd` package (`docs/time-frequency.md`).

## Results on the cohort

From whole-run MVMD, ROI selection, blocks with a 5 s lag, and Pearson FC per mode (3780 edge features, 2646 of them resolvable in a block):

| Test | Result |
|---|---|
| Face vs shape, paired, 14 subjects; all 16384 sign flips enumerated (exact) | no edge survives FWER (smallest p = 0.73) or FDR (q < 0.05: none) |
| Anhedonic vs control, face blocks, 7 vs 7, 5000 permutations | no edge survives FWER (smallest p = 0.71) or FDR (none) |

With 14 subjects these tests have little power, so the null results say nothing about the thesis's classification findings, which used other features (DenseNet on spectra). They show that the full chain, from parcellated series to corrected group statistics, runs from one pipeline file and one dataset spec.

## Not reproduced

- **DenseNet feature extraction and kNN / random forest classification:** these sit outside modewise by design. The `FeatureTable` and `Results.stack` handovers provide their inputs.
- **Parcellation from volumes:** the thesis's parcellated series is the input. modewise can parcellate volumes (`docs/volumes.md`, matched against nilearn), but rebuilding from the fMRIPrep runs needs gigabytes of downloads and is not part of this test.
