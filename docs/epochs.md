# Epochs and decompositions

How modewise cuts signals into epochs (task blocks, trials) when frequency components are involved.

## Why it matters

An epoch of length `T` seconds cannot show a component whose period is longer than `T`. Inside the epoch, such a component looks like a trend or an offset, not an oscillation, and its power in that epoch depends mostly on where in its cycle the epoch happens to fall (Leonardi & Van De Ville, 2015; Hindriks et al., 2016).

This applies to every modality. In EEG the periods are milliseconds and the epochs seconds, so it matters mainly for slow rhythms (delta, below 4 Hz) in short trials. In fMRI the periods are tens of seconds and blocks are short, so it matters almost everywhere below about 0.05 Hz. For example, in a task with 18 s blocks:

| Band (Zuo et al., 2010) | Range | Period | Fits in an 18 s block? |
|---|---|---|---|
| slow-5 | 0.010–0.027 Hz | 37–100 s | no |
| slow-4 | 0.027–0.073 Hz | 14–37 s | only above ~0.055 Hz |
| slow-3 | 0.073–0.198 Hz | 5–14 s | yes |

## How modewise epochs

**Decompose the full run, then cut.** Decompositions and transforms run on the whole continuous recording, and `epoch` cuts the results afterwards, as is common in EEG and MEG time–frequency analysis. Short windows lose about half a filter or wavelet length at each edge (Torrence & Compo, 1998); cutting afterwards keeps those edge effects at the ends of the run, not at every epoch boundary. Decomposers accept only full-run series, so a pipeline that decomposes epochs fails when it is planned. Values inside an epoch equal the matching samples of the full-run result exactly.

**The haemodynamic lag is explicit.** `epoch` shifts every event by `lag_s` seconds. The BOLD response peaks several seconds after a stimulus, and its timing varies across subjects and regions (Handwerker, Ollinger & D'Esposito, 2004), so there is no default: set `lag_s` (4–6 s is typical for fMRI; 0 for EEG) and record it with the results. The decomposition is cached, so rerunning with other lags (for example 0, 2, 4, 6 s) recomputes only `epoch` and the steps after it, which makes sensitivity cheap to report.

**Components that do not fit are flagged.** Each epoch records its length, and each mode or frequency bin gets a `resolvable` flag: it is resolvable when frequency × epoch length ≥ `min_cycles` (1 by default, the window rule of Leonardi & Van De Ville, 2015). For time–frequency maps with a cone of influence (`cwt`), the whole epoch must also lie outside the cone. Raise `min_cycles` for a stricter rule; Cohen (2019) discusses how many cycles a wavelet analysis needs. Feature extraction skips unresolvable epochs by default ([features](features.md)), and connectivity carries the flags along ([connectivity](connectivity.md)).

## Conventions

- **Time base:** sample `i` is at `i / fs` seconds, and event onsets are seconds from the first stored sample, as in BIDS `events.tsv` files (Gorgolewski et al., 2016).
- **Windows:** an epoch holds the samples whose times fall in `[start, end)`, so back-to-back blocks never share a sample. A window that starts between samples begins at the first sample at or after `onset + lag_s`.
- **Window length:** `window = "duration"` (the default) uses each event's own duration (blocks); `window = { tmin_s = -0.2, tmax_s = 0.5 }` uses a fixed window around each onset (trials), as MNE's `tmin` and `tmax` do (Gramfort et al., 2013).
- **Edges:** an epoch that reaches outside the recording, or holds fewer than two samples, is dropped and listed with a reason. Nothing is padded or mirrored.
- **Trials into blocks:** `merge_gap_s` merges consecutive events with the same label when the gap between them is at most that many seconds.
- **Missing values:** `events.tsv` rows with `n/a` onset or duration are rejected, not guessed.

```toml
[pipeline]
name = "blocks"
inputs = { bold = "series", trials = "events" }

[[step]]
id = "modes"
op = "mvmd"
input = "bold"
params = { num_modes = 10, alpha = 1000.0 }

[[step]]
id = "blocks"
op = "epoch"
input = { data = "modes", events = "trials" }
params = { lag_s = 5.0, merge_gap_s = 2.0, min_cycles = 1.0 }
```

## Machine learning

Epochs cut from one full-run decomposition share information, through the decomposition and through slow autocorrelation. When features from epochs go into a classifier, keep a whole run (or subject) on one side of each split, and prefer repeated random splits with nested cross-validation for tuning (Varoquaux et al., 2017).

## References

- Cohen, M. X. (2019). A better way to define and describe Morlet wavelets for time-frequency analysis. *NeuroImage*, 199, 81–86. https://doi.org/10.1016/j.neuroimage.2019.05.048
- Gorgolewski, K. J., Auer, T., Calhoun, V. D., et al. (2016). The brain imaging data structure, a format for organizing and describing outputs of neuroimaging experiments. *Scientific Data*, 3, 160044. https://doi.org/10.1038/sdata.2016.44
- Gramfort, A., Luessi, M., Larson, E., et al. (2013). MEG and EEG data analysis with MNE-Python. *Frontiers in Neuroscience*, 7, 267. https://doi.org/10.3389/fnins.2013.00267
- Handwerker, D. A., Ollinger, J. M., & D'Esposito, M. (2004). Variation of BOLD hemodynamic responses across subjects and brain regions and their effects on statistical analyses. *NeuroImage*, 21(4), 1639–1651. https://doi.org/10.1016/j.neuroimage.2003.11.029
- Hindriks, R., Adhikari, M. H., Murayama, Y., et al. (2016). Can sliding-window correlations reveal dynamic functional connectivity in resting-state fMRI? *NeuroImage*, 127, 242–256. https://doi.org/10.1016/j.neuroimage.2015.11.055
- Leonardi, N., & Van De Ville, D. (2015). On spurious and real fluctuations of dynamic functional connectivity during rest. *NeuroImage*, 104, 430–436. https://doi.org/10.1016/j.neuroimage.2014.09.007
- Torrence, C., & Compo, G. P. (1998). A practical guide to wavelet analysis. *Bulletin of the American Meteorological Society*, 79(1), 61–78.
- Varoquaux, G., Raamana, P. R., Engemann, D. A., et al. (2017). Assessing and tuning brain decoders: Cross-validation, caveats, and guidelines. *NeuroImage*, 145, 166–179. https://doi.org/10.1016/j.neuroimage.2016.10.038
- Zuo, X.-N., Di Martino, A., Kelly, C., et al. (2010). The oscillating brain: Complex and reliable. *NeuroImage*, 49(2), 1432–1445. https://doi.org/10.1016/j.neuroimage.2009.09.037
