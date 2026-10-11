# Operation reference

<!-- Generated from the operations' own schemas by `modewise ops --markdown`; do not edit.
Regenerate with `MODEWISE_UPDATE_DOCS=1 cargo test -p modewise_operations --test reference`. -->

Every step in a pipeline names one of these operations. Parameters go in the step's `params` table; anything not listed is rejected, and missing parameters take the defaults shown. `modewise ops --json` gives the same information as JSON Schema, for tools and agents.

## `ceemdan`

CEEMDAN, improved form: noise-assisted EMD whose modes sum to the signal (docs/emd.md).

- **Inputs:** `signal` (series)
- **Outputs:** `modes` (decomposition)

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `count` | integer | — | s_number: consecutive siftings with extrema and zero crossings within one (2 by default). |
| `epsilon` | number | `0.2` | ceemdan only: noise scale ε. |
| `max_imfs` | integer | — | Stop after this many IMFs (the rest is the residual); all by default, at most ⌊log₂ n⌋ − 1 for eemd. |
| `sd` | number | `0.05` | rilling: the envelope mean relative to the mode amplitude must stay below this on all but a fraction tol of samples. |
| `sd2` | number | `0.5` | rilling: the same ratio must stay below this everywhere (0.5; memd 0.75). |
| `seed` | integer | `0` | eemd and ceemdan: noise seed. |
| `siftings` | integer | — | fixed: siftings per IMF (10 by default). |
| `stop` | `rilling`, `s_number`, `fixed` | `"rilling"` | When sifting stops: `rilling` (thresholds sd, sd2, tol), `s_number` (count) or `fixed` (siftings); `fixed` for eemd, `rilling` otherwise. |
| `tol` | number | `0.05` | rilling: the fraction of samples allowed above sd (0.05; memd 0.075). |
| `trials` | integer | `100` | eemd and ceemdan: noise realisations. |

## `cwt`

Morlet continuous wavelet transform power on log-spaced frequencies, with a cone of influence (docs/time-frequency.md).

- **Inputs:** `signal` (series)
- **Outputs:** `power` (time_frequency)

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `f_max` | number | required | Highest frequency in Hz, at most Nyquist. |
| `f_min` | number | required | Lowest frequency in Hz, above 0. The transform analyses the geometric centres of log-spaced bins. |
| `n_bins` | integer | required | Number of log-spaced frequencies. |
| `omega0` | number | `6.0` | Morlet wavenumber (Torrence and Compo use 6). |
| `rectify` | boolean | `false` | Divide power by scale (Liu et al. 2007; single source, so off by default). |

## `eemd`

Ensemble EMD: the mean of EMDs of the signal plus seeded white noise (docs/emd.md).

- **Inputs:** `signal` (series)
- **Outputs:** `modes` (decomposition)

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `count` | integer | — | s_number: consecutive siftings with extrema and zero crossings within one (2 by default). |
| `max_imfs` | integer | — | Stop after this many IMFs (the rest is the residual); all by default, at most ⌊log₂ n⌋ − 1 for eemd. |
| `noise_sd` | number | `0.2` | eemd only: noise SD relative to the signal's. |
| `sd` | number | — | rilling: the envelope mean relative to the mode amplitude must stay below this on all but a fraction tol of samples. |
| `sd2` | number | — | rilling: the same ratio must stay below this everywhere (0.5; memd 0.75). |
| `seed` | integer | `0` | eemd and ceemdan: noise seed. |
| `siftings` | integer | `10` | fixed: siftings per IMF (10 by default). |
| `stop` | `rilling`, `s_number`, `fixed` | `"fixed"` | When sifting stops: `rilling` (thresholds sd, sd2, tol), `s_number` (count) or `fixed` (siftings); `fixed` for eemd, `rilling` otherwise. |
| `tol` | number | — | rilling: the fraction of samples allowed above sd (0.05; memd 0.075). |
| `trials` | integer | `100` | eemd and ceemdan: noise realisations. |

## `emd`

Empirical mode decomposition of each channel on its own (docs/emd.md).

- **Inputs:** `signal` (series)
- **Outputs:** `modes` (decomposition)

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `count` | integer | — | s_number: consecutive siftings with extrema and zero crossings within one (2 by default). |
| `max_imfs` | integer | — | Stop after this many IMFs (the rest is the residual); all by default, at most ⌊log₂ n⌋ − 1 for eemd. |
| `sd` | number | `0.05` | rilling: the envelope mean relative to the mode amplitude must stay below this on all but a fraction tol of samples. |
| `sd2` | number | `0.5` | rilling: the same ratio must stay below this everywhere (0.5; memd 0.75). |
| `siftings` | integer | — | fixed: siftings per IMF (10 by default). |
| `stop` | `rilling`, `s_number`, `fixed` | `"rilling"` | When sifting stops: `rilling` (thresholds sd, sd2, tol), `s_number` (count) or `fixed` (siftings); `fixed` for eemd, `rilling` otherwise. |
| `tol` | number | `0.05` | rilling: the fraction of samples allowed above sd (0.05; memd 0.075). |

## `epoch`

Cuts windows around events out of a full-run series, decomposition, instantaneous attributes or time–frequency map, with an explicit haemodynamic lag (docs/epochs.md).

- **Inputs:** `data` (series or decomposition or instantaneous or time_frequency), `events` (events)
- **Outputs:** `epochs` (epochs)

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `lag_s` | number | required | Required: shift added to event times (haemodynamic delay in fMRI; 0 for EEG). |
| `conditions` | list of string | — | Keep only these event labels; all by default. |
| `merge_gap_s` | number | — | Merge consecutive same-label events (trials) into blocks when gaps are at most this long. |
| `min_cycles` | number | `1.0` | A mode or frequency is resolvable in an epoch when frequency × epoch length ≥ this, and, for maps with a cone of influence, the whole epoch is outside it (provisional default; docs/epochs.md). |
| `window` | `duration` or table (`tmax_s`, `tmin_s`) | `"duration"` | `"duration"` uses each event's own length (blocks); `{ tmin_s = -0.2, tmax_s = 0.5 }` uses a fixed window around each onset (trials). |

## `functional_connectivity`

Functional connectivity per component (and per epoch): Pearson correlation or phase locking value (docs/connectivity.md).

- **Inputs:** `data` (series or decomposition or epochs)
- **Outputs:** `matrices` (functional_connectivity)

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `measure` | `pearson`, `plv` | required | `pearson` (correlation) or `plv` (phase locking value). |
| `fisher_z` | boolean | `false` | Return atanh(r) (pearson only); the diagonal becomes NaN. |

## `hilbert`

Instantaneous amplitude and frequency of every mode from its analytic signal (docs/time-frequency.md).

- **Inputs:** `modes` (decomposition)
- **Outputs:** `attributes` (instantaneous)

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `padding` | `mirror`, `none` | `"mirror"` | Extend the signal before the FFT (`mirror`); `none` matches scipy.signal.hilbert. |

## `hilbert_spectrum`

Hilbert spectrum: instantaneous amplitude or power of the modes binned on a frequency grid over time (docs/time-frequency.md).

- **Inputs:** `attributes` (instantaneous)
- **Outputs:** `power` (time_frequency)

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `f_max` | number | required | Highest edge of the frequency grid in Hz. |
| `f_min` | number | required | Lowest edge of the frequency grid in Hz (above 0 for a log grid). |
| `n_bins` | integer | required | Number of bins, each [low, high). |
| `scale` | `linear`, `log` | `"linear"` | `linear` or `log` bins (log bins report geometric centres). |
| `weight` | `power`, `amplitude` | `"power"` | `power` (amplitude²) or `amplitude`. |

## `memd`

Multivariate EMD: the same IMFs in every channel, from projections on many directions (docs/emd.md).

- **Inputs:** `signal` (series)
- **Outputs:** `modes` (decomposition)

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `count` | integer | — | s_number: consecutive siftings with extrema and zero crossings within one (2 by default). |
| `max_imfs` | integer | — | Stop after this many IMFs (the rest is the residual); all by default, at most ⌊log₂ n⌋ − 1 for eemd. |
| `num_directions` | integer | `64` | memd only: projection directions; use at least twice the channels. |
| `sd` | number | `0.075` | rilling: the envelope mean relative to the mode amplitude must stay below this on all but a fraction tol of samples. |
| `sd2` | number | `0.75` | rilling: the same ratio must stay below this everywhere (0.5; memd 0.75). |
| `siftings` | integer | — | fixed: siftings per IMF (10 by default). |
| `stop` | `rilling`, `s_number`, `fixed` | `"rilling"` | When sifting stops: `rilling` (thresholds sd, sd2, tol), `s_number` (count) or `fixed` (siftings); `fixed` for eemd, `rilling` otherwise. |
| `tol` | number | `0.075` | rilling: the fraction of samples allowed above sd (0.05; memd 0.075). |

## `modwt_mra`

MODWT multiresolution analysis: octave-band details and a smooth that sum to each channel (docs/wavelets.md).

- **Inputs:** `signal` (series)
- **Outputs:** `modes` (decomposition)

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `boundary` | `periodic`, `reflection` | `"periodic"` | `periodic` (as the MODWT is defined) or `reflection` (fewer edge effects for non-periodic series). |
| `levels` | integer | — | Deepest level; by default the deepest whose filter fits the series. |
| `wavelet` | string | `"sym4"` | sym4 is MATLAB's default and Percival and Walden's LA(8). |

## `mvmd`

Multivariate variational mode decomposition: K modes shared by all channels (Rehman and Aftab 2019).

- **Inputs:** `signal` (series)
- **Outputs:** `modes` (decomposition)

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `alpha` | number | required | Bandwidth penalty; larger means narrower modes. |
| `num_modes` | integer | required | Number of modes K. |
| `init` | `zero`, `linear`, `exponential`, `uniform` | `"zero"` | Starting centre frequencies: `zero`, `linear` (evenly over [0, 0.5], as Dmocrito/mvmd), `exponential`, or `uniform` ((0.5/K)·k, as the original MATLAB VMD and MVMD codes and vmdpy). |
| `max_iterations` | integer | `500` | Iteration cap. |
| `tau` | number | `0.01` | Dual ascent step; 0 lets noise stay out of the modes. |
| `tolerance` | number | `0.001` | Stop when the mean squared update falls below this. |

## `na_mvmd`

Noise-assisted MVMD: MVMD with white-noise channels appended, less sensitive to K (Zisou et al. 2021).

- **Inputs:** `signal` (series)
- **Outputs:** `modes` (decomposition)

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `alpha` | number | required | Bandwidth penalty; larger means narrower modes. |
| `num_modes` | integer | required | Number of modes K. |
| `init` | `zero`, `linear`, `exponential`, `uniform` | `"zero"` | Starting centre frequencies: `zero`, `linear` (evenly over [0, 0.5], as Dmocrito/mvmd), `exponential`, or `uniform` ((0.5/K)·k, as the original MATLAB VMD and MVMD codes and vmdpy). |
| `max_iterations` | integer | `500` | Iteration cap. |
| `noise_channels` | integer | `1` | na_mvmd only: white-noise channels appended before decomposing. |
| `noise_sd` | number | `0.8` | na_mvmd only: noise SD relative to the unit-SD signal channels. |
| `seed` | integer | `0` | na_mvmd only: seed for the noise. |
| `tau` | number | `0.01` | Dual ascent step; 0 lets noise stay out of the modes. |
| `tolerance` | number | `0.001` | Stop when the mean squared update falls below this. |

## `reconstruct`

Sums chosen modes of a decomposition back into a series, for example to keep one band.

- **Inputs:** `modes` (decomposition)
- **Outputs:** `signal` (series)

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `modes` | list of integer | required | Mode indices to sum, counted from the lowest frequency. |

## `region_average`

Averages channels by region, hemisphere or tag (functional connectivity in Fisher z) (docs/regions.md).

- **Inputs:** `data` (series or decomposition or time_frequency or epochs or functional_connectivity), `channels` (channels)
- **Outputs:** `averaged` (series or decomposition or time_frequency or epochs or functional_connectivity)

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `by` | `region`, `hemisphere`, `tag` | required | Group channels by `region`, `hemisphere` or `tag`. |
| `split_hemispheres` | boolean | `false` | Keep left and right apart: `PFCm_left`, `PFCm_right`. |
| `tag` | string | — | Tag key to group by when `by = "tag"`, for example `yeo17`. |

## `select`

Keeps the channels whose name, region, hemisphere or tags match the rules (docs/regions.md).

- **Inputs:** `data` (series or decomposition or instantaneous or time_frequency or epochs or functional_connectivity or loadings or channels), `channels` (channels)
- **Outputs:** `selected` (series or decomposition or instantaneous or time_frequency or epochs or functional_connectivity or loadings or channels)

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `rules` | list of table (`hemisphere`, `name`, `region`, `tags`) | required | Channels matching any rule are kept. A rule has any of `name`, `region`, `hemisphere` (`left`/`right`) and `tags` (a table); all of them must match. Text matches exactly, or with `*` wildcards. |

## `spectrum`

Power spectral density per channel as one frame: a periodogram, or Welch's method with segments (as scipy.signal).

- **Inputs:** `signal` (series)
- **Outputs:** `power` (time_frequency)

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `detrend` | `none`, `constant`, `linear` | `"constant"` | Per segment: `none`, `constant` (mean) or `linear`. |
| `overlap_s` | number | — | Overlap between segments in seconds; half a segment when absent. |
| `segment_s` | number | — | Welch segment length in seconds; the whole run when absent. |
| `window` | `boxcar`, `hann` | `"boxcar"` | `boxcar` or `hann` (periodic). |

## `stft`

Short-time Fourier transform power with a periodic Hann window.

- **Inputs:** `signal` (series)
- **Outputs:** `power` (time_frequency)

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `hop` | integer | required | Step between frames in samples. |
| `window_len` | integer | required | Window length in samples (periodic Hann). |

## `vlmd`

Variational latent mode decomposition: modes of a few latent signals and sparse channel loadings (docs/vlmd.md).

- **Inputs:** `signal` (series)
- **Outputs:** `modes` (decomposition), `latent_modes` (decomposition), `loadings` (loadings)

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `num_modes` | integer | required | Number of modes K. |
| `alpha` | number | `100.0` | Bandwidth penalty; larger means narrower modes. |
| `max_iterations` | integer | `1000` | Iteration cap. |
| `num_latents` | integer | — | Number of latent components L, at most the number of channels; one per channel by default. |
| `reg_lambda` | number | `0.01` | Sparsity of the loadings, on the reference code's scale. |
| `reg_rho` | number | `1.0` | ADMM penalty. |
| `tau` | number | `0.9` | Dual step, in [0, 1]. |
| `tolerance` | number | `0.001` | Stop when the summed squared relative change of the centre frequencies falls to this. |

## `vmd`

Variational mode decomposition of each channel on its own (Dragomiretskiy and Zosso 2014).

- **Inputs:** `signal` (series)
- **Outputs:** `modes` (decomposition)

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `alpha` | number | required | Bandwidth penalty; larger means narrower modes. |
| `num_modes` | integer | required | Number of modes K. |
| `init` | `zero`, `linear`, `exponential`, `uniform` | `"zero"` | Starting centre frequencies: `zero`, `linear` (evenly over [0, 0.5], as Dmocrito/mvmd), `exponential`, or `uniform` ((0.5/K)·k, as the original MATLAB VMD and MVMD codes and vmdpy). |
| `max_iterations` | integer | `500` | Iteration cap. |
| `tau` | number | `0.01` | Dual ascent step; 0 lets noise stay out of the modes. |
| `tolerance` | number | `0.001` | Stop when the mean squared update falls below this. |
