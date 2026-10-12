# Pipelines

A pipeline is a list of named steps. Write it in TOML, or pass the same structure as a Python dict to `modewise.Pipeline.from_dict`. Mistakes are reported before anything runs: unknown operations, parameter typos, invalid values, references to missing steps, wrong data kinds, and cycles.

```toml
[pipeline]
name = "blocks"                                   # names the results folder
inputs = { bold = "series", trials = "events" }   # or a list of names: all series

[[step]]
id = "modes"            # any name without `.`
op = "mvmd"
input = "bold"          # one input: a name; several: a table of port = reference
params = { num_modes = 10, alpha = 1000.0, init = "exponential" }

[[step]]
id = "blocks"
op = "epoch"
input = { data = "modes", events = "trials" }
params = { lag_s = 5.0 }
```

**References** name a pipeline input, a step (when it has one output), or `step.output`.

**Caching.** Each result is stored under a hash of the operation, its version, its parameters (with defaults filled in) and the keys of its inputs. Inputs are keyed by their content but not stored, since they stay where they were read from. Rerunning with a changed parameter recomputes that step and the steps after it, and nothing else. Every run records a manifest: parameters, data keys, what was cached, and timings.

**Keeping outputs.** By default every step's outputs are stored. Add `keep = false` to a step whose outputs are only an intermediate, such as modes that a later step turns into connectivity or band powers. Its outputs then live in memory during the run and take no disk space; later steps still run and are cached as usual. The step is computed again only when a later step has to run, for example after its parameters change. A step with `keep = false` that no other step uses is refused, since its result would be lost, and reading its output afterwards says it was not stored.

```toml
[[step]]
id = "modes"
op = "modwt_mra"
input = "bold"
params = { levels = 6 }
keep = false              # hundreds of MB per cohort, needed only for the next step

[[step]]
id = "fc"
op = "functional_connectivity"
input = "modes"
params = { measure = "pearson" }
```

**Data kinds:** `series` (channels × samples), `decomposition` (modes × channels × samples), `instantaneous` (amplitude and frequency, modes × channels × samples), `time_frequency` (channels × frequencies × frames), `events`, `epochs`. Only `series` and `events` can be pipeline inputs.

## Operations

`modewise ops` lists the operations with their ports, and the [operation reference](reference/operations.md) lists every parameter with its type and default (`modewise ops --json` gives the same as JSON Schema). This section explains what each operation does.

### `mvmd`: multivariate variational mode decomposition

`signal` (series) → `modes` (decomposition). Parameters: [`mvmd`](reference/operations.md#mvmd).

MVMD splits all channels together into K modes, each a band-limited oscillation around a centre frequency shared by every channel (Rehman & Aftab, 2019). It solves for the modes and centre frequencies jointly with the alternating direction method of multipliers, working on the half spectrum of the mirror-extended signal. `alpha` sets how narrow the modes are, and `tau` how strictly they must add up to the signal (`0` lets noise stay out of the modes). Modes come out from low to high frequency, aligned across channels.

### `vmd`: variational mode decomposition, per channel

`signal` (series) → `modes` (decomposition). Parameters: [`vmd`](reference/operations.md#vmd).

VMD (Dragomiretskiy & Zosso, 2014) is the one-channel case of MVMD, so modewise runs the same solver on each channel on its own. Mode k is the k-th lowest frequency of its channel, so modes are **not aligned across channels**: `channel_centre_frequencies_hz` (mode × channel) holds each channel's frequencies, and a mode's `centre_frequencies_hz` is their median. Use `mvmd` when mode k must be the same band in every channel, for example for connectivity per mode.

### `na_mvmd`: noise-assisted MVMD

`signal` (series) → `modes` (decomposition). Parameters: [`na_mvmd`](reference/operations.md#na_mvmd).

Noise-assisted MVMD (Zisou, Apostolidis & Hadjileontiadis, 2021) scales each channel to unit standard deviation, appends channels of white noise, runs MVMD, then drops the noise channels and scales the modes back. The noise gives surplus modes somewhere to settle, so a K larger than the number of components no longer splits a component across several modes. The noise is seeded, so runs are reproducible.

### `emd`, `eemd`, `ceemdan` and `memd`: empirical mode decomposition

`signal` (series) → `modes` (decomposition). Parameters: [`emd`](reference/operations.md#emd), [`eemd`](reference/operations.md#eemd), [`ceemdan`](reference/operations.md#ceemdan), [`memd`](reference/operations.md#memd).

Empirical mode decomposition (Huang et al., 1998) sifts out oscillations from fast to slow until only a trend is left. `eemd` (Wu & Huang, 2009) and `ceemdan` (Colominas, Schlotthauer & Torres, 2014) add white noise to reduce mode mixing, and `memd` (Rehman & Mandic, 2010) sifts all channels together so every channel gets the same modes. Mode 0 is the residual. [EMD](emd.md) explains the sifting, the stop rules and the output layout.

### `modwt_mra`: wavelet multiresolution analysis

`signal` (series) → `modes` (decomposition). Parameters: [`modwt_mra`](reference/operations.md#modwt_mra).

The maximal overlap discrete wavelet transform (Percival & Walden, 2000) splits each channel into octave bands, details D₁…D_J and a smooth S_J, that add up to the signal. Every channel uses the same filter bank, so modes are aligned. [Wavelets](wavelets.md) lists the bands at common sampling rates.

### `vlmd`: variational latent mode decomposition

`signal` (series) → `modes` (decomposition), `latent_modes` (decomposition), `loadings` (loadings). Refer to them as `step.modes`, `step.latent_modes` and `step.loadings`. Parameters: [`vlmd`](reference/operations.md#vlmd).

VLMD (Morante & Rehman, 2025) describes many channels with a few latent signals, decomposes those into modes, and maps them back to the channels through sparse loadings. The channel modes add up to the latent fit rather than exactly to the signal. [VLMD](vlmd.md) has the details.

### `stft`: short-time Fourier transform

`signal` (series) → `power` (time_frequency, |X|²). Parameters: [`stft`](reference/operations.md#stft).

Power of a sliding periodic Hann window. Frames start at sample 0, incomplete frames at the end are dropped, and each frame's time is the centre of its window.

### `spectrum`: power spectral density

`signal` (series) → `power` (time_frequency with one frame, power per Hz). Parameters: [`spectrum`](reference/operations.md#spectrum).

By default a periodogram of the whole run with the mean removed, which is also the spectrum ALFF is computed from (Zang et al., 2007). With `segment_s` it is Welch's method (Welch, 1967): the average of overlapping, windowed segments, which trades frequency resolution for a steadier estimate. The density is one-sided: every bin except DC and an even-length Nyquist counts twice.

### `hilbert`: instantaneous amplitude and frequency

`modes` (decomposition) → `attributes` (instantaneous: amplitude and frequency, mode × channel × sample). Parameters: [`hilbert`](reference/operations.md#hilbert).

The analytic signal of each mode gives its instantaneous amplitude (`|z|`) and frequency (the centred difference of the unwrapped phase), as in Huang et al. (1998) and Boashash (1992). [Time–frequency maps](time-frequency.md) explains the conventions.

### `hilbert_spectrum`: power by instantaneous frequency

`attributes` (instantaneous) → `power` (time_frequency, channel × frequency bin × sample). Parameters: [`hilbert_spectrum`](reference/operations.md#hilbert_spectrum).

Each sample's power (or amplitude) goes to the bin of its instantaneous frequency, summed over modes: the Hilbert spectrum of Huang et al. (1998). Samples outside the grid are dropped.

### `cwt`: continuous wavelet transform

`signal` (series) → `power` (time_frequency, channel × frequency × sample, with a cone-of-influence mask). Parameters: [`cwt`](reference/operations.md#cwt).

Complex Morlet wavelets after Torrence & Compo (1998), on log-spaced frequencies. The signal is zero-padded to the next power of two. The cone of influence (e-folding time √2·s) marks values near the edges as unreliable, and `epoch` treats a frequency as resolvable only where the whole epoch is outside it.

### `functional_connectivity`: undirected dependence between channels

`data` (series, decomposition, or epochs of either) → `matrices` (connectivity, window × component × channel × channel). Parameters: [`functional_connectivity`](reference/operations.md#functional_connectivity).

Pearson correlation or the phase locking value (Lachaux et al., 1999), per mode and per epoch. The matrices show statistical dependence, not direction or cause. PLV phases come from the full run before epochs are cut. For one matrix per band, `reconstruct` the band's modes first. [Connectivity](connectivity.md) has the details.

### `select`: keep chosen regions

`data` (series, decomposition, instantaneous, time–frequency, epochs, functional connectivity, loadings or channels) and `channels` (channels) → `selected` (same kind as `data`). Parameters: [`select`](reference/operations.md#select).

Placed before a decomposition, it decomposes only the selected regions; placed after, it keeps their whole-brain modes. [Regions](regions.md) compares the two.

### `region_average`: one signal or matrix per region

`data` (series, decomposition, time–frequency, epochs or functional connectivity) and `channels` → `averaged` (same kind as `data`). Parameters: [`region_average`](reference/operations.md#region_average).

Signals are averaged across each group's channels. Connectivity becomes group × group, with Pearson r averaged as Fisher z.

### `reconstruct`: sum chosen modes

`modes` (decomposition) → `signal` (series). Parameters: [`reconstruct`](reference/operations.md#reconstruct).

Adds the chosen modes back into a series, for example to keep one band.

### `epoch`: cut a full-run result around events

`data` (series, decomposition, instantaneous, or a time–frequency map with one column per sample: CWT or Hilbert spectrum, not STFT) and `events` (events) → `epochs`. Parameters: [`epoch`](reference/operations.md#epoch).

An epoch holds the samples whose times (`i / fs`) fall in `[onset + lag, end + lag)`. Windows outside the recording are dropped and listed with a reason, never padded. [Epochs](epochs.md) explains the rules.

## References

- Boashash, B. (1992). Estimating and interpreting the instantaneous frequency of a signal. *Proceedings of the IEEE*, 80(4), 520–538. https://doi.org/10.1109/5.135376
- Colominas, M. A., Schlotthauer, G., & Torres, M. E. (2014). Improved complete ensemble EMD: A suitable tool for biomedical signal processing. *Biomedical Signal Processing and Control*, 14, 19–29. https://doi.org/10.1016/j.bspc.2014.06.009
- Dragomiretskiy, K., & Zosso, D. (2014). Variational mode decomposition. *IEEE Transactions on Signal Processing*, 62(3), 531–544. https://doi.org/10.1109/TSP.2013.2288675
- Huang, N. E., et al. (1998). The empirical mode decomposition and the Hilbert spectrum for nonlinear and non-stationary time series analysis. *Proceedings of the Royal Society A*, 454, 903–995. https://doi.org/10.1098/rspa.1998.0193
- Lachaux, J.-P., Rodriguez, E., Martinerie, J., & Varela, F. J. (1999). Measuring phase synchrony in brain signals. *Human Brain Mapping*, 8(4), 194–208.
- Morante, M., & Rehman, N. ur (2025). Variational latent mode decomposition. arXiv:2505.17797.
- Percival, D. B., & Walden, A. T. (2000). *Wavelet Methods for Time Series Analysis*. Cambridge University Press.
- Rehman, N. ur, & Aftab, H. (2019). Multivariate variational mode decomposition. *IEEE Transactions on Signal Processing*, 67(23), 6039–6052. https://doi.org/10.1109/TSP.2019.2951223
- Rehman, N., & Mandic, D. P. (2010). Multivariate empirical mode decomposition. *Proceedings of the Royal Society A*, 466(2117), 1291–1302. https://doi.org/10.1098/rspa.2009.0502
- Torrence, C., & Compo, G. P. (1998). A practical guide to wavelet analysis. *Bulletin of the American Meteorological Society*, 79(1), 61–78.
- Welch, P. D. (1967). The use of fast Fourier transform for the estimation of power spectra. *IEEE Transactions on Audio and Electroacoustics*, 15(2), 70–73. https://doi.org/10.1109/TAU.1967.1161901
- Wu, Z., & Huang, N. E. (2009). Ensemble empirical mode decomposition: A noise-assisted data analysis method. *Advances in Adaptive Data Analysis*, 1(1), 1–41. https://doi.org/10.1142/S1793536909000047
- Zang, Y.-F., et al. (2007). Altered baseline brain activity in children with ADHD revealed by resting-state functional MRI. *Brain and Development*, 29(2), 83–91. https://doi.org/10.1016/j.braindev.2006.07.002
- Zisou, C. A., Apostolidis, G. K., & Hadjileontiadis, L. J. (2021). Noise-assisted multivariate variational mode decomposition. In *Proceedings of IEEE ICASSP 2021*.
