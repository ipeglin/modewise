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

**Caching.** Each result is stored under a hash of the operation, its version, its parameters (with defaults filled in) and the keys of its inputs. Inputs are keyed by their content. Rerunning with a changed parameter recomputes that step and the steps after it, and nothing else. Every run records a manifest: parameters, data keys, what was cached, and timings.

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

`modewise ops` lists them with their ports. [reference/operations.md](reference/operations.md) is generated from the code and lists every parameter with its type and default; `modewise ops --json` gives the same as JSON Schema. Each method's evidence follows the policy in `TESTING.md`: several independent checks, no single source trusted alone.

### `mvmd`: multivariate variational mode decomposition

`signal` (series) → `modes` (decomposition)

**Parameters:** [`mvmd`](reference/operations.md#mvmd) in the generated reference.

**Method:** Rehman & Aftab (2019), *IEEE Transactions on Signal Processing*.

**Evidence:**
- matches the reference implementation [Dmocrito/mvmd](https://github.com/Dmocrito/mvmd) to better than 1e-12
- recovers the frequencies of synthetic tones
- reproduces the legacy thesis outputs

Defaults follow the reference implementation. The thesis used `alpha = 1000`, `init = "exponential"`, `tolerance = 1e-8` and `tau = 1e-3` (see `examples/tcp/blocks.toml`).

### `vmd`: variational mode decomposition, per channel

`signal` (series) → `modes` (decomposition). Parameters as `mvmd`.

**Method:** Dragomiretskiy & Zosso (2014). VMD is MVMD with one channel (Rehman & Aftab 2019), so modewise runs the MVMD solver on each channel on its own. Mode k is the k-th lowest frequency of its channel, so modes are **not aligned across channels**: `channel_centre_frequencies_hz` (mode × channel) gives each channel's frequency, and a mode's `centre_frequencies_hz` is the median over channels. Comparing mode k across channels (for example connectivity per mode) compares different bands; use `mvmd` when modes must line up.

**Evidence:**
- matches [vmdpy](https://github.com/vrcarva/vmdpy) per channel within 1e-6 at a fixed iteration count, on noise-free tones
- one known difference: vmdpy, like the original VMD.m, drops the Nyquist bin. With broadband noise the two then differ by about 1e-4 (relative), and a tone at Nyquist is lost in vmdpy but kept here.
- recovers each channel's own tones, including one at Nyquist

### `na_mvmd`: noise-assisted MVMD

`signal` (series) → `modes` (decomposition). Parameters as `mvmd`, plus:

**Parameters:** [`na_mvmd`](reference/operations.md#na_mvmd) in the generated reference.

**Method:** Zisou, Apostolidis & Hadjileontiadis (ICASSP 2021), following their `na_mvmd.m`: scale each channel to unit SD (ddof 1), append the noise channels, run MVMD, drop the noise channels and scale the modes back. The noise excites MVMD's filter-bank behaviour, so surplus modes settle on noise instead of splitting a real component; the choice of K matters less. The paper's centre-frequency update, the centroid of the generalised cross-spectrum, reduces for one snapshot to Σ_c Σ_f f|û_c|² / Σ_c Σ_f |û_c|², which is the pooled update the MVMD reference code already uses. Their repository has no licence, so modewise follows the paper and does not copy the code.

**Evidence:**
- equals `mvmd` on the scaled signal with the same noise channels, then rescaled
- with K above the number of tones, MVMD splits the tones across every mode, while NA-MVMD moves surplus modes to the noise

The thesis configuration had NA-MVMD as an option (one channel, SD 0.8) but never used it for results.

### `emd`, `eemd`, `ceemdan` and `memd`: empirical mode decomposition

`signal` (series) → `modes` (decomposition). `emd`, `eemd` and `ceemdan` decompose each channel on its own; `memd` decomposes all channels together (at least 3).

**Parameters:** [`emd`](reference/operations.md#emd), [`eemd`](reference/operations.md#eemd), [`ceemdan`](reference/operations.md#ceemdan), [`memd`](reference/operations.md#memd) in the generated reference.

**Method:** Huang et al. (1998); Wu & Huang (2009); Colominas, Schlotthauer & Torres (2014); Rehman & Mandic (2010). `docs/emd.md` gives the sources for each choice, the output layout (mode 0 is the residual; `emd` modes are aligned by IMF index from the top), and where modewise differs from memd.m.

**Evidence:**
- `emd` matches PyEMD within 1e-9 at a fixed number of siftings
- `memd` matches MEMD-Python within 1e-8
- `eemd` and `ceemdan` match the authors' procedures (eemd.m, iceemdan.m) on the same noise within 1e-9
- the envelope splines match scipy within 1e-9

### `modwt_mra`: wavelet multiresolution analysis

`signal` (series) → `modes` (decomposition), each channel on its own with the same filter bank (aligned).

**Parameters:** [`modwt_mra`](reference/operations.md#modwt_mra) in the generated reference.

**Method:** maximal overlap DWT, Percival & Walden (2000). Mode 0 is S_J and the last mode D₁. `docs/wavelets.md` gives the sources and the bands at common sampling rates.

**Evidence:**
- matches PyWavelets `mra(transform="swt")` within 1e-10 for five wavelets
- additive at any length, shift-invariant, energy-preserving

### `vlmd`: variational latent mode decomposition

`signal` (series) → `modes` (decomposition), `latent_modes` (decomposition), `loadings` (loadings). Refer to them as `step.modes`, `step.latent_modes` and `step.loadings`.

**Parameters:** [`vlmd`](reference/operations.md#vlmd) in the generated reference.

**Method:** Morante & Rehman (2025), arXiv:2505.17797. The channel modes are `Θ⁽ᵏ⁾A`: they sum to the latent fit, not exactly to the signal. `docs/vlmd.md` lists where the paper and its reference code disagree, and which one modewise follows.

**Evidence:**
- matches the reference implementation [Dmocrito/vlmd](https://github.com/Dmocrito/vlmd) within 1e-9, with its loading step solved exactly
- the loading step matches scikit-learn's `Lasso` within 1e-9
- recovers the frequencies of tones mixed across channels

### `stft`: short-time Fourier transform

`signal` (series) → `power` (time_frequency, |X|²)

**Parameters:** [`stft`](reference/operations.md#stft) in the generated reference.

Frames start at sample 0, and incomplete frames at the end are dropped. Each frame's time is the centre of its window.

**Evidence:** matches `scipy.signal.stft` to 1e-9 of peak power, and puts a pure tone's peak in the right bin.

### `spectrum`: power spectral density

`signal` (series) → `power` (time_frequency with one frame; power per Hz)

**Parameters:** [`spectrum`](reference/operations.md#spectrum) in the generated reference.

The defaults are a periodogram: scipy `periodogram`'s defaults, and the whole-run FFT used for ALFF (Zang et al. 2007). With `segment_s`, it is Welch's method; scipy `welch` uses a Hann window by default, so set `window = "hann"` to match. Density scaling is one-sided: every bin except DC and an even-length Nyquist counts twice.

**Evidence:** matches `scipy.signal.periodogram` and `scipy.signal.welch` (boxcar and Hann, constant, linear and no detrend, even and odd segments) within 1e-10 relative.

### `hilbert`: instantaneous amplitude and frequency

`modes` (decomposition) → `attributes` (instantaneous: amplitude and frequency, mode × channel × sample)

**Parameters:** [`hilbert`](reference/operations.md#hilbert) in the generated reference.

**Method:** amplitude is `|z|`. Frequency is the centred difference of the unwrapped phase.

**Evidence:**
- the analytic signal matches scipy to 1e-12
- amplitude and frequency match the `emd` package to 1e-9
- a chirp is tracked without lag
- no wrapping up to Nyquist

Conventions and sources are in `docs/time-frequency.md`.

### `hilbert_spectrum`: power by instantaneous frequency

`attributes` (instantaneous) → `power` (time_frequency, channel × frequency bin × sample)

**Parameters:** [`hilbert_spectrum`](reference/operations.md#hilbert_spectrum) in the generated reference.

Each sample's weight goes to the bin of its instantaneous frequency, summed over modes. Samples outside the grid are dropped.

**Evidence:** matches `emd.spectra.hilberthuang` to 1e-9 of the peak.

### `cwt`: continuous wavelet transform

`signal` (series) → `power` (time_frequency, channel × frequency × sample, with a cone-of-influence mask)

**Parameters:** [`cwt`](reference/operations.md#cwt) in the generated reference.

**Method:** complex Morlet after Torrence & Compo (1998). The signal is zero-padded to the next power of two. The cone of influence (e-folding time √2·s) marks values near the edges as unreliable, and `epoch` treats a frequency as resolvable only where the whole epoch is outside it.

**Evidence:** matches [pycwt](https://github.com/regeirk/pycwt) in power (1e-9 of peak), frequencies and cone-of-influence mask; recovers the frequency of a tone.

### `functional_connectivity`: undirected dependence between channels

`data` (series, decomposition, or epochs of either) → `matrices` (connectivity, window × component × channel × channel)

**Parameters:** [`functional_connectivity`](reference/operations.md#functional_connectivity) in the generated reference.

**Method and sources:** `docs/connectivity.md`. The matrices show statistical dependence, not direction or cause (effective connectivity is a different claim). PLV phases come from the full run before epochs are cut. Epoch results carry the epochs' `resolvable` flags. For one matrix per slow band, `reconstruct` the band's modes first.

**Evidence:**
- Pearson matches `numpy.corrcoef`, and PLV matches a direct scipy implementation, both within 1e-12, for whole runs and windows
- a quarter-cycle lag gives PLV ≈ 1 and r ≈ 0
- matrices are symmetric and bounded, with a unit diagonal (property tests)

### `select`: keep chosen regions

`data` (series, decomposition, instantaneous, time–frequency, epochs, functional connectivity, loadings or channels) and `channels` (channels) → `selected` (same kind as `data`)

**Parameters:** [`select`](reference/operations.md#select) in the generated reference.

Placed before a decomposition, it decomposes only the selected regions; placed after, it keeps their whole-brain modes. `docs/regions.md` compares the two.

### `region_average`: one signal or matrix per region

`data` (series, decomposition, time–frequency, epochs or functional connectivity) and `channels` → `averaged` (same kind as `data`)

**Parameters:** [`region_average`](reference/operations.md#region_average) in the generated reference.

Signals are averaged across the group's channels. Connectivity becomes group × group, with Pearson r averaged as Fisher z. Details are in `docs/regions.md`.

### `reconstruct`: sum chosen modes

`modes` (decomposition) → `signal` (series)

**Parameters:** [`reconstruct`](reference/operations.md#reconstruct) in the generated reference.

### `epoch`: cut a full-run result around events

`data` (series, decomposition, instantaneous, or a time-frequency map with one column per sample: CWT or Hilbert spectrum, not STFT) and `events` (events) → `epochs`

**Parameters:** [`epoch`](reference/operations.md#epoch) in the generated reference.

An epoch holds the samples whose times (`i / fs`) fall in `[onset + lag, end + lag)`. Windows outside the recording are dropped and listed with a reason, never padded. Rules, conventions and sources are in `docs/epochs.md`.
