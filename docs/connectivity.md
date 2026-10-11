# Connectivity

How modewise measures dependence between channels, for a whole run, per mode and per epoch, and how strong the evidence is for each choice. No single source is trusted on its own (see `TESTING.md`).

## Functional and effective connectivity

These are different claims (Friston 1994; Friston 2011, *Brain Connectivity* 1(1)):

| | Functional connectivity (FC) | Effective connectivity (EC) |
|---|---|---|
| What it says | two signals are statistically dependent | one region influences another |
| Direction | none: the matrix is symmetric | A → B differs from B → A |
| Examples | correlation, coherence, phase locking value | dynamic causal modelling, Granger causality, transfer entropy |
| In modewise | `functional_connectivity` (this document) | not yet; a separate type and step when built |

An FC matrix shows **dependence**, not direction or cause. A high PLV or correlation between two regions can come from a common driver, from either influencing the other, or from shared noise.

EC on fMRI has caveats of its own. Haemodynamic responses differ between regions, so lag-based methods such as Granger causality on BOLD can point the wrong way (David et al. 2008; Smith et al. 2011). Any EC method added later needs that discussion before it is offered.

## Measures

| Measure | Definition | Sources | Notes |
|---|---|---|---|
| `pearson` | Pearson correlation of each pair of channels over the window | universal; every FC toolbox; legacy thesis code | A channel with zero variance gets NaN. |
| `pearson` with `fisher_z` | `atanh(r)`, for averaging and statistics | Silver & Dunlap 1987; Corey, Dunlap & Burke 1998; Strube 1988. Schulze 2004 finds the bias difference small | The diagonal (r = 1) is NaN. An off-diagonal \|r\| = 1 gives ±∞, reported as is. |
| `plv` | phase locking value `|mean_t e^{i(φa − φb)}|` | Lachaux et al. 1999; Glerean et al. 2012 (fMRI); Honari et al. 2021, NeuroImage 228 | Needs narrow-band input: use it on modes, not on raw signals. Narrow-band null data already give high values (about 0.84 in Honari et al. 2021), so a PLV only means something against a null. |

**Phases come from the full run.** The analytic signal (mirror padding, `docs/time-frequency.md`) is computed on the whole recording, then cut to each epoch, so epoch edges add no Hilbert edge effects. This follows the same principle as `docs/epochs.md`: transform first, cut afterwards.

**Mode decomposition for phase synchrony.** Phase measures need narrow-band signals. Honari & Lindquist (2022, arXiv:2203.13955) compared EMD, bivariate EMD, NA-MEMD and MVMD for time-varying phase synchronisation in fMRI, and MVMD did best. That is a single study; band-pass filtering remains the common alternative.

## In Python

```python
import numpy as np
import modewise

fs = 1.25
t = np.arange(400) / fs
rng = np.random.default_rng(0)
shared = np.sin(2 * np.pi * 0.05 * t)
x = np.vstack([shared + 0.3 * rng.standard_normal(t.size) for _ in range(3)])
fc = modewise.functional_connectivity(modewise.Series(x, fs, labels=["a", "b", "c"]))
print(fc.values.shape, fc.values[0, 0, 0, 1].round(2))    # (window, component, ch, ch); r near 0.85
per_mode = modewise.functional_connectivity(modewise.mvmd(x, fs, num_modes=2, alpha=2000.0), "plv")
print(per_mode.values.shape)                               # one matrix per mode
```

## Shapes

Every result has the layout window × component × channel × channel:

| Input | Windows | Components |
|---|---|---|
| series | 1 | 1 |
| decomposition | 1 | one per mode (with its centre frequency) |
| epochs of a series | one per epoch | 1 |
| epochs of a decomposition | one per epoch | one per mode, with the epochs' `resolvable` flags |

## Short windows

A window shorter than the slowest period in the signal gives spurious fluctuations in correlation (Leonardi & Van De Ville 2015). Zalesky & Breakspear (2015) show that shorter windows can work with careful statistics. The epochs' `resolvable` flags (`min_cycles`, default 1 cycle per window) apply exactly this rule, and `functional_connectivity` carries them along. Treat matrices of unresolvable components as unreliable.

## Band connectivity

Slow bands are a consensus convention:
- slow-5: 0.010–0.027 Hz
- slow-4: 0.027–0.073 Hz
- slow-3: 0.073–0.198 Hz
- slow-2: 0.198–0.5 Hz

Sources: Penttonen & Buzsáki 2003; Buzsáki & Draguhn 2004; Zuo et al. 2010.

To get one matrix per band, sum the modes whose centre frequencies fall in the band (`reconstruct`), then run `functional_connectivity`. Summing modes stands in for the band-pass filtering used in slow-band studies.

## Not adopted from the thesis

- **Clipping r to ±0.9999 before Fisher z.** No source supports the clip. modewise reports ±∞ instead and leaves the diagonal undefined.
- **Averaging per-mode Fisher z within a band.** This answers a different question from band-pass connectivity, and only the thesis used it. It can be done in Python from the per-mode matrices and `centre_frequencies_hz`.
- **Connectivity of blocks decomposed separately.** Fixed by design: decompose the full run, then epoch (`docs/epochs.md`).

## Planned

- Instantaneous phase synchrony, as cos Δφ(t) (Honari et al. 2021 recommend it over PLV), together with dynamic FC.
- Coherence, wavelet coherence, and amplitude envelope correlation.
- Group statistics (S8): Welch t, permutation tests, FDR, and the network-based statistic.
