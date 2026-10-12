# Connectivity

How modewise measures dependence between channels, for a whole run, per mode and per epoch.

## Functional and effective connectivity

These are different claims (Friston, 1994; Friston, 2011):

| | Functional connectivity (FC) | Effective connectivity (EC) |
|---|---|---|
| What it says | two signals are statistically dependent | one region influences another |
| Direction | none: the matrix is symmetric | A → B differs from B → A |
| Examples | correlation, coherence, phase locking value | dynamic causal modelling, Granger causality, transfer entropy |
| In modewise | `functional_connectivity` (this document) | not yet; a separate type and step when built |

An FC matrix shows **dependence**, not direction or cause. A high PLV or correlation between two regions can come from a common driver, from either influencing the other, or from shared noise.

EC on fMRI has caveats of its own. Haemodynamic responses differ between regions, so lag-based methods such as Granger causality on BOLD can point the wrong way (David et al., 2008; Smith et al., 2011).

## Measures

| Measure | Definition | Notes |
|---|---|---|
| `pearson` | Pearson correlation of each pair of channels over the window | A channel with zero variance gets NaN. |
| `pearson` with `fisher_z` | `atanh(r)`, which makes correlations approximately normal, so they can be averaged and tested (Silver & Dunlap, 1987; Corey, Dunlap & Burke, 1998) | The diagonal (r = 1) is NaN. An off-diagonal \|r\| = 1 gives ±∞, reported as is. |
| `plv` | phase locking value `|mean_t e^{i(φa − φb)}|` (Lachaux et al., 1999), used on fMRI by Glerean et al. (2012) | Needs narrow-band input: use it on modes, not on raw signals. Narrow-band noise already gives high values (about 0.84 in Honari et al., 2021), so compare a PLV against a null. |

**Phases come from the full run.** The analytic signal (mirror padding, `docs/time-frequency.md`) is computed on the whole recording, then cut to each epoch, so epoch edges add no Hilbert edge effects. This follows the same principle as `docs/epochs.md`: transform first, cut afterwards.

**Mode decomposition for phase synchrony.** Phase measures need narrow-band signals. Mode decompositions give narrow-band components without choosing a band-pass filter first. Honari & Lindquist (2022) compared EMD, bivariate EMD, NA-MEMD and MVMD for time-varying phase synchronisation in fMRI, and MVMD did best.

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

A window shorter than the slowest period in the signal gives spurious fluctuations in correlation (Leonardi & Van De Ville, 2015); Zalesky & Breakspear (2015) show how careful statistics can make shorter windows usable. The epochs' `resolvable` flags (`min_cycles`, default 1 cycle per window) apply this rule, and `functional_connectivity` carries them along, so matrices of components that do not fit their epoch can be left out.

## Band connectivity

The slow bands of Penttonen & Buzsáki (2003) and Buzsáki & Draguhn (2004), as used in fMRI by Zuo et al. (2010):
- slow-5: 0.010–0.027 Hz
- slow-4: 0.027–0.073 Hz
- slow-3: 0.073–0.198 Hz
- slow-2: 0.198–0.5 Hz

To get one matrix per band, sum the modes whose centre frequencies fall in the band (`reconstruct`), then run `functional_connectivity`. Summing modes stands in for the band-pass filtering used in slow-band studies.

## Planned

- Instantaneous phase synchrony, as cos Δφ(t) (Honari et al., 2021), together with dynamic FC.
- Coherence, wavelet coherence, and amplitude envelope correlation.

## References

- Buzsáki, G., & Draguhn, A. (2004). Neuronal oscillations in cortical networks. *Science*, 304(5679), 1926–1929. https://doi.org/10.1126/science.1099745
- Corey, D. M., Dunlap, W. P., & Burke, M. J. (1998). Averaging correlations: Expected values and bias in combined Pearson r and Fisher's z transformations. *Journal of General Psychology*, 125(3), 245–261. https://doi.org/10.1080/00221309809595548
- David, O., Guillemain, I., Saillet, S., et al. (2008). Identifying neural drivers with functional MRI: An electrophysiological validation. *PLoS Biology*, 6(12), e315. https://doi.org/10.1371/journal.pbio.0060315
- Friston, K. J. (1994). Functional and effective connectivity in neuroimaging: A synthesis. *Human Brain Mapping*, 2(1–2), 56–78. https://doi.org/10.1002/hbm.460020107
- Friston, K. J. (2011). Functional and effective connectivity: A review. *Brain Connectivity*, 1(1), 13–36. https://doi.org/10.1089/brain.2011.0008
- Glerean, E., Salmi, J., Lahnakoski, J. M., Jääskeläinen, I. P., & Sams, M. (2012). Functional magnetic resonance imaging phase synchronization as a measure of dynamic functional connectivity. *Brain Connectivity*, 2(2), 91–101. https://doi.org/10.1089/brain.2011.0068
- Honari, H., Choe, A. S., & Lindquist, M. A. (2021). Evaluating phase synchronization methods in fMRI: A comparison study and new approaches. *NeuroImage*, 228, 117704. https://doi.org/10.1016/j.neuroimage.2020.117704
- Honari, H., & Lindquist, M. A. (2022). Mode decomposition-based time-varying phase synchronization for fMRI. arXiv:2203.13955. https://arxiv.org/abs/2203.13955
- Lachaux, J.-P., Rodriguez, E., Martinerie, J., & Varela, F. J. (1999). Measuring phase synchrony in brain signals. *Human Brain Mapping*, 8(4), 194–208.
- Leonardi, N., & Van De Ville, D. (2015). On spurious and real fluctuations of dynamic functional connectivity during rest. *NeuroImage*, 104, 430–436. https://doi.org/10.1016/j.neuroimage.2014.09.007
- Penttonen, M., & Buzsáki, G. (2003). Natural logarithmic relationship between brain oscillators. *Thalamus & Related Systems*, 2(2), 145–152. https://doi.org/10.1016/S1472-9288(03)00007-4
- Silver, N. C., & Dunlap, W. P. (1987). Averaging correlation coefficients: Should Fisher's z transformation be used? *Journal of Applied Psychology*, 72(1), 146–148. https://doi.org/10.1037/0021-9010.72.1.146
- Smith, S. M., Miller, K. L., Salimi-Khorshidi, G., et al. (2011). Network modelling methods for FMRI. *NeuroImage*, 54(2), 875–891. https://doi.org/10.1016/j.neuroimage.2010.08.063
- Zalesky, A., & Breakspear, M. (2015). Towards a statistical test for functional connectivity dynamics. *NeuroImage*, 114, 466–470. https://doi.org/10.1016/j.neuroimage.2015.03.047
- Zuo, X.-N., Di Martino, A., Kelly, C., et al. (2010). The oscillating brain: Complex and reliable. *NeuroImage*, 49(2), 1432–1445. https://doi.org/10.1016/j.neuroimage.2009.09.037
