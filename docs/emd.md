# Empirical mode decomposition

Empirical mode decomposition (EMD) splits a signal into intrinsic mode functions (IMFs):
oscillations that it sifts out one at a time, from the fastest to the slowest, until only a trend
is left (Huang et al., 1998). modewise runs EMD, EEMD and CEEMDAN on each channel, and MEMD on all
channels together, on one sifting engine.

```python
import numpy as np
import modewise

t = np.arange(600) / 10.0
x = np.vstack([np.sin(2 * np.pi * 1.5 * t) + np.sin(2 * np.pi * 0.2 * t)])
imfs = modewise.emd(x, 10.0)
print(imfs.n_modes, imfs.channel_centre_frequencies_hz[-2:, 0].round(2))   # IMF 2 and IMF 1: about 0.2 and 1.5 Hz
assert np.allclose(imfs.modes.sum(axis=0), x)                               # mode 0 is the residual
```

## Sifting

To find one IMF, modewise repeatedly subtracts the mean of the signal's upper and lower envelopes
until a stop rule holds. The IMF is then subtracted from the signal, and the next one is sifted
from what remains. The decomposition ends when the remainder has fewer than three extrema, or
becomes negligible (below 1e-10 of the signal's largest value).

- **Extrema** are where the first difference changes sign; a flat run counts once, at its middle.
- **Envelopes** are cubic splines through the maxima and through the minima, with not-a-knot ends.
- **Ends:** the two nearest extrema are mirrored about the end extremum or the end sample, so the
  envelopes reach the first and last samples without swinging wildly (Rilling, Flandrin &
  Gonçalvès, 2003).

## Stop rules

| `stop` | Sifting stops | Default |
|---|---|---|
| `rilling` | once the envelope mean, relative to the mode amplitude, is below `sd` on all but a fraction `tol` of samples and below `sd2` everywhere (Rilling, Flandrin & Gonçalvès, 2003) | `emd`, `ceemdan`: 0.05, 0.5, 0.05; `memd`: 0.075, 0.75, 0.075 |
| `s_number` | after `count` consecutive siftings in which the numbers of extrema and zero crossings differ by at most one (Huang et al., 2003) | `count = 2` |
| `fixed` | after a fixed number of siftings | `siftings = 10` (Wu & Huang, 2009); the default for `eemd` |

Sifting one IMF stops after 1000 iterations at most.

## Multivariate EMD

MEMD (Rehman & Mandic, 2010) sifts all channels together. It projects the multichannel signal on
many directions on the unit sphere (a Hammersley set), finds the extrema of each projection,
interpolates all channels at those points, and averages the envelopes over the directions. Every
channel then has the same IMFs, which makes IMF k comparable across channels. Use at least twice as
many directions as channels (`num_directions`, 64 by default); modewise warns otherwise. MEMD needs
at least three channels; use `emd` for fewer.

## Noise-assisted EMD

Plain EMD can mix scales: a burst of a fast rhythm drags parts of a slower one into the same IMF.
Adding white noise gives the sifting a uniform reference across scales.

- **EEMD** (Wu & Huang, 2009) averages the IMFs of many noisy copies of the signal: `trials` copies
  with white noise of standard deviation `noise_sd` times the signal's, 10 siftings per IMF, and at
  most ⌊log₂ n⌋ − 1 IMFs. The modes add up to the signal plus the average noise, which shrinks as
  1/√`trials`.
- **CEEMDAN** in its improved form (Colominas, Schlotthauer & Torres, 2014; after Torres et al.,
  2011) adds noise stage by stage. Each mode is the current residual minus the average local mean
  (what is left after one IMF) of that residual plus a scaled IMF of a noise realisation. The noise
  scale is `epsilon` times the residual's standard deviation. The modes add up to the signal exactly.

Both draw their noise from a seeded generator, one stream per channel (`seed`), so results are
reproducible.

## Output

- **`emd`, `eemd` and `ceemdan`:** channels can give different numbers of IMFs, so modes are
  aligned by IMF index from the top: the last mode is every channel's IMF 1 (the fastest), mode 0
  is the residual, and a channel with fewer IMFs has zero modes just above its residual. IMF k of
  two channels need not be the same band, so these decompositions are not aligned across channels.
  Each channel keeps its own centre frequencies (`channel_centre_frequencies_hz`, NaN for a missing
  mode), and a mode's frequency is their median.
- **`memd`:** every channel has the same IMFs, aligned across channels.
- **Centre frequency:** the power-weighted mean frequency of a mode's spectrum, pooled over channels
  for `memd`.
- The modes, residual included, add up to the signal (for `eemd`, up to the averaged noise).

## Comparing with other implementations

- PyEMD's `CEEMDAN` follows the original 2011 form for the first mode, scales its noise
  differently, and defaults to `epsilon = 0.005`; modewise follows Colominas et al.'s `iceemdan.m`,
  with `epsilon = 0.2`.
- PyEMD's `EEMD` scales its default noise by the signal's range (0.05 × (max − min)); modewise uses
  Wu & Huang's 0.2 times the standard deviation.
- `memd.m` and its Python port treat a projection as having no extrema when all its values are below
  1e-5, which affects slow modes; modewise does not.

## References

- Colominas, M. A., Schlotthauer, G., & Torres, M. E. (2014). Improved complete ensemble EMD: A suitable tool for biomedical signal processing. *Biomedical Signal Processing and Control*, 14, 19–29. https://doi.org/10.1016/j.bspc.2014.06.009
- Huang, N. E., et al. (1998). The empirical mode decomposition and the Hilbert spectrum for nonlinear and non-stationary time series analysis. *Proceedings of the Royal Society A*, 454, 903–995. https://doi.org/10.1098/rspa.1998.0193
- Huang, N. E., Wu, M.-L. C., Long, S. R., et al. (2003). A confidence limit for the empirical mode decomposition and Hilbert spectral analysis. *Proceedings of the Royal Society A*, 459, 2317–2345. https://doi.org/10.1098/rspa.2003.1123
- Rehman, N., & Mandic, D. P. (2010). Multivariate empirical mode decomposition. *Proceedings of the Royal Society A*, 466(2117), 1291–1302. https://doi.org/10.1098/rspa.2009.0502
- Rilling, G., Flandrin, P., & Gonçalvès, P. (2003). On empirical mode decomposition and its algorithms. In *IEEE-EURASIP Workshop on Nonlinear Signal and Image Processing (NSIP-03)*.
- Torres, M. E., Colominas, M. A., Schlotthauer, G., & Flandrin, P. (2011). A complete ensemble empirical mode decomposition with adaptive noise. In *Proceedings of IEEE ICASSP 2011*, 4144–4147. https://doi.org/10.1109/ICASSP.2011.5947265
- Wu, Z., & Huang, N. E. (2009). Ensemble empirical mode decomposition: A noise-assisted data analysis method. *Advances in Adaptive Data Analysis*, 1(1), 1–41. https://doi.org/10.1142/S1793536909000047
