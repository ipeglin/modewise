# Wavelet multiresolution analysis

`modwt_mra` splits each channel into octave bands with the maximal overlap discrete wavelet transform (MODWT): details D₁…D_J and a smooth S_J. They add up to the signal, so they work like any other decomposition in modewise (Hilbert spectra, connectivity per mode, epochs).

```python
import numpy as np
import modewise

fs = 1.25
x = np.random.default_rng(0).standard_normal((2, 488)).cumsum(axis=1)   # a drifting series
mra = modewise.modwt_mra(x, fs, levels=5, boundary="reflection")
print(mra.n_modes, mra.centre_frequencies_hz.round(3))      # S5, D5, …, D1 from low to high
assert np.allclose(mra.modes.sum(axis=0), x)
```

## How it works

The maximal overlap discrete wavelet transform (Percival & Walden, 2000, ch. 5) filters the series
with a wavelet filter and a scaling filter at every level j, spacing the filter taps 2^{j−1}
samples apart instead of downsampling. Each detail D_j is then rebuilt from level j's wavelet
coefficients alone, and the smooth S_J from the last scaling coefficients, so the details and the
smooth add up exactly to the series (Percival & Mofjeld, 1997). Because nothing is downsampled, the
analysis works for any length and is shift-invariant: shifting the series shifts every band by the
same amount.

- **Wavelets:** haar, db1–db10, sym2–sym10 and coif1–coif5. The default `sym4` is the least
  asymmetric filter of length 8, LA(8) in Percival & Walden. The wavelet filter follows from the
  scaling filter by the quadrature mirror relation.
- **Boundary:** `periodic` treats the series as circular, as the transform is defined.
  `reflection` analyses the series followed by its reverse and keeps the first n samples (Percival
  & Walden, 2000). Use it for series that are not periodic, such as BOLD runs with a drift, where
  the periodic boundary would join the last sample to the first.
- **Depth:** by default the deepest level J whose equivalent filter, (2^J − 1)(L − 1) + 1
  samples for a filter of length L, still fits the series; at most ⌊log₂ n⌋.

## Output

Mode 0 is S_J, then D_J, …, D₁, so modes go from low to high frequency as in every modewise
decomposition. D_j covers roughly [fs/2^{j+1}, fs/2^j]: at a repetition time of 0.8 s (1.25 Hz),
D₁ is 0.31–0.63 Hz, D₃ 0.08–0.16 Hz and D₅ 0.02–0.04 Hz. Every channel uses the same filter bank,
so modes are aligned across channels. The centre frequency is the power-weighted spectral centroid,
pooled over channels.

The filter coefficients come from PyWavelets (MIT licence; see `THIRD_PARTY_NOTICES.md`).

## References

- Percival, D. B., & Mofjeld, H. O. (1997). Analysis of subtidal coastal sea level fluctuations using wavelets. *Journal of the American Statistical Association*, 92(439), 868–880. https://doi.org/10.2307/2965551
- Percival, D. B., & Walden, A. T. (2000). *Wavelet Methods for Time Series Analysis*. Cambridge University Press.
- Lee, G. R., Gommers, R., Waselewski, F., Wohlfahrt, K., & O'Leary, A. (2019). PyWavelets: A Python package for wavelet analysis. *Journal of Open Source Software*, 4(36), 1237. https://doi.org/10.21105/joss.01237
