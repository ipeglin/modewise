# Time–frequency maps

How modewise describes signals and modes over time and frequency: instantaneous amplitude and
frequency from the Hilbert transform, the Hilbert spectrum, and the Morlet continuous wavelet
transform.

## Instantaneous amplitude and frequency

`hilbert` turns each mode into its analytic signal z = x + i·H(x), computed by FFT: the negative
frequencies are set to zero and the positive ones doubled (Marple, 1999).

- **Amplitude** is |z| (Huang et al., 1998).
- **Frequency** is the rate of change of the unwrapped phase φ, by a centred difference:
  (φ[n+1] − φ[n−1]) · fs / 4π, with a straight-line fit through three points at the two ends. A
  centred difference has no time lag, and phase steps of one sample keep it valid up to Nyquist.
  Negative values are reported as they are.

Instantaneous frequency is meaningful for narrow-band signals with one component at a time
(Boashash, 1992), which is what the modes of a decomposition are meant to be.

**Edges.** The FFT treats a signal as periodic, so a jump between its last and first sample would
distort the analytic signal near both ends. By default (`padding = "mirror"`) modewise appends the
reversed signal before the FFT, which removes the jump; `padding = "none"` uses the signal as it is.
Instantaneous frequency is least reliable within about one period of either end, which is one more
reason to work on the full run and cut epochs afterwards ([epochs](epochs.md)).

## Hilbert spectrum

`hilbert_spectrum` puts each sample's power (amplitude², or amplitude with `weight = "amplitude"`)
into the frequency bin of its instantaneous frequency and sums over modes (Huang et al., 1998).
Bins are half-open, [low, high), on a linear or logarithmic grid; samples outside the grid are left
out.

```python
import numpy as np
import modewise

fs = 1.25
t = np.arange(500) / fs
x = np.vstack([np.sin(2 * np.pi * 0.03 * t) + 0.5 * np.sin(2 * np.pi * 0.15 * t)])
modes = modewise.mvmd(x, fs, num_modes=2, alpha=2000.0, init="linear", tau=0.0)
attributes = modewise.hilbert(modes)                       # amplitude and frequency per sample
print(np.median(attributes.frequency_hz[1, 0]))            # about 0.15 Hz
spectrum = modewise.hilbert_spectrum(attributes, 0.01, 0.25, 24, scale="log")
print(spectrum.power.shape)                                # (channel, frequency, sample)
```

## Morlet wavelets

`cwt` convolves each channel with complex Morlet wavelets, following Torrence & Compo (1998):

- **Wavelet:** a Gaussian-windowed complex exponential with wavenumber ω0 (6 by default),
  normalised to unit energy, applied by FFT. The signal is zero-padded to the next power of two.
- **Frequencies** are the geometric centres of log-spaced bins between `f_min` and `f_max`. Each
  frequency f is analysed at the scale whose equivalent Fourier period, 4πs / (ω0 + √(2 + ω0²)),
  equals 1/f.
- **Cone of influence:** near the edges, the wavelet reaches past the recording. Values within the
  e-folding time √2·s of either end are marked unreliable (`reliable`), and `epoch` treats a
  frequency as resolvable only where the whole epoch is outside the cone.
- **Rectification:** wavelet power is biased towards low frequencies; `rectify = true` divides it
  by the scale to remove that bias (Liu, Liang & Weisberg, 2007). It is off by default.

```python
scalogram = modewise.cwt(x, fs, 0.02, 0.25, 32)
peak = scalogram.frequencies_hz[np.argmax(scalogram.power[0].mean(axis=1))]
print(round(peak, 3), scalogram.reliable.mean())           # the strongest frequency; share inside the cone
```

## References

- Boashash, B. (1992). Estimating and interpreting the instantaneous frequency of a signal. *Proceedings of the IEEE*, 80(4), 520–538. https://doi.org/10.1109/5.135376
- Huang, N. E., et al. (1998). The empirical mode decomposition and the Hilbert spectrum for nonlinear and non-stationary time series analysis. *Proceedings of the Royal Society A*, 454, 903–995. https://doi.org/10.1098/rspa.1998.0193
- Liu, Y., Liang, X. S., & Weisberg, R. H. (2007). Rectification of the bias in the wavelet power spectrum. *Journal of Atmospheric and Oceanic Technology*, 24(12), 2093–2102. https://doi.org/10.1175/2007JTECHO511.1
- Marple, S. L. (1999). Computing the discrete-time "analytic" signal via FFT. *IEEE Transactions on Signal Processing*, 47(9), 2600–2603. https://doi.org/10.1109/78.782222
- Torrence, C., & Compo, G. P. (1998). A practical guide to wavelet analysis. *Bulletin of the American Meteorological Society*, 79(1), 61–78.
