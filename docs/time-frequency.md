# Instantaneous attributes and time-frequency maps

How modewise turns signals and modes into time-frequency descriptions, which conventions it follows, and how strong the evidence is for each. No single source is trusted on its own (see `TESTING.md`).

## Hilbert: instantaneous amplitude and frequency (S7a, implemented)

| Choice | Sources | Strength |
|---|---|---|
| Analytic signal by FFT: zero the negative frequencies, double the positive ones | [Marple 1999, IEEE TSP 47(9)](https://www.mathworks.com/help/signal/ref/hilbert.html) (as cited by MATLAB); [scipy.signal.hilbert](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.hilbert.html) | consensus; MATLAB and scipy implement the same method |
| Amplitude = `|z|` | Huang et al. 1998; the `emd` package (Quinn et al. 2021); legacy thesis code | consensus |
| Frequency = centred difference of the unwrapped phase, `(φ[n+1] − φ[n−1]) · fs / 4π`, using one-sample phase steps so it stays valid up to Nyquist. At the edges, the slope of a straight line through 3 points | [`emd`](https://joss.theoj.org/papers/10.21105/joss.02977) uses a 3-point Savitzky-Golay derivative of the unwrapped phase, which is the same estimator. By default `emd` also median-filters the phase over 5 samples, zero-padded at the edges, which changes the last samples; modewise does not smooth. The legacy code used a backward difference, which lags by half a sample | two implementations; the centred form has no time lag (tested with a chirp); the agreement check against `emd` caught a first version that wrapped above fs/4 |
| Instantaneous frequency is only meaningful for narrow-band, single-component signals (Bedrosian and Nuttall conditions) | Huang 2009 via secondary sources ([JADE, arXiv:2604.14185](https://arxiv.org/pdf/2604.14185); [noise and IF, arXiv:1008.4176](https://arxiv.org/pdf/1008.4176)) | consensus. Modes are near narrow-band by construction. Negative frequencies are reported, not clamped. |
| Hilbert spectrum: put each sample's power (amplitude²) in the frequency bin of its instantaneous frequency; bins are `[low, high)`; values outside the grid are dropped and counted | Huang et al. 1998; `emd.spectra.hilberthuang` (default `mode='power'`, half-open bins) | two implementations |

**Edges.** The FFT treats a signal as periodic, so a jump between the last and first sample distorts the analytic signal near both ends. By default (`padding = "mirror"`), modewise appends the reversed signal before the FFT, which removes the jump. A test shows this matches a split symmetric extension exactly, because the FFT method is unaffected by rotating the sequence. `padding = "none"` matches scipy exactly. Instantaneous frequency is least reliable within about one period of either end. That is one more reason to epoch only after working on the full run (`docs/epochs.md`).

**Not adopted from the thesis.** The legacy code applied `log1p` to amplitudes and divided by each channel's maximum. That is feature scaling for a neural network, not part of the method. No independent source makes it a default, so modewise leaves it to the ML step.

### In Python

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

## Wavelets (S7b, implemented)

| Choice | Sources | Strength |
|---|---|---|
| Complex Morlet with ω0 = 6, unit-energy normalisation, computed by FFT | [Torrence & Compo 1998](https://experts.colorado.edu/display/pubid_107382); legacy thesis code; MNE uses Morlet as well | consensus on Morlet; the ω0 and normalisation conventions differ between tools, so record which one is used |
| Cone of influence for Morlet: e-folding time √2·s | Torrence & Compo's reference IDL code ([wavelet.pro](https://themis.ssl.berkeley.edu/socware/tdas_2_00/idl/ssl_general/tools/tplot/wvlt/wavelet.pro)); [MathWorks](https://www.mathworks.com/help/wavelet/ug/boundary-effects-and-the-cone-of-influence.html); a pycwt-derived implementation | consensus |
| Equivalent Fourier period = 4πs / (ω0 + √(2 + ω0²)) | Torrence & Compo 1998, after Meyers et al. 1993; the same code | consensus |
| Divide power by scale to remove the bias towards low frequencies | [Liu, Liang & Weisberg 2007, JAOT 24(12)](https://new6.marine.usf.edu/Liu/Papers/Liu_etal_2007_JAOT_wavelet.pdf) | single source, widely cited. Offered as an option, not a default, until more evidence is gathered. |
| Wavelet width described by full width at half maximum rather than "cycles" | [Cohen 2019, NeuroImage 199](https://www.biorxiv.org/content/10.1101/397182.full.pdf) | single source; an option to consider |

**Implemented conventions** (all as in pycwt, which follows Torrence & Compo):
- frequencies are chosen by the user (geometric centres of log-spaced bins), and scales are `1/(λf)`
- the signal is zero-padded to the next power of two (pycwt does this on its scipy path, so the fixture is generated without pyFFTW)
- the Nyquist bin counts as a negative frequency (numpy `fftfreq`)
- the cone of influence uses the distance to the record boundary half a sample beyond the end samples: `λ/√2 · δt · (min(i, n−1−i) + 0.5)`

**Evidence:** power within 1e-9 of peak, identical frequencies, and an identical mask for an even-length and a power-of-two length signal. The comparison caught two mismatches before agreement: the Nyquist sign and the half-sample offset in the cone.

### Morlet CWT in Python

```python
scalogram = modewise.cwt(x, fs, 0.02, 0.25, 32)
peak = scalogram.frequencies_hz[np.argmax(scalogram.power[0].mean(axis=1))]
print(round(peak, 3), scalogram.reliable.mean())           # the strongest frequency; share inside the cone
```
