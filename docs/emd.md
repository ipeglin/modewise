# Empirical mode decomposition

modewise runs EMD, EEMD and CEEMDAN (per channel) and MEMD (multivariate) on one sifting engine. EMD is that engine with one channel and the single direction [1], so both share every choice below.

```python
import numpy as np
import modewise

t = np.arange(600) / 10.0
x = np.vstack([np.sin(2 * np.pi * 1.5 * t) + np.sin(2 * np.pi * 0.2 * t)])
imfs = modewise.emd(x, 10.0)
print(imfs.n_modes, imfs.channel_centre_frequencies_hz[-2:, 0].round(2))   # IMF 2 and IMF 1: about 0.2 and 1.5 Hz
assert np.allclose(imfs.modes.sum(axis=0), x)                               # mode 0 is the residual
```

## Choices and sources

| Choice | Sources | Strength |
|---|---|---|
| Sifting: subtract the mean of the upper and lower envelopes until a stop rule holds; the residual has fewer than 3 extrema | Huang et al. (1998); Rilling emd.m; memd.m; PyEMD | consensus |
| Envelopes: cubic splines with not-a-knot ends (two knots: a line; three: a parabola) | MATLAB `spline` (emd.m, memd.m); scipy `CubicSpline` (MEMD-Python, PyEMD for four or more knots) | consensus. PyEMD's three-knot spline is a natural spline instead; modewise follows MATLAB and scipy. |
| Ends: mirror the two nearest extrema about the end extremum or the end sample | Rilling, Flandrin & Gonçalvès (2003); emd.m and memd.m `boundary_conditions`; PyEMD "simple" | consensus |
| Extrema: sign changes of the first difference; a plateau counts once, at its middle | memd.m `local_peaks`; PyEMD "simple" (rounds the middle differently) | consensus; plateaus are rare in real-valued data |
| MEMD envelopes: project on Hammersley directions on the unit sphere, interpolate all channels at each projection's extrema, average over directions | Rehman & Mandic (2010); memd.m; MEMD-Python | consensus |
| Mode amplitude for the stop rule: ‖upper − lower‖ / 2, averaged over directions | memd.m; emd.m (one channel: \|upper − lower\| / 2) | consensus |

### Stop rules

| `stop` | Rule | Default | Sources |
|---|---|---|---|
| `rilling` | σ = ‖envelope mean‖ / amplitude; stop once σ < `sd` on all but a fraction `tol` of samples and σ < `sd2` everywhere | EMD (0.05, 0.5, 0.05) as emd.m; MEMD (0.075, 0.75, 0.075) as memd.m | Rilling et al. (2003); emd.m; memd.m |
| `s_number` | stop after `count` consecutive siftings in which extrema and zero crossings differ by at most one, in at least one direction | `count = 2`, as memd.m | Huang et al. (2003); memd.m `fix_h` |
| `fixed` | a fixed number of siftings per IMF | `siftings = 10`, as Wu & Huang | Wu & Huang (2009); PyEMD `FIXE` |

PyEMD's default stop (a mix of scaled variance, a sum of squared relative changes and an energy ratio) is not described in a paper, so modewise does not offer it. Sifting stops after 1000 iterations per IMF, as in the reference codes. A mode below 1e-10 of the signal's largest value ends the decomposition, as memd.m.

### Where modewise differs from a reference, and why

- **memd.m's flat-projection rule.** memd.m (and its Python port) gives a projection no extrema when every value is below 1e-5. That treats any wholly negative projection as flat, which happens for slow modes. Neither emd.m nor PyEMD does this. modewise does not either. With the rule on (hidden switch, for comparisons only), modewise matches MEMD-Python exactly; without it, the fast IMFs still match and the slow ones differ.
- **Errors in envelope building** (extrema that cannot be mirrored) stop sifting with a zero envelope mean, as memd.m's `try`/`catch` does. The Python port checks the first sample with a 1-based index (`lsym == 1`); modewise uses the first sample.

## Noise-assisted EMD

Both methods decompose each channel on its own, with noise from a seeded ChaCha8 stream per channel (`seed`), so runs are reproducible.

| Choice | Sources | Strength |
|---|---|---|
| **EEMD:** the mean over `trials` of EMDs of the signal plus white noise with SD `noise_sd`·SD(signal); 10 siftings per IMF; at most ⌊log₂ n⌋ − 1 IMFs; IMFs averaged by index | Wu & Huang (2009) and their eemd.m (noise 0.2, 100 trials) | consensus between paper and code. PyEMD's default noise is 0.05 × (max − min), a different scale; modewise follows the authors. |
| **CEEMDAN:** the improved form. Mode 1 is x − ⟨M(x + ε·E₁(w)/SD(E₁(w)))⟩; mode k is r_{k−1} − ⟨M(r_{k−1} + ε·SD(r_{k−1})·E_k(w))⟩, with M the local mean (what is left after one IMF) and E_k the k-th EMD mode of a noise realisation; stop when the residual has no IMF | Colominas, Schlotthauer & Torres (2014) and their iceemdan.m (SNRFlag 1, ε = 0.2) | consensus between paper and code |

Where references differ:

- **PyEMD's CEEMDAN** computes mode 1 as the mean of first IMFs (the original 2011 form, which keeps some noise in mode 1), scales every noise mode by the SD of the noise's first IMF, and defaults to ε = 0.005. modewise follows iceemdan.m.
- **Wu & Huang's eemd.m** always extracts ⌊log₂ n⌋ − 1 IMFs, even after the residual stops oscillating, and uses its own end treatment. modewise treats that number as a cap and uses the shared engine (Rilling mirroring).
- **iceemdan.m**, for mode 1, subtracts the signal's first IMF, or the whole signal when it has none; with noise added it always has one. modewise takes the local mean throughout, which is the same whenever an IMF exists.
- **iceemdan.m** uses a noise realisation's residual as its last "mode" when a stage needs more modes than the noise had; modewise does the same. A realisation without that many rows contributes M(r) without noise, also as iceemdan.m.

EEMD modes sum to the signal plus the mean added noise, which shrinks as 1/√trials. CEEMDAN modes sum to the signal exactly.

## Output layout

- **`emd`, `eemd` and `ceemdan`:** channels can give different numbers of IMFs, so modes are aligned by IMF index from the top: the last mode is every channel's IMF 1 (highest frequency), mode 0 is the residual, and a channel with fewer IMFs has zero modes just above its residual. IMF k of two channels need not be the same band, so the decomposition is not aligned across channels. Each channel keeps its own centre frequencies (NaN for a missing mode); a mode's frequency is their median.
- **`memd`:** every channel has the same IMFs, aligned across channels. MEMD needs at least 3 channels, as memd.m; use `emd` for fewer. Use at least twice as many directions as channels; modewise warns otherwise.
- **Centre frequency:** the power-weighted mean frequency of a mode's one-sided spectrum, pooled over channels for `memd` (the pooled centroid MVMD uses).
- Modes, residual included, sum to the signal exactly.

## Evidence

| Check | Result |
|---|---|
| Splines vs scipy `CubicSpline` (not-a-knot), 2 to 20 knots, uneven spacing | within 1e-9 |
| `emd` vs PyEMD with 10 siftings per IMF, 3 IMFs, two signals | within 1e-9 |
| `memd` vs MEMD-Python, 3 channels (the special direction set) and 4 channels, `rilling` and `s_number` | within 1e-8 with memd.m's flat-projection rule; fast IMFs within 1e-8 without it |
| Two tones in one channel | separated into IMF 1 and IMF 2 at the right frequencies; extrema and zero crossings agree within 2 |
| A tone shared by several channels in `memd` | in the same IMF of each channel, at the right amplitude |
| A ramp | no IMFs, all residual |
| `eemd` vs Wu & Huang's procedure on PyEMD's EMD (10 siftings), same 20 noise realisations | within 1e-9 |
| `ceemdan` vs a numpy transcription of iceemdan.m on PyEMD's EMD, same noise | within 1e-9; modes sum to the signal within 1e-12 |
| Huang's intermittency test (bursts of a fast tone on a slow one): best mode-to-slow-tone correlation | EMD 0.935, EEMD 0.996, CEEMDAN 0.999 |

memd.m (from Mandic's site) was read to confirm that the Python port follows it; MATLAB was not run.
