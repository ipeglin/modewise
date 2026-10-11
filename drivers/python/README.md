# modewise

Data-adaptive signal decomposition for multichannel recordings (fMRI first; EEG, MEG and fNIRS by
design), with a Rust core, this Python package and a command line.

- **Decompositions:** MVMD, noise-assisted MVMD, VMD, VLMD, EMD, EEMD, CEEMDAN, MEMD and the
  MODWT multiresolution analysis.
- **Analysis:** spectra and band amplitude (ALFF), Hilbert and Morlet time–frequency maps,
  functional connectivity, epochs with an explicit haemodynamic lag, regions from atlases, NIfTI
  parcellation, and permutation statistics across a cohort.
- **Pipelines** in TOML or Python run over whole datasets, cached by content and recorded.

```python
import numpy as np
import modewise

t = np.arange(600) / 1.25
x = np.vstack([np.sin(2 * np.pi * 0.02 * t) + np.sin(2 * np.pi * 0.2 * t)])
result = modewise.mvmd(x, 1.25, num_modes=2, alpha=2000.0, init="linear")
print(result.centre_frequencies_hz)
```

Documentation: https://ipeglin.github.io/modewise/ · Source: https://github.com/ipeglin/modewise ·
Licence: BSD-3-Clause.
