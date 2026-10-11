"""Regenerate CWT fixtures from pycwt, an independent implementation of Torrence & Compo (1998).

pycwt pads to the next power of two only on its scipy path (no pyfftw installed), which is
the convention modewise follows.

Usage: uv run --no-project -p 3.12 --with pycwt --with scipy python generate.py
"""

import json
from pathlib import Path

import numpy as np
import pycwt
from pycwt import helpers

assert getattr(helpers, "_FFT_NEXT_POW2", False), "pycwt must use its scipy path (uninstall pyfftw)"

FS = 10.0
rng = np.random.default_rng(3)
fixtures = {}
for n in (300, 256):
    t = np.arange(n) / FS
    x = np.sin(2 * np.pi * 0.8 * t) + 0.5 * np.sin(2 * np.pi * 2.5 * t) + 0.2 * rng.standard_normal(n)
    # Log grid 0.2..4 Hz, 24 bins; modewise analyses the bin centres.
    edges = 0.2 * (4.0 / 0.2) ** (np.arange(25) / 24)
    centres = np.sqrt(edges[:-1] * edges[1:])
    wave, scales, freqs, coi, _, _ = pycwt.cwt(x, 1 / FS, wavelet=pycwt.Morlet(6), freqs=centres)
    power = np.abs(wave) ** 2
    fixtures[str(n)] = {
        "fs": FS,
        "signal": x.tolist(),
        "centres": centres.tolist(),
        "power": power.tolist(),
        "scales": scales.tolist(),
        "reliable": (1 / freqs[:, None] <= coi[None, :]).astype(int).tolist(),
    }
(Path(__file__).parent / "pycwt.json").write_text(json.dumps(fixtures))
print("power shape", np.asarray(fixtures["300"]["power"]).shape)
