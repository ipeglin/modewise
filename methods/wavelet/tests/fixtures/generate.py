"""Regenerate the MODWT MRA fixture from PyWavelets `mra(transform="swt")`.

Usage: uv run --no-project --with PyWavelets --with numpy python generate.py

pywt's undecimated MRA (periodization, norm=True) is the MODWT MRA of Percival and Mofjeld
(1997); it needs a length divisible by 2^level.
"""

import json
from pathlib import Path

import numpy as np
import pywt

HERE = Path(__file__).parent
rng = np.random.default_rng(5)
n = 256
t = np.arange(n)
x = np.vstack([
    np.sin(2 * np.pi * 0.3 * t) + 0.7 * np.sin(2 * np.pi * 0.04 * t) + 0.01 * t,
    rng.standard_normal(n),
])
cases = []
for wavelet, level in (("haar", 5), ("db4", 4), ("sym4", 4), ("coif2", 3), ("db10", 2)):
    parts = [pywt.mra(row, wavelet, level=level, transform="swt") for row in x]
    # pywt order: [S_J, D_J, ..., D_1].
    cases.append(dict(wavelet=wavelet, level=level, modes=[[np.asarray(p[k]).tolist() for p in parts] for k in range(level + 1)]))
(HERE / "mra.json").write_text(json.dumps(dict(signal=x.tolist(), cases=cases)))
print(len(cases), "cases")
