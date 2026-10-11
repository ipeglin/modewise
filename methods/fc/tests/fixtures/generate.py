"""Regenerate connectivity fixtures with numpy and scipy, independently of modewise.

Pearson: numpy.corrcoef. Phase locking value: |mean exp(i(φa − φb))| with φ from
scipy.signal.hilbert applied to the signal followed by its reverse (modewise's mirror padding),
computed on the whole run and then cut to each window.

Usage: uv run --no-project -p 3.12 --with numpy --with scipy python generate.py
"""

import json
from pathlib import Path

import numpy as np
from scipy.signal import hilbert

rng = np.random.default_rng(5)
n = 200
t = np.arange(n)
base = np.cos(2 * np.pi * 0.05 * t)
x = np.vstack([
    base + 0.2 * rng.standard_normal(n),
    np.cos(2 * np.pi * 0.05 * t + 0.8) + 0.2 * rng.standard_normal(n),  # locked, phase-shifted
    -base + 0.3 * rng.standard_normal(n),                                 # anti-correlated
    np.cos(2 * np.pi * 0.13 * t) + 0.2 * rng.standard_normal(n),          # other frequency
    rng.standard_normal(n),                                               # noise
])
windows = [(10, 60), (100, 180)]


def phases(sig):
    z = hilbert(np.concatenate([sig, sig[:, ::-1]], axis=1), axis=1)[:, : sig.shape[1]]
    return np.exp(1j * np.angle(z))


def plv(u):
    return np.abs(u @ u.conj().T) / u.shape[1]


u = phases(x)
fixture = {
    "signal": x.tolist(),
    "windows": windows,
    "pearson": np.corrcoef(x).tolist(),
    "pearson_windows": [np.corrcoef(x[:, a:b]).tolist() for a, b in windows],
    "plv": plv(u).tolist(),
    "plv_windows": [plv(u[:, a:b]).tolist() for a, b in windows],
}
(Path(__file__).parent / "reference.json").write_text(json.dumps(fixture))
print(np.round(np.array(fixture["plv"]), 2))
