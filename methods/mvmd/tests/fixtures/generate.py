"""Regenerate MVMD fixtures from the reference implementation.

Usage: uv run --no-project --with numpy python generate.py <path to Dmocrito/mvmd clone>
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, sys.argv[1])
from mvmd import mvmd  # noqa: E402

HERE = Path(__file__).parent


def two_tone():
    fs, n = 256.0, 256
    t = np.arange(n) / n
    a = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 20 * t)
    b = 0.8 * np.sin(2 * np.pi * 5 * t) + 0.3 * np.sin(2 * np.pi * 20 * t)
    return np.vstack([a, b]), fs, dict(num_modes=2, alpha=2000.0, init=1, tolerance=1e-7, tau=0.0)


def bold_like():
    fs, n = 1.25, 301
    t = np.arange(n) / fs
    rng = np.random.default_rng(13369326)
    tones = [np.sin(2 * np.pi * f * t) for f in (0.02, 0.08, 0.2)]
    weights = [(1.0, 0.5, 0.2), (0.6, 1.0, 0.3), (0.3, 0.4, 1.0)]
    x = np.vstack([sum(w * s for w, s in zip(ws, tones)) for ws in weights])
    x += 0.1 * rng.standard_normal(x.shape)
    return x, fs, dict(num_modes=4, alpha=1000.0, init=2, tolerance=1e-6, tau=1e-2)


for name, case in [("two_tone", two_tone), ("bold_like", bold_like)]:
    x, fs, params = case()
    modes, _, omega = mvmd(x, sampling_rate=1, **params)
    fixture = dict(
        sampling_rate_hz=fs,
        params=params,
        signal=x.tolist(),
        modes=modes.tolist(),
        centre_frequencies_hz=(omega[-1] * fs).tolist(),
        iterations=int(omega.shape[0]),
    )
    (HERE / f"{name}.json").write_text(json.dumps(fixture))
    print(name, "iterations:", fixture["iterations"], "freqs:", fixture["centre_frequencies_hz"])
