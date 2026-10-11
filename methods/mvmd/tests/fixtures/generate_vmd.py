"""Regenerate the VMD fixture from vmdpy (MIT), run per channel for a fixed 499 iterations.

Usage: uv run --no-project --with vmdpy --with numpy python generate_vmd.py
"""

import json
from pathlib import Path

import numpy as np
from vmdpy import VMD

HERE = Path(__file__).parent

fs, n = 2.0, 400
t = np.arange(n) / fs
# Noise-free on purpose: vmdpy, like the original VMD.m, drops the Nyquist bin, which our MVMD
# solver keeps. Broadband noise puts energy there and moves the results apart by about 1e-4.
x = np.vstack([
    np.cos(2 * np.pi * 0.1 * t) + 0.5 * np.cos(2 * np.pi * 0.42 * t) + 0.2 * np.cos(2 * np.pi * 0.7 * t),
    0.7 * np.cos(2 * np.pi * 0.05 * t) + np.cos(2 * np.pi * 0.3 * t + 1.0),
])

cases = []
for init, tau in ((1, 0.0), (0, 0.01)):
    modes, freqs = [], []
    for row in x:
        # tol = 0 runs vmdpy's full 499 updates, so the comparison does not depend on how each
        # code scales its stopping rule.
        u, _, omega = VMD(row, 2000.0, tau, 3, 0, init, 0.0)
        order = np.argsort(omega[-1])
        modes.append(u[order])
        freqs.append(omega[-1][order] * fs)
    cases.append(dict(
        params=dict(num_modes=3, alpha=2000.0, init=init, tau=tau, iterations=int(omega.shape[0])),
        modes=np.stack(modes, axis=1).tolist(),
        channel_centre_frequencies_hz=np.array(freqs).T.tolist(),
    ))
    print("init", init, "tau", tau, "iterations", omega.shape[0], "freqs", np.round(np.array(freqs), 4).tolist())

(HERE / "vmd.json").write_text(json.dumps(dict(sampling_rate_hz=fs, signal=x.tolist(), cases=cases)))
