"""Regenerate the Welch/periodogram fixture from scipy.signal.

Usage: uv run --no-project --with scipy --with numpy python generate_welch.py
"""

import json
from pathlib import Path

import numpy as np
from scipy import signal

HERE = Path(__file__).parent
rng = np.random.default_rng(11)
fs = 1.25
t = np.arange(301) / fs
x = np.vstack([np.sin(2 * np.pi * 0.05 * t) + 0.3 * rng.standard_normal(t.size) + 0.01 * t,
               rng.standard_normal(t.size)])
cases = []
f, p = signal.periodogram(x, fs, axis=-1)
cases.append(dict(name="periodogram", segment_len=None, overlap=None, window="boxcar", detrend="constant", frequencies=f.tolist(), power=p.tolist()))
for name, nperseg, noverlap, window, detrend in (
    ("welch_hann_linear", 100, None, "hann", "linear"),
    ("welch_boxcar_none", 64, 16, "boxcar", False),
    ("welch_odd", 75, 30, "hann", "constant"),
):
    f, p = signal.welch(x, fs, window=window, nperseg=nperseg, noverlap=noverlap, detrend=detrend, axis=-1)
    cases.append(dict(name=name, segment_len=nperseg, overlap=noverlap, window=window,
                      detrend=detrend or "none", frequencies=f.tolist(), power=p.tolist()))
(HERE / "welch.json").write_text(json.dumps(dict(sampling_rate_hz=fs, signal=x.tolist(), cases=cases)))
print(len(cases), "cases")
