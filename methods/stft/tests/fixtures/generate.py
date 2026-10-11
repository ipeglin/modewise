"""Regenerate STFT fixtures from scipy.

Usage: uv run --no-project --with numpy --with scipy python generate.py
"""

import json
from pathlib import Path

import numpy as np
from scipy.signal import get_window, stft

fs, n, window_len, hop = 10.0, 200, 32, 8
t = np.arange(n) / fs
rng = np.random.default_rng(7)
x = np.vstack([np.sin(2 * np.pi * 1.25 * t), np.cos(2 * np.pi * 3.0 * t) + 0.2 * rng.standard_normal(n)])

freqs, times, z = stft(x, fs=fs, window="hann", nperseg=window_len, noverlap=window_len - hop,
                       boundary=None, padded=False, detrend=False, scaling="spectrum")
power = np.abs(z * get_window("hann", window_len).sum()) ** 2

fixture = dict(sampling_rate_hz=fs, window_len=window_len, hop=hop, signal=x.tolist(),
               frequencies_hz=freqs.tolist(), times_s=times.tolist(), power=power.tolist())
(Path(__file__).parent / "scipy_two_channel.json").write_text(json.dumps(fixture))
print("power shape", power.shape)
