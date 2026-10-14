"""Regenerate nuisance regression fixtures with nilearn and, independently, numpy least squares.

nilearn: `signal.clean` with standardize=None, filter="cosine" for a high-pass, and the
confounds given. With detrend=True it removes a linear trend from signals and confounds before
projecting out the confounds, which equals one fit on [1, t, cosines, confounds]. With
detrend=False nilearn centres the confounds and keeps each channel's mean; modewise always
removes the mean, so the test compares against nilearn's output minus its mean.

numpy: `lstsq` on an explicit design whose cosines come from scipy's orthonormal inverse DCT.

Usage: uv run --no-project -p 3.12 --with nilearn==0.14.1 --with scipy python generate.py
"""

import json
from pathlib import Path

import nilearn
import numpy as np
from nilearn.signal import clean
from scipy.fft import idct

rng = np.random.default_rng(11)
n, t_r, high_pass = 200, 0.8, 0.01
t = np.arange(n)
motion = np.cumsum(rng.standard_normal((n, 4)), axis=0) * 0.05
confounds = np.column_stack([motion, 2 * motion[:, 0] - motion[:, 1]])  # last column collinear
signals = (
    np.column_stack([
        np.sin(2 * np.pi * 0.05 * t * t_r),
        np.cos(2 * np.pi * 0.08 * t * t_r),
        rng.standard_normal(n),
    ])
    + motion @ rng.standard_normal((4, 3))
    + 0.002 * t[:, None]
    + 5.0
)


def lstsq_residuals(detrend, cosines):
    columns = [np.ones(n)]
    if detrend:
        columns.append(t - t.mean())
    order = int(np.floor(2 * n * high_pass * t_r)) if cosines else 0
    columns += [idct(np.eye(n)[k], norm="ortho") for k in range(1, order + 1)]
    design = np.column_stack(columns + [confounds])
    beta, *_ = np.linalg.lstsq(design, signals, rcond=None)
    return signals - design @ beta


cases = []
for detrend, cosines in [(True, True), (True, False), (False, True)]:
    kwargs = {"filter": "cosine", "high_pass": high_pass} if cosines else {"filter": False}
    out = clean(signals, detrend=detrend, standardize=None, confounds=confounds, t_r=t_r, **kwargs)
    cases.append({
        "detrend": detrend,
        "high_pass_hz": high_pass if cosines else None,
        "nilearn": (out - out.mean(axis=0)).T.tolist(),
        "lstsq": lstsq_residuals(detrend, cosines).T.tolist(),
    })

fixture = {
    "source": f"nilearn {nilearn.__version__} signal.clean; numpy {np.__version__} lstsq",
    "sampling_rate_hz": 1 / t_r,
    "signals": signals.T.tolist(),
    "confounds": confounds.T.tolist(),
    "cases": cases,
}
Path(__file__).with_name("reference.json").write_text(json.dumps(fixture))
