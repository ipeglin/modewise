"""Regenerate Hilbert fixtures from two independent implementations.

scipy.signal.hilbert gives the analytic signal; the `emd` package (Quinn et al. 2021)
gives instantaneous amplitude, frequency and the Hilbert-Huang spectrum.

Usage: uv run --no-project -p 3.12 --with emd==0.8.1 --with scipy python generate.py
"""

import json
from pathlib import Path

import emd
import numpy as np
from scipy.signal import hilbert

FS = 50.0
rng = np.random.default_rng(42)


def modes(n):
    t = np.arange(n) / FS
    am_tone = (1.0 + 0.5 * np.sin(2 * np.pi * 0.4 * t)) * np.cos(2 * np.pi * 5.0 * t)
    chirp = np.cos(2 * np.pi * (2.0 * t + 0.6 * t**2))
    return np.vstack([am_tone, chirp])


fixtures = {}
for n in (400, 401):
    x = modes(n)
    analytic = hilbert(x, axis=1)
    # smooth_phase=None: emd by default median-filters the phase (zero-padded at the edges),
    # which changes the last samples; we compare the bare estimator.
    IP, IF, IA = emd.spectra.frequency_transform(x.T, FS, "hilbert", smooth_phase=None)
    edges = np.linspace(0.0, 15.0, 31)
    _, spec = emd.spectra.hilberthuang(IF, IA, edges=edges, sum_time=False, sum_imfs=True, mode="power")
    fixtures[str(n)] = {
        "fs": FS,
        "modes": x.tolist(),
        "analytic_real": analytic.real.tolist(),
        "analytic_imag": analytic.imag.tolist(),
        "amplitude": IA.T.tolist(),
        "frequency_hz": IF.T.tolist(),
        "edges": edges.tolist(),
        "spectrum": np.asarray(spec.todense() if hasattr(spec, "todense") else spec).tolist(),
    }

(Path(__file__).parent / "reference.json").write_text(json.dumps(fixtures))
print("emd", emd.__version__, "spectrum shape", np.asarray(fixtures["400"]["spectrum"]).shape)
