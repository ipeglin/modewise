"""Regenerate the EEMD and CEEMDAN fixtures from given noise realisations.

Usage: uv run --no-project --with EMD-signal --with numpy python generate_ensemble.py

The inner EMD is PyEMD (Apache-2.0) with 10 siftings per IMF and its absolute stop thresholds
off, so it stops only when no oscillation is left, as modewise does.
- EEMD: Wu and Huang's eemd.m procedure, the mean over trials of each IMF index.
- CEEMDAN: a line-by-line numpy transcription of Colominas et al.'s iceemdan.m (SNRFlag 1).
"""

import json
from pathlib import Path

import numpy as np
from PyEMD import EMD

HERE = Path(__file__).parent
rng = np.random.default_rng(31)
emd = EMD(FIXE=10, range_thr=0.0, total_power_thr=0.0)


def run(y, max_imf=-1):
    emd.emd(y.copy(), max_imf=max_imf)
    imfs, residue = emd.get_imfs_and_residue()
    return [m for m in imfs], residue


n = 300
t = np.arange(n)
x = np.cos(2 * np.pi * 0.17 * t) + 0.8 * np.cos(2 * np.pi * 0.04 * t + 0.5) + 0.002 * t
noises = rng.standard_normal((20, n))

# EEMD
k = 4
scaled = 0.2 * np.std(x, ddof=1) * noises
trials = [run(x + w, k) for w in scaled]
count = max(len(imfs) for imfs, _ in trials)
eemd_imfs = [np.mean([imfs[j] if j < len(imfs) else np.zeros(n) for imfs, _ in trials], axis=0) for j in range(count)]
eemd_residual = np.mean([r for _, r in trials], axis=0)

# CEEMDAN (iceemdan.m)
eps = 0.2
sd = np.std(x, ddof=1)
xn = x / sd
noise_modes = []
for w in noises:
    imfs, residue = run(w)
    noise_modes.append(imfs + [residue])


def local_mean(y):
    imfs, residue = run(y, 1)
    return residue


def first_imf_or_all(y):
    imfs, residue = run(y, 1)
    return imfs[0] if imfs else residue


aux = np.zeros(n)
for e in noise_modes:
    xi = xn + eps * e[0] / np.std(e[0], ddof=1)
    aux += (xi - first_imf_or_all(xi)) / len(noise_modes)
modes = [xn - aux]
r = aux
kk = 1
while run(r, 1)[0]:
    aux = np.zeros(n)
    for e in noise_modes:
        if len(e) >= kk + 1:
            aux += local_mean(r + np.std(r, ddof=1) * eps * e[kk]) / len(noise_modes)
        else:
            aux += local_mean(r) / len(noise_modes)
    modes.append(r - aux)
    r = aux
    kk += 1

(HERE / "ensemble.json").write_text(json.dumps(dict(
    signal=x.tolist(), noises=noises.tolist(),
    eemd=dict(max_imfs=k, noise_sd=0.2, imfs=[m.tolist() for m in eemd_imfs], residual=eemd_residual.tolist()),
    ceemdan=dict(epsilon=eps, imfs=[(m * sd).tolist() for m in modes], residual=(r * sd).tolist()),
)))
print("eemd", len(eemd_imfs), "imfs; ceemdan", len(modes), "imfs; noise modes", sorted({len(e) for e in noise_modes}))
