"""Regenerate the EMD and MEMD fixtures.

Usage: uv run --no-project --with EMD-signal --with numpy --with scipy python generate.py <dir with MEMD_all.py>

- Splines: scipy CubicSpline (not-a-knot), as MATLAB spline.
- EMD: PyEMD (Apache-2.0) with a fixed number of siftings per IMF.
- MEMD: MEMD-Python (github.com/mariogrune/MEMD-Python-, no licence; run here, not copied),
  a port of Rehman and Mandic's memd.m.
"""

import json
import sys
from pathlib import Path

import numpy as np
from PyEMD import EMD
from scipy.interpolate import CubicSpline

sys.path.insert(0, sys.argv[1])
from MEMD_all import memd  # noqa: E402

HERE = Path(__file__).parent
rng = np.random.default_rng(20261011)

splines = []
for k in (2, 3, 4, 7, 20):
    gaps = rng.uniform(0.5, 3.0, k - 1)
    knots = np.concatenate([[-rng.uniform(0, 2)], -rng.uniform(0, 2) + np.cumsum(gaps)])
    knots[-1] = max(knots[-1], 30.0)
    knots = np.sort(knots)
    n = int(np.floor(knots[-1])) + 1
    values = rng.standard_normal((k, 2))
    splines.append(dict(knots=knots.tolist(), values=values.tolist(), n=n,
                        expected=CubicSpline(knots, values, bc_type="not-a-knot")(np.arange(n)).tolist()))


def tones(n, specs):
    t = np.arange(n)
    return np.vstack([sum(a * np.cos(2 * np.pi * f * t + p) for a, f, p in spec) for spec in specs])


x = tones(512, [[(1.0, 0.11, 0.0), (0.6, 0.03, 1.0), (0.3, 0.005, 0.0)],
                [(0.8, 0.2, 0.5), (1.0, 0.045, 0.0)]])
x[0] += 0.002 * np.arange(512)
emd = EMD(FIXE=10, nbsym=2, spline_kind="cubic", extrema_detection="simple")
emd_cases = []
for row in x:
    out = emd.emd(row.copy(), max_imf=3)
    emd_cases.append(dict(signal=row.tolist(), imfs=out[:-1].tolist(), residual=out[-1].tolist()))
    print("emd", len(out) - 1, "imfs")

memd_cases = []
for name, channels, args in (
    ("four_stop", 4, ()),
    ("three_stop", 3, ()),
    ("four_fix_h", 4, ("fix_h", 2)),
):
    specs = [[(1.0 - 0.15 * c, 0.12, 0.3 * c), (0.5 + 0.1 * c, 0.03, 0.0), (0.2, 0.007, c)] for c in range(channels)]
    y = tones(400, specs) + 0.05 * rng.standard_normal((channels, 400))
    q = np.asarray(memd(y.T.copy(), 64, *args)) if args else np.asarray(memd(y.T.copy(), 64))
    # q: modes × channels × samples, IMF 1 first, residual last.
    memd_cases.append(dict(name=name, signal=y.tolist(), stop=args[0] if args else "stop",
                           count=args[1] if args else None, modes=q.tolist()))
    print("memd", name, q.shape)

# Boundary mirroring: PyEMD's prepare_points_simple on the same extrema, for random integer
# signals (many ties between the end samples and the nearest extrema).
mirror = EMD(nbsym=2, extrema_detection="simple")
boundary_cases = []
while len(boundary_cases) < 300:
    n = int(rng.integers(8, 30))
    y = rng.integers(-3, 4, n).astype(float)
    t = np.arange(n, dtype=float)
    max_pos, max_val, min_pos, min_val, _ = mirror.find_extrema(t, y)
    if len(max_pos) + len(min_pos) < 3 or len(max_pos) == 0 or len(min_pos) == 0:
        continue
    try:
        tmax, tmin = mirror.prepare_points_simple(t, y, max_pos, max_val, min_pos, min_val)
        knots = dict(max_t=tmax[0].tolist(), max_z=tmax[1].tolist(), min_t=tmin[0].tolist(), min_z=tmin[1].tolist())
    except Exception:
        knots = None
    boundary_cases.append(dict(signal=y.tolist(), indmax=max_pos.astype(int).tolist(),
                               indmin=min_pos.astype(int).tolist(), knots=knots))

(HERE / "emd.json").write_text(json.dumps(dict(splines=splines, emd=emd_cases, memd=memd_cases, boundary=boundary_cases)))
print("boundary", sum(c["knots"] is None for c in boundary_cases), "raised")
