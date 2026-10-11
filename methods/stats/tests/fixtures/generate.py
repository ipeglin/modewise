"""Regenerate statistics fixtures from independent implementations.

statsmodels OLS (t and p of one coefficient), scipy Welch t-test and incomplete beta,
statsmodels multipletests (BH, BY), an exact sign-flip enumeration in numpy, and bctpy's
get_components for network-based statistic components.

Usage: uv run --no-project -p 3.12 --with numpy --with scipy --with statsmodels --with bctpy python generate.py
"""

import itertools
import json
from pathlib import Path

import bct
import numpy as np
import statsmodels.api as sm
from scipy import special, stats
from statsmodels.stats.multitest import multipletests

rng = np.random.default_rng(8)
out = {}

# GLM: intercept, group dummy, centred age; 3 features, the second with a real group effect.
n = 24
group = np.repeat([1.0, 0.0], n // 2)
age = rng.normal(40, 10, n)
design = np.column_stack([np.ones(n), group, age - age.mean()])
y = rng.standard_normal((n, 3))
y[:, 1] += 1.2 * group
y[:, 2] += 0.05 * (age - age.mean())
fits = [sm.OLS(y[:, f], design).fit() for f in range(3)]
out["glm"] = {
    "design": design.tolist(),
    "y": y.tolist(),
    "t": [fit.tvalues[1] for fit in fits],
    "p_two_sided": [fit.pvalues[1] for fit in fits],
    "t_age": [fit.tvalues[2] for fit in fits],
}

# Welch: unequal sizes and variances.
a = rng.normal(0.5, 2.0, 9)
b = rng.normal(0.0, 0.7, 15)
welch = stats.ttest_ind(a, b, equal_var=False)
out["welch"] = {"y": np.concatenate([a, b]).tolist(), "in_a": [True] * 9 + [False] * 15,
                "t": float(welch.statistic), "p": float(welch.pvalue), "df": float(welch.df)}

# Incomplete beta and t tails.
cases = [(0.5, 0.5, 0.3), (2.0, 5.0, 0.2), (10.0, 0.5, 0.95), (30.0, 40.0, 0.4), (0.1, 3.0, 0.01)]
out["betainc"] = [{"a": a_, "b": b_, "x": x_, "value": float(special.betainc(a_, b_, x_))} for a_, b_, x_ in cases]
out["t_upper"] = [{"t": t_, "df": d_, "value": float(stats.t.sf(t_, d_))} for t_, d_ in [(2.1, 5), (-1.3, 12), (4.7, 3.5), (0.2, 100)]]

# FDR.
p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216])
out["fdr"] = {"p": p.tolist(),
              "bh": multipletests(p, method="fdr_bh")[1].tolist(),
              "by": multipletests(p, method="fdr_by")[1].tolist()}

# Exact sign flips: paired differences of 8 subjects, one-sample t on the mean, two-sided.
d = rng.normal(0.4, 1.0, (8, 2))


def t_one(x):
    return x.mean(0) / (x.std(0, ddof=1) / np.sqrt(len(x)))


obs = np.abs(t_one(d))
flips = np.array(list(itertools.product([1, -1], repeat=8)))
null = np.array([np.abs(t_one(d * s[:, None])) for s in flips])
out["sign_flip"] = {"y": d.tolist(),
                    "p": ((null >= obs - 1e-12).sum(0) / len(flips)).tolist(),
                    "p_fwer": [float((null.max(1) >= o - 1e-12).sum() / len(flips)) for o in obs]}

# Freedman–Lane with a nuisance covariate: test the intercept of [1, centred covariate] by
# flipping the signs of the covariate-only model's residuals, every one of 2^8 flips.
cov = rng.normal(0, 1, 8)
cov -= cov.mean()
fl_design = np.column_stack([np.ones(8), cov])
fl_y = rng.normal(0.5, 1.0, (8, 2)) + np.outer(cov, [0.8, -0.3])


def t_intercept(y):
    beta, *_ = np.linalg.lstsq(fl_design, y, rcond=None)
    resid = y - fl_design @ beta
    sigma2 = (resid**2).sum(0) / (8 - 2)
    return beta[0] / np.sqrt(sigma2 * np.linalg.inv(fl_design.T @ fl_design)[0, 0])


z = cov[:, None]
residual = fl_y - z @ np.linalg.lstsq(z, fl_y, rcond=None)[0]
fl_obs = t_intercept(fl_y)
fl_null = np.array([t_intercept(residual * s[:, None]) for s in flips])
out["freedman_lane"] = {
    "design": fl_design.tolist(), "y": fl_y.tolist(), "t": fl_obs.tolist(),
    "p_both": ((np.abs(fl_null) >= np.abs(fl_obs) - 1e-12).sum(0) / 256).tolist(),
    "p_greater": ((fl_null >= fl_obs - 1e-12).sum(0) / 256).tolist(),
}

# NBS components of a thresholded symmetric map.
nodes = 7
tmap = np.zeros((nodes, nodes))
for (i, j, v) in [(0, 1, 3.5), (1, 2, 4.1), (2, 0, 3.2), (3, 4, 3.9), (5, 6, 2.0), (4, 6, 1.1)]:
    tmap[i, j] = tmap[j, i] = v
adjacency = (tmap >= 3.0).astype(float)
membership, sizes = bct.get_components(adjacency)
components = []
for c in range(1, int(membership.max()) + 1):
    members = np.flatnonzero(membership == c)
    edges = [[int(i), int(j)] for i in members for j in members if i < j and adjacency[i, j]]
    if edges:
        components.append(sorted(edges))
out["nbs"] = {"t": tmap.tolist(), "threshold": 3.0, "components": sorted(components)}

(Path(__file__).parent / "reference.json").write_text(json.dumps(out, indent=1))
print("written", {k: len(v) if isinstance(v, (list, dict)) else v for k, v in out.items()})
