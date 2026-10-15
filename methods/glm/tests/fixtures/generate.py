"""Reference values for modewise_glm.

Run: uv run --no-project -p 3.12 --with nilearn==0.14.1 --with statsmodels==0.15.0 python generate.py

1. Task regressors from nilearn's `compute_regressor` (hrf_model="spm") on a fine grid. nilearn
   builds them by discrete convolution, so they match the closed form only approximately; its
   impulses have the area of one fine-grid bin, so they are divided by that bin width here.
2. Betas and t for a fixed design from statsmodels: OLS, and GLS with the AR(1) covariance
   ρ^|i−j| (exact Prais–Winsten whitening), with ρ from nilearn's `_yule_walker` on the OLS
   residuals. nilearn's own `ARModel` with the same ρ is stored too: it leaves the first row
   unscaled, so it differs slightly.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from nilearn.glm.first_level.design_matrix import make_first_level_design_matrix
from nilearn.glm.first_level.first_level import _yule_walker
from nilearn.glm.first_level.hemodynamic_models import compute_regressor
from nilearn.glm.regression import ARModel
from scipy.linalg import toeplitz

OVERSAMPLING = 2000
T_R = 0.8
N = 150


def regressors() -> list[dict]:
    frame_times = np.arange(N) * T_R
    cases = []
    for name, onsets, durations in [
        ("blocks", [10.0, 50.3, 90.0], [18.0, 18.0, 6.5]),
        ("impulses", [5.0, 33.3, 61.7], [0.0, 0.0, 0.0]),
    ]:
        condition = np.array([onsets, durations, np.ones(len(onsets))])
        values, _ = compute_regressor(condition, "spm", frame_times, oversampling=OVERSAMPLING)
        values = values[:, 0]
        if name == "impulses":
            values = values / (T_R / OVERSAMPLING)
        cases.append({"name": name, "onsets": onsets, "durations": durations, "values": values.tolist()})
    return cases


def fit() -> dict:
    rng = np.random.default_rng(7)
    frame_times = np.arange(N) * T_R

    events = pd.DataFrame({
        "onset": [6.0, 30.0, 54.0, 78.0, 102.0, 18.0, 42.0, 66.0, 90.0],
        "duration": [8.0] * 9,
        "trial_type": ["face"] * 5 + ["shape"] * 4,
    })
    confounds = rng.normal(size=(N, 2))
    x = make_first_level_design_matrix(
        frame_times, events, hrf_model="spm", drift_model="cosine", high_pass=0.01,
        add_regs=confounds, add_reg_names=["conf1", "conf2"],
    )
    task = x[["face", "shape"]].to_numpy()
    nuisance = x.drop(columns=["face", "shape"]).to_numpy()
    design = np.column_stack([task, nuisance])

    n_channels, rho_true = 4, 0.4
    noise = np.zeros((N, n_channels))
    e = rng.normal(size=(N, n_channels))
    noise[0] = e[0]
    for t in range(1, N):
        noise[t] = rho_true * noise[t - 1] + e[t]
    beta = rng.normal(size=(design.shape[1], n_channels))
    y = design @ beta + noise

    ols = [sm.OLS(y[:, c], design).fit() for c in range(n_channels)]
    residuals = np.column_stack([r.resid for r in ols])
    rho = _yule_walker(residuals.T, 1)[:, 0]
    gls = [sm.GLS(y[:, c], design, sigma=toeplitz(rho[c] ** np.arange(N))).fit() for c in range(n_channels)]
    nilearn_ar = [ARModel(design, rho[c]).fit(y[:, c]) for c in range(n_channels)]
    take = lambda rows, attr: [[float(getattr(r, attr)[j]) for r in rows] for j in range(2)]
    return {
        "signal": y.T.tolist(),
        "task": task.T.tolist(),
        "nuisance": nuisance.tolist(),
        "ols_betas": take(ols, "params"),
        "ols_t": take(ols, "tvalues"),
        "df": float(ols[0].df_resid),
        "rho": rho.tolist(),
        "ar1_betas": take(gls, "params"),
        "ar1_t": take(gls, "tvalues"),
        "nilearn_ar1_betas": [[float(r.theta[j]) for r in nilearn_ar] for j in range(2)],
    }


if __name__ == "__main__":
    out = {"t_r": T_R, "n": N, "regressors": regressors(), "fit": fit()}
    (Path(__file__).parent / "reference.json").write_text(json.dumps(out))
