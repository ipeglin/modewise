"""Regenerate VLMD fixtures.

The reference VLMD (github.com/Dmocrito/vlmd) is GPL-3 and is not vendored. Clone it and pass
its folder. Its LASSO step is replaced by an exactly solved LASSO without intercept, the
paper's Eq. (10); see docs/vlmd.md for why.

Usage:
  git clone https://github.com/Dmocrito/vlmd /path/to/vlmd   # tested at d5cbd2b
  uv run --no-project -p 3.12 --with numpy --with scikit-learn python generate.py /path/to/vlmd
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from sklearn import linear_model

reference = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(reference))
import vlmd as ref  # noqa: E402

commit = subprocess.run(["git", "-C", str(reference), "rev-parse", "--short", "HEAD"],
                        capture_output=True, text=True).stdout.strip()


class ExactLasso(linear_model.Lasso):
    def __init__(self, alpha=1.0, warm_start=False, **_ignored):
        super().__init__(alpha=alpha, warm_start=warm_start, fit_intercept=False,
                         max_iter=1_000_000, tol=1e-14)


ref.lin_model.Lasso = ExactLasso
here = Path(__file__).parent

# LASSO alone: scikit-learn at a tight tolerance, several penalties.
rng = np.random.default_rng(11)
z = rng.standard_normal((4, 60))
z[3] = 0.6 * z[0] + 0.4 * rng.standard_normal(60)  # correlated latents
x = rng.standard_normal((3, 4)) @ z + 0.3 * rng.standard_normal((3, 60))
lasso_cases = []
for penalty in (0.0, 1e-3, 0.05, 0.3):
    model = linear_model.Lasso(alpha=penalty, fit_intercept=False, tol=1e-14, max_iter=1_000_000)
    model.fit(z.T, x.T)
    lasso_cases.append({"penalty": penalty, "loadings": model.coef_.T.tolist()})
(here / "lasso.json").write_text(json.dumps({"latents": z.tolist(), "signal": x.tolist(), "cases": lasso_cases}))


def lmd_signal(seed, n_channels, n_latents, freqs, n=300, noise=0.05):
    """X = Σ_k Θ⁽ᵏ⁾A with AM tones (cycles per sample) and a sparse A."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    theta = np.array([[(1 + 0.3 * np.sin(2 * np.pi * 0.004 * t + rng.uniform(0, 6)))
                       * rng.uniform(0.5, 1.5) * np.cos(2 * np.pi * f * t + rng.uniform(0, 6))
                       for _ in range(n_latents)] for f in freqs])
    a = rng.uniform(-1, 1, (n_latents, n_channels)) * (rng.uniform(size=(n_latents, n_channels)) > 0.5)
    a[np.arange(n_latents), np.arange(n_latents)] = 1.0
    return np.einsum("klt,lc->ct", theta, a) + noise * rng.standard_normal((n_channels, n))


cases = {}
for name, seed, channels, latents, freqs, params in [
    ("defaults", 1, 4, None, (0.02, 0.11, 0.3), {"num_modes": 3}),
    ("sparse", 2, 6, 3, (0.05, 0.2), {"num_modes": 2, "alpha": 500, "reg_lambda": 0.5,
                                       "reg_rho": 0.6, "tau": 0.5, "tolerance": 1e-6}),
    ("odd_length", 3, 3, 2, (0.07, 0.25), {"num_modes": 3, "alpha": 300, "max_iter": 40}),
]:
    n = 301 if name == "odd_length" else 300
    x = lmd_signal(seed, channels, latents or channels, freqs, n=n)
    kwargs = dict(params)
    latent_modes, loadings, omega, _ = ref.vlmd(x, kwargs.pop("num_modes"), latents, **kwargs)
    cases[name] = {
        "signal": x.tolist(),
        "num_latents": latents,
        "params": params,
        "iterations": len(omega),
        "centres": omega[-1].tolist(),
        "loadings": loadings.tolist(),
        "latent_modes": latent_modes.tolist(),
        "modes": np.einsum("klt,lc->kct", latent_modes, loadings).tolist(),
    }
    print(name, "iterations", len(omega), "centres", np.round(omega[-1], 4))
(here / "reference.json").write_text(json.dumps({"reference_commit": commit, "cases": cases}))
