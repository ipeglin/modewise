import numpy as np
import pytest

import modewise

FS = 1.25


def mixed():
    t = np.arange(500) / FS
    slow, fast = np.cos(2 * np.pi * 0.05 * t), np.sin(2 * np.pi * 0.25 * t)
    return np.vstack([slow, 0.5 * slow + 0.5 * fast, fast])


def test_vlmd_returns_channel_modes_latent_modes_and_loadings():
    modes, latent, loadings = modewise.vlmd(
        mixed(), FS, 2, num_latents=2, alpha=2000.0, tolerance=1e-6, labels=["a", "b", "c"]
    )
    assert modes.modes.shape == (2, 3, 500) and latent.modes.shape == (2, 2, 500)
    assert np.allclose(modes.centre_frequencies_hz, [0.05, 0.25], atol=0.005)
    assert modes.method == latent.method == "vlmd"
    assert loadings.values.shape == (2, 3) and np.all(np.abs(loadings.values) <= 1)
    assert loadings.channel_labels == ["a", "b", "c"] and loadings.latent_labels == ["latent_0", "latent_1"]
    rebuilt = np.einsum("klt,lc->kct", latent.modes, loadings.values)
    assert np.allclose(rebuilt, modes.modes)


def test_vlmd_in_a_pipeline_exposes_every_output():
    spec = {
        "pipeline": {"name": "latent", "inputs": {"x": "series"}},
        "step": [{"id": "split", "op": "vlmd", "input": "x", "params": {"num_modes": 2, "num_latents": 2}}],
    }
    run = modewise.Pipeline.from_dict(spec).run({"x": modewise.Series(mixed(), FS)})
    assert isinstance(run.output("split.loadings"), modewise.Loadings)
    assert isinstance(run.output("split.latent_modes"), modewise.Decomposition)


def test_loadings_convert_to_xarray():
    xr = pytest.importorskip("xarray")
    _, _, loadings = modewise.vlmd(mixed(), FS, 1, num_latents=2)
    array = modewise.to_xarray(loadings)
    assert isinstance(array, xr.DataArray) and array.dims == ("latent", "channel")


def test_invalid_vlmd_settings():
    for kwargs in ({"num_latents": 4}, {"tau": 1.5}, {"reg_rho": 0.0}):
        with pytest.raises(ValueError):
            modewise.vlmd(mixed(), FS, 2, **kwargs)
