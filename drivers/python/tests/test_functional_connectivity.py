import numpy as np
import pytest

import modewise

FS = 1.0


def signals():
    rng = np.random.default_rng(2)
    t = np.arange(300)
    slow = np.cos(2 * np.pi * 0.05 * t)
    return np.vstack([slow, -slow + 0.1 * rng.standard_normal(300), rng.standard_normal(300)])


def test_series_pearson_matches_numpy_and_fisher_z_is_atanh():
    x = signals()
    r = modewise.functional_connectivity(modewise.Series(x, FS))
    assert r.values.shape == (1, 1, 3, 3) and r.measure == "pearson"
    np.testing.assert_allclose(r.values[0, 0], np.corrcoef(x), atol=1e-12)
    z = modewise.functional_connectivity(modewise.Series(x, FS), fisher_z=True)
    off = ~np.eye(3, dtype=bool)
    with np.errstate(divide="ignore"):  # numpy's own arctanh(1) on the diagonal
        expected = np.arctanh(np.corrcoef(x))
    np.testing.assert_allclose(z.values[0, 0][off], expected[off], atol=1e-12)
    assert np.all(np.isnan(np.diag(z.values[0, 0]))) and z.measure == "pearson_fisher_z"


def test_epochs_of_modes_give_per_epoch_matrices_with_cycle_flags():
    modes = modewise.mvmd(signals(), FS, 2, 2000.0)
    events = modewise.Events([50.0, 150.0], [40.0, 40.0], ["a", "b"])
    spec = {
        "pipeline": {"name": "fc", "inputs": {"x": "series", "ev": "events"}},
        "step": [
            {"id": "modes", "op": "mvmd", "input": "x", "params": {"num_modes": 2, "alpha": 2000.0}},
            {"id": "blocks", "op": "epoch", "input": {"data": "modes", "events": "ev"}, "params": {"lag_s": 0.0}},
            {"id": "fc", "op": "functional_connectivity", "input": "blocks", "params": {"measure": "plv"}},
        ],
    }
    run = modewise.Pipeline.from_dict(spec).run({"x": modewise.Series(signals(), FS), "ev": events})
    fc = run.output("fc")
    assert fc.values.shape == (2, 2, 3, 3) and fc.measure == "plv"
    assert fc.window_labels == ["a", "b"] and list(fc.window_onsets_s) == [50.0, 150.0]
    assert fc.resolvable.shape == (2, 2) and fc.centre_frequencies_hz.shape == (2,)
    direct = modewise.functional_connectivity(run.output("blocks"), "plv")
    np.testing.assert_array_equal(direct.values, fc.values)
    assert modewise.functional_connectivity(modes).values.shape == (1, 2, 3, 3)


def test_connectivity_converts_to_xarray():
    xr = pytest.importorskip("xarray")
    array = modewise.to_xarray(modewise.functional_connectivity(modewise.Series(signals(), FS)))
    assert isinstance(array, xr.DataArray) and array.dims == ("window", "component", "channel_a", "channel_b")


def test_invalid_connectivity_requests():
    series = modewise.Series(signals(), FS)
    with pytest.raises(ValueError, match="pearson only"):
        modewise.functional_connectivity(series, "plv", fisher_z=True)
    with pytest.raises(ValueError, match="measure"):
        modewise.functional_connectivity(series, "coherence")
    with pytest.raises(TypeError):
        modewise.functional_connectivity(signals())
