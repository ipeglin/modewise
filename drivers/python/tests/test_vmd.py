import json
from pathlib import Path

import numpy as np
import pytest

import modewise

FIXTURE = Path(__file__).parents[3] / "methods" / "mvmd" / "tests" / "fixtures" / "vmd.json"


def test_vmd_matches_vmdpy_per_channel():
    f = json.loads(FIXTURE.read_text())
    case = f["cases"][0]
    p = case["params"]
    d = modewise.vmd(np.array(f["signal"]), f["sampling_rate_hz"], p["num_modes"], p["alpha"],
                     init="uniform", tau=p["tau"], tolerance=1e-300, max_iterations=p["iterations"] - 1)
    np.testing.assert_allclose(d.modes, case["modes"], atol=1e-6)
    np.testing.assert_allclose(d.channel_centre_frequencies_hz, case["channel_centre_frequencies_hz"], atol=1e-4)
    assert d.method == "vmd" and not d.aligned_across_channels
    np.testing.assert_allclose(d.centre_frequencies_hz, np.median(d.channel_centre_frequencies_hz, axis=1))


def test_na_mvmd_drops_noise_channels_and_is_seeded():
    t = np.arange(300)
    x = np.vstack([3 * np.cos(2 * np.pi * 0.05 * t), 0.2 * np.cos(2 * np.pi * 0.2 * t)])
    a = modewise.na_mvmd(x, 1.0, 2, 2000.0, init="uniform", tau=0.0)
    b = modewise.na_mvmd(x, 1.0, 2, 2000.0, init="uniform", tau=0.0)
    c = modewise.na_mvmd(x, 1.0, 2, 2000.0, init="uniform", tau=0.0, seed=1)
    assert a.modes.shape == (2, 2, 300) and a.aligned_across_channels and a.channel_centre_frequencies_hz is None
    np.testing.assert_array_equal(a.modes, b.modes)
    assert not np.array_equal(a.modes, c.modes)
    with pytest.raises(ValueError, match="noise_channels"):
        modewise.na_mvmd(x, 1.0, 2, 2000.0, noise_channels=0)


def test_external_per_channel_decomposition_round_trips():
    modes = np.zeros((2, 3, 10))
    per_channel = np.array([[0.1, 0.2, np.nan], [0.3, 0.4, 0.5]])
    d = modewise.Decomposition(modes, None, 1.0, channel_centre_frequencies_hz=per_channel)
    np.testing.assert_allclose(d.centre_frequencies_hz, [0.15, 0.4])
    assert not d.aligned_across_channels
    with pytest.raises(ValueError, match="exactly one"):
        modewise.Decomposition(modes, [0.1, 0.2], 1.0, channel_centre_frequencies_hz=per_channel)
    with pytest.raises(ValueError, match="exactly one"):
        modewise.Decomposition(modes, None, 1.0)
    with pytest.raises(ValueError, match="uniform"):
        modewise.vmd(np.ones((1, 10)), 1.0, 2, 10.0, init="random")
