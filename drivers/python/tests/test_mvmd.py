import json
from pathlib import Path

import numpy as np
import pytest

import modewise

FIXTURES = Path(__file__).parents[3] / "methods" / "mvmd" / "tests" / "fixtures"
INIT_NAMES = {0: "zero", 1: "linear", 2: "exponential"}


@pytest.mark.parametrize("name", ["two_tone", "bold_like"])
def test_matches_reference(name):
    fixture = json.loads((FIXTURES / f"{name}.json").read_text())
    params = dict(fixture["params"])
    params["init"] = INIT_NAMES[params["init"]]

    result = modewise.mvmd(np.array(fixture["signal"]), fixture["sampling_rate_hz"], **params)

    np.testing.assert_allclose(result.centre_frequencies_hz, fixture["centre_frequencies_hz"], atol=1e-4)
    expected = np.array(fixture["modes"])
    for ours, ref in zip(result.modes, expected):
        assert np.linalg.norm(ours - ref) / np.linalg.norm(ref) <= 1e-6


def test_result_metadata():
    x = np.random.default_rng(0).standard_normal((2, 200))
    result = modewise.mvmd(x, 1.25, 3, 1000.0, labels=["amy_l", "amy_r"])
    assert result.modes.shape == (3, 2, 200)
    assert result.n_modes == 3
    assert result.labels == ["amy_l", "amy_r"]
    assert result.sampling_rate == 1.25
    assert result.method == "mvmd"
    assert np.all(np.diff(result.centre_frequencies_hz) >= 0)


def test_accepts_lists_and_integers():
    result = modewise.mvmd([[1, 2, 3, 4, 3, 2, 1, 0]], 1.0, 2, 100.0)
    assert result.modes.dtype == np.float64


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(num_modes=0),
        dict(alpha=-1.0),
        dict(init="random"),
        dict(sampling_rate=0.0),
        dict(signal=np.array([[1.0, np.nan, 2.0]])),
        dict(labels=["a", "b"]),
    ],
)
def test_invalid_input_raises_value_error(kwargs):
    args = dict(signal=np.ones((1, 10)), sampling_rate=1.0, num_modes=2, alpha=100.0) | kwargs
    with pytest.raises(ValueError):
        modewise.mvmd(**args)
