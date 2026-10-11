import json
from pathlib import Path

import numpy as np
import pytest

import modewise

FIXTURE = Path(__file__).parents[3] / "methods" / "stft" / "tests" / "fixtures" / "scipy_two_channel.json"


def test_matches_scipy():
    f = json.loads(FIXTURE.read_text())
    result = modewise.stft(np.array(f["signal"]), f["sampling_rate_hz"], f["window_len"], f["hop"])
    np.testing.assert_allclose(result.frequencies_hz, f["frequencies_hz"])
    np.testing.assert_allclose(result.times_s, f["times_s"])
    expected = np.array(f["power"])
    np.testing.assert_allclose(result.power, expected, atol=1e-9 * expected.max())


def test_too_short_signal_raises_value_error():
    with pytest.raises(ValueError, match="too short"):
        modewise.stft(np.ones((1, 10)), 1.0, window_len=16, hop=4)
