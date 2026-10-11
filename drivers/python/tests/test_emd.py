import numpy as np
import pytest

import modewise


def tones(n, specs):
    t = np.arange(n) / 10.0
    return np.vstack([sum(a * np.sin(2 * np.pi * f * t) for a, f in spec) for spec in specs])


def test_emd_and_memd_sum_back_and_find_the_fast_tone():
    x = tones(600, [[(1.0, 1.5), (1.0, 0.2)], [(1.0, 1.5)], [(0.5, 1.5), (1.0, 0.2)]])
    d = modewise.emd(x, 10.0)
    np.testing.assert_allclose(d.modes.sum(axis=0), x, atol=1e-12)
    assert d.method == "emd" and not d.aligned_across_channels
    assert abs(d.channel_centre_frequencies_hz[-1, 0] - 1.5) < 0.05
    m = modewise.memd(x, 10.0, num_directions=16)
    np.testing.assert_allclose(m.modes.sum(axis=0), x, atol=1e-12)
    assert m.aligned_across_channels and m.channel_centre_frequencies_hz is None
    assert abs(m.centre_frequencies_hz[-1] - 1.5) < 0.1
    fixed = modewise.emd(x, 10.0, stop="fixed", siftings=5, max_imfs=1)
    assert fixed.n_modes == 2


def test_bad_settings_raise():
    x = tones(100, [[(1.0, 1.0)]])
    with pytest.raises(ValueError, match="stop must be"):
        modewise.emd(x, 10.0, stop="cauchy")
    with pytest.raises(ValueError, match="3 channels"):
        modewise.memd(x, 10.0)
    with pytest.raises(ValueError):
        modewise.emd(x, 10.0, thresholds=(0.0, 0.5, 0.05))


def test_eemd_and_ceemdan_are_seeded_and_ceemdan_sums_back():
    x = tones(400, [[(1.0, 1.5), (1.0, 0.2)], [(1.0, 0.7)]])
    c = modewise.ceemdan(x, 10.0, trials=10, seed=4)
    np.testing.assert_allclose(c.modes.sum(axis=0), x, atol=1e-12)
    np.testing.assert_array_equal(c.modes, modewise.ceemdan(x, 10.0, trials=10, seed=4).modes)
    e = modewise.eemd(x, 10.0, trials=10, max_imfs=3)
    assert e.method == "eemd" and e.n_modes <= 4
    assert not np.array_equal(e.modes, modewise.eemd(x, 10.0, trials=10, max_imfs=3, seed=1).modes)
    with pytest.raises(ValueError, match="trials"):
        modewise.eemd(x, 10.0, trials=0)
    with pytest.raises(ValueError, match="epsilon"):
        modewise.ceemdan(x, 10.0, epsilon=0.0)
