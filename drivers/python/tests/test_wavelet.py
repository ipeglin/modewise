import numpy as np
import pytest

import modewise


def test_modwt_mra_is_additive_and_matches_pywavelets_layout():
    t = np.arange(256)
    x = np.vstack([np.sin(0.4 * t) + 0.01 * t, np.cos(0.1 * t)])
    d = modewise.modwt_mra(x, 1.0, wavelet="db4", levels=3)
    assert d.modes.shape == (4, 2, 256) and d.method == "modwt_mra" and d.aligned_across_channels
    np.testing.assert_allclose(d.modes.sum(axis=0), x, atol=1e-10)
    assert d.centre_frequencies_hz[-1] > d.centre_frequencies_hz[1]
    r = modewise.modwt_mra(x, 1.0, boundary="reflection")
    np.testing.assert_allclose(r.modes.sum(axis=0), x, atol=1e-10)


def test_bad_settings_raise():
    x = np.ones((1, 64))
    with pytest.raises(ValueError, match="boundary"):
        modewise.modwt_mra(x, 1.0, boundary="zero")
    with pytest.raises(ValueError, match="unknown"):
        modewise.modwt_mra(x, 1.0, wavelet="morlet")
