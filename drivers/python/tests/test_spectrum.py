import numpy as np
import pytest

import modewise


def test_spectrum_is_a_periodogram_by_default_and_welch_with_segments():
    t = np.arange(400) / 2.0
    x = np.vstack([np.sin(2 * np.pi * 0.1 * t), np.random.default_rng(0).standard_normal(400)])
    p = modewise.spectrum(x, 2.0)
    assert p.power.shape == (2, 201, 1) and p.method == "spectrum"
    assert abs(p.frequencies_hz[np.argmax(p.power[0, :, 0])] - 0.1) < 1e-9
    # Parseval for the density: the integral of a one-sided PSD is the variance.
    assert abs(p.power[1, :, 0].sum() * (2.0 / 400) - x[1].var()) < 0.02 * x[1].var()
    w = modewise.spectrum(x, 2.0, segment_s=50.0, window="hann", detrend="linear")
    assert w.power.shape == (2, 51, 1)
    with pytest.raises(ValueError, match="window"):
        modewise.spectrum(x, 2.0, window="hamming")
    with pytest.raises(ValueError, match="detrend"):
        modewise.spectrum(x, 2.0, detrend="quadratic")


def test_band_amplitude_feature_table_and_paired_test(tmp_path):
    h5py = pytest.importorskip("h5py")
    rng = np.random.default_rng(1)
    t = np.arange(400) / 1.25
    for subject in ("01", "02", "03", "04", "05", "06"):
        folder = tmp_path / "data" / f"sub-{subject}"
        folder.mkdir(parents=True)
        # Channel "a" carries a strong 0.05 Hz rhythm; channel "b" is white noise.
        x = np.vstack([3 * np.sin(2 * np.pi * 0.05 * t + rng.uniform(0, 6)), np.zeros_like(t)]) + rng.standard_normal((2, t.size))
        with h5py.File(folder / f"sub-{subject}_task-rest_bold.h5", "w") as f:
            f["signal"] = x
    dataset = modewise.Dataset.from_toml(
        'root = "data"\n[[input]]\nname = "bold"\npattern = "sub-{subject}/sub-{subject}_task-{task}_bold.h5"\n'
        'format = "hdf5"\ndataset = "signal"\nsampling_rate_hz = 1.25\n', tmp_path)
    pipeline = modewise.Pipeline.from_dict({
        "pipeline": {"name": "alff", "inputs": ["bold"]},
        "step": [{"id": "psd", "op": "spectrum", "input": "bold"}],
    })
    pipeline.run_dataset(dataset, tmp_path / "results", progress=False)
    saved = modewise.Results(tmp_path / "results")
    bands = {"slow4": (0.027, 0.073), "slow5": (0.01, 0.027)}
    table = saved.feature_table("alff", "psd", "band_amplitude", bands=bands)
    assert table.feature_names == ["ch0", "ch1"] and table.conditions[:2] == ["slow4", "slow5"]
    report = modewise.test_paired(table, ("slow4", "slow5"), n_permutations=64, seed=0)
    assert report.t[0] > 3 and report.exhaustive and report.p_permutation[0] == pytest.approx(2 / 64), "slow-4 in every subject: the smallest two-sided exact p"
    relative = saved.feature_table("alff", "psd", "band_amplitude", bands=bands, relative=True)
    assert np.all((relative.values > 0) & (relative.values < 1))
    with pytest.raises(ValueError, match="needs bands"):
        saved.feature_table("alff", "psd", "band_amplitude")
    with pytest.raises(ValueError, match="only apply"):
        saved.feature_table("alff", "psd", "edges", bands=bands)
