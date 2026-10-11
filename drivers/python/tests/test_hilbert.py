import numpy as np
import pytest

import modewise

FS = 50.0


def two_tones():
    t = np.arange(500) / FS
    return np.vstack([np.sin(2 * np.pi * 2.0 * t) + 0.5 * np.sin(2 * np.pi * 12.0 * t)] * 2)


def test_hilbert_gives_mode_frequencies_and_amplitudes():
    modes = modewise.mvmd(two_tones(), FS, 2, 2000.0, init="linear", tau=0.0, tolerance=1e-7)
    attrs = modewise.hilbert(modes)
    assert isinstance(attrs, modewise.Instantaneous)
    assert attrs.amplitude.shape == attrs.frequency_hz.shape == modes.modes.shape
    middle = attrs.frequency_hz[:, 0, 100:400]
    assert abs(np.median(middle[0]) - 2.0) < 0.05 and abs(np.median(middle[1]) - 12.0) < 0.1
    assert abs(np.median(attrs.amplitude[1, 0, 100:400]) - 0.5) < 0.02
    np.testing.assert_array_equal(attrs.centre_frequencies_hz, modes.centre_frequencies_hz)

    spectrum = modewise.hilbert_spectrum(attrs, 0.5, 20.0, 39, scale="log")
    assert spectrum.power.shape == (2, 39, 500) and spectrum.method == "hilbert_spectrum"
    peak_bins = spectrum.power[0, :, 250].argsort()[-2:]
    peak_freqs = sorted(spectrum.frequencies_hz[peak_bins])
    assert abs(peak_freqs[0] - 2.0) < 0.3 and abs(peak_freqs[1] - 12.0) < 1.5


def test_function_and_pipeline_agree():
    series = modewise.Series(two_tones(), FS)
    spec = {
        "pipeline": {"name": "h", "inputs": ["x"]},
        "step": [
            {"id": "modes", "op": "mvmd", "input": "x", "params": {"num_modes": 2, "alpha": 2000.0, "init": "linear"}},
            {"id": "attrs", "op": "hilbert", "input": "modes", "params": {"padding": "none"}},
        ],
    }
    run = modewise.Pipeline.from_dict(spec).run({"x": series})
    direct = modewise.hilbert(run.output("modes"), padding="none")
    np.testing.assert_array_equal(run.output("attrs").frequency_hz, direct.frequency_hz)


def test_epochs_of_attributes_return_both_arrays():
    series = modewise.Series(two_tones(), FS)
    events = modewise.Events([1.0, 4.0], [2.0, 2.0], ["a", "b"])
    spec = {
        "pipeline": {"name": "h", "inputs": {"x": "series", "ev": "events"}},
        "step": [
            {"id": "modes", "op": "mvmd", "input": "x", "params": {"num_modes": 2, "alpha": 2000.0, "init": "linear"}},
            {"id": "attrs", "op": "hilbert", "input": "modes"},
            {"id": "blocks", "op": "epoch", "input": {"data": "attrs", "events": "ev"}, "params": {"lag_s": 0.0}},
        ],
    }
    run = modewise.Pipeline.from_dict(spec).run({"x": series, "ev": events})
    blocks = run.output("blocks")
    first = blocks.data(0)
    assert set(first) == {"amplitude", "frequency_hz"} and first["amplitude"].shape == (2, 2, 100)
    assert isinstance(blocks.source, modewise.Instantaneous)
    assert blocks.resolvable.all(), "2 s blocks hold 4 cycles of 2 Hz"


def test_invalid_options_raise_value_error():
    modes = modewise.mvmd(two_tones(), FS, 2, 2000.0)
    with pytest.raises(ValueError):
        modewise.hilbert(modes, padding="zero")
    attrs = modewise.hilbert(modes)
    for kwargs in ({"scale": "mel"}, {"weight": "energy"}):
        with pytest.raises(ValueError):
            modewise.hilbert_spectrum(attrs, 0.5, 20.0, 10, **kwargs)
    with pytest.raises(ValueError):
        modewise.hilbert_spectrum(attrs, 0.0, 20.0, 10, scale="log")
