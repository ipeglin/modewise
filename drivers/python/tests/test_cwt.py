import numpy as np
import pytest

import modewise

FS = 20.0


def test_cwt_finds_tones_and_marks_the_cone_of_influence():
    t = np.arange(400) / FS
    x = np.vstack([np.sin(2 * np.pi * 2.0 * t), np.sin(2 * np.pi * 5.0 * t)])
    scalogram = modewise.cwt(x, FS, 0.5, 8.0, 40, labels=["slow", "fast"])
    assert scalogram.power.shape == (2, 40, 400) and scalogram.method == "cwt"
    assert scalogram.sampling_rate == FS and scalogram.reliable.shape == (40, 400)
    for channel, tone in enumerate((2.0, 5.0)):
        peak = scalogram.frequencies_hz[scalogram.power[channel, :, 200].argmax()]
        assert abs(peak - tone) < 0.3
    lowest = scalogram.reliable[0]
    assert not lowest[0] and not lowest[-1] and lowest[200], "edges unreliable, middle reliable at 0.5 Hz"

    rectified = modewise.cwt(x, FS, 0.5, 8.0, 40, rectify=True)
    assert not np.allclose(rectified.power, scalogram.power)


def test_stft_maps_have_no_sampling_rate_or_mask():
    power = modewise.stft(np.ones((1, 64)), 10.0, 16, 8)
    assert power.sampling_rate is None and power.reliable is None


def test_cwt_epochs_through_a_pipeline():
    t = np.arange(600) / 1.25
    series = modewise.Series(np.sin(2 * np.pi * 0.2 * t)[None, :], 1.25)
    events = modewise.Events([0.0, 200.0], [18.0, 18.0], ["a", "b"])
    spec = {
        "pipeline": {"name": "w", "inputs": {"x": "series", "ev": "events"}},
        "step": [
            {"id": "scalogram", "op": "cwt", "input": "x", "params": {"f_min": 0.01, "f_max": 0.6, "n_bins": 30}},
            {"id": "blocks", "op": "epoch", "input": {"data": "scalogram", "events": "ev"}, "params": {"lag_s": 4.0}},
        ],
    }
    run = modewise.Pipeline.from_dict(spec).run({"x": series, "ev": events})
    blocks = run.output("blocks")
    assert isinstance(blocks.source, modewise.TimeFrequencyMap)
    assert blocks.data(1).shape[:2] == (1, 30)
    fast = int(np.argmin(np.abs(blocks.source.frequencies_hz - 0.2)))
    assert not blocks.resolvable[0, fast] and blocks.resolvable[1, fast]


def test_invalid_cwt_settings():
    x = np.ones((1, 64))
    for kwargs in ({"omega0": 2.0},):
        with pytest.raises(ValueError):
            modewise.cwt(x, 10.0, 0.5, 4.0, 8, **kwargs)
    with pytest.raises(ValueError, match="outside"):
        modewise.cwt(x, 10.0, 0.5, 8.0, 8)
