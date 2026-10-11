import numpy as np
import pytest

import modewise

SPEC = """
[pipeline]
name = "band-power"
inputs = ["bold"]

[[step]]
id = "modes"
op = "mvmd"
input = "bold"
params = {{ num_modes = 3, alpha = 2000.0, init = "linear", tau = 0.0, tolerance = 1e-7 }}

[[step]]
id = "slow"
op = "reconstruct"
input = "modes"
params = {{ modes = {modes} }}

[[step]]
id = "spectrum"
op = "stft"
input = "slow"
params = {{ window_len = 128, hop = 32 }}
"""


def bold():
    t = np.arange(1000) / 100.0
    tones = [np.sin(2 * np.pi * f * t) for f in (2.0, 12.0, 30.0)]
    return np.vstack([tones[0] + tones[1] + 0.3 * tones[2], 0.6 * tones[0] + tones[1] + 0.3 * tones[2]])


def test_runs_and_returns_typed_outputs():
    pipe = modewise.Pipeline.from_toml(SPEC.format(modes="[0]"))
    assert pipe.steps == ["modes", "slow", "spectrum"]
    run = pipe.run({"bold": bold()}, sampling_rate=100.0, labels={"bold": ["left", "right"]})

    assert set(run.outputs) == {"modes.modes", "slow.signal", "spectrum.power"}
    modes = run.output("modes")
    assert isinstance(modes, modewise.Decomposition) and modes.method == "mvmd"
    assert modes.labels == ["left", "right"]
    slow = run.output("slow.signal")
    assert isinstance(slow, modewise.Series) and slow.data.shape == (2, 1000)
    spectrum = run.output("spectrum")
    assert isinstance(spectrum, modewise.TimeFrequencyMap) and spectrum.method == "stft"


def test_shared_store_reuses_unchanged_steps():
    store = modewise.MemoryStore()
    first = modewise.Pipeline.from_toml(SPEC.format(modes="[0]")).run({"bold": bold()}, 100.0, store=store)
    assert first.cached == {"modes": False, "slow": False, "spectrum": False}

    again = modewise.Pipeline.from_toml(SPEC.format(modes="[0]")).run({"bold": bold()}, 100.0, store=store)
    assert all(again.cached.values())

    changed = modewise.Pipeline.from_toml(SPEC.format(modes="[0, 1]")).run({"bold": bold()}, 100.0, store=store)
    assert changed.cached == {"modes": True, "slow": False, "spectrum": False}


def test_manifest_records_resolved_parameters():
    run = modewise.Pipeline.from_toml(SPEC.format(modes="[0]")).run({"bold": bold()}, 100.0)
    steps = {s["id"]: s for s in run.manifest["steps"]}
    assert steps["modes"]["params"]["max_iterations"] == 500
    assert steps["modes"]["op_version"] == modewise.__version__


def test_from_dict_matches_toml():
    spec = {
        "pipeline": {"name": "d", "inputs": ["x"]},
        "step": [{"id": "power", "op": "stft", "input": "x", "params": {"window_len": 16, "hop": 4}}],
    }
    run = modewise.Pipeline.from_dict(spec).run({"x": np.ones((1, 64))}, 1.0)
    assert run.output("power").power.shape == (1, 9, 13)


def test_from_file(tmp_path):
    path = tmp_path / "p.toml"
    path.write_text(SPEC.format(modes="[0]"))
    assert modewise.Pipeline.from_file(path).name == "band-power"


@pytest.mark.parametrize(
    "spec",
    [
        SPEC.format(modes="[0]").replace('op = "stft"', 'op = "nope"'),
        SPEC.format(modes="[0]").replace("num_modes = 3", "num_modes = 0"),
        SPEC.format(modes="[0]").replace('input = "slow"', 'input = "spectrum"'),
    ],
    ids=["unknown op", "invalid params", "cycle"],
)
def test_invalid_spec_raises_value_error(spec):
    with pytest.raises(ValueError):
        modewise.Pipeline.from_toml(spec)


def test_run_errors():
    pipe = modewise.Pipeline.from_toml(SPEC.format(modes="[7]"))
    with pytest.raises(ValueError, match="not provided"):
        pipe.run({}, 100.0)
    with pytest.raises(RuntimeError, match="slow"):
        pipe.run({"bold": bold()}, 100.0)
    run = modewise.Pipeline.from_toml(SPEC.format(modes="[0]")).run({"bold": bold()}, 100.0)
    with pytest.raises(KeyError):
        run.output("missing")


def test_to_xarray():
    xr = pytest.importorskip("xarray")
    run = modewise.Pipeline.from_toml(SPEC.format(modes="[0]")).run(
        {"bold": bold()}, 100.0, labels={"bold": ["left", "right"]}
    )
    modes = modewise.to_xarray(run.output("modes"))
    assert modes.dims == ("mode", "channel", "time")
    assert list(modes.channel.values) == ["left", "right"]
    power = modewise.to_xarray(run.output("spectrum"))
    assert isinstance(power, xr.DataArray) and power.dims == ("channel", "frequency", "time")
    assert modewise.to_xarray(run.output("slow")).dims == ("channel", "time")


def test_operations_describe_themselves():
    ops = {op["name"]: op for op in modewise.operations()}
    assert {"mvmd", "spectrum", "epoch", "select"} <= set(ops)
    spectrum = ops["spectrum"]
    assert spectrum["summary"] and spectrum["inputs"][0] == {"name": "signal", "kinds": ["series"]}
    assert spectrum["params"]["properties"]["window"]["default"] == "boxcar"
    assert ops["mvmd"]["params"]["required"] == ["num_modes", "alpha"]


def test_unkept_steps_feed_later_steps_without_being_stored():
    pipeline = modewise.Pipeline.from_dict({"pipeline": {"name": "k", "inputs": ["x"]}, "step": [
        {"id": "modes", "op": "modwt_mra", "input": "x", "params": {"levels": 3}, "keep": False},
        {"id": "fc", "op": "functional_connectivity", "input": "modes", "params": {"measure": "pearson"}}]})
    run = pipeline.run({"x": modewise.Series(np.random.default_rng(0).standard_normal((2, 64)), 1.0)})
    assert run.output("fc").values.shape == (1, 4, 2, 2)
    assert [s["stored"] for s in run.manifest["steps"]] == [False, True]
    with pytest.raises(ValueError, match="keep = false"):
        run.output("modes")
    with pytest.raises(ValueError, match="no step uses"):
        modewise.Pipeline.from_dict({"pipeline": {"name": "k", "inputs": ["x"]}, "step": [
            {"id": "modes", "op": "modwt_mra", "input": "x", "keep": False}]})
