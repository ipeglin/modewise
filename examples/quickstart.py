"""modewise quick start. Runs as-is (synthetic data), for example in Spyder or `python examples/quickstart.py`.

1. Decompose one recording with a function call.
2. Run a pipeline: full-run MVMD, then epochs around events (docs/epochs.md).
3. Look at the results.
"""

import numpy as np

import modewise

FS = 1.25  # Hz; TR = 0.8 s
t = np.arange(600) / FS
slow, fast = np.sin(2 * np.pi * 0.02 * t), np.sin(2 * np.pi * 0.2 * t)
bold = modewise.Series(np.vstack([slow + fast, 0.5 * slow + fast]), FS, labels=["left", "right"])

# 1. Function call: channels x samples in, modes out.
result = modewise.mvmd(bold.data, FS, num_modes=2, alpha=2000.0, init="linear", tau=0.0, tolerance=1e-7)
print("centre frequencies (Hz):", result.centre_frequencies_hz.round(3))
print("modes shape (mode, channel, time):", result.modes.shape)

# 2. Pipeline: decompose the whole run, then cut 18 s blocks with a 4 s haemodynamic lag.
blocks = modewise.Events(
    onsets_s=[20.0 + 40.0 * b for b in range(10)],
    durations_s=[18.0] * 10,
    labels=["face" if b % 2 == 0 else "shape" for b in range(10)],
)
PIPELINE = """
    [pipeline]
    name = "quickstart"
    inputs = { bold = "series", blocks = "events" }

    [[step]]
    id = "modes"
    op = "mvmd"
    input = "bold"
    params = { num_modes = 2, alpha = 2000.0, init = "linear", tau = 0.0, tolerance = 1e-7 }

    [[step]]
    id = "epochs"
    op = "epoch"
    input = { data = "modes", events = "blocks" }
    params = { lag_s = LAG }
    """
pipeline = modewise.Pipeline.from_toml(PIPELINE.replace("LAG", "4.0"))
store = modewise.MemoryStore()  # reuse unchanged steps across runs in this session
run = pipeline.run({"bold": bold, "blocks": blocks}, store=store)

# 3. Results.
epochs = run.output("epochs")
print(f"{len(epochs)} epochs, lag {epochs.lag_s} s, {len(epochs.dropped)} dropped")
print("resolvable (epoch x mode), first 3 epochs:\n", epochs.resolvable[:3])
print("epoch 0 data shape (mode, channel, time):", epochs.data(0).shape)

# Lag sensitivity: only the epoch step reruns; the decomposition comes from the store.
for lag in (2.0, 6.0):
    swept = modewise.Pipeline.from_toml(PIPELINE.replace("LAG", str(lag))).run({"bold": bold, "blocks": blocks}, store=store)
    print(f"lag {lag} s: steps cached {swept.cached}")
