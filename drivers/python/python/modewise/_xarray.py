import numpy as np

from ._core import FunctionalConnectivity, Decomposition, Loadings, Series, TimeFrequencyMap


def to_xarray(result):
    """Converts a Series, Decomposition, TimeFrequencyMap, Loadings or FunctionalConnectivity to a labelled xarray.DataArray.

    Needs xarray: pip install "modewise[xarray]".
    """
    import xarray as xr

    if isinstance(result, Decomposition):
        n_samples = result.modes.shape[2]
        return xr.DataArray(
            result.modes,
            dims=("mode", "channel", "time"),
            coords={
                "mode": np.arange(result.n_modes),
                "centre_frequency_hz": ("mode", result.centre_frequencies_hz),
                "channel": result.labels,
                "time": np.arange(n_samples) / result.sampling_rate,
            },
            attrs={"method": result.method, "sampling_rate": result.sampling_rate},
        )
    if isinstance(result, TimeFrequencyMap):
        return xr.DataArray(
            result.power,
            dims=("channel", "frequency", "time"),
            coords={"channel": result.labels, "frequency": result.frequencies_hz, "time": result.times_s},
            attrs={"method": result.method},
        )
    if isinstance(result, Series):
        n_samples = result.data.shape[1]
        return xr.DataArray(
            result.data,
            dims=("channel", "time"),
            coords={"channel": result.labels, "time": np.arange(n_samples) / result.sampling_rate},
            attrs={"sampling_rate": result.sampling_rate},
        )
    if isinstance(result, FunctionalConnectivity):
        n_windows, n_components = result.values.shape[:2]
        coords = {
            "window": np.arange(n_windows),
            "component": np.arange(n_components),
            "channel_a": result.labels,
            "channel_b": result.labels,
        }
        if result.centre_frequencies_hz is not None:
            coords["centre_frequency_hz"] = ("component", result.centre_frequencies_hz)
        if result.window_labels is not None:
            coords["window_label"] = ("window", result.window_labels)
            coords["window_onset_s"] = ("window", result.window_onsets_s)
        return xr.DataArray(
            result.values,
            dims=("window", "component", "channel_a", "channel_b"),
            coords=coords,
            attrs={"measure": result.measure},
        )
    if isinstance(result, Loadings):
        return xr.DataArray(
            result.values,
            dims=("latent", "channel"),
            coords={"latent": result.latent_labels, "channel": result.channel_labels},
        )
    raise TypeError(f"cannot convert {type(result).__name__} to xarray")
