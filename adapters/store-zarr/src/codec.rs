//! Artifact ⇄ Zarr group conversion.
//!
//! Arrays hold the numbers; group attributes (under `modewise`) hold the metadata.
//! Epochs store their full-run source in a `source` child group plus a table of windows.

use std::collections::BTreeMap;
use std::sync::Arc;

use modewise_brain::{ChannelInfo, ChannelTable, Hemisphere, Region};
use modewise_connectivity::{FunctionalConnectivity, Measure, WindowInfo};
use modewise_decomposition::{Decomposition, InstantaneousModes, Loadings};
use modewise_epochs::{
    DroppedEvent, EpochParams, EpochSource, EpochWindow, Epochs, Event, Events, Window,
};
use modewise_pipeline::{Artifact, StoreError};
use modewise_signals::{MultichannelSeries, SamplingRate};
use modewise_spectral::TimeFrequencyMap;
use ndarray::{ArrayD, Ix2, Ix3};
use serde_json::{Map, Value, json};
use zarrs::array::{Array, ArrayBuilder, data_type};
use zarrs::filesystem::FilesystemStore;
use zarrs::group::{Group, GroupBuilder};

use crate::store_error;

type Storage = Arc<FilesystemStore>;

pub(crate) fn write(storage: &Storage, node: &str, artifact: &Artifact) -> Result<(), StoreError> {
    let attrs = match artifact {
        Artifact::Channels(t) => {
            let channels: Vec<Value> = t
                .channels()
                .iter()
                .map(|c| {
                    json!({
                        "name": c.name(),
                        "region": c.region().map(Region::as_str),
                        "hemisphere": c.hemisphere().map(Hemisphere::as_str),
                        "tags": c.tags(),
                    })
                })
                .collect();
            json!({ "kind": "channels", "channels": channels })
        }
        Artifact::Series(s) => {
            write_array(
                storage,
                &format!("{node}/data"),
                s.data().to_owned().into_dyn(),
                &["channel", "time"],
            )?;
            json!({
                "kind": "series",
                "labels": s.labels(),
                "sampling_rate_hz": s.sampling_rate().hz(),
            })
        }
        Artifact::FunctionalConnectivity(c) => {
            write_array(
                storage,
                &format!("{node}/values"),
                c.values().to_owned().into_dyn(),
                &["window", "component", "channel", "channel"],
            )?;
            if let Some(r) = c.resolvable() {
                let flags = r.mapv(|ok| if ok { 1.0 } else { 0.0 }).into_dyn();
                write_array(
                    storage,
                    &format!("{node}/resolvable"),
                    flags,
                    &["window", "component"],
                )?;
            }
            let windows = c.windows().map(|w| {
                w.iter()
                    .map(|w| json!({ "label": w.label, "event_onset_s": w.event_onset_s }))
                    .collect::<Vec<_>>()
            });
            json!({
                "kind": "functional_connectivity",
                "measure": c.measure().name(),
                "labels": c.labels(),
                "centre_frequencies_hz": c.centre_frequencies_hz(),
                "windows": windows,
                "has_resolvable": c.resolvable().is_some(),
            })
        }
        Artifact::Loadings(l) => {
            write_array(
                storage,
                &format!("{node}/values"),
                l.values().to_owned().into_dyn(),
                &["latent", "channel"],
            )?;
            json!({
                "kind": "loadings",
                "latent_labels": l.latent_labels(),
                "channel_labels": l.channel_labels(),
            })
        }
        Artifact::Decomposition(d) => {
            write_array(
                storage,
                &format!("{node}/modes"),
                d.modes().to_owned().into_dyn(),
                &["mode", "channel", "time"],
            )?;
            if let Some(per_channel) = d.channel_centre_frequencies_hz() {
                write_array(
                    storage,
                    &format!("{node}/channel_centre_frequencies_hz"),
                    per_channel.to_owned().into_dyn(),
                    &["mode", "channel"],
                )?;
            }
            json!({
                "kind": "decomposition",
                "labels": d.labels(),
                "sampling_rate_hz": d.sampling_rate().hz(),
                "centre_frequencies_hz": d.centre_frequencies_hz(),
                "aligned_across_channels": d.aligned_across_channels(),
                "has_channel_centre_frequencies": d.channel_centre_frequencies_hz().is_some(),
            })
        }
        Artifact::TimeFrequency(m) => {
            write_array(
                storage,
                &format!("{node}/power"),
                m.power().to_owned().into_dyn(),
                &["channel", "frequency", "time"],
            )?;
            if let Some(mask) = m.reliable() {
                let as_numbers = mask.mapv(|ok| if ok { 1.0 } else { 0.0 }).into_dyn();
                write_array(
                    storage,
                    &format!("{node}/reliable"),
                    as_numbers,
                    &["frequency", "time"],
                )?;
            }
            json!({
                "kind": "time_frequency",
                "labels": m.labels(),
                "frequencies_hz": m.frequencies_hz(),
                "times_s": m.times_s(),
                "sampling_rate_hz": m.sampling_rate().map(|fs| fs.hz()),
                "has_reliable_mask": m.reliable().is_some(),
            })
        }
        Artifact::Instantaneous(m) => {
            let dims = ["mode", "channel", "time"];
            write_array(
                storage,
                &format!("{node}/amplitude"),
                m.amplitude().to_owned().into_dyn(),
                &dims,
            )?;
            write_array(
                storage,
                &format!("{node}/frequency_hz"),
                m.frequency_hz().to_owned().into_dyn(),
                &dims,
            )?;
            json!({
                "kind": "instantaneous",
                "labels": m.labels(),
                "sampling_rate_hz": m.sampling_rate().hz(),
                "centre_frequencies_hz": m.centre_frequencies_hz(),
            })
        }
        Artifact::Events(e) => events_json(e),
        Artifact::Epochs(e) => {
            let source = match e.source() {
                EpochSource::Series(s) => Artifact::Series(s.clone()),
                EpochSource::Decomposition(d) => Artifact::Decomposition(d.clone()),
                EpochSource::Instantaneous(m) => Artifact::Instantaneous(m.clone()),
                EpochSource::TimeFrequency(m) => Artifact::TimeFrequency(m.clone()),
            };
            write(storage, &format!("{node}/source"), &source)?;
            epochs_json(e)
        }
    };
    let mut group = GroupBuilder::new()
        .build(storage.clone(), node)
        .map_err(store_error)?;
    *group.attributes_mut() = Map::from_iter([("modewise".to_string(), attrs)]);
    group.store_metadata().map_err(store_error)
}

/// One chunk per index of the first axis, so a single mode or channel reads cheaply.
/// No compression: on real MVMD output, gzip saved nothing on float64 and slowed reads 5x.
/// About this many bytes per chunk file: large enough that an artifact is a handful of files, small
/// enough that reading part of one stays cheap.
const CHUNK_BYTES: u64 = 4 << 20;

/// Whole rows of the first axis, as many as fit in `CHUNK_BYTES` (at least one).
fn chunk_shape(shape: &[u64]) -> Vec<u64> {
    let mut chunk: Vec<u64> = shape.iter().map(|&n| n.max(1)).collect();
    if let Some((first, rest)) = chunk.split_first_mut() {
        let row_bytes = rest.iter().product::<u64>() * 8;
        *first = (CHUNK_BYTES / row_bytes.max(1)).clamp(1, (*first).max(1));
    }
    chunk
}

fn write_array(
    storage: &Storage,
    path: &str,
    data: ArrayD<f64>,
    dims: &[&str],
) -> Result<(), StoreError> {
    let shape: Vec<u64> = data.shape().iter().map(|&n| n as u64).collect();
    let chunk = chunk_shape(&shape);
    let array = ArrayBuilder::new(shape, chunk, data_type::float64(), 0.0f64)
        .dimension_names(Some(dims.iter().copied()))
        .build(storage.clone(), path)
        .map_err(store_error)?;
    array.store_metadata().map_err(store_error)?;
    array
        .store_array_subset(&array.subset_all(), data.as_standard_layout().into_owned())
        .map_err(store_error)
}

fn events_json(events: &Events) -> Value {
    let e = events.as_slice();
    json!({
        "kind": "events",
        "onsets_s": e.iter().map(|e| e.onset_s).collect::<Vec<_>>(),
        "durations_s": e.iter().map(|e| e.duration_s).collect::<Vec<_>>(),
        "labels": e.iter().map(|e| e.label.as_str()).collect::<Vec<_>>(),
    })
}

fn epochs_json(epochs: &Epochs) -> Value {
    let p = epochs.params();
    let window = match p.window {
        Window::Duration => json!({"type": "duration"}),
        Window::Fixed { tmin_s, tmax_s } => {
            json!({"type": "fixed", "tmin_s": tmin_s, "tmax_s": tmax_s})
        }
    };
    json!({
        "kind": "epochs",
        "params": {"lag_s": p.lag_s, "window": window, "conditions": p.conditions, "min_cycles": p.min_cycles},
        "windows": epochs.windows().iter().map(|w| json!({
            "label": w.label, "event_onset_s": w.event_onset_s, "start": w.start, "end": w.end,
        })).collect::<Vec<_>>(),
        "dropped": epochs.dropped().iter().map(|d| json!({
            "label": d.label, "event_onset_s": d.event_onset_s, "reason": d.reason,
        })).collect::<Vec<_>>(),
    })
}

fn read_array(storage: &Storage, path: &str) -> Result<ArrayD<f64>, StoreError> {
    let array = Array::open(storage.clone(), path).map_err(store_error)?;
    array
        .retrieve_array_subset(&array.subset_all())
        .map_err(store_error)
}

/// Reads a typed value from the metadata, naming the node and field on failure.
fn field<T: serde::de::DeserializeOwned>(
    attrs: &Value,
    node: &str,
    key: &str,
) -> Result<T, StoreError> {
    serde_json::from_value(attrs[key].clone())
        .map_err(|e| StoreError(format!("{node}: {key}: {e}")))
}

pub(crate) fn read(storage: &Storage, node: &str) -> Result<Artifact, StoreError> {
    let group = Group::open(storage.clone(), node).map_err(store_error)?;
    let attrs = group
        .attributes()
        .get("modewise")
        .cloned()
        .ok_or_else(|| StoreError(format!("{node} is not a modewise artifact")))?;
    let fs = || {
        field::<f64>(&attrs, node, "sampling_rate_hz")
            .and_then(|hz| SamplingRate::new(hz).map_err(store_error))
    };
    match attrs["kind"].as_str() {
        Some("channels") => {
            let rows = attrs["channels"]
                .as_array()
                .ok_or_else(|| StoreError(format!("{node}: channels: missing")))?;
            let channels = rows
                .iter()
                .map(|row| {
                    let mut c = ChannelInfo::new(field::<String>(row, node, "name")?);
                    if let Some(r) = row["region"].as_str() {
                        c = c.with_region(Region::new(r));
                    }
                    if let Some(h) = row["hemisphere"].as_str() {
                        let side = Hemisphere::parse(h)
                            .ok_or_else(|| StoreError(format!("{node}: hemisphere {h:?}")))?;
                        c = c.with_hemisphere(side);
                    }
                    for (key, value) in field::<BTreeMap<String, String>>(row, node, "tags")? {
                        c = c.with_tag(key, value);
                    }
                    Ok(c)
                })
                .collect::<Result<Vec<_>, StoreError>>()?;
            ChannelTable::new(channels)
                .map(Artifact::Channels)
                .map_err(store_error)
        }
        Some("series") => {
            let data = read_array(storage, &format!("{node}/data"))?
                .into_dimensionality::<Ix2>()
                .map_err(store_error)?;
            MultichannelSeries::new(data, fs()?, field(&attrs, node, "labels")?)
                .map(Artifact::Series)
                .map_err(store_error)
        }
        Some("functional_connectivity") => {
            let values = read_array(storage, &format!("{node}/values"))?
                .into_dimensionality::<ndarray::Ix4>()
                .map_err(store_error)?;
            let name: String = field(&attrs, node, "measure")?;
            let measure = Measure::from_name(&name)
                .ok_or_else(|| StoreError(format!("{node}: unknown measure {name:?}")))?;
            let mut c =
                FunctionalConnectivity::new(values, measure, field(&attrs, node, "labels")?)
                    .map_err(store_error)?;
            if let Some(hz) = field::<Option<Vec<f64>>>(&attrs, node, "centre_frequencies_hz")? {
                c = c.with_centre_frequencies(hz).map_err(store_error)?;
            }
            if let Some(windows) = attrs["windows"].as_array() {
                let info = windows
                    .iter()
                    .map(|w| {
                        Ok(WindowInfo {
                            label: field(w, node, "label")?,
                            event_onset_s: field(w, node, "event_onset_s")?,
                        })
                    })
                    .collect::<Result<Vec<_>, StoreError>>()?;
                let resolvable = if attrs["has_resolvable"].as_bool().unwrap_or(false) {
                    let flags = read_array(storage, &format!("{node}/resolvable"))?
                        .into_dimensionality::<Ix2>()
                        .map_err(store_error)?;
                    Some(flags.mapv(|v| v > 0.5))
                } else {
                    None
                };
                c = c.with_windows(info, resolvable).map_err(store_error)?;
            }
            Ok(Artifact::FunctionalConnectivity(c))
        }
        Some("loadings") => {
            let values = read_array(storage, &format!("{node}/values"))?
                .into_dimensionality::<Ix2>()
                .map_err(store_error)?;
            Loadings::new(
                values,
                field(&attrs, node, "latent_labels")?,
                field(&attrs, node, "channel_labels")?,
            )
            .map(Artifact::Loadings)
            .map_err(store_error)
        }
        Some("decomposition") => {
            let modes = read_array(storage, &format!("{node}/modes"))?
                .into_dimensionality::<Ix3>()
                .map_err(store_error)?;
            let aligned = attrs["aligned_across_channels"].as_bool().unwrap_or(false);
            let labels = field(&attrs, node, "labels")?;
            let per_channel = format!("{node}/channel_centre_frequencies_hz");
            // Per-channel methods keep NaN for missing modes; JSON attributes cannot hold NaN,
            // so their mode frequencies are recomputed from the array.
            let decomposition = if attrs["has_channel_centre_frequencies"]
                .as_bool()
                .unwrap_or(false)
            {
                let frequencies = read_array(storage, &per_channel)?
                    .into_dimensionality::<Ix2>()
                    .map_err(store_error)?;
                Decomposition::per_channel(modes, frequencies, fs()?, labels)
            } else {
                Decomposition::new(
                    modes,
                    field(&attrs, node, "centre_frequencies_hz")?,
                    fs()?,
                    labels,
                    aligned,
                )
            };
            decomposition
                .map(Artifact::Decomposition)
                .map_err(store_error)
        }
        Some("time_frequency") => {
            let power = read_array(storage, &format!("{node}/power"))?
                .into_dimensionality::<Ix3>()
                .map_err(store_error)?;
            let mut map = TimeFrequencyMap::new(
                power,
                field(&attrs, node, "frequencies_hz")?,
                field(&attrs, node, "times_s")?,
                field(&attrs, node, "labels")?,
            )
            .map_err(store_error)?;
            if let Some(hz) = attrs["sampling_rate_hz"].as_f64() {
                map = map
                    .with_sampling_rate(SamplingRate::new(hz).map_err(store_error)?)
                    .map_err(store_error)?;
            }
            if attrs["has_reliable_mask"].as_bool().unwrap_or(false) {
                let mask = read_array(storage, &format!("{node}/reliable"))?
                    .into_dimensionality::<Ix2>()
                    .map_err(store_error)?
                    .mapv(|v| v > 0.5);
                map = map.with_reliable(mask).map_err(store_error)?;
            }
            Ok(Artifact::TimeFrequency(map))
        }
        Some("instantaneous") => {
            let cube = |name: &str| -> Result<ndarray::Array3<f64>, StoreError> {
                read_array(storage, &format!("{node}/{name}"))?
                    .into_dimensionality::<Ix3>()
                    .map_err(store_error)
            };
            InstantaneousModes::new(
                cube("amplitude")?,
                cube("frequency_hz")?,
                field(&attrs, node, "centre_frequencies_hz")?,
                fs()?,
                field(&attrs, node, "labels")?,
            )
            .map(Artifact::Instantaneous)
            .map_err(store_error)
        }
        Some("events") => {
            let onsets: Vec<f64> = field(&attrs, node, "onsets_s")?;
            let durations: Vec<f64> = field(&attrs, node, "durations_s")?;
            let labels: Vec<String> = field(&attrs, node, "labels")?;
            let events = onsets
                .into_iter()
                .zip(durations)
                .zip(labels)
                .map(|((onset_s, duration_s), label)| Event {
                    onset_s,
                    duration_s,
                    label,
                })
                .collect();
            Events::new(events)
                .map(Artifact::Events)
                .map_err(store_error)
        }
        Some("epochs") => read_epochs(storage, node, &attrs),
        other => Err(StoreError(format!(
            "{node}: unknown artifact kind {other:?}"
        ))),
    }
}

fn read_epochs(storage: &Storage, node: &str, attrs: &Value) -> Result<Artifact, StoreError> {
    let source = match read(storage, &format!("{node}/source"))? {
        Artifact::Series(s) => EpochSource::Series(s),
        Artifact::Decomposition(d) => EpochSource::Decomposition(d),
        Artifact::Instantaneous(m) => EpochSource::Instantaneous(m),
        Artifact::TimeFrequency(m) => EpochSource::TimeFrequency(m),
        other => {
            return Err(StoreError(format!(
                "{node}: epochs cannot hold a {} source",
                other.kind()
            )));
        }
    };
    let p = &attrs["params"];
    let window = match p["window"]["type"].as_str() {
        Some("duration") => Window::Duration,
        Some("fixed") => Window::Fixed {
            tmin_s: field(&p["window"], node, "tmin_s")?,
            tmax_s: field(&p["window"], node, "tmax_s")?,
        },
        other => {
            return Err(StoreError(format!(
                "{node}: unknown epoch window {other:?}"
            )));
        }
    };
    let params = EpochParams {
        lag_s: field(p, node, "lag_s")?,
        window,
        conditions: field(p, node, "conditions")?,
        min_cycles: field(p, node, "min_cycles")?,
    };
    let windows = attrs["windows"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|w| {
            Ok(EpochWindow {
                label: field(w, node, "label")?,
                event_onset_s: field(w, node, "event_onset_s")?,
                start: field(w, node, "start")?,
                end: field(w, node, "end")?,
            })
        })
        .collect::<Result<Vec<_>, StoreError>>()?;
    let dropped = attrs["dropped"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|d| {
            Ok(DroppedEvent {
                label: field(d, node, "label")?,
                event_onset_s: field(d, node, "event_onset_s")?,
                reason: field(d, node, "reason")?,
            })
        })
        .collect::<Result<Vec<_>, StoreError>>()?;
    Epochs::from_parts(source, windows, dropped, params)
        .map(Artifact::Epochs)
        .map_err(store_error)
}

#[cfg(test)]
mod chunks {
    use zarrs::filesystem::FilesystemStore;

    use super::*;

    #[test]
    fn chunks_hold_whole_rows_up_to_a_few_megabytes() {
        assert_eq!(
            chunk_shape(&[434, 488]),
            [434, 488],
            "a 1.7 MB series is one chunk"
        );
        assert_eq!(
            chunk_shape(&[7, 434, 488]),
            [2, 434, 488],
            "1.7 MB modes: two per chunk"
        );
        assert_eq!(
            chunk_shape(&[3, 5000, 5000]),
            [1, 5000, 5000],
            "a row larger than the target stays whole"
        );
        assert_eq!(
            chunk_shape(&[0, 4]),
            [1, 4],
            "empty axes still get a chunk of one"
        );
        assert_eq!(chunk_shape(&[10]), [10]);
    }

    #[test]
    fn arrays_written_one_row_per_chunk_still_read() {
        // Results folders from before size-based chunking used one row of the first axis per chunk.
        let dir =
            std::env::temp_dir().join(format!("modewise-zarr-old-chunks-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let storage: Storage = Arc::new(FilesystemStore::new(&dir).unwrap());
        let data = ndarray::Array2::from_shape_fn((3, 5), |(r, c)| (r * 10 + c) as f64).into_dyn();
        let array = ArrayBuilder::new(vec![3, 5], vec![1, 5], data_type::float64(), 0.0f64)
            .build(storage.clone(), "/old")
            .unwrap();
        array.store_metadata().unwrap();
        array
            .store_array_subset(&array.subset_all(), data.clone())
            .unwrap();
        assert_eq!(read_array(&storage, "/old").unwrap(), data);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
