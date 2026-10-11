use std::path::Path;

use hdf5_pure::{AttrValue, FileBuilder, GroupBuilder};
use modewise_epochs::{EpochSource, Epochs, Events, Window};
use modewise_pipeline::{Artifact, ArtifactStore, RunManifest};
use modewise_store_zarr::ZarrStore;

use crate::{ResultsError, io};

/// One file per recording: `/<step>/<output>/` groups, plus the run manifest as a root attribute.
pub(crate) fn write_recording(
    store: &ZarrStore,
    recording: &str,
    manifest: &RunManifest,
    path: &Path,
) -> Result<(), ResultsError> {
    let mut file = FileBuilder::new();
    file.set_attr("recording", AttrValue::String(recording.to_string()));
    file.set_attr("pipeline", AttrValue::String(manifest.pipeline.clone()));
    file.set_attr("manifest", AttrValue::String(manifest.to_json()));

    for step in &manifest.steps {
        let mut step_group = file.create_group(&step.id);
        for (output, key) in &step.outputs {
            let artifact = store.get(key)?.ok_or_else(|| ResultsError::MissingOutput {
                reference: format!("{}.{output}", step.id),
                available: vec![],
            })?;
            let mut group = step_group.create_group(output);
            write_artifact(&mut group, &artifact).map_err(|e| io(path, e))?;
            step_group.add_group(group.finish());
        }
        file.add_group(step_group.finish());
    }
    file.write(path).map_err(|e| io(path, e))
}

fn strings(group: &mut GroupBuilder, name: &str, values: &[&str]) -> Result<(), String> {
    group
        .create_dataset(name)
        .with_strings(values)
        .map_err(|e| e.to_string())?;
    Ok(())
}

fn write_artifact(group: &mut GroupBuilder, artifact: &Artifact) -> Result<(), String> {
    let shape = |s: &[usize]| s.iter().map(|&n| n as u64).collect::<Vec<_>>();
    group.set_attr("kind", AttrValue::String(artifact.kind().to_string()));
    let labels = match artifact {
        Artifact::Channels(t) => {
            let channels = t.channels();
            let column = |values: Vec<String>, name: &str, group: &mut GroupBuilder| {
                strings(
                    group,
                    name,
                    &values.iter().map(String::as_str).collect::<Vec<_>>(),
                )
            };
            let regions = channels
                .iter()
                .map(|c| c.region().map_or("", |r| r.as_str()).to_string())
                .collect();
            column(regions, "regions", group)?;
            let sides = channels
                .iter()
                .map(|c| c.hemisphere().map_or("", |h| h.as_str()).to_string())
                .collect();
            column(sides, "hemispheres", group)?;
            let keys: std::collections::BTreeSet<&String> =
                channels.iter().flat_map(|c| c.tags().keys()).collect();
            for key in keys {
                let values = channels
                    .iter()
                    .map(|c| c.tag(key).unwrap_or_default().to_string())
                    .collect();
                column(values, &format!("tag_{key}"), group)?;
            }
            return column(t.names(), "labels", group);
        }
        Artifact::Series(s) => {
            let data = s.data().as_standard_layout().into_owned();
            group
                .create_dataset("data")
                .with_f64_data(data.as_slice().unwrap_or_default())
                .with_shape(&shape(data.shape()));
            group.set_attr("sampling_rate_hz", AttrValue::F64(s.sampling_rate().hz()));
            s.labels()
        }
        Artifact::FunctionalConnectivity(c) => {
            let values = c.values().as_standard_layout().into_owned();
            group
                .create_dataset("values")
                .with_f64_data(values.as_slice().unwrap_or_default())
                .with_shape(&shape(values.shape()));
            group.set_attr("measure", AttrValue::String(c.measure().name().into()));
            if let Some(hz) = c.centre_frequencies_hz() {
                group
                    .create_dataset("centre_frequencies_hz")
                    .with_f64_data(hz);
            }
            if let Some(windows) = c.windows() {
                let labels: Vec<&str> = windows.iter().map(|w| w.label.as_str()).collect();
                strings(group, "window_labels", &labels)?;
                let onsets: Vec<f64> = windows.iter().map(|w| w.event_onset_s).collect();
                group
                    .create_dataset("window_onsets_s")
                    .with_f64_data(&onsets);
            }
            if let Some(r) = c.resolvable() {
                let flags: Vec<u8> = r.iter().map(|&ok| u8::from(ok)).collect();
                group
                    .create_dataset("resolvable")
                    .with_u8_data(&flags)
                    .with_shape(&shape(r.shape()));
            }
            c.labels()
        }
        Artifact::Loadings(l) => {
            let values = l.values().as_standard_layout().into_owned();
            group
                .create_dataset("values")
                .with_f64_data(values.as_slice().unwrap_or_default())
                .with_shape(&shape(values.shape()));
            let latents: Vec<&str> = l.latent_labels().iter().map(String::as_str).collect();
            strings(group, "latent_labels", &latents)?;
            l.channel_labels()
        }
        Artifact::Decomposition(d) => {
            let modes = d.modes().as_standard_layout().into_owned();
            group
                .create_dataset("modes")
                .with_f64_data(modes.as_slice().unwrap_or_default())
                .with_shape(&shape(modes.shape()));
            group
                .create_dataset("centre_frequencies_hz")
                .with_f64_data(d.centre_frequencies_hz());
            if let Some(per_channel) = d.channel_centre_frequencies_hz() {
                let per_channel = per_channel.as_standard_layout().into_owned();
                group
                    .create_dataset("channel_centre_frequencies_hz")
                    .with_f64_data(per_channel.as_slice().unwrap_or_default())
                    .with_shape(&shape(per_channel.shape()));
            }
            group.set_attr("sampling_rate_hz", AttrValue::F64(d.sampling_rate().hz()));
            d.labels()
        }
        Artifact::TimeFrequency(m) => {
            let power = m.power().as_standard_layout().into_owned();
            group
                .create_dataset("power")
                .with_f64_data(power.as_slice().unwrap_or_default())
                .with_shape(&shape(power.shape()));
            group
                .create_dataset("frequencies_hz")
                .with_f64_data(m.frequencies_hz());
            group.create_dataset("times_s").with_f64_data(m.times_s());
            if let Some(mask) = m.reliable() {
                let flags: Vec<u8> = mask.iter().map(|&ok| u8::from(ok)).collect();
                group
                    .create_dataset("reliable")
                    .with_u8_data(&flags)
                    .with_shape(&shape(mask.shape()));
            }
            if let Some(fs) = m.sampling_rate() {
                group.set_attr("sampling_rate_hz", AttrValue::F64(fs.hz()));
            }
            m.labels()
        }
        Artifact::Instantaneous(m) => {
            let amplitude = m.amplitude().as_standard_layout().into_owned();
            let frequency = m.frequency_hz().as_standard_layout().into_owned();
            group
                .create_dataset("amplitude")
                .with_f64_data(amplitude.as_slice().unwrap_or_default())
                .with_shape(&shape(amplitude.shape()));
            group
                .create_dataset("frequency_hz")
                .with_f64_data(frequency.as_slice().unwrap_or_default())
                .with_shape(&shape(frequency.shape()));
            group
                .create_dataset("centre_frequencies_hz")
                .with_f64_data(m.centre_frequencies_hz());
            group.set_attr("sampling_rate_hz", AttrValue::F64(m.sampling_rate().hz()));
            m.labels()
        }
        Artifact::Events(e) => return write_events(group, e),
        Artifact::Epochs(e) => return write_epochs(group, e),
    };
    let labels: Vec<&str> = labels.iter().map(String::as_str).collect();
    strings(group, "labels", &labels)
}

fn write_events(group: &mut GroupBuilder, events: &Events) -> Result<(), String> {
    let e = events.as_slice();
    group
        .create_dataset("onset_s")
        .with_f64_data(&e.iter().map(|e| e.onset_s).collect::<Vec<_>>());
    group
        .create_dataset("duration_s")
        .with_f64_data(&e.iter().map(|e| e.duration_s).collect::<Vec<_>>());
    strings(
        group,
        "event_labels",
        &e.iter().map(|e| e.label.as_str()).collect::<Vec<_>>(),
    )
}

/// The full-run source goes in `source/`; windows are sample ranges into it (0-based, end exclusive).
fn write_epochs(group: &mut GroupBuilder, epochs: &Epochs) -> Result<(), String> {
    let source = match epochs.source() {
        EpochSource::Series(s) => Artifact::Series(s.clone()),
        EpochSource::Decomposition(d) => Artifact::Decomposition(d.clone()),
        EpochSource::Instantaneous(m) => Artifact::Instantaneous(m.clone()),
        EpochSource::TimeFrequency(m) => Artifact::TimeFrequency(m.clone()),
    };
    let mut source_group = group.create_group("source");
    write_artifact(&mut source_group, &source)?;
    group.add_group(source_group.finish());

    let w = epochs.windows();
    group
        .create_dataset("start_sample")
        .with_u64_data(&w.iter().map(|w| w.start as u64).collect::<Vec<_>>());
    group
        .create_dataset("end_sample")
        .with_u64_data(&w.iter().map(|w| w.end as u64).collect::<Vec<_>>());
    group
        .create_dataset("event_onset_s")
        .with_f64_data(&w.iter().map(|w| w.event_onset_s).collect::<Vec<_>>());
    strings(
        group,
        "epoch_labels",
        &w.iter().map(|w| w.label.as_str()).collect::<Vec<_>>(),
    )?;
    if let Some(resolvable) = epochs.resolvable() {
        let flags: Vec<u8> = resolvable.iter().map(|&r| u8::from(r)).collect();
        group
            .create_dataset("resolvable")
            .with_u8_data(&flags)
            .with_shape(&[resolvable.nrows() as u64, resolvable.ncols() as u64]);
    }
    let d = epochs.dropped();
    if !d.is_empty() {
        group
            .create_dataset("dropped_onset_s")
            .with_f64_data(&d.iter().map(|d| d.event_onset_s).collect::<Vec<_>>());
        strings(
            group,
            "dropped_labels",
            &d.iter().map(|d| d.label.as_str()).collect::<Vec<_>>(),
        )?;
        strings(
            group,
            "dropped_reasons",
            &d.iter().map(|d| d.reason.as_str()).collect::<Vec<_>>(),
        )?;
    }
    let p = epochs.params();
    group.set_attr("lag_s", AttrValue::F64(p.lag_s));
    group.set_attr("min_cycles", AttrValue::F64(p.min_cycles));
    let window = match p.window {
        Window::Duration => "duration".to_string(),
        Window::Fixed { tmin_s, tmax_s } => format!("fixed [{tmin_s}, {tmax_s}) s"),
    };
    group.set_attr("window", AttrValue::String(window));
    Ok(())
}

#[cfg(test)]
mod tests {
    use hdf5_pure::{File, FileBuilder};
    use modewise_epochs::{Event, Events};
    use modewise_pipeline::Artifact;

    use super::write_artifact;

    #[test]
    fn writes_events_as_onset_duration_and_label_datasets() {
        let events = Events::new(vec![Event {
            onset_s: 1.0,
            duration_s: 2.0,
            label: "go".into(),
        }])
        .unwrap();
        let mut file = FileBuilder::new();
        let mut group = file.create_group("events");
        write_artifact(&mut group, &Artifact::Events(events)).unwrap();
        file.add_group(group.finish());
        let path =
            std::env::temp_dir().join(format!("modewise-export-events-{}.h5", std::process::id()));
        file.write(&path).unwrap();

        let h5 = File::open(&path).unwrap();
        assert_eq!(
            h5.dataset("events/onset_s").unwrap().read_f64().unwrap(),
            [1.0]
        );
        assert_eq!(
            h5.dataset("events/duration_s").unwrap().read_f64().unwrap(),
            [2.0]
        );
        assert!(h5.dataset("events/event_labels").is_ok());
    }

    #[test]
    fn writes_instantaneous_amplitude_and_frequency() {
        use modewise_decomposition::InstantaneousModes;
        use modewise_signals::SamplingRate;
        use ndarray::Array3;

        let attrs = InstantaneousModes::new(
            Array3::from_elem((2, 1, 3), 1.5),
            Array3::from_elem((2, 1, 3), 0.2),
            vec![0.1, 0.3],
            SamplingRate::new(1.25).unwrap(),
            vec!["a".into()],
        )
        .unwrap();
        let mut file = FileBuilder::new();
        let mut group = file.create_group("attrs");
        write_artifact(&mut group, &Artifact::Instantaneous(attrs)).unwrap();
        file.add_group(group.finish());
        let path =
            std::env::temp_dir().join(format!("modewise-export-inst-{}.h5", std::process::id()));
        file.write(&path).unwrap();

        let h5 = File::open(&path).unwrap();
        assert_eq!(
            h5.dataset("attrs/amplitude").unwrap().shape().unwrap(),
            [2, 1, 3]
        );
        assert_eq!(
            h5.dataset("attrs/frequency_hz")
                .unwrap()
                .read_f64()
                .unwrap(),
            [0.2; 6]
        );
        assert_eq!(
            h5.dataset("attrs/centre_frequencies_hz")
                .unwrap()
                .read_f64()
                .unwrap(),
            [0.1, 0.3]
        );
        assert!(h5.dataset("attrs/labels").is_ok());
    }

    #[test]
    fn writes_per_channel_centre_frequencies() {
        use modewise_decomposition::Decomposition;
        use modewise_signals::SamplingRate;
        use ndarray::{Array3, array};

        let d = Decomposition::per_channel(
            Array3::zeros((2, 2, 4)),
            array![[0.1, 0.2], [0.3, 0.5]],
            SamplingRate::new(1.0).unwrap(),
            vec!["a".into(), "b".into()],
        )
        .unwrap();
        let mut file = FileBuilder::new();
        let mut group = file.create_group("modes");
        write_artifact(&mut group, &Artifact::Decomposition(d)).unwrap();
        file.add_group(group.finish());
        let path =
            std::env::temp_dir().join(format!("modewise-export-vmd-{}.h5", std::process::id()));
        file.write(&path).unwrap();

        let h5 = File::open(&path).unwrap();
        let per_channel = h5.dataset("modes/channel_centre_frequencies_hz").unwrap();
        assert_eq!(per_channel.shape().unwrap(), [2, 2]);
        assert_eq!(per_channel.read_f64().unwrap(), [0.1, 0.2, 0.3, 0.5]);
        assert_eq!(
            h5.dataset("modes/centre_frequencies_hz")
                .unwrap()
                .read_f64()
                .unwrap(),
            [0.15000000000000002, 0.4]
        );
    }

    #[test]
    fn writes_connectivity_with_windows_and_flags() {
        use modewise_connectivity::{FunctionalConnectivity, Measure, WindowInfo};
        use ndarray::{Array2, Array4};

        let labels = vec!["a".to_string(), "b".to_string()];
        let c = FunctionalConnectivity::new(
            Array4::ones((2, 3, 2, 2)),
            Measure::PhaseLockingValue,
            labels,
        )
        .unwrap()
        .with_centre_frequencies(vec![0.01, 0.05, 0.1])
        .unwrap()
        .with_windows(
            (0..2)
                .map(|i| WindowInfo {
                    label: "x".into(),
                    event_onset_s: i as f64,
                })
                .collect(),
            Some(Array2::from_elem((2, 3), true)),
        )
        .unwrap();
        let mut file = FileBuilder::new();
        let mut group = file.create_group("fc");
        write_artifact(&mut group, &Artifact::FunctionalConnectivity(c)).unwrap();
        file.add_group(group.finish());
        let path =
            std::env::temp_dir().join(format!("modewise-export-fc-{}.h5", std::process::id()));
        file.write(&path).unwrap();
        let h5 = File::open(&path).unwrap();
        assert_eq!(
            h5.dataset("fc/values").unwrap().shape().unwrap(),
            [2, 3, 2, 2]
        );
        assert_eq!(
            h5.dataset("fc/resolvable").unwrap().shape().unwrap(),
            [2, 3]
        );
        for name in [
            "fc/centre_frequencies_hz",
            "fc/window_labels",
            "fc/window_onsets_s",
            "fc/labels",
        ] {
            assert!(h5.dataset(name).is_ok(), "{name}");
        }
        assert!(
            h5.group("fc")
                .unwrap()
                .attrs()
                .unwrap()
                .contains_key("measure")
        );
    }

    #[test]
    fn writes_channel_tables_as_string_columns() {
        use modewise_brain::{ChannelInfo, ChannelTable, Hemisphere, Region};

        let table = ChannelTable::new(vec![
            ChannelInfo::new("a")
                .with_region(Region::new("PFCm"))
                .with_hemisphere(Hemisphere::Left)
                .with_tag("yeo17", "DefaultB"),
            ChannelInfo::new("vermis"),
        ])
        .unwrap();
        let mut file = FileBuilder::new();
        let mut group = file.create_group("ch");
        write_artifact(&mut group, &Artifact::Channels(table)).unwrap();
        file.add_group(group.finish());
        let path =
            std::env::temp_dir().join(format!("modewise-export-ch-{}.h5", std::process::id()));
        file.write(&path).unwrap();
        let h5 = File::open(&path).unwrap();
        for name in ["ch/labels", "ch/regions", "ch/hemispheres", "ch/tag_yeo17"] {
            assert!(h5.dataset(name).is_ok(), "{name}");
        }
    }

    #[test]
    fn writes_loadings_with_both_label_sets() {
        use modewise_decomposition::Loadings;
        use ndarray::Array2;

        let loadings = Loadings::new(
            Array2::from_shape_fn((2, 3), |(l, c)| (l * 3 + c) as f64),
            vec!["latent_0".into(), "latent_1".into()],
            vec!["a".into(), "b".into(), "c".into()],
        )
        .unwrap();
        let mut file = FileBuilder::new();
        let mut group = file.create_group("a");
        write_artifact(&mut group, &Artifact::Loadings(loadings)).unwrap();
        file.add_group(group.finish());
        let path = std::env::temp_dir().join(format!(
            "modewise-export-loadings-{}.h5",
            std::process::id()
        ));
        file.write(&path).unwrap();
        let h5 = File::open(&path).unwrap();
        assert_eq!(h5.dataset("a/values").unwrap().shape().unwrap(), [2, 3]);
        assert!(h5.dataset("a/latent_labels").is_ok() && h5.dataset("a/labels").is_ok());
    }

    #[test]
    fn writes_time_frequency_mask_and_sampling_rate() {
        use modewise_signals::SamplingRate;
        use modewise_spectral::TimeFrequencyMap;
        use ndarray::{Array2, Array3};

        let map = TimeFrequencyMap::new(
            Array3::zeros((1, 2, 3)),
            vec![0.1, 0.2],
            vec![0.0, 0.5, 1.0],
            vec!["a".into()],
        )
        .unwrap()
        .with_sampling_rate(SamplingRate::new(2.0).unwrap())
        .unwrap()
        .with_reliable(Array2::from_elem((2, 3), true))
        .unwrap();
        let mut file = FileBuilder::new();
        let mut group = file.create_group("tf");
        write_artifact(&mut group, &Artifact::TimeFrequency(map)).unwrap();
        file.add_group(group.finish());
        let path =
            std::env::temp_dir().join(format!("modewise-export-tf-{}.h5", std::process::id()));
        file.write(&path).unwrap();
        let h5 = File::open(&path).unwrap();
        assert_eq!(h5.dataset("tf/reliable").unwrap().shape().unwrap(), [2, 3]);
        assert!(
            h5.group("tf")
                .unwrap()
                .attrs()
                .unwrap()
                .contains_key("sampling_rate_hz")
        );
    }
}
