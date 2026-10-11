#![allow(clippy::unwrap_used)]

use std::path::PathBuf;

use hdf5_pure::FileBuilder;
use modewise_hdf5::{Hdf5Error, Layout, read_series};
use modewise_signals::SamplingRate;

fn write_file(name: &str, build: impl FnOnce(&mut FileBuilder)) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("modewise-hdf5-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(name);
    let mut builder = FileBuilder::new();
    build(&mut builder);
    builder.write(&path).unwrap();
    path
}

fn fs() -> SamplingRate {
    SamplingRate::new(1.25).unwrap()
}

#[test]
fn reads_f32_channels_by_time_with_default_labels() {
    let path = write_file("f32.h5", |b| {
        let mut g = b.create_group("01fmri_parcellation");
        g.create_dataset("full_run_std")
            .with_f32_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .with_shape(&[2, 3]);
        b.add_group(g.finish());
    });
    let series = read_series(
        &path,
        "01fmri_parcellation/full_run_std",
        Layout::ChannelsByTime,
        fs(),
        None,
    )
    .unwrap();
    assert_eq!(series.n_channels(), 2);
    assert_eq!(series.data().row(1).to_vec(), [4.0, 5.0, 6.0]);
    assert_eq!(series.labels(), ["ch0", "ch1"]);
    assert_eq!(series.sampling_rate(), fs());
}

#[test]
fn transposes_time_by_channels() {
    let path = write_file("tbc.h5", |b| {
        b.create_dataset("x")
            .with_f64_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .with_shape(&[3, 2]);
    });
    let labels = Some(vec!["a".to_string(), "b".to_string()]);
    let series = read_series(&path, "x", Layout::TimeByChannels, fs(), labels).unwrap();
    assert_eq!(series.data().row(0).to_vec(), [1.0, 3.0, 5.0]);
    assert_eq!(series.labels(), ["a", "b"]);
}

#[test]
fn rejects_bad_files_and_datasets() {
    let path = write_file("bad.h5", |b| {
        b.create_dataset("flat")
            .with_f64_data(&[1.0, 2.0, 3.0])
            .with_shape(&[3]);
        b.create_dataset("nan")
            .with_f64_data(&[1.0, f64::NAN])
            .with_shape(&[1, 2]);
    });
    let read = |name: &str| read_series(&path, name, Layout::ChannelsByTime, fs(), None);
    assert!(matches!(
        read("flat"),
        Err(Hdf5Error::NotTwoDimensional { .. })
    ));
    assert!(matches!(read("missing"), Err(Hdf5Error::Dataset { .. })));
    assert!(matches!(read("nan"), Err(Hdf5Error::Signal(_))));
    let missing_file = read_series(
        &path.with_file_name("none.h5"),
        "x",
        Layout::ChannelsByTime,
        fs(),
        None,
    );
    assert!(matches!(missing_file, Err(Hdf5Error::Open { .. })));
}

#[test]
fn rejects_text_datasets() {
    let path = write_file("text.h5", |b| {
        b.create_dataset("names").with_strings(&["a", "b"]).unwrap();
    });
    let err = read_series(&path, "names", Layout::ChannelsByTime, fs(), None);
    assert!(matches!(err, Err(Hdf5Error::NotNumeric { .. })));
}
