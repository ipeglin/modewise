//! Any text must be rejected or discovered against an empty folder, never panic.
#![no_main]

use libfuzzer_sys::fuzz_target;
use modewise_dataset::{Dataset, DatasetSpec};

fuzz_target!(|data: &[u8]| {
    let Ok(text) = std::str::from_utf8(data) else { return };
    if let Ok(spec) = DatasetSpec::from_toml(text) {
        let empty = std::env::temp_dir().join("modewise-fuzz-empty");
        let _ = std::fs::create_dir_all(&empty);
        let _ = Dataset::discover(spec, &empty);
    }
});
