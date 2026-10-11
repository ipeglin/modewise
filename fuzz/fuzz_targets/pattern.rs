//! Any pattern must be rejected or usable: fill and find must not panic.
#![no_main]

use std::collections::BTreeMap;

use libfuzzer_sys::fuzz_target;
use modewise_dataset::Pattern;

fuzz_target!(|data: &[u8]| {
    let Ok(text) = std::str::from_utf8(data) else { return };
    if let Ok(pattern) = Pattern::parse(text) {
        let values: BTreeMap<String, String> = pattern.entities().into_iter().map(|e| (e, "x1".to_string())).collect();
        let _ = pattern.fill(&values);
        let _ = pattern.find(&std::env::temp_dir().join("modewise-fuzz-missing"));
    }
});
