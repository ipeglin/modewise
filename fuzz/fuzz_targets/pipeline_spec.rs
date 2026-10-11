//! Any text, as TOML or JSON, must be rejected or planned, never panic.
#![no_main]

use libfuzzer_sys::fuzz_target;
use modewise_operations::builtin_registry;
use modewise_pipeline::{PipelineSpec, Plan};

fuzz_target!(|data: &[u8]| {
    let Ok(text) = std::str::from_utf8(data) else { return };
    let registry = builtin_registry();
    for spec in [PipelineSpec::from_toml(text), PipelineSpec::from_json(text)].into_iter().flatten() {
        let _ = Plan::new(&spec, &registry);
    }
});
