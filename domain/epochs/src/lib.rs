//! Events and epochs: cutting full-run results into windows around events.
//!
//! Rules and sources: `docs/epochs.md`. Times follow BIDS: sample `i` of a
//! recording is at `i / fs` seconds, and event onsets are in seconds from the
//! first stored sample.

mod epochs;
mod events;

pub use epochs::{DroppedEvent, EpochError, EpochParams, EpochSource, EpochWindow, Epochs, Window};
pub use events::{Event, Events, EventsError};
