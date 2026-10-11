//! What each channel is: its name, region, hemisphere and other labels, and selections over them.
//!
//! Modality-generic: a channel can be an fMRI parcel, an EEG electrode or an fNIRS optode.
//! Modelled on the parcel labels of an earlier rewrite of the thesis analysis code.

mod selection;
mod table;

pub use selection::{Pattern, Rule, Selection};
pub use table::{ChannelError, ChannelInfo, ChannelTable, Hemisphere, Region};
