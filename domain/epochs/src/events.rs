use thiserror::Error;

#[derive(Debug, Error, PartialEq)]
pub enum EventsError {
    #[error("event {index}: onset must be finite, got {value}")]
    InvalidOnset { index: usize, value: f64 },
    #[error("event {index}: duration must be finite and >= 0, got {value}")]
    InvalidDuration { index: usize, value: f64 },
    #[error("event {index}: label must not be empty")]
    EmptyLabel { index: usize },
}

/// One event, as in a BIDS `events.tsv` row: onset and duration in seconds.
#[derive(Debug, Clone, PartialEq)]
pub struct Event {
    pub onset_s: f64,
    pub duration_s: f64,
    pub label: String,
}

/// Events sorted by onset.
#[derive(Debug, Clone, PartialEq)]
pub struct Events(Vec<Event>);

impl Events {
    pub fn new(mut events: Vec<Event>) -> Result<Self, EventsError> {
        for (index, e) in events.iter().enumerate() {
            if !e.onset_s.is_finite() {
                return Err(EventsError::InvalidOnset {
                    index,
                    value: e.onset_s,
                });
            }
            if !(e.duration_s.is_finite() && e.duration_s >= 0.0) {
                return Err(EventsError::InvalidDuration {
                    index,
                    value: e.duration_s,
                });
            }
            if e.label.is_empty() {
                return Err(EventsError::EmptyLabel { index });
            }
        }
        events.sort_by(|a, b| a.onset_s.total_cmp(&b.onset_s));
        Ok(Self(events))
    }

    pub fn as_slice(&self) -> &[Event] {
        &self.0
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Merges consecutive events with the same label into one when the next one starts
    /// within `max_gap_s` of the previous end, for example trials into task blocks.
    pub fn merge_consecutive(&self, max_gap_s: f64) -> Self {
        let mut merged: Vec<Event> = Vec::new();
        for e in &self.0 {
            match merged.last_mut() {
                Some(last)
                    if last.label == e.label
                        && e.onset_s - (last.onset_s + last.duration_s) <= max_gap_s =>
                {
                    let end = (last.onset_s + last.duration_s).max(e.onset_s + e.duration_s);
                    last.duration_s = end - last.onset_s;
                }
                _ => merged.push(e.clone()),
            }
        }
        Self(merged)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ev(onset_s: f64, duration_s: f64, label: &str) -> Event {
        Event {
            onset_s,
            duration_s,
            label: label.into(),
        }
    }

    #[test]
    fn sorts_by_onset_and_validates() {
        let events = Events::new(vec![ev(5.0, 1.0, "b"), ev(1.0, 0.0, "a")]).unwrap();
        assert_eq!(events.as_slice()[0].label, "a");
        assert_eq!(events.len(), 2);
        assert!(!events.is_empty());
        assert!(Events::new(vec![]).unwrap().is_empty());
        assert!(matches!(
            Events::new(vec![ev(f64::NAN, 1.0, "a")]),
            Err(EventsError::InvalidOnset { .. })
        ));
        assert!(matches!(
            Events::new(vec![ev(0.0, -1.0, "a")]),
            Err(EventsError::InvalidDuration { .. })
        ));
        assert!(matches!(
            Events::new(vec![ev(0.0, f64::INFINITY, "a")]),
            Err(EventsError::InvalidDuration { .. })
        ));
        assert_eq!(
            Events::new(vec![ev(0.0, 1.0, "")]),
            Err(EventsError::EmptyLabel { index: 0 })
        );
    }

    #[test]
    fn merging_uses_the_gap_in_seconds_and_keeps_the_longest_end() {
        let gap_two = Events::new(vec![ev(96.0, 2.0, "a"), ev(100.0, 2.0, "a")]).unwrap();
        assert_eq!(
            gap_two.merge_consecutive(1.5).len(),
            2,
            "a 2 s gap is larger than 1.5 s"
        );
        let nested = Events::new(vec![ev(0.0, 10.0, "a"), ev(2.0, 1.0, "a")]).unwrap();
        let merged = nested.merge_consecutive(0.0);
        assert_eq!(
            (merged.len(), merged.as_slice()[0].duration_s),
            (1, 10.0),
            "a nested event does not shorten the block"
        );
    }

    #[test]
    fn merges_trials_into_blocks() {
        let trials = Events::new(vec![
            ev(0.0, 2.0, "face"),
            ev(3.0, 2.0, "face"),
            ev(6.0, 2.0, "face"),
            ev(20.0, 2.0, "shape"),
            ev(23.0, 2.0, "shape"),
            ev(40.0, 2.0, "face"),
        ])
        .unwrap();
        let blocks = trials.merge_consecutive(1.5);
        let summary: Vec<(f64, f64, &str)> = blocks
            .as_slice()
            .iter()
            .map(|e| (e.onset_s, e.duration_s, e.label.as_str()))
            .collect();
        assert_eq!(
            summary,
            [
                (0.0, 8.0, "face"),
                (20.0, 5.0, "shape"),
                (40.0, 2.0, "face")
            ]
        );
        assert_eq!(
            trials.merge_consecutive(0.5).len(),
            6,
            "gaps of 1 s exceed 0.5 s"
        );
    }
}
