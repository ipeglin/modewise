//! BIDS `events.tsv` reader. Onsets and durations are seconds from the first stored
//! sample (BIDS specification, task events).

use std::path::Path;

use modewise_epochs::{Event, Events};

pub fn read(path: &Path, label_column: &str) -> Result<Events, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("events file {}: {e}", path.display()))?;
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let header: Vec<&str> = lines
        .next()
        .ok_or_else(|| format!("{} is empty", path.display()))?
        .split('\t')
        .map(str::trim)
        .collect();
    let column = |name: &str| {
        header.iter().position(|h| *h == name).ok_or_else(|| {
            format!(
                "{} has no {name:?} column (columns: {})",
                path.display(),
                header.join(", ")
            )
        })
    };
    let (onset, duration, label) = (column("onset")?, column("duration")?, column(label_column)?);

    let mut events = Vec::new();
    for (row, line) in lines.enumerate() {
        let cells: Vec<&str> = line.split('\t').map(str::trim).collect();
        let cell = |i: usize, name: &str| {
            cells
                .get(i)
                .copied()
                .ok_or_else(|| format!("{} row {}: missing {name}", path.display(), row + 1))
        };
        let number = |i: usize, name: &str| -> Result<f64, String> {
            let value = cell(i, name)?;
            if value == "n/a" {
                return Err(format!(
                    "{} row {}: {name} is n/a; such rows cannot be epoched",
                    path.display(),
                    row + 1
                ));
            }
            value.parse().map_err(|_| {
                format!(
                    "{} row {}: {name} {value:?} is not a number",
                    path.display(),
                    row + 1
                )
            })
        };
        events.push(Event {
            onset_s: number(onset, "onset")?,
            duration_s: number(duration, "duration")?,
            label: cell(label, label_column)?.to_string(),
        });
    }
    Events::new(events).map_err(|e| format!("{}: {e}", path.display()))
}
