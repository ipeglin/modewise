//! Tab-separated time series: a header row, one column per series and one row per sample, as
//! in BIDS derivatives such as fMRIPrep's `*_desc-confounds_timeseries.tsv`.

use std::path::Path;

use modewise_signals::{MultichannelSeries, SamplingRate};
use ndarray::Array2;

/// Reads the columns matching `patterns` (names, or globs with `*` and `?`), in pattern order
/// and then file order. `n/a` in the first row (derivatives and framewise displacement have no
/// first value) takes the second row's value, as nilearn's `load_confounds` does; `n/a`
/// anywhere else is an error.
pub fn read(
    path: &Path,
    patterns: &[String],
    fs: SamplingRate,
) -> Result<MultichannelSeries, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("time series file {}: {e}", path.display()))?;
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let header: Vec<&str> = lines
        .next()
        .ok_or_else(|| format!("{} is empty", path.display()))?
        .split('\t')
        .map(str::trim)
        .collect();
    let mut chosen: Vec<usize> = Vec::new();
    for pattern in patterns {
        let found: Vec<usize> = (0..header.len())
            .filter(|&i| matches(pattern, header[i]))
            .collect();
        if found.is_empty() {
            return Err(format!(
                "{}: no column matches {pattern:?} (columns: {})",
                path.display(),
                header.join(", ")
            ));
        }
        for i in found {
            if !chosen.contains(&i) {
                chosen.push(i);
            }
        }
    }

    let rows: Vec<Vec<&str>> = lines
        .map(|l| l.split('\t').map(str::trim).collect())
        .collect();
    let mut data = Array2::zeros((chosen.len(), rows.len()));
    for (r, cells) in rows.iter().enumerate() {
        for (c, &column) in chosen.iter().enumerate() {
            let place = || {
                format!(
                    "{} row {}, column {:?}",
                    path.display(),
                    r + 1,
                    header[column]
                )
            };
            let mut cell = *cells
                .get(column)
                .ok_or_else(|| format!("{}: missing", place()))?;
            if cell == "n/a" && r == 0 {
                cell = rows
                    .get(1)
                    .and_then(|next| next.get(column))
                    .copied()
                    .unwrap_or(cell);
            }
            data[[c, r]] = match cell {
                "n/a" => return Err(format!("{}: n/a is only filled in the first row", place())),
                value => value
                    .parse()
                    .map_err(|_| format!("{}: {value:?} is not a number", place()))?,
            };
        }
    }
    let labels = chosen.iter().map(|&i| header[i].to_string()).collect();
    MultichannelSeries::new(data, fs, labels).map_err(|e| format!("{}: {e}", path.display()))
}

/// Shell-style match: `*` is any run of characters, `?` any one character.
fn matches(pattern: &str, name: &str) -> bool {
    let (p, n): (Vec<char>, Vec<char>) = (pattern.chars().collect(), name.chars().collect());
    let (mut i, mut j, mut star, mut mark) = (0, 0, None, 0);
    while j < n.len() {
        if i < p.len() && (p[i] == '?' || p[i] == n[j]) {
            i += 1;
            j += 1;
        } else if i < p.len() && p[i] == '*' {
            star = Some(i);
            mark = j;
            i += 1;
        } else if let Some(s) = star {
            i = s + 1;
            mark += 1;
            j = mark;
        } else {
            return false;
        }
    }
    p[i..].iter().all(|&c| c == '*')
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    fn write(text: &str) -> std::path::PathBuf {
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "modewise-timeseries-{}-{n}.tsv",
            std::process::id()
        ));
        std::fs::write(&path, text).unwrap();
        path
    }

    fn hz() -> SamplingRate {
        SamplingRate::new(1.25).unwrap()
    }

    const CONFOUNDS: &str = "global_signal\ttrans_x\ttrans_x_derivative1\ttrans_y\tframewise_displacement\n\
        10\t0.1\tn/a\t0.2\tn/a\n\
        11\t0.3\t0.2\t0.1\t0.4\n\
        12\t0.2\t-0.1\t0.0\t0.3\n";

    #[test]
    fn globs_choose_columns_in_pattern_then_file_order() {
        let file = write(CONFOUNDS);
        let patterns = ["framewise_displacement".to_string(), "trans_*".to_string()];
        let s = read(&file, &patterns, hz()).unwrap();
        assert_eq!(
            s.labels(),
            [
                "framewise_displacement",
                "trans_x",
                "trans_x_derivative1",
                "trans_y"
            ]
        );
        assert_eq!(s.n_samples(), 3);
        assert_eq!(s.sampling_rate().hz(), 1.25);
    }

    #[test]
    fn first_row_n_a_takes_the_second_value() {
        let file = write(CONFOUNDS);
        let s = read(&file, &["*derivative1".into(), "trans_x".into()], hz()).unwrap();
        assert_eq!(s.data().row(0).to_vec(), vec![0.2, 0.2, -0.1]);
        assert_eq!(s.data().row(1).to_vec(), vec![0.1, 0.3, 0.2]);
    }

    #[test]
    fn a_column_matched_twice_is_kept_once() {
        let file = write(CONFOUNDS);
        let s = read(&file, &["trans_x".into(), "trans_?".into()], hz()).unwrap();
        assert_eq!(s.labels(), ["trans_x", "trans_y"]);
    }

    #[test]
    fn errors_name_the_problem() {
        let cases = [
            (CONFOUNDS, "rot_*", "no column matches \"rot_*\""),
            ("a\n1\nn/a\n", "a", "n/a is only filled in the first row"),
            ("a\nx\n", "a", "\"x\" is not a number"),
            ("a\tb\n1\n", "b", "missing"),
            ("", "a", "is empty"),
        ];
        for (text, pattern, message) in cases {
            let file = write(text);
            let err = read(&file, &[pattern.into()], hz()).unwrap_err();
            assert!(err.contains(message), "{err}");
        }
        let err = read(Path::new("/no/such.tsv"), &["a".into()], hz()).unwrap_err();
        assert!(err.contains("time series file"), "{err}");
    }

    #[test]
    fn glob_matching() {
        for (pattern, name, expected) in [
            ("a*", "abc", true),
            ("*c", "abc", true),
            ("a?c", "abc", true),
            ("a*b*c", "axxbyyc", true),
            ("a*d", "abc", false),
            ("abc", "ab", false),
            ("*", "", true),
        ] {
            assert_eq!(matches(pattern, name), expected, "{pattern} {name}");
        }
    }
}
