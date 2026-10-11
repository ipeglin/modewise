use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use regex::Regex;

use crate::DatasetError;

/// A matched file and its entity values, in pattern order.
pub type Match = (PathBuf, Vec<(String, String)>);

/// A relative path with `{entity}` placeholders.
#[derive(Debug, Clone)]
pub struct Pattern {
    text: String,
    segments: Vec<Segment>,
}

#[derive(Debug, Clone)]
struct Segment {
    regex: Regex,
    /// Placeholder names, in capture-group order.
    names: Vec<String>,
}

impl Pattern {
    pub fn parse(text: &str) -> Result<Self, DatasetError> {
        let error = |message: &str| DatasetError::Pattern {
            pattern: text.to_string(),
            message: message.to_string(),
        };
        if text.is_empty() || text.starts_with('/') {
            return Err(error("must be a non-empty relative path"));
        }
        let placeholder = Regex::new(r"\{([^{}]*)\}").map_err(|e| error(&e.to_string()))?;
        let valid_name =
            Regex::new(r"^[A-Za-z_][A-Za-z0-9_]*$").map_err(|e| error(&e.to_string()))?;
        let segments = text
            .split('/')
            .map(|segment| {
                let mut source = String::from("^");
                let mut names = Vec::new();
                let mut last = 0;
                for capture in placeholder.captures_iter(segment) {
                    let (Some(whole), Some(name)) = (capture.get(0), capture.get(1)) else {
                        continue;
                    };
                    if !valid_name.is_match(name.as_str()) {
                        return Err(error(&format!("invalid placeholder {{{}}}", name.as_str())));
                    }
                    source.push_str(&regex::escape(&segment[last..whole.start()]));
                    source.push_str("([^/]+?)");
                    names.push(name.as_str().to_string());
                    last = whole.end();
                }
                let rest = &segment[last..];
                if rest.contains('{') || rest.contains('}') {
                    return Err(error("unbalanced braces"));
                }
                source.push_str(&regex::escape(rest));
                source.push('$');
                let regex = Regex::new(&source).map_err(|e| error(&e.to_string()))?;
                Ok(Segment { regex, names })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            text: text.to_string(),
            segments,
        })
    }

    /// Placeholder names in order of first appearance.
    /// Entity values from text matching a one-segment pattern, such as `sub-{subject}` against
    /// `sub-01`. `None` when the text does not match or the pattern has several segments.
    pub fn capture(&self, text: &str) -> Option<BTreeMap<String, String>> {
        let [segment] = self.segments.as_slice() else {
            return None;
        };
        let captures = segment.regex.captures(text)?;
        Some(
            segment
                .names
                .iter()
                .enumerate()
                .map(|(i, name)| {
                    (
                        name.clone(),
                        captures.get(i + 1).map_or("", |m| m.as_str()).to_string(),
                    )
                })
                .collect(),
        )
    }

    pub fn entities(&self) -> Vec<String> {
        let mut seen = Vec::new();
        for name in self.segments.iter().flat_map(|s| &s.names) {
            if !seen.contains(name) {
                seen.push(name.clone());
            }
        }
        seen
    }

    /// Every file under `root` that matches, sorted by path, with its entity values.
    pub fn find(&self, root: &Path) -> Result<Vec<Match>, DatasetError> {
        let mut found = Vec::new();
        self.walk(root, 0, &BTreeMap::new(), &mut found)?;
        found.sort_by(|a, b| a.0.cmp(&b.0));
        let order = self.entities();
        Ok(found
            .into_iter()
            .map(|(path, values)| {
                let ordered = order
                    .iter()
                    .map(|k| (k.clone(), values[k].clone()))
                    .collect();
                (path, ordered)
            })
            .collect())
    }

    /// The path for given entity values.
    pub fn fill(&self, values: &BTreeMap<String, String>) -> Result<PathBuf, DatasetError> {
        let mut path = self.text.clone();
        for name in self.entities() {
            let value = values.get(&name).ok_or_else(|| DatasetError::Pattern {
                pattern: self.text.clone(),
                message: format!("no value for {{{name}}}"),
            })?;
            path = path.replace(&format!("{{{name}}}"), value);
        }
        Ok(PathBuf::from(path))
    }

    fn walk(
        &self,
        dir: &Path,
        depth: usize,
        values: &BTreeMap<String, String>,
        found: &mut Vec<(PathBuf, BTreeMap<String, String>)>,
    ) -> Result<(), DatasetError> {
        let segment = &self.segments[depth];
        let last = depth + 1 == self.segments.len();
        let entries = match std::fs::read_dir(dir) {
            Ok(entries) => entries,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(e) => {
                return Err(DatasetError::Io {
                    path: dir.display().to_string(),
                    message: e.to_string(),
                });
            }
        };
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            let Some(captures) = segment.regex.captures(&name) else {
                continue;
            };
            let mut values = values.clone();
            let consistent = segment.names.iter().enumerate().all(|(i, key)| {
                let value = captures.get(i + 1).map_or("", |m| m.as_str()).to_string();
                values.entry(key.clone()).or_insert_with(|| value.clone()) == &value
            });
            if !consistent {
                continue;
            }
            let path = entry.path();
            if last {
                if path.is_file() {
                    found.push((path, values));
                }
            } else if path.is_dir() {
                self.walk(&path, depth + 1, &values, found)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lists_entities_in_order_and_fills_paths() {
        let p = Pattern::parse("sub-{subject}/func/sub-{subject}_task-{task}_bold.h5").unwrap();
        assert_eq!(p.entities(), ["subject", "task"]);
        let values = BTreeMap::from([
            ("subject".into(), "01".into()),
            ("task".into(), "rest".into()),
        ]);
        assert_eq!(
            p.fill(&values).unwrap(),
            PathBuf::from("sub-01/func/sub-01_task-rest_bold.h5")
        );
    }

    #[test]
    fn fill_needs_every_entity() {
        let p = Pattern::parse("sub-{subject}_task-{task}.h5").unwrap();
        let values = BTreeMap::from([("subject".to_string(), "01".to_string())]);
        assert!(matches!(p.fill(&values), Err(DatasetError::Pattern { .. })));
    }

    #[cfg(unix)]
    #[test]
    fn unreadable_folder_is_an_error_not_an_empty_result() {
        use std::os::unix::fs::PermissionsExt;
        let root =
            std::env::temp_dir().join(format!("modewise-pattern-perm-{}", std::process::id()));
        let locked = root.join("sub-01");
        std::fs::create_dir_all(&locked).unwrap();
        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o000)).unwrap();
        let result = Pattern::parse("sub-{subject}/x.h5").unwrap().find(&root);
        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o755)).unwrap();
        assert!(matches!(result, Err(DatasetError::Io { .. })), "{result:?}");
    }

    #[test]
    fn rejects_invalid_patterns() {
        for text in ["", "/abs/{x}.h5", "a/{1x}.h5", "a/{x.h5", "a/{}.h5"] {
            assert!(Pattern::parse(text).is_err(), "accepted {text:?}");
        }
    }
}
