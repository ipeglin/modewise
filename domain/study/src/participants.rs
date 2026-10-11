use std::collections::BTreeMap;

use crate::StudyError;

/// Information about each subject (group, age, scores), keyed by the subject entity of the
/// recordings. Values are kept as text; numbers are parsed when asked for. A value that a table
/// marks as missing is simply absent.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Participants {
    columns: Vec<String>,
    rows: BTreeMap<String, BTreeMap<String, String>>,
}

impl Participants {
    /// One table: its column names and, per subject, the present values.
    pub fn new(
        columns: Vec<String>,
        rows: BTreeMap<String, BTreeMap<String, String>>,
    ) -> Result<Self, StudyError> {
        if let Some(unknown) = rows
            .values()
            .flat_map(|r| r.keys())
            .find(|k| !columns.contains(k))
        {
            return Err(StudyError::UnknownColumn(unknown.clone()));
        }
        let mut seen = std::collections::BTreeSet::new();
        if let Some(dup) = columns.iter().find(|c| !seen.insert(*c)) {
            return Err(StudyError::DuplicateColumn(dup.clone()));
        }
        Ok(Self { columns, rows })
    }

    /// Joins on subject. Subjects in either table are kept; column names must not clash.
    pub fn join(mut self, other: Self) -> Result<Self, StudyError> {
        if let Some(clash) = other.columns.iter().find(|c| self.columns.contains(c)) {
            return Err(StudyError::DuplicateColumn(clash.clone()));
        }
        self.columns.extend(other.columns);
        for (subject, values) in other.rows {
            self.rows.entry(subject).or_default().extend(values);
        }
        Ok(self)
    }

    pub fn columns(&self) -> &[String] {
        &self.columns
    }

    pub fn subjects(&self) -> impl Iterator<Item = &str> {
        self.rows.keys().map(String::as_str)
    }

    pub fn text(&self, subject: &str, column: &str) -> Result<Option<&str>, StudyError> {
        if !self.columns.iter().any(|c| c == column) {
            return Err(StudyError::UnknownColumn(column.to_string()));
        }
        Ok(self
            .rows
            .get(subject)
            .and_then(|r| r.get(column))
            .map(String::as_str))
    }

    pub fn number(&self, subject: &str, column: &str) -> Result<Option<f64>, StudyError> {
        match self.text(subject, column)? {
            None => Ok(None),
            Some(v) => v
                .trim()
                .parse()
                .map(Some)
                .map_err(|_| StudyError::NotANumber {
                    subject: subject.to_string(),
                    column: column.to_string(),
                    value: v.to_string(),
                }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table(columns: &[&str], rows: &[(&str, &[(&str, &str)])]) -> Participants {
        let rows = rows
            .iter()
            .map(|(s, values)| {
                (
                    s.to_string(),
                    values
                        .iter()
                        .map(|(k, v)| (k.to_string(), v.to_string()))
                        .collect(),
                )
            })
            .collect();
        Participants::new(columns.iter().map(|c| c.to_string()).collect(), rows).unwrap()
    }

    #[test]
    fn values_are_text_numbers_or_absent() {
        let p = table(
            &["group", "age"],
            &[
                ("01", &[("group", "Patient"), ("age", "20.5")]),
                ("02", &[("group", "Control")]),
            ],
        );
        assert_eq!(p.text("01", "group").unwrap(), Some("Patient"));
        assert_eq!(p.number("01", "age").unwrap(), Some(20.5));
        assert_eq!(p.number("02", "age").unwrap(), None, "missing value");
        assert_eq!(p.number("99", "age").unwrap(), None, "unknown subject");
        assert!(matches!(
            p.number("01", "group"),
            Err(StudyError::NotANumber { .. })
        ));
        assert_eq!(
            p.text("01", "height").unwrap_err(),
            StudyError::UnknownColumn("height".into())
        );
        assert_eq!(p.subjects().collect::<Vec<_>>(), ["01", "02"]);
        assert_eq!(p.columns(), ["group", "age"]);
    }

    #[test]
    fn tables_join_on_subject_and_refuse_clashing_columns() {
        let base = table(&["group"], &[("01", &[("group", "Patient")])]);
        let scores = table(
            &["shaps"],
            &[("01", &[("shaps", "4")]), ("03", &[("shaps", "1")])],
        );
        let joined = base.clone().join(scores).unwrap();
        assert_eq!(joined.number("01", "shaps").unwrap(), Some(4.0));
        assert_eq!(joined.text("03", "group").unwrap(), None);
        assert_eq!(joined.subjects().count(), 2);
        assert_eq!(
            base.clone().join(base).unwrap_err(),
            StudyError::DuplicateColumn("group".into())
        );
        let bad_rows = BTreeMap::from([(
            "01".to_string(),
            BTreeMap::from([("x".to_string(), "1".to_string())]),
        )]);
        assert!(Participants::new(vec!["y".into()], bad_rows).is_err());
        assert!(Participants::new(vec!["y".into(), "y".into()], BTreeMap::new()).is_err());
    }
}
