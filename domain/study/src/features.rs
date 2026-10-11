use std::collections::{BTreeMap, BTreeSet};

use ndarray::{Array2, ArrayView1, ArrayView2, Axis};

use crate::{Participants, StudyError};

/// One row of a feature table: a recording, and the condition when features were computed per
/// condition from epochs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Observation {
    pub recording: String,
    pub subject: Option<String>,
    pub condition: Option<String>,
}

/// Observations × named features, with participant information alongside: the handover to
/// statistics and to machine learning.
#[derive(Debug, Clone, PartialEq)]
pub struct FeatureTable {
    observations: Vec<Observation>,
    names: Vec<String>,
    values: Array2<f64>,
    metadata: BTreeMap<String, Vec<Option<String>>>,
}

impl FeatureTable {
    pub fn new(
        observations: Vec<Observation>,
        names: Vec<String>,
        values: Array2<f64>,
    ) -> Result<Self, StudyError> {
        if values.dim() != (observations.len(), names.len()) {
            return Err(StudyError::Shape(format!(
                "values are {:?} but there are {} observations and {} features",
                values.dim(),
                observations.len(),
                names.len()
            )));
        }
        let mut seen = BTreeSet::new();
        if let Some(dup) = names.iter().find(|n| !seen.insert(*n)) {
            return Err(StudyError::DuplicateFeature(dup.clone()));
        }
        Ok(Self {
            observations,
            names,
            values,
            metadata: BTreeMap::new(),
        })
    }

    /// Adds every participants column, matched on each observation's subject.
    pub fn with_participants(mut self, participants: &Participants) -> Result<Self, StudyError> {
        for column in participants.columns() {
            let values = self
                .observations
                .iter()
                .map(|o| match &o.subject {
                    Some(s) => participants.text(s, column).map(|v| v.map(str::to_string)),
                    None => Ok(None),
                })
                .collect::<Result<Vec<_>, _>>()?;
            self.metadata.insert(column.clone(), values);
        }
        Ok(self)
    }

    /// Adds or replaces a metadata column: one value (or none) per observation.
    pub fn with_column(
        mut self,
        name: &str,
        values: Vec<Option<String>>,
    ) -> Result<Self, StudyError> {
        if values.len() != self.observations.len() {
            return Err(StudyError::Shape(format!(
                "column {name:?} has {} values for {} observations",
                values.len(),
                self.observations.len()
            )));
        }
        self.metadata.insert(name.to_string(), values);
        Ok(self)
    }

    pub fn observations(&self) -> &[Observation] {
        &self.observations
    }

    pub fn feature_names(&self) -> &[String] {
        &self.names
    }

    pub fn values(&self) -> ArrayView2<'_, f64> {
        self.values.view()
    }

    pub fn feature(&self, name: &str) -> Option<ArrayView1<'_, f64>> {
        self.names
            .iter()
            .position(|n| n == name)
            .map(|i| self.values.index_axis(Axis(1), i))
    }

    pub fn metadata_columns(&self) -> impl Iterator<Item = &str> {
        self.metadata.keys().map(String::as_str)
    }

    pub fn metadata(&self, column: &str) -> Result<&[Option<String>], StudyError> {
        self.metadata
            .get(column)
            .map(Vec::as_slice)
            .ok_or_else(|| StudyError::UnknownColumn(column.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn obs(recording: &str, subject: Option<&str>, condition: Option<&str>) -> Observation {
        Observation {
            recording: recording.into(),
            subject: subject.map(Into::into),
            condition: condition.map(Into::into),
        }
    }

    #[test]
    fn shapes_and_names_are_checked() {
        let rows = vec![obs("r1", Some("01"), None), obs("r2", Some("02"), None)];
        assert!(FeatureTable::new(rows.clone(), vec!["a".into()], Array2::zeros((2, 2))).is_err());
        assert_eq!(
            FeatureTable::new(
                rows.clone(),
                vec!["a".into(), "a".into()],
                Array2::zeros((2, 2))
            )
            .unwrap_err(),
            StudyError::DuplicateFeature("a".into())
        );
        let t = FeatureTable::new(
            rows,
            vec!["a".into(), "b".into()],
            ndarray::array![[1.0, 2.0], [3.0, 4.0]],
        )
        .unwrap();
        assert_eq!(t.feature("b").unwrap().to_vec(), [2.0, 4.0]);
        assert!(t.feature("c").is_none());
        assert_eq!(
            (
                t.observations().len(),
                t.feature_names().len(),
                t.values().dim()
            ),
            (2, 2, (2, 2))
        );
    }

    #[test]
    fn participants_join_on_each_observations_subject() {
        let p = Participants::new(
            vec!["group".into()],
            BTreeMap::from([(
                "01".to_string(),
                BTreeMap::from([("group".to_string(), "Patient".to_string())]),
            )]),
        )
        .unwrap();
        let rows = vec![
            obs("r1", Some("01"), Some("face")),
            obs("r1", Some("01"), Some("shape")),
            obs("r2", Some("02"), None),
            obs("r3", None, None),
        ];
        let t = FeatureTable::new(rows, vec!["a".into()], Array2::zeros((4, 1)))
            .unwrap()
            .with_participants(&p)
            .unwrap();
        assert_eq!(
            t.metadata("group").unwrap(),
            [
                Some("Patient".to_string()),
                Some("Patient".to_string()),
                None,
                None
            ]
        );
        assert_eq!(t.metadata_columns().collect::<Vec<_>>(), ["group"]);
        assert!(t.metadata("age").is_err());
    }

    #[test]
    fn columns_can_be_added_and_replaced_but_must_fit() {
        let rows = vec![
            Observation {
                recording: "r1".into(),
                subject: Some("01".into()),
                condition: None,
            },
            Observation {
                recording: "r2".into(),
                subject: Some("02".into()),
                condition: None,
            },
        ];
        let t = FeatureTable::new(rows, vec!["x".into()], ndarray::array![[1.0], [2.0]]).unwrap();
        let t = t
            .with_column("group", vec![Some("a".into()), Some("b".into())])
            .unwrap();
        let t = t
            .with_column("group", vec![Some("c".into()), None])
            .unwrap();
        assert_eq!(t.metadata("group").unwrap(), [Some("c".to_string()), None]);
        assert!(t.with_column("age", vec![None]).is_err());
    }
}
