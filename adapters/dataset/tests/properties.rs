#![allow(clippy::unwrap_used)]

use std::collections::BTreeMap;

use modewise_dataset::Pattern;
use proptest::prelude::*;

const PATTERN: &str = "sub-{subject}/ses-{session}/sub-{subject}_ses-{session}_task-{task}_bold.h5";

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

    /// BIDS labels are alphanumeric; any such values must be found again by `find`.
    /// Lowercase only: on case-insensitive filesystems (macOS default) `sub-E` and
    /// `sub-e` are the same folder, so labels differing only in case cannot coexist.
    #[test]
    fn filled_paths_are_found_with_the_same_values(
        recordings in prop::collection::btree_set(("[a-z0-9]{1,6}", "[a-z0-9]{1,4}", "[a-z0-9]{1,8}"), 1..6),
        run in any::<u32>(),
    ) {
        let root = std::env::temp_dir().join(format!("modewise-prop-pattern-{}-{run}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let pattern = Pattern::parse(PATTERN).unwrap();
        let mut expected = Vec::new();
        for (subject, session, task) in &recordings {
            let values = BTreeMap::from([
                ("subject".to_string(), subject.clone()),
                ("session".to_string(), session.clone()),
                ("task".to_string(), task.clone()),
            ]);
            let path = root.join(pattern.fill(&values).unwrap());
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, b"").unwrap();
            expected.push(vec![
                ("subject".to_string(), subject.clone()),
                ("session".to_string(), session.clone()),
                ("task".to_string(), task.clone()),
            ]);
        }
        let mut found: Vec<_> = pattern.find(&root).unwrap().into_iter().map(|(_, e)| e).collect();
        found.sort();
        expected.sort();
        expected.dedup();
        std::fs::remove_dir_all(&root).unwrap();
        prop_assert_eq!(found, expected);
    }
}
