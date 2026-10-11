use std::collections::BTreeMap;
use std::sync::Arc;

use crate::Operation;

/// Operations available to pipelines, by name.
#[derive(Default, Clone)]
pub struct Registry {
    operations: BTreeMap<&'static str, Arc<dyn Operation>>,
}

impl Registry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Replaces any operation already registered under the same name.
    pub fn register(&mut self, operation: Arc<dyn Operation>) {
        self.operations
            .insert(operation.descriptor().name, operation);
    }

    pub fn get(&self, name: &str) -> Option<Arc<dyn Operation>> {
        self.operations.get(name).cloned()
    }

    pub fn names(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.operations.keys().copied()
    }
}

#[cfg(test)]
mod tests {
    use crate::test_support::ops;

    #[test]
    fn lists_registered_names_in_order() {
        let names: Vec<_> = ops().registry.names().collect();
        assert_eq!(names, ["liar", "needs_modes", "pair", "scale", "split"]);
    }
}
