//! Network-based statistic: connected components of suprathreshold edges (Zalesky et al. 2010).

/// Where an edge feature sits: which network (component k of the connectivity) and which nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Edge {
    pub network: usize,
    pub a: usize,
    pub b: usize,
}

/// A connected set of suprathreshold edges within one network.
#[derive(Debug, Clone, PartialEq)]
pub struct Component {
    pub network: usize,
    /// Indices of the edge features in the component.
    pub edges: Vec<usize>,
    /// Permutation p of the component's size against the maximum size under the null.
    pub p: f64,
}

/// Components (as lists of edge feature indices) among edges whose statistic reaches `threshold`.
pub(crate) fn components(
    stat: &[f64],
    edges: &[Option<Edge>],
    threshold: f64,
) -> Vec<(usize, Vec<usize>)> {
    let nodes = edges
        .iter()
        .flatten()
        .map(|e| e.a.max(e.b) + 1)
        .max()
        .unwrap_or(0);
    let networks = edges
        .iter()
        .flatten()
        .map(|e| e.network + 1)
        .max()
        .unwrap_or(0);
    let mut parent: Vec<usize> = (0..nodes * networks).collect();
    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }
    let supra: Vec<(usize, Edge)> = edges
        .iter()
        .enumerate()
        .filter_map(|(f, e)| e.filter(|_| stat[f] >= threshold).map(|e| (f, e)))
        .collect();
    for (_, e) in &supra {
        let (x, y) = (
            find(&mut parent, e.network * nodes + e.a),
            find(&mut parent, e.network * nodes + e.b),
        );
        parent[x] = y;
    }
    let mut groups: std::collections::BTreeMap<usize, (usize, Vec<usize>)> =
        std::collections::BTreeMap::new();
    for (f, e) in supra {
        let root = find(&mut parent, e.network * nodes + e.a);
        groups
            .entry(root)
            .or_insert((e.network, Vec::new()))
            .1
            .push(f);
    }
    groups.into_values().collect()
}

/// The size (in edges) of the largest component.
pub(crate) fn largest(stat: &[f64], edges: &[Option<Edge>], threshold: f64) -> usize {
    components(stat, edges, threshold)
        .iter()
        .map(|(_, e)| e.len())
        .max()
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn components_follow_shared_nodes_within_a_network() {
        let e = |network, a, b| Some(Edge { network, a, b });
        // Network 0: 0-1, 1-2 (joined), 3-4 (separate); network 1: 0-1 (separate network).
        let edges = [
            e(0, 0, 1),
            e(0, 1, 2),
            e(0, 3, 4),
            e(1, 0, 1),
            None,
            e(0, 2, 3),
        ];
        let stat = [3.5, 4.0, 3.1, 5.0, 9.0, 1.0];
        let mut found = components(&stat, &edges, 3.0);
        found.sort_by_key(|(_, e)| e.clone());
        assert_eq!(found, [(0, vec![0, 1]), (0, vec![2]), (1, vec![3])]);
        assert_eq!(largest(&stat, &edges, 3.0), 2);
        let bridged = [3.5, 4.0, 3.1, 5.0, 9.0, 3.0];
        assert_eq!(
            largest(&bridged, &edges, 3.0),
            4,
            "edge 2-3 joins both into one"
        );
        assert_eq!(largest(&stat, &edges, 10.0), 0);
        assert_eq!(largest(&[], &[], 1.0), 0);
    }
}
