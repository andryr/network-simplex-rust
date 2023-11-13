/*
Network simplex algorithm, adapted from networkx library implementation (https://networkx.org/documentation/stable/_modules/networkx/algorithms/flow/networksimplex.html)
*/
use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;
use std::iter::repeat;
use std::usize;

#[derive(Clone, Debug)]
struct Node {
    supply: f64,
}

#[derive(Clone, Debug)]
struct Edge {
    start: usize,
    end: usize,
    capacity: f64,
    cost: f64,
}

#[derive(Clone, Debug)]
pub struct Graph {
    nodes: Vec<Node>,
    edges: Vec<Edge>,
}

#[derive(Clone, Debug)]
pub struct GraphBuilder<T: Eq + Hash + Debug> {
    nodes: Vec<Node>,
    node_label_to_index: HashMap<T, usize>,
    edges: Vec<Edge>,
}

impl<T: Eq + Hash + Debug> GraphBuilder<T> {
    pub fn new() -> GraphBuilder<T> {
        GraphBuilder {
            nodes: vec![],
            node_label_to_index: HashMap::new(),
            edges: vec![],
        }
    }

    pub fn add_node(&mut self, label: T, supply: f64) {
        self.nodes.push(Node {
            supply
        });
        self.node_label_to_index.insert(label, self.nodes.len() - 1);
    }

    pub fn add_edge(&mut self, label_u: T, label_v: T, capacity: f64, cost: f64) {
        let u = self.get_node_or_create(label_u);
        let v = self.get_node_or_create(label_v);
        self.edges.push(Edge {
            start: u,
            end: v,
            capacity,
            cost,
        });
    }

    fn get_node_or_create(&mut self, label: T) -> usize {
        match self.node_label_to_index.get(&label) {
            Some(u) => {
                *u
            }
            None => {
                self.nodes.push(Node {
                    supply: 0.0
                });
                self.node_label_to_index.insert(label, self.nodes.len() - 1);
                self.nodes.len() - 1
            }
        }
    }

    fn build(&self) -> Graph {
        Graph {
            nodes: self.nodes.clone(),
            edges: self.edges.clone(),
        }
    }
}

impl Graph {
    pub fn new() -> Graph {
        Graph {
            nodes: vec![],
            edges: vec![],
        }
    }

    pub fn add_node(&mut self, supply: f64) {
        self.nodes.push(Node {
            supply
        });
    }

    pub fn add_edge(&mut self, u: usize, v: usize, capacity: f64, cost: f64) {
        self.edges.push(Edge {
            start: u,
            end: v,
            capacity,
            cost,
        });
    }
}

#[derive(Debug)]
struct Solution<'a> {
    graph: &'a Graph,
    edge_count: usize,
    potentials: Vec<f64>,
    next_node_dft: Vec<usize>,
    prev_node_dft: Vec<usize>,
    last_descendent_dft: Vec<usize>,
    parent_nodes: Vec<Option<usize>>,
    parent_edges: Vec<Option<usize>>,
    subtree_sizes: Vec<usize>,
    flow: Vec<f64>,
    eps: f64,
    block_start: usize,
}

struct SubtreeIterator<'a> {
    current: Option<usize>,
    next_node_dft: &'a Vec<usize>,
    last: usize,
}

impl SubtreeIterator<'_> {
    fn init<'a>(u: usize, next_node_dft: &'a Vec<usize>, last_descendant_dft: &'a Vec<usize>) -> SubtreeIterator<'a> {
        SubtreeIterator {
            current: Some(u),
            next_node_dft,
            last: last_descendant_dft[u],
        }
    }
}

impl Iterator for SubtreeIterator<'_> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        let cur = self.current;
        match cur {
            Some(node) => {
                if node != self.last {
                    self.current = Some(self.next_node_dft[node]);
                } else {
                    self.current = None;
                }
                Some(node)
            }
            _ => None
        }
    }
}

impl Solution<'_> {
    fn new(graph: &mut Graph, eps: f64) -> Solution {
        let n = graph.nodes.len();
        let m = graph.edges.len();

        let cost_sum = graph.edges.iter().map(|edge| edge.cost.abs()).sum();
        let capacity_sum = graph.edges.iter().map(|edge| edge.capacity).filter(|capacity| *capacity < f64::INFINITY).sum();
        let supplies: Vec<f64> = graph.nodes.iter().map(|node| node.supply).collect();
        let faux_inf = 3.0 * supplies.iter().chain(vec![cost_sum, capacity_sum].iter()).max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        // Add artificial root and edges
        for (i, u) in graph.nodes.iter().enumerate() {
            let edge = if u.supply < -eps {
                Edge {
                    start: n,
                    end: i,
                    capacity: faux_inf,
                    cost: faux_inf,
                }
            } else {
                Edge {
                    start: i,
                    end: n,
                    capacity: faux_inf,
                    cost: faux_inf,
                }
            };
            graph.edges.push(edge);
        }
        let bfs = Solution {
            graph,
            edge_count: m,
            potentials: graph.nodes.iter().map(|node| {
                if node.supply >= 0.0 {
                    faux_inf
                } else {
                    -faux_inf
                }
            }).collect(),
            next_node_dft: (1..n).chain(vec![n, 0]).collect(),
            prev_node_dft: vec![n].iter().copied().chain(0..n).collect(),
            last_descendent_dft: (0..n).chain(vec![n - 1]).collect(),
            parent_nodes: repeat(Some(n)).take(n).chain(vec![None].iter().copied()).collect(),
            parent_edges: (m..(m + n)).map(|i| Some(i)).collect(),
            subtree_sizes: repeat(1).take(n).chain(vec![n + 1].iter().copied()).collect(),
            flow: repeat(0.0).take(m).chain(graph.nodes.iter().map(|node| { node.supply.abs() })).collect(),
            eps,
            block_start: 0,
        };
        bfs
    }

    fn lca(&self, u: usize, v: usize) -> usize {
        let mut u = u;
        let mut v = v;
        let mut size_u = self.subtree_sizes[u];
        let mut size_v = self.subtree_sizes[v];
        loop {
            while size_u < size_v {
                u = self.parent_nodes[u].unwrap();
                size_u = self.subtree_sizes[u];
            }
            while size_u > size_v {
                v = self.parent_nodes[v].unwrap();
                size_v = self.subtree_sizes[v];
            }
            if size_u == size_v {
                if u != v {
                    u = self.parent_nodes[u].unwrap();
                    size_u = self.subtree_sizes[u];
                    v = self.parent_nodes[v].unwrap();
                    size_v = self.subtree_sizes[v];
                } else {
                    return u;
                }
            }
        }
    }

    fn trace_path(&self, u: usize, v: usize) -> (Vec<usize>, Vec<usize>) {
        let mut u = u;
        let mut nodes = vec![u];
        let mut edges: Vec<usize> = vec![];
        while u != v {
            edges.push(self.parent_edges[u].unwrap());
            u = self.parent_nodes[u].unwrap();
            nodes.push(u);
        }
        (nodes, edges)
    }

    fn find_cycle(&self, i: usize, u: usize, v: usize) -> (Vec<usize>, Vec<usize>) {
        let w = self.lca(u, v);
        let (mut nodes, mut edges) = self.trace_path(u, w);
        nodes.reverse();
        edges.reverse();
        if *edges != [i] {
            edges.push(i);
        }
        let (mut nodes_r, edges_r) = self.trace_path(v, w);
        nodes_r.remove(nodes_r.len() - 1);
        nodes.extend(nodes_r.iter());
        edges.extend(edges_r.iter());
        (nodes, edges)
    }

    fn augment_flow(&mut self, nodes: &Vec<usize>, edges: &Vec<usize>, flow: f64) {
        for i in 0..edges.len() {
            let edge = edges[i];
            let node = nodes[i];
            if self.graph.edges[edge].start == node {
                self.flow[edge] += flow
            } else {
                self.flow[edge] -= flow
            }
        }
    }

    fn trace_subtree(&mut self, u: usize) -> SubtreeIterator {
        SubtreeIterator::init(u, &self.next_node_dft, &self.last_descendent_dft)
    }

    fn remove_edge(&mut self, u: usize, v: usize) {
        let size_v = self.subtree_sizes[v];
        let prev_v = self.prev_node_dft[v];
        let last_v = self.last_descendent_dft[v];
        let next_last_v = self.next_node_dft[last_v];
        // Remove (u, v).
        self.parent_nodes[v] = None;
        self.parent_edges[v] = None;
        // Remove the subtree rooted at v from the depth-first thread.
        self.next_node_dft[prev_v] = next_last_v;
        self.prev_node_dft[next_last_v] = prev_v;
        self.next_node_dft[last_v] = v;
        self.prev_node_dft[v] = last_v;
        // Update the subtree sizes and last descendants of the (old) acenstors of v.
        let mut u = Some(u);
        while !u.is_none() {
            let u_ = u.unwrap();
            self.subtree_sizes[u_] -= size_v;
            if self.last_descendent_dft[u_] == last_v {
                self.last_descendent_dft[u_] = prev_v;
            }
            u = self.parent_nodes[u_];
        }
    }

    fn make_root(&mut self, u: usize) {
        let ancestors = self.get_ancestors(u);

        for (v, w) in ancestors.iter().zip(ancestors.iter().skip(1)) {
            let size_v = self.subtree_sizes[*v];
            let mut last_v = self.last_descendent_dft[*v];
            let prev_w = self.prev_node_dft[*w];
            let last_w = self.last_descendent_dft[*w];
            let next_last_w = self.next_node_dft[last_w];
            // Make v a child of w
            self.parent_nodes[*v] = Some(*w);
            self.parent_nodes[*w] = None;
            self.parent_edges[*v] = self.parent_edges[*w];
            self.parent_edges[*w] = None;
            self.subtree_sizes[*v] = size_v - self.subtree_sizes[*w];
            self.subtree_sizes[*w] = size_v;
            // Remove the subtree rooted at w from the depth-first thread.
            self.next_node_dft[prev_w] = next_last_w;
            self.prev_node_dft[next_last_w] = prev_w;
            self.next_node_dft[last_w] = *w;
            self.prev_node_dft[*w] = last_w;
            if last_v == last_w {
                self.last_descendent_dft[*v] = prev_w;
                last_v = prev_w;
            }
            // Add the remaining parts of the subtree rooted at v as a subtree
            // of w in the depth-first thread.
            self.prev_node_dft[*v] = last_w;
            self.next_node_dft[last_w] = *v;
            self.next_node_dft[last_v] = *w;
            self.prev_node_dft[*w] = last_v;
            self.last_descendent_dft[*w] = last_v;
        }
    }

    fn get_ancestors(&self, u: usize) -> Vec<usize> {
        let mut u = Some(u);
        let mut ancestors: Vec<usize> = vec![];
        while !u.is_none() {
            let u_ = u.unwrap();
            ancestors.push(u_);
            u = self.parent_nodes[u_];
        }
        ancestors.reverse();
        ancestors
    }

    fn add_edge(&mut self, i: usize, u: usize, v: usize) {
        let last_u = self.last_descendent_dft[u];
        let next_last_u = self.next_node_dft[last_u];
        let size_v = self.subtree_sizes[v];
        let last_v = self.last_descendent_dft[v];
        // Make v a child of u.
        self.parent_nodes[v] = Some(u);
        self.parent_edges[v] = Some(i);
        // Insert the subtree rooted at v into the depth-first thread.
        self.next_node_dft[last_u] = v;
        self.prev_node_dft[v] = last_u;
        self.prev_node_dft[next_last_u] = last_v;
        self.next_node_dft[last_v] = next_last_u;
        // Update the subtree sizes and last descendants of the (new) ancestors
        // of v.
        let mut u = Some(u);
        while !u.is_none() {
            let u_ = u.unwrap();
            self.subtree_sizes[u_] += size_v;
            if self.last_descendent_dft[u_] == last_u {
                self.last_descendent_dft[u_] = last_v;
            }
            u = self.parent_nodes[u_];
        }
    }

    fn update_potentials(&mut self, i: usize, p: usize, q: usize) {
        let d = if q == self.graph.edges[i].end {
            self.potentials[p] - self.graph.edges[i].cost - self.potentials[q]
        } else {
            self.potentials[p] + self.graph.edges[i].cost - self.potentials[q]
        };
        let last = self.last_descendent_dft[q];
        let mut u = q;
        loop {
            self.potentials[u] += d;
            if u == last {
                break;
            }
            u = self.next_node_dft[u];
        }
    }

    fn reduced_cost(&self, i: usize) -> f64 {
        let c = self.graph.edges[i].cost - self.potentials[self.graph.edges[i].start] + self.potentials[self.graph.edges[i].end];
        if self.flow[i].abs() < self.eps {
            c
        } else {
            -c
        }
    }

    fn find_entering_edge(&mut self, block_size: usize, num_blocks: usize) -> Option<(usize, usize, usize)> {
        let mut m = 0;
        while m < num_blocks {
            let mut block_end = self.block_start + block_size;

            let edges = if block_end <= self.edge_count {
                Box::new(self.block_start..block_end) as Box<dyn Iterator<Item=usize>>
            } else {
                block_end -= self.edge_count;
                Box::new((self.block_start..self.edge_count).chain(0..block_end)) as Box<dyn Iterator<Item=usize>>
            };
            self.block_start = block_end;

            let i = argmin(edges, |i| self.reduced_cost(i)).unwrap();
            let c = self.reduced_cost(i);

            if c >= -self.eps {
                m += 1;
            } else {
                let edge = &self.graph.edges[i];
                let (u, v) = if self.flow[i].abs() < self.eps {
                    (edge.start, edge.end)
                } else {
                    (edge.end, edge.start)
                };
                return Some((i, u, v));
            }
        }
        None
    }

    fn residual_capacity(&self, i: usize, p: usize) -> f64 {
        let edge = &self.graph.edges[i];
        let flow = self.flow[i];
        if edge.start == p {
            edge.capacity - flow
        } else {
            flow
        }
    }

    fn find_leaving_edge(&self, nodes: &Vec<usize>, edges: &Vec<usize>) -> (usize, usize, usize) {
        let nodes_rev = nodes.iter().rev();
        let edges_rev = edges.iter().rev();

        let ind = edges_rev.zip(nodes_rev);
        let (j, s) = argmin(ind,
                            |(t, u)| self.residual_capacity(*t, *u)).unwrap();

        let t = if self.graph.edges[*j].start == *s {
            self.graph.edges[*j].end
        } else {
            self.graph.edges[*j].start
        };
        (*j, *s, t)
    }
}

fn argmin<S: Copy>(edges: impl Iterator<Item=S>, func: impl Fn(S) -> f64) -> Option<S> {
    let mut argmin: Option<S> = None;
    let mut min = f64::INFINITY;
    for i in edges {
        let value = func(i);
        if value < min {
            min = value;
            argmin = Some(i);
        }
    }
    argmin
}

pub fn network_simplex(graph: &Graph, eps: f64) -> Vec<f64> {
    let mut graph = graph.clone();
    let mut solution = Solution::new(&mut graph, eps);

    // Pivot loop
    let block_size = (solution.edge_count as f64).sqrt().ceil() as usize;
    let num_blocks = (solution.edge_count + block_size - 1) / block_size;
    let mut next = solution.find_entering_edge(block_size, num_blocks);
    while next.is_some() {
        let (i, mut u, mut v) = next.unwrap();

        let (nodes, edges) = solution.find_cycle(i, u, v);
        let (j, mut s, mut t) = solution.find_leaving_edge(&nodes, &edges);
        solution.augment_flow(&nodes, &edges, solution.residual_capacity(j, s));

        if i != j {
            let parent_t = solution.parent_nodes[t];
            if parent_t.is_none() || parent_t.unwrap() != s {
                (s, t) = (t, s);
            }
            let ind1 = edges.iter().position(|&k| k == i).unwrap();
            let ind2 = edges.iter().position(|&k| k == j).unwrap();
            if ind1 > ind2 {
                (u, v) = (v, u);
            }
            solution.remove_edge(s, t);
            solution.make_root(v);
            solution.add_edge(i, u, v);
            solution.update_potentials(i, u, v);
        }
        next = solution.find_entering_edge(block_size, num_blocks);
    }
    solution.flow.drain(solution.edge_count..solution.flow.len());
    solution.flow
}

#[cfg(test)]
mod tests {
    use ndarray_rand::rand;
    use ndarray_rand::rand::random;

    use crate::lp::network_simplex::{Graph, GraphBuilder, network_simplex};

    #[test]
    fn test_ns() {
        let mut graph = GraphBuilder::new();
        graph.add_node(String::from("a"), 5.0);
        graph.add_node(String::from("d"), -5.0);
        graph.add_edge(String::from("a"), String::from("b"), 4.0, 3.0);
        graph.add_edge(String::from("a"), String::from("c"), 10.0, 6.0);
        graph.add_edge(String::from("b"), String::from("d"), 9.0, 1.0);
        graph.add_edge(String::from("c"), String::from("d"), 5.0, 2.0);
        let flow = super::network_simplex(&graph.build(), 10e-12);
        assert_eq!(vec![4.0, 1.0, 4.0, 1.0], flow);
    }

    #[test]
    fn test_ns2() {
        let mut graph = GraphBuilder::new();
        graph.add_node(String::from("p"), 4.0);
        graph.add_node(String::from("q"), -2.0);
        graph.add_node(String::from("a"), 2.0);
        graph.add_node(String::from("d"), 1.0);
        graph.add_node(String::from("t"), -2.0);
        graph.add_node(String::from("w"), -3.0);
        graph.add_edge(String::from("p"), String::from("q"), 5.0, 7.0);
        graph.add_edge(String::from("p"), String::from("a"), 4.0, 1.0);
        graph.add_edge(String::from("q"), String::from("d"), 3.0, 2.0);
        graph.add_edge(String::from("t"), String::from("q"), 2.0, 1.0);
        graph.add_edge(String::from("a"), String::from("t"), 4.0, 2.0);
        graph.add_edge(String::from("d"), String::from("w"), 4.0, 3.0);
        graph.add_edge(String::from("t"), String::from("w"), 1.0, 4.0);
        let flow = super::network_simplex(&graph.build(), 10e-12);
        assert_eq!(vec![2.0, 2.0, 1.0, 1.0, 4.0, 2.0, 1.0], flow);
    }
}
