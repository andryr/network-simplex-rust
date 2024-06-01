use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;

#[derive(Clone, Debug)]
pub struct Node {
    pub supply: f64,
}

#[derive(Clone, Debug)]
pub struct Edge {
    pub start: usize,
    pub end: usize,
    pub capacity: f64,
    pub cost: f64,
}

#[derive(Clone, Debug)]
pub struct Graph {
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
}

#[derive(Clone, Debug)]
pub struct GraphBuilder<T: Eq + Hash + Debug> {
    pub nodes: Vec<Node>,
    pub node_label_to_index: HashMap<T, usize>,
    pub edges: Vec<Edge>,
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

    pub fn get_node_or_create(&mut self, label: T) -> usize {
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

    pub fn build(&self) -> Graph {
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