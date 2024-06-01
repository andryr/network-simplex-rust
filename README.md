# Network Simplex Rust
This repository contains a Rust implementation of the Network Simplex algorithm, which is used for solving minimum-cost flow problems in network optimization.

## Usage
Edit your `Cargo.toml` file to add `network-simplex` to your dependencies.
```toml
[dependencies]
network-simplex = { git = "https://github.com/andryr/network-simplex-rust.git" }
```
Here is a simple example to get you started:
```rust
let mut graph = GraphBuilder::new();
// Add nodes with their supply
graph.add_node(String::from("a"), 5.0);
graph.add_node(String::from("d"), -5.0);
// Add edges with their cost and capacity
// Nodes for which you don't specify supply will have their supply value set to 0
graph.add_edge(String::from("a"), String::from("b"), 4.0, 3.0);
graph.add_edge(String::from("a"), String::from("c"), 10.0, 6.0);
graph.add_edge(String::from("b"), String::from("d"), 9.0, 1.0);
graph.add_edge(String::from("c"), String::from("d"), 5.0, 2.0);
// The solver takes two arguments: the graph and an epsilon value for floating point comparison
let flow = solve_min_cost_flow(&graph.build(), 10e-12);
// Print the optimal flow
println!("{:?}", flow);
```
