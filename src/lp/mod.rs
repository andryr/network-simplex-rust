mod network_simplex;

use ndarray::prelude::*;
use crate::lp::network_simplex::{Graph, network_simplex};

pub fn solve(u: &Array1<f64>, v: &Array1<f64>, cost_matrix: &Array2<f64>, eps: f64) -> Array2<f64> {
    let mut graph: Graph = Graph::new();
    let m = u.len();
    let n = v.len();

    for i in 0..m {
        graph.add_node(u[i]);
    }
    for j in 0..n {
        graph.add_node(-v[j]);
    }

    for i in 0..m {
        for j in 0..n {
            graph.add_edge(i, m + j, f64::INFINITY, cost_matrix[[i, j]]);
        }
    }

    let flow = Array1::from_vec(network_simplex(&graph, eps));

    let mut result: Array2<f64> = Array::zeros((m, n));

    for i in 0..m {
        for j in 0..n {
            result[[i, j]] = flow[i * n + j];
        }
    }

    result
}

mod tests {
    use approx::assert_abs_diff_eq;
    use ndarray::{Array, array, Array1, Array2, Axis};
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand::SeedableRng;
    use ndarray_rand::rand_distr::Normal;
    use rand_isaac::isaac64::Isaac64Rng;
    use crate::lp::solve;

    #[test]
    fn test_solve() {
        let mut rng = Isaac64Rng::seed_from_u64(0);

        let n = 300;
        let x = Array::random_using((n, 2), Normal::new(0.0, 1.0).unwrap(), &mut rng);
        let x2 = (&x * &x).sum_axis(Axis(1)).into_shape((n, 1)).unwrap();

        let cost_matrix = (&x2 + &x2.t()) - 2.0 * x.dot(&x.t());

        let u: Array1<f64> = Array1::ones(n) / (n as f64);

        let result = solve(&u, &u, &cost_matrix, 1e-8);



        assert_abs_diff_eq!(result.sum_axis(Axis(0)), u);
        assert_abs_diff_eq!(result.sum_axis(Axis(1)), u);
    }
}

