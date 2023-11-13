use ndarray::{Array1, Array2, s};

pub fn euclidean_distance(x: &Array2<f64>, y: &Array2<f64>, squared: bool) -> Array2<f64> {
    let n = x.shape()[0];
    let mut sq_dist: Array2<f64> = Array2::zeros((n, n));

    for i in 0..n {
        for j in 0..n {
            let z: Array1<f64> = x.slice(s![i, ..]).to_owned() - y.slice(s![j, ..]).to_owned();
            sq_dist[[i, j]] = z.mapv_into(|w| w * w).sum();
        }
    }

    if squared {
        sq_dist
    } else {
        sq_dist.mapv_into(|x| x.sqrt())
    }
}

#[cfg(test)]
mod tests {
    use ndarray::prelude::*;

    #[test]
    fn test_euclidean_distances() {
        let x = Array2::<f64>::zeros((3, 5));
        let y = Array2::from_elem((3, 5), 5.0);
        // let y = DMatrix::from_element(3, 5, 5.0);

        let distance = super::euclidean_distance(&x, &y, false);

        // println!("euclidean_distances: {:?}", distance);

        // squared = true
        // let truth = array![
        //             [125.0, 125.0, 125.0],
        //             [125.0, 125.0, 125.0],
        //             [125.0, 125.0, 125.0]];

        // squared = false
        let truth = array![
            [11.180339887498949, 11.180339887498949, 11.180339887498949],
            [11.180339887498949, 11.180339887498949, 11.180339887498949],
            [11.180339887498949, 11.180339887498949, 11.180339887498949]
        ];

        assert_eq!(distance, truth);
    }
}
