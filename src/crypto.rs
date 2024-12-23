use ark_ff::Field;
use rand::Rng;

use crate::dtypes::Vector;

pub fn generate_s<F: Field>(dim: usize) -> Vector<F> {
    let mut rng = rand::thread_rng();
    Vector::new((0..dim).map(|_| F::rand(&mut rng)).collect())
}

pub fn generate_e<F: Field>(dim: usize, theta: f64) -> Vector<F> {
    let mut rng = rand::thread_rng();
    Vector::new((0..dim).map(|_| F::from(discrete_gaussian::sample_vartime(theta, &mut rng))).collect())
}
