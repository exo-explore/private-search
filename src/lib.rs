pub mod dtypes;
pub mod crypto;
pub mod utils;

// use crate::dtypes::{Matrix, Vector};
// use ark_ff::{Field, PrimeField};
// use rand::{rngs::StdRng, Rng, SeedableRng};

// struct PublicParams {
//     pub dim: usize,
//     pub ciphertext_modulus: u64,
//     pub plaintext_modulus: u64,
//     pub stdev: f64,
// }

// struct Database<F: Field> {
//     pub data: Matrix<F>,
// }

// impl<F: Field> Database<F> {
//     pub fn new(data: Vector<F>) {
//         let n = data.len();
//         let sqrt_n = (n as f64).sqrt() as usize;
//         assert!(sqrt_n * sqrt_n == n, "Data length must be a perfect square");
//     }
// }

// pub struct Server<F: Field> {
//     pub db: Database<F>,
// }

// pub struct Client<F: Field> {
//     a_matrix: Matrix<F>,
//     params: PublicParams,
// }

// impl<F: Field> Client<F> {
//     pub fn new(params: PublicParams) -> Self {
//         let mut rng = rand::thread_rng();
//         let a_matrix = Matrix::new(vec![vec![F::rand(&mut rng); params.dim]; params.dim]);
//         Self { a_matrix, params }
//     }

//     pub fn query(&self, index: usize, dim: usize) {
//         let (i_row, i_col) = (index / dim, index % dim);
//         let s = crypto::generate_s::<F>(dim);
//         let e = crypto::generate_e::<F>(dim, self.params.stdev);
//         let mut u_icol = Vector::new(vec![F::zero(); dim]);
        
//         u_icol[index] = F::one();
//         let delta = self.params.ciphertext_modulus / self.params.plaintext_modulus;

//         let q_u = &self.a_matrix.mul_vector(&s) + &u_icol.mul_scalar(F::from(delta));
        

//     }
// }
