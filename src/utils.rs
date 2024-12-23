use ark_ff::{BigInteger, PrimeField};

use crate::dtypes::{Matrix, Vector};

#[derive(Debug, Clone)]
pub struct Params<Fp: PrimeField> {
    pub a: Matrix<Fp>, // A matrix
    pub p: u64, // Plaintext modulus
    pub n: usize, // LWE secret length
    pub m: usize, // Number of samples
    pub stdev: f64, // Standard deviation
}

pub fn generate_example_params<Fp: PrimeField>(n: usize, m: usize, stdev: f64) -> Params<Fp> {
    let a = Matrix::from_random(n, m);
    let p = 2;
    Params { a, p, n, m, stdev }
}

fn to_u64<Fp: PrimeField>(x: Fp) -> u64 {
    let x_bytes = x.into_bigint().to_bytes_le();
    u64::from_le_bytes(x_bytes[0..8].try_into().unwrap())
}

fn get_q_over_p<Fp: PrimeField>(params: &Params<Fp>) -> u64 {
    let q = to_u64(Fp::from_bigint(Fp::MODULUS).unwrap());
    q / params.p
}

pub fn encrypt<Fp: PrimeField>(params: &Params<Fp>, secret: &Vector<Fp>, e: &Vector<Fp>, plaintext: Vector<Fp>) -> (Matrix<Fp>, Vector<Fp>) {
    // Check that the secret has the correct length
    assert_eq!(secret.len(), params.n, "Secret length must match params.n");

    // Check that the error vector has correct length 
    assert_eq!(e.len(), params.m, "Error vector length must match params.m");

    // Check that plaintext is within range of plaintext modulus
    assert!(plaintext.len() == params.m, "Plaintext length must match params.m");

    let a_s = params.a.mul_vec(secret);
    let b = &a_s + &e;
    let delta = Fp::from(get_q_over_p(params));
    let c = &b + &plaintext.mul_scalar(delta);
    (params.a.clone(), c)
}

pub fn decrypt<Fp: PrimeField>(params: &Params<Fp>, secret: &Vector<Fp>, hint: &Matrix<Fp>, ciphertext: Vector<Fp>, index: usize) -> Fp {
    assert!(secret.len() == params.n, "Secret length must match params.n");
    assert!(ciphertext.len() == params.m, "Ciphertext length must match params.m");

    let delta = get_q_over_p(params);
    let a_s = hint.mul_vec(secret);
    let c_minus_a_s = &ciphertext - &a_s;
    let noised = c_minus_a_s[index];
    
    let noised_u64 = to_u64(noised);
    let denoised = ((noised_u64 as f64 / delta as f64).round() * delta as f64) as u64;
    let result = denoised % params.p;
    
    Fp::from(result)
}
