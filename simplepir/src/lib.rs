use nalgebra::{DMatrix, DVector};
use num_bigint::{BigInt, RandBigInt};
use num_traits::{One, Signed, Zero};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use rand_distr::{Distribution, Normal};

#[derive(Debug, Clone)]
pub struct SimplePIRParams {
    pub n: usize,  // LWE dimension
    pub m: usize,  // Matrix dimension
    pub q: BigInt, // LWE modulus
    pub p: BigInt, // Plaintext modulus
    std_dev: f64,  // Standard deviation for error
    seed: u64,     // Random seed for reproducibility
}

pub fn gen_params(m: usize, n: usize, mod_power: u32) -> SimplePIRParams {
    let mut rng = rand::thread_rng();
    SimplePIRParams {
        n,
        m,
        q: BigInt::one() << 64,
        p: BigInt::one() << mod_power,
        std_dev: 3.2,
        seed: rng.gen(),
    }
}

pub fn gen_matrix_a(seed: u64, m: usize, n: usize, q: u64) -> DMatrix<BigInt> {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let data: Vec<BigInt> = (0..m * n).map(|_| rng.gen_bigint(q).abs()).collect();
    DMatrix::from_vec(m, n, data)
}

pub fn gen_secret(q: u64, n: usize, seed: Option<u64>) -> DVector<BigInt> {
    let mut rng = match seed {
        Some(s) => ChaCha20Rng::seed_from_u64(s),
        None => ChaCha20Rng::from_entropy(),
    };

    let data: Vec<BigInt> = (0..n).map(|_| rng.gen_bigint(q).abs()).collect();
    DVector::from_vec(data)
}

pub fn gen_hint(
    params: &SimplePIRParams,
    db: &DMatrix<BigInt>,
) -> (DMatrix<BigInt>, DMatrix<BigInt>) {
    let a = gen_matrix_a(params.seed, params.m, params.n, params.q.bits());
    let modulus = &params.q.clone();

    // Matrix multiplication with modulo
    let mut hint = DMatrix::zeros(db.nrows(), a.ncols());
    for i in 0..db.nrows() {
        for j in 0..a.ncols() {
            let mut sum = BigInt::zero();
            for k in 0..db.ncols() {
                sum = (sum + (db[(i, k)].clone() * a[(k, j)].clone()) % modulus) % modulus;
            }
            hint[(i, j)] = sum;
        }
    }

    (hint, a)
}

pub fn encrypt(
    params: &SimplePIRParams,
    v: &DVector<BigInt>,
    a: &DMatrix<BigInt>,
    s: &DVector<BigInt>,
) -> DVector<BigInt> {
    let modulus = &params.q.clone();
    let delta = modulus / &params.p;

    // Generate Gaussian error
    let normal = Normal::new(0.0, params.std_dev).unwrap();
    let mut rng = rand::thread_rng();
    let e: DVector<BigInt> = DVector::from_iterator(
        params.m,
        (0..params.m)
            .map(|_| (BigInt::from(normal.sample(&mut rng).round() as i64) * &params.p) % modulus),
    );

    // Compute As
    let mut as_prod = DVector::zeros(params.m);
    for i in 0..params.m {
        let mut sum = BigInt::zero();
        for j in 0..params.n {
            sum = (sum + (&a[(i, j)] * &s[j]) % modulus) % modulus
        }
        as_prod[i] = sum;
    }

    let mut result = DVector::<BigInt>::zeros(params.m);
    result
        .iter_mut()
        .zip(as_prod.iter().zip(e.iter().zip(v.iter())))
        .for_each(|(res, (as_val, (e_val, v_val)))| {
            let scaled_v = (&delta * v_val) % modulus;
            *res = (as_val + e_val + scaled_v) % modulus;
        });

    result
}

pub fn generate_query(
    params: &SimplePIRParams,
    v: &DVector<BigInt>,
    a: &DMatrix<BigInt>,
) -> (DVector<BigInt>, DVector<BigInt>) {
    assert_eq!(v.len(), params.m, "Vector dimension mismatch");

    let s = gen_secret(params.q.bits(), params.n, None);
    let query = encrypt(params, v, a, &s);

    (s, query)
}

pub fn process_query(db: &DMatrix<BigInt>, query: &DVector<BigInt>, q: BigInt) -> DVector<BigInt> {
    let mut result = DVector::zeros(db.nrows());
    let modulus = &q.clone();
    for i in 0..db.nrows() {
        let mut sum = BigInt::zero();
        for j in 0..db.ncols() {
            sum = (sum + (&db[(i, j)] * &query[j]) % modulus) % modulus;
        }
        result[i] = sum;
    }
    result
}

pub fn recover(
    hint: &DMatrix<BigInt>,
    s: &DVector<BigInt>,
    answer: &DVector<BigInt>,
    params: &SimplePIRParams,
) -> DVector<BigInt> {
    let modulus = &params.q.clone();
    let delta = modulus / &params.p;
    let half_p: BigInt = &params.p >> 1;

    let mut hint_s = DVector::zeros(answer.len());
    for i in 0..answer.len() {
        let mut sum = BigInt::zero();
        for j in 0..s.len() {
            sum = ((&sum + &hint[(i, j)] * &s[j]) % modulus + modulus) % modulus;
        }
        hint_s[i] = sum;
    }

    let mut decrypted = DVector::zeros(answer.len());
    for i in 0..answer.len() {
        let diff = ((&answer[i] + modulus - &hint_s[i]) % modulus + modulus) % modulus;
        let raw = &diff / &delta;
        let centered = if raw >= half_p { raw - &params.p } else { raw };
        decrypted[i] = centered;
    }
    decrypted
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper function to check if two vectors are approximately equal
    fn is_approximately_equal(
        v1: &DVector<BigInt>,
        v2: &DVector<BigInt>,
        tolerance: &BigInt,
    ) -> bool {
        if v1.len() != v2.len() {
            return false;
        }

        for i in 0..v1.len() {
            let diff = (&v1[i] - &v2[i]).abs();
            if diff > *tolerance {
                return false;
            }
        }
        true
    }

    #[test]
    fn test_pir() {
        let matrix_height = 10;
        let matrix_width = 10;
        let max_val_bits = 12;

        // Create random test data with both positive and negative values
        let mut rng = rand::thread_rng();
        let d_data: Vec<BigInt> = (0..matrix_height * matrix_width)
            .map(|_| {
                let val = rng.gen_bigint(max_val_bits);
                // Randomly decide if the value should be negative
                if rng.gen_bool(0.5) {
                    -val
                } else {
                    val
                }
            })
            .collect();
        let d = DMatrix::from_vec(matrix_height, matrix_width, d_data);

        let v_data: Vec<BigInt> = (0..matrix_width)
            .map(|_| {
                let val = rng.gen_bigint(max_val_bits);
                if rng.gen_bool(0.5) {
                    -val
                } else {
                    val
                }
            })
            .collect();
        let v = DVector::from_vec(v_data);

        // Rest of the test remains the same
        let expected = {
            let mut result = DVector::zeros(matrix_height);
            for i in 0..matrix_height {
                let mut sum = BigInt::zero();
                for j in 0..matrix_width {
                    sum += &d[(i, j)] * &v[j];
                }
                result[i] = sum;
            }
            result
        };

        // Test system
        let params = gen_params(matrix_height, 2048, 30);
        let (hint, a) = gen_hint(&params, &d);
        let (s, query) = generate_query(&params, &v, &a);
        let answer = process_query(&d, &query, params.q.clone());
        let result = recover(&hint, &s, &answer, &params);

        // Define tolerance - adjust this value based on your needs
        let tolerance = BigInt::from(10000);

        // Compare results with tolerance
        println!("expected: {:?}", expected);
        println!("result: {:?}", result);
        assert!(
            is_approximately_equal(&expected, &result, &tolerance),
            "Test failed: Results don't match within tolerance"
        );
        println!("Success: Test passed!");
    }

    #[test]
    fn test_row_retrieval() {
        let matrix_height = 10;
        let matrix_width = 10;
        let max_val_bits = 12;

        // Create random test data
        let mut rng = rand::thread_rng();
        let d_data: Vec<BigInt> = (0..matrix_height * matrix_width)
            .map(|_| rng.gen_bigint(max_val_bits).abs())
            .collect();
        let d = DMatrix::from_vec(matrix_height, matrix_width, d_data);

        let target_row = rng.gen_range(0..matrix_width);
        let mut v = DVector::<BigInt>::zeros(matrix_width);
        v[target_row] = BigInt::one();

        // Expected result
        let expected = {
            let mut result = DVector::zeros(matrix_height);
            for i in 0..matrix_height {
                result[i] = d[(i, target_row)].clone();
            }
            result
        };

        // Test system
        let params = gen_params(matrix_height, 2048, 17);
        let (hint, a) = gen_hint(&params, &d);
        let (s, query) = generate_query(&params, &v, &a);
        let answer = process_query(&d, &query, params.q.clone());
        let result = recover(&hint, &s, &answer, &params);

        // Define tolerance - adjust this value based on your needs
        let tolerance = BigInt::from(10);

        // Compare results with tolerance
        assert!(
            is_approximately_equal(&expected, &result, &tolerance),
            "Test failed: Results don't match within tolerance"
        );
        println!("Success: Test passed!");
    }
}
