use ark_ff::{Field, PrimeField};
use rand::{thread_rng, SeedableRng};
use std::ops::{Add, Mul, Neg, Sub};

#[derive(Clone, Debug)]
pub struct Matrix<F: Field> {
    pub data: Vec<Vec<F>>,
}

impl<F: Field> Matrix<F> {
    pub fn new(data: Vec<Vec<F>>) -> Self {
        Self { data }
    }

    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![vec![F::zero(); cols]; rows],
        }
    }

    pub fn from_vector(rows: usize, cols: usize, vector: Vec<F>) -> Self {
        assert!(vector.len() == cols);
        Self {
            data: vec![vector; rows],
        }
    }

    pub fn from_random(rows: usize, cols: usize) -> Self {
        let mut rng = thread_rng();

        Self {
            data: (0..rows)
                .map(|_| (0..cols).map(|_| F::rand(&mut rng)).collect())
                .collect(),
        }
    }

    pub fn add(&self, other: &Self) -> Self {
        assert_eq!(self.data.len(), other.data.len());
        assert_eq!(self.data[0].len(), other.data[0].len());
        Self {
            data: self
                .data
                .iter()
                .zip(other.data.iter())
                .map(|(a, b)| a.iter().zip(b.iter()).map(|(x, y)| *x + *y).collect())
                .collect(),
        }
    }

    pub fn sub(&self, other: &Self) -> Self {
        assert_eq!(self.rows(), other.rows());
        assert_eq!(self.cols(), other.cols());
        Self {
            data: self
                .data
                .iter()
                .zip(other.data.iter())
                .map(|(a, b)| a.iter().zip(b.iter()).map(|(x, y)| *x - *y).collect())
                .collect(),
        }
    }

    pub fn neg(&self) -> Self {
        Self {
            data: self
                .data
                .iter()
                .map(|row| row.iter().map(|x| -(*x)).collect())
                .collect(),
        }
    }

    pub fn mul(&self, other: &Self) -> Self {
        assert_eq!(self.cols(), other.rows());
        let mut result = Matrix::zeros(self.rows(), other.cols());
        for i in 0..self.rows() {
            for j in 0..other.cols() {
                for k in 0..self.cols() {
                    result.data[i][j] += self.data[i][k] * other.data[k][j];
                }
            }
        }
        result
    }

    pub fn mul_scalar(&self, scalar: F) -> Self {
        Self {
            data: self
                .data
                .iter()
                .map(|row| row.iter().map(|x| *x * scalar).collect())
                .collect(),
        }
    }

    pub fn mul_vec(&self, vector: &Vector<F>) -> Vector<F> {
        assert_eq!(self.cols(), vector.data.len());
        let mut result = Vector::new(vec![F::zero(); self.rows()]);
        for i in 0..self.rows() {
            for j in 0..self.cols() {
                result.data[i] += self.data[i][j] * vector.data[j];
            }
        }
        result
    }

    #[inline]
    pub fn rows(&self) -> usize {
        self.data.len()
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.data[0].len()
    }

    #[inline]
    pub fn dim(&self) -> usize {
        assert!(self.rows() == self.cols()); // Database is always square
        self.rows()
    }
}

impl<F: Field> Add for &Matrix<F> {
    type Output = Matrix<F>;
    fn add(self, other: &Matrix<F>) -> Matrix<F> {
        self.add(other)
    }
}

impl<F: Field> Sub for &Matrix<F> {
    type Output = Matrix<F>;
    fn sub(self, other: &Matrix<F>) -> Matrix<F> {
        self.sub(other)
    }
}

impl<F: Field> Neg for &Matrix<F> {
    type Output = Matrix<F>;
    fn neg(self) -> Matrix<F> {
        self.neg()
    }
}

impl<F: Field> Mul for &Matrix<F> {
    type Output = Matrix<F>;
    fn mul(self, other: &Matrix<F>) -> Matrix<F> {
        self.mul(other)
    }
}

#[derive(Clone, Debug)]
pub struct Vector<F: Field> {
    data: Vec<F>,
}

impl<F: Field> Vector<F> {
    pub fn new(data: Vec<F>) -> Self {
        Self { data }
    }

    pub fn from_random(len: usize, seed: Option<u64>) -> Self {
        let mut rng = match seed {
            Some(s) => rand::rngs::StdRng::seed_from_u64(s),
            None => rand::rngs::StdRng::from_entropy(),
        };

        Self {
            data: (0..len).map(|_| F::rand(&mut rng)).collect(),
        }
    }

    pub fn add(&self, other: &Self) -> Self {
        assert_eq!(self.data.len(), other.data.len());
        Self {
            data: self
                .data
                .iter()
                .zip(other.data.iter())
                .map(|(a, b)| *a + *b)
                .collect(),
        }
    }

    pub fn sub(&self, other: &Self) -> Self {
        assert_eq!(self.data.len(), other.data.len());
        Self {
            data: self
                .data
                .iter()
                .zip(other.data.iter())
                .map(|(a, b)| *a - *b)
                .collect(),
        }
    }

    pub fn dot(&self, other: &Self) -> F {
        assert_eq!(self.data.len(), other.data.len());
        self.data.iter().zip(other.data.iter()).map(|(a, b)| *a * *b).sum()
    }

    pub fn neg(&self) -> Self {
        Self {
            data: self.data.iter().map(|x| -(*x)).collect(),
        }
    }

    pub fn mul_scalar(&self, scalar: F) -> Self {
        Self {
            data: self.data.iter().map(|x| *x * scalar).collect(),
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }
}

impl<F: Field> std::ops::Index<usize> for Vector<F> {
    type Output = F;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<F: Field> std::ops::IndexMut<usize> for Vector<F> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}


impl<F: Field> Add for &Vector<F> {
    type Output = Vector<F>;
    fn add(self, other: &Vector<F>) -> Vector<F> {
        self.add(other)
    }
}

impl<F: Field> Sub for &Vector<F> {
    type Output = Vector<F>;
    fn sub(self, other: &Vector<F>) -> Vector<F> {
        self.sub(other)
    }
}

impl<F: Field> Neg for &Vector<F> {
    type Output = Vector<F>;
    fn neg(self) -> Vector<F> {
        self.neg()
    }
}
