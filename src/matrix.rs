use rand::{thread_rng, Rng};
use std::f64::consts::E;
use std::fmt::{Debug, Display, Formatter, Result};
use std::ops::{Add, Mul, Sub};

#[derive(Clone, Debug)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<Vec<f64>>,
}

impl Matrix {
    pub fn zeros(rows: usize, cols: usize) -> Matrix {
        Matrix {
            rows,
            cols,
            data: vec![vec![0.0; cols]; rows],
        }
    }

    pub fn random(rows: usize, cols: usize, low: f64, high: f64) -> Matrix {
        let mut rng = thread_rng();

        let mut res = Matrix::zeros(rows, cols);
        for i in 0..rows {
            for j in 0..cols {
                res.data[i][j] = rng.gen_range(low..high);
            }
        }

        res
    }

    pub fn rand(&mut self, low: f64, high: f64) {
        let mut rng = thread_rng();

        for i in 0..self.rows {
            for j in 0..self.cols {
                self.data[i][j] = rng.gen_range(low..high);
            }
        }
    }

    pub fn from(data: Vec<Vec<f64>>) -> Matrix {
        Matrix {
            rows: data.len(),
            cols: data[0].len(),
            data,
        }
    }
    pub fn apply_sigmoid(&mut self) {
        for i in 0..self.rows {
            for j in 0..self.cols {
                self.data[i][j] = 1.0 / (1.0 + E.powf(-1.0 * (self.get_value(i, j))));
            }
        }
    }

    pub fn row(&self, rows: usize) -> Matrix {
        let data = &self.data[rows];

        Matrix {
            rows: 1,
            cols: data.len(),
            data: vec![data.clone()],
        }
    }

    pub fn get_value(&self, rows: usize, cols: usize) -> f64 {
        self.data[rows][cols]
    }

    pub fn set_value(&mut self, rows: usize, cols: usize, value: f64) {
        self.data[rows][cols] = value;
    }

    pub fn increase_value(&mut self, rows: usize, cols: usize, value: f64) {
        self.data[rows][cols] = self.get_value(rows, cols) + value;
    }
    pub fn decrease_value(&mut self, rows: usize, cols: usize, value: f64) {
        self.data[rows][cols] = self.get_value(rows, cols) - value;
    }
}

impl Display for Matrix {
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(
            f,
            "Matrix {{\n{}\n}}",
            (&self.data)
                .into_iter()
                .map(|row| "  ".to_string()
                    + &row
                        .into_iter()
                        .map(|value| value.to_string())
                        .collect::<Vec<String>>()
                        .join(" "))
                .collect::<Vec<String>>()
                .join("\n")
        )
    }
}

impl Add for &Matrix {
    type Output = Matrix;
    fn add(self, other: Self) -> Self::Output {
        if self.rows != other.rows || self.cols != other.cols {
            panic!("Attempted to add matrix of incorrect dimensions");
        }

        let mut res = Matrix::zeros(self.rows, self.cols);

        for i in 0..self.rows {
            for j in 0..self.cols {
                res.data[i][j] = self.data[i][j] + other.data[i][j];
            }
        }

        res
    }
}

impl Mul for &Matrix {
    type Output = Matrix;
    fn mul(self, other: Self) -> Self::Output {
        if self.cols != other.rows {
            panic!("Attempted to multiply by matrix of incorrect dimensions");
        }

        let mut res = Matrix::zeros(self.rows, other.cols);

        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self.data[i][k] * other.data[k][j];
                }

                res.data[i][j] = sum;
            }
        }

        res
    }
}

impl Sub for &Matrix {
    type Output = Matrix;
    fn sub(self, other: Self) -> Self::Output {
        if self.rows != other.rows || self.cols != other.cols {
            panic!("Attempted to subtract matrix of incorrect dimensions");
        }

        let mut res = Matrix::zeros(self.rows, self.cols);

        for i in 0..self.rows {
            for j in 0..self.cols {
                res.data[i][j] = self.data[i][j] - other.data[i][j];
            }
        }

        res
    }
}
