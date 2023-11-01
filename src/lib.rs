use matrix::Matrix;

pub mod gates;
pub mod matrix;

#[derive(Debug, Clone)]
pub struct NeuralNetwork {
    count: usize,
    weights: Vec<Matrix>,
    biases: Vec<Matrix>,
    pub activations: Vec<Matrix>,
}

impl NeuralNetwork {
    pub fn new(architechture: Vec<usize>) -> Self {
        let mut weights = Vec::new();
        let mut biases = Vec::new();
        let mut activations = Vec::new();

        activations.push(Matrix::zeros(1, architechture[0]));
        for i in 1..architechture.len() {
            weights.push(Matrix::zeros(activations[i - 1].cols, architechture[i]));
            biases.push(Matrix::zeros(1, architechture[i]));
            activations.push(Matrix::zeros(1, architechture[i]));
        }

        Self {
            count: architechture.len(),
            weights,
            biases,
            activations,
        }
    }
    pub fn rand(&mut self, low: f64, high: f64) {
        for i in 0..self.count - 1 {
            self.weights[i].rand(low, high);
            self.biases[i].rand(low, high);
        }
    }

    pub fn forward(&mut self) {
        for i in 0..self.count - 1 {
            self.activations[i + 1] = &self.activations[i] * &self.weights[i];
            self.activations[i + 1] = &self.activations[i + 1] + &self.biases[i];
            self.activations[i + 1].apply_sigmoid();
        }
    }

    pub fn cost(&mut self, training_input: &Matrix, training_output: &Matrix) -> f64 {
        let mut cost = 0.0;
        for i in 0..training_input.rows {
            let x = training_input.row(i);
            let y = training_output.row(i);

            self.activations[0] = x;
            self.forward();
            for j in 0..training_output.cols {
                let d = &self.activations[self.count - 1].get_value(0, j) - y.get_value(0, j);

                cost += d * d;
            }

            self.activations[self.count - 1] = y;
        }

        cost / training_input.rows as f64
    }

    pub fn diff(
        &mut self,
        training_input: &Matrix,
        training_output: &Matrix,
        eps: f64,
    ) -> NeuralNetwork {
        let mut output = self.clone();
        let c = self.cost(&training_input, &training_output);
        for i in 0..self.count - 1 {
            for j in 0..self.weights[i].rows {
                for k in 0..self.weights[i].cols {
                    let saved = self.weights[i].get_value(j, k);
                    self.weights[i].increase_value(j, k, eps);
                    output.weights[i].set_value(
                        j,
                        k,
                        (self.cost(&training_input, &training_output) - c) / eps,
                    );
                    self.weights[i].set_value(j, k, saved);
                }
            }
        }

        for i in 0..self.count - 1 {
            for j in 0..self.biases[i].rows {
                for k in 0..self.biases[i].cols {
                    let saved = self.biases[i].get_value(j, k);
                    self.biases[i].increase_value(j, k, eps);
                    output.biases[i].set_value(
                        j,
                        k,
                        (self.cost(&training_input, &training_output) - c) / eps,
                    );
                    self.biases[i].set_value(j, k, saved);
                }
            }
        }

        output
    }

    pub fn learn(&mut self, gradiant: &NeuralNetwork, rate: f64) {
        for i in 0..self.count - 1 {
            for j in 0..self.weights[i].rows {
                for k in 0..self.weights[i].cols {
                    self.weights[i].decrease_value(
                        j,
                        k,
                        rate * gradiant.weights[i].get_value(j, k),
                    );
                }
            }
        }

        for i in 0..self.count - 1 {
            for j in 0..self.biases[i].rows {
                for k in 0..self.biases[i].cols {
                    self.biases[i].decrease_value(j, k, rate * gradiant.biases[i].get_value(j, k));
                }
            }
        }
    }

    pub fn output(&self) -> &Matrix {
        &self.activations[self.count - 1]
    }
}
