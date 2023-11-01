use super::Matrix;
use std::f64::consts::E;

#[derive(Debug)]
pub struct Or {
    x: Matrix,
    w: Matrix,
    b: Matrix,
    trained: bool,
}

impl Or {
    pub fn new() -> Self {
        let x = Matrix::zeros(2, 1);
        let w = Matrix::random(1, 2, 0.0, 1.0);
        let b = Matrix::random(1, 1, 0.0, 1.0);

        Self {
            x,
            w,
            b,
            trained: false,
        }
    }

    fn forward(&mut self, x1: f64, x2: f64) -> f64 {
        self.x.set_value(0, 0, x1);
        self.x.set_value(1, 0, x2);

        let mut a1 = &self.w * &self.x;
        a1 = &a1 + &self.b;

        1.0 / (1.0 + E.powf(-1.0 * a1.get_value(0, 0)))
    }

    fn cost(&mut self, training_input: &Matrix, training_output: &Matrix) -> f64 {
        let mut cost = 0.0;
        for i in 0..training_input.rows {
            let x1 = training_input.get_value(i, 0);
            let x2 = training_input.get_value(i, 1);
            let y = training_output.get_value(i, 0);

            let d = y - self.forward(x1, x2);
            cost += d * d;
        }

        cost / training_input.rows as f64
    }

    fn finite_diff(&mut self, training_input: &Matrix, training_output: &Matrix, eps: f64) -> Or {
        let mut output = Or::new();
        let c = self.cost(training_input, training_output);

        for i in 0..self.w.rows {
            for j in 0..self.w.cols {
                let saved = self.w.get_value(i, j);
                self.w.increase_value(i, j, eps);
                output
                    .w
                    .set_value(i, j, (self.cost(training_input, training_output) - c) / eps);
                self.w.set_value(i, j, saved);
            }
        }

        for i in 0..self.b.rows {
            for j in 0..self.b.cols {
                let saved = self.b.get_value(i, j);
                self.b.increase_value(i, j, eps);
                output
                    .b
                    .set_value(i, j, (self.cost(training_input, training_output) - c) / eps);
                self.b.set_value(i, j, saved);
            }
        }

        output
    }

    fn learn(&mut self, gradiant: &Or, rate: f64) {
        for i in 0..self.w.rows {
            for j in 0..self.w.cols {
                self.w
                    .decrease_value(i, j, rate * gradiant.w.get_value(i, j));
            }
        }

        for i in 0..self.b.rows {
            for j in 0..self.b.cols {
                self.b
                    .decrease_value(i, j, rate * gradiant.b.get_value(i, j));
            }
        }
    }

    pub fn train(&mut self) {
        let eps = 0.1;
        let rate = 0.1;
        let training_input = Matrix::from(vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ]);

        let training_output = Matrix::from(vec![vec![0.0], vec![1.0], vec![1.0], vec![1.0]]);

        for _ in 0..1000 * 1000 {
            let gradiant = self.finite_diff(&training_input, &training_output, eps);
            self.learn(&gradiant, rate);
        }

        self.trained = true;
        println!("Succesfuly trained the \"Or Gate\"! :)");
    }

    pub fn predict(&mut self, x1: bool, x2: bool) -> Result<f64, String> {
        if self.trained {
            return Ok(self.forward(f64::from(x1), f64::from(x2)));
        } else {
            return Err(String::from("Please train it"));
        }
    }
}
