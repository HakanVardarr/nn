use nn::matrix::Matrix;
use nn::NeuralNetwork;

fn main() {
    let eps = 0.1;
    let rate = 0.1;
    let mut nn = NeuralNetwork::new(vec![2, 4, 3, 1]);

    nn.rand(0.0, 1.0);

    let training_input = Matrix::from(vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ]);

    let training_output = Matrix::from(vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]]);
    println!("Cost = {}", nn.cost(&training_input, &training_output));

    for _ in 0..1000 * 100 {
        let gradiant = nn.diff(&training_input, &training_output, eps);
        nn.learn(&gradiant, rate);
    }
    println!("Cost = {}", nn.cost(&training_input, &training_output));
    for i in 0..2 {
        for j in 0..2 {
            let i = i as f64;
            let j = j as f64;

            nn.activations[0] = Matrix::from(vec![vec![i, j]]);
            nn.forward();

            let output = nn.output();
            println!("{i} ^ {j} = {}", output.get_value(0, 0));
        }
    }
}
