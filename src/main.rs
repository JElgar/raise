#![feature(array_zip)]

use std::cmp::Ordering;
use rand::Rng;

type Value = f32;

#[derive(Clone)]
struct TrainingInput<const NUM_INPUTS: usize, const NUM_OUTPUTS: usize> {
    input_values: [Value; NUM_INPUTS],
    expected_outputs: [Value; NUM_OUTPUTS],
}

fn activation_function(input: Value) -> Value {
    1.0 / (1.0 + (-input).exp())
}

fn activation_function_derivative(input: Value) -> Value {
    let activation = activation_function(input);
    activation * (1.0 - activation)
}

/// Loss value for a single node
fn node_loss(output_activation: &Value, expected_output: &Value) -> Value {
    let error = output_activation - expected_output;
    error * error
}

/// Partial derivative of the loss function with respect to the activation of an output node  
fn node_loss_derivative(output_activation: &Value, expected_output: &Value) -> Value {
    2.0 * (output_activation - expected_output)
}


fn create_random_weights<const NUM_INPUTS: usize, const NUM_OUTPUTS: usize>() -> [[Value; NUM_OUTPUTS]; NUM_INPUTS] {
    let mut rng = rand::thread_rng();
    let mut weights = [[1.0; NUM_OUTPUTS]; NUM_INPUTS];

    // There are many options for how we initalize the weights. For now (given we are using the
    // sigmoid activation function) we want to keep the inital values reasonably small to make sure
    // the graident doesnt become too small.
    weights.iter_mut().flatten().for_each(|weight| *weight = rng.gen_range(-1.0..1.0) / (NUM_INPUTS as f32).sqrt());
    weights
}

trait Layer {
    fn calculate_outputs(&mut self, inputs: &[Value]) -> Vec<Value>;
    fn apply_gradients(&mut self, learn_rate: Value);
    fn calculate_output_layer_node_values(&self, expected_outputs: &[Value]) -> Vec<Value>;
    fn update_gradients(&mut self, node_values: Vec<Value>);
}

struct InnerLayer<const NODES_IN: usize, const NODES_OUT: usize> {
    weights: [[Value; NODES_OUT]; NODES_IN],
    biases: [Value; NODES_OUT],
    
    cost_gradient_weights: [[Value; NODES_OUT]; NODES_IN],
    cost_gradient_biases: [Value; NODES_OUT],
    inputs: [Value; NODES_IN],
    activations: [Value; NODES_OUT],
}

impl<const NODES_IN: usize, const NODES_OUT: usize> InnerLayer<NODES_IN, NODES_OUT> {
    fn new() -> Self {
        return InnerLayer {
            weights: create_random_weights(),
            biases: [0.0; NODES_OUT],
            cost_gradient_weights: [[1.0; NODES_OUT]; NODES_IN],
            cost_gradient_biases: [0.0; NODES_OUT],

            inputs: [0.0; NODES_IN],
            activations: [0.0; NODES_OUT],
        };
    }

    fn calculate_outputs_sized(&mut self, inputs: [Value; NODES_IN]) -> [Value; NODES_OUT] {
        let mut activations = [1.0; NODES_OUT];
        for out_node in 0..NODES_OUT {
            activations[out_node] = activation_function(
                self.biases[out_node]
                    + (0..NODES_IN)
                        .map(|in_node| inputs[in_node] * self.weights[in_node][out_node])
                        .sum::<Value>(),
            );
        }
        self.inputs = inputs;
        self.activations = activations;
        activations
    }
}

impl<const NODES_IN: usize, const NODES_OUT: usize> Layer for InnerLayer<NODES_IN, NODES_OUT> {
    fn calculate_outputs(&mut self, inputs: &[Value]) -> Vec<Value> {
        self.activations = self.calculate_outputs_sized(inputs.try_into().unwrap())
            .into();
        self.activations.to_vec()
    }

    fn apply_gradients(&mut self, learn_rate: Value) {
        self.biases.iter_mut().zip(self.cost_gradient_biases).for_each(|(bias, grad)| *bias -= grad * learn_rate);
        self.weights.iter_mut().flatten().zip(self.cost_gradient_weights.iter().flatten()).for_each(|(weight, grad)| {
            *weight -= grad * learn_rate
        });
    }

    fn calculate_output_layer_node_values(&self, expected_outputs: &[Value]) -> Vec<Value> {
        self.activations.iter().zip(expected_outputs.iter()).zip(self.inputs).map(|((activation_value, expected_value), input_value)| {
            node_loss_derivative(activation_value, expected_value) * activation_function_derivative(input_value)
        }).collect()
    }
    
    fn calculate_hidden_layer_node_values(&self, previous_layer_node_values: &[Value]) -> Vec<Value> {
        (0..NODES_OUT).map(|node_index| {
            previous_layer_node_values.iter().map(||)
        })
    }

    fn update_gradients(&mut self, node_values: Vec<Value>) {
        for out_node in 0..NODES_OUT {
            for in_node in 0..NODES_IN {
                // Add partial derivate: loss / weight of connection
                self.cost_gradient_weights[in_node][out_node] += self.inputs[in_node] * node_values[out_node];
            }

            // Add partial derivate: loss / bias
            self.cost_gradient_biases[out_node] += node_values[out_node];
        }
    }
}

struct Network<const NODES_IN: usize, const NODES_OUT: usize> {
    // Probably need some macro schenanigans to get these values from the array [3, 4];
    // Probably also worth storing this array, if possible in the generics I guess?
    layers: Vec<Box<dyn Layer>>,
}

impl<const NODES_IN: usize, const NODES_OUT: usize> Network<NODES_IN, NODES_OUT> {
    fn new(first_layer: InnerLayer<NODES_IN, NODES_OUT>) -> Self {
        let mut layers: Vec<Box<dyn Layer>> = Vec::new();
        layers.push(Box::new(first_layer));
        Network { layers }
    }

    fn calculate_outputs(&mut self, inputs: &[Value; NODES_IN]) -> [Value; NODES_OUT] {
        let outputs: Vec<Value> = self.layers.iter_mut().fold((*inputs).into(), |new_inputs, layer| {
            return layer.calculate_outputs(&new_inputs);
        });
        outputs.try_into().unwrap()
    }

    fn classify(&mut self, inputs: &[Value; NODES_IN]) -> usize {
        let outputs = self.calculate_outputs(inputs);
        outputs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .map(|(index, _)| index)
            .expect("Cannot have empty output layer")
    }

    fn single_input_loss(&mut self, TrainingInput { input_values, expected_outputs }: &TrainingInput<NODES_IN, NODES_OUT>) -> Value {
        let outputs = self.calculate_outputs(input_values);
        expected_outputs.zip(outputs).iter().fold(0.0, |loss, (expected_output, output)| loss + node_loss(output, expected_output))
    }
    
    fn loss(&mut self, inputs: &[TrainingInput<NODES_IN, NODES_OUT>]) -> Value {
        inputs.iter().map(|input| self.single_input_loss(input)).sum::<Value>() / inputs.len() as f32
    }

    /// Returns a new network with the new layer appened
    /// The new layers must have the same number of inputs as the current netowkr has outputs
    fn push<const NEW_LAYER_NODES_OUT: usize>(
        self,
        layer: InnerLayer<NODES_OUT, NEW_LAYER_NODES_OUT>,
    ) -> Network<NODES_IN, NEW_LAYER_NODES_OUT> {
        let new_layers = {
            let mut layers = self.layers;
            layers.push(Box::new(layer));
            layers
        };
        Network { layers: new_layers }
    }
    
    fn update_all_gradients(&mut self, TrainingInput { input_values, expected_outputs }: &TrainingInput<NODES_IN, NODES_OUT>) {
        self.calculate_outputs(input_values);

        let output_layer = self.layers.last_mut().unwrap();
        let node_values = output_layer.calculate_output_layer_node_values(expected_outputs);
        output_layer.update_gradients(node_values);
    }

    fn learn(&mut self, inputs: &[TrainingInput<NODES_IN, NODES_OUT>]) {
    }
}

macro_rules! create_network {
    ($a:expr, $b:expr $(,$c:expr)*) => {
        {
            let network = Network::new(InnerLayer::<$a, $b>::new());
            create_network!(@push network, $b $(,$c)*)
        }
    };
    // (@push $a:expr) => {
    // };
    (@push $network:ident, $a:expr, $b:expr $(,$c:expr)*) => {
        {
            let network = $network.push(InnerLayer::<$a, $b>::new());
            create_network!(@push network, $b $(,$c)*)
        }
    };
    // If there is only one layer size left this is the ouput size of the previous layer so we can do nothing
    (@push $network:ident, $a:expr) => {
        $network
    }
}

fn main() {
    // Manually create network
    let layer_1 = InnerLayer::<1, 2>::new();
    let layer_2 = InnerLayer::<2, 5>::new();
    let mut manual_network = Network::new(layer_1).push(layer_2);

    let outputs = manual_network.calculate_outputs(&[1.0]);
    let class = manual_network.classify(&[1.0]);
    println!("Outputs are: {:?}", outputs);
    println!("Class is: {:?}", class);

    let input = TrainingInput::<2, 5> {
        expected_outputs: [1.0, 0.0, 0.0, 0.0, 0.0],
        input_values: [1.0, 2.0],
    };

    let inputs = vec![input.clone(), input.clone()];

    let mut network = create_network!(2, 3, 5);
    let outputs = network.calculate_outputs(&input.input_values);
    let class = network.classify(&input.input_values);
    let loss = network.single_input_loss(&input);
    let inputs_loss = network.loss(&inputs);
    println!("Outputs are: {:?}", outputs);
    println!("Class is: {:?}", class);
    println!("Loss is: {:?}", loss);
    println!("Inputs loss is: {:?}", inputs_loss);
}
