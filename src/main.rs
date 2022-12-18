#![feature(generic_const_exprs)]

fn combine_arrays<T, Default, const A1_SIZE: usize, const A2_SIZE: usize, const OUTPUT_SIZE: usize>(a1: [T; A1_SIZE], a2: [T; A2_SIZE]) -> [T; A1_SIZE + A2_SIZE]  where T: Copy, T: std::default::Default {
    let mut whole: [T; A1_SIZE + A2_SIZE] = [T::default(); A1_SIZE + A2_SIZE];
    let (one, two) = whole.split_at_mut(a1.len());
    one.copy_from_slice(&a1);
    two.copy_from_slice(&a2);
    whole
}

trait Layer {
    fn calculate_outputs(&self, inputs: &[u32]) -> Vec<u32>;
}

struct LayerFixed<const NODES_IN: usize, const NODES_OUT: usize> {
    weights: [[u32; NODES_OUT]; NODES_IN],
    biases: [u32; NODES_OUT],
}

impl<const NODES_IN: usize, const NODES_OUT: usize> LayerFixed<NODES_IN, NODES_OUT> {
    fn new() -> Self {
        return LayerFixed {
            weights: [[0; NODES_OUT]; NODES_IN],
            biases: [0; NODES_OUT],
        }
    }

    fn calculate_outputs_sized(&self, inputs: [u32; NODES_IN]) -> [u32; NODES_OUT] {
        let mut outputs = [0; NODES_OUT];
        for out_node in 0..NODES_OUT {
            outputs[out_node] = self.biases[out_node] + (0..NODES_IN).map(|in_node| {
                inputs[in_node] * self.weights[in_node][out_node]
            }).sum::<u32>();
        }
        outputs
    }
}

impl<const NODES_IN: usize, const NODES_OUT: usize> Layer for LayerFixed<NODES_IN, NODES_OUT> {
    fn calculate_outputs(&self, inputs: &[u32]) -> Vec<u32> {
        self.calculate_outputs_sized(inputs.try_into().unwrap()).into()
    }
}

// struct Network<const NUM_LAYERS: usize> {
//     layers: [Layer; NUM_LAYERS]
// }

// struct Network {
//     type ThisLayer<NEW_NODES_OUT: usize>: Layer<NODES_IN, NODES_OUT>;
//     type NextLayer<const NODES_IN: usize, const NODES_OUT: usize>: Layer<NODES_IN, NODES_OUT>;
//     // type NextLayer: u32;
// 
//     // layers: [Layer; NUM_LAYERS]
// }

struct Network<const NODES_IN: usize, const NODES_OUT: usize, const NUM_LAYERS: usize> {
    // Probably need some macro schenanigans to get these values from the array [3, 4];
    // Probably also worth storing this array, if possible in the generics I guess?
    layers: [Box<dyn Layer>; NUM_LAYERS],
}


impl<const NODES_IN: usize, const NODES_OUT: usize, const NUM_LAYERS: usize> Network<NODES_IN, NODES_OUT, NUM_LAYERS> {
    fn calculate_outputs(&self, inputs: [u32; NODES_IN]) -> [u32; NODES_OUT] {
        let a: Vec<u32> = self.layers.iter().fold(inputs.into(), |new_inputs, layer| return layer.calculate_outputs(&new_inputs));
        a.try_into().unwrap()
    }

    /// Returns a new network with the new layer appened
    /// The new layers must have the same number of inputs as the current netowkr has outputs
    fn push<const NEW_LAYER_NODES_OUT: usize>(self, layer: LayerFixed<NODES_OUT, NEW_LAYER_NODES_OUT>) -> Network<NODES_IN, NEW_LAYER_NODES_OUT, {NUM_LAYERS + 1}> {
        let a = [Box::new(layer)];
        Network {
            layers: combine_arrays(self.layers, a)
        }
    }
}

fn main() {
    println!("Hello, world!");
}
