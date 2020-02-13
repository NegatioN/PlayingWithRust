extern crate circular_queue;
extern crate ndarray;
extern crate ndarray_rand;
use circular_queue::CircularQueue;
use ndarray::{s, Array, Array1};
use ndarray_rand::rand::distributions::Uniform;
use ndarray_rand::RandomExt;

use std::collections::HashMap;

fn make_lookup<'a>(arr: &[&'a str]) -> HashMap<&'a str, i32> {
    let mut words: HashMap<&str, i32> = HashMap::new();

    let mut counter = 0;
    for word in arr.iter() {
        if !words.contains_key(word) {
            words.insert(word, counter);
            counter += 1;
        }
    }

    words
}

fn reverse_lookup<'a>(lookup: &HashMap<&'a str, i32>) -> HashMap<i32, &'a str> {
    let mut words: HashMap<i32, &str> = HashMap::new();
    for (k, v) in lookup.iter() {
        words.insert(*v, k);
    }
    words
}

fn sigmoid(x: &f32) -> f32 {
    1. / (1. + (-x).exp())
}

fn deriv_logsigmoid(x: &f32) -> f32 {
    1. / (1. + x.exp())
}

struct Network {
    output_embeddings: Array1<f32>,
    hidden_embeddings: Array1<f32>,
    vec_size: i32,
    num_words: i32,
    lr: f32,
    hidden_grads: Vec<Array1<f32>>,
    output_grads: Vec<Array1<f32>>,
    hidden_grads_offsets: Vec<i32>,
    output_grads_offsets: Vec<i32>,
}

impl Network {
    fn new(num_words: i32, vec_size: i32, lr: f32) -> Self {
        let vec_max_val = 0.5 / num_words as f32;
        let udist = Uniform::from(-vec_max_val..vec_max_val);
        let num_elements = num_words * vec_size;
        let mut output_embeddings = Array1::<f32>::zeros(num_elements as usize);
        let mut hidden_embeddings = Array::random(num_elements as usize, udist);
        let mut hidden_grads = Vec::new();
        let mut output_grads = Vec::new();
        let mut hidden_grads_offsets = Vec::new();
        let mut output_grads_offsets = Vec::new();
        Network {
            output_embeddings,
            hidden_embeddings,
            vec_size,
            num_words,
            lr,
            hidden_grads,
            output_grads,
            hidden_grads_offsets,
            output_grads_offsets,
        }
    }

    fn forward(&mut self, ind: &i32, tar_ind: &i32, label: f32) -> f32 {
        //TODO confirm values goes to correct embedding spaces.
        let vec_offset = ind * self.vec_size;
        let tar_offset = tar_ind * self.vec_size;
        let vec = self
            .hidden_embeddings
            .slice(s![vec_offset..vec_offset + self.vec_size]);
        // multiply by label to get inverted vecs if negative sample
        let tar_vec = label
            * &self
                .output_embeddings
                .slice(s![tar_offset..tar_offset + self.vec_size]);

        let dot = tar_vec.dot(&vec);
        //TODO implement robust version of logsigmoid
        //TODO WRITE TESTS, especially for functions like sigmoid
        let sig = sigmoid(&dot);
        let loss = -sig.log2();

        //gradient of dot product = lr * (label - score)
        //positives want to be labeled 1, and negatives will be inverted, and move towards -1
        let grad = self.lr * (1. - deriv_logsigmoid(&loss));
        self.hidden_grads.push(grad * &tar_vec);
        self.output_grads.push(grad * &vec);
        self.hidden_grads_offsets.push(tar_offset);
        self.output_grads_offsets.push(vec_offset);
        loss
    }

    fn backward(&mut self) {
        for _ in 0..self.hidden_grads.len() {
            let offset = self.hidden_grads_offsets.pop();
            let vec = self.hidden_grads.pop();
            self.output_embeddings[offset..offset + self.vec_size] += &vec;
            self.output_embeddings
                .slice_mut(s![offset..offset + self.vec_size]) += &vec;
        }
        //TODO in backward just apply updates from all stored offsets and grads?
    }
}

fn main() {
    let ex_sentence = [
        "i", "walked", "on", "the", "magical", "white", "shore", "in", "my", "white", "shoes",
        "and", "white", "sky",
    ];
    const NUM_WORDS: i32 = 1000;
    const VEC_SIZE: i32 = 100;
    const WINDOW_SIZE: i32 = 1;
    const NUM_NEG: i32 = 3;
    const LR: f32 = 0.001;
    const EPOCHS: usize = 5;

    let mut net = Network::new(NUM_WORDS, VEC_SIZE, LR);
    let words: HashMap<&str, i32> = make_lookup(&ex_sentence);
    let rev: HashMap<i32, &str> = reverse_lookup(&words);

    let mapped_sentence = ex_sentence
        .iter()
        .map(|x| words.get(x).unwrap())
        .collect::<Vec<&i32>>();

    println!("{:?}", mapped_sentence);
    let max_s = mapped_sentence.len() as i32;

    let mut losses = CircularQueue::with_capacity(500);
    for e in 0..EPOCHS {
        for i in 0..max_s {
            let target = mapped_sentence[i as usize];
            for j in i - WINDOW_SIZE..i + WINDOW_SIZE + 1 {
                // Only take valid samples, and not target ind
                if j >= 0 && j < max_s && j != i {
                    let pos = mapped_sentence[j as usize];
                    for z in 0..NUM_NEG + 1 {
                        let vec_ind: &i32;
                        let label: f32;
                        if z == 0 {
                            vec_ind = pos;
                            label = 1.;
                        } else {
                            // negatives can be anything except pos and target
                            // should be drawn according to distribution and params
                            vec_ind = &8; // TODO DRAW NEGATIVE RANDOMLY
                            label = -1.;
                        }
                        //TODO keep losses to display progress.
                        let loss = net.forward(vec_ind, target, label);
                        losses.push(loss);

                        //SUPER-TODO, we need to keep the lists of gradients outside the model
                        // and then join them when the threads get joined?
                    }
                }
            }
            println!(
                "Loss: {}",
                losses.iter().fold(0., |sum, i| sum + i) / losses.len() as f32
            );
        }
    }
    //words.insert()

    // let vecs: CsMat<f32> = CsMat::zero((num_words, vec_size));
    // println!("{}", vecs.cols());
    // println!("{}", vecs.to_dense().slice(s![5, 0, 0]));

    // Do calculation on targ --> pos_exs and neg_exs. Return sparse vector
    // apply this sparse vector to global scope.
}
