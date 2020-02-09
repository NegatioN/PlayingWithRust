extern crate ndarray;
extern crate ndarray_rand;
use ndarray::{s, Array, Array1, ArrayBase, ArrayViewMut};
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

struct Network {
    output_embeddings: Array1<f32>,
    hidden_embeddings: Array1<f32>,
    vec_size: i32,
    num_words: i32,
}

impl Network {
    fn new(num_words: i32, vec_size: i32) -> Self {
        let vec_max_val = 0.5 / num_words as f32;
        let udist = Uniform::from(-vec_max_val..vec_max_val);
        let num_elements = num_words * vec_size;
        let mut output_embeddings = Array1::<f32>::zeros(num_elements as usize);
        let mut hidden_embeddings = Array::random(num_elements as usize, udist);
        Network {
            output_embeddings,
            hidden_embeddings,
            vec_size,
            num_words,
        }
    }

    fn forward(&mut self, ind: &i32, tar_ind: &i32) -> f32 {
        let vec = self
            .output_embeddings
            .slice(s![ind * self.vec_size..(ind + 1) * self.vec_size]);
        let tar_vec = self
            .output_embeddings
            .slice(s![tar_ind * self.vec_size..(tar_ind + 1) * self.vec_size]);
        tar_vec.dot(&vec)
    }

    // the opposite vector * error
    // the error equals logsigmoid(res) distance from 0
    //the paper says pos +
    //fn backward(&mut self, posneg_gradients){

    //}
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

    let mut net = Network::new(NUM_WORDS, VEC_SIZE);
    let words: HashMap<&str, i32> = make_lookup(&ex_sentence);
    let rev: HashMap<i32, &str> = reverse_lookup(&words);

    let mapped_sentence = ex_sentence
        .iter()
        .map(|x| words.get(x).unwrap())
        .collect::<Vec<&i32>>();

    println!("{:?}", mapped_sentence);

    //let select_output = |tar| output_embeddings.slice(s![tar * VEC_SIZE..(tar + 1) * VEC_SIZE]);
    //let select_hidden = |tar| hidden_embeddings.slice_mut(s![tar * VEC_SIZE..(tar + 1) * VEC_SIZE]);

    let max_s = mapped_sentence.len() as i32;
    for i in 0..max_s {
        let target = mapped_sentence[i as usize];
        //let target_vec = select_vec(target, hidden_embeddings, &VEC_SIZE);
        let mut local_gradient = Array1::<f32>::zeros(VEC_SIZE as usize); //neu1e
        for j in i - WINDOW_SIZE..i + WINDOW_SIZE + 1 {
            // Only take valid samples, and not target ind
            if j >= 0 && j < max_s && j != i {
                let pos = mapped_sentence[j as usize];
                for z in 0..NUM_NEG + 1 {
                    let vec_ind: &i32;
                    if z == 0 {
                        vec_ind = pos;
                    } else {
                        // negatives can be anything except pos and target
                        // should be drawn according to distribution and params
                        vec_ind = &8; // TODO DRAW NEGATIVE RANDOMLY
                    }
                    let res = net.forward(vec_ind, target);

                    //WHAT DO I NEED TO STORE TO KEEP FOR BACKWARDS??
                    // result of dot-product
                    // to have access to the vector
                    // the accumulated gradient over pos, neg * targets?
                    // the target index to apply gradient on

                    //let vec = select_output(vec_ind);
                    //let res = target_vec.dot(&vec);
                    let g = res; //TODO more complex gradient
                                 //TODO COMBINE LOCAL GRADIENT WITH VECTOR
                                 //local_gradient += &(g * &vec);
                    println!("rvec {} * {}: {:?}", i, j, res);
                }
            }
        }
        println!("Grad: {:?}", local_gradient);
    }
    //words.insert()

    // let vecs: CsMat<f32> = CsMat::zero((num_words, vec_size));
    // println!("{}", vecs.cols());
    // println!("{}", vecs.to_dense().slice(s![5, 0, 0]));

    // Do calculation on targ --> pos_exs and neg_exs. Return sparse vector
    // apply this sparse vector to global scope.
}
