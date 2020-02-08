extern crate ndarray;
extern crate ndarray_rand;
use ndarray::{s, Array, Array1, ArrayView, ArrayViewMut};
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

fn main() {
    let ex_sentence = [
        "i", "walked", "on", "the", "magical", "white", "shore", "in", "my", "white", "shoes",
        "and", "white", "sky",
    ];
    const NUM_WORDS: i32 = 1000;
    const VEC_SIZE: i32 = 100;
    const NUM_ELEMENTS: i32 = NUM_WORDS * VEC_SIZE;
    const WINDOW_SIZE: i32 = 1;

    let udist = Uniform::from(-0.5..0.5);
    let mut output_embeddings = Array1::<f32>::zeros(NUM_ELEMENTS as usize);
    let mut hidden_embeddings = Array::random(NUM_ELEMENTS as usize, udist);
    println!("{:?}", hidden_embeddings);

    let words: HashMap<&str, i32> = make_lookup(&ex_sentence);
    let rev: HashMap<i32, &str> = reverse_lookup(&words);

    let mapped_sentence = ex_sentence
        .iter()
        .map(|x| words.get(x).unwrap())
        .collect::<Vec<&i32>>();

    println!("{:?}", mapped_sentence);

    let select_output = |tar| output_embeddings.slice(s![tar * VEC_SIZE..(tar + 1) * VEC_SIZE]);
    //let select_hidden = |tar| hidden_embeddings.slice(s![tar * VEC_SIZE..(tar + 1) * VEC_SIZE]);

    let max_s = mapped_sentence.len() as i32;
    for i in 0..max_s {
        let target_vec = select_output(mapped_sentence[i as usize]);
        let mut local_gradient = Array1::<f32>::zeros(VEC_SIZE as usize); //neu1e
        let mut test = Array1::<f32>::ones(VEC_SIZE as usize);
        for j in i - WINDOW_SIZE..i + WINDOW_SIZE + 1 {
            // Only take valid samples, and not target ind
            if j >= 0 && j < max_s && j != i {
                let pos_vec = select_output(mapped_sentence[j as usize]);
                let res = target_vec.dot(&pos_vec);

                let g = res; //TODO more complex gradient
                test.fill(g);
                //TODO COMBINE LOCAL GRADIENT WITH VECTOR
                local_gradient += test * pos_vec;
                println!("rvec {} * {}: {:?}", i, j, res);
                //TODO also calculate negative samples
                // update accumulated gradient of
            }
        }
    }
    //words.insert()

    // let vecs: CsMat<f32> = CsMat::zero((num_words, vec_size));
    // println!("{}", vecs.cols());
    // println!("{}", vecs.to_dense().slice(s![5, 0, 0]));

    // Do calculation on targ --> pos_exs and neg_exs. Return sparse vector
    // apply this sparse vector to global scope.
}
