extern crate ndarray;
use ndarray::s;
use ndarray::Array1;

use std::collections::HashMap;

fn make_lookup<'a>(arr: &[&'a str]) -> HashMap<&'a str, i32> {
    let mut words: HashMap<&str, i32> = HashMap::new();

    let mut counter = 0;
    for word in arr.iter() {
        if !words.contains_key(word) {
            words.insert(word.clone(), counter);
            counter += 1;
        }
    }

    words
}

fn reverse_lookup(lookup: HashMap<&str, i32>) -> HashMap<i32, &str> {
    let mut words: HashMap<i32, &str> = HashMap::new();
    for (k,v) in lookup.iter() {
        words.insert(*v, k);
    }
    words
}

fn main() {
    let ex_sentence = ["i", "walked", "on", "the", "magical", "white", "shore", "in", "my", "white", "shoes", "and", "white", "sky"];
    const NUM_WORDS: usize = 1000;
    const VEC_SIZE: usize = 100;
    const NUM_ELEMENTS: usize = NUM_WORDS * VEC_SIZE;


    let mut test = Array1::<f64>::zeros(NUM_ELEMENTS);
    println!("{:?}", test);
    test += 1.0;
    println!("{:?}", test);
    println!("{:?}", test.slice(s![..32]));


    let words: HashMap<&str, i32> = make_lookup(&ex_sentence);
    let rev = reverse_lookup(words);
    //words.insert()

    // let vecs: CsMat<f32> = CsMat::zero((num_words, vec_size));
    // println!("{}", vecs.cols());
    // println!("{}", vecs.to_dense().slice(s![5, 0, 0]));
}
