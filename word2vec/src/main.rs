extern crate ndarray;
use ndarray::s;
use ndarray::array;
use ndarray::Array1;
use ndarray::aview1;


fn main() {
    const NUM_WORDS: usize = 1000;
    const VEC_SIZE: usize = 100;
    const NUM_ELEMENTS: usize = NUM_WORDS * VEC_SIZE;

    let mut test = Array1::<f64>::zeros(NUM_ELEMENTS);
    println!("{:?}", test);
    test += 1.0;
    println!("{:?}", test);
    println!("{:?}", test.slice(s![..32]));

    // let vecs: CsMat<f32> = CsMat::zero((num_words, vec_size));
    // println!("{}", vecs.cols());
    // println!("{}", vecs.to_dense().slice(s![5, 0, 0]));
}
