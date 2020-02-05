extern crate ndarray;

use ndarray::array;
use ndarray::aview1;

fn main() {
    const NUM_WORDS: usize = 1000;
    const VEC_SIZE: usize = 100;
    const NUM_ELEMENTS: usize = NUM_WORDS * VEC_SIZE;

    let a1 = array![1, 2, 3, 4];

    println!("{}", a1.shape()[0]);

    let data = [0.0; NUM_ELEMENTS];

    println!("{:?}", &data[500..550]);

    // Create a 2D array view from borrowed data
    let a2d = aview1(&data).into_shape((NUM_WORDS, VEC_SIZE)).unwrap();

    println!("{}", a2d);
    println!("{} {}", a2d.shape()[0], a2d.shape()[1]);

    // let vecs: CsMat<f32> = CsMat::zero((num_words, vec_size));
    // println!("{}", vecs.cols());
    // println!("{}", vecs.to_dense().slice(s![5, 0, 0]));
}
