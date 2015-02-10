#![feature(core)]
extern crate fftw3;

extern crate num;
extern crate strided;

fn main() {

    let mut in_ = [0f64; 10];
    let mut out = [num::Complex::new(0f64, 0f64); 10];
    let p
        = fftw3::Planner::new()
        .input(in_.as_mut_slice())
        .output(out.as_mut_slice())
        ._1d(10);

    let mut p = p.plan()
        .unwrap();

    for (i, place) in p.input().iter_mut().enumerate() {
        *place = i as f64
    }
    p.execute();
    println!("{:?}", p.output());
}
