#![feature(globs)]
extern crate fftw3;

use fftw3::FftwVec;
use fftw3::builder::*;

fn main() {
    let input = FftwVec::zeros(16);
    let output = FftwVec::zeros(9);

    let mut plan = Planner::new()
        .rigor(Rigor::Estimate)
        .direction(Direction::Forward)
        .r2c(input, output)
        .plan()
        .ok().expect("failed to create plan");

    for v in plan.input().iter_mut() { *v = 1.0 }
    println!("{:?}", plan.input());

    plan.execute();

    println!("{:?}", plan.output());

    let mut inverse = Planner::new()
        .c2r(FftwVec::zeros(9), FftwVec::zeros(16))
        .plan()
        .ok().expect("failed to create plan");
    for (a, b) in inverse.input().iter_mut().zip(plan.output().unwrap().iter()) {
        *a = *b
    }
    inverse.execute();
    println!("{:?}", inverse.output());
}
