#![feature(os, env, path)]
extern crate fftw3;

use fftw3::wisdom;

fn main() {
    let n: usize = std::env::args().nth(1).unwrap().into_string().unwrap().parse().ok().expect("./basic integer");
    let p = Path::new(format!("wisdom-{}.fftw", n));
    let loaded = wisdom::import_from_file(p.clone());

    let mut plan = fftw3::Plan::r2c_1d(1 << n);
    plan.execute();

    if !loaded { wisdom::export_to_file(p); }
}
