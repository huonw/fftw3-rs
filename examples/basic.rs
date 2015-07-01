extern crate fftw3;

use std::path::Path;
use fftw3::wisdom;

fn main() {
    let n: usize = std::env::args().nth(1).unwrap().parse().ok().expect("./basic integer");
    let title = format!("wisdom-{}.fftw", n);
    let p = Path::new(&title);
    let loaded = wisdom::import_from_file(p.clone());

    let mut plan = fftw3::Plan::r2c_1d(1 << n);
    plan.execute();

    if !loaded { wisdom::export_to_file(p); }
}
