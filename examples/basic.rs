extern crate fftw3;

use fftw3::wisdom;

fn main() {
    let n: uint = from_str(std::os::args()[1].as_slice()).expect("./basic integer");
    let p = Path::new(format!("wisdom-{}.fftw", n));
    let loaded = wisdom::import_from_file(p.clone());

    let mut plan = fftw3::Plan::r2c_1d(1 << n);
    plan.execute();

    if !loaded { wisdom::export_to_file(p); }
}
