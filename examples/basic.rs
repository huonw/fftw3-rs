extern crate fftw;

use fftw::{wisdom, plan};

fn main() {
    let n: uint = from_str(std::os::args()[1].as_slice()).expect("./basic integer");
    let p = Path::new(format!("wisdom-{}.fftw", n));
    let loaded = wisdom::import_from_file(p.clone());

    let mut plan = plan::Plan::r2c_1d(1 << n);
    plan.execute();

    if !loaded { wisdom::export_to_file(p); }
}
