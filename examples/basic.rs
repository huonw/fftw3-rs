extern mod fftw;

use fftw::{wisdom, plan};

fn main() {
    let n: uint = from_str(*std::os::args().get(1).expect("./basic integer")).expect("./basic integer");
    let p = Path::new(format!("wisdom-{}.fftw", n));
    let loaded = wisdom::import_from_file(p.clone());

    let mut plan = plan::Plan::r2c_1d(1 << n);
    plan.execute();

    if !loaded { wisdom::export_to_file(p); }
}
