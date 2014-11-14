extern crate "pkg-config" as pkg_config;
fn main() {
    match pkg_config::find_library("fftw3") {
        Ok(()) => return,
        Err(..) => {}
    }

    println!("failed to find libfftw3");
    std::os::set_exit_status(1);
}
