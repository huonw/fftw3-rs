extern crate "pkg-config" as pkg_config;
fn main() {
    match pkg_config::find_library("fftw3") {
        Ok(()) => return,
        Err(..) => {}
    }

    println!("failed to find libfftw3, and automatically building it is currently unimplemented.");
    std::os::set_exit_status(1);
}
