#[crate_id="fftw"];
#[crate_type="lib"];
#[feature(globs)];

extern mod extra;

pub mod ffi;
pub mod plan;
pub mod mem;
pub mod wisdom;

pub mod lock {
    use std::unstable::finally::Finally;
    use std::unstable::mutex::{Mutex, MUTEX_INIT};

    static mut LOCK: Mutex = MUTEX_INIT;

    pub fn run<A>(f: || -> A) -> A {
        unsafe {LOCK.lock();}
        f.finally(|| unsafe {LOCK.unlock()})
    }
}


#[test]
fn test() {
    let mut p = plan::Plan::r2c_1d(4);
    p.input()[0] = 1.0;
    p.input()[1] = 1.0;

    println!("{:?}", p.execute());
}
