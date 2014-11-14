#![crate_type="lib"]
#![feature(globs, macro_rules, unsafe_destructor)]

extern crate libc;
extern crate sync;
extern crate num;

extern crate "fftw3-sys" as ffi;

pub use mem::FftwVec;
pub use plan::{Plan, RawPlan};

mod plan;
mod mem;

pub mod builder;

pub mod wisdom;
pub mod lock;


#[test]
fn test() {
    let mut p = plan::Plan::r2c_1d(4);
    p.input()[0] = 1.0;
    p.input()[1] = 1.0;

    println!("{}", p.execute());
}
