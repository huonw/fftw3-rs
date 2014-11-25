#![crate_type="lib"]
#![feature(default_type_params, associated_types)]
#![feature(globs, macro_rules, unsafe_destructor, tuple_indexing, phase)]

#[phase(plugin, link)]
extern crate log;
extern crate libc;
extern crate sync;
extern crate num;
extern crate strided;

extern crate "fftw3-sys" as ffi;

pub use mem::FftwVec;
pub use plan::{Plan, RawPlan};

mod plan;
mod mem;

pub mod builder;
pub mod builder2;

pub mod wisdom;
pub mod lock;

pub mod traits;

#[test]
fn test() {
    let mut p = plan::Plan::r2c_1d(4);
    p.input()[0] = 1.0;
    p.input()[1] = 1.0;

    println!("{}", p.execute());
}
