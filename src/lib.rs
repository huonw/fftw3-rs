#![crate_type="lib"]

#[macro_use]
extern crate lazy_static;
#[macro_use]
extern crate log;
extern crate libc;
extern crate num;
extern crate strided;

extern crate fftw3_sys as ffi;

pub use mem::FftwVec;
pub use plan::{Plan, RawPlan};
pub use builder2::Planner;

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

    println!("{:?}", p.execute());
}
