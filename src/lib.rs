#![crate_type="lib"]

//! High-level bindings to [FFTW3](http://fftw.org/): Fastest Fourier
//! Transform in the West.
//!
//! At the moment, this only provides one-dimensional
//! complex-to-complex transforms via `c2c_1d`, with no explicit plan
//! reuse.
//!
//! There are some modules that provide helpers to make using the
//! low-level interface manually slightly nicer and safer, in
//! particular the `lock` module assists with keeping the use of FFTW3
//! threadsafe, and the `plan` module gives a nicer interface to
//! creating and using plans.

extern crate libc;
extern crate sync;
extern crate num;

extern crate "fftw3-sys" as ffi;

use std::{error, fmt, mem};
use num::Complex;

pub mod plan;
pub mod lock;

/// Errors that can occur while planning or running a fourier transform.
pub enum FftError {
    /// The input slice was too long for the the output (elements are
    /// the lengths of those respective slices).
    InsufficientSpace(uint, uint),
    /// FFTW returned an error (a null plan) when planning.
    FailedPlan,
}

impl fmt::Show for FftError {
    fn fmt(&self, fmtr: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            FftError::InsufficientSpace(in_, out) => {
                write!(fmtr, "insufficient output space: given {}, needed {}", out, in_)
            }
            FftError::FailedPlan => write!(fmtr, "planning failed"),
        }
    }
}

impl error::Error for FftError {
    fn description(&self) -> &str {
        match *self {
            FftError::InsufficientSpace(_, _) => "insufficient output space",
            FftError::FailedPlan => "planning failed"
        }
    }
    fn detail(&self) -> Option<String> {
        match *self {
            FftError::InsufficientSpace(in_, out) => Some(format!("given {}, needed {}", out, in_)),
            FftError::FailedPlan => None,
        }
    }
}

/// Perform a complex-to-complex one-dimensional Fourier transform of
/// `in_`, writing to `out`.
///
/// The direction is controlled by `forward`: `true` for the forward
/// transform (that is, multiplying by `exp(-2πi/N)`) and `false` for
/// the inverse (`exp(2πi/N)`). This uses the ESTIMATE rigor for
/// making a plan to avoid overwriting `in_` and `out` while planning,
/// hence this definitely does not achieve the peak performance
/// possible with FFTW.
///
/// An error is returned if `out` is shorter than `in_`, since there
/// is insufficient space to perform the transform, or if the FFTW3
/// planning function does not succeed.
///
/// # Example
///
/// ```rust
/// extern crate fftw3;
/// extern crate num;
/// use num::Complex;
///
/// fn main() {
///     let input = [Complex::new(2.0, 0.0), Complex::new(1.0, 1.0),
///                  Complex::new(0.0, 3.0), Complex::new(2.0, 4.0)];
///     let mut output = [Complex::new(0.0, 0.0), .. 4];
///
///     fftw3::c2c_1d(&input, &mut output, true).unwrap(); // panic on error
///
///     println!("the transform of {} is {}", input.as_slice(), output.as_slice());
/// }
/// ```
pub fn c2c_1d(in_: &[Complex<f64>], out: &mut [Complex<f64>],
              forward: bool) -> Result<(), FftError> {
    assert!(in_.len() < (1u << (8*mem::size_of::<libc::c_int>() - 1)) - 1);

    if out.len() < in_.len() {
        return Err(FftError::InsufficientSpace(in_.len(), out.len()))
    }

    let sign = if forward { ffi::FFTW_FORWARD } else { ffi::FFTW_BACKWARD };

    unsafe {
        let plan = plan::RawPlan::new(|| {
            ffi::fftw_plan_dft_1d(in_.len() as libc::c_int,
                                  // this is only correct because we use ESTIMATE
                                  in_.as_ptr() as *mut ffi::fftw_complex,
                                  out.as_mut_ptr() as *mut ffi::fftw_complex,
                                  sign,
                                  ffi::FFTW_ESTIMATE)
        });

        match plan {
            None => Err(FftError::FailedPlan),
            Some(mut p) => {
                p.execute();
                Ok(())
            }
        }
    }
}


#[cfg(test)]
mod tests {
    use num::Complex;
    use std::rand;
    #[test]
    fn c2c_1d_roundtrip() {
        let input = Vec::from_fn(1000, |_| Complex::new(rand::random(),
                                                        rand::random()));
        let mut output = Vec::from_elem(1000, Complex::new(0.0,0.0));
        let mut output2 = Vec::from_elem(1000, Complex::new(0.0,0.0));

        super::c2c_1d(&*input, &mut *output, true).unwrap();
        super::c2c_1d(&*output, &mut *output2, false).unwrap();

        for (x, raw_y) in input.iter().zip(output2.iter()) {
            let y = raw_y.unscale(1000.0);
            if (*x - y).norm() >= 1e-10 {
                panic!("roundtrip elements too far apart: {}, {}", *x, y)
            }
        }
    }
}
