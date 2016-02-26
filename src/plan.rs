use {ffi, lock};
use std::ops::DerefMut;
use mem::FftwVec;

use num::complex::Complex64;

/// A thin wrapper around the internal FFTW plan type. Prefer `Plan`
/// if possible.
pub struct RawPlan {
    plan: ffi::fftw_plan
}
impl RawPlan {
    /// Create a `RawPlan` from the output of `f`.
    ///
    /// This executes `f` inside a lock since FFTW plan creation is
    /// not threadsafe.
    pub fn new<F: FnOnce() -> ffi::fftw_plan>(f: F) -> Option<RawPlan> {
        let plan = lock::run(f);

        if plan.is_null() {
            None
        } else {
            Some(RawPlan { plan: plan })
        }
    }

    /// Create a `RawPlan` directly from an `fftw_plan`, with no
    /// synchronisation or checks. Prefer `RawPlan::new` where possible.
    pub unsafe fn new_unchecked(plan: ffi::fftw_plan) -> RawPlan {
        RawPlan { plan: plan }
    }

    /// Print information about the plan to stdout.
    pub fn debug_print(&self) {
        unsafe {ffi::fftw_print_plan(self.plan);}
    }

    /// Execute the plan
    pub unsafe fn execute(&mut self) {
        ffi::fftw_execute(self.plan)
    }
}

impl Drop for RawPlan {
    fn drop(&mut self) {
        unsafe {ffi::fftw_destroy_plan(self.plan)}
    }
}

/// The structure representing the computation of an FFT.
///
/// Manages a native FFTW3 plan, and access to the associated input and output
/// arrays.
pub struct Plan<In, Out> {
    raw: RawPlan,
    in_: In,
    out: Out
}

impl<In, Out> Plan<In, Out> {
    /// Get immutable access to the output of the plan.
    pub fn take_out(self) -> Out {
        self.out
    }

    /// Get immutable access to the input of the plan.
    pub fn take_in(self) -> In {
        self.in_
    }
}

impl<In: DerefMut, Out> Plan<In, Out> {
    /// Get a mutable reference to the plan's input.
    ///
    /// This function is usually used to fill the input buffer with a new set
    /// of data before [executing](struct.Plan.html#method.execute).
    pub fn input<'a>(&'a mut self) -> &'a mut In::Target {
        &mut *self.in_
    }
}

impl<In, Out: DerefMut> Plan<In, Out> {
    /// Execute the plan.
    ///
    /// Compute the FFT of the plan's input buffer.
    /// Returns a mutable reference to the output buffer.
    pub fn execute<'a>(&'a mut self) -> &'a mut Out::Target {
        unsafe {
            self.raw.execute()
        }

        &mut *self.out
    }
}

impl<In: DerefMut<Target = [f64]>, Out: DerefMut<Target = [Complex64]>> Plan<In, Out> {
    /// Create a real to complex plan with preallocated buffers.
    ///
    /// # Panics
    /// If the length of `in` is more than twice the length of `out`, we will panic.
    /// This is because we need to have at least `in.len()` elements worth of output,
    /// and every `Complex64` is 2 elements.
    pub fn r2c_1d_prealloc(mut in_: In, mut out: Out) -> Plan<In, Out> {
        let plan = {
            let n = in_.len();
            if out.len() < n / 2 + 1 {
                panic!("Plan::r2c_prealloc: `out` has length {}, but requires at least {}",
                      out.len(), n / 2 + 1)
            }
            RawPlan::new(|| unsafe {
                    ffi::fftw_plan_dft_r2c_1d(n as i32,
                                              in_.as_mut_ptr(),
                                              out.as_mut_ptr() as *mut ffi::fftw_complex,
                                              //ffi::FFTW_MEASURE
                                              ffi::FFTW_ESTIMATE
                                              )
                }).unwrap()
        };
        Plan { raw: plan, in_: in_, out: out }
    }
}
impl Plan<FftwVec<f64>, FftwVec<Complex64>> {
    /// A transformation from real numbers to complex numbers.
    ///
    /// See [`r2c_1d_prealloc`](struct.Plan.html#method.r2c_1d_prealloc) for more information
    pub fn r2c_1d(n: usize) -> Plan<FftwVec<f64>, FftwVec<Complex64>> {
        unsafe {
            let (in_, out) = (FftwVec::uninit(n), FftwVec::uninit(n / 2 + 1));

            Plan::r2c_1d_prealloc(in_, out)
        }
    }
}
impl<In: DerefMut<Target = [Complex64]>, Out: DerefMut<Target = [f64]>> Plan<In, Out> {
    /// Creates a complex to real plan with preallocated buffers.
    ///
    /// # Panics
    /// If the length of `in` is more than twice the length of `out`, we will panic.
    /// This is because we need to have at least `in.len()` elements worth of output,
    /// and every `Complex64` is 2 elements.
    pub fn c2r_1d_prealloc(mut in_: In, mut out: Out) -> Plan<In, Out> {
        let plan = {
            let n = out.len();
            if in_.len() > n / 2 + 1 {
                panic!("Plan::r2c_prealloc: `in_` has length {}, but requires at most {}",
                      in_.len(), n / 2 + 1)
            }
            RawPlan::new(|| unsafe {
                    ffi::fftw_plan_dft_c2r_1d(n as i32,
                                              in_.as_mut_ptr() as *mut ffi::fftw_complex,
                                              out.as_mut_ptr(),
                                              ffi::FFTW_ESTIMATE)
                }).unwrap()
        };
        Plan { raw: plan, in_: in_, out: out }
    }
}
impl Plan<FftwVec<Complex64>, FftwVec<f64>> {
    /// Creates a complex to real transformation
    ///
    /// See [c2r_1d_prealloc](struct.Plan.html#method.c2r_1d_prealloc) for more information
    pub fn c2r_1d(n: usize) -> Plan<FftwVec<Complex64>, FftwVec<f64>> {
        unsafe {
            let (in_, out) = (FftwVec::uninit(n / 2 + 1), FftwVec::uninit(n));

            Plan::c2r_1d_prealloc(in_, out)
        }
    }
}

impl<In: DerefMut<Target = [Complex64]>, Out: DerefMut<Target = [Complex64]>> Plan<In, Out> {
    /// Create a complex to complex plan with preallocated buffers.
    ///
    /// # Panics
    /// If the length of `in` is more than the length of `out`, we will panic.
    /// This is because we need to have at least `in.len()` elements worth of output.
    pub fn c2c_1d_prealloc(mut in_: In, mut out: Out) -> Plan<In, Out> {
        let plan = {
            let n = in_.len();
            if out.len() < n {
                panic!("Plan::r2c_prealloc: `out` has length {}, but requires at least {}",
                      out.len(), n)
            }
            RawPlan::new(|| unsafe {
                    ffi::fftw_plan_dft_1d(n as i32,
                                          in_.as_mut_ptr() as *mut ffi::fftw_complex,
                                          out.as_mut_ptr() as *mut ffi::fftw_complex,
                                          ffi::FFTW_FORWARD,
                                          ffi::FFTW_ESTIMATE
                                          //ffi::FFTW_MEASURE
                                          )
                }).unwrap()
        };
        Plan { raw: plan, in_: in_, out: out }
    }
}
impl Plan<FftwVec<Complex64>, FftwVec<Complex64>> {
    /// Creates a complex to complex transformation
    ///
    /// See [c2c_1d_prealloc](struct.Plan.html#method.c2c_1d_prealloc) for more information
    pub fn c2c_1d(n: usize) -> Plan<FftwVec<Complex64>, FftwVec<Complex64>> {
        unsafe {
            let (in_, out) = (FftwVec::uninit(n), FftwVec::uninit(n));

            Plan::c2c_1d_prealloc(in_, out)
        }
    }
}
