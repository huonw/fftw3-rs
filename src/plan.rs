use {ffi, lock};
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
    pub fn new(f: || -> ffi::fftw_plan) -> RawPlan {
        let plan = lock::run(f);
        if plan.is_null() {
            panic!("RawPlan::new: created a NULL plan");
        }

        RawPlan { plan: plan }
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
pub struct Plan<In, Out> {
    raw: RawPlan,
    in_: In,
    out: Out
}

impl<In, Out> Plan<In, Out> {
    pub fn take_out(self) -> Out {
        self.out
    }

    pub fn take_in(self) -> In {
        self.in_
    }
}

impl<I, In: DerefMut<[I]>, Out> Plan<In, Out> {
    pub fn input<'a>(&'a mut self) -> &'a mut [I] {
        &mut *self.in_
    }
}

impl<In, O, Out: DerefMut<[O]>> Plan<In, Out> {
    pub fn execute<'a>(&'a mut self) -> &'a mut [O] {
        unsafe {
            self.raw.execute()
        }

        &mut *self.out
    }
}

impl<In: DerefMut<[f64]>, Out: DerefMut<[Complex64]>> Plan<In, Out> {
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
                })
        };
        Plan { raw: plan, in_: in_, out: out }
    }
}
impl Plan<FftwVec<f64>, FftwVec<Complex64>> {
    pub fn r2c_1d(n: uint) -> Plan<FftwVec<f64>, FftwVec<Complex64>> {
        unsafe {
            let (in_, out) = (FftwVec::uninit(n), FftwVec::uninit(n / 2 + 1));

            Plan::r2c_1d_prealloc(in_, out)
        }
    }
}
impl<In: DerefMut<[Complex64]>, Out: DerefMut<[f64]>> Plan<In, Out> {
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
                })
        };
        Plan { raw: plan, in_: in_, out: out }
    }
}
impl Plan<FftwVec<Complex64>, FftwVec<f64>> {
    pub fn c2r_1d(n: uint) -> Plan<FftwVec<Complex64>, FftwVec<f64>> {
        unsafe {
            let (in_, out) = (FftwVec::uninit(n / 2 + 1), FftwVec::uninit(n));

            Plan::c2r_1d_prealloc(in_, out)
        }
    }
}

impl<In: DerefMut<[Complex64]>, Out: DerefMut<[Complex64]>> Plan<In, Out> {
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
                })
        };
        Plan { raw: plan, in_: in_, out: out }
    }
}
impl Plan<FftwVec<Complex64>, FftwVec<Complex64>> {
    pub fn c2c_1d(n: uint) -> Plan<FftwVec<Complex64>, FftwVec<Complex64>> {
        unsafe {
            let (in_, out) = (FftwVec::uninit(n), FftwVec::uninit(n));

            Plan::c2c_1d_prealloc(in_, out)
        }
    }
}
