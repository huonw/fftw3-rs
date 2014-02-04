use {ffi, lock};
use mem::{BackingStorage, FFTWVec};

use extra::complex::Complex64;

pub struct RawPlan {
    priv plan: ffi::fftw_plan
}
impl RawPlan {
    /// Create a `RawPlan` from the output of `f`.
    ///
    /// This executes `f` inside a lock since FFTW plan creation is
    /// not threadsafe.
    pub fn new(f: || -> ffi::fftw_plan) -> RawPlan {
        let plan = lock::run(f);
        if plan.is_null() {
            fail!("RawPlan::new: created a NULL plan");
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

pub struct Plan<In, Out> {
    priv raw: RawPlan,
    priv in_: In,
    priv out: Out
}

impl<I, In: BackingStorage<I>, Out> Plan<In, Out> {
    pub fn input<'a>(&'a mut self) -> &'a mut [I] {
        self.in_.storage_slice()
    }
}

impl<In, O, Out: BackingStorage<O>> Plan<In, Out> {
    pub fn execute<'a>(&'a mut self) -> &'a [O] {
        unsafe {
            self.raw.execute()
        }

        self.out.storage_slice().as_slice()
    }
}

impl<In: BackingStorage<f64>, Out: BackingStorage<Complex64>> Plan<In, Out> {
    pub fn r2c_1d_prealloc(mut in_: In, mut out: Out) -> Plan<In, Out> {
        let plan = {
            let islice = in_.storage_slice();
            let oslice = out.storage_slice();

            let n = islice.len();
            if oslice.len() < n / 2 + 1 {
                fail!("Plan::r2c_prealloc: `out` has length {}, but requires at least {}",
                      oslice.len(), n / 2 + 1)
            }
            RawPlan::new(|| unsafe {
                    ffi::fftw_plan_dft_r2c_1d(n as i32,
                                              islice.as_mut_ptr(),
                                              oslice.as_mut_ptr() as *mut ffi::fftw_complex,
                                              ffi::FFTW_MEASURE)
                })
        };
        Plan { raw: plan, in_: in_, out: out }
    }
}
impl Plan<FFTWVec<f64>, FFTWVec<Complex64>> {
    pub fn r2c_1d(n: uint) -> Plan<FFTWVec<f64>, FFTWVec<Complex64>> {
        unsafe {
            let (in_, out) = (FFTWVec::uninit(n), FFTWVec::uninit(n / 2 + 1));

            Plan::r2c_1d_prealloc(in_, out)
        }
    }
}
