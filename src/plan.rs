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
    pub fn new(f: || -> ffi::fftw_plan) -> Option<RawPlan> {
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

    pub unsafe fn execute(&mut self) {
        ffi::fftw_execute(self.plan)
    }
}

impl Drop for RawPlan {
    fn drop(&mut self) {
        unsafe {ffi::fftw_destroy_plan(self.plan)}
    }
}
