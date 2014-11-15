use ffi;
use libc::{c_uint, c_int, c_void};
use num::complex::Complex64;

use plan::RawPlan;

/// How much effort FFTW should put into computing the best strategy
/// to use.
///
/// The `FFTW_WISDOM_ONLY` rigor level is replaced by the
pub enum Rigor {
    Estimate,
    Measure,
    Patient,
    Exhaustive,
}
impl Rigor {
    fn flags(self) -> c_uint {
        match self {
            Estimate => ffi::FFTW_ESTIMATE,
            Measure => ffi::FFTW_MEASURE,
            Patient => ffi::FFTW_PATIENT,
            Exhaustive => ffi::FFTW_EXHAUSTIVE,
        }
    }
}

/// The direction of the transform to perform..
pub enum Direction {
    Forward, Backward
}

/// Control the basic properties of a set of transforms.
pub struct Planner {
    rigor: Rigor,
    wisdom_restriction: bool,
    direction: Direction,
}

impl Planner {
    /// Construct a new planner with default values.
    ///
    /// This defaults to a forward transform with estimate rigor.
    pub fn new() -> Planner {
        Planner {
            rigor: Estimate,
            wisdom_restriction: false,
            direction: Forward,
        }
    }

    /// Set the rigor to use for this plan.
    pub fn rigor(&mut self, r: Rigor) -> &mut Planner {
        self.rigor = r;
        self
    }
    /// Set whether the planner should only be successfully created if
    /// there exists wisdom created with at least the rigor level set.
    pub fn wisdom_restriction(&mut self, wisdom_only: bool) -> &mut Planner {
        self.wisdom_restriction = wisdom_only;
        self
    }

    /// Set the direction of the transform to perform.
    pub fn direction(&mut self, direction: Direction) -> &mut Planner {
        self.direction = direction;
        self
    }

    fn flags(&self) -> c_uint {
        self.rigor.flags() | if self.wisdom_restriction {
            ffi::FFTW_WISDOM_ONLY
        } else {
            0
        }
    }
    fn dir(&self) -> c_int {
        match self.direction {
            Forward => ffi::FFTW_FORWARD,
            Backward => ffi::FFTW_BACKWARD,
        }
    }

    pub fn c2c<I, O>(&self, in_: I, out: O) -> PlanMem<I, O>
        where I: DerefMut<[Complex64]>, O: DerefMut<[Complex64]>
    {
        assert!(in_.len() <= out.len());
        PlanMem {
            plan: *self,
            n: in_.len() as c_int,
            in_: in_,
            out: Some(out),
            planner: c2c,
        }
    }
    pub fn c2r<I, O>(&self, in_: I, out: O) -> PlanMem<I, O>
        where I: DerefMut<[Complex64]>, O: DerefMut<[f64]>
    {
        assert!(in_.len() <= out.len() / 2 + 1);
        PlanMem {
            plan: *self,
            n: 2 * (in_.len() as c_int - 1),
            in_: in_,
            out: Some(out),
            planner: c2r,
        }
    }
    pub fn r2c<I, O>(&self, in_: I, out: O) -> PlanMem<I, O>
        where I: DerefMut<[f64]>, O: DerefMut<[Complex64]>
    {
        assert!(in_.len() / 2 + 1 <= out.len());
        PlanMem {
            plan: *self,
            n: in_.len() as c_int,
            in_: in_,
            out: Some(out),
            planner: r2c,
        }
    }
    #[cfg(r2r_is_hard)]
    pub fn r2r<I, O>(&self, in_: I, out: O) -> PlanMem<I, O>
        where I: DerefMut<[f64]>, O: DerefMut<[f64]>
    {
        assert!(in_.len() <= out.len());
        PlanMem {
            plan: *self,
            n: in_.len() as c_int,
            in_: in_,
            out: Some(out),
            planner: r2r,
        }
    }
}

unsafe fn c2c(n: c_int, in_: *mut c_void, out: *mut c_void,
              sign: c_int, flags: c_uint) -> ffi::fftw_plan {
    ffi::fftw_plan_dft_1d(n,
                          in_ as *mut ffi::fftw_complex, out as *mut ffi::fftw_complex,
                          sign, flags)
}
unsafe fn r2c(n: c_int, in_: *mut c_void, out: *mut c_void,
              _sign: c_int, flags: c_uint) -> ffi::fftw_plan {
    ffi::fftw_plan_dft_r2c_1d(n,
                              in_ as *mut f64, out as *mut ffi::fftw_complex,
                              flags)
}
unsafe fn c2r(n: c_int, in_: *mut c_void, out: *mut c_void,
              _sign: c_int, flags: c_uint) -> ffi::fftw_plan {
    ffi::fftw_plan_dft_c2r_1d(n,
                              in_ as *mut ffi::fftw_complex, out as *mut f64,
                              flags)
}
#[cfg(r2r_is_hard)]
unsafe fn r2r(n: c_int, in_: *mut c_void, out: *mut c_void,
              sign: c_int, flags: c_uint) -> ffi::fftw_plan {
    ffi::fftw_plan_r2r_1d(n,
                          in_ as *mut f64, out as *mut f64,
                          sign, flags)
}

pub struct PlanMem<I, O> {
    plan: Planner,
    n: c_int,
    in_: I,
    out: Option<O>,
    planner: unsafe fn(c_int, *mut c_void, *mut c_void, c_int, c_uint) -> ffi::fftw_plan,
}

impl<X, Y, I: DerefMut<[X]>, O: DerefMut<[Y]>> PlanMem<I, O> {
    pub fn plan(mut self) -> Result<Planned<I, O>, PlanMem<I, O>> {
        let in_ptr = self.in_.as_mut_ptr() as *mut c_void;
        let out_ptr = match self.out {
            None => in_ptr,
            Some(ref mut o) => o.as_mut_ptr() as *mut c_void,
        };

        let plan = RawPlan::new(|| unsafe {
            (self.planner)(self.n,
                           in_ptr,
                           out_ptr,
                           self.plan.dir(),
                           self.plan.flags())
        });

        match plan {
            None => Err(self),
            Some(p) => Ok(Planned { mem: self, plan: p })
        }
    }
}

pub struct Planned<I, O> {
    mem: PlanMem<I, O>,
    plan: RawPlan,
}

impl<X, Y, I: DerefMut<[X]>, O: DerefMut<[Y]>> Planned<I, O> {
    pub fn input(&mut self) -> &mut [X] {
        &mut *self.mem.in_
    }
    pub fn output(&mut self) -> Option<&mut [Y]> {
        self.mem.out.as_mut().map(|o| &mut **o)
    }

    pub fn execute(&mut self) {
        unsafe {
            self.plan.execute()
        }
    }
}
