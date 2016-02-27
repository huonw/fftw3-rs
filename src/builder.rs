//! This module is an interface to the `fftw` plan building machinery.
//!

use ffi;
use libc::{c_uint, c_int, c_void};
use num::complex::Complex64;
use std::ops::DerefMut;

use plan::RawPlan;

/// How much effort FFTW should put into computing the best strategy
/// to use.
///
/// The `FFTW_WISDOM_ONLY` rigor level is replaced by the
#[derive(Clone, Copy)]
#[allow(missing_docs)]
pub enum Rigor {
    Estimate,
    Measure,
    Patient,
    Exhaustive,
}
impl Rigor {
    fn flags(self) -> c_uint {
        match self {
            Rigor::Estimate => ffi::FFTW_ESTIMATE,
            Rigor::Measure => ffi::FFTW_MEASURE,
            Rigor::Patient => ffi::FFTW_PATIENT,
            Rigor::Exhaustive => ffi::FFTW_EXHAUSTIVE,
        }
    }
}

/// The direction of the transform to perform.
#[derive(Clone, Copy)]
#[allow(missing_docs)]
pub enum Direction {
    Forward, Backward
}

/// Control the basic properties of a set of transforms.
pub struct Planner {
    rigor: Rigor,
    wisdom_restriction: bool,
    direction: Direction,

    #[allow(dead_code)]
    dims: Vec<Dim>,
    #[allow(dead_code)]
    howmany: Vec<Dim>,
}

impl Planner {
    /// Construct a new planner with default values.
    ///
    /// This defaults to a forward transform with estimate rigor.
    pub fn new() -> Planner {
        Planner {
            rigor: Rigor::Estimate,
            wisdom_restriction: false,
            direction: Direction::Forward,
            dims: vec![],
            howmany: vec![],
        }
    }

    /// Set the rigor to use for this plan.
    pub fn rigor(mut self, r: Rigor) -> Planner {
        self.rigor = r;
        self
    }
    /// Set whether the planner should only be successfully created if
    /// there exists wisdom created with at least the rigor level set.
    pub fn wisdom_restriction(mut self, wisdom_only: bool) -> Planner {
        self.wisdom_restriction = wisdom_only;
        self
    }

    /// Set the direction of the transform to perform.
    pub fn direction(mut self, direction: Direction) -> Planner {
        self.direction = direction;
        self
    }

    #[cfg(higher_dim_is_hard)]
    pub fn dims_row_major(mut self, dims: Vec<uint>) -> Planner {
        assert!(dims.len() >= 1);

        self.dims = dims.iter().map(|n| Dim { n: *n, in_stride: 1, out_stride: 1 });
        for i in range(0, self.dims.len() - 1).rev() {
            let stride = self.dims[i + 1].n * self.dims[i + 1].in_stride;
            self.dims[i].in_stride = stride;
            self.dims[i].out_stride = stride;
        }

        self
    }
    #[cfg(higher_dim_is_hard)]
    pub fn multiples(mut self, number: uint) -> PlanMem<I, O> {
        unimplemented!();
        self.how_many = Contiguous(vec![number]);
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
            Direction::Forward => ffi::FFTW_FORWARD,
            Direction::Backward => ffi::FFTW_BACKWARD,
        }
    }

    /// Create a plan that performs its transformation in place using the settings
    /// from this planner
    pub fn inplace(self) -> InPlacePlanner {
        InPlacePlanner { plan: self }
    }

    /// Create plan memory for a complex to complex transformation
    pub fn c2c<I, O>(self, in_: I, out: O) -> PlanMem<I, O>
        where I: DerefMut<Target = [Complex64]>, O: DerefMut<Target = [Complex64]>
    {
        assert!(in_.len() <= 0x7F_FF_FF_FF);
        assert!(in_.len() <= out.len());
        let dims = vec![Dim { n: in_.len(), in_stride: 1, out_stride: 1 }];
        PlanMem {
            plan: self,
            in_: in_,
            out: Some(out),
            planner: c2c,

            dims: dims,
            how_many: vec![],
        }
    }

    /// Create plan memory for a complex to real transformation
    pub fn c2r<I, O>(self, in_: I, out: O) -> PlanMem<I, O>
        where I: DerefMut<Target = [Complex64]>, O: DerefMut<Target = [f64]>
    {
        assert!(in_.len() <= 0x7F_FF_FF_FF);
        assert!(in_.len() <= out.len() / 2 + 1);
        let dims = vec![Dim { n: 2 * (in_.len() - 1), in_stride: 1, out_stride: 1 }];
        PlanMem {
            plan: self,
            in_: in_,
            out: Some(out),
            planner: c2r,

            dims: dims,
            how_many: vec![],
        }
    }

    /// Create plan memory for a real to complex transformation
    pub fn r2c<I, O>(self, in_: I, out: O) -> PlanMem<I, O>
        where I: DerefMut<Target = [f64]>, O: DerefMut<Target = [Complex64]>
    {
        assert!(in_.len() <= 0x7F_FF_FF_FF);
        assert!(in_.len() / 2 + 1 <= out.len());
        let dims = vec![Dim { n: in_.len(), in_stride: 1, out_stride: 1 }];
        PlanMem {
            plan: self,
            in_: in_,
            out: Some(out),
            planner: r2c,

            dims: dims,
            how_many: vec![],
        }
    }
    #[cfg(r2r_is_hard)]
    pub fn r2r<I, O>(self, in_: I, out: O) -> PlanMem<I, O>
        where I: DerefMut<[f64]>, O: DerefMut<[f64]>
    {
        assert!(in_.len() <= 0x7F_FF_FF_FF);
        assert!(in_.len() <= out.len());
        let dims = vec![Dim { n: in_.len(), in_stride: 1, out_stride: 1 }];
        PlanMem {
            plan: *self,
            in_: in_,
            out: Some(out),
            planner: r2r,

            dims: dims,
            how_many: vec![],
        }
    }
}

/// Planner for plans that perform their operation in-place
pub struct InPlacePlanner {
    plan: Planner
}

impl InPlacePlanner {
    /// Create an in-place complex to complex plan
    pub fn c2c<I>(self, in_: I) -> PlanMem<I, I>
        where I: DerefMut<Target = [Complex64]>
    {
        assert!(in_.len() <= 0x7F_FF_FF_FF);
        let dims = vec![Dim { n: in_.len(), in_stride: 1, out_stride: 1 }];
        PlanMem {
            plan: self.plan,
            in_: in_,
            out: None,
            planner: c2c,

            dims: dims,
            how_many: vec![],
        }
    }
}

type GuruPlanner =
    unsafe fn(rank: c_int, dims: *const ffi::fftw_iodim64,
              howmany_rank: c_int, howmany_dims: *const ffi::fftw_iodim64,
              in_: *mut c_void, out: *mut c_void,
              sign: c_int, flags: c_uint) -> ffi::fftw_plan;

unsafe fn c2c(rank: c_int, dims: *const ffi::fftw_iodim64,
              howmany_rank: c_int, howmany_dims: *const ffi::fftw_iodim64,
              in_: *mut c_void, out: *mut c_void,
              sign: c_int, flags: c_uint) -> ffi::fftw_plan {
    ffi::fftw_plan_guru64_dft(rank, dims,
                            howmany_rank, howmany_dims,
                            in_ as *mut _, out as *mut _,
                            sign, flags)
}

unsafe fn r2c(rank: c_int, dims: *const ffi::fftw_iodim64,
              howmany_rank: c_int, howmany_dims: *const ffi::fftw_iodim64,
              in_: *mut c_void, out: *mut c_void,
              _sign: c_int, flags: c_uint) -> ffi::fftw_plan {
    ffi::fftw_plan_guru64_dft_r2c(rank, dims,
                                howmany_rank, howmany_dims,
                                in_ as *mut _, out as *mut _,
                                flags)
}
unsafe fn c2r(rank: c_int, dims: *const ffi::fftw_iodim64,
              howmany_rank: c_int, howmany_dims: *const ffi::fftw_iodim64,
              in_: *mut c_void, out: *mut c_void,
              _sign: c_int, flags: c_uint) -> ffi::fftw_plan {
    ffi::fftw_plan_guru64_dft_c2r(rank, dims,
                                howmany_rank, howmany_dims,
                                in_ as *mut _, out as *mut _,
                                flags)
}
#[cfg(r2r_is_hard)]
unsafe fn r2r(n: c_int, in_: *mut c_void, out: *mut c_void,
              sign: c_int, flags: c_uint) -> ffi::fftw_plan {
    ffi::fftw_plan_r2r_1d(n,
                          in_ as *mut f64, out as *mut f64,
                          sign, flags)
}


/// Represents a dimnsion along which to take a transformation.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Dim {
    /// The size of the dimension
    pub n: usize,
    /// The stride of the dimension in the input array
    pub in_stride: usize,
    /// The stride of the dimension in the output array
    pub out_stride: usize,
}

/// Represents the structure of a plan's input and output.
pub struct PlanMem<I, O> {
    /// A planner which can produce a plan which can 
    plan: Planner,
    dims: Vec<Dim>,
    #[allow(dead_code)]
    how_many: Vec<Dim>,
    in_: I,
    out: Option<O>,
    planner: GuruPlanner
}

impl<X, Y, I: DerefMut<Target = [X]>, O: DerefMut<Target = [Y]>> PlanMem<I, O> {
    /// Produce a plan from a completed PlanMem
    pub fn plan(mut self) -> Result<Planned<I, O>, PlanMem<I, O>> {
        let plan;
        {
            let in_ptr = self.in_.as_mut_ptr() as *mut c_void;
            let out_ptr = match self.out {
                None => in_ptr,
                Some(ref mut o) => o.as_mut_ptr() as *mut c_void,
            };
            assert!(self.dims.len() == 1);

            plan = RawPlan::new(|| unsafe {
                (self.planner)(
                    self.dims.len() as c_int,
                    self.dims.as_ptr() as *const ffi::fftw_iodim64,
                    0, [].as_ptr(),
                    in_ptr,
                    out_ptr,
                    self.plan.dir(),
                    self.plan.flags())
            });
        }
        match plan {
            None => Err(self),
            Some(p) => Ok(Planned { mem: self, plan: p })
        }
    }
}

/// Represents a successful plan
pub struct Planned<I, O> {
    mem: PlanMem<I, O>,
    plan: RawPlan,
}

impl<I: DerefMut, O: DerefMut> Planned<I, O> {
    /// Get a mutable reference to our input storage.
    pub fn input(&mut self) -> &mut I::Target {
        &mut *self.mem.in_
    }

    /// Get a mutable reference to our output storage.
    pub fn output(&mut self) -> Option<&mut O::Target> {
        self.mem.out.as_mut().map(|o| &mut **o)
    }

    /// Run a completed plan
    pub fn execute(&mut self) {
        unsafe {
            self.plan.execute()
        }
    }
}

#[cfg(test)]
mod tests {
    use std::mem;
    use libc::ptrdiff_t;
    use ffi::Struct_fftw_iodim64_do_not_use_me;
    use super::Dim;

    #[test]
    fn iodims_are_compatible() {
        // handle 32-bit and 64-bit platforms properly
        let n = 0x0102_0304_0506_0708u64 as usize;
        let is = 0x090A_0B0C_0D0E_0F00u64 as usize;
        let os = 0x1122_3344_5566_7788u64 as usize;

        let d = Dim { n: n, in_stride: is, out_stride: os };
        let f = Struct_fftw_iodim64_do_not_use_me {
            n: n as ptrdiff_t, is: is as ptrdiff_t, os: os as ptrdiff_t
        };
        type T = (usize, usize, usize);
        unsafe {
            assert_eq!(mem::transmute::<_, T>(d), mem::transmute::<_, T>(f));
        }
    }
}
