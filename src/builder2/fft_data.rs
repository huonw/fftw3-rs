use num::Complex;

use libc::c_int;
use ffi;

use strided::{MutStrided, MutStride};

use plan::RawPlan;
use super::{FftData,  Meta, Secret, Inplace, Io, FftSpec, PlanResult, do_plan,
            Dim, PlanningError, Ready, R2R};

impl<T: FftData<T>, I: MutStrided<Elem = T>> FftSpec for Inplace<I> {
    type Input = T;
    type Output = T;
    #[doc(hidden)]
    unsafe fn plan(&mut self, meta: &Meta) -> PlanResult<RawPlan> {
        FftData::plan(self.in_out.as_stride_mut(), None::<MutStride<T>>, meta)
    }

    #[doc(hidden)]
    fn secret() -> Secret { Secret(()) }
}
impl<U, T: FftData<U>, I: MutStrided<Elem = T>, O: MutStrided<Elem = U>> FftSpec for Io<I, O> {
    type Input = T;
    type Output = U;
    #[doc(hidden)]
    unsafe fn plan(&mut self, meta: &Meta) -> PlanResult<RawPlan> {
        let in_ = self.in_.as_stride_mut();
        let out = self.out.as_stride_mut();

        FftData::plan(in_, Some(out), meta)
    }
    #[doc(hidden)]
    fn secret() -> Secret { Secret(()) }
}



impl FftData<Complex<f64>> for Complex<f64> {
    type State = Ready;

    #[doc(hidden)]
    unsafe fn plan(mut in_: MutStride<Complex<f64>>, mut out: Option<MutStride<Complex<f64>>>,
                   meta: &Meta)-> PlanResult<RawPlan> {
        let in_ptr = in_.as_mut_ptr() as *mut _;
        let in_len = in_.len();
        let in_stride = in_.stride();
        let (out_ptr, out_len, out_stride) = match out {
            Some(ref mut o) => (o.as_mut_ptr() as *mut _, o.len(), o.stride()),
            None => (in_ptr, in_len, in_stride),
        };

        let use_default_length = meta.dims.is_empty();
        let default = [Dim { n: in_len, in_stride: in_stride, out_stride: out_stride }];

        let required_len = if use_default_length {
            in_len
        } else {
            Dim::size(&*meta.dims).0
        };

        if in_len < required_len || out_len < required_len {
            // insufficient space
            return Err(PlanningError::BufferTooSmall(in_len, out_len, meta.dims.clone()))
        }

        let (rank,dims) = if use_default_length {
            (1, default.as_ptr() as *const _)
        } else {
            (meta.dims.len() as c_int, meta.dims.as_ptr() as *const _)
        };

        do_plan(|| {
            ffi::fftw_plan_guru64_dft(
                rank, dims,
                meta.howmany.len() as c_int, meta.howmany.as_ptr() as *const _,
                in_ptr, out_ptr,
                meta.direction.sign(), meta.rigor.flags())
        })
    }

    #[doc(hidden)]
    fn secret() -> Secret { Secret(()) }
}


impl FftData<Complex<f64>> for f64 {
    type State = Ready;

    #[doc(hidden)]
    unsafe fn plan(mut in_: MutStride<f64>, mut out: Option<MutStride<Complex<f64>>>,
                   meta: &Meta) -> PlanResult<RawPlan> {
        let in_ptr = in_.as_mut_ptr();
        let in_len = in_.len();

        let (out_ptr, out_len) = match out {
            Some(ref mut o) => (o.as_mut_ptr() as *mut _, o.len()),
            None => unimplemented!(/*"in-place r2c is hard"*/)
        };

        let use_default_length = meta.dims.is_empty();
        if use_default_length {
            return Err(PlanningError::NoLengthNoDefault)
        }

        let (r_size, c_size) = Dim::size(&*meta.dims);
        if in_len < r_size || out_len < c_size {
            return Err(PlanningError::BufferTooSmall(in_len, out_len, meta.dims.clone()));
        }

        do_plan(|| {
            ffi::fftw_plan_guru64_dft_r2c(
                meta.dims.len() as c_int, meta.dims.as_ptr() as *const _,
                meta.howmany.len() as c_int, meta.howmany.as_ptr() as *const _,
                in_ptr, out_ptr,
                meta.rigor.flags())
        })
    }

    #[doc(hidden)]
    fn secret() -> Secret { Secret(()) }
}
impl FftData<f64> for Complex<f64> {
    type State = Ready;

    #[doc(hidden)]
    unsafe fn plan(mut in_: MutStride<Complex<f64>>, mut out: Option<MutStride<f64>>,
                   meta: &Meta) -> PlanResult<RawPlan> {
        let in_ptr = in_.as_mut_ptr() as *mut _;
        let in_len = in_.len();

        let (out_ptr, out_len) = match out {
            Some(ref mut o) => (o.as_mut_ptr(), o.len()),
            None => unimplemented!(/*"in-place c2r is hard"*/)
        };

        let use_default_length = meta.dims.is_empty();
        if use_default_length {
            return Err(PlanningError::NoLengthNoDefault)
        }

        let (r_size, c_size) = Dim::size(&*meta.dims);
        if in_len < c_size || out_len < r_size {
            return Err(PlanningError::BufferTooSmall(in_len, out_len, meta.dims.clone()));
        }

        do_plan(|| {
            ffi::fftw_plan_guru64_dft_c2r(
                meta.dims.len() as c_int, meta.dims.as_ptr() as *const _,
                meta.howmany.len() as c_int, meta.howmany.as_ptr() as *const _,
                in_ptr, out_ptr,
                meta.rigor.flags())
        })
    }

    #[doc(hidden)]
    fn secret() -> Secret { Secret(()) }
}

impl FftData<f64> for f64 {
    type State = R2R;

    #[doc(hidden)]
    unsafe fn plan(mut in_: MutStride<f64>, mut out: Option<MutStride<f64>>,
                   meta: &Meta) -> PlanResult<RawPlan> {
        let in_ptr = in_.as_mut_ptr() as *mut _;
        let in_len = in_.len();
        let in_stride = in_.stride();
        let (out_ptr, out_len, out_stride) = match out {
            Some(ref mut o) => (o.as_mut_ptr() as *mut _, o.len(), o.stride()),
            None => (in_ptr, in_len, in_stride),
        };

        let use_default_length = meta.dims.is_empty();
        let default = [Dim { n: in_len, in_stride: in_stride, out_stride: out_stride }];

        let required_len = if use_default_length {
            in_len
        } else {
            Dim::size(&*meta.dims).0
        };
        if in_len < required_len || out_len < required_len {
            // insufficient space
            return Err(PlanningError::BufferTooSmall(in_len, out_len, meta.dims.clone()))
        }
        let (rank,dims) = if use_default_length {
            (1, default.as_ptr() as *const _)
        } else {
            (meta.dims.len() as c_int, meta.dims.as_ptr() as *const _)
        };

        assert!(meta.r2r_kinds.len() == rank as usize || meta.r2r_kinds.len() == 1);
        let kinds = meta.r2r_kinds.as_ptr();
        do_plan(|| {
            ffi::fftw_plan_guru64_r2r(
                rank, dims,
                meta.howmany.len() as c_int, meta.howmany.as_ptr() as *const _,
                in_ptr, out_ptr,
                kinds, meta.rigor.flags()
                )
        })
    }

    #[doc(hidden)]
    fn secret() -> Secret { Secret(()) }
}

/*
trait Split<T> {
    fn split(&mut self) -> (T, T);
}
impl<T> Split<*mut T> for Complex<T> {
    fn split(&mut self) -> (*mut T, *mut T) {
        let ptr = self as *mut _ as *mut T;
        unsafe {
            (ptr, ptr.offset(1))
        }
    }
}
impl<T, U: Split<T>> Split<T> for *mut U {
    fn split(&mut self) -> (T,T) {
        unsafe {
            (**self).split()
        }
    }
}


impl FftData<(*mut f64, *mut f64)> for (*mut f64, *mut f64) {
    unsafe fn plan(self, out: (*mut f64, *mut f64), meta: &Meta) {}
}

impl<U: Split<*mut f64>, W: Split<*mut f64>> FftData<U> for W {
    unsafe fn plan(mut self, mut out: U, meta: &Meta) {
        self.split().plan(out.split(), meta)
    }
}

// c2r split
impl FftData<*mut f64> for (*mut f64, *mut f64) {
    unsafe fn plan(self, out: *mut f64, meta: &Meta) {}
}
impl<U: Split<*mut f64>> FftData<*mut f64> for U {
    unsafe fn plan(mut self, out: *mut f64, meta: &Meta) {
        self.split().plan(out, meta)
    }
}

// r2c split
impl FftData<(*mut f64, *mut f64)> for *mut f64 {
    unsafe fn plan(self, out: (*mut f64, *mut f64), meta: &Meta) {}
}
impl<U: Split<*mut f64>> FftData<U> for *mut f64 {
    unsafe fn plan(self, mut out: U, meta: &Meta) {
        self.plan(out.split(), meta)
    }
}
*/
