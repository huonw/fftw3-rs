#![allow(warnings)]
use ffi;
use libc::{c_uint, c_int, c_void};
use num::complex::{Complex, Complex64};
use std::marker::PhantomData;
use strided::{MutStrided, Strided, MutStride};

use plan::RawPlan;

/// Values for which `[Self] -> [Target]` works as a transform.
pub trait FftData<Target> : Sized {
    type State;
    #[doc(hidden)]
    unsafe fn plan(in_: MutStride<Self>, out: Option<MutStride<Target>>,
                   meta: &Meta) -> PlanResult<RawPlan>;

    #[doc(hidden)]
    fn secret() -> Secret;
}

mod fft_data;


/// How much effort FFTW should put into computing the best strategy
/// to use.
///
/// The `FFTW_WISDOM_ONLY` rigor level is replaced by the
#[derive(Clone, Copy)]
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

/// The direction of the transform to perform..
#[derive(Clone, Copy)]
pub enum Direction {
    Forward, Backward
}
impl Direction {
    fn sign(self) -> c_int {
        match self {
            Direction::Forward => ffi::FFTW_FORWARD,
            Direction::Backward => ffi::FFTW_BACKWARD,
        }
    }
}

pub struct Begin(());
pub struct Input<I> {
    in_: I
}
pub struct Io<I, O> {
    in_: I,
    out: O
}
pub struct Inplace<I> {
    in_out: I
}

pub struct R2R(());
pub struct Ready(());

#[repr(C)]
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Dim {
    pub n: usize,
    pub in_stride: usize,
    pub out_stride: usize,
}

impl Dim {
    // TODO: should be num_elems?
    fn size(ds: &[Dim]) -> (usize, usize) {
        assert!(ds.len() > 0);
        let head = ds[.. ds.len() - 1].iter().fold(1, |m, d| m * d.n);
        let last = ds.last().expect("no dims in Dim::num_elems");
        (head * last.n, head * (last.n / 2 + 1))
    }

    // TODO: the checks in fft_size.rs need to use this
    fn array_size(ds: &[Dim], c2r: bool, r2c: bool) -> (usize, usize) {
        debug_assert!(!(c2r && r2c));
        assert!(ds.len() > 0);

        let (in_, out) = ds[.. ds.len() - 1].iter().fold((0, 0), |(a,b), d| {
            (a + d.n * d.in_stride, b + d.n * d.out_stride)
        });
        let last = ds.last().unwrap();
        if c2r {
            (in_ + (last.n / 2 + 1) * last.in_stride,
             out + last.n * last.out_stride)
        } else if r2c {
            (in_ + last.n * last.out_stride,
             out + (last.n / 2 + 1) * last.out_stride)
        } else {
            (in_, out)
        }
    }
}

#[derive(Clone, Copy)]
pub enum R2rKind {
    R2ch,
    Hc2r,
    Dht,
    Dct00,
    Dct01,
    Dct10,
    Dct11,
    Dst00,
    Dst01,
    Dst10,
    Dst11,
}
impl R2rKind {
    fn as_fftw(self) -> c_uint {
        match self {
            R2rKind::R2ch => ffi::FFTW_R2HC,
            R2rKind::Hc2r => ffi::FFTW_HC2R,
            R2rKind::Dht => ffi::FFTW_DHT,
            R2rKind::Dct00 => ffi::FFTW_REDFT00,
            R2rKind::Dct01 => ffi::FFTW_REDFT01,
            R2rKind::Dct10 => ffi::FFTW_REDFT10,
            R2rKind::Dct11 => ffi::FFTW_REDFT11,
            R2rKind::Dst00 => ffi::FFTW_RODFT00,
            R2rKind::Dst01 => ffi::FFTW_RODFT01,
            R2rKind::Dst10 => ffi::FFTW_RODFT10,
            R2rKind::Dst11 => ffi::FFTW_RODFT11,
        }
    }
}

#[doc(hidden)]
pub struct Meta {
    rigor: Rigor,
    wisdom_restriction: bool,
    direction: Direction,

    in_stride: usize,
    out_stride: usize,

    r2r_kinds: Vec<c_uint>,
    overall_dims: Vec<Dim>,
    dims: Vec<Dim>,
    howmany: Vec<Dim>,
}

#[derive(Debug)]
pub enum PlanningError {
    FftwError,
    NoLengthNoDefault,
    BufferTooSmall(usize, usize, Vec<Dim>),
}

pub type PlanResult<T> = Result<T, PlanningError>;

fn do_plan<F: FnOnce() -> ffi::fftw_plan>(f: F) -> PlanResult<RawPlan> {
    match RawPlan::new(f) {
        Some(p) => Ok(p),
        None => Err(PlanningError::FftwError),
    }
}

/// This is designed to stop the must-be-public traits from being able
/// to be implemented externally, because that would be rather
/// strange.
#[doc(hidden)]
pub struct Secret(());

pub struct Planner<Stage, State> {
    meta: Meta,
    data: Stage,
    _marker: PhantomData<State>
}


impl Planner<Begin, Begin> {
    pub fn new() -> Planner<Begin, Begin> {
        Planner {
            meta: Meta {
                rigor: Rigor::Estimate,
                wisdom_restriction: false,
                direction: Direction::Forward,

                r2r_kinds: vec![],

                overall_dims: vec![],
                dims: vec![],
                howmany: vec![],

                in_stride: 1,
                out_stride: 1,
            },

            data: Begin(()),

            _marker: PhantomData
        }
    }
}

impl<Y> Planner<Begin, Y> {
    pub fn input<I: MutStrided>(mut self, in_: I) -> Planner<Input<I>, Begin> {
        self.meta.in_stride = in_.stride();

        Planner {
            meta: self.meta,
            data: Input { in_: in_ },
            _marker: PhantomData
        }
    }
}

impl<X, Y> Planner<X, Y> {
    /// Set the rigor to use for this plan.
    pub fn rigor(mut self, r: Rigor) -> Planner<X, Y> {
        self.meta.rigor = r;
        self
    }
    /// Set whether the planner should only be successfully created if
    /// there exists wisdom created with at least the rigor level set.
    pub fn wisdom_restriction(mut self, wisdom_only: bool) -> Planner<X, Y> {
        self.meta.wisdom_restriction = wisdom_only;
        self
    }

    /// Set the direction of the transform to perform.
    pub fn direction(mut self, direction: Direction) -> Planner<X, Y> {
        self.meta.direction = direction;
        self
    }
}

impl<I: MutStrided, Y> Planner<Input<I>, Y> {
    pub fn output<O: MutStrided>(mut self, out: O)
                                       -> Planner<Io<I, O>,
                                                  <I::Elem as FftData<O::Elem>>::State>
        where O: MutStrided,
              I::Elem: FftData<O::Elem>
    {

        self.meta.out_stride = out.stride();

        Planner {
            meta: self.meta,
            data: Io { in_: self.data.in_, out: out },
            _marker: PhantomData
        }
    }
}

impl<I: MutStrided, Y> Planner<Input<I>, Y>
    where <I as Strided>::Elem: FftData<I::Elem>
{
    pub fn inplace(mut self) -> Planner<Inplace<I>, <I::Elem as FftData<I::Elem>>::State>  {
        self.meta.out_stride = self.meta.in_stride;

        Planner {
            meta: self.meta,
            data: Inplace { in_out: self.data.in_ },
            _marker: PhantomData
        }
    }
}
impl<X, Y> Planner<X, Y>
    where X: HasInput + HasOutput,
        <X as HasInput>::I: MutStrided, <X as HasOutput>::O: MutStrided,
        <<X as HasInput>::I as Strided>::Elem: FftData<<<X as HasOutput>::O as Strided>::Elem>
{

    pub fn _1d(mut self, n: usize) -> Planner<X, Y> {
        self.nd(&[n])
    }

    pub fn _2d(mut self, n0: usize, n1: usize) -> Planner<X, Y> {
        self.nd(&[n0, n1])
    }

    pub fn _3d(mut self, n0: usize, n1: usize, n2: usize) -> Planner<X, Y> {
        self.nd(&[n0, n1, n2])
    }

    pub fn nd(mut self, dims: &[usize]) -> Planner<X, Y> {
        assert!(dims.len() > 0, "Planner.nd: empty dimensions");
        self.meta.dims.clear();
        self.meta.dims.extend(dims.iter().map(|n| Dim { n: *n, in_stride: 0, out_stride: 0 }));

        let mut stride = 1;
        for place in self.meta.dims.iter_mut().rev() {
            place.in_stride = stride;
            place.out_stride = stride;
            stride *= place.n;
        }
        self
    }

    pub fn nd_subarray(mut self, dims: &[(usize, usize, usize)]) -> Planner<X, Y> {
        assert!(dims.len() > 0,
                "Planner.nd_subarray: empty dimensions");

        self.meta.dims.clear();
        self.meta.dims.extend(dims.iter().map(|n| Dim { n: n.0 , in_stride: 0, out_stride: 0 }));

        let mut in_stride = 1;
        let mut out_stride = 1;
        for (place, &(_, in_, out)) in self.meta.dims.iter_mut().zip(dims.iter()).rev() {
            place.in_stride = in_stride;
            place.out_stride = out_stride;

            in_stride *= in_;
            out_stride *= out;
        }
        debug!("dimensions & strides: {:?}", self.meta.dims);
        self
    }
}

pub trait HasInput {
    type I;
    fn input(&mut self) -> &mut Self::I;
}
impl<I> HasInput for Input<I> {
    type I = I;
    fn input(&mut self) -> &mut I { &mut self.in_ }
}
impl<I> HasInput for Inplace<I> {
    type I = I;
    fn input(&mut self) -> &mut I { &mut self.in_out }
}
impl<I, O> HasInput for Io<I, O> {
    type I = I;
    fn input(&mut self) -> &mut I { &mut self.in_ }
}

pub trait HasOutput {
    type O;
    fn output(&mut self) -> &mut Self::O;
}
impl<I> HasOutput for Inplace<I> {
    type O = I;
    fn output(&mut self) -> &mut I { &mut self.in_out }
}
impl<I, O> HasOutput for Io<I,O> {
    type O = O;
    fn output(&mut self) -> &mut O { &mut self.out }
}

// add associated item here for the target result, to make r2r kind always required.
pub trait FftSpec {
    type Input;
    type Output;
    #[doc(hidden)]
    unsafe fn plan(&mut self, meta: &Meta) -> PlanResult<RawPlan>;

    #[doc(hidden)]
    fn secret() -> Secret;
}

impl<X: FftSpec<Input=f64, Output=f64>> Planner<X, R2R> {
    #[cfg(ices)]
    pub fn r2r_kind(mut self, kind: R2rKind) -> Planner<X, Ready> {
        self.r2r_kinds(&[kind])
    }
    pub fn r2r_kinds(mut self, kinds: &[R2rKind]) -> Planner<X, Ready> {
        assert!(kinds.len() > 0, "Planner.r2r_kinds: no kinds");

        self.meta.r2r_kinds.clear();
        self.meta.r2r_kinds.extend(kinds.iter().map(|k| k.as_fftw()));

        Planner {
            meta: self.meta,
            data: self.data,
            _marker: PhantomData
        }
    }
}

impl<X: FftSpec> Planner<X, Ready> {
    pub fn plan(mut self) -> Result<Plan<X>, PlanningError> {
        // space things out appropriately for the backing array.
        for d in self.meta.dims.iter_mut() {
            d.in_stride *= self.meta.in_stride;
            d.out_stride *= self.meta.out_stride;
        }

        match unsafe {self.data.plan(&self.meta)} {
            Ok(p) => Ok(Plan { planner: self, plan: p }),
            Err(e) => Err(e)
        }
    }
}

pub struct Plan<X> {
    planner: Planner<X, Ready>,
    plan: RawPlan,
}

impl<I: MutStrided> Plan<Inplace<I>>
    where <I as Strided>::Elem: FftData<I::Elem>
{
    pub fn in_out(&mut self) -> &mut I {
        &mut self.planner.data.in_out
    }
}
impl<I, O> Plan<Io<I, O>> {
    pub fn input(&mut self) -> &mut I {
        &mut self.planner.data.in_
    }
    pub fn output(&mut self) -> &mut O {
        &mut self.planner.data.out
    }
}

impl<X> Plan<X> {
    pub fn execute(&mut self) {
        unsafe {
            self.plan.execute()
        }
    }

    pub fn debug_print(&self) {
        self.plan.debug_print()
    }
}
