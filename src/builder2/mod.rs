#![allow(warnings)]
use ffi;
use libc::{c_uint, c_int, c_void};
use num::complex::{Complex, Complex64};
use std::marker::PhantomData;
use strided::{MutStrided, Strided, MutStride};

use plan::RawPlan;

/// Values for which `[Self] -> [Target]` works as a transform.
pub trait FftData<Target> {
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
impl Direction {
    fn sign(self) -> c_int {
        match self {
            Direction::Forward => ffi::FFTW_FORWARD,
            Direction::Backward => ffi::FFTW_BACKWARD,
        }
    }
}

/// Used a placeholder before the plan has been filled out.
///
/// See [`Planner::new`](struct.Planner.html#method.new).
pub struct Begin(());

///
/// See [`Planner::input`](struct.Planner.html#method.input).
pub struct Input<I> {
    in_: I
}

/// A stage where both input and output have been configured.
///
/// See [`Planner::output`](struct.Planner.html#method.output).
pub struct Io<I, O> {
    in_: I,
    out: O
}

/// A stage where both input and output have been configured.
///
/// See [`Planner::inplace`](struct.Planner.html#method.inplace).
pub struct Inplace<I> {
    in_out: I
}

/// Represents that the planenr is only interested in real to real transforms
pub struct R2R(());
/// Represents that we have chosen a specific type of transform and are ready to build.
pub struct Ready(());

/// Represents a dimnsion along which to take a transformation.
#[repr(C)]
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Dim {
    /// The size of the dimension
    pub n: usize,
    /// The stride of the dimension in the input array
    pub in_stride: usize,
    /// The stride of the dimension in the output array
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
/// Represents what type of real to real transformation we want to be doing.
pub enum R2rKind {
    /// A forward FFT which computes a half complex transform.
    ///
    /// Since the FFT of a real series is always symetrical, this mode can use
    /// an output array of the same size as the input. It stores the real
    /// component of the output in the first half of the output array and the
    /// immaginary component in the second half. Imaginary components are
    /// stored in reverse order, with `i[0]` in the last position of the output
    /// array.
    R2hc,
    /// The inverse of `R2hc` above. Expects the same encoding scheme.
    Hc2r,
    /// Computes the discrete Hartley transform. Is its own inverse
    Dht,
    /// Computes the REDDFT00 transform, also known as the DCT-I transform. Is its own inverse.
    Dct00,
    /// Computes the REDDFT01 transform, also known as the DCT-III transform. Inverse is the `Dct10`.
    Dct01,
    /// Computes the REDDFT10 transform, also known as the DCT-II transform. Inverse is the `Dct01`.
    Dct10,
    /// Computes the REDDFT11 transform, also known as the DCT-IV transform. Inverse is the `Dct11`.
    Dct11,
    /// Computes the RODDFT00 transform, also known as the DST-I transform. Is its own inverse.
    Dst00,
    /// Computes the RODDFT01 transform, also known as the DST-III transform. Inverse is the `Dst10`.
    Dst01,
    /// Computes the RODDFT10 transform, also known as the DST-II transform. Inverse is the `Dst01`.
    Dst10,
    /// Computes the RODDFT11 transform, also known as the DST-IV transform. Inverse is the `Dst11`.
    Dst11,
}
impl R2rKind {
    /// Internal utility to convert our enums to native FFTW enums.
    fn as_fftw(self) -> c_uint {
        match self {
            R2rKind::R2hc => ffi::FFTW_R2HC,
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

/// All the ways planning can fail.
#[derive(Debug)]
pub enum PlanningError {
    /// An error occured inside of the fftw library.
    FftwError,
    /// Currently unused.
    NoLengthNoDefault,
    /// Raised when the output buffer provided is too small for the provided input.
    BufferTooSmall(usize, usize, Vec<Dim>),
}

/// A `Result` alias representing result of trying to create a plan.
pub type PlanResult<T> = Result<T, PlanningError>;

/// Try to create a raw plan
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

/// High level interface to building FFTW3 plans.
///
/// The first parameter is refered to as the "Stage" of the planner.
/// Planners progress through the following stages:
///
///   * [`Begin`](struct.Begin.html): The initial stage, indicating that nothing has been set up
///   * [`Input<I>`](struct.Input.html): Set by calling [`Planner::input`](struct.Planner.html#method.input).
///   Indicates that the planner knows what sort of input (`f64`, `Complex64`,
///   etc.) the final plan will be consuming.
///   * [`Inplace<I>`](struct.Inplace.html) or [`Io<I, O>`](struct.Io.html): Indicates that we know what sort of output
///   the planner will be producing.  If the eventual plan will be in place, we transition to the `Inplace<I>` stage
///   using [`Planner::inplace`](struct.Planner.html#method.inplace). Otherwise, we configure the output buffer by
///   calling [`Planner::output`](struct.Planner.html#method.output).
pub struct Planner<Stage, State> {
    #[doc(hidden)]
    meta: Meta,
    #[doc(hidden)]
    data: Stage,
    #[doc(hidden)]
    _marker: PhantomData<State>
}


impl Planner<Begin, Begin> {
    /// Create a planner where we are in the beginning stage and state.
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
    /// Set the input format for the planner.
    ///
    /// Transitions from the [`Begin`](struct.Begin.html) stage to the [`Input`](struct.Input.html) stage.
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
    /// Set the output configuration of the planner.
    ///
    /// Transitions from the [`Input<I>`](struct.Input.html) stage to the [`Io`](struct.Io.html) stage.
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
    /// Configure the planner to operate in place
    ///
    /// Transitions from the [`Input<I>`](struct.Input.html) stage to the [`Inplace<I>`](struct.Inplace.html) stage.
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
    /// Declare that we are interested in a 1d FFT
    ///
    /// `n` is the size of the dimension.
    ///
    /// Requires that the planner be in one of the terminal stages ([`Io<I, O>`](struct.Io.html) or
    /// [`Inplace<I>`](struct.Inplace.html)).
    pub fn _1d(mut self, n: usize) -> Planner<X, Y> {
        self.nd(&[n])
    }

    /// Declare that we are interested in a 2d FFT
    ///
    /// `n0` and `n1` are the size of the first and second dimensions respectively.
    ///
    /// Requires that the planner be in one of the terminal stages ([`Io<I, O>`](struct.Io.html) or
    /// [`Inplace<I>`](struct.Inplace.html)).
    pub fn _2d(mut self, n0: usize, n1: usize) -> Planner<X, Y> {
        self.nd(&[n0, n1])
    }

    /// Declare that we are interested in a 3d FFT
    ///
    /// `n0`, `n1` and `n2` are the size of the first, second, and third dimensions respectively.
    ///
    /// Requires that the planner be in one of the terminal stages ([`Io<I, O>`](struct.Io.html) or
    /// [`Inplace<I>`](struct.Inplace.html)).
    pub fn _3d(mut self, n0: usize, n1: usize, n2: usize) -> Planner<X, Y> {
        self.nd(&[n0, n1, n2])
    }

    /// Declare that we are interested in a n-dimensional FFT
    ///
    /// `dims` specifies the size of each of the dimensions.
    ///
    /// Requires that the planner be in one of the terminal stages ([`Io<I, O>`](struct.Io.html) or
    /// [`Inplace<I>`](struct.Inplace.html)).
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

    /// Declare that we are performing an FFT with dimensions outlined by `dims`
    ///
    /// `dims` is a sliice of `usize` tuples. The first element is the size of the dimension.
    /// The second element is the input stride, and the third element is the output stride.
    /// Strides of 0 mean tight packing.
    ///
    /// Requires that the planner be in one of the terminal stages ([`Io<I, O>`](struct.Io.html) or
    /// [`Inplace<I>`](struct.Inplace.html)).
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

/// Used to indicate that a stage has had the input array configured.
///
/// See [`Planner::new`](struct.Planner.Html#new) for more detailed information about planner stages.
pub trait HasInput {
    /// The type of the input storage
    type I;
    /// Get a mutable reference to the input storage.
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

/// Used to indicate that a stage has had the outpu array configured.
///
/// See [`Planner::new`](struct.Planner.Html#new) for more detailed information about planner stages.
pub trait HasOutput {
    /// The type of the output storage
    type O;
    /// Get a mutable reference to the output storage
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
/// Agregate specification information for a plan. Used to produce raw plans.
pub trait FftSpec {
    /// The input buffer type
    type Input;
    /// The output buffer type
    type Output;
    #[doc(hidden)]
    unsafe fn plan(&mut self, meta: &Meta) -> PlanResult<RawPlan>;

    #[doc(hidden)]
    fn secret() -> Secret;
}

impl<X: FftSpec<Input=f64, Output=f64>> Planner<X, R2R> {
    /// Produce a 1d real to real plan
    #[cfg(ices)]
    pub fn r2r_kind(mut self, kind: R2rKind) -> Planner<X, Ready> {
        self.r2r_kinds(&[kind])
    }
    /// Produce a n-dimensional real to real plan
    ///
    /// # Panics
    /// We assert that `kinds.len()` > 0
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
    /// Produce a plan
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

/// A completed plan.
///
/// See the [`Planner`](struct.Planner.html) for information about creating plans.
pub struct Plan<X> {
    planner: Planner<X, Ready>,
    plan: RawPlan,
}

impl<I: MutStrided> Plan<Inplace<I>>
    where <I as Strided>::Elem: FftData<I::Elem>
{
    /// A more ergonomic way of getting the combined input/output buffer for inplace plans.
    ///
    /// See [`Plan::input`](struct.Plan.html#method.input) and [`Plan::ouput`](struct.Plan.html#method.output) for
    /// getting the input and output buffers of other types of plans.
    pub fn in_out(&mut self) -> &mut I {
        &mut self.planner.data.in_out
    }
}
impl<I, O> Plan<Io<I, O>> {
    /// Get a mutable reference to the plan's inpu buffer.
    ///
    /// This function is usually used to fill the input buffer with a new set
    /// of data before [executing](struct.Plan.html#method.execute).
    pub fn input(&mut self) -> &mut I {
        &mut self.planner.data.in_
    }

    /// Get a mutable reference to the plan's output buffer.
    pub fn output(&mut self) -> &mut O {
        &mut self.planner.data.out
    }
}

impl<X> Plan<X> {
    /// Execute a plan.
    ///
    /// Compute the FFT of the plan's input buffer.
    /// Returns a mutable reference to the output buffer.
    pub fn execute(&mut self) {
        unsafe {
            self.plan.execute()
        }
    }

    /// Debug printing of the plan
    pub fn debug_print(&self) {
        self.plan.debug_print()
    }
}
