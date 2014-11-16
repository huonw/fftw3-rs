use num::Complex;
use std::num;


pub trait Zero {
    fn zero() -> Self;
}

macro_rules! impls {
    ($($zero: expr: $($ty: ty),*);*) => {
        $($(impl Zero for $ty {
            fn zero() -> $ty {
                $zero
            }
        })*)*
    }
}

impls! {
    0: int, i8, i16, i32, i64, uint, u8, u16, u32, u64;
    0.0: f32, f64
}

impl<T: Zero + num::Float> Zero for Complex<T> {
    fn zero() -> Complex<T> {
        Complex::new(Zero::zero(), Zero::zero())
    }
}
