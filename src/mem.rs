//! Memory backing.

use ffi;
use libc;
use std::{mem, num, ops, ptr, raw};

struct RawVec<T> {
    dat: *mut T,
    len: uint
}
impl<T> RawVec<T> {
    unsafe fn uninit(n: uint) -> RawVec<T> {
        let size = n.checked_mul(&mem::size_of::<T>()).expect("FFTWVec::uninit: size overflow");
        let dat = ffi::fftw_malloc(size as libc::size_t);

        if dat.is_null() {
            panic!("FFTWVec::uninit: fftw_malloc failed");
        }

        RawVec { dat: dat as *mut T, len: n }
    }
}

impl<T> Deref<[T]> for RawVec<T> {
    fn deref(&self) -> &[T] {
        unsafe {mem::transmute(raw::Slice { data: self.dat as *const T, len: self.len })}
    }
}
impl<T> DerefMut<[T]> for RawVec<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        unsafe {mem::transmute(raw::Slice { data: self.dat as *const T, len: self.len })}
    }
}

#[unsafe_destructor]
impl<T> Drop for RawVec<T> {
    fn drop(&mut self) {
        unsafe {ffi::fftw_free(self.dat as *mut libc::c_void)}
    }
}

/// A non-resizable vector allocated using FFTWs allocator.
///
/// This implements `Deref<[T]>` and `DerefMut<[T]>` and so can be
/// used nearly-interchangeably with slices.
pub struct FFTWVec<T> {
    dat: RawVec<T>
}

impl<T> FFTWVec<T> {
    /// Allocate a `FFTWVec` without initialising the elements at all.
    pub unsafe fn uninit(n: uint) -> FFTWVec<T> {
        FFTWVec { dat: RawVec::uninit(n) }
    }
}
impl<T> Deref<[T]> for FFTWVec<T> {
    fn deref(&self) -> &[T] {
        &*self.dat
    }
}
impl<T> DerefMut<[T]> for FFTWVec<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        &mut *self.dat
    }
}

impl<T> ops::Slice<uint, [T]> for FFTWVec<T> {
    #[inline]
    fn as_slice_<'a>(&'a self) -> &'a [T] {
        &**self
    }

    #[inline]
    fn slice_from_or_fail<'a>(&'a self, start: &uint) -> &'a [T] {
        (**self).slice_from_or_fail(start)
    }

    #[inline]
    fn slice_to_or_fail<'a>(&'a self, end: &uint) -> &'a [T] {
        (**self).slice_to_or_fail(end)
    }
    #[inline]
    fn slice_or_fail<'a>(&'a self, start: &uint, end: &uint) -> &'a [T] {
        (**self).slice_or_fail(start, end)
    }
}
impl<T> ops::SliceMut<uint, [T]> for FFTWVec<T> {
    #[inline]
    fn as_mut_slice_<'a>(&'a mut self) -> &'a mut [T] {
        &mut **self
    }

    #[inline]
    fn slice_from_or_fail_mut<'a>(&'a mut self, start: &uint) -> &'a mut [T] {
        (**self).slice_from_or_fail_mut(start)
    }

    #[inline]
    fn slice_to_or_fail_mut<'a>(&'a mut self, end: &uint) -> &'a mut [T] {
        (**self).slice_to_or_fail_mut(end)
    }
    #[inline]
    fn slice_or_fail_mut<'a>(&'a mut self, start: &uint, end: &uint) -> &'a mut [T] {
        (**self).slice_or_fail_mut(start, end)
    }
}

#[unsafe_destructor]
impl<T> Drop for FFTWVec<T> {
    fn drop(&mut self) {
        // free everything
        for p in self.iter() {
            unsafe {ptr::read(p);}
        }
    }
}

struct PartialVec<T> {
    dat: RawVec<T>,
    idx: uint
}

impl<T: num::Zero> FFTWVec<T> {
    /// Allocate a `FFTWVec` of length `n` containing zeros.
    pub fn zeros(n: uint) -> FFTWVec<T> {
        let mut v = PartialVec {
            dat: unsafe {RawVec::uninit(n)},
            idx: 0,
        };

        while v.idx < n {
            unsafe {
                ptr::write(v.dat.dat.offset(v.idx as int), num::zero());
            }
            v.idx += 1
        }

        unsafe {
            let ret = FFTWVec {
                dat: ptr::read(&v.dat)
            };
            mem::forget(v);
            ret
        }
    }
}

#[unsafe_destructor]
impl<T> Drop for PartialVec<T> {
    fn drop(&mut self) {
        for p in self.dat.as_slice().slice_to(self.idx).iter() {
            unsafe {ptr::read(p);}
        }
    }
}

#[cfg(test)]
mod tests {
    use mem::FFTWVec;

    #[test]
    fn fftw_vec() {
        let mut v = FFTWVec::<uint>::zeros(100);
        for (i, x) in v.as_mut_slice().iter_mut().enumerate() {
            *x = i;
        }
        let mut i = 0;
        for x in v.as_slice().iter() {
            assert_eq!(*x, i);
            i += 1;
        }
        assert_eq!(i, 100);
    }
}
