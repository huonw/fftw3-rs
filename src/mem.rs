//! Memory backing.

use ffi;
use traits::Zero;
use libc;
use std::{mem, ptr, slice};
use std::ops::{Deref, DerefMut};

struct RawVec<T> {
    dat: *mut T,
    len: usize
}
impl<T> RawVec<T> {
    unsafe fn uninit(n: usize) -> RawVec<T> {
        let size = n.checked_mul(mem::size_of::<T>()).expect("FftwVec::uninit: size overflow");
        let dat = ffi::fftw_malloc(size as libc::size_t);

        if dat.is_null() {
            panic!("FftwVec::uninit: fftw_malloc failed");
        }

        RawVec { dat: dat as *mut T, len: n }
    }
}

impl<T> Deref for RawVec<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        unsafe {
            slice::from_raw_parts(self.dat as *const T, self.len)
        }
    }
}
impl<T> DerefMut for RawVec<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        unsafe {
            slice::from_raw_parts_mut(self.dat as *mut T, self.len)
        }
    }
}

impl<T> Drop for RawVec<T> {
    fn drop(&mut self) {
        unsafe {ffi::fftw_free(self.dat as *mut libc::c_void)}
    }
}

/// A non-resizable vector allocated using FFTWs allocator.
///
/// This implements `Deref<[T]>` and `DerefMut<[T]>` and so can be
/// used nearly-interchangeably with slices.
pub struct FftwVec<T> {
    dat: RawVec<T>
}

impl<T> FftwVec<T> {
    /// Allocate a `FftwVec` without initialising the elements at all.
    pub unsafe fn uninit(n: usize) -> FftwVec<T> {
        FftwVec { dat: RawVec::uninit(n) }
    }
}
impl<T> Deref for FftwVec<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        &*self.dat
    }
}
impl<T> DerefMut for FftwVec<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        &mut *self.dat
    }
}

impl<T> Drop for FftwVec<T> {
    fn drop(&mut self) {
        // free everything
        for p in self.iter() {
            unsafe {ptr::read(p);}
        }
    }
}

struct PartialVec<T> {
    dat: RawVec<T>,
    idx: usize
}

impl<T: Zero> FftwVec<T> {
    /// Allocate a `FftwVec` of length `n` containing zeros.
    pub fn zeros(n: usize) -> FftwVec<T> {
        let mut v = PartialVec {
            dat: unsafe {RawVec::uninit(n)},
            idx: 0,
        };

        while v.idx < n {
            unsafe {
                ptr::write(v.dat.dat.offset(v.idx as isize), Zero::zero());
            }
            v.idx += 1
        }

        unsafe {
            let ret = FftwVec {
                dat: ptr::read(&v.dat)
            };
            mem::forget(v);
            ret
        }
    }
}

impl<T> Drop for PartialVec<T> {
    fn drop(&mut self) {
        for p in self.dat[..self.idx].iter() {
            unsafe {ptr::read(p);}
        }
    }
}

#[cfg(test)]
mod tests {
    use mem::FftwVec;

    #[test]
    fn fftw_vec() {
        let mut v = FftwVec::<usize>::zeros(100);
        for (i, x) in v.iter_mut().enumerate() {
            *x = i;
        }
        let mut i = 0;
        for x in v.iter() {
            assert_eq!(*x, i);
            i += 1;
        }
        assert_eq!(i, 100);
    }
}
