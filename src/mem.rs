use ffi;
use libc;
use std::{mem, num, ptr, raw};

/// Types that can be used as to store data passed to and from FFTW.
pub trait BackingStorage<T> {
    /// Get a pointer to that data.
    fn storage_slice<'a>(&'a mut self) -> &'a mut [T];
}

impl<'x, T> BackingStorage<T> for &'x mut [T] {
    fn storage_slice<'a>(&'a mut self) -> &'a mut [T] { self.as_mut_slice() }
}
impl<T> BackingStorage<T> for Vec<T> {
    fn storage_slice<'a>(&'a mut self) -> &'a mut [T] { self.as_mut_slice() }
}

/// A non-resizable vector allocated using FFTWs allocator.
pub struct FFTWVec<T> {
    dat: *mut T,
    len: uint
}

impl<T> FFTWVec<T> {
    /// Allocate a `FFTWVec` without initialising the elements at all.
    pub unsafe fn uninit(n: uint) -> FFTWVec<T> {
        let size = n.checked_mul(&mem::size_of::<T>()).expect("FFTWVec::uninit: size overflow");
        let dat = ffi::fftw_malloc(size as libc::size_t);

        if dat.is_null() {
            panic!("FFTWVec::uninit: fftw_malloc failed");
        }

        FFTWVec { dat: dat as *mut T, len: n }
    }

    /// View the contents as an immutable slice.
    pub fn as_slice<'a>(&'a self) -> &'a [T] {
        unsafe {mem::transmute(raw::Slice { data: self.dat as *const T, len: self.len })}
    }
    /// View the contents as a mutable slice.
    pub fn as_mut_slice<'a>(&'a mut self) -> &'a mut [T] {
        unsafe {mem::transmute(raw::Slice { data: self.dat as *const T, len: self.len })}
    }
}

impl<T: num::Zero> FFTWVec<T> {
    /// Allocate a `FFTWVec` of length `n` containing zeros.
    pub fn zeros(n: uint) -> FFTWVec<T> {
        let mut v = unsafe {FFTWVec::uninit(n)};
        for elem in v.as_mut_slice().iter_mut() {
            *elem = num::zero();
        }

        v
    }
}
#[unsafe_destructor]
impl<T> Drop for FFTWVec<T> {
    fn drop(&mut self) {
        // free everything
        for p in self.as_slice().iter() {
            unsafe {ptr::read(p);}
        }
        unsafe {ffi::fftw_free(self.dat as *mut libc::c_void)}
    }
}

impl<T> BackingStorage<T> for FFTWVec<T> {
    fn storage_slice<'a>(&'a mut self) -> &'a mut [T] {
        self.as_mut_slice()
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
