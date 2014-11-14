//! Some functions in FFTW are not thread-safe, and one should ensure
//! that only one thread is executing these at a time. This module
//! provides a lock for this purpose.

use sync::mutex::{StaticMutex, MUTEX_INIT};

/// Hold this lock when doing anything thread-unsafe with FFTW.
pub static LOCK: StaticMutex = MUTEX_INIT;

/// Run the given closure inside the critical section of the FFTW
/// lock.
pub fn run<A>(f: || -> A) -> A {
    let _g = LOCK.lock();
    f()
}
