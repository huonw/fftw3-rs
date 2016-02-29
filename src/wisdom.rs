use std::path::{Path, PathBuf};
use {ffi, lock};
use std::ffi::CString;

/// Import and export FFTW wisdom implicitly.
///
/// The destructor will save wisdom to the file from which it was
/// loaded.
///
/// The `wisdom` macro performs this automatically.
///
/// # Example
///
/// ```rust,ignore
/// {
///    wisdom!(Path::new("./wise.fftw"));
///    // ... perform FFTs with the assistance of that wisdom
///
/// } // any new wisdom is automatically saved.
/// ```
pub struct WisdomGuard {
    p: PathBuf
}
impl WisdomGuard {
    /// Load wisdom from `p`, and save it automatically on clean-up.
    ///
    /// Failure to load is ignored, so one can supply a wisdom file
    /// that does not exist (yet) and it will be created on the next run.
    pub fn import(p: &Path) -> WisdomGuard {
        import_from_file(p);
        WisdomGuard { p: p.to_path_buf() }
    }
}
impl Drop for WisdomGuard {
    fn drop(&mut self) {
        export_to_file(self.p.as_path());
    }
}

#[macro_export]
macro_rules! wisdom {
    ($p: expr) => { let _guard = fftw3::wisdom::WisdomGuard::import($p); }
}

unsafe fn native_file_name(p: &Path) -> *const i8 {
    CString::new(p.as_os_str().to_str()
        .expect("fftw3: Path expected to be valid unicode"))
        .unwrap() // should never panic since we just made the string
        .as_ptr()
}

/// Attempt to load the system's wisdom.
pub fn import_from_system() -> bool {
    unsafe {
        lock::run(|| ffi::fftw_import_system_wisdom() != 0)
    }
}

/// Attempt to save wisdom to `p`.
pub fn export_to_file(p: &Path) -> bool {
    unsafe {
        let v = native_file_name(p);
        lock::run(|| ffi::fftw_export_wisdom_to_filename(v) != 0)
    }
}

/// Attempt to load wisdom from `p`.
pub fn import_from_file(p: &Path) -> bool {
    unsafe {
        let v = native_file_name(p);
        lock::run(|| ffi::fftw_import_wisdom_from_filename(v) != 0)
    }
}
