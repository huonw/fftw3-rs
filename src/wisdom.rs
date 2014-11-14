use {ffi, lock};

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
    p: Path
}
impl WisdomGuard {
    /// Load wisdom from `p`, and save it automatically on clean-up.
    ///
    /// Failure to load is ignored, so one can supply a wisdom file
    /// that does not exist (yet) and it will be created on the next run.
    pub fn import(p: Path) -> WisdomGuard {
        import_from_file(p.clone());
        WisdomGuard { p: p }
    }
}
impl Drop for WisdomGuard {
    fn drop(&mut self) {
        export_to_file(self.p.clone());
    }
}

#[macro_export]
macro_rules! wisdom {
    ($p: expr) => { let _guard = fftw3::wisdom::WisdomGuard::import($p); }
}


/// Attempt to load the system's wisdom.
pub fn import_from_system() -> bool {
    unsafe {
        lock::run(|| ffi::fftw_import_system_wisdom() != 0)
    }
}

/// Attempt to save wisdom to `p`.
pub fn export_to_file(p: Path) -> bool {
    let mut v = p.into_vec();
    v.push(0);
    unsafe {
        lock::run(|| ffi::fftw_export_wisdom_to_filename(v.as_ptr() as *const i8) != 0)
    }
}

/// Attempt to load wisdom from `p`.
pub fn import_from_file(p: Path) -> bool {
    let mut v = p.into_vec();
    v.push(0);
    unsafe {
        lock::run(|| ffi::fftw_import_wisdom_from_filename(v.as_ptr() as *const i8) != 0)
    }
}
