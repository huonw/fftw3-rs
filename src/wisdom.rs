use {ffi, lock};

/// Attempt to load the system's wisdom.
pub fn import_from_system() -> bool {
    unsafe {
        lock::run(|| ffi::fftw_import_system() != 0)
    }
}

/// Attempt to save wisdom to `p`.
pub fn export_to_file(p: Path) -> bool {
    let mut v = p.into_vec();
    v.push(0);
    unsafe {
        lock::run(|| ffi::fftw_export_to_filename(v.as_ptr() as *i8) != 0)
    }
}

/// Attempt to load wisdom from `p`.
pub fn import_from_file(p: Path) -> bool {
    let mut v = p.into_vec();
    v.push(0);
    unsafe {
        lock::run(|| ffi::fftw_import_from_filename(v.as_ptr() as *i8) != 0)
    }
}
