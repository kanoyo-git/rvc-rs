use std::ffi::{CStr, c_char, c_int, c_uint, c_ulong, c_void};
use std::ptr;
use std::sync::OnceLock;

use anyhow::{Result, ensure};
use libloading::Library;

#[repr(C)]
struct IoSpec {
    input_type: c_int,
    output_type: c_int,
    scale: f64,
    reserved: *mut c_void,
    flags: c_ulong,
}

#[repr(C)]
struct QualitySpec {
    precision: f64,
    phase_response: f64,
    passband_end: f64,
    stopband_begin: f64,
    reserved: *mut c_void,
    flags: c_ulong,
}

type OneShot = unsafe extern "C" fn(
    f64,
    f64,
    c_uint,
    *const c_void,
    usize,
    *mut usize,
    *mut c_void,
    usize,
    *mut usize,
    *const IoSpec,
    *const QualitySpec,
    *const c_void,
) -> *const c_char;

pub(super) struct Soxr {
    _library: Library,
    one_shot: OneShot,
    io: unsafe extern "C" fn(c_int, c_int) -> IoSpec,
    quality: unsafe extern "C" fn(c_ulong, c_ulong) -> QualitySpec,
}

impl Soxr {
    pub(super) fn get() -> Option<&'static Self> {
        static LIBRARY: OnceLock<Option<Soxr>> = OnceLock::new();
        LIBRARY.get_or_init(Self::open).as_ref()
    }

    fn open() -> Option<Self> {
        #[cfg(all(unix, not(target_os = "macos")))]
        const NAMES: &[&str] = &["libsoxr.so.0", "libsoxr.so"];
        #[cfg(target_os = "macos")]
        const NAMES: &[&str] = &[
            "libsoxr.0.dylib",
            "libsoxr.dylib",
            "/opt/homebrew/lib/libsoxr.0.dylib",
        ];
        #[cfg(windows)]
        const NAMES: &[&str] = &["libsoxr.dll", "libsoxr-0.dll", "soxr.dll"];

        NAMES.iter().find_map(|name| {
            use std::ffi::OsStr;
            let name = OsStr::new(name);
            // These filenames and symbols use the stable libsoxr C ABI.
            unsafe {
                #[cfg(unix)]
                let library = Library::new(name).ok()?;
                #[cfg(windows)]
                let library: Library = libloading::os::windows::Library::load_with_flags(
                    name,
                    libloading::os::windows::LOAD_LIBRARY_SEARCH_DEFAULT_DIRS,
                )
                .ok()?
                .into();
                Some(Self {
                    one_shot: *library.get(b"soxr_oneshot\0").ok()?,
                    io: *library.get(b"soxr_io_spec\0").ok()?,
                    quality: *library.get(b"soxr_quality_spec\0").ok()?,
                    _library: library,
                })
            }
        })
    }

    pub(super) fn resample(
        &self,
        left: &[f32],
        right: &[f32],
        from: u32,
        to: u32,
    ) -> Result<[Vec<f32>; 2]> {
        const FLOAT32_PLANAR: c_int = 4;
        const HIGH_QUALITY: c_ulong = 4;
        let output_frames =
            usize::try_from((left.len() as u128 * u128::from(to)).div_ceil(u128::from(from)))?;
        let mut result = [vec![0.0f32; output_frames], vec![0.0f32; output_frames]];
        let input = [left.as_ptr(), right.as_ptr()];
        let mut output = [result[0].as_mut_ptr(), result[1].as_mut_ptr()];
        let mut consumed = 0;
        let mut written = 0;
        // Each pointer in the planar arrays addresses exactly the declared number of samples.
        let error = unsafe {
            let io = (self.io)(FLOAT32_PLANAR, FLOAT32_PLANAR);
            let quality = (self.quality)(HIGH_QUALITY, 0);
            (self.one_shot)(
                f64::from(from),
                f64::from(to),
                2,
                input.as_ptr().cast(),
                left.len(),
                &mut consumed,
                output.as_mut_ptr().cast(),
                output_frames,
                &mut written,
                &io,
                &quality,
                ptr::null(),
            )
        };
        if !error.is_null() {
            let detail = unsafe { CStr::from_ptr(error) };
            anyhow::bail!("VR SoXR resampling failed: {}", detail.to_string_lossy());
        }
        ensure!(
            consumed == left.len() && written <= output_frames,
            "VR SoXR resampling returned invalid lengths"
        );
        Ok(result)
    }
}
