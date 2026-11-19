use std::ffi::{CStr, c_char, c_int};
use std::sync::OnceLock;

use libloading::Library;
use serde::Serialize;
use tch::Cuda;

#[derive(Debug, Serialize)]
pub struct CudaProfile {
    pub name: String,
    pub capability: (i32, i32),
    pub memory_gib: f64,
    pub fp16: bool,
}

impl CudaProfile {
    pub fn eligible(&self) -> bool {
        self.memory_gib >= 4.0 && self.capability >= (6, 1)
    }
}

static PROFILES: OnceLock<Vec<Option<CudaProfile>>> = OnceLock::new();
static BACKEND: OnceLock<Option<Library>> = OnceLock::new();

pub fn cuda_device_count() -> usize {
    // CUDA registers LibTorch hooks on load. Downstream executables can lose
    // the unreferenced backend library to their linker's as-needed behavior.
    BACKEND.get_or_init(load_backend);
    if Cuda::is_available() {
        Cuda::device_count().max(0) as usize
    } else {
        0
    }
}

fn load_backend() -> Option<Library> {
    #[cfg(target_os = "linux")]
    {
        use libloading::os::unix::{Library as UnixLibrary, RTLD_GLOBAL, RTLD_NOW};
        // This resolves alongside the other LibTorch libraries at runtime.
        unsafe {
            UnixLibrary::open(Some("libtorch_cuda.so"), RTLD_NOW | RTLD_GLOBAL)
                .ok()
                .map(Into::into)
        }
    }
    #[cfg(target_os = "windows")]
    {
        // The matching CUDA backend is in the same LibTorch distribution.
        unsafe { Library::new("torch_cuda.dll").ok() }
    }
    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
    {
        None
    }
}

pub fn cuda_profiles() -> &'static [Option<CudaProfile>] {
    PROFILES.get_or_init(|| {
        let count = cuda_device_count();
        if count == 0 {
            return Vec::new();
        }
        let Some(driver) = driver_library() else {
            return (0..count).map(|_| None).collect();
        };
        let Ok(driver) = Driver::load(&driver) else {
            return (0..count).map(|_| None).collect();
        };
        (0..count).map(|index| driver.profile(index)).collect()
    })
}

pub fn cuda_profile(index: usize) -> Option<&'static CudaProfile> {
    cuda_profiles().get(index)?.as_ref()
}

fn driver_library() -> Option<Library> {
    #[cfg(target_os = "windows")]
    {
        use libloading::os::windows::{LOAD_LIBRARY_SEARCH_SYSTEM32, Library as WindowsLibrary};
        // The NVIDIA driver DLL is installed in System32.
        unsafe {
            WindowsLibrary::load_with_flags("nvcuda.dll", LOAD_LIBRARY_SEARCH_SYSTEM32)
                .ok()
                .map(Into::into)
        }
    }
    #[cfg(target_os = "linux")]
    {
        // This is the platform NVIDIA driver SONAME.
        unsafe { Library::new("libcuda.so.1").ok() }
    }
    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
    {
        None
    }
}

type DeviceGet = unsafe extern "system" fn(*mut c_int, c_int) -> c_int;
type Attribute = unsafe extern "system" fn(*mut c_int, c_int, c_int) -> c_int;
type Memory = unsafe extern "system" fn(*mut usize, c_int) -> c_int;
type Name = unsafe extern "system" fn(*mut c_char, c_int, c_int) -> c_int;

struct Driver<'a> {
    device_get: libloading::Symbol<'a, DeviceGet>,
    attribute: libloading::Symbol<'a, Attribute>,
    memory: libloading::Symbol<'a, Memory>,
    name: libloading::Symbol<'a, Name>,
}

impl<'a> Driver<'a> {
    fn load(library: &'a Library) -> anyhow::Result<Self> {
        type Initialize = unsafe extern "system" fn(u32) -> c_int;
        // cuda.h defines these signatures. CUDA has already been probed by LibTorch.
        unsafe {
            let init = library.get::<Initialize>(b"cuInit\0")?;
            anyhow::ensure!(init(0) == 0, "initializing the CUDA driver");
            Ok(Self {
                device_get: library.get(b"cuDeviceGet\0")?,
                attribute: library.get(b"cuDeviceGetAttribute\0")?,
                memory: library.get(b"cuDeviceTotalMem_v2\0")?,
                name: library.get(b"cuDeviceGetName\0")?,
            })
        }
    }

    fn profile(&self, index: usize) -> Option<CudaProfile> {
        let mut device = 0;
        let mut major = 0;
        let mut minor = 0;
        let mut memory = 0;
        let mut name = [0 as c_char; 256];
        const CC_MAJOR: c_int = 75;
        const CC_MINOR: c_int = 76;
        // Every output pointer refers to storage sized for the API result.
        let result = unsafe {
            (self.device_get)(&mut device, c_int::try_from(index).ok()?) == 0
                && (self.attribute)(&mut major, CC_MAJOR, device) == 0
                && (self.attribute)(&mut minor, CC_MINOR, device) == 0
                && (self.memory)(&mut memory, device) == 0
                && (self.name)(name.as_mut_ptr(), name.len() as c_int, device) == 0
        };
        if !result {
            return None;
        }
        name[name.len() - 1] = 0;
        // The fixed-size name buffer ends in NUL even on driver truncation.
        let name = unsafe { CStr::from_ptr(name.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        let memory_gib = memory as f64 / (1u64 << 30) as f64 + 0.4;
        let capability = (major, minor);
        let is_16_series = capability == (7, 5)
            && name
                .as_bytes()
                .windows(4)
                .any(|bytes| bytes.starts_with(b"16") && bytes[2..].iter().all(u8::is_ascii_digit));
        let fp16 = capability > (6, 1) && !is_16_series;
        Some(CudaProfile {
            name,
            capability,
            memory_gib,
            fp16,
        })
    }
}
