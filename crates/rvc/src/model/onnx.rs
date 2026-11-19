use std::cell::RefCell;
use std::ffi::OsString;
use std::fmt::Debug;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use anyhow::{Context, Result, anyhow, bail, ensure};
use libloading::Library;
use ort::{
    ep::DirectML,
    session::Session,
    sys::{ORT_API_VERSION, OrtApiBase},
    value::{PrimitiveTensorElementType, TensorRef},
};
use tch::{Device, Tensor, kind::Element};
use tempfile::TempDir;

pub(crate) mod graph;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Provider {
    Cpu,
    DirectMl(i32),
}

impl Provider {
    pub(crate) fn selected(setting: &str, value: Option<OsString>) -> Result<Option<Self>> {
        value
            .map(|value| {
                let value = value
                    .to_str()
                    .with_context(|| format!("{setting} must be text"))?;
                Self::parse(setting, value)
            })
            .transpose()
    }

    pub(crate) fn parse(setting: &str, value: &str) -> Result<Self> {
        let provider = match value {
            "cpu" => Self::Cpu,
            "directml" => Self::DirectMl(0),
            value if value.starts_with("directml:") => {
                let adapter: i32 = value["directml:".len()..]
                    .parse()
                    .context("invalid DirectML adapter index")?;
                ensure!(adapter >= 0, "DirectML adapter index must be non-negative");
                Self::DirectMl(adapter)
            }
            _ => bail!("{setting} must be cpu, directml or directml:N"),
        };
        ensure!(
            matches!(provider, Self::Cpu) || cfg!(target_os = "windows"),
            "DirectML requires Windows"
        );
        Ok(provider)
    }

    pub(crate) fn session(self, path: &Path) -> Result<Session> {
        runtime()?;
        let threads = tch::get_num_threads().max(1) as usize;
        let mut builder = Session::builder()?
            .with_intra_threads(threads)
            .map_err(ort::Error::<()>::from)?;
        if let Self::DirectMl(adapter) = self {
            let directml = DirectML::default()
                .with_device_id(adapter)
                .build()
                .error_on_failure();
            builder = builder
                .with_memory_pattern(false)
                .map_err(ort::Error::<()>::from)?
                .with_parallel_execution(false)
                .map_err(ort::Error::<()>::from)?
                .with_execution_providers([directml])
                .map_err(ort::Error::<()>::from)?;
        }
        builder
            .commit_from_file(path)
            .with_context(|| format!("loading ONNX model {}", path.display()))
    }
}

pub(crate) struct GraphSession {
    pub(crate) session: RefCell<Session>,
    _directory: Option<TempDir>,
}

impl GraphSession {
    pub(crate) fn open(provider: Provider, path: &Path) -> Result<Self> {
        Ok(Self {
            session: RefCell::new(provider.session(path)?),
            _directory: None,
        })
    }

    pub(crate) fn load(
        provider: Provider,
        name: &str,
        export: impl FnOnce(&Path, &Path) -> Result<()>,
    ) -> Result<Self> {
        runtime()?;
        let directory = tempfile::Builder::new()
            .prefix(&format!("rvc-{name}-"))
            .tempdir()?;
        let path = directory.path().join(format!("{name}.onnx"));
        export(directory.path(), &path)?;
        let session = provider.session(&path)?;
        Ok(Self {
            session: RefCell::new(session),
            _directory: Some(directory),
        })
    }
}

pub(crate) struct TensorInput<T> {
    dimensions: Vec<i64>,
    values: Vec<T>,
}

impl<T: Element + Copy + Debug + PrimitiveTensorElementType + 'static> TensorInput<T> {
    pub(crate) fn new(tensor: &Tensor) -> Result<Self> {
        let dimensions = tensor.size();
        let tensor = tensor
            .f_to_device(Device::Cpu)?
            .f_contiguous()?
            .f_view([-1])?;
        Ok(Self {
            dimensions,
            values: Vec::try_from(tensor)?,
        })
    }

    pub(crate) fn view(&self) -> Result<TensorRef<'_, T>> {
        Ok(TensorRef::from_array_view((
            self.dimensions.as_slice(),
            self.values.as_slice(),
        ))?)
    }
}

pub(crate) fn runtime() -> Result<()> {
    static RUNTIME: OnceLock<std::result::Result<Library, String>> = OnceLock::new();
    RUNTIME
        .get_or_init(|| initialize_runtime().map_err(|error| format!("{error:#}")))
        .as_ref()
        .map(|_| ())
        .map_err(|error| anyhow!("{error}"))
}

fn initialize_runtime() -> Result<Library> {
    let path = std::env::var_os("RVC_ONNX_RUNTIME")
        .map(PathBuf::from)
        .context("RVC_ONNX_RUNTIME must point to the ONNX Runtime shared library")?;
    let path = path
        .canonicalize()
        .with_context(|| format!("resolving {}", path.display()))?;
    // Hold the library until process exit, so the registered C API stays valid.
    let library = open_library(&path).with_context(|| format!("loading {}", path.display()))?;
    let api = unsafe {
        type GetBase = unsafe extern "system" fn() -> *const OrtApiBase;
        let get_base = library.get::<GetBase>(b"OrtGetApiBase\0")?;
        let base = get_base()
            .as_ref()
            .context("ONNX Runtime returned no C API")?;
        (base.GetApi)(ORT_API_VERSION)
            .as_ref()
            .with_context(|| format!("ONNX Runtime does not support C API {ORT_API_VERSION}"))?
            .clone()
    };
    ensure!(
        ort::set_api(api),
        "an ONNX Runtime backend has already been configured"
    );
    ort::init().with_telemetry(false).commit();
    Ok(library)
}

fn open_library(path: &Path) -> Result<Library> {
    #[cfg(target_os = "windows")]
    {
        use libloading::os::windows::{
            LOAD_LIBRARY_SEARCH_DEFAULT_DIRS, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR,
            Library as WindowsLibrary,
        };
        Ok(unsafe {
            WindowsLibrary::load_with_flags(
                path,
                LOAD_LIBRARY_SEARCH_DEFAULT_DIRS | LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR,
            )?
        }
        .into())
    }
    #[cfg(not(target_os = "windows"))]
    {
        Ok(unsafe { Library::new(path)? })
    }
}
