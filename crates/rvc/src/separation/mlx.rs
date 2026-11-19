use std::cell::RefCell;
use std::ffi::{CStr, c_char, c_void};
use std::path::PathBuf;
use std::ptr;
use std::sync::OnceLock;

use anyhow::{Context, Result, anyhow, bail, ensure};
use libloading::Library;
use serde_json::Value;
use tch::{Device, Kind, Tensor};

// MLX C 0.8 uses one opaque pointer for arrays, streams, strings and vectors.
#[derive(Clone, Copy)]
#[repr(C)]
struct Handle(*mut c_void);

impl Handle {
    const EMPTY: Self = Self(ptr::null_mut());
}

#[derive(Clone, Copy)]
#[repr(i32)]
pub(super) enum DType {
    I32 = 7,
    F16 = 9,
    F32 = 10,
}

pub(super) fn selected_dtype(
    document: &Value,
    device: Device,
    supported: bool,
    model: &str,
) -> Result<Option<DType>> {
    let field = &document["inference"]["mps_model_backend"];
    let backend = field.as_str();
    ensure!(
        field.is_null() || backend.is_some(),
        "mps_model_backend must be text"
    );
    match backend {
        Some("torch") => return Ok(None),
        Some("mlx_full") => ensure!(supported, "{model} does not support this MLX configuration"),
        Some(value) => bail!("unknown mps_model_backend {value:?}"),
        None if supported
            && device == Device::Mps
            && std::env::var_os("RVC_MLX_LIBRARY").is_some() => {}
        None => return Ok(None),
    }
    ensure!(
        matches!(device, Device::Cpu | Device::Mps),
        "MLX requires CPU or MPS"
    );
    compute_dtype(&document["inference"]["mps_model_compute_dtype"], device).map(Some)
}

pub(super) fn compute_dtype(dtype: &Value, device: Device) -> Result<DType> {
    match dtype.as_str() {
        Some("float16") => Ok(DType::F16),
        Some("float32") => Ok(DType::F32),
        None if dtype.is_null() => Ok(if device == Device::Cpu {
            DType::F32
        } else {
            DType::F16
        }),
        _ => bail!("mps_model_compute_dtype must be float16 or float32"),
    }
}

type Constructor = unsafe extern "C" fn() -> Handle;
type Release = unsafe extern "C" fn(Handle) -> i32;
type Unary = unsafe extern "C" fn(*mut Handle, Handle, Handle) -> i32;
type Binary = unsafe extern "C" fn(*mut Handle, Handle, Handle, Handle) -> i32;
type Shape = unsafe extern "C" fn(*mut Handle, Handle, *const i32, usize, Handle) -> i32;
type Axis = unsafe extern "C" fn(*mut Handle, Handle, i32, Handle) -> i32;
type OnError = unsafe extern "C" fn(*const c_char, *mut c_void);
type SetError =
    unsafe extern "C" fn(OnError, *mut c_void, Option<unsafe extern "C" fn(*mut c_void)>);

macro_rules! functions {
    ($($name:ident: $signature:ty => $symbol:literal),+ $(,)?) => {
        struct Api {
            $($name: $signature,)+
            _library: Library,
        }
        impl Api {
            fn load() -> Result<Self> {
                let path = std::env::var_os("RVC_MLX_LIBRARY").map(PathBuf::from)
                    .context("RVC_MLX_LIBRARY must point to MLX C 0.8")?;
                let path = path.canonicalize().with_context(|| format!("resolving {}", path.display()))?;
                // The ABI used below is the MLX C 0.8 public header.
                let library = unsafe { Library::new(&path) }.with_context(|| format!("loading {}", path.display()))?;
                let api = Self {
                    $($name: *unsafe { library.get::<$signature>($symbol) }?,)+
                    _library: library,
                };
                unsafe { (api.set_error)(record_error, ptr::null_mut(), None) };
                let mut version = Handle::EMPTY;
                let result = status(unsafe { (api.version)(&mut version) });
                let text = if result.is_ok() && !version.0.is_null() {
                    let data = unsafe { (api.string_data)(version) };
                    (!data.is_null()).then(|| unsafe { CStr::from_ptr(data).to_string_lossy().into_owned() })
                } else {
                    None
                };
                if !version.0.is_null() {
                    unsafe { (api.string_free)(version) };
                }
                result?;
                ensure!(text.as_deref() == Some("0.32.4"), "MLX C 0.8 with MLX 0.32.4 is required, found {text:?}");
                Ok(api)
            }
        }
    };
}

functions! {
    set_error: SetError => b"mlx_set_error_handler",
    cpu_stream: Constructor => b"mlx_default_cpu_stream_new",
    gpu_stream: Constructor => b"mlx_default_gpu_stream_new",
    stream_free: Release => b"mlx_stream_free",
    version: unsafe extern "C" fn(*mut Handle) -> i32 => b"mlx_version",
    string_data: unsafe extern "C" fn(Handle) -> *const c_char => b"mlx_string_data",
    clear_cache: unsafe extern "C" fn() -> i32 => b"mlx_clear_cache",
    string_free: Release => b"mlx_string_free",
    array_data: unsafe extern "C" fn(*const c_void, *const i32, i32, DType) -> Handle => b"mlx_array_new_data",
    scalar: unsafe extern "C" fn(f32) -> Handle => b"mlx_array_new_float",
    complex: unsafe extern "C" fn(f32, f32) -> Handle => b"mlx_array_new_complex",
    free: Release => b"mlx_array_free",
    eval: Release => b"mlx_array_eval",
    size: unsafe extern "C" fn(Handle) -> usize => b"mlx_array_size",
    ndim: unsafe extern "C" fn(Handle) -> usize => b"mlx_array_ndim",
    shape: unsafe extern "C" fn(Handle) -> *const i32 => b"mlx_array_shape",
    floats: unsafe extern "C" fn(Handle) -> *const f32 => b"mlx_array_data_float32",
    cast: unsafe extern "C" fn(*mut Handle, Handle, DType, Handle) -> i32 => b"mlx_astype",
    contiguous: unsafe extern "C" fn(*mut Handle, Handle, bool, Handle) -> i32 => b"mlx_contiguous",
    reshape: Shape => b"mlx_reshape",
    transpose: Shape => b"mlx_transpose_axes",
    strided: unsafe extern "C" fn(*mut Handle, Handle, *const i32, usize, *const i64, usize, usize, Handle) -> i32 => b"mlx_as_strided",
    broadcast: Shape => b"mlx_broadcast_to",
    slice: unsafe extern "C" fn(*mut Handle, Handle, *const i32, usize, *const i32, usize, *const i32, usize, Handle) -> i32 => b"mlx_slice",
    arange: unsafe extern "C" fn(*mut Handle, f64, f64, f64, DType, Handle) -> i32 => b"mlx_arange",
    zeros: unsafe extern "C" fn(*mut Handle, *const i32, usize, DType, Handle) -> i32 => b"mlx_zeros",
    take: unsafe extern "C" fn(*mut Handle, Handle, Handle, i32, Handle) -> i32 => b"mlx_take_axis",
    scatter_add: unsafe extern "C" fn(*mut Handle, Handle, Handle, Handle, i32, Handle) -> i32 => b"mlx_scatter_add_axis",
    stack: Axis => b"mlx_stack_axis",
    concat: Axis => b"mlx_concatenate_axis",
    vec_new: unsafe extern "C" fn(*const Handle, usize) -> Handle => b"mlx_vector_array_new_data",
    vec_free: Release => b"mlx_vector_array_free",
    add: Binary => b"mlx_add",
    sub: Binary => b"mlx_subtract",
    mul: Binary => b"mlx_multiply",
    div: Binary => b"mlx_divide",
    max: Binary => b"mlx_maximum",
    min: Binary => b"mlx_minimum",
    logaddexp: Binary => b"mlx_logaddexp",
    matmul: Binary => b"mlx_matmul",
    square: Unary => b"mlx_square",
    sqrt: Unary => b"mlx_sqrt",
    log: Unary => b"mlx_log",
    rsqrt: Unary => b"mlx_rsqrt",
    expm1: Unary => b"mlx_expm1",
    sin: Unary => b"mlx_sin",
    cos: Unary => b"mlx_cos",
    tanh: Unary => b"mlx_tanh",
    real: Unary => b"mlx_real",
    imag: Unary => b"mlx_imag",
    erf: Unary => b"mlx_erf",
    sigmoid: Unary => b"mlx_sigmoid",
    sum: unsafe extern "C" fn(*mut Handle, Handle, i32, bool, Handle) -> i32 => b"mlx_sum_axis",
    amax: unsafe extern "C" fn(*mut Handle, Handle, i32, bool, Handle) -> i32 => b"mlx_max_axis",
    mean: unsafe extern "C" fn(*mut Handle, Handle, *const i32, usize, bool, Handle) -> i32 => b"mlx_mean_axes",
    conv1d: unsafe extern "C" fn(*mut Handle, Handle, Handle, i32, i32, i32, i32, Handle) -> i32 => b"mlx_conv1d",
    upsample1d: unsafe extern "C" fn(*mut Handle, Handle, Handle, i32, i32, i32, i32, i32, Handle) -> i32 => b"mlx_conv_transpose1d",
    conv2d: unsafe extern "C" fn(*mut Handle, Handle, Handle, i32, i32, i32, i32, i32, i32, i32, Handle) -> i32 => b"mlx_conv2d",
    upsample2d: unsafe extern "C" fn(*mut Handle, Handle, Handle, i32, i32, i32, i32, i32, i32, i32, i32, i32, Handle) -> i32 => b"mlx_conv_transpose2d",
    softmax: unsafe extern "C" fn(*mut Handle, Handle, i32, bool, Handle) -> i32 => b"mlx_softmax_axis",
    attention: unsafe extern "C" fn(*mut Handle, Handle, Handle, Handle, f32, *const c_char, Handle, Handle, bool, Handle) -> i32 => b"mlx_fast_scaled_dot_product_attention",
    fft: unsafe extern "C" fn(*mut Handle, Handle, i32, i32, i32, Handle) -> i32 => b"mlx_fft_rfft",
    ifft: unsafe extern "C" fn(*mut Handle, Handle, i32, i32, i32, Handle) -> i32 => b"mlx_fft_irfft",
}

thread_local! { static ERROR: RefCell<Option<String>> = const { RefCell::new(None) }; }
static API: OnceLock<std::result::Result<Api, String>> = OnceLock::new();

pub(super) fn clear_cache() -> Result<()> {
    if let Some(Ok(api)) = API.get() {
        status(unsafe { (api.clear_cache)() })?;
    }
    Ok(())
}

unsafe extern "C" fn record_error(message: *const c_char, _: *mut c_void) {
    if !message.is_null() {
        let message = unsafe { CStr::from_ptr(message) }
            .to_string_lossy()
            .into_owned();
        let _ = ERROR.try_with(move |error| {
            if let Ok(mut error) = error.try_borrow_mut() {
                *error = Some(message);
            }
        });
    }
}

fn error() -> anyhow::Error {
    anyhow!(
        "{}",
        ERROR
            .with(|error| error.borrow_mut().take())
            .unwrap_or_else(|| "MLX operation failed".to_owned())
    )
}

fn status(code: i32) -> Result<()> {
    if code == 0 { Ok(()) } else { Err(error()) }
}

pub(super) struct Array {
    raw: Handle,
    api: &'static Api,
}

// The handle owns a C++ array reference. Moving the owner transfers access.
unsafe impl Send for Array {}

impl Array {
    pub(super) fn shape(&self) -> &[i32] {
        let dimensions = unsafe { (self.api.ndim)(self.raw) };
        if dimensions == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts((self.api.shape)(self.raw), dimensions) }
        }
    }

    pub(super) fn eval(&self) -> Result<()> {
        status(unsafe { (self.api.eval)(self.raw) })
    }
}

impl Drop for Array {
    fn drop(&mut self) {
        if !self.raw.0.is_null() {
            unsafe { (self.api.free)(self.raw) };
        }
    }
}

pub(super) struct Stream {
    raw: Handle,
    api: &'static Api,
}

// An MLX stream is an index on a specific device, copied by value in C.
// The pipeline moves the stream owner to a worker and processes calls serially.
unsafe impl Send for Stream {}

macro_rules! binary_ops {
    ($($name:ident),+ $(,)?) => {$(
        pub(super) fn $name(&self, left: &Array, right: &Array) -> Result<Array> {
            self.result(|out| unsafe { (self.api.$name)(out, left.raw, right.raw, self.raw) })
        }
    )+};
}

macro_rules! unary_ops {
    ($($name:ident),+ $(,)?) => {$(
        pub(super) fn $name(&self, array: &Array) -> Result<Array> {
            self.result(|out| unsafe { (self.api.$name)(out, array.raw, self.raw) })
        }
    )+};
}

impl Stream {
    pub(super) fn new(device: Device) -> Result<Self> {
        let api = API
            .get_or_init(|| Api::load().map_err(|error| format!("{error:#}")))
            .as_ref()
            .map_err(|error| anyhow!("{error}"))?;
        ensure!(
            matches!(device, Device::Cpu | Device::Mps),
            "MLX requires CPU or an Apple GPU"
        );
        let raw = unsafe {
            if device == Device::Mps {
                (api.gpu_stream)()
            } else {
                (api.cpu_stream)()
            }
        };
        if raw.0.is_null() {
            return Err(error());
        }
        Ok(Self { raw, api })
    }

    fn array(&self, raw: Handle) -> Result<Array> {
        if raw.0.is_null() {
            Err(error())
        } else {
            Ok(Array { raw, api: self.api })
        }
    }

    fn result(&self, run: impl FnOnce(*mut Handle) -> i32) -> Result<Array> {
        let mut owned = Array {
            raw: Handle::EMPTY,
            api: self.api,
        };
        status(run(&mut owned.raw))?;
        ensure!(!owned.raw.0.is_null(), "MLX returned an empty array");
        Ok(owned)
    }

    pub(super) fn upload(&self, tensor: &Tensor, dtype: DType) -> Result<Array> {
        let shape = tensor
            .size()
            .into_iter()
            .map(i32::try_from)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let buffer = Vec::<f32>::try_from(
            tensor
                .f_to_device(Device::Cpu)?
                .f_to_kind(Kind::Float)?
                .f_contiguous()?
                .f_view([-1])?,
        )?;
        let array = self.float32(&buffer, &shape)?;
        self.cast(&array, dtype)
    }

    pub(super) fn float32(&self, data: &[f32], shape: &[i32]) -> Result<Array> {
        self.copy_data(data.as_ptr().cast(), data.len(), shape, DType::F32)
    }

    pub(super) fn indices(&self, data: &[i32], shape: &[i32]) -> Result<Array> {
        self.copy_data(data.as_ptr().cast(), data.len(), shape, DType::I32)
    }

    fn copy_data(
        &self,
        data: *const c_void,
        len: usize,
        shape: &[i32],
        dtype: DType,
    ) -> Result<Array> {
        let count = shape
            .iter()
            .try_fold(1usize, |n, &d| n.checked_mul(usize::try_from(d).ok()?));
        ensure!(
            count == Some(len),
            "MLX data does not match the given shape"
        );
        let rank = i32::try_from(shape.len())?;
        self.array(unsafe { (self.api.array_data)(data, shape.as_ptr(), rank, dtype) })
    }

    pub(super) fn to_floats(&self, array: &Array) -> Result<Vec<f32>> {
        let array = self.cast(array, DType::F32)?;
        let array =
            self.result(|out| unsafe { (self.api.contiguous)(out, array.raw, false, self.raw) })?;
        array.eval()?;
        let length = unsafe { (self.api.size)(array.raw) };
        let data = unsafe { (self.api.floats)(array.raw) };
        ensure!(!data.is_null(), "MLX returned no float data");
        let samples = unsafe { std::slice::from_raw_parts(data, length) };
        Ok(samples.to_vec())
    }

    pub(super) fn scalar(&self, value: f32, dtype: DType) -> Result<Array> {
        let array = self.array(unsafe { (self.api.scalar)(value) })?;
        self.cast(&array, dtype)
    }

    pub(super) fn cast(&self, array: &Array, dtype: DType) -> Result<Array> {
        self.result(|out| unsafe { (self.api.cast)(out, array.raw, dtype, self.raw) })
    }

    pub(super) fn reshape(&self, array: &Array, shape: &[i32]) -> Result<Array> {
        self.result(|out| unsafe {
            (self.api.reshape)(out, array.raw, shape.as_ptr(), shape.len(), self.raw)
        })
    }

    pub(super) fn transpose(&self, array: &Array, axes: &[i32]) -> Result<Array> {
        self.result(|out| unsafe {
            (self.api.transpose)(out, array.raw, axes.as_ptr(), axes.len(), self.raw)
        })
    }

    pub(super) fn strided(&self, array: &Array, shape: &[i32], strides: &[i64]) -> Result<Array> {
        self.result(|out| unsafe {
            (self.api.strided)(
                out,
                array.raw,
                shape.as_ptr(),
                shape.len(),
                strides.as_ptr(),
                strides.len(),
                0,
                self.raw,
            )
        })
    }

    pub(super) fn broadcast(&self, array: &Array, shape: &[i32]) -> Result<Array> {
        self.result(|out| unsafe {
            (self.api.broadcast)(out, array.raw, shape.as_ptr(), shape.len(), self.raw)
        })
    }

    pub(super) fn slice(
        &self,
        array: &Array,
        axis: usize,
        start: i32,
        stop: i32,
        step: i32,
    ) -> Result<Array> {
        let mut stops = array.shape().to_vec();
        ensure!(
            axis < stops.len(),
            "MLX slice axis {axis} is outside the array"
        );
        let mut starts = vec![0; stops.len()];
        let mut steps = vec![1; stops.len()];
        starts[axis] = start;
        stops[axis] = stop;
        steps[axis] = step;
        self.result(|out| unsafe {
            (self.api.slice)(
                out,
                array.raw,
                starts.as_ptr(),
                starts.len(),
                stops.as_ptr(),
                stops.len(),
                steps.as_ptr(),
                steps.len(),
                self.raw,
            )
        })
    }

    pub(super) fn select(&self, array: &Array, axis: usize, index: i32) -> Result<Array> {
        let slice = self.slice(array, axis, index, index + 1, 1)?;
        let mut shape = slice.shape().to_vec();
        shape.remove(axis);
        self.reshape(&slice, &shape)
    }

    pub(super) fn arange(&self, end: i32, dtype: DType) -> Result<Array> {
        self.result(|out| unsafe {
            (self.api.arange)(out, 0., f64::from(end), 1., dtype, self.raw)
        })
    }

    pub(super) fn zeros(&self, shape: &[i32]) -> Result<Array> {
        self.result(|out| unsafe {
            (self.api.zeros)(out, shape.as_ptr(), shape.len(), DType::F32, self.raw)
        })
    }

    pub(super) fn take(&self, array: &Array, indices: &Array, axis: i32) -> Result<Array> {
        self.result(|out| unsafe { (self.api.take)(out, array.raw, indices.raw, axis, self.raw) })
    }

    pub(super) fn scatter_add(
        &self,
        base: &Array,
        indices: &Array,
        values: &Array,
        axis: i32,
    ) -> Result<Array> {
        self.result(|out| unsafe {
            (self.api.scatter_add)(out, base.raw, indices.raw, values.raw, axis, self.raw)
        })
    }

    pub(super) fn fft(&self, array: &Array, fft: i32, normalized: bool) -> Result<Array> {
        self.result(|out| unsafe {
            (self.api.fft)(out, array.raw, fft, -1, i32::from(normalized), self.raw)
        })
    }

    pub(super) fn ifft(&self, array: &Array, fft: i32, normalized: bool) -> Result<Array> {
        self.result(|out| unsafe {
            (self.api.ifft)(out, array.raw, fft, -1, i32::from(normalized), self.raw)
        })
    }

    pub(super) fn complex(&self, real: &Array, imaginary: &Array) -> Result<Array> {
        let i = self.array(unsafe { (self.api.complex)(0.0, 1.0) })?;
        self.add(real, &self.mul(imaginary, &i)?)
    }

    pub(super) fn stack(&self, arrays: &[Array], axis: i32) -> Result<Array> {
        self.join(arrays, axis, self.api.stack)
    }

    pub(super) fn concat(&self, arrays: &[Array], axis: i32) -> Result<Array> {
        self.join(arrays, axis, self.api.concat)
    }

    fn join(&self, arrays: &[Array], axis: i32, operation: Axis) -> Result<Array> {
        let raw: Vec<_> = arrays.iter().map(|array| array.raw).collect();
        let vector = unsafe { (self.api.vec_new)(raw.as_ptr(), raw.len()) };
        if vector.0.is_null() {
            return Err(error());
        }
        let result = self.result(|out| unsafe { operation(out, vector, axis, self.raw) });
        unsafe { (self.api.vec_free)(vector) };
        result
    }

    pub(super) fn sum_keep(&self, array: &Array, axis: i32) -> Result<Array> {
        self.result(|out| unsafe { (self.api.sum)(out, array.raw, axis, true, self.raw) })
    }

    pub(super) fn max_keep(&self, array: &Array, axis: i32) -> Result<Array> {
        self.result(|out| unsafe { (self.api.amax)(out, array.raw, axis, true, self.raw) })
    }

    pub(super) fn mean_keep(&self, array: &Array, axes: &[i32]) -> Result<Array> {
        self.result(|out| unsafe {
            (self.api.mean)(out, array.raw, axes.as_ptr(), axes.len(), true, self.raw)
        })
    }

    pub(super) fn conv1d(
        &self,
        input: &Array,
        weight: &Array,
        pad: i32,
        groups: i32,
    ) -> Result<Array> {
        self.convolution1d(input, weight, 1, pad, 1, groups)
    }

    pub(super) fn convolution1d(
        &self,
        input: &Array,
        weight: &Array,
        stride: i32,
        pad: i32,
        dilation: i32,
        groups: i32,
    ) -> Result<Array> {
        self.result(|out| unsafe {
            (self.api.conv1d)(
                out, input.raw, weight.raw, stride, pad, dilation, groups, self.raw,
            )
        })
    }

    pub(super) fn upsample1d(&self, input: &Array, weight: &Array, stride: i32) -> Result<Array> {
        self.result(|out| unsafe {
            (self.api.upsample1d)(out, input.raw, weight.raw, stride, 0, 1, 0, 1, self.raw)
        })
    }

    pub(super) fn conv2d(
        &self,
        input: &Array,
        weight: &Array,
        stride: [i32; 2],
        pad: [i32; 2],
    ) -> Result<Array> {
        self.convolution2d(input, weight, stride, pad, [1, 1], 1)
    }

    pub(super) fn convolution2d(
        &self,
        input: &Array,
        weight: &Array,
        stride: [i32; 2],
        pad: [i32; 2],
        dilation: [i32; 2],
        groups: i32,
    ) -> Result<Array> {
        self.result(|out| unsafe {
            (self.api.conv2d)(
                out,
                input.raw,
                weight.raw,
                stride[0],
                stride[1],
                pad[0],
                pad[1],
                dilation[0],
                dilation[1],
                groups,
                self.raw,
            )
        })
    }

    pub(super) fn upsample2d(
        &self,
        input: &Array,
        weight: &Array,
        stride: [i32; 2],
    ) -> Result<Array> {
        self.result(|out| unsafe {
            (self.api.upsample2d)(
                out, input.raw, weight.raw, stride[0], stride[1], 0, 0, 1, 1, 0, 0, 1, self.raw,
            )
        })
    }

    pub(super) fn softmax(&self, array: &Array, axis: i32) -> Result<Array> {
        self.result(|out| unsafe { (self.api.softmax)(out, array.raw, axis, true, self.raw) })
    }

    pub(super) fn attention(
        &self,
        query: &Array,
        key: &Array,
        value: &Array,
        scale: f32,
    ) -> Result<Array> {
        self.result(|out| unsafe {
            (self.api.attention)(
                out,
                query.raw,
                key.raw,
                value.raw,
                scale,
                c"".as_ptr(),
                Handle::EMPTY,
                Handle::EMPTY,
                false,
                self.raw,
            )
        })
    }

    binary_ops!(add, sub, mul, div, max, min, logaddexp, matmul);
    unary_ops!(
        square, sqrt, log, rsqrt, expm1, sin, cos, tanh, real, imag, erf, sigmoid
    );
}

impl Drop for Stream {
    fn drop(&mut self) {
        unsafe { (self.api.stream_free)(self.raw) };
    }
}
