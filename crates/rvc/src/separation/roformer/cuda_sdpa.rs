#[cfg(rvc_cuda_sdpa)]
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result, bail};
use serde_json::Value;
use tch::{Device, Tensor};

#[derive(Clone, Copy, Debug)]
pub(in crate::separation) enum Policy {
    Auto,
    Default,
    Cudnn,
    Flash,
    Efficient,
    Math,
}

impl Policy {
    pub(in crate::separation) fn parse(value: &Value) -> Result<Self> {
        if value.is_null() {
            return Ok(Self::Cudnn);
        }
        let name = value
            .as_str()
            .context("cuda_attention_backend must be text")?
            .to_ascii_lowercase()
            .replace('-', "_");
        Ok(match name.as_str() {
            "auto" => Self::Auto,
            "default" | "torch" | "sdpa" => Self::Default,
            "cudnn" | "cudnn_attn" | "cudnn_attention" => Self::Cudnn,
            "flash" | "flash_attention" => Self::Flash,
            "efficient" | "mem_efficient" | "memory_efficient" | "xformers" => Self::Efficient,
            "math" => Self::Math,
            _ => bail!("unsupported CUDA attention backend {name:?}"),
        })
    }
}

pub(super) struct Backend {
    policy: Policy,
    #[cfg(rvc_cuda_sdpa)]
    no_cudnn_plan: AtomicBool,
}

impl Backend {
    pub(super) fn new(policy: Policy) -> Self {
        Self {
            policy,
            #[cfg(rvc_cuda_sdpa)]
            no_cudnn_plan: AtomicBool::new(false),
        }
    }

    pub(super) fn forward(&self, query: &Tensor, key: &Tensor, value: &Tensor) -> Result<Tensor> {
        if matches!(query.device(), Device::Cuda(_)) {
            if matches!(self.policy, Policy::Math) {
                return math(query, key, value);
            }
            #[cfg(rvc_cuda_sdpa)]
            if let Some(result) = self.forward_native(|kernel| kernel.forward(query, key, value))? {
                return Ok(result);
            }
        }
        Ok(Tensor::f_scaled_dot_product_attention(
            query,
            key,
            value,
            None::<&Tensor>,
            0.0,
            false,
            None,
            false,
        )?)
    }

    #[cfg(rvc_cuda_sdpa)]
    fn forward_native(
        &self,
        mut dispatch: impl FnMut(Kernel) -> Result<KernelOutput>,
    ) -> Result<Option<Tensor>> {
        let candidates = match self.policy {
            Policy::Auto => &[Kernel::Cudnn, Kernel::Efficient][..],
            Policy::Cudnn => &[Kernel::Cudnn],
            Policy::Flash => &[Kernel::Flash],
            Policy::Efficient => &[Kernel::Efficient],
            Policy::Default | Policy::Math => &[],
        };
        for &kernel in candidates {
            if matches!(kernel, Kernel::Cudnn) && self.no_cudnn_plan.load(Ordering::Relaxed) {
                continue;
            }
            match dispatch(kernel)? {
                KernelOutput::Ready(result) => return Ok(Some(result)),
                KernelOutput::Ineligible => (),
                KernelOutput::NoCudnnPlan => self.no_cudnn_plan.store(true, Ordering::Relaxed),
            }
        }
        Ok(None)
    }
}

fn math(query: &Tensor, key: &Tensor, value: &Tensor) -> Result<Tensor> {
    let (result, _) = Tensor::f_internal_scaled_dot_product_attention_math(
        query,
        key,
        value,
        None::<&Tensor>,
        0.0,
        false,
        None::<&Tensor>,
        None,
        false,
    )?;
    Ok(result)
}

#[cfg(rvc_cuda_sdpa)]
#[derive(Clone, Copy, Debug)]
#[repr(i32)]
enum Kernel {
    Cudnn = 0,
    Flash = 1,
    Efficient = 2,
}

#[cfg(rvc_cuda_sdpa)]
enum KernelOutput {
    Ready(Tensor),
    Ineligible,
    NoCudnnPlan,
}

#[cfg(rvc_cuda_sdpa)]
impl Kernel {
    fn forward(self, query: &Tensor, key: &Tensor, value: &Tensor) -> Result<KernelOutput> {
        use std::ffi::{CStr, c_char, c_void};

        unsafe extern "C" {
            fn rvc_roformer_cuda_sdpa(
                query: *const c_void,
                key: *const c_void,
                value: *const c_void,
                kernel: i32,
                output: *mut *mut c_void,
                no_cudnn_plan: *mut i32,
            ) -> *const c_char;
        }
        let mut output = std::ptr::null_mut();
        let mut no_cudnn_plan = 0;
        // Each live tch handle points to a LibTorch tensor. The bridge leaves
        // the output empty for an ineligible kernel and transfers ownership
        // of a newly allocated tensor on success.
        let error = unsafe {
            rvc_roformer_cuda_sdpa(
                query.as_ptr().cast(),
                key.as_ptr().cast(),
                value.as_ptr().cast(),
                self as i32,
                &mut output,
                &mut no_cudnn_plan,
            )
        };
        if !error.is_null() {
            let message = unsafe { CStr::from_ptr(error) }.to_string_lossy();
            bail!("{self:?} CUDA attention failed: {message}");
        }
        Ok(if no_cudnn_plan != 0 {
            KernelOutput::NoCudnnPlan
        } else if output.is_null() {
            KernelOutput::Ineligible
        } else {
            KernelOutput::Ready(unsafe { Tensor::from_ptr(output.cast()) })
        })
    }
}

#[cfg(all(test, rvc_cuda_sdpa))]
mod tests {
    use super::*;
    use tch::{Cuda, Kind, no_grad};

    #[test]
    fn unavailable_cudnn_plan_is_disabled_per_layer() -> Result<()> {
        let backend = Backend::new(Policy::Auto);
        let mut cudnn_attempts = 0;
        let mut efficient_attempts = 0;
        for _ in 0..2 {
            let result = backend.forward_native(|kernel| match kernel {
                Kernel::Cudnn => {
                    cudnn_attempts += 1;
                    Ok(KernelOutput::NoCudnnPlan)
                }
                Kernel::Efficient => {
                    efficient_attempts += 1;
                    Ok(KernelOutput::Ineligible)
                }
                Kernel::Flash => unreachable!(),
            })?;
            assert!(result.is_none());
        }
        assert_eq!(cudnn_attempts, 1);
        assert_eq!(efficient_attempts, 2);
        let second_layer = Backend::new(Policy::Cudnn);
        assert!(
            second_layer
                .forward_native(|_| bail!("CUDA out of memory"))
                .is_err()
        );
        assert!(!second_layer.no_cudnn_plan.load(Ordering::Relaxed));
        Ok(())
    }

    #[test]
    fn eligible_native_kernels_match_math_for_half_precision() -> Result<()> {
        if !Cuda::is_available() {
            return Ok(());
        }
        no_grad(|| {
            let options = (Kind::Half, Device::Cuda(0));
            let query = Tensor::f_randn([2, 4, 128, 64], options)?;
            let key = Tensor::f_randn([2, 4, 128, 64], options)?;
            let value = Tensor::f_randn([2, 4, 128, 64], options)?;
            let reference = math(&query, &key, &value)?;
            for kernel in [Kernel::Cudnn, Kernel::Flash, Kernel::Efficient] {
                let KernelOutput::Ready(actual) = kernel.forward(&query, &key, &value)? else {
                    eprintln!("{kernel:?} is ineligible on this GPU");
                    continue;
                };
                let delta = actual
                    .f_sub(&reference)?
                    .f_abs()?
                    .f_max()?
                    .double_value(&[]);
                eprintln!("{kernel:?} FP16 max difference {delta}");
                assert!(delta < 0.004, "{kernel:?} diverged from math: {delta}");
            }
            Ok(())
        })
    }
}
