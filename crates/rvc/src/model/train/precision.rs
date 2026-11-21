use anyhow::{Result, ensure};
use tch::{Device, Kind};

use crate::model::weights;

unsafe extern "C" {
    fn rvc_enter_autocast(precision: std::ffi::c_int) -> std::ffi::c_int;
    fn rvc_leave_autocast(previous: std::ffi::c_int);
}

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub(super) enum Precision {
    Fp32,
    Fp16,
    Bf16,
}

impl Precision {
    pub(super) fn choose(devices: &[Device], fp16: Option<bool>, bf16: bool) -> Result<Self> {
        ensure!(
            !(bf16 && fp16 == Some(true)),
            "select FP16 or BF16, not both"
        );
        let mut automatic_fp16 = true;
        let mut all_cuda = true;
        for &device in devices {
            if let Device::Cuda(index) = device {
                let profile = weights::cuda_profile(index);
                if let Some(profile) = profile {
                    ensure!(
                        profile.eligible(),
                        "CUDA device {index} does not meet the training memory or capability requirements"
                    );
                }
                if bf16 {
                    ensure!(
                        profile.is_some_and(|profile| profile.capability.0 >= 8),
                        "BF16 training requires CUDA compute capability 8.0 or higher on device {index}"
                    );
                }
                automatic_fp16 &= profile.is_some_and(|profile| profile.fp16);
            } else {
                all_cuda = false;
            }
        }
        ensure!(all_cuda || !bf16, "BF16 training requires CUDA devices");
        ensure!(
            all_cuda || fp16 != Some(true),
            "FP16 training requires CUDA devices"
        );
        Ok(if bf16 {
            Self::Bf16
        } else if all_cuda && fp16.unwrap_or(automatic_fp16) {
            Self::Fp16
        } else {
            Self::Fp32
        })
    }

    pub(super) fn name(self) -> &'static str {
        match self {
            Self::Fp32 => "FP32",
            Self::Fp16 => "FP16 mixed precision",
            Self::Bf16 => "BF16 mixed precision",
        }
    }

    pub(super) fn kind(self) -> Kind {
        match self {
            Self::Fp32 => Kind::Float,
            Self::Fp16 => Kind::Half,
            Self::Bf16 => Kind::BFloat16,
        }
    }

    pub(super) fn run<T>(self, operation: impl FnOnce() -> T) -> T {
        struct Restore(std::ffi::c_int);
        impl Drop for Restore {
            fn drop(&mut self) {
                // The saved mode is restored on the same thread, including on unwind.
                unsafe { rvc_leave_autocast(self.0) };
            }
        }
        let _restore = Restore(unsafe { rvc_enter_autocast(self as i32) });
        operation()
    }
}

pub(super) struct GradScaler {
    factor: Option<f32>,
    finite_steps: usize,
}

impl GradScaler {
    pub(super) fn new(enabled: bool) -> Self {
        Self {
            factor: enabled.then_some(65_536.0),
            finite_steps: 0,
        }
    }

    pub(super) fn factor(&self) -> Option<f32> {
        self.factor
    }

    pub(super) fn update(&mut self, overflow: bool) {
        let Some(scale) = self.factor.as_mut() else {
            return;
        };
        if overflow {
            *scale *= 0.5;
            self.finite_steps = 0;
        } else {
            self.finite_steps += 1;
            if self.finite_steps == 2000 {
                let grown = *scale * 2.0;
                if grown.is_finite() {
                    *scale = grown;
                }
                self.finite_steps = 0;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::GradScaler;

    #[test]
    fn grows_after_2000_consecutive_steps_and_backs_off_on_overflow() {
        let mut scale = GradScaler::new(true);
        for _ in 0..1999 {
            scale.update(false);
        }
        assert_eq!(scale.factor(), Some(65_536.0));
        scale.update(false);
        assert_eq!(scale.factor(), Some(131_072.0));
        scale.update(true);
        assert_eq!(scale.factor(), Some(65_536.0));
        scale.update(false);
        assert_eq!(scale.factor(), Some(65_536.0));
    }
}
