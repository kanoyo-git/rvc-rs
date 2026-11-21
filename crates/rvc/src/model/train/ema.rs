//! Exponential moving average of generator weights for exported models.

use anyhow::{Result, ensure};
use tch::Tensor;

unsafe extern "C" {
    fn rvc_foreach_lerp(
        values: *const *mut std::ffi::c_void,
        targets: *const *mut std::ffi::c_void,
        count: usize,
        weight: f64,
        error: *mut std::ffi::c_char,
        error_length: usize,
    ) -> std::ffi::c_int;
}

/// Shadow copies of the generator parameters. Exports use these weights,
/// which smooths the step-to-step fluctuation of adversarial training.
pub(super) struct Ema {
    decay: f64,
    updates: u64,
    shadow: Vec<(String, Tensor)>,
}

impl Ema {
    pub(super) fn new(parameters: &[(String, Tensor)], decay: f64) -> Result<Self> {
        ensure!(
            decay > 0.0 && decay < 1.0,
            "EMA decay must be between 0 and 1"
        );
        let shadow = tch::no_grad(|| {
            parameters
                .iter()
                .map(|(name, value)| Ok((name.clone(), value.f_detach_copy()?)))
                .collect::<Result<Vec<_>>>()
        })?;
        Ok(Self {
            decay,
            updates: 0,
            shadow,
        })
    }

    /// Moves the shadow weights towards the current parameters. The decay warms
    /// up as `(1 + n) / (10 + n)`, so early averages do not lag the pretrained start.
    pub(super) fn update(&mut self, parameters: &[(String, Tensor)]) -> Result<()> {
        ensure!(
            parameters.len() == self.shadow.len(),
            "EMA parameter count changed"
        );
        let warm = (1.0 + self.updates as f64) / (10.0 + self.updates as f64);
        let decay = self.decay.min(warm);
        self.updates += 1;
        let shadow: Vec<_> = self
            .shadow
            .iter()
            .map(|(_, tensor)| tensor.as_ptr().cast_mut().cast())
            .collect();
        let current: Vec<_> = parameters
            .iter()
            .map(|(_, tensor)| tensor.as_ptr().cast_mut().cast())
            .collect();
        let mut error = [0; 512];
        let ok = unsafe {
            rvc_foreach_lerp(
                shadow.as_ptr(),
                current.as_ptr(),
                shadow.len(),
                1.0 - decay,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if ok == 0 {
            // The bridge always terminates its diagnostic and the buffer starts zeroed.
            let error = unsafe { std::ffi::CStr::from_ptr(error.as_ptr()) }.to_string_lossy();
            anyhow::bail!("updating EMA weights: {error}");
        }
        Ok(())
    }

    pub(super) fn weights(&self) -> Vec<(String, Tensor)> {
        self.shadow
            .iter()
            .map(|(name, tensor)| (name.clone(), tensor.shallow_clone()))
            .collect()
    }
}
