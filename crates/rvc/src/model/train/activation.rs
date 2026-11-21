use std::ffi::{CStr, c_char, c_void};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::{ptr, slice};

use anyhow::{Result, bail};
use tch::Tensor;

type Run = unsafe extern "C" fn(
    *mut c_void,
    *const *const c_void,
    usize,
    *mut c_char,
    usize,
) -> *mut c_void;
type Drop = unsafe extern "C" fn(*mut c_void);

unsafe extern "C" {
    fn rvc_checkpoint(
        context: *mut c_void,
        run: Run,
        drop: Drop,
        inputs: *const *const c_void,
        count: usize,
        error: *mut c_char,
        error_length: usize,
    ) -> *mut c_void;
}

type Forward = Box<dyn Fn(&[Tensor]) -> Result<Tensor> + Send + Sync>;

/// The replayed stage is deterministic and takes all differentiable tensors
/// explicitly. LibTorch owns the callback until the graph is released.
pub(in crate::model) fn checkpoint(
    inputs: &[&Tensor],
    forward: impl Fn(&[Tensor]) -> Result<Tensor> + Send + Sync + 'static,
) -> Result<Tensor> {
    let context = Box::into_raw(Box::new(Box::new(forward) as Forward)).cast();
    let inputs: Vec<_> = inputs.iter().map(|tensor| tensor.as_ptr().cast()).collect();
    let mut error = [0; 1024];
    // The inputs live for this call. LibTorch makes shallow copies and calls
    // drop_forward exactly once, including if it fails to construct the node.
    let output = unsafe {
        rvc_checkpoint(
            context,
            run_forward,
            drop_forward,
            inputs.as_ptr(),
            inputs.len(),
            error.as_mut_ptr(),
            error.len(),
        )
    };
    if output.is_null() {
        // The bridge always terminates its diagnostic and the buffer starts zeroed.
        let error = unsafe { CStr::from_ptr(error.as_ptr()) }.to_string_lossy();
        bail!("checkpointing decoder stage: {error}");
    }
    // An owned, heap-allocated LibTorch tensor is transferred across the bridge.
    Ok(unsafe { Tensor::from_ptr(output.cast()) })
}

unsafe extern "C" fn run_forward(
    context: *mut c_void,
    inputs: *const *const c_void,
    count: usize,
    error: *mut c_char,
    error_length: usize,
) -> *mut c_void {
    let result = catch_unwind(AssertUnwindSafe(|| {
        // The bridge retains both the boxed callback and each tensor for this call.
        let forward = unsafe { &*context.cast::<Forward>() };
        let tensors = unsafe { slice::from_raw_parts(inputs, count) }
            .iter()
            .map(|&tensor| unsafe { Tensor::clone_from_ptr(tensor.cast_mut().cast()) })
            .collect::<Vec<_>>();
        forward(&tensors)
    }));
    let message = match result {
        Ok(Ok(mut output)) => {
            let pointer = output.as_mut_ptr().cast();
            std::mem::forget(output);
            return pointer;
        }
        Ok(Err(error)) => format!("{error:#}"),
        Err(_) => "decoder stage panicked".into(),
    };
    let length = message.len().min(error_length - 1);
    // LibTorch provides the writeable, non-empty diagnostic buffer.
    unsafe {
        ptr::copy_nonoverlapping(message.as_ptr(), error.cast::<u8>(), length);
        *error.add(length) = 0;
    }
    ptr::null_mut()
}

unsafe extern "C" fn drop_forward(context: *mut c_void) {
    // This is the sole owner obtained from Box::into_raw in checkpoint().
    drop(unsafe { Box::from_raw(context.cast::<Forward>()) });
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use tch::{Device, Kind, Tensor};

    use super::*;

    #[test]
    fn replays_decoder_stage_and_preserves_gradients() -> Result<()> {
        let options = (Kind::Double, Device::Cpu);
        let x = Tensor::f_full([2, 3], 0.2, options)?.f_set_requires_grad(true)?;
        let w = Tensor::f_full([3, 4], -0.3, options)?.f_set_requires_grad(true)?;
        let constant = Tensor::f_full([4], 0.4, options)?;
        let forward = |x: &Tensor, w: &Tensor, constant: &Tensor| -> Result<Tensor> {
            Ok(x.f_matmul(w)?.f_add(constant)?.f_tanh()?)
        };
        let expected = forward(&x, &w, &constant)?;
        let calls = Arc::new(AtomicUsize::new(0));
        let saved = {
            let calls = Arc::clone(&calls);
            checkpoint(&[&x, &w, &constant], move |tensors| {
                calls.fetch_add(1, Ordering::Relaxed);
                forward(&tensors[0], &tensors[1], &tensors[2])
            })?
        };
        assert!(saved.f_equal(&expected)?);
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        let gradients = |out: Tensor| {
            Tensor::f_run_backward(
                &[out.f_square()?.f_sum(Kind::Double)?],
                &[&x, &w],
                false,
                false,
            )
        };
        for (actual, expected) in gradients(saved)?.iter().zip(gradients(expected)?) {
            assert!(actual.f_allclose(&expected, 0.0, 1e-12, false)?);
        }
        assert_eq!(calls.load(Ordering::Relaxed), 2);
        Ok(())
    }

    #[test]
    #[ignore = "requires a CUDA device with BF16 support"]
    fn replays_autocast_in_bf16_during_backward() -> Result<()> {
        use crate::model::train::precision::Precision;

        let device = Device::Cuda(0);
        let amp = Precision::choose(&[device], None, true)?;
        let x = Tensor::f_full([8, 16], 0.221, (Kind::Float, device))?.f_set_requires_grad(true)?;
        let w = Tensor::f_full([16, 8], -0.33, (Kind::Float, device))?.f_set_requires_grad(true)?;
        let expected = amp.run(|| x.f_matmul(&w))?;
        let replayed = amp.run(|| {
            checkpoint(&[&x, &w], |tensors| {
                let product = tensors[0].f_matmul(&tensors[1])?;
                anyhow::ensure!(
                    product.kind() == Kind::BFloat16,
                    "autocast did not use BF16"
                );
                Ok(product)
            })
        })?;
        assert!(expected.f_equal(&replayed)?);
        let gradients = |output: &Tensor| {
            Tensor::f_run_backward(&[output.f_sum(Kind::Float)?], &[&x, &w], false, false)
        };
        for (actual, expected) in gradients(&replayed)?.iter().zip(gradients(&expected)?) {
            assert!(actual.f_equal(&expected)?);
        }
        Ok(())
    }
}
