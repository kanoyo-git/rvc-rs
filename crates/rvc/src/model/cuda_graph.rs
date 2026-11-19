#[cfg(not(feature = "cuda-graph"))]
use anyhow::Result;
#[cfg(not(feature = "cuda-graph"))]
use tch::{Device, Tensor};

#[cfg(not(feature = "cuda-graph"))]
#[derive(Default)]
pub(super) struct Cache;

#[cfg(not(feature = "cuda-graph"))]
impl Cache {
    pub(super) fn run(
        &mut self,
        _key: [i64; 4],
        input: &[&Tensor],
        _shared: Option<&Self>,
        inference: impl Fn(&[&Tensor]) -> Result<Tensor>,
    ) -> Result<Tensor> {
        inference(input)
    }
}

#[cfg(not(feature = "cuda-graph"))]
pub(super) fn prepare<T>(_device: Device, work: impl FnOnce() -> Result<T>) -> Result<T> {
    work()
}

#[cfg(feature = "cuda-graph")]
pub(super) use enabled::{Cache, prepare, trim};

#[cfg(feature = "cuda-graph")]
mod enabled {
    use std::ffi::{CStr, c_char, c_void};
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::ptr::{self, NonNull};

    use anyhow::{Result, anyhow};
    use tch::{Device, Tensor};

    type Operation = unsafe extern "C" fn(*mut c_void, *mut c_char, usize) -> i32;

    unsafe extern "C" {
        fn rvc_cuda_graph_create(
            device: i32,
            shared: *const c_void,
            error: *mut c_char,
            capacity: usize,
        ) -> *mut c_void;
        fn rvc_cuda_graph_enter(graph: *mut c_void, error: *mut c_char, capacity: usize) -> i32;
        fn rvc_cuda_graph_leave(graph: *mut c_void, error: *mut c_char, capacity: usize) -> i32;
        fn rvc_cuda_graph_begin(graph: *mut c_void, error: *mut c_char, capacity: usize) -> i32;
        fn rvc_cuda_graph_end(graph: *mut c_void, error: *mut c_char, capacity: usize) -> i32;
        fn rvc_cuda_graph_replay(graph: *mut c_void, error: *mut c_char, capacity: usize) -> i32;
        fn rvc_cuda_graph_delete(graph: *mut c_void, error: *mut c_char, capacity: usize) -> i32;
        fn rvc_cuda_graph_trim(error: *mut c_char, capacity: usize) -> i32;
    }

    pub(in crate::model) fn trim() {
        let mut error: [c_char; 768] = [0; 768];
        // No graph is capturing when the worker finishes its warmup block.
        let status = unsafe { rvc_cuda_graph_trim(error.as_mut_ptr(), error.len()) };
        if status != 0 {
            eprintln!("{:#}", graph_error(&error));
        }
    }

    pub(in crate::model) fn prepare<T>(
        device: Device,
        work: impl FnOnce() -> Result<T>,
    ) -> Result<T> {
        let Device::Cuda(device) = device else {
            return work();
        };
        // Queue background CUDA work on its own stream. Before publishing the
        // result, order the default stream after everything queued here.
        let scope = match Graph::new(device, None).and_then(|scope| {
            scope.call(rvc_cuda_graph_enter)?;
            Ok(scope)
        }) {
            Ok(scope) => scope,
            Err(error) => {
                eprintln!("{error:#}. Preparing the estimator on the default CUDA stream.");
                return work();
            }
        };
        let result = work();
        let left = scope.call(rvc_cuda_graph_leave);
        let value = result?;
        left?;
        Ok(value)
    }

    struct Graph(NonNull<c_void>);

    // The arena is owned by a model and only operated by the model worker.
    // Each operation installs and restores the calling thread's CUDA stream.
    unsafe impl Send for Graph {}

    impl Graph {
        fn new(device: usize, shared: Option<&Self>) -> Result<Self> {
            let mut error: [c_char; 768] = [0; 768];
            // The bridge only writes within the supplied error buffer.
            let graph = unsafe {
                rvc_cuda_graph_create(
                    device.try_into()?,
                    shared.map_or(ptr::null(), |root| root.0.as_ptr()),
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            NonNull::new(graph)
                .map(Self)
                .ok_or_else(|| graph_error(&error))
        }

        fn call(&self, operation: Operation) -> Result<()> {
            let mut error: [c_char; 768] = [0; 768];
            // These functions borrow an arena that is live until Graph is dropped.
            let status = unsafe { operation(self.0.as_ptr(), error.as_mut_ptr(), error.len()) };
            if status == 0 {
                Ok(())
            } else {
                Err(graph_error(&error))
            }
        }
    }

    impl Drop for Graph {
        fn drop(&mut self) {
            let _ = self.call(rvc_cuda_graph_delete);
        }
    }

    fn graph_error(buffer: &[c_char; 768]) -> anyhow::Error {
        // The bridge NUL-terminates every error string.
        let message = unsafe { CStr::from_ptr(buffer.as_ptr()) }.to_string_lossy();
        anyhow!("CUDA Graph: {message}")
    }

    struct Capture {
        graph: Graph,
        key: [i64; 4],
        input: Vec<Tensor>,
        output: Tensor,
    }

    #[derive(Default)]
    pub(in crate::model) struct Cache {
        capture: Option<Capture>,
        disabled: bool,
    }

    impl Cache {
        pub(in crate::model) fn clear(&mut self) {
            self.capture = None;
        }

        pub(in crate::model) fn run(
            &mut self,
            key: [i64; 4],
            input: &[&Tensor],
            shared: Option<&Cache>,
            inference: impl Fn(&[&Tensor]) -> Result<Tensor>,
        ) -> Result<Tensor> {
            let Some(Device::Cuda(device)) = input.first().map(|tensor| tensor.device()) else {
                return inference(input);
            };
            if self.disabled
                || std::env::var_os("RVC_CUDA_GRAPH").is_some_and(|value| value == "0")
                || std::env::var_os("RVC_PROFILE").is_some_and(|value| value == "1")
            {
                return inference(input);
            }
            let pool = shared
                .and_then(|root| root.capture.as_ref())
                .map(|root| &root.graph);
            match self.captured(device, key, input, pool, &inference) {
                Ok(output) => Ok(output),
                Err(error) => {
                    self.capture = None;
                    self.disabled = true;
                    eprintln!("{error:#}. Continuing with eager CUDA for this model.");
                    inference(input)
                }
            }
        }

        fn captured(
            &mut self,
            device: usize,
            key: [i64; 4],
            input: &[&Tensor],
            pool: Option<&Graph>,
            inference: &impl Fn(&[&Tensor]) -> Result<Tensor>,
        ) -> Result<Tensor> {
            if let Some(capture) = &mut self.capture
                && capture.key == key
                && capture.input.len() == input.len()
                && capture.input.iter().zip(input).all(|(left, right)| {
                    left.size() == right.size()
                        && left.kind() == right.kind()
                        && left.device() == right.device()
                })
            {
                for (storage, source) in capture.input.iter_mut().zip(input) {
                    storage.f_copy_(source)?;
                }
                capture.graph.call(rvc_cuda_graph_replay)?;
                return Ok(capture.output.shallow_clone());
            }
            self.capture = None;
            let mut storage = input
                .iter()
                .map(|tensor| tensor.f_empty_like())
                .collect::<tch::Result<Vec<_>>>()?;
            for (tensor, source) in storage.iter_mut().zip(input) {
                tensor.f_copy_(source)?;
            }
            let args: Vec<_> = storage.iter().collect();
            let graph = Graph::new(device, pool)?;
            graph.call(rvc_cuda_graph_enter)?;
            let warmed = run_guarded(inference, &args);
            let left = graph.call(rvc_cuda_graph_leave);
            let _ = warmed?;
            left?;
            trim();

            graph.call(rvc_cuda_graph_begin)?;
            let output = run_guarded(inference, &args);
            let ended = graph.call(rvc_cuda_graph_end);
            let output = output?;
            ended?;
            graph.call(rvc_cuda_graph_replay)?;
            let result = output.shallow_clone();
            self.capture = Some(Capture {
                graph,
                key,
                input: storage,
                output,
            });
            Ok(result)
        }
    }

    fn run_guarded(
        inference: &impl Fn(&[&Tensor]) -> Result<Tensor>,
        input: &[&Tensor],
    ) -> Result<Tensor> {
        match catch_unwind(AssertUnwindSafe(|| inference(input))) {
            Ok(result) => result,
            Err(error) => {
                let detail = error
                    .downcast_ref::<String>()
                    .map(String::as_str)
                    .or_else(|| error.downcast_ref::<&'static str>().copied())
                    .unwrap_or("inference failed during capture");
                Err(anyhow!("{detail}"))
            }
        }
    }
}
