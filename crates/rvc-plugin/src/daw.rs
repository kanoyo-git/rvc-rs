use std::path::PathBuf;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicU8, AtomicU32, AtomicU64, Ordering},
};
use std::thread::{self, JoinHandle};
use std::time::Duration;
use std::{
    cell::UnsafeCell,
    ops::Add,
    panic::{AssertUnwindSafe, catch_unwind},
    ptr,
};

use anyhow::{Context, Result};
use rtrb::{Consumer, Producer, PushError, RingBuffer};

use rvc::realtime::{
    device::StreamFormat,
    worker::{AudioWorker, Capture, Playback},
};

pub const PARAMETER_COUNT: usize = 12;
pub const ENGINE: usize = 0;
pub const BLOCK_MS: usize = 6;
const F0_METHOD: usize = 9;
const SAMPLE_RATE_MIN: u32 = 8_000;
const SAMPLE_RATE_MAX: u32 = 384_000;
const MAX_DAW_LATENCY: usize = SAMPLE_RATE_MAX as usize * 2;

#[derive(Clone, Copy)]
pub struct Parameter {
    pub name: &'static str,
    pub unit: &'static str,
    min: f32,
    max: f32,
    step: f32,
    pub default: f32,
}

const fn parameter(
    name: &'static str,
    unit: &'static str,
    min: f32,
    max: f32,
    step: f32,
    default: f32,
) -> Parameter {
    Parameter {
        name,
        unit,
        min,
        max,
        step,
        default,
    }
}

pub const PARAMETERS: [Parameter; PARAMETER_COUNT] = [
    parameter("Engine", "", 0., 1., 1., 0.),
    parameter("Pitch", "st", -24., 24., 1., 12.),
    parameter("Formant", "st", -12., 12., 0.01, 0.),
    parameter("Index", "", 0., 1., 0.001, 0.),
    parameter("RMS Mix", "", 0., 1., 0.001, 0.5),
    parameter("Gate", "dB", -60., 0., 0.1, -60.),
    parameter("Block", "ms", 20., 1_000., 1., 130.),
    parameter("Crossfade", "ms", 10., 100., 1., 80.),
    parameter("Context", "ms", 500., 3_000., 1., 2_000.),
    parameter("F0 Method", "", 0., 2., 1., 0.),
    parameter("Mix", "%", 0., 100., 0.1, 100.),
    parameter("Output", "dB", -18., 12., 0.1, 0.),
];

impl Parameter {
    pub(crate) fn range(self) -> std::ops::RangeInclusive<f32> {
        self.min..=self.max
    }

    pub(crate) fn increment(self) -> f32 {
        self.step
    }

    pub fn normalize(self, value: f32) -> f32 {
        (value.clamp(self.min, self.max) - self.min) / (self.max - self.min)
    }

    pub(crate) fn denormalize(self, value: f32) -> f32 {
        let value = self.min + value.clamp(0.0, 1.0) * (self.max - self.min);
        let steps = ((value - self.min) / self.step).round();
        (steps * self.step + self.min).clamp(self.min, self.max)
    }

    pub(crate) fn step_count(self) -> i32 {
        if self.step >= 1.0 {
            ((self.max - self.min) / self.step) as i32
        } else {
            0
        }
    }

    pub fn display(self, index: usize, value: f32) -> String {
        if index == ENGINE {
            return if value >= 0.5 { "On" } else { "Off" }.into();
        }
        if index == F0_METHOD {
            return ["RMVPE", "FCPE", "PM"][value as usize].into();
        }
        let digits = if self.step >= 1.0 {
            0
        } else if self.step >= 0.1 {
            1
        } else if self.step >= 0.01 {
            2
        } else {
            3
        };
        format!("{value:.digits$}")
    }
}

#[derive(Clone, Copy, Debug)]
pub struct LiveControls {
    pub pitch: i32,
    pub formant: f32,
    pub index_rate: f32,
    pub rms_mix: f32,
    pub threshold_db: f32,
    pub f0_method: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ModelPaths {
    pub model: PathBuf,
    pub index: PathBuf,
    pub rvc_root: PathBuf,
    /// Kept in legacy preset state so saved DAW sessions can round-trip.
    pub legacy_python: PathBuf,
}

#[derive(Clone, Debug)]
pub struct PluginModelSettings {
    pub format: StreamFormat,
    pub crossfade_ms: u32,
    pub extra_ms: u32,
    pub paths: ModelPaths,
    pub initial_controls: LiveControls,
}

pub type Processor =
    Box<dyn FnMut(&[f32], &mut [f32], bool, LiveControls) -> Result<()> + Send + 'static>;
pub type ProcessorFactory =
    Box<dyn FnMut(PluginModelSettings) -> Result<Processor> + Send + 'static>;

#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum WorkerStatus {
    Off,
    Loading,
    Ready,
    Error,
}

pub struct PluginControls {
    values: [AtomicU32; PARAMETER_COUNT],
    generation: AtomicU64,
    rate: AtomicU32,
    active: AtomicBool,
    bypass: AtomicBool,
    stop: AtomicBool,
    channels: AtomicU8,
    paths: Mutex<ModelPaths>,
    status: AtomicU8,
    inference_us: AtomicU64,
    dropped_blocks: AtomicU64,
    worker_error: Mutex<Option<String>>,
}

impl PluginControls {
    fn new(paths: ModelPaths) -> Self {
        Self {
            values: PARAMETERS.map(|spec| AtomicU32::new(spec.default.to_bits())),
            generation: AtomicU64::new(1),
            rate: AtomicU32::new(48_000),
            active: AtomicBool::new(false),
            bypass: AtomicBool::new(false),
            stop: AtomicBool::new(false),
            channels: AtomicU8::new(0x22),
            paths: Mutex::new(paths),
            status: AtomicU8::new(WorkerStatus::Off as u8),
            inference_us: AtomicU64::new(0),
            dropped_blocks: AtomicU64::new(0),
            worker_error: Mutex::new(None),
        }
    }

    pub fn get(&self, index: usize) -> Option<f32> {
        self.values
            .get(index)
            .map(|value| f32::from_bits(value.load(Ordering::Relaxed)))
    }

    fn value(&self, index: usize) -> f32 {
        f32::from_bits(self.values[index].load(Ordering::Relaxed))
    }

    pub fn set_normalized(&self, index: usize, normalized: f32) {
        let Some(parameter) = PARAMETERS.get(index) else {
            return;
        };
        if !normalized.is_finite() {
            return;
        }
        let value = parameter.denormalize(normalized);
        let old = self.values[index].swap(value.to_bits(), Ordering::Relaxed);
        let retry_pitch = index == F0_METHOD && self.status() == WorkerStatus::Error;
        if old != value.to_bits() && (matches!(index, ENGINE | BLOCK_MS..=8) || retry_pitch) {
            self.changed();
        }
    }

    pub fn bypassed(&self) -> bool {
        self.bypass.load(Ordering::Relaxed)
    }

    pub fn sample_rate(&self) -> u32 {
        self.rate.load(Ordering::Relaxed)
    }

    pub fn set_sample_rate(&self, rate: u32) {
        if (SAMPLE_RATE_MIN..=SAMPLE_RATE_MAX).contains(&rate)
            && self.rate.swap(rate, Ordering::Relaxed) != rate
        {
            self.changed();
        }
    }

    pub fn set_active(&self, active: bool) {
        if self.active.swap(active, Ordering::Relaxed) != active {
            self.changed();
        }
    }

    pub fn set_bypass(&self, bypass: bool) {
        if self.bypass.swap(bypass, Ordering::Relaxed) != bypass {
            self.changed();
        }
    }

    pub fn set_layout(&self, inputs: usize, outputs: usize) -> bool {
        if !matches!((inputs, outputs), (1, 1) | (1, 2) | (2, 2)) {
            return false;
        }
        self.channels
            .store((inputs as u8) << 4 | outputs as u8, Ordering::Relaxed);
        true
    }

    pub fn layout(&self) -> (usize, usize) {
        let layout = self.channels.load(Ordering::Relaxed);
        (usize::from(layout >> 4), usize::from(layout & 0x0f))
    }

    pub fn latency_frames(&self) -> usize {
        let rate = self.sample_rate();
        let grid = (rate / 100) as f64;
        let blocks = (f64::from(self.value(BLOCK_MS)) * f64::from(rate) / (1_000.0 * grid)).round();
        (blocks * grid) as usize * 2
    }

    pub fn snapshot(&self) -> (ModelPaths, [f32; PARAMETER_COUNT]) {
        let paths = self
            .paths
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .clone();
        (paths, std::array::from_fn(|index| self.value(index)))
    }

    pub(crate) fn set_model_paths(&self, paths: ModelPaths) -> bool {
        let mut current = self.paths.lock().unwrap_or_else(|err| err.into_inner());
        if *current == paths {
            return false;
        }
        *current = paths;
        let request_was_enabled =
            self.values[ENGINE].swap(0.0f32.to_bits(), Ordering::Relaxed) != 0.0f32.to_bits();
        self.changed();
        request_was_enabled
    }

    pub(crate) fn performance(&self) -> (f32, u64) {
        (
            self.inference_us.load(Ordering::Relaxed) as f32 / 1000.0,
            self.dropped_blocks.load(Ordering::Relaxed),
        )
    }

    pub fn restore(&self, paths: ModelPaths, values: [f32; PARAMETER_COUNT]) {
        *self.paths.lock().unwrap_or_else(|err| err.into_inner()) = paths;
        for (index, value) in values.into_iter().enumerate() {
            let spec = PARAMETERS[index];
            let finite = if value.is_finite() {
                value
            } else {
                spec.default
            };
            let quantized = spec.denormalize(spec.normalize(finite));
            self.values[index].store(quantized.to_bits(), Ordering::Relaxed);
        }
        self.changed();
    }

    pub fn last_error(&self) -> Option<String> {
        self.worker_error
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .clone()
    }

    pub fn status(&self) -> WorkerStatus {
        match self.status.load(Ordering::Acquire) {
            1 => WorkerStatus::Loading,
            2 => WorkerStatus::Ready,
            3 => WorkerStatus::Error,
            _ => WorkerStatus::Off,
        }
    }

    pub(crate) fn engine_display(&self, requested: bool) -> &'static str {
        if !requested {
            return "Off";
        }
        match self.status() {
            WorkerStatus::Off => "On",
            WorkerStatus::Loading => "Loading",
            WorkerStatus::Ready => "Ready",
            WorkerStatus::Error => "Error",
        }
    }

    fn changed(&self) {
        self.generation.fetch_add(1, Ordering::Release);
    }

    fn requested(&self) -> bool {
        self.active.load(Ordering::Relaxed)
            && !self.bypass.load(Ordering::Relaxed)
            && self.value(ENGINE) >= 0.5
    }

    fn live(&self) -> LiveControls {
        LiveControls {
            pitch: self.value(1) as i32,
            formant: self.value(2),
            index_rate: self.value(3),
            rms_mix: self.value(4),
            threshold_db: self.value(5),
            f0_method: self.value(F0_METHOD) as usize,
        }
    }

    fn factory_settings(&self) -> Result<PluginModelSettings> {
        Ok(PluginModelSettings {
            format: StreamFormat::new(self.sample_rate(), self.value(BLOCK_MS) as u32)?,
            crossfade_ms: self.value(7) as u32,
            extra_ms: self.value(8) as u32,
            paths: self
                .paths
                .lock()
                .unwrap_or_else(|err| err.into_inner())
                .clone(),
            initial_controls: self.live(),
        })
    }

    fn report_error(&self, error: Option<anyhow::Error>) {
        if error.is_some() {
            self.status
                .store(WorkerStatus::Error as u8, Ordering::Release);
        }
        if let Some(error) = &error {
            eprintln!("RVC Realtime: {error:#}");
        }
        *self
            .worker_error
            .lock()
            .unwrap_or_else(|err| err.into_inner()) = error.map(|error| format!("{error:#}"));
    }
}

struct Endpoints {
    generation: u64,
    capture: Capture,
    playback: Playback,
}

pub struct PluginWorker {
    controls: Arc<PluginControls>,
    thread: Option<JoinHandle<()>>,
}

impl Drop for PluginWorker {
    fn drop(&mut self) {
        self.controls.stop.store(true, Ordering::Release);
        if let Some(worker) = self.thread.take() {
            worker.thread().unpark();
            let _ = worker.join();
        }
    }
}

/// Audio callback state shared by VST frontends; all model setup and endpoint
/// destruction run on the manager thread.
pub struct PluginAudio {
    controls: Arc<PluginControls>,
    incoming: Consumer<Endpoints>,
    retired: Producer<Endpoints>,
    current: Option<Endpoints>,
    dry: Vec<[f32; 2]>,
    cursor: usize,
    history_length: usize,
    blend: f32,
}

pub fn start(
    paths: ModelPaths,
    factory: ProcessorFactory,
) -> Result<(Arc<PluginControls>, PluginAudio, PluginWorker)> {
    let controls = Arc::new(PluginControls::new(paths));
    let (publish, incoming) = RingBuffer::new(2);
    let (retired, discard) = RingBuffer::new(2);
    let status = Arc::clone(&controls);
    let thread = thread::Builder::new()
        .name("rvc-plugin-control".into())
        .spawn(move || manage(status, factory, publish, discard))
        .context("start DAW control worker")?;
    let audio = PluginAudio {
        controls: Arc::clone(&controls),
        incoming,
        retired,
        current: None,
        dry: vec![[0.0; 2]; MAX_DAW_LATENCY + 1],
        cursor: 0,
        history_length: 0,
        blend: 0.0,
    };
    let worker = PluginWorker {
        controls: Arc::clone(&controls),
        thread: Some(thread),
    };
    Ok((controls, audio, worker))
}

fn manage(
    controls: Arc<PluginControls>,
    mut factory: ProcessorFactory,
    mut publish: Producer<Endpoints>,
    mut discard: Consumer<Endpoints>,
) {
    let mut seen = 0;
    let mut worker = None::<(AudioWorker, usize)>;
    let mut pending = None;
    while !controls.stop.load(Ordering::Acquire) {
        #[cfg(all(target_os = "windows", not(test)))]
        let mut started_paths = None;
        while let Ok(retired) = discard.pop() {
            drop(retired);
        }
        let requested = controls.generation.load(Ordering::Acquire);
        if requested != seen {
            seen = requested;
            drop(worker.take());
            pending = None;
            controls.report_error(None);
            controls.inference_us.store(0, Ordering::Relaxed);
            controls.dropped_blocks.store(0, Ordering::Relaxed);
            if controls.requested() {
                controls
                    .status
                    .store(WorkerStatus::Loading as u8, Ordering::Release);
                let result = controls.factory_settings().and_then(|config| {
                    let frames = config.format.block_frames;
                    #[cfg(all(target_os = "windows", not(test)))]
                    let model_paths = config.paths.clone();
                    let mut process = factory(config)?;
                    let params = Arc::clone(&controls);
                    let (worker, capture, playback) =
                        AudioWorker::start_ready(frames, move |input, output, gap| {
                            process(input, output, gap, params.live())
                        })?;
                    #[cfg(all(target_os = "windows", not(test)))]
                    {
                        started_paths = Some(model_paths);
                    }
                    Ok((worker, capture, playback, frames))
                });
                match result {
                    Ok((next_worker, capture, playback, frames)) => {
                        controls
                            .status
                            .store(WorkerStatus::Ready as u8, Ordering::Release);
                        worker = Some((next_worker, frames));
                        pending = Some(Endpoints {
                            generation: seen,
                            capture,
                            playback,
                        });
                    }
                    Err(error) => controls.report_error(Some(error)),
                }
            } else {
                controls
                    .status
                    .store(WorkerStatus::Off as u8, Ordering::Release);
            }
        }

        if let Some(staged) = pending.take()
            && let Err(PushError::Full(staged)) = publish.push(staged)
        {
            pending = Some(staged);
        }

        #[cfg(all(target_os = "windows", not(test)))]
        if let Some(paths) = started_paths
            && controls.generation.load(Ordering::Acquire) == seen
            && !controls.stop.load(Ordering::Acquire)
            && controls.requested()
            && let Err(error) = crate::windows_paths::save(&paths)
        {
            use std::io::Write as _;

            let _ = writeln!(
                std::io::stderr(),
                "RVC Realtime: could not save model paths: {error:#}"
            );
        }

        if let Some((audio_worker, block)) = worker.as_ref() {
            if let Err(error) = audio_worker.check() {
                drop(worker.take());
                controls.report_error(Some(error));
            } else {
                let stats = audio_worker.stats();
                controls
                    .inference_us
                    .store(stats.last_inference_us, Ordering::Relaxed);
                let dropped = stats.dropped_input_frames.div_ceil(*block as u64)
                    + stats.dropped_output_blocks;
                controls.dropped_blocks.store(dropped, Ordering::Relaxed);
            }
        }
        thread::park_timeout(Duration::from_millis(5));
    }
}

impl PluginAudio {
    /// Called by one host audio thread at a time. Returning false indicates
    /// the caller should copy its dry input at its native sample precision.
    pub fn render(
        &mut self,
        frames: usize,
        read: impl Fn(usize) -> [f32; 2],
        mut write: impl FnMut(usize, [f32; 2]),
    ) -> bool {
        self.take_latest();
        let generation = self.controls.generation.load(Ordering::Acquire);
        let Some(active) = self
            .current
            .as_mut()
            .filter(|active| active.generation == generation)
        else {
            self.blend = 0.0;
            self.history_length = 0;
            return false;
        };
        if !self.controls.requested() || self.controls.status() != WorkerStatus::Ready {
            self.blend = 0.0;
            self.history_length = 0;
            return false;
        }
        let latency = self.controls.latency_frames().min(self.dry.len() - 1);
        let mix = self.controls.value(10) / 100.0;
        let gain = 10.0_f32.powf(self.controls.value(11) / 20.0);
        self.blend = (self.blend + frames as f32 / 2048.0).min(1.0);
        active.capture.write((0..frames).map(|frame| {
            let [left, right] = read(frame);
            0.5 * (left + right)
        }));
        let mut frame = 0;
        active.playback.read(frames, |wet| {
            let input = read(frame);
            self.dry[self.cursor] = input;
            let earlier = (self.cursor + self.dry.len() - latency) % self.dry.len();
            let dry = if self.history_length >= latency {
                self.dry[earlier]
            } else {
                [0.0; 2]
            };
            self.cursor = (self.cursor + 1) % self.dry.len();
            self.history_length = (self.history_length + 1).min(self.dry.len());
            let wet = wet * gain * mix;
            write(
                frame,
                std::array::from_fn(|channel| {
                    input[channel] * (1.0 - self.blend)
                        + (dry[channel] * (1.0 - mix) + wet) * self.blend
                }),
            );
            frame += 1;
        });
        true
    }

    fn take_latest(&mut self) {
        for _ in 0..self.incoming.slots() {
            if let Some(previous) = self.current.take()
                && let Err(PushError::Full(previous)) = self.retired.push(previous)
            {
                self.current = Some(previous);
                return;
            }
            if let Ok(incoming) = self.incoming.pop() {
                self.current = Some(incoming);
                self.history_length = 0;
                self.blend = 0.0;
            }
        }
    }
}

pub(crate) struct AudioSlot {
    busy: AtomicBool,
    audio: UnsafeCell<PluginAudio>,
}

// SAFETY: the only mutable access to the audio state holds the non-blocking
// exclusive claim from `with_audio`. Control callbacks never access it.
unsafe impl Sync for AudioSlot {}

impl AudioSlot {
    pub(crate) fn new(audio: PluginAudio) -> Self {
        Self {
            busy: AtomicBool::new(false),
            audio: UnsafeCell::new(audio),
        }
    }

    fn with_audio(&self, render: impl FnOnce(&mut PluginAudio) -> bool) -> bool {
        if self
            .busy
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            return false;
        }
        struct Claim<'a>(&'a AtomicBool);

        impl Drop for Claim<'_> {
            fn drop(&mut self) {
                self.0.store(false, Ordering::Release);
            }
        }

        let _claim = Claim(&self.busy);
        // SAFETY: the successful atomic claim is exclusive for the lifetime
        // of this reference, including during unwinding.
        render(unsafe { &mut *self.audio.get() })
    }
}

pub(crate) trait NativeSample: Copy + Default + Add<Output = Self> {
    fn to_f32(self) -> f32;
    fn from_f32(sample: f32) -> Self;
}

impl NativeSample for f32 {
    fn to_f32(self) -> f32 {
        self
    }
    fn from_f32(sample: f32) -> Self {
        sample
    }
}

impl NativeSample for f64 {
    fn to_f32(self) -> f32 {
        self as f32
    }
    fn from_f32(sample: f32) -> Self {
        f64::from(sample)
    }
}

/// # Safety
/// Channel-pointer arrays have the negotiated counts. Each non-null channel
/// contains at least `frames` accessible samples; outputs are writable.
pub(crate) unsafe fn render_channels<T: NativeSample>(
    audio: &AudioSlot,
    layout: (usize, usize),
    input: *const *const T,
    output: *mut *mut T,
    frames: usize,
    accumulate: bool,
) {
    if frames == 0 || output.is_null() {
        return;
    }
    let (inputs, outputs) = layout;
    // SAFETY: the host owns channel-pointer arrays of the negotiated sizes.
    // A host can disconnect a channel by passing a null data pointer.
    let out = unsafe {
        [
            *output,
            if outputs == 2 {
                *output.add(1)
            } else {
                ptr::null_mut()
            },
        ]
    };
    let [mut left, mut right] = [ptr::null(), ptr::null()];
    if !input.is_null() {
        // SAFETY: input is a live audio pointer array with at least one slot.
        left = unsafe { *input };
        right = if inputs == 1 {
            left
        } else {
            unsafe { *input.add(1) }
        };
    }
    let read = |frame: usize| -> [T; 2] {
        [left, right].map(|channel| {
            if channel.is_null() {
                T::default()
            } else {
                // SAFETY: non-null channels contain all of this callback's
                // sample frames. Read before writing supports in-place I/O.
                unsafe { channel.add(frame).read() }
            }
        })
    };
    let write = |frame: usize, data: [T; 2]| {
        for (channel, sample) in out.into_iter().zip(data) {
            if channel.is_null() {
                continue;
            }
            // SAFETY: a connected output contains all sample frames. The host
            // keeps its audio buffers valid until this callback returns.
            unsafe {
                let destination = channel.add(frame);
                destination.write(if accumulate {
                    destination.read() + sample
                } else {
                    sample
                });
            }
        }
    };

    let converted = (!left.is_null() || !right.is_null())
        && catch_unwind(AssertUnwindSafe(|| {
            audio.with_audio(|audio| {
                audio.render(
                    frames,
                    |frame| read(frame).map(T::to_f32),
                    |frame, stereo| write(frame, stereo.map(T::from_f32)),
                )
            })
        }))
        .unwrap_or(false);
    if !converted {
        for frame in 0..frames {
            write(frame, read(frame));
        }
    }
}

pub(crate) const MAX_STATE_BYTES: usize = 128 * 1024;

pub(crate) fn model_state(
    paths: ModelPaths,
    parameters: [f32; PARAMETER_COUNT],
) -> Option<Vec<u8>> {
    let mut out = Vec::new();
    for path in [
        &paths.model,
        &paths.index,
        &paths.rvc_root,
        &paths.legacy_python,
    ] {
        let text = path.to_str()?.as_bytes();
        if out.len().checked_add(text.len())? > MAX_STATE_BYTES {
            return None;
        }
        out.extend(i32::try_from(text.len()).ok()?.to_le_bytes());
        out.extend(text);
    }
    for param in parameters {
        out.extend(f64::from(param).to_le_bytes());
    }
    (out.len() <= MAX_STATE_BYTES).then_some(out)
}

pub(crate) fn state_path(data: &mut &[u8]) -> Option<PathBuf> {
    let size = i32::from_le_bytes(data.split_off(..4)?.try_into().ok()?);
    let length = usize::try_from(size).ok()?;
    let value = data.split_off(..length)?;
    if value.contains(&0) {
        return None;
    }
    Some(PathBuf::from(std::str::from_utf8(value).ok()?))
}

pub(crate) fn parse_model_state(mut data: &[u8]) -> Option<(ModelPaths, [f32; PARAMETER_COUNT])> {
    let paths = ModelPaths {
        model: state_path(&mut data)?,
        index: state_path(&mut data)?,
        rvc_root: state_path(&mut data)?,
        legacy_python: state_path(&mut data)?,
    };
    let mut parameters = [0.0; PARAMETER_COUNT];
    for param in &mut parameters {
        *param = f64::from_le_bytes(data.split_off(..8)?.try_into().ok()?) as f32;
        if !param.is_finite() {
            return None;
        }
    }
    data.is_empty().then_some((paths, parameters))
}
