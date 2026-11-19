use std::sync::{
    Arc, OnceLock,
    atomic::{AtomicBool, AtomicU64, Ordering},
};

use anyhow::{Context, Result, anyhow, bail, ensure};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{
    BufferSize, Device, DeviceId, FromSample, Host, HostId, I24, SampleFormat, SizedSample,
    SupportedBufferSize, SupportedStreamConfig, SupportedStreamConfigRange, U24,
};
use serde::Serialize;

use super::worker::{AudioWorker, Capture, Playback, WorkerStats};

#[cfg(target_os = "windows")]
mod wasapi;
#[cfg(any(target_os = "windows", test))]
mod wasapi_pcm;

#[derive(Clone, Debug)]
pub struct DeviceOptions {
    pub host: Option<String>,
    pub input: Option<String>,
    pub output: Option<String>,
    /// A rate required by both endpoints. When omitted, negotiate a common rate.
    pub sample_rate: Option<u32>,
    pub block_ms: u32,
    pub wasapi_exclusive: bool,
}

impl Default for DeviceOptions {
    fn default() -> Self {
        Self {
            host: None,
            input: None,
            output: None,
            sample_rate: None,
            block_ms: 250,
            wasapi_exclusive: false,
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct StreamFormat {
    pub sample_rate: u32,
    pub block_frames: usize,
}

impl StreamFormat {
    pub fn new(sample_rate: u32, block_ms: u32) -> Result<Self> {
        ensure!(
            (8_000..=384_000).contains(&sample_rate),
            "audio sample rate {sample_rate} Hz is out of range"
        );
        ensure!(
            (10..=2_000).contains(&block_ms),
            "audio block duration must be between 10 and 2000 ms"
        );
        let period = sample_rate / 100;
        let block_periods = (f64::from(block_ms) * f64::from(sample_rate)
            / (1000.0 * f64::from(period)))
        .round_ties_even() as usize;
        Ok(Self {
            sample_rate,
            block_frames: block_periods * period as usize,
        })
    }
}

#[derive(Debug, Serialize)]
pub struct AudioDevice {
    pub id: String,
    pub name: String,
    pub is_default: bool,
}

#[derive(Debug, Serialize)]
pub struct DeviceCatalog {
    pub host: String,
    pub available_hosts: Vec<String>,
    pub inputs: Vec<AudioDevice>,
    pub outputs: Vec<AudioDevice>,
}

pub fn devices(host_name: Option<&str>) -> Result<DeviceCatalog> {
    let host = open_host(host_name, None, None)?;
    let default_input = host.default_input_device().and_then(|d| d.id().ok());
    let default_output = host.default_output_device().and_then(|d| d.id().ok());
    let inputs = host
        .input_devices()
        .context("enumerate audio inputs")?
        .filter_map(|d| device_info(d, default_input.as_ref()))
        .collect();
    let outputs = host
        .output_devices()
        .context("enumerate audio outputs")?
        .filter_map(|d| device_info(d, default_output.as_ref()))
        .collect();
    Ok(DeviceCatalog {
        host: host.id().to_string(),
        available_hosts: cpal::available_hosts()
            .into_iter()
            .map(|id| id.to_string())
            .collect(),
        inputs,
        outputs,
    })
}

fn device_info(device: Device, default: Option<&DeviceId>) -> Option<AudioDevice> {
    let id = device.id().ok()?;
    let name = device
        .description()
        .map(|description| description.name().to_string())
        .unwrap_or_else(|_| id.id().to_owned());
    Some(AudioDevice {
        is_default: default == Some(&id),
        name,
        id: id.to_string(),
    })
}

fn open_host(
    name: Option<&str>,
    input: Option<&DeviceId>,
    output: Option<&DeviceId>,
) -> Result<Host> {
    let selected = name
        .map(str::parse::<HostId>)
        .transpose()
        .context("invalid audio host name")?
        .or_else(|| input.or(output).map(DeviceId::host));
    let host = match selected {
        Some(id) => cpal::host_from_id(id).with_context(|| format!("open audio host {id}"))?,
        None => cpal::default_host(),
    };
    for id in [input, output].into_iter().flatten() {
        ensure!(
            id.host() == host.id(),
            "audio device {id} doesn't belong to host {}",
            host.id()
        );
    }
    Ok(host)
}

fn choose_device(host: &Host, id: Option<&DeviceId>, capture: bool) -> Result<Device> {
    let direction = if capture { "input" } else { "output" };
    match id {
        Some(id) => host
            .device_by_id(id)
            .ok_or_else(|| anyhow!("audio {direction} device {id} is unavailable")),
        None => {
            let default = if capture {
                host.default_input_device()
            } else {
                host.default_output_device()
            };
            default.ok_or_else(|| anyhow!("no default audio {direction}"))
        }
    }
}

#[derive(Default)]
struct StreamStatus {
    xruns: AtomicU64,
    error_claimed: AtomicBool,
    error: OnceLock<anyhow::Error>,
}

impl StreamStatus {
    fn capture_error(&self, error: cpal::Error) {
        if error.kind() == cpal::ErrorKind::Xrun {
            self.xrun();
        } else {
            self.record_error(error.into());
        }
    }

    fn xrun(&self) {
        self.xruns.fetch_add(1, Ordering::Relaxed);
    }

    fn record_error(&self, error: anyhow::Error) {
        if !self.error_claimed.swap(true, Ordering::Relaxed) {
            let _ = self.error.set(error);
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct RealtimeStats {
    pub worker: WorkerStats,
    pub device_xruns: u64,
}

/// Keep this handle alive while capturing. Users of a GUI/CLI can poll
/// `check()` and display `stats()` on the control thread.
pub struct RealtimeStream {
    devices: Streams,
    worker: AudioWorker,
    status: Arc<StreamStatus>,
    format: StreamFormat,
}

enum Streams {
    Shared {
        _streams: Box<SharedStreams>,
    },
    #[cfg(target_os = "windows")]
    Exclusive {
        _stream: wasapi::ExclusiveStream,
    },
}

struct SharedStreams {
    _capture: cpal::Stream,
    _playback: cpal::Stream,
}

impl RealtimeStream {
    /// The initializer is called once after device negotiation. The returned
    /// processor runs exclusively on the worker thread.
    pub fn start<F>(
        options: &DeviceOptions,
        make_processor: impl FnOnce(StreamFormat) -> Result<F>,
    ) -> Result<Self>
    where
        F: FnMut(&[f32], &mut [f32], bool) -> Result<()> + Send + 'static,
    {
        let input_id = options
            .input
            .as_deref()
            .map(str::parse::<DeviceId>)
            .transpose()
            .context("invalid audio input ID")?;
        let output_id = options
            .output
            .as_deref()
            .map(str::parse::<DeviceId>)
            .transpose()
            .context("invalid audio output ID")?;
        let host = open_host(
            options.host.as_deref(),
            input_id.as_ref(),
            output_id.as_ref(),
        )?;
        #[cfg(target_os = "windows")]
        if options.wasapi_exclusive && host.id() == HostId::Wasapi {
            return wasapi::start(
                options,
                input_id.as_ref().map(DeviceId::id),
                output_id.as_ref().map(DeviceId::id),
                make_processor,
            );
        }
        let input = choose_device(&host, input_id.as_ref(), true)?;
        let output = choose_device(&host, output_id.as_ref(), false)?;
        let (input_config, output_config) = negotiate(&input, &output, options.sample_rate)?;
        let format = StreamFormat::new(input_config.sample_rate(), options.block_ms)?;
        // PulseAudio's default playback request can span several model blocks.
        // Keep transfers shorter than the queue shared with the model worker.
        #[cfg(target_os = "linux")]
        let period = (host.id() == HostId::PulseAudio)
            .then_some((format.sample_rate / 50).min(format.block_frames as u32));
        #[cfg(not(target_os = "linux"))]
        let period = None;

        let (worker, capture_endpoint, playback_endpoint) =
            AudioWorker::start_ready(format.block_frames, make_processor(format)?)?;
        let status = Arc::new(StreamStatus::default());
        let capture = open_stream(
            &input,
            &input_config,
            period,
            Endpoint::Capture(capture_endpoint),
            Arc::clone(&status),
        )
        .context("open audio input stream")?;
        let playback = open_stream(
            &output,
            &output_config,
            period,
            Endpoint::Playback(playback_endpoint),
            Arc::clone(&status),
        )
        .context("open audio output stream")?;

        playback.play().context("start audio output stream")?;
        capture.play().context("start audio input stream")?;
        Ok(Self {
            devices: Streams::Shared {
                _streams: Box::new(SharedStreams {
                    _capture: capture,
                    _playback: playback,
                }),
            },
            worker,
            status,
            format,
        })
    }

    pub fn format(&self) -> StreamFormat {
        self.format
    }

    pub fn stats(&self) -> RealtimeStats {
        RealtimeStats {
            worker: self.worker.stats(),
            device_xruns: self.status.xruns.load(Ordering::Relaxed),
        }
    }

    pub fn check(&self) -> Result<()> {
        if let Some(error) = self.status.error.get() {
            bail!("audio device failed: {error:#}");
        }
        self.worker.check()
    }

    pub fn stop(self) -> Result<()> {
        let status = self.check();
        drop(self.devices);
        self.worker.stop()?;
        status
    }
}

fn negotiate(
    input: &Device,
    output: &Device,
    requested_rate: Option<u32>,
) -> Result<(SupportedStreamConfig, SupportedStreamConfig)> {
    let input_default = input.default_input_config().context("audio input config")?;
    let output_default = output
        .default_output_config()
        .context("audio output config")?;
    let inputs: Vec<_> = input
        .supported_input_configs()
        .context("read supported audio input formats")?
        .collect();
    let outputs: Vec<_> = output
        .supported_output_configs()
        .context("read supported audio output formats")?
        .collect();

    let defaults = [
        input_default.sample_rate(),
        output_default.sample_rate(),
        48_000,
        44_100,
        32_000,
        40_000,
        96_000,
        16_000,
        22_050,
        8_000,
    ];
    let rates = requested_rate
        .as_ref()
        .map_or(defaults.as_slice(), std::slice::from_ref);
    for &rate in rates {
        if !(8_000..=384_000).contains(&rate) {
            continue;
        }
        if let (Some(input_config), Some(output_config)) = (
            config_at_rate(&inputs, &input_default, rate),
            config_at_rate(&outputs, &output_default, rate),
        ) {
            return Ok((input_config, output_config));
        }
    }
    if let Some(rate) = requested_rate {
        bail!("audio input and output do not share a PCM configuration at {rate} Hz");
    }
    // Drivers occasionally advertise only a nonstandard interval with neither
    // endpoint's preferred configuration inside it.
    for left in &inputs {
        for right in &outputs {
            let low = left
                .min_sample_rate()
                .max(right.min_sample_rate())
                .max(8_000);
            let high = left
                .max_sample_rate()
                .min(right.max_sample_rate())
                .min(384_000);
            if low > high {
                continue;
            }
            if let (Some(input_config), Some(output_config)) = (
                config_at_rate(&inputs, &input_default, low),
                config_at_rate(&outputs, &output_default, low),
            ) {
                return Ok((input_config, output_config));
            }
        }
    }
    bail!("audio input and output have no shared PCM sample rate")
}

fn config_at_rate(
    configs: &[SupportedStreamConfigRange],
    default: &SupportedStreamConfig,
    rate: u32,
) -> Option<SupportedStreamConfig> {
    if default.sample_rate() == rate && pcm(default.sample_format()) {
        return Some(*default);
    }
    configs
        .iter()
        .filter(|config| pcm(config.sample_format()))
        .filter_map(|config| config.try_with_sample_rate(rate))
        .min_by_key(|config| {
            (
                config.channels().abs_diff(default.channels()),
                config.sample_format() != default.sample_format(),
                config.sample_format() != SampleFormat::F32,
            )
        })
}

fn pcm(format: SampleFormat) -> bool {
    matches!(
        format,
        SampleFormat::F32
            | SampleFormat::F64
            | SampleFormat::I8
            | SampleFormat::I16
            | SampleFormat::I24
            | SampleFormat::I32
            | SampleFormat::I64
            | SampleFormat::U8
            | SampleFormat::U16
            | SampleFormat::U24
            | SampleFormat::U32
            | SampleFormat::U64
    )
}

enum Endpoint {
    Capture(Capture),
    Playback(Playback),
}

fn open_stream(
    device: &Device,
    supported: &SupportedStreamConfig,
    period: Option<u32>,
    endpoint: Endpoint,
    status: Arc<StreamStatus>,
) -> Result<cpal::Stream> {
    let format = supported.sample_format();
    let mut config = supported.config();
    if let (Some(period), SupportedBufferSize::Range { min, max }) =
        (period, supported.buffer_size())
    {
        config.buffer_size = BufferSize::Fixed(period.clamp(*min, *max));
    }
    macro_rules! stream_types {
        ($($format:ident => $sample:ty),+ $(,)?) => {
            match format {
                $(SampleFormat::$format => typed_stream::<$sample>(device, config, endpoint, status),)+
                _ => Err(anyhow!("unsupported audio format {format}")),
            }
        }
    }
    stream_types!(
        F32 => f32, F64 => f64,
        I8 => i8, I16 => i16, I24 => I24, I32 => i32, I64 => i64,
        U8 => u8, U16 => u16, U24 => U24, U32 => u32, U64 => u64,
    )
}

fn typed_stream<T>(
    device: &Device,
    config: cpal::StreamConfig,
    endpoint: Endpoint,
    status: Arc<StreamStatus>,
) -> Result<cpal::Stream>
where
    T: SizedSample + FromSample<f32>,
    f32: FromSample<T>,
{
    let channels = usize::from(config.channels);
    ensure!(channels > 0, "audio device reported zero channels");
    let error_handler = move |error| status.capture_error(error);
    let stream = match endpoint {
        Endpoint::Capture(mut capture) => device.build_input_stream(
            config,
            move |data: &[T], _| {
                capture.write(data.chunks_exact(channels).map(|frame| {
                    frame
                        .iter()
                        .map(|&value| value.to_sample::<f32>())
                        .sum::<f32>()
                        / channels as f32
                }));
            },
            error_handler,
            None,
        )?,
        Endpoint::Playback(mut playback) => device.build_output_stream(
            config,
            move |data: &mut [T], _| {
                let mut frames = data.chunks_exact_mut(channels);
                playback.read(frames.len(), |sample| {
                    if let Some(frame) = frames.next() {
                        // The endpoint quantizers expect [-1, 1).
                        let sample = T::from_sample(sample.clamp(-1.0, 1.0 - f32::EPSILON));
                        frame.fill(sample);
                    }
                });
                frames.into_remainder().fill(T::EQUILIBRIUM);
            },
            error_handler,
            None,
        )?,
    };
    Ok(stream)
}
