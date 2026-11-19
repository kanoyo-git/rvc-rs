pub mod context;
pub mod device;
mod noise;
mod resampler;
mod sola;
pub mod worker;

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, ensure};
use clap::Args as ClapArgs;
use serde::{Deserialize, Serialize};

use crate::audio;
use crate::catalog::{ModelInfo, Paths};
use crate::model::{self, ConversionSettings, Rvc};
use context::{RealtimeContext, RealtimeOptions};
use device::{DeviceOptions, RealtimeStream};

#[derive(ClapArgs)]
pub struct Args {
    #[arg(long, required_unless_present = "monitor")]
    model: Option<PathBuf>,
    #[arg(long)]
    speaker_id: Option<i64>,
    #[arg(long)]
    index: Option<PathBuf>,
    #[arg(long, default_value_t = 0.0)]
    index_rate: f32,
    #[arg(long, default_value_t = 0.0, allow_negative_numbers = true)]
    pitch: f32,
    #[arg(long, default_value_t = 0.0, allow_negative_numbers = true)]
    formant: f32,
    #[arg(long, default_value = "rmvpe", value_enum)]
    f0_method: model::PitchMethod,
    #[arg(long)]
    host: Option<String>,
    #[arg(long)]
    input_device: Option<String>,
    #[arg(long)]
    output_device: Option<String>,
    #[arg(long)]
    wasapi_exclusive: bool,
    #[arg(long, conflicts_with = "model_rate")]
    sample_rate: Option<u32>,
    #[arg(long, conflicts_with = "monitor")]
    model_rate: bool,
    #[arg(long, default_value_t = 250)]
    block_ms: u32,
    #[arg(long, default_value_t = 50)]
    crossfade_ms: u32,
    #[arg(long, default_value_t = 2500)]
    extra_ms: u32,
    #[arg(long, default_value_t = -60.0, allow_negative_numbers = true)]
    threshold_db: f32,
    #[arg(long, default_value_t = 0.0)]
    rms_mix: f32,
    #[arg(long, default_value = "auto")]
    device: String,
    #[arg(long)]
    monitor: bool,
    #[arg(long)]
    input_noise_reduction: bool,
    #[arg(long)]
    output_noise_reduction: bool,
    #[arg(long)]
    seconds: Option<f64>,
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(default)]
pub struct Request {
    pub model: Option<PathBuf>,
    pub speaker_id: Option<i64>,
    pub index: Option<PathBuf>,
    pub host: Option<String>,
    pub input_device: Option<String>,
    pub output_device: Option<String>,
    pub wasapi_exclusive: bool,
    pub sample_rate: Option<u32>,
    pub model_rate: bool,
    pub block_ms: u32,
    pub crossfade_ms: u32,
    pub extra_ms: u32,
    pub threshold_db: f32,
    pub rms_mix: f32,
    pub monitor: bool,
    pub input_noise_reduction: bool,
    pub output_noise_reduction: bool,
    pub device: String,
    pub settings: ConversionSettings,
}

#[derive(Clone, Copy, Deserialize)]
pub struct LiveSettings {
    pub threshold_db: f32,
    pub rms_mix: f32,
    pub monitor: bool,
    pub input_noise_reduction: bool,
    pub output_noise_reduction: bool,
    pub settings: ConversionSettings,
}

#[derive(Clone)]
pub struct Controls {
    pending: Arc<Mutex<Pending>>,
    can_convert: bool,
}

#[derive(Default)]
struct Pending {
    update: Option<LiveSettings>,
    error: Option<String>,
    active_f0_method: Option<model::PitchMethod>,
}

impl Controls {
    pub fn update(&self, update: LiveSettings) -> Result<()> {
        update.settings.validate()?;
        ensure!(
            (-60.0..=0.0).contains(&update.threshold_db) && (0.0..=1.0).contains(&update.rms_mix),
            "invalid audio threshold or RMS mix"
        );
        ensure!(
            update.monitor || self.can_convert,
            "select a voice model and start audio again to enable conversion"
        );
        let mut pending = self
            .pending
            .lock()
            .map_err(|_| anyhow::anyhow!("realtime controls unavailable"))?;
        pending.update = Some(update);
        pending.error = None;
        Ok(())
    }

    pub fn error(&self) -> Option<String> {
        self.pending.lock().ok()?.error.clone()
    }

    pub fn active_f0_method(&self) -> Option<model::PitchMethod> {
        self.pending.lock().ok()?.active_f0_method
    }
}

impl Default for Request {
    fn default() -> Self {
        let options = RealtimeOptions::default();
        Self {
            model: None,
            speaker_id: None,
            index: None,
            host: None,
            input_device: None,
            output_device: None,
            wasapi_exclusive: false,
            sample_rate: None,
            model_rate: false,
            block_ms: 250,
            crossfade_ms: options.crossfade_ms,
            extra_ms: options.extra_ms,
            threshold_db: options.threshold_db,
            rms_mix: options.rms_mix,
            monitor: false,
            input_noise_reduction: false,
            output_noise_reduction: false,
            device: "auto".to_owned(),
            settings: ConversionSettings {
                index_rate: 0.0,
                ..Default::default()
            },
        }
    }
}

pub fn profile(paths: &Paths) -> Result<Request> {
    let location = paths.root.join("configs/realtime-rust.json");
    if location.is_file() {
        return serde_json::from_reader(std::fs::File::open(&location)?)
            .with_context(|| format!("reading the realtime profile {}", location.display()));
    }
    let location = paths.root.join("configs/config.json");
    if !location.is_file() {
        return Ok(Request::default());
    }
    let source: serde_json::Value = serde_json::from_reader(std::fs::File::open(&location)?)
        .with_context(|| {
            format!(
                "reading the original realtime profile {}",
                location.display()
            )
        })?;
    let number = |key: &str, default: f32| source[key].as_f64().map_or(default, |n| n as f32);
    let field = |key: &str| {
        source[key]
            .as_str()
            .map(str::trim)
            .filter(|s| !s.is_empty())
    };
    let millis = |key: &str, default: u32| {
        source[key]
            .as_f64()
            .map_or(default, |seconds| (seconds * 1000.0).round() as u32)
    };
    let default = Request::default();
    let model = field("pth_path").map(PathBuf::from);
    let model = model.filter(|path| paths.model(path).is_ok());
    Ok(Request {
        model,
        index: field("index_path").map(PathBuf::from),
        host: field("sg_hostapi").map(str::to_owned),
        input_device: field("sg_input_device").map(str::to_owned),
        output_device: field("sg_output_device").map(str::to_owned),
        wasapi_exclusive: source["sg_wasapi_exclusive"].as_bool().unwrap_or_default(),
        model_rate: field("sr_type") == Some("sr_model"),
        block_ms: millis("block_time", default.block_ms),
        crossfade_ms: millis("crossfade_length", default.crossfade_ms),
        extra_ms: millis("extra_time", default.extra_ms),
        threshold_db: number("threhold", default.threshold_db),
        rms_mix: number("rms_mix_rate", default.rms_mix),
        settings: ConversionSettings {
            pitch: number("pitch", default.settings.pitch),
            formant: number("formant", default.settings.formant),
            index_rate: number("index_rate", default.settings.index_rate),
            f0_method: match field("f0method") {
                Some("pm") => model::PitchMethod::Pm,
                Some("fcpe") => model::PitchMethod::Fcpe,
                _ => model::PitchMethod::Rmvpe,
            },
            ..default.settings
        },
        ..default
    })
}

pub fn save_profile(paths: &Paths, profile: &Request) -> Result<()> {
    let path = paths.root.join("configs/realtime-rust.json");
    let raw = serde_json::to_vec_pretty(profile)?;
    audio::write_file(&path, &raw, true)
}

pub fn start(paths: &Paths, request: Request) -> Result<RealtimeStream> {
    start_controlled(paths, request).map(|(stream, _)| stream)
}

pub fn start_controlled(paths: &Paths, request: Request) -> Result<(RealtimeStream, Controls)> {
    let (engine, index_source) = if request.model.is_some() || !request.monitor {
        let path = paths.model(request.model.as_deref().context("select a voice model")?)?;
        let speaker = ModelInfo::read(&path)?.speaker(request.speaker_id)?;
        let index = paths.require_index(
            &path,
            speaker,
            request.index.as_deref(),
            request.settings.index_rate,
        )?;
        let engine = Rvc::load(
            paths,
            &path,
            Some(speaker),
            index.as_deref(),
            request.settings,
            model::weights::device(&request.device)?,
        )?;
        (Some(engine), Some((path, speaker, index.is_some())))
    } else {
        (None, None)
    };
    let controls = Controls {
        pending: Arc::new(Mutex::new(Pending {
            active_f0_method: engine.as_ref().and_then(Rvc::pitch_method),
            ..Pending::default()
        })),
        can_convert: engine.is_some(),
    };
    let pending = Arc::clone(&controls.pending);
    let paths = paths.clone();
    let sample_rate = if request.model_rate {
        engine.as_ref().map(Rvc::rate)
    } else {
        request.sample_rate
    };
    let device = DeviceOptions {
        host: request.host,
        input: request.input_device,
        output: request.output_device,
        sample_rate,
        block_ms: request.block_ms,
        wasapi_exclusive: request.wasapi_exclusive,
    };
    let options = RealtimeOptions {
        crossfade_ms: request.crossfade_ms,
        extra_ms: request.extra_ms,
        threshold_db: request.threshold_db,
        rms_mix: request.rms_mix,
    };
    let stream = RealtimeStream::start(&device, |format| {
        let mut engine = engine;
        let mut index_source = index_source;
        let mut context =
            RealtimeContext::new(format, engine.as_ref().map_or(48_000, Rvc::rate), options)?;
        context.set_noise_reduction(
            request.input_noise_reduction,
            request.output_noise_reduction,
        )?;
        let block = format.block_frames / (format.sample_rate / 100) as usize;
        let mut settings = request.settings;
        let mut monitor = request.monitor;
        Ok(move |input: &[f32], output: &mut [f32], discontinuity| {
            let update = pending
                .lock()
                .ok()
                .and_then(|mut pending| pending.update.take());
            if let Some(update) = update {
                let apply = (|| -> Result<()> {
                    if let Some(engine) = &mut engine {
                        if let Some((path, speaker, loaded)) = &mut index_source
                            && update.settings.index_rate > 0.0
                            && !*loaded
                        {
                            let index = paths
                                .require_index(
                                    path,
                                    *speaker,
                                    request.index.as_deref(),
                                    update.settings.index_rate,
                                )?
                                .context("no retrieval index for realtime conversion")?;
                            engine.set_index(&index)?;
                            *loaded = true;
                        }
                        if update.monitor != monitor {
                            engine.reset_stream();
                        }
                    }
                    context.set_levels(update.threshold_db, update.rms_mix)?;
                    context.set_noise_reduction(
                        update.input_noise_reduction,
                        update.output_noise_reduction,
                    )?;
                    if update.monitor != monitor {
                        context.reset();
                    }
                    monitor = update.monitor;
                    settings = update.settings;
                    Ok(())
                })();
                if let Ok(mut pending) = pending.lock() {
                    pending.error = apply.err().map(|error| format!("{error:#}"));
                }
            }
            if let Some(engine) = &mut engine
                && !monitor
            {
                let previous = engine.pitch_method();
                if let Err(error) = engine.update_pitch_method(&paths, settings.f0_method) {
                    settings.f0_method = previous.unwrap_or_default();
                    if let Ok(mut pending) = pending.lock() {
                        pending.error = Some(format!("{error:#}"));
                    }
                }
                if previous != engine.pitch_method()
                    && let Ok(mut pending) = pending.lock()
                {
                    pending.active_f0_method = engine.pitch_method();
                }
                if discontinuity {
                    engine.reset_stream();
                }
                context.process(input, output, discontinuity, |samples, skip, frames| {
                    engine.convert_window(samples, settings, skip, frames, block)
                })
            } else {
                context.monitor(input, output, discontinuity)
            }
        })
    })?;
    Ok((stream, controls))
}

pub fn run(paths: &Paths, args: Args) -> Result<()> {
    let duration = args.seconds.map(Duration::try_from_secs_f64).transpose()?;
    ensure!(
        duration.is_none_or(|d| !d.is_zero()),
        "stream duration must be positive"
    );
    let stream = start(
        paths,
        Request {
            model: args.model,
            speaker_id: args.speaker_id,
            index: args.index,
            host: args.host,
            input_device: args.input_device,
            output_device: args.output_device,
            wasapi_exclusive: args.wasapi_exclusive,
            sample_rate: args.sample_rate,
            model_rate: args.model_rate,
            block_ms: args.block_ms,
            crossfade_ms: args.crossfade_ms,
            extra_ms: args.extra_ms,
            threshold_db: args.threshold_db,
            rms_mix: args.rms_mix,
            monitor: args.monitor,
            device: args.device,
            input_noise_reduction: args.input_noise_reduction,
            output_noise_reduction: args.output_noise_reduction,
            settings: ConversionSettings {
                pitch: args.pitch,
                formant: args.formant,
                f0_method: args.f0_method,
                index_rate: args.index_rate,
                ..Default::default()
            },
        },
    )?;
    eprintln!(
        "Audio started at {} Hz, {} samples per block",
        stream.format().sample_rate,
        stream.format().block_frames
    );
    let begun = Instant::now();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    runtime.block_on(async {
        let mut interval = tokio::time::interval(Duration::from_millis(500));
        let stop = tokio::signal::ctrl_c();
        tokio::pin!(stop);
        loop {
            tokio::select! {
                result = &mut stop => break result.map_err(anyhow::Error::from),
                _ = interval.tick() => {
                    stream.check()?;
                    if duration.is_some_and(|d| begun.elapsed() >= d) { break Ok(()); }
                }
            }
        }
    })?;
    println!("{}", serde_json::to_string(&stream.stats())?);
    stream.stop()
}
