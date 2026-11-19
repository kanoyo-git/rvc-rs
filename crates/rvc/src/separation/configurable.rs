//! Local PyMSS models described by a checkpoint and a YAML configuration.

#[path = "apollo.rs"]
mod apollo;
#[path = "bandit.rs"]
mod bandit;
#[path = "bandit_v2.rs"]
mod bandit_v2;
#[path = "scnet.rs"]
mod scnet;

use std::collections::{BTreeMap, HashMap, HashSet};
use std::ops::ControlFlow;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tch::{Device, Tensor};

use super::demucs::{Demucs, legacy};
use super::mdx23c::{self, Mdx23c};
use super::overlap::{Geometry, Progress, demix, demix_uniform, residual};
use super::roformer::Roformer;
use super::vr::Vr;
use super::{
    AudioEncoding, JobProgress, PcmLayout, SAMPLE_RATE, demix_error, job_progress, persist_stem,
    plan_outputs, read_audio, stage_stem,
};
use crate::audio::Format;
use crate::checkpoint::Checkpoint;
use scnet::Scnet;

pub struct ConfiguredOptions<'a> {
    pub model_type: &'a str,
    pub checkpoint: &'a Path,
    pub config: &'a Path,
    pub stems: &'a [String],
    pub device: Device,
    pub additional_devices: &'a [Device],
    pub format: Format,
    pub overwrite: bool,
    pub chunk_size: Option<usize>,
    pub overlap_size: Option<usize>,
    pub save_as_folder: bool,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(transparent)]
pub struct InferenceParams(BTreeMap<String, Value>);

const COMMON_PARAMS: &[&str] = &["normalize", "enable_tta", "batch_size", "use_amp"];
const TRACK_PARAMS: &[&str] = &[
    "chunk_size",
    "overlap_size",
    "standardize",
    "stem_batch_size",
    "model_dtype",
];
const VR_PARAMS: &[&str] = &[
    "window_size",
    "aggression",
    "enable_post_process",
    "post_process_threshold",
    "high_end_process",
    "fuse_conv_bn",
    "use_channels_last",
    "mps_model_backend",
    "mps_model_compute_dtype",
];
const LEGACY_PARAMS: &[&str] = &["shifts", "split", "overlap"];

impl InferenceParams {
    pub fn parse(overrides: &[String]) -> Result<Self> {
        let mut params = Self::default();
        for entry in overrides {
            let (key, raw) = entry
                .split_once('=')
                .context("expected PyMSS parameter key=value")?;
            ensure!(!key.is_empty(), "PyMSS parameter name is empty");
            let value = if raw.eq_ignore_ascii_case("true") {
                Value::Bool(true)
            } else if raw.eq_ignore_ascii_case("false") {
                Value::Bool(false)
            } else if let Ok(value) = raw.parse::<i64>() {
                Value::from(value)
            } else if let Some(value) = raw
                .parse::<f64>()
                .ok()
                .and_then(serde_json::Number::from_f64)
            {
                Value::Number(value)
            } else {
                Value::String(raw.to_owned())
            };
            params.0.insert(key.to_owned(), value);
        }
        Ok(params)
    }

    pub fn overlay(&self, overrides: &Self) -> Self {
        let mut result = self.clone();
        result.0.extend(overrides.0.clone());
        result
    }

    pub fn resolve_device(&mut self, request: &str) -> Result<Device> {
        let (backend, adapter) = request.split_once(':').unwrap_or((request, "0"));
        if matches!(backend, "privateuseone" | "directml" | "dml") {
            let provider = format!("directml:{adapter}");
            #[cfg(not(feature = "onnx"))]
            bail!("{provider} requires the onnx build feature");
            #[cfg(feature = "onnx")]
            {
                use crate::model::onnx::Provider;
                let directml = Provider::parse("device", &provider)?;
                if let Some(value) = self.get("onnx_provider") {
                    let value = value.as_str().context("onnx_provider must be text")?;
                    ensure!(
                        Provider::parse("onnx_provider", value)? == directml,
                        "--device and onnx_provider select different DirectML adapters"
                    );
                }
                self.0
                    .insert("onnx_provider".to_owned(), Value::String(provider));
                return Ok(Device::Cpu);
            }
        }
        if request == "mlx" {
            let device = super::device("mps")?;
            ensure!(
                std::env::var_os("RVC_MLX_LIBRARY").is_some()
                    || self
                        .get("mps_model_backend")
                        .is_some_and(|value| value == "torch"),
                "--device mlx requires RVC_MLX_LIBRARY"
            );
            return Ok(device);
        }
        super::device(request)
    }

    pub fn device_label(&self, device: Device) -> String {
        match device {
            Device::Cpu => self
                .get("onnx_provider")
                .and_then(Value::as_str)
                .filter(|provider| provider.starts_with("directml"))
                .unwrap_or("cpu")
                .to_owned(),
            Device::Cuda(i) => format!("cuda:{i}"),
            Device::Mps => "mps".to_owned(),
            Device::Vulkan => "vulkan".to_owned(),
        }
    }

    pub fn with_tta(mut self, tta: bool) -> Self {
        if tta && self.get("enable_tta").is_none() {
            self.0.insert("enable_tta".to_owned(), Value::Bool(true));
        }
        self
    }

    fn with_geometry(mut self, chunk: Option<usize>, overlap: Option<usize>) -> Self {
        for (key, value) in [("chunk_size", chunk), ("overlap_size", overlap)] {
            if let Some(value) = value {
                self.0.insert(key.to_owned(), Value::from(value));
            }
        }
        self
    }

    pub fn supported(model_type: &str) -> Vec<&'static str> {
        let mut keys = COMMON_PARAMS.to_vec();
        if model_type == "vr" {
            keys.extend_from_slice(VR_PARAMS);
            if cfg!(feature = "onnx") {
                keys.push("onnx_provider");
            }
        } else {
            keys.extend_from_slice(TRACK_PARAMS);
            keys.push("mps_mlx_clear_cache");
            if legacy::is_legacy(model_type) {
                keys.extend_from_slice(LEGACY_PARAMS);
            }
            if matches!(
                model_type,
                "bs_roformer" | "bs_roformer_hyperace" | "mel_band_roformer"
            ) {
                keys.extend_from_slice(&[
                    "mask_mode",
                    "cuda_attention_backend",
                    "mps_attention_backend",
                    "mps_mlx_min_tokens",
                ]);
            }
            if matches!(
                model_type,
                "bs_roformer"
                    | "bs_roformer_hyperace"
                    | "mel_band_roformer"
                    | "mdx23c"
                    | "scnet"
                    | "apollo"
                    | "bandit"
                    | "bandit_v2"
                    | "htdemucs"
            ) {
                keys.extend_from_slice(&["mps_model_backend", "mps_model_compute_dtype"]);
            }
            if cfg!(feature = "onnx")
                && (legacy::is_legacy(model_type)
                    || matches!(
                        model_type,
                        "mdx23c"
                            | "bs_roformer"
                            | "bs_roformer_hyperace"
                            | "mel_band_roformer"
                            | "scnet"
                            | "apollo"
                            | "bandit"
                            | "bandit_v2"
                            | "htdemucs"
                    ))
            {
                keys.push("onnx_provider");
            }
        }
        keys
    }

    pub fn validate(&self, model_type: &str) -> Result<()> {
        let keys = Self::supported(model_type);
        for (name, value) in &self.0 {
            if value.is_null() {
                continue;
            }
            // Preserve the earlier native `chunk_size` alias for VR window size.
            ensure!(
                keys.contains(&name.as_str()) || model_type == "vr" && name == "chunk_size",
                "PyMSS parameter {name:?} is unavailable for {model_type} in the native runtime"
            );
            match name.as_str() {
                "chunk_size" | "window_size" | "batch_size" => {
                    ensure!(
                        self.integer(name)?.is_some_and(|value| value > 0),
                        "PyMSS parameter {name:?} must be positive"
                    );
                }
                "overlap_size" | "shifts" | "stem_batch_size" | "mps_mlx_min_tokens" => {
                    self.integer(name)?;
                }
                "aggression" => {
                    let amount = self.number(name)?.unwrap_or_default();
                    ensure!(
                        (0.0..=100.0).contains(&amount),
                        "VR aggression must be between 0 and 100"
                    );
                }
                "post_process_threshold" => {
                    let amount = self.number(name)?.unwrap_or_default();
                    ensure!(
                        (0.0..=1.0).contains(&amount),
                        "invalid VR post-processing threshold"
                    );
                }
                "overlap" => {
                    let amount = self.number(name)?.unwrap_or_default();
                    ensure!(
                        (0.0..1.0).contains(&amount),
                        "Demucs overlap must be at least zero and below one"
                    );
                }
                "mask_mode" => ensure!(
                    matches!(value.as_str(), Some("no_segm" | "full" | "segm_only")),
                    "invalid HyperACE mask mode"
                ),
                "cuda_attention_backend" => {
                    super::roformer::cuda_sdpa::Policy::parse(value)?;
                }
                "mps_attention_backend" => {
                    super::roformer::MpsAttention::parse(value)?;
                }
                "mps_model_backend" => ensure!(
                    matches!(value.as_str(), Some("torch" | "mlx_full")),
                    "mps_model_backend must be torch or mlx_full"
                ),
                "mps_model_compute_dtype" => ensure!(
                    matches!(value.as_str(), Some("float16" | "float32")),
                    "mps_model_compute_dtype must be float16 or float32"
                ),
                "onnx_provider" => ensure!(value.is_string(), "onnx_provider must be text"),
                "model_dtype" => {
                    let half = super::roformer::requests_half(value);
                    ensure!(
                        matches!(
                            value.as_str(),
                            Some("auto" | "float32" | "fp32" | "torch.float32")
                        ) || half
                            && cfg!(feature = "onnx")
                            && matches!(
                                model_type,
                                "bs_roformer" | "bs_roformer_hyperace" | "mel_band_roformer"
                            ),
                        "use CUDA use_amp, or model_dtype=float16 with RoFormer ONNX"
                    );
                }
                _ => {
                    self.boolean(name)?;
                }
            }
        }
        Ok(())
    }

    fn get(&self, key: &str) -> Option<&Value> {
        self.0.get(key).filter(|value| !value.is_null())
    }

    fn integer(&self, key: &str) -> Result<Option<usize>> {
        self.get(key)
            .map(|value| {
                let number = value
                    .as_u64()
                    .or_else(|| {
                        value
                            .as_f64()
                            .filter(|n| (0.0..u64::MAX as f64).contains(n) && n.fract() == 0.0)
                            .map(|n| n as u64)
                    })
                    .or_else(|| value.as_str().and_then(|s| s.trim().parse().ok()));
                Ok(usize::try_from(number.with_context(|| {
                    format!("PyMSS parameter {key:?} must be a nonnegative integer")
                })?)?)
            })
            .transpose()
    }

    fn boolean(&self, key: &str) -> Result<Option<bool>> {
        self.get(key)
            .map(|value| {
                value
                    .as_bool()
                    .with_context(|| format!("PyMSS parameter {key:?} must be true or false"))
            })
            .transpose()
    }

    fn number(&self, key: &str) -> Result<Option<f64>> {
        self.get(key)
            .map(|value| {
                value
                    .as_f64()
                    .or_else(|| value.as_str().and_then(|s| s.trim().parse().ok()))
                    .filter(|amount| amount.is_finite())
                    .with_context(|| format!("PyMSS parameter {key:?} must be numeric"))
            })
            .transpose()
    }
}

fn set_inference(document: &mut Value, key: &str, value: Value) -> Result<()> {
    if document["inference"].is_null() {
        document["inference"] = serde_json::json!({});
    }
    document["inference"]
        .as_object_mut()
        .context("invalid PyMSS inference configuration")?
        .insert(key.to_owned(), value);
    Ok(())
}

#[derive(Debug, Serialize)]
pub struct StemFile {
    pub name: String,
    pub path: PathBuf,
}

#[derive(Debug, Serialize)]
pub struct ConfiguredStemFiles {
    pub input: PathBuf,
    pub stems: Vec<StemFile>,
}

#[derive(Deserialize)]
struct CommonConfig {
    audio: AudioConfig,
    training: TrainingConfig,
    #[serde(default)]
    inference: InferenceConfig,
}

#[derive(Deserialize)]
struct AudioConfig {
    sample_rate: Option<u32>,
    chunk_size: Option<usize>,
}

#[derive(Deserialize)]
struct TrainingConfig {
    #[serde(default)]
    instruments: Vec<String>,
    target_instrument: Option<String>,
}

#[derive(Default, Deserialize)]
struct InferenceConfig {
    overlap_size: Option<usize>,
    batch_size: Option<usize>,
    stem_batch_size: Option<usize>,
    // In existing PyMSS YAML, this field standardizes input audio.
    normalize: Option<bool>,
}

enum Model {
    Apollo(apollo::Apollo),
    Bandit(bandit::Bandit),
    Demucs(Box<Demucs>),
    LegacyDemucs(Box<legacy::Legacy>),
    Mdx(Mdx23c),
    Roformer(Roformer),
    Scnet(Scnet),
    Vr(Box<Vr>),
}

impl Model {
    fn load(
        document: &Value,
        model_type: &str,
        weights: &HashMap<String, Tensor>,
        device: Device,
    ) -> Result<Self> {
        match model_type {
            "apollo" => Ok(Self::Apollo(apollo::Apollo::load(
                document, weights, device,
            )?)),
            "bandit" => Ok(Self::Bandit(bandit::Bandit::load_v1(
                document, weights, device,
            )?)),
            "bandit_v2" => Ok(Self::Bandit(bandit_v2::load(document, weights, device)?)),
            "scnet" => Ok(Self::Scnet(Scnet::load(document, weights, device)?)),
            "mdx23c" => Ok(Self::Mdx(Mdx23c::load(
                mdx23c::ModelConfig::from_document(document)?,
                weights,
                device,
                document,
            )?)),
            "bs_roformer_hyperace" => Ok(Self::Roformer(Roformer::load_hyperace(
                document, weights, device,
            )?)),
            "bs_roformer" if weights.keys().any(|name| name.contains(".segm.")) => Ok(
                Self::Roformer(Roformer::load_hyperace(document, weights, device)?),
            ),
            "bs_roformer" | "mel_band_roformer" => Ok(Self::Roformer(Roformer::load_configured(
                document,
                model_type == "mel_band_roformer",
                weights,
                device,
            )?)),
            _ => bail!("unsupported local PyMSS model type {model_type:?}"),
        }
    }

    fn channels(&self) -> Result<usize> {
        match self {
            Self::Apollo(_) | Self::Bandit(_) => Ok(2),
            Self::Demucs(model) => Ok(model.channels()),
            Self::LegacyDemucs(model) => Ok(model.channels()),
            Self::Vr(model) => Ok(model.channels()),
            Self::Mdx(model) => {
                let channels = model.input_channels();
                ensure!(
                    channels == model.output_channels(),
                    "MDX23C's PyMSS inverse transform requires a stereo model for track separation"
                );
                Ok(channels)
            }
            Self::Roformer(model) => Ok(model.channels()),
            Self::Scnet(model) => Ok(model.channels()),
        }
    }

    fn stems(&self) -> usize {
        match self {
            Self::Apollo(_) => 1,
            Self::Bandit(model) => model.num_stems(),
            Self::Demucs(model) => model.stems(),
            Self::LegacyDemucs(model) => model.stems().len(),
            Self::Vr(model) => model.stems().len(),
            Self::Mdx(model) => model.num_stems(),
            Self::Roformer(model) => model.num_stems(),
            Self::Scnet(model) => model.num_stems(),
        }
    }

    fn predict_batch(&self, audio: &[f32], count: usize) -> Result<Vec<f32>> {
        match self {
            Self::Apollo(model) => model.predict_batch(audio, count),
            Self::Bandit(model) => model.predict_batch(audio, count),
            Self::Demucs(model) => model.predict_batch(audio, count),
            Self::LegacyDemucs(_) | Self::Vr(_) => bail!("model requires a complete audio track"),
            Self::Mdx(model) => model.predict_batch(audio, count),
            Self::Roformer(model) => model.predict_batch(audio, count),
            Self::Scnet(model) => model.predict_batch(audio, count),
        }
    }
}

#[derive(Clone, Copy)]
enum Source {
    Prediction(usize),
    Residual,
}

#[derive(Clone, Copy)]
struct Selection<'a> {
    name: &'a str,
    source: Source,
}

impl TrainingConfig {
    fn target(&self) -> Option<&str> {
        self.target_instrument
            .as_deref()
            .filter(|name| !name.is_empty())
    }

    fn model_stems(&self) -> usize {
        if self.target().is_some() {
            1
        } else {
            self.instruments.len()
        }
    }

    fn outputs(&self) -> Vec<Selection<'_>> {
        if let Some(target) = self.target() {
            let mut outputs = vec![Selection {
                name: target,
                source: Source::Prediction(0),
            }];
            if let Some(other) = self.instruments.iter().find(|name| *name != target) {
                outputs.push(Selection {
                    name: other,
                    source: Source::Residual,
                });
            }
            outputs
        } else {
            self.instruments
                .iter()
                .enumerate()
                .map(|(index, name)| Selection {
                    name,
                    source: Source::Prediction(index),
                })
                .collect()
        }
    }

    fn select(&self, names: &[String]) -> Result<Vec<Selection<'_>>> {
        let outputs = self.outputs();
        ensure!(!outputs.is_empty(), "PyMSS config has no instruments");
        let mut by_name = HashMap::with_capacity(outputs.len());
        for output in &outputs {
            ensure!(
                by_name
                    .insert(output.name.to_lowercase(), *output)
                    .is_none(),
                "duplicate PyMSS instrument name {:?}",
                output.name,
            );
        }
        if names.is_empty() {
            return Ok(outputs);
        }
        let mut selected = Vec::with_capacity(names.len());
        let mut seen = HashSet::new();
        for name in names {
            let normalized = name.to_lowercase();
            let output = by_name.get(&normalized).with_context(|| {
                let choices = outputs
                    .iter()
                    .map(|item| item.name)
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("unknown PyMSS output stem {name:?}; available stems: {choices}")
            })?;
            if seen.insert(normalized) {
                selected.push(*output);
            }
        }
        Ok(selected)
    }
}

/// Loaded PyMSS weights and configuration reusable across local files or PCM requests.
pub struct ConfiguredModel {
    model: Model,
    replicas: Vec<Mutex<Model>>,
    layout: PcmLayout,
    training: TrainingConfig,
    geometry: Geometry,
    stem_batch_size: usize,
    standardize: bool,
    enable_tta: bool,
    normalize: bool,
    autocast: bool,
    clear_mlx_cache: bool,
    legacy_geometry: (Option<usize>, Option<usize>),
    legacy_split: bool,
    legacy_shifts: usize,
}

impl ConfiguredModel {
    pub fn load(
        model_type: &str,
        checkpoint: &Path,
        config: &Path,
        device: Device,
        chunk_size: Option<usize>,
        overlap_size: Option<usize>,
    ) -> Result<Self> {
        let params = InferenceParams::default().with_geometry(chunk_size, overlap_size);
        Self::load_with_params(model_type, checkpoint, config, device, &params)
    }

    pub fn load_with_params(
        model_type: &str,
        checkpoint: &Path,
        config: &Path,
        device: Device,
        params: &InferenceParams,
    ) -> Result<Self> {
        params.validate(model_type)?;
        let chunk_size = params.integer("chunk_size")?;
        let overlap_size = params.integer("overlap_size")?;
        let enable_tta = params.boolean("enable_tta")?;
        let standardize = params.boolean("standardize")?;
        let normalize = params.boolean("normalize")?.unwrap_or(false);
        let stem_batch_size = params.integer("stem_batch_size")?;
        let amp = params.boolean("use_amp")?;
        let cuda = matches!(device, Device::Cuda(_));
        if legacy::is_legacy(model_type) {
            let legacy =
                legacy::Legacy::load(checkpoint, config, device, params.get("onnx_provider"))?;
            let layout = PcmLayout {
                sample_rate: legacy.rate(),
                channels: legacy.channels(),
            };
            let frames = legacy.segment();
            let training = TrainingConfig {
                instruments: legacy.stems().to_vec(),
                target_instrument: None,
            };
            let document = if checkpoint != config && config.is_file() {
                load_document(config)?
            } else {
                Value::Null
            };
            let settings = &document["inference"];
            let legacy_shifts = params
                .integer("shifts")?
                .or(settings["shifts"]
                    .as_u64()
                    .map(usize::try_from)
                    .transpose()?)
                .unwrap_or(0);
            let legacy_split = params
                .boolean("split")?
                .or(settings["split"].as_bool())
                .unwrap_or(true);
            let fraction = params.number("overlap")?.or(settings["overlap"].as_f64());
            if let Some(value) = fraction {
                ensure!(
                    (0.0..1.0).contains(&value),
                    "invalid legacy Demucs overlap fraction"
                );
            }
            let overlap = overlap_size.or_else(|| {
                fraction.map(|fraction| {
                    let segment = chunk_size.unwrap_or(frames);
                    segment - ((1.0 - fraction) * segment as f64) as usize
                })
            });
            return Ok(Self {
                model: Model::LegacyDemucs(Box::new(legacy)),
                replicas: Vec::new(),
                layout,
                training,
                geometry: Geometry {
                    frames_per_chunk: frames,
                    overlap: frames / 4,
                    batch_size: 1,
                },
                stem_batch_size: stem_batch_size
                    .or(settings["stem_batch_size"]
                        .as_u64()
                        .map(usize::try_from)
                        .transpose()?)
                    .unwrap_or(0),
                standardize: standardize
                    .or(settings["normalize"].as_bool())
                    .unwrap_or(false),
                enable_tta: enable_tta
                    .or(settings["enable_tta"].as_bool())
                    .unwrap_or(false),
                normalize,
                autocast: cuda
                    && amp
                        .or(document["training"]["use_amp"].as_bool())
                        .unwrap_or(false),
                clear_mlx_cache: false,
                legacy_geometry: (chunk_size, overlap),
                legacy_split,
                legacy_shifts,
            });
        }
        if model_type == "vr" {
            let mut document = if checkpoint != config && config.is_file() {
                load_document(config)?
            } else {
                serde_json::json!({})
            };
            for key in VR_PARAMS
                .iter()
                .chain(["batch_size", "enable_tta", "use_amp", "onnx_provider"].iter())
            {
                if let Some(value) = params.get(key) {
                    let value = match *key {
                        "window_size" | "batch_size" => {
                            Value::from(params.integer(key)?.unwrap_or_default())
                        }
                        "aggression" => Value::from(params.number(key)?.unwrap_or_default() as i32),
                        "post_process_threshold" => {
                            Value::from(params.number(key)?.unwrap_or_default())
                        }
                        _ => value.clone(),
                    };
                    set_inference(&mut document, key, value)?;
                }
            }
            let mut options = if document["inference"].is_null() {
                super::vr::Options::default()
            } else {
                super::vr::Options::deserialize(&document["inference"])?
            };
            if let Some(frames) = chunk_size {
                options.window_size = frames;
            }
            ensure!(
                overlap_size.is_none(),
                "VR overlap is determined by its network"
            );
            ensure!(
                Vr::supports(checkpoint),
                "unknown VR checkpoint {}",
                checkpoint.display()
            );
            let vr = Vr::load(checkpoint, device, options, &document)?;
            let instruments = vr.stems().map(str::to_owned).into_iter().collect();
            let layout = PcmLayout {
                sample_rate: vr.rate(),
                channels: vr.channels(),
            };
            return Ok(Self {
                model: Model::Vr(Box::new(vr)),
                replicas: Vec::new(),
                layout,
                training: TrainingConfig {
                    instruments,
                    target_instrument: None,
                },
                geometry: Geometry {
                    frames_per_chunk: 1,
                    overlap: 0,
                    batch_size: 1,
                },
                stem_batch_size: 0,
                standardize: false,
                enable_tta: false,
                normalize,
                autocast: false,
                clear_mlx_cache: false,
                legacy_geometry: (None, None),
                legacy_split: false,
                legacy_shifts: 0,
            });
        }
        let mut document = load_document(config)?;
        for key in [
            "mask_mode",
            "cuda_attention_backend",
            "mps_attention_backend",
            "mps_mlx_min_tokens",
            "mps_model_backend",
            "mps_model_compute_dtype",
            "onnx_provider",
            "model_dtype",
        ] {
            if let Some(value) = params.get(key) {
                set_inference(&mut document, key, value.clone())?;
            }
        }
        let common =
            CommonConfig::deserialize(&document).context("invalid PyMSS common configuration")?;
        common.training.select(&[])?;
        let frames_per_chunk = chunk_size
            .or_else(|| {
                (model_type == "htdemucs")
                    .then(|| {
                        let rate = document["training"]["samplerate"].as_f64()?;
                        let segment = document["training"]["segment"].as_f64()?;
                        Some((rate * segment) as usize)
                    })
                    .flatten()
            })
            .or(common.audio.chunk_size)
            .context("PyMSS config has no audio.chunk_size; specify a chunk size")?;
        let geometry = Geometry {
            frames_per_chunk,
            overlap: overlap_size
                .or(common.inference.overlap_size)
                .unwrap_or(frames_per_chunk / 2),
            batch_size: params
                .integer("batch_size")?
                .or(common.inference.batch_size)
                .unwrap_or(1),
        };
        ensure!(
            geometry.overlap < geometry.frames_per_chunk,
            "expected 0 <= PyMSS overlap size < chunk size"
        );
        ensure!(geometry.batch_size > 0, "PyMSS batch size must be positive");
        let sample_rate = if model_type == "htdemucs" {
            document["training"]["samplerate"]
                .as_u64()
                .map(u32::try_from)
                .transpose()?
                .or(common.audio.sample_rate)
                .unwrap_or(SAMPLE_RATE)
        } else {
            common.audio.sample_rate.unwrap_or(SAMPLE_RATE)
        };
        ensure!(sample_rate > 0, "PyMSS sample rate must be positive");
        let checkpoint = if model_type == "apollo" {
            apollo::checkpoint_path(checkpoint)
        } else {
            checkpoint.to_path_buf()
        };
        let model = if model_type == "htdemucs" {
            Model::Demucs(Box::new(Demucs::from_yaml(
                &document,
                &checkpoint,
                device,
                geometry.frames_per_chunk,
            )?))
        } else {
            let weight_device = if cuda
                && matches!(
                    model_type,
                    "bs_roformer" | "bs_roformer_hyperace" | "mel_band_roformer"
                ) {
                device
            } else {
                Device::Cpu
            };
            let weights = checkpoint_weights(&checkpoint, weight_device)?;
            if params
                .get("mask_mode")
                .is_some_and(|mode| mode != "no_segm")
            {
                ensure!(
                    model_type == "bs_roformer_hyperace"
                        || model_type == "bs_roformer"
                            && weights.keys().any(|name| name.contains(".segm.")),
                    "mask_mode requires a HyperACE checkpoint"
                );
            }
            Model::load(&document, model_type, &weights, device)
                .with_context(|| format!("loading {}", checkpoint.display()))?
        };
        let channels = model.channels()?;
        let expected = common.training.model_stems();
        ensure!(
            model.stems() == expected,
            "model produces {} stems but YAML training configuration expects {expected}",
            model.stems()
        );
        Ok(Self {
            model,
            replicas: Vec::new(),
            layout: PcmLayout {
                sample_rate,
                channels,
            },
            training: common.training,
            geometry,
            stem_batch_size: stem_batch_size
                .or(common.inference.stem_batch_size)
                .unwrap_or(0),
            standardize: standardize.or(common.inference.normalize).unwrap_or(false),
            enable_tta: enable_tta
                .or(document["inference"]["enable_tta"].as_bool())
                .unwrap_or(false),
            normalize,
            autocast: cuda
                && amp
                    .or(document["inference"]["use_amp"].as_bool())
                    .or(document["training"]["use_amp"].as_bool())
                    .unwrap_or(false),
            clear_mlx_cache: (device == Device::Mps
                || document["inference"]["mps_model_backend"] == "mlx_full")
                && params
                    .boolean("mps_mlx_clear_cache")?
                    .or(document["inference"]["mps_mlx_clear_cache"].as_bool())
                    .unwrap_or(false),
            legacy_geometry: (None, None),
            legacy_split: false,
            legacy_shifts: 0,
        })
    }

    pub fn load_on_devices(
        model_type: &str,
        checkpoint: &Path,
        config: &Path,
        device: Device,
        additional_devices: &[Device],
        params: &InferenceParams,
    ) -> Result<Self> {
        let mut primary = Self::load_with_params(model_type, checkpoint, config, device, params)?;
        if matches!(primary.model, Model::LegacyDemucs(_) | Model::Vr(_)) {
            return Ok(primary);
        }
        for &device in additional_devices {
            let replica = Self::load_with_params(model_type, checkpoint, config, device, params)
                .with_context(|| format!("creating PyMSS replica on {device:?}"))?;
            primary.replicas.push(Mutex::new(replica.model));
        }
        Ok(primary)
    }

    pub fn rate(&self) -> u32 {
        self.layout.sample_rate
    }

    pub fn channels(&self) -> usize {
        self.layout.channels
    }

    pub fn output_channels(&self) -> usize {
        match &self.model {
            Model::Vr(model) if model.stems()[1] == "Aspiration" => 1,
            _ => self.channels(),
        }
    }

    pub fn stems(&self) -> Vec<String> {
        self.training
            .outputs()
            .into_iter()
            .map(|stem| stem.name.to_owned())
            .collect()
    }

    /// Input and output buffers are channel-major at the configured rate.
    pub fn separate_planar(
        &self,
        pcm: &[f32],
        selected: &[String],
        mut progress: impl FnMut(JobProgress) -> ControlFlow<()>,
    ) -> Result<Vec<(String, Vec<f32>)>> {
        let selections = self.training.select(selected)?;
        let groups = self.stem_groups(&selections);
        let count = groups.len();
        let mut stems = Vec::with_capacity(selections.len());
        for (index, group) in groups.enumerate() {
            let tracks = self.separate_audio(pcm, group, |status| {
                progress(job_progress(0, 1, combine_progress(status, index, count)))
            })?;
            for selection in group {
                stems.push((
                    selection.name.to_owned(),
                    tracks.stem(selection.source)?.to_vec(),
                ));
            }
        }
        Ok(stems)
    }

    fn stem_groups<'a, 'b>(
        &self,
        selected: &'a [Selection<'b>],
    ) -> std::slice::Chunks<'a, Selection<'b>> {
        let len = if self.normalize || self.stem_batch_size == 0 {
            selected.len()
        } else {
            self.stem_batch_size.min(selected.len())
        };
        selected.chunks(len)
    }

    fn separate_audio(
        &self,
        mix: &[f32],
        selections: &[Selection<'_>],
        mut progress: impl FnMut(Progress) -> ControlFlow<()>,
    ) -> Result<Separated> {
        let channels = self.channels();
        ensure!(
            !mix.is_empty()
                && mix.len().is_multiple_of(channels)
                && mix.iter().all(|sample| sample.is_finite()),
            "expected nonempty, finite channel-major separation input"
        );
        if let Model::Vr(model) = &self.model {
            let [primary, secondary] = model.separate_planar(mix, self.rate(), progress)?;
            ensure!(
                primary.channels == self.output_channels()
                    && secondary.channels == primary.channels
                    && primary.samples.len() == secondary.samples.len()
                    && self.training.instruments == [primary.name, secondary.name],
                "inconsistent VR stem metadata"
            );
            let stem_len = primary.samples.len();
            let mut estimates = primary.samples;
            estimates.extend(secondary.samples);
            let mut audio = Separated {
                estimates,
                sources: vec![0, 1],
                stem_len,
                remaining: None,
            };
            if self.normalize {
                audio.normalize(&self.training.outputs())?;
            }
            return Ok(audio);
        }
        let prepared = self
            .standardize
            .then(|| Standardized::new(mix, channels))
            .transpose()?;
        let input = prepared
            .as_ref()
            .map_or(mix, |input| input.audio.as_slice());
        let mut sources = if matches!(self.model, Model::LegacyDemucs(_)) {
            (0..self.model.stems()).collect::<Vec<_>>()
        } else {
            selections
                .iter()
                .map(|selection| match selection.source {
                    Source::Prediction(index) => index,
                    Source::Residual => 0,
                })
                .collect()
        };
        sources.sort_unstable();
        sources.dedup();
        let runs = if self.enable_tta { 3 } else { 1 };
        let mut run = |buffer: &[f32], index: usize| {
            self.predict_track(buffer, &sources, |status| {
                progress(combine_progress(status, index, runs))
            })
        };
        let mut estimates = run(input, 0)?;
        if self.enable_tta {
            let frames = input.len() / channels;
            let mut variant: Vec<_> = input
                .chunks_exact(frames)
                .rev()
                .flatten()
                .copied()
                .collect();
            let swapped = run(&variant, 1)?;
            for (base, swapped) in estimates
                .chunks_exact_mut(input.len())
                .zip(swapped.chunks_exact(input.len()))
            {
                for (sample, flipped) in base
                    .iter_mut()
                    .zip(swapped.chunks_exact(frames).rev().flatten())
                {
                    *sample += *flipped;
                }
            }
            variant
                .iter_mut()
                .zip(input)
                .for_each(|(out, sample)| *out = -*sample);
            let negative = run(&variant, 2)?;
            for (out, negative) in estimates.iter_mut().zip(negative) {
                *out = (*out - negative) / 3.0;
            }
        }
        if let Some(input) = prepared {
            input.restore(&mut estimates);
        }
        let remaining = if selections
            .iter()
            .any(|entry| matches!(entry.source, Source::Residual))
        {
            Some(
                residual(mix, &estimates[..mix.len()])
                    .context("separation changed the track duration")?,
            )
        } else {
            None
        };
        let mut audio = Separated {
            estimates,
            sources,
            remaining,
            stem_len: mix.len(),
        };
        if self.normalize {
            audio.normalize(selections)?;
        }
        Ok(audio)
    }

    fn predict_track(
        &self,
        mix: &[f32],
        sources: &[usize],
        mut progress: impl FnMut(Progress) -> ControlFlow<()>,
    ) -> Result<Vec<f32>> {
        let channels = self.channels();
        let predict = |chunk: &[f32], count| self.predict_batch(chunk, count);
        match &self.model {
            Model::LegacyDemucs(legacy) => {
                let demix =
                    |buffer: &[f32], progress: &mut dyn FnMut(Progress) -> ControlFlow<()>| {
                        let geometry = if self.legacy_split {
                            self.legacy_geometry
                        } else {
                            (Some(buffer.len() / channels), Some(0))
                        };
                        tch::autocast(self.autocast, || {
                            legacy.demix(buffer, geometry.0, geometry.1, progress)
                        })
                    };
                if self.legacy_shifts == 0 {
                    return demix(mix, &mut progress);
                }
                let frames = mix.len() / channels;
                let extent = self.rate() as usize / 2;
                let mut estimates = vec![
                    0.0;
                    mix.len()
                        .checked_mul(self.model.stems())
                        .context("Demucs output is too large")?
                ];
                for step in 0..self.legacy_shifts {
                    let random = Tensor::f_randint(
                        i64::try_from(extent + 1)?,
                        [],
                        (tch::Kind::Int64, Device::Cpu),
                    )?;
                    let leading = extent - usize::try_from(random.f_int64_value(&[])?)?;
                    let shifted_frames = frames
                        .checked_add(leading)
                        .context("Demucs input is too long")?;
                    let mut shifted = vec![
                        0.0;
                        shifted_frames
                            .checked_mul(channels)
                            .context("Demucs input is too large")?
                    ];
                    for (buffer, input) in shifted
                        .chunks_exact_mut(shifted_frames)
                        .zip(mix.chunks_exact(frames))
                    {
                        buffer[leading..].copy_from_slice(input);
                    }
                    let separated = demix(&shifted, &mut |status| {
                        progress(combine_progress(status, step, self.legacy_shifts))
                    })?;
                    for (result, plane) in estimates
                        .chunks_exact_mut(frames)
                        .zip(separated.chunks_exact(shifted_frames))
                    {
                        for (out, sample) in result.iter_mut().zip(&plane[leading..]) {
                            *out += sample;
                        }
                    }
                }
                for sample in &mut estimates {
                    *sample /= self.legacy_shifts as f32;
                }
                Ok(estimates)
            }
            Model::Demucs(_) => demix_uniform(
                mix,
                channels,
                self.model.stems(),
                sources,
                self.geometry,
                predict,
                progress,
            )
            .map_err(demix_error),
            _ => demix(
                mix,
                channels,
                self.model.stems(),
                sources,
                self.geometry,
                predict,
                progress,
            )
            .map_err(demix_error),
        }
    }

    fn predict_batch(&self, audio: &[f32], count: usize) -> Result<Vec<f32>> {
        let extra = self.replicas.len().min(count - 1);
        let amp = self.autocast;
        if extra == 0 {
            let output = tch::autocast(amp, || self.model.predict_batch(audio, count))?;
            if self.clear_mlx_cache {
                super::mlx::clear_cache()?;
            }
            return Ok(output);
        }
        let item = audio.len() / count;
        let width = count.div_ceil(extra + 1);
        let mut windows = audio.chunks(width * item);
        let first = windows.next().context("missing primary inference batch")?;
        std::thread::scope(|scope| {
            let workers: Vec<_> = self
                .replicas
                .iter()
                .zip(windows)
                .map(|(replica, chunk)| {
                    scope.spawn(move || {
                        let model = replica
                            .lock()
                            .map_err(|_| anyhow::anyhow!("PyMSS replica stopped"))?;
                        tch::autocast(amp, || model.predict_batch(chunk, chunk.len() / item))
                    })
                })
                .collect();
            let mut output = tch::autocast(amp, || self.model.predict_batch(first, width));
            for worker in workers {
                // Join all workers before returning any error so a replica can be reused.
                let result = worker
                    .join()
                    .map_err(|_| anyhow::anyhow!("PyMSS replica panicked"))
                    .and_then(|r| r);
                output = output.and_then(|mut batch| {
                    batch.extend(result?);
                    Ok(batch)
                });
            }
            output
        })
    }
}

fn combine_progress(status: Progress, index: usize, runs: usize) -> Progress {
    Progress {
        chunks_done: index
            .saturating_mul(status.chunks_total)
            .saturating_add(status.chunks_done),
        chunks_total: runs.saturating_mul(status.chunks_total),
    }
}

struct Separated {
    estimates: Vec<f32>,
    sources: Vec<usize>,
    remaining: Option<Vec<f32>>,
    stem_len: usize,
}

impl Separated {
    fn prediction_range(&self, index: usize) -> Result<std::ops::Range<usize>> {
        let slot = self
            .sources
            .binary_search(&index)
            .ok()
            .context("missing model prediction for requested stem")?;
        Ok(slot * self.stem_len..(slot + 1) * self.stem_len)
    }

    fn stem(&self, source: Source) -> Result<&[f32]> {
        match source {
            Source::Prediction(index) => Ok(&self.estimates[self.prediction_range(index)?]),
            Source::Residual => self.remaining.as_deref().context("missing residual stem"),
        }
    }

    fn normalize(&mut self, selected: &[Selection<'_>]) -> Result<()> {
        let mut peak = 0_f32;
        for selection in selected {
            for sample in self.stem(selection.source)? {
                peak = peak.max(sample.abs());
            }
        }
        if peak > 0.0 && peak.is_finite() {
            let gain = 10_f64.powf(-0.01 / 20.0) / f64::from(peak);
            for selection in selected {
                let stem = match selection.source {
                    Source::Prediction(index) => {
                        let range = self.prediction_range(index)?;
                        &mut self.estimates[range]
                    }
                    Source::Residual => self
                        .remaining
                        .as_deref_mut()
                        .context("missing residual stem")?,
                };
                for sample in stem {
                    *sample = (f64::from(*sample) * gain) as f32;
                }
            }
        }
        Ok(())
    }
}

/// Separates local input with a PyMSS model and configuration.
///
/// Paths point to local files. If `stems` is empty, every configured output
/// is saved. A single-target model uses the unprocessed mix minus its estimate
/// for the first other instrument specified in the YAML.
pub fn separate_configured(
    root: &Path,
    input: &Path,
    output: &Path,
    options: ConfiguredOptions<'_>,
    progress: impl FnMut(JobProgress) -> ControlFlow<()>,
) -> Result<Vec<ConfiguredStemFiles>> {
    separate_configured_with_params(
        root,
        input,
        output,
        options,
        &InferenceParams::default(),
        &AudioEncoding::default(),
        progress,
    )
}

pub fn separate_configured_with_params(
    root: &Path,
    input: &Path,
    output: &Path,
    options: ConfiguredOptions<'_>,
    params: &InferenceParams,
    encoding: &AudioEncoding,
    mut progress: impl FnMut(JobProgress) -> ControlFlow<()>,
) -> Result<Vec<ConfiguredStemFiles>> {
    let params = params
        .clone()
        .with_geometry(options.chunk_size, options.overlap_size);
    let model = ConfiguredModel::load_on_devices(
        options.model_type,
        options.checkpoint,
        options.config,
        options.device,
        options.additional_devices,
        &params,
    )?;
    let selections = model.training.select(options.stems)?;
    let destinations = selections
        .iter()
        .map(|stem| (stem.name, output))
        .collect::<Vec<_>>();
    let jobs = plan_outputs(
        input,
        &destinations,
        options.format,
        options.overwrite,
        options.save_as_folder,
    )?;
    let layout = model.layout;
    let mut results = Vec::with_capacity(jobs.len());
    for (index, (input, paths)) in jobs.iter().enumerate() {
        let mix = read_audio(root, input, layout)?;
        let output_layout = PcmLayout {
            channels: model.output_channels(),
            ..layout
        };
        let groups = model.stem_groups(&selections);
        let count = groups.len();
        let mut staged = Vec::with_capacity(selections.len());
        let mut destinations = paths.iter();
        for (group_index, group) in groups.enumerate() {
            let separated = model
                .separate_audio(&mix, group, |status| {
                    progress(job_progress(
                        index,
                        jobs.len(),
                        combine_progress(status, group_index, count),
                    ))
                })
                .with_context(|| format!("separating {}", input.display()))?;
            for (selection, path) in group.iter().zip(destinations.by_ref()) {
                staged.push((
                    path,
                    stage_stem(
                        root,
                        path,
                        separated.stem(selection.source)?,
                        output_layout,
                        options.format,
                        encoding,
                    )?,
                ));
            }
        }
        for (path, temporary) in staged {
            persist_stem(temporary, path, options.overwrite)?;
        }
        let stems = paths
            .iter()
            .zip(&selections)
            .map(|(path, stem)| StemFile {
                name: stem.name.to_owned(),
                path: path.clone(),
            })
            .collect();
        results.push(ConfiguredStemFiles {
            input: input.clone(),
            stems,
        });
    }
    Ok(results)
}

struct Standardized {
    audio: Vec<f32>,
    mean: f32,
    sigma: f32,
}

impl Standardized {
    fn new(audio: &[f32], channels: usize) -> Result<Self> {
        let frames = audio.len() / channels;
        let mono = |n| {
            (0..channels)
                .map(|channel| f64::from(audio[channel * frames + n]))
                .sum::<f64>()
                / channels as f64
        };
        let mean = (0..frames).map(mono).sum::<f64>() / frames as f64;
        let variance = (0..frames).map(|n| (mono(n) - mean).powi(2)).sum::<f64>() / frames as f64;
        let mean = mean as f32;
        let sigma = variance.sqrt() as f32;
        ensure!(
            sigma.is_finite(),
            "cannot standardize audio with nonfinite variance"
        );
        let audio = if sigma == 0.0 {
            vec![0.0; audio.len()]
        } else {
            audio.iter().map(|sample| (sample - mean) / sigma).collect()
        };
        Ok(Self { audio, mean, sigma })
    }

    fn restore(self, audio: &mut [f32]) {
        for sample in audio {
            *sample = *sample * self.sigma + self.mean;
        }
    }
}

pub(super) fn checkpoint_weights(
    checkpoint: &Path,
    device: Device,
) -> Result<HashMap<String, Tensor>> {
    let mut checkpoint = Checkpoint::open(checkpoint)?;
    let section = ["state", "state_dict", "model_state_dict", "model"]
        .into_iter()
        .find(|key| checkpoint.has(key));
    checkpoint.weights(section, device)
}

pub(super) fn load_document(path: &Path) -> Result<Value> {
    let file = std::fs::File::open(path)
        .with_context(|| format!("opening PyMSS config {}", path.display()))?;
    let mut yaml: serde_yaml::Value = serde_yaml::from_reader(file)
        .with_context(|| format!("parsing PyMSS config {}", path.display()))?;
    yaml.apply_merge().context("invalid PyMSS YAML merge")?;
    serde_json::to_value(strip_tags(yaml)?).context("invalid PyMSS YAML values")
}

fn strip_tags(value: serde_yaml::Value) -> Result<serde_yaml::Value> {
    use serde_yaml::Value as Yaml;
    match value {
        Yaml::Tagged(tagged) => {
            ensure!(
                tagged.tag == "python/tuple" || tagged.tag == "tag:yaml.org,2002:python/tuple",
                "unsupported PyMSS YAML tag {}",
                tagged.tag
            );
            ensure!(
                matches!(tagged.value, Yaml::Sequence(_)),
                "PyMSS tuple tag must contain a sequence"
            );
            strip_tags(tagged.value)
        }
        Yaml::Sequence(items) => Ok(Yaml::Sequence(
            items.into_iter().map(strip_tags).collect::<Result<_>>()?,
        )),
        Yaml::Mapping(items) => Ok(Yaml::Mapping(
            items
                .into_iter()
                .map(|(key, value)| Ok((strip_tags(key)?, strip_tags(value)?)))
                .collect::<Result<_>>()?,
        )),
        value => Ok(value),
    }
}
