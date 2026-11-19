//! Rust inference for the BS-RoFormer and Mel-Band-RoFormer models.
//!
//! The layer names, attention, spectral transforms and mask handling follow
//! PyMSS Core’s original BS-RoFormer implementation. See `licenses/pymss-core.txt`.

#[path = "hyperace.rs"]
mod hyperace;

use std::collections::HashMap;

use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use tch::{Device, Kind, Tensor, no_grad};

pub(super) mod cuda_sdpa;
mod mlx_network;
mod mlx_spectral;
#[cfg(feature = "onnx")]
mod onnx_network;

pub(super) use mlx_network::AttentionMode as MpsAttention;
use mlx_network::MlxNetwork;

use super::{Acceleration, SAMPLE_RATE};
use hyperace::{MaskMode, Segmentation};

#[derive(Debug)]
pub enum Bands {
    Split(Vec<i64>),
    Mel { count: i64, sample_rate: u32 },
}

#[derive(Debug)]
pub struct ModelConfig {
    pub bands: Bands,
    pub channels: i64,
    pub stems: i64,
    pub embedding_dim: i64,
    pub depth: usize,
    pub time_depth: usize,
    pub frequency_depth: usize,
    pub heads: i64,
    pub head_dim: i64,
    pub flash_attention: bool,
    pub fft: i64,
    pub hop: i64,
    pub window: i64,
    pub normalized_stft: bool,
    pub zero_dc: bool,
    pub match_input_length: bool,
}

pub(super) fn requests_half(value: &serde_json::Value) -> bool {
    matches!(value.as_str(), Some("float16" | "fp16" | "torch.float16"))
}

#[derive(Deserialize)]
struct ModelFields {
    dim: i64,
    depth: usize,
    #[serde(default)]
    stereo: bool,
    num_stems: Option<i64>,
    time_transformer_depth: Option<usize>,
    freq_transformer_depth: Option<usize>,
    heads: Option<i64>,
    dim_head: Option<i64>,
    flash_attn: Option<bool>,
    stft_n_fft: Option<i64>,
    stft_hop_length: Option<i64>,
    stft_win_length: Option<i64>,
    #[serde(default)]
    stft_normalized: bool,
    freqs_per_bands: Option<Vec<i64>>,
    num_bands: Option<i64>,
    sample_rate: Option<u32>,
    zero_dc: Option<bool>,
    #[serde(default)]
    match_input_audio_length: bool,
}

impl ModelConfig {
    pub fn from_document(document: &serde_json::Value, mel: bool) -> Result<Self> {
        let fields = ModelFields::deserialize(&document["model"])
            .context("invalid RoFormer model configuration")?;
        let bands = if mel {
            Bands::Mel {
                count: fields.num_bands.unwrap_or(60),
                sample_rate: fields.sample_rate.unwrap_or(SAMPLE_RATE),
            }
        } else {
            Bands::Split(fields.freqs_per_bands.unwrap_or_else(default_split_bands))
        };
        Ok(Self {
            bands,
            channels: if fields.stereo { 2 } else { 1 },
            stems: fields.num_stems.unwrap_or(1),
            embedding_dim: fields.dim,
            depth: fields.depth,
            time_depth: fields.time_transformer_depth.unwrap_or(2),
            frequency_depth: fields.freq_transformer_depth.unwrap_or(2),
            heads: fields.heads.unwrap_or(8),
            head_dim: fields.dim_head.unwrap_or(64),
            flash_attention: fields.flash_attn.unwrap_or(true),
            fft: fields.stft_n_fft.unwrap_or(2048),
            hop: fields.stft_hop_length.unwrap_or(512),
            window: fields.stft_win_length.unwrap_or(2048),
            normalized_stft: fields.stft_normalized,
            zero_dc: mel && fields.zero_dc.unwrap_or(true),
            match_input_length: !mel || fields.match_input_audio_length,
        })
    }

    fn validate(&self) -> Result<()> {
        ensure!(
            matches!(self.channels, 1 | 2)
                && (1..=128).contains(&self.stems)
                && (1..=8192).contains(&self.embedding_dim)
                && (1..=128).contains(&self.depth)
                && self.time_depth <= 128
                && self.frequency_depth <= 128
                && (1..=128).contains(&self.heads)
                && (2..=1024).contains(&self.head_dim)
                && self.head_dim % 2 == 0
                && (2..=65536).contains(&self.fft)
                && (1..=self.fft).contains(&self.window)
                && (1..=self.window).contains(&self.hop),
            "invalid RoFormer model geometry"
        );
        Ok(())
    }
}

fn default_split_bands() -> Vec<i64> {
    [
        (2, 24),
        (4, 12),
        (12, 8),
        (24, 8),
        (48, 8),
        (128, 1),
        (129, 1),
    ]
    .into_iter()
    .flat_map(|(width, count)| std::iter::repeat_n(width, count))
    .collect()
}

struct Weights<'a> {
    tensors: &'a HashMap<String, Tensor>,
    device: Device,
    cuda_attention: cuda_sdpa::Policy,
    mps_attention: Option<mlx_network::BridgeSettings>,
}

impl Weights<'_> {
    fn tensor(&self, key: &str, shape: &[i64]) -> Result<Tensor> {
        let tensor = self
            .tensors
            .get(key)
            .with_context(|| format!("missing RoFormer weight {key}"))?;
        ensure!(
            tensor.size() == shape,
            "RoFormer weight {key} has shape {:?}, expected {shape:?}",
            tensor.size()
        );
        ensure!(
            matches!(tensor.kind(), Kind::Float | Kind::Half | Kind::BFloat16),
            "unsupported RoFormer dtype for {key}"
        );
        ensure!(
            tensor.f_isfinite()?.f_all()?.f_int64_value(&[])? != 0,
            "nonfinite RoFormer weight {key}"
        );
        Ok(tensor.f_to_device(self.device)?.f_to_kind(Kind::Float)?)
    }

    fn linear(&self, key: &str, input: i64, output: i64, bias: bool) -> Result<Linear> {
        let weight = self.tensor(&format!("{key}.weight"), &[output, input])?;
        let bias = bias
            .then(|| self.tensor(&format!("{key}.bias"), &[output]))
            .transpose()?;
        Ok(Linear { weight, bias })
    }
}

struct Linear {
    weight: Tensor,
    bias: Option<Tensor>,
}

impl Linear {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        Ok(input.f_linear(&self.weight, self.bias.as_ref())?)
    }
}

fn normalize(input: &Tensor, gamma: &Tensor) -> Result<Tensor> {
    let dimension = gamma.size()[0] as f64;
    let length = input
        .f_norm_scalaropt_dim(2.0, [-1], true)?
        .f_clamp_min(1e-12)?;
    Ok(input
        .f_div(&length)?
        .f_mul_scalar(dimension.sqrt())?
        .f_mul(gamma)?)
}

struct Attention {
    gamma: Tensor,
    frequencies: Tensor,
    qkv: Linear,
    gates: Linear,
    output: Linear,
    heads: i64,
    head_dim: i64,
    flash: bool,
    cuda_attention: cuda_sdpa::Backend,
}

impl Attention {
    fn load(weights: &Weights<'_>, key: &str, config: &ModelConfig) -> Result<Self> {
        let ModelConfig {
            heads,
            head_dim,
            embedding_dim: dim,
            ..
        } = *config;
        let inner = heads * head_dim;
        Ok(Self {
            gamma: weights.tensor(&format!("{key}.norm.gamma"), &[dim])?,
            frequencies: weights.tensor(&format!("{key}.rotary_embed.freqs"), &[head_dim / 2])?,
            qkv: weights.linear(
                &format!("{key}.to_qkv"),
                dim,
                3 * inner,
                weights.tensors.contains_key(&format!("{key}.to_qkv.bias")),
            )?,
            gates: weights.linear(&format!("{key}.to_gates"), dim, heads, true)?,
            output: weights.linear(
                &format!("{key}.to_out.0"),
                inner,
                dim,
                weights
                    .tensors
                    .contains_key(&format!("{key}.to_out.0.bias")),
            )?,
            heads,
            head_dim,
            flash: config.flash_attention,
            cuda_attention: cuda_sdpa::Backend::new(weights.cuda_attention),
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.forward_with(input, |query, key, value| {
            self.cuda_attention.forward(query, key, value)
        })
    }

    fn forward_with(
        &self,
        input: &Tensor,
        attend: impl FnOnce(&Tensor, &Tensor, &Tensor) -> Result<Tensor>,
    ) -> Result<Tensor> {
        let input = normalize(input, &self.gamma)?;
        let (batch, length, _) = input.size3()?;
        let qkv =
            self.qkv
                .forward(&input)?
                .f_reshape([batch, length, 3, self.heads, self.head_dim])?;
        let positions = Tensor::f_arange(length, (Kind::Float, input.device()))?;
        let angles = positions
            .f_reshape([1, length, 1, 1])?
            .f_mul(&self.frequencies)?;
        let kind = qkv.kind();
        let (cosine, sine) = (
            angles.f_cos()?.f_to_kind(kind)?,
            angles.f_sin()?.f_to_kind(kind)?,
        );
        let query = rotate(&qkv.f_select(2, 0)?, &cosine, &sine)?.f_transpose(1, 2)?;
        let key = rotate(&qkv.f_select(2, 1)?, &cosine, &sine)?.f_transpose(1, 2)?;
        let value = qkv.f_select(2, 2)?.f_transpose(1, 2)?;
        let attended = if self.flash {
            attend(&query, &key, &value)?
        } else {
            let scale = (self.head_dim as f64).powf(-0.5);
            query
                .f_matmul(&key.f_transpose(-2, -1)?)?
                .f_mul_scalar(scale)?
                .f_softmax(-1, None)?
                .f_matmul(&value)?
        };
        let gate = self.gates.forward(&input)?.f_unsqueeze(-1)?.f_sigmoid()?;
        self.output.forward(
            &attended
                .f_transpose(1, 2)?
                .f_mul(&gate)?
                .f_flatten(-2, -1)?,
        )
    }
}

fn rotate(input: &Tensor, cosine: &Tensor, sine: &Tensor) -> Result<Tensor> {
    let even = input.f_slice(-1, 0, None, 2)?;
    let odd = input.f_slice(-1, 1, None, 2)?;
    let real = even.f_mul(cosine)?.f_sub(&odd.f_mul(sine)?)?;
    let imaginary = odd.f_mul(cosine)?.f_add(&even.f_mul(sine)?)?;
    Ok(Tensor::f_stack(&[real, imaginary], -1)?.f_flatten(-2, -1)?)
}

struct AttentionBlock {
    attention: Attention,
    gamma: Tensor,
    expand: Linear,
    contract: Linear,
}

impl AttentionBlock {
    fn load(weights: &Weights<'_>, key: &str, config: &ModelConfig) -> Result<Self> {
        let dim = config.embedding_dim;
        Ok(Self {
            attention: Attention::load(weights, &format!("{key}.0"), config)?,
            gamma: weights.tensor(&format!("{key}.1.net.0.gamma"), &[dim])?,
            expand: weights.linear(&format!("{key}.1.net.1"), dim, 4 * dim, true)?,
            contract: weights.linear(&format!("{key}.1.net.4"), 4 * dim, dim, true)?,
        })
    }

    fn feed_forward(&self, input: &Tensor, attended: &Tensor) -> Result<Tensor> {
        let residual = input.f_add(attended)?;
        let expanded = self.expand.forward(&normalize(&residual, &self.gamma)?)?;
        let projected = self.contract.forward(&expanded.f_gelu("none")?)?;
        Ok(residual.f_add(&projected)?)
    }
}

struct Transformer {
    blocks: Vec<AttentionBlock>,
    gamma: Option<Tensor>,
    mlx: Option<mlx_network::MlxBridge>,
}

impl Transformer {
    fn load(
        weights: &Weights<'_>,
        key: &str,
        config: &ModelConfig,
        depth: usize,
        normalize_output: bool,
    ) -> Result<Self> {
        let blocks = (0..depth)
            .map(|index| AttentionBlock::load(weights, &format!("{key}.layers.{index}"), config))
            .collect::<Result<Vec<_>>>()?;
        let gamma = normalize_output
            .then(|| weights.tensor(&format!("{key}.norm.gamma"), &[config.embedding_dim]))
            .transpose()?;
        let mut transformer = Self {
            blocks,
            gamma,
            mlx: None,
        };
        transformer.mlx = weights
            .mps_attention
            .map(|settings| mlx_network::MlxBridge::load(&transformer, weights.device, settings))
            .transpose()?;
        Ok(transformer)
    }

    fn forward(&self, value: Tensor) -> Result<Tensor> {
        if let Some(mlx) = self.mlx.as_ref().filter(|mlx| mlx.accepts(&value)) {
            return mlx.forward(self, value);
        }
        self.forward_blocks(value, |_, block, input| block.forward(input))
    }

    fn forward_blocks(
        &self,
        mut value: Tensor,
        mut attend: impl FnMut(usize, &Attention, &Tensor) -> Result<Tensor>,
    ) -> Result<Tensor> {
        for (index, block) in self.blocks.iter().enumerate() {
            let attended = attend(index, &block.attention, &value)?;
            value = block.feed_forward(&value, &attended)?;
        }
        if let Some(gamma) = &self.gamma {
            value = normalize(&value, gamma)?;
        }
        Ok(value)
    }
}

struct MaskMlp(Vec<Linear>);

impl MaskMlp {
    fn load(weights: &Weights<'_>, key: &str, dim: i64, frequency_width: i64) -> Result<Self> {
        let mut input_size = dim;
        let mut layers = Vec::new();
        for index in (0..=32).step_by(2) {
            let prefix = format!("{key}.{index}");
            let [output_size, _] = weights
                .tensors
                .get(&format!("{prefix}.weight"))
                .with_context(|| format!("missing RoFormer MLP weight {prefix}.weight"))?
                .size()[..]
            else {
                anyhow::bail!("invalid RoFormer MLP weight {prefix}.weight");
            };
            layers.push(weights.linear(&prefix, input_size, output_size, true)?);
            if !weights
                .tensors
                .contains_key(&format!("{key}.{}.weight", index + 2))
            {
                ensure!(
                    output_size == 2 * frequency_width,
                    "invalid RoFormer mask width at {key}"
                );
                return Ok(Self(layers));
            }
            input_size = output_size;
        }
        anyhow::bail!("RoFormer mask estimator exceeds maximum supported depth");
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let (last, hidden) = self
            .0
            .split_last()
            .context("empty RoFormer mask estimator")?;
        let mut value = input.shallow_clone();
        for layer in hidden {
            value = layer.forward(&value)?.f_tanh()?;
        }
        Ok(last.forward(&value)?.f_glu(-1)?)
    }
}

struct BandProjection {
    width: i64,
    gamma: Tensor,
    linear: Linear,
}

struct MelSelection {
    indices: Tensor,
    coverage: Tensor,
}

pub struct Roformer {
    config: ModelConfig,
    device: Device,
    spectral_device: Device,
    network: Network,
    mel: Option<MelSelection>,
    window: Tensor,
}

enum Network {
    Torch(TorchNetwork),
    Mlx(MlxNetwork),
    #[cfg(feature = "onnx")]
    Onnx(onnx_network::Network),
}

struct TorchNetwork {
    projections: Vec<BandProjection>,
    blocks: Vec<[Transformer; 2]>,
    final_gamma: Option<Tensor>,
    masks: Vec<Vec<MaskMlp>>,
    mask_mode: MaskMode,
    segmentation: Option<Vec<Segmentation>>,
}

impl Roformer {
    pub fn load(
        config: ModelConfig,
        tensors: &HashMap<String, Tensor>,
        device: Device,
    ) -> Result<Self> {
        Self::load_with_masks(
            config,
            tensors,
            device,
            MaskMode::Standard,
            &serde_json::Value::Null,
        )
    }

    pub(super) fn load_configured(
        document: &serde_json::Value,
        mel: bool,
        tensors: &HashMap<String, Tensor>,
        device: Device,
    ) -> Result<Self> {
        Self::load_with_masks(
            ModelConfig::from_document(document, mel)?,
            tensors,
            device,
            MaskMode::Standard,
            document,
        )
    }

    pub fn load_hyperace(
        document: &serde_json::Value,
        tensors: &HashMap<String, Tensor>,
        device: Device,
    ) -> Result<Self> {
        let config = ModelConfig::from_document(document, false)?;
        let mode = document["inference"]
            .get("mask_mode")
            .map(MaskMode::deserialize)
            .transpose()
            .context("invalid HyperACE inference.mask_mode")?
            .unwrap_or_default();
        Self::load_with_masks(config, tensors, device, mode, document)
    }

    fn load_with_masks(
        config: ModelConfig,
        tensors: &HashMap<String, Tensor>,
        device: Device,
        mask_mode: MaskMode,
        document: &serde_json::Value,
    ) -> Result<Self> {
        config.validate()?;
        let inference = &document["inference"];
        let mps_mode = MpsAttention::parse(&inference["mps_attention_backend"])?;
        let backend = if device == Device::Mps
            && mps_mode.is_some()
            && inference["mps_model_backend"].is_null()
            && inference["onnx_provider"].is_null()
        {
            Acceleration::Torch
        } else {
            Acceleration::configured(document, device, true, "RoFormer")?
        };
        let mps_attention = if device == Device::Mps && matches!(backend, Acceleration::Torch) {
            mps_mode
                .map(|mode| {
                    let minimum = &inference["mps_mlx_min_tokens"];
                    let minimum = if minimum.is_null() {
                        128
                    } else {
                        minimum
                            .as_i64()
                            .filter(|value| *value >= 0)
                            .context("mps_mlx_min_tokens must be a nonnegative integer")?
                    };
                    let dtype =
                        super::mlx::compute_dtype(&inference["mps_model_compute_dtype"], device)?;
                    let dtype = if mask_mode == MaskMode::Standard {
                        dtype
                    } else {
                        super::mlx::DType::F32
                    };
                    Ok::<_, anyhow::Error>(mlx_network::BridgeSettings {
                        mode,
                        minimum,
                        dtype,
                    })
                })
                .transpose()?
        } else {
            None
        };
        let half = requests_half(&document["inference"]["model_dtype"]);
        if half {
            #[cfg(not(feature = "onnx"))]
            anyhow::bail!("RoFormer model_dtype=float16 requires the onnx build feature");
            #[cfg(feature = "onnx")]
            ensure!(
                matches!(backend, Acceleration::Onnx(_)),
                "RoFormer model_dtype=float16 requires onnx_provider"
            );
        }
        let spectral_device = if device == Device::Mps {
            Device::Cpu
        } else {
            device
        };
        let weight_device = backend.weight_device(device);
        let weights = Weights {
            tensors,
            device: weight_device,
            cuda_attention: cuda_sdpa::Policy::parse(
                &document["inference"]["cuda_attention_backend"],
            )?,
            mps_attention,
        };
        let ModelConfig {
            embedding_dim: dim,
            channels,
            stems,
            fft,
            ..
        } = config;
        let bins = fft / 2 + 1;

        let (widths, mel) = match &config.bands {
            Bands::Split(widths) => {
                ensure!(
                    widths.len() > 1
                        && widths.iter().all(|width| (1..=bins).contains(width))
                        && widths.iter().sum::<i64>() == bins,
                    "RoFormer band widths must partition the FFT bins"
                );
                (widths.clone(), None)
            }
            Bands::Mel { count, sample_rate } => {
                let layout = mel_bands(*count, *sample_rate, bins)?;
                let indices = layout
                    .indices
                    .into_iter()
                    .flat_map(|bin| (0..channels).map(move |channel| bin * channels + channel))
                    .collect::<Vec<_>>();
                let coverage = layout
                    .coverage
                    .into_iter()
                    .flat_map(|count| (0..channels).map(move |_| count as f32))
                    .collect::<Vec<_>>();
                let selection = MelSelection {
                    indices: Tensor::f_from_slice(&indices)?.f_to_device(spectral_device)?,
                    coverage: Tensor::f_from_slice(&coverage)?
                        .f_to_device(spectral_device)?
                        .f_reshape([1, -1, 1])?,
                };
                (layout.widths, Some(selection))
            }
        };

        let mut masks: Vec<Vec<MaskMlp>> = (0..stems)
            .map(|_| Vec::with_capacity(widths.len()))
            .collect();
        let segmentation = if mask_mode == MaskMode::Standard {
            None
        } else {
            ensure!(
                (bins * channels) % 2 == 0,
                "HyperACE segmentation requires a real/imaginary mask width divisible by four"
            );
            Some(
                (0..stems)
                    .map(|stem| {
                        Segmentation::load(
                            &weights,
                            &format!("mask_estimators.{stem}.segm"),
                            dim,
                            widths.len() as i64,
                            bins * channels / 2,
                        )
                    })
                    .collect::<Result<_>>()?,
            )
        };
        let mut projections = Vec::with_capacity(widths.len());
        for (index, bins_in_band) in widths.into_iter().enumerate() {
            let width = 2 * channels * bins_in_band;
            let key = format!("band_split.to_features.{index}");
            projections.push(BandProjection {
                width,
                gamma: weights.tensor(&format!("{key}.0.gamma"), &[width])?,
                linear: weights.linear(&format!("{key}.1"), width, dim, true)?,
            });
            for (stem, estimators) in masks.iter_mut().enumerate() {
                estimators.push(MaskMlp::load(
                    &weights,
                    &format!("mask_estimators.{stem}.to_freqs.{index}.0"),
                    dim,
                    width,
                )?);
            }
        }
        let mut blocks = Vec::with_capacity(config.depth);
        for index in 0..config.depth {
            let time = Transformer::load(
                &weights,
                &format!("layers.{index}.0"),
                &config,
                config.time_depth,
                mel.is_some(),
            )?;
            let frequency = Transformer::load(
                &weights,
                &format!("layers.{index}.1"),
                &config,
                config.frequency_depth,
                mel.is_some(),
            )?;
            blocks.push([time, frequency]);
        }
        let final_gamma = mel
            .is_none()
            .then(|| weights.tensor("final_norm.gamma", &[dim]))
            .transpose()?;
        let window =
            Tensor::f_hann_window_periodic(config.window, true, (Kind::Float, spectral_device))?;
        let torch = TorchNetwork {
            projections,
            blocks,
            final_gamma,
            masks,
            mask_mode,
            segmentation,
        };
        let network = match backend {
            Acceleration::Torch => Network::Torch(torch),
            Acceleration::Mlx(dtype) => Network::Mlx(MlxNetwork::load(
                &torch,
                device,
                dtype,
                &config,
                mel.as_ref(),
                &window,
            )?),
            #[cfg(feature = "onnx")]
            Acceleration::Onnx(provider) => {
                // Keep the full segmentation cascade at the same precision as MLX.
                Network::Onnx(onnx_network::Network::load(
                    provider,
                    &torch,
                    dim,
                    half && mask_mode == MaskMode::Standard,
                )?)
            }
        };
        Ok(Self {
            config,
            device: weight_device,
            spectral_device,
            network,
            mel,
            window,
        })
    }

    pub fn channels(&self) -> usize {
        self.config.channels as usize
    }

    pub fn num_stems(&self) -> usize {
        self.config.stems as usize
    }

    pub(super) fn predict_batch(&self, audio: &[f32], count: usize) -> Result<Vec<f32>> {
        let channels = self.channels();
        let samples = super::window_samples(audio, count, channels)?;
        ensure!(
            samples > self.config.fft / 2,
            "RoFormer input must exceed half the FFT size"
        );
        let samples = match &self.network {
            Network::Torch(network) => no_grad(|| {
                self.forward_spectrum(audio, count as i64, samples, |input, frames| {
                    network.forward(input, count as i64, frames, self.config.embedding_dim)
                })
            })?,
            #[cfg(feature = "onnx")]
            Network::Onnx(network) => no_grad(|| {
                self.forward_spectrum(audio, count as i64, samples, |input, _| {
                    network.estimate(input)
                })
            })?,
            Network::Mlx(network) => network.predict(audio, count, samples as usize)?,
        };
        ensure!(
            samples.iter().all(|x| x.is_finite()),
            "nonfinite samples in RoFormer output"
        );
        Ok(samples)
    }

    fn forward_spectrum(
        &self,
        audio: &[f32],
        batch: i64,
        samples: i64,
        estimate: impl FnOnce(&Tensor, i64) -> Result<Tensor>,
    ) -> Result<Vec<f32>> {
        let ModelConfig {
            channels,
            stems,
            fft,
            hop,
            window,
            normalized_stft,
            zero_dc,
            match_input_length,
            ..
        } = self.config;
        let bins = fft / 2 + 1;
        // Reflection followed by uncentered STFT is PyTorch's centered reflect
        // mode. The tch binding otherwise also supplies align_to_window,
        // which the LibTorch API disallows with center = true.
        let padded = Tensor::f_from_slice(audio)?
            .f_reshape([-1, samples])?
            .f_to_device(self.spectral_device)?
            .f_reflection_pad1d([fft / 2, fft / 2])?;
        let complex = padded.f_stft_center(
            fft,
            hop,
            window,
            Some(&self.window),
            false,
            "reflect",
            normalized_stft,
            true,
            true,
            false,
        )?;
        let frames = complex.size()[2];
        let spectrum = complex
            .f_reshape([batch, channels, bins, frames])?
            .f_transpose(1, 2)?
            .f_reshape([batch, bins * channels, frames])?;
        let selected = if let Some(mel) = &self.mel {
            spectrum.f_index_select(1, &mel.indices)?
        } else {
            spectrum.shallow_clone()
        };
        let selected = selected
            .f_view_as_real()?
            .f_permute([0, 2, 1, 3])?
            .f_reshape([batch, frames, -1])?
            .f_to_device(self.device)?;
        let masks = estimate(&selected, frames)?
            .f_to_device(self.spectral_device)?
            .f_reshape([batch, stems, frames, -1, 2])?
            .f_view_as_complex()?
            .f_transpose(2, 3)?;
        let weight = if let Some(mel) = &self.mel {
            Tensor::f_zeros(
                [batch, stems, bins * channels, frames],
                (masks.kind(), self.spectral_device),
            )?
            .f_index_add(2, &mel.indices, &masks)?
            .f_div(&mel.coverage)?
        } else {
            masks
        };
        let complex = spectrum
            .f_unsqueeze(1)?
            .f_mul(&weight)?
            .f_reshape([batch, stems, bins, channels, frames])?
            .f_transpose(2, 3)?
            .f_reshape([batch * stems * channels, bins, frames])?;
        let complex = if zero_dc {
            let dc_bin = Tensor::f_from_slice(&[0i64])?.f_to_device(self.spectral_device)?;
            complex.f_index_fill(1, &dc_bin, 0.0)?
        } else {
            complex
        };
        let output = complex
            .f_istft(
                fft,
                hop,
                window,
                Some(&self.window),
                true,
                normalized_stft,
                true,
                match_input_length.then_some(samples),
                false,
            )?
            .f_to_device(Device::Cpu)?
            .f_reshape([-1])?;
        Ok(Vec::<f32>::try_from(output)?)
    }
}

impl TorchNetwork {
    fn forward(&self, selected: &Tensor, batch: i64, frames: i64, dim: i64) -> Result<Tensor> {
        let mut mapped = Vec::with_capacity(self.projections.len());
        let mut offset = 0;
        for projection in &self.projections {
            let slice = selected.f_narrow(-1, offset, projection.width)?;
            mapped.push(
                projection
                    .linear
                    .forward(&normalize(&slice, &projection.gamma)?)?,
            );
            offset += projection.width;
        }
        let mut embeddings = Tensor::f_stack(&mapped, 1)?;
        let bands = embeddings.size()[1];
        for [time, frequency] in &self.blocks {
            embeddings = time
                .forward(embeddings.f_reshape([batch * bands, frames, dim])?)?
                .f_reshape([batch, bands, frames, dim])?
                .f_transpose(1, 2)?;
            embeddings = frequency
                .forward(embeddings.f_reshape([batch * frames, bands, dim])?)?
                .f_reshape([batch, frames, bands, dim])?
                .f_transpose(1, 2)?;
        }
        if let Some(gamma) = &self.final_gamma {
            embeddings = normalize(&embeddings, gamma)?;
        }
        ensure!(
            embeddings.size() == [batch, bands, frames, dim],
            "invalid RoFormer embedding dimensions"
        );
        let segmentations = self
            .segmentation
            .as_ref()
            .map(|estimators| {
                let input = embeddings.f_permute([0, 3, 2, 1])?;
                estimators
                    .iter()
                    .map(|estimator| estimator.forward(&input))
                    .collect::<Result<Vec<_>>>()
            })
            .transpose()?;
        let estimates = if self.mask_mode == MaskMode::SegmentationOnly {
            segmentations.context("missing HyperACE segmentation masks")?
        } else {
            self.mlp_estimates(&embeddings, segmentations.as_deref())?
        };
        Ok(Tensor::f_stack(&estimates, 1)?)
    }

    fn mlp_estimates(
        &self,
        embeddings: &Tensor,
        segmentations: Option<&[Tensor]>,
    ) -> Result<Vec<Tensor>> {
        let mut estimates = Vec::with_capacity(self.masks.len());
        for (stem, model_stem) in self.masks.iter().enumerate() {
            let mut masks = Vec::with_capacity(model_stem.len());
            for (index, estimator) in model_stem.iter().enumerate() {
                masks.push(estimator.forward(&embeddings.f_select(1, index as i64)?)?);
            }
            let mask = Tensor::f_cat(&masks, -1)?;
            estimates.push(if let Some(segmentations) = segmentations {
                mask.f_add(&segmentations[stem])?
            } else {
                mask
            });
        }
        Ok(estimates)
    }
}

struct MelBands {
    indices: Vec<i64>,
    widths: Vec<i64>,
    coverage: Vec<i64>,
}

fn mel_bands(count: i64, sample_rate: u32, bins: i64) -> Result<MelBands> {
    ensure!(
        count >= 2 && count <= bins && sample_rate > 0,
        "invalid RoFormer mel-band geometry"
    );
    let linear_scale = 200.0 / 3.0;
    let transition_mel = 1000.0 / linear_scale;
    let log_step = 6.4_f64.ln() / 27.0;
    let limit_hz = f64::from(sample_rate) / 2.0;
    let limit_mel = if limit_hz >= 1000.0 {
        transition_mel + (limit_hz / 1000.0).ln() / log_step
    } else {
        limit_hz / linear_scale
    };
    let edges: Vec<f64> = (0..count + 2)
        .map(|index| {
            let mel = index as f64 * (limit_mel / (count + 1) as f64);
            if mel >= transition_mel {
                1000.0 * (log_step * (mel - transition_mel)).exp()
            } else {
                linear_scale * mel
            }
        })
        .collect();

    let mut indices = Vec::new();
    let mut widths = Vec::with_capacity(count as usize);
    let mut coverage = vec![0i64; bins as usize];
    for (band, edges) in edges.windows(3).enumerate() {
        let prior = indices.len();
        for bin in 0..bins {
            let hz = bin as f64 * (limit_hz / (bins - 1) as f64);
            if (hz > edges[0] && hz < edges[2])
                || (band == 0 && bin == 0)
                || (band + 1 == count as usize && bin == bins - 1)
            {
                indices.push(bin);
                coverage[bin as usize] += 1;
            }
        }
        ensure!(indices.len() > prior, "RoFormer mel band {band} is empty");
        widths.push((indices.len() - prior) as i64);
    }
    ensure!(
        coverage.iter().all(|&count| count > 0),
        "RoFormer mel bands leave FFT bins uncovered"
    );
    Ok(MelBands {
        indices,
        widths,
        coverage,
    })
}
