//! MDX23C inference with the PyMSS TFC-TDF v3 network and subband STFT.
//! See `licenses/pymss-core.txt`.

use std::collections::HashMap;

use anyhow::{Context, Result, bail, ensure};
use serde::Deserialize;
use tch::{Device, Kind, Tensor, no_grad};

use super::mlx;

mod mlx_network;
#[cfg(feature = "onnx")]
mod onnx_network;

#[derive(Deserialize, Debug)]
pub struct ModelConfig {
    audio: AudioConfig,
    model: NetworkConfig,
    training: StemConfig,
}

#[derive(Deserialize, Debug)]
struct AudioConfig {
    n_fft: i64,
    hop_length: i64,
    dim_f: i64,
    num_channels: i64,
}

#[derive(Deserialize, Debug)]
struct NetworkConfig {
    norm: String,
    act: String,
    num_subbands: i64,
    num_scales: usize,
    // The network transposes the spectrum so this tuple is (time, frequency).
    scale: [i64; 2],
    num_blocks_per_scale: usize,
    num_channels: i64,
    growth: i64,
    bottleneck_factor: i64,
}

#[derive(Deserialize, Debug)]
struct StemConfig {
    target_instrument: Option<String>,
    #[serde(default)]
    instruments: Vec<String>,
}

impl ModelConfig {
    pub fn from_document(document: &serde_json::Value) -> Result<Self> {
        Self::deserialize(document).context("invalid MDX23C audio, model or training configuration")
    }

    fn stems(&self) -> usize {
        if self
            .training
            .target_instrument
            .as_deref()
            .is_some_and(|name| !name.is_empty())
        {
            1
        } else {
            self.training.instruments.len()
        }
    }

    fn validate(&self) -> Result<i64> {
        let audio = &self.audio;
        let net = &self.model;
        ensure!(
            (2..=65_536).contains(&audio.n_fft)
                && (1..=audio.n_fft).contains(&audio.hop_length)
                && (1..=audio.n_fft / 2 + 1).contains(&audio.dim_f)
                && matches!(audio.num_channels, 1 | 2)
                && (1..=128).contains(&self.stems()),
            "invalid MDX23C audio or output geometry"
        );
        ensure!(
            (1..=audio.dim_f).contains(&net.num_subbands)
                && audio.dim_f % net.num_subbands == 0
                && net.num_scales <= 16
                && (1..=128).contains(&net.num_blocks_per_scale)
                && net.scale.iter().all(|scale| (1..=128).contains(scale))
                && (1..=8192).contains(&net.num_channels)
                && (-8192..=8192).contains(&net.growth)
                && net.bottleneck_factor > 0,
            "invalid MDX23C network geometry"
        );
        let mut channels = net.num_channels;
        let mut width = audio.dim_f / net.num_subbands;
        for _ in 0..net.num_scales {
            ensure!(
                width % net.scale[1] == 0,
                "MDX23C frequency stride does not divide the subband width"
            );
            width /= net.scale[1];
            channels += net.growth;
            ensure!(
                (1..=8192).contains(&channels),
                "invalid MDX23C scale channel count"
            );
        }
        ensure!(
            width >= net.bottleneck_factor,
            "MDX23C bottleneck width is too small"
        );
        net.scale[0]
            .checked_pow(net.num_scales as u32)
            .context("MDX23C time stride overflow")
    }
}

#[derive(Clone, Copy)]
enum Activation {
    Gelu,
    Relu,
    Elu(f64),
}

impl Activation {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "gelu" => Ok(Self::Gelu),
            "relu" => Ok(Self::Relu),
            value if value.starts_with("elu") => {
                let alpha: f64 = value[3..].parse().context("invalid MDX23C ELU alpha")?;
                ensure!(alpha.is_finite(), "MDX23C ELU alpha must be finite");
                Ok(Self::Elu(alpha))
            }
            _ => bail!("unknown MDX23C activation {value:?}"),
        }
    }

    fn apply(self, input: &Tensor) -> Result<Tensor> {
        Ok(match self {
            Self::Gelu => input.f_gelu("none")?,
            Self::Relu => input.f_relu()?,
            Self::Elu(1.0) => input.f_elu()?,
            Self::Elu(alpha) => {
                let negative = input.f_clamp_max(0.0)?.f_expm1()?.f_mul_scalar(alpha)?;
                input.f_where_self(&input.f_gt(0.0)?, &negative)?
            }
        })
    }
}

#[derive(Clone, Copy)]
enum NormKind {
    Identity,
    Batch,
    Instance,
    Group(i64),
}

impl NormKind {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "BatchNorm" => Ok(Self::Batch),
            "InstanceNorm" => Ok(Self::Instance),
            value if value.contains("GroupNorm") => {
                let groups = value
                    .replace("GroupNorm", "")
                    .parse::<i64>()
                    .context("invalid MDX23C group count")?;
                ensure!(groups > 0, "MDX23C group count must be positive");
                Ok(Self::Group(groups))
            }
            _ => Ok(Self::Identity),
        }
    }
}

enum Norm {
    Identity,
    Batch {
        gain: Tensor,
        bias: Tensor,
        mean: Tensor,
        variance: Tensor,
    },
    Instance {
        gain: Tensor,
        bias: Tensor,
    },
    Group {
        groups: i64,
        gain: Tensor,
        bias: Tensor,
    },
}

impl Norm {
    fn apply(&self, input: &Tensor) -> Result<Tensor> {
        const EPSILON: f64 = 1e-5;
        Ok(match self {
            Self::Identity => input.shallow_clone(),
            Self::Batch {
                gain,
                bias,
                mean,
                variance,
            } => input.f_batch_norm(
                Some(gain),
                Some(bias),
                Some(mean),
                Some(variance),
                false,
                0.1,
                EPSILON,
                true,
            )?,
            Self::Instance { gain, bias } => input.f_instance_norm(
                Some(gain),
                Some(bias),
                None,
                None,
                true,
                0.1,
                EPSILON,
                true,
            )?,
            Self::Group { groups, gain, bias } => {
                input.f_group_norm(*groups, Some(gain), Some(bias), EPSILON, true)?
            }
        })
    }
}

struct Loader<'a> {
    weights: &'a HashMap<String, Tensor>,
    device: Device,
    norm: NormKind,
    act: Activation,
    config: &'a NetworkConfig,
}

impl Loader<'_> {
    fn tensor(&self, key: &str, shape: &[i64]) -> Result<Tensor> {
        let tensor = self
            .weights
            .get(key)
            .with_context(|| format!("MDX23C checkpoint missing tensor {key}"))?;
        ensure!(
            tensor.size() == shape,
            "MDX23C tensor {key} has shape {:?}, expected {shape:?}",
            tensor.size()
        );
        ensure!(
            matches!(
                tensor.kind(),
                Kind::Float | Kind::Half | Kind::BFloat16 | Kind::Double
            ),
            "unsupported MDX23C tensor kind for {key}"
        );
        Ok(tensor.f_to_kind(Kind::Float)?.f_to_device(self.device)?)
    }

    fn norm(&self, key: &str, channels: i64) -> Result<Norm> {
        if matches!(self.norm, NormKind::Identity) {
            return Ok(Norm::Identity);
        }
        let get = |name| self.tensor(&format!("{key}.{name}"), &[channels]);
        let gain = get("weight")?;
        let bias = get("bias")?;
        Ok(match self.norm {
            NormKind::Batch => Norm::Batch {
                gain,
                bias,
                mean: get("running_mean")?,
                variance: get("running_var")?,
            },
            NormKind::Instance => Norm::Instance { gain, bias },
            NormKind::Group(groups) => {
                ensure!(
                    channels % groups == 0,
                    "MDX23C group count does not divide channels at {key}"
                );
                Norm::Group { groups, gain, bias }
            }
            NormKind::Identity => Norm::Identity,
        })
    }

    fn normalized(&self, key: &str, channels: i64, shape: &[i64]) -> Result<Normalized> {
        Ok(Normalized {
            norm: self.norm(&format!("{key}.0"), channels)?,
            filter: self.tensor(&format!("{key}.2.weight"), shape)?,
            act: self.act,
        })
    }

    fn stack(&self, key: &str, channels_in: i64, channels: i64, width: i64) -> Result<Vec<Block>> {
        let mut input = channels_in;
        let reduced = width / self.config.bottleneck_factor;
        let mut blocks = Vec::with_capacity(self.config.num_blocks_per_scale);
        for layer in 0..self.config.num_blocks_per_scale {
            let key = format!("{key}.blocks.{layer}");
            let frequency_in =
                self.normalized(&format!("{key}.tdf"), channels, &[reduced, width])?;
            let frequency_out = Normalized {
                norm: self.norm(&format!("{key}.tdf.3"), channels)?,
                filter: self.tensor(&format!("{key}.tdf.5.weight"), &[width, reduced])?,
                act: self.act,
            };
            blocks.push(Block {
                first: self.normalized(&format!("{key}.tfc1"), input, &[channels, input, 3, 3])?,
                last: self.normalized(
                    &format!("{key}.tfc2"),
                    channels,
                    &[channels, channels, 3, 3],
                )?,
                frequency_in,
                frequency_out,
                shortcut: self
                    .tensor(&format!("{key}.shortcut.weight"), &[channels, input, 1, 1])?,
            });
            input = channels;
        }
        Ok(blocks)
    }
}

struct Normalized {
    norm: Norm,
    filter: Tensor,
    act: Activation,
}

impl Normalized {
    fn pre(&self, x: &Tensor) -> Result<Tensor> {
        self.act.apply(&self.norm.apply(x)?)
    }

    fn convolution(&self, x: &Tensor, stride: [i64; 2], pad: i64) -> Result<Tensor> {
        convolution(&self.pre(x)?, &self.filter, stride, pad)
    }

    fn upsample(&self, x: &Tensor, scale: [i64; 2]) -> Result<Tensor> {
        Ok(self.pre(x)?.f_conv_transpose2d(
            &self.filter,
            None::<&Tensor>,
            scale,
            [0, 0],
            [0, 0],
            1,
            [1, 1],
        )?)
    }

    fn linear(&self, x: &Tensor) -> Result<Tensor> {
        Ok(self.pre(x)?.f_linear(&self.filter, None::<&Tensor>)?)
    }
}

fn convolution(input: &Tensor, filter: &Tensor, stride: [i64; 2], pad: i64) -> Result<Tensor> {
    Ok(input.f_conv2d(filter, None::<&Tensor>, stride, [pad, pad], [1, 1], 1)?)
}

struct Block {
    first: Normalized,
    last: Normalized,
    frequency_in: Normalized,
    frequency_out: Normalized,
    shortcut: Tensor,
}

fn run_stack(stack: &[Block], mut x: Tensor) -> Result<Tensor> {
    for block in stack {
        let short = convolution(&x, &block.shortcut, [1, 1], 0)?;
        let hidden = block.first.convolution(&x, [1, 1], 1)?;
        let spectral = block
            .frequency_out
            .linear(&block.frequency_in.linear(&hidden)?)?;
        x = block
            .last
            .convolution(&hidden.f_add(&spectral)?, [1, 1], 1)?
            .f_add(&short)?;
    }
    Ok(x)
}

pub struct Mdx23c {
    config: ModelConfig,
    time_stride: i64,
    backend: Backend,
}

enum Backend {
    Torch(Box<TorchNetwork>),
    Mlx(Box<mlx_network::Network>),
    #[cfg(feature = "onnx")]
    Onnx(onnx_network::Network),
}

struct TorchNetwork {
    device: Device,
    window: Tensor,
    first: Tensor,
    encoders: Vec<(Vec<Block>, Normalized)>,
    bottleneck: Vec<Block>,
    decoders: Vec<(Normalized, Vec<Block>)>,
    final_in: Tensor,
    final_out: Tensor,
    activation: Activation,
}

impl Mdx23c {
    pub fn load(
        config: ModelConfig,
        weights: &HashMap<String, Tensor>,
        device: Device,
        document: &serde_json::Value,
    ) -> Result<Self> {
        let time_stride = config.validate()?;
        let net = &config.model;
        let activation = Activation::parse(&net.act)?;
        let mlx_dtype = mlx::selected_dtype(document, device, true, "MDX23C")?;
        let provider = &document["inference"]["onnx_provider"];
        let onnx = provider.as_str();
        ensure!(
            provider.is_null() || onnx.is_some(),
            "onnx_provider must be text"
        );
        #[cfg(not(feature = "onnx"))]
        ensure!(
            onnx.is_none(),
            "onnx_provider requires the onnx build feature"
        );
        if onnx.is_some() {
            ensure!(
                device == Device::Cpu && mlx_dtype.is_none(),
                "MDX23C ONNX requires --device cpu and a disabled MLX backend"
            );
        }
        #[cfg(feature = "onnx")]
        let onnx = onnx
            .map(|name| crate::model::onnx::Provider::parse("onnx_provider", name))
            .transpose()?;
        let weight_device = if mlx_dtype.is_some() {
            Device::Cpu
        } else {
            device
        };
        let loader = Loader {
            weights,
            device: weight_device,
            norm: NormKind::parse(&net.norm)?,
            act: activation,
            config: net,
        };
        let input_channels = 2 * config.audio.num_channels * net.num_subbands;
        let mut channels = net.num_channels;
        let mut width = config.audio.dim_f / net.num_subbands;
        let [time_scale, frequency_scale] = net.scale;

        let first = loader.tensor("first_conv.weight", &[channels, input_channels, 1, 1])?;
        let mut encoders = Vec::with_capacity(net.num_scales);
        for layer in 0..net.num_scales {
            let key = format!("encoder_blocks.{layer}");
            let stack = loader.stack(&format!("{key}.tfc_tdf"), channels, channels, width)?;
            let next = channels + net.growth;
            let down = loader.normalized(
                &format!("{key}.downscale.conv"),
                channels,
                &[next, channels, time_scale, frequency_scale],
            )?;
            encoders.push((stack, down));
            channels = next;
            width /= frequency_scale;
        }
        let bottleneck = loader.stack("bottleneck_block", channels, channels, width)?;
        let mut decoders = Vec::with_capacity(net.num_scales);
        for layer in 0..net.num_scales {
            let key = format!("decoder_blocks.{layer}");
            let next = channels - net.growth;
            let up = loader.normalized(
                &format!("{key}.upscale.conv"),
                channels,
                &[channels, next, time_scale, frequency_scale],
            )?;
            width *= frequency_scale;
            let stack = loader.stack(&format!("{key}.tfc_tdf"), 2 * next, next, width)?;
            decoders.push((up, stack));
            channels = next;
        }
        let final_in = loader.tensor(
            "final_conv.0.weight",
            &[channels, channels + input_channels, 1, 1],
        )?;
        let final_out = loader.tensor(
            "final_conv.2.weight",
            &[config.stems() as i64 * input_channels, channels, 1, 1],
        )?;
        let spectral_device = if weight_device == Device::Mps {
            Device::Cpu
        } else {
            weight_device
        };
        let window = Tensor::f_hann_window_periodic(
            config.audio.n_fft,
            true,
            (Kind::Float, spectral_device),
        )?;
        let torch = TorchNetwork {
            device: weight_device,
            window,
            first,
            encoders,
            bottleneck,
            decoders,
            final_in,
            final_out,
            activation,
        };
        #[cfg(feature = "onnx")]
        if let Some(provider) = onnx {
            let backend = Backend::Onnx(onnx_network::Network::load(&config, &torch, provider)?);
            return Ok(Self {
                config,
                time_stride,
                backend,
            });
        }
        let backend = if let Some(dtype) = mlx_dtype {
            Backend::Mlx(Box::new(mlx_network::Network::load(
                &config, &torch, device, dtype,
            )?))
        } else {
            Backend::Torch(Box::new(torch))
        };
        Ok(Self {
            config,
            time_stride,
            backend,
        })
    }

    pub fn input_channels(&self) -> usize {
        self.config.audio.num_channels as usize
    }

    /// PyMSS SubbandSTFT.inverse always returns two channels per stem.
    pub fn output_channels(&self) -> usize {
        2
    }

    pub fn num_stems(&self) -> usize {
        self.config.stems()
    }

    pub(super) fn predict_batch(&self, audio: &[f32], count: usize) -> Result<Vec<f32>> {
        let channels = self.input_channels();
        let samples = super::window_samples(audio, count, channels)?;
        ensure!(
            samples > self.config.audio.n_fft / 2,
            "MDX23C input is too short for centered STFT"
        );
        let fft = self.config.audio.n_fft;
        let frames = (samples + 2 * (fft / 2) - fft) / self.config.audio.hop_length + 1;
        ensure!(
            frames % self.time_stride == 0,
            "MDX23C time stride does not divide the window's {frames} STFT frames"
        );
        let output = match &self.backend {
            Backend::Torch(model) => {
                self.spectral_pass(audio, count, samples, &model.window, |mix| {
                    model.forward(mix, self.config.model.scale)
                })?
            }
            #[cfg(feature = "onnx")]
            Backend::Onnx(model) => {
                self.spectral_pass(audio, count, samples, &model.window, |mix| model.run(&mix))?
            }
            Backend::Mlx(model) => model.run(audio, count, samples as usize)?,
        };
        ensure!(
            output.iter().all(|sample| sample.is_finite()),
            "MDX23C produced nonfinite audio"
        );
        Ok(output)
    }

    fn spectral_pass(
        &self,
        audio: &[f32],
        count: usize,
        samples: i64,
        window: &Tensor,
        forward: impl FnOnce(Tensor) -> Result<Tensor>,
    ) -> Result<Vec<f32>> {
        no_grad(|| {
            let input = Tensor::f_from_slice(audio)?
                .f_reshape([-1, samples])?
                .f_to_device(window.device())?;
            let batch = i64::try_from(count)?;
            let mix = self.config.analyze(&input, batch, window)?;
            let frames = mix.size()[3];
            let output = forward(mix)?;
            let audio = self.config.synthesize(&output, batch, frames, window)?;
            let flat = audio.f_to_device(Device::Cpu)?.f_reshape([-1])?;
            Ok(Vec::<f32>::try_from(flat)?)
        })
    }
}

impl ModelConfig {
    fn analyze(&self, input: &Tensor, batch: i64, window: &Tensor) -> Result<Tensor> {
        let AudioConfig {
            n_fft: fft,
            hop_length: hop,
            dim_f,
            num_channels: channels,
        } = self.audio;
        let half_window = fft / 2;
        let padded = input.f_reflection_pad1d([half_window, half_window])?;
        let complex = padded.f_stft_center(
            fft,
            hop,
            fft,
            Some(window),
            false,
            "reflect",
            false,
            true,
            true,
            false,
        )?;
        let frames = complex.size()[2];
        let subbands = self.model.num_subbands;
        Ok(complex
            .f_view_as_real()?
            .f_permute([0, 3, 1, 2])?
            .f_narrow(2, 0, dim_f)?
            .f_reshape([batch, 2 * channels * subbands, dim_f / subbands, frames])?)
    }

    fn synthesize(
        &self,
        output: &Tensor,
        batch: i64,
        frames: i64,
        window: &Tensor,
    ) -> Result<Tensor> {
        let AudioConfig {
            n_fft: fft,
            hop_length: hop,
            dim_f,
            num_channels: channels,
        } = self.audio;
        let bins = fft / 2 + 1;
        let stems = self.stems() as i64;
        let output = output
            .f_to_device(window.device())?
            .f_reshape([batch * stems * channels, 2, dim_f, frames])?
            .f_constant_pad_nd([0, 0, 0, bins - dim_f])?
            .f_permute([0, 2, 3, 1])?
            .f_contiguous()?
            .f_view_as_complex()?;
        let restored =
            output.f_istft(fft, hop, fft, Some(window), true, false, true, None, false)?;
        Ok(restored.f_reshape([batch, stems, 2, -1])?)
    }
}

impl TorchNetwork {
    fn forward(&self, mix: Tensor, scale: [i64; 2]) -> Result<Tensor> {
        let mix = mix.f_to_device(self.device)?;
        let first = convolution(&mix, &self.first, [1, 1], 0)?;
        let core = self
            .forward_core(first.f_transpose(2, 3)?, scale)?
            .f_transpose(2, 3)?;
        let product = core.f_mul(&first)?;
        let hidden = convolution(
            &Tensor::f_cat(&[mix, product], 1)?,
            &self.final_in,
            [1, 1],
            0,
        )?;
        convolution(&self.activation.apply(&hidden)?, &self.final_out, [1, 1], 0)
    }

    fn forward_core(&self, mut value: Tensor, scale: [i64; 2]) -> Result<Tensor> {
        let mut skips = Vec::with_capacity(self.encoders.len());
        for (stack, down) in &self.encoders {
            let hidden = run_stack(stack, value)?;
            value = down.convolution(&hidden, scale, 0)?;
            skips.push(hidden);
        }
        value = run_stack(&self.bottleneck, value)?;
        for ((up, stack), skip) in self.decoders.iter().zip(skips.into_iter().rev()) {
            let upscaled = up.upsample(&value, scale)?;
            value = run_stack(stack, Tensor::f_cat(&[upscaled, skip], 1)?)?;
        }
        Ok(value)
    }
}
