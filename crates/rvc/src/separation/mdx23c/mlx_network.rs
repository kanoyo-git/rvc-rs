use anyhow::{Context, Result, ensure};
use tch::{Device, Tensor};

use super::super::mlx::{Array, DType, Stream};
use super::super::mlx_stft::Stft;
use super::{Activation, ModelConfig, TorchNetwork};

pub(super) struct Network {
    stream: Stream,
    dtype: DType,
    stft: Stft,
    channels: i32,
    stems: i32,
    dim_f: i32,
    subbands: i32,
    scale: [i32; 2],
    first: Array,
    encoders: Vec<(Vec<Block>, Layer)>,
    bottleneck: Vec<Block>,
    decoders: Vec<(Layer, Vec<Block>)>,
    final_in: Array,
    final_out: Array,
    activation: Activation,
}

impl Network {
    pub(super) fn load(
        config: &ModelConfig,
        torch: &TorchNetwork,
        device: Device,
        dtype: DType,
    ) -> Result<Self> {
        let stream = Stream::new(device)?;
        let stft = Stft::new(
            &stream,
            config.audio.n_fft,
            config.audio.hop_length,
            false,
            &torch.window,
        )?;
        let encoders = torch
            .encoders
            .iter()
            .map(|(blocks, down)| {
                Ok((
                    load_stack(&stream, blocks, dtype)?,
                    Layer::load(&stream, down, Kernel::Convolution, dtype)?,
                ))
            })
            .collect::<Result<_>>()?;
        let bottleneck = load_stack(&stream, &torch.bottleneck, dtype)?;
        let decoders = torch
            .decoders
            .iter()
            .map(|(up, blocks)| {
                Ok((
                    Layer::load(&stream, up, Kernel::Upsample, dtype)?,
                    load_stack(&stream, blocks, dtype)?,
                ))
            })
            .collect::<Result<_>>()?;
        let first = conv_weight(&stream, &torch.first, dtype)?;
        let final_in = conv_weight(&stream, &torch.final_in, dtype)?;
        let final_out = conv_weight(&stream, &torch.final_out, dtype)?;
        Ok(Self {
            stream,
            dtype,
            stft,
            channels: i32::try_from(config.audio.num_channels)?,
            stems: i32::try_from(config.stems())?,
            dim_f: i32::try_from(config.audio.dim_f)?,
            subbands: i32::try_from(config.model.num_subbands)?,
            scale: [
                i32::try_from(config.model.scale[0])?,
                i32::try_from(config.model.scale[1])?,
            ],
            first,
            encoders,
            bottleneck,
            decoders,
            final_in,
            final_out,
            activation: torch.activation,
        })
    }

    pub(super) fn run(&self, audio: &[f32], batch: usize, samples: usize) -> Result<Vec<f32>> {
        let stream = &self.stream;
        let batch = i32::try_from(batch)?;
        let spectrum = self.stft.analyze(stream, audio, samples)?;
        let frames = spectrum.shape()[1];
        let spectrum = stream.transpose(&spectrum, &[0, 2, 1])?;
        let spectrum = stream.slice(&spectrum, 1, 0, self.dim_f, 1)?;
        let real = stream.real(&spectrum)?;
        let imaginary = stream.imag(&spectrum)?;
        let spectrum = stream.stack(&[real, imaginary], 1)?;
        let mix = stream.reshape(
            &spectrum,
            &[
                batch,
                2 * self.channels * self.subbands,
                self.dim_f / self.subbands,
                frames,
            ],
        )?;
        let mix = stream.cast(&mix, self.dtype)?;
        let first = convolve(stream, &mix, &self.first, [1, 1], 0)?;
        let core = stream.transpose(&first, &[0, 1, 3, 2])?;
        let core = self.run_core(core)?;
        let core = stream.transpose(&core, &[0, 1, 3, 2])?;
        let product = stream.mul(&core, &first)?;
        let hidden = stream.concat(&[mix, product], 1)?;
        let hidden = convolve(stream, &hidden, &self.final_in, [1, 1], 0)?;
        let hidden = activate(stream, &hidden, self.activation, self.dtype)?;
        let spectrum = convolve(stream, &hidden, &self.final_out, [1, 1], 0)?;
        let tracks = batch
            .checked_mul(self.stems * self.channels)
            .context("MDX23C output track count overflow")?;
        let spectrum = stream.reshape(&spectrum, &[tracks, 2, self.dim_f, frames])?;
        let spectrum = stream.cast(&spectrum, DType::F32)?;
        let spectrum = if self.dim_f < self.stft.bins {
            let tail = stream.zeros(&[tracks, 2, self.stft.bins - self.dim_f, frames])?;
            stream.concat(&[spectrum, tail], 2)?
        } else {
            spectrum
        };
        let spectrum = stream.complex(
            &stream.select(&spectrum, 1, 0)?,
            &stream.select(&spectrum, 1, 1)?,
        )?;
        let spectrum = stream.transpose(&spectrum, &[0, 2, 1])?;
        let audio = self.stft.synthesize(stream, &spectrum, None)?;
        let length = i64::from(audio.shape()[1]) * i64::from(self.channels);
        ensure!(
            length % 2 == 0,
            "MDX23C cannot reshape its output to stereo"
        );
        stream.to_floats(&audio)
    }

    fn run_core(&self, mut value: Array) -> Result<Array> {
        let stream = &self.stream;
        let mut skips = Vec::with_capacity(self.encoders.len());
        for (blocks, down) in &self.encoders {
            let hidden = run_stack(stream, blocks, value, self.dtype)?;
            value = down.convolution(stream, &hidden, self.scale, 0, self.dtype)?;
            skips.push(hidden);
        }
        value = run_stack(stream, &self.bottleneck, value, self.dtype)?;
        for ((up, blocks), skip) in self.decoders.iter().zip(skips.into_iter().rev()) {
            let upscaled = up.upsample(stream, &value, self.scale, self.dtype)?;
            let joined = stream.concat(&[upscaled, skip], 1)?;
            value = run_stack(stream, blocks, joined, self.dtype)?;
        }
        Ok(value)
    }
}

fn conv_weight(stream: &Stream, weight: &Tensor, dtype: DType) -> Result<Array> {
    stream.upload(&weight.f_permute([0, 2, 3, 1])?, dtype)
}

fn convolve(
    stream: &Stream,
    input: &Array,
    weight: &Array,
    stride: [i32; 2],
    pad: i32,
) -> Result<Array> {
    let input = stream.transpose(input, &[0, 2, 3, 1])?;
    let result = stream.conv2d(&input, weight, stride, [pad, pad])?;
    stream.transpose(&result, &[0, 3, 1, 2])
}

enum Kernel {
    Convolution,
    Upsample,
    Linear,
}

struct Layer {
    norm: Norm,
    filter: Array,
    activation: Activation,
}

impl Layer {
    fn load(
        stream: &Stream,
        layer: &super::Normalized,
        kernel: Kernel,
        dtype: DType,
    ) -> Result<Self> {
        let filter = match kernel {
            Kernel::Convolution => conv_weight(stream, &layer.filter, dtype)?,
            Kernel::Upsample => stream.upload(&layer.filter.f_permute([1, 2, 3, 0])?, dtype)?,
            Kernel::Linear => stream.upload(&layer.filter.f_transpose(0, 1)?, dtype)?,
        };
        Ok(Self {
            norm: Norm::load(stream, &layer.norm, dtype)?,
            filter,
            activation: layer.act,
        })
    }

    fn pre(&self, stream: &Stream, input: &Array, dtype: DType) -> Result<Array> {
        let value = self.norm.apply(stream, input, dtype)?;
        activate(stream, &value, self.activation, dtype)
    }

    fn convolution(
        &self,
        stream: &Stream,
        input: &Array,
        stride: [i32; 2],
        pad: i32,
        dtype: DType,
    ) -> Result<Array> {
        convolve(
            stream,
            &self.pre(stream, input, dtype)?,
            &self.filter,
            stride,
            pad,
        )
    }

    fn upsample(
        &self,
        stream: &Stream,
        input: &Array,
        stride: [i32; 2],
        dtype: DType,
    ) -> Result<Array> {
        let input = self.pre(stream, input, dtype)?;
        let input = stream.transpose(&input, &[0, 2, 3, 1])?;
        let value = stream.upsample2d(&input, &self.filter, stride)?;
        stream.transpose(&value, &[0, 3, 1, 2])
    }

    fn linear(&self, stream: &Stream, input: &Array, dtype: DType) -> Result<Array> {
        stream.matmul(&self.pre(stream, input, dtype)?, &self.filter)
    }
}

struct Block {
    first: Layer,
    last: Layer,
    frequency_in: Layer,
    frequency_out: Layer,
    shortcut: Array,
}

fn load_stack(stream: &Stream, blocks: &[super::Block], dtype: DType) -> Result<Vec<Block>> {
    blocks
        .iter()
        .map(|block| {
            Ok(Block {
                first: Layer::load(stream, &block.first, Kernel::Convolution, dtype)?,
                last: Layer::load(stream, &block.last, Kernel::Convolution, dtype)?,
                frequency_in: Layer::load(stream, &block.frequency_in, Kernel::Linear, dtype)?,
                frequency_out: Layer::load(stream, &block.frequency_out, Kernel::Linear, dtype)?,
                shortcut: conv_weight(stream, &block.shortcut, dtype)?,
            })
        })
        .collect()
}

fn run_stack(stream: &Stream, blocks: &[Block], mut value: Array, dtype: DType) -> Result<Array> {
    for block in blocks {
        let shortcut = convolve(stream, &value, &block.shortcut, [1, 1], 0)?;
        let hidden = block.first.convolution(stream, &value, [1, 1], 1, dtype)?;
        let frequency = block.frequency_in.linear(stream, &hidden, dtype)?;
        let frequency = block.frequency_out.linear(stream, &frequency, dtype)?;
        let merged = stream.add(&hidden, &frequency)?;
        let last = block.last.convolution(stream, &merged, [1, 1], 1, dtype)?;
        value = stream.add(&shortcut, &last)?;
    }
    Ok(value)
}

fn activate(stream: &Stream, input: &Array, act: Activation, dtype: DType) -> Result<Array> {
    let zero = stream.scalar(0.0, dtype)?;
    match act {
        Activation::Relu => stream.max(input, &zero),
        Activation::Elu(alpha) => {
            let positive = stream.max(input, &zero)?;
            let negative = stream.expm1(&stream.min(input, &zero)?)?;
            let negative = stream.mul(&negative, &stream.scalar(alpha as f32, dtype)?)?;
            stream.add(&positive, &negative)
        }
        Activation::Gelu => {
            let input = stream.cast(input, DType::F32)?;
            let scale = stream.scalar(std::f32::consts::FRAC_1_SQRT_2, DType::F32)?;
            let sigmoid = stream.erf(&stream.mul(&input, &scale)?)?;
            let shifted = stream.add(&sigmoid, &stream.scalar(1.0, DType::F32)?)?;
            let half = stream.mul(&input, &stream.scalar(0.5, DType::F32)?)?;
            stream.cast(&stream.mul(&half, &shifted)?, dtype)
        }
    }
}

enum Norm {
    Identity,
    Batch {
        mean: Array,
        inverse_std: Array,
        affine: Affine,
    },
    Sample {
        groups: Option<i32>,
        affine: Affine,
    },
}

struct Affine {
    gain: Array,
    bias: Array,
}

impl Affine {
    fn load(stream: &Stream, gain: &Tensor, bias: &Tensor, dtype: DType) -> Result<Self> {
        Ok(Self {
            gain: stream.upload(&gain.f_reshape([1, -1, 1, 1])?, dtype)?,
            bias: stream.upload(&bias.f_reshape([1, -1, 1, 1])?, dtype)?,
        })
    }

    fn apply(&self, stream: &Stream, input: &Array, dtype: DType) -> Result<Array> {
        let input = stream.cast(input, dtype)?;
        stream.add(&stream.mul(&input, &self.gain)?, &self.bias)
    }
}

impl Norm {
    fn load(stream: &Stream, norm: &super::Norm, dtype: DType) -> Result<Self> {
        Ok(match norm {
            super::Norm::Identity => Self::Identity,
            super::Norm::Batch {
                gain,
                bias,
                mean,
                variance,
            } => {
                let inverse_std = variance.f_add_scalar(1e-5)?.f_rsqrt()?;
                Self::Batch {
                    mean: stream.upload(&mean.f_reshape([1, -1, 1, 1])?, DType::F32)?,
                    inverse_std: stream
                        .upload(&inverse_std.f_reshape([1, -1, 1, 1])?, DType::F32)?,
                    affine: Affine::load(stream, gain, bias, dtype)?,
                }
            }
            super::Norm::Instance { gain, bias } => Self::Sample {
                groups: None,
                affine: Affine::load(stream, gain, bias, dtype)?,
            },
            super::Norm::Group { groups, gain, bias } => Self::Sample {
                groups: Some(i32::try_from(*groups)?),
                affine: Affine::load(stream, gain, bias, dtype)?,
            },
        })
    }

    fn apply(&self, stream: &Stream, input: &Array, dtype: DType) -> Result<Array> {
        match self {
            Self::Identity => stream.cast(input, dtype),
            Self::Batch {
                mean,
                inverse_std,
                affine,
            } => {
                let input = stream.cast(input, DType::F32)?;
                let centered = stream.sub(&input, mean)?;
                let normalized = stream.mul(&centered, inverse_std)?;
                affine.apply(stream, &normalized, dtype)
            }
            Self::Sample { groups, affine } => {
                let input = stream.cast(input, DType::F32)?;
                let shape = input.shape();
                let normalized = if let Some(groups) = groups {
                    let grouped = stream.reshape(
                        &input,
                        &[shape[0], *groups, shape[1] / groups, shape[2], shape[3]],
                    )?;
                    let grouped = normalize(stream, &grouped, &[2, 3, 4])?;
                    stream.reshape(&grouped, shape)?
                } else {
                    normalize(stream, &input, &[2, 3])?
                };
                affine.apply(stream, &normalized, dtype)
            }
        }
    }
}

fn normalize(stream: &Stream, input: &Array, axes: &[i32]) -> Result<Array> {
    let mean = stream.mean_keep(input, axes)?;
    let centered = stream.sub(input, &mean)?;
    let variance = stream.mean_keep(&stream.square(&centered)?, axes)?;
    let variance = stream.add(&variance, &stream.scalar(1e-5, DType::F32)?)?;
    stream.mul(&centered, &stream.rsqrt(&variance)?)
}
