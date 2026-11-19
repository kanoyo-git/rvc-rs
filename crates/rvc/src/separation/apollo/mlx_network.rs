use anyhow::Result;
use tch::Device;

use super::super::super::{
    mlx::{Array, DType, Stream},
    mlx_layers::{Affine, Layout, glu, silu},
    mlx_stft::Stft,
};
use super::{BANDS, HEADS, RMS_EPSILON, TorchNetwork};

const BAND_COUNT: i32 = BANDS as i32;
const PREFIX: i32 = BAND_COUNT - 1;
const HEAD_COUNT: i32 = HEADS as i32;

pub(super) struct Network {
    stream: Stream,
    dtype: DType,
    stft: Stft,
    width: i32,
    tail_width: i32,
    prefix_input: Packed,
    tail_input: Projection,
    layers: Vec<Band>,
    prefix_output: Packed,
    tail_output: Projection,
}

impl Network {
    pub(super) fn load(torch: &TorchNetwork, device: Device, dtype: DType) -> Result<Self> {
        let stream = Stream::new(device)?;
        let spectral = &torch.spectral;
        let stft = Stft::new(
            &stream,
            spectral.fft,
            spectral.fft / 2,
            false,
            &spectral.window,
        )?;
        let prefix_input = Packed::load(&stream, &torch.prefix_input, dtype)?;
        let tail_input = Projection::load(&stream, &torch.tail_input, dtype)?;
        let prefix_output = Packed::load(&stream, &torch.prefix_output, dtype)?;
        let tail_output = Projection::load(&stream, &torch.tail_output, dtype)?;
        let layers = torch
            .network
            .iter()
            .map(|layer| Band::load(&stream, layer, dtype))
            .collect::<Result<_>>()?;
        Ok(Self {
            stream,
            dtype,
            stft,
            width: i32::try_from(spectral.width)?,
            tail_width: i32::try_from(spectral.tail_width)?,
            prefix_input,
            tail_input,
            layers,
            prefix_output,
            tail_output,
        })
    }

    pub(super) fn run(&self, audio: &[f32], samples: usize) -> Result<Vec<f32>> {
        let stream = &self.stream;
        let spectrum = self.stft.analyze(stream, audio, samples)?;
        let [tracks, frames, _] = <[i32; 3]>::try_from(spectrum.shape())?;
        let spectrum = stream.transpose(&spectrum, &[0, 2, 1])?;
        let prefix_bins = PREFIX * self.width;
        let prefix = stream.slice(&spectrum, 1, 0, prefix_bins, 1)?;
        let prefix = stream.reshape(&prefix, &[tracks, PREFIX, self.width, frames])?;
        let tail = stream.slice(&spectrum, 1, prefix_bins, self.stft.bins, 1)?;
        let prefix = normalize_band(stream, &prefix, 2)?;
        let prefix = self
            .prefix_input
            .run(stream, &stream.cast(&prefix, self.dtype)?)?;
        let tail = normalize_band(stream, &tail, 1)?;
        let tail = self
            .tail_input
            .run(stream, &stream.cast(&tail, self.dtype)?)?;
        let tail = stream.reshape(&tail, &[tracks, 1, -1, frames])?;
        let mut value = stream.concat(&[prefix, tail], 1)?;
        for layer in &self.layers {
            value = layer.run(stream, &value)?;
        }
        let prefix = stream.slice(&value, 1, 0, PREFIX, 1)?;
        let prefix = self.prefix_output.run(stream, &prefix)?;
        let prefix = glu(stream, &prefix, 2)?;
        let prefix = stream.reshape(&prefix, &[tracks, PREFIX, 2, self.width, frames])?;
        let prefix = stream.transpose(&prefix, &[0, 2, 1, 3, 4])?;
        let prefix = stream.reshape(&prefix, &[tracks, 2, prefix_bins, frames])?;
        let tail = self
            .tail_output
            .run(stream, &stream.select(&value, 1, PREFIX)?)?;
        let tail = stream.reshape(
            &glu(stream, &tail, 1)?,
            &[tracks, 2, self.tail_width, frames],
        )?;
        let estimate = stream.cast(&stream.concat(&[prefix, tail], 2)?, DType::F32)?;
        let estimate = stream.complex(
            &stream.select(&estimate, 1, 0)?,
            &stream.select(&estimate, 1, 1)?,
        )?;
        let estimate = stream.transpose(&estimate, &[0, 2, 1])?;
        let wave = self.stft.synthesize(stream, &estimate, Some(samples))?;
        stream.to_floats(&wave)
    }
}

fn normalize_band(stream: &Stream, input: &Array, axis: i32) -> Result<Array> {
    let real = stream.real(input)?;
    let imaginary = stream.imag(input)?;
    let squared = stream.add(&stream.square(&real)?, &stream.square(&imaginary)?)?;
    let power = stream.sum_keep(&squared, axis)?;
    let power = stream.add(&power, &stream.scalar(f32::EPSILON, DType::F32)?)?;
    let power = stream.sqrt(&power)?;
    stream.concat(
        &[
            stream.div(&real, &power)?,
            stream.div(&imaginary, &power)?,
            stream.log(&power)?,
        ],
        axis,
    )
}

fn rms(stream: &Stream, input: &Array, norm: &Array, axis: i32, dtype: DType) -> Result<Array> {
    let input = stream.cast(input, DType::F32)?;
    let square = stream.square(&input)?;
    let mean = stream.mean_keep(&square, &[axis])?;
    let mean = stream.add(&mean, &stream.scalar(RMS_EPSILON as f32, DType::F32)?)?;
    let normalized = stream.mul(&input, &stream.rsqrt(&mean)?)?;
    stream.mul(&stream.cast(&normalized, dtype)?, norm)
}

struct Pointwise {
    weight: Array,
    bias: Option<Array>,
}

impl Pointwise {
    fn load(stream: &Stream, layer: &super::Pointwise, dtype: DType) -> Result<Self> {
        let weight = stream.upload(&layer.weight.f_transpose(0, 1)?, dtype)?;
        let bias = layer
            .bias
            .as_ref()
            .map(|bias| stream.upload(bias, dtype))
            .transpose()?;
        Ok(Self { weight, bias })
    }

    fn run(&self, stream: &Stream, input: &Array) -> Result<Array> {
        let input = stream.transpose(input, &[0, 2, 1])?;
        let value = stream.matmul(&input, &self.weight)?;
        let value = if let Some(bias) = &self.bias {
            stream.add(&value, bias)?
        } else {
            value
        };
        stream.transpose(&value, &[0, 2, 1])
    }
}

struct Projection {
    norm: Array,
    linear: Pointwise,
    dtype: DType,
}

impl Projection {
    fn load(stream: &Stream, layer: &super::Projection, dtype: DType) -> Result<Self> {
        Ok(Self {
            norm: stream.upload(&layer.norm.f_reshape([1, -1, 1])?, dtype)?,
            linear: Pointwise::load(stream, &layer.linear, dtype)?,
            dtype,
        })
    }

    fn run(&self, stream: &Stream, input: &Array) -> Result<Array> {
        self.linear
            .run(stream, &rms(stream, input, &self.norm, 1, self.dtype)?)
    }
}

struct Packed {
    norm: Array,
    weight: Array,
    bias: Array,
    dtype: DType,
}

impl Packed {
    fn load(stream: &Stream, layer: &super::PackedProjection, dtype: DType) -> Result<Self> {
        Ok(Self {
            norm: stream.upload(&layer.norm, dtype)?,
            weight: stream.upload(&layer.weight.f_permute([0, 2, 1])?, dtype)?,
            bias: stream.upload(&layer.bias, dtype)?,
            dtype,
        })
    }

    fn run(&self, stream: &Stream, input: &Array) -> Result<Array> {
        let [batch, bands, channels, frames] = <[i32; 4]>::try_from(input.shape())?;
        let input = rms(stream, input, &self.norm, 2, self.dtype)?;
        let input = stream.reshape(&input, &[batch, bands * channels, frames])?;
        let input = stream.transpose(&input, &[0, 2, 1])?;
        let value = stream.conv1d(&input, &self.weight, 0, bands)?;
        let value = stream.add(&value, &self.bias)?;
        let value = stream.transpose(&value, &[0, 2, 1])?;
        stream.reshape(&value, &[batch, bands, -1, frames])
    }
}

struct Attention {
    norm: Array,
    qkv: Pointwise,
    projection: Pointwise,
    cosine: Array,
    sine: Array,
    minus_one: Array,
    mlp: Projection,
    mlp_output: Pointwise,
    head_dim: i32,
    dtype: DType,
}

impl Attention {
    fn load(stream: &Stream, layer: &super::Attention, dtype: DType) -> Result<Self> {
        Ok(Self {
            norm: stream.upload(&layer.norm.f_reshape([1, -1, 1])?, dtype)?,
            qkv: Pointwise::load(stream, &layer.qkv, dtype)?,
            projection: Pointwise::load(stream, &layer.projection, dtype)?,
            cosine: stream.upload(&layer.cosine, dtype)?,
            sine: stream.upload(&layer.sine, dtype)?,
            minus_one: stream.scalar(-1.0, dtype)?,
            mlp: Projection::load(stream, &layer.mlp, dtype)?,
            mlp_output: Pointwise::load(stream, &layer.mlp_output, dtype)?,
            head_dim: i32::try_from(layer.head_dim)?,
            dtype,
        })
    }

    fn rotate(&self, stream: &Stream, input: &Array) -> Result<Array> {
        let [batch, heads, frames, dim] = <[i32; 4]>::try_from(input.shape())?;
        let pairs = stream.reshape(input, &[batch, heads, frames, dim / 2, 2])?;
        let even = stream.select(&pairs, 4, 0)?;
        let odd = stream.select(&pairs, 4, 1)?;
        let swapped = stream.stack(&[stream.mul(&odd, &self.minus_one)?, even], 4)?;
        let swapped = stream.reshape(&swapped, input.shape())?;
        let direct = stream.mul(input, &self.cosine)?;
        stream.add(&direct, &stream.mul(&swapped, &self.sine)?)
    }

    fn run(&self, stream: &Stream, input: &Array) -> Result<Array> {
        let [batch, _, frames] = <[i32; 3]>::try_from(input.shape())?;
        let normalized = rms(stream, input, &self.norm, 1, self.dtype)?;
        let weights = self.qkv.run(stream, &normalized)?;
        let weights = stream.reshape(&weights, &[batch, HEAD_COUNT, 3 * self.head_dim, frames])?;
        let weights = stream.transpose(&weights, &[0, 1, 3, 2])?;
        let q = self.rotate(stream, &stream.slice(&weights, 3, 0, self.head_dim, 1)?)?;
        let k = self.rotate(
            stream,
            &stream.slice(&weights, 3, self.head_dim, 2 * self.head_dim, 1)?,
        )?;
        let v = stream.slice(&weights, 3, 2 * self.head_dim, 3 * self.head_dim, 1)?;
        let value = stream.attention(&q, &k, &v, (self.head_dim as f32).sqrt().recip())?;
        let value = stream.transpose(&value, &[0, 1, 3, 2])?;
        let value = stream.reshape(&value, &[batch, -1, frames])?;
        let value = stream.add(input, &self.projection.run(stream, &value)?)?;
        let gate = silu(stream, &self.mlp.run(stream, &value)?)?;
        let mid = gate.shape()[1] / 2;
        let left = silu(stream, &stream.slice(&gate, 1, 0, mid, 1)?)?;
        let right = stream.slice(&gate, 1, mid, 2 * mid, 1)?;
        let product = stream.mul(&left, &right)?;
        stream.add(&value, &self.mlp_output.run(stream, &product)?)
    }
}

struct Sequence {
    convolution: Affine,
    norm: Array,
    expand: Pointwise,
    contract: Pointwise,
    dtype: DType,
}

impl Sequence {
    fn load(stream: &Stream, layer: &super::SequenceBlock, dtype: DType) -> Result<Self> {
        Ok(Self {
            convolution: Affine::load(stream, &layer.conv, &layer.bias, Layout::Conv1d, dtype)?,
            norm: stream.upload(&layer.norm.f_reshape([1, -1, 1])?, dtype)?,
            expand: Pointwise::load(stream, &layer.expand, dtype)?,
            contract: Pointwise::load(stream, &layer.contract, dtype)?,
            dtype,
        })
    }

    fn run(&self, stream: &Stream, input: &Array) -> Result<Array> {
        let depthwise = self
            .convolution
            .conv1d(stream, input, 3, self.convolution.channels())?;
        let value = rms(stream, &depthwise, &self.norm, 1, self.dtype)?;
        let value = silu(stream, &self.expand.run(stream, &value)?)?;
        stream.add(input, &self.contract.run(stream, &value)?)
    }
}

struct Band {
    attention: Attention,
    sequence: Vec<Sequence>,
}

impl Band {
    fn load(stream: &Stream, layer: &super::BandNetwork, dtype: DType) -> Result<Self> {
        Ok(Self {
            attention: Attention::load(stream, &layer.attention, dtype)?,
            sequence: layer
                .sequence
                .iter()
                .map(|layer| Sequence::load(stream, layer, dtype))
                .collect::<Result<_>>()?,
        })
    }

    fn run(&self, stream: &Stream, input: &Array) -> Result<Array> {
        let [batch, bands, features, frames] = <[i32; 4]>::try_from(input.shape())?;
        let value = stream.transpose(input, &[0, 3, 2, 1])?;
        let value = stream.reshape(&value, &[batch * frames, features, bands])?;
        let value = self.attention.run(stream, &value)?;
        let value = stream.reshape(&value, &[batch, frames, features, bands])?;
        let value = stream.transpose(&value, &[0, 3, 2, 1])?;
        let mut value = stream.reshape(&value, &[batch * bands, features, frames])?;
        for layer in &self.sequence {
            value = layer.run(stream, &value)?;
        }
        stream.reshape(&value, &[batch, bands, features, frames])
    }
}
