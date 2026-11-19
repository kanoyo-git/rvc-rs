use anyhow::{Result, bail, ensure};
use tch::{Device, Tensor};

use super::super::super::{
    mlx::{Array, DType, Stream},
    mlx_layers::{Affine, Cell, Layout, Rnn, gelu, glu, silu},
    mlx_stft::{Padding, Stft},
};
use super::{Activation, CHANNELS, RnnType, TorchNetwork, Version};

pub(super) struct Network {
    stream: Stream,
    dtype: DType,
    analysis: Stft,
    inverse: Stft,
    padding: Padding,
    center: bool,
    version: Version,
    projections: Vec<Projection>,
    recurrent: Vec<Recurrent>,
    masks: Vec<Mask>,
    scatter: Option<Array>,
    complex: bool,
    activation: Activation,
}

impl Network {
    pub(super) fn load(model: &TorchNetwork, device: Device, dtype: DType) -> Result<Self> {
        let stream = Stream::new(device)?;
        let spectrum = &model.spectral;
        let pad_window = |window: &Tensor| -> Result<Stft> {
            let side = (spectrum.fft - spectrum.length) / 2;
            let window = window.f_constant_pad_nd([side, spectrum.fft - spectrum.length - side])?;
            Stft::new(
                &stream,
                spectrum.fft,
                spectrum.hop,
                spectrum.normalized,
                &window,
            )
        };
        let analysis = pad_window(&spectrum.window)?;
        let inverse = pad_window(&spectrum.inverse)?;
        let padding = if spectrum.center {
            match spectrum.pad_mode.as_str() {
                "constant" => Padding::Constant,
                "reflect" => Padding::Reflect,
                "replicate" => Padding::Replicate,
                "circular" => Padding::Circular,
                mode => bail!("unknown Bandit STFT padding {mode:?}"),
            }
        } else {
            Padding::None
        };
        let projections = model
            .projections
            .iter()
            .map(|layer| {
                Ok(Projection {
                    norm: pack(&stream, &layer.norm, Layout::Channels, dtype)?,
                    linear: pack(&stream, &layer.linear, Layout::Linear, dtype)?,
                    start: i32::try_from(layer.band.start)?,
                    width: i32::try_from(layer.band.width())?,
                })
            })
            .collect::<Result<_>>()?;
        let recurrent = model
            .recurrent
            .iter()
            .map(|layer| {
                let cell = match layer.rnn_type {
                    RnnType::Lstm => Cell::Lstm,
                    RnnType::Gru => Cell::Gru,
                    RnnType::Tanh => Cell::Tanh,
                };
                Ok(Recurrent {
                    norm: pack(&stream, &layer.norm, Layout::Channels, dtype)?,
                    recurrent: Rnn::load(&stream, &layer.params, layer.hidden, dtype, cell)?,
                    output: pack(&stream, &layer.output, Layout::Linear, dtype)?,
                })
            })
            .collect::<Result<_>>()?;
        let masks = model
            .masks
            .iter()
            .map(|stem| Mask::load(&stream, stem, dtype))
            .collect::<Result<_>>()?;
        let scatter = if model.bands.overlap {
            let indices = model
                .bands
                .ranges
                .iter()
                .flat_map(|band| band.start..band.end)
                .map(i32::try_from)
                .collect::<std::result::Result<Vec<_>, _>>()?;
            Some(stream.indices(&indices, &[1, 1, i32::try_from(indices.len())?, 1])?)
        } else {
            None
        };
        Ok(Self {
            stream,
            dtype,
            analysis,
            inverse,
            padding,
            center: spectrum.center,
            version: model.version,
            projections,
            recurrent,
            masks,
            scatter,
            complex: model.mask_config.complex,
            activation: model.mask_config.activation.clone(),
        })
    }

    pub(super) fn run(&self, audio: &[f32], batch: usize, samples: usize) -> Result<Vec<f32>> {
        let stream = &self.stream;
        let spectrum = self
            .analysis
            .analyze_with_padding(stream, audio, samples, self.padding)?;
        let parts = stream.stack(&[stream.real(&spectrum)?, stream.imag(&spectrum)?], 3)?;
        let part_axis = if self.version == Version::V1 { 3 } else { 2 };
        let parts = if self.version == Version::V1 {
            stream.transpose(&parts, &[0, 1, 3, 2])?
        } else {
            parts
        };
        let parts = stream.cast(&parts, self.dtype)?;
        let bands = self
            .projections
            .iter()
            .map(|layer| layer.run(stream, &parts, part_axis))
            .collect::<Result<Vec<_>>>()?;
        let mut features = stream.stack(&bands, 1)?;
        for layer in &self.recurrent {
            features = layer.run(stream, &features)?;
            features = stream.transpose(&features, &[0, 2, 1, 3])?;
        }
        let plane = samples * CHANNELS;
        let item = plane * self.masks.len();
        let mut output = vec![0.; batch * item];
        for (stem, mask) in self.masks.iter().enumerate() {
            let masks = mask.run(
                stream,
                &features,
                self.dtype,
                &self.activation,
                self.complex,
            )?;
            let masks = if let Some(indices) = &self.scatter {
                let [tracks, time, _, values] = <[i32; 4]>::try_from(masks.shape())?;
                let instructions = stream.broadcast(indices, masks.shape())?;
                let base = stream.zeros(&[tracks, time, self.analysis.bins, values])?;
                stream.scatter_add(&base, &instructions, &masks, 2)?
            } else {
                masks
            };
            let first = stream.select(&masks, 3, 0)?;
            let mask = if self.complex {
                stream.complex(&first, &stream.select(&masks, 3, 1)?)?
            } else {
                first
            };
            let estimate = stream.mul(&spectrum, &mask)?;
            let wave =
                self.inverse
                    .synthesize_centered(stream, &estimate, Some(samples), self.center)?;
            let wave = stream.to_floats(&wave)?;
            ensure!(
                wave.len() == batch * plane,
                "unexpected Bandit output shape"
            );
            for (range, samples) in output.chunks_exact_mut(item).zip(wave.chunks_exact(plane)) {
                range[plane * stem..plane * (stem + 1)].copy_from_slice(samples);
            }
        }
        Ok(output)
    }
}

fn pack(stream: &Stream, layer: &super::Affine, layout: Layout, dtype: DType) -> Result<Affine> {
    Affine::load(stream, &layer.weight, &layer.bias, layout, dtype)
}

struct Projection {
    norm: Affine,
    linear: Affine,
    start: i32,
    width: i32,
}

impl Projection {
    fn run(&self, stream: &Stream, input: &Array, axis: usize) -> Result<Array> {
        let bands = stream.slice(input, axis, self.start, self.start + self.width, 1)?;
        let shape = input.shape();
        let value = stream.reshape(&bands, &[shape[0], shape[1], 2 * self.width])?;
        let value = self.norm.layer_norm(stream, &value)?;
        self.linear.linear(stream, &value)
    }
}

struct Recurrent {
    norm: Affine,
    recurrent: Rnn,
    output: Affine,
}

impl Recurrent {
    fn run(&self, stream: &Stream, input: &Array) -> Result<Array> {
        let [batch, lanes, steps, embedding] = <[i32; 4]>::try_from(input.shape())?;
        let value = self.norm.layer_norm(stream, input)?;
        let value = stream.reshape(&value, &[batch * lanes, steps, embedding])?;
        let value = self.recurrent.run(stream, &value)?;
        let value = stream.reshape(&value, &[batch, lanes, steps, -1])?;
        stream.add(input, &self.output.linear(stream, &value)?)
    }
}

struct Mask {
    bands: Vec<MaskBand>,
}

struct MaskBand {
    norm: Affine,
    hidden: Affine,
    output: Affine,
    weight: Option<Array>,
    width: i32,
}

impl Mask {
    fn load(stream: &Stream, stem: &super::StemMask, dtype: DType) -> Result<Self> {
        let bands = stem
            .bands
            .iter()
            .zip(&stem.frequencies)
            .map(|(layer, weight)| {
                Ok(MaskBand {
                    norm: pack(stream, &layer.norm, Layout::Channels, dtype)?,
                    hidden: pack(stream, &layer.hidden, Layout::Linear, dtype)?,
                    output: pack(stream, &layer.output, Layout::Linear, dtype)?,
                    weight: weight
                        .as_ref()
                        .map(|weight| stream.upload(weight, DType::F32))
                        .transpose()?,
                    width: i32::try_from(layer.width)?,
                })
            })
            .collect::<Result<_>>()?;
        Ok(Self { bands })
    }

    fn run(
        &self,
        stream: &Stream,
        features: &Array,
        dtype: DType,
        activation: &Activation,
        complex: bool,
    ) -> Result<Array> {
        let [tracks, _, time, _] = <[i32; 4]>::try_from(features.shape())?;
        let fields = 1 + i32::from(complex);
        let values = self
            .bands
            .iter()
            .enumerate()
            .map(|(index, layer)| {
                let value = stream.select(features, 1, i32::try_from(index)?)?;
                let value = layer.norm.layer_norm(stream, &value)?;
                let value = layer.hidden.linear(stream, &value)?;
                let value = activate(stream, value, activation, dtype)?;
                let value = layer.output.linear(stream, &value)?;
                let value = stream.reshape(
                    &glu(stream, &value, 2)?,
                    &[tracks, time, layer.width, fields],
                )?;
                let value = stream.cast(&value, DType::F32)?;
                if let Some(weight) = &layer.weight {
                    stream.mul(&value, weight)
                } else {
                    Ok(value)
                }
            })
            .collect::<Result<Vec<_>>>()?;
        stream.concat(&values, 2)
    }
}

fn activate(stream: &Stream, input: Array, activation: &Activation, dtype: DType) -> Result<Array> {
    match activation {
        Activation::Tanh => stream.tanh(&input),
        Activation::Relu => stream.max(&input, &stream.scalar(0.0, dtype)?),
        Activation::Relu6 => {
            let lower = stream.max(&input, &stream.scalar(0.0, dtype)?)?;
            stream.min(&lower, &stream.scalar(6.0, dtype)?)
        }
        Activation::Silu => silu(stream, &input),
        Activation::Sigmoid => stream.sigmoid(&input),
        Activation::Mish => {
            let softplus = stream.logaddexp(&input, &stream.scalar(0., dtype)?)?;
            stream.mul(&input, &stream.tanh(&softplus)?)
        }
        Activation::Gelu(approximation) => match approximation.as_str() {
            "none" => gelu(stream, &input, dtype),
            "tanh" => {
                let input = stream.cast(&input, DType::F32)?;
                let cube = stream.mul(&input, &stream.square(&input)?)?;
                let cubic = stream.mul(&cube, &stream.scalar(0.044715, DType::F32)?)?;
                let polynomial = stream.add(&input, &cubic)?;
                let scale = stream.scalar((2.0 / std::f32::consts::PI).sqrt(), DType::F32)?;
                let value = stream.tanh(&stream.mul(&polynomial, &scale)?)?;
                let value = stream.add(&value, &stream.scalar(1.0, DType::F32)?)?;
                let input = stream.mul(&input, &stream.scalar(0.5, DType::F32)?)?;
                stream.cast(&stream.mul(&input, &value)?, dtype)
            }
            value => bail!("unsupported Bandit GELU approximation {value:?}"),
        },
        Activation::LeakyRelu(slope) | Activation::Elu(slope) => {
            let zero = stream.scalar(0.0, dtype)?;
            let positive = stream.max(&input, &zero)?;
            let negative = stream.min(&input, &zero)?;
            let negative = if matches!(activation, Activation::Elu(_)) {
                stream.expm1(&negative)?
            } else {
                negative
            };
            let negative = stream.mul(&negative, &stream.scalar(*slope as f32, dtype)?)?;
            stream.add(&positive, &negative)
        }
        Activation::Identity => Ok(input),
    }
}
