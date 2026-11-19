use anyhow::{Context, Result};
use tch::Device;

use super::super::super::{
    mlx::{Array, DType, Stream},
    mlx_layers::{Affine, Cell, Layout, Rnn, gelu, glu, silu},
    mlx_stft::Stft,
};
use super::{Config, TorchNetwork, band_padding};

pub(super) struct Network {
    stream: Stream,
    dtype: DType,
    stft: Stft,
    encoder: Vec<Encoder>,
    separator: Vec<[Recurrent; 2]>,
    decoder: Vec<Decoder>,
}

impl Network {
    pub(super) fn load(
        config: &Config,
        torch: &TorchNetwork,
        device: Device,
        dtype: DType,
    ) -> Result<Self> {
        let stream = Stream::new(device)?;
        let side = (config.nfft - config.win_size) / 2;
        let window = torch
            .window
            .f_constant_pad_nd([side, config.nfft - config.win_size - side])?;
        let stft = Stft::new(
            &stream,
            config.nfft,
            config.hop_size,
            config.normalized,
            &window,
        )?;
        let load =
            |value: &super::Affine, layout: Layout| load_affine(&stream, value, layout, dtype);
        let encoder = torch
            .encoders
            .iter()
            .map(|layer| -> Result<_> {
                let bands = layer
                    .bands
                    .iter()
                    .map(|band| -> Result<_> {
                        let blocks = band
                            .layers
                            .iter()
                            .map(|block| -> Result<_> {
                                Ok(Block {
                                    norm: load(&block.norm, Layout::Channels)?,
                                    expand: load(&block.expand, Layout::Conv1d)?,
                                    depthwise: load(&block.depthwise, Layout::Conv1d)?,
                                    hidden_norm: load(&block.hidden_norm, Layout::Channels)?,
                                    project: load(&block.project, Layout::Conv1d)?,
                                })
                            })
                            .collect::<Result<_>>()?;
                        Ok(Band {
                            down: load(&band.down, Layout::Conv2d)?,
                            blocks,
                        })
                    })
                    .collect::<Result<_>>()?;
                Ok(Encoder {
                    bands,
                    global: load(&layer.global, Layout::Conv2d)?,
                })
            })
            .collect::<Result<_>>()?;
        let recurrent = |path: &super::RecurrentPath| -> Result<Recurrent> {
            Ok(Recurrent {
                norm: load(&path.norm, Layout::Channels)?,
                lstm: Rnn::load(&stream, &path.params, path.hidden, dtype, Cell::Lstm)?,
                project: load(&path.project, Layout::Linear)?,
            })
        };
        let separator = torch
            .dual_path
            .iter()
            .map(|path| Ok([recurrent(&path.frequency)?, recurrent(&path.time)?]))
            .collect::<Result<_>>()?;
        let decoder = torch
            .decoders
            .iter()
            .map(|layer| -> Result<_> {
                Ok(Decoder {
                    fusion: load(&layer.fusion, Layout::Conv2d)?,
                    up: layer
                        .up
                        .iter()
                        .map(|affine| load(affine, Layout::Transpose2d))
                        .collect::<Result<_>>()?,
                })
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            stream,
            dtype,
            stft,
            encoder,
            separator,
            decoder,
        })
    }

    pub(super) fn run(
        &self,
        config: &Config,
        audio: &[f32],
        batch: usize,
        samples: i64,
        padding: i64,
    ) -> Result<Vec<f32>> {
        let stream = &self.stream;
        let samples = usize::try_from(samples)?;
        let padding = usize::try_from(padding)?;
        let length = samples
            .checked_add(padding)
            .context("SCNet sample length overflow")?;
        let tracks = audio.len() / samples;
        let count = tracks
            .checked_mul(length)
            .context("SCNet audio size overflow")?;
        let mut padded = Vec::with_capacity(count);
        for track in audio.chunks_exact(samples) {
            padded.extend_from_slice(track);
            padded.resize(padded.len() + padding, 0.0);
        }
        let spectrum = self.stft.analyze(stream, &padded, length)?;
        let time = spectrum.shape()[1];
        let freq = self.stft.bins;
        let batch = i32::try_from(batch)?;
        let channels = i32::try_from(config.audio_channels)?;
        let stems = i32::try_from(config.sources.len())?;
        let spectrum = stream.transpose(&spectrum, &[0, 2, 1])?;
        let components = stream.stack(&[stream.real(&spectrum)?, stream.imag(&spectrum)?], 1)?;
        let mut value = stream.reshape(&components, &[batch, 2 * channels, freq, time])?;
        value = stream.cast(&value, self.dtype)?;
        let mut skips = Vec::with_capacity(self.encoder.len());
        for layer in &self.encoder {
            let (compressed, skip) = layer.run(stream, config, &value, self.dtype)?;
            value = compressed;
            skips.push(skip);
        }
        for (index, paths) in self.separator.iter().enumerate() {
            value = separate(stream, &value, paths)?;
            let frames = value.shape()[3];
            value = if index % 2 == 0 {
                let value = stream.cast(&value, DType::F32)?;
                let spectrum = stream.fft(&value, frames, true)?;
                stream.concat(&[stream.real(&spectrum)?, stream.imag(&spectrum)?], 1)?
            } else {
                let channels = value.shape()[1];
                let real = stream.slice(&value, 1, 0, channels / 2, 1)?;
                let imaginary = stream.slice(&value, 1, channels / 2, channels, 1)?;
                let real = stream.cast(&real, DType::F32)?;
                let imaginary = stream.cast(&imaginary, DType::F32)?;
                stream.ifft(&stream.complex(&real, &imaginary)?, 2 * (frames - 1), true)?
            };
            value = stream.cast(&value, self.dtype)?;
        }
        for (layer, skip) in self.decoder.iter().zip(skips.into_iter().rev()) {
            value = layer.run(stream, config, &value, skip)?;
        }
        let tracks = batch
            .checked_mul(channels * stems)
            .context("SCNet track count overflow")?;
        let value = stream.reshape(&value, &[tracks, 2, freq, time])?;
        let value = stream.cast(&value, DType::F32)?;
        let spectrum =
            stream.complex(&stream.select(&value, 1, 0)?, &stream.select(&value, 1, 1)?)?;
        let spectrum = stream.transpose(&spectrum, &[0, 2, 1])?;
        let audio = self.stft.synthesize(stream, &spectrum, Some(samples))?;
        stream.to_floats(&audio)
    }
}

fn load_affine(
    stream: &Stream,
    value: &super::Affine,
    layout: Layout,
    dtype: DType,
) -> Result<Affine> {
    Affine::load(stream, &value.weight, &value.bias, layout, dtype)
}

struct Block {
    norm: Affine,
    expand: Affine,
    depthwise: Affine,
    hidden_norm: Affine,
    project: Affine,
}

impl Block {
    fn run(&self, stream: &Stream, value: &Array, padding: i32) -> Result<Array> {
        let hidden = self.norm.group_norm_one(stream, value)?;
        let hidden = self.expand.conv1d(stream, &hidden, padding, 1)?;
        let hidden = glu(stream, &hidden, 1)?;
        let hidden = self
            .depthwise
            .conv1d(stream, &hidden, padding, self.depthwise.channels())?;
        let hidden = self.hidden_norm.group_norm_one(stream, &hidden)?;
        let swish = silu(stream, &hidden)?;
        stream.add(value, &self.project.conv1d(stream, &swish, 0, 1)?)
    }
}

struct Band {
    down: Affine,
    blocks: Vec<Block>,
}

struct Encoder {
    bands: Vec<Band>,
    global: Affine,
}

struct Skip {
    values: Array,
    widths: [i32; 3],
    original_widths: [i64; 3],
}

impl Encoder {
    fn run(
        &self,
        stream: &Stream,
        config: &Config,
        input: &Array,
        dtype: DType,
    ) -> Result<(Array, Skip)> {
        let original_widths = config.split_bands(i64::from(input.shape()[2]))?;
        let mut widths = [0; 3];
        let mut start = 0;
        let mut bands = Vec::with_capacity(3);
        for (index, band) in self.bands.iter().enumerate() {
            let width = original_widths[index];
            let stride = config.band_stride[index];
            let pad = band_padding(width, stride, config.band_kernel[index]);
            let stop = start + i32::try_from(width)?;
            let band_input = stream.slice(input, 2, start, stop, 1)?;
            let band_input = pad_band(stream, band_input, i32::try_from(pad)?, dtype)?;
            let reduced =
                band.down
                    .conv2d(stream, &band_input, [i32::try_from(stride)?, 1], [0, 0])?;
            let [batch, channels, freq, time] = <[i32; 4]>::try_from(reduced.shape())?;
            let reduced = stream.transpose(&reduced, &[0, 2, 1, 3])?;
            let mut value = stream.reshape(&reduced, &[batch * freq, channels, time])?;
            for block in &band.blocks {
                value = block.run(stream, &value, i32::try_from(config.conv_kernel / 2)?)?;
            }
            let value = stream.reshape(&value, &[batch, freq, channels, time])?;
            let value = stream.transpose(&value, &[0, 2, 1, 3])?;
            bands.push(gelu(stream, &value, dtype)?);
            widths[index] = freq;
            start = stop;
        }
        let values = stream.concat(&bands, 2)?;
        let combined = self.global.conv2d(stream, &values, [1, 1], [1, 1])?;
        Ok((
            combined,
            Skip {
                values,
                widths,
                original_widths,
            },
        ))
    }
}

fn pad_band(stream: &Stream, input: Array, pad: i32, dtype: DType) -> Result<Array> {
    if pad == 0 {
        return Ok(input);
    }
    let mut shape = input.shape().to_vec();
    shape[2] = pad / 2;
    let left = stream.cast(&stream.zeros(&shape)?, dtype)?;
    shape[2] = pad - pad / 2;
    let right = stream.cast(&stream.zeros(&shape)?, dtype)?;
    stream.concat(&[left, input, right], 2)
}

struct Recurrent {
    norm: Affine,
    lstm: Rnn,
    project: Affine,
}

impl Recurrent {
    fn run(&self, stream: &Stream, value: &Array) -> Result<Array> {
        self.project.linear(stream, &self.lstm.run(stream, value)?)
    }
}

fn separate(stream: &Stream, input: &Array, paths: &[Recurrent; 2]) -> Result<Array> {
    let [batch, channels, freq, time] = <[i32; 4]>::try_from(input.shape())?;
    let frequency = paths[0].norm.group_norm_one(stream, input)?;
    let frequency = stream.transpose(&frequency, &[0, 3, 2, 1])?;
    let frequency = stream.reshape(&frequency, &[batch * time, freq, channels])?;
    let frequency = paths[0].run(stream, &frequency)?;
    let frequency = stream.reshape(&frequency, &[batch, time, freq, channels])?;
    let frequency = stream.transpose(&frequency, &[0, 3, 2, 1])?;
    let hidden = stream.add(input, &frequency)?;
    let temporal = paths[1].norm.group_norm_one(stream, &hidden)?;
    let temporal = stream.transpose(&temporal, &[0, 2, 3, 1])?;
    let temporal = stream.reshape(&temporal, &[batch * freq, time, channels])?;
    let temporal = paths[1].run(stream, &temporal)?;
    let temporal = stream.reshape(&temporal, &[batch, freq, time, channels])?;
    let temporal = stream.transpose(&temporal, &[0, 3, 1, 2])?;
    stream.add(&hidden, &temporal)
}

struct Decoder {
    fusion: Affine,
    up: Vec<Affine>,
}

impl Decoder {
    fn run(&self, stream: &Stream, config: &Config, input: &Array, skip: Skip) -> Result<Array> {
        let merged = stream.add(input, &skip.values)?;
        let [batch, channels, freq, time] = <[i32; 4]>::try_from(merged.shape())?;
        let merged = stream.reshape(&merged, &[batch, 1, channels, freq, time])?;
        let merged = stream.broadcast(&merged, &[batch, 2, channels, freq, time])?;
        let merged = stream.reshape(&merged, &[batch, channels * 2, freq, time])?;
        let fused = self.fusion.conv2d(stream, &merged, [1, 1], [1, 1])?;
        let fused = glu(stream, &fused, 1)?;
        let mut start = 0;
        let mut bands = Vec::with_capacity(3);
        for (index, up) in self.up.iter().enumerate() {
            let stop = start + skip.widths[index];
            let section = stream.slice(&fused, 2, start, stop, 1)?;
            let stride = i32::try_from(config.band_stride[index])?;
            let section = up.upsample2d(stream, &section, [stride, 1])?;
            let original = i32::try_from(skip.original_widths[index])?;
            let trim = (original - section.shape()[2]).abs() / 2;
            bands.push(stream.slice(&section, 2, trim, trim + original, 1)?);
            start = stop;
        }
        stream.concat(&bands, 2)
    }
}
