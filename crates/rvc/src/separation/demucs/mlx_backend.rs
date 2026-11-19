use std::collections::HashMap;

use anyhow::{Context, Result, ensure};
use tch::{Device, Tensor};

use super::super::{
    mlx::{Array, DType, Stream},
    mlx_layers::{gelu, glu, stable_mean, standardize},
    mlx_stft::Stft,
};
use super::{Config, Demucs, Stage};

pub(super) struct Network {
    parameters: Parameters,
    stft: Stft,
}

pub(super) struct Parameters {
    pub(super) stream: Stream,
    pub(super) dtype: DType,
    weights: HashMap<String, Array>,
}

impl Network {
    pub(super) fn load(
        weights: HashMap<String, Tensor>,
        config: &Config,
        window: &Tensor,
        device: Device,
        dtype: DType,
    ) -> Result<Self> {
        let stream = Stream::new(device)?;
        let stft = Stft::new(&stream, config.nfft, config.nfft / 4, true, window)?;
        let weights = weights
            .into_iter()
            .map(|(name, value)| {
                let transpose_conv = name.ends_with(".conv_tr.weight");
                let packed = match (value.dim(), transpose_conv) {
                    (3, false) => value.f_permute([0, 2, 1])?,
                    (3, true) => value.f_permute([1, 2, 0])?,
                    (4, false) => value.f_permute([0, 2, 3, 1])?,
                    (4, true) => value.f_permute([1, 2, 3, 0])?,
                    _ => value,
                };
                Ok((name, stream.upload(&packed, dtype)?))
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            parameters: Parameters {
                stream,
                dtype,
                weights,
            },
            stft,
        })
    }

    pub(super) fn run(
        &self,
        model: &Demucs,
        input: &[f32],
        count: usize,
        samples: i64,
    ) -> Result<Vec<f32>> {
        let params = &self.parameters;
        let stream = &params.stream;
        let batch = i32::try_from(count)?;
        let channels = i32::try_from(model.channels)?;
        let stems = i32::try_from(model.stems)?;
        let original = i32::try_from(samples)?;
        let length = if model.config.use_train_segment {
            original.max(i32::try_from(model.training_frames)?)
        } else {
            original
        };
        let tracks = batch
            .checked_mul(channels)
            .context("Demucs input batch overflow")?;
        let sample_count = (tracks as usize)
            .checked_mul(length as usize)
            .context("Demucs input size overflow")?;
        let mut audio = Vec::with_capacity(sample_count);
        for values in input.chunks_exact(original as usize) {
            audio.extend_from_slice(values);
            audio.resize(audio.len() + (length - original) as usize, 0.0);
        }
        let spectrum = self.spectrum(model, &audio, length)?;
        let [_, frames, frequencies] = <[i32; 3]>::try_from(spectrum.shape())?;
        let spectrum = stream.transpose(&spectrum, &[0, 2, 1])?;
        let value = stream.stack(&[stream.real(&spectrum)?, stream.imag(&spectrum)?], 1)?;
        let value = stream.reshape(&value, &[batch, channels * 2, frequencies, frames])?;
        let (mut spectrogram, scale, offset) = params.normalize(&value, &[1, 2, 3])?;
        let mix = stream.float32(&audio, &[batch, channels, length])?;
        let (mut waveform, wave_scale, wave_offset) = params.normalize(&mix, &[1, 2])?;
        let mut wave_skips = Vec::with_capacity(model.time_encoders.len());
        let mut wave_lengths = Vec::with_capacity(model.time_encoders.len());
        let mut skips = Vec::with_capacity(model.encoders.len());
        for (index, encoder) in model.encoders.iter().enumerate() {
            let mut inject = None;
            if let Some(time) = model.time_encoders.get(index) {
                wave_lengths.push(waveform.shape()[2]);
                waveform = params.encode(
                    time,
                    &format!("tencoder.{index}"),
                    &waveform,
                    None,
                    &model.config,
                )?;
                if time.empty() {
                    let [batch, channels, frames] = <[i32; 3]>::try_from(waveform.shape())?;
                    inject = Some(stream.reshape(&waveform, &[batch, channels, 1, frames])?);
                } else {
                    wave_skips.push(stream.reshape(&waveform, waveform.shape())?);
                }
            }
            let before = last(&spectrogram)?;
            spectrogram = params.encode(
                encoder,
                &format!("encoder.{index}"),
                &spectrogram,
                inject.as_ref(),
                &model.config,
            )?;
            if index == 0
                && model.config.freq_emb != 0.0
                && let Some(embedding) = params.weights.get("freq_emb.embedding.weight")
            {
                let [bins, channels] = <[i32; 2]>::try_from(embedding.shape())?;
                let embedding = stream.transpose(embedding, &[1, 0])?;
                let embedding = stream.reshape(&embedding, &[1, channels, bins, 1])?;
                let gain = stream.scalar(
                    (model.config.freq_emb * model.config.emb_scale) as f32,
                    params.dtype,
                )?;
                spectrogram = stream.add(&spectrogram, &stream.mul(&embedding, &gain)?)?;
            }
            skips.push((stream.reshape(&spectrogram, spectrogram.shape())?, before));
        }
        if let Some(transformer) = &model.transformer {
            (spectrogram, waveform) = transformer.forward_mlx(params, spectrogram, waveform)?;
        }
        for (decode_index, (index, (skip, before))) in
            skips.into_iter().enumerate().rev().enumerate()
        {
            let (next, pre) = params.decode(
                &model.encoders[index],
                &format!("decoder.{decode_index}"),
                spectrogram,
                Some(&skip),
                before,
                index == 0,
                &model.config,
            )?;
            spectrogram = next;
            if let Some(time) = model.time_encoders.get(index) {
                let length = wave_lengths.pop().context("missing MLX waveform length")?;
                let name = format!("tdecoder.{}", model.time_encoders.len() - index - 1);
                let (input, skip) = if time.empty() {
                    (stream.select(&pre, 2, 0)?, None)
                } else {
                    (waveform, wave_skips.pop())
                };
                (waveform, _) = params.decode(
                    time,
                    &name,
                    input,
                    skip.as_ref(),
                    length,
                    index == 0,
                    &model.config,
                )?;
            }
        }
        let spectrogram = stream.reshape(&spectrogram, &[batch, stems, -1, frequencies, frames])?;
        let spectrogram = denormalize(stream, &spectrogram, &scale, &offset)?;
        let frequency_audio =
            self.waveform(model, &spectrogram, batch * channels * stems, length)?;
        let waveform = stream.reshape(&waveform, &[batch, stems, channels, length])?;
        let waveform = denormalize(stream, &waveform, &wave_scale, &wave_offset)?;
        let combined = stream.add(&frequency_audio, &waveform)?;
        let combined = stream.slice(&combined, 3, 0, original, 1)?;
        let result = stream.to_floats(&combined)?;
        ensure!(
            result.iter().all(|value| value.is_finite()),
            "MLX Demucs produced nonfinite audio"
        );
        Ok(result)
    }

    fn spectrum(&self, model: &Demucs, audio: &[f32], length: i32) -> Result<Array> {
        let stream = &self.parameters.stream;
        let hop = i32::try_from(model.config.nfft / 4)?;
        let frames = (length + hop - 1) / hop;
        let pad = hop / 2 * 3;
        let tail = pad + frames * hop - length;
        let padded_length = usize::try_from(length + pad + tail)?;
        let mut padded = Vec::new();
        for track in audio.chunks_exact(length as usize) {
            if model.config.hybrid_old {
                padded.resize(padded.len() + pad as usize, 0.0);
                padded.extend_from_slice(track);
                padded.resize(padded.len() + tail as usize, 0.0);
            } else {
                reflect(track, pad as usize, tail as usize, &mut padded);
            }
        }
        let spectrum = self.stft.analyze(stream, &padded, padded_length)?;
        let spectrum = stream.slice(&spectrum, 1, 2, 2 + frames, 1)?;
        stream.slice(&spectrum, 2, 0, self.stft.bins - 1, 1)
    }

    fn waveform(&self, model: &Demucs, value: &Array, tracks: i32, length: i32) -> Result<Array> {
        let stream = &self.parameters.stream;
        let [batch, stems, _, bins, frames] = <[i32; 5]>::try_from(value.shape())?;
        let value = stream.reshape(value, &[tracks, 2, bins, frames])?;
        let last_bin = stream.zeros(&[tracks, 2, 1, frames])?;
        let value = stream.concat(&[value, last_bin], 2)?;
        let left = stream.zeros(&[tracks, 2, bins + 1, 2])?;
        let right = stream.zeros(&[tracks, 2, bins + 1, 2])?;
        let value = stream.concat(&[left, value, right], 3)?;
        let value = stream.complex(&stream.select(&value, 1, 0)?, &stream.select(&value, 1, 1)?)?;
        let value = stream.transpose(&value, &[0, 2, 1])?;
        let hop = i32::try_from(model.config.nfft / 4)?;
        let start = if model.config.hybrid_old {
            0
        } else {
            hop / 2 * 3
        };
        let padded = ((length + hop - 1) / hop) * hop + 2 * start;
        let audio = self
            .stft
            .synthesize(stream, &value, Some(usize::try_from(padded)?))?;
        let audio = stream.slice(&audio, 1, start, start + length, 1)?;
        stream.reshape(
            &audio,
            &[batch, stems, i32::try_from(model.channels)?, length],
        )
    }
}

fn reflect(input: &[f32], left: usize, right: usize, output: &mut Vec<f32>) {
    let extra = (left.max(right) + 1).saturating_sub(input.len());
    if extra == 0 {
        output.extend(crate::audio::reflect(input, left, right));
        return;
    }
    let extra_right = right.min(extra);
    let extra_left = extra - extra_right;
    let mut padded = vec![0.0; extra_left];
    padded.extend_from_slice(input);
    padded.resize(padded.len() + extra_right, 0.0);
    output.extend(crate::audio::reflect(
        &padded,
        left - extra_left,
        right - extra_right,
    ));
}

fn last(value: &Array) -> Result<i32> {
    value
        .shape()
        .last()
        .copied()
        .context("MLX Demucs array has no sequence dimension")
}

fn denormalize(stream: &Stream, x: &Array, scale: &Array, offset: &Array) -> Result<Array> {
    let mut shape = scale.shape().to_vec();
    shape.insert(1, 1);
    let scale = stream.reshape(scale, &shape)?;
    let offset = stream.reshape(offset, &shape)?;
    let x = stream.cast(x, DType::F32)?;
    stream.add(&stream.mul(&x, &scale)?, &offset)
}

impl Parameters {
    pub(super) fn get(&self, key: &str) -> Result<&Array> {
        self.weights
            .get(key)
            .with_context(|| format!("Demucs checkpoint is missing {key}"))
    }

    pub(super) fn optional(&self, key: &str) -> Option<&Array> {
        self.weights.get(key)
    }

    fn bias(&self, value: Array, name: &str) -> Result<Array> {
        if let Some(bias) = self.optional(&format!("{name}.bias")) {
            self.stream.add(&value, bias)
        } else {
            Ok(value)
        }
    }

    pub(super) fn linear_parameters(
        &self,
        input: &Array,
        weight: &Array,
        bias: Option<&Array>,
    ) -> Result<Array> {
        let weight = self.stream.transpose(weight, &[1, 0])?;
        let result = self.stream.matmul(input, &weight)?;
        if let Some(bias) = bias {
            self.stream.add(&result, bias)
        } else {
            Ok(result)
        }
    }

    pub(super) fn linear(&self, input: &Array, name: &str) -> Result<Array> {
        self.linear_parameters(
            input,
            self.get(&format!("{name}.weight"))?,
            self.optional(&format!("{name}.bias")),
        )
    }

    pub(super) fn conv1d(
        &self,
        input: &Array,
        name: &str,
        stride: i32,
        pad: i32,
        dilation: i32,
    ) -> Result<Array> {
        let stream = &self.stream;
        let weight = self.get(&format!("{name}.weight"))?;
        let native = stream.transpose(input, &[0, 2, 1])?;
        let value = stream.convolution1d(&native, weight, stride, pad, dilation, 1)?;
        stream.transpose(&self.bias(value, name)?, &[0, 2, 1])
    }

    fn conv2d(&self, input: &Array, name: &str, stride: [i32; 2], pad: [i32; 2]) -> Result<Array> {
        let stream = &self.stream;
        let weight = self.get(&format!("{name}.weight"))?;
        let native = stream.transpose(input, &[0, 2, 3, 1])?;
        let value = stream.conv2d(&native, weight, stride, pad)?;
        stream.transpose(&self.bias(value, name)?, &[0, 3, 1, 2])
    }

    fn upsample(&self, input: &Array, name: &str, stride: i32, frequency: bool) -> Result<Array> {
        let stream = &self.stream;
        let weight = self.get(&format!("{name}.weight"))?;
        if frequency {
            let native = stream.transpose(input, &[0, 2, 3, 1])?;
            let value = stream.upsample2d(&native, weight, [stride, 1])?;
            stream.transpose(&self.bias(value, name)?, &[0, 3, 1, 2])
        } else {
            let native = stream.transpose(input, &[0, 2, 1])?;
            let value = stream.upsample1d(&native, weight, stride)?;
            stream.transpose(&self.bias(value, name)?, &[0, 2, 1])
        }
    }

    pub(super) fn normalize_layer(&self, name: &str, input: &Array, groups: i64) -> Result<Array> {
        let stream = &self.stream;
        let Some(weight) = self.optional(&format!("{name}.weight")) else {
            return stream.reshape(input, input.shape());
        };
        let bias = self.get(&format!("{name}.bias"))?;
        let shape = input.shape();
        let (value, weight, bias) = if groups > 0 {
            let groups = i32::try_from(groups)?;
            ensure!(
                shape[1] % groups == 0,
                "invalid Demucs group count in {name}"
            );
            let grouped = stream.reshape(input, &[shape[0], groups, -1])?;
            let group = standardize(stream, &grouped, &[-1], 1e-5)?;
            let value = stream.reshape(&group, shape)?;
            let mut affine = vec![1; shape.len()];
            affine[1] = shape[1];
            (
                value,
                stream.reshape(weight, &affine)?,
                stream.reshape(bias, &affine)?,
            )
        } else {
            (
                standardize(stream, input, &[-1], 1e-5)?,
                stream.reshape(weight, weight.shape())?,
                stream.reshape(bias, bias.shape())?,
            )
        };
        let value = stream.cast(&value, self.dtype)?;
        stream.add(&stream.mul(&value, &weight)?, &bias)
    }

    fn normalize(&self, value: &Array, axes: &[i32]) -> Result<(Array, Array, Array)> {
        let stream = &self.stream;
        let offset = stable_mean(stream, value, axes)?;
        let shifted = stream.sub(value, &offset)?;
        let var = stable_mean(stream, &stream.square(&shifted)?, axes)?;
        let samples = axes
            .iter()
            .map(|&axis| f64::from(value.shape()[axis as usize]))
            .product::<f64>();
        let correction = stream.scalar((samples / (samples - 1.0)) as f32, DType::F32)?;
        let scale = stream.sqrt(&stream.mul(&var, &correction)?)?;
        let denominator = stream.add(&scale, &stream.scalar(1e-5, DType::F32)?)?;
        let value = stream.div(&shifted, &denominator)?;
        Ok((stream.cast(&value, self.dtype)?, scale, offset))
    }

    fn encode(
        &self,
        stage: &Stage,
        name: &str,
        input: &Array,
        inject: Option<&Array>,
        config: &Config,
    ) -> Result<Array> {
        let stream = &self.stream;
        let stride = i32::try_from(stage.stride)?;
        let pad = i32::try_from(stage.pad)?;
        let name_conv = format!("{name}.conv");
        let value = if stage.frequency {
            self.conv2d(input, &name_conv, [stride, 1], [pad, 0])?
        } else {
            let length = last(input)?;
            let value = stream.reshape(input, &[input.shape()[0], -1, length])?;
            let remainder = length % stride;
            let value = if remainder == 0 {
                value
            } else {
                let tail =
                    stream.zeros(&[value.shape()[0], value.shape()[1], stride - remainder])?;
                let tail = stream.cast(&tail, self.dtype)?;
                stream.concat(&[value, tail], 2)?
            };
            self.conv1d(&value, &name_conv, stride, pad, 1)?
        };
        if stage.empty() {
            return Ok(value);
        }
        let value = if let Some(inject) = inject {
            stream.add(&value, inject)?
        } else {
            value
        };
        let value = self.normalize_layer(&format!("{name}.norm1"), &value, config.norm_groups)?;
        let value = gelu(stream, &value, self.dtype)?;
        let value = self.residual(stage, name, value, config.dconv_depth)?;
        self.rewrite(stage, name, value, config.norm_groups, "norm2")
    }

    #[allow(clippy::too_many_arguments)]
    fn decode(
        &self,
        stage: &Stage,
        name: &str,
        mut value: Array,
        skip: Option<&Array>,
        length: i32,
        is_last: bool,
        config: &Config,
    ) -> Result<(Array, Array)> {
        let stream = &self.stream;
        let name_conv = format!("{name}.conv_tr");
        if stage.frequency && value.shape().len() == 3 {
            let weight = self.get(&format!("{name_conv}.weight"))?;
            let input_channels = last(weight)?;
            value = stream.reshape(
                &value,
                &[value.shape()[0], input_channels, -1, value.shape()[2]],
            )?;
        }
        if !stage.empty() {
            value = stream.add(&value, skip.context("missing MLX Demucs skip")?)?;
            value = self.rewrite(stage, name, value, config.norm_groups, "norm1")?;
            value = self.residual(stage, name, value, config.dconv_depth)?;
        }
        let up = self.upsample(
            &value,
            &name_conv,
            i32::try_from(stage.stride)?,
            stage.frequency,
        )?;
        let up = self.normalize_layer(&format!("{name}.norm2"), &up, config.norm_groups)?;
        let pad = i32::try_from(stage.pad)?;
        let end = if stage.frequency {
            up.shape()[2] - pad
        } else {
            pad + length
        };
        let up = stream.slice(&up, 2, pad, end, 1)?;
        let up = if is_last {
            up
        } else {
            gelu(stream, &up, self.dtype)?
        };
        Ok((up, value))
    }

    fn rewrite(
        &self,
        stage: &Stage,
        name: &str,
        value: Array,
        groups: i64,
        norm: &str,
    ) -> Result<Array> {
        let key = format!("{name}.rewrite");
        let Some(kernel) = self.optional(&format!("{key}.weight")) else {
            return Ok(value);
        };
        let dimensions = kernel.shape();
        let value = if stage.frequency {
            self.conv2d(&value, &key, [1, 1], [dimensions[1] / 2, dimensions[2] / 2])?
        } else {
            self.conv1d(&value, &key, 1, dimensions[1] / 2, 1)?
        };
        let value = self.normalize_layer(&format!("{name}.{norm}"), &value, groups)?;
        glu(&self.stream, &value, 1)
    }

    fn residual(&self, stage: &Stage, name: &str, mut value: Array, depth: i64) -> Result<Array> {
        let prefix = format!("{name}.dconv");
        if self
            .optional(&format!("{prefix}.layers.0.0.weight"))
            .is_none()
        {
            return Ok(value);
        }
        let stream = &self.stream;
        let geometry = if stage.frequency {
            let [batch, channels, frequency, time] = <[i32; 4]>::try_from(value.shape())?;
            value = stream.transpose(&value, &[0, 2, 1, 3])?;
            value = stream.reshape(&value, &[batch * frequency, channels, time])?;
            Some([batch, frequency, channels, time])
        } else {
            None
        };
        for index in 0..depth.unsigned_abs() {
            let name = format!("{prefix}.layers.{index}");
            let dilation = 1 << index;
            let y = self.conv1d(&value, &format!("{name}.0"), 1, dilation, dilation)?;
            let y = self.normalize_layer(&format!("{name}.1"), &y, 1)?;
            let y = gelu(stream, &y, self.dtype)?;
            let y = self.conv1d(&y, &format!("{name}.3"), 1, 0, 1)?;
            let y = self.normalize_layer(&format!("{name}.4"), &y, 1)?;
            let y = glu(stream, &y, 1)?;
            let scale = stream.reshape(self.get(&format!("{name}.6.scale"))?, &[-1, 1])?;
            value = stream.add(&value, &stream.mul(&scale, &y)?)?;
        }
        if let Some(geometry) = geometry {
            value = stream.transpose(&stream.reshape(&value, &geometry)?, &[0, 2, 1, 3])?;
        }
        Ok(value)
    }
}
