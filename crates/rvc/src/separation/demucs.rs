//! Native time/frequency Demucs models described by the bundled PyMSS configs.

mod layers;
pub(super) mod legacy;
mod mlx_backend;
#[cfg(feature = "onnx")]
mod onnx_backend;
mod transformer;

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use serde_json::Value;
use tch::{Device, Kind, Tensor};

use super::Acceleration;
use crate::checkpoint::Checkpoint;
use crate::model::weights::Weights;

use layers::Stage;
use transformer::Transformer;

#[derive(Deserialize)]
#[serde(default)]
struct Config {
    nfft: i64,
    depth: usize,
    kernel_size: i64,
    time_stride: i64,
    stride: i64,
    norm_groups: i64,
    dconv_depth: i64,
    freq_emb: f64,
    emb_scale: f64,
    num_subbands: i64,
    wiener_iters: i64,
    end_iters: Option<i64>,
    cac: bool,
    multi_freqs: Option<Vec<f64>>,
    multi_freqs_depth: usize,
    t_layers: usize,
    bottom_channels: i64,
    use_train_segment: bool,
    hybrid: bool,
    hybrid_old: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            nfft: 4096,
            depth: 4,
            kernel_size: 8,
            time_stride: 2,
            stride: 4,
            norm_groups: 4,
            dconv_depth: 2,
            freq_emb: 0.2,
            emb_scale: 10.0,
            num_subbands: 1,
            wiener_iters: 0,
            end_iters: None,
            cac: true,
            multi_freqs: None,
            multi_freqs_depth: 3,
            t_layers: 5,
            bottom_channels: 0,
            use_train_segment: false,
            hybrid: true,
            hybrid_old: false,
        }
    }
}

pub(super) struct Demucs {
    config: Config,
    encoders: Vec<Stage>,
    time_encoders: Vec<Stage>,
    transformer: Option<Transformer>,
    channels: usize,
    stems: usize,
    training_frames: i64,
    legacy: bool,
    backend: Backend,
}

enum Backend {
    Torch(TorchNetwork),
    Mlx(mlx_backend::Network),
    #[cfg(feature = "onnx")]
    Onnx(onnx_backend::Network),
}

struct TorchNetwork {
    weights: Weights,
    device: Device,
    window: Tensor,
}

impl Demucs {
    pub(super) fn from_yaml(
        document: &Value,
        path: &Path,
        device: Device,
        window_frames: usize,
    ) -> Result<Self> {
        let training = &document["training"];
        let config = &document["htdemucs"];
        let channels = training["channels"].as_u64().unwrap_or(2) as usize;
        let stems = training["instruments"]
            .as_array()
            .context("HTDemucs requires instrument names")?
            .len();
        let rate = training["samplerate"]
            .as_u64()
            .or(document["audio"]["sample_rate"].as_u64())
            .unwrap_or(44100);
        let segment = training["segment"].as_f64().unwrap_or(10.0);
        let weights = Checkpoint::open(path)?.demucs_weights(Device::Cpu)?;
        Self::load(
            config,
            weights,
            channels,
            stems,
            rate,
            segment,
            false,
            device,
            Some((document, window_frames)),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn load(
        document: &Value,
        weights: HashMap<String, Tensor>,
        channels: usize,
        stems: usize,
        rate: u64,
        segment: f64,
        legacy: bool,
        device: Device,
        inference: Option<(&Value, usize)>,
    ) -> Result<Self> {
        let mut config =
            Config::deserialize(document).context("invalid Demucs model configuration")?;
        if legacy {
            config.t_layers = 0;
            if document.get("depth").is_none() {
                config.depth = 6;
            }
            if document.get("multi_freqs_depth").is_none() {
                config.multi_freqs_depth = 2;
            }
        }
        ensure!(
            matches!(channels, 1 | 2)
                && (1..=32).contains(&stems)
                && (1..=384000).contains(&rate)
                && segment.is_finite()
                && (0.01..=600.0).contains(&segment),
            "invalid Demucs audio metadata"
        );
        ensure!(
            (16..=16384).contains(&config.nfft)
                && config.nfft % 8 == 0
                && (1..=12).contains(&config.depth)
                && (2..=64).contains(&config.kernel_size)
                && (1..=16).contains(&config.stride)
                && (1..=16).contains(&config.time_stride)
                && (1..=256).contains(&config.norm_groups)
                && (-16..=16).contains(&config.dconv_depth)
                && config.t_layers <= 32
                && (1..=16).contains(&config.num_subbands)
                && (config.nfft / 2) % config.num_subbands == 0,
            "invalid Demucs network geometry"
        );
        ensure!(
            config.cac || (!legacy && config.wiener_iters < 0),
            "this Demucs configuration requires unsupported Wiener filtering"
        );
        let training_frames = (segment * rate as f64) as i64;
        let mut encoders = Vec::with_capacity(config.depth);
        let mut time_encoders = Vec::with_capacity(config.depth);
        let mut frequencies = config.nfft / 2;
        for index in 0..config.depth {
            let freq = frequencies > 1;
            let joins_waveform = freq && frequencies <= config.kernel_size;
            let (stride, width, pad) = if joins_waveform {
                (config.stride, frequencies, 0)
            } else if freq {
                (config.stride, config.kernel_size, config.kernel_size / 4)
            } else {
                (
                    config.time_stride,
                    config.time_stride * 2,
                    config.time_stride / 2,
                )
            };
            let split = config
                .multi_freqs
                .as_deref()
                .filter(|ratios| index < config.multi_freqs_depth && !ratios.is_empty());
            encoders.push(Stage::new(freq, stride, width, pad, false, split)?);
            if config.hybrid && freq {
                time_encoders.push(Stage::new(
                    false,
                    config.stride,
                    config.kernel_size,
                    config.kernel_size / 4,
                    joins_waveform,
                    None,
                )?);
            }
            if freq {
                frequencies = if joins_waveform {
                    1
                } else {
                    frequencies / config.stride
                };
                ensure!(
                    frequencies > 0,
                    "Demucs encoders exhaust the frequency dimension"
                );
            }
        }
        let transformer = (config.t_layers > 0)
            .then(|| Transformer::new(document, config.t_layers, config.bottom_channels))
            .transpose()?;
        let spectral_device = if device == Device::Mps {
            Device::Cpu
        } else {
            device
        };
        let window =
            Tensor::f_hann_window_periodic(config.nfft, true, (Kind::Float, spectral_device))?;
        let supports_mlx = !legacy
            && config.hybrid
            && config.num_subbands == 1
            && config.cac
            && config.wiener_iters == 0
            && config.end_iters.is_none_or(|iters| iters == 0)
            && config.multi_freqs.as_ref().is_none_or(Vec::is_empty)
            && transformer.as_ref().is_none_or(Transformer::supports_mlx);
        let acceleration = inference
            .map(|(document, _)| {
                Acceleration::configured(document, device, supports_mlx, "HTDemucs")
            })
            .transpose()?
            .unwrap_or(Acceleration::Torch);
        let backend = if let Acceleration::Mlx(dtype) = acceleration {
            Backend::Mlx(mlx_backend::Network::load(
                weights, &config, &window, device, dtype,
            )?)
        } else {
            let weights = weights
                .into_iter()
                .map(|(key, tensor)| Ok((key, tensor.f_to_device(device)?)))
                .collect::<Result<_>>()?;
            Backend::Torch(TorchNetwork {
                weights: Weights::new(weights),
                device,
                window,
            })
        };
        let demucs = Self {
            config,
            encoders,
            time_encoders,
            transformer,
            channels,
            stems,
            legacy,
            training_frames,
            backend,
        };
        #[cfg(feature = "onnx")]
        let demucs = match (acceleration, &demucs.backend) {
            (Acceleration::Onnx(provider), Backend::Torch(torch)) => {
                let (_, window_frames) = inference.context("missing HTDemucs output geometry")?;
                let onnx = onnx_backend::Network::load(&demucs, torch, provider, window_frames)?;
                Self {
                    backend: Backend::Onnx(onnx),
                    ..demucs
                }
            }
            _ => demucs,
        };
        Ok(demucs)
    }

    pub(super) fn stems(&self) -> usize {
        self.stems
    }
    pub(super) fn channels(&self) -> usize {
        self.channels
    }

    pub(super) fn predict_batch(&self, audio: &[f32], count: usize) -> Result<Vec<f32>> {
        let frames = super::window_samples(audio, count, self.channels)?;
        match &self.backend {
            Backend::Torch(torch) => {
                let x = Tensor::f_from_slice(audio)?
                    .f_reshape([count as i64, self.channels as i64, frames])?
                    .f_to_device(torch.device)?;
                let prediction = tch::no_grad(|| {
                    self.forward(&torch.window, torch.device, x, |spectrum, wave| {
                        self.estimate(torch, spectrum, wave)
                    })
                })?;
                Ok(Vec::<f32>::try_from(
                    prediction.f_to_device(Device::Cpu)?.f_reshape([-1])?,
                )?)
            }
            Backend::Mlx(mlx) => mlx.run(self, audio, count, frames),
            #[cfg(feature = "onnx")]
            Backend::Onnx(onnx) => {
                let x = Tensor::f_from_slice(audio)?.f_reshape([
                    count as i64,
                    self.channels as i64,
                    frames,
                ])?;
                let prediction = tch::no_grad(|| {
                    self.forward(&onnx.window, Device::Cpu, x, |spectrum, wave| {
                        onnx.estimate(&spectrum, wave)
                    })
                })?;
                Ok(Vec::<f32>::try_from(prediction.f_reshape([-1])?)?)
            }
        }
    }

    fn spectrum(&self, window: &Tensor, device: Device, mix: &Tensor) -> Result<Tensor> {
        let nfft = self.config.nfft;
        let hop = nfft / 4;
        let sample_length = mix.size()[2];
        let frames = (sample_length + hop - 1) / hop;
        let pad = hop / 2 * 3;
        let mut input = mix.f_to_device(window.device())?;
        if self.config.hybrid {
            let right = pad + frames * hop - sample_length;
            input = if self.config.hybrid_old {
                input.f_constant_pad_nd([pad, right])?
            } else {
                reflect(&input, pad, right)?
            };
        }
        let input = reflect(&input.f_flatten(0, 1)?.f_unsqueeze(0)?, nfft / 2, nfft / 2)?
            .f_squeeze_dim(0)?;
        let mut spec = input
            .f_stft_center(
                nfft,
                hop,
                nfft,
                Some(window),
                false,
                "reflect",
                true,
                true,
                true,
                false,
            )?
            .f_narrow(1, 0, nfft / 2)?;
        if self.config.hybrid {
            spec = spec.f_narrow(2, 2, frames)?;
        }
        let count = spec.size()[2];
        Ok(spec
            .f_reshape([mix.size()[0], self.channels as i64, nfft / 2, count])?
            .f_to_device(device)?)
    }

    fn waveform(
        &self,
        window: &Tensor,
        device: Device,
        spec: &Tensor,
        length: i64,
    ) -> Result<Tensor> {
        let nfft = self.config.nfft;
        let hop = nfft / 4;
        let mut spec = spec
            .f_to_device(window.device())?
            .f_constant_pad_nd([0, 0, 0, 1])?;
        let mut padded_length = length;
        let mut start = 0;
        if self.config.hybrid {
            spec = spec.f_constant_pad_nd([2, 2])?;
            padded_length = ((length + hop - 1) / hop) * hop;
            if !self.config.hybrid_old {
                start = hop / 2 * 3;
                padded_length += 2 * start;
            }
        }
        let frames = spec.size()[4];
        let wav = spec.f_reshape([-1, nfft / 2 + 1, frames])?.f_istft(
            nfft,
            hop,
            nfft,
            Some(window),
            true,
            true,
            true,
            padded_length,
            false,
        )?;
        Ok(wav
            .f_narrow(1, start, length)?
            .f_reshape([
                spec.size()[0],
                self.stems as i64,
                self.channels as i64,
                length,
            ])?
            .f_to_device(device)?)
    }

    fn forward(
        &self,
        window: &Tensor,
        device: Device,
        mut mix: Tensor,
        estimate: impl FnOnce(Tensor, Option<&Tensor>) -> Result<(Tensor, Option<Tensor>)>,
    ) -> Result<Tensor> {
        let original_length = mix.size()[2];
        if self.config.use_train_segment && original_length < self.training_frames {
            mix = mix.f_constant_pad_nd([0, self.training_frames - original_length])?;
        }
        let length = mix.size()[2];
        let spec = self.spectrum(window, device, &mix)?;
        let (batch, freqs, frames) = (spec.size()[0], spec.size()[2], spec.size()[3]);
        let mut x = if self.config.cac {
            spec.f_view_as_real()?
                .f_permute([0, 1, 4, 2, 3])?
                .f_reshape([batch, (self.channels * 2) as i64, freqs, frames])?
        } else {
            spec.f_abs()?
        };
        if self.config.num_subbands > 1 {
            x = x.f_reshape([batch, -1, freqs / self.config.num_subbands, frames])?;
        }
        let (std, mean) = x.f_std_mean_dim(&[1, 2, 3][..], true, true)?;
        x = x.f_sub(&mean)?.f_div(&std.f_add_scalar(1e-5)?)?;
        let time = self
            .config
            .hybrid
            .then(|| -> Result<_> {
                let (std, mean) = mix.f_std_mean_dim(&[1, 2][..], true, true)?;
                let normalized = mix.f_sub(&mean)?.f_div(&std.f_add_scalar(1e-5)?)?;
                Ok((normalized, std, mean))
            })
            .transpose()?;
        let (x, xt) = estimate(x, time.as_ref().map(|(wave, _, _)| wave))?;
        let stems = x.f_reshape([batch, self.stems as i64, -1, freqs, frames])?;
        let restored = stems
            .f_mul(&std.f_unsqueeze(1)?)?
            .f_add(&mean.f_unsqueeze(1)?)?;
        let output = if self.config.cac {
            restored
                .f_reshape([
                    batch,
                    self.stems as i64,
                    self.channels as i64,
                    2,
                    freqs,
                    frames,
                ])?
                .f_permute([0, 1, 2, 4, 5, 3])?
                .f_contiguous()?
                .f_view_as_complex()?
        } else {
            let phase = spec.f_div(&spec.f_abs()?.f_add_scalar(1e-8)?)?;
            phase.f_unsqueeze(1)?.f_mul(&restored)?
        };
        let mut pcm = self.waveform(window, device, &output, length)?;
        if let (Some(y), Some((_, std, mean))) = (xt, time) {
            let y = y
                .f_reshape([batch, self.stems as i64, self.channels as i64, length])?
                .f_mul(&std.f_unsqueeze(1)?)?
                .f_add(&mean.f_unsqueeze(1)?)?;
            pcm = pcm.f_add(&y)?;
        }
        Ok(pcm.f_narrow(-1, 0, original_length)?)
    }

    fn estimate(
        &self,
        torch: &TorchNetwork,
        mut x: Tensor,
        time: Option<&Tensor>,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let mut wave_skips = Vec::with_capacity(self.time_encoders.len());
        let mut wave_lengths = Vec::with_capacity(self.time_encoders.len());
        let mut skips = Vec::with_capacity(self.encoders.len());
        let mut xt = time.map(Tensor::shallow_clone);
        for (index, encoder) in self.encoders.iter().enumerate() {
            let mut inject = None;
            if let Some(time_encoder) = self.time_encoders.get(index) {
                let wave_in = xt.as_ref().context("missing Demucs waveform branch")?;
                wave_lengths.push(wave_in.size()[2]);
                let y = time_encoder.encode(
                    &torch.weights,
                    &format!("tencoder.{index}"),
                    wave_in,
                    None,
                    &self.config,
                    self.legacy,
                )?;
                if time_encoder.empty() {
                    inject = Some(y.f_unsqueeze(2)?);
                } else {
                    wave_skips.push(y.shallow_clone());
                }
                xt = Some(y);
            }
            let before = x
                .size()
                .last()
                .copied()
                .context("invalid Demucs encoder shape")?;
            x = encoder.encode(
                &torch.weights,
                &format!("encoder.{index}"),
                &x,
                inject.as_ref(),
                &self.config,
                self.legacy,
            )?;
            if index == 0
                && self.config.freq_emb != 0.0
                && torch.weights.contains("freq_emb.embedding.weight")
            {
                let embedding = torch
                    .weights
                    .get("freq_emb.embedding.weight")?
                    .f_transpose(0, 1)?
                    .f_unsqueeze(0)?
                    .f_unsqueeze(-1)?;
                x = x.f_add(
                    &embedding.f_mul_scalar(self.config.freq_emb * self.config.emb_scale)?,
                )?;
            }
            skips.push((x.shallow_clone(), before));
        }
        if let Some(transformer) = &self.transformer {
            let y = xt.as_ref().context("missing transformer waveform input")?;
            let (spec, wave) = transformer.forward(&torch.weights, &x, y)?;
            x = spec;
            xt = Some(wave);
        }
        if self.legacy {
            x = x.f_zeros_like()?;
            if let Some(y) = &mut xt {
                *y = y.f_zeros_like()?;
            }
        }
        for (decode_index, (index, (skip, before))) in
            skips.into_iter().enumerate().rev().enumerate()
        {
            let (y, pre) = self.encoders[index].decode(
                &torch.weights,
                &format!("decoder.{decode_index}"),
                &x,
                Some(&skip),
                before,
                index == 0,
                &self.config,
                self.legacy,
            )?;
            x = y;
            if let Some(time_decoder) = self.time_encoders.get(index) {
                let audio_length = wave_lengths
                    .pop()
                    .context("missing waveform decoder length")?;
                let path = format!("tdecoder.{}", self.time_encoders.len() - index - 1);
                let (wave, _) = if time_decoder.empty() {
                    time_decoder.decode(
                        &torch.weights,
                        &path,
                        &pre.f_squeeze_dim(2)?,
                        None,
                        audio_length,
                        index == 0,
                        &self.config,
                        self.legacy,
                    )?
                } else {
                    let wave_skip = wave_skips.pop().context("missing waveform skip tensor")?;
                    time_decoder.decode(
                        &torch.weights,
                        &path,
                        xt.as_ref().context("missing waveform decoder input")?,
                        Some(&wave_skip),
                        audio_length,
                        index == 0,
                        &self.config,
                        self.legacy,
                    )?
                };
                xt = Some(wave);
            }
        }
        Ok((x, xt))
    }
}

fn reflect(x: &Tensor, left: i64, right: i64) -> Result<Tensor> {
    let last = x
        .size()
        .last()
        .copied()
        .context("missing spectral input length")?;
    let extra = (left.max(right) - last + 1).max(0);
    let extra_right = right.min(extra);
    let extra_left = extra - extra_right;
    let x = if extra == 0 {
        x.shallow_clone()
    } else {
        x.f_constant_pad_nd([extra_left, extra_right])?
    };
    Ok(x.f_reflection_pad1d([left - extra_left, right - extra_right])?)
}
