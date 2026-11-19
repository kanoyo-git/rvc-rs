//! SCNet encoder, dual-path recurrent separator and decoder from PyMSS.
//! See `licenses/pymss-core.txt`.

use std::collections::HashMap;

use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use tch::{Device, Kind, Tensor, no_grad};

use super::super::Acceleration;

#[path = "scnet/mlx_network.rs"]
mod mlx_network;
#[cfg(feature = "onnx")]
#[path = "scnet/onnx_network.rs"]
mod onnx_network;

#[derive(Debug, Deserialize)]
#[serde(default)]
struct Config {
    sources: Vec<String>,
    audio_channels: i64,
    dims: Vec<i64>,
    nfft: i64,
    hop_size: i64,
    win_size: i64,
    normalized: bool,
    #[serde(rename = "band_SR")]
    band_ratios: [f64; 3],
    band_stride: [i64; 3],
    band_kernel: [i64; 3],
    conv_depths: [i64; 3],
    compress: f64,
    conv_kernel: i64,
    num_dplayer: usize,
    expand: i64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            sources: ["drums", "bass", "other", "vocals"]
                .map(str::to_owned)
                .to_vec(),
            audio_channels: 2,
            dims: vec![4, 32, 64, 128],
            nfft: 4096,
            hop_size: 1024,
            win_size: 4096,
            normalized: true,
            band_ratios: [0.175, 0.392, 0.433],
            band_stride: [1, 4, 16],
            band_kernel: [3, 4, 16],
            conv_depths: [3, 2, 1],
            compress: 4.0,
            conv_kernel: 3,
            num_dplayer: 6,
            expand: 1,
        }
    }
}

impl Config {
    fn validate(&self) -> Result<()> {
        ensure!(
            matches!(self.audio_channels, 1 | 2)
                && (2..=16).contains(&self.dims.len())
                && self.dims[0] == 2 * self.audio_channels
                && self.dims.iter().all(|d| (1..=8192).contains(d))
                && (1..=128).contains(&self.sources.len()),
            "invalid SCNet channel or source configuration"
        );
        ensure!(
            (2..=65536).contains(&self.nfft)
                && self.nfft % 2 == 0
                && (1..=self.nfft).contains(&self.win_size)
                && (1..=self.win_size).contains(&self.hop_size),
            "invalid SCNet STFT configuration"
        );
        ensure!(
            self.band_ratios
                .iter()
                .all(|ratio| ratio.is_finite() && *ratio > 0.0)
                && self.band_ratios[0] + self.band_ratios[1] < 1.0
                && self
                    .band_stride
                    .iter()
                    .all(|value| (1..=1024).contains(value))
                && self
                    .band_kernel
                    .iter()
                    .all(|value| (1..=1024).contains(value))
                && self
                    .conv_depths
                    .iter()
                    .all(|value| (-128..=128).contains(value))
                && self.compress.is_finite()
                && self.compress > 0.0
                && (1..=1024).contains(&self.conv_kernel)
                && self.conv_kernel % 2 == 1
                && self.num_dplayer <= 128
                && self.num_dplayer.is_multiple_of(2)
                && (1..=16).contains(&self.expand),
            "invalid SCNet sparse-band or dual-path geometry"
        );
        Ok(())
    }

    fn split_bands(&self, frequencies: i64) -> Result<[i64; 3]> {
        let low = (frequencies as f64 * self.band_ratios[0]).ceil() as i64;
        let mid = (frequencies as f64 * (self.band_ratios[0] + self.band_ratios[1])).ceil() as i64;
        let widths = [low, mid - low, frequencies - mid];
        ensure!(
            widths.iter().all(|width| *width > 0),
            "SCNet frequency split has an empty band"
        );
        Ok(widths)
    }

    fn sample_padding(&self, samples: i64) -> Result<i64> {
        let hop = self.hop_size;
        let mut padding = hop - samples % hop;
        if ((samples + padding) / hop) % 2 == 0 {
            padding += hop;
        }
        ensure!(
            samples + padding > self.nfft / 2,
            "SCNet input is too short for centered STFT"
        );
        Ok(padding)
    }
}

struct Weights<'a> {
    tensors: &'a HashMap<String, Tensor>,
    device: Device,
}

impl Weights<'_> {
    fn tensor(&self, key: &str, shape: &[i64]) -> Result<Tensor> {
        let tensor = self
            .tensors
            .get(key)
            .with_context(|| format!("SCNet checkpoint is missing {key}"))?;
        ensure!(
            tensor.size() == shape,
            "SCNet tensor {key} has shape {:?}, expected {shape:?}",
            tensor.size()
        );
        ensure!(
            matches!(
                tensor.kind(),
                Kind::Float | Kind::Half | Kind::BFloat16 | Kind::Double
            ),
            "SCNet tensor {key} is not floating point"
        );
        Ok(tensor.f_to_kind(Kind::Float)?.f_to_device(self.device)?)
    }

    fn affine(&self, key: &str, weight_shape: &[i64], output: i64) -> Result<Affine> {
        Ok(Affine {
            weight: self.tensor(&format!("{key}.weight"), weight_shape)?,
            bias: self.tensor(&format!("{key}.bias"), &[output])?,
        })
    }
}

struct Affine {
    weight: Tensor,
    bias: Tensor,
}

impl Affine {
    fn group_norm(&self, x: &Tensor) -> Result<Tensor> {
        Ok(x.f_group_norm(1, Some(&self.weight), Some(&self.bias), 1e-5, true)?)
    }

    fn conv1d(&self, x: &Tensor, pad: i64, groups: i64) -> Result<Tensor> {
        Ok(x.f_conv1d(&self.weight, Some(&self.bias), 1, pad, 1, groups)?)
    }

    fn conv2d(&self, x: &Tensor, stride: [i64; 2], padding: [i64; 2]) -> Result<Tensor> {
        Ok(x.f_conv2d(&self.weight, Some(&self.bias), stride, padding, [1, 1], 1)?)
    }

    fn upsample(&self, x: &Tensor, stride: i64) -> Result<Tensor> {
        Ok(x.f_conv_transpose2d(
            &self.weight,
            Some(&self.bias),
            [stride, 1],
            [0, 0],
            [0, 0],
            1,
            [1, 1],
        )?)
    }

    fn linear(&self, x: &Tensor) -> Result<Tensor> {
        Ok(x.f_linear(&self.weight, Some(&self.bias))?)
    }
}

struct GatedConvolution {
    norm: Affine,
    expand: Affine,
    depthwise: Affine,
    hidden_norm: Affine,
    project: Affine,
}

impl GatedConvolution {
    fn load(
        weights: &Weights<'_>,
        key: &str,
        channels: i64,
        hidden: i64,
        kernel: i64,
    ) -> Result<Self> {
        Ok(Self {
            norm: weights.affine(&format!("{key}.0"), &[channels], channels)?,
            expand: weights.affine(
                &format!("{key}.1"),
                &[2 * hidden, channels, kernel],
                2 * hidden,
            )?,
            depthwise: weights.affine(&format!("{key}.3"), &[hidden, 1, kernel], hidden)?,
            hidden_norm: weights.affine(&format!("{key}.4"), &[hidden], hidden)?,
            project: weights.affine(&format!("{key}.6"), &[channels, hidden, 1], channels)?,
        })
    }

    fn forward(&self, x: &Tensor, padding: i64) -> Result<Tensor> {
        let normalized = self.norm.group_norm(x)?;
        let hidden = self.expand.conv1d(&normalized, padding, 1)?.f_glu(1)?;
        let groups = self.depthwise.bias.size()[0];
        let hidden = self.depthwise.conv1d(&hidden, padding, groups)?;
        let hidden = self.hidden_norm.group_norm(&hidden)?;
        let swish = hidden.f_mul(&hidden.f_sigmoid()?)?;
        Ok(x.f_add(&self.project.conv1d(&swish, 0, 1)?)?)
    }
}

struct BandEncoder {
    down: Affine,
    layers: Vec<GatedConvolution>,
    stride: i64,
    kernel: i64,
}

fn band_padding(width: i64, stride: i64, kernel: i64) -> i64 {
    if stride == 1 {
        kernel - stride
    } else {
        (stride - width % stride) % stride
    }
}

struct Encoder {
    bands: Vec<BandEncoder>,
    global: Affine,
}

struct Skip {
    values: Tensor,
    widths: [i64; 3],
    original_widths: [i64; 3],
}

impl Encoder {
    fn load(
        weights: &Weights<'_>,
        config: &Config,
        index: usize,
        input: i64,
        channels: i64,
    ) -> Result<Self> {
        let mut bands = Vec::with_capacity(3);
        let inner = (channels as f64 / config.compress) as i64;
        ensure!(
            (1..=8192).contains(&inner) || config.conv_depths.iter().all(|depth| *depth == 0),
            "invalid SCNet compressed convolution size"
        );
        for band in 0..3 {
            let kernel = config.band_kernel[band];
            let stride = config.band_stride[band];
            let down = weights.affine(
                &format!("encoder.{index}.SDlayer.convs.{band}"),
                &[channels, input, kernel, 1],
                channels,
            )?;
            let layers = (0..config.conv_depths[band].unsigned_abs())
                .map(|layer| {
                    GatedConvolution::load(
                        weights,
                        &format!("encoder.{index}.conv_modules.{band}.layers.{layer}"),
                        channels,
                        inner,
                        config.conv_kernel,
                    )
                })
                .collect::<Result<_>>()?;
            bands.push(BandEncoder {
                down,
                layers,
                stride,
                kernel,
            });
        }
        let global = weights.affine(
            &format!("encoder.{index}.globalconv"),
            &[channels, channels, 3, 3],
            channels,
        )?;
        Ok(Self { bands, global })
    }

    fn forward(&self, input: &Tensor, config: &Config) -> Result<(Tensor, Skip)> {
        let (_, _, frequencies, _) = input.size4()?;
        let original_widths = config.split_bands(frequencies)?;
        let mut bands = Vec::with_capacity(3);
        let mut widths = [0; 3];
        let mut start = 0;
        for (i, band) in self.bands.iter().enumerate() {
            let width = original_widths[i];
            let total_pad = band_padding(width, band.stride, band.kernel);
            let extracted = input.f_narrow(2, start, width)?.f_constant_pad_nd([
                0,
                0,
                total_pad / 2,
                total_pad - total_pad / 2,
            ])?;
            let encoded = band.down.conv2d(&extracted, [band.stride, 1], [0, 0])?;
            let (batch, channels, reduced, time) = encoded.size4()?;
            let mut value =
                encoded
                    .f_permute([0, 2, 1, 3])?
                    .f_reshape([batch * reduced, channels, time])?;
            for layer in &band.layers {
                value = layer.forward(&value, config.conv_kernel / 2)?;
            }
            let value = value
                .f_reshape([batch, reduced, channels, time])?
                .f_permute([0, 2, 1, 3])?
                .f_gelu("none")?;
            widths[i] = reduced;
            bands.push(value);
            start += width;
        }
        let values = Tensor::f_cat(&bands, 2)?;
        let combined = self.global.conv2d(&values, [1, 1], [1, 1])?;
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

struct Decoder {
    fusion: Affine,
    up: Vec<Affine>,
}

impl Decoder {
    fn load(
        weights: &Weights<'_>,
        config: &Config,
        index: usize,
        channels: i64,
        output: i64,
    ) -> Result<Self> {
        let fusion = weights.affine(
            &format!("decoder.{index}.0.conv"),
            &[2 * channels, 2 * channels, 3, 3],
            2 * channels,
        )?;
        let up = (0..3)
            .map(|band| {
                weights.affine(
                    &format!("decoder.{index}.1.convtrs.{band}"),
                    &[channels, output, config.band_kernel[band], 1],
                    output,
                )
            })
            .collect::<Result<_>>()?;
        Ok(Self { fusion, up })
    }

    fn forward(&self, input: &Tensor, skip: Skip, config: &Config) -> Result<Tensor> {
        let merged = input.f_add(&skip.values)?.f_repeat([1, 2, 1, 1])?;
        let fused = self.fusion.conv2d(&merged, [1, 1], [1, 1])?.f_glu(1)?;
        let mut start = 0;
        let mut bands = Vec::with_capacity(3);
        for (index, up) in self.up.iter().enumerate() {
            let width = skip.widths[index];
            let upscaled =
                up.upsample(&fused.f_narrow(2, start, width)?, config.band_stride[index])?;
            let original = skip.original_widths[index];
            let trim = (original - upscaled.size()[2]).abs() / 2;
            bands.push(upscaled.f_slice(2, trim, trim + original, 1)?);
            start += width;
        }
        Ok(Tensor::f_cat(&bands, 2)?)
    }
}

struct RecurrentPath {
    norm: Affine,
    params: Vec<Tensor>,
    project: Affine,
    hidden: i64,
}

impl RecurrentPath {
    fn load(
        weights: &Weights<'_>,
        prefix: &str,
        axis: usize,
        channels: i64,
        hidden: i64,
    ) -> Result<Self> {
        let mut params = Vec::with_capacity(8);
        let key = format!("{prefix}.lstm_layers.{axis}");
        for direction in ["", "_reverse"] {
            let get = |field, shape: &[i64]| {
                weights.tensor(&format!("{key}.{field}_l0{direction}"), shape)
            };
            params.push(get("weight_ih", &[4 * hidden, channels])?);
            params.push(get("weight_hh", &[4 * hidden, hidden])?);
            params.push(get("bias_ih", &[4 * hidden])?);
            params.push(get("bias_hh", &[4 * hidden])?);
        }
        Ok(Self {
            norm: weights.affine(
                &format!("{prefix}.norm_layers.{axis}"),
                &[channels],
                channels,
            )?,
            params,
            project: weights.affine(
                &format!("{prefix}.linear_layers.{axis}"),
                &[channels, hidden * 2],
                channels,
            )?,
            hidden,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let state = Tensor::f_zeros(
            [2, input.size()[0], self.hidden],
            (input.kind(), input.device()),
        )?;
        let params = self.params.iter().collect::<Vec<_>>();
        let (value, _, _) =
            input.f_lstm(&[&state, &state], &params, true, 1, 0.0, false, true, true)?;
        self.project.linear(&value)
    }
}

struct DualPath {
    frequency: RecurrentPath,
    time: RecurrentPath,
}

impl DualPath {
    fn load(weights: &Weights<'_>, index: usize, channels: i64, hidden: i64) -> Result<Self> {
        let prefix = format!("separation_net.dp_modules.{index}");
        Ok(Self {
            frequency: RecurrentPath::load(weights, &prefix, 0, channels, hidden)?,
            time: RecurrentPath::load(weights, &prefix, 1, channels, hidden)?,
        })
    }

    fn forward(&self, input: Tensor) -> Result<Tensor> {
        let (batch, channels, frequency, time) = input.size4()?;
        let across_bins = self
            .frequency
            .norm
            .group_norm(&input)?
            .f_transpose(1, 3)?
            .f_reshape([batch * time, frequency, channels])?;
        let residual = self
            .frequency
            .forward(&across_bins)?
            .f_reshape([batch, time, frequency, channels])?
            .f_transpose(1, 3)?
            .f_add(&input)?;
        let across_frames = self
            .time
            .norm
            .group_norm(&residual)?
            .f_transpose(1, 2)?
            .f_reshape([batch * frequency, channels, time])?
            .f_transpose(1, 2)?;
        Ok(self
            .time
            .forward(&across_frames)?
            .f_transpose(1, 2)?
            .f_reshape([batch, frequency, channels, time])?
            .f_transpose(1, 2)?
            .f_add(&residual)?)
    }
}

pub struct Scnet {
    config: Config,
    backend: Backend,
}

enum Backend {
    Torch(TorchNetwork),
    Mlx(mlx_network::Network),
    #[cfg(feature = "onnx")]
    Onnx(onnx_network::Network),
}

struct TorchNetwork {
    device: Device,
    window: Tensor,
    encoders: Vec<Encoder>,
    decoders: Vec<Decoder>,
    dual_path: Vec<DualPath>,
}

impl Scnet {
    pub fn load(
        document: &serde_json::Value,
        tensors: &HashMap<String, Tensor>,
        device: Device,
    ) -> Result<Self> {
        let config =
            Config::deserialize(&document["model"]).context("invalid SCNet model configuration")?;
        config.validate()?;
        let acceleration = Acceleration::configured(document, device, true, "SCNet")?;
        let weight_device = acceleration.weight_device(device);
        let weights = Weights {
            tensors,
            device: weight_device,
        };
        let mut encoders = Vec::with_capacity(config.dims.len() - 1);
        let mut decoders = Vec::with_capacity(config.dims.len() - 1);
        for (index, pair) in config.dims.windows(2).enumerate() {
            encoders.push(Encoder::load(&weights, &config, index, pair[0], pair[1])?);
        }
        for (index, pair) in config.dims.windows(2).rev().enumerate() {
            let output = if index + 2 == config.dims.len() {
                pair[0] * config.sources.len() as i64
            } else {
                pair[0]
            };
            decoders.push(Decoder::load(&weights, &config, index, pair[1], output)?);
        }
        let channels = *config
            .dims
            .last()
            .context("SCNet has no encoder channels")?;
        let dual_path = (0..config.num_dplayer)
            .map(|i| {
                let dim = channels * (1 + i as i64 % 2);
                DualPath::load(&weights, i, dim, dim * config.expand)
            })
            .collect::<Result<_>>()?;
        let spectral_device = if weight_device == Device::Mps {
            Device::Cpu
        } else {
            weight_device
        };
        // SCNet does not pass a window to torch.stft or torch.istft.
        let window = Tensor::f_ones([config.win_size], (Kind::Float, spectral_device))?;
        let torch = TorchNetwork {
            device: weight_device,
            window,
            encoders,
            decoders,
            dual_path,
        };
        let backend = match acceleration {
            Acceleration::Torch => Backend::Torch(torch),
            Acceleration::Mlx(dtype) => {
                Backend::Mlx(mlx_network::Network::load(&config, &torch, device, dtype)?)
            }
            #[cfg(feature = "onnx")]
            Acceleration::Onnx(provider) => {
                Backend::Onnx(onnx_network::Network::load(&config, torch, provider)?)
            }
        };
        Ok(Self { config, backend })
    }

    pub fn channels(&self) -> usize {
        self.config.audio_channels as usize
    }

    pub fn num_stems(&self) -> usize {
        self.config.sources.len()
    }

    pub(super) fn predict_batch(&self, audio: &[f32], count: usize) -> Result<Vec<f32>> {
        let channels = self.channels();
        let samples = super::super::window_samples(audio, count, channels)?;
        let padding = self.config.sample_padding(samples)?;
        let wave = match &self.backend {
            Backend::Torch(model) => no_grad(|| {
                spectral_pass(
                    &self.config,
                    &model.window,
                    audio,
                    count as i64,
                    samples,
                    padding,
                    |features| model.predict_features(&self.config, features),
                )
            })?,
            #[cfg(feature = "onnx")]
            Backend::Onnx(model) => no_grad(|| {
                spectral_pass(
                    &self.config,
                    &model.window,
                    audio,
                    count as i64,
                    samples,
                    padding,
                    |features| model.predict_features(features),
                )
            })?,
            Backend::Mlx(model) => model.run(&self.config, audio, count, samples, padding)?,
        };
        ensure!(
            wave.iter().all(|sample| sample.is_finite()),
            "SCNet produced nonfinite audio"
        );
        Ok(wave)
    }
}

impl TorchNetwork {
    fn predict_features(&self, config: &Config, input: Tensor) -> Result<Tensor> {
        let spectral_device = self.window.device();
        let mut value = input.f_to_device(self.device)?;
        let mut skips = Vec::with_capacity(self.encoders.len());
        for encoder in &self.encoders {
            let (compressed, skip) = encoder.forward(&value, config)?;
            value = compressed;
            skips.push(skip);
        }
        for (index, path) in self.dual_path.iter().enumerate() {
            let recurrent = path.forward(value)?.f_to_device(spectral_device)?;
            value = separator_transform(&recurrent, index % 2 == 0)?.f_to_device(self.device)?;
        }
        for (decoder, skip) in self.decoders.iter().zip(skips.into_iter().rev()) {
            value = decoder.forward(&value, skip, config)?;
        }
        Ok(value)
    }
}

fn separator_transform(recurrent: &Tensor, forward: bool) -> Result<Tensor> {
    if forward {
        let spectrum = recurrent.f_fft_rfft(None, 3, "ortho")?;
        Ok(Tensor::f_cat(&[spectrum.f_real()?, spectrum.f_imag()?], 1)?)
    } else {
        let base = recurrent.size()[1] / 2;
        let re = recurrent.f_narrow(1, 0, base)?;
        let im = recurrent.f_narrow(1, base, base)?;
        Ok(Tensor::f_complex(&re, &im)?.f_fft_irfft(None, 3, "ortho")?)
    }
}

fn spectral_pass(
    config: &Config,
    window: &Tensor,
    audio: &[f32],
    batch: i64,
    samples: i64,
    padding: i64,
    predict: impl FnOnce(Tensor) -> Result<Tensor>,
) -> Result<Vec<f32>> {
    let Config {
        nfft,
        hop_size: hop,
        win_size: win,
        normalized,
        audio_channels: channels,
        ..
    } = *config;
    let device = window.device();
    let padded = Tensor::f_from_slice(audio)?
        .f_reshape([-1, samples])?
        .f_to_device(device)?
        .f_constant_pad_nd([0, padding])?
        .f_reflection_pad1d([nfft / 2, nfft / 2])?;
    let stft = padded.f_stft_center(
        nfft,
        hop,
        win,
        Some(window),
        false,
        "reflect",
        normalized,
        true,
        true,
        false,
    )?;
    let (_, frequency, time) = stft.size3()?;
    let features = stft.f_view_as_real()?.f_permute([0, 3, 1, 2])?.f_reshape([
        batch,
        channels * 2,
        frequency,
        time,
    ])?;
    let resolved = predict(features)?
        .f_to_device(device)?
        .f_reshape([
            batch * config.sources.len() as i64 * channels,
            2,
            frequency,
            time,
        ])?
        .f_permute([0, 2, 3, 1])?
        .f_contiguous()?
        .f_view_as_complex()?;
    let wave = resolved.f_istft(
        nfft,
        hop,
        win,
        Some(window),
        true,
        normalized,
        true,
        None,
        false,
    )?;
    let wave = wave
        .f_narrow(1, 0, samples)?
        .f_to_device(Device::Cpu)?
        .f_reshape([-1])?;
    Ok(Vec::<f32>::try_from(wave)?)
}
