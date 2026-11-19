//! Apollo spectral restoration from the bundled PyMSS model.
//! See `licenses/pymss-core.txt`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use serde_json::Value;
use tch::{Device, Kind, Tensor, no_grad};

use super::super::Acceleration;

#[path = "apollo/mlx_network.rs"]
mod mlx_network;
#[cfg(feature = "onnx")]
#[path = "apollo/onnx_network.rs"]
mod onnx_network;

const BANDS: i64 = 80;
const HEADS: i64 = 8;
const ROTARY_WINDOW: i64 = 100;
const RMS_EPSILON: f64 = 1e-5;

pub(super) fn checkpoint_path(path: &Path) -> PathBuf {
    if path.extension().is_some() {
        let sidecar = path.with_extension("pymss_state_dict.pt");
        if sidecar.is_file() {
            return sidecar;
        }
    }
    let mut sidecar = path.as_os_str().to_os_string();
    sidecar.push(".pymss_state_dict.pt");
    let sidecar = PathBuf::from(sidecar);
    if sidecar.is_file() {
        return sidecar;
    }
    path.to_path_buf()
}

#[derive(Deserialize)]
struct Config {
    sr: u32,
    win: u32,
    layer: usize,
}

struct Weights<'a> {
    values: &'a HashMap<String, Tensor>,
    device: Device,
}

impl Weights<'_> {
    fn tensor(&self, name: &str, shape: &[i64]) -> Result<Tensor> {
        let value = self
            .values
            .get(name)
            .with_context(|| format!("Apollo checkpoint is missing {name}"))?;
        ensure!(
            value.size() == shape && value.is_floating_point(),
            "invalid Apollo tensor {name}: {:?} {:?}, expected floating point {shape:?}",
            value.kind(),
            value.size(),
        );
        Ok(value.f_to_kind(Kind::Float)?.f_to_device(self.device)?)
    }

    fn pointwise(&self, name: &str, input: i64, output: i64, bias: bool) -> Result<Pointwise> {
        Ok(Pointwise {
            weight: self
                .tensor(&format!("{name}.weight"), &[output, input, 1])?
                .f_squeeze_dim(2)?,
            bias: bias
                .then(|| self.tensor(&format!("{name}.bias"), &[output]))
                .transpose()?,
        })
    }
}

struct Pointwise {
    weight: Tensor,
    bias: Option<Tensor>,
}

impl Pointwise {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        Ok(input
            .f_transpose(1, 2)?
            .f_linear(&self.weight, self.bias.as_ref())?
            .f_transpose(1, 2)?)
    }
}

fn rms(input: &Tensor, weight: &Tensor) -> Result<Tensor> {
    Ok(input
        .f_transpose(1, 2)?
        .f_rms_norm(weight.size(), None::<&Tensor>, RMS_EPSILON)?
        .f_transpose(1, 2)?
        .f_mul(&weight.f_reshape([1, -1, 1])?)?)
}

struct Projection {
    norm: Tensor,
    linear: Pointwise,
}

impl Projection {
    fn load(weights: &Weights, name: &str, input: i64, output: i64, bias: bool) -> Result<Self> {
        Ok(Self {
            norm: weights.tensor(&format!("{name}.0.weight"), &[input])?,
            linear: weights.pointwise(&format!("{name}.1"), input, output, bias)?,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.linear.forward(&rms(input, &self.norm)?)
    }
}

struct PackedProjection {
    norm: Tensor,
    weight: Tensor,
    bias: Tensor,
}

impl PackedProjection {
    fn load(weights: &Weights, name: &str, input: i64, output: i64) -> Result<Self> {
        let prefix = (0..BANDS - 1)
            .map(|band| Projection::load(weights, &format!("{name}.{band}"), input, output, true))
            .collect::<Result<Vec<_>>>()?;
        let norms: Vec<_> = prefix.iter().map(|value| &value.norm).collect();
        let matrices: Vec<_> = prefix.iter().map(|value| &value.linear.weight).collect();
        let biases: Vec<_> = prefix
            .iter()
            .map(|value| value.linear.bias.as_ref().context("missing Apollo bias"))
            .collect::<Result<_>>()?;
        Ok(Self {
            norm: Tensor::f_stack(&norms, 0)?.f_reshape([1, BANDS - 1, input, 1])?,
            weight: Tensor::f_cat(&matrices, 0)?.f_unsqueeze(2)?,
            bias: Tensor::f_cat(&biases, 0)?,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let (batch, bands, channels, frames) = input.size4()?;
        let magnitude = input
            .f_pow_tensor_scalar(2)?
            .f_mean_dim([2].as_slice(), true, None)?
            .f_add_scalar(RMS_EPSILON)?
            .f_rsqrt()?;
        Ok(input
            .f_mul(&magnitude)?
            .f_mul(&self.norm)?
            .f_reshape([batch, bands * channels, frames])?
            .f_conv1d(&self.weight, Some(&self.bias), 1, 0, 1, bands)?
            .f_reshape([batch, bands, -1, frames])?)
    }
}

struct Attention {
    norm: Tensor,
    qkv: Pointwise,
    projection: Pointwise,
    cosine: Tensor,
    sine: Tensor,
    reverse: Tensor,
    mlp: Projection,
    mlp_output: Pointwise,
    head_dim: i64,
}

impl Attention {
    fn load(weights: &Weights, name: &str, features: i64) -> Result<Self> {
        let head_dim = features / HEADS;
        let hidden = HEADS * head_dim;
        let rotary = |key| -> Result<Tensor> {
            Ok(weights
                .tensor(&format!("{name}.{key}"), &[ROTARY_WINDOW, head_dim])?
                .f_narrow(0, 0, BANDS)?)
        };
        Ok(Self {
            norm: weights.tensor(&format!("{name}.input_norm.weight"), &[features])?,
            qkv: weights.pointwise(&format!("{name}.weight"), features, 3 * hidden, false)?,
            projection: weights.pointwise(&format!("{name}.output"), hidden, features, false)?,
            cosine: rotary("cos_freq")?,
            sine: rotary("sin_freq")?,
            reverse: Tensor::f_from_slice(&[-1_f32, 1.0])?.f_to_device(weights.device)?,
            mlp: Projection::load(
                weights,
                &format!("{name}.MLP"),
                features,
                8 * features,
                false,
            )?,
            mlp_output: weights.pointwise(
                &format!("{name}.MLP_output"),
                4 * features,
                features,
                false,
            )?,
            head_dim,
        })
    }

    fn rotate(&self, input: &Tensor) -> Result<Tensor> {
        let shape = input.size();
        let swapped = input
            .f_reshape([-1, self.head_dim / 2, 2])?
            .f_flip([-1])?
            .f_mul(&self.reverse)?
            .f_reshape(shape.as_slice())?;
        Ok(input
            .f_mul(&self.cosine)?
            .f_add(&swapped.f_mul(&self.sine)?)?
            .f_contiguous()?)
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let (batch, _, frames) = input.size3()?;
        let weights = self
            .qkv
            .forward(&rms(input, &self.norm)?)?
            .f_reshape([batch, HEADS, self.head_dim * 3, frames])?
            .f_transpose(-2, -1)?;
        let parts = weights.f_split(self.head_dim, -1)?;
        let output = Tensor::f_scaled_dot_product_attention(
            &self.rotate(&parts[0])?,
            &self.rotate(&parts[1])?,
            &parts[2],
            None::<&Tensor>,
            0.0,
            false,
            None,
            false,
        )?;
        let output = self
            .projection
            .forward(&output.f_transpose(-2, -1)?.f_reshape([batch, -1, frames])?)?
            .f_add(input)?;
        let gate = self.mlp.forward(&output)?.f_silu()?.f_chunk(2, 1)?;
        Ok(output.f_add(
            &self
                .mlp_output
                .forward(&gate[0].f_silu()?.f_mul(&gate[1])?)?,
        )?)
    }
}

struct SequenceBlock {
    conv: Tensor,
    bias: Tensor,
    features: i64,
    norm: Tensor,
    expand: Pointwise,
    contract: Pointwise,
}

impl SequenceBlock {
    fn load(weights: &Weights, name: &str, features: i64) -> Result<Self> {
        Ok(Self {
            conv: weights.tensor(&format!("{name}.0.weight"), &[features, 1, 7])?,
            bias: weights.tensor(&format!("{name}.0.bias"), &[features])?,
            features,
            norm: weights.tensor(&format!("{name}.1.weight"), &[features])?,
            expand: weights.pointwise(&format!("{name}.2"), features, 4 * features, true)?,
            contract: weights.pointwise(&format!("{name}.4"), 4 * features, features, true)?,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let depthwise = input.f_conv1d(&self.conv, Some(&self.bias), 1, 3, 1, self.features)?;
        let expanded = self
            .expand
            .forward(&rms(&depthwise, &self.norm)?)?
            .f_silu()?;
        Ok(input.f_add(&self.contract.forward(&expanded)?)?)
    }
}

struct BandNetwork {
    attention: Attention,
    sequence: Vec<SequenceBlock>,
}

impl BandNetwork {
    fn load(weights: &Weights, layer: usize, features: i64) -> Result<Self> {
        let name = format!("net.{layer}");
        Ok(Self {
            attention: Attention::load(weights, &format!("{name}.band_net"), features)?,
            sequence: (0..3)
                .map(|index| {
                    SequenceBlock::load(
                        weights,
                        &format!("{name}.seq_net.blocks.{index}.conv"),
                        features,
                    )
                })
                .collect::<Result<_>>()?,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let (batch, bands, features, frames) = input.size4()?;
        let along_bands =
            input
                .f_permute([0, 3, 2, 1])?
                .f_reshape([batch * frames, features, bands])?;
        let along_time = self
            .attention
            .forward(&along_bands)?
            .f_reshape([batch, frames, features, bands])?
            .f_permute([0, 3, 2, 1])?
            .f_reshape([batch * bands, features, frames])?;
        let output = self
            .sequence
            .iter()
            .try_fold(along_time, |value, block| block.forward(&value))?;
        Ok(output.f_reshape([batch, bands, features, frames])?)
    }
}

pub(super) struct Apollo {
    fft: i64,
    backend: Backend,
}

enum Backend {
    Torch(Box<TorchNetwork>),
    Mlx(Box<mlx_network::Network>),
    #[cfg(feature = "onnx")]
    Onnx(Box<onnx_network::Network>),
}

struct Spectral {
    window: Tensor,
    fft: i64,
    width: i64,
    tail_width: i64,
}

struct TorchNetwork {
    device: Device,
    spectral: Spectral,
    prefix_input: PackedProjection,
    tail_input: Projection,
    network: Vec<BandNetwork>,
    prefix_output: PackedProjection,
    tail_output: Projection,
}

impl Apollo {
    pub(super) fn load(
        document: &Value,
        weights: &HashMap<String, Tensor>,
        device: Device,
    ) -> Result<Self> {
        let config =
            Config::deserialize(&document["model"]).context("invalid Apollo config.model")?;
        let fft = i64::try_from(u64::from(config.sr) * u64::from(config.win) / 1000)?;
        // The catalog's v2 checkpoint stores 192 features while its YAML still says 256.
        let (features, _, _) = weights
            .get("BN.0.1.weight")
            .context("Apollo checkpoint is missing BN.0.1.weight")?
            .size3()?;
        ensure!(
            (160..=65536).contains(&fft)
                && (1..=8192).contains(&features)
                && (config.layer == 0 || features % (HEADS * 2) == 0)
                && config.layer <= 128,
            "invalid Apollo model dimensions"
        );
        let width = fft / (2 * BANDS);
        let tail_width = fft / 2 + 1 - width * (BANDS - 1);
        let acceleration = Acceleration::configured(document, device, true, "Apollo")?;
        let weight_device = acceleration.weight_device(device);
        let weights = Weights {
            values: weights,
            device: weight_device,
        };
        let spectral_device = if weight_device == Device::Mps {
            Device::Cpu
        } else {
            weight_device
        };
        let torch = TorchNetwork {
            device: weight_device,
            spectral: Spectral {
                fft,
                window: Tensor::f_hann_window(fft, (Kind::Float, spectral_device))?,
                width,
                tail_width,
            },
            prefix_input: PackedProjection::load(&weights, "BN", width * 2 + 1, features)?,
            tail_input: Projection::load(&weights, "BN.79", tail_width * 2 + 1, features, true)?,
            network: (0..config.layer)
                .map(|index| BandNetwork::load(&weights, index, features))
                .collect::<Result<_>>()?,
            prefix_output: PackedProjection::load(&weights, "output", features, width * 4)?,
            tail_output: Projection::load(&weights, "output.79", features, tail_width * 4, true)?,
        };
        let backend = match acceleration {
            Acceleration::Torch => Backend::Torch(Box::new(torch)),
            Acceleration::Mlx(dtype) => {
                Backend::Mlx(Box::new(mlx_network::Network::load(&torch, device, dtype)?))
            }
            #[cfg(feature = "onnx")]
            Acceleration::Onnx(provider) => {
                Backend::Onnx(Box::new(onnx_network::Network::load(torch, provider)?))
            }
        };
        Ok(Self { fft, backend })
    }

    pub(super) fn predict_batch(&self, audio: &[f32], count: usize) -> Result<Vec<f32>> {
        let channels = 2;
        let samples = super::super::window_samples(audio, count, channels)?;
        ensure!(
            samples > self.fft / 2,
            "Apollo input is too short for centered STFT"
        );
        let wave = match &self.backend {
            Backend::Torch(model) => no_grad(|| {
                model
                    .spectral
                    .predict(audio, samples, |prefix, tail| model.estimate(prefix, tail))
            })?,
            #[cfg(feature = "onnx")]
            Backend::Onnx(model) => no_grad(|| {
                model
                    .spectral
                    .predict(audio, samples, |prefix, tail| model.estimate(prefix, tail))
            })?,
            Backend::Mlx(model) => model.run(audio, samples as usize)?,
        };
        ensure!(
            wave.iter().all(|value| value.is_finite()),
            "Apollo produced nonfinite audio"
        );
        Ok(wave)
    }
}

impl Spectral {
    fn normalize_band(spec: &Tensor, axis: i64) -> Result<Tensor> {
        let power = spec
            .f_abs()?
            .f_pow_tensor_scalar(2)?
            .f_sum_dim_intlist([axis].as_slice(), true, None)?
            .f_add_scalar(f32::EPSILON as f64)?
            .f_sqrt()?;
        Ok(Tensor::f_cat(
            &[
                spec.f_real()?.f_div(&power)?,
                spec.f_imag()?.f_div(&power)?,
                power.f_log()?,
            ],
            axis,
        )?)
    }

    fn predict(
        &self,
        audio: &[f32],
        samples: i64,
        estimate: impl FnOnce(Tensor, Tensor) -> Result<Tensor>,
    ) -> Result<Vec<f32>> {
        let input = Tensor::f_from_slice(audio)?.f_reshape([-1, samples])?;
        let channels = input.size()[0];
        let spectral_device = self.window.device();
        let padded = input
            .f_to_device(spectral_device)?
            .f_reflection_pad1d([self.fft / 2, self.fft / 2])?;
        let spectrum = padded.f_stft_center(
            self.fft,
            self.fft / 2,
            self.fft,
            Some(&self.window),
            false,
            "reflect",
            false,
            true,
            true,
            false,
        )?;
        let frames = spectrum.size()[2];
        let prefix_bins = (BANDS - 1) * self.width;
        let prefix = spectrum.f_narrow(1, 0, prefix_bins)?.f_reshape([
            channels,
            BANDS - 1,
            self.width,
            frames,
        ])?;
        let tail = spectrum.f_narrow(1, prefix_bins, self.tail_width)?;
        let output = estimate(
            Self::normalize_band(&prefix, 2)?,
            Self::normalize_band(&tail, 1)?,
        )?
        .f_to_device(spectral_device)?;
        let spectrum = Tensor::f_complex(&output.f_select(1, 0)?, &output.f_select(1, 1)?)?;
        let wave = spectrum.f_istft(
            self.fft,
            self.fft / 2,
            self.fft,
            Some(&self.window),
            true,
            false,
            true,
            samples,
            false,
        )?;
        Ok(Vec::<f32>::try_from(
            wave.f_to_device(Device::Cpu)?.f_reshape([-1])?,
        )?)
    }
}

impl TorchNetwork {
    fn estimate(&self, prefix: Tensor, tail: Tensor) -> Result<Tensor> {
        let (channels, _, _, frames) = prefix.size4()?;
        let Spectral {
            width, tail_width, ..
        } = self.spectral;
        let prefix_bins = (BANDS - 1) * width;
        let prefix = self
            .prefix_input
            .forward(&prefix.f_to_device(self.device)?)?;
        let tail = self.tail_input.forward(&tail.f_to_device(self.device)?)?;
        let mut features = Tensor::f_cat(&[prefix, tail.f_unsqueeze(1)?], 1)?;
        for layer in &self.network {
            features = layer.forward(&features)?;
        }
        let prefix = self
            .prefix_output
            .forward(&features.f_narrow(1, 0, BANDS - 1)?)?
            .f_glu(2)?
            .f_reshape([channels, BANDS - 1, 2, width, frames])?
            .f_transpose(1, 2)?
            .f_reshape([channels, 2, prefix_bins, frames])?;
        let tail = self
            .tail_output
            .forward(&features.f_select(1, BANDS - 1)?)?
            .f_glu(1)?
            .f_reshape([channels, 2, tail_width, frames])?;
        Ok(Tensor::f_cat(&[prefix, tail], 2)?)
    }
}
