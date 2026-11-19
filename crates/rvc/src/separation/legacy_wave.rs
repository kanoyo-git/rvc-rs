//! Waveform inference for the bundled legacy PyMSS Demucs and ConvTasNet checkpoints.
//! See licenses/pymss-core.txt.

use std::collections::HashMap;

use anyhow::{Context, Result, bail, ensure};
use serde::Deserialize;
use serde_json::Value;
use tch::{Device, Kind, Tensor, no_grad};

use super::Acceleration;
use crate::model::weights::Weights;

#[cfg(feature = "onnx")]
#[path = "legacy_wave/onnx_network.rs"]
pub(super) mod onnx_network;

const FOUR_STEMS: &[&str] = &["drums", "bass", "other", "vocals"];
const TWO_STEMS: &[&str] = &["vocals", "non_vocals"];
const NORM_EPSILON: f64 = 1e-5;

pub(super) struct LegacyWave {
    network: Network,
    weights: Weights,
    rate: u32,
    channels: usize,
    stems: Vec<String>,
    segment: usize,
    device: Device,
    #[cfg(feature = "onnx")]
    onnx: Option<onnx_network::Network>,
}

enum Network {
    Demucs(Demucs),
    Tasnet(Tasnet),
}

#[derive(Clone, Deserialize, serde::Serialize)]
#[serde(untagged)]
enum Sources {
    Names(Vec<String>),
    Count(usize),
}

impl Sources {
    fn into_names(self) -> Result<Vec<String>> {
        let names = match self {
            Self::Names(names) => names,
            Self::Count(2) => TWO_STEMS.iter().map(|name| (*name).to_owned()).collect(),
            Self::Count(4) => FOUR_STEMS.iter().map(|name| (*name).to_owned()).collect(),
            Self::Count(count) if (1..=32).contains(&count) => {
                (0..count).map(|index| format!("source_{index}")).collect()
            }
            Self::Count(count) => bail!("unsupported legacy waveform source count {count}"),
        };
        ensure!(
            (1..=32).contains(&names.len()) && names.iter().all(|name| !name.trim().is_empty()),
            "expected 1 to 32 nonempty legacy waveform stem names"
        );
        Ok(names)
    }
}

impl LegacyWave {
    pub(super) fn load(
        kind: &str,
        config: &Value,
        state: &HashMap<String, Tensor>,
        device: Device,
        acceleration: Acceleration,
    ) -> Result<Self> {
        let (network, rate, channels, stems, segment) = match kind {
            "demucs" | "demucs_v3" => {
                let is_v3 = kind == "demucs_v3";
                let mut args = serde_json::to_value(DemucsArgs::defaults(is_v3))?;
                overlay_config(&mut args, config)?;
                let args: DemucsArgs = serde_json::from_value(args)?;
                let stems = args
                    .sources
                    .clone()
                    .context("waveform Demucs v3 requires source names or a source count")?
                    .into_names()?;
                let segment = if is_v3 {
                    let frames = args.segment * f64::from(args.samplerate);
                    ensure!(
                        frames.is_finite() && (1.0..=i32::MAX as f64).contains(&frames),
                        "invalid waveform Demucs v3 segment"
                    );
                    frames as usize
                } else {
                    args.segment_length
                };
                let (rate, channels) = (args.samplerate, args.audio_channels);
                let network = Demucs::load(args, is_v3, stems.len(), state)?;
                (Network::Demucs(network), rate, channels, stems, segment)
            }
            "conv_tasnet" => {
                let args: TasnetArgs = serde_json::from_value(config.clone())?;
                let stems = args
                    .sources
                    .clone()
                    .unwrap_or(Sources::Count(args.source_count))
                    .into_names()?;
                let (rate, channels, segment) =
                    (args.samplerate, args.audio_channels, args.segment_length);
                let network = Tasnet::load(args, stems.len(), state)?;
                (Network::Tasnet(network), rate, channels, stems, segment)
            }
            _ => bail!("unsupported legacy waveform architecture {kind:?}"),
        };
        ensure!(
            (1..=2).contains(&channels) && rate > 0 && (1..=i32::MAX as usize).contains(&segment),
            "invalid legacy waveform audio geometry"
        );
        let device = acceleration.weight_device(device);
        let values = state
            .iter()
            .map(|(name, tensor)| {
                let tensor = tensor.f_to_device(device)?;
                let tensor = if tensor.is_floating_point() {
                    tensor.f_to_kind(Kind::Float)?
                } else {
                    tensor
                };
                Ok((name.clone(), tensor))
            })
            .collect::<Result<HashMap<_, _>>>()?;
        Ok(Self {
            network,
            weights: Weights::new(values),
            rate,
            channels,
            stems,
            segment,
            device,
            #[cfg(feature = "onnx")]
            onnx: match acceleration {
                Acceleration::Onnx(provider) => Some(onnx_network::Network::new(provider)),
                _ => None,
            },
        })
    }

    pub(super) fn rate(&self) -> u32 {
        self.rate
    }

    pub(super) fn channels(&self) -> usize {
        self.channels
    }

    pub(super) fn stems(&self) -> &[String] {
        &self.stems
    }

    pub(super) fn segment_length(&self) -> usize {
        self.segment
    }

    pub(super) fn valid_length(&self, frames: usize) -> Result<usize> {
        ensure!(
            (1..=i32::MAX as usize).contains(&frames),
            "invalid legacy waveform input length"
        );
        match &self.network {
            Network::Demucs(model) => usize::try_from(model.valid_length(frames as i64)?)
                .context("waveform Demucs input length exceeds address space"),
            Network::Tasnet(_) => Ok(frames),
        }
    }

    /// The input is [channel][frame]; the output is [stem][channel][frame].
    pub(super) fn predict_window(&self, samples: &[f32]) -> Result<Vec<f32>> {
        let frames = self.input_frames(samples)?;
        self.predict(samples, frames, self.valid_length(frames)?)
    }

    /// Surrounding track samples can supply the alignment padding.
    pub(super) fn predict_aligned(&self, samples: &[f32], frames: usize) -> Result<Vec<f32>> {
        let padded = self.input_frames(samples)?;
        ensure!(
            padded == self.valid_length(frames)?,
            "legacy waveform alignment length {padded} is invalid for {frames} frames"
        );
        self.predict(samples, frames, padded)
    }

    fn input_frames(&self, samples: &[f32]) -> Result<usize> {
        ensure!(
            !samples.is_empty() && samples.len().is_multiple_of(self.channels),
            "invalid legacy waveform input channels"
        );
        Ok(samples.len() / self.channels)
    }

    fn predict(&self, samples: &[f32], frames: usize, aligned: usize) -> Result<Vec<f32>> {
        no_grad(|| {
            let input = Tensor::f_from_slice(samples)?
                .f_view([1, self.channels as i64, -1])?
                .f_to_device(self.device)?;
            let input = center_pad(&input, aligned as i64)?;
            let output = self.estimate(&input)?;
            let output = center_trim(&output, frames as i64)?.f_to_device(Device::Cpu)?;
            ensure!(
                output.numel() == frames * self.channels * self.stems.len(),
                "legacy waveform prediction shape {:?} does not match channels and stems",
                output.size()
            );
            Ok(Vec::<f32>::try_from(&output.f_reshape([-1])?)?)
        })
    }

    fn estimate(&self, input: &Tensor) -> Result<Tensor> {
        match &self.network {
            Network::Demucs(model) => model.forward(input, |value| {
                #[cfg(feature = "onnx")]
                if let Some(onnx) = &self.onnx {
                    return onnx.demucs(model, &self.weights, &value);
                }
                model.estimate(&self.weights, value)
            }),
            Network::Tasnet(model) => {
                let stems = self.stems.len() as i64;
                #[cfg(feature = "onnx")]
                if let Some(onnx) = &self.onnx {
                    return onnx.tasnet(model, &self.weights, input, stems);
                }
                model.forward(&self.weights, input, stems)
            }
        }
    }
}

fn overlay_config(defaults: &mut Value, config: &Value) -> Result<()> {
    let table = config
        .as_object()
        .context("expected legacy waveform constructor arguments")?;
    let defaults = defaults
        .as_object_mut()
        .expect("serialized waveform arguments are an object");
    defaults.extend(
        table
            .iter()
            .map(|(name, value)| (name.clone(), value.clone())),
    );
    Ok(())
}

fn last_len(x: &Tensor) -> i64 {
    *x.size().last().unwrap_or(&0)
}

fn center_pad(x: &Tensor, length: i64) -> Result<Tensor> {
    let pad = length - last_len(x);
    ensure!(pad >= 0, "legacy waveform pad target shorter than input");
    if pad == 0 {
        return Ok(x.shallow_clone());
    }
    Ok(x.f_constant_pad_nd([pad / 2, pad - pad / 2])?)
}

fn center_trim(x: &Tensor, length: i64) -> Result<Tensor> {
    let excess = last_len(x) - length;
    ensure!(
        excess >= 0,
        "legacy waveform prediction shorter than {length} frames"
    );
    if excess == 0 {
        return Ok(x.shallow_clone());
    }
    Ok(x.f_narrow(-1, excess / 2, length)?)
}

#[derive(Deserialize, serde::Serialize)]
struct DemucsArgs {
    sources: Option<Sources>,
    audio_channels: usize,
    channels: i64,
    depth: usize,
    rewrite: bool,
    glu: bool,
    resample: bool,
    upsample: Option<bool>,
    kernel_size: i64,
    stride: i64,
    growth: f64,
    lstm_layers: i64,
    context: i64,
    normalize: bool,
    gelu: bool,
    norm_starts: usize,
    norm_groups: i64,
    dconv_mode: u8,
    dconv_depth: i32,
    dconv_comp: f64,
    dconv_attn: usize,
    dconv_lstm: usize,
    samplerate: u32,
    segment_length: usize,
    segment: f64,
}

struct Demucs {
    args: DemucsArgs,
    is_v3: bool,
    widths: Vec<i64>,
}

impl DemucsArgs {
    fn defaults(is_v3: bool) -> Self {
        Self {
            sources: if is_v3 { None } else { Some(Sources::Count(4)) },
            audio_channels: 2,
            channels: 64,
            depth: 6,
            rewrite: true,
            glu: true,
            resample: true,
            upsample: None,
            kernel_size: 8,
            stride: 4,
            growth: 2.0,
            lstm_layers: if is_v3 { 0 } else { 2 },
            context: if is_v3 { 1 } else { 3 },
            normalize: is_v3,
            gelu: is_v3,
            norm_starts: 4,
            norm_groups: 4,
            dconv_mode: if is_v3 { 1 } else { 0 },
            dconv_depth: 2,
            dconv_comp: 4.0,
            dconv_attn: 4,
            dconv_lstm: 4,
            samplerate: 44_100,
            segment_length: 4 * 10 * 44_100,
            segment: 40.0,
        }
    }

    fn activation(&self, value: Tensor) -> Result<Tensor> {
        if self.gelu {
            Ok(value.f_gelu("none")?)
        } else {
            Ok(value.f_relu()?)
        }
    }

    fn gate(&self, value: Tensor) -> Result<Tensor> {
        if self.glu {
            Ok(value.f_glu(1)?)
        } else {
            Ok(value.f_relu()?)
        }
    }
}

impl Demucs {
    fn load(
        mut args: DemucsArgs,
        is_v3: bool,
        stems: usize,
        state: &HashMap<String, Tensor>,
    ) -> Result<Self> {
        ensure!(
            (1..=20).contains(&args.depth)
                && (1..=256).contains(&args.kernel_size)
                && (1..=64).contains(&args.stride)
                && args.growth.is_finite()
                && args.growth > 0.0
                && (0..=8).contains(&args.lstm_layers)
                && (if is_v3 { 0..=32 } else { 1..=32 }).contains(&args.context),
            "invalid waveform Demucs layer geometry"
        );
        ensure!(
            !is_v3
                || ((1..=64).contains(&args.norm_groups)
                    && args.dconv_mode <= 3
                    && args.dconv_depth.unsigned_abs() <= 16
                    && args.dconv_comp > 0.0
                    && args.dconv_comp.is_finite()),
            "unsupported Demucs v3 normalization or residual geometry"
        );
        let mut widths = Vec::with_capacity(args.depth);
        let mut input = args.audio_channels as i64;
        let mut width = args.channels;
        for stage in 0..args.depth {
            ensure!(
                (1..=16_384).contains(&width),
                "invalid waveform Demucs layer {stage} width"
            );
            let key = format!("encoder.{stage}.0.weight");
            check_tensor(state, &key, &[width, input, args.kernel_size])?;
            let index = if is_v3 {
                3 * usize::from(args.rewrite) + usize::from(args.dconv_mode & 2 != 0)
            } else {
                2 * usize::from(args.rewrite)
            };
            let reverse = args.depth - 1 - stage;
            let key = format!("decoder.{reverse}.{index}.weight");
            let out = if stage == 0 {
                input * stems as i64
            } else {
                input
            };
            check_tensor(state, &key, &[width, out, args.kernel_size])?;
            widths.push(width);
            input = width;
            width = (width as f64 * args.growth) as i64;
        }
        if !is_v3 {
            args.resample = args.upsample.unwrap_or(args.resample);
            args.gelu = false;
            args.dconv_mode = 0;
        }
        Ok(Self {
            args,
            is_v3,
            widths,
        })
    }

    fn valid_length(&self, frames: i64) -> Result<i64> {
        let args = &self.args;
        let mut length = frames * if args.resample { 2 } else { 1 };
        for _ in 0..args.depth {
            length =
                ((length - args.kernel_size + args.stride - 1).div_euclid(args.stride) + 1).max(1);
            if !self.is_v3 {
                length += args.context - 1;
            }
        }
        for _ in 0..args.depth {
            length = (length - 1)
                .checked_mul(args.stride)
                .and_then(|length| length.checked_add(args.kernel_size))
                .context("waveform Demucs aligned input length overflow")?;
        }
        if args.resample {
            length = (length + 1) / 2;
        }
        ensure!(
            (frames..=i32::MAX as i64).contains(&length),
            "waveform Demucs aligned length {length} is invalid"
        );
        Ok(length)
    }

    fn forward(
        &self,
        input: &Tensor,
        estimate: impl FnOnce(Tensor) -> Result<Tensor>,
    ) -> Result<Tensor> {
        let args = &self.args;
        let frames = last_len(input);
        let (mut x, norm) = if args.normalize {
            let mono = input.f_mean_dim([1].as_slice(), true, None)?;
            let mean = mono.f_mean_dim([-1].as_slice(), true, None)?;
            let std = mono.f_std_dim([-1].as_slice(), true, true)?;
            (
                input
                    .f_sub(&mean)?
                    .f_div(&std.f_add_scalar(NORM_EPSILON)?)?,
                Some((mean, std)),
            )
        } else if self.is_v3 {
            (input.shallow_clone(), None)
        } else {
            (input.f_div_scalar(1.0 + NORM_EPSILON)?, None)
        };
        let aligned = if self.is_v3 {
            self.valid_length(frames)?
        } else {
            frames
        };
        x = center_pad(&x, aligned)?;
        if args.resample {
            x = x.f_upsample_linear1d([aligned * 2], false, None)?;
        }
        x = estimate(x)?;
        if args.resample {
            x = x.f_upsample_linear1d([aligned], false, None)?;
        }
        if let Some((mean, std)) = norm {
            x = x.f_mul(&std)?.f_add(&mean)?;
        }
        if self.is_v3 {
            x = center_trim(&x, frames)?;
        }
        Ok(x)
    }

    fn estimate(&self, weights: &Weights, mut x: Tensor) -> Result<Tensor> {
        let args = &self.args;
        let mut skip = Vec::with_capacity(args.depth);
        for index in 0..args.depth {
            x = self.encode(weights, &x, index)?;
            skip.push(x.shallow_clone());
        }
        if args.lstm_layers > 0 {
            x = blstm(weights, &x, "lstm", args.lstm_layers, false)?;
        }
        for (index, layer) in skip.into_iter().enumerate().rev() {
            x = x.f_add(&center_trim(&layer, last_len(&x))?)?;
            x = self.decode(weights, &x, index)?;
        }
        Ok(x)
    }

    fn encode(&self, weights: &Weights, x: &Tensor, index: usize) -> Result<Tensor> {
        let args = &self.args;
        let stem = format!("encoder.{index}");
        let mut x = weights.conv(x, &format!("{stem}.0"), args.stride, 0, 1, 1)?;
        let group_norm = self.is_v3 && index >= args.norm_starts;
        if group_norm {
            x = group(weights, &x, &format!("{stem}.1"), args.norm_groups)?;
        }
        x = args.activation(x)?;
        let mut next = if self.is_v3 { 3 } else { 2 };
        if args.dconv_mode & 1 != 0 {
            x = self.dconv(weights, x, &format!("{stem}.{next}"), index)?;
            next += 1;
        }
        if args.rewrite {
            x = weights.conv(&x, &format!("{stem}.{next}"), 1, 0, 1, 1)?;
            if group_norm {
                x = group(
                    weights,
                    &x,
                    &format!("{stem}.{}", next + 1),
                    args.norm_groups,
                )?;
            }
            x = args.gate(x)?;
        }
        Ok(x)
    }

    fn decode(&self, weights: &Weights, input: &Tensor, index: usize) -> Result<Tensor> {
        let args = &self.args;
        let stem = format!("decoder.{}", args.depth - 1 - index);
        let mut x = input.shallow_clone();
        let mut next = 0;
        let group_norm = self.is_v3 && index >= args.norm_starts;
        if args.rewrite {
            let padding = if self.is_v3 { args.context } else { 0 };
            x = weights.conv(&x, &format!("{stem}.0"), 1, padding, 1, 1)?;
            if group_norm {
                x = group(weights, &x, &format!("{stem}.1"), args.norm_groups)?;
            }
            x = args.gate(x)?;
            next = if self.is_v3 { 3 } else { 2 };
        }
        if args.dconv_mode & 2 != 0 {
            x = self.dconv(weights, x, &format!("{stem}.{next}"), index)?;
            next += 1;
        }
        x = weights.deconv(&x, &format!("{stem}.{next}"), args.stride, 0)?;
        if index > 0 {
            if group_norm {
                x = group(
                    weights,
                    &x,
                    &format!("{stem}.{}", next + 1),
                    args.norm_groups,
                )?;
            }
            x = args.activation(x)?;
        }
        Ok(x)
    }

    fn dconv(&self, weights: &Weights, x: Tensor, name: &str, index: usize) -> Result<Tensor> {
        let args = &self.args;
        let hidden = (self.widths[index] as f64 / args.dconv_comp) as i64;
        ensure!(
            hidden > 0,
            "invalid waveform Demucs residual channels at {name}"
        );
        residual_stack(
            weights,
            x,
            name,
            args.dconv_depth,
            index >= args.dconv_lstm,
            index >= args.dconv_attn,
        )
    }
}

pub(super) fn residual_stack(
    weights: &Weights,
    mut x: Tensor,
    name: &str,
    depth: i32,
    recurrent: bool,
    attention: bool,
) -> Result<Tensor> {
    ensure!(
        depth.unsigned_abs() <= 30,
        "unsupported Demucs residual depth {depth}"
    );
    for layer in 0..depth.unsigned_abs() {
        let prefix = format!("{name}.layers.{layer}");
        let dilation = if depth > 0 { 1 << layer } else { 1 };
        let mut y = weights.conv(&x, &format!("{prefix}.0"), 1, dilation, dilation, 1)?;
        y = group(weights, &y, &format!("{prefix}.1"), 1)?.f_gelu("none")?;
        let mut projection = 3;
        if recurrent {
            y = blstm(weights, &y, &format!("{prefix}.{projection}"), 2, true)?;
            projection += 1;
        }
        if attention {
            y = local_attention(weights, &y, &format!("{prefix}.{projection}"))?;
            projection += 1;
        }
        y = weights.conv(&y, &format!("{prefix}.{projection}"), 1, 0, 1, 1)?;
        y = group(weights, &y, &format!("{prefix}.{}", projection + 1), 1)?.f_glu(1)?;
        let scale = weights.get(&format!("{prefix}.{}.scale", projection + 3))?;
        x = x.f_add(&y.f_mul(&scale.f_view([1, -1, 1])?)?)?;
    }
    Ok(x)
}

fn check_tensor(state: &HashMap<String, Tensor>, name: &str, expected: &[i64]) -> Result<()> {
    let value = state
        .get(name)
        .with_context(|| format!("legacy waveform checkpoint is missing {name}"))?;
    ensure!(
        value.is_floating_point() && value.size() == expected,
        "invalid legacy waveform tensor {name}, expected floating point {expected:?} (got {:?} {:?})",
        value.kind(),
        value.size()
    );
    Ok(())
}

fn group(weights: &Weights, x: &Tensor, name: &str, groups: i64) -> Result<Tensor> {
    Ok(x.f_group_norm(
        groups,
        Some(weights.get(&format!("{name}.weight"))?),
        Some(weights.get(&format!("{name}.bias"))?),
        NORM_EPSILON,
        true,
    )?)
}

pub(super) fn blstm(
    weights: &Weights,
    input: &Tensor,
    name: &str,
    layers: i64,
    framed: bool,
) -> Result<Tensor> {
    const WIDTH: i64 = 200;
    const STRIDE: i64 = WIDTH / 2;
    const CROP: i64 = STRIDE / 2;
    let (batch, channels, length) = input.size3()?;
    let frames = if framed && length > WIDTH {
        // Include the final partial frame so the residual keeps its original length.
        let pad = (STRIDE - (length - WIDTH) % STRIDE) % STRIDE;
        Some(
            input
                .f_constant_pad_nd([0, pad])?
                .f_unfold(-1, WIDTH, STRIDE)?
                .f_permute([0, 2, 1, 3])?,
        )
    } else {
        None
    };
    let x = match &frames {
        Some(frames) => frames.f_reshape([-1, channels, WIDTH])?,
        None => input.shallow_clone(),
    }
    .f_transpose(1, 2)?;
    let mut params = Vec::with_capacity(layers as usize * 8);
    for layer in 0..layers {
        for direction in ["", "_reverse"] {
            for part in ["weight_ih", "weight_hh", "bias_ih", "bias_hh"] {
                params.push(weights.get(&format!("{name}.lstm.{part}_l{layer}{direction}"))?);
            }
        }
    }
    let initial = Tensor::f_zeros([layers * 2, x.size()[0], channels], (x.kind(), x.device()))?;
    let (x, _, _) = x.f_lstm(
        &[&initial, &initial],
        &params,
        true,
        layers,
        0.0,
        false,
        true,
        true,
    )?;
    let x = weights
        .linear(&x, &format!("{name}.linear"))?
        .f_transpose(1, 2)?;
    if let Some(frames) = frames {
        let count = frames.size()[1];
        let frames = x.f_reshape([batch, count, channels, WIDTH])?;
        let mut pieces = Vec::with_capacity(count as usize);
        for index in 0..count {
            let start = if index == 0 { 0 } else { CROP };
            let end = if index == count - 1 {
                WIDTH
            } else {
                WIDTH - CROP
            };
            pieces.push(
                frames
                    .f_select(1, index)?
                    .f_narrow(-1, start, end - start)?,
            );
        }
        let x = Tensor::f_cat(&pieces, -1)?;
        Ok(input.f_add(&x.f_narrow(-1, 0, length)?)?)
    } else if framed {
        Ok(input.f_add(&x)?)
    } else {
        Ok(x)
    }
}

fn local_attention(weights: &Weights, x: &Tensor, name: &str) -> Result<Tensor> {
    const HEADS: i64 = 4;
    const DECAYS: i64 = 4;
    let (batch, channels, time) = x.size3()?;
    ensure!(
        channels % HEADS == 0,
        "invalid waveform Demucs attention channels"
    );
    let projections = |key: &str| {
        weights
            .conv(x, &format!("{name}.{key}"), 1, 0, 1, 1)?
            .f_view([batch, HEADS, -1, time])
            .map_err(anyhow::Error::from)
    };
    let query = projections("query")?;
    let key = projections("key")?;
    let content = projections("content")?;
    let opts = (x.kind(), x.device());
    let positions = Tensor::f_arange(time, opts)?;
    let delta = positions
        .f_unsqueeze(1)?
        .f_sub(&positions.f_unsqueeze(0)?)?;
    let decay = Tensor::f_arange_start(1, DECAYS + 1, opts)?.f_view([-1, 1, 1])?;
    let kernel = delta
        .f_abs()?
        .f_mul(&decay)?
        .f_div_scalar(-(DECAYS as f64).sqrt())?;
    let strength = projections("query_decay")?.f_sigmoid()?.f_div_scalar(2.0)?;
    let similarity = Tensor::f_einsum("bhct,bhcs->bhts", &[key, query], None::<&[i64]>)?
        .f_div_scalar(((channels / HEADS) as f64).sqrt())?;
    let bias = Tensor::f_einsum("fts,bhfs->bhts", &[kernel, strength], None::<&[i64]>)?;
    let diagonal = Tensor::f_eye(time, (Kind::Bool, x.device()))?;
    let attention = similarity
        .f_add(&bias)?
        .f_masked_fill(&diagonal, -100.0)?
        .f_softmax(2, None)?;
    let value = Tensor::f_einsum("bhts,bhct->bhcs", &[attention, content], None::<&[i64]>)?
        .f_reshape([batch, channels, time])?;
    Ok(x.f_add(&weights.conv(&value, &format!("{name}.proj"), 1, 0, 1, 1)?)?)
}

#[derive(Deserialize)]
#[serde(default)]
struct TasnetArgs {
    sources: Option<Sources>,
    #[serde(rename = "N")]
    channels_encoder: i64,
    #[serde(rename = "L")]
    kernel: i64,
    #[serde(rename = "B")]
    bottleneck: i64,
    #[serde(rename = "H")]
    hidden: i64,
    #[serde(rename = "P")]
    residual_kernel: i64,
    #[serde(rename = "X")]
    blocks: usize,
    #[serde(rename = "R")]
    repeats: usize,
    #[serde(rename = "C")]
    source_count: usize,
    audio_channels: usize,
    norm_type: String,
    causal: bool,
    mask_nonlinear: String,
    samplerate: u32,
    segment_length: usize,
}

impl Default for TasnetArgs {
    fn default() -> Self {
        Self {
            sources: None,
            channels_encoder: 256,
            kernel: 20,
            bottleneck: 256,
            hidden: 512,
            residual_kernel: 3,
            blocks: 8,
            repeats: 4,
            source_count: 4,
            audio_channels: 2,
            norm_type: "gLN".into(),
            causal: false,
            mask_nonlinear: "relu".into(),
            samplerate: 44_100,
            segment_length: 44_100 * 2 * 4,
        }
    }
}

struct Tasnet {
    args: TasnetArgs,
    norm: TasnetNorm,
    softmax: bool,
}

#[derive(Clone, Copy)]
enum TasnetNorm {
    Channel,
    Global,
    Identity,
    Batch,
}

impl TasnetNorm {
    fn forward(self, weights: &Weights, x: &Tensor, name: &str) -> Result<Tensor> {
        let axes: &[i64] = match self {
            Self::Channel => &[1],
            Self::Global => &[1, 2],
            Self::Identity => return Ok(x.shallow_clone()),
            Self::Batch => {
                return Ok(x.f_batch_norm(
                    Some(weights.get(&format!("{name}.weight"))?),
                    Some(weights.get(&format!("{name}.bias"))?),
                    Some(weights.get(&format!("{name}.running_mean"))?),
                    Some(weights.get(&format!("{name}.running_var"))?),
                    false,
                    0.1,
                    NORM_EPSILON,
                    true,
                )?);
            }
        };
        let gamma = weights.get(&format!("{name}.gamma"))?;
        let beta = weights.get(&format!("{name}.beta"))?;
        let centered = x.f_sub(&x.f_mean_dim(axes, true, None)?)?;
        let scale = centered
            .f_square()?
            .f_mean_dim(axes, true, None)?
            .f_add_scalar(1e-8)?
            .f_rsqrt()?;
        Ok(centered.f_mul(&scale)?.f_mul(gamma)?.f_add(beta)?)
    }
}

impl Tasnet {
    fn load(args: TasnetArgs, stems: usize, state: &HashMap<String, Tensor>) -> Result<Self> {
        ensure!(
            (1..=16_384).contains(&args.channels_encoder)
                && (2..=256).contains(&args.kernel)
                && (1..=16_384).contains(&args.bottleneck)
                && (1..=16_384).contains(&args.hidden)
                && (1..=31).contains(&args.residual_kernel)
                && args.residual_kernel % 2 == 1
                && (!args.causal || args.residual_kernel > 1)
                && (1..=20).contains(&args.blocks)
                && (1..=32).contains(&args.repeats),
            "invalid legacy ConvTasNet layer geometry"
        );
        let channels = args.audio_channels as i64;
        let n = args.channels_encoder;
        check_tensor(
            state,
            "encoder.conv1d_U.weight",
            &[n, channels, args.kernel],
        )?;
        check_tensor(
            state,
            "decoder.basis_signals.weight",
            &[channels * args.kernel, n],
        )?;
        check_tensor(
            state,
            "separator.network.1.weight",
            &[args.bottleneck, n, 1],
        )?;
        check_tensor(
            state,
            "separator.network.3.weight",
            &[stems as i64 * n, args.bottleneck, 1],
        )?;
        let norm = match args.norm_type.as_str() {
            "cLN" => TasnetNorm::Channel,
            "gLN" => TasnetNorm::Global,
            "id" => TasnetNorm::Identity,
            _ => TasnetNorm::Batch,
        };
        let softmax = match args.mask_nonlinear.as_str() {
            "relu" => false,
            "softmax" => true,
            other => bail!("unsupported legacy ConvTasNet mask function {other:?}"),
        };
        Ok(Self {
            args,
            norm,
            softmax,
        })
    }

    fn forward(&self, weights: &Weights, input: &Tensor, stems: i64) -> Result<Tensor> {
        let args = &self.args;
        ensure!(
            last_len(input) >= args.kernel,
            "legacy ConvTasNet requires at least {} frames",
            args.kernel
        );
        let hop = args.kernel / 2;
        let encoded = weights
            .conv(input, "encoder.conv1d_U", hop, 0, 1, 1)?
            .f_relu()?;
        let (batch, features, frames) = encoded.size3()?;
        let prefix = "separator.network";
        let mut x = TasnetNorm::Channel.forward(weights, &encoded, &format!("{prefix}.0"))?;
        x = weights.conv(&x, &format!("{prefix}.1"), 1, 0, 1, 1)?;
        for repeat in 0..args.repeats {
            for block in 0..args.blocks {
                let name = format!("{prefix}.2.{repeat}.{block}.net");
                let mut y = weights
                    .conv(&x, &format!("{name}.0"), 1, 0, 1, 1)?
                    .f_prelu(weights.get(&format!("{name}.1.weight"))?)?;
                y = self.norm.forward(weights, &y, &format!("{name}.2"))?;
                y = self.depthwise(weights, &y, &format!("{name}.3.net"), block)?;
                x = x.f_add(&y)?;
            }
        }
        let score = weights
            .conv(&x, &format!("{prefix}.3"), 1, 0, 1, 1)?
            .f_view([batch, stems, features, frames])?;
        let mask = if self.softmax {
            score.f_softmax(1, None)?
        } else {
            score.f_relu()?
        };
        let source = encoded.f_unsqueeze(1)?.f_mul(&mask)?.f_transpose(2, 3)?;
        let channels = args.audio_channels as i64;
        let source = weights
            .linear(&source, "decoder.basis_signals")?
            .f_view([batch, stems, frames, channels, args.kernel])?
            .f_transpose(2, 3)?
            .f_reshape([batch, stems, channels, -1])?;
        let opts = (Kind::Int64, input.device());
        let indices = Tensor::f_arange(frames, opts)?
            .f_mul_scalar(hop)?
            .f_unsqueeze(1)?
            .f_add(&Tensor::f_arange(args.kernel, opts)?)?
            .f_reshape([-1])?;
        let length = (frames - 1) * hop + args.kernel;
        let decoded = Tensor::f_zeros(
            [batch, stems, channels, length],
            (input.kind(), input.device()),
        )?
        .f_index_add(-1, &indices, &source)?;
        let target = last_len(input);
        if length <= target {
            Ok(decoded.f_constant_pad_nd([0, target - length])?)
        } else {
            Ok(decoded.f_narrow(-1, 0, target)?)
        }
    }

    fn depthwise(&self, weights: &Weights, x: &Tensor, name: &str, block: usize) -> Result<Tensor> {
        let args = &self.args;
        let dilation = 1 << block;
        let spread = (args.residual_kernel - 1) * dilation;
        let padding = if args.causal { spread } else { spread / 2 };
        let mut x = weights.conv(x, &format!("{name}.0"), 1, padding, dilation, args.hidden)?;
        let activation = if args.causal {
            x = x.f_narrow(-1, 0, last_len(&x) - padding)?;
            2
        } else {
            1
        };
        x = x.f_prelu(weights.get(&format!("{name}.{activation}.weight"))?)?;
        x = self
            .norm
            .forward(weights, &x, &format!("{name}.{}", activation + 1))?;
        weights.conv(&x, &format!("{name}.{}", activation + 2), 1, 0, 1, 1)
    }
}
