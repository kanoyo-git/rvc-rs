#[cfg(feature = "onnx")]
mod onnx_decoder;
#[cfg(feature = "onnx")]
mod onnx_flow;
#[cfg(feature = "onnx")]
mod onnx_prior;
#[cfg(test)]
mod tests;
pub mod training;

use std::cell::RefCell;
use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use serde_json::Value;
use tch::{Device, Kind, Tensor};

use super::cuda_graph::Cache;
use super::weights::{Weights, leaky};
use crate::catalog::ModelInfo;
use crate::checkpoint::Checkpoint;

const SLOPE: f64 = 0.1;

pub struct Synth {
    pub info: ModelInfo,
    pub rate: u32,
    weights: Weights,
    config: Config,
    #[cfg(feature = "onnx")]
    onnx: Onnx,
    pub(super) graph: RefCell<Cache>,
}

#[cfg(feature = "onnx")]
#[derive(Default)]
struct Onnx {
    prior: Option<onnx_prior::Prior>,
    flow: Option<onnx_flow::Flow>,
    decoder: Option<onnx_decoder::Decoder>,
}

struct Config {
    inter: i64,
    hidden: i64,
    heads: i64,
    encoder_layers: i64,
    encoder_kernel: i64,
    dropout: f64,
    resblock: u8,
    kernels: Vec<i64>,
    dilations: Vec<Vec<i64>>,
    upsample_rates: Vec<i64>,
    upsample_kernels: Vec<i64>,
}

impl Config {
    fn read(value: Value, rate: u32) -> Result<Self> {
        let list = value
            .as_array()
            .filter(|list| list.len() == 18)
            .context("invalid RVC generator configuration")?;
        let number = |index: usize, max: i64| -> Result<i64> {
            list[index]
                .as_i64()
                .filter(|n| (1..=max).contains(n))
                .with_context(|| format!("invalid RVC config field {index}"))
        };
        let numbers = |value: &Value| -> Result<Vec<i64>> {
            value
                .as_array()
                .filter(|list| (1..=12).contains(&list.len()))
                .context("expected a list in RVC generator configuration")?
                .iter()
                .map(|v| {
                    v.as_i64()
                        .filter(|n| (1..=1024).contains(n))
                        .context("invalid RVC config list item")
                })
                .collect()
        };
        let resblock = match list[9].as_str() {
            Some("1") => 1,
            Some("2") => 2,
            _ => bail!("unknown RVC residual block type"),
        };
        let config = Self {
            inter: number(2, 1024)?,
            hidden: number(3, 1024)?,
            heads: number(5, 32)?,
            encoder_layers: number(6, 32)?,
            encoder_kernel: number(7, 63)?,
            dropout: list[8]
                .as_f64()
                .filter(|value| (0.0..1.0).contains(value))
                .context("invalid encoder dropout")?,
            resblock,
            kernels: numbers(&list[10])?,
            dilations: list[11]
                .as_array()
                .context("missing residual dilations")?
                .iter()
                .map(numbers)
                .collect::<Result<_>>()?,
            upsample_rates: numbers(&list[12])?,
            upsample_kernels: numbers(&list[14])?,
        };
        let hop = config
            .upsample_rates
            .iter()
            .try_fold(1_i64, |a, b| a.checked_mul(*b));
        ensure!(
            hop == Some(i64::from(rate) / 100),
            "generator upsample product does not match sample rate"
        );
        ensure!(
            config.inter % 2 == 0
                && config.hidden % config.heads == 0
                && config.kernels.len() == config.dilations.len()
                && config.kernels.iter().all(|n| n % 2 == 1)
                && config
                    .dilations
                    .iter()
                    .all(|d| d.len() >= if resblock == 1 { 3 } else { 2 })
                && config.upsample_rates.len() == config.upsample_kernels.len()
                && config
                    .upsample_rates
                    .iter()
                    .zip(&config.upsample_kernels)
                    .all(|(stride, kernel)| stride <= kernel),
            "inconsistent channel, kernel or dilation configuration"
        );
        Ok(config)
    }

    fn decoder_context(&self, hop: i64) -> Option<i64> {
        let depth = if self.resblock == 1 { 3 } else { 2 };
        let extra = i64::from(self.resblock == 1);
        let residual = self
            .kernels
            .iter()
            .zip(&self.dilations)
            .map(|(&kernel, dilations)| {
                (kernel - 1) / 2
                    * dilations
                        .iter()
                        .take(depth)
                        .map(|dilation| dilation + extra)
                        .sum::<i64>()
            })
            .max()?;
        // Indices changed by cropping can spread across each convolution.
        // Count the affected samples starting at conv_pre, then conv_post.
        let mut context = 3;
        for (&stride, &kernel) in self.upsample_rates.iter().zip(&self.upsample_kernels) {
            let difference = kernel - stride;
            if difference % 2 != 0 {
                return None;
            }
            context = context * stride + difference / 2 + residual;
        }
        Some((context + 3 + hop - 1) / hop)
    }
}

#[derive(Clone, Copy, Default)]
pub struct Window {
    pub skip: i64,
    pub frames: i64,
    pub stretched_frames: i64,
    pub trim_frames: i64,
}

impl Synth {
    pub fn load(path: &Path, device: Device, kind: Kind) -> Result<Self> {
        let [prior, flow, decoder] =
            ["RVC_PRIOR_ONNX", "RVC_FLOW_ONNX", "RVC_DECODER_ONNX"].map(super::onnx_selection);
        #[cfg(not(feature = "onnx"))]
        if let Some(value) = prior.or(flow).or(decoder) {
            bail!(
                "generator provider {} requires the onnx build feature",
                value.to_string_lossy()
            );
        }
        let mut checkpoint = Checkpoint::open(path)?;
        let info = ModelInfo::from_checkpoint(&checkpoint)?;
        ensure!(
            matches!(info.version.as_str(), "v1" | "v2"),
            "unsupported RVC version {}",
            info.version
        );
        let rate = info
            .sample_rate
            .filter(|rate| matches!(rate, 32_000 | 40_000 | 48_000))
            .context("unsupported RVC sample rate")?;
        let config = Config::read(
            checkpoint
                .metadata("config")?
                .context("missing generator config")?,
            rate,
        )?;
        let mut weights = Weights::new(checkpoint.weights(Some("weight"), Device::Cpu)?);
        weights.remove_weight_norm(0)?;
        #[cfg(feature = "onnx")]
        let onnx = {
            use super::onnx::Provider;

            let prior = Provider::selected("RVC_PRIOR_ONNX", prior)?
                .map(|provider| onnx_prior::Prior::load(&weights, &config, info.uses_f0, provider))
                .transpose()?;
            if prior.is_some() {
                weights.retain(|name| !name.starts_with("enc_p."));
            }
            let flow = Provider::selected("RVC_FLOW_ONNX", flow)?
                .map(|provider| onnx_flow::Flow::load(&weights, &config, provider))
                .transpose()?;
            if flow.is_some() {
                weights.retain(|name| !name.starts_with("flow."));
            }
            let decoder = Provider::selected("RVC_DECODER_ONNX", decoder)?
                .map(|provider| {
                    onnx_decoder::Decoder::load(&weights, &config, info.uses_f0, provider)
                })
                .transpose()?;
            if decoder.is_some() {
                weights
                    .retain(|name| !name.starts_with("dec.") || name.starts_with("dec.m_source."));
            }
            Onnx {
                prior,
                flow,
                decoder,
            }
        };
        weights.move_to(device, kind)?;
        weights.prepare_cpu_convolutions(device)?;
        Ok(Self {
            info,
            rate,
            weights,
            config,
            #[cfg(feature = "onnx")]
            onnx,
            graph: RefCell::default(),
        })
    }

    pub fn infer(
        &self,
        phone: &Tensor,
        pitch: Option<(&Tensor, &Tensor)>,
        speaker: i64,
        window: Window,
    ) -> Result<Tensor> {
        let feature_frames = phone.size()[1];
        ensure!(
            window.skip >= 0
                && window.frames > 0
                && window.stretched_frames > 0
                && window.skip + window.frames <= feature_frames,
            "invalid generator render window"
        );
        ensure!(
            pitch.is_some() == self.info.uses_f0,
            "pitch inputs do not match generator"
        );
        let ids = Tensor::full([1], speaker, (Kind::Int64, phone.device()));
        let speaker = self.speaker(&ids)?;
        let flow_head = (window.skip - 24).max(0);
        let coarse = pitch.map(|(coarse, _)| coarse);
        let (mean, logs, mask) = self.inference_prior(phone, coarse, flow_head)?;
        // Keep the seeded noise in logical channel order across convolution layouts.
        let noise = Tensor::randn(mean.size(), (mean.kind(), mean.device()));
        let latent = (&mean + logs.exp() * noise * 0.66666) * &mask;
        let z = self.flow(latent, &mask, &speaker, true)?;
        let z = z.narrow(2, window.skip - flow_head, window.frames);
        let f0 = pitch.map(|(_, hz)| hz.narrow(1, window.skip, window.frames));
        self.decode(
            &z,
            f0.as_ref(),
            &speaker,
            window.stretched_frames,
            window.trim_frames,
        )
    }

    pub(super) fn infer_realtime(
        &self,
        phone: &Tensor,
        pitch: Option<(&Tensor, &Tensor)>,
        speaker: i64,
        window: Window,
        pool: &Cache,
    ) -> Result<Tensor> {
        #[cfg(feature = "onnx")]
        if self.onnx.prior.is_some() || self.onnx.flow.is_some() || self.onnx.decoder.is_some() {
            return self.infer(phone, pitch, speaker, window);
        }
        let key = [speaker, window.skip, window.frames, window.stretched_frames];
        let mut graph = self.graph.borrow_mut();
        match pitch {
            Some((coarse, hz)) => graph.run(key, &[phone, coarse, hz], Some(pool), |args| {
                self.infer(args[0], Some((args[1], args[2])), speaker, window)
            }),
            None => graph.run(key, &[phone], Some(pool), |args| {
                self.infer(args[0], None, speaker, window)
            }),
        }
    }

    fn speaker(&self, ids: &Tensor) -> Result<Tensor> {
        Ok(self
            .weights
            .get("emb_g.weight")?
            .f_index_select(0, &ids.view([-1]))?
            .unsqueeze(-1))
    }

    fn inference_prior(
        &self,
        phone: &Tensor,
        pitch: Option<&Tensor>,
        skip: i64,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        #[cfg(feature = "onnx")]
        if let Some(prior) = &self.onnx.prior {
            return prior.stats(phone, pitch, skip);
        }
        let lengths = Tensor::full([1], phone.size()[1], (Kind::Int64, phone.device()));
        self.prior(phone, pitch, &lengths, skip, false)
    }

    fn prior(
        &self,
        phone: &Tensor,
        pitch: Option<&Tensor>,
        lengths: &Tensor,
        skip: i64,
        training: bool,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let w = &self.weights;
        let c = &self.config;
        let size = phone.size();
        let frames = size[1];
        let mut x = w.linear(phone, "enc_p.emb_phone")?;
        if let Some(coarse) = pitch {
            x = x.f_add(
                &w.get("enc_p.emb_pitch.weight")?
                    .f_index_select(0, &coarse.reshape([-1]))?
                    .view([size[0], frames, c.hidden]),
            )?;
        }
        x = leaky(&(x * (c.hidden as f64).sqrt()), SLOPE).transpose(1, 2);
        let mask = sequence_mask(lengths, frames, x.kind());
        let attention_mask = (&mask.unsqueeze(2) * &mask.unsqueeze(-1)).eq(0);
        x = x.f_mul(&mask)?;
        for layer in 0..c.encoder_layers {
            let prefix = "enc_p.encoder";
            let attn = self
                .attention(
                    &x,
                    &attention_mask,
                    &format!("{prefix}.attn_layers.{layer}"),
                    training,
                )?
                .dropout(c.dropout, training);
            x = w.rvc_norm(&(x + attn), &format!("{prefix}.norm_layers_1.{layer}"))?;
            let ffn = format!("{prefix}.ffn_layers.{layer}");
            let padding = [(c.encoder_kernel - 1) / 2, c.encoder_kernel / 2];
            let y = w
                .conv(
                    &(&x * &mask).constant_pad_nd(padding),
                    &format!("{ffn}.conv_1"),
                    1,
                    0,
                    1,
                    1,
                )?
                .relu()
                .dropout(c.dropout, training);
            let y = w.conv(
                &(y * &mask).constant_pad_nd(padding),
                &format!("{ffn}.conv_2"),
                1,
                0,
                1,
                1,
            )?;
            let y = (y * &mask).dropout(c.dropout, training);
            x = w.rvc_norm(&(x + y), &format!("{prefix}.norm_layers_2.{layer}"))?;
        }
        let mask = mask.narrow(2, skip, frames - skip);
        let stats = w.conv(
            &(x.narrow(2, skip, frames - skip) * &mask),
            "enc_p.proj",
            1,
            0,
            1,
            1,
        )? * &mask;
        Ok((
            stats.narrow(1, 0, c.inter),
            stats.narrow(1, c.inter, c.inter),
            mask,
        ))
    }

    fn attention(&self, x: &Tensor, mask: &Tensor, name: &str, training: bool) -> Result<Tensor> {
        let w = &self.weights;
        let heads = self.config.heads;
        let b = x.size()[0];
        let t = x.size()[2];
        let channels = x.size()[1] / heads;
        let project = |suffix| -> Result<Tensor> {
            Ok(w.conv(x, &format!("{name}.conv_{suffix}"), 1, 0, 1, 1)?
                .view([b, heads, channels, t])
                .transpose(2, 3))
        };
        let q = project('q')? / (channels as f64).sqrt();
        let k = project('k')?;
        let v = project('v')?;
        let relative = |suffix| -> Result<Tensor> {
            let emb = w.get(&format!("{name}.emb_rel_{suffix}"))?;
            let window = (emb.size()[1] + 1) / 2;
            let pad = (t - window).max(0);
            let begin = (window - t).max(0);
            Ok(emb
                .constant_pad_nd([0, 0, pad, pad])
                .narrow(1, begin, 2 * t - 1)
                .unsqueeze(0))
        };
        let rel = q.matmul(&relative('k')?.transpose(-2, -1));
        let rel = rel
            .constant_pad_nd([0, 1])
            .view([b, heads, t * 2 * t])
            .constant_pad_nd([0, t - 1])
            .view([b, heads, t + 1, 2 * t - 1])
            .narrow(2, 0, t)
            .narrow(3, t - 1, t);
        let attention = (q.matmul(&k.transpose(2, 3)) + rel)
            .masked_fill(mask, -1e4)
            .softmax(-1, x.kind())
            .dropout(self.config.dropout, training);
        let relative_weights = attention
            .constant_pad_nd([0, t - 1])
            .view([b, heads, t * (2 * t - 1)])
            .constant_pad_nd([t, 0])
            .view([b, heads, t, 2 * t])
            .narrow(3, 1, 2 * t - 1);
        let result = attention.matmul(&v) + relative_weights.matmul(&relative('v')?);
        w.conv(
            &result
                .transpose(2, 3)
                .contiguous()
                .view([b, heads * channels, t]),
            &format!("{name}.conv_o"),
            1,
            0,
            1,
            1,
        )
    }

    fn flow(
        &self,
        mut x: Tensor,
        mask: &Tensor,
        speaker: &Tensor,
        reverse: bool,
    ) -> Result<Tensor> {
        #[cfg(feature = "onnx")]
        if reverse && let Some(flow) = &self.onnx.flow {
            return flow.reverse(&x, speaker);
        }
        let w = &self.weights;
        let half = self.config.inter / 2;
        let layers = if reverse { [6, 4, 2, 0] } else { [0, 2, 4, 6] };
        for layer in layers {
            if reverse {
                x = x.flip([1]);
            }
            let name = format!("flow.flows.{layer}");
            let x0 = x.narrow(1, 0, half);
            let hidden = w.conv(&x0, &format!("{name}.pre"), 1, 0, 1, 1)? * mask;
            let sum = self.wavenet(hidden, mask, speaker, &format!("{name}.enc"), 3)?;
            let mean = w.conv(&sum, &format!("{name}.post"), 1, 0, 1, 1)? * mask;
            let x1 = x.narrow(1, half, half);
            let x1 = if reverse {
                (x1 - mean) * mask
            } else {
                mean + x1 * mask
            };
            x = Tensor::cat(&[x0, x1], 1);
            if !reverse {
                x = x.flip([1]);
            }
        }
        Ok(x)
    }

    fn wavenet(
        &self,
        mut hidden: Tensor,
        mask: &Tensor,
        speaker: &Tensor,
        name: &str,
        depth: i64,
    ) -> Result<Tensor> {
        let w = &self.weights;
        let width = hidden.size()[1];
        let cond = w.conv(speaker, &format!("{name}.cond_layer"), 1, 0, 1, 1)?;
        let mut sum = Tensor::zeros_like(&hidden);
        for i in 0..depth {
            let a = w.conv(&hidden, &format!("{name}.in_layers.{i}"), 1, 2, 1, 1)?
                + cond.narrow(1, 2 * width * i, 2 * width);
            let gate = a.narrow(1, 0, width).tanh() * a.narrow(1, width, width).sigmoid();
            let skip = w.conv(&gate, &format!("{name}.res_skip_layers.{i}"), 1, 0, 1, 1)?;
            if i < depth - 1 {
                hidden = (hidden + skip.narrow(1, 0, width)) * mask;
                sum = sum.f_add(&skip.narrow(1, width, width))?;
            } else {
                sum = sum.f_add(&skip)?;
            }
        }
        Ok(sum * mask)
    }

    fn decode(
        &self,
        latent: &Tensor,
        pitch: Option<&Tensor>,
        speaker: &Tensor,
        frames: i64,
        trim_frames: i64,
    ) -> Result<Tensor> {
        let c = &self.config;
        let w = &self.weights;
        let hop = i64::from(self.rate) / 100;
        let resize = |x: &Tensor, len: i64| -> Tensor {
            if x.size()[2] == len {
                x.shallow_clone()
            } else {
                x.upsample_linear1d([len], false, None)
            }
        };
        let excitation = pitch
            .map(|pitch| -> Result<Tensor> {
                let steps = Tensor::arange_start(1, hop + 1, (Kind::Float, pitch.device()));
                let radians = pitch.unsqueeze(-1) / f64::from(self.rate) * steps;
                let phase = (radians.select(2, hop - 1).unsqueeze(-1) + 0.5).fmod(1.0) - 0.5;
                let phase = phase
                    .cumsum(1, Kind::Float)
                    .fmod(1.0)
                    .constant_pad_nd([0, 0, 1, -1]);
                // SineGen draws a phase for each harmonic, then zeros the fundamental.
                // RVC uses only the fundamental, but the draw is part of its RNG sequence.
                let _ = Tensor::rand([1, 1, 1], (Kind::Float, pitch.device()));
                let sine = ((radians + phase).view([pitch.size()[0], -1, 1])
                    * std::f64::consts::TAU)
                    .sin()
                    * 0.1;
                let voiced = pitch
                    .gt(0.0)
                    .repeat_interleave_self_int(hop, 1, Some(pitch.size()[1] * hop))
                    .unsqueeze(-1)
                    .to_kind(Kind::Float);
                let noise_amp = &voiced * 0.003 + (1.0 - &voiced) * (0.1 / 3.0);
                let excitation: Tensor = &sine * voiced + Tensor::randn_like(&sine) * noise_amp;
                let harmonics = w
                    .linear(&excitation.to_kind(latent.kind()), "dec.m_source.l_linear")?
                    .tanh()
                    .transpose(1, 2);
                Ok(resize(&harmonics, frames * hop))
            })
            .transpose()?;
        let crop = if trim_frames > 0 && frames > 2 * trim_frames {
            c.decoder_context(hop)
                .map_or(0, |context| (trim_frames - context).max(0))
        } else {
            0
        };
        let latent = resize(latent, frames);
        let latent = if crop == 0 {
            latent
        } else {
            latent.narrow(2, crop, frames - 2 * crop)
        };
        let output = self.decode_convolutions(&latent, excitation.as_ref(), speaker, crop, hop)?;
        if trim_frames == 0 {
            return Ok(output);
        }
        let pad = (trim_frames - crop) * hop;
        let length = output.size()[2] - 2 * pad;
        ensure!(
            length > 0,
            "conversion segment is too short after removing context"
        );
        Ok(output.f_narrow(2, pad, length)?)
    }

    fn decode_convolutions(
        &self,
        latent: &Tensor,
        excitation: Option<&Tensor>,
        speaker: &Tensor,
        crop: i64,
        hop: i64,
    ) -> Result<Tensor> {
        #[cfg(feature = "onnx")]
        if let Some(decoder) = &self.onnx.decoder {
            let cropped = excitation
                .filter(|_| crop != 0)
                .map(|source| source.f_narrow(2, crop * hop, latent.size()[2] * hop))
                .transpose()?;
            return decoder.run(latent, speaker, cropped.as_ref().or(excitation));
        }
        let c = &self.config;
        let w = &self.weights;
        let mut x = w.conv(latent, "dec.conv_pre", 1, 3, 1, 1)?
            + w.conv(speaker, "dec.cond", 1, 0, 1, 1)?;
        let mut stage_hop = 1;
        for (i, (&stride, &kernel)) in c.upsample_rates.iter().zip(&c.upsample_kernels).enumerate()
        {
            stage_hop *= stride;
            x = w.deconv(
                &leaky(&x, SLOPE),
                &format!("dec.ups.{i}"),
                stride,
                (kernel - stride) / 2,
            )?;
            if let Some(source) = excitation {
                let stride = hop / stage_hop;
                let padding = stride / 2;
                let name = format!("dec.noise_convs.{i}");
                // The full excitation retains phase and noise from the
                // discarded frames. Take only the taps for this window.
                let (restricted, padding) = if crop == 0 {
                    (None, padding)
                } else {
                    let shape = w.get(&format!("{name}.weight"))?.size();
                    let kernel = shape.get(2).context("invalid excitation kernel")?;
                    let length = (x.size()[2] - 1) * stride + kernel;
                    (Some(source.f_narrow(2, crop * hop - padding, length)?), 0)
                };
                x = x.f_add(&w.conv(
                    restricted.as_ref().unwrap_or(source),
                    &name,
                    stride,
                    padding,
                    1,
                    1,
                )?)?;
            }
            let mut combined = Tensor::zeros_like(&x);
            for (j, (&kernel, dilations)) in c.kernels.iter().zip(&c.dilations).enumerate() {
                let layer = format!("dec.resblocks.{}", i * c.kernels.len() + j);
                let mut branch = x.shallow_clone();
                let depth = if c.resblock == 1 { 3 } else { 2 };
                for (n, &dilation) in dilations.iter().take(depth).enumerate() {
                    let group = if c.resblock == 1 { "convs1" } else { "convs" };
                    let mut y = w.conv(
                        &leaky(&branch, SLOPE),
                        &format!("{layer}.{group}.{n}"),
                        1,
                        (kernel - 1) * dilation / 2,
                        dilation,
                        1,
                    )?;
                    if c.resblock == 1 {
                        y = w.conv(
                            &leaky(&y, SLOPE),
                            &format!("{layer}.convs2.{n}"),
                            1,
                            (kernel - 1) / 2,
                            1,
                            1,
                        )?;
                    }
                    branch = branch.f_add(&y)?;
                }
                combined += branch;
            }
            x = combined / c.kernels.len() as f64;
        }
        Ok(w.conv(&leaky(&x, 0.01), "dec.conv_post", 1, 3, 1, 1)?
            .tanh())
    }
}

fn sequence_mask(lengths: &Tensor, frames: i64, kind: Kind) -> Tensor {
    Tensor::arange(frames, (Kind::Int64, lengths.device()))
        .view([1, 1, frames])
        .lt_tensor(&lengths.view([-1, 1, 1]))
        .to_kind(kind)
}
