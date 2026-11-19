use anyhow::{Result, bail, ensure};
use rand::Rng;
use serde_json::Value;
use tch::{Device, Kind, Tensor};

use crate::model::weights::Weights;

use super::layers::group_norm;

#[path = "transformer/mlx_transformer.rs"]
mod mlx_transformer;
#[cfg(feature = "onnx")]
#[path = "transformer/onnx_transformer.rs"]
mod onnx_transformer;

const PREFIX: &str = "crosstransformer";

pub(super) struct Transformer {
    layers: usize,
    bottleneck: i64,
    heads: i64,
    cross_first: bool,
    pre_norm: bool,
    norm_in_groups: i64,
    groups: i64,
    output_groups: i64,
    gelu: bool,
    embedding: Embedding,
    period: f64,
    position_scale: f64,
}

enum Embedding {
    Sin { random_shift: i64 },
    Cape { center: bool },
    Scaled,
}

impl Transformer {
    pub(super) fn supports_mlx(&self) -> bool {
        matches!(self.embedding, Embedding::Sin { .. })
    }

    pub(super) fn new(config: &Value, layers: usize, bottleneck: i64) -> Result<Self> {
        ensure!(
            !flag(config, "t_sparse_self_attn", false)
                && !flag(config, "t_sparse_cross_attn", false),
            "sparse Demucs attention is unsupported by the original local model"
        );
        let heads = config["t_heads"].as_i64().unwrap_or(8);
        let period = config["t_max_period"].as_f64().unwrap_or(10_000.0);
        let position_scale = config["t_weight_pos_embed"].as_f64().unwrap_or(1.0);
        let random_shift = config["t_sin_random_shift"].as_i64().unwrap_or(0);
        ensure!(
            (1..=128).contains(&heads)
                && (1.0..=1e12).contains(&period)
                && position_scale.is_finite()
                && (0..=1_000_000).contains(&random_shift)
                && (0..=16384).contains(&bottleneck),
            "invalid Demucs transformer parameters"
        );
        let embedding = match config["t_emb"].as_str().unwrap_or("sin") {
            "sin" => Embedding::Sin { random_shift },
            "cape" => Embedding::Cape {
                center: flag(config, "t_cape_mean_normalize", true),
            },
            "scaled" => Embedding::Scaled,
            name => bail!("unsupported Demucs positional embedding {name:?}"),
        };
        let groups = groups(&config["t_group_norm"], false);
        let norm_in_groups = if flag(config, "t_norm_in", true) {
            0
        } else {
            groups_for_input(config)
        };
        Ok(Self {
            layers,
            bottleneck,
            heads,
            groups,
            norm_in_groups,
            embedding,
            period,
            position_scale,
            cross_first: flag(config, "t_cross_first", false),
            pre_norm: flag(config, "t_norm_first", true),
            output_groups: self::groups(&config["t_norm_out"], true),
            gelu: flag(config, "t_gelu", true),
        })
    }

    pub(super) fn forward(
        &self,
        weights: &Weights,
        x: &Tensor,
        wave: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let dimensions = x.size();
        ensure!(
            dimensions.len() == 4,
            "Demucs transformer expected a frequency tensor"
        );
        let (batch, channels, freqs, steps) =
            (dimensions[0], dimensions[1], dimensions[2], dimensions[3]);
        let mut signal = wave.shallow_clone();
        let mut spectral = if self.bottleneck == 0 {
            x.shallow_clone()
        } else {
            signal = weights.conv(wave, "channel_upsampler_t", 1, 0, 1, 1)?;
            weights
                .conv(
                    &x.f_reshape([batch, channels, freqs * steps])?,
                    "channel_upsampler",
                    1,
                    0,
                    1,
                    1,
                )?
                .f_reshape([batch, self.bottleneck, freqs, steps])?
        };
        let dim = spectral.size()[1];
        ensure!(
            dim >= 4 && dim % 4 == 0 && dim % self.heads == 0 && signal.size()[1] == dim,
            "Demucs transformer dimensions do not match its attention heads"
        );
        let pos = self.position_2d(freqs, steps, dim, spectral.device())?;
        let pos_t = self.position_1d(weights, signal.size()[2], dim, spectral.device())?;
        spectral = spectral
            .f_permute([0, 3, 2, 1])?
            .f_reshape([batch, steps * freqs, dim])?;
        spectral = self
            .norm(
                weights,
                &format!("{PREFIX}.norm_in"),
                &spectral,
                self.norm_in_groups,
            )?
            .f_add(&pos.f_mul_scalar(self.position_scale)?)?;
        signal = self
            .norm(
                weights,
                &format!("{PREFIX}.norm_in_t"),
                &signal.f_transpose(1, 2)?,
                self.norm_in_groups,
            )?
            .f_add(&pos_t.f_mul_scalar(self.position_scale)?)?;
        for index in 0..self.layers {
            let frequency_name = format!("{PREFIX}.layers.{index}");
            let temporal_name = format!("{PREFIX}.layers_t.{index}");
            let cross = (index % 2 == 1) != self.cross_first;
            let frequency_keys = cross.then_some(&signal);
            let temporal_keys = cross.then_some(&spectral);
            let next_frequency = self.block(weights, &frequency_name, &spectral, frequency_keys)?;
            signal = self.block(weights, &temporal_name, &signal, temporal_keys)?;
            spectral = next_frequency;
        }
        spectral = spectral
            .f_reshape([batch, steps, freqs, dim])?
            .f_permute([0, 3, 2, 1])?;
        signal = signal.f_transpose(1, 2)?;
        if self.bottleneck != 0 {
            spectral = weights
                .conv(
                    &spectral.f_reshape([batch, dim, freqs * steps])?,
                    "channel_downsampler",
                    1,
                    0,
                    1,
                    1,
                )?
                .f_reshape([batch, channels, freqs, steps])?;
            signal = weights.conv(&signal, "channel_downsampler_t", 1, 0, 1, 1)?;
        }
        Ok((spectral, signal))
    }

    fn block(
        &self,
        weights: &Weights,
        name: &str,
        q: &Tensor,
        cross: Option<&Tensor>,
    ) -> Result<Tensor> {
        let first = format!("{name}.norm1");
        let second = format!("{name}.norm2");
        let attention = format!(
            "{name}.{}",
            if cross.is_some() {
                "cross_attn"
            } else {
                "self_attn"
            }
        );
        let mut x;
        if self.pre_norm {
            let query = self.norm(weights, &first, q, self.groups)?;
            let keys = cross
                .map(|k| self.norm(weights, &second, k, self.groups))
                .transpose()?;
            let y = self.attend(weights, &attention, &query, keys.as_ref())?;
            x = q.f_add(&self.scale(weights, name, "gamma_1", y)?)?;
            let norm = if cross.is_some() {
                format!("{name}.norm3")
            } else {
                second
            };
            let input = self.norm(weights, &norm, &x, self.groups)?;
            x =
                x.f_add(&self.scale(weights, name, "gamma_2", self.ff(weights, name, &input)?)?)?;
            if self.output_groups > 0 {
                x = self.norm(weights, &format!("{name}.norm_out"), &x, self.output_groups)?;
            }
        } else {
            let y = self.attend(weights, &attention, q, cross)?;
            x = self.norm(
                weights,
                &first,
                &q.f_add(&self.scale(weights, name, "gamma_1", y)?)?,
                self.groups,
            )?;
            let y = self.ff(weights, name, &x)?;
            x = self.norm(
                weights,
                &second,
                &x.f_add(&self.scale(weights, name, "gamma_2", y)?)?,
                self.groups,
            )?;
        }
        Ok(x)
    }

    fn attend(
        &self,
        weights: &Weights,
        name: &str,
        query: &Tensor,
        cross: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (batch, steps, dim) = (query.size()[0], query.size()[1], query.size()[2]);
        let w = weights.get(&format!("{name}.in_proj_weight"))?;
        let b = weights.get(&format!("{name}.in_proj_bias"))?;
        let (q, k, v) = if let Some(cross) = cross {
            let q = query.f_linear(&w.f_narrow(0, 0, dim)?, Some(&b.f_narrow(0, 0, dim)?))?;
            let kv = cross.f_linear(
                &w.f_narrow(0, dim, dim * 2)?,
                Some(&b.f_narrow(0, dim, dim * 2)?),
            )?;
            (q, kv.f_narrow(2, 0, dim)?, kv.f_narrow(2, dim, dim)?)
        } else {
            let qkv = query.f_linear(w, Some(b))?;
            (
                qkv.f_narrow(2, 0, dim)?,
                qkv.f_narrow(2, dim, dim)?,
                qkv.f_narrow(2, 2 * dim, dim)?,
            )
        };
        let heads = |x: Tensor| -> Result<Tensor> {
            Ok(x.f_reshape([batch, -1, self.heads, dim / self.heads])?
                .f_transpose(1, 2)?)
        };
        let (q, k, v) = (heads(q)?, heads(k)?, heads(v)?);
        let result = Tensor::f_scaled_dot_product_attention(
            &q,
            &k,
            &v,
            None::<&Tensor>,
            0.0,
            false,
            None,
            false,
        )?
        .f_transpose(1, 2)?
        .f_reshape([batch, steps, dim])?;
        weights.linear(&result, &format!("{name}.out_proj"))
    }

    fn ff(&self, weights: &Weights, name: &str, x: &Tensor) -> Result<Tensor> {
        let y = weights.linear(x, &format!("{name}.linear1"))?;
        let y = if self.gelu {
            y.f_gelu("none")?
        } else {
            y.f_relu()?
        };
        weights.linear(&y, &format!("{name}.linear2"))
    }

    fn scale(&self, weights: &Weights, name: &str, key: &str, x: Tensor) -> Result<Tensor> {
        let Ok(weight) = weights.get(&format!("{name}.{key}.scale")) else {
            return Ok(x);
        };
        Ok(x.f_mul(weight)?)
    }

    fn norm(&self, weights: &Weights, name: &str, x: &Tensor, groups: i64) -> Result<Tensor> {
        if !weights.contains(&format!("{name}.weight")) {
            return Ok(x.shallow_clone());
        }
        if groups > 0 {
            let y = group_norm(weights, name, &x.f_transpose(1, 2)?, groups)?;
            Ok(y.f_transpose(1, 2)?)
        } else {
            weights.norm(x, name, 1e-5)
        }
    }

    fn position_2d(&self, freqs: i64, frames: i64, dim: i64, device: Device) -> Result<Tensor> {
        let cpu = (Kind::Float, Device::Cpu);
        let half = dim / 2;
        let decay = Tensor::f_arange(half / 2, cpu)?
            .f_mul_scalar(-2.0 * self.period.ln() / half as f64)?
            .f_exp()?;
        let positions =
            |n| -> Result<Tensor> { Ok(Tensor::f_arange(n, cpu)?.f_unsqueeze(-1)?.f_mul(&decay)?) };
        let horiz = positions(frames)?;
        let vert = positions(freqs)?;
        let horiz = Tensor::f_stack(&[horiz.f_sin()?, horiz.f_cos()?], -1)?
            .f_reshape([frames, 1, half])?
            .f_expand([-1, freqs, -1], false)?;
        let vert = Tensor::f_stack(&[vert.f_sin()?, vert.f_cos()?], -1)?
            .f_reshape([1, freqs, half])?
            .f_expand([frames, -1, -1], false)?;
        Ok(Tensor::f_cat(&[horiz, vert], -1)?
            .f_reshape([1, frames * freqs, dim])?
            .f_to_device(device)?)
    }

    fn position_1d(
        &self,
        weights: &Weights,
        frames: i64,
        dim: i64,
        device: Device,
    ) -> Result<Tensor> {
        if matches!(self.embedding, Embedding::Scaled) {
            return Ok(weights
                .get(&format!("{PREFIX}.position_embeddings.embedding.weight"))?
                .f_narrow(0, 0, frames)?
                .f_unsqueeze(0)?
                .f_mul_scalar(3.0)?);
        }
        self.sinusoidal_position_1d(frames, dim, device)
    }

    fn sinusoidal_position_1d(&self, frames: i64, dim: i64, device: Device) -> Result<Tensor> {
        let half = dim / 2;
        let options = (Kind::Float, device);
        let denominator = Tensor::f_arange(half, options)?
            .f_mul_scalar(self.period.ln() / (half - 1) as f64)?
            .f_exp()?;
        let shift = match self.embedding {
            Embedding::Sin { random_shift: 0 } => 0.0,
            Embedding::Sin { random_shift } => rand::rng().random_range(0..=random_shift) as f64,
            Embedding::Cape { center: true } => -(frames - 1) as f64 / 2.0,
            _ => 0.0,
        };
        let phase = Tensor::f_arange(frames, options)?
            .f_add_scalar(shift)?
            .f_unsqueeze(-1)?
            .f_div(&denominator)?;
        Ok(Tensor::f_cat(&[phase.f_cos()?, phase.f_sin()?], -1)?.f_unsqueeze(0)?)
    }
}

fn flag(value: &Value, key: &str, default: bool) -> bool {
    value[key]
        .as_bool()
        .or_else(|| value[key].as_i64().map(|n| n != 0))
        .unwrap_or(default)
}

fn groups(value: &Value, default: bool) -> i64 {
    value
        .as_i64()
        .unwrap_or_else(|| i64::from(value.as_bool().unwrap_or(default)))
}

fn groups_for_input(config: &Value) -> i64 {
    groups(&config["t_norm_in_group"], false)
}
