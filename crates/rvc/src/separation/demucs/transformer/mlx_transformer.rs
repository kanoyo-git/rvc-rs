use anyhow::{Result, ensure};
use tch::Device;

use super::super::super::{mlx::Array, mlx_layers::gelu};
use super::super::mlx_backend::Parameters;
use super::{PREFIX, Transformer};

impl Transformer {
    pub(in crate::separation::demucs) fn forward_mlx(
        &self,
        parameters: &Parameters,
        mut spectral: Array,
        mut wave: Array,
    ) -> Result<(Array, Array)> {
        let stream = &parameters.stream;
        let [batch, channels, frequencies, steps] = <[i32; 4]>::try_from(spectral.shape())?;
        let bottleneck = i32::try_from(self.bottleneck)?;
        if bottleneck != 0 {
            wave = parameters.conv1d(&wave, "channel_upsampler_t", 1, 0, 1)?;
            spectral = stream.reshape(&spectral, &[batch, channels, frequencies * steps])?;
            spectral = parameters.conv1d(&spectral, "channel_upsampler", 1, 0, 1)?;
            spectral = stream.reshape(&spectral, &[batch, bottleneck, frequencies, steps])?;
        }
        let dim = spectral.shape()[1];
        let heads = i32::try_from(self.heads)?;
        ensure!(
            dim >= 4 && dim % 4 == 0 && dim % heads == 0 && wave.shape()[1] == dim,
            "Demucs transformer dimensions do not match its attention heads"
        );
        let position = self.position_2d(
            i64::from(frequencies),
            i64::from(steps),
            i64::from(dim),
            Device::Cpu,
        )?;
        let position = stream.upload(&position, parameters.dtype)?;
        let time =
            self.sinusoidal_position_1d(i64::from(wave.shape()[2]), i64::from(dim), Device::Cpu)?;
        let time = stream.upload(&time, parameters.dtype)?;
        let scale = stream.scalar(self.position_scale as f32, parameters.dtype)?;
        spectral = stream.transpose(&spectral, &[0, 3, 2, 1])?;
        spectral = stream.reshape(&spectral, &[batch, steps * frequencies, dim])?;
        spectral = self.norm_mlx(
            parameters,
            &format!("{PREFIX}.norm_in"),
            &spectral,
            self.norm_in_groups,
        )?;
        spectral = stream.add(&spectral, &stream.mul(&position, &scale)?)?;
        wave = stream.transpose(&wave, &[0, 2, 1])?;
        wave = self.norm_mlx(
            parameters,
            &format!("{PREFIX}.norm_in_t"),
            &wave,
            self.norm_in_groups,
        )?;
        wave = stream.add(&wave, &stream.mul(&time, &scale)?)?;
        for index in 0..self.layers {
            let cross = (index % 2 == 1) != self.cross_first;
            let next = self.block_mlx(
                parameters,
                &format!("{PREFIX}.layers.{index}"),
                &spectral,
                cross.then_some(&wave),
            )?;
            wave = self.block_mlx(
                parameters,
                &format!("{PREFIX}.layers_t.{index}"),
                &wave,
                cross.then_some(&spectral),
            )?;
            spectral = next;
        }
        spectral = stream.reshape(&spectral, &[batch, steps, frequencies, dim])?;
        spectral = stream.transpose(&spectral, &[0, 3, 2, 1])?;
        wave = stream.transpose(&wave, &[0, 2, 1])?;
        if bottleneck != 0 {
            spectral = stream.reshape(&spectral, &[batch, dim, frequencies * steps])?;
            spectral = parameters.conv1d(&spectral, "channel_downsampler", 1, 0, 1)?;
            spectral = stream.reshape(&spectral, &[batch, channels, frequencies, steps])?;
            wave = parameters.conv1d(&wave, "channel_downsampler_t", 1, 0, 1)?;
        }
        Ok((spectral, wave))
    }

    fn norm_mlx(
        &self,
        parameters: &Parameters,
        name: &str,
        input: &Array,
        groups: i64,
    ) -> Result<Array> {
        if groups > 0 {
            let stream = &parameters.stream;
            let input = stream.transpose(input, &[0, 2, 1])?;
            let input = parameters.normalize_layer(name, &input, groups)?;
            stream.transpose(&input, &[0, 2, 1])
        } else {
            parameters.normalize_layer(name, input, 0)
        }
    }

    fn block_mlx(
        &self,
        parameters: &Parameters,
        name: &str,
        query: &Array,
        cross: Option<&Array>,
    ) -> Result<Array> {
        let stream = &parameters.stream;
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
        let scale = |value: Array, key| {
            if let Some(weight) = parameters.optional(&format!("{name}.{key}.scale")) {
                stream.mul(&value, weight)
            } else {
                Ok(value)
            }
        };
        if self.pre_norm {
            let normalized = self.norm_mlx(parameters, &first, query, self.groups)?;
            let keys = cross
                .map(|key| self.norm_mlx(parameters, &second, key, self.groups))
                .transpose()?;
            let connection = self.attend_mlx(parameters, &attention, &normalized, keys.as_ref())?;
            let value = stream.add(query, &scale(connection, "gamma_1")?)?;
            let next = if cross.is_some() {
                format!("{name}.norm3")
            } else {
                second
            };
            let hidden = self.norm_mlx(parameters, &next, &value, self.groups)?;
            let next = self.feed_forward_mlx(parameters, name, &hidden)?;
            let value = stream.add(&value, &scale(next, "gamma_2")?)?;
            if self.output_groups > 0 {
                self.norm_mlx(
                    parameters,
                    &format!("{name}.norm_out"),
                    &value,
                    self.output_groups,
                )
            } else {
                Ok(value)
            }
        } else {
            let connection = self.attend_mlx(parameters, &attention, query, cross)?;
            let value = stream.add(query, &scale(connection, "gamma_1")?)?;
            let value = self.norm_mlx(parameters, &first, &value, self.groups)?;
            let next = self.feed_forward_mlx(parameters, name, &value)?;
            let value = stream.add(&value, &scale(next, "gamma_2")?)?;
            self.norm_mlx(parameters, &second, &value, self.groups)
        }
    }

    fn attend_mlx(
        &self,
        parameters: &Parameters,
        name: &str,
        query: &Array,
        cross: Option<&Array>,
    ) -> Result<Array> {
        let stream = &parameters.stream;
        let [batch, steps, dim] = <[i32; 3]>::try_from(query.shape())?;
        let weight = parameters.get(&format!("{name}.in_proj_weight"))?;
        let bias = parameters.get(&format!("{name}.in_proj_bias"))?;
        let (q, k, v) = if let Some(context) = cross {
            let query_weight = stream.slice(weight, 0, 0, dim, 1)?;
            let query_bias = stream.slice(bias, 0, 0, dim, 1)?;
            let context_weight = stream.slice(weight, 0, dim, 3 * dim, 1)?;
            let context_bias = stream.slice(bias, 0, dim, 3 * dim, 1)?;
            let q = parameters.linear_parameters(query, &query_weight, Some(&query_bias))?;
            let kv = parameters.linear_parameters(context, &context_weight, Some(&context_bias))?;
            (
                q,
                stream.slice(&kv, 2, 0, dim, 1)?,
                stream.slice(&kv, 2, dim, 2 * dim, 1)?,
            )
        } else {
            let qkv = parameters.linear_parameters(query, weight, Some(bias))?;
            (
                stream.slice(&qkv, 2, 0, dim, 1)?,
                stream.slice(&qkv, 2, dim, 2 * dim, 1)?,
                stream.slice(&qkv, 2, 2 * dim, 3 * dim, 1)?,
            )
        };
        let heads = i32::try_from(self.heads)?;
        let head_dim = dim / heads;
        let split = |value: &Array| {
            let view = stream.reshape(value, &[batch, -1, heads, head_dim])?;
            stream.transpose(&view, &[0, 2, 1, 3])
        };
        let result = stream.attention(
            &split(&q)?,
            &split(&k)?,
            &split(&v)?,
            (head_dim as f32).sqrt().recip(),
        )?;
        let result = stream.transpose(&result, &[0, 2, 1, 3])?;
        let result = stream.reshape(&result, &[batch, steps, dim])?;
        parameters.linear(&result, &format!("{name}.out_proj"))
    }

    fn feed_forward_mlx(
        &self,
        parameters: &Parameters,
        name: &str,
        input: &Array,
    ) -> Result<Array> {
        let stream = &parameters.stream;
        let value = parameters.linear(input, &format!("{name}.linear1"))?;
        let value = if self.gelu {
            gelu(stream, &value, parameters.dtype)?
        } else {
            stream.max(&value, &stream.scalar(0.0, parameters.dtype)?)?
        };
        parameters.linear(&value, &format!("{name}.linear2"))
    }
}
