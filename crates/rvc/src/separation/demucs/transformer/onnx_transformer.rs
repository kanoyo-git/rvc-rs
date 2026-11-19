use anyhow::{Result, ensure};
use tch::{Device, Kind, Tensor};

use super::{Embedding, PREFIX, Transformer};
use crate::model::onnx::graph::Attribute;
use crate::separation::demucs::onnx_backend::{Export, Feature};
use crate::separation::onnx_layers::{concat, concat_strings, linear, reshape, slice, transpose};

impl Transformer {
    pub(in crate::separation::demucs) fn export_onnx(
        &self,
        model: &mut Export<'_>,
        spectrum: &Feature,
        wave: &Feature,
    ) -> Result<(Feature, Feature)> {
        let [channels, bins, frames] = <[i64; 3]>::try_from(spectrum.shape.as_slice())?;
        let mut time = wave.clone();
        let mut frequency = spectrum.clone();
        if self.bottleneck != 0 {
            time = model.conv("channel_upsampler_t", &time, &[1], &[0; 2], 1, false)?;
            let flat = model.reshape(&frequency, &[channels, bins * frames]);
            let flat = model.conv("channel_upsampler", &flat, &[1], &[0; 2], 1, false)?;
            frequency = model.reshape(&flat, &[self.bottleneck, bins, frames]);
        }
        let dim = frequency.shape[0];
        let samples = time.shape[1];
        ensure!(
            dim >= 4 && dim % 4 == 0 && dim % self.heads == 0 && time.shape[0] == dim,
            "HTDemucs transformer dimensions do not match its heads"
        );
        let x = transpose(&mut model.graph, &frequency.value, &[0, 3, 2, 1]);
        let x = reshape(&mut model.graph, &x, &[0, frames * bins, dim]);
        let mut x = Feature::new(x, [frames * bins, dim]);
        let y = transpose(&mut model.graph, &time.value, &[0, 2, 1]);
        let mut y = Feature::new(y, [samples, dim]);
        x = norm(model, &format!("{PREFIX}.norm_in"), &x, self.norm_in_groups)?;
        y = norm(
            model,
            &format!("{PREFIX}.norm_in_t"),
            &y,
            self.norm_in_groups,
        )?;
        let pos = self
            .position_2d(bins, frames, dim, Device::Cpu)?
            .f_mul_scalar(self.position_scale)?;
        let pos = model.graph.tensor(&pos)?;
        let time_pos = self.time_position(model, samples, dim)?;
        x.value = model.graph.op("Add", &[&x.value, &pos], vec![]);
        y.value = model.graph.op("Add", &[&y.value, &time_pos], vec![]);

        for layer in 0..self.layers {
            let cross = (layer % 2 == 1) != self.cross_first;
            let next = self.export_block(
                model,
                &format!("{PREFIX}.layers.{layer}"),
                &x,
                cross.then_some(&y),
            )?;
            y = self.export_block(
                model,
                &format!("{PREFIX}.layers_t.{layer}"),
                &y,
                cross.then_some(&x),
            )?;
            x = next;
        }
        let x = reshape(&mut model.graph, &x.value, &[0, frames, bins, dim]);
        let x = transpose(&mut model.graph, &x, &[0, 3, 2, 1]);
        let mut x = Feature::new(x, [dim, bins, frames]);
        let y = transpose(&mut model.graph, &y.value, &[0, 2, 1]);
        let mut y = Feature::new(y, [dim, samples]);
        if self.bottleneck != 0 {
            let flat = model.reshape(&x, &[dim, bins * frames]);
            let flat = model.conv("channel_downsampler", &flat, &[1], &[0; 2], 1, false)?;
            x = model.reshape(&flat, &[channels, bins, frames]);
            y = model.conv("channel_downsampler_t", &y, &[1], &[0; 2], 1, false)?;
        }
        Ok((x, y))
    }

    fn time_position(&self, model: &mut Export<'_>, length: i64, dim: i64) -> Result<String> {
        if let Embedding::Sin { random_shift } = self.embedding
            && random_shift > 0
        {
            let graph = &mut model.graph;
            let options = (Kind::Float, Device::Cpu);
            let half = dim / 2;
            let decay = Tensor::f_arange(half, options)?
                .f_mul_scalar(self.period.ln() / (half - 1) as f64)?
                .f_exp()?;
            let decay = graph.tensor(&decay)?;
            let step =
                graph.tensor(&Tensor::f_arange(length, options)?.f_reshape([1, length, 1])?)?;
            let shift = graph.op(
                "RandomUniform",
                &[],
                vec![
                    Attribute::ints("shape", &[1]),
                    Attribute::float("low", 0.),
                    Attribute::float("high", (random_shift + 1) as f32),
                ],
            );
            let shift = graph.op("Floor", &[&shift], vec![]);
            let phase = graph.op("Add", &[&step, &shift], vec![]);
            let phase = graph.op("Div", &[&phase, &decay], vec![]);
            let cos = graph.op("Cos", &[&phase], vec![]);
            let sin = graph.op("Sin", &[&phase], vec![]);
            let position = concat(graph, &[&cos, &sin], -1);
            let scale = graph.scale(self.position_scale as f32);
            return Ok(graph.op("Mul", &[&position, &scale], vec![]));
        }
        let pos = self
            .position_1d(model.weights, length, dim, Device::Cpu)?
            .f_mul_scalar(self.position_scale)?;
        model.graph.tensor(&pos)
    }

    fn export_block(
        &self,
        model: &mut Export<'_>,
        key: &str,
        q: &Feature,
        cross: Option<&Feature>,
    ) -> Result<Feature> {
        let first = format!("{key}.norm1");
        let second = format!("{key}.norm2");
        let attend = format!(
            "{key}.{}",
            if cross.is_some() {
                "cross_attn"
            } else {
                "self_attn"
            }
        );
        let mut value;
        if self.pre_norm {
            let query = norm(model, &first, q, self.groups)?;
            let cond = cross
                .map(|key| norm(model, &second, key, self.groups))
                .transpose()?;
            let y = attention(model, &attend, self.heads, &query, cond.as_ref())?;
            value = residual(model, key, "gamma_1", q, y)?;
            let method = if cross.is_some() {
                format!("{key}.norm3")
            } else {
                second
            };
            let source = norm(model, &method, &value, self.groups)?;
            let y = self.ff_onnx(model, key, &source.value)?;
            value = residual(model, key, "gamma_2", &value, y)?;
            if self.output_groups > 0 {
                value = norm(
                    model,
                    &format!("{key}.norm_out"),
                    &value,
                    self.output_groups,
                )?;
            }
        } else {
            let y = attention(model, &attend, self.heads, q, cross)?;
            value = residual(model, key, "gamma_1", q, y)?;
            value = norm(model, &first, &value, self.groups)?;
            let y = self.ff_onnx(model, key, &value.value)?;
            value = residual(model, key, "gamma_2", &value, y)?;
            value = norm(model, &second, &value, self.groups)?;
        }
        Ok(value)
    }

    fn ff_onnx(&self, model: &mut Export<'_>, key: &str, input: &str) -> Result<String> {
        let y = project(model, &format!("{key}.linear1"), input)?;
        let y = model
            .graph
            .op(if self.gelu { "Gelu" } else { "Relu" }, &[&y], vec![]);
        project(model, &format!("{key}.linear2"), &y)
    }
}

fn attention(
    model: &mut Export<'_>,
    key: &str,
    heads: i64,
    query: &Feature,
    cross: Option<&Feature>,
) -> Result<String> {
    let graph = &mut model.graph;
    let dim = query.shape[1];
    let weight = model.weights.get(&format!("{key}.in_proj_weight"))?;
    let bias = model.weights.get(&format!("{key}.in_proj_bias"))?;
    let (q, k, v) = if let Some(cross) = cross {
        let q = linear(
            graph,
            &query.value,
            &weight.f_narrow(0, 0, dim)?,
            Some(&bias.f_narrow(0, 0, dim)?),
        )?;
        let kv = linear(
            graph,
            &cross.value,
            &weight.f_narrow(0, dim, 2 * dim)?,
            Some(&bias.f_narrow(0, dim, 2 * dim)?),
        )?;
        (
            q,
            slice(graph, &kv, -1, 0, dim, 1),
            slice(graph, &kv, -1, dim, 2 * dim, 1),
        )
    } else {
        let qkv = linear(graph, &query.value, weight, Some(bias))?;
        (
            slice(graph, &qkv, -1, 0, dim, 1),
            slice(graph, &qkv, -1, dim, 2 * dim, 1),
            slice(graph, &qkv, -1, 2 * dim, 3 * dim, 1),
        )
    };
    let depth = dim / heads;
    let q = reshape(graph, &q, &[0, 0, heads, depth]);
    let q = transpose(graph, &q, &[0, 2, 1, 3]);
    let k = reshape(graph, &k, &[0, 0, heads, depth]);
    let k = transpose(graph, &k, &[0, 2, 3, 1]);
    let v = reshape(graph, &v, &[0, 0, heads, depth]);
    let v = transpose(graph, &v, &[0, 2, 1, 3]);
    let scale = graph.scale((depth as f32).sqrt().recip());
    // A full score matrix grows quadratically with the window. Independent
    // query blocks preserve the softmax over all keys and bound scratch space.
    const QUERY_BLOCK: i64 = 384;
    let steps = query.shape[0];
    let mut blocks = Vec::new();
    for start in (0..steps).step_by(QUERY_BLOCK as usize) {
        let end = (start + QUERY_BLOCK).min(steps);
        let q = if steps <= QUERY_BLOCK {
            q.clone()
        } else {
            slice(graph, &q, 2, start, end, 1)
        };
        let score = graph.op("MatMul", &[&q, &k], vec![]);
        let score = graph.op("Mul", &[&score, &scale], vec![]);
        let probability = graph.op("Softmax", &[&score], vec![Attribute::int("axis", -1)]);
        blocks.push(graph.op("MatMul", &[&probability, &v], vec![]));
    }
    let result = if blocks.len() == 1 {
        blocks.remove(0)
    } else {
        concat_strings(graph, &blocks, 2)
    };
    let result = transpose(graph, &result, &[0, 2, 1, 3]);
    let result = reshape(graph, &result, &[0, 0, dim]);
    project(model, &format!("{key}.out_proj"), &result)
}

fn project(model: &mut Export<'_>, name: &str, value: &str) -> Result<String> {
    linear(
        &mut model.graph,
        value,
        model.weights.get(&format!("{name}.weight"))?,
        model.weights.get(&format!("{name}.bias")).ok(),
    )
}

fn residual(
    model: &mut Export<'_>,
    root: &str,
    name: &str,
    source: &Feature,
    value: String,
) -> Result<Feature> {
    let graph = &mut model.graph;
    let value = if let Ok(scale) = model.weights.get(&format!("{root}.{name}.scale")) {
        let scale = graph.tensor(scale)?;
        graph.op("Mul", &[&value, &scale], vec![])
    } else {
        value
    };
    Ok(source.with_value(graph.op("Add", &[&source.value, &value], vec![])))
}

fn norm(model: &mut Export<'_>, name: &str, value: &Feature, groups: i64) -> Result<Feature> {
    let Ok(weight) = model.weights.get(&format!("{name}.weight")) else {
        return Ok(value.clone());
    };
    if groups > 0 {
        let x = transpose(&mut model.graph, &value.value, &[0, 2, 1]);
        let x = Feature::new(x, [value.shape[1], value.shape[0]]);
        let norm = model.group_norm(name, &x, groups)?;
        let out = transpose(&mut model.graph, &norm.value, &[0, 2, 1]);
        return Ok(value.with_value(out));
    }
    let gamma = model.graph.tensor(weight)?;
    let beta = model
        .graph
        .tensor(model.weights.get(&format!("{name}.bias"))?)?;
    let result = model.graph.op(
        "LayerNormalization",
        &[&value.value, &gamma, &beta],
        vec![Attribute::float("epsilon", 1e-5)],
    );
    Ok(value.with_value(result))
}
