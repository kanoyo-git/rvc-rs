use anyhow::{Result, ensure};
use tch::Tensor;

use super::mlx::{Array, DType, Stream};

pub(super) enum Layout {
    Channels,
    Linear,
    Conv1d,
    Conv2d,
    Transpose2d,
}

pub(super) struct Affine {
    weight: Array,
    bias: Array,
    dtype: DType,
}

impl Affine {
    pub(super) fn load(
        stream: &Stream,
        weight: &Tensor,
        bias: &Tensor,
        layout: Layout,
        dtype: DType,
    ) -> Result<Self> {
        let weight = match layout {
            Layout::Channels => weight.shallow_clone(),
            Layout::Linear => weight.f_transpose(0, 1)?,
            Layout::Conv1d => weight.f_permute([0, 2, 1])?,
            Layout::Conv2d => weight.f_permute([0, 2, 3, 1])?,
            Layout::Transpose2d => weight.f_permute([1, 2, 3, 0])?,
        };
        Ok(Self {
            weight: stream.upload(&weight, dtype)?,
            bias: stream.upload(bias, dtype)?,
            dtype,
        })
    }

    pub(super) fn channels(&self) -> i32 {
        self.bias.shape()[0]
    }

    pub(super) fn linear(&self, stream: &Stream, input: &Array) -> Result<Array> {
        stream.add(&stream.matmul(input, &self.weight)?, &self.bias)
    }

    pub(super) fn conv1d(
        &self,
        stream: &Stream,
        input: &Array,
        pad: i32,
        groups: i32,
    ) -> Result<Array> {
        let input = stream.transpose(input, &[0, 2, 1])?;
        let value = stream.conv1d(&input, &self.weight, pad, groups)?;
        let value = stream.add(&value, &self.bias)?;
        stream.transpose(&value, &[0, 2, 1])
    }

    pub(super) fn conv2d(
        &self,
        stream: &Stream,
        input: &Array,
        stride: [i32; 2],
        pad: [i32; 2],
    ) -> Result<Array> {
        let input = stream.transpose(input, &[0, 2, 3, 1])?;
        let value = stream.conv2d(&input, &self.weight, stride, pad)?;
        let value = stream.add(&value, &self.bias)?;
        stream.transpose(&value, &[0, 3, 1, 2])
    }

    pub(super) fn upsample2d(
        &self,
        stream: &Stream,
        input: &Array,
        stride: [i32; 2],
    ) -> Result<Array> {
        let input = stream.transpose(input, &[0, 2, 3, 1])?;
        let value = stream.upsample2d(&input, &self.weight, stride)?;
        let value = stream.add(&value, &self.bias)?;
        stream.transpose(&value, &[0, 3, 1, 2])
    }

    pub(super) fn group_norm_one(&self, stream: &Stream, input: &Array) -> Result<Array> {
        let axes: Vec<_> = (1..input.shape().len() as i32).collect();
        let mut shape = vec![1; input.shape().len()];
        shape[1] = self.channels();
        let normalized = standardize(stream, input, &axes, 1e-5)?;
        let normalized = stream.cast(&normalized, self.dtype)?;
        let scale = stream.reshape(&self.weight, &shape)?;
        let bias = stream.reshape(&self.bias, &shape)?;
        stream.add(&stream.mul(&normalized, &scale)?, &bias)
    }

    pub(super) fn layer_norm(&self, stream: &Stream, input: &Array) -> Result<Array> {
        let value = standardize(stream, input, &[-1], 1e-5)?;
        let value = stream.cast(&value, self.dtype)?;
        stream.add(&stream.mul(&value, &self.weight)?, &self.bias)
    }
}

pub(super) fn standardize(stream: &Stream, input: &Array, axes: &[i32], eps: f32) -> Result<Array> {
    let input = stream.cast(input, DType::F32)?;
    let mean = stable_mean(stream, &input, axes)?;
    let centered = stream.sub(&input, &mean)?;
    let variance = stable_mean(stream, &stream.square(&centered)?, axes)?;
    let variance = stream.add(&variance, &stream.scalar(eps, DType::F32)?)?;
    stream.mul(&centered, &stream.rsqrt(&variance)?)
}

pub(super) fn stable_mean(stream: &Stream, input: &Array, axes: &[i32]) -> Result<Array> {
    let shape = input.shape();
    let rank = i32::try_from(shape.len())?;
    let Some(&axis) = axes.first() else {
        return stream.mean_keep(input, axes);
    };
    let first = if axis < 0 { rank + axis } else { axis };
    if first < 0
        || first >= rank
        || !axes
            .iter()
            .map(|&axis| axis.rem_euclid(rank))
            .eq(first..rank)
    {
        return stream.mean_keep(input, axes);
    }
    let first = first as usize;
    let length = shape[first..]
        .iter()
        .try_fold(1_i32, |n, &k| n.checked_mul(k));
    let Some(count) = length.filter(|&count| count > 4096) else {
        return stream.mean_keep(input, axes);
    };

    // Bound float32 accumulation error on long audio and large feature maps.
    const BLOCK: i32 = 128;
    let mut flat = shape[..first].to_vec();
    flat.push(count);
    let mut value = stream.reshape(input, &flat)?;
    let mut size = count;
    while size > BLOCK {
        let groups = size / BLOCK;
        let split = groups * BLOCK;
        let mut chunked = flat[..first].to_vec();
        chunked.extend([groups, BLOCK]);
        let main = if split == size {
            stream.reshape(&value, &chunked)?
        } else {
            stream.reshape(&stream.slice(&value, first, 0, split, 1)?, &chunked)?
        };
        flat[first] = groups;
        let partial = stream.reshape(&stream.sum_keep(&main, -1)?, &flat)?;
        if split < size {
            let tail = stream.slice(&value, first, split, size, 1)?;
            let tail = stream.sum_keep(&tail, -1)?;
            value = stream.concat(&[partial, tail], i32::try_from(first)?)?;
            flat[first] += 1;
        } else {
            value = partial;
        }
        size = flat[first];
    }
    let sum = stream.sum_keep(&value, -1)?;
    let average = stream.div(&sum, &stream.scalar(count as f32, DType::F32)?)?;
    let mut output = shape.to_vec();
    output[first..].fill(1);
    stream.reshape(&average, &output)
}

pub(super) fn gelu(stream: &Stream, input: &Array, dtype: DType) -> Result<Array> {
    let input = stream.cast(input, DType::F32)?;
    let scale = stream.scalar(std::f32::consts::FRAC_1_SQRT_2, DType::F32)?;
    let value = stream.erf(&stream.mul(&input, &scale)?)?;
    let shifted = stream.add(&value, &stream.scalar(1.0, DType::F32)?)?;
    let half = stream.mul(&input, &stream.scalar(0.5, DType::F32)?)?;
    stream.cast(&stream.mul(&half, &shifted)?, dtype)
}

pub(super) fn glu(stream: &Stream, input: &Array, axis: usize) -> Result<Array> {
    let length = input.shape()[axis];
    let left = stream.slice(input, axis, 0, length / 2, 1)?;
    let right = stream.slice(input, axis, length / 2, length, 1)?;
    stream.mul(&left, &stream.sigmoid(&right)?)
}

pub(super) fn silu(stream: &Stream, input: &Array) -> Result<Array> {
    stream.mul(input, &stream.sigmoid(input)?)
}

pub(super) fn resize_bilinear(
    stream: &Stream,
    input: Array,
    size: [i32; 2],
    dtype: DType,
    align_corners: bool,
) -> Result<Array> {
    let input = interpolate_axis(stream, input, 2, size[0], dtype, align_corners)?;
    interpolate_axis(stream, input, 3, size[1], dtype, align_corners)
}

fn interpolate_axis(
    stream: &Stream,
    input: Array,
    axis: usize,
    output: i32,
    dtype: DType,
    align_corners: bool,
) -> Result<Array> {
    let size = input.shape()[axis];
    if size == output {
        return Ok(input);
    }
    let mut shape = input.shape().to_vec();
    shape[axis] = output;
    if size == 1 {
        return stream.broadcast(&input, &shape);
    }
    let (ratio, offset) = if align_corners {
        (f64::from(size - 1) / f64::from((output - 1).max(1)), 0.0)
    } else {
        (f64::from(size) / f64::from(output), 0.5)
    };
    let positions = (0..output)
        .map(|i| ((f64::from(i) + offset) * ratio - offset).clamp(0.0, f64::from(size - 1)));
    let starts = positions
        .clone()
        .map(|position| position.floor() as i32)
        .collect::<Vec<_>>();
    let ends = starts
        .iter()
        .map(|&i| (i + 1).min(size - 1))
        .collect::<Vec<_>>();
    let fractions = positions
        .map(|position| position.fract() as f32)
        .collect::<Vec<_>>();
    let start = stream.indices(&starts, &[output])?;
    let stop = stream.indices(&ends, &[output])?;
    let left = stream.cast(&stream.take(&input, &start, axis as i32)?, DType::F32)?;
    let right = stream.cast(&stream.take(&input, &stop, axis as i32)?, DType::F32)?;
    let mut bands = vec![1; shape.len()];
    bands[axis] = output;
    let fraction = stream.float32(&fractions, &bands)?;
    let other = stream.sub(&stream.scalar(1., DType::F32)?, &fraction)?;
    let blended = stream.add(&stream.mul(&left, &other)?, &stream.mul(&right, &fraction)?)?;
    stream.cast(&blended, dtype)
}

#[derive(Clone, Copy)]
pub(super) enum Cell {
    Lstm,
    Gru,
    Tanh,
}

pub(super) struct Rnn {
    forward: Direction,
    backward: Option<Direction>,
}

impl Rnn {
    pub(super) fn load(
        stream: &Stream,
        params: &[Tensor],
        hidden: i64,
        dtype: DType,
        cell: Cell,
    ) -> Result<Self> {
        ensure!(matches!(params.len(), 4 | 8), "invalid MLX RNN parameters");
        let forward = Direction::load(stream, &params[..4], hidden, dtype, cell)?;
        let backward = (params.len() == 8)
            .then(|| Direction::load(stream, &params[4..], hidden, dtype, cell))
            .transpose()?;
        Ok(Self { forward, backward })
    }

    pub(super) fn run(&self, stream: &Stream, input: &Array) -> Result<Array> {
        let forward = self.forward.run(stream, input, false)?;
        if let Some(backward) = &self.backward {
            stream.concat(&[forward, backward.run(stream, input, true)?], 2)
        } else {
            Ok(forward)
        }
    }
}

struct Direction {
    input: Affine,
    recurrent: Affine,
    hidden: i32,
    cell: Cell,
}

impl Direction {
    fn load(
        stream: &Stream,
        params: &[Tensor],
        hidden: i64,
        dtype: DType,
        cell: Cell,
    ) -> Result<Self> {
        Ok(Self {
            input: Affine::load(stream, &params[0], &params[2], Layout::Linear, dtype)?,
            recurrent: Affine::load(stream, &params[1], &params[3], Layout::Linear, dtype)?,
            hidden: i32::try_from(hidden)?,
            cell,
        })
    }

    fn run(&self, stream: &Stream, input: &Array, reverse: bool) -> Result<Array> {
        let batch = input.shape()[0];
        let frames = input.shape()[1];
        let terms = self.input.linear(stream, input)?;
        let shape = [batch, self.hidden];
        let initial = stream.cast(&stream.zeros(&shape)?, self.input.dtype)?;
        let mut cell = stream.reshape(&initial, &shape)?;
        let mut outputs = Vec::with_capacity(frames as usize);
        for frame in 0..frames {
            let frame = if reverse { frames - frame - 1 } else { frame };
            let projected = stream.select(&terms, 1, frame)?;
            let previous = outputs.last().unwrap_or(&initial);
            let recurrent = self.recurrent.linear(stream, previous)?;
            let value = match self.cell {
                Cell::Lstm => {
                    let gates = stream.add(&projected, &recurrent)?;
                    let gates = stream.reshape(&gates, &[batch, 4, self.hidden])?;
                    let input = stream.sigmoid(&stream.select(&gates, 1, 0)?)?;
                    let forget = stream.sigmoid(&stream.select(&gates, 1, 1)?)?;
                    let candidate = stream.tanh(&stream.select(&gates, 1, 2)?)?;
                    let output = stream.sigmoid(&stream.select(&gates, 1, 3)?)?;
                    cell = stream.add(
                        &stream.mul(&cell, &forget)?,
                        &stream.mul(&candidate, &input)?,
                    )?;
                    stream.mul(&stream.tanh(&cell)?, &output)?
                }
                Cell::Gru => {
                    let shape = [batch, 3, self.hidden];
                    let projected = stream.reshape(&projected, &shape)?;
                    let recurrent = stream.reshape(&recurrent, &shape)?;
                    let gate = |slot| -> Result<Array> {
                        let left = stream.select(&projected, 1, slot)?;
                        let right = stream.select(&recurrent, 1, slot)?;
                        stream.sigmoid(&stream.add(&left, &right)?)
                    };
                    let reset = gate(0)?;
                    let update = gate(1)?;
                    let recurrent = stream.mul(&reset, &stream.select(&recurrent, 1, 2)?)?;
                    let hidden = stream.add(&stream.select(&projected, 1, 2)?, &recurrent)?;
                    let hidden = stream.tanh(&hidden)?;
                    let next = stream.mul(&update, &stream.sub(previous, &hidden)?)?;
                    stream.add(&hidden, &next)?
                }
                Cell::Tanh => stream.tanh(&stream.add(&projected, &recurrent)?)?,
            };
            outputs.push(value);
        }
        if reverse {
            outputs.reverse();
        }
        stream.stack(&outputs, 1)
    }
}
