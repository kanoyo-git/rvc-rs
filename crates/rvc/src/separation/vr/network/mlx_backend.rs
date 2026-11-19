use anyhow::{Result, ensure};
use tch::{Device, Tensor};

use super::super::super::{
    mlx::{Array, DType, Stream},
    mlx_layers::{Affine, Cell, Layout, Rnn, resize_bilinear},
};
use super::{Activation, TorchNetwork};

pub(super) struct Network {
    stream: Stream,
    dtype: DType,
    topology: Topology,
    output: Conv,
}

enum Topology {
    Legacy {
        low: Box<Subnetwork>,
        high: Box<Subnetwork>,
        bridge2: Box<Block>,
        bridge3: Box<Block>,
        stage2: Box<Subnetwork>,
        stage3: Box<Subnetwork>,
    },
    Modern {
        low1: Box<Subnetwork>,
        high1: Box<Subnetwork>,
        low2: Box<Subnetwork>,
        high2: Box<Subnetwork>,
        stage3: Box<Subnetwork>,
    },
}

impl Network {
    pub(super) fn load(torch: &TorchNetwork, device: Device, dtype: DType) -> Result<Self> {
        let stream = Stream::new(device)?;
        let subnet = |value| Subnetwork::load(&stream, value, dtype).map(Box::new);
        let topology = match &torch.topology {
            super::Topology::Legacy {
                low,
                high,
                bridge2,
                bridge3,
                stage2,
                stage3,
            } => Topology::Legacy {
                low: subnet(low)?,
                high: subnet(high)?,
                bridge2: Box::new(Block::load(&stream, bridge2, dtype)?),
                bridge3: Box::new(Block::load(&stream, bridge3, dtype)?),
                stage2: subnet(stage2)?,
                stage3: subnet(stage3)?,
            },
            super::Topology::Modern {
                low1,
                high1,
                low2,
                high2,
                stage3,
            } => Topology::Modern {
                low1: subnet(low1)?,
                high1: subnet(high1)?,
                low2: subnet(low2)?,
                high2: subnet(high2)?,
                stage3: subnet(stage3)?,
            },
        };
        let output = Conv::load(&stream, &torch.output, dtype)?;
        Ok(Self {
            stream,
            dtype,
            topology,
            output,
        })
    }

    pub(super) fn predict_mask(&self, input: &Tensor, bins: i64, offset: i64) -> Result<Tensor> {
        let stream = &self.stream;
        let input = stream.upload(input, self.dtype)?;
        let [batch, _, _, frames] = <[i32; 4]>::try_from(input.shape())?;
        let bins = i32::try_from(bins)?;
        let offset = i32::try_from(offset)?;
        let input = stream.slice(&input, 2, 0, bins, 1)?;
        let lower = stream.slice(&input, 2, 0, bins / 2, 1)?;
        let upper = stream.slice(&input, 2, bins / 2, bins, 1)?;
        let features = match &self.topology {
            Topology::Legacy {
                low,
                high,
                bridge2,
                bridge3,
                stage2,
                stage3,
            } => {
                let lower = low.run(stream, &lower)?;
                let upper = high.run(stream, &upper)?;
                let joined = stream.concat(&[lower, upper], 2)?;
                let stage2_input = stream.concat(
                    &[
                        stream.reshape(&input, input.shape())?,
                        stream.reshape(&joined, joined.shape())?,
                    ],
                    1,
                )?;
                let refine = stage2.run(stream, &bridge2.run(stream, &stage2_input)?)?;
                let refine = stream.concat(&[input, joined, refine], 1)?;
                stage3.run(stream, &bridge3.run(stream, &refine)?)?
            }
            Topology::Modern {
                low1,
                high1,
                low2,
                high2,
                stage3,
            } => {
                let lower1 = low1.run(stream, &lower)?;
                let upper1 = high1.run(stream, &upper)?;
                let joined = stream.concat(
                    &[
                        stream.reshape(&lower1, lower1.shape())?,
                        stream.reshape(&upper1, upper1.shape())?,
                    ],
                    2,
                )?;
                let lower2 = low2.run(stream, &stream.concat(&[lower, lower1], 1)?)?;
                let upper2 = high2.run(stream, &stream.concat(&[upper, upper1], 1)?)?;
                let refined = stream.concat(&[lower2, upper2], 2)?;
                stage3.run(stream, &stream.concat(&[input, joined, refined], 1)?)?
            }
        };
        let mask = stream.sigmoid(&self.output.run(stream, &features)?)?;
        ensure!(
            mask.shape() == [batch, 2, bins, frames],
            "invalid VR MLX mask shape"
        );
        let mask = stream.slice(&mask, 3, offset, frames - offset, 1)?;
        let pad = stream.slice(&mask, 2, bins - 1, bins, 1)?;
        let mask = stream.concat(&[mask, pad], 2)?;
        let values = stream.to_floats(&mask)?;
        let dimensions = [batch, 2, bins + 1, frames - 2 * offset].map(i64::from);
        Ok(Tensor::f_from_slice(&values)?.f_view(dimensions)?)
    }
}

struct Conv {
    weight: Array,
    bias: Option<Array>,
    stride: [i32; 2],
    padding: [i32; 2],
    dilation: [i32; 2],
    groups: i32,
}

impl Conv {
    fn load(stream: &Stream, source: &super::Conv, dtype: DType) -> Result<Self> {
        let pair = |[a, b]: [i64; 2]| -> Result<_> { Ok([i32::try_from(a)?, i32::try_from(b)?]) };
        Ok(Self {
            weight: stream.upload(&source.weight.f_permute([0, 2, 3, 1])?, dtype)?,
            bias: source
                .bias
                .as_ref()
                .map(|bias| stream.upload(bias, dtype))
                .transpose()?,
            stride: pair(source.geometry.stride)?,
            padding: pair(source.geometry.padding)?,
            dilation: pair(source.geometry.dilation)?,
            groups: i32::try_from(source.groups)?,
        })
    }

    fn run(&self, stream: &Stream, input: &Array) -> Result<Array> {
        let input = stream.transpose(input, &[0, 2, 3, 1])?;
        let output = stream.convolution2d(
            &input,
            &self.weight,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )?;
        let output = if let Some(bias) = &self.bias {
            stream.add(&output, bias)?
        } else {
            output
        };
        stream.transpose(&output, &[0, 3, 1, 2])
    }
}

struct Norm {
    scale: Array,
    shift: Array,
    dtype: DType,
}

impl Norm {
    fn load(stream: &Stream, source: &super::Norm, dtype: DType, shape: &[i64]) -> Result<Self> {
        let scale = source
            .weight
            .f_mul(&source.variance.f_add_scalar(1e-5)?.f_rsqrt()?)?;
        let shift = source.bias.f_sub(&source.mean.f_mul(&scale)?)?;
        Ok(Self {
            scale: stream.upload(&scale.f_reshape(shape)?, DType::F32)?,
            shift: stream.upload(&shift.f_reshape(shape)?, DType::F32)?,
            dtype,
        })
    }

    fn run(&self, stream: &Stream, input: &Array) -> Result<Array> {
        let input = stream.cast(input, DType::F32)?;
        let value = stream.add(&stream.mul(&input, &self.scale)?, &self.shift)?;
        stream.cast(&value, self.dtype)
    }
}

struct Block {
    depth: Option<Conv>,
    point: Conv,
    norm: Option<Norm>,
    act: Activation,
    zero: Array,
    slope: Array,
}

impl Block {
    fn load(stream: &Stream, source: &super::Block, dtype: DType) -> Result<Self> {
        Ok(Self {
            depth: source
                .depth
                .as_ref()
                .map(|depth| Conv::load(stream, depth, dtype))
                .transpose()?,
            point: Conv::load(stream, &source.point, dtype)?,
            norm: source
                .norm
                .as_ref()
                .map(|norm| Norm::load(stream, norm, dtype, &[1, -1, 1, 1]))
                .transpose()?,
            act: source.act,
            zero: stream.scalar(0.0, dtype)?,
            slope: stream.scalar(0.01, dtype)?,
        })
    }

    fn run(&self, stream: &Stream, input: &Array) -> Result<Array> {
        let depth = self
            .depth
            .as_ref()
            .map(|op| op.run(stream, input))
            .transpose()?;
        let value = self.point.run(stream, depth.as_ref().unwrap_or(input))?;
        let value = if let Some(norm) = &self.norm {
            norm.run(stream, &value)?
        } else {
            value
        };
        let positive = stream.max(&value, &self.zero)?;
        match self.act {
            Activation::Relu => Ok(positive),
            Activation::Leaky => {
                let negative = stream.min(&value, &self.zero)?;
                stream.add(&positive, &stream.mul(&negative, &self.slope)?)
            }
        }
    }
}

struct Pyramid {
    pool: Block,
    paths: Vec<Block>,
    merge: Block,
}

impl Pyramid {
    fn load(stream: &Stream, source: &super::Pyramid, dtype: DType) -> Result<Self> {
        Ok(Self {
            pool: Block::load(stream, &source.pool, dtype)?,
            paths: source
                .paths
                .iter()
                .map(|path| Block::load(stream, path, dtype))
                .collect::<Result<_>>()?,
            merge: Block::load(stream, &source.merge, dtype)?,
        })
    }

    fn run(&self, stream: &Stream, input: &Array, dtype: DType) -> Result<Array> {
        let [_, _, bins, frames] = <[i32; 4]>::try_from(input.shape())?;
        let pooled = self.pool.run(stream, &stream.mean_keep(input, &[2])?)?;
        let pooled = resize_bilinear(stream, pooled, [bins, frames], dtype, true)?;
        let mut features = Vec::with_capacity(1 + self.paths.len());
        features.push(pooled);
        for path in &self.paths {
            features.push(path.run(stream, input)?);
        }
        self.merge.run(stream, &stream.concat(&features, 1)?)
    }
}

struct Recurrent {
    conv: Block,
    recurrent: Rnn,
    linear: Affine,
    norm: Norm,
    zero: Array,
}

impl Recurrent {
    fn load(stream: &Stream, source: &super::Recurrent, dtype: DType) -> Result<Self> {
        Ok(Self {
            conv: Block::load(stream, &source.conv, dtype)?,
            recurrent: Rnn::load(stream, &source.parameters, source.hidden, dtype, Cell::Lstm)?,
            linear: Affine::load(stream, &source.weight, &source.bias, Layout::Linear, dtype)?,
            norm: Norm::load(stream, &source.norm, dtype, &[-1])?,
            zero: stream.scalar(0.0, dtype)?,
        })
    }

    fn run(&self, stream: &Stream, input: &Array) -> Result<Array> {
        let [batch, _, bins, frames] = <[i32; 4]>::try_from(input.shape())?;
        let value = self.conv.run(stream, input)?;
        let value = stream.select(&value, 1, 0)?;
        let value = stream.transpose(&value, &[0, 2, 1])?;
        let value = self.recurrent.run(stream, &value)?;
        let value = self.linear.linear(stream, &value)?;
        let value = stream.max(&self.norm.run(stream, &value)?, &self.zero)?;
        let value = stream.transpose(&value, &[0, 2, 1])?;
        stream.reshape(&value, &[batch, 1, bins, frames])
    }
}

struct Subnetwork {
    first: Option<Block>,
    down: Vec<[Block; 2]>,
    pyramid: Pyramid,
    up: Vec<Block>,
    recurrent: Option<Recurrent>,
    last: Option<Block>,
    dtype: DType,
}

impl Subnetwork {
    fn load(stream: &Stream, source: &super::Subnetwork, dtype: DType) -> Result<Self> {
        let block = |src: &super::Block| Block::load(stream, src, dtype);
        Ok(Self {
            first: source.first.as_ref().map(block).transpose()?,
            down: source
                .down
                .iter()
                .map(|[first, second]| Ok([block(first)?, block(second)?]))
                .collect::<Result<_>>()?,
            pyramid: Pyramid::load(stream, &source.aspp, dtype)?,
            up: source.up.iter().map(block).collect::<Result<_>>()?,
            recurrent: source
                .recurrent
                .as_ref()
                .map(|value| Recurrent::load(stream, value, dtype))
                .transpose()?,
            last: source.last.as_ref().map(block).transpose()?,
            dtype,
        })
    }

    fn run(&self, stream: &Stream, input: &Array) -> Result<Array> {
        let mut hidden = if let Some(first) = &self.first {
            first.run(stream, input)?
        } else {
            stream.reshape(input, input.shape())?
        };
        let mut skips = Vec::with_capacity(self.down.len());
        for [first, second] in &self.down {
            let features = first.run(stream, &hidden)?;
            let downsampled = second.run(stream, &features)?;
            skips.push(if self.first.is_some() {
                hidden
            } else {
                features
            });
            hidden = downsampled;
        }
        hidden = self.pyramid.run(stream, &hidden, self.dtype)?;
        for (index, (block, skip)) in self.up.iter().zip(skips.into_iter().rev()).enumerate() {
            if let Some(recurrent) = self
                .recurrent
                .as_ref()
                .filter(|_| index + 1 == self.up.len())
            {
                let context = recurrent.run(stream, &hidden)?;
                hidden = stream.concat(&[hidden, context], 1)?;
            }
            let [_, _, bins, frames] = <[i32; 4]>::try_from(hidden.shape())?;
            hidden = resize_bilinear(stream, hidden, [2 * bins, 2 * frames], self.dtype, true)?;
            let [_, _, skip_bins, skip_frames] = <[i32; 4]>::try_from(skip.shape())?;
            ensure!(
                skip_bins == 2 * bins && skip_frames >= 2 * frames,
                "invalid VR MLX skip shape"
            );
            let crop = (skip_frames - 2 * frames) / 2;
            let skip = stream.slice(&skip, 3, crop, crop + 2 * frames, 1)?;
            hidden = block.run(stream, &stream.concat(&[hidden, skip], 1)?)?;
        }
        if let Some(last) = &self.last {
            last.run(stream, &hidden)
        } else {
            Ok(hidden)
        }
    }
}
