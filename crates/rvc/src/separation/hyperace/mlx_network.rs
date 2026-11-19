use anyhow::{Context, Result};

use super::super::super::{
    mlx::{Array, DType, Stream},
    mlx_layers::{resize_bilinear, silu, standardize},
};
use super::super::mlx_network::MlxLinear;
use super::{HEADS, HYPEREDGES, Segmentation};

pub(in crate::separation::roformer) struct Segmenter {
    dtype: DType,
    stem: Activated,
    encoders: Vec<(Activated, Nested)>,
    fusion: Fusion,
    decoders: Vec<Decoder>,
    decoded: Nested,
    upsample: Vec<Shuffle>,
    output: Conv,
    target_bins: i32,
}

impl Segmenter {
    pub(in crate::separation::roformer) fn load(
        stream: &Stream,
        model: &Segmentation,
        dtype: DType,
    ) -> Result<Self> {
        Ok(Self {
            dtype,
            stem: Activated::load(stream, &model.backbone.stem, dtype)?,
            encoders: model
                .backbone
                .stages
                .iter()
                .map(|stage| {
                    Ok((
                        Activated::load(stream, &stage.downsample, dtype)?,
                        Nested::load(stream, &stage.block, dtype)?,
                    ))
                })
                .collect::<Result<_>>()?,
            fusion: Fusion::load(stream, &model.fusion, dtype)?,
            decoders: model
                .decoder
                .levels
                .iter()
                .map(|stage| Decoder::load(stream, stage, dtype))
                .collect::<Result<_>>()?,
            decoded: Nested::load(stream, &model.decoder.output, dtype)?,
            upsample: model
                .output
                .blocks
                .iter()
                .map(|block| Shuffle::load(stream, block, dtype))
                .collect::<Result<_>>()?,
            output: Conv::load(stream, &model.output.output, dtype)?,
            target_bins: i32::try_from(model.output.target_bins)?,
        })
    }

    pub(in crate::separation::roformer) fn run(
        &self,
        stream: &Stream,
        input: &Array,
    ) -> Result<Array> {
        let [batch, _, frames, _] = <[i32; 4]>::try_from(input.shape())?;
        let mut value = self.stem.run(stream, input)?;
        let mut levels = Vec::with_capacity(self.encoders.len());
        for (down, stage) in &self.encoders {
            value = stage.run(stream, &down.run(stream, &value)?)?;
            levels.push(stream.reshape(&value, value.shape())?);
        }
        let graph = self.fusion.run(stream, &levels, self.dtype)?;
        let mut decoded = None;
        for (stage, encoder) in self.decoders.iter().zip(levels).rev() {
            decoded = Some(stage.run(stream, &encoder, &graph, decoded, self.dtype)?);
        }
        let value = self.decoded.run(
            stream,
            &decoded.context("HyperACE decoder needs encoder features")?,
        )?;
        let bands = value.shape()[3];
        let mut value = resize_bilinear(stream, value, [frames, bands], self.dtype, false)?;
        for stage in &self.upsample {
            value = stage.run(stream, &value)?;
        }
        let value = resize_bilinear(stream, value, [frames, self.target_bins], self.dtype, false)?;
        let value = self.output.run(stream, &value)?;
        let value = stream.transpose(&value, &[0, 2, 3, 1])?;
        stream.reshape(&value, &[batch, frames, -1])
    }
}

struct Conv {
    weight: Array,
    stride: [i32; 2],
    pad: i32,
    groups: i32,
}

impl Conv {
    fn load(stream: &Stream, model: &super::Conv2d, dtype: DType) -> Result<Self> {
        Ok(Self {
            weight: stream.upload(&model.weight.f_permute([0, 2, 3, 1])?, dtype)?,
            stride: [
                i32::try_from(model.stride[0])?,
                i32::try_from(model.stride[1])?,
            ],
            pad: i32::try_from(model.padding)?,
            groups: i32::try_from(model.groups)?,
        })
    }

    fn run(&self, stream: &Stream, input: &Array) -> Result<Array> {
        let input = stream.transpose(input, &[0, 2, 3, 1])?;
        let value = stream.convolution2d(
            &input,
            &self.weight,
            self.stride,
            [self.pad; 2],
            [1; 2],
            self.groups,
        )?;
        stream.transpose(&value, &[0, 3, 1, 2])
    }
}

struct Norm {
    weight: Array,
    bias: Array,
    dtype: DType,
}

impl Norm {
    fn load(stream: &Stream, model: &super::InstanceNorm, dtype: DType) -> Result<Self> {
        Ok(Self {
            weight: stream.upload(&model.weight.f_reshape([1, -1, 1, 1])?, dtype)?,
            bias: stream.upload(&model.bias.f_reshape([1, -1, 1, 1])?, dtype)?,
            dtype,
        })
    }

    fn activate(&self, stream: &Stream, input: &Array) -> Result<Array> {
        let value = standardize(stream, input, &[2, 3], 1e-8)?;
        let value = stream.cast(&value, self.dtype)?;
        let value = stream.add(&stream.mul(&value, &self.weight)?, &self.bias)?;
        silu(stream, &value)
    }
}

struct Activated {
    depthwise: Option<Conv>,
    project: Conv,
    norm: Norm,
}

impl Activated {
    fn load(stream: &Stream, model: &super::ActivatedConv, dtype: DType) -> Result<Self> {
        Ok(Self {
            depthwise: model
                .depthwise
                .as_ref()
                .map(|conv| Conv::load(stream, conv, dtype))
                .transpose()?,
            project: Conv::load(stream, &model.project, dtype)?,
            norm: Norm::load(stream, &model.norm, dtype)?,
        })
    }

    fn run(&self, stream: &Stream, input: &Array) -> Result<Array> {
        let depthwise = self
            .depthwise
            .as_ref()
            .map(|conv| conv.run(stream, input))
            .transpose()?;
        let value = self
            .project
            .run(stream, depthwise.as_ref().unwrap_or(input))?;
        self.norm.activate(stream, &value)
    }
}

struct Depthwise {
    first: Activated,
    second: Activated,
    project: Activated,
    residual: Vec<[Activated; 2]>,
}

impl Depthwise {
    fn load(stream: &Stream, model: &super::DepthwiseC3, dtype: DType) -> Result<Self> {
        Ok(Self {
            first: Activated::load(stream, &model.first, dtype)?,
            second: Activated::load(stream, &model.second, dtype)?,
            project: Activated::load(stream, &model.project, dtype)?,
            residual: model
                .bottlenecks
                .iter()
                .map(|block| {
                    Ok([
                        Activated::load(stream, &block.first, dtype)?,
                        Activated::load(stream, &block.second, dtype)?,
                    ])
                })
                .collect::<Result<_>>()?,
        })
    }

    fn run(&self, stream: &Stream, input: &Array) -> Result<Array> {
        let mut first = self.first.run(stream, input)?;
        for [left, right] in &self.residual {
            let branch = left.run(stream, &first)?;
            first = stream.add(&first, &right.run(stream, &branch)?)?;
        }
        let split = stream.concat(&[first, self.second.run(stream, input)?], 1)?;
        self.project.run(stream, &split)
    }
}

struct Nested {
    reduce: Activated,
    block: Depthwise,
    project: Activated,
}

impl Nested {
    fn load(stream: &Stream, model: &super::NestedDepthwiseC3, dtype: DType) -> Result<Self> {
        Ok(Self {
            reduce: Activated::load(stream, &model.reduce, dtype)?,
            block: Depthwise::load(stream, &model.nested, dtype)?,
            project: Activated::load(stream, &model.project, dtype)?,
        })
    }

    fn run(&self, stream: &Stream, input: &Array) -> Result<Array> {
        let value = self.reduce.run(stream, input)?;
        let value = self.block.run(stream, &value)?;
        self.project.run(stream, &value)
    }
}

struct Hypergraph {
    prototype: Array,
    context: MlxLinear,
    query: MlxLinear,
    edges: MlxLinear,
    vertices: MlxLinear,
    dtype: DType,
}

impl Hypergraph {
    fn load(stream: &Stream, graph: &super::Hypergraph, dtype: DType) -> Result<Self> {
        Ok(Self {
            prototype: stream.upload(&graph.global_prototype, dtype)?,
            context: MlxLinear::load(stream, &graph.context, dtype)?,
            query: MlxLinear::load(stream, &graph.query, dtype)?,
            edges: MlxLinear::load(stream, &graph.edges, dtype)?,
            vertices: MlxLinear::load(stream, &graph.vertices, dtype)?,
            dtype,
        })
    }

    fn run(&self, stream: &Stream, input: &Array) -> Result<Array> {
        let [batch, channels, frames, bands] = <[i32; 4]>::try_from(input.shape())?;
        let positions = frames * bands;
        let heads = HEADS as i32;
        let edges = HYPEREDGES as i32;
        let dim = channels / heads;
        let flat = stream.reshape(input, &[batch, channels, positions])?;
        let mean = stream.reshape(&stream.mean_keep(&flat, &[2])?, &[batch, channels])?;
        let max = stream.reshape(&stream.max_keep(&flat, 2)?, &[batch, channels])?;
        let context = stream.concat(&[mean, max], 1)?;
        let context = self.context.forward(stream, &context)?;
        let context = stream.reshape(&context, &[batch, edges, channels])?;
        let prototype = stream.add(&self.prototype, &context)?;
        let prototype = stream.reshape(&prototype, &[batch, edges, heads, dim])?;
        let prototype = stream.transpose(&prototype, &[0, 2, 3, 1])?;
        let features = stream.transpose(&flat, &[0, 2, 1])?;
        let query = self.query.forward(stream, &features)?;
        let query = stream.reshape(&query, &[batch, positions, heads, dim])?;
        let query = stream.transpose(&query, &[0, 2, 1, 3])?;
        let attention = stream.matmul(&query, &prototype)?;
        let scale = stream.scalar((dim as f32).sqrt().recip(), self.dtype)?;
        let attention = stream.mean_keep(&stream.mul(&attention, &scale)?, &[1])?;
        let attention = stream.reshape(&attention, &[batch, positions, edges])?;
        let attention = stream.softmax(&stream.transpose(&attention, &[0, 2, 1])?, 2)?;
        let edge = stream.matmul(&attention, &features)?;
        let edge = silu(stream, &self.edges.forward(stream, &edge)?)?;
        let vertex = stream.transpose(&attention, &[0, 2, 1])?;
        let vertex = stream.matmul(&vertex, &edge)?;
        let vertex = silu(stream, &self.vertices.forward(stream, &vertex)?)?;
        let value = stream.transpose(&stream.add(&features, &vertex)?, &[0, 2, 1])?;
        stream.reshape(&value, &[batch, channels, frames, bands])
    }
}

struct GraphBranch {
    first: Activated,
    second: Activated,
    graph: Hypergraph,
    project: Activated,
}

impl GraphBranch {
    fn load(stream: &Stream, model: &super::HypergraphC3, dtype: DType) -> Result<Self> {
        Ok(Self {
            first: Activated::load(stream, &model.first, dtype)?,
            second: Activated::load(stream, &model.second, dtype)?,
            graph: Hypergraph::load(stream, &model.graph, dtype)?,
            project: Activated::load(stream, &model.project, dtype)?,
        })
    }

    fn run(&self, stream: &Stream, input: &Array) -> Result<Array> {
        let graph = self.graph.run(stream, &self.second.run(stream, input)?)?;
        let value = stream.concat(&[graph, self.first.run(stream, input)?], 1)?;
        self.project.run(stream, &value)
    }
}

struct Fusion {
    input: Activated,
    graphs: [GraphBranch; 2],
    graph_merge: Activated,
    residual: Depthwise,
    output: Activated,
    high_end: i32,
    low_end: i32,
}

impl Fusion {
    fn load(stream: &Stream, model: &super::MultiscaleFusion, dtype: DType) -> Result<Self> {
        let [first, second] = &model.high_order;
        let [high, low, _] = model.channel_split;
        Ok(Self {
            input: Activated::load(stream, &model.inputs, dtype)?,
            graphs: [
                GraphBranch::load(stream, first, dtype)?,
                GraphBranch::load(stream, second, dtype)?,
            ],
            graph_merge: Activated::load(stream, &model.high_fusion, dtype)?,
            residual: Depthwise::load(stream, &model.low_order, dtype)?,
            output: Activated::load(stream, &model.output, dtype)?,
            high_end: i32::try_from(high)?,
            low_end: i32::try_from(high + low)?,
        })
    }

    fn run(&self, stream: &Stream, levels: &[Array], dtype: DType) -> Result<Array> {
        let [_, _, frames, bands] = <[i32; 4]>::try_from(levels[2].shape())?;
        let scaled = levels
            .iter()
            .map(|level| {
                let level = stream.reshape(level, level.shape())?;
                resize_bilinear(stream, level, [frames, bands], dtype, false)
            })
            .collect::<Result<Vec<_>>>()?;
        let joined = self.input.run(stream, &stream.concat(&scaled, 1)?)?;
        let high = stream.slice(&joined, 1, 0, self.high_end, 1)?;
        let low = stream.slice(&joined, 1, self.high_end, self.low_end, 1)?;
        let rest = stream.slice(&joined, 1, self.low_end, joined.shape()[1], 1)?;
        let graphs = [
            self.graphs[0].run(stream, &high)?,
            self.graphs[1].run(stream, &high)?,
        ];
        let high = self.graph_merge.run(stream, &stream.concat(&graphs, 1)?)?;
        let low = self.residual.run(stream, &low)?;
        self.output
            .run(stream, &stream.concat(&[high, low, rest], 1)?)
    }
}

struct Decoder {
    graph: Activated,
    skip: Activated,
    fusion: Array,
    upsample: Option<Nested>,
}

impl Decoder {
    fn load(stream: &Stream, model: &super::DecoderLevel, dtype: DType) -> Result<Self> {
        Ok(Self {
            graph: Activated::load(stream, &model.hypergraph, dtype)?,
            skip: Activated::load(stream, &model.skip, dtype)?,
            fusion: stream.upload(&model.fusion, dtype)?,
            upsample: model
                .upsample
                .as_ref()
                .map(|block| Nested::load(stream, block, dtype))
                .transpose()?,
        })
    }

    fn run(
        &self,
        stream: &Stream,
        encoder: &Array,
        graph: &Array,
        previous: Option<Array>,
        dtype: DType,
    ) -> Result<Array> {
        let [_, _, frames, bands] = <[i32; 4]>::try_from(encoder.shape())?;
        let shape = [frames, bands];
        let skip = self.skip.run(stream, encoder)?;
        let value = if let (Some(up), Some(previous)) = (&self.upsample, previous) {
            let value = resize_bilinear(stream, previous, shape, dtype, false)?;
            stream.add(&skip, &up.run(stream, &value)?)?
        } else {
            skip
        };
        let graph = resize_bilinear(
            stream,
            stream.reshape(graph, graph.shape())?,
            shape,
            dtype,
            false,
        )?;
        let graph = self.graph.run(stream, &graph)?;
        stream.add(&value, &stream.mul(&graph, &self.fusion)?)
    }
}

struct TimeFreq {
    input: (Norm, Conv),
    output: (Norm, Conv),
    down: (Norm, MlxLinear),
    up: (Norm, MlxLinear),
    residual: Conv,
}

impl TimeFreq {
    fn load(stream: &Stream, model: &super::TimeFreqBlock, dtype: DType) -> Result<Self> {
        let conv = |layer: &super::TimeFreqConv| -> Result<_> {
            Ok((
                Norm::load(stream, &layer.norm, dtype)?,
                Conv::load(stream, &layer.conv, dtype)?,
            ))
        };
        let projection = |layer: &super::TimeDistributedProjection| -> Result<_> {
            Ok((
                Norm::load(stream, &layer.norm, dtype)?,
                MlxLinear::load(stream, &layer.linear, dtype)?,
            ))
        };
        Ok(Self {
            input: conv(&model.input)?,
            output: conv(&model.output)?,
            down: projection(&model.down)?,
            up: projection(&model.up)?,
            residual: Conv::load(stream, &model.residual, dtype)?,
        })
    }

    fn run(&self, stream: &Stream, input: &Array) -> Result<Array> {
        let convolved = self
            .input
            .1
            .run(stream, &self.input.0.activate(stream, input)?)?;
        let reduced = self
            .down
            .1
            .forward(stream, &self.down.0.activate(stream, &convolved)?)?;
        let reduced = self
            .up
            .1
            .forward(stream, &self.up.0.activate(stream, &reduced)?)?;
        let mixed = stream.add(&convolved, &reduced)?;
        let result = self
            .output
            .1
            .run(stream, &self.output.0.activate(stream, &mixed)?)?;
        stream.add(&result, &self.residual.run(stream, input)?)
    }
}

struct Shuffle {
    conv: Activated,
    refinement: [TimeFreq; 2],
}

impl Shuffle {
    fn load(stream: &Stream, model: &super::FreqPixelShuffle, dtype: DType) -> Result<Self> {
        let [first, second] = &model.refinements;
        Ok(Self {
            conv: Activated::load(stream, &model.conv, dtype)?,
            refinement: [
                TimeFreq::load(stream, first, dtype)?,
                TimeFreq::load(stream, second, dtype)?,
            ],
        })
    }

    fn run(&self, stream: &Stream, input: &Array) -> Result<Array> {
        let value = self.conv.run(stream, input)?;
        let [batch, channels, frames, bins] = <[i32; 4]>::try_from(value.shape())?;
        let value = stream.reshape(&value, &[batch, channels / 2, 2, frames, bins])?;
        let value = stream.transpose(&value, &[0, 1, 3, 4, 2])?;
        let mut value = stream.reshape(&value, &[batch, channels / 2, frames, 2 * bins])?;
        for layer in &self.refinement {
            value = layer.run(stream, &value)?;
        }
        Ok(value)
    }
}
