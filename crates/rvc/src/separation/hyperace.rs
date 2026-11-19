//! BS-RoFormer HyperACE segmentation branch from the bundled PyMSS source.
//! See `licenses/pymss-core.txt`.

use anyhow::{Context, Result};
use serde::Deserialize;
use tch::Tensor;

use super::{Linear, Weights};

#[path = "hyperace/mlx_network.rs"]
pub(super) mod mlx_network;

#[cfg(feature = "onnx")]
#[path = "hyperace/onnx_network.rs"]
pub(super) mod onnx_network;

const ENCODER_CHANNELS: [i64; 4] = [256, 384, 512, 768];
const STEM_CHANNELS: i64 = 64;
const HYPEREDGES: i64 = 32;
const HEADS: i64 = 8;
const MASK_CHANNELS: i64 = 4;

#[derive(Clone, Copy, Default, Deserialize, PartialEq, Eq)]
pub(super) enum MaskMode {
    #[default]
    #[serde(rename = "no_segm")]
    Standard,
    #[serde(rename = "full")]
    Full,
    #[serde(rename = "segm_only")]
    SegmentationOnly,
}

struct Conv2d {
    weight: Tensor,
    stride: [i64; 2],
    padding: i64,
    groups: i64,
}

impl Conv2d {
    fn load(
        weights: &Weights,
        name: &str,
        input: i64,
        output: i64,
        kernel: i64,
        stride: [i64; 2],
        groups: i64,
    ) -> Result<Self> {
        Ok(Self {
            weight: weights.tensor(
                &format!("{name}.weight"),
                &[output, input / groups, kernel, kernel],
            )?,
            stride,
            padding: kernel / 2,
            groups,
        })
    }

    fn pointwise(weights: &Weights, name: &str, input: i64, output: i64) -> Result<Self> {
        Self::load(weights, name, input, output, 1, [1, 1], 1)
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(x.f_conv2d(
            &self.weight,
            None::<&Tensor>,
            self.stride,
            [self.padding; 2],
            [1; 2],
            self.groups,
        )?)
    }
}

struct InstanceNorm {
    weight: Tensor,
    bias: Tensor,
}

impl InstanceNorm {
    fn load(weights: &Weights, name: &str, channels: i64) -> Result<Self> {
        Ok(Self {
            weight: weights.tensor(&format!("{name}.weight"), &[channels])?,
            bias: weights.tensor(&format!("{name}.bias"), &[channels])?,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        Ok(input.f_instance_norm(
            Some(&self.weight),
            Some(&self.bias),
            None,
            None,
            true,
            0.1,
            1e-8,
            true,
        )?)
    }
}

struct ActivatedConv {
    depthwise: Option<Conv2d>,
    project: Conv2d,
    norm: InstanceNorm,
}

impl ActivatedConv {
    fn pointwise(weights: &Weights, name: &str, input: i64, output: i64) -> Result<Self> {
        Ok(Self {
            depthwise: None,
            project: Conv2d::pointwise(weights, &format!("{name}.conv"), input, output)?,
            norm: InstanceNorm::load(weights, &format!("{name}.bn"), output)?,
        })
    }

    fn depthwise(
        weights: &Weights,
        name: &str,
        input: i64,
        output: i64,
        stride: [i64; 2],
    ) -> Result<Self> {
        Ok(Self {
            depthwise: Some(Conv2d::load(
                weights,
                &format!("{name}.dwconv"),
                input,
                input,
                3,
                stride,
                input,
            )?),
            project: Conv2d::pointwise(weights, &format!("{name}.pwconv"), input, output)?,
            norm: InstanceNorm::load(weights, &format!("{name}.bn"), output)?,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let depthwise = self
            .depthwise
            .as_ref()
            .map(|layer| layer.forward(input))
            .transpose()?;
        let projected = self.project.forward(depthwise.as_ref().unwrap_or(input))?;
        Ok(self.norm.forward(&projected)?.f_silu()?)
    }
}

struct DepthwiseBottleneck {
    first: ActivatedConv,
    second: ActivatedConv,
}

impl DepthwiseBottleneck {
    fn load(weights: &Weights, name: &str, channels: i64) -> Result<Self> {
        Ok(Self {
            first: ActivatedConv::depthwise(
                weights,
                &format!("{name}.dsconv1"),
                channels,
                channels,
                [1; 2],
            )?,
            second: ActivatedConv::depthwise(
                weights,
                &format!("{name}.dsconv2"),
                channels,
                channels,
                [1; 2],
            )?,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        Ok(input.f_add(&self.second.forward(&self.first.forward(input)?)?)?)
    }
}

struct DepthwiseC3 {
    first: ActivatedConv,
    second: ActivatedConv,
    project: ActivatedConv,
    bottlenecks: Vec<DepthwiseBottleneck>,
}

impl DepthwiseC3 {
    fn load(weights: &Weights, name: &str, channels: i64, depth: usize) -> Result<Self> {
        Ok(Self {
            first: ActivatedConv::pointwise(weights, &format!("{name}.cv1"), channels, channels)?,
            second: ActivatedConv::pointwise(weights, &format!("{name}.cv2"), channels, channels)?,
            project: ActivatedConv::pointwise(
                weights,
                &format!("{name}.cv3"),
                2 * channels,
                channels,
            )?,
            bottlenecks: (0..depth)
                .map(|index| {
                    DepthwiseBottleneck::load(weights, &format!("{name}.m.{index}"), channels)
                })
                .collect::<Result<_>>()?,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let mut first = self.first.forward(input)?;
        for bottleneck in &self.bottlenecks {
            first = bottleneck.forward(&first)?;
        }
        self.project
            .forward(&Tensor::f_cat(&[first, self.second.forward(input)?], 1)?)
    }
}

struct NestedDepthwiseC3 {
    reduce: ActivatedConv,
    nested: DepthwiseC3,
    project: ActivatedConv,
}

impl NestedDepthwiseC3 {
    fn load(weights: &Weights, name: &str, input: i64, output: i64, depth: usize) -> Result<Self> {
        let hidden = output / 2;
        Ok(Self {
            reduce: ActivatedConv::pointwise(weights, &format!("{name}.cv1"), input, hidden)?,
            nested: DepthwiseC3::load(weights, &format!("{name}.m"), hidden, depth)?,
            project: ActivatedConv::pointwise(weights, &format!("{name}.cv2"), hidden, output)?,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.project
            .forward(&self.nested.forward(&self.reduce.forward(input)?)?)
    }
}

struct Hypergraph {
    global_prototype: Tensor,
    context: Linear,
    query: Linear,
    edges: Linear,
    vertices: Linear,
}

impl Hypergraph {
    fn load(weights: &Weights, name: &str, channels: i64) -> Result<Self> {
        let generation = format!("{name}.adaptive_hyperedge_gen");
        let convolution = format!("{name}.hypergraph_conv");
        Ok(Self {
            global_prototype: weights.tensor(
                &format!("{generation}.global_proto"),
                &[HYPEREDGES, channels],
            )?,
            context: weights.linear(
                &format!("{generation}.context_mapper"),
                2 * channels,
                HYPEREDGES * channels,
                false,
            )?,
            query: weights.linear(
                &format!("{generation}.query_proj"),
                channels,
                channels,
                false,
            )?,
            edges: weights.linear(&format!("{convolution}.W_e"), channels, channels, false)?,
            vertices: weights.linear(&format!("{convolution}.W_v"), channels, channels, false)?,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let (batch, channels, frames, bands) = input.size4()?;
        let positions = frames * bands;
        let head_dim = channels / HEADS;
        let flat = input.f_flatten(2, 3)?;
        let context = Tensor::f_cat(
            &[
                flat.f_adaptive_avg_pool1d([1])?.f_squeeze_dim(2)?,
                flat.f_adaptive_max_pool1d([1])?.0.f_squeeze_dim(2)?,
            ],
            1,
        )?;
        let prototype = self
            .global_prototype
            .f_unsqueeze(0)?
            .f_add(
                &self
                    .context
                    .forward(&context)?
                    .f_reshape([batch, HYPEREDGES, channels])?,
            )?
            .f_reshape([batch, HYPEREDGES, HEADS, head_dim])?
            .f_permute([0, 2, 3, 1])?;
        let features = flat.f_transpose(1, 2)?;
        let query = self
            .query
            .forward(&features)?
            .f_reshape([batch, positions, HEADS, head_dim])?
            .f_permute([0, 2, 1, 3])?;
        let attention = query
            .f_matmul(&prototype)?
            .f_mul_scalar((head_dim as f64).powf(-0.5))?
            .f_mean_dim([1].as_slice(), false, None)?
            .f_transpose(1, 2)?
            .f_softmax(-1, None)?;
        let edge_values = self.edges.forward(&attention.f_bmm(&features)?)?.f_silu()?;
        let vertex_values = self
            .vertices
            .forward(&attention.f_transpose(1, 2)?.f_bmm(&edge_values)?)?
            .f_silu()?;
        Ok(features
            .f_add(&vertex_values)?
            .f_transpose(1, 2)?
            .f_reshape([batch, channels, frames, bands])?)
    }
}

struct HypergraphC3 {
    first: ActivatedConv,
    second: ActivatedConv,
    graph: Hypergraph,
    project: ActivatedConv,
}

impl HypergraphC3 {
    fn load(weights: &Weights, name: &str, channels: i64) -> Result<Self> {
        Ok(Self {
            first: ActivatedConv::pointwise(weights, &format!("{name}.cv1"), channels, channels)?,
            second: ActivatedConv::pointwise(weights, &format!("{name}.cv2"), channels, channels)?,
            graph: Hypergraph::load(weights, &format!("{name}.ahc"), channels)?,
            project: ActivatedConv::pointwise(
                weights,
                &format!("{name}.cv3"),
                channels * 2,
                channels,
            )?,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let graph = self.graph.forward(&self.second.forward(input)?)?;
        self.project
            .forward(&Tensor::f_cat(&[graph, self.first.forward(input)?], 1)?)
    }
}

fn resized(input: &Tensor, frames: i64, bands: i64) -> Result<Tensor> {
    Ok(input.f_upsample_bilinear2d([frames, bands], false, None, None)?)
}

struct MultiscaleFusion {
    inputs: ActivatedConv,
    high_order: [HypergraphC3; 2],
    high_fusion: ActivatedConv,
    low_order: DepthwiseC3,
    output: ActivatedConv,
    channel_split: [i64; 3],
}

impl MultiscaleFusion {
    fn load(weights: &Weights, name: &str) -> Result<Self> {
        let channels = ENCODER_CHANNELS[2];
        let high = channels / 2;
        let low = channels / 4;
        Ok(Self {
            inputs: ActivatedConv::pointwise(
                weights,
                &format!("{name}.fuse_conv"),
                ENCODER_CHANNELS.iter().sum(),
                channels,
            )?,
            high_order: [
                HypergraphC3::load(weights, &format!("{name}.high_order_branch.0"), high)?,
                HypergraphC3::load(weights, &format!("{name}.high_order_branch.1"), high)?,
            ],
            high_fusion: ActivatedConv::pointwise(
                weights,
                &format!("{name}.high_order_fuse"),
                high * 2,
                high,
            )?,
            low_order: DepthwiseC3::load(weights, &format!("{name}.low_order_branch.0"), low, 1)?,
            output: ActivatedConv::pointwise(
                weights,
                &format!("{name}.final_fuse"),
                channels,
                channels,
            )?,
            channel_split: [high, low, channels - high - low],
        })
    }

    fn forward(&self, levels: &[Tensor]) -> Result<Tensor> {
        let (_, _, frames, bands) = levels[2].size4()?;
        let scales = [
            resized(&levels[0], frames, bands)?,
            resized(&levels[1], frames, bands)?,
            levels[2].shallow_clone(),
            resized(&levels[3], frames, bands)?,
        ];
        let split = self
            .inputs
            .forward(&Tensor::f_cat(&scales, 1)?)?
            .f_split_with_sizes(self.channel_split, 1)?;
        let high = [
            self.high_order[0].forward(&split[0])?,
            self.high_order[1].forward(&split[0])?,
        ];
        let joined = [
            self.high_fusion.forward(&Tensor::f_cat(&high, 1)?)?,
            self.low_order.forward(&split[1])?,
            split[2].shallow_clone(),
        ];
        self.output.forward(&Tensor::f_cat(&joined, 1)?)
    }
}

struct EncoderStage {
    downsample: ActivatedConv,
    block: NestedDepthwiseC3,
}

struct Backbone {
    stem: ActivatedConv,
    stages: Vec<EncoderStage>,
}

impl Backbone {
    fn load(weights: &Weights, name: &str, input: i64) -> Result<Self> {
        let stem = ActivatedConv::depthwise(
            weights,
            &format!("{name}.stem"),
            input,
            STEM_CHANNELS,
            [2, 1],
        )?;
        let mut stages = Vec::with_capacity(ENCODER_CHANNELS.len());
        let mut previous = STEM_CHANNELS;
        for (index, (channels, depth)) in ENCODER_CHANNELS.into_iter().zip([2, 4, 4, 2]).enumerate()
        {
            let key = format!("{name}.p{}", index + 2);
            let stride = if index < 2 { [2, 1] } else { [2, 2] };
            stages.push(EncoderStage {
                downsample: ActivatedConv::depthwise(
                    weights,
                    &format!("{key}.0"),
                    previous,
                    channels,
                    stride,
                )?,
                block: NestedDepthwiseC3::load(
                    weights,
                    &format!("{key}.1"),
                    channels,
                    channels,
                    depth,
                )?,
            });
            previous = channels;
        }
        Ok(Self { stem, stages })
    }

    fn forward(&self, input: &Tensor) -> Result<Vec<Tensor>> {
        let mut feature = self.stem.forward(input)?;
        let mut levels = Vec::with_capacity(self.stages.len());
        for stage in &self.stages {
            feature = stage.block.forward(&stage.downsample.forward(&feature)?)?;
            levels.push(feature.shallow_clone());
        }
        Ok(levels)
    }
}

struct DecoderLevel {
    hypergraph: ActivatedConv,
    skip: ActivatedConv,
    fusion: Tensor,
    upsample: Option<NestedDepthwiseC3>,
}

struct Decoder {
    levels: Vec<DecoderLevel>,
    output: NestedDepthwiseC3,
}

impl Decoder {
    fn load(weights: &Weights, name: &str) -> Result<Self> {
        let mut levels = Vec::with_capacity(ENCODER_CHANNELS.len());
        let hypergraph = ENCODER_CHANNELS[2];
        for (index, channels) in ENCODER_CHANNELS.into_iter().enumerate() {
            let id = index + 2;
            let upsample = ENCODER_CHANNELS
                .get(index + 1)
                .map(|&previous| {
                    NestedDepthwiseC3::load(
                        weights,
                        &format!("{name}.up_d{}", id + 1),
                        previous,
                        channels,
                        1,
                    )
                })
                .transpose()?;
            levels.push(DecoderLevel {
                hypergraph: ActivatedConv::pointwise(
                    weights,
                    &format!("{name}.h_to_d{id}"),
                    hypergraph,
                    channels,
                )?,
                skip: ActivatedConv::pointwise(
                    weights,
                    &format!("{name}.skip_p{id}"),
                    channels,
                    channels,
                )?,
                fusion: weights
                    .tensor(&format!("{name}.fusion_d{id}.gamma"), &[1, channels, 1, 1])?,
                upsample,
            });
        }
        let channels = ENCODER_CHANNELS[0];
        Ok(Self {
            levels,
            output: NestedDepthwiseC3::load(
                weights,
                &format!("{name}.final_d2"),
                channels,
                channels,
                1,
            )?,
        })
    }

    fn forward(&self, encoded: &[Tensor], hypergraph: &Tensor) -> Result<Tensor> {
        let mut decoded = None;
        for (level, encoder) in self.levels.iter().zip(encoded).rev() {
            let (_, _, frames, bands) = encoder.size4()?;
            let skip = level.skip.forward(encoder)?;
            let feature = match (&level.upsample, &decoded) {
                (Some(upsample), Some(previous)) => upsample
                    .forward(&resized(previous, frames, bands)?)?
                    .f_add(&skip)?,
                _ => skip,
            };
            let graph = level
                .hypergraph
                .forward(&resized(hypergraph, frames, bands)?)?;
            decoded = Some(feature.f_add(&graph.f_mul(&level.fusion)?)?);
        }
        self.output
            .forward(&decoded.context("HyperACE decoder needs encoder features")?)
    }
}

struct TimeFreqConv {
    norm: InstanceNorm,
    conv: Conv2d,
}

impl TimeFreqConv {
    fn load(weights: &Weights, name: &str, channels: i64) -> Result<Self> {
        Ok(Self {
            norm: InstanceNorm::load(weights, &format!("{name}.0"), channels)?,
            conv: Conv2d::load(
                weights,
                &format!("{name}.2"),
                channels,
                channels,
                3,
                [1; 2],
                1,
            )?,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.conv.forward(&self.norm.forward(input)?.f_silu()?)
    }
}

struct TimeDistributedProjection {
    norm: InstanceNorm,
    linear: Linear,
}

impl TimeDistributedProjection {
    fn load(
        weights: &Weights,
        name: &str,
        channels: i64,
        index: usize,
        input: i64,
        output: i64,
    ) -> Result<Self> {
        Ok(Self {
            norm: InstanceNorm::load(weights, &format!("{name}.{index}"), channels)?,
            linear: weights.linear(&format!("{name}.{}", index + 2), input, output, false)?,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.linear.forward(&self.norm.forward(input)?.f_silu()?)
    }
}

struct TimeFreqBlock {
    input: TimeFreqConv,
    output: TimeFreqConv,
    down: TimeDistributedProjection,
    up: TimeDistributedProjection,
    residual: Conv2d,
}

impl TimeFreqBlock {
    fn load(weights: &Weights, name: &str, channels: i64, bands: i64) -> Result<Self> {
        let distributed = format!("{name}.tdf");
        let low = bands / 4;
        Ok(Self {
            input: TimeFreqConv::load(weights, &format!("{name}.tfc1"), channels)?,
            output: TimeFreqConv::load(weights, &format!("{name}.tfc2"), channels)?,
            down: TimeDistributedProjection::load(weights, &distributed, channels, 0, bands, low)?,
            up: TimeDistributedProjection::load(weights, &distributed, channels, 3, low, bands)?,
            residual: Conv2d::pointwise(weights, &format!("{name}.shortcut"), channels, channels)?,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let convolution = self.input.forward(input)?;
        let distributed = self.up.forward(&self.down.forward(&convolution)?)?;
        Ok(self
            .output
            .forward(&convolution.f_add(&distributed)?)?
            .f_add(&self.residual.forward(input)?)?)
    }
}

struct FreqPixelShuffle {
    conv: ActivatedConv,
    refinements: [TimeFreqBlock; 2],
}

impl FreqPixelShuffle {
    fn load(weights: &Weights, name: &str, channels: i64, bands: i64) -> Result<Self> {
        Ok(Self {
            conv: ActivatedConv::depthwise(
                weights,
                &format!("{name}.conv"),
                channels,
                channels,
                [1, 1],
            )?,
            refinements: [
                TimeFreqBlock::load(
                    weights,
                    &format!("{name}.out_conv.blocks.0"),
                    channels / 2,
                    bands,
                )?,
                TimeFreqBlock::load(
                    weights,
                    &format!("{name}.out_conv.blocks.1"),
                    channels / 2,
                    bands,
                )?,
            ],
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let output = self.conv.forward(input)?;
        let (batch, channels, frames, bands) = output.size4()?;
        let reordered = output
            .f_reshape([batch, channels / 2, 2, frames, bands])?
            .f_permute([0, 1, 3, 4, 2])?
            .f_reshape([batch, channels / 2, frames, bands * 2])?;
        self.refinements
            .iter()
            .try_fold(reordered, |value, layer| layer.forward(&value))
    }
}

struct UpsampleHead {
    blocks: Vec<FreqPixelShuffle>,
    output: Conv2d,
    target_bins: i64,
}

impl UpsampleHead {
    fn load(weights: &Weights, name: &str, bands: i64, target_bins: i64) -> Result<Self> {
        let channels = ENCODER_CHANNELS[0];
        Ok(Self {
            blocks: (1..=4)
                .map(|index| {
                    FreqPixelShuffle::load(
                        weights,
                        &format!("{name}.block{index}"),
                        channels >> (index - 1),
                        bands << index,
                    )
                })
                .collect::<Result<_>>()?,
            output: Conv2d::load(
                weights,
                &format!("{name}.final_conv"),
                channels / 16,
                MASK_CHANNELS,
                3,
                [1, 1],
                1,
            )?,
            target_bins,
        })
    }

    fn forward(&self, mut input: Tensor) -> Result<Tensor> {
        for stage in &self.blocks {
            input = stage.forward(&input)?;
        }
        let (_, _, frames, bins) = input.size4()?;
        if bins != self.target_bins {
            input = resized(&input, frames, self.target_bins)?;
        }
        self.output.forward(&input)
    }
}

pub(super) struct Segmentation {
    backbone: Backbone,
    fusion: MultiscaleFusion,
    decoder: Decoder,
    output: UpsampleHead,
}

impl Segmentation {
    pub(super) fn load(
        weights: &Weights,
        name: &str,
        input: i64,
        bands: i64,
        out_bins: i64,
    ) -> Result<Self> {
        Ok(Self {
            backbone: Backbone::load(weights, &format!("{name}.backbone"), input)?,
            fusion: MultiscaleFusion::load(weights, &format!("{name}.hyperace"))?,
            decoder: Decoder::load(weights, &format!("{name}.decoder"))?,
            output: UpsampleHead::load(weights, &format!("{name}.upsample_head"), bands, out_bins)?,
        })
    }

    /// The input has shape [batch, embedding, time, band]. Output interleaves the
    /// STFT channels and their real/imaginary values identically to the MLP.
    pub(super) fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let (batch, _, frames, _) = input.size4()?;
        let features = self.backbone.forward(input)?;
        let decoded = self
            .decoder
            .forward(&features, &self.fusion.forward(&features)?)?;
        let decoded = resized(&decoded, frames, decoded.size()[3])?;
        Ok(self
            .output
            .forward(decoded)?
            .f_permute([0, 2, 3, 1])?
            .f_reshape([batch, frames, -1])?)
    }
}
