use anyhow::{Context, Result};

use super::{
    ActivatedConv, Conv2d, DepthwiseC3, HEADS, HYPEREDGES, Hypergraph, HypergraphC3, InstanceNorm,
    MultiscaleFusion, Segmentation, TimeFreqBlock,
};
use crate::model::onnx::graph::{Attribute, Builder};
use crate::separation::roformer::onnx_network::{concat, linear, reshape, transpose};

pub(in crate::separation::roformer) fn forward(
    graph: &mut Builder,
    model: &Segmentation,
    input: &str,
) -> Result<String> {
    let mut feature = activated(graph, &model.backbone.stem, input)?;
    let mut encoded = Vec::with_capacity(model.backbone.stages.len());
    for stage in &model.backbone.stages {
        feature = activated(graph, &stage.downsample, &feature)?;
        feature = nested(graph, &stage.block, &feature)?;
        encoded.push(feature.clone());
    }
    let fused = multiscale(graph, &model.fusion, &encoded)?;
    let mut decoded = None;
    for (stage, skip) in model.decoder.levels.iter().zip(&encoded).rev() {
        let skip_projected = activated(graph, &stage.skip, skip)?;
        let feature = match (&stage.upsample, decoded.as_deref()) {
            (Some(layer), Some(previous)) => {
                let value = resize_like(graph, previous, skip);
                let value = nested(graph, layer, &value)?;
                graph.op("Add", &[&value, &skip_projected], vec![])
            }
            _ => skip_projected,
        };
        let context = resize_like(graph, &fused, skip);
        let context = activated(graph, &stage.hypergraph, &context)?;
        let scale = graph.tensor(&stage.fusion)?;
        let context = graph.op("Mul", &[&context, &scale], vec![]);
        decoded = Some(graph.op("Add", &[&feature, &context], vec![]));
    }
    let decoded = decoded.context("HyperACE decoder needs encoder features")?;
    let decoded = nested(graph, &model.decoder.output, &decoded)?;
    let mut value = resize_like(graph, &decoded, input);
    let mut bins = 0;
    for block in &model.output.blocks {
        let channels = block.conv.project.weight.size()[0] / 2;
        bins = block.refinements[0].down.linear.weight.size()[1];
        value = activated(graph, &block.conv, &value)?;
        value = reshape(graph, &value, &[0, channels, 2, -1, bins / 2]);
        value = transpose(graph, &value, &[0, 1, 3, 4, 2]);
        value = reshape(graph, &value, &[0, channels, -1, bins]);
        for refine in &block.refinements {
            value = time_freq(graph, refine, &value)?;
        }
    }
    let target = model.output.target_bins;
    if bins != target {
        let shape = graph.op("Shape", &[&value], vec![]);
        let geometry = slice(graph, &shape, 0, 0, 3);
        let target = graph.integers(&[1], &[target]);
        let size = concat(graph, &[&geometry, &target], 0);
        value = resize(graph, &value, &size);
    }
    value = convolution(graph, &model.output.output, &value)?;
    value = transpose(graph, &value, &[0, 2, 3, 1]);
    Ok(reshape(graph, &value, &[0, 0, -1]))
}

fn convolution(graph: &mut Builder, layer: &Conv2d, input: &str) -> Result<String> {
    let weight = graph.tensor(&layer.weight)?;
    Ok(graph.op(
        "Conv",
        &[input, &weight],
        vec![
            Attribute::ints("strides", &layer.stride),
            Attribute::ints("pads", &[layer.padding; 4]),
            Attribute::int("group", layer.groups),
        ],
    ))
}

fn normalize(graph: &mut Builder, layer: &InstanceNorm, input: &str) -> Result<String> {
    let weight = graph.tensor(&layer.weight)?;
    let bias = graph.tensor(&layer.bias)?;
    Ok(graph.op(
        "InstanceNormalization",
        &[input, &weight, &bias],
        vec![Attribute::float("epsilon", 1e-8)],
    ))
}

fn silu(graph: &mut Builder, input: &str) -> String {
    let gate = graph.op("Sigmoid", &[input], vec![]);
    graph.op("Mul", &[input, &gate], vec![])
}

fn activated(graph: &mut Builder, layer: &ActivatedConv, input: &str) -> Result<String> {
    let depth = layer
        .depthwise
        .as_ref()
        .map(|depth| convolution(graph, depth, input))
        .transpose()?;
    let value = convolution(graph, &layer.project, depth.as_deref().unwrap_or(input))?;
    let value = normalize(graph, &layer.norm, &value)?;
    Ok(silu(graph, &value))
}

fn depthwise(graph: &mut Builder, layer: &DepthwiseC3, input: &str) -> Result<String> {
    let mut first = activated(graph, &layer.first, input)?;
    for bottleneck in &layer.bottlenecks {
        let inner = activated(graph, &bottleneck.first, &first)?;
        let inner = activated(graph, &bottleneck.second, &inner)?;
        first = graph.op("Add", &[&first, &inner], vec![]);
    }
    let second = activated(graph, &layer.second, input)?;
    let merged = concat(graph, &[&first, &second], 1);
    activated(graph, &layer.project, &merged)
}

fn nested(graph: &mut Builder, layer: &super::NestedDepthwiseC3, input: &str) -> Result<String> {
    let value = activated(graph, &layer.reduce, input)?;
    let value = depthwise(graph, &layer.nested, &value)?;
    activated(graph, &layer.project, &value)
}

fn hypergraph(graph: &mut Builder, layer: &Hypergraph, input: &str) -> Result<String> {
    let channels = layer.query.weight.size()[0];
    let head_dim = channels / HEADS;
    let axes = graph.integers(&[2], &[2, 3]);
    let mean = graph.op(
        "ReduceMean",
        &[input, &axes],
        vec![Attribute::int("keepdims", 0)],
    );
    let max = graph.op(
        "ReduceMax",
        &[input, &axes],
        vec![Attribute::int("keepdims", 0)],
    );
    let context = concat(graph, &[&mean, &max], 1);
    let context = linear(graph, &context, &layer.context)?;
    let context = reshape(graph, &context, &[0, HYPEREDGES, channels]);
    let prototype = graph.tensor(&layer.global_prototype)?;
    let prototype = graph.op("Add", &[&context, &prototype], vec![]);
    let prototype = reshape(graph, &prototype, &[0, HYPEREDGES, HEADS, head_dim]);
    let prototype = transpose(graph, &prototype, &[0, 2, 3, 1]);
    let flat = reshape(graph, input, &[0, channels, -1]);
    let features = transpose(graph, &flat, &[0, 2, 1]);
    let query = linear(graph, &features, &layer.query)?;
    let query = reshape(graph, &query, &[0, 0, HEADS, head_dim]);
    let query = transpose(graph, &query, &[0, 2, 1, 3]);
    let affinity = graph.op("MatMul", &[&query, &prototype], vec![]);
    let scale = graph.scale((head_dim as f32).sqrt().recip());
    let affinity = graph.op("Mul", &[&affinity, &scale], vec![]);
    let axes = graph.integers(&[1], &[1]);
    let incidence = graph.op(
        "ReduceMean",
        &[&affinity, &axes],
        vec![Attribute::int("keepdims", 0)],
    );
    let incidence = transpose(graph, &incidence, &[0, 2, 1]);
    let incidence = graph.op("Softmax", &[&incidence], vec![Attribute::int("axis", -1)]);
    let edges = graph.op("MatMul", &[&incidence, &features], vec![]);
    let edges = linear(graph, &edges, &layer.edges)?;
    let edges = silu(graph, &edges);
    let incidence = transpose(graph, &incidence, &[0, 2, 1]);
    let vertices = graph.op("MatMul", &[&incidence, &edges], vec![]);
    let vertices = linear(graph, &vertices, &layer.vertices)?;
    let vertices = silu(graph, &vertices);
    let vertices = graph.op("Add", &[&features, &vertices], vec![]);
    let vertices = transpose(graph, &vertices, &[0, 2, 1]);
    let shape = graph.op("Shape", &[input], vec![]);
    Ok(graph.op("Reshape", &[&vertices, &shape], vec![]))
}

fn hypergraph_c3(graph: &mut Builder, layer: &HypergraphC3, input: &str) -> Result<String> {
    let features = activated(graph, &layer.second, input)?;
    let features = hypergraph(graph, &layer.graph, &features)?;
    let shortcut = activated(graph, &layer.first, input)?;
    let value = concat(graph, &[&features, &shortcut], 1);
    activated(graph, &layer.project, &value)
}

fn multiscale(graph: &mut Builder, layer: &MultiscaleFusion, encoded: &[String]) -> Result<String> {
    let reference = &encoded[2];
    let first = resize_like(graph, &encoded[0], reference);
    let second = resize_like(graph, &encoded[1], reference);
    let fourth = resize_like(graph, &encoded[3], reference);
    let value = concat(graph, &[&first, &second, reference, &fourth], 1);
    let value = activated(graph, &layer.inputs, &value)?;
    let [high, low, rest] = layer.channel_split;
    let high_features = slice(graph, &value, 1, 0, high);
    let low_features = slice(graph, &value, 1, high, high + low);
    let rest = slice(graph, &value, 1, high + low, high + low + rest);
    let high_first = hypergraph_c3(graph, &layer.high_order[0], &high_features)?;
    let high_second = hypergraph_c3(graph, &layer.high_order[1], &high_features)?;
    let high_features = concat(graph, &[&high_first, &high_second], 1);
    let high_features = activated(graph, &layer.high_fusion, &high_features)?;
    let low_features = depthwise(graph, &layer.low_order, &low_features)?;
    let combined = concat(graph, &[&high_features, &low_features, &rest], 1);
    activated(graph, &layer.output, &combined)
}

fn time_freq(graph: &mut Builder, block: &TimeFreqBlock, input: &str) -> Result<String> {
    let value = normalize(graph, &block.input.norm, input)?;
    let value = silu(graph, &value);
    let value = convolution(graph, &block.input.conv, &value)?;
    let distributed = normalize(graph, &block.down.norm, &value)?;
    let distributed = silu(graph, &distributed);
    let distributed = linear(graph, &distributed, &block.down.linear)?;
    let distributed = normalize(graph, &block.up.norm, &distributed)?;
    let distributed = silu(graph, &distributed);
    let distributed = linear(graph, &distributed, &block.up.linear)?;
    let value = graph.op("Add", &[&value, &distributed], vec![]);
    let value = normalize(graph, &block.output.norm, &value)?;
    let value = silu(graph, &value);
    let value = convolution(graph, &block.output.conv, &value)?;
    let residual = convolution(graph, &block.residual, input)?;
    Ok(graph.op("Add", &[&value, &residual], vec![]))
}

fn resize_like(graph: &mut Builder, input: &str, reference: &str) -> String {
    let shape = graph.op("Shape", &[input], vec![]);
    let batch_channels = slice(graph, &shape, 0, 0, 2);
    let shape = graph.op("Shape", &[reference], vec![]);
    let lattice = slice(graph, &shape, 0, 2, 4);
    let size = concat(graph, &[&batch_channels, &lattice], 0);
    resize(graph, input, &size)
}

fn resize(graph: &mut Builder, input: &str, size: &str) -> String {
    graph.op(
        "Resize",
        &[input, "", "", size],
        vec![
            Attribute::text("mode", "linear"),
            Attribute::text("coordinate_transformation_mode", "half_pixel"),
        ],
    )
}

fn slice(graph: &mut Builder, value: &str, axis: i64, start: i64, end: i64) -> String {
    let axes = graph.integers(&[1], &[axis]);
    let start = graph.integers(&[1], &[start]);
    let end = graph.integers(&[1], &[end]);
    graph.op("Slice", &[value, &start, &end, &axes], vec![])
}
