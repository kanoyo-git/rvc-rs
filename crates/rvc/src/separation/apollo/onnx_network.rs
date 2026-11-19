use std::path::Path;

use anyhow::{Result, ensure};
use tch::Tensor;

use super::{
    Attention, BANDS, HEADS, PackedProjection, Pointwise, Projection, RMS_EPSILON, Spectral,
    TorchNetwork,
};
use crate::model::onnx::{
    GraphSession, Provider, TensorInput,
    graph::{Attribute, Builder, Dimension},
};
use crate::separation::onnx_layers::{
    concat, convolution, dimensions, glu, reduce, reshape, select, slice, swish, transpose,
    unsqueeze,
};

pub(super) struct Network {
    pub(super) spectral: Spectral,
    model: GraphSession,
}

impl Network {
    pub(super) fn load(torch: TorchNetwork, provider: Provider) -> Result<Self> {
        let model = GraphSession::load(provider, "apollo", |directory, path| {
            export(&torch, directory, path)
        })?;
        Ok(Self {
            spectral: torch.spectral,
            model,
        })
    }

    pub(super) fn estimate(&self, prefix: Tensor, tail: Tensor) -> Result<Tensor> {
        let (batch, _, _, frames) = prefix.size4()?;
        let expected = [batch, 2, self.spectral.fft / 2 + 1, frames];
        let prefix = TensorInput::<f32>::new(&prefix)?;
        let tail = TensorInput::<f32>::new(&tail)?;
        let mut session = self.model.session.borrow_mut();
        let result = session.run(ort::inputs![prefix.view()?, tail.view()?])?;
        let (shape, values) = result[0].try_extract_tensor::<f32>()?;
        ensure!(
            **shape == expected,
            "invalid Apollo ONNX spectrum {shape:?}"
        );
        Ok(Tensor::f_from_slice(values)?.f_view(expected)?)
    }
}

fn export(model: &TorchNetwork, directory: &Path, path: &Path) -> Result<()> {
    let Spectral {
        fft,
        width,
        tail_width,
        ..
    } = model.spectral;
    let features = model.tail_input.linear.weight.size()[0];
    let mut graph = Builder::new(
        directory,
        "PyMSS Apollo",
        vec![
            Dimension::variable("channels"),
            Dimension::fixed(BANDS - 1),
            Dimension::fixed(2 * width + 1),
            Dimension::variable("frames"),
        ],
    )?;
    graph.input(
        "tail",
        vec![
            Dimension::variable("channels"),
            Dimension::fixed(2 * tail_width + 1),
            Dimension::variable("frames"),
        ],
    );
    let prefix = packed_projection(&mut graph, &model.prefix_input, "input")?;
    let tail = projection(&mut graph, &model.tail_input, "tail")?;
    let tail = unsqueeze(&mut graph, &tail, &[1]);
    let mut value = concat(&mut graph, &[&prefix, &tail], 1);
    let batch = dimensions(&mut graph, "input", 0, 1);
    let frames = dimensions(&mut graph, "input", 3, 4);
    let inferred = graph.integers(&[1], &[-1]);
    let geometry = graph.integers(&[2], &[BANDS, features]);
    let flipped = graph.integers(&[2], &[features, BANDS]);
    let feature_count = graph.integers(&[1], &[features]);
    let along_band = concat(&mut graph, &[&batch, &frames, &flipped], 0);
    let along_time = concat(&mut graph, &[&inferred, &feature_count, &frames], 0);
    let stage_output = concat(&mut graph, &[&batch, &geometry, &frames], 0);
    for layer in &model.network {
        value = transpose(&mut graph, &value, &[0, 3, 2, 1]);
        value = reshape(&mut graph, &value, &[-1, features, BANDS]);
        value = attention(&mut graph, &layer.attention, &value)?;
        value = graph.op("Reshape", &[&value, &along_band], vec![]);
        value = transpose(&mut graph, &value, &[0, 3, 2, 1]);
        value = graph.op("Reshape", &[&value, &along_time], vec![]);
        for block in &layer.sequence {
            let convolved = convolution(
                &mut graph,
                "Conv",
                &value,
                &block.conv,
                Some(&block.bias),
                vec![
                    Attribute::ints("pads", &[3, 3]),
                    Attribute::int("group", block.features),
                ],
            )?;
            let normalized = rms(&mut graph, &convolved, &block.norm, 1)?;
            let expanded = pointwise(&mut graph, &block.expand, &normalized)?;
            let expanded = swish(&mut graph, &expanded);
            let contracted = pointwise(&mut graph, &block.contract, &expanded)?;
            value = graph.op("Add", &[&value, &contracted], vec![]);
        }
        value = graph.op("Reshape", &[&value, &stage_output], vec![]);
    }
    let prefix = slice(&mut graph, &value, 1, 0, BANDS - 1, 1);
    let prefix = packed_projection(&mut graph, &model.prefix_output, &prefix)?;
    let prefix = glu(&mut graph, &prefix, 2, 2 * width);
    let prefix = reshape(&mut graph, &prefix, &[0, BANDS - 1, 2, width, -1]);
    let prefix = transpose(&mut graph, &prefix, &[0, 2, 1, 3, 4]);
    let prefix = reshape(&mut graph, &prefix, &[0, 2, (BANDS - 1) * width, -1]);
    let tail = select(&mut graph, &value, 1, BANDS - 1);
    let tail = projection(&mut graph, &model.tail_output, &tail)?;
    let tail = glu(&mut graph, &tail, 1, 2 * tail_width);
    let tail = reshape(&mut graph, &tail, &[0, 2, tail_width, -1]);
    let output = concat(&mut graph, &[&prefix, &tail], 2);
    graph.finish(
        path,
        output,
        vec![
            Dimension::variable("channels"),
            Dimension::fixed(2),
            Dimension::fixed(fft / 2 + 1),
            Dimension::variable("frames"),
        ],
    )
}

fn rms(graph: &mut Builder, value: &str, norm: &Tensor, axis: i64) -> Result<String> {
    let squared = graph.op("Mul", &[value, value], vec![]);
    let mean = reduce(graph, "ReduceMean", &squared, &[axis], true);
    let eps = graph.scale(RMS_EPSILON as f32);
    let std = graph.op("Add", &[&mean, &eps], vec![]);
    let std = graph.op("Sqrt", &[&std], vec![]);
    let value = graph.op("Div", &[value, &std], vec![]);
    let norm = graph.tensor(&norm.f_reshape(if axis == 1 {
        vec![1, -1, 1]
    } else {
        norm.size()
    })?)?;
    Ok(graph.op("Mul", &[&value, &norm], vec![]))
}

fn pointwise(graph: &mut Builder, layer: &Pointwise, input: &str) -> Result<String> {
    convolution(
        graph,
        "Conv",
        input,
        &layer.weight.f_unsqueeze(2)?,
        layer.bias.as_ref(),
        vec![],
    )
}

fn projection(graph: &mut Builder, layer: &Projection, input: &str) -> Result<String> {
    let value = rms(graph, input, &layer.norm, 1)?;
    pointwise(graph, &layer.linear, &value)
}

fn packed_projection(graph: &mut Builder, layer: &PackedProjection, input: &str) -> Result<String> {
    let value = rms(graph, input, &layer.norm, 2)?;
    let value = reshape(graph, &value, &[0, (BANDS - 1) * layer.norm.size()[2], -1]);
    let value = convolution(
        graph,
        "Conv",
        &value,
        &layer.weight,
        Some(&layer.bias),
        vec![Attribute::int("group", BANDS - 1)],
    )?;
    Ok(reshape(
        graph,
        &value,
        &[0, BANDS - 1, layer.bias.size()[0] / (BANDS - 1), -1],
    ))
}

fn attention(graph: &mut Builder, layer: &Attention, input: &str) -> Result<String> {
    let dim = layer.head_dim;
    let normalized = rms(graph, input, &layer.norm, 1)?;
    let qkv = pointwise(graph, &layer.qkv, &normalized)?;
    let qkv = reshape(graph, &qkv, &[0, HEADS, dim * 3, BANDS]);
    let qkv = transpose(graph, &qkv, &[0, 1, 3, 2]);
    let cosine = graph.tensor(&layer.cosine)?;
    let sine = graph.tensor(&layer.sine)?;
    let swaps = (0..dim).map(|index| index ^ 1).collect::<Vec<_>>();
    let swaps = graph.integers(&[dim], &swaps);
    let signs = graph.tensor(&layer.reverse.f_repeat([dim / 2])?)?;
    let rotate = |graph: &mut Builder, value: &str| {
        let reverse = graph.op("Gather", &[value, &swaps], vec![Attribute::int("axis", -1)]);
        let reverse = graph.op("Mul", &[&reverse, &signs], vec![]);
        let cosine_term = graph.op("Mul", &[value, &cosine], vec![]);
        let sine_term = graph.op("Mul", &[&reverse, &sine], vec![]);
        graph.op("Add", &[&cosine_term, &sine_term], vec![])
    };
    let query = slice(graph, &qkv, -1, 0, dim, 1);
    let query = rotate(graph, &query);
    let key = slice(graph, &qkv, -1, dim, 2 * dim, 1);
    let key = rotate(graph, &key);
    let key = transpose(graph, &key, &[0, 1, 3, 2]);
    let value = slice(graph, &qkv, -1, 2 * dim, 3 * dim, 1);
    let similarity = graph.op("MatMul", &[&query, &key], vec![]);
    let scale = graph.scale((dim as f32).sqrt().recip());
    let scaled = graph.op("Mul", &[&similarity, &scale], vec![]);
    let weights = graph.op("Softmax", &[&scaled], vec![Attribute::int("axis", -1)]);
    let attended = graph.op("MatMul", &[&weights, &value], vec![]);
    let attended = transpose(graph, &attended, &[0, 1, 3, 2]);
    let attended = reshape(graph, &attended, &[0, HEADS * dim, BANDS]);
    let output = pointwise(graph, &layer.projection, &attended)?;
    let output = graph.op("Add", &[&output, input], vec![]);
    let inner = projection(graph, &layer.mlp, &output)?;
    let inner = swish(graph, &inner);
    let width = layer.mlp.linear.weight.size()[0] / 2;
    let left = slice(graph, &inner, 1, 0, width, 1);
    let left = swish(graph, &left);
    let right = slice(graph, &inner, 1, width, 2 * width, 1);
    let gated = graph.op("Mul", &[&left, &right], vec![]);
    let projected = pointwise(graph, &layer.mlp_output, &gated)?;
    Ok(graph.op("Add", &[&output, &projected], vec![]))
}
