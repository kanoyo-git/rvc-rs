use std::path::Path;

use anyhow::{Context, Result, ensure};
use tch::Tensor;

use super::{Attention, Linear, MaskMlp, MaskMode, TorchNetwork, Transformer};
use crate::model::onnx::{
    GraphSession, Provider, TensorInput,
    graph::{Attribute, Builder, Dimension},
};

pub(super) struct Network {
    model: GraphSession,
    stems: i64,
    features: i64,
}

#[derive(Clone, Copy)]
struct Precision {
    half: bool,
}

impl Precision {
    fn tensor(self, graph: &mut Builder, weight: &Tensor) -> Result<String> {
        if self.half {
            graph.half_tensor(weight)
        } else {
            graph.tensor(weight)
        }
    }

    fn cast(self, graph: &mut Builder, value: &str, to_half: bool) -> String {
        if self.half {
            graph.op(
                "Cast",
                &[value],
                vec![Attribute::int("to", if to_half { 10 } else { 1 })],
            )
        } else {
            value.to_owned()
        }
    }

    fn scale(self, graph: &mut Builder, value: f32) -> Result<String> {
        if self.half {
            graph.half_tensor(&Tensor::f_from_slice(&[value])?.f_reshape([])?)
        } else {
            Ok(graph.scale(value))
        }
    }

    fn linear(self, graph: &mut Builder, value: &str, layer: &Linear) -> Result<String> {
        let weight = self.tensor(graph, &layer.weight.f_transpose(0, 1)?)?;
        let value = graph.op("MatMul", &[value, &weight], vec![]);
        if let Some(bias) = &layer.bias {
            let bias = self.tensor(graph, bias)?;
            Ok(graph.op("Add", &[&value, &bias], vec![]))
        } else {
            Ok(value)
        }
    }
}

impl Network {
    pub(super) fn load(
        provider: Provider,
        torch: &TorchNetwork,
        dim: i64,
        half: bool,
    ) -> Result<Self> {
        let stems = torch.masks.len() as i64;
        let features = torch.projections.iter().map(|band| band.width).sum();
        let model = GraphSession::load(provider, "roformer", |directory, path| {
            export(
                torch,
                dim,
                features,
                stems,
                Precision { half },
                directory,
                path,
            )
        })?;
        Ok(Self {
            model,
            stems,
            features,
        })
    }

    pub(super) fn estimate(&self, input: &Tensor) -> Result<Tensor> {
        let (batch, frames, _) = input.size3()?;
        let expected = [batch, self.stems, frames, self.features];
        let input = TensorInput::<f32>::new(input)?;
        let mut model = self.model.session.borrow_mut();
        let output = model.run(ort::inputs![input.view()?])?;
        let (shape, values) = output[0].try_extract_tensor::<f32>()?;
        ensure!(
            **shape == expected,
            "invalid RoFormer ONNX mask shape {shape:?}"
        );
        Ok(Tensor::f_from_slice(values)?.f_view(expected)?)
    }
}

fn export(
    model: &TorchNetwork,
    dim: i64,
    features: i64,
    stems: i64,
    precision: Precision,
    directory: &Path,
    path: &Path,
) -> Result<()> {
    let mut graph = Builder::new(
        directory,
        "PyMSS RoFormer",
        vec![
            Dimension::variable("batch"),
            Dimension::variable("frames"),
            Dimension::fixed(features),
        ],
    )?;
    let input = precision.cast(&mut graph, "input", true);
    let mut bands = Vec::with_capacity(model.projections.len());
    let mut offset = 0;
    for band in &model.projections {
        let slice = slice(&mut graph, &input, offset, offset + band.width, 1);
        let value = normalize(&mut graph, &slice, &band.gamma, precision)?;
        let value = precision.linear(&mut graph, &value, &band.linear)?;
        bands.push(unsqueeze(&mut graph, &value, &[1]));
        offset += band.width;
    }
    let band_count = bands.len() as i64;
    let shape = graph.op("Shape", &["input"], vec![]);
    let batch = dimension(&mut graph, &shape, 0, &[1]);
    let frames = dimension(&mut graph, &shape, 1, &[1]);
    let bands_dim = graph.integers(&[1], &[band_count]);
    let channel_dim = graph.integers(&[1], &[dim]);
    let inferred = graph.integers(&[1], &[-1]);
    let band_first = concat(&mut graph, &[&batch, &bands_dim, &frames, &channel_dim], 0);
    let time_first = concat(&mut graph, &[&batch, &frames, &bands_dim, &channel_dim], 0);
    let time_input = concat(&mut graph, &[&inferred, &frames, &channel_dim], 0);
    let freq_input = graph.integers(&[3], &[-1, band_count, dim]);
    let mut value = concat_strings(&mut graph, &bands, 1);

    for [time, frequency] in &model.blocks {
        value = graph.op("Reshape", &[&value, &time_input], vec![]);
        value = transformer(&mut graph, time, value, precision)?;
        value = graph.op("Reshape", &[&value, &band_first], vec![]);
        value = transpose(&mut graph, &value, &[0, 2, 1, 3]);
        value = graph.op("Reshape", &[&value, &freq_input], vec![]);
        value = transformer(&mut graph, frequency, value, precision)?;
        value = graph.op("Reshape", &[&value, &time_first], vec![]);
        value = transpose(&mut graph, &value, &[0, 2, 1, 3]);
    }
    if let Some(gamma) = &model.final_gamma {
        value = normalize(&mut graph, &value, gamma, precision)?;
    }
    let embeddings = if model.mask_mode == MaskMode::SegmentationOnly {
        Vec::new()
    } else {
        (0..band_count)
            .map(|index| select(&mut graph, &value, 1, index))
            .collect()
    };
    let segmentations = model
        .segmentation
        .as_ref()
        .map(|models| {
            let features = transpose(&mut graph, &value, &[0, 3, 2, 1]);
            models
                .iter()
                .map(|model| super::hyperace::onnx_network::forward(&mut graph, model, &features))
                .collect::<Result<Vec<_>>>()
        })
        .transpose()?;
    let mut output = Vec::with_capacity(model.masks.len());
    if model.mask_mode == MaskMode::SegmentationOnly {
        for mask in segmentations.context("missing HyperACE segmentation masks")? {
            output.push(unsqueeze(&mut graph, &mask, &[1]));
        }
    } else {
        for (stem, estimators) in model.masks.iter().enumerate() {
            let masks = estimators
                .iter()
                .zip(&embeddings)
                .map(|(estimator, input)| mask(&mut graph, estimator, input, precision))
                .collect::<Result<Vec<_>>>()?;
            let mut mask = concat_strings(&mut graph, &masks, -1);
            if let Some(segmentation) = &segmentations {
                mask = graph.op("Add", &[&mask, &segmentation[stem]], vec![]);
            }
            output.push(unsqueeze(&mut graph, &mask, &[1]));
        }
    }
    let output = concat_strings(&mut graph, &output, 1);
    let output = precision.cast(&mut graph, &output, false);
    graph.finish(
        path,
        output,
        vec![
            Dimension::variable("batch"),
            Dimension::fixed(stems),
            Dimension::variable("frames"),
            Dimension::fixed(features),
        ],
    )
}

fn transformer(
    graph: &mut Builder,
    layer: &Transformer,
    mut value: String,
    precision: Precision,
) -> Result<String> {
    for block in &layer.blocks {
        let attended = attention(graph, &block.attention, &value, precision)?;
        let residual = graph.op("Add", &[&value, &attended], vec![]);
        let expanded = normalize(graph, &residual, &block.gamma, precision)?;
        let expanded = precision.linear(graph, &expanded, &block.expand)?;
        let activated = graph.op("Gelu", &[&expanded], vec![]);
        let contracted = precision.linear(graph, &activated, &block.contract)?;
        value = graph.op("Add", &[&residual, &contracted], vec![]);
    }
    if let Some(gamma) = &layer.gamma {
        value = normalize(graph, &value, gamma, precision)?;
    }
    Ok(value)
}

fn attention(
    graph: &mut Builder,
    layer: &Attention,
    input: &str,
    precision: Precision,
) -> Result<String> {
    let input = normalize(graph, input, &layer.gamma, precision)?;
    let qkv = precision.linear(graph, &input, &layer.qkv)?;
    let qkv = reshape(graph, &qkv, &[0, 0, 3, layer.heads, layer.head_dim]);
    let shape = graph.op("Shape", &[&input], vec![]);
    let frames = dimension(graph, &shape, 1, &[]);
    let zero = graph.integers(&[], &[0]);
    let one = graph.integers(&[], &[1]);
    let positions = graph.op("Range", &[&zero, &frames, &one], vec![]);
    let positions = graph.op("Cast", &[&positions], vec![Attribute::int("to", 1)]);
    let positions = unsqueeze(graph, &positions, &[0, 2, 3]);
    let frequencies = graph.tensor(&layer.frequencies)?;
    let angles = graph.op("Mul", &[&positions, &frequencies], vec![]);
    let cosine = graph.op("Cos", &[&angles], vec![]);
    let sine = graph.op("Sin", &[&angles], vec![]);
    let cosine = precision.cast(graph, &cosine, true);
    let sine = precision.cast(graph, &sine, true);

    let query = select(graph, &qkv, 2, 0);
    let query = rotate(graph, &query, &cosine, &sine, layer.head_dim);
    let query = transpose(graph, &query, &[0, 2, 1, 3]);
    let key = select(graph, &qkv, 2, 1);
    let key = rotate(graph, &key, &cosine, &sine, layer.head_dim);
    let key = transpose(graph, &key, &[0, 2, 3, 1]);
    let value = select(graph, &qkv, 2, 2);
    let value = transpose(graph, &value, &[0, 2, 1, 3]);
    let scale = precision.scale(graph, (layer.head_dim as f32).sqrt().recip())?;
    let similarity = graph.op("MatMul", &[&query, &key], vec![]);
    let similarity = graph.op("Mul", &[&similarity, &scale], vec![]);
    let probability = graph.op("Softmax", &[&similarity], vec![Attribute::int("axis", -1)]);
    let attended = graph.op("MatMul", &[&probability, &value], vec![]);
    let attended = transpose(graph, &attended, &[0, 2, 1, 3]);
    let gate = precision.linear(graph, &input, &layer.gates)?;
    let gate = unsqueeze(graph, &gate, &[-1]);
    let gate = graph.op("Sigmoid", &[&gate], vec![]);
    let attended = graph.op("Mul", &[&attended, &gate], vec![]);
    let attended = reshape(graph, &attended, &[0, 0, layer.heads * layer.head_dim]);
    precision.linear(graph, &attended, &layer.output)
}

fn rotate(graph: &mut Builder, value: &str, cos: &str, sin: &str, dim: i64) -> String {
    let even = slice(graph, value, 0, dim, 2);
    let odd = slice(graph, value, 1, dim, 2);
    let even_cos = graph.op("Mul", &[&even, cos], vec![]);
    let odd_sin = graph.op("Mul", &[&odd, sin], vec![]);
    let odd_cos = graph.op("Mul", &[&odd, cos], vec![]);
    let even_sin = graph.op("Mul", &[&even, sin], vec![]);
    let real = graph.op("Sub", &[&even_cos, &odd_sin], vec![]);
    let imaginary = graph.op("Add", &[&odd_cos, &even_sin], vec![]);
    let real = unsqueeze(graph, &real, &[-1]);
    let imaginary = unsqueeze(graph, &imaginary, &[-1]);
    let interleaved = concat(graph, &[&real, &imaginary], -1);
    reshape(graph, &interleaved, &[0, 0, 0, dim])
}

fn mask(
    graph: &mut Builder,
    estimator: &MaskMlp,
    input: &str,
    precision: Precision,
) -> Result<String> {
    let (last, hidden) = estimator
        .0
        .split_last()
        .context("empty RoFormer mask estimator")?;
    let mut value = input.to_owned();
    for layer in hidden {
        value = precision.linear(graph, &value, layer)?;
        value = graph.op("Tanh", &[&value], vec![]);
    }
    let value = precision.linear(graph, &value, last)?;
    let width = last.weight.size()[0] / 2;
    let activation = slice(graph, &value, 0, width, 1);
    let gate = slice(graph, &value, width, 2 * width, 1);
    let gate = graph.op("Sigmoid", &[&gate], vec![]);
    Ok(graph.op("Mul", &[&activation, &gate], vec![]))
}

fn normalize(
    graph: &mut Builder,
    value: &str,
    gamma: &Tensor,
    precision: Precision,
) -> Result<String> {
    let dim = gamma.size()[0] as f32;
    let gamma = precision.tensor(graph, gamma)?;
    // The normalization floor is smaller than the least positive FP16 value.
    let value = precision.cast(graph, value, false);
    let axes = graph.integers(&[1], &[-1]);
    let epsilon = graph.scale(1e-12);
    let scale = graph.scale(dim.sqrt());
    let norm = graph.op("ReduceL2", &[&value, &axes], vec![]);
    let norm = graph.op("Max", &[&norm, &epsilon], vec![]);
    let value = graph.op("Div", &[&value, &norm], vec![]);
    let value = graph.op("Mul", &[&value, &scale], vec![]);
    let value = precision.cast(graph, &value, true);
    Ok(graph.op("Mul", &[&value, &gamma], vec![]))
}

pub(super) fn linear(graph: &mut Builder, value: &str, layer: &Linear) -> Result<String> {
    Precision { half: false }.linear(graph, value, layer)
}

pub(super) fn transpose(graph: &mut Builder, value: &str, axes: &[i64]) -> String {
    graph.op("Transpose", &[value], vec![Attribute::ints("perm", axes)])
}

fn unsqueeze(graph: &mut Builder, value: &str, axes: &[i64]) -> String {
    let axes = graph.integers(&[axes.len() as i64], axes);
    graph.op("Unsqueeze", &[value, &axes], vec![])
}

pub(super) fn reshape(graph: &mut Builder, value: &str, shape: &[i64]) -> String {
    let shape = graph.integers(&[shape.len() as i64], shape);
    graph.op("Reshape", &[value, &shape], vec![])
}

fn select(graph: &mut Builder, value: &str, axis: i64, index: i64) -> String {
    let index = graph.integers(&[], &[index]);
    graph.op(
        "Gather",
        &[value, &index],
        vec![Attribute::int("axis", axis)],
    )
}

fn dimension(graph: &mut Builder, shape: &str, index: i64, rank: &[i64]) -> String {
    let index = graph.integers(rank, &[index]);
    graph.op("Gather", &[shape, &index], vec![])
}

fn slice(graph: &mut Builder, value: &str, start: i64, end: i64, step: i64) -> String {
    let start = graph.integers(&[1], &[start]);
    let end = graph.integers(&[1], &[end]);
    let axis = graph.integers(&[1], &[-1]);
    let step = graph.integers(&[1], &[step]);
    graph.op("Slice", &[value, &start, &end, &axis, &step], vec![])
}

pub(super) fn concat(graph: &mut Builder, values: &[&str], axis: i64) -> String {
    graph.op("Concat", values, vec![Attribute::int("axis", axis)])
}

fn concat_strings(graph: &mut Builder, values: &[String], axis: i64) -> String {
    let values = values.iter().map(String::as_str).collect::<Vec<_>>();
    concat(graph, &values, axis)
}
