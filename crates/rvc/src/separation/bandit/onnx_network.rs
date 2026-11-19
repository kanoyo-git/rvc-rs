use std::path::Path;

use anyhow::{Context, Result, ensure};
use tch::Tensor;

use super::{
    Activation, Affine, MaskConfig, ResidualRnn, RnnType, Spectral, TorchNetwork, Version,
};
use crate::model::onnx::{
    GraphSession, Provider, TensorInput,
    graph::{Attribute, Builder, Dimension},
};
use crate::separation::onnx_layers::{
    Recurrent, concat, concat_strings, dimensions, glu, linear, recurrent, reshape, select, slice,
    swish, transpose, unsqueeze,
};

pub(super) struct Network {
    spectral: Spectral,
    version: Version,
    stems: usize,
    components: i64,
    model: GraphSession,
}

impl Network {
    pub(super) fn load(torch: TorchNetwork, provider: Provider) -> Result<Self> {
        let components = 1 + i64::from(torch.mask_config.complex);
        let model = GraphSession::load(provider, "bandit", |directory, path| {
            export(&torch, components, directory, path)
        })?;
        Ok(Self {
            spectral: torch.spectral,
            version: torch.version,
            stems: torch.masks.len(),
            components,
            model,
        })
    }

    pub(super) fn run(&self, audio: &[f32], count: usize, samples: i64) -> Result<Vec<f32>> {
        let input = Tensor::f_from_slice(audio)?.f_reshape([-1, samples])?;
        let spectrum = self.spectral.stft(&input)?;
        let (planes, bins, frames) = spectrum.size3()?;
        let order = match self.version {
            Version::V1 => [0, 2, 3, 1],
            Version::V2 => [0, 2, 1, 3],
        };
        let source = TensorInput::<f32>::new(&spectrum.f_view_as_real()?.f_permute(order)?)?;
        let mut session = self.model.session.borrow_mut();
        let output = session.run(ort::inputs![source.view()?])?;
        let expected = [planes, bins, frames, self.components];
        let masks = (0..self.stems).map(|stem| {
            let (shape, values) = output[stem].try_extract_tensor::<f32>()?;
            ensure!(
                **shape == expected,
                "invalid Bandit ONNX mask shape {shape:?}"
            );
            let mask = Tensor::f_from_slice(values)?.f_view(expected)?;
            if self.components == 2 {
                Ok(mask.f_view_as_complex()?)
            } else {
                Ok(mask.f_squeeze_dim(-1)?)
            }
        });
        self.spectral.render(&spectrum, count, samples, masks)
    }
}

fn export(model: &TorchNetwork, components: i64, directory: &Path, path: &Path) -> Result<()> {
    let bins = model.spectral.fft / 2 + 1;
    let (axis, width, parts) = match model.version {
        Version::V1 => (3, 2, bins),
        Version::V2 => (2, bins, 2),
    };
    let mut graph = Builder::new(
        directory,
        "PyMSS Bandit",
        vec![
            Dimension::variable("channels"),
            Dimension::variable("frames"),
            Dimension::fixed(width),
            Dimension::fixed(parts),
        ],
    )?;
    let mut bands = Vec::with_capacity(model.projections.len());
    for band in &model.projections {
        let value = slice(&mut graph, "input", axis, band.band.start, band.band.end, 1);
        let value = reshape(&mut graph, &value, &[0, 0, 2 * band.band.width()]);
        let value = normalize(&mut graph, &band.norm, &value)?;
        let value = affine(&mut graph, &band.linear, &value)?;
        bands.push(unsqueeze(&mut graph, &value, &[1]));
    }
    let mut features = concat_strings(&mut graph, &bands, 1);
    for layer in &model.recurrent {
        features = residual_rnn(&mut graph, layer, &features)?;
        features = transpose(&mut graph, &features, &[0, 2, 1, 3]);
    }

    let overlap = model.bands.overlap.then(|| {
        let indices = model
            .bands
            .ranges
            .iter()
            .flat_map(|band| band.start..band.end)
            .collect::<Vec<_>>();
        graph.integers(&[indices.len() as i64, 1], &indices)
    });
    let planes_frames = dimensions(&mut graph, "input", 0, 2);
    let bin_dim = graph.integers(&[1], &[bins]);
    let width_dim = graph.integers(&[1], &[components]);
    let output_shape = concat(&mut graph, &[&bin_dim, &planes_frames, &width_dim], 0);
    let mut predictions = Vec::with_capacity(model.masks.len());
    for stem in &model.masks {
        let mut local_masks = Vec::with_capacity(stem.bands.len());
        for (index, band) in stem.bands.iter().enumerate() {
            let input = select(&mut graph, &features, 1, index as i64);
            let value = normalize(&mut graph, &band.norm, &input)?;
            let value = affine(&mut graph, &band.hidden, &value)?;
            let value = activation(&mut graph, &model.mask_config, value);
            let value = affine(&mut graph, &band.output, &value)?;
            let value = glu(&mut graph, &value, -1, components * band.width);
            let mut value = reshape(&mut graph, &value, &[0, 0, band.width, components]);
            if let Some(weights) = &stem.frequencies[index] {
                let scale = graph.tensor(&weights.f_reshape([1, 1, band.width, 1])?)?;
                value = graph.op("Mul", &[&value, &scale], vec![]);
            }
            local_masks.push(value);
        }
        let value = concat_strings(&mut graph, &local_masks, 2);
        let value = if let Some(indices) = &overlap {
            let updates = transpose(&mut graph, &value, &[2, 0, 1, 3]);
            let empty = graph.op("ConstantOfShape", &[&output_shape], vec![]);
            let value = graph.op(
                "ScatterND",
                &[&empty, indices, &updates],
                vec![Attribute::text("reduction", "add")],
            );
            transpose(&mut graph, &value, &[1, 0, 2, 3])
        } else {
            transpose(&mut graph, &value, &[0, 2, 1, 3])
        };
        predictions.push(value);
    }
    let last = predictions
        .pop()
        .context("Bandit needs at least one stem")?;
    let shape = || {
        vec![
            Dimension::variable("channels"),
            Dimension::fixed(bins),
            Dimension::variable("frames"),
            Dimension::fixed(components),
        ]
    };
    for value in predictions {
        graph.output(value, shape());
    }
    graph.finish(path, last, shape())
}

fn residual_rnn(graph: &mut Builder, layer: &ResidualRnn, input: &str) -> Result<String> {
    let kind = match layer.rnn_type {
        RnnType::Lstm => Recurrent::Lstm,
        RnnType::Gru => Recurrent::Gru,
        RnnType::Tanh => Recurrent::Tanh,
    };
    let embedding = layer.norm.bias.size()[0];
    let step_count = dimensions(graph, input, 2, 3);
    let array_shape = dimensions(graph, input, 0, 4);
    let inferred = graph.integers(&[1], &[-1]);
    let columns = graph.integers(&[1], &[embedding]);
    let sequence_shape = concat(graph, &[&inferred, &step_count, &columns], 0);
    let value = normalize(graph, &layer.norm, input)?;
    let value = graph.op("Reshape", &[&value, &sequence_shape], vec![]);
    let value = recurrent(
        graph,
        &value,
        &layer.params,
        layer.hidden,
        layer.bidirectional,
        kind,
    )?;
    let value = affine(graph, &layer.output, &value)?;
    let value = graph.op("Reshape", &[&value, &array_shape], vec![]);
    Ok(graph.op("Add", &[input, &value], vec![]))
}

fn normalize(graph: &mut Builder, layer: &Affine, input: &str) -> Result<String> {
    let scale = graph.tensor(&layer.weight)?;
    let bias = graph.tensor(&layer.bias)?;
    Ok(graph.op(
        "LayerNormalization",
        &[input, &scale, &bias],
        vec![Attribute::float("epsilon", 1e-5)],
    ))
}

fn affine(graph: &mut Builder, layer: &Affine, input: &str) -> Result<String> {
    linear(graph, input, &layer.weight, Some(&layer.bias))
}

fn activation(graph: &mut Builder, config: &MaskConfig, value: String) -> String {
    let (op, attrs) = match &config.activation {
        Activation::Tanh => ("Tanh", vec![]),
        Activation::Relu => ("Relu", vec![]),
        Activation::Silu => return swish(graph, &value),
        Activation::Sigmoid => ("Sigmoid", vec![]),
        Activation::Mish => ("Mish", vec![]),
        Activation::Gelu(mode) => ("Gelu", vec![Attribute::text("approximate", mode)]),
        Activation::LeakyRelu(slope) => {
            ("LeakyRelu", vec![Attribute::float("alpha", *slope as f32)])
        }
        Activation::Elu(alpha) => ("Elu", vec![Attribute::float("alpha", *alpha as f32)]),
        Activation::Identity => return value,
        Activation::Relu6 => {
            let low = graph.scale(0.);
            let high = graph.scale(6.);
            return graph.op("Clip", &[&value, &low, &high], vec![]);
        }
    };
    graph.op(op, &[&value], attrs)
}
