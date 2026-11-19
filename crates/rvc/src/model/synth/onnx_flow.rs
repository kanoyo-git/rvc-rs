use std::path::Path;

use anyhow::{Result, ensure};
use tch::Tensor;

use super::{Config, Weights};
use crate::model::onnx::{
    GraphSession, Provider, TensorInput,
    graph::{Attribute, Builder, Dimension, Layers},
};

pub(super) struct Flow(GraphSession);

impl Flow {
    pub(super) fn load(weights: &Weights, config: &Config, provider: Provider) -> Result<Self> {
        GraphSession::load(provider, "flow", |directory, path| {
            export(weights, config, directory, path)
        })
        .map(Self)
    }

    pub(super) fn reverse(&self, latent: &Tensor, speaker: &Tensor) -> Result<Tensor> {
        let dimensions = latent.size();
        let input = TensorInput::<f32>::new(latent)?;
        let speaker = TensorInput::<f32>::new(speaker)?;
        let mut session = self.0.session.borrow_mut();
        let output = session.run(ort::inputs![input.view()?, speaker.view()?])?;
        let (shape, data) = output[0].try_extract_tensor::<f32>()?;
        ensure!(
            &**shape == dimensions.as_slice(),
            "invalid ONNX flow output shape {shape:?}"
        );
        Ok(Tensor::f_from_slice(data)?
            .f_view(dimensions.as_slice())?
            .f_to_device(latent.device())?
            .f_to_kind(latent.kind())?)
    }
}

fn export(weights: &Weights, config: &Config, directory: &Path, path: &Path) -> Result<()> {
    let fixed = Dimension::fixed;
    let latent_shape = || vec![fixed(1), fixed(config.inter), Dimension::variable("frames")];
    let mut model = Layers {
        graph: Builder::new(directory, "RVC reverse flow", latent_shape())?,
        weights,
    };
    let speaker = weights.get("emb_g.weight")?.size()[1];
    model
        .graph
        .input("speaker", vec![fixed(1), fixed(speaker), fixed(1)]);
    let flip = model.graph.integers(
        &[config.inter],
        &(0..config.inter).rev().collect::<Vec<_>>(),
    );
    let mut latent = "input".to_owned();
    let half = config.inter / 2;
    // Inference supplies one unpadded sequence, so its flow mask is all ones.
    for layer in [6, 4, 2, 0] {
        let name = format!("flow.flows.{layer}");
        let flipped = model
            .graph
            .op("Gather", &[&latent, &flip], vec![Attribute::int("axis", 1)]);
        let first = channels(&mut model.graph, &flipped, 0, half);
        let second = channels(&mut model.graph, &flipped, half, config.inter);
        let hidden = model.conv(&first, &format!("{name}.pre"), 1, 0, 1)?;
        let width = weights.get(&format!("{name}.pre.weight"))?.size()[0];
        let sum = wavenet(&mut model, hidden, width, &format!("{name}.enc"))?;
        let mean = model.conv(&sum, &format!("{name}.post"), 1, 0, 1)?;
        let second = model.graph.op("Sub", &[&second, &mean], vec![]);
        latent = model.graph.op(
            "Concat",
            &[&first, &second],
            vec![Attribute::int("axis", 1)],
        );
    }
    model.graph.finish(path, latent, latent_shape())
}

fn channels(graph: &mut Builder, value: &str, start: i64, end: i64) -> String {
    let start = graph.integers(&[1], &[start]);
    let end = graph.integers(&[1], &[end]);
    let axis = graph.integers(&[1], &[1]);
    graph.op("Slice", &[value, &start, &end, &axis], vec![])
}

fn wavenet(model: &mut Layers<'_>, mut hidden: String, width: i64, name: &str) -> Result<String> {
    let condition = model.conv("speaker", &format!("{name}.cond_layer"), 1, 0, 1)?;
    let mut skips = Vec::with_capacity(3);
    for layer in 0..3 {
        let input = model.conv(&hidden, &format!("{name}.in_layers.{layer}"), 1, 2, 1)?;
        let offset = 2 * width * layer;
        let cond = channels(&mut model.graph, &condition, offset, offset + 2 * width);
        let input = model.graph.op("Add", &[&input, &cond], vec![]);
        let a = channels(&mut model.graph, &input, 0, width);
        let b = channels(&mut model.graph, &input, width, 2 * width);
        let a = model.graph.op("Tanh", &[&a], vec![]);
        let b = model.graph.op("Sigmoid", &[&b], vec![]);
        let gate = model.graph.op("Mul", &[&a, &b], vec![]);
        let skip = model.conv(&gate, &format!("{name}.res_skip_layers.{layer}"), 1, 0, 1)?;
        if layer == 2 {
            skips.push(skip);
        } else {
            let residual = channels(&mut model.graph, &skip, 0, width);
            hidden = model.graph.op("Add", &[&hidden, &residual], vec![]);
            skips.push(channels(&mut model.graph, &skip, width, 2 * width));
        }
    }
    let sum = model.graph.op("Add", &[&skips[0], &skips[1]], vec![]);
    Ok(model.graph.op("Add", &[&sum, &skips[2]], vec![]))
}
