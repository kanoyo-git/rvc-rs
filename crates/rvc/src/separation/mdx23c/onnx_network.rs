use std::path::Path;

use anyhow::{Result, ensure};
use tch::Tensor;

use crate::model::onnx::{
    GraphSession, Provider, TensorInput,
    graph::{Attribute, Builder, Dimension},
};

use super::{Activation, Block, ModelConfig, Norm, Normalized, TorchNetwork};

pub(super) struct Network {
    graph: GraphSession,
    pub(super) window: Tensor,
    stems: i64,
}

impl Network {
    pub(super) fn load(
        config: &ModelConfig,
        torch: &TorchNetwork,
        provider: Provider,
    ) -> Result<Self> {
        let graph = GraphSession::load(provider, "mdx23c", |directory, path| {
            export(config, torch, directory, path)
        })?;
        Ok(Self {
            graph,
            window: torch.window.shallow_clone(),
            stems: config.stems() as i64,
        })
    }

    pub(super) fn run(&self, spectrum: &Tensor) -> Result<Tensor> {
        let mut expected = spectrum.size();
        expected[1] *= self.stems;
        let input = TensorInput::<f32>::new(spectrum)?;
        let mut session = self.graph.session.borrow_mut();
        let output = session.run(ort::inputs![input.view()?])?;
        let (shape, data) = output[0].try_extract_tensor::<f32>()?;
        ensure!(
            &**shape == expected.as_slice(),
            "invalid ONNX MDX23C output shape {shape:?}"
        );
        Ok(Tensor::f_from_slice(data)?.f_view(expected.as_slice())?)
    }
}

fn export(config: &ModelConfig, model: &TorchNetwork, directory: &Path, path: &Path) -> Result<()> {
    let fixed = Dimension::fixed;
    let channels = 2 * config.audio.num_channels * config.model.num_subbands;
    let frequency = config.audio.dim_f / config.model.num_subbands;
    let shape = |channels| {
        vec![
            Dimension::variable("batch"),
            fixed(channels),
            fixed(frequency),
            Dimension::variable("frames"),
        ]
    };
    let mut graph = Builder::new(directory, "PyMSS MDX23C", shape(channels))?;
    let first = conv(&mut graph, "input", &model.first, [1, 1], 0, "Conv")?;
    let mut value = swap_axes(&mut graph, &first);
    let scale = config.model.scale;
    let mut skips = Vec::with_capacity(model.encoders.len());
    for (blocks, down) in &model.encoders {
        let hidden = run_stack(&mut graph, blocks, value)?;
        let input = pre(&mut graph, down, &hidden)?;
        value = conv(&mut graph, &input, &down.filter, scale, 0, "Conv")?;
        skips.push(hidden);
    }
    value = run_stack(&mut graph, &model.bottleneck, value)?;
    for ((up, blocks), skip) in model.decoders.iter().zip(skips.into_iter().rev()) {
        let value_in = pre(&mut graph, up, &value)?;
        let upscaled = conv(&mut graph, &value_in, &up.filter, scale, 0, "ConvTranspose")?;
        let joined = concatenate(&mut graph, &upscaled, &skip);
        value = run_stack(&mut graph, blocks, joined)?;
    }
    let value = swap_axes(&mut graph, &value);
    let product = graph.op("Mul", &[&value, &first], vec![]);
    let joined = concatenate(&mut graph, "input", &product);
    let hidden = conv(&mut graph, &joined, &model.final_in, [1, 1], 0, "Conv")?;
    let hidden = activate(&mut graph, &hidden, model.activation);
    let result = conv(&mut graph, &hidden, &model.final_out, [1, 1], 0, "Conv")?;
    graph.finish(path, result, shape(channels * config.stems() as i64))
}

fn run_stack(graph: &mut Builder, blocks: &[Block], mut value: String) -> Result<String> {
    for block in blocks {
        let skip = conv(graph, &value, &block.shortcut, [1, 1], 0, "Conv")?;
        let active = pre(graph, &block.first, &value)?;
        let hidden = conv(graph, &active, &block.first.filter, [1, 1], 1, "Conv")?;
        let frequency = linear(graph, &block.frequency_in, &hidden)?;
        let frequency = linear(graph, &block.frequency_out, &frequency)?;
        let merged = graph.op("Add", &[&hidden, &frequency], vec![]);
        let active = pre(graph, &block.last, &merged)?;
        let last = conv(graph, &active, &block.last.filter, [1, 1], 1, "Conv")?;
        value = graph.op("Add", &[&last, &skip], vec![]);
    }
    Ok(value)
}

fn swap_axes(graph: &mut Builder, input: &str) -> String {
    graph.op(
        "Transpose",
        &[input],
        vec![Attribute::ints("perm", &[0, 1, 3, 2])],
    )
}

fn concatenate(graph: &mut Builder, left: &str, right: &str) -> String {
    graph.op("Concat", &[left, right], vec![Attribute::int("axis", 1)])
}

fn pre(graph: &mut Builder, layer: &Normalized, input: &str) -> Result<String> {
    let value = normalize(graph, input, &layer.norm)?;
    Ok(activate(graph, &value, layer.act))
}

fn activate(graph: &mut Builder, input: &str, act: Activation) -> String {
    match act {
        Activation::Gelu => graph.op("Gelu", &[input], vec![]),
        Activation::Relu => graph.op("Relu", &[input], vec![]),
        Activation::Elu(alpha) => graph.op(
            "Elu",
            &[input],
            vec![Attribute::float("alpha", alpha as f32)],
        ),
    }
}

fn conv(
    graph: &mut Builder,
    input: &str,
    weight: &Tensor,
    stride: [i64; 2],
    pad: i64,
    op: &str,
) -> Result<String> {
    let weight = graph.tensor(weight)?;
    Ok(graph.op(
        op,
        &[input, &weight],
        vec![
            Attribute::ints("strides", &stride),
            Attribute::ints("pads", &[pad; 4]),
        ],
    ))
}

fn linear(graph: &mut Builder, layer: &Normalized, input: &str) -> Result<String> {
    let active = pre(graph, layer, input)?;
    let weight = graph.tensor(&layer.filter.f_transpose(0, 1)?)?;
    Ok(graph.op("MatMul", &[&active, &weight], vec![]))
}

fn normalize(graph: &mut Builder, input: &str, norm: &Norm) -> Result<String> {
    const EPSILON: f32 = 1e-5;
    Ok(match norm {
        Norm::Identity => input.to_owned(),
        Norm::Batch {
            gain,
            bias,
            mean,
            variance,
        } => {
            let gain = graph.tensor(gain)?;
            let bias = graph.tensor(bias)?;
            let mean = graph.tensor(mean)?;
            let variance = graph.tensor(variance)?;
            graph.op(
                "BatchNormalization",
                &[input, &gain, &bias, &mean, &variance],
                vec![Attribute::float("epsilon", EPSILON)],
            )
        }
        Norm::Instance { gain, bias } => {
            let gain = graph.tensor(gain)?;
            let bias = graph.tensor(bias)?;
            graph.op(
                "InstanceNormalization",
                &[input, &gain, &bias],
                vec![Attribute::float("epsilon", EPSILON)],
            )
        }
        Norm::Group { groups, gain, bias } => {
            let original_shape = graph.op("Shape", &[input], vec![]);
            let shape = graph.integers(&[3], &[0, *groups, -1]);
            let grouped = graph.op("Reshape", &[input, &shape], vec![]);
            let axis = graph.integers(&[1], &[-1]);
            let mean = graph.op("ReduceMean", &[&grouped, &axis], vec![]);
            let centered = graph.op("Sub", &[&grouped, &mean], vec![]);
            let squared = graph.op("Mul", &[&centered, &centered], vec![]);
            let variance = graph.op("ReduceMean", &[&squared, &axis], vec![]);
            let epsilon = graph.scale(EPSILON);
            let adjusted = graph.op("Add", &[&variance, &epsilon], vec![]);
            let deviation = graph.op("Sqrt", &[&adjusted], vec![]);
            let normalized = graph.op("Div", &[&centered, &deviation], vec![]);
            let normalized = graph.op("Reshape", &[&normalized, &original_shape], vec![]);
            let gain = graph.tensor(&gain.f_reshape([1, -1, 1, 1])?)?;
            let bias = graph.tensor(&bias.f_reshape([1, -1, 1, 1])?)?;
            let scaled = graph.op("Mul", &[&normalized, &gain], vec![]);
            graph.op("Add", &[&scaled, &bias], vec![])
        }
    })
}
