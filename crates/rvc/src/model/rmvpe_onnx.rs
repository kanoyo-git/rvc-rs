use std::path::Path;

use anyhow::{Result, ensure};
use tch::{Device, Kind, Tensor};

use super::onnx::{
    GraphSession, Provider, TensorInput,
    graph::{Attribute, Builder, Dimension, Layers},
};
use super::rmvpe::{BINS, GRU_WIDTH, MEL_BANDS, MelSpectrogram, decode};
use super::weights::Weights;

pub(super) struct RmvpeOnnx {
    graph: GraphSession,
    features: MelSpectrogram,
}

impl RmvpeOnnx {
    pub(super) fn load(models: &Path, provider: &str) -> Result<Self> {
        let provider = Provider::parse("RVC_RMVPE_ONNX", provider)?;
        let checkpoint = models.join("rmvpe.pt");
        let graph = if checkpoint.is_file() {
            GraphSession::load(provider, "rmvpe", |directory, path| {
                let weights = Weights::load(&checkpoint, None, Device::Cpu)?;
                export(&weights, directory, path)
            })?
        } else {
            let path = models.join("rmvpe.onnx");
            ensure!(
                path.is_file(),
                "RMVPE weights (rmvpe.pt or rmvpe.onnx) not found in {}",
                models.display()
            );
            GraphSession::open(provider, &path)?
        };
        let session = graph.session.borrow();
        ensure!(
            session.inputs().len() == 1 && session.outputs().len() == 1,
            "RMVPE ONNX requires one mel input and one salience output"
        );
        drop(session);
        Ok(Self {
            graph,
            features: MelSpectrogram::new(Device::Cpu)?,
        })
    }

    pub(super) fn infer(&self, samples: &[f32], threshold: f32) -> Result<Vec<f32>> {
        let input = self.features.input(samples)?;
        let (mel, frames) = self.features.extract(&input, Kind::Float, Device::Cpu)?;
        let padded = mel.size()[2];
        let mel = TensorInput::<f32>::new(&mel)?;
        let mut session = self.graph.session.borrow_mut();
        let output = session.run(ort::inputs![mel.view()?])?;
        let (shape, salience) = output[0].try_extract_tensor::<f32>()?;
        ensure!(
            **shape == [1, padded, BINS as i64],
            "invalid RMVPE ONNX output shape {shape:?}"
        );
        decode(&salience[..frames as usize * BINS], threshold)
    }
}

fn export(weights: &Weights, directory: &Path, path: &Path) -> Result<()> {
    let fixed = Dimension::fixed;
    let frames = || Dimension::variable("frames");
    let mut model = Layers {
        graph: Builder::new(
            directory,
            "RVC RMVPE",
            vec![fixed(1), fixed(MEL_BANDS), frames()],
        )?,
        weights,
    };
    let mel = model.transpose("input", &[0, 2, 1]);
    let axis = model.graph.integers(&[1], &[1]);
    let mel = model.graph.op("Unsqueeze", &[&mel, &axis], vec![]);
    let mut value = normalize(&mut model, &mel, "unet.encoder.bn")?;
    let mut skips = Vec::with_capacity(5);
    for layer in 0..5 {
        value = residual_stack(
            &mut model,
            value,
            &format!("unet.encoder.layers.{layer}.conv"),
        )?;
        let pooled = model.graph.op(
            "AveragePool",
            &[&value],
            vec![
                Attribute::ints("kernel_shape", &[2, 2]),
                Attribute::ints("strides", &[2, 2]),
            ],
        );
        skips.push(value);
        value = pooled;
    }
    for layer in 0..4 {
        value = residual_stack(
            &mut model,
            value,
            &format!("unet.intermediate.layers.{layer}.conv"),
        )?;
    }
    for (layer, skip) in skips.into_iter().rev().enumerate() {
        let name = format!("unet.decoder.layers.{layer}");
        let expanded = model.convolution(
            "ConvTranspose",
            &value,
            &format!("{name}.conv1.0"),
            vec![
                Attribute::ints("pads", &[1, 1, 1, 1]),
                Attribute::ints("strides", &[2, 2]),
                Attribute::ints("output_padding", &[1, 1]),
            ],
        )?;
        let expanded = normalize(&mut model, &expanded, &format!("{name}.conv1.1"))?;
        let expanded = model.graph.op("Relu", &[&expanded], vec![]);
        let joined = model.graph.op(
            "Concat",
            &[&expanded, &skip],
            vec![Attribute::int("axis", 1)],
        );
        value = residual_stack(&mut model, joined, &format!("{name}.conv2"))?;
    }
    let value = convolution(&mut model, &value, "cnn", 1)?;
    let value = model.transpose(&value, &[2, 0, 1, 3]);
    let input = model.graph.integers(&[3], &[-1, 1, 3 * MEL_BANDS]);
    let value = model.graph.op("Reshape", &[&value, &input], vec![]);
    let value = bidirectional_gru(&mut model, &value)?;
    let value = model.transpose(&value, &[2, 0, 1, 3]);
    let shape = model.graph.integers(&[3], &[1, -1, 2 * GRU_WIDTH]);
    let value = model.graph.op("Reshape", &[&value, &shape], vec![]);
    let value = model.linear(&value, "fc.1")?;
    let salience = model.graph.op("Sigmoid", &[&value], vec![]);
    model
        .graph
        .finish(path, salience, vec![fixed(1), frames(), fixed(BINS as i64)])
}

fn convolution(model: &mut Layers<'_>, input: &str, name: &str, pad: i64) -> Result<String> {
    model.convolution(
        "Conv",
        input,
        name,
        vec![Attribute::ints("pads", &[pad; 4])],
    )
}

fn normalize(model: &mut Layers<'_>, input: &str, name: &str) -> Result<String> {
    let normalization = ["weight", "bias", "running_mean", "running_var"].map(|key| {
        model
            .graph
            .tensor(model.weights.get(&format!("{name}.{key}"))?)
    });
    let [scale, bias, mean, variance] = normalization;
    Ok(model.graph.op(
        "BatchNormalization",
        &[input, &scale?, &bias?, &mean?, &variance?],
        vec![Attribute::float("epsilon", 1e-5)],
    ))
}

fn residual_stack(model: &mut Layers<'_>, mut value: String, name: &str) -> Result<String> {
    for block in 0..4 {
        let layer = format!("{name}.{block}");
        let mut hidden = value.clone();
        for (conv, norm) in [(0, 1), (3, 4)] {
            hidden = convolution(model, &hidden, &format!("{layer}.conv.{conv}"), 1)?;
            hidden = normalize(model, &hidden, &format!("{layer}.conv.{norm}"))?;
            hidden = model.graph.op("Relu", &[&hidden], vec![]);
        }
        let shortcut = format!("{layer}.shortcut");
        if model.weights.contains(&format!("{shortcut}.weight")) {
            value = convolution(model, &value, &shortcut, 0)?;
        }
        value = model.graph.op("Add", &[&value, &hidden], vec![]);
    }
    Ok(value)
}

fn bidirectional_gru(model: &mut Layers<'_>, input: &str) -> Result<String> {
    // ONNX stores update, reset and hidden gates, in that order.
    let order = |name, direction| -> Result<Tensor> {
        let weight = model
            .weights
            .get(&format!("fc.0.gru.{name}_l0{direction}"))?;
        let gates = [1, 0, 2].map(|i| weight.narrow(0, i * GRU_WIDTH, GRU_WIDTH));
        Ok(Tensor::f_cat(&gates, 0)?)
    };
    let stack = |name| -> Result<Tensor> {
        Ok(Tensor::f_stack(
            &[order(name, "")?, order(name, "_reverse")?],
            0,
        )?)
    };
    let weights = model.graph.tensor(&stack("weight_ih")?)?;
    let recurrent = model.graph.tensor(&stack("weight_hh")?)?;
    let bias = Tensor::f_cat(&[stack("bias_ih")?, stack("bias_hh")?], 1)?;
    let bias = model.graph.tensor(&bias)?;
    Ok(model.graph.op(
        "GRU",
        &[input, &weights, &recurrent, &bias],
        vec![
            Attribute::text("direction", "bidirectional"),
            Attribute::int("hidden_size", GRU_WIDTH),
            Attribute::int("linear_before_reset", 1),
        ],
    ))
}
