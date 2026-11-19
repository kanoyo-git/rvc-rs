use std::cell::RefCell;
use std::path::Path;

use anyhow::{Result, ensure};
use ort::{session::Session, value::TensorRef};
use tch::Tensor;
use tempfile::TempDir;

use super::fcpe::{CHANNELS, CONFORMER_LAYERS, MEL_BANDS, PITCH_BINS};
use super::onnx::{
    Provider,
    graph::{Attribute, Builder, Dimension, Layers},
};
use super::weights::Weights;

pub(super) struct FcpeOnnx {
    session: RefCell<Session>,
    _directory: TempDir,
}

impl FcpeOnnx {
    pub(super) fn load(weights: &Weights, provider: &str) -> Result<Self> {
        let provider = Provider::parse("RVC_FCPE_ONNX", provider)?;
        let directory = tempfile::Builder::new().prefix("rvc-fcpe-").tempdir()?;
        let path = directory.path().join("fcpe.onnx");
        export(weights, directory.path(), &path)?;
        let session = provider.session(&path)?;
        Ok(Self {
            session: RefCell::new(session),
            _directory: directory,
        })
    }

    pub(super) fn salience(&self, mel: &Tensor) -> Result<Tensor> {
        let frames = mel.size()[1];
        let values = Vec::<f32>::try_from(mel.f_contiguous()?.f_view([-1])?)?;
        let mel = TensorRef::from_array_view(([1, frames, MEL_BANDS], values.as_slice()))?;
        let mut session = self.session.borrow_mut();
        let output = session.run(ort::inputs![mel])?;
        let (shape, data) = output[0].try_extract_tensor::<f32>()?;
        ensure!(
            **shape == [1, frames, PITCH_BINS],
            "invalid FCPE ONNX output shape {shape:?}"
        );
        Ok(Tensor::f_from_slice(data)?.f_view([1, frames, PITCH_BINS])?)
    }
}

fn export(weights: &Weights, directory: &Path, path: &Path) -> Result<()> {
    let mut model = Layers {
        graph: Builder::new(
            directory,
            "FCPE",
            vec![
                Dimension::fixed(1),
                Dimension::variable("frames"),
                Dimension::fixed(MEL_BANDS),
            ],
        )?,
        weights,
    };
    let input = model.transpose("input", &[0, 2, 1]);
    let first = model.conv(&input, "input_stack.0", 1, 1, 1)?;
    let group_shape = model.graph.integers(&[3], &[1, 4, -1]);
    let flat = model.graph.op("Reshape", &[&first, &group_shape], vec![]);
    let ones = model.graph.floats(&[4], &[1.0; 4]);
    let zeros = model.graph.floats(&[4], &[0.0; 4]);
    let normalized = model.graph.op(
        "InstanceNormalization",
        &[&flat, &ones, &zeros],
        vec![Attribute::float("epsilon", 1e-5)],
    );
    let shape = model.graph.integers(&[3], &[1, CHANNELS, -1]);
    let normalized = model.graph.op("Reshape", &[&normalized, &shape], vec![]);
    let weight = model.graph.tensor(
        &weights
            .get("input_stack.1.weight")?
            .f_view([1, CHANNELS, 1])?,
    )?;
    let bias = model.graph.tensor(
        &weights
            .get("input_stack.1.bias")?
            .f_view([1, CHANNELS, 1])?,
    )?;
    let normalized = model.graph.op("Mul", &[&normalized, &weight], vec![]);
    let normalized = model.graph.op("Add", &[&normalized, &bias], vec![]);
    let activated = model.graph.op(
        "LeakyRelu",
        &[&normalized],
        vec![Attribute::float("alpha", 0.01)],
    );
    let hidden = model.conv(&activated, "input_stack.3", 1, 1, 1)?;
    let mut hidden = model.transpose(&hidden, &[0, 2, 1]);
    let axis = model.graph.integers(&[1], &[1]);
    let start = model.graph.integers(&[1], &[0]);
    let midpoint = model.graph.integers(&[1], &[2 * CHANNELS]);
    let end = model.graph.integers(&[1], &[4 * CHANNELS]);
    for layer in 0..CONFORMER_LAYERS {
        let name = format!("net.encoder_layers.{layer}.conformer.net");
        let branch = model.norm("LayerNormalization", &hidden, &format!("{name}.0"))?;
        let branch = model.transpose(&branch, &[0, 2, 1]);
        let branch = model.conv(&branch, &format!("{name}.2"), 1, 0, 1)?;
        let values = model
            .graph
            .op("Slice", &[&branch, &start, &midpoint, &axis], vec![]);
        let gate = model
            .graph
            .op("Slice", &[&branch, &midpoint, &end, &axis], vec![]);
        let gate = model.graph.op("Sigmoid", &[&gate], vec![]);
        let gated = model.graph.op("Mul", &[&values, &gate], vec![]);
        let filtered = model.conv(&gated, &format!("{name}.4.conv"), 1, 15, 2 * CHANNELS)?;
        let gate = model.graph.op("Sigmoid", &[&filtered], vec![]);
        let gated = model.graph.op("Mul", &[&filtered, &gate], vec![]);
        let branch = model.conv(&gated, &format!("{name}.6"), 1, 0, 1)?;
        let branch = model.transpose(&branch, &[0, 2, 1]);
        hidden = model.graph.op("Add", &[&hidden, &branch], vec![]);
    }
    let normalized = model.norm("LayerNormalization", &hidden, "norm")?;
    let output = model.linear(&normalized, "output_proj")?;
    let output = model.graph.op("Sigmoid", &[&output], vec![]);
    model.graph.finish(
        path,
        output,
        vec![
            Dimension::fixed(1),
            Dimension::variable("frames"),
            Dimension::fixed(PITCH_BINS),
        ],
    )
}
