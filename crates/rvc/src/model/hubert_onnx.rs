use std::cell::RefCell;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use ort::{session::Session, value::TensorRef};
use tch::Tensor;
use tempfile::TempDir;

use super::hubert::load_weights;
use super::onnx::{
    Provider,
    graph::{Attribute, Builder, Dimension, Layers},
    runtime,
};

pub(super) struct HubertOnnx {
    checkpoint: PathBuf,
    provider: Provider,
    loaded: RefCell<Option<Model>>,
}

struct Model {
    v2: bool,
    session: Session,
    _directory: TempDir,
}

impl HubertOnnx {
    pub(super) fn new(checkpoint: PathBuf, provider: &str) -> Result<Self> {
        let provider = Provider::parse("RVC_CONTENTVEC_ONNX", provider)?;
        ensure!(
            checkpoint.is_file(),
            "ContentVec checkpoint not found: {}",
            checkpoint.display()
        );
        runtime()?;
        Ok(Self {
            checkpoint,
            provider,
            loaded: RefCell::default(),
        })
    }

    pub(super) fn extract(&self, audio: &[f32], v2: bool) -> Result<Tensor> {
        ensure!(
            audio.len() >= 400,
            "HuBERT requires at least 400 input samples"
        );
        let mut loaded = self.loaded.borrow_mut();
        if loaded.as_ref().is_none_or(|model| model.v2 != v2) {
            *loaded = None;
            let directory = tempfile::Builder::new()
                .prefix("rvc-contentvec-")
                .tempdir()?;
            let path = directory.path().join("contentvec.onnx");
            export(&self.checkpoint, directory.path(), &path, v2)?;
            let session = self.provider.session(&path)?;
            *loaded = Some(Model {
                v2,
                session,
                _directory: directory,
            });
        }
        let model = loaded
            .as_mut()
            .context("ContentVec ONNX model is unavailable")?;
        let input = TensorRef::from_array_view(([1, 1, i64::try_from(audio.len())?], audio))?;
        let output = model.session.run(ort::inputs![input])?;
        let (shape, samples) = output[0].try_extract_tensor::<f32>()?;
        let width = if v2 { 768 } else { 256 };
        let [1, frames, actual_width] = **shape else {
            anyhow::bail!("invalid ContentVec output shape {shape:?}")
        };
        ensure!(
            actual_width == width && frames > 0,
            "invalid ContentVec output shape {shape:?}"
        );
        Ok(Tensor::f_from_slice(samples)?.f_view([1, frames, width])?)
    }
}

fn export(checkpoint: &Path, directory: &Path, path: &Path, v2: bool) -> Result<()> {
    let weights = load_weights(checkpoint)?;
    let mut model = Layers {
        graph: Builder::new(
            directory,
            "ContentVec",
            vec![
                Dimension::fixed(1),
                Dimension::fixed(1),
                Dimension::variable("samples"),
            ],
        )?,
        weights: &weights,
    };
    let mut features = "input".to_owned();
    for (index, stride) in [5, 2, 2, 2, 2, 2, 2].into_iter().enumerate() {
        let layer = format!("feature_extractor.conv_layers.{index}");
        features = model.conv(&features, &format!("{layer}.conv"), stride, 0, 1)?;
        if index == 0 {
            features = model.norm(
                "InstanceNormalization",
                &features,
                &format!("{layer}.layer_norm"),
            )?;
        }
        features = model.graph.op("Gelu", &[&features], vec![]);
    }
    features = model.transpose(&features, &[0, 2, 1]);
    features = model.norm(
        "LayerNormalization",
        &features,
        "feature_projection.layer_norm",
    )?;
    features = model.linear(&features, "feature_projection.projection")?;
    let transposed = model.transpose(&features, &[0, 2, 1]);
    let position = model.conv(&transposed, "encoder.pos_conv_embed.conv", 1, 64, 16)?;
    let start = model.graph.integers(&[1], &[0]);
    let end = model.graph.integers(&[1], &[-1]);
    let axis = model.graph.integers(&[1], &[2]);
    let position = model
        .graph
        .op("Slice", &[&position, &start, &end, &axis], vec![]);
    let position = model.transpose(&position, &[0, 2, 1]);
    let position = model.graph.op("Gelu", &[&position], vec![]);
    features = model.graph.op("Add", &[&features, &position], vec![]);
    features = model.norm("LayerNormalization", &features, "encoder.layer_norm")?;
    let qkv_shape = model.graph.integers(&[5], &[1, -1, 3, 12, 64]);
    let output_shape = model.graph.integers(&[3], &[1, -1, 768]);
    let projection = [0, 1, 2].map(|part| model.graph.integers(&[], &[part]));
    let scale = model.graph.scale(0.125);
    for index in 0..if v2 { 12 } else { 9 } {
        let layer = format!("encoder.layers.{index}");
        let prefix = format!("{layer}.attention");
        let qkv = model.linear(&features, &format!("{prefix}.qkv_proj"))?;
        let qkv = model.graph.op("Reshape", &[&qkv, &qkv_shape], vec![]);
        let [q, k, v] = projection.each_ref().map(|part| {
            let value = model
                .graph
                .op("Gather", &[&qkv, part], vec![Attribute::int("axis", 2)]);
            model.transpose(&value, &[0, 2, 1, 3])
        });
        let q = model.graph.op("Mul", &[&q, &scale], vec![]);
        let k = model.transpose(&k, &[0, 1, 3, 2]);
        let attention = model.graph.op("MatMul", &[&q, &k], vec![]);
        let attention = model
            .graph
            .op("Softmax", &[&attention], vec![Attribute::int("axis", -1)]);
        let attention = model.graph.op("MatMul", &[&attention, &v], vec![]);
        let attention = model.transpose(&attention, &[0, 2, 1, 3]);
        let attention = model
            .graph
            .op("Reshape", &[&attention, &output_shape], vec![]);
        let attention = model.linear(&attention, &format!("{prefix}.out_proj"))?;
        features = model.graph.op("Add", &[&features, &attention], vec![]);
        features = model.norm(
            "LayerNormalization",
            &features,
            &format!("{layer}.layer_norm"),
        )?;
        let dense = model.linear(
            &features,
            &format!("{layer}.feed_forward.intermediate_dense"),
        )?;
        let dense = model.graph.op("Gelu", &[&dense], vec![]);
        let dense = model.linear(&dense, &format!("{layer}.feed_forward.output_dense"))?;
        features = model.graph.op("Add", &[&features, &dense], vec![]);
        features = model.norm(
            "LayerNormalization",
            &features,
            &format!("{layer}.final_layer_norm"),
        )?;
    }
    if !v2 {
        features = model.linear(&features, "final_proj")?;
    }
    model.graph.finish(
        path,
        features,
        vec![
            Dimension::fixed(1),
            Dimension::variable("frames"),
            Dimension::fixed(if v2 { 768 } else { 256 }),
        ],
    )
}
