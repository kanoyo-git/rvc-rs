use std::collections::HashMap;
use std::path::Path;

use anyhow::{Result, ensure};
use tch::Tensor;

use super::{Config, SLOPE, Weights};
use crate::model::onnx::{
    GraphSession, Provider, TensorInput,
    graph::{Attribute, Builder, Dimension, Layers},
};

pub(super) struct Prior {
    graph: GraphSession,
    channels: i64,
}

impl Prior {
    pub(super) fn load(
        weights: &Weights,
        config: &Config,
        uses_f0: bool,
        provider: Provider,
    ) -> Result<Self> {
        let graph = GraphSession::load(provider, "prior", |directory, path| {
            export(weights, config, uses_f0, directory, path)
        })?;
        Ok(Self {
            graph,
            channels: config.inter,
        })
    }

    pub(super) fn stats(
        &self,
        phone: &Tensor,
        pitch: Option<&Tensor>,
        skip: i64,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let frames = phone.size()[1];
        let phones = TensorInput::<f32>::new(phone)?;
        let pitch = pitch.map(TensorInput::<i64>::new).transpose()?;
        let mut session = self.graph.session.borrow_mut();
        let output = if let Some(pitch) = &pitch {
            session.run(ort::inputs![phones.view()?, pitch.view()?])?
        } else {
            session.run(ort::inputs![phones.view()?])?
        };
        let (shape, data) = output[0].try_extract_tensor::<f32>()?;
        let expected = [1, 2 * self.channels, frames];
        ensure!(
            **shape == expected,
            "invalid ONNX prior output shape {shape:?}"
        );
        let stats = Tensor::f_from_slice(data)?
            .f_view(expected)?
            .f_narrow(2, skip, frames - skip)?
            .f_to_device(phone.device())?
            .f_to_kind(phone.kind())?;
        Ok((
            stats.narrow(1, 0, self.channels),
            stats.narrow(1, self.channels, self.channels),
            Tensor::ones([1, 1, frames - skip], (phone.kind(), phone.device())),
        ))
    }
}

fn export(
    weights: &Weights,
    config: &Config,
    uses_f0: bool,
    directory: &Path,
    path: &Path,
) -> Result<()> {
    let fixed = Dimension::fixed;
    let frames = || Dimension::variable("frames");
    let input = weights.get("enc_p.emb_phone.weight")?.size()[1];
    let mut model = Layers {
        graph: Builder::new(
            directory,
            "RVC prior",
            vec![fixed(1), frames(), fixed(input)],
        )?,
        weights,
    };
    let mut value = model.linear("input", "enc_p.emb_phone")?;
    if uses_f0 {
        model.graph.integer_input("pitch", vec![fixed(1), frames()]);
        let table = model.graph.tensor(weights.get("enc_p.emb_pitch.weight")?)?;
        let pitch = model.graph.op("Gather", &[&table, "pitch"], vec![]);
        value = model.graph.op("Add", &[&value, &pitch], vec![]);
    }
    let scale = model.graph.scale((config.hidden as f32).sqrt());
    let value = model.graph.op("Mul", &[&value, &scale], vec![]);
    let value = model.graph.op(
        "LeakyRelu",
        &[&value],
        vec![Attribute::float("alpha", SLOPE as f32)],
    );
    let mut value = model.transpose(&value, &[0, 2, 1]);
    let mut attention = Attention::new(&mut model.graph, config);
    let padding = [(config.encoder_kernel - 1) / 2, config.encoder_kernel / 2];
    for layer in 0..config.encoder_layers {
        let prefix = "enc_p.encoder";
        let name = format!("{prefix}.attn_layers.{layer}");
        let attended = attention.forward(&mut model, &value, &name)?;
        value = norm_residual(
            &mut model,
            &value,
            &attended,
            &format!("{prefix}.norm_layers_1.{layer}"),
        )?;
        let name = format!("{prefix}.ffn_layers.{layer}");
        let dense = model.convolution(
            "Conv",
            &value,
            &format!("{name}.conv_1"),
            vec![Attribute::ints("pads", &padding)],
        )?;
        let dense = model.graph.op("Relu", &[&dense], vec![]);
        let dense = model.convolution(
            "Conv",
            &dense,
            &format!("{name}.conv_2"),
            vec![Attribute::ints("pads", &padding)],
        )?;
        value = norm_residual(
            &mut model,
            &value,
            &dense,
            &format!("{prefix}.norm_layers_2.{layer}"),
        )?;
    }
    let stats = model.conv(&value, "enc_p.proj", 1, 0, 1)?;
    model.graph.finish(
        path,
        stats,
        vec![fixed(1), fixed(config.inter * 2), frames()],
    )
}

fn norm_residual(model: &mut Layers<'_>, left: &str, right: &str, name: &str) -> Result<String> {
    let sum = model.graph.op("Add", &[left, right], vec![]);
    let sum = model.transpose(&sum, &[0, 2, 1]);
    let scale = model
        .graph
        .tensor(model.weights.get(&format!("{name}.gamma"))?)?;
    let bias = model
        .graph
        .tensor(model.weights.get(&format!("{name}.beta"))?)?;
    let value = model.graph.op(
        "LayerNormalization",
        &[&sum, &scale, &bias],
        vec![Attribute::float("epsilon", 1e-5)],
    );
    Ok(model.transpose(&value, &[0, 2, 1]))
}

struct Attention {
    head_shape: String,
    flat_shape: String,
    output_shape: String,
    scale: String,
    zero_float: String,
    positions: PositionIndex,
    relative: HashMap<i64, RelativePositions>,
}

struct PositionIndex {
    frames: String,
    last_frame: String,
    query: String,
    distance: String,
    zero: String,
}

struct RelativePositions {
    offsets: String,
    valid_offsets: String,
    keys: String,
    valid_keys: String,
}

impl Attention {
    fn new(graph: &mut Builder, config: &Config) -> Self {
        let heads = config.heads;
        let width = config.hidden / heads;
        let head_shape = graph.integers(&[4], &[1, heads, width, -1]);
        let flat_shape = graph.integers(&[3], &[1, heads, -1]);
        let output_shape = graph.integers(&[3], &[1, config.hidden, -1]);
        let scale = graph.scale(1.0 / (width as f32).sqrt());
        let zero_float = graph.scale(0.0);
        Self {
            head_shape,
            flat_shape,
            output_shape,
            scale,
            zero_float,
            positions: PositionIndex::new(graph),
            relative: HashMap::new(),
        }
    }

    fn project(&self, model: &mut Layers<'_>, value: &str, name: &str) -> Result<String> {
        let value = model.conv(value, name, 1, 0, 1)?;
        let value = model
            .graph
            .op("Reshape", &[&value, &self.head_shape], vec![]);
        Ok(model.transpose(&value, &[0, 1, 3, 2]))
    }

    fn forward(&mut self, model: &mut Layers<'_>, value: &str, name: &str) -> Result<String> {
        let q = self.project(model, value, &format!("{name}.conv_q"))?;
        let q = model.graph.op("Mul", &[&q, &self.scale], vec![]);
        let k = self.project(model, value, &format!("{name}.conv_k"))?;
        let k = model.transpose(&k, &[0, 1, 3, 2]);
        let v = self.project(model, value, &format!("{name}.conv_v"))?;
        let dot = model.graph.op("MatMul", &[&q, &k], vec![]);
        let rel_k = model.weights.get(&format!("{name}.emb_rel_k"))?;
        let rel_v = model.weights.get(&format!("{name}.emb_rel_v"))?;
        let offsets = rel_k.size()[1];
        ensure!(
            offsets > 0 && offsets % 2 == 1 && rel_k.size() == rel_v.size(),
            "invalid relative attention table in {name}"
        );
        let rel_k = model.graph.tensor(&rel_k.transpose(1, 2).unsqueeze(0))?;
        let rel_v = model.graph.tensor(&rel_v.unsqueeze(0))?;
        let positions = self
            .relative
            .entry(offsets)
            .or_insert_with(|| self.positions.for_table(&mut model.graph, offsets));
        let band = model.graph.op("MatMul", &[&q, &rel_k], vec![]);
        let band = model
            .graph
            .op("Reshape", &[&band, &self.flat_shape], vec![]);
        let relative = gather(&mut model.graph, &band, &positions.offsets);
        let relative = model.graph.op(
            "Where",
            &[&positions.valid_offsets, &relative, &self.zero_float],
            vec![],
        );
        let logits = model.graph.op("Add", &[&dot, &relative], vec![]);
        let weights = model.graph.op("Softmax", &[&logits], vec![]);
        let direct = model.graph.op("MatMul", &[&weights, &v], vec![]);
        let weights = model
            .graph
            .op("Reshape", &[&weights, &self.flat_shape], vec![]);
        let band = gather(&mut model.graph, &weights, &positions.keys);
        let band = model.graph.op(
            "Where",
            &[&positions.valid_keys, &band, &self.zero_float],
            vec![],
        );
        let relative = model.graph.op("MatMul", &[&band, &rel_v], vec![]);
        let result = model.graph.op("Add", &[&direct, &relative], vec![]);
        let result = model.transpose(&result, &[0, 1, 3, 2]);
        let result = model
            .graph
            .op("Reshape", &[&result, &self.output_shape], vec![]);
        model.conv(&result, &format!("{name}.conv_o"), 1, 0, 1)
    }
}

impl PositionIndex {
    fn new(graph: &mut Builder) -> Self {
        let zero = graph.integers(&[], &[0]);
        let one = graph.integers(&[], &[1]);
        let shape = graph.op("Shape", &["input"], vec![]);
        let frames = graph.op("Gather", &[&shape, &one], vec![]);
        let last_frame = graph.op("Sub", &[&frames, &one], vec![]);
        let key = graph.op("Range", &[&zero, &frames, &one], vec![]);
        let axis = graph.integers(&[1], &[1]);
        let query = graph.op("Unsqueeze", &[&key, &axis], vec![]);
        let distance = graph.op("Sub", &[&key, &query], vec![]);
        Self {
            frames,
            last_frame,
            query,
            distance,
            zero,
        }
    }

    fn for_table(&self, graph: &mut Builder, count: i64) -> RelativePositions {
        let center = count / 2;
        let center_index = graph.integers(&[], &[center]);
        let count_index = graph.integers(&[], &[count]);
        let last_offset = graph.integers(&[], &[count - 1]);
        // Table column d contains the relative position key - query = d - center.
        let offset = graph.op("Add", &[&self.distance, &center_index], vec![]);
        let (offset, valid_offsets) = bounds(graph, &offset, &self.zero, &last_offset);
        let row = graph.op("Mul", &[&self.query, &count_index], vec![]);
        let offsets = graph.op("Add", &[&row, &offset], vec![]);
        let distances: Vec<_> = (-center..=center).collect();
        let distances = graph.integers(&[count], &distances);
        let keys = graph.op("Add", &[&self.query, &distances], vec![]);
        let (keys, valid_keys) = bounds(graph, &keys, &self.zero, &self.last_frame);
        let row = graph.op("Mul", &[&self.query, &self.frames], vec![]);
        let keys = graph.op("Add", &[&row, &keys], vec![]);
        RelativePositions {
            offsets,
            valid_offsets,
            keys,
            valid_keys,
        }
    }
}

fn bounds(graph: &mut Builder, index: &str, low: &str, high: &str) -> (String, String) {
    let begin = graph.op("GreaterOrEqual", &[index, low], vec![]);
    let end = graph.op("LessOrEqual", &[index, high], vec![]);
    let valid = graph.op("And", &[&begin, &end], vec![]);
    let clipped = graph.op("Clip", &[index, low, high], vec![]);
    (clipped, valid)
}

fn gather(graph: &mut Builder, input: &str, indices: &str) -> String {
    graph.op("Gather", &[input, indices], vec![Attribute::int("axis", 2)])
}
