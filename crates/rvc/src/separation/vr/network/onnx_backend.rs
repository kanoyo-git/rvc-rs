use std::path::Path;

use anyhow::{Result, ensure};
use tch::Tensor;

use super::{Activation, Block, Conv, Norm, Recurrent, Subnetwork, Topology, TorchNetwork};
use crate::model::onnx::{
    GraphSession, Provider, TensorInput,
    graph::{Attribute, Builder, Dimension},
};

pub(super) struct Network {
    model: GraphSession,
    width: i64,
    frames: i64,
}

impl Network {
    pub(super) fn load(
        torch: &TorchNetwork,
        provider: Provider,
        bins: i64,
        frames: i64,
        offset: i64,
    ) -> Result<Self> {
        let model = GraphSession::load(provider, "uvr-vr", |directory, path| {
            export(torch, bins, frames, offset, directory, path)
        })?;
        Ok(Self {
            model,
            width: bins + 1,
            frames: frames - 2 * offset,
        })
    }

    pub(super) fn predict_mask(&self, input: &Tensor) -> Result<Tensor> {
        let expected = [input.size()[0], 2, self.width, self.frames];
        let input = TensorInput::<f32>::new(input)?;
        let mut model = self.model.session.borrow_mut();
        let output = model.run(ort::inputs![input.view()?])?;
        let (shape, values) = output[0].try_extract_tensor::<f32>()?;
        ensure!(
            **shape == expected,
            "invalid UVR VR ONNX mask shape {shape:?}"
        );
        Ok(Tensor::f_from_slice(values)?.f_view(expected)?)
    }
}

#[derive(Clone)]
struct Plane {
    value: String,
    geometry: [i64; 3],
}

impl Plane {
    fn new(value: String, geometry: [i64; 3]) -> Self {
        Self { value, geometry }
    }
}

fn dimensions(geometry: [i64; 3]) -> Vec<Dimension> {
    std::iter::once(Dimension::variable("batch"))
        .chain(geometry.into_iter().map(Dimension::fixed))
        .collect()
}

fn export(
    model: &TorchNetwork,
    bins: i64,
    frames: i64,
    offset: i64,
    directory: &Path,
    path: &Path,
) -> Result<()> {
    let shape = [2, bins + 1, frames];
    let mut graph = Builder::new(directory, "PyMSS UVR VR", dimensions(shape))?;
    let input = Plane::new("input".to_owned(), shape);
    let input = narrow(&mut graph, &input, 2, 0, bins);
    let lower = narrow(&mut graph, &input, 2, 0, bins / 2);
    let upper = narrow(&mut graph, &input, 2, bins / 2, bins);
    let features = match &model.topology {
        Topology::Legacy {
            low,
            high,
            bridge2,
            bridge3,
            stage2,
            stage3,
        } => {
            let low = subnet(&mut graph, low, &lower)?;
            let high = subnet(&mut graph, high, &upper)?;
            let split = concatenate(&mut graph, &[&low, &high], 2);
            let refine = concatenate(&mut graph, &[&input, &split], 1);
            let refine = block(&mut graph, bridge2, &refine)?;
            let refine = subnet(&mut graph, stage2, &refine)?;
            let merged = concatenate(&mut graph, &[&input, &split, &refine], 1);
            let merged = block(&mut graph, bridge3, &merged)?;
            subnet(&mut graph, stage3, &merged)?
        }
        Topology::Modern {
            low1,
            high1,
            low2,
            high2,
            stage3,
        } => {
            let low = subnet(&mut graph, low1, &lower)?;
            let high = subnet(&mut graph, high1, &upper)?;
            let first = concatenate(&mut graph, &[&low, &high], 2);
            let low_input = concatenate(&mut graph, &[&lower, &low], 1);
            let high_input = concatenate(&mut graph, &[&upper, &high], 1);
            let low = subnet(&mut graph, low2, &low_input)?;
            let high = subnet(&mut graph, high2, &high_input)?;
            let second = concatenate(&mut graph, &[&low, &high], 2);
            let merged = concatenate(&mut graph, &[&input, &first, &second], 1);
            subnet(&mut graph, stage3, &merged)?
        }
    };
    let mask = convolution(&mut graph, &model.output, &features)?;
    ensure!(
        mask.geometry == [2, bins, frames],
        "invalid exported VR mask shape"
    );
    let mask = narrow(&mut graph, &mask, 3, offset, frames - offset);
    let edge = narrow(&mut graph, &mask, 2, bins - 1, bins);
    let mask = concatenate(&mut graph, &[&mask, &edge], 2);
    let result = graph.op("Sigmoid", &[&mask.value], vec![]);
    graph.finish(path, result, dimensions(mask.geometry))
}

fn narrow(graph: &mut Builder, input: &Plane, axis: usize, start: i64, end: i64) -> Plane {
    let dims = graph.integers(&[1], &[axis as i64]);
    let start_node = graph.integers(&[1], &[start]);
    let end_node = graph.integers(&[1], &[end]);
    let value = graph.op(
        "Slice",
        &[&input.value, &start_node, &end_node, &dims],
        vec![],
    );
    let mut geometry = input.geometry;
    geometry[axis - 1] = end - start;
    Plane::new(value, geometry)
}

fn concatenate(graph: &mut Builder, inputs: &[&Plane], axis: usize) -> Plane {
    let mut geometry = inputs[0].geometry;
    geometry[axis - 1] = inputs.iter().map(|input| input.geometry[axis - 1]).sum();
    let values = inputs
        .iter()
        .map(|input| input.value.as_str())
        .collect::<Vec<_>>();
    Plane::new(
        graph.op("Concat", &values, vec![Attribute::int("axis", axis as i64)]),
        geometry,
    )
}

fn convolution(graph: &mut Builder, layer: &Conv, input: &Plane) -> Result<Plane> {
    let [out, _, height, width] = <[i64; 4]>::try_from(layer.weight.size().as_slice())?;
    let stride = layer.geometry.stride;
    let pad = layer.geometry.padding;
    let dilation = layer.geometry.dilation;
    let weight = graph.tensor(&layer.weight)?;
    let bias = layer
        .bias
        .as_ref()
        .map(|bias| graph.tensor(bias))
        .transpose()?;
    let mut values = vec![input.value.as_str(), weight.as_str()];
    if let Some(bias) = &bias {
        values.push(bias);
    }
    let value = graph.op(
        "Conv",
        &values,
        vec![
            Attribute::ints("strides", &stride),
            Attribute::ints("pads", &[pad[0], pad[1], pad[0], pad[1]]),
            Attribute::ints("dilations", &dilation),
            Attribute::int("group", layer.groups),
        ],
    );
    let dimension = |axis, kernel| {
        (input.geometry[axis + 1] + 2 * pad[axis] - dilation[axis] * (kernel - 1) - 1)
            / stride[axis]
            + 1
    };
    Ok(Plane::new(
        value,
        [out, dimension(0, height), dimension(1, width)],
    ))
}

fn normalize(graph: &mut Builder, layer: &Norm, input: &str) -> Result<String> {
    let scale = graph.tensor(&layer.weight)?;
    let bias = graph.tensor(&layer.bias)?;
    let mean = graph.tensor(&layer.mean)?;
    let variance = graph.tensor(&layer.variance)?;
    Ok(graph.op(
        "BatchNormalization",
        &[input, &scale, &bias, &mean, &variance],
        vec![Attribute::float("epsilon", 1e-5)],
    ))
}

fn block(graph: &mut Builder, layer: &Block, input: &Plane) -> Result<Plane> {
    let depth = layer
        .depth
        .as_ref()
        .map(|depth| convolution(graph, depth, input))
        .transpose()?;
    let mut value = convolution(graph, &layer.point, depth.as_ref().unwrap_or(input))?;
    if let Some(norm) = &layer.norm {
        value.value = normalize(graph, norm, &value.value)?;
    }
    value.value = match layer.act {
        Activation::Relu => graph.op("Relu", &[&value.value], vec![]),
        Activation::Leaky => graph.op(
            "LeakyRelu",
            &[&value.value],
            vec![Attribute::float("alpha", 0.01)],
        ),
    };
    Ok(value)
}

fn align_corners(graph: &mut Builder, input: &Plane, bins: i64, frames: i64) -> Plane {
    let scale = graph.floats(
        &[4],
        &[
            1.,
            1.,
            bins as f32 / input.geometry[1] as f32,
            frames as f32 / input.geometry[2] as f32,
        ],
    );
    Plane::new(
        graph.op(
            "Resize",
            &[&input.value, "", &scale],
            vec![
                Attribute::text("mode", "linear"),
                Attribute::text("coordinate_transformation_mode", "align_corners"),
            ],
        ),
        [input.geometry[0], bins, frames],
    )
}

fn subnet(graph: &mut Builder, model: &Subnetwork, input: &Plane) -> Result<Plane> {
    let mut value = if let Some(first) = &model.first {
        block(graph, first, input)?
    } else {
        input.clone()
    };
    let mut skips = Vec::with_capacity(model.down.len());
    for [first, second] in &model.down {
        let features = block(graph, first, &value)?;
        let downsampled = block(graph, second, &features)?;
        skips.push(if model.first.is_some() {
            value
        } else {
            features
        });
        value = downsampled;
    }
    let [channels, bins, frames] = value.geometry;
    let pool_axis = graph.integers(&[1], &[2]);
    let pool = graph.op("ReduceMean", &[&value.value, &pool_axis], vec![]);
    let pool = block(
        graph,
        &model.aspp.pool,
        &Plane::new(pool, [channels, 1, frames]),
    )?;
    let pool = align_corners(graph, &pool, bins, frames);
    let mut branches = vec![pool];
    for path in &model.aspp.paths {
        branches.push(block(graph, path, &value)?);
    }
    let merged = concatenate(graph, &branches.iter().collect::<Vec<_>>(), 1);
    value = block(graph, &model.aspp.merge, &merged)?;
    for (index, (layer, skip)) in model.up.iter().zip(skips.into_iter().rev()).enumerate() {
        if let Some(context) = model
            .recurrent
            .as_ref()
            .filter(|_| index + 1 == model.up.len())
        {
            let context = recurrent(graph, context, &value)?;
            value = concatenate(graph, &[&value, &context], 1);
        }
        let [_, bins, frames] = value.geometry;
        let resized = align_corners(graph, &value, 2 * bins, 2 * frames);
        ensure!(
            skip.geometry[1] == bins * 2 && skip.geometry[2] >= frames * 2,
            "invalid UVR VR skip dimensions"
        );
        let start = (skip.geometry[2] - 2 * frames) / 2;
        let skip = narrow(graph, &skip, 3, start, start + 2 * frames);
        let tensor = concatenate(graph, &[&resized, &skip], 1);
        value = block(graph, layer, &tensor)?;
    }
    if let Some(last) = &model.last {
        value = block(graph, last, &value)?;
    }
    Ok(value)
}

fn recurrent(graph: &mut Builder, layer: &Recurrent, input: &Plane) -> Result<Plane> {
    let [_, bins, frames] = input.geometry;
    let convolution = block(graph, &layer.conv, input)?;
    let channel = graph.integers(&[1], &[1]);
    let source = graph.op("Squeeze", &[&convolution.value, &channel], vec![]);
    let source = graph.op(
        "Transpose",
        &[&source],
        vec![Attribute::ints("perm", &[2, 0, 1])],
    );
    let reorder = |input: &Tensor| -> Result<Tensor> {
        let gates = [0, 3, 1, 2].map(|gate| input.narrow(0, gate * layer.hidden, layer.hidden));
        Ok(Tensor::f_cat(&gates, 0)?)
    };
    let parameter = |index| -> Result<Tensor> {
        Ok(Tensor::f_stack(
            &[
                reorder(&layer.parameters[index])?,
                reorder(&layer.parameters[index + 4])?,
            ],
            0,
        )?)
    };
    let initial = graph.tensor(&parameter(0)?)?;
    let hidden = graph.tensor(&parameter(1)?)?;
    let bias = graph.tensor(&Tensor::f_cat(&[parameter(2)?, parameter(3)?], 1)?)?;
    let sequence = graph.op(
        "LSTM",
        &[&source, &initial, &hidden, &bias],
        vec![
            Attribute::text("direction", "bidirectional"),
            Attribute::int("hidden_size", layer.hidden),
        ],
    );
    let sequence = graph.op(
        "Transpose",
        &[&sequence],
        vec![Attribute::ints("perm", &[0, 2, 1, 3])],
    );
    let shape = graph.integers(&[2], &[-1, layer.hidden * 2]);
    let sequence = graph.op("Reshape", &[&sequence, &shape], vec![]);
    let weights = graph.tensor(&layer.weight.f_transpose(0, 1)?)?;
    let bias = graph.tensor(&layer.bias)?;
    let sequence = graph.op("MatMul", &[&sequence, &weights], vec![]);
    let sequence = graph.op("Add", &[&sequence, &bias], vec![]);
    let sequence = normalize(graph, &layer.norm, &sequence)?;
    let sequence = graph.op("Relu", &[&sequence], vec![]);
    let shape = graph.integers(&[4], &[frames, -1, 1, bins]);
    let sequence = graph.op("Reshape", &[&sequence, &shape], vec![]);
    let value = graph.op(
        "Transpose",
        &[&sequence],
        vec![Attribute::ints("perm", &[1, 2, 3, 0])],
    );
    Ok(Plane::new(value, [1, bins, frames]))
}
