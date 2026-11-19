use anyhow::{Context, Result, ensure};
use std::cell::RefCell;
use tch::{Device, Kind, Tensor};

use super::{Demucs, Tasnet, TasnetNorm};
use crate::model::onnx::{
    GraphSession, Provider, TensorInput,
    graph::{Attribute, Builder, Dimension},
};
use crate::model::weights::Weights;
use crate::separation::onnx_layers::{
    Recurrent, concat_strings, convolution, glu, linear, recurrent, reduce, reshape, select, slice,
    transpose, unsqueeze,
};

pub(super) struct Network {
    provider: Provider,
    session: RefCell<Option<(i64, GraphSession)>>,
}

impl Network {
    pub(super) fn new(provider: Provider) -> Self {
        Self {
            provider,
            session: RefCell::new(None),
        }
    }

    pub(super) fn demucs(
        &self,
        model: &Demucs,
        weights: &Weights,
        input: &Tensor,
    ) -> Result<Tensor> {
        self.run(weights, input, "waveform-demucs", |graph, input| {
            graph.demucs(model, input)
        })
    }

    pub(super) fn tasnet(
        &self,
        model: &Tasnet,
        weights: &Weights,
        input: &Tensor,
        stems: i64,
    ) -> Result<Tensor> {
        self.run(weights, input, "conv-tasnet", |graph, input| {
            graph.tasnet(model, input, stems)
        })
    }

    fn run(
        &self,
        weights: &Weights,
        input: &Tensor,
        name: &str,
        export: impl FnOnce(&mut Layers<'_>, Series) -> Result<(String, Vec<i64>)>,
    ) -> Result<Tensor> {
        let (_, channels, length) = input.size3()?;
        let mut cached = self.session.borrow_mut();
        if cached
            .as_ref()
            .is_none_or(|(length_used, _)| *length_used != length)
        {
            *cached = None;
            let model = GraphSession::load(self.provider, name, |directory, path| {
                let mut graph = Builder::new(directory, name, dimensions(&[channels, length]))?;
                let mut layers = Layers {
                    graph: &mut graph,
                    weights,
                };
                let input = Series::new("input", channels, length);
                let (value, shape) = export(&mut layers, input)?;
                graph.finish(path, value, dimensions(&shape))
            })?;
            *cached = Some((length, model));
        }
        let (_, model) = cached.as_ref().context("absent legacy ONNX model")?;
        let input = TensorInput::<f32>::new(input)?;
        let mut session = model.session.borrow_mut();
        let output = session.run(ort::inputs![input.view()?])?;
        let (shape, values) = output[0].try_extract_tensor::<f32>()?;
        Ok(Tensor::f_from_slice(values)?.f_view(&shape[..])?)
    }
}

fn dimensions(shape: &[i64]) -> Vec<Dimension> {
    std::iter::once(Dimension::variable("batch"))
        .chain(shape.iter().copied().map(Dimension::fixed))
        .collect()
}

#[derive(Clone)]
pub(in crate::separation) struct Series {
    pub(in crate::separation) value: String,
    channels: i64,
    length: i64,
}

impl Series {
    pub(in crate::separation) fn new(value: impl Into<String>, channels: i64, length: i64) -> Self {
        Self {
            value: value.into(),
            channels,
            length,
        }
    }
}

pub(in crate::separation) struct Layers<'a> {
    pub(in crate::separation) graph: &'a mut Builder,
    pub(in crate::separation) weights: &'a Weights,
}

impl Layers<'_> {
    fn demucs(&mut self, model: &Demucs, mut x: Series) -> Result<(String, Vec<i64>)> {
        let args = &model.args;
        let mut skips = Vec::with_capacity(args.depth);
        for index in 0..args.depth {
            let key = format!("encoder.{index}");
            x = self.conv(&format!("{key}.0"), &x, args.stride, 0, 1, 1)?;
            let normalize = model.is_v3 && index >= args.norm_starts;
            if normalize {
                x = self.group(&format!("{key}.1"), x, args.norm_groups)?;
            }
            x = self.activation(x, args.gelu);
            let mut next = if model.is_v3 { 3 } else { 2 };
            if args.dconv_mode & 1 != 0 {
                x = self.residual(
                    x,
                    &format!("{key}.{next}"),
                    args.dconv_depth,
                    index >= args.dconv_lstm,
                    index >= args.dconv_attn,
                )?;
                next += 1;
            }
            if args.rewrite {
                x = self.conv(&format!("{key}.{next}"), &x, 1, 0, 1, 1)?;
                if normalize {
                    x = self.group(&format!("{key}.{}", next + 1), x, args.norm_groups)?;
                }
                x = self.gate(x, args.glu);
            }
            skips.push(x.clone());
        }
        if args.lstm_layers > 0 {
            x = self.blstm(&x, "lstm", args.lstm_layers, false)?;
        }
        for (index, skip) in skips.into_iter().enumerate().rev() {
            ensure!(
                skip.length >= x.length,
                "invalid waveform Demucs skip length"
            );
            let key = format!("decoder.{}", args.depth - 1 - index);
            let begin = (skip.length - x.length) / 2;
            let skip = slice(self.graph, &skip.value, -1, begin, begin + x.length, 1);
            x.value = self.graph.op("Add", &[&x.value, &skip], vec![]);
            let normalize = model.is_v3 && index >= args.norm_starts;
            let mut next = 0;
            if args.rewrite {
                let pad = if model.is_v3 { args.context } else { 0 };
                x = self.conv(&format!("{key}.0"), &x, 1, pad, 1, 1)?;
                if normalize {
                    x = self.group(&format!("{key}.1"), x, args.norm_groups)?;
                }
                x = self.gate(x, args.glu);
                next = if model.is_v3 { 3 } else { 2 };
            }
            if args.dconv_mode & 2 != 0 {
                x = self.residual(
                    x,
                    &format!("{key}.{next}"),
                    args.dconv_depth,
                    index >= args.dconv_lstm,
                    index >= args.dconv_attn,
                )?;
                next += 1;
            }
            x = self.deconv(&format!("{key}.{next}"), &x, args.stride)?;
            if index > 0 {
                if normalize {
                    x = self.group(&format!("{key}.{}", next + 1), x, args.norm_groups)?;
                }
                x = self.activation(x, args.gelu);
            }
        }
        Ok((x.value, vec![x.channels, x.length]))
    }

    pub(in crate::separation) fn residual(
        &mut self,
        mut x: Series,
        key: &str,
        depth: i32,
        recurrent: bool,
        attention: bool,
    ) -> Result<Series> {
        for level in 0..depth.unsigned_abs() {
            let dilation = if depth > 0 { 1 << level } else { 1 };
            let name = format!("{key}.layers.{level}");
            let mut y = self.conv(&format!("{name}.0"), &x, 1, dilation, dilation, 1)?;
            y = self.group(&format!("{name}.1"), y, 1)?;
            y = self.activation(y, true);
            let mut next = 3;
            if recurrent {
                y = self.blstm(&y, &format!("{name}.{next}"), 2, true)?;
                next += 1;
            }
            if attention {
                y = self.local_attention(&y, &format!("{name}.{next}"))?;
                next += 1;
            }
            y = self.conv(&format!("{name}.{next}"), &y, 1, 0, 1, 1)?;
            y = self.group(&format!("{name}.{}", next + 1), y, 1)?;
            y = self.gate(y, true);
            let scale = self
                .weights
                .get(&format!("{name}.{}.scale", next + 3))?
                .f_view([1, -1, 1])?;
            let scale = self.graph.tensor(&scale)?;
            let scaled = self.graph.op("Mul", &[&scale, &y.value], vec![]);
            x.value = self.graph.op("Add", &[&x.value, &scaled], vec![]);
        }
        Ok(x)
    }

    fn blstm(&mut self, input: &Series, name: &str, layers: i64, framed: bool) -> Result<Series> {
        const WIDTH: i64 = 200;
        const STRIDE: i64 = WIDTH / 2;
        const CROP: i64 = STRIDE / 2;
        let mut value = input.value.clone();
        let mut count = 1;
        if framed && input.length > WIDTH {
            count = (input.length - WIDTH - 1) / STRIDE + 2;
            let pad = (count - 1) * STRIDE + WIDTH - input.length;
            value = pad_right(self.graph, &value, 3, pad);
            let positions = (0..count)
                .flat_map(|frame| {
                    let first = frame * STRIDE;
                    first..first + WIDTH
                })
                .collect::<Vec<_>>();
            let positions = self.graph.integers(&[count, WIDTH], &positions);
            value = self.graph.op(
                "Gather",
                &[&value, &positions],
                vec![Attribute::int("axis", -1)],
            );
            value = transpose(self.graph, &value, &[0, 2, 1, 3]);
            value = reshape(self.graph, &value, &[-1, input.channels, WIDTH]);
        }
        value = transpose(self.graph, &value, &[0, 2, 1]);
        for layer in 0..layers {
            let mut params = Vec::with_capacity(8);
            for direction in ["", "_reverse"] {
                for field in ["weight_ih", "weight_hh", "bias_ih", "bias_hh"] {
                    params.push(
                        self.weights
                            .get(&format!("{name}.lstm.{field}_l{layer}{direction}"))?
                            .shallow_clone(),
                    );
                }
            }
            value = recurrent(
                self.graph,
                &value,
                &params,
                input.channels,
                true,
                Recurrent::Lstm,
            )?;
        }
        value = self.linear(&format!("{name}.linear"), &value)?;
        value = transpose(self.graph, &value, &[0, 2, 1]);
        if count > 1 {
            let frames = reshape(self.graph, &value, &[-1, count, input.channels, WIDTH]);
            let mut chunks = Vec::with_capacity(count as usize);
            for index in 0..count {
                let begin = if index == 0 { 0 } else { CROP };
                let end = if index == count - 1 {
                    WIDTH
                } else {
                    WIDTH - CROP
                };
                let frame = select(self.graph, &frames, 1, index);
                chunks.push(slice(self.graph, &frame, -1, begin, end, 1));
            }
            value = concat_strings(self.graph, &chunks, -1);
            value = slice(self.graph, &value, -1, 0, input.length, 1);
        }
        if framed {
            value = self.graph.op("Add", &[&input.value, &value], vec![]);
        }
        Ok(Series::new(value, input.channels, input.length))
    }

    fn local_attention(&mut self, input: &Series, name: &str) -> Result<Series> {
        const HEADS: i64 = 4;
        const BLOCK: i64 = 384;
        let length = input.length;
        let width = input.channels / HEADS;
        let mut project = |key: &str, width| {
            let value = self.conv(&format!("{name}.{key}"), input, 1, 0, 1, 1)?;
            Ok::<String, anyhow::Error>(reshape(
                self.graph,
                &value.value,
                &[0, HEADS, width, length],
            ))
        };
        let q = project("query", width)?;
        let k = project("key", width)?;
        let v = project("content", width)?;
        let s = project("query_decay", 4)?;
        let graph = &mut *self.graph;
        let q = transpose(graph, &q, &[0, 1, 3, 2]);
        let v = transpose(graph, &v, &[0, 1, 3, 2]);
        let s = graph.op("Sigmoid", &[&s], vec![]);
        let rate = graph.floats(&[1, 1, 4, 1], &[-0.25, -0.5, -0.75, -1.]);
        let s = graph.op("Mul", &[&s, &rate], vec![]);
        let s = reduce(graph, "ReduceSum", &s, &[2], true);
        let s = transpose(graph, &s, &[0, 1, 3, 2]);
        let scale = graph.scale((width as f32).sqrt().recip());
        let zero = graph.scale(0.);
        let diag = graph.scale(-100.);
        let coords = (0..length).map(|i| i as f32).collect::<Vec<_>>();
        let coords = graph.floats(&[length], &coords);
        let columns = unsqueeze(graph, &coords, &[0]);
        let mut chunks = Vec::new();
        for begin in (0..length).step_by(BLOCK as usize) {
            let end = (begin + BLOCK).min(length);
            let rows = slice(graph, &coords, 0, begin, end, 1);
            let rows = unsqueeze(graph, &rows, &[1]);
            let delta = graph.op("Sub", &[&rows, &columns], vec![]);
            let delta = graph.op("Abs", &[&delta], vec![]);
            let diagonal = graph.op("Equal", &[&delta, &zero], vec![]);
            let strength = slice(graph, &s, 2, begin, end, 1);
            let bias = graph.op("Mul", &[&strength, &delta], vec![]);
            let queries = slice(graph, &q, 2, begin, end, 1);
            let score = graph.op("MatMul", &[&queries, &k], vec![]);
            let score = graph.op("Mul", &[&score, &scale], vec![]);
            let score = graph.op("Add", &[&score, &bias], vec![]);
            let score = graph.op("Where", &[&diagonal, &diag, &score], vec![]);
            let score = graph.op("Softmax", &[&score], vec![Attribute::int("axis", -1)]);
            chunks.push(graph.op("MatMul", &[&score, &v], vec![]));
        }
        let value = concat_strings(graph, &chunks, 2);
        let value = transpose(graph, &value, &[0, 1, 3, 2]);
        let value = reshape(graph, &value, &[0, input.channels, length]);
        let source = Series::new(value, input.channels, length);
        let mut out = self.conv(&format!("{name}.proj"), &source, 1, 0, 1, 1)?;
        out.value = self.graph.op("Add", &[&input.value, &out.value], vec![]);
        Ok(out)
    }

    fn tasnet(&mut self, model: &Tasnet, input: Series, stems: i64) -> Result<(String, Vec<i64>)> {
        let args = &model.args;
        let hop = args.kernel / 2;
        let encoded = self.conv("encoder.conv1d_U", &input, hop, 0, 1, 1)?;
        let encoded = self.activation(encoded, false);
        let frames = encoded.length;
        let features = encoded.channels;
        let prefix = "separator.network";
        let mut x = self.tasnet_norm(TasnetNorm::Channel, &format!("{prefix}.0"), &encoded)?;
        x = self.conv(&format!("{prefix}.1"), &x, 1, 0, 1, 1)?;
        for repeat in 0..args.repeats {
            for index in 0..args.blocks {
                let name = format!("{prefix}.2.{repeat}.{index}.net");
                let mut y = self.conv(&format!("{name}.0"), &x, 1, 0, 1, 1)?;
                y = self.prelu(&format!("{name}.1"), y)?;
                y = self.tasnet_norm(model.norm, &format!("{name}.2"), &y)?;
                let dilation = 1 << index;
                let spread = (args.residual_kernel - 1) * dilation;
                let pad = if args.causal { spread } else { spread / 2 };
                let inner = format!("{name}.3.net");
                y = self.conv(&format!("{inner}.0"), &y, 1, pad, dilation, args.hidden)?;
                let next = if args.causal {
                    y.length -= pad;
                    y.value = slice(self.graph, &y.value, -1, 0, y.length, 1);
                    2
                } else {
                    1
                };
                y = self.prelu(&format!("{inner}.{next}"), y)?;
                y = self.tasnet_norm(model.norm, &format!("{inner}.{}", next + 1), &y)?;
                y = self.conv(&format!("{inner}.{}", next + 2), &y, 1, 0, 1, 1)?;
                x.value = self.graph.op("Add", &[&x.value, &y.value], vec![]);
            }
        }
        let scores = self.conv(&format!("{prefix}.3"), &x, 1, 0, 1, 1)?;
        let scores = reshape(self.graph, &scores.value, &[0, stems, features, frames]);
        let mask = if model.softmax {
            self.graph
                .op("Softmax", &[&scores], vec![Attribute::int("axis", 1)])
        } else {
            self.graph.op("Relu", &[&scores], vec![])
        };
        let encoded = unsqueeze(self.graph, &encoded.value, &[1]);
        let source = self.graph.op("Mul", &[&encoded, &mask], vec![]);
        let source = reshape(self.graph, &source, &[-1, features, frames]);
        let channels = input.channels;
        let basis = self
            .weights
            .kernel("decoder.basis_signals")?
            .f_reshape([channels, args.kernel, features])?
            .f_permute([2, 0, 1])?;
        let mut decoded = convolution(
            self.graph,
            "ConvTranspose",
            &source,
            &basis,
            None,
            vec![Attribute::ints("strides", &[hop])],
        )?;
        let output_length = (frames - 1) * hop + args.kernel;
        if let Ok(bias) = self.weights.get("decoder.basis_signals.bias") {
            let pads = Tensor::f_ones([1, 1, frames], (Kind::Float, Device::Cpu))?;
            let bias = pads.f_conv_transpose1d(
                &bias.f_reshape([1, channels, args.kernel])?,
                None::<&Tensor>,
                hop,
                0,
                0,
                1,
                1,
            )?;
            let bias = self.graph.tensor(&bias)?;
            decoded = self.graph.op("Add", &[&decoded, &bias], vec![]);
        }
        let length = input.length;
        let decoded = if output_length > length {
            slice(self.graph, &decoded, -1, 0, length, 1)
        } else {
            pad_right(self.graph, &decoded, 3, length - output_length)
        };
        let output = reshape(self.graph, &decoded, &[-1, stems, channels, length]);
        Ok((output, vec![stems, channels, length]))
    }

    fn conv(
        &mut self,
        key: &str,
        x: &Series,
        stride: i64,
        pad: i64,
        dilation: i64,
        group: i64,
    ) -> Result<Series> {
        let weight = self.weights.kernel(key)?;
        let [channels, _, kernel] = <[i64; 3]>::try_from(weight.size().as_slice())?;
        let extent = (kernel - 1) * dilation + 1;
        ensure!(
            x.length + 2 * pad >= extent,
            "legacy input is too short at {key}"
        );
        let length = (x.length + 2 * pad - extent) / stride + 1;
        let value = convolution(
            self.graph,
            "Conv",
            &x.value,
            &weight,
            self.weights.get(&format!("{key}.bias")).ok(),
            vec![
                Attribute::ints("strides", &[stride]),
                Attribute::ints("pads", &[pad; 2]),
                Attribute::ints("dilations", &[dilation]),
                Attribute::int("group", group),
            ],
        )?;
        Ok(Series::new(value, channels, length))
    }

    fn deconv(&mut self, key: &str, x: &Series, stride: i64) -> Result<Series> {
        let weight = self.weights.kernel(key)?;
        let [_, channels, kernel] = <[i64; 3]>::try_from(weight.size().as_slice())?;
        let value = convolution(
            self.graph,
            "ConvTranspose",
            &x.value,
            &weight,
            self.weights.get(&format!("{key}.bias")).ok(),
            vec![Attribute::ints("strides", &[stride])],
        )?;
        Ok(Series::new(
            value,
            channels,
            (x.length - 1) * stride + kernel,
        ))
    }

    fn linear(&mut self, name: &str, x: &str) -> Result<String> {
        linear(
            self.graph,
            x,
            &self.weights.kernel(name)?,
            self.weights.get(&format!("{name}.bias")).ok(),
        )
    }

    fn activation(&mut self, mut input: Series, gelu: bool) -> Series {
        input.value = self
            .graph
            .op(if gelu { "Gelu" } else { "Relu" }, &[&input.value], vec![]);
        input
    }

    fn gate(&mut self, mut x: Series, use_glu: bool) -> Series {
        if use_glu {
            x.channels /= 2;
            x.value = glu(self.graph, &x.value, 1, x.channels);
            x
        } else {
            self.activation(x, false)
        }
    }

    fn group(&mut self, name: &str, mut x: Series, groups: i64) -> Result<Series> {
        ensure!(
            x.channels % groups == 0,
            "invalid legacy normalization at {name}"
        );
        let value = reshape(self.graph, &x.value, &[0, groups, -1]);
        let value = normalize(self.graph, &value, &[2], 1e-5);
        let value = reshape(self.graph, &value, &[0, x.channels, x.length]);
        let gamma = self
            .weights
            .get(&format!("{name}.weight"))?
            .f_reshape([1, -1, 1])?;
        let beta = self
            .weights
            .get(&format!("{name}.bias"))?
            .f_reshape([1, -1, 1])?;
        let gamma = self.graph.tensor(&gamma)?;
        let beta = self.graph.tensor(&beta)?;
        let value = self.graph.op("Mul", &[&value, &gamma], vec![]);
        x.value = self.graph.op("Add", &[&value, &beta], vec![]);
        Ok(x)
    }

    fn prelu(&mut self, name: &str, mut x: Series) -> Result<Series> {
        let weight = self
            .weights
            .get(&format!("{name}.weight"))?
            .f_reshape([1, -1, 1])?;
        let weight = self.graph.tensor(&weight)?;
        x.value = self.graph.op("PRelu", &[&x.value, &weight], vec![]);
        Ok(x)
    }

    fn tasnet_norm(&mut self, kind: TasnetNorm, name: &str, x: &Series) -> Result<Series> {
        let graph = &mut *self.graph;
        let (value, axes) = match kind {
            TasnetNorm::Identity => return Ok(x.clone()),
            TasnetNorm::Channel => (&x.value, &[1][..]),
            TasnetNorm::Global => (&x.value, &[1, 2][..]),
            TasnetNorm::Batch => {
                let mut inputs = vec![x.value.clone()];
                for field in ["weight", "bias", "running_mean", "running_var"] {
                    inputs.push(graph.tensor(self.weights.get(&format!("{name}.{field}"))?)?);
                }
                let inputs = inputs.iter().map(String::as_str).collect::<Vec<_>>();
                let value = graph.op(
                    "BatchNormalization",
                    &inputs,
                    vec![Attribute::float("epsilon", 1e-5)],
                );
                return Ok(Series::new(value, x.channels, x.length));
            }
        };
        let value = normalize(graph, value, axes, 1e-8);
        let gamma = graph.tensor(self.weights.get(&format!("{name}.gamma"))?)?;
        let beta = graph.tensor(self.weights.get(&format!("{name}.beta"))?)?;
        let value = graph.op("Mul", &[&value, &gamma], vec![]);
        let value = graph.op("Add", &[&value, &beta], vec![]);
        Ok(Series::new(value, x.channels, x.length))
    }
}

fn normalize(graph: &mut Builder, x: &str, axes: &[i64], epsilon: f32) -> String {
    let mean = reduce(graph, "ReduceMean", x, axes, true);
    let value = graph.op("Sub", &[x, &mean], vec![]);
    let square = graph.op("Mul", &[&value, &value], vec![]);
    let var = reduce(graph, "ReduceMean", &square, axes, true);
    let epsilon = graph.scale(epsilon);
    let variance = graph.op("Add", &[&var, &epsilon], vec![]);
    let std = graph.op("Sqrt", &[&variance], vec![]);
    graph.op("Div", &[&value, &std], vec![])
}

fn pad_right(graph: &mut Builder, x: &str, rank: usize, width: i64) -> String {
    if width == 0 {
        return x.to_owned();
    }
    let mut pads = vec![0; rank * 2];
    pads[rank * 2 - 1] = width;
    let pads = graph.integers(&[pads.len() as i64], &pads);
    graph.op("Pad", &[x, &pads], vec![])
}
