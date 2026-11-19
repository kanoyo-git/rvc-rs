use std::path::Path;

use anyhow::{Context, Result, ensure};
use ort::session::SessionInputValue;
use ort::value::DynValue;
use tch::Tensor;

use super::{Affine, Config, DualPath, TorchNetwork, band_padding, separator_transform};
use crate::model::onnx::{
    GraphSession, Provider, TensorInput,
    graph::{Attribute, Builder, Dimension},
};
use crate::separation::onnx_layers::{
    Recurrent, concat, concat_strings, convolution, dimensions, glu, linear, recurrent, reduce,
    reshape, slice, swish, transpose,
};

pub(super) struct Network {
    pub(super) window: Tensor,
    encoder: GraphSession,
    separator: Vec<GraphSession>,
    decoder: GraphSession,
    stages: usize,
}

impl Network {
    pub(super) fn load(config: &Config, torch: TorchNetwork, provider: Provider) -> Result<Self> {
        let mut frequency = config.nfft / 2 + 1;
        let mut geometry = Vec::with_capacity(torch.encoders.len());
        for (channels, encoder) in config.dims[1..].iter().copied().zip(&torch.encoders) {
            let widths = config.split_bands(frequency)?;
            let mut encoded = [0; 3];
            let mut padding = [0; 3];
            for (band, layer) in encoder.bands.iter().enumerate() {
                padding[band] = band_padding(widths[band], layer.stride, layer.kernel);
                let padded = widths[band] + padding[band];
                ensure!(
                    padded >= layer.kernel,
                    "SCNet band is smaller than its convolution"
                );
                encoded[band] = (padded - layer.kernel) / layer.stride + 1;
                ensure!(
                    layer.kernel + (encoded[band] - 1) * layer.stride >= widths[band],
                    "SCNet band cannot be restored to its original width"
                );
            }
            frequency = encoded.iter().sum();
            geometry.push(StageGeometry {
                widths,
                encoded,
                padding,
                channels,
            });
        }
        let encoder = GraphSession::load(provider, "scnet-encoder", |directory, path| {
            export_encoder(config, &torch, &geometry, directory, path)
        })?;
        let separator = torch
            .dual_path
            .iter()
            .map(|layer| {
                GraphSession::load(provider, "scnet-separator", |directory, path| {
                    export_dual_path(layer, frequency, directory, path)
                })
            })
            .collect::<Result<_>>()?;
        let decoder = GraphSession::load(provider, "scnet-decoder", |directory, path| {
            export_decoder(config, &torch, &geometry, directory, path)
        })?;
        Ok(Self {
            window: torch.window,
            encoder,
            separator,
            decoder,
            stages: geometry.len(),
        })
    }

    pub(super) fn predict_features(&self, input: Tensor) -> Result<Tensor> {
        let input = TensorInput::<f32>::new(&input)?;
        let (mut value, skips) = {
            let mut session = self.encoder.session.borrow_mut();
            let output = session.run(ort::inputs![input.view()?])?;
            let skips = (0..self.stages)
                .map(|index| read_tensor(&output[index]))
                .collect::<Result<Vec<_>>>()?;
            (read_tensor(&output[self.stages])?, skips)
        };
        for (index, path) in self.separator.iter().enumerate() {
            let input = TensorInput::<f32>::new(&value)?;
            let mut session = path.session.borrow_mut();
            let output = session.run(ort::inputs![input.view()?])?;
            value = separator_transform(&read_tensor(&output[0])?, index % 2 == 0)?;
        }
        let values = std::iter::once(&value)
            .chain(&skips)
            .map(TensorInput::<f32>::new)
            .collect::<Result<Vec<_>>>()?;
        let inputs = values
            .iter()
            .map(|value| Ok(SessionInputValue::from(value.view()?)))
            .collect::<Result<Vec<_>>>()?;
        let mut session = self.decoder.session.borrow_mut();
        let output = session.run(inputs.as_slice())?;
        read_tensor(&output[0])
    }
}

fn read_tensor(output: &DynValue) -> Result<Tensor> {
    let (shape, values) = output.try_extract_tensor::<f32>()?;
    ensure!(
        shape.len() == 4,
        "invalid SCNet ONNX feature shape {shape:?}"
    );
    Ok(Tensor::f_from_slice(values)?.f_view(&shape[..])?)
}

struct StageGeometry {
    widths: [i64; 3],
    encoded: [i64; 3],
    padding: [i64; 3],
    channels: i64,
}

fn shape(channels: i64, bins: i64) -> Vec<Dimension> {
    vec![
        Dimension::variable("batch"),
        Dimension::fixed(channels),
        Dimension::fixed(bins),
        Dimension::variable("frames"),
    ]
}

fn export_encoder(
    config: &Config,
    torch: &TorchNetwork,
    stages: &[StageGeometry],
    directory: &Path,
    path: &Path,
) -> Result<()> {
    let mut graph = Builder::new(
        directory,
        "PyMSS SCNet encoder",
        shape(2 * config.audio_channels, config.nfft / 2 + 1),
    )?;
    let mut value = "input".to_owned();
    let frames = dimensions(&mut graph, &value, 3, 4);
    let batch = dimensions(&mut graph, &value, 0, 1);
    for (encoder, stage) in torch.encoders.iter().zip(stages) {
        let mut bands = Vec::with_capacity(3);
        let mut offset = 0;
        for (index, band) in encoder.bands.iter().enumerate() {
            let width = stage.widths[index];
            let source = slice(&mut graph, &value, 2, offset, offset + width, 1);
            let pad = stage.padding[index];
            let source = conv(
                &mut graph,
                "Conv",
                &band.down,
                &source,
                &[band.stride, 1],
                &[pad / 2, 0, pad - pad / 2, 0],
                1,
            )?;
            let source = transpose(&mut graph, &source, &[0, 2, 1, 3]);
            let merged = graph.integers(&[2], &[-1, stage.channels]);
            let flat_shape = concat(&mut graph, &[&merged, &frames], 0);
            let mut source = graph.op("Reshape", &[&source, &flat_shape], vec![]);
            let half = config.conv_kernel / 2;
            for layer in &band.layers {
                let hidden = layer.depthwise.bias.size()[0];
                let inner = group_norm(&mut graph, &layer.norm, &source, 3)?;
                let inner = conv(
                    &mut graph,
                    "Conv",
                    &layer.expand,
                    &inner,
                    &[1],
                    &[half; 2],
                    1,
                )?;
                let inner = glu(&mut graph, &inner, 1, hidden);
                let inner = conv(
                    &mut graph,
                    "Conv",
                    &layer.depthwise,
                    &inner,
                    &[1],
                    &[half; 2],
                    hidden,
                )?;
                let inner = group_norm(&mut graph, &layer.hidden_norm, &inner, 3)?;
                let inner = swish(&mut graph, &inner);
                let inner = conv(&mut graph, "Conv", &layer.project, &inner, &[1], &[0; 2], 1)?;
                source = graph.op("Add", &[&source, &inner], vec![]);
            }
            let axes = graph.integers(&[2], &[stage.encoded[index], stage.channels]);
            let restored = concat(&mut graph, &[&batch, &axes, &frames], 0);
            let source = graph.op("Reshape", &[&source, &restored], vec![]);
            let source = transpose(&mut graph, &source, &[0, 2, 1, 3]);
            bands.push(graph.op("Gelu", &[&source], vec![]));
            offset += width;
        }
        let skip = concat_strings(&mut graph, &bands, 2);
        value = conv(
            &mut graph,
            "Conv",
            &encoder.global,
            &skip,
            &[1, 1],
            &[1; 4],
            1,
        )?;
        graph.output(skip, shape(stage.channels, stage.encoded.iter().sum()));
    }
    let last = stages
        .last()
        .context("SCNet has no encoded feature stages")?;
    graph.finish(path, value, shape(last.channels, last.encoded.iter().sum()))
}

fn export_decoder(
    config: &Config,
    torch: &TorchNetwork,
    stages: &[StageGeometry],
    directory: &Path,
    path: &Path,
) -> Result<()> {
    let last = stages
        .last()
        .context("SCNet has no decoded feature stages")?;
    let mut graph = Builder::new(
        directory,
        "PyMSS SCNet decoder",
        shape(last.channels, last.encoded.iter().sum()),
    )?;
    for (index, stage) in stages.iter().enumerate() {
        graph.input(
            &format!("skip_{index}"),
            shape(stage.channels, stage.encoded.iter().sum()),
        );
    }
    let mut value = "input".to_owned();
    for (index, decoder) in torch.decoders.iter().enumerate() {
        let reverse = stages.len() - 1 - index;
        let stage = &stages[reverse];
        let skip = format!("skip_{reverse}");
        let fused = graph.op("Add", &[&value, &skip], vec![]);
        let fused = concat(&mut graph, &[&fused, &fused], 1);
        let fused = conv(
            &mut graph,
            "Conv",
            &decoder.fusion,
            &fused,
            &[1, 1],
            &[1; 4],
            1,
        )?;
        let fused = glu(&mut graph, &fused, 1, stage.channels);
        let mut offset = 0;
        let mut bands = Vec::with_capacity(3);
        for (index, layer) in decoder.up.iter().enumerate() {
            let width = stage.encoded[index];
            let stride = config.band_stride[index];
            let source = slice(&mut graph, &fused, 2, offset, offset + width, 1);
            let source = conv(
                &mut graph,
                "ConvTranspose",
                layer,
                &source,
                &[stride, 1],
                &[0; 4],
                1,
            )?;
            let actual = (width - 1) * stride + config.band_kernel[index];
            let original = stage.widths[index];
            let trim = (actual - original) / 2;
            bands.push(slice(&mut graph, &source, 2, trim, trim + original, 1));
            offset += width;
        }
        value = concat_strings(&mut graph, &bands, 2);
    }
    graph.finish(
        path,
        value,
        shape(
            2 * config.audio_channels * config.sources.len() as i64,
            config.nfft / 2 + 1,
        ),
    )
}

fn export_dual_path(model: &DualPath, bins: i64, directory: &Path, path: &Path) -> Result<()> {
    let channels = model.frequency.norm.bias.size()[0];
    let mut graph = Builder::new(directory, "PyMSS SCNet separator", shape(channels, bins))?;
    let batch = dimensions(&mut graph, "input", 0, 1);
    let frames = dimensions(&mut graph, "input", 3, 4);
    let band_width = graph.integers(&[1], &[bins]);
    let channel_width = graph.integers(&[1], &[channels]);
    let inferred = graph.integers(&[1], &[-1]);

    let along_bins = group_norm(&mut graph, &model.frequency.norm, "input", 4)?;
    let along_bins = transpose(&mut graph, &along_bins, &[0, 3, 2, 1]);
    let along_bins = reshape(&mut graph, &along_bins, &[-1, bins, channels]);
    let along_bins = recurrent(
        &mut graph,
        &along_bins,
        &model.frequency.params,
        model.frequency.hidden,
        true,
        Recurrent::Lstm,
    )?;
    let projection = &model.frequency.project;
    let along_bins = linear(
        &mut graph,
        &along_bins,
        &projection.weight,
        Some(&projection.bias),
    )?;
    let bins_shape = concat(
        &mut graph,
        &[&batch, &frames, &band_width, &channel_width],
        0,
    );
    let along_bins = graph.op("Reshape", &[&along_bins, &bins_shape], vec![]);
    let along_bins = transpose(&mut graph, &along_bins, &[0, 3, 2, 1]);
    let residual = graph.op("Add", &["input", &along_bins], vec![]);

    let along_frames = group_norm(&mut graph, &model.time.norm, &residual, 4)?;
    let along_frames = transpose(&mut graph, &along_frames, &[0, 2, 3, 1]);
    let flat_shape = concat(&mut graph, &[&inferred, &frames, &channel_width], 0);
    let along_frames = graph.op("Reshape", &[&along_frames, &flat_shape], vec![]);
    let along_frames = recurrent(
        &mut graph,
        &along_frames,
        &model.time.params,
        model.time.hidden,
        true,
        Recurrent::Lstm,
    )?;
    let projection = &model.time.project;
    let along_frames = linear(
        &mut graph,
        &along_frames,
        &projection.weight,
        Some(&projection.bias),
    )?;
    let full_shape = concat(
        &mut graph,
        &[&batch, &band_width, &frames, &channel_width],
        0,
    );
    let along_frames = graph.op("Reshape", &[&along_frames, &full_shape], vec![]);
    let along_frames = transpose(&mut graph, &along_frames, &[0, 3, 1, 2]);
    let result = graph.op("Add", &[&residual, &along_frames], vec![]);
    graph.finish(path, result, shape(channels, bins))
}

fn group_norm(graph: &mut Builder, layer: &Affine, input: &str, rank: i64) -> Result<String> {
    let axes = (1..rank).collect::<Vec<_>>();
    let mean = reduce(graph, "ReduceMean", input, &axes, true);
    let centered = graph.op("Sub", &[input, &mean], vec![]);
    let squared = graph.op("Mul", &[&centered, &centered], vec![]);
    let variance = reduce(graph, "ReduceMean", &squared, &axes, true);
    let epsilon = graph.scale(1e-5);
    let biased = graph.op("Add", &[&variance, &epsilon], vec![]);
    let std = graph.op("Sqrt", &[&biased], vec![]);
    let norm = graph.op("Div", &[&centered, &std], vec![]);
    let mut shape = vec![1; rank as usize];
    shape[1] = layer.bias.size()[0];
    let weight = graph.tensor(&layer.weight.f_reshape(&shape[..])?)?;
    let bias = graph.tensor(&layer.bias.f_reshape(&shape[..])?)?;
    let value = graph.op("Mul", &[&norm, &weight], vec![]);
    Ok(graph.op("Add", &[&value, &bias], vec![]))
}

fn conv(
    graph: &mut Builder,
    op: &str,
    layer: &Affine,
    input: &str,
    strides: &[i64],
    pads: &[i64],
    groups: i64,
) -> Result<String> {
    convolution(
        graph,
        op,
        input,
        &layer.weight,
        Some(&layer.bias),
        vec![
            Attribute::ints("strides", strides),
            Attribute::ints("pads", pads),
            Attribute::int("group", groups),
        ],
    )
}
