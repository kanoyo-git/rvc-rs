use std::cell::RefCell;
use std::path::Path;

use anyhow::{Context, Result, ensure};
use tch::Tensor;

use super::{Config, Demucs, TorchNetwork};
use crate::model::onnx::{
    GraphSession, Provider, TensorInput,
    graph::{Attribute, Builder, Dimension},
};
use crate::model::weights::Weights;
use crate::separation::onnx_layers::{convolution, reduce, reshape};

#[path = "onnx_stages.rs"]
mod stages;

pub(super) struct Network {
    pub(super) window: Tensor,
    model: GraphSession,
    input_shape: [i64; 3],
}

pub(super) struct LegacySession {
    provider: Provider,
    network: RefCell<Option<(usize, Network)>>,
}

impl LegacySession {
    pub(super) fn new(provider: Provider) -> Self {
        Self {
            provider,
            network: RefCell::new(None),
        }
    }

    pub(super) fn predict(&self, demucs: &Demucs, audio: &[f32]) -> Result<Vec<f32>> {
        let super::Backend::Torch(torch) = &demucs.backend else {
            anyhow::bail!("legacy Demucs has no CPU weights to export");
        };
        let frames = super::super::window_samples(audio, 1, demucs.channels)?;
        let length = usize::try_from(frames)?;
        let mut cached = self.network.borrow_mut();
        if cached.as_ref().is_none_or(|(frames, _)| *frames != length) {
            *cached = None;
            let network = Network::load(demucs, torch, self.provider, length)?;
            *cached = Some((length, network));
        }
        let (_, network) = cached
            .as_ref()
            .context("absent legacy Demucs ONNX session")?;
        let mix = Tensor::f_from_slice(audio)?.f_reshape([1, demucs.channels as i64, frames])?;
        let output = tch::no_grad(|| {
            demucs.forward(&network.window, tch::Device::Cpu, mix, |tensor, wave| {
                network.estimate(&tensor, wave)
            })
        })?;
        Ok(Vec::<f32>::try_from(output.f_reshape([-1])?)?)
    }
}

impl Network {
    pub(super) fn load(
        demucs: &Demucs,
        torch: &TorchNetwork,
        provider: Provider,
        window_frames: usize,
    ) -> Result<Self> {
        let frames = i64::try_from(window_frames)?;
        ensure!(frames > 0, "HTDemucs needs a positive window length");
        let length = if demucs.config.use_train_segment {
            frames.max(demucs.training_frames)
        } else {
            frames
        };
        let config = &demucs.config;
        let hop = config.nfft / 4;
        let bins = config.nfft / 2 / config.num_subbands;
        let channels = demucs.channels as i64 * config.num_subbands * (1 + i64::from(config.cac));
        let steps = if config.hybrid {
            (length - 1) / hop + 1
        } else {
            length / hop + 1
        };
        let input_shape = [channels, bins, steps];
        let model = GraphSession::load(provider, "htdemucs", |directory, path| {
            export(demucs, &torch.weights, input_shape, length, directory, path)
        })?;
        Ok(Self {
            window: torch.window.shallow_clone(),
            model,
            input_shape,
        })
    }

    pub(super) fn estimate(
        &self,
        spectrum: &Tensor,
        wave: Option<&Tensor>,
    ) -> Result<(Tensor, Option<Tensor>)> {
        ensure!(
            spectrum.size()[1..] == self.input_shape,
            "HTDemucs ONNX window does not match the model's chunk size"
        );
        let spectral = TensorInput::<f32>::new(spectrum)?;
        let waveform = wave.map(TensorInput::<f32>::new).transpose()?;
        let mut session = self.model.session.borrow_mut();
        let output = if let Some(waveform) = &waveform {
            session.run(ort::inputs![spectral.view()?, waveform.view()?])?
        } else {
            session.run(ort::inputs![spectral.view()?])?
        };
        let convert = |index: usize| -> Result<Tensor> {
            let (shape, values) = output[index].try_extract_tensor::<f32>()?;
            Ok(Tensor::f_from_slice(values)?.f_view(&shape[..])?)
        };
        let waveform = wave.map(|_| convert(1)).transpose()?;
        Ok((convert(0)?, waveform))
    }
}

#[derive(Clone)]
pub(super) struct Feature {
    pub(super) value: String,
    /// Shape excluding the dynamic batch dimension.
    pub(super) shape: Vec<i64>,
}

impl Feature {
    pub(super) fn new(value: impl Into<String>, shape: impl Into<Vec<i64>>) -> Self {
        Self {
            value: value.into(),
            shape: shape.into(),
        }
    }

    pub(super) fn with_value(&self, value: String) -> Self {
        Self::new(value, self.shape.clone())
    }

    fn length(&self) -> i64 {
        self.shape[self.shape.len() - 1]
    }
}

pub(super) struct Export<'a> {
    pub(super) graph: Builder,
    pub(super) weights: &'a Weights,
    config: &'a Config,
    legacy: bool,
}

fn dimensions(shape: &[i64]) -> Vec<Dimension> {
    std::iter::once(Dimension::variable("batch"))
        .chain(shape.iter().copied().map(Dimension::fixed))
        .collect()
}

fn export(
    demucs: &Demucs,
    weights: &Weights,
    input_shape: [i64; 3],
    length: i64,
    directory: &Path,
    path: &Path,
) -> Result<()> {
    let config = &demucs.config;
    let graph = Builder::new(directory, "PyMSS HTDemucs", dimensions(&input_shape))?;
    let mut model = Export {
        graph,
        weights,
        config,
        legacy: demucs.legacy,
    };
    let mut x = Feature::new("input", input_shape);
    let mut wave = config.hybrid.then(|| {
        let shape = [demucs.channels as i64, length];
        model.graph.input("wave", dimensions(&shape));
        Feature::new("wave", shape)
    });
    let mut skips = Vec::with_capacity(demucs.encoders.len());
    let mut time_skips = Vec::with_capacity(demucs.time_encoders.len());
    for (index, encoder) in demucs.encoders.iter().enumerate() {
        let before = x.length();
        let mut inject = None;
        if let Some(stage) = demucs.time_encoders.get(index) {
            let input = wave.as_ref().context("absent HTDemucs waveform branch")?;
            let y = model.encode(stage, &format!("tencoder.{index}"), input, None)?;
            time_skips.push((y.clone(), input.length()));
            if stage.empty() {
                let shape = [y.shape[0], 1, y.length()];
                inject = Some(model.reshape(&y, &shape));
            }
            wave = Some(y);
        }
        x = model.encode(encoder, &format!("encoder.{index}"), &x, inject.as_ref())?;
        if index == 0 && config.freq_emb != 0. && weights.contains("freq_emb.embedding.weight") {
            let weight = weights
                .get("freq_emb.embedding.weight")?
                .f_transpose(0, 1)?
                .f_unsqueeze(0)?
                .f_unsqueeze(-1)?
                .f_mul_scalar(config.freq_emb * config.emb_scale)?;
            let weight = model.graph.tensor(&weight)?;
            x.value = model.graph.op("Add", &[&x.value, &weight], vec![]);
        }
        skips.push((x.clone(), before));
    }
    if let Some(transformer) = &demucs.transformer {
        let time = wave
            .as_ref()
            .context("absent HTDemucs transformer waveform branch")?;
        let (spectral, waveform) = transformer.export_onnx(&mut model, &x, time)?;
        x = spectral;
        wave = Some(waveform);
    }
    if demucs.legacy {
        x.value = model.zeroes(&x);
        if let Some(time) = &mut wave {
            time.value = model.zeroes(time);
        }
    }
    for (decode_index, (index, (skip, length))) in skips.into_iter().enumerate().rev().enumerate() {
        let (spectral, pre) = model.decode(
            &demucs.encoders[index],
            &format!("decoder.{decode_index}"),
            &x,
            Some(&skip),
            length,
            index == 0,
        )?;
        x = spectral;
        if let Some(stage) = demucs.time_encoders.get(index) {
            let (skip, samples) = time_skips.pop().context("absent HTDemucs waveform skip")?;
            let path = format!("tdecoder.{}", demucs.time_encoders.len() - index - 1);
            let (decoded, _) = if stage.empty() {
                ensure!(pre.shape[1] == 1, "invalid HTDemucs time branch injection");
                let time = model.reshape(&pre, &[pre.shape[0], pre.length()]);
                model.decode(stage, &path, &time, None, samples, index == 0)?
            } else {
                let time = wave
                    .as_ref()
                    .context("absent HTDemucs waveform decoder input")?;
                model.decode(stage, &path, time, Some(&skip), samples, index == 0)?
            };
            wave = Some(decoded);
        }
    }
    if let Some(wave) = wave {
        model.graph.output(x.value, dimensions(&x.shape));
        model
            .graph
            .finish(path, wave.value, dimensions(&wave.shape))
    } else {
        model.graph.finish(path, x.value, dimensions(&x.shape))
    }
}

impl Export<'_> {
    fn zeroes(&mut self, input: &Feature) -> String {
        let shape = self.graph.op("Shape", &[&input.value], vec![]);
        self.graph.op("ConstantOfShape", &[&shape], vec![])
    }

    pub(super) fn reshape(&mut self, input: &Feature, shape: &[i64]) -> Feature {
        let batch = std::iter::once(0)
            .chain(shape.iter().copied())
            .collect::<Vec<_>>();
        Feature::new(
            reshape(&mut self.graph, &input.value, &batch),
            shape.to_vec(),
        )
    }

    pub(super) fn conv(
        &mut self,
        key: &str,
        input: &Feature,
        strides: &[i64],
        pads: &[i64],
        dilation: i64,
        transpose: bool,
    ) -> Result<Feature> {
        let weight = self.weights.get(&format!("{key}.weight"))?;
        let bias = self.weights.get(&format!("{key}.bias")).ok();
        let kernel = weight.size();
        ensure!(
            strides.len() + 2 == kernel.len() && strides.len() + 1 == input.shape.len(),
            "invalid HTDemucs convolution dimensions at {key}"
        );
        let output_channels = kernel[usize::from(transpose)];
        let mut shape = vec![output_channels];
        let kernel_dilations = vec![dilation; strides.len()];
        for (axis, stride) in strides.iter().enumerate() {
            let pad = pads[axis] + pads[axis + strides.len()];
            let extent = (kernel[axis + 2] - 1) * dilation + 1;
            let input = input.shape[axis + 1];
            if transpose {
                shape.push((input - 1) * stride + extent - pad);
            } else {
                ensure!(
                    input + pad >= extent,
                    "HTDemucs convolution input is too short at {key}"
                );
                shape.push((input + pad - extent) / stride + 1);
            }
        }
        let value = convolution(
            &mut self.graph,
            if transpose { "ConvTranspose" } else { "Conv" },
            &input.value,
            weight,
            bias,
            vec![
                Attribute::ints("strides", strides),
                Attribute::ints("pads", pads),
                Attribute::ints("dilations", &kernel_dilations),
            ],
        )?;
        Ok(Feature::new(value, shape))
    }

    pub(super) fn group_norm(
        &mut self,
        key: &str,
        input: &Feature,
        groups: i64,
    ) -> Result<Feature> {
        let Ok(weight) = self.weights.get(&format!("{key}.weight")) else {
            return Ok(input.clone());
        };
        let channels = input.shape[0];
        ensure!(
            groups > 0 && channels % groups == 0,
            "invalid HTDemucs group normalization"
        );
        let mut broadcast = vec![1; input.shape.len() + 1];
        broadcast[1] = channels;
        let bias = self.weights.get(&format!("{key}.bias"))?;
        let gamma = self.graph.tensor(&weight.f_reshape(&broadcast[..])?)?;
        let beta = self.graph.tensor(&bias.f_reshape(&broadcast[..])?)?;
        let length = input.shape.iter().product::<i64>() / groups;
        let value = reshape(&mut self.graph, &input.value, &[0, groups, -1]);
        let mean = group_mean(&mut self.graph, &value, length);
        let value = self.graph.op("Sub", &[&value, &mean], vec![]);
        let square = self.graph.op("Mul", &[&value, &value], vec![]);
        let variance = group_mean(&mut self.graph, &square, length);
        let epsilon = self.graph.scale(1e-5);
        let variance = self.graph.op("Add", &[&variance, &epsilon], vec![]);
        let std = self.graph.op("Sqrt", &[&variance], vec![]);
        let normalized = self.graph.op("Div", &[&value, &std], vec![]);
        let out = self.reshape(&input.with_value(normalized), &input.shape);
        let value = self.graph.op("Mul", &[&out.value, &gamma], vec![]);
        Ok(input.with_value(self.graph.op("Add", &[&value, &beta], vec![])))
    }

    fn gelu(&mut self, input: &Feature) -> Feature {
        input.with_value(self.graph.op("Gelu", &[&input.value], vec![]))
    }
}

fn group_mean(graph: &mut Builder, input: &str, length: i64) -> String {
    const BLOCK: i64 = 128;
    if length <= BLOCK * 8 {
        return reduce(graph, "ReduceMean", input, &[2], true);
    }
    // Normalize long HTDemucs features with small reductions to avoid losing
    // precision in a single sum over several million values.
    let blocks = (length - 1) / BLOCK + 1;
    let padding = blocks * BLOCK - length;
    let value = if padding == 0 {
        input.to_owned()
    } else {
        let pads = graph.integers(&[6], &[0, 0, 0, 0, 0, padding]);
        graph.op("Pad", &[input, &pads], vec![])
    };
    let value = reshape(graph, &value, &[0, 0, blocks, BLOCK]);
    let averages = reduce(graph, "ReduceMean", &value, &[3], false);
    let mean = reduce(graph, "ReduceMean", &averages, &[2], true);
    if padding == 0 {
        mean
    } else {
        let scale = graph.scale((blocks * BLOCK) as f32 / length as f32);
        graph.op("Mul", &[&mean, &scale], vec![])
    }
}
