use std::path::Path;

use anyhow::{Result, ensure};
use tch::Tensor;

use super::{Config, SLOPE, Weights};
use crate::model::onnx::{
    GraphSession, Provider, TensorInput,
    graph::{Attribute, Builder, Dimension, Layers},
};

pub(super) struct Decoder(GraphSession);

impl Decoder {
    pub(super) fn load(
        weights: &Weights,
        config: &Config,
        uses_f0: bool,
        provider: Provider,
    ) -> Result<Self> {
        GraphSession::load(provider, "decoder", |directory, path| {
            export(weights, config, uses_f0, directory, path)
        })
        .map(Self)
    }

    pub(super) fn run(
        &self,
        latent: &Tensor,
        speaker: &Tensor,
        source: Option<&Tensor>,
    ) -> Result<Tensor> {
        let latent = TensorInput::<f32>::new(latent)?;
        let speaker = TensorInput::<f32>::new(speaker)?;
        let source = source.map(TensorInput::<f32>::new).transpose()?;
        let latent = latent.view()?;
        let speaker = speaker.view()?;
        let mut session = self.0.session.borrow_mut();
        let output = if let Some(source) = &source {
            session.run(ort::inputs![latent, speaker, source.view()?])?
        } else {
            session.run(ort::inputs![latent, speaker])?
        };
        let (shape, data) = output[0].try_extract_tensor::<f32>()?;
        let [1, 1, samples] = **shape else {
            anyhow::bail!("invalid ONNX decoder output shape {shape:?}")
        };
        ensure!(samples > 0, "empty ONNX decoder output");
        Ok(Tensor::f_from_slice(data)?.f_view([1, 1, samples])?)
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
    let dynamic = Dimension::variable;
    let mut model = Layers {
        graph: Builder::new(
            directory,
            "RVC decoder",
            vec![fixed(1), fixed(config.inter), dynamic("frames")],
        )?,
        weights,
    };
    let channels = weights.get("dec.cond.weight")?.size()[1];
    model
        .graph
        .input("speaker", vec![fixed(1), fixed(channels), fixed(1)]);
    if uses_f0 {
        model
            .graph
            .input("excitation", vec![fixed(1), fixed(1), dynamic("samples")]);
    }
    let latent = model.conv("input", "dec.conv_pre", 1, 3, 1)?;
    let condition = model.conv("speaker", "dec.cond", 1, 0, 1)?;
    let mut value = model.graph.op("Add", &[&latent, &condition], vec![]);
    let branches = config.kernels.len();
    let divisor = model.graph.scale(branches as f32);
    let mut source_stride = config.upsample_rates.iter().product::<i64>();
    for (stage, (&stride, &kernel)) in config
        .upsample_rates
        .iter()
        .zip(&config.upsample_kernels)
        .enumerate()
    {
        source_stride /= stride;
        let pad = (kernel - stride) / 2;
        let active = leaky(&mut model, &value, SLOPE);
        value = model.convolution(
            "ConvTranspose",
            &active,
            &format!("dec.ups.{stage}"),
            vec![
                Attribute::ints("strides", &[stride]),
                Attribute::ints("pads", &[pad, pad]),
            ],
        )?;
        if uses_f0 {
            let projected = model.conv(
                "excitation",
                &format!("dec.noise_convs.{stage}"),
                source_stride,
                source_stride / 2,
                1,
            )?;
            value = model.graph.op("Add", &[&value, &projected], vec![]);
        }
        let mut residuals = Vec::with_capacity(branches);
        for (index, (&kernel, dilations)) in
            config.kernels.iter().zip(&config.dilations).enumerate()
        {
            let layer = format!("dec.resblocks.{}", stage * branches + index);
            let mut branch = value.clone();
            let depth = if config.resblock == 1 { 3 } else { 2 };
            for (index, &dilation) in dilations.iter().take(depth).enumerate() {
                let group = if config.resblock == 1 {
                    "convs1"
                } else {
                    "convs"
                };
                let active = leaky(&mut model, &branch, SLOPE);
                let padding = (kernel - 1) * dilation / 2;
                let mut residual = model.convolution(
                    "Conv",
                    &active,
                    &format!("{layer}.{group}.{index}"),
                    vec![
                        Attribute::ints("pads", &[padding, padding]),
                        Attribute::ints("dilations", &[dilation]),
                    ],
                )?;
                if config.resblock == 1 {
                    let active = leaky(&mut model, &residual, SLOPE);
                    residual = model.conv(
                        &active,
                        &format!("{layer}.convs2.{index}"),
                        1,
                        (kernel - 1) / 2,
                        1,
                    )?;
                }
                branch = model.graph.op("Add", &[&branch, &residual], vec![]);
            }
            residuals.push(branch);
        }
        let refs = residuals.iter().map(String::as_str).collect::<Vec<_>>();
        let total = model.graph.op("Sum", &refs, vec![]);
        value = model.graph.op("Div", &[&total, &divisor], vec![]);
    }
    let active = leaky(&mut model, &value, 0.01);
    let waveform = model.conv(&active, "dec.conv_post", 1, 3, 1)?;
    let waveform = model.graph.op("Tanh", &[&waveform], vec![]);
    model
        .graph
        .finish(path, waveform, vec![fixed(1), fixed(1), dynamic("samples")])
}

fn leaky(model: &mut Layers<'_>, value: &str, slope: f64) -> String {
    model.graph.op(
        "LeakyRelu",
        &[value],
        vec![Attribute::float("alpha", slope as f32)],
    )
}
