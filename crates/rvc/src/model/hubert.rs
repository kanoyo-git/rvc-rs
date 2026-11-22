use std::borrow::Cow;
use std::cell::RefCell;
use std::fs::File;
use std::path::Path;

use anyhow::{Context, Result, ensure};
use tch::{Device, Kind, Tensor};

use super::cuda_graph::Cache;
use super::weights::Weights;

pub struct Hubert {
    backend: Backend,
    device: Device,
    kind: Kind,
    normalize_input: bool,
    layers: Layers,
    pub(super) graph: RefCell<Cache>,
}

enum Backend {
    Torch(Weights),
    #[cfg(feature = "onnx")]
    Onnx(super::hubert_onnx::HubertOnnx),
}

impl Hubert {
    pub(super) fn load(
        directory: &Path,
        device: Device,
        kind: Kind,
        normalize_input: bool,
    ) -> Result<Self> {
        let binary = directory.join("pytorch_model.bin");
        let checkpoint = if binary.is_file() {
            binary
        } else {
            directory.join("model.safetensors")
        };
        let normalize_input = normalize_input && needs_normalization(directory)?;
        let layers = layers(directory)?;
        let backend = if let Some(value) = super::onnx_selection("RVC_CONTENTVEC_ONNX") {
            #[cfg(feature = "onnx")]
            {
                let value = value.to_str().context("RVC_CONTENTVEC_ONNX must be text")?;
                Backend::Onnx(super::hubert_onnx::HubertOnnx::new(checkpoint, value)?)
            }
            #[cfg(not(feature = "onnx"))]
            anyhow::bail!(
                "ContentVec provider {} requires the onnx build feature",
                value.to_string_lossy()
            );
        } else {
            let mut weights = load_weights(&checkpoint)?;
            weights.move_to(device, kind)?;
            weights.prepare_cpu_convolutions(device)?;
            Backend::Torch(weights)
        };
        Ok(Self {
            backend,
            device,
            kind,
            normalize_input,
            layers,
            graph: RefCell::default(),
        })
    }

    pub fn extract(&self, audio: &[f32], v2: bool) -> Result<Tensor> {
        let audio = self.preprocess(audio);
        match &self.backend {
            Backend::Torch(weights) => self.forward(weights, self.input(&audio)?, v2),
            #[cfg(feature = "onnx")]
            Backend::Onnx(session) => self.extract_onnx(session, &audio, v2),
        }
    }

    pub(super) fn extract_realtime(&self, audio: &[f32], v2: bool) -> Result<Tensor> {
        let audio = self.preprocess(audio);
        match &self.backend {
            Backend::Torch(weights) => self.extract_realtime_torch(weights, &audio, v2),
            #[cfg(feature = "onnx")]
            Backend::Onnx(session) => self.extract_onnx(session, &audio, v2),
        }
    }

    fn preprocess<'a>(&self, samples: &'a [f32]) -> Cow<'a, [f32]> {
        if !self.normalize_input || samples.is_empty() {
            return Cow::Borrowed(samples);
        }
        let mean = samples.iter().map(|v| f64::from(*v)).sum::<f64>() / samples.len() as f64;
        let variance = samples
            .iter()
            .map(|&v| (f64::from(v) - mean).powi(2))
            .sum::<f64>()
            / samples.len() as f64;
        let scale = (variance + 1e-5).sqrt().recip();
        Cow::Owned(
            samples
                .iter()
                .map(|&v| ((f64::from(v) - mean) * scale) as f32)
                .collect(),
        )
    }

    #[cfg(feature = "onnx")]
    fn extract_onnx(
        &self,
        backend: &super::hubert_onnx::HubertOnnx,
        audio: &[f32],
        v2: bool,
    ) -> Result<Tensor> {
        Ok(backend
            .extract(audio, v2)?
            .f_to_device(self.device)?
            .f_to_kind(self.kind)?)
    }

    fn extract_realtime_torch(&self, weights: &Weights, audio: &[f32], v2: bool) -> Result<Tensor> {
        let x = self.input(audio)?;
        self.graph
            .borrow_mut()
            .run([i64::from(v2), 0, 0, 0], &[&x], None, |args| {
                self.forward(weights, args[0].shallow_clone(), v2)
            })
    }

    fn input(&self, audio: &[f32]) -> Result<Tensor> {
        ensure!(
            audio.len() >= 400,
            "HuBERT requires at least 400 input samples"
        );
        Ok(Tensor::f_from_slice(audio)?
            .view([1, 1, -1])
            .to_device(self.device)
            .to_kind(self.kind))
    }

    fn forward(&self, w: &Weights, x: Tensor, v2: bool) -> Result<Tensor> {
        let count = match self.layers {
            Layers::Stable(n) | Layers::Chosen(n) => n,
            Layers::Default if v2 => 12,
            Layers::Default => 9,
        };
        let x = self.layers.encode(w, x, count, |_, _| Ok(()))?;
        // A stored projection maps other encoders onto RVC's 768 content channels.
        if w.contains("pca.weight") {
            return w.linear(&x, "pca");
        }
        if !matches!(self.layers, Layers::Default) {
            return Ok(x);
        }
        if v2 {
            Ok(x)
        } else {
            w.linear(&x, "final_proj")
        }
    }
}

/// Runs the first `layers` HuBERT encoder layers on `[batch, 1, samples]` audio.
///
/// `tap` receives each layer's `[batch, frames, 768]` output (zero-based layer index).
pub(super) fn encode(
    w: &Weights,
    mut x: Tensor,
    layers: usize,
    mut tap: impl FnMut(usize, &Tensor) -> Result<()>,
) -> Result<Tensor> {
    let mut measure = super::phase_timer(x.device());
    for (i, stride) in [5, 2, 2, 2, 2, 2, 2].into_iter().enumerate() {
        let layer = format!("feature_extractor.conv_layers.{i}");
        x = w.conv(&x, &format!("{layer}.conv"), stride, 0, 1, 1)?;
        if i == 0 {
            let name = format!("{layer}.layer_norm");
            x = x.f_group_norm(
                x.size()[1],
                Some(w.get(&format!("{name}.weight"))?),
                Some(w.get(&format!("{name}.bias"))?),
                1e-5,
                true,
            )?;
        }
        x = x.gelu("none");
    }
    measure("hubert-convolutions");
    x = w.norm(&x.transpose(1, 2), "feature_projection.layer_norm", 1e-5)?;
    x = w.linear(&x, "feature_projection.projection")?;
    let positions = w.conv(
        &x.transpose(1, 2),
        "encoder.pos_conv_embed.conv",
        1,
        64,
        1,
        16,
    )?;
    let positions = positions
        .narrow(2, 0, x.size()[1])
        .transpose(1, 2)
        .gelu("none");
    x = w.norm(&(x + positions), "encoder.layer_norm", 1e-5)?;
    measure("hubert-projection-and-position");
    let batch = x.size()[0];
    for i in 0..layers {
        let layer = format!("encoder.layers.{i}");
        let prefix = format!("{layer}.attention");
        let projections = w
            .linear(&x, &format!("{prefix}.qkv_proj"))?
            .f_view([batch, x.size()[1], 3, 12, 64])?
            .f_permute([2, 0, 3, 1, 4])?;
        let qkv = projections.f_unbind(0)?;
        let attn = Tensor::f_scaled_dot_product_attention(
            &qkv[0],
            &qkv[1],
            &qkv[2],
            None::<&Tensor>,
            0.0,
            false,
            None,
            false,
        )?
        .transpose(1, 2)
        .reshape([batch, -1, 768]);
        let attn = w.linear(&attn, &format!("{prefix}.out_proj"))?;
        x = w.norm(&(x + attn), &format!("{layer}.layer_norm"), 1e-5)?;
        let hidden = w
            .linear(&x, &format!("{layer}.feed_forward.intermediate_dense"))?
            .gelu("none");
        let hidden = w.linear(&hidden, &format!("{layer}.feed_forward.output_dense"))?;
        x = w.norm(&(x + hidden), &format!("{layer}.final_layer_norm"), 1e-5)?;
        tap(i, &x)?;
    }
    measure("hubert-transformer");
    Ok(x)
}

/// Runs the first `layers` layers of a pre-LN wav2vec 2.0 encoder (layer-norm convolutions,
/// `do_stable_layer_norm`) on `[batch, 1, samples]` audio and returns the last layer's output
/// before the final encoder norm, as `hidden_states[layers]` in Transformers.
fn encode_stable(
    w: &Weights,
    mut x: Tensor,
    layers: usize,
    mut tap: impl FnMut(usize, &Tensor) -> Result<()>,
) -> Result<Tensor> {
    for (i, stride) in [5, 2, 2, 2, 2, 2, 2].into_iter().enumerate() {
        let layer = format!("feature_extractor.conv_layers.{i}");
        x = w.conv(&x, &format!("{layer}.conv"), stride, 0, 1, 1)?;
        x = w
            .norm(&x.transpose(1, 2), &format!("{layer}.layer_norm"), 1e-5)?
            .transpose(1, 2)
            .gelu("none");
    }
    x = w.norm(&x.transpose(1, 2), "feature_projection.layer_norm", 1e-5)?;
    x = w.linear(&x, "feature_projection.projection")?;
    let positions = w.conv(
        &x.transpose(1, 2),
        "encoder.pos_conv_embed.conv",
        1,
        64,
        1,
        16,
    )?;
    x = &x
        + positions
            .narrow(2, 0, x.size()[1])
            .transpose(1, 2)
            .gelu("none");
    let (batch, width) = (x.size()[0], x.size()[2]);
    let heads = 16;
    for i in 0..layers {
        let layer = format!("encoder.layers.{i}");
        let prefix = format!("{layer}.attention");
        let hidden = w.norm(&x, &format!("{layer}.layer_norm"), 1e-5)?;
        let projections = w
            .linear(&hidden, &format!("{prefix}.qkv_proj"))?
            .f_view([batch, x.size()[1], 3, heads, width / heads])?
            .f_permute([2, 0, 3, 1, 4])?;
        let qkv = projections.f_unbind(0)?;
        let attn = Tensor::f_scaled_dot_product_attention(
            &qkv[0],
            &qkv[1],
            &qkv[2],
            None::<&Tensor>,
            0.0,
            false,
            None,
            false,
        )?
        .transpose(1, 2)
        .reshape([batch, -1, width]);
        x = x.f_add(&w.linear(&attn, &format!("{prefix}.out_proj"))?)?;
        let hidden = w.norm(&x, &format!("{layer}.final_layer_norm"), 1e-5)?;
        let hidden = w
            .linear(&hidden, &format!("{layer}.feed_forward.intermediate_dense"))?
            .gelu("none");
        x = x.f_add(&w.linear(&hidden, &format!("{layer}.feed_forward.output_dense"))?)?;
        tap(i, &x)?;
    }
    Ok(x)
}

#[derive(Clone, Copy)]
pub(super) enum Layers {
    /// RVC's layer: 12 for v2, 9 plus `final_proj` for v1.
    Default,
    /// `rvc_layer` from config.json on a post-LN HuBERT encoder.
    Chosen(usize),
    /// Pre-LN wav2vec 2.0 (`do_stable_layer_norm`) run up to this layer.
    Stable(usize),
}

impl Layers {
    pub(super) fn encode(
        self,
        weights: &Weights,
        input: Tensor,
        count: usize,
        tap: impl FnMut(usize, &Tensor) -> Result<()>,
    ) -> Result<Tensor> {
        match self {
            Self::Stable(_) => encode_stable(weights, input, count, tap),
            _ => encode(weights, input, count, tap),
        }
    }
}

/// Reads the output layer from config.json. `rvc_layer` overrides RVC's choice; pre-LN
/// wav2vec 2.0 encoders default to layer 18, where a Russian CTC model keeps phonetics most
/// linearly separable in our probe.
pub(super) fn layers(directory: &Path) -> Result<Layers> {
    let path = directory.join("config.json");
    let Ok(file) = File::open(&path) else {
        return Ok(Layers::Default);
    };
    let config: serde_json::Value =
        serde_json::from_reader(file).with_context(|| format!("reading {}", path.display()))?;
    let chosen = config["rvc_layer"]
        .as_u64()
        .map(usize::try_from)
        .transpose()?;
    Ok(
        if config["do_stable_layer_norm"].as_bool().unwrap_or(false) {
            Layers::Stable(chosen.unwrap_or(18))
        } else {
            chosen.map_or(Layers::Default, Layers::Chosen)
        },
    )
}

pub(super) fn needs_normalization(directory: &Path) -> Result<bool> {
    let path = directory.join("preprocessor_config.json");
    match File::open(&path) {
        Ok(file) => {
            let config: serde_json::Value = serde_json::from_reader(file)
                .with_context(|| format!("reading {}", path.display()))?;
            Ok(config["do_normalize"].as_bool().unwrap_or(false))
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error).with_context(|| format!("opening {}", path.display())),
    }
}

// The project's ContentVec checkpoint is a 12-layer HuBERT Base.
pub(super) fn load_weights(checkpoint: &Path) -> Result<Weights> {
    let mut weights = if checkpoint
        .extension()
        .is_some_and(|ext| ext == "safetensors")
    {
        Weights::new(
            Tensor::read_safetensors(checkpoint)?
                .into_iter()
                .map(|(name, tensor)| {
                    (
                        name.strip_prefix("hubert.")
                            .or_else(|| name.strip_prefix("wav2vec2."))
                            .map(str::to_owned)
                            .unwrap_or(name),
                        tensor,
                    )
                })
                .collect(),
        )
    } else {
        Weights::load(checkpoint, None, Device::Cpu)?
    };
    weights.remove_weight_norm(2)?;
    let mut layer = 0;
    while weights.contains(&format!("encoder.layers.{layer}.attention.q_proj.weight")) {
        weights.pack_attention(&format!("encoder.layers.{layer}.attention"))?;
        layer += 1;
    }
    Ok(weights)
}
