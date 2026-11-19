use std::cell::RefCell;
use std::path::Path;

use anyhow::{Result, ensure};
use tch::{Device, Kind, Tensor};

use super::cuda_graph::Cache;
use super::weights::Weights;

pub struct Hubert {
    backend: Backend,
    device: Device,
    kind: Kind,
    pub(super) graph: RefCell<Cache>,
}

enum Backend {
    Torch(Weights),
    #[cfg(feature = "onnx")]
    Onnx(super::hubert_onnx::HubertOnnx),
}

impl Hubert {
    pub fn load(directory: &Path, device: Device) -> Result<Self> {
        Self::load_with_kind(directory, device, Kind::Float)
    }

    pub fn load_with_kind(directory: &Path, device: Device, kind: Kind) -> Result<Self> {
        let checkpoint = directory.join("pytorch_model.bin");
        let backend = if let Some(value) = super::onnx_selection("RVC_CONTENTVEC_ONNX") {
            #[cfg(feature = "onnx")]
            {
                use anyhow::Context;
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
            graph: RefCell::default(),
        })
    }

    pub fn extract(&self, audio: &[f32], v2: bool) -> Result<Tensor> {
        match &self.backend {
            Backend::Torch(weights) => self.forward(weights, self.input(audio)?, v2),
            #[cfg(feature = "onnx")]
            Backend::Onnx(session) => self.extract_onnx(session, audio, v2),
        }
    }

    pub(super) fn extract_realtime(&self, audio: &[f32], v2: bool) -> Result<Tensor> {
        match &self.backend {
            Backend::Torch(weights) => self.extract_realtime_torch(weights, audio, v2),
            #[cfg(feature = "onnx")]
            Backend::Onnx(session) => self.extract_onnx(session, audio, v2),
        }
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

    fn forward(&self, w: &Weights, mut x: Tensor, v2: bool) -> Result<Tensor> {
        let mut measure = super::phase_timer(self.device);
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
        for i in 0..if v2 { 12 } else { 9 } {
            let layer = format!("encoder.layers.{i}");
            let prefix = format!("{layer}.attention");
            let projections = w
                .linear(&x, &format!("{prefix}.qkv_proj"))?
                .f_view([1, x.size()[1], 3, 12, 64])?
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
            .reshape([1, -1, 768]);
            let attn = w.linear(&attn, &format!("{prefix}.out_proj"))?;
            x = w.norm(&(x + attn), &format!("{layer}.layer_norm"), 1e-5)?;
            let hidden = w
                .linear(&x, &format!("{layer}.feed_forward.intermediate_dense"))?
                .gelu("none");
            let hidden = w.linear(&hidden, &format!("{layer}.feed_forward.output_dense"))?;
            x = w.norm(&(x + hidden), &format!("{layer}.final_layer_norm"), 1e-5)?;
        }
        measure("hubert-transformer");
        if v2 {
            Ok(x)
        } else {
            w.linear(&x, "final_proj")
        }
    }
}

// The project's ContentVec checkpoint is a 12-layer HuBERT Base.
pub(super) fn load_weights(checkpoint: &Path) -> Result<Weights> {
    let mut weights = Weights::load(checkpoint, None, Device::Cpu)?;
    weights.remove_weight_norm(2)?;
    for layer in 0..12 {
        weights.pack_attention(&format!("encoder.layers.{layer}.attention"))?;
    }
    Ok(weights)
}
