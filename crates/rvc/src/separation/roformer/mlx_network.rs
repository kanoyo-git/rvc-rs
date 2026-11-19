use anyhow::{Context, Result, bail};
use serde_json::Value;
use tch::{Device, Tensor};

use super::super::mlx::{Array, DType, Stream};
use super::hyperace::mlx_network::Segmenter;
use super::mlx_spectral::Spectral;
use super::{
    Attention, AttentionBlock, Linear, MaskMode, MelSelection, ModelConfig, TorchNetwork,
    Transformer,
};

pub(super) struct MlxNetwork {
    stream: Stream,
    dtype: DType,
    dim: i32,
    projections: Vec<Projection>,
    blocks: Vec<[MlxTransformer; 2]>,
    final_gamma: Option<Array>,
    masks: Vec<Vec<Vec<MlxLinear>>>,
    mask_mode: MaskMode,
    segmentations: Option<Vec<Segmenter>>,
    spectral: Spectral,
}

struct Projection {
    width: i32,
    gamma: Array,
    linear: MlxLinear,
}

impl MlxNetwork {
    pub(super) fn load(
        torch: &TorchNetwork,
        device: Device,
        dtype: DType,
        config: &ModelConfig,
        mel: Option<&MelSelection>,
        window: &Tensor,
    ) -> Result<Self> {
        let stream = Stream::new(device)?;
        // HyperACE's cascade of instance normalizations amplifies half-precision
        // errors in the embeddings as well as in the segmentation layers.
        let dtype = if torch.segmentation.is_some() {
            DType::F32
        } else {
            dtype
        };
        let dim = i32::try_from(config.embedding_dim)?;
        let spectral = Spectral::new(&stream, config, mel, window)?;
        let projections = torch
            .projections
            .iter()
            .map(|band| {
                Ok(Projection {
                    width: i32::try_from(band.width)?,
                    gamma: upload_weight(&stream, &band.gamma, dtype)?,
                    linear: MlxLinear::load(&stream, &band.linear, dtype)?,
                })
            })
            .collect::<Result<_>>()?;
        let blocks = torch
            .blocks
            .iter()
            .map(|[time, frequency]| {
                Ok([
                    MlxTransformer::load(&stream, time, dtype)?,
                    MlxTransformer::load(&stream, frequency, dtype)?,
                ])
            })
            .collect::<Result<_>>()?;
        let final_gamma = torch
            .final_gamma
            .as_ref()
            .map(|gamma| upload_weight(&stream, gamma, dtype))
            .transpose()?;
        let masks = torch
            .masks
            .iter()
            .map(|stem| {
                stem.iter()
                    .map(|layers| {
                        layers
                            .0
                            .iter()
                            .map(|linear| MlxLinear::load(&stream, linear, dtype))
                            .collect()
                    })
                    .collect::<Result<_>>()
            })
            .collect::<Result<_>>()?;
        let segmentations = torch
            .segmentation
            .as_ref()
            .map(|estimators| {
                estimators
                    .iter()
                    .map(|estimator| Segmenter::load(&stream, estimator, dtype))
                    .collect::<Result<_>>()
            })
            .transpose()?;
        Ok(Self {
            stream,
            dtype,
            dim,
            projections,
            blocks,
            final_gamma,
            masks,
            mask_mode: torch.mask_mode,
            segmentations,
            spectral,
        })
    }

    pub(super) fn predict(&self, audio: &[f32], batch: usize, samples: usize) -> Result<Vec<f32>> {
        let (spectrum, features) = self.spectral.analyze(&self.stream, audio, batch, samples)?;
        let masks = self.masks(&features)?;
        self.spectral
            .render(&self.stream, &spectrum, &masks, samples)
    }

    fn masks(&self, input: &Array) -> Result<Array> {
        let stream = &self.stream;
        let input = stream.cast(input, self.dtype)?;
        let [batch, frames, _] = input.shape() else {
            bail!("RoFormer needs a three-dimensional spectrum")
        };
        let (batch, frames) = (*batch, *frames);
        let dim = self.dim;
        let bands = i32::try_from(self.projections.len())?;
        let time_batch = batch
            .checked_mul(bands)
            .context("RoFormer time batch overflow")?;
        let frequency_batch = batch
            .checked_mul(frames)
            .context("RoFormer frequency batch overflow")?;
        let mut mapped = Vec::with_capacity(self.projections.len());
        let mut offset = 0;
        for projection in &self.projections {
            let features = stream.slice(&input, 2, offset, offset + projection.width, 1)?;
            let features = normalize(stream, &features, &projection.gamma, self.dtype)?;
            mapped.push(projection.linear.forward(stream, &features)?);
            offset += projection.width;
        }
        let mut value = stream.stack(&mapped, 1)?;
        for [time, frequency] in &self.blocks {
            value = time.forward(
                stream,
                stream.reshape(&value, &[time_batch, frames, dim])?,
                self.dtype,
            )?;
            value = stream.transpose(
                &stream.reshape(&value, &[batch, bands, frames, dim])?,
                &[0, 2, 1, 3],
            )?;
            value = frequency.forward(
                stream,
                stream.reshape(&value, &[frequency_batch, bands, dim])?,
                self.dtype,
            )?;
            value = stream.transpose(
                &stream.reshape(&value, &[batch, frames, bands, dim])?,
                &[0, 2, 1, 3],
            )?;
        }
        if let Some(gamma) = &self.final_gamma {
            value = normalize(stream, &value, gamma, self.dtype)?;
        }
        let segmentations = self
            .segmentations
            .as_ref()
            .map(|estimators| {
                let features = stream.transpose(&value, &[0, 3, 2, 1])?;
                estimators
                    .iter()
                    .map(|estimator| estimator.run(stream, &features))
                    .collect::<Result<Vec<_>>>()
            })
            .transpose()?;
        if self.mask_mode == MaskMode::SegmentationOnly {
            let segmentations = segmentations.context("missing HyperACE segmentation masks")?;
            return stream.stack(&segmentations, 1);
        }
        let stems = self
            .masks
            .iter()
            .enumerate()
            .map(|(index, stem)| {
                let masks = stem
                    .iter()
                    .enumerate()
                    .map(|(band, layers)| {
                        let mut band = stream.select(&value, 1, i32::try_from(band)?)?;
                        let (last, hidden) = layers
                            .split_last()
                            .context("empty RoFormer mask estimator")?;
                        for linear in hidden {
                            band = stream.tanh(&linear.forward(stream, &band)?)?;
                        }
                        let gates = last.forward(stream, &band)?;
                        let width = *gates
                            .shape()
                            .last()
                            .context("RoFormer mask estimator output is scalar")?;
                        let left = stream.slice(&gates, 2, 0, width / 2, 1)?;
                        let right = stream.slice(&gates, 2, width / 2, width, 1)?;
                        stream.mul(&left, &stream.sigmoid(&right)?)
                    })
                    .collect::<Result<Vec<_>>>()?;
                let mask = stream.concat(&masks, -1)?;
                if let Some(segmentations) = &segmentations {
                    stream.add(&mask, &segmentations[index])
                } else {
                    Ok(mask)
                }
            })
            .collect::<Result<Vec<_>>>()?;
        stream.stack(&stems, 1)
    }
}

fn upload_weight(stream: &Stream, weights: &Tensor, dtype: DType) -> Result<Array> {
    let array = stream.upload(weights, dtype)?;
    array.eval()?;
    Ok(array)
}

pub(super) struct MlxLinear {
    weight: Array,
    bias: Option<Array>,
}

impl MlxLinear {
    pub(super) fn load(stream: &Stream, linear: &Linear, dtype: DType) -> Result<Self> {
        Ok(Self {
            weight: upload_weight(stream, &linear.weight.f_transpose(0, 1)?, dtype)?,
            bias: linear
                .bias
                .as_ref()
                .map(|bias| upload_weight(stream, bias, dtype))
                .transpose()?,
        })
    }

    pub(super) fn forward(&self, stream: &Stream, input: &Array) -> Result<Array> {
        let value = stream.matmul(input, &self.weight)?;
        match &self.bias {
            Some(bias) => stream.add(&value, bias),
            None => Ok(value),
        }
    }
}

fn normalize(stream: &Stream, input: &Array, gamma: &Array, dtype: DType) -> Result<Array> {
    let input = stream.cast(input, DType::F32)?;
    let length = stream.sqrt(&stream.sum_keep(&stream.square(&input)?, -1)?)?;
    let length = stream.max(&length, &stream.scalar(1e-12, DType::F32)?)?;
    let scale = stream.scalar((gamma.shape()[0] as f32).sqrt(), DType::F32)?;
    let value = stream.mul(&stream.div(&input, &length)?, &scale)?;
    stream.mul(&stream.cast(&value, dtype)?, gamma)
}

struct MlxTransformer {
    blocks: Vec<MlxBlock>,
    gamma: Option<Array>,
}

impl MlxTransformer {
    fn load(stream: &Stream, model: &Transformer, dtype: DType) -> Result<Self> {
        let blocks = model
            .blocks
            .iter()
            .map(|block| MlxBlock::load(stream, block, dtype))
            .collect::<Result<_>>()?;
        let gamma = model
            .gamma
            .as_ref()
            .map(|gamma| upload_weight(stream, gamma, dtype))
            .transpose()?;
        Ok(Self { blocks, gamma })
    }

    fn forward(&self, stream: &Stream, mut value: Array, dtype: DType) -> Result<Array> {
        for block in &self.blocks {
            value = block.forward(stream, &value, dtype)?;
        }
        match &self.gamma {
            Some(gamma) => normalize(stream, &value, gamma, dtype),
            None => Ok(value),
        }
    }
}

struct MlxAttention {
    gamma: Array,
    frequencies: Array,
    qkv: MlxLinear,
    gates: MlxLinear,
    output: MlxLinear,
    heads: i32,
    head_dim: i32,
    flash: bool,
}

impl MlxAttention {
    fn load(stream: &Stream, attention: &Attention, dtype: DType) -> Result<Self> {
        Ok(Self {
            gamma: upload_weight(stream, &attention.gamma, dtype)?,
            frequencies: upload_weight(stream, &attention.frequencies, DType::F32)?,
            qkv: MlxLinear::load(stream, &attention.qkv, dtype)?,
            gates: MlxLinear::load(stream, &attention.gates, dtype)?,
            output: MlxLinear::load(stream, &attention.output, dtype)?,
            heads: i32::try_from(attention.heads)?,
            head_dim: i32::try_from(attention.head_dim)?,
            flash: attention.flash,
        })
    }

    fn forward(&self, stream: &Stream, input: &Array, dtype: DType) -> Result<Array> {
        let [batch, steps, _] = input.shape() else {
            bail!("invalid RoFormer attention input")
        };
        let (batch, steps) = (*batch, *steps);
        let input = normalize(stream, input, &self.gamma, dtype)?;
        let qkv = self.qkv.forward(stream, &input)?;
        let qkv = stream.reshape(&qkv, &[batch, steps, 3, self.heads, self.head_dim])?;
        let positions = stream.reshape(&stream.arange(steps, DType::F32)?, &[1, steps, 1, 1])?;
        let angles = stream.mul(&positions, &self.frequencies)?;
        let cosine = stream.cast(&stream.cos(&angles)?, dtype)?;
        let sine = stream.cast(&stream.sin(&angles)?, dtype)?;
        let query = rotate(stream, &stream.select(&qkv, 2, 0)?, &cosine, &sine)?;
        let key = rotate(stream, &stream.select(&qkv, 2, 1)?, &cosine, &sine)?;
        let query = stream.transpose(&query, &[0, 2, 1, 3])?;
        let key = stream.transpose(&key, &[0, 2, 1, 3])?;
        let value = stream.transpose(&stream.select(&qkv, 2, 2)?, &[0, 2, 1, 3])?;
        let scale = (self.head_dim as f32).sqrt().recip();
        let attention = if self.flash {
            stream.attention(&query, &key, &value, scale)?
        } else {
            let scores = stream.matmul(&query, &stream.transpose(&key, &[0, 1, 3, 2])?)?;
            let scores = stream.mul(&scores, &stream.scalar(scale, dtype)?)?;
            stream.matmul(&stream.softmax(&scores, -1)?, &value)?
        };
        let gate = stream.sigmoid(&self.gates.forward(stream, &input)?)?;
        let gate = stream.reshape(&gate, &[batch, steps, self.heads, 1])?;
        let attention = stream.mul(&stream.transpose(&attention, &[0, 2, 1, 3])?, &gate)?;
        self.output.forward(
            stream,
            &stream.reshape(&attention, &[batch, steps, self.heads * self.head_dim])?,
        )
    }
}

struct MlxBlock {
    attention: MlxAttention,
    gamma: Array,
    expand: MlxLinear,
    contract: MlxLinear,
}

impl MlxBlock {
    fn load(stream: &Stream, block: &AttentionBlock, dtype: DType) -> Result<Self> {
        Ok(Self {
            attention: MlxAttention::load(stream, &block.attention, dtype)?,
            gamma: upload_weight(stream, &block.gamma, dtype)?,
            expand: MlxLinear::load(stream, &block.expand, dtype)?,
            contract: MlxLinear::load(stream, &block.contract, dtype)?,
        })
    }

    fn forward(&self, stream: &Stream, input: &Array, dtype: DType) -> Result<Array> {
        let value = stream.add(input, &self.attention.forward(stream, input, dtype)?)?;
        let hidden = self
            .expand
            .forward(stream, &normalize(stream, &value, &self.gamma, dtype)?)?;
        let hidden = super::super::mlx_layers::gelu(stream, &hidden, dtype)?;
        let output = self.contract.forward(stream, &hidden)?;
        stream.add(&value, &output)
    }
}

#[derive(Clone, Copy, Debug)]
pub(in crate::separation) enum AttentionMode {
    DotProduct,
    Layer,
    Transformer,
}

impl AttentionMode {
    pub(in crate::separation) fn parse(value: &Value) -> Result<Option<Self>> {
        if value.is_null() {
            return Ok(None);
        }
        let mode = value
            .as_str()
            .context("mps_attention_backend must be text")?
            .to_ascii_lowercase();
        Ok(match mode.as_str() {
            "torch" => None,
            "mlx" => Some(Self::DotProduct),
            "mlx_attention" => Some(Self::Layer),
            "mlx_transformer" => Some(Self::Transformer),
            _ => bail!("unknown MPS attention backend {mode:?}"),
        })
    }
}

#[derive(Clone, Copy)]
pub(super) struct BridgeSettings {
    pub(super) mode: AttentionMode,
    pub(super) minimum: i64,
    pub(super) dtype: DType,
}

pub(super) struct MlxBridge {
    stream: Stream,
    settings: BridgeSettings,
    module: BridgedModule,
}

enum BridgedModule {
    DotProduct,
    Layers(Vec<MlxAttention>),
    Transformer(MlxTransformer),
}

impl MlxBridge {
    pub(super) fn load(
        model: &Transformer,
        device: Device,
        settings: BridgeSettings,
    ) -> Result<Self> {
        let stream = Stream::new(device)?;
        let dtype = settings.dtype;
        let module = match settings.mode {
            AttentionMode::DotProduct => BridgedModule::DotProduct,
            AttentionMode::Layer => BridgedModule::Layers(
                model
                    .blocks
                    .iter()
                    .map(|block| MlxAttention::load(&stream, &block.attention, dtype))
                    .collect::<Result<_>>()?,
            ),
            AttentionMode::Transformer => {
                BridgedModule::Transformer(MlxTransformer::load(&stream, model, dtype)?)
            }
        };
        Ok(Self {
            stream,
            settings,
            module,
        })
    }

    pub(super) fn accepts(&self, input: &Tensor) -> bool {
        input.size()[1] >= self.settings.minimum
    }

    pub(super) fn forward(&self, model: &Transformer, value: Tensor) -> Result<Tensor> {
        let stream = &self.stream;
        let dtype = self.settings.dtype;
        match &self.module {
            BridgedModule::DotProduct => model.forward_blocks(value, |_, block, input| {
                block.forward_with(input, |query, key, value| {
                    let reference = query;
                    let query = stream.upload(query, dtype)?;
                    let key = stream.upload(key, dtype)?;
                    let value = stream.upload(value, dtype)?;
                    let scale = (block.head_dim as f32).sqrt().recip();
                    let output = stream.attention(&query, &key, &value, scale)?;
                    self.to_tensor(&output, reference)
                })
            }),
            BridgedModule::Layers(layers) => model.forward_blocks(value, |index, block, input| {
                if !block.flash {
                    return block.forward(input);
                }
                let array = stream.upload(input, dtype)?;
                let output = layers[index].forward(stream, &array, dtype)?;
                self.to_tensor(&output, input)
            }),
            BridgedModule::Transformer(transformer) => {
                let array = stream.upload(&value, dtype)?;
                let output = transformer.forward(stream, array, dtype)?;
                self.to_tensor(&output, &value)
            }
        }
    }

    fn to_tensor(&self, output: &Array, reference: &Tensor) -> Result<Tensor> {
        let shape: Vec<_> = output.shape().iter().copied().map(i64::from).collect();
        let values = self.stream.to_floats(output)?;
        Ok(Tensor::f_from_slice(&values)?
            .f_view(shape.as_slice())?
            .f_to_kind(reference.kind())?
            .f_to_device(reference.device())?)
    }
}

fn rotate(stream: &Stream, input: &Array, cosine: &Array, sine: &Array) -> Result<Array> {
    let width = input.shape()[3];
    let even = stream.slice(input, 3, 0, width, 2)?;
    let odd = stream.slice(input, 3, 1, width, 2)?;
    let real = stream.sub(&stream.mul(&even, cosine)?, &stream.mul(&odd, sine)?)?;
    let imaginary = stream.add(&stream.mul(&odd, cosine)?, &stream.mul(&even, sine)?)?;
    stream.reshape(&stream.stack(&[real, imaginary], -1)?, input.shape())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::Kind;

    #[test]
    #[ignore = "requires the optional MLX C CPU runtime"]
    fn partial_mlx_paths_match_a_torch_transformer() -> Result<()> {
        let options = (Kind::Float, Device::Cpu);
        let linear = |input, output| Linear {
            weight: Tensor::randn([output, input], options) * 0.1,
            bias: Some(Tensor::zeros([output], options)),
        };
        let mut model = Transformer {
            blocks: vec![AttentionBlock {
                attention: Attention {
                    gamma: Tensor::ones([32], options),
                    frequencies: Tensor::rand([8], options),
                    qkv: linear(32, 96),
                    gates: linear(32, 2),
                    output: linear(32, 32),
                    heads: 2,
                    head_dim: 16,
                    flash: true,
                    cuda_attention: super::super::cuda_sdpa::Backend::new(
                        super::super::cuda_sdpa::Policy::Default,
                    ),
                },
                gamma: Tensor::ones([32], options),
                expand: linear(32, 128),
                contract: linear(128, 32),
            }],
            gamma: Some(Tensor::ones([32], options)),
            mlx: None,
        };
        let input = Tensor::randn([3, 128, 32], options);
        let short = input.f_narrow(1, 0, 127)?;
        let expected = model.forward(input.shallow_clone())?;
        let expected_short = model.forward(short.shallow_clone())?;
        for mode in [
            AttentionMode::DotProduct,
            AttentionMode::Layer,
            AttentionMode::Transformer,
        ] {
            for (dtype, tolerance) in [(DType::F32, 2e-5), (DType::F16, 6e-3)] {
                model.mlx = Some(MlxBridge::load(
                    &model,
                    Device::Cpu,
                    BridgeSettings {
                        mode,
                        minimum: 128,
                        dtype,
                    },
                )?);
                let result = model.forward(input.shallow_clone())?;
                let error = result.f_sub(&expected)?.f_abs()?.f_max()?.double_value(&[]);
                eprintln!(
                    "MLX {mode:?}, dtype {} max difference {error}",
                    dtype as i32
                );
                assert!(error < tolerance);
                let bypassed = model.forward(short.shallow_clone())?;
                assert!(bypassed.f_equal(&expected_short)?);
            }
        }
        Ok(())
    }
}
