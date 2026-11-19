//! Waveform discriminators used by the RVC v1 and v2 trainers.

use std::collections::HashMap;

use anyhow::{Context, Result, ensure};
use tch::{Device, Kind, Tensor, no_grad};

const PERIODS: [i64; 8] = [2, 3, 5, 7, 11, 17, 23, 37];

pub struct DiscriminatorOutput {
    pub scores: Vec<Tensor>,
    pub features: Vec<Vec<Tensor>>,
}

pub struct Discriminator {
    branches: Vec<Branch>,
}

struct Branch {
    period: Option<i64>,
    layers: Vec<NormalizedConv>,
}

#[derive(Clone, Copy)]
struct Geometry {
    channels_in: i64,
    channels_out: i64,
    kernel: i64,
    stride: i64,
    padding: i64,
    groups: i64,
    period: bool,
}

struct NormalizedConv {
    name: String,
    geometry: Geometry,
    bias: Tensor,
    weight: ConvWeight,
}

enum ConvWeight {
    GainDirection {
        gain: Tensor,
        direction: Tensor,
    },
    Spectral {
        original: Tensor,
        u: Tensor,
        v: Tensor,
    },
}

impl Discriminator {
    pub fn new(v2: bool, spectral: bool, device: Device) -> Result<Self> {
        Self::build(v2, spectral, device, None)
    }

    pub fn from_weights(
        v2: bool,
        spectral: bool,
        device: Device,
        weights: HashMap<String, Tensor>,
    ) -> Result<Self> {
        Self::build(v2, spectral, device, Some(weights))
    }

    fn build(
        v2: bool,
        spectral: bool,
        device: Device,
        mut weights: Option<HashMap<String, Tensor>>,
    ) -> Result<Self> {
        let periods = &PERIODS[..if v2 { 8 } else { 6 }];
        let mut branches = Vec::with_capacity(periods.len() + 1);
        for (index, period) in std::iter::once(None)
            .chain(periods.iter().copied().map(Some))
            .enumerate()
        {
            let specs = if period.is_some() {
                periodic_convs()
            } else {
                scale_convs()
            };
            let mut layers = Vec::with_capacity(specs.len());
            for (layer_index, geometry) in specs.iter().enumerate() {
                let suffix = if layer_index + 1 == specs.len() {
                    "conv_post".to_owned()
                } else {
                    format!("convs.{layer_index}")
                };
                let name = format!("discriminators.{index}.{suffix}");
                let layer =
                    NormalizedConv::build(name, *geometry, spectral, device, weights.as_mut())?;
                layers.push(layer);
            }
            branches.push(Branch { period, layers });
        }
        if let Some(weights) = weights {
            ensure!(
                weights.is_empty(),
                "discriminator checkpoint contains unexpected tensors"
            );
        }
        Ok(Self { branches })
    }

    /// Grad-enabled tensor handles in the original parameter registration order.
    pub fn parameters(&self) -> Vec<(String, Tensor)> {
        self.state(false)
    }

    pub fn state_dict(&self) -> Vec<(String, Tensor)> {
        self.state(true)
    }

    fn state(&self, include_buffers: bool) -> Vec<(String, Tensor)> {
        let mut tensors = Vec::new();
        for branch in &self.branches {
            for layer in &branch.layers {
                layer.append_state(&mut tensors, include_buffers);
            }
        }
        tensors
    }

    /// Both inputs use [batch, 1, samples] layout.
    pub fn forward(
        &mut self,
        real: &Tensor,
        generated: &Tensor,
        training: bool,
    ) -> Result<(DiscriminatorOutput, DiscriminatorOutput)> {
        let mut positive = DiscriminatorOutput {
            scores: Vec::new(),
            features: Vec::new(),
        };
        let mut negative = DiscriminatorOutput {
            scores: Vec::new(),
            features: Vec::new(),
        };
        for branch in &mut self.branches {
            let (score, features) = branch.forward(real, training)?;
            positive.scores.push(score);
            positive.features.push(features);
            let (score, features) = branch.forward(generated, training)?;
            negative.scores.push(score);
            negative.features.push(features);
        }
        Ok((positive, negative))
    }
}

impl Branch {
    fn forward(&mut self, audio: &Tensor, training: bool) -> Result<(Tensor, Vec<Tensor>)> {
        let (batch, channels, frames) = audio.size3()?;
        ensure!(
            batch > 0 && channels == 1 && frames > 0,
            "expected a nonempty batch of mono waveforms"
        );
        let mut hidden = if let Some(period) = self.period {
            let padding = (period - frames.rem_euclid(period)) % period;
            let padded = if padding == 0 {
                audio.shallow_clone()
            } else {
                audio.f_reflection_pad1d([0, padding])?
            };
            padded.f_reshape([batch, channels, (frames + padding) / period, period])?
        } else {
            audio.shallow_clone()
        };
        let mut features = Vec::with_capacity(self.layers.len());
        let (post, convolutions) = self
            .layers
            .split_last_mut()
            .context("missing discriminator post convolution")?;
        for layer in convolutions {
            let filtered = layer.forward(&hidden, training)?;
            let negative = filtered.f_mul_scalar(0.1)?;
            hidden = filtered.f_where_self(&filtered.f_gt(0.0)?, &negative)?;
            features.push(hidden.shallow_clone());
        }
        hidden = post.forward(&hidden, training)?;
        let score = hidden.f_flatten(1, -1)?;
        features.push(hidden);
        Ok((score, features))
    }
}

impl NormalizedConv {
    fn build(
        name: String,
        geometry: Geometry,
        spectral: bool,
        device: Device,
        imported: Option<&mut HashMap<String, Tensor>>,
    ) -> Result<Self> {
        let mut shape = vec![
            geometry.channels_out,
            geometry.channels_in / geometry.groups,
            geometry.kernel,
        ];
        if geometry.period {
            shape.push(1);
        }
        let fan_in: i64 = shape[1..].iter().product();
        let mut gain_shape = vec![1; shape.len()];
        gain_shape[0] = geometry.channels_out;
        let (bias, weight) = if let Some(imported) = imported {
            let mut get = |suffix, shape: &[i64], trainable| {
                read_tensor(imported, &name, suffix, shape, device, trainable)
            };
            let bias = get("bias", &[geometry.channels_out], true)?;
            let weight = if spectral {
                ConvWeight::Spectral {
                    original: get("weight_orig", &shape, true)?,
                    u: get("weight_u", &[geometry.channels_out], false)?,
                    v: get("weight_v", &[fan_in], false)?,
                }
            } else {
                ConvWeight::GainDirection {
                    gain: get("weight_g", &gain_shape, true)?,
                    direction: get("weight_v", &shape, true)?,
                }
            };
            (bias, weight)
        } else {
            let bound = 1.0 / (fan_in as f64).sqrt();
            let options = (Kind::Float, device);
            let direction = Tensor::f_empty(&shape[..], options)?.f_uniform_(-bound, bound)?;
            let bias = Tensor::f_empty([geometry.channels_out], options)?
                .f_uniform_(-bound, bound)?
                .f_set_requires_grad(true)?;
            let weight = if spectral {
                ConvWeight::Spectral {
                    original: direction.f_set_requires_grad(true)?,
                    u: unit(&Tensor::f_randn([geometry.channels_out], options)?)?,
                    v: unit(&Tensor::f_randn([fan_in], options)?)?,
                }
            } else {
                let gain = direction
                    .f_reshape([geometry.channels_out, fan_in])?
                    .f_norm_scalaropt_dim(2.0, [1], false)?
                    .f_reshape(&gain_shape[..])?
                    .f_set_requires_grad(true)?;
                ConvWeight::GainDirection {
                    gain,
                    direction: direction.f_set_requires_grad(true)?,
                }
            };
            (bias, weight)
        };
        Ok(Self {
            name,
            geometry,
            bias,
            weight,
        })
    }

    fn forward(&mut self, input: &Tensor, training: bool) -> Result<Tensor> {
        let weight = self.weight.normalized(training)?;
        let Geometry {
            stride,
            padding,
            groups,
            period,
            ..
        } = self.geometry;
        if period {
            Ok(input.f_conv2d(
                &weight,
                Some(&self.bias),
                [stride, 1],
                [padding, 0],
                [1, 1],
                groups,
            )?)
        } else {
            Ok(input.f_conv1d(&weight, Some(&self.bias), stride, padding, 1, groups)?)
        }
    }

    fn append_state(&self, tensors: &mut Vec<(String, Tensor)>, include_buffers: bool) {
        let mut push = |name, weight: &Tensor| {
            tensors.push((format!("{}.{name}", self.name), weight.shallow_clone()))
        };
        push("bias", &self.bias);
        match &self.weight {
            ConvWeight::GainDirection { gain, direction } => {
                push("weight_g", gain);
                push("weight_v", direction);
            }
            ConvWeight::Spectral { original, u, v } => {
                push("weight_orig", original);
                if include_buffers {
                    push("weight_u", u);
                    push("weight_v", v);
                }
            }
        }
    }
}

impl ConvWeight {
    fn normalized(&mut self, training: bool) -> Result<Tensor> {
        match self {
            Self::GainDirection { gain, direction } => {
                Ok(Tensor::f_internal_weight_norm(direction, gain, 0)?)
            }
            Self::Spectral { original, u, v } => {
                let matrix = original.f_flatten(1, -1)?;
                if training {
                    no_grad(|| -> Result<()> {
                        v.f_copy_(&unit(&matrix.f_transpose(0, 1)?.f_mv(u)?)?)?;
                        u.f_copy_(&unit(&matrix.f_mv(v)?)?)?;
                        Ok(())
                    })?;
                }
                // The real and generated passes update the same buffers before
                // backward. Each weight expression must save its own values.
                let u = u.f_detach_copy()?;
                let v = v.f_detach_copy()?;
                let sigma = u.f_dot(&matrix.f_mv(&v)?)?;
                Ok(original.f_div(&sigma)?)
            }
        }
    }
}

fn unit(vector: &Tensor) -> Result<Tensor> {
    Ok(vector.f_div(&vector.f_norm()?.f_clamp_min(1e-12)?)?)
}

fn read_tensor(
    weights: &mut HashMap<String, Tensor>,
    prefix: &str,
    name: &str,
    shape: &[i64],
    device: Device,
    trainable: bool,
) -> Result<Tensor> {
    let key = format!("{prefix}.{name}");
    let weight = weights
        .remove(&key)
        .with_context(|| format!("discriminator weight {key} is missing"))?;
    ensure!(
        weight.size() == shape,
        "discriminator weight {key} has shape {:?}, expected {shape:?}",
        weight.size()
    );
    ensure!(
        matches!(
            weight.kind(),
            Kind::Float | Kind::Half | Kind::BFloat16 | Kind::Double
        ),
        "unsupported dtype for discriminator weight {key}"
    );
    Ok(weight
        .f_detach()?
        .f_to_device(device)?
        .f_to_kind(Kind::Float)?
        .f_set_requires_grad(trainable)?)
}

fn scale_convs() -> Vec<Geometry> {
    [
        (1, 16, 15, 1, 1),
        (16, 64, 41, 4, 4),
        (64, 256, 41, 4, 16),
        (256, 1024, 41, 4, 64),
        (1024, 1024, 41, 4, 256),
        (1024, 1024, 5, 1, 1),
        (1024, 1, 3, 1, 1),
    ]
    .into_iter()
    .map(
        |(channels_in, channels_out, kernel, stride, groups)| Geometry {
            channels_in,
            channels_out,
            kernel,
            stride,
            padding: kernel / 2,
            groups,
            period: false,
        },
    )
    .collect()
}

fn periodic_convs() -> Vec<Geometry> {
    [
        (1, 32, 5, 3),
        (32, 128, 5, 3),
        (128, 512, 5, 3),
        (512, 1024, 5, 3),
        (1024, 1024, 5, 1),
        (1024, 1, 3, 1),
    ]
    .into_iter()
    .map(|(channels_in, channels_out, kernel, stride)| Geometry {
        channels_in,
        channels_out,
        kernel,
        stride,
        padding: kernel / 2,
        groups: 1,
        period: true,
    })
    .collect()
}
