use std::collections::HashMap;

use anyhow::{Context, Result, ensure};
use tch::{Device, Tensor};

use super::super::mlx::DType;

#[path = "network/mlx_backend.rs"]
mod mlx_backend;
#[cfg(feature = "onnx")]
#[path = "network/onnx_backend.rs"]
mod onnx_backend;

pub(super) struct Network {
    backend: Backend,
    modern: bool,
    pub offset: i64,
    spatial_factor: i64,
    bins: i64,
}

enum Backend {
    Torch(TorchNetwork),
    Mlx(Box<mlx_backend::Network>),
    #[cfg(feature = "onnx")]
    Onnx(Box<onnx_backend::Network>),
}

pub(super) enum Acceleration {
    Torch,
    Mlx(Device, DType),
    #[cfg(feature = "onnx")]
    Onnx(crate::model::onnx::Provider),
}

impl Acceleration {
    pub(super) fn host_input(&self) -> bool {
        !matches!(self, Self::Torch)
    }
}

struct TorchNetwork {
    topology: Topology,
    output: Conv,
}

enum Topology {
    Legacy {
        low: Box<Subnetwork>,
        high: Box<Subnetwork>,
        bridge2: Box<Block>,
        bridge3: Box<Block>,
        stage2: Box<Subnetwork>,
        stage3: Box<Subnetwork>,
    },
    Modern {
        low1: Box<Subnetwork>,
        high1: Box<Subnetwork>,
        low2: Box<Subnetwork>,
        high2: Box<Subnetwork>,
        stage3: Box<Subnetwork>,
    },
}

#[derive(Clone, Copy)]
struct Architecture {
    modern: bool,
    downsamples: i64,
    pyramid_paths: i64,
}

impl Network {
    pub(super) fn load(
        weights: &HashMap<String, Tensor>,
        bins: i64,
        fuse: bool,
        channels_last: bool,
        frames: i64,
        acceleration: Acceleration,
    ) -> Result<Self> {
        let modern = weights.contains_key("stg1_low_band_net.0.enc1.conv.0.weight");
        let downsamples =
            if !modern && weights.contains_key("stg1_low_band_net.enc5.conv1.conv.0.weight") {
                5
            } else {
                4
            };
        let pyramid_paths = if !modern
            && weights.contains_key("stg1_low_band_net.aspp.conv7.conv.0.weight")
        {
            7
        } else if !modern && weights.contains_key("stg1_low_band_net.aspp.conv6.conv.0.weight") {
            6
        } else {
            5
        };
        let architecture = Architecture {
            modern,
            downsamples,
            pyramid_paths,
        };
        let spatial_factor = 1 << downsamples;
        let offset = if modern { 64 } else { 128 };
        ensure!(
            bins > 0 && bins % (2 * spatial_factor) == 0,
            "VR frequency bins are incompatible with the network downsampling factor"
        );
        ensure!(
            frames > 2 * offset && frames % spatial_factor == 0,
            "VR window must exceed {} frames and be divisible by {spatial_factor}",
            2 * offset
        );
        let builder = Builder {
            weights,
            architecture,
            fuse,
            channels_last,
        };
        let topology = if modern {
            Topology::Modern {
                low1: builder.subnet("stg1_low_band_net", true)?,
                high1: builder.subnet("stg1_high_band_net", false)?,
                low2: builder.subnet("stg2_low_band_net", true)?,
                high2: builder.subnet("stg2_high_band_net", false)?,
                stage3: builder.subnet("stg3_full_band_net", false)?,
            }
        } else {
            Topology::Legacy {
                low: builder.subnet("stg1_low_band_net", false)?,
                high: builder.subnet("stg1_high_band_net", false)?,
                bridge2: Box::new(builder.block(
                    "stg2_bridge",
                    Geometry::POINTWISE,
                    Activation::Relu,
                    false,
                )?),
                bridge3: Box::new(builder.block(
                    "stg3_bridge",
                    Geometry::POINTWISE,
                    Activation::Relu,
                    false,
                )?),
                stage2: builder.subnet("stg2_full_band_net", false)?,
                stage3: builder.subnet("stg3_full_band_net", false)?,
            }
        };
        let torch = TorchNetwork {
            topology,
            output: builder.conv("out", Geometry::POINTWISE, 1)?,
        };
        let backend = match acceleration {
            Acceleration::Torch => Backend::Torch(torch),
            Acceleration::Mlx(device, dtype) => {
                Backend::Mlx(Box::new(mlx_backend::Network::load(&torch, device, dtype)?))
            }
            #[cfg(feature = "onnx")]
            Acceleration::Onnx(provider) => Backend::Onnx(Box::new(onnx_backend::Network::load(
                &torch, provider, bins, frames, offset,
            )?)),
        };
        Ok(Self {
            backend,
            modern,
            offset,
            spatial_factor,
            bins,
        })
    }

    pub(super) fn modern(&self) -> bool {
        self.modern
    }

    pub(super) fn host_input(&self) -> bool {
        !matches!(self.backend, Backend::Torch(_))
    }

    pub(super) fn predict_mask(&self, input: &Tensor) -> Result<Tensor> {
        let (_, channels, freqs, frames) = input.size4()?;
        ensure!(
            channels == 2
                && freqs == self.bins + 1
                && frames > 2 * self.offset
                && frames % self.spatial_factor == 0,
            "invalid VR network input shape"
        );
        match &self.backend {
            Backend::Torch(torch) => torch.predict_mask(input, self.bins, self.offset, frames),
            Backend::Mlx(mlx) => mlx.predict_mask(input, self.bins, self.offset),
            #[cfg(feature = "onnx")]
            Backend::Onnx(onnx) => onnx.predict_mask(input),
        }
    }
}

impl TorchNetwork {
    fn predict_mask(&self, input: &Tensor, bins: i64, offset: i64, frames: i64) -> Result<Tensor> {
        let input = input.f_narrow(2, 0, bins)?;
        let half = bins / 2;
        let lower = input.f_narrow(2, 0, half)?;
        let upper = input.f_narrow(2, half, half)?;
        let features = match &self.topology {
            Topology::Legacy {
                low,
                high,
                bridge2,
                bridge3,
                stage2,
                stage3,
            } => {
                let split = Tensor::f_cat(&[low.run(&lower)?, high.run(&upper)?], 2)?;
                let refine = stage2.run(&bridge2.run(&Tensor::f_cat(&[&input, &split], 1)?)?)?;
                stage3.run(&bridge3.run(&Tensor::f_cat(&[input, split, refine], 1)?)?)?
            }
            Topology::Modern {
                low1,
                high1,
                low2,
                high2,
                stage3,
            } => {
                let lower1 = low1.run(&lower)?;
                let upper1 = high1.run(&upper)?;
                let split1 = Tensor::f_cat(&[&lower1, &upper1], 2)?;
                let lower2 = low2.run(&Tensor::f_cat(&[lower, lower1], 1)?)?;
                let upper2 = high2.run(&Tensor::f_cat(&[upper, upper1], 1)?)?;
                let split2 = Tensor::f_cat(&[lower2, upper2], 2)?;
                stage3.run(&Tensor::f_cat(&[input, split1, split2], 1)?)?
            }
        };
        let mask = self.output.run(&features)?.f_sigmoid()?;
        ensure!(
            mask.size()[1] == 2 && mask.size()[2] == bins,
            "invalid VR output shape"
        );
        Ok(mask
            .f_replication_pad2d([0, 0, 0, 1])?
            .f_narrow(3, offset, frames - 2 * offset)?)
    }
}

struct Builder<'a> {
    weights: &'a HashMap<String, Tensor>,
    architecture: Architecture,
    fuse: bool,
    channels_last: bool,
}

impl Builder<'_> {
    fn tensor(&self, name: &str) -> Result<Tensor> {
        self.weights
            .get(name)
            .map(Tensor::shallow_clone)
            .with_context(|| format!("VR checkpoint is missing {name}"))
    }

    fn conv(&self, prefix: &str, geometry: Geometry, groups: i64) -> Result<Conv> {
        let mut weight = self.tensor(&format!("{prefix}.weight"))?;
        let (out, input, height, width) = weight
            .size4()
            .with_context(|| format!("{prefix} is not a 2D convolution"))?;
        let kernel = if geometry.pointwise { 1 } else { 3 };
        ensure!(
            input > 0
                && out > 0
                && groups > 0
                && out % groups == 0
                && height == kernel
                && width == kernel,
            "unexpected VR convolution geometry for {prefix}"
        );
        if self.channels_last {
            weight = channels_last(&weight)?;
        }
        Ok(Conv {
            weight,
            bias: None,
            geometry,
            groups,
        })
    }

    fn norm(&self, prefix: &str, channels: i64) -> Result<Norm> {
        let layer = Norm {
            weight: self.tensor(&format!("{prefix}.weight"))?,
            bias: self.tensor(&format!("{prefix}.bias"))?,
            mean: self.tensor(&format!("{prefix}.running_mean"))?,
            variance: self.tensor(&format!("{prefix}.running_var"))?,
        };
        ensure!(
            [&layer.weight, &layer.bias, &layer.mean, &layer.variance]
                .into_iter()
                .all(|weight| weight.size() == [channels]),
            "unexpected VR batch normalization geometry for {prefix}"
        );
        Ok(layer)
    }

    fn block(
        &self,
        prefix: &str,
        geometry: Geometry,
        act: Activation,
        separable: bool,
    ) -> Result<Block> {
        let mut depth = None;
        let mut point = self.conv(
            &format!("{prefix}.conv.{}", if separable { 1 } else { 0 }),
            if separable {
                Geometry::POINTWISE
            } else {
                geometry
            },
            1,
        )?;
        if separable {
            let channels = point.weight.size()[1];
            let conv = self.conv(&format!("{prefix}.conv.0"), geometry, channels)?;
            ensure!(
                conv.weight.size()[..2] == [channels, 1],
                "unexpected VR depthwise convolution geometry for {prefix}"
            );
            depth = Some(conv);
        }
        let norm = self.norm(
            &format!("{prefix}.conv.{}", if separable { 2 } else { 1 }),
            point.weight.size()[0],
        )?;
        let norm = if self.fuse {
            let scale = norm
                .weight
                .f_mul(&norm.variance.f_add_scalar(1e-5)?.f_rsqrt()?)?;
            point.weight = point.weight.f_mul(&scale.f_reshape([-1, 1, 1, 1])?)?;
            point.bias = Some(norm.bias.f_sub(&norm.mean.f_mul(&scale)?)?);
            None
        } else {
            Some(norm)
        };
        Ok(Block {
            depth,
            point,
            norm,
            act,
        })
    }

    fn subnet(&self, prefix: &str, output_bridge: bool) -> Result<Box<Subnetwork>> {
        let modern = self.architecture.modern;
        let name = if output_bridge {
            format!("{prefix}.0")
        } else {
            prefix.to_owned()
        };
        let first = modern
            .then(|| {
                self.block(
                    &format!("{name}.enc1"),
                    Geometry::STANDARD,
                    Activation::Relu,
                    false,
                )
            })
            .transpose()?;
        let mut down = Vec::new();
        let pairs = (1..=self.architecture.downsamples).map(|i| if modern { i + 1 } else { i });
        for index in pairs {
            let first_geometry = if modern {
                Geometry::DOWNSAMPLE
            } else {
                Geometry::STANDARD
            };
            let second_geometry = if modern {
                Geometry::STANDARD
            } else {
                Geometry::DOWNSAMPLE
            };
            let enc = format!("{name}.enc{index}");
            down.push([
                self.block(
                    &format!("{enc}.conv1"),
                    first_geometry,
                    Activation::Leaky,
                    false,
                )?,
                self.block(
                    &format!("{enc}.conv2"),
                    second_geometry,
                    Activation::Leaky,
                    false,
                )?,
            ]);
        }
        let aspp = self.pyramid(&format!("{name}.aspp"))?;
        let decode_name = if modern { "conv1" } else { "conv" };
        let up = (1..=self.architecture.downsamples)
            .rev()
            .map(|index| {
                self.block(
                    &format!("{name}.dec{index}.{decode_name}"),
                    Geometry::STANDARD,
                    Activation::Relu,
                    false,
                )
            })
            .collect::<Result<_>>()?;
        let recurrent = modern
            .then(|| self.lstm(&format!("{name}.lstm_dec2")))
            .transpose()?;
        let last = output_bridge
            .then(|| {
                self.block(
                    &format!("{prefix}.1"),
                    Geometry::POINTWISE,
                    Activation::Relu,
                    false,
                )
            })
            .transpose()?;
        Ok(Box::new(Subnetwork {
            first,
            down,
            aspp,
            up,
            recurrent,
            last,
        }))
    }

    fn pyramid(&self, prefix: &str) -> Result<Pyramid> {
        let modern = self.architecture.modern;
        let pool = self.block(
            &format!("{prefix}.conv1.1"),
            Geometry::POINTWISE,
            Activation::Relu,
            false,
        )?;
        let linear = self.block(
            &format!("{prefix}.conv2"),
            Geometry::POINTWISE,
            Activation::Relu,
            false,
        )?;
        let mut paths = vec![linear];
        for path in 3..=self.architecture.pyramid_paths {
            let dilation = match path {
                3 => [4, if modern { 2 } else { 4 }],
                4 => [8, if modern { 4 } else { 8 }],
                _ if modern => [12, 6],
                _ => [16, 16],
            };
            // In seven-path checkpoints conv6 and conv7 are the same module.
            let saved = if path == 6 && self.architecture.pyramid_paths == 7 {
                7
            } else {
                path
            };
            let geometry = Geometry {
                dilation,
                padding: dilation,
                ..Geometry::STANDARD
            };
            paths.push(self.block(
                &format!("{prefix}.conv{saved}"),
                geometry,
                Activation::Relu,
                !modern,
            )?);
        }
        let bottleneck = if modern { "bottleneck" } else { "bottleneck.0" };
        let merge = self.block(
            &format!("{prefix}.{bottleneck}"),
            Geometry::POINTWISE,
            Activation::Relu,
            false,
        )?;
        Ok(Pyramid { pool, paths, merge })
    }

    fn lstm(&self, prefix: &str) -> Result<Recurrent> {
        let conv = self.block(
            &format!("{prefix}.conv"),
            Geometry::POINTWISE,
            Activation::Relu,
            false,
        )?;
        let mut parameters = Vec::new();
        for direction in ["", "_reverse"] {
            for parameter in ["weight_ih", "weight_hh", "bias_ih", "bias_hh"] {
                parameters.push(self.tensor(&format!("{prefix}.lstm.{parameter}_l0{direction}"))?);
            }
        }
        let (_, hidden) = parameters[1].size2()?;
        let (features, width) = parameters[0].size2()?;
        ensure!(
            hidden > 0 && features == hidden * 4 && parameters[1].size() == [features, hidden],
            "invalid VR LSTM parameter geometry for {prefix}"
        );
        for (part, forward) in parameters[4..].iter().zip(&parameters[..4]) {
            ensure!(
                part.size() == forward.size(),
                "VR LSTM direction widths differ"
            );
        }
        ensure!(
            parameters[2].size() == [features] && parameters[3].size() == [features],
            "invalid VR LSTM bias geometry for {prefix}"
        );
        let weight = self.tensor(&format!("{prefix}.dense.0.weight"))?;
        let bias = self.tensor(&format!("{prefix}.dense.0.bias"))?;
        ensure!(
            weight.size() == [width, 2 * hidden] && bias.size() == [width],
            "invalid VR LSTM projection"
        );
        let norm = self.norm(&format!("{prefix}.dense.1"), width)?;
        Ok(Recurrent {
            conv,
            parameters,
            weight,
            bias,
            norm,
            hidden,
        })
    }
}

#[derive(Clone, Copy)]
enum Activation {
    Relu,
    Leaky,
}

#[derive(Clone, Copy)]
struct Geometry {
    stride: [i64; 2],
    padding: [i64; 2],
    dilation: [i64; 2],
    pointwise: bool,
}

impl Geometry {
    const STANDARD: Self = Self {
        stride: [1, 1],
        padding: [1, 1],
        dilation: [1, 1],
        pointwise: false,
    };
    const POINTWISE: Self = Self {
        padding: [0, 0],
        pointwise: true,
        ..Self::STANDARD
    };
    const DOWNSAMPLE: Self = Self {
        stride: [2, 2],
        ..Self::STANDARD
    };
}

struct Conv {
    weight: Tensor,
    bias: Option<Tensor>,
    geometry: Geometry,
    groups: i64,
}

impl Conv {
    fn run(&self, input: &Tensor) -> Result<Tensor> {
        let Geometry {
            stride,
            padding,
            dilation,
            ..
        } = self.geometry;
        Ok(input.f_conv2d(
            &self.weight,
            self.bias.as_ref(),
            stride,
            padding,
            dilation,
            self.groups,
        )?)
    }
}

struct Norm {
    weight: Tensor,
    bias: Tensor,
    mean: Tensor,
    variance: Tensor,
}

impl Norm {
    fn run(&self, input: &Tensor) -> Result<Tensor> {
        Ok(input.f_batch_norm(
            Some(&self.weight),
            Some(&self.bias),
            Some(&self.mean),
            Some(&self.variance),
            false,
            0.1,
            1e-5,
            true,
        )?)
    }
}

struct Block {
    depth: Option<Conv>,
    point: Conv,
    norm: Option<Norm>,
    act: Activation,
}

impl Block {
    fn run(&self, input: &Tensor) -> Result<Tensor> {
        let depth = self
            .depth
            .as_ref()
            .map(|layer| layer.run(input))
            .transpose()?;
        let output = self.point.run(depth.as_ref().unwrap_or(input))?;
        let output = match &self.norm {
            Some(norm) => norm.run(&output)?,
            None => output,
        };
        Ok(match self.act {
            Activation::Relu => output.f_relu()?,
            Activation::Leaky => output.f_leaky_relu()?,
        })
    }
}

struct Pyramid {
    pool: Block,
    paths: Vec<Block>,
    merge: Block,
}

impl Pyramid {
    fn run(&self, input: &Tensor) -> Result<Tensor> {
        let (_, _, height, width) = input.size4()?;
        let pooled = input.f_mean_dim(&[2i64][..], true, None)?;
        let mut features = vec![self.pool.run(&pooled)?.f_upsample_bilinear2d(
            [height, width],
            true,
            None,
            None,
        )?];
        for path in &self.paths {
            features.push(path.run(input)?);
        }
        self.merge.run(&Tensor::f_cat(&features, 1)?)
    }
}

struct Recurrent {
    conv: Block,
    parameters: Vec<Tensor>,
    weight: Tensor,
    bias: Tensor,
    norm: Norm,
    hidden: i64,
}

impl Recurrent {
    fn run(&self, input: &Tensor) -> Result<Tensor> {
        let (n, _, bins, frames) = input.size4()?;
        let sequence = self.conv.run(input)?.f_select(1, 0)?.f_permute([2, 0, 1])?;
        let state = Tensor::f_zeros([2, n, self.hidden], (sequence.kind(), sequence.device()))?;
        let parameters: Vec<_> = self.parameters.iter().collect();
        let (features, _, _) = sequence.f_lstm(
            &[&state, &state],
            &parameters,
            true,
            1,
            0.0,
            false,
            true,
            false,
        )?;
        let projected = features
            .f_reshape([-1, self.hidden * 2])?
            .f_linear(&self.weight, Some(&self.bias))?;
        Ok(self
            .norm
            .run(&projected)?
            .f_relu()?
            .f_reshape([frames, n, 1, bins])?
            .f_permute([1, 2, 3, 0])?)
    }
}

struct Subnetwork {
    first: Option<Block>,
    down: Vec<[Block; 2]>,
    aspp: Pyramid,
    up: Vec<Block>,
    recurrent: Option<Recurrent>,
    last: Option<Block>,
}

impl Subnetwork {
    fn run(&self, input: &Tensor) -> Result<Tensor> {
        let mut hidden = match &self.first {
            Some(layer) => layer.run(input)?,
            None => input.shallow_clone(),
        };
        let mut skip = Vec::with_capacity(self.down.len());
        for [first, second] in &self.down {
            let features = first.run(&hidden)?;
            let downsampled = second.run(&features)?;
            skip.push(if self.first.is_some() {
                hidden
            } else {
                features
            });
            hidden = downsampled;
        }
        hidden = self.aspp.run(&hidden)?;
        for (step, (decoder, shortcut)) in self.up.iter().zip(skip.into_iter().rev()).enumerate() {
            if let Some(recurrent) = self
                .recurrent
                .as_ref()
                .filter(|_| step + 1 == self.up.len())
            {
                let context = recurrent.run(&hidden)?;
                hidden = Tensor::f_cat(&[hidden, context], 1)?;
            }
            let (_, _, bins, frames) = hidden.size4()?;
            hidden = hidden.f_upsample_bilinear2d([2 * bins, 2 * frames], true, None, None)?;
            let (_, _, shortcut_bins, shortcut_frames) = shortcut.size4()?;
            ensure!(
                shortcut_bins == 2 * bins && shortcut_frames >= 2 * frames,
                "VR skip tensor is smaller than the upsampled window"
            );
            let shortcut = shortcut.f_narrow(3, (shortcut_frames - 2 * frames) / 2, 2 * frames)?;
            hidden = decoder.run(&Tensor::f_cat(&[hidden, shortcut], 1)?)?;
        }
        match &self.last {
            Some(layer) => layer.run(&hidden),
            None => Ok(hidden),
        }
    }
}

pub(super) fn channels_last(input: &Tensor) -> Result<Tensor> {
    Ok(input
        .f_permute([0, 2, 3, 1])?
        .f_contiguous()?
        .f_permute([0, 3, 1, 2])?)
}
