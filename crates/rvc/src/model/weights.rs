use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use tch::{Device, Kind, Tensor};

use crate::checkpoint::Checkpoint;

mod cuda;
pub use cuda::{cuda_device_count, cuda_profile, cuda_profiles};

// Keep LibTorch ahead of FAISS's system BLAS when ELF resolves the Fortran
// BLAS symbols exported by both. LibTorch 2.7 exports at::get_num_threads().
#[cfg(all(target_os = "linux", feature = "index-training"))]
#[link(name = "torch_cpu")]
unsafe extern "C" {
    #[link_name = "_ZN2at15get_num_threadsEv"]
    fn torch_cpu_threads() -> std::ffi::c_int;
}

pub struct Weights {
    tensors: HashMap<String, Tensor>,
    cpu_channels_last: bool,
}

impl Weights {
    pub fn load(path: &Path, section: Option<&str>, device: Device) -> Result<Self> {
        Ok(Self::new(Checkpoint::open(path)?.weights(section, device)?))
    }

    pub fn new(weights: HashMap<String, Tensor>) -> Self {
        Self {
            tensors: weights,
            cpu_channels_last: false,
        }
    }

    pub fn get(&self, name: &str) -> Result<&Tensor> {
        self.tensors
            .get(name)
            .with_context(|| format!("checkpoint tensor {name} is missing"))
    }

    pub fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    #[cfg(feature = "onnx")]
    pub(super) fn retain(&mut self, mut keep: impl FnMut(&str) -> bool) {
        self.tensors.retain(|name, _| keep(name));
    }

    pub fn move_to(&mut self, device: Device, kind: Kind) -> Result<()> {
        for tensor in self.tensors.values_mut() {
            let dtype = if tensor.is_floating_point() {
                kind
            } else {
                tensor.kind()
            };
            *tensor = tensor.f_to_device_(device, dtype, false, false)?;
        }
        Ok(())
    }

    pub(super) fn prepare_cpu_convolutions(&mut self, device: Device) -> Result<()> {
        if device != Device::Cpu || tch::get_num_threads() < 2 || !tch::utils::has_mkldnn() {
            return Ok(());
        }
        for (name, tensor) in &mut self.tensors {
            if name.ends_with(".weight") {
                match tensor.dim() {
                    3 => {
                        *tensor = tensor
                            .f_transpose(1, 2)?
                            .f_contiguous()?
                            .f_transpose(1, 2)?;
                    }
                    4 => {
                        *tensor = tensor
                            .f_permute([0, 2, 3, 1])?
                            .f_contiguous()?
                            .f_permute([0, 3, 1, 2])?;
                    }
                    _ => {}
                }
            }
        }
        self.cpu_channels_last = true;
        Ok(())
    }

    pub(crate) fn kernel(&self, name: &str) -> Result<Tensor> {
        if let Some(kernel) = self.tensors.get(&format!("{name}.weight")) {
            Ok(kernel.shallow_clone())
        } else {
            let g = self.get(&format!("{name}.weight_g"))?;
            let v = self.get(&format!("{name}.weight_v"))?;
            Ok(Tensor::f_internal_weight_norm(v, g, 0)?)
        }
    }

    pub fn remove_weight_norm(&mut self, axis: i64) -> Result<()> {
        let names: Vec<_> = self
            .tensors
            .keys()
            .filter_map(|name| {
                name.strip_suffix(".weight_g")
                    .map(|base| (base.to_owned(), ".weight_g", ".weight_v"))
                    .or_else(|| {
                        name.strip_suffix(".parametrizations.weight.original0")
                            .map(|base| {
                                (
                                    base.to_owned(),
                                    ".parametrizations.weight.original0",
                                    ".parametrizations.weight.original1",
                                )
                            })
                    })
            })
            .collect();
        for (base, magnitude, direction) in names {
            let g = self.get(&format!("{base}{magnitude}"))?;
            let v = self.get(&format!("{base}{direction}"))?;
            let reduce: Vec<_> = (0..v.dim() as i64)
                .filter(|&dimension| dimension != axis.rem_euclid(v.dim() as i64))
                .collect();
            let norm = v
                .f_square()?
                .f_sum_dim_intlist(&reduce[..], true, Kind::Float)?
                .f_sqrt()?;
            let weight = v.f_mul(&g.f_div(&norm)?)?;
            self.tensors.insert(format!("{base}.weight"), weight);
            self.tensors.remove(&format!("{base}{magnitude}"));
            self.tensors.remove(&format!("{base}{direction}"));
        }
        Ok(())
    }

    pub fn linear(&self, x: &Tensor, name: &str) -> Result<Tensor> {
        Ok(x.f_linear(
            self.get(&format!("{name}.weight"))?,
            self.tensors.get(&format!("{name}.bias")),
        )?)
    }

    pub(super) fn pack_attention(&mut self, name: &str) -> Result<()> {
        let projections = ["q", "k", "v"].map(|projection| format!("{name}.{projection}_proj"));
        if projections
            .iter()
            .any(|key| !self.contains(&format!("{key}.weight")))
        {
            return Ok(());
        }
        let mut weights = Vec::with_capacity(3);
        let mut biases = Vec::with_capacity(3);
        for key in projections {
            let weight = self
                .tensors
                .remove(&format!("{key}.weight"))
                .with_context(|| format!("checkpoint tensor {key}.weight is missing"))?;
            ensure!(weight.dim() == 2, "invalid {key} weight dimensions");
            let bias = match self.tensors.remove(&format!("{key}.bias")) {
                Some(bias) => bias,
                None => Tensor::f_zeros([weight.size()[0]], (weight.kind(), weight.device()))?,
            };
            weights.push(weight);
            biases.push(bias);
        }
        self.tensors.insert(
            format!("{name}.qkv_proj.weight"),
            Tensor::f_cat(&weights, 0)?,
        );
        self.tensors
            .insert(format!("{name}.qkv_proj.bias"), Tensor::f_cat(&biases, 0)?);
        Ok(())
    }

    pub fn norm(&self, x: &Tensor, name: &str, epsilon: f64) -> Result<Tensor> {
        self.layer_norm(x, name, epsilon, "weight", "bias")
    }

    pub fn rvc_norm(&self, x: &Tensor, name: &str) -> Result<Tensor> {
        Ok(self
            .layer_norm(&x.transpose(1, 2), name, 1e-5, "gamma", "beta")?
            .transpose(1, 2))
    }

    fn layer_norm(
        &self,
        x: &Tensor,
        name: &str,
        epsilon: f64,
        weight: &str,
        bias: &str,
    ) -> Result<Tensor> {
        let weight = self.get(&format!("{name}.{weight}"))?;
        let bias = self.get(&format!("{name}.{bias}"))?;
        Ok(x.f_layer_norm(weight.size(), Some(weight), Some(bias), epsilon, true)?)
    }

    pub fn conv(
        &self,
        x: &Tensor,
        name: &str,
        stride: i64,
        pad: i64,
        dilation: i64,
        groups: i64,
    ) -> Result<Tensor> {
        let weight = self.kernel(name)?;
        let bias = self.tensors.get(&format!("{name}.bias"));
        if self.cpu_channels_last {
            // A unit-height view selects LibTorch's channels-last CPU kernel.
            return Ok(x
                .f_unsqueeze(2)?
                .f_conv2d(
                    &weight.f_unsqueeze(2)?,
                    bias,
                    [1, stride],
                    [0, pad],
                    [1, dilation],
                    groups,
                )?
                .f_squeeze_dim(2)?);
        }
        Ok(x.f_conv1d(&weight, bias, stride, pad, dilation, groups)?)
    }

    pub fn conv2(&self, x: &Tensor, name: &str, pad: i64) -> Result<Tensor> {
        Ok(x.f_conv2d(
            self.get(&format!("{name}.weight"))?,
            self.tensors.get(&format!("{name}.bias")),
            [1, 1],
            [pad, pad],
            [1, 1],
            1,
        )?)
    }

    pub fn deconv(&self, x: &Tensor, name: &str, stride: i64, pad: i64) -> Result<Tensor> {
        let weight = self.kernel(name)?;
        let bias = self.tensors.get(&format!("{name}.bias"));
        if self.cpu_channels_last {
            return Ok(x
                .f_unsqueeze(2)?
                .f_conv_transpose2d(
                    &weight.f_unsqueeze(2)?,
                    bias,
                    [1, stride],
                    [0, pad],
                    [0, 0],
                    1,
                    [1, 1],
                )?
                .f_squeeze_dim(2)?);
        }
        Ok(x.f_conv_transpose1d(&weight, bias, stride, pad, 0, 1, 1)?)
    }
}

pub fn device(requested: &str) -> Result<Device> {
    #[cfg(all(target_os = "linux", feature = "index-training"))]
    {
        // Reads LibTorch's existing CPU thread count and anchors the link.
        unsafe { torch_cpu_threads() };
    }
    match requested {
        "auto" => {
            let profiles = cuda_profiles();
            let best = profiles
                .iter()
                .enumerate()
                .filter_map(|(index, profile)| {
                    profile
                        .as_ref()
                        .filter(|profile| profile.eligible())
                        .map(|profile| (index, profile))
                })
                .max_by(|(index_a, a), (index_b, b)| {
                    a.capability
                        .cmp(&b.capability)
                        .then_with(|| a.memory_gib.total_cmp(&b.memory_gib))
                        .then_with(|| index_b.cmp(index_a))
                });
            Ok(best.map_or_else(
                || {
                    if !profiles.is_empty() && profiles.iter().all(Option::is_none) {
                        Device::Cuda(0)
                    } else {
                        Device::Cpu
                    }
                },
                |(index, _)| Device::Cuda(index),
            ))
        }
        "cpu" => Ok(Device::Cpu),
        "mps" => {
            ensure!(
                tch::utils::has_mps(),
                "Metal Performance Shaders are unavailable"
            );
            Ok(Device::Mps)
        }
        "cuda" => device("cuda:0"),
        id if id.starts_with("cuda:") => {
            let index: usize = id[5..].parse().context("invalid CUDA device number")?;
            ensure!(
                index < cuda_device_count(),
                "CUDA device {index} is unavailable"
            );
            Ok(Device::Cuda(index))
        }
        _ => bail!("unsupported device {requested}; use auto, cpu, cuda:N or mps"),
    }
}

pub fn leaky(x: &Tensor, slope: f64) -> Tensor {
    // Multiplying x by the piecewise slope is LeakyReLU itself. The ATen
    // kernel exposes the slope here, including the derivative at zero.
    x.leaky_relu_backward(x, slope, false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn leaky_relu_keeps_the_configured_slope_and_gradient() {
        let x = Tensor::from_slice(&[-2.0_f32, 0.0, 3.0]).set_requires_grad(true);
        let result = leaky(&x, 0.1);
        assert_eq!(Vec::<f32>::try_from(&result).unwrap(), [-0.2, 0.0, 3.0]);
        result.sum(Kind::Float).backward();
        assert_eq!(Vec::<f32>::try_from(x.grad()).unwrap(), [0.1, 0.1, 1.0]);
    }

    #[test]
    fn cpu_convolution_layout_preserves_groups_dilation_and_upsampling() -> Result<()> {
        let values = |shape: &[i64]| {
            (Tensor::arange(shape.iter().product::<i64>(), (Kind::Float, Device::Cpu)) / 32.0)
                .view(shape)
        };
        let mut weights = Weights::new(HashMap::from([
            ("conv.weight".to_owned(), values(&[4, 2, 3])),
            ("conv.bias".to_owned(), values(&[4])),
            ("up.weight".to_owned(), values(&[4, 3, 4])),
        ]));
        let input = values(&[1, 4, 13]);
        let convolution = weights.conv(&input, "conv", 2, 2, 2, 2)?;
        let upsampled = weights.deconv(&input, "up", 2, 1)?;
        weights.prepare_cpu_convolutions(Device::Cpu)?;
        let packed = weights.conv(&input, "conv", 2, 2, 2, 2)?;
        let packed_up = weights.deconv(&input, "up", 2, 1)?;
        assert!(convolution.allclose(&packed, 1e-5, 1e-6, false));
        assert!(upsampled.allclose(&packed_up, 1e-5, 1e-6, false));
        Ok(())
    }
}
