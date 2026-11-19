use anyhow::{Context, Result, ensure};
use tch::Tensor;

use crate::model::weights::Weights;

use super::Config;

pub(super) struct Stage {
    pub(super) frequency: bool,
    pub(super) stride: i64,
    pub(super) width: i64,
    pub(super) pad: i64,
    empty: bool,
    pub(super) bands: Vec<f64>,
}

impl Stage {
    pub(super) fn new(
        frequency: bool,
        stride: i64,
        width: i64,
        pad: i64,
        empty: bool,
        bands: Option<&[f64]>,
    ) -> Result<Self> {
        let bands = bands.unwrap_or_default();
        ensure!(
            bands.len() <= 32
                && (bands.is_empty() || frequency)
                && bands.iter().all(|&ratio| 0.0 < ratio && ratio < 1.0)
                && bands.windows(2).all(|pair| pair[0] < pair[1]),
            "invalid Demucs frequency bands"
        );
        Ok(Self {
            frequency,
            stride,
            width,
            pad,
            empty,
            bands: bands.to_vec(),
        })
    }

    pub(super) fn empty(&self) -> bool {
        self.empty
    }

    pub(super) fn encode(
        &self,
        weights: &Weights,
        name: &str,
        x: &Tensor,
        inject: Option<&Tensor>,
        config: &Config,
        legacy: bool,
    ) -> Result<Tensor> {
        if self.bands.is_empty() {
            return self.encode_band(weights, name, x, inject, config, self.pad, legacy);
        }
        ensure!(
            inject.is_none(),
            "Demucs cannot inject waveform into a split frequency layer"
        );
        let freqs = x.size()[2];
        let mut start = 0;
        let mut outputs = Vec::new();
        let pad = self.width / 4;
        for (index, ratio) in self
            .bands
            .iter()
            .copied()
            .chain(std::iter::once(1.0))
            .enumerate()
        {
            let first = if index == 0 { pad } else { 0 };
            let last = if ratio == 1.0 { pad } else { 0 };
            let end = if ratio == 1.0 {
                freqs
            } else {
                let target = (freqs as f64 * ratio).round_ties_even() as i64;
                let span = target - start + first;
                let frames = ((span - self.width) as f64 / self.stride as f64 + 1.0)
                    .round_ties_even() as i64;
                start + (frames - 1) * self.stride + self.width - first
            };
            ensure!(
                start < end && end <= freqs,
                "Demucs encoder frequency band is empty or out of range"
            );
            let band = x
                .f_narrow(2, start, end - start)?
                .f_constant_pad_nd([0, 0, first, last])?;
            outputs.push(self.encode_band(
                weights,
                &format!("{name}.layers.{index}"),
                &band,
                None,
                config,
                0,
                legacy,
            )?);
            start = end - self.width + self.stride;
            ensure!(start >= 0, "invalid Demucs frequency overlap");
        }
        Ok(Tensor::f_cat(&outputs, 2)?)
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_band(
        &self,
        weights: &Weights,
        name: &str,
        x: &Tensor,
        inject: Option<&Tensor>,
        config: &Config,
        pad: i64,
        legacy: bool,
    ) -> Result<Tensor> {
        let key = format!("{name}.conv");
        let mut y = if self.frequency {
            conv2(weights, &key, x, [self.stride, 1], [pad, 0])?
        } else {
            let mut x = x.f_reshape([
                x.size()[0],
                -1,
                x.size().last().copied().context("empty Demucs shape")?,
            ])?;
            let remainder = x.size()[2] % self.stride;
            if remainder != 0 {
                x = x.f_constant_pad_nd([0, self.stride - remainder])?;
            }
            weights.conv(&x, &key, self.stride, pad, 1, 1)?
        };
        if self.empty {
            return Ok(y);
        }
        if let Some(inject) = inject {
            y = y.f_add(inject)?;
        }
        y = group_norm(weights, &format!("{name}.norm1"), &y, config.norm_groups)?
            .f_gelu("none")?;
        y = self.temporal_residual(weights, name, y, config.dconv_depth, legacy)?;
        self.rewrite(weights, name, y, config.norm_groups, "norm2")
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn decode(
        &self,
        weights: &Weights,
        name: &str,
        x: &Tensor,
        skip: Option<&Tensor>,
        length: i64,
        last: bool,
        config: &Config,
        legacy: bool,
    ) -> Result<(Tensor, Tensor)> {
        if self.bands.is_empty() {
            return self.decode_band(
                weights, name, x, skip, length, last, self.pad, config, legacy,
            );
        }
        let skip = skip.context("missing Demucs frequency band skip input")?;
        let freqs = skip.size()[2];
        let half_stride = self.stride / 2;
        let mut start = 0;
        let mut outputs: Vec<Tensor> = Vec::new();
        for (index, ratio) in self
            .bands
            .iter()
            .copied()
            .chain(std::iter::once(1.0))
            .enumerate()
        {
            let end = if ratio == 1.0 {
                freqs
            } else {
                (freqs as f64 * ratio).round_ties_even() as i64
            };
            ensure!(start < end, "empty Demucs decoder band");
            let name = format!("{name}.layers.{index}");
            let (mut band, _) = self.decode_band(
                weights,
                &name,
                &x.f_narrow(2, start, end - start)?,
                Some(&skip.f_narrow(2, start, end - start)?),
                length,
                true,
                0,
                config,
                legacy,
            )?;
            if let Some(previous) = outputs.last_mut() {
                let bias = weights
                    .get(&format!("{name}.conv_tr.bias"))?
                    .f_view([1, -1, 1, 1])?;
                let offset = previous.size()[2] - self.stride;
                let mut seam = previous.f_narrow(2, offset, self.stride)?;
                let _ = seam.f_add_(&band.f_narrow(2, 0, self.stride)?.f_sub(&bias)?)?;
                band = band.f_narrow(2, self.stride, band.size()[2] - self.stride)?;
            }
            let trim_left = if index == 0 { half_stride } else { 0 };
            let trim_right = if ratio == 1.0 { half_stride } else { 0 };
            outputs.push(band.f_narrow(2, trim_left, band.size()[2] - trim_left - trim_right)?);
            start = end;
        }
        let y = Tensor::f_cat(&outputs, 2)?;
        Ok((if last { y } else { y.f_gelu("none")? }, Tensor::new()))
    }

    #[allow(clippy::too_many_arguments)]
    fn decode_band(
        &self,
        weights: &Weights,
        name: &str,
        x: &Tensor,
        skip: Option<&Tensor>,
        length: i64,
        last: bool,
        pad: i64,
        config: &Config,
        legacy: bool,
    ) -> Result<(Tensor, Tensor)> {
        let kernel = format!("{name}.conv_tr.weight");
        let mut x = x.shallow_clone();
        if self.frequency && x.dim() == 3 {
            x = x.f_reshape([
                x.size()[0],
                weights.get(&kernel)?.size()[0],
                -1,
                x.size()[2],
            ])?;
        }
        if !self.empty {
            x = x.f_add(skip.context("missing Demucs skip tensor")?)?;
            x = self.rewrite(weights, name, x, config.norm_groups, "norm1")?;
            x = self.temporal_residual(weights, name, x, config.dconv_depth, legacy)?;
        }
        let key = format!("{name}.conv_tr");
        let decoded = if self.frequency {
            x.f_conv_transpose2d(
                weights.get(&kernel)?,
                weights.get(&format!("{key}.bias")).ok(),
                [self.stride, 1],
                [0, 0],
                [0, 0],
                1,
                [1, 1],
            )?
        } else {
            weights.deconv(&x, &key, self.stride, 0)?
        };
        let mut y = group_norm(
            weights,
            &format!("{name}.norm2"),
            &decoded,
            config.norm_groups,
        )?;
        if self.frequency {
            y = y.f_narrow(2, pad, y.size()[2] - pad * 2)?;
        } else {
            y = y.f_narrow(2, pad, length)?;
        }
        Ok((if last { y } else { y.f_gelu("none")? }, x))
    }

    fn rewrite(
        &self,
        weights: &Weights,
        name: &str,
        x: Tensor,
        groups: i64,
        norm: &str,
    ) -> Result<Tensor> {
        let key = format!("{name}.rewrite");
        let Ok(kernel) = weights.get(&format!("{key}.weight")) else {
            return Ok(x);
        };
        let dimensions = kernel.size();
        let y = if self.frequency {
            ensure!(
                dimensions.len() == 4,
                "invalid Demucs frequency rewrite shape"
            );
            conv2(
                weights,
                &key,
                &x,
                [1, 1],
                [dimensions[2] / 2, dimensions[3] / 2],
            )?
        } else {
            ensure!(dimensions.len() == 3, "invalid Demucs time rewrite shape");
            weights.conv(&x, &key, 1, dimensions[2] / 2, 1, 1)?
        };
        Ok(group_norm(weights, &format!("{name}.{norm}"), &y, groups)?.f_glu(1)?)
    }

    fn temporal_residual(
        &self,
        weights: &Weights,
        name: &str,
        x: Tensor,
        depth: i64,
        legacy: bool,
    ) -> Result<Tensor> {
        let prefix = format!("{name}.dconv");
        if !weights.contains(&format!("{prefix}.layers.0.0.weight")) {
            return Ok(x);
        }
        if self.frequency {
            let (b, c, f, t) = (x.size()[0], x.size()[1], x.size()[2], x.size()[3]);
            let series = x.f_permute([0, 2, 1, 3])?.f_reshape([b * f, c, t])?;
            let hidden = residual(weights, &prefix, series, depth, legacy)?;
            Ok(hidden.f_view([b, f, c, t])?.f_transpose(1, 2)?)
        } else {
            residual(weights, &prefix, x, depth, legacy)
        }
    }
}

fn conv2(
    weights: &Weights,
    name: &str,
    x: &Tensor,
    stride: [i64; 2],
    pad: [i64; 2],
) -> Result<Tensor> {
    Ok(x.f_conv2d(
        weights.get(&format!("{name}.weight"))?,
        weights.get(&format!("{name}.bias")).ok(),
        stride,
        pad,
        [1, 1],
        1,
    )?)
}

pub(super) fn group_norm(weights: &Weights, name: &str, x: &Tensor, groups: i64) -> Result<Tensor> {
    let Ok(weight) = weights.get(&format!("{name}.weight")) else {
        return Ok(x.shallow_clone());
    };
    Ok(x.f_group_norm(
        groups,
        Some(weight),
        Some(weights.get(&format!("{name}.bias"))?),
        1e-5,
        true,
    )?)
}

fn residual(
    weights: &Weights,
    prefix: &str,
    mut x: Tensor,
    depth: i64,
    legacy: bool,
) -> Result<Tensor> {
    if legacy {
        let recurrent = weights.contains(&format!("{prefix}.layers.0.3.lstm.weight_ih_l0"));
        let attention_index = if recurrent { 4 } else { 3 };
        let attention =
            weights.contains(&format!("{prefix}.layers.0.{attention_index}.query.weight"));
        return crate::separation::legacy_wave::residual_stack(
            weights,
            x,
            prefix,
            depth as i32,
            recurrent,
            attention,
        );
    }
    for d in 0..depth.unsigned_abs() {
        let base = format!("{prefix}.layers.{d}");
        let dilation = 1 << d;
        let mut y = weights.conv(&x, &format!("{base}.0"), 1, dilation, dilation, 1)?;
        y = group_norm(weights, &format!("{base}.1"), &y, 1)?.f_gelu("none")?;
        y = weights.conv(&y, &format!("{base}.3"), 1, 0, 1, 1)?;
        y = group_norm(weights, &format!("{base}.4"), &y, 1)?.f_glu(1)?;
        let scale = weights.get(&format!("{base}.6.scale"))?.f_unsqueeze(-1)?;
        x = x.f_add(&scale.f_mul(&y)?)?;
    }
    Ok(x)
}
