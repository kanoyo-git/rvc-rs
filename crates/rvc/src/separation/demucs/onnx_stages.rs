use anyhow::{Context, Result, ensure};

use super::{Export, Feature};
use crate::separation::demucs::layers::Stage;
use crate::separation::legacy_wave::onnx_network::{Layers, Series};
use crate::separation::onnx_layers::{concat_strings, glu, slice, transpose};

impl Export<'_> {
    pub(super) fn encode(
        &mut self,
        stage: &Stage,
        name: &str,
        x: &Feature,
        inject: Option<&Feature>,
    ) -> Result<Feature> {
        if stage.bands.is_empty() {
            return self.encode_band(stage, name, x, inject, [stage.pad; 2]);
        }
        ensure!(
            inject.is_none(),
            "HTDemucs cannot inject waveforms into a split layer"
        );
        let bins = x.shape[1];
        let mut start = 0;
        let mut encoded = Vec::with_capacity(stage.bands.len() + 1);
        let pad = stage.width / 4;
        for (index, ratio) in stage
            .bands
            .iter()
            .copied()
            .chain(std::iter::once(1.))
            .enumerate()
        {
            let front = if index == 0 { pad } else { 0 };
            let tail = if ratio == 1. { pad } else { 0 };
            let end = if ratio == 1. {
                bins
            } else {
                let target = (bins as f64 * ratio).round_ties_even() as i64;
                let span = target - start + front;
                let frames = ((span - stage.width) as f64 / stage.stride as f64 + 1.0)
                    .round_ties_even() as i64;
                start + (frames - 1) * stage.stride + stage.width - front
            };
            let band = self.narrow(x, 2, start, end)?;
            encoded.push(self.encode_band(
                stage,
                &format!("{name}.layers.{index}"),
                &band,
                None,
                [front, tail],
            )?);
            start = end - stage.width + stage.stride;
        }
        self.merge_bands(encoded)
    }

    fn encode_band(
        &mut self,
        stage: &Stage,
        name: &str,
        x: &Feature,
        inject: Option<&Feature>,
        [front, tail]: [i64; 2],
    ) -> Result<Feature> {
        let key = format!("{name}.conv");
        let mut y = if stage.frequency {
            self.conv(&key, x, &[stage.stride, 1], &[front, 0, tail, 0], 1, false)?
        } else {
            let length = x.length();
            let channels = x.shape.iter().take(x.shape.len() - 1).product();
            let x = self.reshape(x, &[channels, length]);
            let right = (stage.stride - length % stage.stride) % stage.stride;
            self.conv(&key, &x, &[stage.stride], &[front, tail + right], 1, false)?
        };
        if stage.empty() {
            return Ok(y);
        }
        if let Some(inject) = inject {
            y.value = self.graph.op("Add", &[&y.value, &inject.value], vec![]);
        }
        let y = self.group_norm(&format!("{name}.norm1"), &y, self.config.norm_groups)?;
        let y = self.gelu(&y);
        let y = self.residual(stage, name, y)?;
        self.rewrite(stage, name, y, "norm2")
    }

    pub(super) fn decode(
        &mut self,
        stage: &Stage,
        name: &str,
        x: &Feature,
        skip: Option<&Feature>,
        length: i64,
        last: bool,
    ) -> Result<(Feature, Feature)> {
        if stage.bands.is_empty() {
            return self.decode_band(stage, name, x, skip, length, last, stage.pad);
        }
        let skip = skip.context("missing HTDemucs frequency band skip")?;
        let bins = skip.shape[1];
        let mut start = 0;
        let mut outputs: Vec<Feature> = Vec::with_capacity(stage.bands.len() + 1);
        for (index, ratio) in stage
            .bands
            .iter()
            .copied()
            .chain(std::iter::once(1.))
            .enumerate()
        {
            let end = (bins as f64 * ratio).round_ties_even() as i64;
            let name = format!("{name}.layers.{index}");
            let source = self.narrow(x, 2, start, end)?;
            let skip = self.narrow(skip, 2, start, end)?;
            let (mut decoded, _) =
                self.decode_band(stage, &name, &source, Some(&skip), length, true, 0)?;
            if let Some(previous) = outputs.last_mut() {
                let offset = previous.shape[1] - stage.stride;
                let trailing = self.narrow(previous, 2, offset, previous.shape[1])?;
                let overlap = self.narrow(&decoded, 2, 0, stage.stride)?;
                let bias = self
                    .weights
                    .get(&format!("{name}.conv_tr.bias"))?
                    .f_reshape([1, -1, 1, 1])?;
                let bias = self.graph.tensor(&bias)?;
                let shift = self.graph.op("Sub", &[&overlap.value, &bias], vec![]);
                let joined = self.graph.op("Add", &[&trailing.value, &shift], vec![]);
                *previous = if offset > 0 {
                    let leading = self.narrow(previous, 2, 0, offset)?;
                    self.merge_bands(vec![leading, trailing.with_value(joined)])?
                } else {
                    trailing.with_value(joined)
                };
                decoded = self.narrow(&decoded, 2, stage.stride, decoded.shape[1])?;
            }
            let front = if index == 0 { stage.stride / 2 } else { 0 };
            let back = if ratio == 1. { stage.stride / 2 } else { 0 };
            outputs.push(self.narrow(&decoded, 2, front, decoded.shape[1] - back)?);
            start = end;
        }
        let mut decoded = self.merge_bands(outputs)?;
        if !last {
            decoded = self.gelu(&decoded);
        }
        Ok((decoded, x.clone()))
    }

    #[allow(clippy::too_many_arguments)]
    fn decode_band(
        &mut self,
        stage: &Stage,
        name: &str,
        x: &Feature,
        skip: Option<&Feature>,
        length: i64,
        last: bool,
        pad: i64,
    ) -> Result<(Feature, Feature)> {
        let kernel = format!("{name}.conv_tr");
        let mut x = if stage.frequency && x.shape.len() == 2 {
            let channels = self.weights.get(&format!("{kernel}.weight"))?.size()[0];
            self.reshape(x, &[channels, x.shape[0] / channels, x.length()])
        } else {
            x.clone()
        };
        if !stage.empty() {
            let skip = skip.context("missing HTDemucs skip tensor")?;
            x.value = self.graph.op("Add", &[&x.value, &skip.value], vec![]);
            x = self.rewrite(stage, name, x, "norm1")?;
            x = self.residual(stage, name, x)?;
        }
        let decoded = if stage.frequency {
            self.conv(&kernel, &x, &[stage.stride, 1], &[0; 4], 1, true)?
        } else {
            self.conv(&kernel, &x, &[stage.stride], &[0; 2], 1, true)?
        };
        let decoded =
            self.group_norm(&format!("{name}.norm2"), &decoded, self.config.norm_groups)?;
        let end = if stage.frequency {
            decoded.shape[1] - pad
        } else {
            pad + length
        };
        let mut y = self.narrow(&decoded, 2, pad, end)?;
        if !last {
            y = self.gelu(&y);
        }
        Ok((y, x))
    }

    fn rewrite(&mut self, stage: &Stage, name: &str, x: Feature, norm: &str) -> Result<Feature> {
        let key = format!("{name}.rewrite");
        let Ok(kernel) = self.weights.get(&format!("{key}.weight")) else {
            return Ok(x);
        };
        let sizes = kernel.size();
        let y = if stage.frequency {
            let freq = sizes[2] / 2;
            let time = sizes[3] / 2;
            self.conv(&key, &x, &[1, 1], &[freq, time, freq, time], 1, false)?
        } else {
            self.conv(&key, &x, &[1], &[sizes[2] / 2; 2], 1, false)?
        };
        let y = self.group_norm(&format!("{name}.{norm}"), &y, self.config.norm_groups)?;
        Ok(self.gate(y))
    }

    fn residual(&mut self, stage: &Stage, name: &str, x: Feature) -> Result<Feature> {
        let prefix = format!("{name}.dconv");
        if !self
            .weights
            .contains(&format!("{prefix}.layers.0.0.weight"))
        {
            return Ok(x);
        }
        let mut y = if stage.frequency {
            let (channels, steps) = (x.shape[0], x.shape[2]);
            let series = transpose(&mut self.graph, &x.value, &[0, 2, 1, 3]);
            let series = crate::separation::onnx_layers::reshape(
                &mut self.graph,
                &series,
                &[-1, channels, steps],
            );
            Feature::new(series, [channels, steps])
        } else {
            x.clone()
        };
        if self.legacy {
            let recurrent = self
                .weights
                .contains(&format!("{prefix}.layers.0.3.lstm.weight_ih_l0"));
            let slot = if recurrent { 4 } else { 3 };
            let attention = self
                .weights
                .contains(&format!("{prefix}.layers.0.{slot}.query.weight"));
            let series = Series::new(y.value, y.shape[0], y.shape[1]);
            let mut layers = Layers {
                graph: &mut self.graph,
                weights: self.weights,
            };
            y.value = layers
                .residual(
                    series,
                    &prefix,
                    self.config.dconv_depth as i32,
                    recurrent,
                    attention,
                )?
                .value;
        } else {
            y = self.residual_layers(y, &prefix)?;
        }
        if stage.frequency {
            let shape = [-1, x.shape[1], x.shape[0], x.length()];
            let value = crate::separation::onnx_layers::reshape(&mut self.graph, &y.value, &shape);
            Ok(x.with_value(transpose(&mut self.graph, &value, &[0, 2, 1, 3])))
        } else {
            Ok(y)
        }
    }

    fn residual_layers(&mut self, mut y: Feature, prefix: &str) -> Result<Feature> {
        for depth in 0..self.config.dconv_depth.unsigned_abs() {
            let base = format!("{prefix}.layers.{depth}");
            let dilation = 1 << depth;
            let z = self.conv(
                &format!("{base}.0"),
                &y,
                &[1],
                &[dilation; 2],
                dilation,
                false,
            )?;
            let z = self.group_norm(&format!("{base}.1"), &z, 1)?;
            let z = self.gelu(&z);
            let z = self.conv(&format!("{base}.3"), &z, &[1], &[0; 2], 1, false)?;
            let z = self.group_norm(&format!("{base}.4"), &z, 1)?;
            let z = self.gate(z);
            let scale = self
                .weights
                .get(&format!("{base}.6.scale"))?
                .f_unsqueeze(-1)?;
            let scale = self.graph.tensor(&scale)?;
            let z = self.graph.op("Mul", &[&scale, &z.value], vec![]);
            y.value = self.graph.op("Add", &[&y.value, &z], vec![]);
        }
        Ok(y)
    }

    fn gate(&mut self, mut x: Feature) -> Feature {
        x.shape[0] /= 2;
        x.value = glu(&mut self.graph, &x.value, 1, x.shape[0]);
        x
    }

    fn narrow(&mut self, x: &Feature, axis: usize, start: i64, end: i64) -> Result<Feature> {
        ensure!(
            0 <= start && start < end && end <= x.shape[axis - 1],
            "invalid HTDemucs band crop"
        );
        let mut shape = x.shape.clone();
        shape[axis - 1] = end - start;
        Ok(Feature::new(
            slice(&mut self.graph, &x.value, axis as i64, start, end, 1),
            shape,
        ))
    }

    fn merge_bands(&mut self, inputs: Vec<Feature>) -> Result<Feature> {
        let mut shape = inputs
            .first()
            .context("missing HTDemucs frequency bands")?
            .shape
            .clone();
        shape[1] = inputs.iter().map(|part| part.shape[1]).sum();
        let values = inputs
            .into_iter()
            .map(|part| part.value)
            .collect::<Vec<_>>();
        Ok(Feature::new(
            concat_strings(&mut self.graph, &values, 2),
            shape,
        ))
    }
}
