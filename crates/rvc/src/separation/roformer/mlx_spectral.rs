use anyhow::{Context, Result, bail};
use tch::{Device, Tensor};

use super::super::mlx::{Array, DType, Stream};
use super::super::mlx_stft::Stft;
use super::{MelSelection, ModelConfig};

pub(super) struct Spectral {
    channels: i32,
    stems: i32,
    bins: i32,
    preserve_length: bool,
    stft: Stft,
    mel: Option<Mel>,
    dc_gate: Option<Array>,
}

struct Mel {
    select: Array,
    scatter: Array,
    coverage: Array,
}

impl Spectral {
    pub(super) fn new(
        stream: &Stream,
        config: &ModelConfig,
        mel: Option<&MelSelection>,
        wnd: &Tensor,
    ) -> Result<Self> {
        let channels = i32::try_from(config.channels)?;
        let bins = i32::try_from(config.fft / 2 + 1)?;
        let side = (config.fft - config.window) / 2;
        let window = wnd.f_constant_pad_nd([side, config.fft - config.window - side])?;
        let stft = Stft::new(
            stream,
            config.fft,
            config.hop,
            config.normalized_stft,
            &window,
        )?;
        let mel = mel
            .map(|mel| -> Result<_> {
                let indices =
                    Vec::<i64>::try_from(mel.indices.f_to_device(Device::Cpu)?.f_view([-1])?)?;
                let indices = indices
                    .into_iter()
                    .map(i32::try_from)
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                let select = stream.indices(&indices, &[i32::try_from(indices.len())?])?;
                let scatter: Vec<_> = indices
                    .iter()
                    .flat_map(|&index| [2 * index, 2 * index + 1])
                    .collect();
                let scatter =
                    stream.indices(&scatter, &[1, 1, 1, i32::try_from(scatter.len())?])?;
                let coverage =
                    Vec::<f32>::try_from(mel.coverage.f_to_device(Device::Cpu)?.f_view([-1])?)?;
                let coverage: Vec<_> = coverage.iter().flat_map(|&count| [count, count]).collect();
                let coverage = stream.float32(&coverage, &[1, 1, 1, bins * channels * 2])?;
                Ok(Mel {
                    select,
                    scatter,
                    coverage,
                })
            })
            .transpose()?;
        let dc_gate = config
            .zero_dc
            .then(|| {
                let mut samples = vec![1.; bins as usize];
                samples[0] = 0.;
                stream.float32(&samples, &[bins])
            })
            .transpose()?;
        Ok(Self {
            channels,
            stems: i32::try_from(config.stems)?,
            bins,
            preserve_length: config.match_input_length,
            stft,
            mel,
            dc_gate,
        })
    }

    pub(super) fn analyze(
        &self,
        stream: &Stream,
        audio: &[f32],
        batch: usize,
        samples: usize,
    ) -> Result<(Array, Array)> {
        let batch = i32::try_from(batch)?;
        let spectrum = self.stft.analyze(stream, audio, samples)?;
        let frames = spectrum.shape()[1];
        let spectrum = stream.reshape(&spectrum, &[batch, self.channels, frames, self.bins])?;
        let spectrum = stream.transpose(&spectrum, &[0, 2, 3, 1])?;
        let spectrum = stream.reshape(&spectrum, &[batch, frames, self.bins * self.channels])?;
        let selected = self
            .mel
            .as_ref()
            .map(|mel| stream.take(&spectrum, &mel.select, 2))
            .transpose()?;
        let selected = selected.as_ref().unwrap_or(&spectrum);
        let features = stream.stack(&[stream.real(selected)?, stream.imag(selected)?], -1)?;
        let features = stream.reshape(&features, &[batch, frames, -1])?;
        Ok((spectrum, features))
    }

    pub(super) fn render(
        &self,
        stream: &Stream,
        spectrum: &Array,
        masks: &Array,
        samples: usize,
    ) -> Result<Vec<f32>> {
        let [batch, frames, _] = spectrum.shape() else {
            bail!("invalid MLX spectrum rank")
        };
        let (batch, frames) = (*batch, *frames);
        let tracks = batch
            .checked_mul(self.stems * self.channels)
            .context("MLX output batch overflow")?;
        let masks = stream.cast(masks, DType::F32)?;
        let masks = if let Some(mel) = &self.mel {
            let base = stream.zeros(&[batch, self.stems, frames, self.bins * self.channels * 2])?;
            let indices = stream.broadcast(&mel.scatter, masks.shape())?;
            let merged = stream.scatter_add(&base, &indices, &masks, 3)?;
            stream.div(&merged, &mel.coverage)?
        } else {
            masks
        };
        let masks = stream.reshape(
            &masks,
            &[batch, self.stems, frames, self.bins * self.channels, 2],
        )?;
        let masks = stream.complex(&stream.select(&masks, 4, 0)?, &stream.select(&masks, 4, 1)?)?;
        let spectrum = stream.reshape(spectrum, &[batch, 1, frames, self.bins * self.channels])?;
        let output = stream.mul(&spectrum, &masks)?;
        let output = stream.reshape(
            &output,
            &[batch, self.stems, frames, self.bins, self.channels],
        )?;
        let output = stream.transpose(&output, &[0, 1, 4, 2, 3])?;
        let output = stream.reshape(&output, &[tracks, frames, self.bins])?;
        let output = match &self.dc_gate {
            Some(gate) => stream.mul(&output, gate)?,
            None => output,
        };
        let audio =
            self.stft
                .synthesize(stream, &output, self.preserve_length.then_some(samples))?;
        stream.to_floats(&audio)
    }
}
