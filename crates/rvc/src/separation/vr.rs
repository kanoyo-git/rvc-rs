//! Native UVR inference for the PyMSS VR architectures.

mod dsp;
mod network;
mod profiles;
mod soxr;

use std::ops::ControlFlow;
use std::path::Path;

use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use tch::{Device, Kind, Tensor};

use super::overlap::Progress;
use crate::checkpoint::Checkpoint;
use network::Network;
use profiles::{Metadata, Profile};

const OUTPUT_RATE: u32 = 44_100;

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(default)]
pub struct Options {
    pub window_size: usize,
    pub batch_size: usize,
    pub aggression: i32,
    pub enable_tta: bool,
    pub enable_post_process: bool,
    pub post_process_threshold: f32,
    pub high_end_process: bool,
    pub use_amp: bool,
    pub fuse_conv_bn: bool,
    pub use_channels_last: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            window_size: 512,
            batch_size: 2,
            aggression: 5,
            enable_tta: false,
            enable_post_process: false,
            post_process_threshold: 0.2,
            high_end_process: false,
            use_amp: true,
            fuse_conv_bn: false,
            use_channels_last: false,
        }
    }
}

pub struct Stem {
    pub name: &'static str,
    pub channels: usize,
    pub samples: Vec<f32>,
}

pub struct Vr {
    model: Network,
    metadata: Metadata,
    profile: Profile,
    options: Options,
    device: Device,
}

impl Vr {
    pub fn supports(checkpoint: &Path) -> bool {
        profiles::metadata(checkpoint).is_some()
    }

    pub fn load(
        checkpoint: &Path,
        device: Device,
        options: Options,
        document: &serde_json::Value,
    ) -> Result<Self> {
        let metadata = profiles::metadata(checkpoint)
            .with_context(|| format!("unsupported UVR VR model {}", checkpoint.display()))?;
        let profile = Profile::parse(metadata.parameters)?;
        ensure!(
            options.batch_size > 0
                && i64::try_from(options.batch_size).is_ok()
                && options.post_process_threshold.is_finite(),
            "invalid VR batch size or post-processing threshold"
        );
        let mlx = super::mlx::selected_dtype(document, device, true, "UVR VR")?;
        let field = &document["inference"]["onnx_provider"];
        let onnx = field.as_str();
        ensure!(
            field.is_null() || onnx.is_some(),
            "onnx_provider must be text"
        );
        #[cfg(not(feature = "onnx"))]
        ensure!(
            onnx.is_none(),
            "onnx_provider requires the onnx build feature"
        );
        let acceleration = mlx.map_or(network::Acceleration::Torch, |dtype| {
            network::Acceleration::Mlx(device, dtype)
        });
        #[cfg(feature = "onnx")]
        let acceleration = if let Some(onnx) = onnx {
            ensure!(
                device == Device::Cpu && mlx.is_none(),
                "UVR VR ONNX requires --device cpu and a disabled MLX backend"
            );
            network::Acceleration::Onnx(crate::model::onnx::Provider::parse("onnx_provider", onnx)?)
        } else {
            acceleration
        };
        let load_device = if acceleration.host_input() {
            Device::Cpu
        } else {
            device
        };
        let weights = Checkpoint::open(checkpoint)?.weights(None, load_device)?;
        let channels_last = options.use_channels_last && matches!(device, Device::Cuda(_));
        let model = Network::load(
            &weights,
            profile.bins,
            options.fuse_conv_bn,
            channels_last,
            i64::try_from(options.window_size)?,
            acceleration,
        )?;
        Ok(Self {
            model,
            metadata,
            profile,
            options,
            device,
        })
    }

    pub fn rate(&self) -> u32 {
        OUTPUT_RATE
    }

    pub fn channels(&self) -> usize {
        2
    }

    pub fn stems(&self) -> [&'static str; 2] {
        [self.metadata.primary, self.metadata.secondary]
    }

    pub(super) fn separate_planar(
        &self,
        pcm: &[f32],
        input_rate: u32,
        mut progress: impl FnMut(Progress) -> ControlFlow<()>,
    ) -> Result<[Stem; 2]> {
        ensure!(
            input_rate > 0
                && !pcm.is_empty()
                && pcm.len().is_multiple_of(2)
                && pcm.iter().all(|sample| sample.is_finite()),
            "VR expects nonempty, finite stereo PCM in channel-major order"
        );
        tch::no_grad(|| {
            let spectral_device = if matches!(self.device, Device::Cuda(_)) {
                self.device
            } else {
                Device::Cpu
            };
            let input = Tensor::f_from_slice(pcm)?
                .f_reshape([2, -1])?
                .f_to_device(spectral_device)?;
            let modern = self.model.modern();
            let analysis = dsp::analysis(
                &input,
                input_rate,
                &self.profile,
                modern,
                self.options.high_end_process,
                self.device == Device::Mps,
            )?;
            let mut mask = self.estimate_mask(&analysis.combined, &mut progress)?;
            mask = self.adjust_aggression(mask)?;
            if self.options.enable_post_process {
                mask = merge_artifacts(&mask, self.options.post_process_threshold)?;
            }
            let primary = finite_spectrum(mask.f_mul(&analysis.combined)?)?;
            let secondary =
                finite_spectrum(mask.f_neg()?.f_add_scalar(1.0)?.f_mul(&analysis.combined)?)?;
            let secondary =
                dsp::synthesis(&secondary, &analysis, &self.profile, modern, self.rate())?;
            if self.metadata.secondary == "Aspiration" {
                let left = secondary.f_select(0, 0)?;
                let right = secondary.f_select(0, 1)?;
                return Ok([
                    output(self.metadata.primary, 1, &right.f_sub(&left)?)?,
                    output(self.metadata.secondary, 1, &left)?,
                ]);
            }
            let primary = dsp::synthesis(&primary, &analysis, &self.profile, modern, self.rate())?;
            Ok([
                output(self.metadata.primary, 2, &primary)?,
                output(self.metadata.secondary, 2, &secondary)?,
            ])
        })
    }

    fn estimate_mask(
        &self,
        spectrum: &Tensor,
        progress: &mut impl FnMut(Progress) -> ControlFlow<()>,
    ) -> Result<Tensor> {
        let mut magnitude = spectrum.f_abs()?;
        let peak = magnitude.f_max()?.f_double_value(&[])?;
        if peak > 0.0 {
            magnitude = magnitude.f_div_scalar(peak)?;
        }
        let frames = magnitude.size()[2];
        let geometry = Window {
            window: self.options.window_size as i64,
            offset: self.model.offset,
        };
        let shift = geometry.roi() / 2;
        let patches = geometry.patches(frames, 0);
        let total = patches
            + if self.options.enable_tta {
                geometry.patches(frames, shift)
            } else {
                0
            };
        let mut reporter = Reporter {
            total: total as usize,
            done: 0,
            progress,
        };
        reporter.emit()?;
        let first = self
            .windows(&magnitude, geometry, 0, &mut reporter)?
            .f_narrow(2, 0, frames)?;
        if self.options.enable_tta {
            let second = self
                .windows(&magnitude, geometry, shift, &mut reporter)?
                .f_narrow(2, shift, frames)?;
            Ok(first.f_add(&second)?.f_mul_scalar(0.5)?)
        } else {
            Ok(first)
        }
    }

    fn windows(
        &self,
        magnitude: &Tensor,
        geometry: Window,
        shift: i64,
        reporter: &mut Reporter<impl FnMut(Progress) -> ControlFlow<()>>,
    ) -> Result<Tensor> {
        let frames = magnitude.size()[2];
        let roi = geometry.roi();
        let count = geometry.patches(frames, shift);
        let padded = magnitude.f_constant_pad_nd([
            geometry.offset + shift,
            roi - frames % roi + geometry.offset + shift,
        ])?;
        let cuda = matches!(self.device, Device::Cuda(_));
        let device = if self.model.host_input() {
            Device::Cpu
        } else {
            self.device
        };
        let mut batches = Vec::new();
        for start in (0..count).step_by(self.options.batch_size) {
            let end = start + (count - start).min(self.options.batch_size as i64);
            let windows = (start..end)
                .map(|i| padded.f_narrow(2, i * roi, geometry.window))
                .collect::<tch::Result<Vec<_>>>()?;
            let mut input = Tensor::f_stack(&windows, 0)?.f_to_device(device)?;
            if cuda && self.options.use_channels_last {
                input = network::channels_last(&input)?;
            }
            let mask = tch::autocast(cuda && self.options.use_amp, || {
                self.model.predict_mask(&input)
            })?
            .f_to_kind(Kind::Float)?
            .f_to_device(magnitude.device())?
            .f_permute([1, 2, 0, 3])?
            .f_reshape([2, self.profile.bins + 1, (end - start) * roi])?;
            batches.push(mask);
            reporter.done += (end - start) as usize;
            reporter.emit()?;
        }
        Ok(Tensor::f_cat(&batches, 2)?)
    }

    fn adjust_aggression(&self, mask: Tensor) -> Result<Tensor> {
        let mut aggression = self.options.aggression as f32 / 50.0;
        if aggression == 0.0 {
            return Ok(mask);
        }
        if matches!(
            self.metadata.primary,
            "Vocals"
                | "Other"
                | "Bass"
                | "Drums"
                | "Guitar"
                | "Piano"
                | "Synthesizer"
                | "Strings"
                | "Woodwinds"
                | "Brass"
                | "Wind Inst"
        ) {
            aggression = 1.0 - aggression;
        }
        let corrections = self
            .profile
            .aggr_correction
            .as_ref()
            .map_or([0.0, 0.0], |correction| [correction.left, correction.right]);
        let bins = mask.size()[1];
        let split = self.profile.bands[&1].crop_stop.min(bins);
        let powers: Vec<_> = corrections
            .into_iter()
            .flat_map(|correction| {
                let amount = aggression + correction;
                (0..bins).map(move |index| 1.0 + if index < split { amount / 3.0 } else { amount })
            })
            .collect();
        let exponent = Tensor::f_from_slice(&powers)?
            .f_reshape([2, bins, 1])?
            .f_to_device(mask.device())?;
        Ok(mask.f_pow(&exponent)?)
    }
}

#[derive(Clone, Copy)]
struct Window {
    window: i64,
    offset: i64,
}

impl Window {
    fn roi(self) -> i64 {
        self.window - 2 * self.offset
    }

    fn patches(self, frames: i64, shift: i64) -> i64 {
        (frames + self.roi() - frames % self.roi() + 2 * shift) / self.roi()
    }
}

struct Reporter<'a, F> {
    total: usize,
    done: usize,
    progress: &'a mut F,
}

impl<F: FnMut(Progress) -> ControlFlow<()>> Reporter<'_, F> {
    fn emit(&mut self) -> Result<()> {
        ensure!(
            (self.progress)(Progress {
                chunks_done: self.done,
                chunks_total: self.total
            })
            .is_continue(),
            "VR separation cancelled"
        );
        Ok(())
    }
}

fn finite_spectrum(spectrum: Tensor) -> Result<Tensor> {
    Ok(spectrum
        .f_view_as_real()?
        .f_nan_to_num(0.0, 0.0, 0.0)?
        .f_view_as_complex()?)
}

fn output(name: &'static str, channels: usize, wave: &Tensor) -> Result<Stem> {
    let samples = Vec::<f32>::try_from(
        wave.f_to_device(Device::Cpu)?
            .f_to_kind(Kind::Float)?
            .f_reshape([-1])?,
    )?;
    ensure!(
        samples.iter().all(|x| x.is_finite()),
        "VR reconstruction produced non-finite audio"
    );
    Ok(Stem {
        name,
        channels,
        samples,
    })
}

fn merge_artifacts(mask: &Tensor, threshold: f32) -> Result<Tensor> {
    const FADE: usize = 32;
    let minima = mask.f_amin([0, 1], false)?.f_to_device(Device::Cpu)?;
    let minima = Vec::<f32>::try_from(minima)?;
    let mut weights = vec![0.0f32; minima.len()];
    let mut start: Option<usize> = None;
    let mut prior_end: Option<usize> = None;
    for (frame, value) in minima
        .into_iter()
        .chain(std::iter::once(f32::NEG_INFINITY))
        .enumerate()
    {
        if value > threshold {
            start.get_or_insert(frame);
        } else if let Some(mut begin) = start.take() {
            let end = frame - 1;
            if end - begin <= 2 * FADE {
                continue;
            }
            if let Some(prior) = prior_end.filter(|&prior| begin - prior < FADE) {
                begin = prior - 2 * FADE;
            }
            let plateau = if begin == 0 {
                0
            } else {
                for (index, weight) in weights[begin..begin + FADE].iter_mut().enumerate() {
                    *weight = index as f32 / (FADE - 1) as f32;
                }
                begin + FADE
            };
            weights[plateau..end - FADE].fill(1.0);
            for (index, weight) in weights[end - FADE..end].iter_mut().enumerate() {
                *weight = 1.0 - index as f32 / (FADE - 1) as f32;
            }
            prior_end = Some(end);
        }
    }
    let weights = Tensor::f_from_slice(&weights)?
        .f_to_device(mask.device())?
        .f_reshape([1, 1, -1])?;
    Ok(mask.f_add(&weights.f_mul(&mask.f_neg()?.f_add_scalar(1.0)?)?)?)
}
