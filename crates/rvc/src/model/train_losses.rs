//! Losses and differentiable spectral transforms used by RVC training.

use anyhow::{Result, ensure};
use tch::{Device, Kind, Tensor};

pub fn discriminator_loss(real: &[Tensor], generated: &[Tensor]) -> Result<Tensor> {
    ensure!(
        !real.is_empty() && real.len() == generated.len(),
        "invalid discriminator scores"
    );
    let mut loss = Tensor::f_zeros([], (Kind::Float, real[0].device()))?;
    for (real, generated) in real.iter().zip(generated) {
        let real_loss = squared_distance_from_one(real)?;
        let generated_loss = generated
            .f_to_kind(Kind::Float)?
            .f_square()?
            .f_mean(Kind::Float)?;
        loss = loss.f_add(&real_loss.f_add(&generated_loss)?)?;
    }
    Ok(loss)
}

pub fn generator_loss(generated: &[Tensor]) -> Result<Tensor> {
    ensure!(!generated.is_empty(), "missing generator scores");
    let mut loss = Tensor::f_zeros([], (Kind::Float, generated[0].device()))?;
    for score in generated {
        loss = loss.f_add(&squared_distance_from_one(score)?)?;
    }
    Ok(loss)
}

fn squared_distance_from_one(score: &Tensor) -> Result<Tensor> {
    Ok(score
        .f_to_kind(Kind::Float)?
        .f_neg()?
        .f_add_scalar(1.0)?
        .f_square()?
        .f_mean(Kind::Float)?)
}

pub fn feature_loss(real: &[Vec<Tensor>], generated: &[Vec<Tensor>]) -> Result<Tensor> {
    ensure!(
        !generated.is_empty() && generated.len() == real.len(),
        "invalid feature-map groups"
    );
    let first = generated[0]
        .first()
        .ok_or_else(|| anyhow::anyhow!("missing feature maps"))?;
    let mut loss = Tensor::f_zeros([], (Kind::Float, first.device()))?;
    for (real_branch, generated_branch) in real.iter().zip(generated) {
        ensure!(
            real_branch.len() == generated_branch.len(),
            "incompatible feature-map groups"
        );
        for (real_map, generated_map) in real_branch.iter().zip(generated_branch) {
            let real_map = real_map.f_to_kind(Kind::Float)?.f_detach()?;
            let generated_map = generated_map.f_to_kind(Kind::Float)?;
            let difference = real_map
                .f_sub(&generated_map)?
                .f_abs()?
                .f_mean(Kind::Float)?;
            loss = loss.f_add(&difference)?;
        }
    }
    Ok(loss.f_mul_scalar(2.0)?)
}

pub fn kl_loss(
    z_p: &Tensor,
    logs_q: &Tensor,
    m_p: &Tensor,
    logs_p: &Tensor,
    mask: &Tensor,
) -> Result<Tensor> {
    let z_p = z_p.f_to_kind(Kind::Float)?;
    let logs_q = logs_q.f_to_kind(Kind::Float)?;
    let m_p = m_p.f_to_kind(Kind::Float)?;
    let logs_p = logs_p.f_to_kind(Kind::Float)?;
    let mask = mask.f_to_kind(Kind::Float)?;
    let variance_precision = logs_p.f_mul_scalar(-2.0)?.f_exp()?;
    let error = z_p
        .f_sub(&m_p)?
        .f_square()?
        .f_mul_scalar(0.5)?
        .f_mul(&variance_precision)?;
    let kl = logs_p.f_sub(&logs_q)?.f_sub_scalar(0.5)?.f_add(&error)?;
    Ok(kl
        .f_mul(&mask)?
        .f_sum(Kind::Float)?
        .f_div(&mask.f_sum(Kind::Float)?)?)
}

#[derive(Debug, Clone, Copy)]
pub struct MelConfig {
    pub fft: i64,
    pub hop: i64,
    pub window: i64,
    pub sample_rate: u32,
    pub mel_bins: i64,
    pub minimum: f64,
    pub maximum: Option<f64>,
}

pub struct MelSpectrogram {
    config: MelConfig,
    basis: Tensor,
    window: Tensor,
}

impl MelSpectrogram {
    pub fn new(config: MelConfig, device: Device) -> Result<Self> {
        let maximum = config
            .maximum
            .unwrap_or(f64::from(config.sample_rate) / 2.0);
        ensure!(
            config.sample_rate > 0
                && config.fft > 1
                && (1..=config.fft).contains(&config.window)
                && (1..=config.fft).contains(&config.hop)
                && config.mel_bins > 0
                && config.minimum.is_finite()
                && config.minimum >= 0.0
                && maximum.is_finite()
                && maximum > config.minimum,
            "invalid training mel spectrogram configuration"
        );
        let values = mel_filterbank(config, maximum);
        let basis = Tensor::f_from_slice(&values)?
            .f_reshape([config.mel_bins, config.fft / 2 + 1])?
            .f_to_device(device)?;
        let spectral_device = if device == Device::Mps {
            Device::Cpu
        } else {
            device
        };
        let window =
            Tensor::f_hann_window_periodic(config.window, true, (Kind::Float, spectral_device))?;
        Ok(Self {
            config,
            basis,
            window,
        })
    }

    /// Linear-frequency amplitude spectrogram from [batch, samples] waveforms.
    pub fn spectrogram(&self, waveform: &Tensor) -> Result<Tensor> {
        self.spectrogram_padded(waveform, false)
    }

    fn spectrogram_padded(&self, waveform: &Tensor, centered: bool) -> Result<Tensor> {
        let (batch, samples) = waveform.size2()?;
        let MelConfig {
            fft, hop, window, ..
        } = self.config;
        let pad = if centered { fft / 2 } else { (fft - hop) / 2 };
        ensure!(
            batch > 0 && samples > pad,
            "waveform is too short for reflection padding"
        );

        let kind = waveform.kind();
        let waveform = waveform.f_to_device(self.window.device())?;
        let waveform =
            if waveform.device() == Device::Cpu && matches!(kind, Kind::Half | Kind::BFloat16) {
                waveform.f_to_kind(Kind::Float)?
            } else {
                waveform
            };
        let window_tensor = self.window.f_to_kind(waveform.kind())?;
        let spectrum = waveform.f_reflection_pad1d([pad, pad])?.f_stft_center(
            fft,
            hop,
            window,
            Some(&window_tensor),
            false,
            "reflect",
            false,
            true,
            true,
            false,
        )?;
        let real = spectrum.f_real()?;
        let imaginary = spectrum.f_imag()?;
        let magnitude = real
            .f_square()?
            .f_add(&imaginary.f_square()?)?
            .f_add_scalar(if centered { 1e-6 } else { 2e-7 })?
            .f_sqrt()?;
        Ok(magnitude
            .f_to_device(self.basis.device())?
            .f_to_kind(kind)?)
    }

    pub fn spec_to_mel(&self, spectrum: &Tensor) -> Result<Tensor> {
        let basis = self
            .basis
            .f_to_device(spectrum.device())?
            .f_to_kind(spectrum.kind())?;
        Ok(basis.f_matmul(spectrum)?.f_clamp_min(2e-6)?.f_log()?)
    }

    pub fn mel_spectrogram(&self, waveform: &Tensor) -> Result<Tensor> {
        self.spec_to_mel(&self.spectrogram(waveform)?)
    }
}

/// NVC's optional multi-scale log-mel loss.
pub(super) struct MultiScaleMel(Vec<MelSpectrogram>);

impl MultiScaleMel {
    pub(super) fn new(rate: u32, device: Device) -> Result<Self> {
        let transforms = [32, 64, 128, 256, 512, 1024, 2048]
            .into_iter()
            .zip([5, 10, 20, 40, 80, 160, 320])
            .map(|(fft, mel_bins)| {
                MelSpectrogram::new(
                    MelConfig {
                        fft,
                        hop: fft / 4,
                        window: fft,
                        sample_rate: rate,
                        mel_bins,
                        minimum: 0.0,
                        maximum: None,
                    },
                    device,
                )
            })
            .collect::<Result<_>>()?;
        Ok(Self(transforms))
    }

    pub(super) fn loss(&self, real: &Tensor, fake: &Tensor) -> Result<Tensor> {
        let real = real.f_squeeze_dim(1)?.f_to_kind(Kind::Float)?;
        let fake = fake.f_squeeze_dim(1)?.f_to_kind(Kind::Float)?;
        let mut loss = Tensor::f_zeros([], (Kind::Float, fake.device()))?;
        for mel in &self.0 {
            let logmel = |waveform: &Tensor| -> Result<Tensor> {
                let amplitude = mel.spectrogram_padded(waveform, true)?;
                Ok(mel
                    .basis
                    .f_matmul(&amplitude)?
                    .f_clamp_min(1e-5)?
                    .f_log10()?)
            };
            let desired = tch::no_grad(|| logmel(&real))?;
            let predicted = logmel(&fake)?;
            loss = loss.f_add(&desired.f_l1_loss(&predicted, tch::Reduction::Mean)?)?;
        }
        Ok(loss)
    }
}

/// Multi-resolution STFT loss (spectral convergence plus log-magnitude L1).
///
/// Linear-frequency magnitudes weight the upper bands as much as the lower ones,
/// unlike the 125-band mel loss, so this term targets high-frequency detail.
pub(super) struct MultiResolutionStft(Vec<(i64, i64, Tensor)>);

impl MultiResolutionStft {
    pub(super) fn new(device: Device) -> Result<Self> {
        let spectral_device = if device == Device::Mps {
            Device::Cpu
        } else {
            device
        };
        let resolutions = [(256, 64), (512, 128), (1024, 256), (2048, 512)]
            .into_iter()
            .map(|(fft, hop)| {
                let window =
                    Tensor::f_hann_window_periodic(fft, true, (Kind::Float, spectral_device))?;
                Ok((fft, hop, window))
            })
            .collect::<Result<_>>()?;
        Ok(Self(resolutions))
    }

    /// Both inputs use [batch, 1, samples] layout.
    pub(super) fn loss(&self, real: &Tensor, fake: &Tensor) -> Result<Tensor> {
        let real = real.f_squeeze_dim(1)?.f_to_kind(Kind::Float)?;
        let fake = fake.f_squeeze_dim(1)?.f_to_kind(Kind::Float)?;
        let mut total = Tensor::f_zeros([], (Kind::Float, fake.device()))?;
        for (fft, hop, window) in &self.0 {
            let magnitude = |waveform: &Tensor| -> Result<Tensor> {
                let waveform = waveform
                    .f_to_device(window.device())?
                    .f_reflection_pad1d([fft / 2, fft / 2])?;
                let spectrum = waveform.f_stft_center(
                    *fft,
                    *hop,
                    *fft,
                    Some(window),
                    false,
                    "reflect",
                    false,
                    true,
                    true,
                    false,
                )?;
                Ok(spectrum
                    .f_real()?
                    .f_square()?
                    .f_add(&spectrum.f_imag()?.f_square()?)?
                    .f_clamp_min(1e-7)?
                    .f_sqrt()?
                    .f_to_device(fake.device())?)
            };
            let desired = tch::no_grad(|| magnitude(&real))?;
            let predicted = magnitude(&fake)?;
            let convergence = desired
                .f_sub(&predicted)?
                .f_square()?
                .f_sum(Kind::Float)?
                .f_sqrt()?
                .f_div(
                    &desired
                        .f_square()?
                        .f_sum(Kind::Float)?
                        .f_sqrt()?
                        .f_clamp_min(1e-7)?,
                )?;
            let log_magnitude = desired
                .f_log()?
                .f_l1_loss(&predicted.f_log()?, tch::Reduction::Mean)?;
            total = total.f_add(&convergence.f_add(&log_magnitude)?)?;
        }
        Ok(total.f_div_scalar(self.0.len() as f64)?)
    }
}

fn mel_filterbank(config: MelConfig, maximum: f64) -> Vec<f32> {
    let scale = 200.0 / 3.0;
    let logarithmic_start = 1000.0 / scale;
    let logarithmic_step = 6.4_f64.ln() / 27.0;
    let hz_to_mel = |hz: f64| {
        if hz >= 1000.0 {
            logarithmic_start + (hz / 1000.0).ln() / logarithmic_step
        } else {
            hz / scale
        }
    };
    let mel_to_hz = |mel: f64| {
        if mel >= logarithmic_start {
            1000.0 * (logarithmic_step * (mel - logarithmic_start)).exp()
        } else {
            mel * scale
        }
    };
    let first = hz_to_mel(config.minimum);
    let last = hz_to_mel(maximum);
    let delta = (last - first) / (config.mel_bins + 1) as f64;
    let edges: Vec<_> = (0..config.mel_bins + 2)
        .map(|i| mel_to_hz(first + i as f64 * delta))
        .collect();
    let bin_rate = f64::from(config.sample_rate) / config.fft as f64;
    edges
        .windows(3)
        .flat_map(|band| {
            let normalize = 2.0 / (band[2] - band[0]);
            (0..=config.fft / 2).map(move |i| {
                let frequency = i as f64 * bin_rate;
                let lower = (frequency - band[0]) / (band[1] - band[0]);
                let upper = (band[2] - frequency) / (band[2] - band[1]);
                let weight = lower.min(upper).max(0.0) as f32;
                (f64::from(weight) * normalize) as f32
            })
        })
        .collect()
}

/// Phonetic (content) loss: L1 distance between frozen HuBERT-family features of
/// real and generated audio.
///
/// The mel and STFT losses weigh every frequency bin by energy, so a blurred
/// consonant or a vowel drifting towards its neighbour costs little. A
/// self-supervised speech encoder separates exactly those phonetic contrasts.
/// With a multilingual encoder such as mHuBERT-147 the loss also covers
/// languages that the English HuBERT used for content features hears poorly.
pub(super) struct PhoneticLoss {
    weights: super::weights::Weights,
    layers: super::hubert::Layers,
    resample: Option<SincResample>,
    normalize: bool,
    taps: Vec<usize>,
}

impl PhoneticLoss {
    /// Encoder layers (zero-based) whose outputs are compared.
    const TAPS: [usize; 3] = [5, 8, 11];

    pub(super) fn new(
        directory: &std::path::Path,
        sample_rate: i64,
        device: Device,
        kind: Kind,
    ) -> Result<Self> {
        let binary = directory.join("pytorch_model.bin");
        let checkpoint = if binary.is_file() {
            binary
        } else {
            directory.join("model.safetensors")
        };
        ensure!(
            checkpoint.is_file(),
            "phonetic loss encoder not found in {}",
            directory.display()
        );
        let mut weights = super::hubert::load_weights(&checkpoint)?;
        weights.move_to(device, kind)?;
        Ok(Self {
            weights,
            layers: super::hubert::layers(directory)?,
            resample: (sample_rate != 16_000)
                .then(|| SincResample::new(sample_rate, 16_000, device))
                .transpose()?,
            normalize: super::hubert::needs_normalization(directory)?,
            taps: Self::TAPS.to_vec(),
        })
    }

    fn features(&self, audio: &Tensor) -> Result<Vec<Tensor>> {
        let mut audio = audio.f_to_kind(Kind::Float)?;
        if let Some(resample) = &self.resample {
            audio = resample.apply(&audio)?;
        }
        if self.normalize {
            let mean = audio.f_mean_dim(-1, true, Kind::Float)?;
            let centered = audio.f_sub(&mean)?;
            let variance = centered.f_square()?.f_mean_dim(-1, true, Kind::Float)?;
            audio = centered.f_div(&variance.f_add_scalar(1e-5)?.f_sqrt()?)?;
        }
        let last = *self.taps.last().expect("phonetic taps");
        let mut outputs = Vec::with_capacity(self.taps.len());
        drop(self.layers.encode(
            &self.weights,
            audio,
            last + 1,
            |layer, hidden| {
                if self.taps.contains(&layer) {
                    outputs.push(hidden.shallow_clone());
                }
                Ok(())
            },
        )?);
        Ok(outputs)
    }

    /// Both inputs use [batch, 1, samples] layout at the training sample rate.
    pub(super) fn loss(&self, real: &Tensor, fake: &Tensor) -> Result<Tensor> {
        let desired = tch::no_grad(|| self.features(real))?;
        let predicted = self.features(fake)?;
        let mut total = Tensor::f_zeros([], (Kind::Float, fake.device()))?;
        for (desired, predicted) in desired.iter().zip(&predicted) {
            total = total.f_add(
                &desired
                    .f_to_kind(Kind::Float)?
                    .f_l1_loss(&predicted.f_to_kind(Kind::Float)?, tch::Reduction::Mean)?,
            )?;
        }
        Ok(total.f_div_scalar(desired.len() as f64)?)
    }
}

/// Differentiable band-limited resampler (Hann-windowed sinc, as in torchaudio).
pub(super) struct SincResample {
    kernel: Tensor,
    orig: i64,
    new: i64,
    width: i64,
}

impl SincResample {
    pub(super) fn new(from: i64, to: i64, device: Device) -> Result<Self> {
        ensure!(from > 0 && to > 0, "invalid resampling rates");
        let divisor = gcd(from, to);
        let (orig, new) = (from / divisor, to / divisor);
        let filter_width = 6.0;
        let base = orig.min(new) as f64 * 0.99;
        let width = (filter_width * orig as f64 / base).ceil() as i64;
        let taps = 2 * width + orig;
        let mut values = Vec::with_capacity((new * taps) as usize);
        for phase in 0..new {
            for index in 0..taps {
                let position = (index - width) as f64 / orig as f64 - phase as f64 / new as f64;
                let t = (position * base).clamp(-filter_width, filter_width);
                let window = (t * std::f64::consts::PI / filter_width / 2.0)
                    .cos()
                    .powi(2);
                let t = t * std::f64::consts::PI;
                let sinc = if t == 0.0 { 1.0 } else { t.sin() / t };
                values.push((sinc * window * base / orig as f64) as f32);
            }
        }
        let kernel = Tensor::f_from_slice(&values)?
            .f_view([new, 1, taps])?
            .f_to_device(device)?;
        Ok(Self {
            kernel,
            orig,
            new,
            width,
        })
    }

    /// Resamples [batch, 1, samples] float audio.
    pub(super) fn apply(&self, audio: &Tensor) -> Result<Tensor> {
        let size = audio.size();
        let (batch, length) = (size[0], size[size.len() - 1]);
        let padded = audio
            .f_view([batch, 1, length])?
            .f_constant_pad_nd([self.width, self.width + self.orig])?;
        let output = padded
            .f_conv1d(&self.kernel, None::<&Tensor>, [self.orig], [0], [1], 1)?
            .f_transpose(1, 2)?
            .f_reshape([batch, 1, -1])?;
        let target = (self.new * length + self.orig - 1) / self.orig;
        Ok(output.f_narrow(2, 0, target)?)
    }
}

fn gcd(mut a: i64, mut b: i64) -> i64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

#[cfg(test)]
mod phonetic_tests {
    use super::*;

    #[test]
    fn sinc_resample_preserves_a_low_tone() -> Result<()> {
        let resample = SincResample::new(40_000, 16_000, Device::Cpu)?;
        let input: Vec<f32> = (0..8_000)
            .map(|n| (2.0 * std::f32::consts::PI * 440.0 * n as f32 / 40_000.0).sin())
            .collect();
        let output = resample.apply(&Tensor::from_slice(&input).view([1, 1, -1]))?;
        assert_eq!(output.size(), [1, 1, 3_200]);
        let output = Vec::<f32>::try_from(output.flatten(0, -1))?;
        let error = output[200..3_000]
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                let n = (i + 200) as f32;
                (v - (2.0 * std::f32::consts::PI * 440.0 * n / 16_000.0).sin()).abs()
            })
            .fold(0.0f32, f32::max);
        assert!(error < 0.01, "max error {error}");
        Ok(())
    }
}
