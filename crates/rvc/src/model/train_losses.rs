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
        let (batch, samples) = waveform.size2()?;
        let MelConfig {
            fft, hop, window, ..
        } = self.config;
        let pad = (fft - hop) / 2;
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
            .f_add_scalar(2e-7)?
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
