//! Source high-band envelope conditioning for the content encoder.
//!
//! HuBERT-style content features are computed from 16 kHz audio and so carry
//! nothing above 8 kHz, where sibilants live. Without it the prior has to
//! guess the color of every "s", "sh" and "ch" from an average, which blurs
//! them together and gives fricatives an over-sharp, generic hiss. These
//! features describe the spectral shape of the source between 2 and 16 kHz
//! and a local level, 100 times per second, and enter the content encoder
//! through `enc_p.emb_bands`, initialized to zero so an older generator starts
//! out unchanged.

use anyhow::{Result, ensure};
use tch::{Kind, Tensor};

/// Band edges in Hz: eight shape bands, then the level channel.
const EDGES: [f64; 9] = [
    2_000.0, 3_000.0, 4_000.0, 5_000.0, 6_300.0, 8_000.0, 10_000.0, 12_500.0, 16_000.0,
];
pub const COUNT: i64 = EDGES.len() as i64;
pub const WEIGHT: &str = "enc_p.emb_bands.weight";
/// Frames over which the level channel finds its local peak (±1 s).
const LEVEL_SPAN: i64 = 201;

/// Features from a linear amplitude spectrogram [batch, bins, frames] of
/// audio at `rate`, as [batch, frames, COUNT].
pub fn from_spectrogram(spec: &Tensor, rate: u32) -> Result<Tensor> {
    let (batch, bins, frames) = spec.size3()?;
    ensure!(bins > 8 && frames > 0, "invalid spectrogram for band features");
    let hz_per_bin = f64::from(rate) / (2.0 * (bins - 1) as f64);
    let bin = |hz: f64| ((hz / hz_per_bin).round() as i64).clamp(1, bins - 1);
    let power = spec.to_kind(Kind::Float).square();
    let sum = |low: i64, high: i64| -> Tensor {
        power
            .narrow(1, low, (high - low).max(1))
            .sum_dim_intlist([1].as_slice(), false, Kind::Float)
    };
    let total = sum(bin(50.0), bin(16_000.0)) + 1e-9;
    let mut channels = Vec::with_capacity(COUNT as usize);
    for pair in EDGES.windows(2) {
        let ratio = sum(bin(pair[0]), bin(pair[1])) / &total;
        // log10 of the band's share, mapped from about [-5, 0] to [-1.7, 1.7].
        channels.push((ratio + 1e-5).log10().clamp(-5.0, 0.0) * (1.0 / 1.5) + 1.7);
    }
    let level = total.log10() * 10.0;
    let peak = level
        .unsqueeze(1)
        .max_pool1d([LEVEL_SPAN], [1], [LEVEL_SPAN / 2], [1], false)
        .squeeze_dim(1);
    channels.push(((level - peak) / 20.0).clamp(-3.0, 0.0) + 1.5);
    let features = Tensor::stack(&channels, 2);
    debug_assert_eq!(features.size(), [batch, frames, COUNT]);
    Ok(features)
}

/// Share of training examples whose bands are computed as if the source
/// stopped at 8 kHz (16 kHz audio, realtime input, low-bitrate MP3), so the
/// model falls back gracefully when a recording has no high band.
const BAND_LIMITED_SHARE: f64 = 0.2;
const BAND_LIMIT_HZ: f64 = 8_000.0;

/// [`from_spectrogram`] for a training batch, band-limiting a random share of examples.
pub fn with_band_limited_examples(spec: &Tensor, rate: u32) -> Result<Tensor> {
    let (batch, bins, _) = spec.size3()?;
    let hz_per_bin = f64::from(rate) / (2.0 * (bins - 1) as f64);
    let cut = ((BAND_LIMIT_HZ / hz_per_bin).round() as i64).clamp(1, bins);
    let options = (spec.kind(), spec.device());
    let lowpass = Tensor::ones([1, cut, 1], options)
        .constant_pad_nd([0, 0, 0, bins - cut]);
    let limited = Tensor::rand([batch, 1, 1], (Kind::Float, spec.device()))
        .lt(BAND_LIMITED_SHARE)
        .to_kind(spec.kind());
    let keep = (lowpass - 1.0) * &limited + 1.0;
    // Leave the spectrogram's own floor (sqrt of its 2e-7 power offset) where the
    // band was removed, as a resampled recording would read.
    from_spectrogram(&(spec * &keep + (1.0 - &keep) * 2e-7_f64.sqrt()), rate)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::Device;

    #[test]
    fn hiss_and_hum_have_opposite_shapes() {
        let rate = 40_000;
        let samples = 40_000;
        let t = Tensor::arange(samples, (Kind::Float, Device::Cpu)) / f64::from(rate);
        let hum = (&t * std::f64::consts::TAU * 220.0).sin();
        tch::manual_seed(0);
        let hiss = Tensor::randn([samples], (Kind::Float, Device::Cpu));
        let config = crate::model::train_losses::MelConfig {
            fft: 2048,
            hop: 400,
            window: 2048,
            sample_rate: rate,
            mel_bins: 80,
            minimum: 0.0,
            maximum: None,
        };
        let mel = crate::model::train_losses::MelSpectrogram::new(config, Device::Cpu).unwrap();
        let shape = |audio: Tensor| {
            let spec = mel.spectrogram(&audio.unsqueeze(0)).unwrap();
            let bands = from_spectrogram(&spec, rate).unwrap();
            assert_eq!(bands.size()[2], COUNT);
            // Highest shape band, averaged over frames.
            bands.select(2, 7).mean(Kind::Float).double_value(&[])
        };
        assert!(shape(hiss) > shape(hum) + 1.0);
    }
}
