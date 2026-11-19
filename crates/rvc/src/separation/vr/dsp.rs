use std::f64::consts::PI;

use anyhow::{Context, Result, ensure};
use tch::{Device, Kind, Tensor};

use super::profiles::{Band, Channels, Profile, Resampler};
use super::soxr::Soxr;

pub(super) struct Spectrogram {
    pub combined: Tensor,
    high_end: Option<Tensor>,
}

pub(super) fn analysis(
    wave: &Tensor,
    rate: u32,
    profile: &Profile,
    modern: bool,
    high_end: bool,
    mps: bool,
) -> Result<Spectrogram> {
    let mut spectra = Vec::with_capacity(profile.bands.len());
    let mut wave = wave.shallow_clone();
    let mut current_rate = rate;
    for band in profile.bands.values().rev() {
        let mode = if mps {
            Resampler::Polyphase
        } else {
            band.resampler
        };
        wave = resample(&wave, current_rate, band.rate, mode)?;
        current_rate = band.rate;
        let input = if modern {
            wave.shallow_clone()
        } else {
            transform_channels(&wave, profile.legacy_channels(), false)?
        };
        let mut spec = stft(&input, *band)?;
        if modern {
            spec = transform_channels(&spec, band.channels, false)?;
        }
        spectra.push((spec, *band));
    }
    let frames = spectra
        .iter()
        .map(|(spec, _)| spec.size()[2])
        .min()
        .context("VR separation has no spectra")?;
    let most_high = if high_end {
        let (spectrum, band) = &spectra[0];
        let width =
            band.fft / 2 - band.crop_stop + profile.pre_filter_stop - profile.pre_filter_start;
        ensure!(
            width >= 0 && width <= band.fft / 2,
            "invalid VR high-frequency restoration width"
        );
        (width > 0)
            .then(|| {
                spectrum
                    .f_narrow(1, band.fft / 2 - width, width)?
                    .f_narrow(2, 0, frames)
            })
            .transpose()?
    } else {
        None
    };
    let mut crops = Vec::with_capacity(spectra.len() + 1);
    let mut cursor = 0;
    for (spec, band) in spectra.into_iter().rev() {
        let width = band.crop_stop - band.crop_start;
        crops.push(
            spec.f_narrow(1, band.crop_start, width)?
                .f_narrow(2, 0, frames)?,
        );
        cursor += width;
    }
    ensure!(
        cursor <= profile.bins,
        "VR band slices exceed the configured bandwidth"
    );
    crops.push(Tensor::f_zeros(
        [2, profile.bins + 1 - cursor, frames],
        (Kind::ComplexFloat, wave.device()),
    )?);
    let combined = pre_filter(Tensor::f_cat(&crops, 1)?, profile, modern)?;
    Ok(Spectrogram {
        combined,
        high_end: most_high,
    })
}

pub(super) fn synthesis(
    spectrum: &Tensor,
    analysis: &Spectrogram,
    profile: &Profile,
    modern: bool,
    target_rate: u32,
) -> Result<Tensor> {
    let frames = spectrum.size()[2];
    let extra = analysis
        .high_end
        .as_ref()
        .map(|input| mirror(spectrum, input, profile))
        .transpose()?;
    let mut wave: Option<Tensor> = None;
    let mut offset = 0;
    let mut bands = profile.bands.values().peekable();
    let mode = if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        Resampler::Polyphase
    } else {
        Resampler::SoxrHq
    };
    while let Some(band) = bands.next() {
        let highest = bands.peek().is_none();
        let width = band.crop_stop - band.crop_start;
        let combined = spectrum.f_narrow(1, offset, width)?;
        let bins = band.fft / 2 + 1;
        let mut local =
            combined.f_constant_pad_nd([0, 0, band.crop_start, bins - band.crop_stop])?;
        if let Some(high) = extra.as_ref().filter(|_| highest) {
            let count = high.size()[1];
            let mut destination = local.f_narrow(1, band.fft / 2 - count, count)?;
            destination.f_copy_(high)?;
        }
        if (offset != 0 || highest) && band.hpf_start > 0 {
            local = filter_high(local, band.hpf_start, band.hpf_stop - 1)?;
        }
        if !highest && band.lpf_start > 0 {
            local = filter_low(local, band.lpf_start, band.lpf_stop)?;
        }
        let restored = if frames < 2 {
            Tensor::f_zeros([2, 0], (Kind::Float, spectrum.device()))?
        } else {
            istft(&local, *band)?
        };
        let channels = if modern {
            band.channels
        } else {
            profile.legacy_channels()
        };
        let restored = transform_channels(&restored, channels, true)?;
        let mut mixed = match wave {
            Some(previous) => {
                ensure!(
                    previous.size() == restored.size(),
                    "VR bands have different reconstructed lengths"
                );
                previous.f_add(&restored)?
            }
            None => restored,
        };
        if let Some(next) = bands.peek() {
            mixed = resample(&mixed, band.rate, next.rate, mode)?;
        }
        wave = Some(mixed);
        offset += width;
    }
    let wave = wave.context("VR separation has no reconstructed bands")?;
    resample(&wave, profile.rate, target_rate, Resampler::SoxrHq)
}

fn stft(input: &Tensor, band: Band) -> Result<Tensor> {
    let hann = Tensor::f_hann_window_periodic(band.fft, true, (Kind::Float, input.device()))?;
    let radius = band.fft / 2;
    Ok(input.f_constant_pad_nd([radius, radius])?.f_stft_center(
        band.fft,
        band.hop,
        band.fft,
        Some(&hann),
        false,
        "constant",
        false,
        true,
        true,
        false,
    )?)
}

fn istft(spectrum: &Tensor, band: Band) -> Result<Tensor> {
    let hann = Tensor::f_hann_window_periodic(band.fft, true, (Kind::Float, spectrum.device()))?;
    Ok(spectrum.f_istft(
        band.fft,
        band.hop,
        band.fft,
        Some(&hann),
        true,
        false,
        true,
        None,
        false,
    )?)
}

fn transform_channels(input: &Tensor, mode: Channels, inverse: bool) -> Result<Tensor> {
    let left = input.f_select(0, 0)?;
    let right = input.f_select(0, 1)?;
    let pair = match (mode, inverse) {
        (Channels::Identity, _) => return Ok(input.shallow_clone()),
        (Channels::Reverse, _) => return Ok(input.f_flip([-1])?),
        (Channels::MidSide, false) => [left.f_add(&right)?.f_mul_scalar(0.5)?, left.f_sub(&right)?],
        (Channels::MidSide, true) => {
            let side = right.f_mul_scalar(0.5)?;
            [left.f_add(&side)?, left.f_sub(&side)?]
        }
        (Channels::MidSideB2, false) => [
            right.f_add(&left.f_mul_scalar(0.5)?)?,
            left.f_sub(&right.f_mul_scalar(0.5)?)?,
        ],
        (Channels::MidSideB2, true) => [
            right.f_mul_scalar(0.8)?.f_add(&left.f_mul_scalar(0.4)?)?,
            left.f_mul_scalar(0.8)?.f_sub(&right.f_mul_scalar(0.4)?)?,
        ],
        (Channels::MidSideC, false) => [
            left.f_add(&right.f_mul_scalar(0.25)?)?,
            right.f_sub(&left.f_mul_scalar(0.25)?)?,
        ],
        (Channels::MidSideC, true) => [
            left.f_div_scalar(1.0625)?
                .f_sub(&right.f_div_scalar(4.25)?)?,
            right
                .f_div_scalar(1.0625)?
                .f_add(&left.f_div_scalar(4.25)?)?,
        ],
        (Channels::StereoN, false) => [
            left.f_add(&right.f_mul_scalar(0.25)?)?
                .f_div_scalar(0.9375)?,
            right
                .f_add(&left.f_mul_scalar(0.25)?)?
                .f_div_scalar(0.9375)?,
        ],
        (Channels::StereoN, true) => [
            left.f_sub(&right.f_mul_scalar(0.25)?)?,
            right.f_sub(&left.f_mul_scalar(0.25)?)?,
        ],
    };
    Ok(Tensor::f_stack(&pair, 0)?)
}

fn multiply_filter(input: Tensor, coefficients: &[f32]) -> Result<Tensor> {
    let mask = Tensor::f_from_slice(coefficients)?
        .f_to_device(input.device())?
        .f_reshape([1, -1, 1])?;
    Ok(input.f_mul(&mask)?)
}

fn filter_low(input: Tensor, start: i64, end: i64) -> Result<Tensor> {
    let bins = input.size()[1];
    ensure!(
        0 < start && start <= bins && end <= bins,
        "invalid VR low-pass frequencies"
    );
    let coefficients: Vec<_> = (0..bins)
        .map(|i| {
            if i >= end {
                0.0
            } else if i < start || start >= end {
                1.0
            } else {
                (end - i - 1) as f32 / (end - start) as f32
            }
        })
        .collect();
    multiply_filter(input, &coefficients)
}

fn filter_high(input: Tensor, start: i64, end: i64) -> Result<Tensor> {
    let bins = input.size()[1];
    ensure!(
        0 <= end && end < start && start < bins,
        "invalid VR high-pass frequencies"
    );
    let coefficients: Vec<_> = (0..bins)
        .map(|i| ((i - end - 1) as f32 / (start - end) as f32).clamp(0.0, 1.0))
        .collect();
    multiply_filter(input, &coefficients)
}

fn pre_filter(input: Tensor, profile: &Profile, modern: bool) -> Result<Tensor> {
    let start = profile.pre_filter_start;
    let stop = profile.pre_filter_stop;
    if start <= 0 {
        return Ok(input);
    }
    if modern || profile.bands.len() == 1 {
        return filter_low(input, start, stop);
    }
    let mut coefficients = vec![1.0f32; input.size()[1] as usize];
    let mut gain = 1.0;
    for index in start + 1..stop {
        gain = 10f64.powf(-((index - start) as f64) * (3.5 - gain) / 20.0);
        coefficients[index as usize] = gain as f32;
    }
    multiply_filter(input, &coefficients)
}

fn mirror(spectrum: &Tensor, input: &Tensor, profile: &Profile) -> Result<Tensor> {
    let height = input.size()[1];
    let bins = spectrum.size()[1];
    let end = profile.pre_filter_start - 10;
    let end = if end < 0 { bins + end } else { end };
    ensure!(
        end >= height && end <= bins,
        "VR mirror frequencies exceed the combined spectrum"
    );
    let magnitude = spectrum
        .f_narrow(1, end - height, height)?
        .f_abs()?
        .f_flip([1])?;
    let mirrored = Tensor::f_polar(&magnitude, &input.f_angle()?)?;
    let keep = input.f_abs()?.f_le_tensor(&magnitude)?;
    Ok(input.f_where_self(&keep, &mirrored)?)
}

fn resample(wave: &Tensor, from: u32, to: u32, mode: Resampler) -> Result<Tensor> {
    if from == to || wave.size()[1] == 0 {
        return Ok(wave.shallow_clone());
    }
    let (_, samples) = wave.size2()?;
    let device = wave.device();
    let values = Vec::<f32>::try_from(
        wave.f_to_device(Device::Cpu)?
            .f_to_kind(Kind::Float)?
            .f_reshape([-1])?,
    )?;
    let (left, right) = values.split_at(samples as usize);
    // The reference falls back to SoXR HQ for sinc/kaiser modes when their
    // optional Python resamplers are unavailable, and to polyphase without SoXR.
    let resampled =
        if let Some(soxr) = Soxr::get().filter(|_| !matches!(mode, Resampler::Polyphase)) {
            soxr.resample(left, right, from, to)?
        } else {
            let filter = Polyphase::new(from, to)?;
            [filter.run(left), filter.run(right)]
        };
    Ok(Tensor::f_stack(
        &[
            Tensor::f_from_slice(&resampled[0])?,
            Tensor::f_from_slice(&resampled[1])?,
        ],
        0,
    )?
    .f_to_device(device)?)
}

struct Polyphase {
    up: i64,
    down: i64,
    half: i64,
    weights: Vec<f64>,
}

impl Polyphase {
    fn new(from: u32, to: u32) -> Result<Self> {
        let common = gcd(from, to);
        let up = i64::from(to / common);
        let down = i64::from(from / common);
        let period = up.max(down);
        let half = 10 * period;
        let cutoff = 1.0 / period as f64;
        let window_scale = bessel_i0(5.0);
        let mut weights = Vec::new();
        weights
            .try_reserve((2 * half + 1) as usize)
            .context("allocating VR polyphase filter")?;
        weights.extend((-half..=half).map(|i| {
            let relative = i as f64 / half as f64;
            let window =
                bessel_i0(5.0 * (1.0 - relative * relative).max(0.0).sqrt()) / window_scale;
            let angle = PI * cutoff * i as f64;
            let sinc = if i == 0 { 1.0 } else { angle.sin() / angle };
            cutoff * sinc * window
        }));
        let normalization = up as f64 / weights.iter().sum::<f64>();
        weights
            .iter_mut()
            .for_each(|weight| *weight *= normalization);
        Ok(Self {
            up,
            down,
            half,
            weights,
        })
    }

    fn run(&self, input: &[f32]) -> Vec<f32> {
        let len = input.len() as i64;
        let output = (len * self.up + self.down - 1) / self.down;
        (0..output)
            .map(|index| {
                let center = index * self.down;
                let start = ((center - self.half + self.up - 1).div_euclid(self.up)).max(0);
                let stop = ((center + self.half).div_euclid(self.up)).min(len - 1);
                (start..=stop)
                    .map(|i| {
                        self.weights[(center + self.half - i * self.up) as usize]
                            * f64::from(input[i as usize])
                    })
                    .sum::<f64>() as f32
            })
            .collect()
    }
}

fn bessel_i0(x: f64) -> f64 {
    let square = (x / 2.0) * (x / 2.0);
    let mut sum = 1.0;
    let mut term = 1.0;
    for i in 1..=20 {
        let i = f64::from(i);
        term *= square / (i * i);
        sum += term;
    }
    sum
}

fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}
