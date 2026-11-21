//! Vocal-production augmentation for training: the processing of modern rap and
//! hyperpop vocals.
//!
//! Some artists' vocals reach the model through a heavy chain: saturation and
//! clipping, brick-wall compression that lifts breaths and room, telephone or
//! mid-forward EQ, chorus and flanger, bit reduction. The content encoder and the
//! pitch tracker were trained on clean speech; on such vocals they lose words and
//! voicing (RMVPE marks only 50-72 % of a processed rap verse voiced, against
//! 78-88 % for clean takes), and the converted voice slurs. Training on clips that
//! carry a random chain of these effects on the input side, while the target stays
//! the clean clip, teaches the generator to pull the words out of the processing.
//!
//! Every effect keeps the length and the pitch of the clip, so the clean target, its
//! F0 and the processed content stay aligned.

use std::path::Path;

use anyhow::{Context, Result, ensure};
use rand::Rng;
use rand::rngs::StdRng;
use rustfft::FftPlanner;
use rustfft::num_complex::Complex;

/// Third-octave band centres (Hz) of a long-term average spectrum profile.
pub const PROFILE_BANDS: [f32; 24] = [
    100.0, 125.0, 160.0, 200.0, 250.0, 315.0, 400.0, 500.0, 630.0, 800.0, 1000.0, 1250.0, 1600.0, 2000.0,
    2500.0, 3150.0, 4000.0, 5000.0, 6300.0, 8000.0, 10000.0, 12500.0, 16000.0, 20000.0,
];

/// Long-term average spectrum of the vocals in `input` (a file, or a folder searched
/// recursively), normalized over [`PROFILE_BANDS`], with the number of files read.
pub fn measure(input: &Path) -> Result<(usize, Vec<f32>)> {
    let files = if input.is_dir() {
        crate::audio::files(input, true)?
    } else {
        vec![input.to_owned()]
    };
    ensure!(!files.is_empty(), "no audio found");
    let mut power = vec![0_f64; PROFILE_BANDS.len()];
    for file in &files {
        let samples = crate::audio::read_mono(file, 40_000)?;
        for (p, db) in power.iter_mut().zip(profile(&samples, 40_000)) {
            *p += 10_f64.powf(f64::from(db) / 10.0);
        }
    }
    let total: f64 = power.iter().sum::<f64>().max(1e-20);
    let profile = power
        .iter()
        .map(|p| (10.0 * (p / total).max(1e-12).log10()) as f32)
        .collect();
    Ok((files.len(), profile))
}

/// Read a profile written by [`measure`] (JSON array, one value per band).
pub fn read_profile(path: &Path) -> Result<Vec<f32>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading effects profile {}", path.display()))?;
    let values: Vec<f32> = serde_json::from_str(&text)
        .with_context(|| format!("parsing effects profile {}", path.display()))?;
    ensure!(
        values.len() == PROFILE_BANDS.len(),
        "an effects profile has {} bands",
        PROFILE_BANDS.len()
    );
    Ok(values)
}

/// Which effects a processed clip carries.
#[derive(Clone, Copy, Debug, Default)]
pub struct Applied {
    pub eq: bool,
    pub telephone: bool,
    pub saturation: bool,
    pub compression: bool,
    pub chorus: bool,
    pub crush: bool,
}

impl Applied {
    pub fn count(self) -> usize {
        [self.eq, self.telephone, self.saturation, self.compression, self.chorus, self.crush]
            .iter()
            .filter(|&&on| on)
            .count()
    }
}

/// Long-term average spectrum of `x` in [`PROFILE_BANDS`], in dB, over frames within
/// 30 dB of the loud part (silence would tilt it toward noise).
pub fn profile(x: &[f32], rate: u32) -> Vec<f32> {
    const SIZE: usize = 4096;
    let mut power = vec![0_f64; PROFILE_BANDS.len()];
    if x.len() < SIZE {
        return vec![0.0; PROFILE_BANDS.len()];
    }
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(SIZE);
    let window: Vec<f32> = (0..SIZE)
        .map(|i| (std::f32::consts::PI * (i as f32 + 0.5) / SIZE as f32).sin().powi(2))
        .collect();
    let starts: Vec<usize> = (0..x.len() - SIZE).step_by(SIZE / 2).collect();
    let level = |s: usize| x[s..s + SIZE].iter().map(|v| v * v).sum::<f32>() / SIZE as f32;
    let mut levels: Vec<f32> = starts.iter().map(|&s| level(s)).collect();
    levels.sort_by(f32::total_cmp);
    let gate = levels.get(levels.len() * 95 / 100).copied().unwrap_or(0.0) * 1e-3;
    let hz = rate as f32 / SIZE as f32;
    for &s in &starts {
        if level(s) < gate {
            continue;
        }
        let mut buffer: Vec<Complex<f32>> =
            x[s..s + SIZE].iter().zip(&window).map(|(v, w)| Complex::new(v * w, 0.0)).collect();
        fft.process(&mut buffer);
        for (k, v) in buffer[..SIZE / 2].iter().enumerate() {
            if let Some(b) = band(k as f32 * hz) {
                power[b] += f64::from(v.norm_sqr());
            }
        }
    }
    let total: f64 = power.iter().sum::<f64>().max(1e-20);
    power.iter().map(|p| (10.0 * (p / total + 1e-12).log10()) as f32).collect()
}

/// Index of the third-octave band holding `f`.
fn band(f: f32) -> Option<usize> {
    let step = 2_f32.powf(1.0 / 6.0);
    PROFILE_BANDS.iter().position(|&c| f >= c / step && f < c * step)
}

/// Process `clean` (at `rate`) with a random chain of production effects. `target`
/// is an optional long-term spectrum (see [`profile`]) that the EQ step pulls the
/// clip toward. At least two effects are applied. The result keeps the clean peak.
pub fn process(clean: &[f32], rate: u32, rng: &mut StdRng, target: Option<&[f32]>) -> (Vec<f32>, Applied) {
    let mut applied = Applied {
        eq: target.is_some() && rng.random_bool(0.6),
        telephone: rng.random_bool(0.35),
        saturation: rng.random_bool(0.6),
        compression: rng.random_bool(0.6),
        chorus: rng.random_bool(0.4),
        crush: rng.random_bool(0.2),
    };
    while applied.count() < 2 {
        match rng.random_range(0..4) {
            0 => applied.saturation = true,
            1 => applied.compression = true,
            2 => applied.telephone = true,
            _ => applied.chorus = true,
        }
    }
    let mut x = clean.to_vec();
    if let (true, Some(target)) = (applied.eq, target) {
        let strength = rng.random_range(0.5..1.0);
        x = match_spectrum(&x, rate, target, strength);
    }
    if applied.telephone {
        let low = rng.random_range(150.0..450.0);
        let high = rng.random_range(3_500.0..8_000.0_f32).min(rate as f32 * 0.45);
        let mid = rng.random_range(1_000.0..3_000.0);
        let boost = rng.random_range(4.0..10.0);
        x = Biquad::highpass(low, rate).run(&x);
        x = Biquad::highpass(low, rate).run(&x);
        x = Biquad::lowpass(high, rate).run(&x);
        x = Biquad::lowpass(high, rate).run(&x);
        x = Biquad::peak(mid, 1.0, boost, rate).run(&x);
    }
    if applied.compression {
        let ratio = rng.random_range(4.0..20.0);
        let threshold = rng.random_range(-30.0..-14.0);
        x = compress(&x, rate, threshold, ratio);
    }
    if applied.saturation {
        let drive = 10_f32.powf(rng.random_range(6.0..24.0) / 20.0);
        let mix = rng.random_range(0.5..1.0);
        x = saturate(&x, drive, mix);
    }
    if applied.chorus {
        let delay = rng.random_range(0.004..0.022);
        let depth = rng.random_range(0.001..0.005);
        let speed = rng.random_range(0.2..3.0);
        let mix = rng.random_range(0.3..0.7);
        x = chorus(&x, rate, delay, depth, speed, mix);
    }
    if applied.crush {
        let bits = rng.random_range(6..11);
        let hold = rng.random_range(1..4);
        x = crush(&x, bits, hold);
    }
    let peak = clean.iter().fold(0_f32, |m, v| m.max(v.abs()));
    let now = x.iter().fold(0_f32, |m, v| m.max(v.abs()));
    if now > 0.0 && x.iter().all(|v| v.is_finite()) {
        let scale = peak.max(1e-4) / now;
        x.iter_mut().for_each(|v| *v *= scale);
    } else {
        x = clean.to_vec();
    }
    (x, applied)
}

/// Static EQ that moves the long-term spectrum of `x` toward `target` by `strength`
/// (at most ±12 dB per band), applied with overlapping FFT frames.
fn match_spectrum(x: &[f32], rate: u32, target: &[f32], strength: f32) -> Vec<f32> {
    const SIZE: usize = 2048;
    const STEP: usize = SIZE / 4;
    let own = profile(x, rate);
    let gains_db: Vec<f32> = own
        .iter()
        .zip(target)
        .map(|(o, t)| ((t - o) * strength).clamp(-12.0, 12.0))
        .collect();
    let hz = rate as f32 / SIZE as f32;
    let bins = SIZE / 2 + 1;
    let gain: Vec<f32> = (0..bins)
        .map(|k| {
            let f = (k as f32 * hz).max(PROFILE_BANDS[0]);
            let b = PROFILE_BANDS.iter().position(|&c| c > f).unwrap_or(PROFILE_BANDS.len());
            let db = if b == 0 {
                gains_db[0]
            } else if b == PROFILE_BANDS.len() {
                gains_db[b - 1]
            } else {
                let (lo, hi) = (PROFILE_BANDS[b - 1].ln(), PROFILE_BANDS[b].ln());
                let t = (f.ln() - lo) / (hi - lo);
                gains_db[b - 1] * (1.0 - t) + gains_db[b] * t
            };
            10_f32.powf(db / 20.0)
        })
        .collect();
    let mut planner = FftPlanner::<f32>::new();
    let (forward, inverse) = (planner.plan_fft_forward(SIZE), planner.plan_fft_inverse(SIZE));
    let window: Vec<f32> = (0..SIZE)
        .map(|i| (std::f32::consts::PI * (i as f32 + 0.5) / SIZE as f32).sin().powi(2))
        .collect();
    let n = x.len();
    let frames = n.div_ceil(STEP) + 3;
    let mut out = vec![0_f32; (frames + 2) * STEP + SIZE];
    let mut norm = vec![0_f32; out.len()];
    for t in 0..frames {
        let start = (t * STEP) as isize - (SIZE / 2) as isize;
        let mut buffer: Vec<Complex<f32>> = (0..SIZE)
            .map(|i| {
                let at = start + i as isize;
                let v = if at >= 0 && (at as usize) < n { x[at as usize] } else { 0.0 };
                Complex::new(v * window[i], 0.0)
            })
            .collect();
        forward.process(&mut buffer);
        for k in 0..bins {
            buffer[k] *= gain[k];
            if k > 0 && k < SIZE - k {
                buffer[SIZE - k] = buffer[k].conj();
            }
        }
        inverse.process(&mut buffer);
        for (i, w) in window.iter().enumerate() {
            out[t * STEP + i] += buffer[i].re / SIZE as f32 * w;
            norm[t * STEP + i] += w * w;
        }
    }
    (0..n)
        .map(|i| {
            let k = i + SIZE / 2;
            if norm[k] > 1e-6 { out[k] / norm[k] } else { 0.0 }
        })
        .collect()
}

/// Soft clipping: `tanh(drive * x)` blended with the dry signal by `mix`.
fn saturate(x: &[f32], drive: f32, mix: f32) -> Vec<f32> {
    let peak = x.iter().fold(0_f32, |m, v| m.max(v.abs())).max(1e-6);
    x.iter()
        .map(|&v| {
            let n = v / peak;
            let wet = (drive * n).tanh() / drive.tanh();
            (wet * mix + n * (1.0 - mix)) * peak
        })
        .collect()
}

/// Feed-forward compressor on a 5 ms RMS envelope (2 ms attack, 60 ms release) with
/// `threshold` in dB under the peak and make-up gain back to the peak.
fn compress(x: &[f32], rate: u32, threshold: f32, ratio: f32) -> Vec<f32> {
    let peak = x.iter().fold(0_f32, |m, v| m.max(v.abs())).max(1e-6);
    let window = (rate as usize / 200).max(1);
    let attack = (-1.0 / (0.002 * rate as f32)).exp();
    let release = (-1.0 / (0.060 * rate as f32)).exp();
    let mut acc = 0_f32;
    let mut env = 0_f32;
    let mut out = Vec::with_capacity(x.len());
    for (i, &v) in x.iter().enumerate() {
        acc += v * v;
        if i >= window {
            acc -= x[i - window] * x[i - window];
        }
        let rms = (acc.max(0.0) / window as f32).sqrt();
        let coeff = if rms > env { attack } else { release };
        env = coeff * env + (1.0 - coeff) * rms;
        let level = 20.0 * (env / peak).max(1e-6).log10();
        let over = (level - threshold).max(0.0);
        let cut = over - over / ratio;
        out.push(v * 10_f32.powf(-cut / 20.0));
    }
    out
}

/// One voice of chorus: a copy delayed by `delay` ± `depth` seconds, swept at
/// `speed` Hz, blended by `mix`.
fn chorus(x: &[f32], rate: u32, delay: f32, depth: f32, speed: f32, mix: f32) -> Vec<f32> {
    let r = rate as f32;
    x.iter()
        .enumerate()
        .map(|(i, &v)| {
            // The sweep may reach past the base delay: never read ahead of the input.
            let lag = ((delay + depth * (std::f32::consts::TAU * speed * i as f32 / r).sin()) * r).max(0.0);
            let pos = i as f32 - lag;
            let wet = if pos >= 0.0 {
                let a = (pos.floor() as usize).min(x.len() - 1);
                let t = pos - a as f32;
                let b = (a + 1).min(x.len() - 1);
                x[a] * (1.0 - t) + x[b] * t
            } else {
                0.0
            };
            v * (1.0 - mix) + wet * mix
        })
        .collect()
}

/// Bit reduction to `bits` and a sample-and-hold of `hold` samples.
fn crush(x: &[f32], bits: u32, hold: usize) -> Vec<f32> {
    let peak = x.iter().fold(0_f32, |m, v| m.max(v.abs())).max(1e-6);
    let steps = (1_u32 << (bits - 1)) as f32;
    let mut out = Vec::with_capacity(x.len());
    let mut held = 0_f32;
    for (i, &v) in x.iter().enumerate() {
        if i % hold.max(1) == 0 {
            held = (v / peak * steps).round() / steps * peak;
        }
        out.push(held);
    }
    out
}

/// RBJ biquad filter.
struct Biquad {
    b: [f32; 3],
    a: [f32; 2],
}

impl Biquad {
    fn new(b0: f32, b1: f32, b2: f32, a0: f32, a1: f32, a2: f32) -> Self {
        Self { b: [b0 / a0, b1 / a0, b2 / a0], a: [a1 / a0, a2 / a0] }
    }

    fn highpass(f: f32, rate: u32) -> Self {
        let w = std::f32::consts::TAU * f / rate as f32;
        let alpha = w.sin() / (2.0 * std::f32::consts::FRAC_1_SQRT_2);
        let c = w.cos();
        Self::new((1.0 + c) / 2.0, -(1.0 + c), (1.0 + c) / 2.0, 1.0 + alpha, -2.0 * c, 1.0 - alpha)
    }

    fn lowpass(f: f32, rate: u32) -> Self {
        let w = std::f32::consts::TAU * f / rate as f32;
        let alpha = w.sin() / (2.0 * std::f32::consts::FRAC_1_SQRT_2);
        let c = w.cos();
        Self::new((1.0 - c) / 2.0, 1.0 - c, (1.0 - c) / 2.0, 1.0 + alpha, -2.0 * c, 1.0 - alpha)
    }

    fn peak(f: f32, q: f32, gain_db: f32, rate: u32) -> Self {
        let a = 10_f32.powf(gain_db / 40.0);
        let w = std::f32::consts::TAU * f / rate as f32;
        let alpha = w.sin() / (2.0 * q);
        let c = w.cos();
        Self::new(1.0 + alpha * a, -2.0 * c, 1.0 - alpha * a, 1.0 + alpha / a, -2.0 * c, 1.0 - alpha / a)
    }

    fn run(&self, x: &[f32]) -> Vec<f32> {
        let (mut x1, mut x2, mut y1, mut y2) = (0_f32, 0_f32, 0_f32, 0_f32);
        x.iter()
            .map(|&v| {
                let y = self.b[0] * v + self.b[1] * x1 + self.b[2] * x2 - self.a[0] * y1 - self.a[1] * y2;
                x2 = x1;
                x1 = v;
                y2 = y1;
                y1 = y;
                y
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn sung(n: usize, rate: u32) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let t = i as f32 / rate as f32;
                let env = (std::f32::consts::PI * t / (n as f32 / rate as f32)).sin();
                (1..40).map(|h| (std::f32::consts::TAU * 180.0 * h as f32 * t).sin() * 0.3 / h as f32).sum::<f32>() * env
            })
            .collect()
    }

    #[test]
    fn processed_clips_keep_length_peak_and_stay_finite() {
        let rate = 40_000;
        let clean = sung(rate as usize * 2, rate);
        let target = profile(&clean, rate).iter().map(|v| v + 3.0).collect::<Vec<_>>();
        let peak = clean.iter().fold(0_f32, |m, v| m.max(v.abs()));
        for seed in 0..24 {
            let mut rng = StdRng::seed_from_u64(seed);
            let (out, applied) = process(&clean, rate, &mut rng, Some(&target));
            assert_eq!(out.len(), clean.len());
            assert!(applied.count() >= 2);
            assert!(out.iter().all(|v| v.is_finite()));
            let now = out.iter().fold(0_f32, |m, v| m.max(v.abs()));
            assert!((now - peak).abs() < 1e-3, "{seed}: {now} vs {peak}");
        }
    }

    #[test]
    fn saturation_adds_harmonics_and_compression_lifts_quiet_parts() {
        let rate = 40_000;
        let tone: Vec<f32> = (0..rate as usize).map(|i| (i as f32 * 0.03).sin() * 0.5).collect();
        let hard = saturate(&tone, 10.0, 1.0);
        // A saturated sine is closer to a square: its RMS rises toward the peak.
        let rms = |x: &[f32]| (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt();
        assert!(rms(&hard) > rms(&tone) * 1.2);
        // Loud then quiet: compression narrows the gap.
        let steps: Vec<f32> = (0..rate as usize * 2)
            .map(|i| (i as f32 * 0.05).sin() * if i < rate as usize { 0.8 } else { 0.08 })
            .collect();
        let squeezed = compress(&steps, rate, -24.0, 10.0);
        let gap = |x: &[f32]| rms(&x[2_000..rate as usize - 2_000]) / rms(&x[rate as usize + 2_000..]);
        assert!(gap(&squeezed) < gap(&steps) * 0.5, "{} {}", gap(&squeezed), gap(&steps));
    }

    #[test]
    fn a_chorus_deeper_than_its_delay_stays_in_bounds() {
        let x: Vec<f32> = (0..40_000).map(|i| (i as f32 * 0.01).sin()).collect();
        let y = chorus(&x, 40_000, 0.004, 0.005, 3.0, 0.7);
        assert_eq!(y.len(), x.len());
        assert!(y.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn spectrum_matching_moves_the_profile() {
        let rate = 40_000;
        let clean = sung(rate as usize * 2, rate);
        let own = profile(&clean, rate);
        // Ask for 6 dB more around 2 kHz relative to the rest.
        let target: Vec<f32> = own
            .iter()
            .zip(PROFILE_BANDS)
            .map(|(v, c)| if (1_600.0..=2_500.0).contains(&c) { v + 6.0 } else { *v })
            .collect();
        let out = match_spectrum(&clean, rate, &target, 1.0);
        let after = profile(&out, rate);
        let lift = |p: &[f32]| p[13] - p[7];
        assert!(lift(&after) - lift(&own) > 3.0, "{} {}", lift(&own), lift(&after));
    }
}
