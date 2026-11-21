//! Unvoiced envelope transfer: natural breaths and sibilants.
//!
//! A converted voice renders breaths and fricatives through the same decoder as
//! vowels. Measured against the source, it adds a low "voiced" murmur under
//! breaths (+3 dB at 300-1000 Hz) and under "s"/"sh" (+4.5 dB), an extra sizzle
//! above 12 kHz (+4-5 dB), and it lifts breaths by about 3 dB relative to the
//! singing. Breaths and fricatives are turbulent noise shaped by the mouth: their
//! spectral shape and loudness carry the articulation, not the singer's identity.
//!
//! On frames where the source has no pitch, this module reshapes the converted
//! spectrum band by band so that its shape and its level relative to the voiced
//! part follow the source. The noise itself (phase, fine texture) stays the
//! model's. Voiced frames pass through unchanged: the analysis and synthesis
//! windows reconstruct them exactly.

use rustfft::FftPlanner;
use rustfft::num_complex::Complex;

/// Largest correction per band and frame, in either direction (±12 dB).
const MAX_BAND_GAIN: f32 = 4.0;
/// Largest level correction of an unvoiced frame (±10 dB).
const MAX_LEVEL_GAIN: f32 = 3.16;

pub struct Source<'a> {
    /// The source recording at the output rate, aligned with the output.
    pub audio: &'a [f32],
    /// F0 per 10 ms frame (0 = unvoiced), aligned with the output.
    pub pitch: &'a [f32],
    /// Highest frequency the source actually carries (8 kHz for 16 kHz input).
    pub bandwidth: f32,
}

/// Clear F0 on frames that are voiceless fricatives in the 16 kHz `audio`.
///
/// RMVPE reports a pitch on a large share of "s", "sh" and breath frames
/// (42 % of sibilant frames on a rap acapella), probably from voicing that
/// bleeds in at the edges. The NSF source then drives harmonics through the
/// hiss, which rings. A frame counts as voiceless when over half of its energy
/// lies between 3.5 and 8 kHz and under 15 % below 1 kHz; voiced fricatives
/// ("z", "zh") keep their low voicing and stay voiced. Breaths are cleared too:
/// noise-like frames (spectral flatness over 0.25 between 300 Hz and 8 kHz, under
/// half of the energy below 1 kHz) 12-45 dB under the loud part of the take. RMVPE
/// gives most of them a pitch, and the NSF source then hums through the breath
/// (+1.3 dB at 300-1000 Hz measured on converted acapellas). `pitch` holds 10 ms
/// frames centered on multiples of 160 samples of `audio`. Returns the frames cleared.
pub fn unvoice_fricatives(audio: &[f32], pitch: &mut [f32]) -> usize {
    const SIZE: usize = 512;
    const HOP: usize = 160;
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(SIZE);
    let window: Vec<f32> = (0..SIZE)
        .map(|i| (std::f32::consts::PI * (i as f32 + 0.5) / SIZE as f32).sin().powi(2))
        .collect();
    let hz = 16_000.0 / SIZE as f32;
    let bin = |f: f32| (f / hz).round() as usize;
    let mut energy = Vec::with_capacity(pitch.len());
    let mut fricative = Vec::with_capacity(pitch.len());
    let mut noisy = Vec::with_capacity(pitch.len());
    for t in 0..pitch.len() {
        let start = (t * HOP) as isize - (SIZE / 2) as isize;
        let mut buffer: Vec<Complex<f32>> = (0..SIZE)
            .map(|i| {
                let at = start + i as isize;
                let v = if at >= 0 && (at as usize) < audio.len() { audio[at as usize] } else { 0.0 };
                Complex::new(v * window[i], 0.0)
            })
            .collect();
        fft.process(&mut buffer);
        let power: Vec<f32> = buffer[..SIZE / 2 + 1].iter().map(|v| v.norm_sqr()).collect();
        let sum = |a: f32, b: f32| power[bin(a)..bin(b)].iter().sum::<f32>();
        let total = sum(80.0, 7_900.0) + 1e-12;
        energy.push(total);
        fricative.push(sum(3_500.0, 7_900.0) / total > 0.5 && sum(80.0, 1_000.0) / total < 0.15);
        let band = &power[bin(300.0)..bin(7_900.0)];
        let mean = band.iter().sum::<f32>() / band.len() as f32 + 1e-20;
        let log_mean = band.iter().map(|p| (p + 1e-20).ln()).sum::<f32>() / band.len() as f32;
        noisy.push(log_mean.exp() / mean > 0.25 && sum(80.0, 1_000.0) / total < 0.5);
    }
    let mut sorted = energy.clone();
    sorted.sort_by(f32::total_cmp);
    let top = sorted.get(sorted.len() * 95 / 100).copied().unwrap_or(0.0);
    let floor = top * 1e-5;
    let breath_ceiling = top * 10_f32.powf(-1.2);
    let breath_floor = top * 10_f32.powf(-4.5);
    let mut cleared = 0;
    for (((hz, &fricative), &noisy), &e) in pitch.iter_mut().zip(&fricative).zip(&noisy).zip(&energy) {
        let breath = noisy && e < breath_ceiling && e > breath_floor;
        if (fricative && e > floor || breath) && *hz > 0.0 {
            *hz = 0.0;
            cleared += 1;
        }
    }
    cleared
}

/// Highest frequency with real content: where the long-term spectrum falls
/// 60 dB under its peak (about 8 kHz for 16 kHz sources, 16 kHz for many MP3s).
pub fn bandwidth(audio: &[f32], rate: u32) -> f32 {
    let size = 4096;
    if audio.len() < size {
        return rate as f32 / 2.0;
    }
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(size);
    let mut power = vec![0_f32; size / 2 + 1];
    let step = (audio.len() / 64).max(size);
    for start in (0..audio.len() - size).step_by(step) {
        let mut buffer: Vec<Complex<f32>> = audio[start..start + size]
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                let w = (std::f32::consts::PI * (i as f32 + 0.5) / size as f32).sin().powi(2);
                Complex::new(v * w, 0.0)
            })
            .collect();
        fft.process(&mut buffer);
        for (p, v) in power.iter_mut().zip(&buffer) {
            *p += v.norm_sqr();
        }
    }
    let peak = power.iter().copied().fold(0_f32, f32::max);
    let last = power.iter().rposition(|&p| p > peak * 1e-6).unwrap_or(power.len() - 1);
    last as f32 * rate as f32 / size as f32
}

/// Reshape unvoiced frames of `output` (at `rate`) toward `source`; `strength`
/// in [0, 1] blends between untouched (0) and fully matched (1).
pub fn transfer(output: &mut [f32], source: &Source<'_>, rate: u32, strength: f32) {
    if strength <= 0.0 || output.is_empty() {
        return;
    }
    let hop = (rate / 100) as usize;
    let window_len = 4 * hop;
    let size = window_len.next_power_of_two();
    let frames = output.len().div_ceil(hop) + 3;
    let window: Vec<f32> = (0..window_len)
        .map(|i| {
            let x = (i as f32 + 0.5) / window_len as f32;
            (std::f32::consts::PI * x).sin().powi(2)
        })
        .collect();
    let mut planner = FftPlanner::<f32>::new();
    let forward = planner.plan_fft_forward(size);
    let inverse = planner.plan_fft_inverse(size);
    let bins = size / 2 + 1;
    let hz = rate as f32 / size as f32;

    // Third-octave bands from 150 Hz up to Nyquist; bands above the source's
    // bandwidth are left alone.
    let mut edges = vec![150.0_f32];
    while *edges.last().unwrap_or(&0.0) < rate as f32 / 2.0 {
        let next = edges.last().copied().unwrap_or(150.0) * 2_f32.powf(1.0 / 3.0);
        edges.push(next.min(rate as f32 / 2.0));
    }
    let band_of: Vec<Option<usize>> = (0..bins)
        .map(|k| {
            let f = k as f32 * hz;
            edges.windows(2).position(|e| f >= e[0] && f < e[1])
        })
        .collect();
    let bands = edges.len() - 1;
    let usable: Vec<bool> = edges.windows(2).map(|e| e[1] <= source.bandwidth * 1.01).collect();

    let spectra = |signal: &[f32]| -> Vec<Vec<Complex<f32>>> {
        (0..frames)
            .map(|t| {
                let start = t as isize * hop as isize - (window_len / 2) as isize;
                let mut buffer = vec![Complex::new(0.0, 0.0); size];
                for (i, w) in window.iter().enumerate() {
                    let at = start + i as isize;
                    if at >= 0 && (at as usize) < signal.len() {
                        buffer[i] = Complex::new(signal[at as usize] * w, 0.0);
                    }
                }
                forward.process(&mut buffer);
                buffer.truncate(bins);
                buffer
            })
            .collect()
    };
    let out_spec = spectra(output);
    let src_spec = spectra(source.audio);
    let band_power = |spectrum: &[Complex<f32>]| -> Vec<f32> {
        let mut power = vec![0_f32; bands];
        for (k, value) in spectrum.iter().enumerate() {
            if let Some(b) = band_of[k] {
                power[b] += value.norm_sqr();
            }
        }
        power
    };
    let total = |power: &[f32]| -> f32 {
        power.iter().zip(&usable).filter(|(_, u)| **u).map(|(p, _)| p).sum::<f32>() + 1e-12
    };
    let src_bands: Vec<Vec<f32>> = src_spec.iter().map(|s| band_power(s)).collect();
    let out_bands: Vec<Vec<f32>> = out_spec.iter().map(|s| band_power(s)).collect();
    let src_level: Vec<f32> = src_bands.iter().map(|p| total(p)).collect();
    let out_level: Vec<f32> = out_bands.iter().map(|p| total(p)).collect();

    // Unvoiced weight per frame, softened over ±1 frame, and gated to frames
    // that carry sound (within 45 dB of the loud part of the source).
    let mut loud: Vec<f32> = src_level.clone();
    loud.sort_by(f32::total_cmp);
    let top = loud[loud.len() * 95 / 100].max(1e-12);
    let unvoiced: Vec<f32> = (0..frames)
        .map(|t| {
            let pitch = source.pitch.get(t).copied().unwrap_or(0.0);
            if pitch <= 0.0 && src_level[t] > top * 10_f32.powf(-4.5) { 1.0 } else { 0.0 }
        })
        .collect();
    let weight: Vec<f32> = (0..frames)
        .map(|t| {
            let around = t.saturating_sub(1)..(t + 2).min(frames);
            let n = around.len() as f32;
            unvoiced[around].iter().sum::<f32>() / n
        })
        .collect();
    // Frames that keep a pitch but whose source is dominated by hiss above 3 kHz: the
    // voiced edges of "s", "sh" and "ch", and voiced fricatives. Only the bands above
    // 3 kHz follow the source there, so the harmonics underneath stay the model's.
    let high_from = centers_from(&edges, 3_000.0);
    let hissy: Vec<f32> = (0..frames)
        .map(|t| {
            let pitch = source.pitch.get(t).copied().unwrap_or(0.0);
            let high: f32 = src_bands[t][high_from..].iter().zip(&usable[high_from..]).filter(|(_, u)| **u).map(|(p, _)| p).sum();
            if pitch > 0.0 && src_level[t] > top * 10_f32.powf(-4.5) && high / src_level[t] > 0.4 { 1.0 } else { 0.0 }
        })
        .collect();
    // The voiced level of each signal anchors the relative loudness of breaths.
    let voiced_mean = |level: &[f32]| -> f32 {
        let (sum, n) = (0..frames)
            .filter(|&t| unvoiced[t] == 0.0 && src_level[t] > top * 10_f32.powf(-2.0))
            .fold((0_f32, 0usize), |(s, n), t| (s + level[t], n + 1));
        if n == 0 { 1.0 } else { sum / n as f32 }
    };
    let (src_voiced, out_voiced) = (voiced_mean(&src_level), voiced_mean(&out_level));

    let centers: Vec<f32> = edges.windows(2).map(|e| (e[0] * e[1]).sqrt()).collect();
    let mut result = vec![0_f32; (frames + 4) * hop + window_len];
    let mut norm = vec![0_f32; result.len()];
    for t in 0..frames {
        let mut spectrum = out_spec[t].clone();
        let w = weight[t] * strength;
        let partial = if w > 0.0 { 0.0 } else { hissy[t] * strength };
        if partial > 0.0 {
            let around = t.saturating_sub(1)..(t + 2).min(frames);
            let gains: Vec<f32> = (0..bands)
                .map(|b| {
                    if b < high_from || !usable[b] {
                        return 1.0;
                    }
                    let src: f32 = around.clone().map(|u| src_bands[u][b]).sum::<f32>() / src_voiced;
                    let out: f32 = around.clone().map(|u| out_bands[u][b]).sum::<f32>() / out_voiced;
                    (src / out.max(1e-12)).sqrt().clamp(1.0 / MAX_BAND_GAIN, MAX_BAND_GAIN)
                })
                .collect();
            apply_band_gains(&mut spectrum, &gains, &centers, hz, partial);
        }
        if w > 0.0 {
            // Three-frame band powers make the per-frame shape estimate stable.
            let smoothed = |bank: &[Vec<f32>], b: usize| -> f32 {
                let around = t.saturating_sub(1)..(t + 2).min(frames);
                around.map(|u| bank[u][b]).sum::<f32>()
            };
            let (src_total, out_total): (f32, f32) = (0..bands)
                .filter(|&b| usable[b])
                .map(|b| (smoothed(&src_bands, b), smoothed(&out_bands, b)))
                .fold((1e-12, 1e-12), |(a, c), (s, o)| (a + s, c + o));
            let level = ((src_level[t] / src_voiced) / (out_level[t] / out_voiced).max(1e-12))
                .sqrt()
                .clamp(1.0 / MAX_LEVEL_GAIN, MAX_LEVEL_GAIN);
            let gains: Vec<f32> = (0..bands)
                .map(|b| {
                    if !usable[b] {
                        return level;
                    }
                    let share = |bank: &[Vec<f32>], total: f32| smoothed(bank, b) / total;
                    let shape = (share(&src_bands, src_total) / share(&out_bands, out_total).max(1e-9))
                        .sqrt()
                        .clamp(1.0 / MAX_BAND_GAIN, MAX_BAND_GAIN);
                    shape * level
                })
                .collect();
            apply_band_gains(&mut spectrum, &gains, &centers, hz, w);
        }
        // Inverse transform of the half spectrum.
        let mut full = vec![Complex::new(0.0, 0.0); size];
        full[..bins].copy_from_slice(&spectrum);
        for k in 1..size - bins + 1 {
            full[size - k] = spectrum[k].conj();
        }
        inverse.process(&mut full);
        let start = t * hop;
        for (i, w) in window.iter().enumerate() {
            result[start + i] += full[i].re / size as f32 * w;
            norm[start + i] += w * w;
        }
    }
    let offset = window_len / 2;
    for (i, sample) in output.iter_mut().enumerate() {
        let n = norm[i + offset];
        if n > 1e-6 {
            *sample = result[i + offset] / n;
        }
    }
}

/// Index of the first third-octave band whose center lies at or above `hz`.
fn centers_from(edges: &[f32], hz: f32) -> usize {
    edges.windows(2).position(|e| (e[0] * e[1]).sqrt() >= hz).unwrap_or(edges.len() - 1)
}

/// Scale `spectrum` by per-band `gains`, interpolated in log-frequency between band
/// centers, raised to the power `weight`.
fn apply_band_gains(spectrum: &mut [Complex<f32>], gains: &[f32], centers: &[f32], hz: f32, weight: f32) {
    let bands = gains.len();
    for (k, value) in spectrum.iter_mut().enumerate() {
        let f = (k as f32 * hz).max(1.0);
        let log_gain = if f <= centers[0] {
            gains[0].ln()
        } else if f >= centers[bands - 1] {
            gains[bands - 1].ln()
        } else {
            let b = centers.iter().position(|&c| c > f).unwrap_or(bands - 1);
            let x = (f.ln() - centers[b - 1].ln()) / (centers[b].ln() - centers[b - 1].ln());
            gains[b - 1].ln() * (1.0 - x) + gains[b].ln() * x
        };
        *value *= (log_gain * weight).exp();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn noise(n: usize, seed: u32) -> Vec<f32> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                state as f32 / u32::MAX as f32 - 0.5
            })
            .collect()
    }

    #[test]
    fn a_quiet_breath_between_sung_notes_loses_its_pitch() {
        // 0.5 s sung tone, 0.5 s soft broadband breath (-20 dB), 0.5 s tone.
        let n = 8_000;
        let breath = noise(n, 7);
        let audio: Vec<f32> = (0..3 * n)
            .map(|i| {
                let tone = (i as f32 * std::f32::consts::TAU * 220.0 / 16_000.0).sin() * 0.5
                    + (i as f32 * std::f32::consts::TAU * 440.0 / 16_000.0).sin() * 0.2;
                if (n..2 * n).contains(&i) { breath[i - n] * 0.1 } else { tone }
            })
            .collect();
        let mut pitch = vec![220.0; 150];
        unvoice_fricatives(&audio, &mut pitch);
        let cleared = |r: std::ops::Range<usize>| pitch[r].iter().filter(|&&hz| hz == 0.0).count();
        assert!(cleared(55..95) >= 38, "breath kept its pitch");
        assert_eq!(cleared(5..45) + cleared(105..145), 0, "sung frames lost their pitch");
    }

    #[test]
    fn hiss_loses_its_pitch_but_a_hummed_z_keeps_it() {
        let n = 16_000;
        let hiss: Vec<f32> = {
            // High-passed noise: a first difference twice.
            let raw = noise(n + 2, 3);
            (0..n).map(|i| raw[i + 2] - 2.0 * raw[i + 1] + raw[i]).collect()
        };
        let mut pitch = vec![180.0; 100];
        assert!(unvoice_fricatives(&hiss, &mut pitch) > 90);
        let z: Vec<f32> = (0..n)
            .map(|i| hiss[i] * 0.5 + (i as f32 * std::f32::consts::TAU * 180.0 / 16_000.0).sin() * 0.6)
            .collect();
        let mut pitch = vec![180.0; 100];
        assert_eq!(unvoice_fricatives(&z, &mut pitch), 0);
    }

    #[test]
    fn voiced_frames_pass_through_unchanged() {
        let rate = 40_000;
        let x: Vec<f32> = (0..rate as usize)
            .map(|i| (i as f32 * 0.05).sin() * 0.3)
            .collect();
        let mut y = x.clone();
        let pitch = vec![200.0; 100];
        let source = Source { audio: &x, pitch: &pitch, bandwidth: 20_000.0 };
        transfer(&mut y, &source, rate, 1.0);
        let error = x.iter().zip(&y).fold(0_f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(error < 1e-4, "{error}");
    }

    #[test]
    fn unvoiced_frames_take_the_source_shape() {
        let rate = 40_000;
        let n = rate as usize;
        // Source: bright hiss. Output: the same hiss plus a loud low hum.
        let hiss = noise(n, 7);
        let source_audio: Vec<f32> = hiss.iter().map(|v| v * 0.2).collect();
        let mut output: Vec<f32> = (0..n)
            .map(|i| source_audio[i] + (i as f32 * std::f32::consts::TAU * 400.0 / rate as f32).sin() * 0.3)
            .collect();
        let pitch = vec![0.0; 100];
        let source = Source { audio: &source_audio, pitch: &pitch, bandwidth: 20_000.0 };
        let hum = |x: &[f32]| -> f32 {
            let mut s = 0_f32;
            let mut c = 0_f32;
            for (i, v) in x.iter().enumerate().skip(4_000).take(32_000) {
                let a = i as f32 * std::f32::consts::TAU * 400.0 / rate as f32;
                s += v * a.sin();
                c += v * a.cos();
            }
            (s * s + c * c).sqrt()
        };
        let before = hum(&output);
        transfer(&mut output, &source, rate, 1.0);
        assert!(hum(&output) < 0.35 * before, "{} {}", hum(&output), before);
    }
}
