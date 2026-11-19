//! Window-corrected autocorrelation with voicing and octave transition costs.

use rustfft::FftPlanner;
use rustfft::num_complex::Complex;

use crate::audio::{ANALYSIS_RATE, FRAME};

struct Candidate {
    hz: f32,
    score: f32,
    previous: usize,
    path: f32,
}

pub fn autocorrelation(wave: &[f32], realtime: bool) -> Vec<f32> {
    let floor = if realtime { 65.0 } else { 50.0 };
    let rate = ANALYSIS_RATE as f32;
    let period = (rate / floor).floor() as usize;
    let half = (3.0 * rate / floor * 0.5).ceil() as usize;
    let mut padded = Vec::new();
    if realtime {
        let right = half.saturating_add(FRAME);
        padded.resize(half, 0.0);
        padded.extend_from_slice(wave);
        padded.resize(half + wave.len() + right, 0.0);
    }
    let samples = if realtime { &padded } else { wave };
    let target = wave.len() / FRAME;
    let len = half * 2;
    if samples.len() < len || target == 0 {
        return vec![0.0; target];
    }
    let frames = (samples.len() - len) / FRAME + 1;
    let start = (samples.len() - len - (frames - 1) * FRAME) / 2;
    let peak = samples.iter().map(|s| s.abs()).fold(0.0, f32::max);
    if peak == 0.0 {
        return vec![0.0; target];
    }
    let fft_len = (len + period + 2).next_power_of_two();
    let mut planner = FftPlanner::<f32>::new();
    let forward = planner.plan_fft_forward(fft_len);
    let inverse = planner.plan_fft_inverse(fft_len);
    let mut spectrum = vec![Complex::default(); fft_len];
    let window: Vec<_> = (0..len)
        .map(|i| {
            (std::f32::consts::TAU * (i + 1) as f32 / (len + 1) as f32)
                .cos()
                .mul_add(-0.5, 0.5)
        })
        .collect();
    let lag_min = (rate / 1100.0).floor() as usize;
    let mut correction = vec![0.0; period + 2];
    for (i, value) in correction.iter_mut().enumerate() {
        *value = window.iter().zip(&window[i..]).map(|(a, b)| a * b).sum();
    }
    let zero_lag = correction[0];
    for coefficient in &mut correction {
        *coefficient = zero_lag / *coefficient;
    }

    let mut lattice: Vec<Vec<Candidate>> = Vec::with_capacity(frames);
    for frame in 0..frames {
        let offset = start + frame * FRAME;
        let segment = &samples[offset..offset + len];
        let central = &segment[half - period / 2..half + period / 2];
        let mean = central.iter().sum::<f32>() / central.len() as f32;
        let local_peak = segment.iter().map(|s| (s - mean).abs()).fold(0.0, f32::max);
        let mut candidates = vec![Candidate {
            hz: 0.0,
            score: 0.6 + (2.0 - local_peak / peak * 1.6 / 0.03).max(0.0),
            previous: 0,
            path: 0.0,
        }];
        if local_peak > 0.0 {
            for ((value, &sample), &weight) in spectrum.iter_mut().zip(segment).zip(&window) {
                *value = Complex::new((sample - mean) * weight, 0.0);
            }
            spectrum[len..].fill(Complex::default());
            forward.process(&mut spectrum);
            for value in &mut spectrum {
                *value = Complex::new(value.norm_sqr(), 0.0);
            }
            inverse.process(&mut spectrum);
            let normalizer = spectrum[0].re;
            let ac = |lag: usize| spectrum[lag].re * correction[lag] / normalizer;
            for lag in lag_min..=period {
                let (left, center, right) = (ac(lag - 1), ac(lag), ac(lag + 1));
                if center > 0.3 && center > left && center >= right {
                    let delta = 0.5 * (left - right) / (left - 2.0 * center + right);
                    let hz = rate / (lag as f32 + delta);
                    if (floor..=1100.0).contains(&hz) {
                        let estimate = center - 0.25 * (left - right) * delta;
                        let strength = if estimate > 1.0 {
                            estimate.recip()
                        } else {
                            estimate
                        };
                        candidates.push(Candidate {
                            hz,
                            score: strength - 0.01 * (1100.0 / hz).log2(),
                            previous: 0,
                            path: 0.0,
                        });
                    }
                }
            }
            candidates[1..].sort_by(|a, b| b.score.total_cmp(&a.score));
            candidates.truncate(15);
        }
        for candidate in &mut candidates {
            if let Some(previous) = lattice.last() {
                let best = previous
                    .iter()
                    .enumerate()
                    .map(|(i, from)| {
                        let transition = match (from.hz > 0.0, candidate.hz > 0.0) {
                            (true, true) => 0.35 * (from.hz / candidate.hz).log2().abs(),
                            (false, false) => 0.0,
                            _ => 0.14,
                        };
                        (i, from.path - transition)
                    })
                    .max_by(|a, b| a.1.total_cmp(&b.1));
                if let Some((id, value)) = best {
                    candidate.path = value + candidate.score;
                    candidate.previous = id;
                }
            } else {
                candidate.path = candidate.score;
            }
        }
        lattice.push(candidates);
    }
    let result = if let Some(last) = lattice.last() {
        let mut choice = (0..last.len())
            .max_by(|&a, &b| last[a].path.total_cmp(&last[b].path))
            .unwrap_or(0);
        let mut result = vec![0.0; frames];
        for (i, frame) in lattice.iter().enumerate().rev() {
            result[i] = frame[choice].hz;
            choice = frame[choice].previous;
        }
        result
    } else {
        Vec::new()
    };
    let left = if realtime {
        0
    } else {
        target.saturating_sub(frames).div_ceil(2)
    };
    let mut output = vec![0.0; left];
    output.extend(result);
    output.resize(if realtime { target + 1 } else { target }, 0.0);
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tracks_periodic_input_and_keeps_silence_unvoiced() {
        let tone: Vec<_> = (0..16_000)
            .map(|i| (std::f32::consts::TAU * i as f32 * 220.0 / 16_000.0).sin())
            .collect();
        let estimates = autocorrelation(&tone, false);
        assert_eq!(estimates.len(), 100);
        assert!(estimates[10..90].iter().all(|hz| (hz - 220.0).abs() < 1.0));
        assert!(
            autocorrelation(&vec![0.0; 16_000], false)
                .iter()
                .all(|&hz| hz == 0.0)
        );
    }
}
