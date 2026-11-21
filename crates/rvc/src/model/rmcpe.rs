//! RMCPE fuses the two trained salience maps on a single 10 ms grid. The
//! waveform tests competing periods, and a joint pitch/voicing path resolves
//! short ambiguities without taking an average between different notes.

use anyhow::{Result, ensure};

use super::RmvpeBackend;
use super::cuda_graph::Cache;
use super::fcpe::Fcpe;
use super::rmvpe::{self, BINS};
use crate::audio::{ANALYSIS_RATE, FRAME};

const FCPE_THRESHOLD: f32 = 0.006;
const RMVPE_THRESHOLD: f32 = 0.03;
const BLOCK_FRAMES: usize = 3_000;
const CONTEXT_FRAMES: usize = 100;

pub(super) struct Rmcpe {
    rmvpe: RmvpeBackend,
    fcpe: Fcpe,
    cents: Vec<f32>,
}

impl Rmcpe {
    pub(super) fn new(rmvpe: RmvpeBackend, fcpe: Fcpe) -> Result<Self> {
        let cents = fcpe.cent_table()?;
        ensure!(
            cents.len() == BINS && cents.windows(2).all(|p| p[0] < p[1]),
            "invalid FCPE pitch grid"
        );
        Ok(Self { rmvpe, fcpe, cents })
    }

    pub(super) fn infer(&self, audio: &[f32], stream: Option<&Cache>) -> Result<Vec<f32>> {
        ensure!(
            !audio.is_empty() && audio.iter().all(|x| x.is_finite()),
            "invalid RMCPE audio"
        );
        let frames = audio.len() / FRAME + 1;
        let mut path = Vec::with_capacity(frames);
        let mut evidence = Vec::with_capacity(frames);
        for begin in (0..frames).step_by(BLOCK_FRAMES) {
            let end = (begin + BLOCK_FRAMES).min(frames);
            let left = begin.saturating_sub(CONTEXT_FRAMES);
            let right = ((end + CONTEXT_FRAMES) * FRAME).min(audio.len());
            let samples = &audio[left * FRAME..right];
            let rmvpe = self.rmvpe.salience(samples, stream)?;
            let fcpe = self.fcpe.infer_salience(samples, stream)?;
            let expected = (samples.len() / FRAME + 1) * BINS;
            ensure!(
                rmvpe.len() == expected && fcpe.len() == expected,
                "RMCPE pitch grids differ in length"
            );
            let row = |t: usize| t * BINS..(t + 1) * BINS;
            for frame in begin..end {
                let t = frame - left;
                // FCPE's STFT is centered half a hop later than RMVPE's. Align
                // the salience, before selecting peaks or taking a voicing decision.
                let before = &fcpe[row(t.saturating_sub(1))];
                let after = &fcpe[row(t)];
                let aligned: [f32; BINS] = std::array::from_fn(|k| (before[k] + after[k]) * 0.5);
                let (choices, models) =
                    candidates(audio, frame, &rmvpe[row(t)], &aligned, &self.cents);
                extend(&mut path, choices);
                evidence.push(models);
            }
        }
        let mut track = backtrack(path);
        refine(audio, &mut track, &evidence);
        Ok(track)
    }

    #[cfg(feature = "cuda-graph")]
    pub(super) fn clear_graph(&mut self) {
        self.rmvpe.clear_graph();
        self.fcpe.graph.get_mut().clear();
    }
}

#[derive(Clone, Copy)]
struct Peak {
    hz: f32,
    strength: f32,
}

fn peaks(row: &[f32], threshold: f32, decode: impl Fn(usize) -> f32) -> Vec<Peak> {
    // Select distinct local modes; shoulders of the same peak are not votes.
    let mut maxima = [None; 2];
    for (i, &p) in row.iter().enumerate() {
        if p < threshold * 0.5
            || (i > 0 && row[i - 1] >= p)
            || (i + 1 < row.len() && row[i + 1] > p)
        {
            continue;
        }
        if maxima[0].is_none_or(|k: usize| p > row[k]) {
            maxima[1] = maxima[0];
            maxima[0] = Some(i);
        } else if maxima[1].is_none_or(|k| p > row[k]) {
            maxima[1] = Some(i);
        }
    }
    let mut modes = Vec::with_capacity(2);
    for bin in maxima.into_iter().flatten() {
        let p = row[bin];
        // Scale the two detectors against their own native voicing thresholds.
        // Sigmoids trained under different losses are not interchangeable.
        modes.push(Peak {
            hz: decode(bin),
            strength: p / (p + threshold),
        });
    }
    modes
}

fn candidates(
    audio: &[f32],
    frame: usize,
    rm: &[f32],
    fc: &[f32],
    cents: &[f32],
) -> (Vec<Node>, Evidence) {
    let rm = peaks(rm, RMVPE_THRESHOLD, |i| rmvpe::centered_pitch(rm, i));
    let fc = peaks(fc, FCPE_THRESHOLD, |i| {
        let mut sum = 0.0;
        let mut weight = 0.0;
        for delta in -4..=4 {
            let k = (i as isize + delta).clamp(0, BINS as isize - 1) as usize;
            sum += fc[k] * cents[k];
            weight += fc[k];
        }
        10.0 * (sum / weight / 1200.0).exp2()
    });
    let mut frequencies: Vec<f32> = Vec::with_capacity(10);
    let mut add = |hz: f32| {
        if (32.0..=1_980.0).contains(&hz)
            && frequencies.iter().all(|x| (hz / x).log2().abs() > 0.012)
        {
            frequencies.push(hz);
        }
    };
    for p in rm.iter().chain(&fc) {
        add(p.hz);
    }
    for p in rm
        .first()
        .into_iter()
        .chain(fc.first())
        .filter(|p| p.strength > 0.58)
    {
        add(p.hz * 0.5);
        add(p.hz * 2.0);
    }
    let center = frame * FRAME;
    let window = &audio[center.saturating_sub(640)..(center + 640).min(audio.len())];
    let mut out = vec![Node::new(0.0, 0.0)];
    for hz in frequencies {
        let vote = |modes: &[Peak]| {
            modes
                .iter()
                .map(|p| {
                    let semitones = 12.0 * (hz / p.hz).log2();
                    p.strength * (-2.0 * semitones * semitones).exp()
                })
                .fold(0.0, f32::max)
        };
        let r = vote(&rm);
        let f = vote(&fc);
        let lag = ANALYSIS_RATE as f32 / hz;
        let periodicity = correlation(window, lag).max(0.0);
        let shorter = correlation(window, lag * 0.5).max(correlation(window, lag / 3.0));
        // A subharmonic also repeats a periodic waveform. Prefer its shorter
        // period only when that period, too, explains almost the whole waveform.
        let duplicate = ((shorter - 0.82) / 0.16).clamp(0.0, 1.0);
        let neural = 0.92 * r.max(f) + 0.53 * r.min(f);
        let score = if r.max(f) < 0.15 {
            // Allow an octave alternative only with near-exact acoustic support.
            (periodicity - 0.9) * 2.5 - duplicate * 1.1
        } else {
            neural + 1.12 * periodicity - 0.95 - 1.35 * duplicate
        };
        out.push(Node::new(hz, score));
    }
    let primary = |modes: &[Peak]| {
        modes
            .first()
            .filter(|p| p.strength > 0.5)
            .map_or(0.0, |p| p.hz)
    };
    let rm_primary = primary(&rm);
    let fc_primary = primary(&fc);
    let agreement =
        rm_primary > 0.0 && fc_primary > 0.0 && (rm_primary / fc_primary).log2().abs() < 0.075;
    (
        out,
        Evidence {
            rm_primary,
            agreement,
        },
    )
}

#[derive(Clone, Copy)]
struct Evidence {
    rm_primary: f32,
    agreement: bool,
}

fn refine(audio: &[f32], track: &mut [f32], evidence: &[Evidence]) {
    let window =
        |i: usize| &audio[(i * FRAME).saturating_sub(320)..(i * FRAME + 320).min(audio.len())];
    let energy: Vec<f32> = (0..track.len())
        .map(|i| {
            let x = window(i);
            x.iter().map(|s| s * s).sum::<f32>() / x.len() as f32
        })
        .collect();
    for (i, (&models, hz)) in evidence.iter().zip(track).enumerate() {
        if *hz == 0.0 {
            continue;
        }
        // A soft consonant beside a vowel can acquire salient bins in both
        // noncausal networks. A short window localizes the actual periodic part.
        let peak = energy[i.saturating_sub(25)..(i + 26).min(energy.len())]
            .iter()
            .copied()
            .fold(0.0, f32::max);
        let level_drop_db = 10.0 * (peak.max(1e-16) / energy[i].max(1e-16)).log10();
        let threshold = 0.48 + 0.45 * (level_drop_db / 24.0).clamp(0.0, 1.0)
            - if models.agreement { 0.15 } else { 0.0 };
        if correlation(window(i), ANALYSIS_RATE as f32 / *hz) < threshold {
            *hz = 0.0;
        } else if models.rm_primary > 0.0 {
            // Retain RMVPE's fine timing when it supports the same local note.
            let cents = 1200.0 * (models.rm_primary / *hz).log2();
            if cents.abs() < 75.0 {
                *hz *= (0.75 * cents / 1200.0).exp2();
            }
        }
    }
}

/// Normalized correlation with a fractional delay. Interpolation matters above
/// 1 kHz, where changing an integer lag by one sample can exceed a semitone.
fn correlation(window: &[f32], lag: f32) -> f32 {
    let n = lag as usize;
    if window.len() <= n + 24 {
        return 0.0;
    }
    let frac = lag - n as f32;
    let pairs = window.len() - n - 1;
    let mut sums = [0.0f32; 5];
    for i in 0..pairs {
        let a = window[i];
        let b = window[i + n] + frac * (window[i + n + 1] - window[i + n]);
        sums[0] += a;
        sums[1] += b;
        sums[2] += a * a;
        sums[3] += b * b;
        sums[4] += a * b;
    }
    let count = pairs as f32;
    let energy_a = (sums[2] - sums[0] * sums[0] / count).max(0.0);
    let energy_b = (sums[3] - sums[1] * sums[1] / count).max(0.0);
    ((sums[4] - sums[0] * sums[1] / count) / (energy_a * energy_b).sqrt().max(1e-12))
        .clamp(-1.0, 1.0)
}

struct Node {
    hz: f32,
    score: f32,
    previous: usize,
}

impl Node {
    fn new(hz: f32, score: f32) -> Self {
        Self {
            hz,
            score,
            previous: 0,
        }
    }
}

fn extend(path: &mut Vec<Vec<Node>>, mut row: Vec<Node>) {
    if let Some(previous) = path.last() {
        for node in &mut row {
            let mut best = f32::NEG_INFINITY;
            for (i, prior) in previous.iter().enumerate() {
                let cost = match (prior.hz > 0.0, node.hz > 0.0) {
                    (true, true) => ((node.hz / prior.hz).log2().abs() * 0.85).min(1.2),
                    (false, false) => 0.0,
                    _ => 0.26,
                };
                if prior.score - cost > best {
                    best = prior.score - cost;
                    node.previous = i;
                }
            }
            node.score += best;
        }
        // Keep path scores near zero irrespective of recording length.
        let max = row
            .iter()
            .map(|n| n.score)
            .fold(f32::NEG_INFINITY, f32::max);
        row.iter_mut().for_each(|n| n.score -= max);
    }
    path.push(row);
}

fn backtrack(path: Vec<Vec<Node>>) -> Vec<f32> {
    let mut idx = path
        .last()
        .and_then(|row| {
            row.iter()
                .enumerate()
                .max_by(|a, b| a.1.score.total_cmp(&b.1.score))
        })
        .map_or(0, |(i, _)| i);
    let mut hz = Vec::with_capacity(path.len());
    for row in path.iter().rev() {
        hz.push(row[idx].hz);
        idx = row[idx].previous;
    }
    hz.reverse();
    hz
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn periodicity_distinguishes_a_missing_fundamental_and_rejects_dc() {
        let samples: Vec<_> = (0..1280)
            .map(|i| {
                let phase = std::f32::consts::TAU * 110.0 * i as f32 / ANALYSIS_RATE as f32;
                (2..=10)
                    .map(|k| (phase * k as f32).sin() / k as f32)
                    .sum::<f32>()
            })
            .collect();
        let lag = ANALYSIS_RATE as f32 / 110.0;
        assert!(correlation(&samples, lag) > 0.995);
        assert!(correlation(&samples, lag / 2.0) < 0.65);
        assert_eq!(correlation(&[0.25; 1280], lag), 0.0);
    }

    #[test]
    fn temporal_path_can_cross_real_octaves_and_go_unvoiced() {
        let mut path = Vec::new();
        for hz in [110.0, 110.0, 110.0, 220.0, 220.0, 220.0, 0.0] {
            let mut row = vec![Node::new(0.0, 0.0)];
            if hz > 0.0 {
                row.extend([Node::new(hz, 1.0), Node::new(330.0, 0.2)]);
            }
            extend(&mut path, row);
        }
        assert_eq!(
            backtrack(path),
            vec![110.0, 110.0, 110.0, 220.0, 220.0, 220.0, 0.0]
        );
    }

    #[test]
    fn voicing_preserves_a_quiet_periodic_note_and_removes_a_dc_tail() {
        let samples: Vec<_> = (0..16_000)
            .map(|i| {
                if i < 8_000 {
                    0.0001 * (std::f32::consts::TAU * 220.0 * i as f32 / 16_000.0).sin()
                } else {
                    0.0
                }
            })
            .collect();
        let mut track = vec![220.0; 101];
        let models = vec![
            Evidence {
                rm_primary: 220.0,
                agreement: true
            };
            track.len()
        ];
        refine(&samples, &mut track, &models);
        assert!(track[5..45].iter().all(|&f| f == 220.0));
        assert!(track[55..].iter().all(|&f| f == 0.0));
    }
}
