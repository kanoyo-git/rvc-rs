//! Voice gate: keep the converted voice from rendering what is not voice.
//!
//! Separated vocal stems carry the separator's mistakes: hi-hat and snare ticks, synth
//! tails, effect sweeps, ghosts of the beat. The voice model renders whatever it is
//! given, so in the pauses those come out as "sung" noises. A small classifier on the
//! content features the model already receives (one 768-d frame per 20 ms, repeated to
//! 10 ms) tells voice (speech, singing, fricatives, breaths, screams) from residue. It was
//! fitted on separation round trips: clean takes mixed with real instrumentals and split
//! again, labelled from the clean take (voice where it sounds, residue where it is silent
//! but the separated input is audible).
//!
//! A frame is attenuated only when the classifier reads it as not voice *and* the input
//! is at least 15 dB under the loud part of the take: residue in the pauses is quiet,
//! and a misread loud syllable must never be cut. Gains move over ~70 ms.

use std::path::Path;

use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use tch::{Device, Kind, Tensor};

/// Voice probability under which a frame counts as not voice.
pub const THRESHOLD: f32 = 0.3;
/// Only frames this far (dB) under the loud part of the input may be attenuated.
pub const QUIET_DB: f32 = 15.0;
/// 10 ms frames of 16 kHz analysis audio.
const FRAME: usize = 160;

#[derive(Deserialize)]
struct Weights {
    w1: Vec<Vec<f32>>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: f32,
}

/// One-hidden-layer classifier: sigmoid(w2 · relu(W1 x + b1) + b2).
pub struct VoiceGate {
    w1: Tensor,
    b1: Tensor,
    w2: Tensor,
    b2: f64,
}

impl VoiceGate {
    /// Load `path` if it exists (None otherwise).
    pub fn load(path: &Path, width: usize, device: Device) -> Result<Option<Self>> {
        if !path.is_file() {
            return Ok(None);
        }
        let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let weights: Weights = serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
        let hidden = weights.b1.len();
        ensure!(
            weights.w1.len() == hidden && weights.w1.iter().all(|row| row.len() == width) && weights.w2.len() == hidden,
            "voice gate {} does not match {width}-wide features",
            path.display()
        );
        let flat: Vec<f32> = weights.w1.into_iter().flatten().collect();
        Ok(Some(Self {
            w1: Tensor::f_from_slice(&flat)?.view([hidden as i64, width as i64]).to_device(device),
            b1: Tensor::f_from_slice(&weights.b1)?.to_device(device),
            w2: Tensor::f_from_slice(&weights.w2)?.to_device(device),
            b2: f64::from(weights.b2),
        }))
    }

    /// Voice probability per frame of `features` ([1, frames, width]).
    pub fn probability(&self, features: &Tensor) -> Result<Vec<f32>> {
        let x = features.squeeze_dim(0).to_kind(Kind::Float);
        let hidden = (x.matmul(&self.w1.tr()) + &self.b1).relu();
        let logit = hidden.matmul(&self.w2) + self.b2;
        Ok(Vec::<f32>::try_from(logit.sigmoid().to_device(Device::Cpu).contiguous())?)
    }
}

/// Per-frame gains (10 ms frames) from voice probabilities and the 16 kHz input `audio`:
/// `attenuation_db` where the smoothed probability is under [`THRESHOLD`] and the frame is
/// [`QUIET_DB`] under the loud part of the input, 1 elsewhere, eased over 7 frames.
pub fn gains(probability: &[f32], audio: &[f32], attenuation_db: f32) -> Vec<f32> {
    let frames = (audio.len() / FRAME).min(probability.len());
    if frames == 0 {
        return Vec::new();
    }
    let level: Vec<f32> = (0..frames)
        .map(|t| {
            let chunk = &audio[t * FRAME..(t + 1) * FRAME];
            10.0 * (chunk.iter().map(|v| v * v).sum::<f32>() / FRAME as f32 + 1e-12).log10()
        })
        .collect();
    let mut sorted = level.clone();
    sorted.sort_by(f32::total_cmp);
    let loud = sorted[sorted.len() * 95 / 100];
    let low = 10_f32.powf(-attenuation_db / 20.0);
    let target: Vec<f32> = (0..frames)
        .map(|t| {
            let around = t.saturating_sub(1)..(t + 2).min(frames);
            let p = probability[around.clone()].iter().sum::<f32>() / around.len() as f32;
            if p < THRESHOLD && level[t] < loud - QUIET_DB { low } else { 1.0 }
        })
        .collect();
    if std::env::var_os("RVC_GATE_DEBUG").is_some() {
        let quiet = (0..frames).filter(|&t| level[t] < loud - QUIET_DB).count();
        let lowered = target.iter().filter(|&&g| g < 1.0).count();
        eprintln!(
            "voice gate: {lowered} of {frames} frames lowered ({:.1} %), {:.1} % of the quiet frames",
            100.0 * lowered as f32 / frames as f32,
            100.0 * lowered as f32 / quiet.max(1) as f32
        );
    }
    (0..frames)
        .map(|t| {
            let around = t.saturating_sub(3)..(t + 4).min(frames);
            target[around.clone()].iter().sum::<f32>() / around.len() as f32
        })
        .collect()
}

/// Scale `output` (at `rate`, aligned with the analysis audio) by 10 ms frame `gains`,
/// interpolated between frame centres. Samples past the last frame keep its gain.
pub fn apply(output: &mut [f32], gains: &[f32], rate: u32) {
    if gains.is_empty() {
        return;
    }
    let step = rate as f32 / 100.0;
    let last = gains.len() - 1;
    for (i, sample) in output.iter_mut().enumerate() {
        let pos = (i as f32 / step - 0.5).max(0.0);
        let a = (pos.floor() as usize).min(last);
        let b = (a + 1).min(last);
        let x = (pos - a as f32).clamp(0.0, 1.0);
        *sample *= gains[a] * (1.0 - x) + gains[b] * x;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_quiet_frames_read_as_residue_are_lowered() {
        // 1 s loud voice, 1 s quiet residue, 1 s loud frames the classifier misreads.
        let n = 16_000;
        let audio: Vec<f32> = (0..3 * n)
            .map(|i| {
                let a = if (n..2 * n).contains(&i) { 0.01 } else { 0.3 };
                (i as f32 * 0.1).sin() * a
            })
            .collect();
        let probability: Vec<f32> = (0..300).map(|t| if t < 100 { 0.95 } else { 0.05 }).collect();
        let g = gains(&probability, &audio, 12.0);
        let mean = |r: std::ops::Range<usize>| g[r.clone()].iter().sum::<f32>() / r.len() as f32;
        assert!(mean(5..95) > 0.999, "voice touched");
        assert!(mean(110..190) < 0.26, "residue kept: {}", mean(110..190));
        assert!(mean(210..290) > 0.999, "a loud misread frame was cut");
        // Applying the gains at another rate keeps the shape.
        let mut out = vec![1.0_f32; 3 * 40_000];
        apply(&mut out, &g, 40_000);
        assert!(out[20_000] > 0.99 && out[60_000] < 0.26 && out[100_000] > 0.99);
    }

    #[test]
    fn the_shipped_gate_loads_and_scores_frames() {
        let path = std::env::temp_dir().join(format!("rvc-gate-{}.json", std::process::id()));
        let weights = serde_json::json!({
            "w1": vec![vec![0.0_f32; 4]; 2], "b1": [1.0, -1.0], "w2": [2.0, 0.0], "b2": 0.0
        });
        std::fs::write(&path, weights.to_string()).unwrap();
        let gate = VoiceGate::load(&path, 4, Device::Cpu).unwrap().unwrap();
        let p = gate.probability(&Tensor::zeros([1, 3, 4], (Kind::Float, Device::Cpu))).unwrap();
        // relu(1) * 2 = 2 -> sigmoid(2) for every frame.
        assert!(p.iter().all(|v| (v - 0.8808).abs() < 1e-3), "{p:?}");
        assert!(VoiceGate::load(&path, 5, Device::Cpu).is_err());
        std::fs::remove_file(&path).ok();
        assert!(VoiceGate::load(&path, 4, Device::Cpu).unwrap().is_none());
    }
}
