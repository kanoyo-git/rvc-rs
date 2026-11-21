// NMVPE's timbre-filter and stable decoder are based on the NVC implementation (MIT).
use std::path::Path;

use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use tch::{Device, Kind};

use super::cuda_graph::Cache;
use super::rmvpe::{BINS, Rmvpe, centered_pitch};
use super::weights::Weights;
use crate::audio::FRAME;
use crate::checkpoint::Checkpoint;

pub(super) struct Nmvpe {
    pub(super) network: Rmvpe,
    decoder: Decoder,
}

#[derive(Deserialize)]
#[serde(default, deny_unknown_fields)]
struct Decoder {
    thred: f32,
    candidate_count: usize,
    emission_temperature: f32,
    jump_cost: f32,
    voiced_switch_cost: f32,
}

impl Default for Decoder {
    fn default() -> Self {
        Self {
            thred: 0.03,
            candidate_count: 12,
            emission_temperature: 0.02,
            jump_cost: 0.06,
            voiced_switch_cost: 2.0,
        }
    }
}

impl Nmvpe {
    pub(super) fn load(path: &Path, device: Device, kind: Kind) -> Result<Self> {
        let mut checkpoint =
            Checkpoint::open(path).with_context(|| format!("loading NMVPE {}", path.display()))?;
        let section = checkpoint.has("model").then_some("model");
        let mut tensors = checkpoint.weights(section, Device::Cpu)?;
        if !tensors.is_empty() && tensors.keys().all(|name| name.starts_with("module.")) {
            tensors = tensors
                .into_iter()
                .map(|(name, tensor)| (name[7..].to_owned(), tensor))
                .collect();
        }
        let filtered = match checkpoint
            .metadata("model_variant")?
            .as_ref()
            .and_then(|value| value.as_str())
        {
            Some("filtered") => true,
            Some("legacy") => false,
            Some(value) => anyhow::bail!("unknown NMVPE model variant {value}"),
            None => tensors.keys().any(|name| name.starts_with("unet.tf.")),
        };
        let decoder = checkpoint
            .metadata("decoder_config")?
            .map(serde_json::from_value::<Decoder>)
            .transpose()?
            .unwrap_or_default();
        ensure!(
            (0.0..1.0).contains(&decoder.thred)
                && decoder.thred > 0.0
                && (1..=BINS).contains(&decoder.candidate_count)
                && decoder.emission_temperature > 0.0
                && decoder.emission_temperature.is_finite()
                && decoder.jump_cost >= 0.0
                && decoder.jump_cost.is_finite()
                && decoder.voiced_switch_cost >= 0.0
                && decoder.voiced_switch_cost.is_finite(),
            "invalid NMVPE decoder configuration"
        );
        Ok(Self {
            network: Rmvpe::from_weights(Weights::new(tensors), device, kind, filtered, true)?,
            decoder,
        })
    }

    pub(super) fn infer(&self, samples: &[f32], graph: Option<&Cache>) -> Result<Vec<f32>> {
        ensure!(!samples.is_empty(), "empty NMVPE input");
        let (power, peak) = samples
            .iter()
            .fold((0.0f64, 0.0f32), |(power, peak), &value| {
                (power + f64::from(value).powi(2), peak.max(value.abs()))
            });
        let rms = (power / samples.len() as f64).sqrt() as f32;
        let gain = if rms > 1e-7 && peak > 1e-7 {
            (0.1 / rms).min(0.99 / peak).min(100.0)
        } else {
            1.0
        };
        const CHUNK: usize = 4096;
        const CONTEXT: usize = 128;
        let frames = samples.len() / FRAME + 1;
        let mut pitch = Vec::with_capacity(frames);
        for start in (0..frames).step_by(CHUNK) {
            let end = (start + CHUNK).min(frames);
            let left = start.saturating_sub(CONTEXT);
            let right = ((end + CONTEXT) * FRAME).min(samples.len());
            let audio: Vec<_> = samples[left * FRAME..right]
                .iter()
                .map(|sample| sample * gain)
                .collect();
            let salience = self.network.infer_salience(&audio, graph)?;
            let f0 = self.decoder.decode(&salience)?;
            pitch.extend_from_slice(&f0[start - left..end - left]);
        }
        Ok(pitch)
    }
}

fn strongest(row: &[f32]) -> usize {
    (1..row.len()).fold(0, |best, i| if row[i] > row[best] { i } else { best })
}

impl Decoder {
    fn decode(&self, salience: &[f32]) -> Result<Vec<f32>> {
        let (frames, rest) = salience.as_chunks::<BINS>();
        ensure!(rest.is_empty(), "NMVPE returned an incomplete pitch frame");
        if frames.is_empty() {
            return Ok(Vec::new());
        }

        let peaks: Vec<_> = frames.iter().map(|row| strongest(row)).collect();
        let uv = self.candidate_count;
        let mut all_candidates = Vec::with_capacity(frames.len());
        let mut history = Vec::with_capacity(frames.len());
        let mut previous = vec![0.0f64; uv + 1];
        for (i, row) in frames.iter().enumerate() {
            let mut candidates = Vec::with_capacity(uv);
            let mut selected = [false; BINS];
            let mut add = |bin: usize| {
                if !selected[bin] && candidates.len() < uv {
                    selected[bin] = true;
                    candidates.push(bin);
                }
            };
            for &bin in &peaks[i.saturating_sub(2)..(i + 3).min(peaks.len())] {
                add(bin);
            }
            let local = |bin: usize| {
                let value = row[bin];
                (bin == 0 || value >= row[bin - 1]) && (bin + 1 == BINS || value >= row[bin + 1])
            };
            let mut ranked: [usize; BINS] = std::array::from_fn(|bin| bin);
            let order = |&a: &usize, &b: &usize| {
                local(b)
                    .cmp(&local(a))
                    .then_with(|| row[b].total_cmp(&row[a]))
                    .then(a.cmp(&b))
            };
            let rank = (uv + 5).min(BINS);
            ranked.select_nth_unstable_by(rank - 1, order);
            ranked[..rank].sort_unstable_by(order);
            for &bin in &ranked[..rank] {
                add(bin);
            }
            debug_assert_eq!(candidates.len(), uv);

            let threshold = f64::from(self.thred);
            let temperature = f64::from(self.emission_temperature);
            let switch = f64::from(self.voiced_switch_cost);
            let mut back = vec![uv; uv + 1];
            let mut score = Vec::with_capacity(uv + 1);
            for (j, &bin) in candidates.iter().enumerate() {
                let mut best = previous[uv] - switch;
                if i > 0 {
                    let bins: &Vec<usize> = &all_candidates[i - 1];
                    for (k, &old) in bins.iter().enumerate() {
                        let offset = old.abs_diff(bin) as f64 * f64::from(self.jump_cost);
                        let option = previous[k] - offset;
                        if option >= best {
                            best = option;
                            back[j] = k;
                        }
                    }
                } else {
                    best = 0.0;
                }
                score.push(best + (f64::from(row[bin]) - threshold) / temperature);
            }
            let mut best = previous[uv];
            for (j, &voiced) in previous[..uv].iter().enumerate() {
                if voiced - switch > best {
                    best = voiced - switch;
                    back[uv] = j;
                }
            }
            score.push(best + (threshold - f64::from(row[peaks[i]])) / temperature);
            previous = score;
            all_candidates.push(candidates);
            history.push(back);
        }
        let mut state = uv;
        for j in 0..uv {
            if previous[j] > previous[state] {
                state = j;
            }
        }
        let mut hz = vec![0.0; frames.len()];
        for i in (0..frames.len()).rev() {
            if state != uv {
                hz[i] = centered_pitch(&frames[i], all_candidates[i][state]);
            }
            state = history[i][state];
        }
        Ok(hz)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stable_pitch_ignores_a_brief_octave_spike_and_tracks_silence() -> Result<()> {
        let mut frames = [[0.0; BINS]; 15];
        for frame in &mut frames[3..12] {
            frame[108] = 0.8;
        }
        frames[7][168] = 0.91;
        let decoded = Decoder::default().decode(frames.as_flattened())?;
        assert_eq!(&decoded[..3], &[0.0; 3]);
        assert_eq!(&decoded[12..], &[0.0; 3]);
        assert!((decoded[7] - decoded[6]).abs() < 1e-5);
        Ok(())
    }
}
