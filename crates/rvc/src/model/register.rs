// Register matching and pitch profile format adapted from NVC Studio (MIT).
use std::collections::BTreeMap;
use std::fs::File;
use std::ops::{ControlFlow, Range};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use tch::{Device, Kind, no_grad};

use super::{Pitch, PitchMethod};
use crate::audio::{self, ANALYSIS_RATE, FRAME};
use crate::catalog::Paths;
use crate::checkpoint::Checkpoint;

const QUANTILES: [f32; 7] = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95];
const WEIGHTS: [f32; 7] = [0.8, 1.0, 1.4, 2.2, 1.4, 1.0, 0.8];
const SHIFTS: [f32; 5] = [-24.0, -12.0, 0.0, 12.0, 24.0];
const MIN_HZ: f32 = 50.0;
const MAX_HZ: f32 = 1100.0;
const BINS: usize = 4096;

#[derive(Deserialize, Serialize)]
pub(crate) struct Profile {
    #[serde(default)]
    name: String,
    #[serde(default)]
    voiced_frames: u64,
    #[serde(default)]
    audio_files: usize,
    #[serde(alias = "quantiles")]
    midi_quantiles: BTreeMap<String, f32>,
}

impl Profile {
    pub(crate) fn load(model: &Path) -> Result<Option<Self>> {
        let sidecar = model.with_extension("pitch.json");
        if sidecar.is_file() {
            return Self::read(&sidecar).map(Some);
        }
        if let Some(value) = Checkpoint::open(model)?.metadata("pitch_profile")?
            && let Ok(profile) = serde_json::from_value::<Self>(value)
            && profile.distribution().is_ok()
        {
            return Ok(Some(profile));
        }

        let directory = model.parent().context("model has no directory")?;
        for path in [
            directory.join("pitch-profile.json"),
            directory.join("pitch_profile.json"),
        ] {
            if path.is_file() {
                return Self::read(&path).map(Some);
            }
        }
        Ok(None)
    }

    pub(crate) fn read(path: &Path) -> Result<Self> {
        let profile: Self = serde_json::from_reader(File::open(path)?)
            .with_context(|| format!("reading pitch profile {}", path.display()))?;
        profile.distribution()?;
        Ok(profile)
    }

    fn distribution(&self) -> Result<[f32; 7]> {
        let mut points: Vec<_> = self
            .midi_quantiles
            .iter()
            .filter_map(|(key, &value)| {
                let quantile: f32 = key.parse().ok()?;
                ((0.0 < quantile && quantile < 1.0) && value.is_finite())
                    .then_some((quantile, value))
            })
            .collect();
        points.sort_by(|(a, _), (b, _)| a.total_cmp(b));
        ensure!(
            points.len() >= 3
                && points.iter().any(|(p, _)| (p - 0.5).abs() < 1e-6)
                && points
                    .windows(2)
                    .all(|pair| pair[0].0 < pair[1].0 && pair[0].1 <= pair[1].1),
            "pitch profile must contain monotonic MIDI quantiles including the median"
        );
        Ok(QUANTILES.map(|q| {
            let i = points.partition_point(|(p, _)| *p < q);
            if i == 0 {
                points[0].1
            } else if i == points.len() {
                points[i - 1].1
            } else {
                let (lo, low) = points[i - 1];
                let (hi, high) = points[i];
                low + (high - low) * (q - lo) / (hi - lo)
            }
        }))
    }
}

pub(crate) fn build(
    paths: &Paths,
    model: &Path,
    source: &Path,
    method: PitchMethod,
    mut progress: impl FnMut(&str) -> ControlFlow<()>,
) -> Result<(PathBuf, Profile)> {
    let files = if source.is_file() {
        vec![source.to_owned()]
    } else {
        audio::files(source, true)?
    };
    ensure!(!files.is_empty(), "no audio files for the pitch profile");
    let mut histogram = [0u64; BINS];
    let min = midi(MIN_HZ);
    let width = (midi(MAX_HZ) - min) / BINS as f32;
    let mut voiced_frames = 0;
    let pitch = no_grad(|| Pitch::load(method, paths, Device::Cpu, Kind::Float))?;
    for (index, file) in files.iter().enumerate() {
        let message = format!("{} / {} · {}", index + 1, files.len(), file.display());
        ensure!(progress(&message).is_continue(), "pitch profile cancelled");
        let audio = audio::read_mono(file, ANALYSIS_RATE)?;
        for chunk in audio.chunks(20 * ANALYSIS_RATE as usize) {
            ensure!(progress(&message).is_continue(), "pitch profile cancelled");
            if chunk.len() <= 512 {
                continue;
            }
            let mut pitch = no_grad(|| pitch.extract(chunk, None))?;
            clean_and_tune(&mut pitch, false);
            for hz in pitch.into_iter().filter(|&hz| voiced(hz)) {
                let bin = (((midi(hz) - min) / width) as usize).min(BINS - 1);
                histogram[bin] += 1;
                voiced_frames += 1;
            }
        }
    }
    ensure!(
        voiced_frames >= 100,
        "need at least one second of voiced audio for a pitch profile"
    );

    let mut start = 0;
    let mut cumulative = 0;
    let mut midi_quantiles = BTreeMap::new();
    for q in [
        0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.98, 0.99, 0.995,
    ] {
        let rank = q * (voiced_frames - 1) as f32;
        while (cumulative + histogram[start]) as f32 <= rank && start < BINS - 1 {
            cumulative += histogram[start];
            start += 1;
        }
        let fraction = (rank - cumulative as f32) / histogram[start].max(1) as f32;
        midi_quantiles.insert(q.to_string(), min + (start as f32 + fraction) * width);
    }
    let profile = Profile {
        name: model
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned(),
        voiced_frames,
        audio_files: files.len(),
        midi_quantiles,
    };
    let output = model.with_extension("pitch.json");
    let mut tmp = tempfile::NamedTempFile::new_in(output.parent().unwrap_or(Path::new(".")))?;
    serde_json::to_writer_pretty(&mut tmp, &profile)?;
    ensure!(
        progress("Saving pitch profile").is_continue(),
        "pitch profile cancelled"
    );
    tmp.persist(&output)?;
    Ok((output, profile))
}

fn voiced(hz: f32) -> bool {
    (MIN_HZ..=MAX_HZ).contains(&hz)
}

fn midi(hz: f32) -> f32 {
    69.0 + 12.0 * (hz / 440.0).log2()
}

fn hz(midi: f32) -> f32 {
    440.0 * ((midi - 69.0) / 12.0).exp2()
}

/// Contiguous voiced runs with short gaps treated as part of the same phrase.
fn phrases(pitch: &[f32], max_gap: usize) -> Vec<Range<usize>> {
    let mut result: Vec<Range<usize>> = Vec::new();
    for (index, _) in pitch.iter().enumerate().filter(|(_, hz)| voiced(**hz)) {
        if let Some(last) = result.last_mut()
            && index - last.end <= max_gap
        {
            last.end = index + 1;
        } else {
            result.push(index..index + 1);
        }
    }
    result
}

fn median_filter<const SIZE: usize>(values: &[f32], index: usize) -> f32 {
    let mut window = std::array::from_fn::<_, SIZE, _>(|offset| {
        values[(index + offset)
            .saturating_sub(SIZE / 2)
            .min(values.len() - 1)]
    });
    window.sort_unstable_by(f32::total_cmp);
    window[SIZE / 2]
}

fn smooth(values: &[f32], seconds: f32) -> Vec<f32> {
    let alpha = 1.0 - (-1.0 / (100.0 * seconds)).exp();
    let filter = |previous: &mut f32, &value: &f32| {
        *previous += alpha * (value - *previous);
        Some(*previous)
    };
    let mut forward: Vec<_> = values.iter().scan(values[0], filter).collect();
    for (forward, backward) in forward
        .iter_mut()
        .rev()
        .zip(values.iter().rev().scan(values[values.len() - 1], filter))
    {
        *forward = (*forward + backward) * 0.5;
    }
    forward
}

fn clean_and_tune(pitch: &mut [f32], tune: bool) {
    for sample in pitch.iter_mut().filter(|hz| !voiced(**hz)) {
        *sample = 0.0;
    }
    for phrase in phrases(pitch, 3) {
        let run = &mut pitch[phrase];
        if run.len() < 3 {
            continue;
        }
        let mut notes = Vec::with_capacity(run.len());
        let mut last = (0, run[0]);
        for (index, &hz) in run.iter().enumerate().skip(1).filter(|(_, hz)| **hz > 0.0) {
            let (lo, left) = last;
            for i in lo..index {
                notes.push(midi(
                    left + (hz - left) * (i - lo) as f32 / (index - lo) as f32,
                ));
            }
            last = (index, hz);
        }
        notes.push(midi(last.1));

        let mut corrected = Vec::with_capacity(notes.len());
        for (index, &value) in notes.iter().enumerate() {
            let middle = median_filter::<5>(&notes, index);
            let diff = value - middle;
            let octave = value - (diff / 12.0).round() * 12.0;
            corrected.push(if diff.abs() > 7.0 && (octave - middle).abs() < 3.0 {
                octave
            } else {
                value
            });
        }
        for (index, frame) in corrected.windows(3).enumerate() {
            let [before, center, after] = [frame[0], frame[1], frame[2]];
            notes[index + 1] = if (center - before).abs() > 4.0
                && (center - after).abs() > 4.0
                && (before - after).abs() < 1.5
            {
                (before + after) * 0.5
            } else {
                center
            };
        }
        notes[0] = corrected[0];
        *notes.last_mut().unwrap() = *corrected.last().unwrap();

        if tune {
            let medians: Vec<_> = (0..notes.len())
                .map(|i| median_filter::<9>(&notes, i))
                .collect();
            let centers = smooth(&medians, 0.035);
            let mut note = centers[0].round();
            let correction: Vec<_> = centers
                .iter()
                .map(|&center| {
                    if (center - note).abs() >= 0.62 {
                        note = center.round();
                    }
                    (note - center).clamp(-0.65, 0.65)
                })
                .collect();
            let length = notes.len();
            let fade = 4.min(length / 2);
            for (i, (note, correction)) in
                notes.iter_mut().zip(smooth(&correction, 0.080)).enumerate()
            {
                let edge = i.min(length - i - 1);
                let strength = if edge < fade {
                    0.25 + 0.75 * edge as f32 / (fade - 1).max(1) as f32
                } else {
                    1.0
                };
                *note += strength * correction;
            }
        }
        for (sample, note) in run.iter_mut().zip(notes) {
            if *sample != 0.0 {
                *sample = hz(note);
            }
        }
    }
}

fn distribution(pitch: &[f32], manual: f32) -> (usize, Option<[f32; 7]>) {
    let mut notes: Vec<_> = pitch
        .iter()
        .copied()
        .filter(|&hz| voiced(hz))
        .map(midi)
        .collect();
    let frames = notes.len();
    if frames < 8 {
        return (frames, None);
    }
    notes.sort_unstable_by(f32::total_cmp);
    let values = QUANTILES.map(|p| {
        let rank = p * (frames - 1) as f32;
        notes[rank.floor() as usize] * (1.0 - rank.fract())
            + notes[rank.ceil() as usize] * rank.fract()
            + manual
    });
    (frames, Some(values))
}

fn cost(source: [f32; 7], target: [f32; 7], shift: f32) -> f32 {
    let distribution: f32 = (0..7)
        .map(|i| (source[i] + shift - target[i]).abs() * WEIGHTS[i])
        .sum::<f32>()
        / WEIGHTS.iter().sum::<f32>();
    let overflow = (target[0] - 2.0 - source[0] - shift).max(0.0)
        + (source[6] + shift - target[6] - 2.0).max(0.0);
    distribution + 1.4 * overflow + 0.1 * shift.abs() / 12.0
}

fn lowest(costs: [f32; 5]) -> usize {
    (0..costs.len())
        .min_by(|&a, &b| costs[a].total_cmp(&costs[b]))
        .unwrap_or(2)
}

/// Keep the song key by changing octaves only and switching across vocal pauses.
pub(super) fn match_pitch(pitch: &mut [f32], manual: f32, profile: Option<&Profile>) -> Result<()> {
    clean_and_tune(pitch, true);
    let manual_ratio = (manual / 12.0).exp2();
    let (_, global) = distribution(pitch, manual);
    let Some(global) = global else {
        pitch.iter_mut().for_each(|hz| *hz *= manual_ratio);
        return Ok(());
    };
    let target = match profile {
        Some(profile) => profile.distribution()?,
        None => global.map(|note| note - global[3] + midi(155.0)),
    };
    let anchor = lowest(SHIFTS.map(|shift| cost(global, target, shift)));
    let runs = phrases(pitch, 13);
    let mut backpointers = Vec::with_capacity(runs.len());
    let mut costs = SHIFTS.map(|shift| 3.0 * (shift - SHIFTS[anchor]).abs() / 12.0);
    for (i, run) in runs.iter().enumerate() {
        let (frames, source) = distribution(&pitch[run.clone()], manual);
        let emission = source.map_or([0.0; 5], |source| {
            let baseline = cost(source, target, 0.0);
            let advantage = 6.0 * (frames as f32 / 100.0).min(1.0);
            SHIFTS.map(|shift| {
                let local = cost(source, target, shift).max(baseline - advantage);
                let uncertain = if shift != 0.0 && baseline - local < 1.0 {
                    2.0
                } else {
                    0.0
                };
                local + uncertain + 0.2 * (shift - SHIFTS[anchor]).abs() / 12.0
            })
        });
        let transition_weight = if i == 0 {
            // The initial costs are already anchored.
            f32::INFINITY
        } else if (run.start - runs[i - 1].end) * FRAME >= ANALYSIS_RATE as usize * 2 / 5 {
            1.0
        } else {
            4.0
        };
        let mut previous = [0; 5];
        costs = std::array::from_fn(|j| {
            if i == 0 {
                return costs[j] + emission[j];
            }
            let candidates =
                std::array::from_fn(|k| costs[k] + transition_weight * j.abs_diff(k) as f32);
            previous[j] = lowest(candidates);
            candidates[previous[j]] + emission[j]
        });
        backpointers.push(previous);
    }
    let mut selected = lowest(std::array::from_fn(|i| {
        costs[i] + 3.0 * i.abs_diff(anchor) as f32
    }));
    for (run, previous) in runs.into_iter().zip(backpointers).rev() {
        let ratio = manual_ratio * (SHIFTS[selected] / 12.0).exp2();
        pitch[run].iter_mut().for_each(|hz| *hz *= ratio);
        selected = previous[selected];
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_matching_preserves_silence_and_moves_an_octave() -> Result<()> {
        let mut profile = Profile {
            name: "test".into(),
            voiced_frames: 100,
            audio_files: 1,
            midi_quantiles: [
                ("0.05".into(), 55.0),
                ("0.5".into(), 55.0),
                ("0.95".into(), 55.0),
            ]
            .into(),
        };
        let mut pitch = [0.0; 250];
        pitch[20..220].fill(hz(67.0));
        match_pitch(&mut pitch, 0.0, Some(&profile))?;
        assert!((midi(pitch[100]) - 55.0).abs() < 0.01);
        assert_eq!(pitch[0], 0.0);
        assert_eq!(pitch[249], 0.0);
        profile.midi_quantiles.insert("0.95".into(), 50.0);
        assert!(profile.distribution().is_err());
        Ok(())
    }
}
