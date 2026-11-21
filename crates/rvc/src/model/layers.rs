//! Layered-vocal augmentation for training.
//!
//! Real song vocals are rarely one clean voice: a lead is stacked with doubles,
//! harmonies and ad-libs, and sits in reverb. RVC extracts one pitch track and
//! one stream of content features from that mixture and the decoder renders
//! whatever it is given, so stacked parts come out metallic and phasey. This
//! module writes a copy of an experiment's clips where each clip carries such
//! layers. Features and F0 are then extracted from the copy, and training
//! (`--layer-aug`) feeds them to the generator while the target stays the
//! clean original, teaching it to render the lead through the stack.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use crate::audio;

/// Directory names shared with preprocessing.
const TARGET_DIR: &str = "0_gt_wavs";
const ANALYSIS_DIR: &str = "1_16k_wavs";

pub struct Report {
    pub clips: usize,
    pub doubled: usize,
    pub backed: usize,
    pub reverberated: usize,
    /// Clips that went through the production-effects chain (see [`super::fx`]).
    pub processed: usize,
    /// Clips with separation residue or instrument bleed mixed in.
    pub bled: usize,
}

/// What the augmented copy carries.
#[derive(Clone, Debug)]
pub struct Options {
    /// Doubles, backing and reverb.
    pub layers: bool,
    /// Production effects on every clip, pulled toward this long-term spectrum when
    /// one is given (see [`super::fx::profile`]).
    pub fx: Option<Option<Vec<f32>>>,
    /// Folder of short clips of what a vocal separator leaves behind in a vocal stem
    /// (see [`bleed`]); files named `inst*` are instrumentals leaking in, mixed lower.
    pub bleed: Option<PathBuf>,
}

impl Default for Options {
    fn default() -> Self {
        Self { layers: true, fx: None, bleed: None }
    }
}

/// Write layered copies of `experiment`'s target-rate clips into `output`.
pub fn generate(experiment: &Path, output: &Path, rate: u32, seed: u64) -> Result<Report> {
    generate_with(experiment, output, rate, seed, &Options::default())
}

/// [`generate`] with a choice of layers and production effects.
pub fn generate_with(experiment: &Path, output: &Path, rate: u32, seed: u64, options: &Options) -> Result<Report> {
    ensure!(
        options.layers || options.fx.is_some() || options.bleed.is_some(),
        "nothing to augment: enable layers, effects or bleed"
    );
    let bleed_clips: Vec<PathBuf> = match &options.bleed {
        Some(dir) => {
            let mut files = audio::files(dir, true)?;
            files.sort();
            ensure!(!files.is_empty(), "no bleed clips in {}", dir.display());
            files
        }
        None => Vec::new(),
    };
    let mut clips = audio::files(&experiment.join(TARGET_DIR), false)?;
    clips.retain(|path| path.extension().is_some_and(|ext| ext == "wav"));
    clips.sort();
    ensure!(clips.len() > 1, "layered augmentation needs at least two clips");
    std::fs::create_dir_all(output.join(TARGET_DIR))?;
    std::fs::create_dir_all(output.join(ANALYSIS_DIR))?;
    let flags: Vec<[bool; 5]> = clips
        .par_iter()
        .enumerate()
        .map(|(index, path)| -> Result<[bool; 5]> {
            let mut rng = StdRng::seed_from_u64(seed ^ (index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let clean = audio::read_mono(path, rate)?;
            let (mut mixed, layered) = if options.layers {
                layer(&clean, &clips, index, rate, &mut rng)?
            } else {
                (clean.clone(), [false; 3])
            };
            let mut processed = false;
            if let Some(target) = &options.fx {
                mixed = super::fx::process(&mixed, rate, &mut rng, target.as_deref()).0;
                processed = true;
            }
            let bled = !bleed_clips.is_empty() && rng.random_bool(0.6);
            if bled {
                let pick = &bleed_clips[rng.random_range(0..bleed_clips.len())];
                bleed(&mut mixed, &clean, pick, rate, &mut rng)?;
            }
            let flags = [layered[0], layered[1], layered[2], processed, bled];
            let name = path.file_name().context("clip has no file name")?;
            write(&output.join(TARGET_DIR).join(name), &mixed, rate)?;
            let analysis = audio::resample(&mixed, rate, audio::ANALYSIS_RATE);
            write(&output.join(ANALYSIS_DIR).join(name), &analysis, audio::ANALYSIS_RATE)?;
            Ok(flags)
        })
        .collect::<Result<_>>()?;
    let count = |i: usize| flags.iter().filter(|f| f[i]).count();
    Ok(Report {
        clips: clips.len(),
        doubled: count(0),
        backed: count(1),
        reverberated: count(2),
        processed: count(3),
        bled: count(4),
    })
}

fn write(path: &PathBuf, samples: &[f32], rate: u32) -> Result<()> {
    let file = std::io::BufWriter::new(std::fs::File::create(path)?);
    audio::write_wav(file, samples, rate, 1, true)
}

fn rms(x: &[f32]) -> f32 {
    (x.iter().map(|v| v * v).sum::<f32>() / x.len().max(1) as f32).sqrt()
}

fn db(value: f32) -> f32 {
    10_f32.powf(value / 20.0)
}

/// `x` replayed `ratio` times higher in pitch (and shorter).
fn shift(x: &[f32], rate: u32, ratio: f64) -> Vec<f32> {
    let to = (f64::from(rate) / ratio).round().max(1.0) as u32;
    audio::resample(x, rate, to)
}

fn layer(
    clean: &[f32],
    clips: &[PathBuf],
    index: usize,
    rate: u32,
    rng: &mut StdRng,
) -> Result<(Vec<f32>, [bool; 3])> {
    let level = rms(clean).max(1e-5);
    let mut double = rng.random_bool(0.6);
    let mut backing = rng.random_bool(0.5);
    let mut reverb = rng.random_bool(0.45);
    if !(double || backing || reverb) {
        double = true;
    }
    // Experiments: RVC_LAYER_PARTS picks the kinds of layer to use ("d", "b", "r").
    if let Ok(parts) = std::env::var("RVC_LAYER_PARTS") {
        double &= parts.contains('d');
        backing &= parts.contains('b');
        reverb &= parts.contains('r');
        if !(double || backing || reverb) {
            double = parts.contains('d');
            backing = !double && parts.contains('b');
            reverb = !double && !backing && parts.contains('r');
        }
    }
    let mut mixed = clean.to_vec();
    if double {
        // A second take: a few cents off, 10-40 ms late, a little quieter.
        let cents = rng.random_range(6.0..22.0) * if rng.random_bool(0.5) { 1.0 } else { -1.0 };
        let copy = shift(clean, rate, 2_f64.powf(cents / 1200.0));
        let delay = (rng.random_range(0.010..0.040) * f64::from(rate)) as usize;
        let gain = db(rng.random_range(-7.0..-1.0));
        for (i, value) in copy.iter().enumerate() {
            if let Some(out) = mixed.get_mut(i + delay) {
                *out += gain * value;
            }
        }
    }
    if backing {
        // Another phrase as a harmony or ad-lib, from any clip but this one.
        let mut other = rng.random_range(0..clips.len() - 1);
        if other >= index {
            other += 1;
        }
        let voice = audio::read_mono(&clips[other], rate)?;
        let semitones = [-12.0, -5.0, 0.0, 3.0, 4.0, 7.0, 12.0][rng.random_range(0..7)];
        let voice = shift(&voice, rate, 2_f64.powf(semitones / 12.0));
        let gain = db(rng.random_range(-16.0..-6.0)) * level / rms(&voice).max(1e-5);
        let offset = rng.random_range(0..mixed.len().max(1)) as isize
            - rng.random_range(0..voice.len().max(1)) as isize;
        // The backing sounds only under the lead. Left in the lead's gaps it would teach
        // the model to silence any voice somewhat quieter than the loudest phrase, which
        // is what screams, growls and rough belts are; they then come out muted.
        let under = lead_presence(clean, rate);
        for (i, value) in voice.iter().enumerate() {
            let at = i as isize + offset;
            if at >= 0
                && let Some(out) = mixed.get_mut(at as usize)
            {
                *out += gain * value * under[at as usize];
            }
        }
    }
    if reverb {
        let decay = rng.random_range(0.4..1.6);
        let wet = schroeder(&mixed, rate, decay);
        let gain = db(rng.random_range(-14.0..-5.0)) * rms(&mixed) / rms(&wet).max(1e-6);
        for (out, value) in mixed.iter_mut().zip(&wet) {
            *out += gain * value;
        }
    }
    // Keep the clean clip's peak so the level stays comparable.
    let peak = clean.iter().fold(0_f32, |m, v| m.max(v.abs()));
    let now = mixed.iter().fold(0_f32, |m, v| m.max(v.abs()));
    if now > 0.0 {
        let scale = peak.max(1e-4) / now;
        mixed.iter_mut().for_each(|v| *v *= scale);
    }
    Ok((mixed, [double, backing, reverb]))
}

/// Mix what a vocal separator leaves behind into `mixed`, over the whole clip.
///
/// Separated vocal stems carry residue: hi-hat and snare ticks, synth tails, 808 rumble,
/// effect sweeps, phasey ghosts of the beat. A model trained only on clean voices
/// renders all of it as voice (random "sung" noises in pauses, a buzz under words).
/// Here a random stretch of `path` (a residue clip, or an instrumental for files named
/// `inst*`) goes under the whole input, pauses included, while the target stays the clean
/// clip, so the generator learns that this is not voice and stays silent on it. Residue
/// sits 8-26 dB under the voice, instrument leakage 20-34 dB under it.
fn bleed(mixed: &mut [f32], clean: &[f32], path: &Path, rate: u32, rng: &mut StdRng) -> Result<()> {
    let noise = audio::read_mono(path, rate)?;
    if noise.is_empty() {
        return Ok(());
    }
    let start = rng.random_range(0..noise.len());
    let instrument = path
        .file_name()
        .and_then(|n| n.to_str())
        .is_some_and(|n| n.starts_with("inst"));
    let db_under = if instrument { rng.random_range(20.0..34.0) } else { rng.random_range(8.0..26.0) };
    // Level against the voiced part of the clip, so long pauses do not lower it.
    let under = lead_presence(clean, rate);
    let (sum, n) = clean
        .iter()
        .zip(&under)
        .filter(|(_, w)| **w > 0.5)
        .fold((0_f64, 0usize), |(s, n), (v, _)| (s + f64::from(v * v), n + 1));
    let voice = if n > 0 { (sum / n as f64).sqrt() as f32 } else { rms(clean) }.max(1e-5);
    let level = rms(&noise).max(1e-6);
    let gain = voice * db(-db_under) / level;
    for (i, out) in mixed.iter_mut().enumerate() {
        *out += gain * noise[(start + i) % noise.len()];
    }
    Ok(())
}

/// Per-sample weight in [0, 1] of where `clean` carries the lead: 1 within 20 dB of
/// its loud part, fading to 0 at 30 dB under it, from 20 ms RMS frames smoothed over
/// about 60 ms so the backing fades in and out instead of switching.
fn lead_presence(clean: &[f32], rate: u32) -> Vec<f32> {
    let hop = (rate as usize / 50).max(1);
    let frames = clean.len().div_ceil(hop).max(1);
    let level: Vec<f32> = (0..frames)
        .map(|t| {
            let chunk = &clean[t * hop..((t + 1) * hop).min(clean.len())];
            20.0 * rms(chunk).max(1e-7).log10()
        })
        .collect();
    let mut sorted = level.clone();
    sorted.sort_by(f32::total_cmp);
    let top = sorted[sorted.len() * 95 / 100];
    let raw: Vec<f32> = level.iter().map(|l| ((l - (top - 30.0)) / 10.0).clamp(0.0, 1.0)).collect();
    let smooth: Vec<f32> = (0..frames)
        .map(|t| {
            let around = t.saturating_sub(1)..(t + 2).min(frames);
            let n = around.len() as f32;
            raw[around].iter().sum::<f32>() / n
        })
        .collect();
    (0..clean.len())
        .map(|i| {
            let x = i as f32 / hop as f32 - 0.5;
            let a = (x.floor().max(0.0) as usize).min(frames - 1);
            let b = (a + 1).min(frames - 1);
            let w = (x - a as f32).clamp(0.0, 1.0);
            smooth[a] * (1.0 - w) + smooth[b] * w
        })
        .collect()
}

/// A plain algorithmic reverb tail (wet only) for `x` at `rate`.
pub fn reverb(x: &[f32], rate: u32, rt60: f64) -> Vec<f32> {
    schroeder(x, rate, rt60)
}

/// Schroeder reverberator: four parallel combs into two allpasses, with a
/// feedback chosen so the tail falls 60 dB in `rt60` seconds.
fn schroeder(x: &[f32], rate: u32, rt60: f64) -> Vec<f32> {
    let scale = f64::from(rate) / 44_100.0;
    let combs = [1557.0, 1617.0, 1491.0, 1422.0].map(|n: f64| (n * scale) as usize);
    let mut out = vec![0_f32; x.len()];
    for delay in combs {
        let gain = 10_f64.powf(-3.0 * delay as f64 / (rt60 * f64::from(rate))) as f32;
        let mut line = vec![0_f32; delay];
        let mut low = 0_f32;
        for (i, &input) in x.iter().enumerate() {
            let slot = i % delay;
            let delayed = line[slot];
            // A gentle low-pass in the loop makes high frequencies die first.
            low = 0.7 * delayed + 0.3 * low;
            line[slot] = input + gain * low;
            out[i] += delayed * 0.25;
        }
    }
    for delay in [556.0, 441.0].map(|n: f64| (n * scale) as usize) {
        let mut line = vec![0_f32; delay];
        for (i, value) in out.iter_mut().enumerate() {
            let slot = i % delay;
            let delayed = line[slot];
            let input = *value;
            line[slot] = input + 0.5 * delayed;
            *value = delayed - 0.5 * input;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reverb_tail_decays() {
        let mut impulse = vec![0_f32; 80_000];
        impulse[0] = 1.0;
        let tail = schroeder(&impulse, 40_000, 0.5);
        let energy = |range: std::ops::Range<usize>| tail[range].iter().map(|v| v * v).sum::<f32>();
        assert!(energy(0..10_000) > 100.0 * energy(30_000..40_000));
    }

    #[test]
    fn backing_is_kept_out_of_the_lead_gaps() {
        // Lead sings for 1 s, rests 1 s, sings 1 s.
        let rate = 40_000;
        let clean: Vec<f32> = (0..3 * rate as usize)
            .map(|i| {
                let on = !(rate as usize..2 * rate as usize).contains(&i);
                if on { (i as f32 * 0.05).sin() * 0.3 } else { 0.0 }
            })
            .collect();
        let under = lead_presence(&clean, rate);
        let mean = |r: std::ops::Range<usize>| under[r.clone()].iter().sum::<f32>() / r.len() as f32;
        assert!(mean(1_200..38_000) > 0.99);
        assert!(mean(44_000..76_000) < 0.01);
        assert!(mean(84_000..118_000) > 0.99);
    }

    #[test]
    fn bleed_fills_the_pauses_below_the_voice() {
        let rate = 40_000;
        let dir = std::env::temp_dir().join(format!("rvc-bleed-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let tick: Vec<f32> = (0..rate as usize).map(|i| if i % 4_000 < 200 { 0.5 } else { 0.0 }).collect();
        let path = dir.join("residue.wav");
        write(&path, &tick, rate).unwrap();
        // Voice for 1 s, then a 1 s pause.
        let clean: Vec<f32> = (0..2 * rate as usize)
            .map(|i| if i < rate as usize { (i as f32 * 0.05).sin() * 0.3 } else { 0.0 })
            .collect();
        let mut mixed = clean.clone();
        let mut rng = StdRng::seed_from_u64(1);
        bleed(&mut mixed, &clean, &path, rate, &mut rng).unwrap();
        let added: Vec<f32> = mixed.iter().zip(&clean).map(|(m, c)| m - c).collect();
        let pause = rms(&added[rate as usize..]);
        assert!(pause > 1e-4, "the pause got no residue");
        let voice = rms(&clean[..rate as usize]);
        assert!(rms(&added) < voice * 0.5, "residue louder than allowed");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn pitch_shift_changes_length_by_ratio() {
        let x = vec![0.1_f32; 40_000];
        let y = shift(&x, 40_000, 2.0);
        assert!((y.len() as i64 - 20_000).abs() < 10);
    }
}
