//! Prepare aligned alternate inputs once, before entering the training graph.
//! Sources, pitch labels and speaker labels in the experiment remain the supervision.
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io;
use std::ops::ControlFlow;
use std::path::{Path, PathBuf};

use crate::model::{
    ConversionSettings, PitchMethod, Rvc, edit, extract,
    hubert::Hubert,
    train_data::{Config, Version},
    weights,
};
use crate::{
    assets, audio,
    catalog::{Embedder, Paths},
    checkpoint::Checkpoint,
};
use anyhow::{Context, Result, ensure};
use rustfft::{FftPlanner, num_complex::Complex32};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tch::{Device, Kind};

const DONOR_AND_FORMANT: u8 = 1;
const FORMANT: u8 = 2;
const SPEAKERS: [i64; 4] = [4, 32, 68, 97];
const WARPS: [f32; 4] = [0.78, 0.85, 1.14, 1.25];

pub(in crate::model) fn version(checkpoint: &Checkpoint) -> Result<u8> {
    let Some(value) = checkpoint.metadata("paired_content")? else {
        return Ok(0);
    };
    match value.as_u64().or_else(|| value.as_bool().map(u64::from)) {
        Some(0) => Ok(0),
        Some(1) => Ok(DONOR_AND_FORMANT),
        Some(2) => Ok(FORMANT),
        _ => anyhow::bail!("unsupported paired_content recipe {value}"),
    }
}

fn hash(path: &Path) -> Result<String> {
    let mut digest = Sha256::new();
    io::copy(
        &mut File::open(path).with_context(|| format!("opening {}", path.display()))?,
        &mut digest,
    )?;
    Ok(format!("{:x}", digest.finalize()))
}

#[derive(Serialize, Deserialize)]
struct Entry {
    source_sha256: String,
    speaker: Option<i64>,
    warp: f32,
    wave_sha256: String,
    features_sha256: String,
}

/// Replicate the two draws made by NumPy's SeedSequence + PCG64 for a 32-bit
/// filename seed. This keeps the donor and warp choices identical to the paired corpus.
fn choices(seed: u32) -> (Option<i64>, f32) {
    let mut hash_constant = 0x43b0d7e5_u32;
    let mut mix_seed = |value: u32| {
        let x = value ^ hash_constant;
        hash_constant = hash_constant.wrapping_mul(0x931e8875);
        let y = x.wrapping_mul(hash_constant);
        y ^ (y >> 16)
    };
    let mut pool = [seed, 0, 0, 0].map(&mut mix_seed);
    for i in 0..4 {
        for j in 0..4 {
            if i != j {
                let y = pool[j]
                    .wrapping_mul(0xca01f9dd)
                    .wrapping_sub(mix_seed(pool[i]).wrapping_mul(0x4973f715));
                pool[j] = y ^ (y >> 16);
            }
        }
    }
    let mut hash_constant = 0x8b51f9dd_u32;
    let mut words = [0_u32; 8];
    for (i, word) in words.iter_mut().enumerate() {
        let x = pool[i % 4] ^ hash_constant;
        hash_constant = hash_constant.wrapping_mul(0x58f38ded);
        let y = x.wrapping_mul(hash_constant);
        *word = y ^ (y >> 16);
    }
    let get64 = |i| u128::from(words[i]) | (u128::from(words[i + 1]) << 32);
    let initial = (get64(0) << 64) | get64(2);
    let increment = (((get64(4) << 64) | get64(6)) << 1) | 1;
    let step = |x: u128| {
        x.wrapping_mul(0x2360ed051fc65da44385df649fccf645)
            .wrapping_add(increment)
    };
    let mut state = step(increment.wrapping_add(initial));
    let mut draw = || {
        state = step(state);
        ((state ^ (state >> 64)) as u64).rotate_right((state >> 122) as u32)
    };
    let uniform = (draw() >> 11) as f64 / ((1_u64 << 53) as f64);
    let warp = (draw() as u32 >> 30) as usize;
    (
        (uniform < 0.58).then(|| SPEAKERS[(seed % 4) as usize]),
        WARPS[warp],
    )
}

struct Warp {
    forward: std::sync::Arc<dyn rustfft::Fft<f32>>,
    inverse: std::sync::Arc<dyn rustfft::Fft<f32>>,
    window: Vec<f32>,
    gaussian: Vec<f32>,
}

impl Warp {
    fn new() -> Self {
        let mut planner = FftPlanner::new();
        let mut gaussian: Vec<f32> = (-20..=20)
            .map(|x| (-0.5 * (x as f32 / 5.0).powi(2)).exp())
            .collect();
        let sum: f32 = gaussian.iter().sum();
        gaussian.iter_mut().for_each(|v| *v /= sum);
        Self {
            forward: planner.plan_fft_forward(1024),
            inverse: planner.plan_fft_inverse(1024),
            window: (0..1024)
                .map(|i| (std::f32::consts::PI * i as f32 / 1024.0).sin().powi(2))
                .collect(),
            gaussian,
        }
    }

    /// Move the smooth log-magnitude envelope. Bin phases and the time grid stay intact.
    fn apply(&self, input: &[f32], target: &[f32], factor: f32) -> Vec<f32> {
        let n = 1024;
        let hop = 256;
        let mut spectrum = vec![Complex32::default(); n];
        let mut scratch = vec![
            Complex32::default();
            self.forward
                .get_inplace_scratch_len()
                .max(self.inverse.get_inplace_scratch_len())
        ];
        let mut amplitude = [0.0_f32; 513];
        let mut env = [0.0_f32; 513];
        let mut output = vec![0.0; input.len()];
        let mut norm = vec![0.0; input.len()];
        for c in (0..=input.len().div_ceil(hop) * hop).step_by(hop) {
            let start = c.saturating_sub(n / 2);
            let skip = (n / 2).saturating_sub(c);
            let count = (n - skip).min(input.len().saturating_sub(start));
            spectrum.fill(Complex32::default());
            for j in 0..count {
                spectrum[skip + j].re = input[start + j] * self.window[skip + j];
            }
            self.forward
                .process_with_scratch(&mut spectrum, &mut scratch);
            for (dest, bin) in amplitude.iter_mut().zip(&spectrum) {
                *dest = (bin.norm() / 512.0 + 1e-7).ln();
            }
            for (k, value) in env.iter_mut().enumerate() {
                *value = self
                    .gaussian
                    .iter()
                    .enumerate()
                    .map(|(j, w)| {
                        let index = (k as isize + j as isize - 20).rem_euclid(1026) as usize;
                        w * amplitude[index.min(1025 - index)]
                    })
                    .sum();
            }
            for k in 0..=n / 2 {
                let source = (k as f32 / factor).min(512.0);
                let low = source as usize;
                let frac = source - low as f32;
                let moved = env[low] + frac * (env[(low + 1).min(512)] - env[low]);
                spectrum[k] *= (moved - env[k]).clamp(-1.6, 1.6).exp();
                if k != 0 && k != n / 2 {
                    spectrum[n - k] = spectrum[k].conj();
                }
            }
            self.inverse
                .process_with_scratch(&mut spectrum, &mut scratch);
            for j in 0..count {
                let win = self.window[j + skip];
                output[start + j] += spectrum[j + skip].re * win / n as f32;
                norm[start + j] += win * win;
            }
        }
        for (s, w) in output.iter_mut().zip(norm) {
            *s /= w.max(1e-12);
        }
        let rms = |a: &[f32]| a.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>().sqrt();
        let peak = output.iter().copied().map(f32::abs).fold(0.0, f32::max);
        let scale =
            (rms(target) / rms(&output).max(1e-8)).min(0.98 / f64::from(peak.max(1e-8))) as f32;
        output.iter_mut().for_each(|v| *v *= scale);
        output
    }
}

/// The paired corpus uses scipy.signal.resample_poly with its default Kaiser(5)
/// low-pass and zero extension before quantizing the 16 kHz feature input.
fn analysis_wave(samples: &[f32], rate: u32) -> Vec<f32> {
    let gcd = |mut a: usize, mut b: usize| {
        while b != 0 {
            (a, b) = (b, a % b);
        }
        a
    };
    let divisor = gcd(rate as usize, audio::ANALYSIS_RATE as usize);
    let up = audio::ANALYSIS_RATE as usize / divisor;
    let down = rate as usize / divisor;
    let max = up.max(down);
    let half = 10 * max;
    let i0 = |x: f64| {
        let (mut term, mut sum) = (1.0, 1.0);
        for k in 1..=24 {
            term *= (x / (2 * k) as f64).powi(2);
            sum += term;
        }
        sum
    };
    let taps: Vec<_> = (0..=2 * half)
        .map(|k| {
            let d = k as f64 - half as f64;
            let theta = std::f64::consts::PI * d / max as f64;
            let sinc = if k == half { 1.0 } else { theta.sin() / theta };
            sinc * i0(5.0 * (1.0 - (d / half as f64).powi(2)).max(0.0).sqrt())
        })
        .collect();
    let gain = up as f64 / taps.iter().sum::<f64>();
    (0..(samples.len() * up).div_ceil(down))
        .map(|n| {
            let center = n * down + half;
            let first = center.saturating_sub(2 * half).div_ceil(up);
            let end = (center / up + 1).min(samples.len());
            let value: f64 = (first..end)
                .map(|j| f64::from(samples[j]) * taps[center - j * up])
                .sum();
            ((value * gain * 32768.0).floor().clamp(-32768.0, 32767.0) / 32768.0) as f32
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
pub(super) fn prepare(
    paths: &Paths,
    experiment: &Path,
    recipe: u8,
    sources: BTreeSet<PathBuf>,
    version: Version,
    embedder: Embedder,
    rate: u32,
    device: Device,
    progress: &mut impl FnMut(&str) -> ControlFlow<()>,
) -> Result<BTreeMap<PathBuf, (PathBuf, PathBuf)>> {
    let root = experiment.join("paired-content");
    fs::create_dir_all(&root)?;
    ensure!(
        matches!(recipe, DONOR_AND_FORMANT | FORMANT),
        "unsupported paired_content recipe {recipe}"
    );
    let feature_encoder = embedder.install(paths, &mut *progress)?;
    let mut directories = vec![feature_encoder.clone()];
    let mut resources = BTreeMap::new();
    let stock = if recipe == DONOR_AND_FORMANT {
        directories.push(Embedder::HubertBase.install(paths, &mut *progress)?);
        assets::pitch(paths, PitchMethod::Rmvpe, progress)?;
        let rmvpe = paths.rmvpe.join("rmvpe.pt");
        resources.insert(rmvpe.strip_prefix(&paths.root)?.to_owned(), hash(&rmvpe)?);
        Some(assets::paired_donor(paths, &root, progress)?)
    } else {
        None
    };
    for directory in directories {
        for name in [
            "config.json",
            "preprocessor_config.json",
            "pytorch_model.bin",
            "model.safetensors",
        ] {
            let path = directory.join(name);
            let key = path.strip_prefix(&paths.root)?;
            if path.is_file() && !resources.contains_key(key) {
                resources.insert(key.to_owned(), hash(&path)?);
            }
        }
    }
    let key = serde_json::to_vec(&serde_json::json!({
        "format": 1, "paired_content": recipe, "donor_sha256": stock.as_ref().map(|(_, sha)|sha),
        "embedder": embedder, "version": version, "rate": rate, "resources": resources,
    }))?;
    let cache = root.join(format!("{:x}", Sha256::digest(&key)));
    fs::create_dir_all(&cache)?;
    let manifest_path = cache.join("manifest.json");
    audio::write_file(&cache.join("resources.json"), &key, true)?;
    let settings = ConversionSettings {
        index_rate: 0.0,
        aperiodic: Some(0.0),
        rms_mix_rate: 0.25,
        ..Default::default()
    };
    let mut donor = None;
    let mut encoder = None;
    let warp = Warp::new();
    let scratch = tempfile::tempdir_in(&root)?;
    let kind = if matches!(device,Device::Cuda(index) if weights::cuda_profile(index).is_none_or(|v|v.fp16))
    {
        Kind::Half
    } else {
        Kind::Float
    };
    let total = sources.len();
    let mut manifest = BTreeMap::new();
    let mut pairs = BTreeMap::new();
    tch::no_grad(|| -> Result<()> {
        for (idx, source) in sources.into_iter().enumerate() {
            let name = source
                .file_name()
                .and_then(|s| s.to_str())
                .context("training WAV needs a UTF-8 filename")?;
            ensure!(
                progress(&format!(
                    "Preparing paired content {}/{}: {}",
                    idx + 1,
                    total,
                    name
                ))
                .is_continue(),
                "paired content preparation cancelled"
            );
            let seed = Sha256::digest(name.as_bytes());
            let seed = u32::from_le_bytes(seed[..4].try_into()?);
            let (speaker, factor) = choices(seed);
            let speaker = speaker.filter(|_| recipe == DONOR_AND_FORMANT);
            let source_sha = hash(&source)?;
            let id = format!(
                "{:x}",
                Sha256::digest(serde_json::to_vec(&(&source_sha, name))?)
            );
            let wave = cache.join(format!("{id}.wav"));
            let phone = cache.join(format!("{id}.npy"));
            let marker = cache.join(format!("{id}.json"));
            let previous: Option<Entry> = if marker.is_file() {
                Some(serde_json::from_reader(File::open(&marker)?)?)
            } else {
                None
            };
            let cached = match &previous {
                Some(e)
                    if e.source_sha256 == source_sha
                        && e.speaker == speaker
                        && e.warp == factor
                        && wave.is_file()
                        && phone.is_file() =>
                {
                    hash(&wave)? == e.wave_sha256 && hash(&phone)? == e.features_sha256
                }
                _ => false,
            };
            if !cached {
                let clean = audio::read_mono(&source, rate)?;
                let driven = if let Some(sid) = speaker {
                    if donor.is_none() {
                        let (stock_path, _) = stock.as_ref().context("missing paired donor")?;
                        let mut ck = Checkpoint::open(stock_path)?;
                        let mut config = Config::load(scratch.path(), Version::V2, 40000)?;
                        config.embedder_model = Some(Embedder::HubertBase);
                        let compact = scratch.path().join("donor.pth");
                        let state = ck.weights(Some("model"), Device::Cpu)?;
                        ensure!(
                            state
                                .get("emb_g.weight")
                                .is_some_and(|t| t.size()[0] == 109),
                            "official paired donor needs its 109-speaker table"
                        );
                        config.model.spk_embed_dim = 109;
                        edit::compact(
                            &compact,
                            &config,
                            state,
                            edit::Export {
                                version: Version::V2,
                                sample_rate: 40000,
                                uses_f0: true,
                                description: "Stock RVC paired training donor",
                                overwrite: false,
                            },
                        )?;
                        donor = Some(Rvc::load(
                            paths,
                            &compact,
                            Some(sid),
                            None,
                            settings,
                            device,
                        )?);
                    }
                    let engine = donor.as_mut().context("paired donor unavailable")?;
                    engine.speaker = sid;
                    tch::manual_seed(i64::from(seed));
                    let input = audio::read_mono(&source, audio::ANALYSIS_RATE)?;
                    let result = engine.convert(&input, settings)?;
                    let mut samples = audio::resample(&result.samples, result.rate, rate);
                    samples.resize(clean.len(), 0.0);
                    samples
                } else {
                    clean.clone()
                };
                let augmented = warp.apply(&driven, &clean, factor);
                audio::write_file(&wave, &audio::encode_wav(&augmented, rate, false)?, true)?;
                if encoder.is_none() {
                    encoder = Some(Hubert::load(&feature_encoder, device, kind, true)?);
                }
                let analysis = analysis_wave(&augmented, rate);
                extract::write_features(
                    encoder.as_ref().context("paired encoder unavailable")?,
                    &analysis,
                    version == Version::V2,
                    version.dimensions() as usize,
                    &phone,
                )?;
            }
            let entry = if cached {
                previous.context("missing paired cache record")?
            } else {
                let entry = Entry {
                    source_sha256: source_sha,
                    speaker,
                    warp: factor,
                    wave_sha256: hash(&wave)?,
                    features_sha256: hash(&phone)?,
                };
                audio::write_file(&marker, &serde_json::to_vec(&entry)?, true)?;
                entry
            };
            manifest.insert(source.clone(), entry);
            pairs.insert(source, (wave, phone));
        }
        Ok(())
    })?;
    audio::write_file(&manifest_path, &serde_json::to_vec_pretty(&manifest)?, true)?;
    Ok(pairs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn choices_match_the_reference_corpus() {
        assert_eq!(choices(0), (None, 1.14));
        assert_eq!(choices(1), (Some(32), 1.25));
        let name = Sha256::digest(b"0_2.wav");
        let seed = u32::from_le_bytes(name[..4].try_into().unwrap());
        assert_eq!(
            choices(seed),
            (Some(SPEAKERS[(name[0] % 4) as usize]), 1.25)
        );
    }

    #[test]
    fn analysis_input_matches_scipy_polyphase_and_pcm16() {
        let mut samples = [0.0; 70];
        samples[15] = 0.6;
        samples[27] = -0.2;
        let result = analysis_wave(&samples, 40000);
        assert_eq!(result.len(), 28);
        for (i, expected) in [
            (6, 7810),
            (7, 92),
            (8, -147),
            (10, -605),
            (11, -2452),
            (17, -31),
        ] {
            assert_eq!((result[i] * 32768.0) as i32, expected);
        }
    }

    #[test]
    fn spectral_warp_stays_on_the_clean_time_grid() {
        let signal: Vec<_> = (0..6413)
            .map(|i| {
                let t = std::f32::consts::TAU * i as f32 / 40_000.0;
                0.2 * (180.0 * t).sin() + 0.05 * (1440.0 * t).sin()
            })
            .collect();
        let warp = Warp::new();
        let baseline = warp.apply(&signal, &signal, 1.0);
        let shifted = warp.apply(&signal, &signal, 0.78);
        assert_eq!(shifted.len(), signal.len());
        assert!(
            baseline
                .iter()
                .zip(&signal)
                .all(|(x, y)| (x - y).abs() < 2e-6)
        );
        assert!(
            shifted
                .iter()
                .zip(&signal)
                .any(|(x, y)| (x - y).abs() > 0.01)
        );
        let energy = |x: &[f32]| x.iter().map(|v| v * v).sum::<f32>();
        assert!((energy(&shifted) / energy(&signal) - 1.0).abs() < 1e-5);
    }
}
