use std::ffi::OsStr;
use std::io::{BufWriter, Cursor, Seek, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use anyhow::{Context, Result, bail, ensure};
use clap::ValueEnum;
use serde::Deserialize;
use tempfile::NamedTempFile;

pub const ANALYSIS_RATE: u32 = 16_000;
pub const FRAME: usize = 160;
const MAX_INPUT_SECONDS: usize = 3_600;

#[derive(Clone, Copy, Debug, Default, Deserialize, ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum Format {
    #[default]
    Wav,
    Flac,
    Mp3,
    M4a,
}

impl Format {
    pub fn from_extension(extension: &OsStr) -> Result<Self> {
        let extension = extension.to_string_lossy().to_ascii_lowercase();
        Self::from_str(&extension, false).map_err(anyhow::Error::msg)
    }

    pub fn extension(self) -> &'static str {
        match self {
            Self::Wav => "wav",
            Self::Flac => "flac",
            Self::Mp3 => "mp3",
            Self::M4a => "m4a",
        }
    }
}

pub fn is_audio(path: &Path) -> bool {
    path.extension().and_then(OsStr::to_str).is_some_and(|ext| {
        matches!(
            ext.to_ascii_lowercase().as_str(),
            "wav"
                | "flac"
                | "mp3"
                | "m4a"
                | "ogg"
                | "opus"
                | "aac"
                | "wma"
                | "mp4"
                | "mkv"
                | "webm"
        )
    })
}

pub fn read_mono(path: &Path, rate: u32) -> Result<Vec<f32>> {
    read_interleaved(path, rate, 1)
}

pub fn probe(path: &Path) -> Result<(u32, u16)> {
    let output = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=sample_rate,channels",
            "-of",
            "json",
        ])
        .arg(path)
        .output()
        .context("starting FFprobe")?;
    ensure!(
        output.status.success(),
        "probing {}: {}",
        path.display(),
        String::from_utf8_lossy(&output.stderr).trim()
    );
    let result: serde_json::Value = serde_json::from_slice(&output.stdout)?;
    let stream = &result["streams"][0];
    let sample_rate: u32 = stream["sample_rate"]
        .as_str()
        .context("no audio sample rate")?
        .parse()?;
    let channels: u16 = stream["channels"]
        .as_u64()
        .context("no audio channels")?
        .try_into()?;
    ensure!(
        (1..=384_000).contains(&sample_rate) && (1..=32).contains(&channels),
        "unsupported audio format"
    );
    Ok((sample_rate, channels))
}

pub fn read_interleaved(path: &Path, rate: u32, channels: u16) -> Result<Vec<f32>> {
    decode(path, rate, channels, None)
}

pub(crate) fn read_denoised(path: &Path, rate: u32, strength: f32) -> Result<Vec<f32>> {
    if strength == 0.0 {
        return read_mono(path, rate);
    }
    let filter = format!("afftdn=nr={:.2}:tn=1", (strength * 24.0).max(0.01));
    decode(path, rate, 1, Some(&filter))
}

fn decode(path: &Path, rate: u32, channels: u16, filter: Option<&str>) -> Result<Vec<f32>> {
    ensure!(
        (1..=384_000).contains(&rate) && (1..=32).contains(&channels),
        "unsupported audio format"
    );
    if filter.is_none()
        && let Some(samples) = read_wav_direct(path, rate, channels)?
    {
        return Ok(samples);
    }
    let mut command = Command::new("ffmpeg");
    command
        .args([
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-protocol_whitelist",
            "file,pipe,crypto,data",
            // FFmpeg sizes its thread pools by the core count. On many-core machines that
            // costs far more than decoding a short clip, which preprocessing and feature
            // extraction do thousands of times.
            "-threads",
            "1",
            "-filter_threads",
            "1",
            "-i",
        ])
        .arg(path);
    if let Some(filter) = filter {
        command.args(["-af", filter]);
    }
    let output = command
        .args([
            "-map",
            "0:a:0",
            "-ac",
            &channels.to_string(),
            "-ar",
            &rate.to_string(),
        ])
        .args(["-t", &(MAX_INPUT_SECONDS + 1).to_string()])
        .args(["-f", "f32le", "-c:a", "pcm_f32le", "pipe:1"])
        .output()
        .context("starting FFmpeg to decode audio")?;
    if !output.status.success() {
        bail!(
            "decoding {}: {}",
            path.display(),
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    let (chunks, remainder) = output.stdout.as_chunks::<4>();
    ensure!(
        remainder.is_empty() && chunks.len().is_multiple_of(channels as usize),
        "FFmpeg returned incomplete PCM"
    );
    ensure!(!chunks.is_empty(), "no audio samples in {}", path.display());
    ensure!(
        chunks.len() / channels as usize <= MAX_INPUT_SECONDS * rate as usize,
        "audio exceeds the {MAX_INPUT_SECONDS}-second limit"
    );
    let samples: Vec<_> = chunks
        .iter()
        .map(|&bytes| f32::from_le_bytes(bytes))
        .collect();
    ensure!(
        samples.iter().all(|v| v.is_finite()),
        "audio contains non-finite samples"
    );
    Ok(samples)
}

/// Reads a PCM or float WAV that already has the requested rate and channel count without
/// starting FFmpeg. Returns `None` for any other file so the caller can decode it with FFmpeg.
fn read_wav_direct(path: &Path, rate: u32, channels: u16) -> Result<Option<Vec<f32>>> {
    if !path
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("wav"))
    {
        return Ok(None);
    }
    let Ok(mut reader) = hound::WavReader::open(path) else {
        return Ok(None);
    };
    let spec = reader.spec();
    if spec.sample_rate != rate || spec.channels != channels {
        return Ok(None);
    }
    let samples: Vec<f32> = match (spec.sample_format, spec.bits_per_sample) {
        (hound::SampleFormat::Float, 32) => reader.samples::<f32>().collect::<Result<_, _>>()?,
        (hound::SampleFormat::Int, bits @ (8 | 16 | 24 | 32)) => {
            let scale = 1.0 / (1_i64 << (bits - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|value| value.map(|value| value as f32 * scale))
                .collect::<Result<_, _>>()?
        }
        _ => return Ok(None),
    };
    ensure!(
        !samples.is_empty(),
        "no audio samples in {}",
        path.display()
    );
    ensure!(
        samples.len() / channels as usize <= MAX_INPUT_SECONDS * rate as usize,
        "audio exceeds the {MAX_INPUT_SECONDS}-second limit"
    );
    ensure!(
        samples.iter().all(|v| v.is_finite()),
        "audio contains non-finite samples"
    );
    Ok(Some(samples))
}

pub fn encode_wav(samples: &[f32], rate: u32, float: bool) -> Result<Vec<u8>> {
    encode_wav_channels(samples, rate, 1, float)
}

pub fn encode_wav_channels(
    samples: &[f32],
    rate: u32,
    channels: u16,
    float: bool,
) -> Result<Vec<u8>> {
    let mut bytes = Cursor::new(Vec::new());
    write_wav(&mut bytes, samples, rate, channels, float)?;
    Ok(bytes.into_inner())
}

pub fn write_wav(
    output: impl Write + Seek,
    samples: &[f32],
    rate: u32,
    channels: u16,
    float: bool,
) -> Result<()> {
    ensure!(
        channels > 0 && samples.len().is_multiple_of(channels as usize),
        "invalid WAV channel or sample count"
    );
    let spec = hound::WavSpec {
        channels,
        sample_rate: rate,
        bits_per_sample: if float { 32 } else { 16 },
        sample_format: if float {
            hound::SampleFormat::Float
        } else {
            hound::SampleFormat::Int
        },
    };
    let mut writer = hound::WavWriter::new(output, spec)?;
    for &sample in samples {
        if float {
            writer.write_sample(sample)?;
        } else {
            writer.write_sample((sample * 32768.0).round() as i16)?;
        }
    }
    writer.finalize()?;
    Ok(())
}

pub fn write_file(path: &Path, bytes: &[u8], overwrite: bool) -> Result<()> {
    let output_dir = output_directory(path);
    std::fs::create_dir_all(output_dir)?;
    let mut temporary = NamedTempFile::new_in(output_dir)?;
    temporary.write_all(bytes)?;
    persist(temporary, path, overwrite)
}

pub fn write_audio(
    path: &Path,
    samples: &[f32],
    rate: u32,
    format: Format,
    overwrite: bool,
) -> Result<()> {
    write_audio_channels(path, samples, rate, 1, format, false, overwrite)
}

pub fn write_audio_channels(
    path: &Path,
    samples: &[f32],
    rate: u32,
    channels: u16,
    format: Format,
    float: bool,
    overwrite: bool,
) -> Result<()> {
    let output_dir = output_directory(path);
    std::fs::create_dir_all(output_dir)?;
    let mut temporary = NamedTempFile::new_in(output_dir)?;
    if matches!(format, Format::Wav) {
        write_wav(
            BufWriter::new(temporary.as_file_mut()),
            samples,
            rate,
            channels,
            float,
        )?;
        return persist(temporary, path, overwrite);
    }
    let wav = encode_wav_channels(samples, rate, channels, float)?;
    let muxer = if matches!(format, Format::M4a) {
        "ipod"
    } else {
        format.extension()
    };
    let mut child = Command::new("ffmpeg")
        .args([
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "wav",
            "-i",
            "pipe:0",
            "-f",
            muxer,
        ])
        .arg(temporary.path())
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .context("starting FFmpeg to encode audio")?;
    let written = child
        .stdin
        .take()
        .context("missing FFmpeg stdin")?
        .write_all(&wav);
    let output = child.wait_with_output()?;
    ensure!(
        output.status.success(),
        "encoding {}: {}",
        path.display(),
        String::from_utf8_lossy(&output.stderr).trim()
    );
    written?;
    persist(temporary, path, overwrite)
}

fn output_directory(path: &Path) -> &Path {
    path.parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
}

fn persist(temporary: NamedTempFile, path: &Path, overwrite: bool) -> Result<()> {
    let result = if overwrite {
        temporary.persist(path)
    } else {
        temporary.persist_noclobber(path)
    };
    result.with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

pub fn files(root: &Path, recursive: bool) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    let mut walker = walkdir::WalkDir::new(root).min_depth(1).sort_by_file_name();
    if !recursive {
        walker = walker.max_depth(1);
    }
    for entry in walker {
        let entry = entry?;
        if entry.path().is_file() && is_audio(entry.path()) {
            files.push(entry.into_path());
        }
    }
    files.sort();
    Ok(files)
}

/// Reflection padding as used by numpy.pad in the original pipelines.
pub fn reflect(samples: &[f32], left: usize, right: usize) -> Vec<f32> {
    if samples.len() <= 1 {
        return vec![samples.first().copied().unwrap_or(0.0); left + samples.len() + right];
    }
    let edge = samples.len() as i64 - 1;
    (-(left as i64)..samples.len() as i64 + right as i64)
        .map(|position| {
            let period = position.rem_euclid(2 * edge);
            samples[period.min(2 * edge - period) as usize]
        })
        .collect()
}

/// Band-limited interpolation for mono audio buffers, independent of FFmpeg.
/// Hold `converted` under the level of `input` plus `allowance_db`, frame by frame
/// (~46 ms at `rate`, each frame judged with its neighbours).
///
/// A voice model renders faint input (a pause, a fade, a quiet echo of the voice) at a
/// full singing level, which sounds like the voice singing where the source is
/// silent, or a howl behind the vocal. Only attenuates; gains are interpolated
/// between frame centres. Both signals must be aligned at `rate`. Returns the deepest
/// cut in dB.
pub fn hold_under(converted: &mut [f32], input: &[f32], rate: u32, allowance_db: f32) -> f32 {
    let size = ((rate as f32 * 0.0464) as usize).max(64);
    let n = converted.len().min(input.len());
    if n == 0 {
        return 0.0;
    }
    let frames = n.div_ceil(size);
    let rms = |x: &[f32], t: usize| -> f32 {
        let chunk = &x[t * size..((t + 1) * size).min(n)];
        (chunk.iter().map(|v| v * v).sum::<f32>() / chunk.len() as f32).sqrt()
    };
    let allowed = 10_f32.powf(allowance_db / 20.0);
    let gains: Vec<f32> = (0..frames)
        .map(|t| {
            let around = t.saturating_sub(1)..(t + 2).min(frames);
            let (i, o) = around.fold((0_f32, 0_f32), |(i, o), u| {
                (i.max(rms(input, u)), o.max(rms(converted, u)))
            });
            if o > i * allowed { (i * allowed / o.max(1e-9)).max(1e-3) } else { 1.0 }
        })
        .collect();
    let deepest = gains.iter().copied().fold(1_f32, f32::min);
    if deepest >= 1.0 {
        return 0.0;
    }
    for (k, sample) in converted[..n].iter_mut().enumerate() {
        let pos = (k as f32 - size as f32 / 2.0) / size as f32;
        let t = pos.floor().max(0.0) as usize;
        let x = (pos - t as f32).clamp(0.0, 1.0);
        let (a, b) = (gains[t.min(frames - 1)], gains[(t + 1).min(frames - 1)]);
        *sample *= a * (1.0 - x) + b * x;
    }
    -20.0 * deepest.log10()
}

pub fn resample(samples: &[f32], from: u32, to: u32) -> Vec<f32> {
    use rayon::prelude::*;
    if from == to || samples.is_empty() {
        return samples.to_vec();
    }
    let output_len = (samples.len() as u64 * u64::from(to)).div_ceil(u64::from(from)) as usize;
    let step = f64::from(from) / f64::from(to);
    let cutoff = step.recip().min(1.0) * 0.95;
    let radius = (24.0 / cutoff).ceil() as i64;
    let taps = 2 * radius as usize;
    // Hann-windowed sinc, normalized to unit gain, at fractional offset `fraction`.
    let kernel = |fraction: f64, out: &mut [f64]| {
        let mut sum = 0.0;
        for (k, tap) in out.iter_mut().enumerate() {
            let distance = fraction + radius as f64 - 1.0 - k as f64;
            let window = 0.5 + 0.5 * (std::f64::consts::PI * distance / radius as f64).cos();
            let angle = std::f64::consts::PI * distance * cutoff;
            *tap = window * if angle.abs() < 1e-12 { 1.0 } else { angle.sin() / angle };
            sum += *tap;
        }
        out.iter_mut().for_each(|tap| *tap /= sum);
    };
    // A rational rate change repeats `phases` fractional offsets: compute each kernel once.
    let divisor = gcd(from, to);
    let (up, down) = (u64::from(to / divisor), u64::from(from / divisor));
    let table = (up <= 1 << 16).then(|| {
        let mut table = vec![0.0; up as usize * taps];
        for (phase, row) in table.chunks_exact_mut(taps).enumerate() {
            kernel(phase as f64 / up as f64, row);
        }
        table
    });
    let last = samples.len() as i64 - 1;
    let render = |output: usize, scratch: &mut Vec<f64>| -> f32 {
        let numerator = output as u64 * down;
        let center = (numerator / up) as i64;
        let phase = (numerator % up) as usize;
        let weights: &[f64] = match &table {
            Some(table) => &table[phase * taps..(phase + 1) * taps],
            None => {
                scratch.resize(taps, 0.0);
                kernel(phase as f64 / up as f64, scratch);
                scratch
            }
        };
        let first = center - radius + 1;
        let mut weighted = 0.0;
        if first >= 0 && first + taps as i64 - 1 <= last {
            let window = &samples[first as usize..first as usize + taps];
            for (w, &x) in weights.iter().zip(window) {
                weighted += w * f64::from(x);
            }
        } else {
            for (k, w) in weights.iter().enumerate() {
                weighted += w * f64::from(samples[(first + k as i64).clamp(0, last) as usize]);
            }
        }
        weighted as f32
    };
    let mut output = vec![0.0_f32; output_len];
    output
        .par_chunks_mut(16_384)
        .enumerate()
        .for_each(|(chunk, values)| {
            let mut scratch = Vec::new();
            for (i, value) in values.iter_mut().enumerate() {
                *value = render(chunk * 16_384 + i, &mut scratch);
            }
        });
    output
}

fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

#[cfg(feature = "inference")]
pub fn conversion_jobs(
    input: PathBuf,
    output: PathBuf,
    format: Option<Format>,
    recursive: bool,
) -> Result<Vec<(PathBuf, PathBuf, Format)>> {
    ensure!(input.exists(), "Input not found: {}", input.display());
    if input.is_file() {
        let extension = if output.is_dir() {
            None
        } else {
            output.extension().map(Format::from_extension).transpose()?
        };
        let format = format.or(extension).unwrap_or_default();
        let target = if extension.is_some() && !output.is_dir() {
            output.with_extension(format.extension())
        } else {
            output
                .join(input.file_name().context("missing input file name")?)
                .with_extension(format.extension())
        };
        return Ok(vec![(input, target, format)]);
    }
    let format = format.unwrap_or_default();
    let files = files(&input, recursive)?;
    ensure!(
        !files.is_empty(),
        "No supported audio files found in: {}",
        input.display()
    );
    files
        .into_iter()
        .map(|file| {
            let relative = file.strip_prefix(&input)?;
            let target = output.join(relative).with_extension(format.extension());
            Ok((file, target, format))
        })
        .collect()
}

#[cfg(test)]
mod direct_wav_tests {
    use super::*;

    #[test]
    fn direct_wav_matches_ffmpeg() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let samples: Vec<f32> = (0..16_000).map(|n| (n as f32 * 0.01).sin() * 0.5).collect();
        for float in [false, true] {
            let path = dir.path().join(format!("clip_{float}.wav"));
            std::fs::write(&path, encode_wav(&samples, 16_000, float)?)?;
            let direct = read_wav_direct(&path, 16_000, 1)?.expect("direct read");
            let mut command = Command::new("ffmpeg");
            let output = command
                .args(["-nostdin", "-loglevel", "error", "-i"])
                .arg(&path)
                .args(["-f", "f32le", "-c:a", "pcm_f32le", "pipe:1"])
                .output()?;
            let reference: Vec<f32> = output
                .stdout
                .as_chunks::<4>()
                .0
                .iter()
                .map(|&b| f32::from_le_bytes(b))
                .collect();
            assert_eq!(direct.len(), reference.len());
            let error = direct
                .iter()
                .zip(&reference)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(error < 1e-6, "max error {error} (float {float})");
            assert!(read_wav_direct(&path, 22_050, 1)?.is_none());
        }
        Ok(())
    }
}

#[cfg(test)]
mod resample_tests {
    use super::resample;

    /// The direct form the polyphase resampler replaced.
    fn reference(samples: &[f32], from: u32, to: u32) -> Vec<f32> {
        let output_len = (samples.len() as u64 * u64::from(to)).div_ceil(u64::from(from)) as usize;
        let step = f64::from(from) / f64::from(to);
        let cutoff = step.recip().min(1.0) * 0.95;
        let radius = (24.0 / cutoff).ceil() as i64;
        (0..output_len)
            .map(|output| {
                let position = output as f64 * step;
                let center = position.floor() as i64;
                let (mut weighted, mut normalization) = (0.0, 0.0);
                for input in center - radius + 1..=center + radius {
                    let distance = position - input as f64;
                    let window = 0.5 + 0.5 * (std::f64::consts::PI * distance / radius as f64).cos();
                    let angle = std::f64::consts::PI * distance * cutoff;
                    let kernel = window * if angle.abs() < 1e-12 { 1.0 } else { angle.sin() / angle };
                    let sample = samples[input.clamp(0, samples.len() as i64 - 1) as usize];
                    weighted += f64::from(sample) * kernel;
                    normalization += kernel;
                }
                (weighted / normalization) as f32
            })
            .collect()
    }

    #[test]
    fn polyphase_matches_direct_form() {
        let x: Vec<f32> = (0..20_000)
            .map(|i| ((i as f32 * 0.013).sin() + (i as f32 * 0.37).cos() * 0.3) * 0.5)
            .collect();
        for (from, to) in [(44_100, 16_000), (16_000, 40_000), (40_000, 44_100), (48_000, 44_100), (40_000, 39_567)] {
            let fast = resample(&x, from, to);
            let slow = reference(&x, from, to);
            assert_eq!(fast.len(), slow.len());
            let error = fast.iter().zip(&slow).fold(0_f32, |m, (a, b)| m.max((a - b).abs()));
            assert!(error < 1e-5, "{from}->{to}: {error}");
        }
    }
}
