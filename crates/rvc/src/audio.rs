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
    ensure!(
        (1..=384_000).contains(&rate) && (1..=32).contains(&channels),
        "unsupported audio format"
    );
    let output = Command::new("ffmpeg")
        .args([
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-protocol_whitelist",
            "file,pipe,crypto,data",
            "-i",
        ])
        .arg(path)
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
pub fn resample(samples: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to || samples.is_empty() {
        return samples.to_vec();
    }
    let output_len = (samples.len() as u64 * u64::from(to)).div_ceil(u64::from(from)) as usize;
    let step = f64::from(from) / f64::from(to);
    let cutoff = step.recip().min(1.0) * 0.95;
    let radius = (24.0 / cutoff).ceil() as i64;
    (0..output_len)
        .map(|output| {
            let position = output as f64 * step;
            let center = position.floor() as i64;
            let mut weighted = 0.0;
            let mut normalization = 0.0;
            for input in center - radius + 1..=center + radius {
                let distance = position - input as f64;
                let window = 0.5 + 0.5 * (std::f64::consts::PI * distance / radius as f64).cos();
                let angle = std::f64::consts::PI * distance * cutoff;
                let kernel = window
                    * if angle.abs() < 1e-12 {
                        1.0
                    } else {
                        angle.sin() / angle
                    };
                let sample = samples[input.clamp(0, samples.len() as i64 - 1) as usize];
                weighted += f64::from(sample) * kernel;
                normalization += kernel;
            }
            (weighted / normalization) as f32
        })
        .collect()
}
