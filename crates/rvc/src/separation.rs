//! PyMSS WebUI separation presets and local batch processing.

mod configurable;
mod demucs;
mod legacy_wave;
mod mdx23c;
mod mlx;
mod mlx_layers;
mod mlx_stft;
#[cfg(feature = "onnx")]
mod onnx_layers;
mod overlap;
mod roformer;
#[cfg(test)]
mod tests;
mod vr;

use std::collections::HashSet;
use std::io::{BufWriter, Write};
use std::ops::ControlFlow;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use anyhow::{Context, Result, bail, ensure};
use clap::ValueEnum;
use serde::Serialize;
use tch::Device;
use tempfile::NamedTempFile;

use crate::audio::{self, Format};
use overlap::{DemixError, Geometry, Progress, demix, residual};
use roformer::{ModelConfig, Roformer};

#[derive(Clone, Copy)]
enum Acceleration {
    Torch,
    Mlx(mlx::DType),
    #[cfg(feature = "onnx")]
    Onnx(crate::model::onnx::Provider),
}

impl Acceleration {
    fn configured(
        document: &serde_json::Value,
        device: Device,
        supports_mlx: bool,
        name: &str,
    ) -> Result<Self> {
        let mlx = mlx::selected_dtype(document, device, supports_mlx, name)?;
        let provider = &document["inference"]["onnx_provider"];
        if !provider.is_null() {
            let provider = provider.as_str().context("onnx_provider must be text")?;
            #[cfg(feature = "onnx")]
            {
                ensure!(
                    device == Device::Cpu && mlx.is_none(),
                    "{name} ONNX requires --device cpu and a disabled MLX backend"
                );
                return Ok(Self::Onnx(crate::model::onnx::Provider::parse(
                    "onnx_provider",
                    provider,
                )?));
            }
            #[cfg(not(feature = "onnx"))]
            bail!("onnx_provider={provider} requires the onnx build feature");
        }
        Ok(mlx.map_or(Self::Torch, Self::Mlx))
    }

    fn weight_device(&self, device: Device) -> Device {
        match self {
            Self::Torch => device,
            _ => Device::Cpu,
        }
    }
}

pub use configurable::{
    ConfiguredModel, ConfiguredOptions, ConfiguredStemFiles, InferenceParams, StemFile,
    separate_configured, separate_configured_with_params,
};

pub use crate::model::weights::device;

pub fn optional_config(model_type: &str) -> bool {
    model_type == "vr" || demucs::legacy::is_legacy(model_type)
}

const SAMPLE_RATE: u32 = 44_100;
const PRESET_PCM: PcmLayout = PcmLayout {
    sample_rate: SAMPLE_RATE,
    channels: 2,
};

fn window_samples(audio: &[f32], count: usize, channels: usize) -> Result<i64> {
    let planes = count
        .checked_mul(channels)
        .context("too many audio channels")?;
    ensure!(
        planes > 0
            && !audio.is_empty()
            && audio.len().is_multiple_of(planes)
            && audio.iter().all(|sample| sample.is_finite()),
        "expected a batch of finite channel-major audio windows"
    );
    Ok(i64::try_from(audio.len() / planes)?)
}

#[derive(Clone, Copy)]
struct PcmLayout {
    sample_rate: u32,
    channels: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum Architecture {
    #[serde(rename = "bs_roformer")]
    BandSplitRoformer,
    #[serde(rename = "mel_band_roformer")]
    MelBandRoformer,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct Preset {
    pub id: &'static str,
    pub label: &'static str,
    pub architecture: Architecture,
    pub checkpoint: &'static str,
    pub config: &'static str,
    pub model_stem: &'static str,
    pub other_stem: &'static str,
    pub suffixes: [&'static str; 2],
    pub batch: usize,
    pub overlap: usize,
}

pub const PRESETS: [Preset; 8] = [
    Preset {
        id: "dereverb-less-aggressive-18.8050",
        label: "去混响",
        architecture: Architecture::MelBandRoformer,
        checkpoint: "dereverb_mel_band_roformer_less_aggressive_anvuew_sdr_18.8050.ckpt",
        config: "dereverb_mel_band_roformer_anvuew.yaml",
        model_stem: "noreverb",
        other_stem: "reverb",
        suffixes: ["noreverb", "reverb"],
        batch: 1,
        overlap: 176_400,
    },
    Preset {
        id: "dereverb-anvuew-19.1729",
        label: "去混响（激进）",
        architecture: Architecture::MelBandRoformer,
        checkpoint: "dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt",
        config: "dereverb_mel_band_roformer_anvuew.yaml",
        model_stem: "noreverb",
        other_stem: "reverb",
        suffixes: ["noreverb", "reverb"],
        batch: 1,
        overlap: 176_400,
    },
    Preset {
        id: "vocals-bs-roformer-368",
        label: "去伴奏",
        architecture: Architecture::BandSplitRoformer,
        checkpoint: "model_bs_roformer_ep_368_sdr_12.9628.ckpt",
        config: "model_bs_roformer_ep_368_sdr_12.9628.yaml",
        model_stem: "vocals",
        other_stem: "instrumental",
        suffixes: ["vocals", "instrumental"],
        batch: 1,
        overlap: 264_600,
    },
    Preset {
        id: "vocals-bs-roformer-317",
        label: "去伴奏（激进）",
        architecture: Architecture::BandSplitRoformer,
        checkpoint: "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
        config: "model_bs_roformer_ep_317_sdr_12.9755.yaml",
        model_stem: "vocals",
        other_stem: "other",
        suffixes: ["vocals", "instrumental"],
        batch: 4,
        overlap: 176_400,
    },
    Preset {
        id: "karaoke-mel-roformer-10.1956",
        label: "提主旋律",
        architecture: Architecture::MelBandRoformer,
        checkpoint: "model_mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt",
        config: "config_mel_band_roformer_karaoke.yaml",
        model_stem: "karaoke",
        other_stem: "other",
        suffixes: ["main_vocal", "off_vocal"],
        batch: 1,
        overlap: 264_600,
    },
    Preset {
        id: "vocals-bs-roformer-ft1-anvuew",
        label: "vocals (BS-RoFormer FT1, anvuew)",
        architecture: Architecture::BandSplitRoformer,
        checkpoint: "bs_roformer_ft1_anvuew_sdr_12.55.ckpt",
        config: "bs_roformer_anvuew.yaml",
        model_stem: "vocals",
        other_stem: "instrument",
        suffixes: ["vocals", "instrumental"],
        batch: 1,
        overlap: 176_400,
    },
    Preset {
        id: "karaoke-bs-roformer-anvuew",
        label: "lead / backing vocals (BS-RoFormer karaoke, anvuew)",
        architecture: Architecture::BandSplitRoformer,
        checkpoint: "karaoke_bs_roformer_anvuew.ckpt",
        config: "karaoke_bs_roformer_anvuew.yaml",
        model_stem: "Vocals",
        other_stem: "Instrumental",
        suffixes: ["lead", "backing"],
        batch: 1,
        overlap: 176_400,
    },
    Preset {
        id: "dereverb-bs-roformer-anvuew-22.5050",
        label: "dereverb (BS-RoFormer, anvuew)",
        architecture: Architecture::BandSplitRoformer,
        checkpoint: "dereverb_bs_roformer_anvuew_sdr_22.5050.ckpt",
        config: "dereverb_bs_roformer_anvuew.yaml",
        model_stem: "noreverb",
        other_stem: "reverb",
        suffixes: ["noreverb", "reverb"],
        batch: 1,
        overlap: 176_400,
    },
];

fn preset(name: &str) -> Option<&'static Preset> {
    if name.is_empty() {
        return PRESETS.first();
    }
    PRESETS
        .iter()
        .find(|item| item.id == name || item.label == name)
}

pub struct Options<'a> {
    pub preset: &'a str,
    pub device: Device,
    pub format: Format,
    pub secondary_output: Option<&'a Path>,
    pub overwrite: bool,
}

#[derive(Debug, Serialize, serde::Deserialize)]
pub struct StemFiles {
    pub input: PathBuf,
    pub primary: PathBuf,
    pub secondary: PathBuf,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct JobProgress {
    /// One-based position of the current input.
    pub file_index: usize,
    pub file_count: usize,
    pub chunks_done: usize,
    pub chunks_total: usize,
}

/// A preset's model kept loaded for several in-memory separations.
pub struct Separator {
    spec: &'static Preset,
    model: Roformer,
}

impl Separator {
    pub fn load(root: &Path, preset_name: &str, device: Device) -> Result<Self> {
        let spec =
            preset(preset_name).with_context(|| format!("unknown separation preset {preset_name}"))?;
        let directory = weights_directory(root);
        let model_path = directory.join(spec.checkpoint);
        let document = configurable::load_document(&directory.join(spec.config))?;
        let weight_device = match device {
            Device::Cuda(_) => device,
            _ => Device::Cpu,
        };
        let weights = configurable::checkpoint_weights(&model_path, weight_device)?;
        let config = ModelConfig::from_document(
            &document,
            spec.architecture == Architecture::MelBandRoformer,
        )?;
        let model = Roformer::load(config, &weights, device)
            .with_context(|| format!("loading separation model {}", model_path.display()))?;
        ensure!(
            model.channels() == PRESET_PCM.channels && model.num_stems() == 1,
            "PyMSS preset requires a stereo model that predicts one stem"
        );
        Ok(Self { spec, model })
    }

    /// Model stem and residual of interleaved stereo audio at 44.1 kHz, interleaved.
    pub fn split(&self, interleaved: &[f32]) -> Result<(Vec<f32>, Vec<f32>)> {
        // The models read channel planes, not interleaved frames.
        let frames = interleaved.len() / 2;
        let mut mix = Vec::with_capacity(frames * 2);
        mix.extend(interleaved.iter().step_by(2).take(frames));
        mix.extend(interleaved.iter().skip(1).step_by(2).take(frames));
        let mix = mix.as_slice();
        let geometry = Geometry {
            frames_per_chunk: 352_800,
            overlap: self.spec.overlap,
            batch_size: 1,
        };
        let primary = demix(
            mix,
            PRESET_PCM.channels,
            1,
            &[0],
            geometry,
            |samples, count| self.model.predict_batch(samples, count),
            |_| ControlFlow::Continue(()),
        )
        .map_err(demix_error)?;
        let secondary =
            residual(mix, &primary).context("separation changed the track duration")?;
        let interleave = |planes: &[f32]| -> Vec<f32> {
            let (left, right) = planes.split_at(planes.len() / 2);
            left.iter().zip(right).flat_map(|(&l, &r)| [l, r]).collect()
        };
        Ok((interleave(&primary), interleave(&secondary)))
    }
}

/// Runs the preset on a file or a directory of audio files. Both stems go to
/// `output` unless `options.secondary_output` selects a separate directory.
/// Progress notifications can cancel before the next inference window.
pub fn separate(
    root: &Path,
    input: &Path,
    output: &Path,
    options: Options<'_>,
    mut progress: impl FnMut(JobProgress) -> ControlFlow<()>,
) -> Result<Vec<StemFiles>> {
    let spec = preset(options.preset)
        .with_context(|| format!("unknown separation preset {}", options.preset))?;
    let jobs = plan_jobs(input, output, &options, spec)?;
    let directory = weights_directory(root);
    let model_path = directory.join(spec.checkpoint);
    let document = configurable::load_document(&directory.join(spec.config))?;
    let weight_device = match options.device {
        Device::Cuda(_) => options.device,
        _ => Device::Cpu,
    };
    let weights = configurable::checkpoint_weights(&model_path, weight_device)?;
    let config = ModelConfig::from_document(
        &document,
        spec.architecture == Architecture::MelBandRoformer,
    )?;
    let model = Roformer::load(config, &weights, options.device)
        .with_context(|| format!("loading separation model {}", model_path.display()))?;
    ensure!(
        model.channels() == PRESET_PCM.channels && model.num_stems() == 1,
        "PyMSS preset requires a stereo model that predicts one stem"
    );
    drop(weights);

    let geometry = Geometry {
        frames_per_chunk: 352_800,
        overlap: spec.overlap,
        batch_size: 1,
    };
    for (index, files) in jobs.iter().enumerate() {
        let input = &files.input;
        let mix = read_audio(root, input, PRESET_PCM)?;
        let primary = demix(
            &mix,
            PRESET_PCM.channels,
            1,
            &[0],
            geometry,
            |samples, count| model.predict_batch(samples, count),
            |status| progress(job_progress(index, jobs.len(), status)),
        )
        .map_err(demix_error)
        .with_context(|| format!("separating {}", input.display()))?;
        let secondary =
            residual(&mix, &primary).context("separation changed the track duration")?;
        write_stems(
            root,
            [
                (files.primary.as_path(), primary.as_slice()),
                (files.secondary.as_path(), secondary.as_slice()),
            ],
            PRESET_PCM,
            options.format,
            options.overwrite,
        )?;
    }
    Ok(jobs)
}

fn job_progress(index: usize, file_count: usize, status: Progress) -> JobProgress {
    JobProgress {
        file_index: index + 1,
        file_count,
        chunks_done: status.chunks_done,
        chunks_total: status.chunks_total,
    }
}

fn demix_error(error: DemixError<anyhow::Error>) -> anyhow::Error {
    match error {
        DemixError::Model(error) => error,
        other => anyhow::Error::msg(other.to_string()),
    }
}

/// Whether the checkpoint and config of `preset` are installed under `root`.
pub fn installed(root: &Path, preset: &Preset) -> bool {
    let directory = weights_directory(root);
    directory.join(preset.checkpoint).is_file() && directory.join(preset.config).is_file()
}

pub(crate) fn weights_directory(root: &Path) -> PathBuf {
    let configured = std::env::var_os("weight_pymss_root")
        .map(PathBuf::from)
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| PathBuf::from("assets/pymss_weights"));
    root.join(configured)
}

fn plan_jobs(
    input: &Path,
    output: &Path,
    options: &Options<'_>,
    spec: &Preset,
) -> Result<Vec<StemFiles>> {
    let secondary = options.secondary_output.unwrap_or(output);
    let destinations = [(spec.suffixes[0], output), (spec.suffixes[1], secondary)];
    plan_outputs(
        input,
        &destinations,
        options.format,
        options.overwrite,
        false,
    )?
    .into_iter()
    .map(|(input, paths)| {
        let [primary, secondary] = paths
            .try_into()
            .map_err(|_| anyhow::anyhow!("incorrect preset output count"))?;
        Ok(StemFiles {
            input,
            primary,
            secondary,
        })
    })
    .collect()
}

fn plan_outputs(
    input: &Path,
    destinations: &[(&str, &Path)],
    format: Format,
    overwrite: bool,
    save_as_folder: bool,
) -> Result<Vec<(PathBuf, Vec<PathBuf>)>> {
    let inputs = if input.is_file() {
        vec![input.to_path_buf()]
    } else if input.is_dir() {
        audio::files(input, false)?
    } else {
        bail!("separation input does not exist: {}", input.display());
    };
    ensure!(
        !inputs.is_empty(),
        "no audio files found in {}",
        input.display()
    );
    let destinations = destinations
        .iter()
        .map(|&(name, directory)| -> Result<_> {
            ensure!(
                !name.trim().is_empty() && !name.contains(['/', '\\', ':', '\0']),
                "separation stem name {name:?} is not a filename suffix"
            );
            std::fs::create_dir_all(directory)
                .with_context(|| format!("creating output directory {}", directory.display()))?;
            Ok((name, directory.canonicalize()?))
        })
        .collect::<Result<Vec<_>>>()?;
    let source_paths = inputs
        .iter()
        .map(|path| path.canonicalize())
        .collect::<std::io::Result<HashSet<_>>>()?;
    let mut output_paths = HashSet::new();
    let mut jobs = Vec::with_capacity(inputs.len());
    for input in inputs {
        let stem = input
            .file_stem()
            .context("input audio filename has no stem")?;
        let mut paths = Vec::with_capacity(destinations.len());
        for (suffix, root) in &destinations {
            let directory = if save_as_folder {
                let directory = root.join(stem);
                std::fs::create_dir_all(&directory)?;
                let directory = directory.canonicalize()?;
                ensure!(
                    directory.starts_with(root),
                    "separation output directory leaves the requested output folder"
                );
                directory
            } else {
                root.clone()
            };
            let mut filename = stem.to_os_string();
            filename.push(format!("_{suffix}.{}", format.extension()));
            let path = directory.join(filename);
            ensure!(
                output_paths.insert(path.clone()),
                "multiple inputs use {}",
                path.display()
            );
            if path.symlink_metadata().is_ok() {
                ensure!(
                    overwrite,
                    "output already exists; use --overwrite: {}",
                    path.display()
                );
                ensure!(
                    !path
                        .canonicalize()
                        .is_ok_and(|resolved| source_paths.contains(&resolved)),
                    "separation output would replace input audio: {}",
                    path.display()
                );
            }
            paths.push(path);
        }
        jobs.push((input, paths));
    }
    Ok(jobs)
}

fn ffmpeg(root: &Path, tool: &str) -> Command {
    let bundled = root.join(format!("{tool}.exe"));
    if cfg!(windows) && bundled.is_file() {
        Command::new(bundled)
    } else {
        Command::new(tool)
    }
}

fn read_audio(root: &Path, path: &Path, layout: PcmLayout) -> Result<Vec<f32>> {
    let input = path.canonicalize()?;
    let probed = ffmpeg(root, "ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=channels",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            "-i",
        ])
        .arg(&input)
        .output()
        .context("starting FFprobe to inspect the audio channels")?;
    ensure!(
        probed.status.success(),
        "reading audio channels in {}: {}",
        input.display(),
        String::from_utf8_lossy(&probed.stderr).trim()
    );
    let channels: usize = std::str::from_utf8(&probed.stdout)?
        .trim()
        .parse()
        .context("expected a mono or stereo audio stream")?;
    ensure!(
        matches!(channels, 1 | 2),
        "expected mono or stereo audio, got {channels} channels"
    );
    let decoded = ffmpeg(root, "ffmpeg")
        .args(["-nostdin", "-hide_banner", "-loglevel", "error", "-i"])
        .arg(&input)
        .args([
            "-map",
            "0:a:0",
            "-vn",
            "-ar",
            &layout.sample_rate.to_string(),
            "-ac",
            &channels.to_string(),
            "-f",
            "f32le",
            "-c:a",
            "pcm_f32le",
            "pipe:1",
        ])
        .output()
        .context("starting FFmpeg to decode separation input")?;
    ensure!(
        decoded.status.success(),
        "decoding {}: {}",
        input.display(),
        String::from_utf8_lossy(&decoded.stderr).trim()
    );
    let (samples, tail) = decoded.stdout.as_chunks::<4>();
    ensure!(
        tail.is_empty() && !samples.is_empty() && samples.len().is_multiple_of(channels),
        "FFmpeg returned empty or incomplete audio"
    );
    let mut audio = Vec::with_capacity(layout.channels * (samples.len() / channels));
    if channels == 2 && layout.channels == 1 {
        for &[left, right] in samples.as_chunks::<2>().0 {
            let l = f32::from_le_bytes(left);
            let r = f32::from_le_bytes(right);
            audio.push(0.5 * (l + r));
        }
    } else {
        for channel in 0..layout.channels {
            audio.extend(
                samples
                    .iter()
                    .skip(channel % channels)
                    .step_by(channels)
                    .copied()
                    .map(f32::from_le_bytes),
            );
        }
    }
    ensure!(
        audio.iter().all(|sample| sample.is_finite()),
        "input audio contains nonfinite samples"
    );
    Ok(audio)
}

fn write_stems<'a>(
    root: &Path,
    stems: impl IntoIterator<Item = (&'a Path, &'a [f32])>,
    layout: PcmLayout,
    format: Format,
    overwrite: bool,
) -> Result<()> {
    write_stems_with_encoding(
        root,
        stems,
        layout,
        format,
        &AudioEncoding::default(),
        overwrite,
    )
}

fn write_stems_with_encoding<'a>(
    root: &Path,
    stems: impl IntoIterator<Item = (&'a Path, &'a [f32])>,
    layout: PcmLayout,
    format: Format,
    encoding: &AudioEncoding,
    overwrite: bool,
) -> Result<()> {
    let encoded = stems
        .into_iter()
        .map(|(path, audio)| {
            Ok((
                path,
                stage_stem(root, path, audio, layout, format, encoding)?,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    for (path, temporary) in encoded {
        persist_stem(temporary, path, overwrite)?;
    }
    Ok(())
}

#[derive(Clone, Copy, ValueEnum)]
pub enum WavDepth {
    #[value(name = "FLOAT")]
    Float,
    #[value(name = "PCM_16")]
    Pcm16,
    #[value(name = "PCM_24")]
    Pcm24,
}

#[derive(Clone, Copy, ValueEnum)]
pub enum FlacDepth {
    #[value(name = "PCM_16")]
    Pcm16,
    #[value(name = "PCM_24")]
    Pcm24,
}

pub struct AudioEncoding {
    pub wav: WavDepth,
    pub flac: FlacDepth,
    pub mp3_bit_rate: u32,
    pub m4a_bit_rate: u32,
    pub m4a_codec: String,
    pub m4a_aac_at_quality: i32,
}

impl Default for AudioEncoding {
    fn default() -> Self {
        Self {
            wav: WavDepth::Float,
            flac: FlacDepth::Pcm24,
            mp3_bit_rate: 320_000,
            m4a_bit_rate: 320_000,
            m4a_codec: "aac".to_owned(),
            m4a_aac_at_quality: 2,
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn write_encoded_track(
    root: &Path,
    path: &Path,
    planar: &[f32],
    sample_rate: u32,
    channels: usize,
    format: Format,
    encoding: &AudioEncoding,
    overwrite: bool,
) -> Result<()> {
    let layout = PcmLayout {
        sample_rate,
        channels,
    };
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)?;
    }
    let staged = stage_stem(root, path, planar, layout, format, encoding)?;
    persist_stem(staged, path, overwrite)
}

fn stage_stem(
    root: &Path,
    path: &Path,
    audio: &[f32],
    layout: PcmLayout,
    format: Format,
    encoding: &AudioEncoding,
) -> Result<NamedTempFile> {
    ensure!(
        (1..=u16::MAX as usize).contains(&layout.channels)
            && layout.sample_rate > 0
            && !audio.is_empty()
            && audio.len().is_multiple_of(layout.channels)
            && audio.iter().all(|sample| sample.is_finite()),
        "invalid output PCM for {}",
        path.display()
    );
    let directory = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let mut temp = NamedTempFile::new_in(directory)?;
    if matches!(format, Format::Wav) {
        let (bits, format) = match encoding.wav {
            WavDepth::Float => (32, hound::SampleFormat::Float),
            WavDepth::Pcm16 => (16, hound::SampleFormat::Int),
            WavDepth::Pcm24 => (24, hound::SampleFormat::Int),
        };
        let spec = hound::WavSpec {
            channels: layout.channels as u16,
            sample_rate: layout.sample_rate,
            bits_per_sample: bits,
            sample_format: format,
        };
        let mut writer = hound::WavWriter::new(BufWriter::new(temp.as_file_mut()), spec)?;
        let frames = audio.len() / layout.channels;
        for frame in 0..frames {
            for channel in 0..layout.channels {
                let sample = audio[channel * frames + frame];
                if format == hound::SampleFormat::Float {
                    writer.write_sample(sample)?;
                } else {
                    let scale = f64::from(1 << (bits - 1));
                    let sample = (f64::from(sample) * scale)
                        .round_ties_even()
                        .clamp(-scale, scale - 1.0);
                    writer.write_sample(sample as i32)?;
                }
            }
        }
        writer.finalize()?;
    } else {
        encode_stem(root, temp.path(), audio, layout, format, encoding)
            .with_context(|| format!("encoding {}", path.display()))?;
    }
    ensure!(
        temp.as_file().metadata()?.len() > 0,
        "no output for {}",
        path.display()
    );
    Ok(temp)
}

fn encode_stem(
    root: &Path,
    temporary: &Path,
    audio: &[f32],
    layout: PcmLayout,
    format: Format,
    encoding: &AudioEncoding,
) -> Result<()> {
    let mp3_rate = encoding.mp3_bit_rate.to_string();
    let m4a_rate = encoding.m4a_bit_rate.to_string();
    let m4a_quality = encoding.m4a_aac_at_quality.to_string();
    let options = match format {
        Format::Wav => bail!("WAV separation output uses hound"),
        Format::Flac => {
            let (samples, depth) = match encoding.flac {
                FlacDepth::Pcm16 => ("s16", "16"),
                FlacDepth::Pcm24 => ("s32", "24"),
            };
            vec![
                "-c:a",
                "flac",
                "-sample_fmt",
                samples,
                "-bits_per_raw_sample",
                depth,
                "-f",
                "flac",
            ]
        }
        Format::Mp3 => vec!["-c:a", "libmp3lame", "-b:a", &mp3_rate, "-f", "mp3"],
        Format::M4a => {
            let mut args = vec![
                "-c:a",
                encoding.m4a_codec.as_str(),
                "-b:a",
                &m4a_rate,
                "-f",
                "ipod",
            ];
            if encoding.m4a_codec == "aac_at" {
                args.extend(["-aac_at_quality", &m4a_quality]);
            }
            args
        }
    };
    let mut child = ffmpeg(root, "ffmpeg")
        .args([
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "f32le",
            "-ar",
            &layout.sample_rate.to_string(),
            "-ac",
            &layout.channels.to_string(),
            "-i",
            "pipe:0",
            "-vn",
        ])
        .args(options)
        .arg(temporary)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .context("starting FFmpeg to encode separation output")?;
    let written = (|| -> Result<()> {
        let mut stdin = BufWriter::new(child.stdin.take().context("FFmpeg has no input pipe")?);
        let frames = audio.len() / layout.channels;
        for frame in 0..frames {
            for channel in 0..layout.channels {
                stdin.write_all(&audio[channel * frames + frame].to_le_bytes())?;
            }
        }
        stdin.flush()?;
        Ok(())
    })();
    let output = child.wait_with_output()?;
    ensure!(
        output.status.success(),
        "FFmpeg: {}",
        String::from_utf8_lossy(&output.stderr).trim()
    );
    written
}

fn persist_stem(temporary: NamedTempFile, path: &Path, overwrite: bool) -> Result<()> {
    let result = if overwrite {
        temporary.persist(path)
    } else {
        temporary.persist_noclobber(path)
    };
    result.with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}
