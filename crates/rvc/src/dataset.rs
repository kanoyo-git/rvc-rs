mod manifest;
mod slicer;

use std::collections::BTreeSet;
use std::f64::consts::PI;
use std::ffi::OsStr;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc;

use anyhow::{Context, Result, ensure};
use rayon::prelude::*;
use serde::Serialize;
use tempfile::TempDir;

use self::slicer::Slicer;
use crate::audio;

pub use manifest::{
    ManifestRowInput, load_manifest, manifest_from_rows, save_manifest, scan_speakers,
};

const GROUND_TRUTH: &str = "0_gt_wavs";
const ANALYSIS_WAV: &str = "1_16k_wavs";
const OVERLAP_SECONDS: f64 = 0.3;

#[derive(Clone, Copy)]
pub enum DatasetSource<'a> {
    Single(&'a Path),
    MultiFolder(&'a Path),
    MultiManifest,
}

pub struct PreprocessRequest<'a> {
    pub source: DatasetSource<'a>,
    pub experiment_dir: &'a Path,
    pub sample_rate: u32,
    pub workers: usize,
    pub segment_seconds: f64,
}

#[derive(Debug, Serialize)]
pub struct FileFailure {
    pub path: PathBuf,
    pub error: String,
}

#[derive(Debug, Serialize)]
pub struct PreprocessReport {
    pub source_files: usize,
    pub written_clips: usize,
    pub skipped_clips: usize,
    pub failed_files: Vec<FileFailure>,
}

pub fn preprocess(
    request: &PreprocessRequest<'_>,
    report: impl FnMut(&str),
) -> Result<PreprocessReport> {
    preprocess_with_cancel(request, || false, report)
}

pub fn preprocess_with_cancel(
    request: &PreprocessRequest<'_>,
    is_cancelled: impl Fn() -> bool + Sync,
    report: impl FnMut(&str),
) -> Result<PreprocessReport> {
    preprocess_with_denoise(request, 0.0, is_cancelled, report)
}

pub(crate) fn preprocess_with_denoise(
    request: &PreprocessRequest<'_>,
    denoise: f32,
    is_cancelled: impl Fn() -> bool + Sync,
    mut report: impl FnMut(&str),
) -> Result<PreprocessReport> {
    ensure!(!is_cancelled(), "dataset preprocessing cancelled");
    ensure!(
        (0.0..=1.0).contains(&denoise),
        "denoise strength must be between 0 and 1"
    );
    let sample_rate = request.sample_rate;
    let segment = request.segment_seconds;
    ensure!(
        sample_rate > 96,
        "sample rate must exceed 96 Hz for the 48 Hz high-pass filter"
    );
    let step = f64::from(sample_rate) * (segment - OVERLAP_SECONDS);
    ensure!(
        segment.is_finite() && step >= 1.0,
        "segment duration must exceed the 0.3-second overlap"
    );
    fs::create_dir_all(request.experiment_dir)?;
    let (sources, pending_manifest) = collect_sources(request)?;
    ensure!(!sources.is_empty(), "no input files to preprocess");
    ensure!(!is_cancelled(), "dataset preprocessing cancelled");

    let staging = TempDir::new_in(request.experiment_dir)?;
    fs::create_dir(staging.path().join(GROUND_TRUTH))?;
    fs::create_dir(staging.path().join(ANALYSIS_WAV))?;
    if let Some(manifest) = pending_manifest {
        save_manifest(staging.path(), &manifest)?;
    }
    let segmenter = Segmenter {
        slicer: Slicer::new(sample_rate),
        filter: Butterworth::highpass(sample_rate),
        staging: staging.path(),
        sample_rate,
        samples_per_segment: (f64::from(sample_rate) * segment) as usize,
        tail_limit: f64::from(sample_rate) * (segment + OVERLAP_SECONDS),
        step,
        denoise,
    };
    let total = sources.len();
    let workers = request.workers.clamp(1, total);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()
        .context("starting dataset workers")?;

    let log = fs::File::create(request.experiment_dir.join("preprocess.log"))?;
    let mut log = BufWriter::new(log);
    let mut emit = |line: &str| -> Result<()> {
        report(line);
        writeln!(&mut log, "{line}")?;
        log.flush()?;
        Ok(())
    };
    emit(&format!(
        "Preprocessing {total} input file(s) with {workers} worker(s)"
    ))?;
    let mut skipped_clips = 0;
    let mut failed_files = Vec::new();
    let is_cancelled = &is_cancelled;
    pool.in_place_scope(|scope| -> Result<()> {
        let (sender, receiver) = mpsc::channel();
        scope.spawn(move |_| {
            sources
                .par_iter()
                .for_each_with(sender, |sender, (path, output_key)| {
                    if is_cancelled() {
                        return;
                    }
                    let result = segmenter.preprocess_file(path, output_key);
                    let _ = sender.send((path.clone(), result));
                });
        });
        for (index, (path, result)) in receiver.into_iter().enumerate() {
            match result {
                Ok(skipped) => skipped_clips += skipped,
                Err(error) => {
                    let message = format!("{error:#}");
                    emit(&format!("Failed {} ({message})", path.display()))?;
                    failed_files.push(FileFailure {
                        path: path.clone(),
                        error: message,
                    });
                }
            }
            let done = index + 1;
            if done == 1 || done == total || done % total.div_ceil(12) == 0 {
                emit(&format!("Processed {done}/{total} ({})", path.display()))?;
            }
        }
        Ok(())
    })?;
    failed_files.sort_by(|a, b| a.path.cmp(&b.path));
    ensure!(!is_cancelled(), "dataset preprocessing cancelled");
    let written_clips = publish_clips(staging.path(), request.experiment_dir, is_cancelled)?;
    let pending_manifest = staging.path().join(manifest::MANIFEST_FILENAME);
    if pending_manifest.is_file() {
        let output = request.experiment_dir.join(manifest::MANIFEST_FILENAME);
        fs::rename(pending_manifest, &output)
            .with_context(|| format!("publishing {}", output.display()))?;
    }
    emit(&format!(
        "Prepared {written_clips} clip(s), skipped {skipped_clips}, failed {} file(s)",
        failed_files.len()
    ))?;
    Ok(PreprocessReport {
        source_files: total,
        written_clips,
        skipped_clips,
        failed_files,
    })
}

type SourceFile = (PathBuf, String);

fn collect_sources(
    request: &PreprocessRequest<'_>,
) -> Result<(Vec<SourceFile>, Option<manifest::Manifest>)> {
    let (manifest, save) = match request.source {
        DatasetSource::Single(root) => {
            let sources = manifest::file_paths(root)?
                .into_iter()
                .filter(|path| path.is_file())
                .enumerate()
                .map(|(index, path)| (path, index.to_string()))
                .collect();
            return Ok((sources, None));
        }
        DatasetSource::MultiFolder(root) => (scan_speakers(root)?, true),
        DatasetSource::MultiManifest => (load_manifest(request.experiment_dir)?, false),
    };
    let sources = manifest
        .entries
        .iter()
        .map(|entry| (entry.path.clone(), entry.output_key.clone()))
        .collect();
    Ok((sources, save.then_some(manifest)))
}

struct Segmenter<'a> {
    slicer: Slicer,
    filter: Butterworth,
    staging: &'a Path,
    sample_rate: u32,
    samples_per_segment: usize,
    tail_limit: f64,
    step: f64,
    denoise: f32,
}

impl Segmenter<'_> {
    fn preprocess_file(&self, path: &Path, output_key: &str) -> Result<usize> {
        let samples = audio::read_denoised(path, self.sample_rate, self.denoise)?;
        let filtered = self.filter.apply(&samples);
        let mut output_id = 0;
        let mut final_tail = None;
        let mut skipped = 0;
        for range in self.slicer.slices(&filtered) {
            let clip = &filtered[range];
            let mut window = 0;
            loop {
                let start = (self.step * f64::from(window)) as usize;
                let remainder = &clip[start..];
                if remainder.len() as f64 <= self.tail_limit {
                    // The original emits only the last silence slice's terminal window.
                    final_tail = Some(remainder);
                    output_id += 1;
                    break;
                }
                let frame = &remainder[..self.samples_per_segment];
                skipped += usize::from(!write_clip(
                    self.staging,
                    frame,
                    self.sample_rate,
                    output_key,
                    output_id,
                )?);
                output_id += 1;
                window += 1;
            }
        }
        let tail = final_tail.context("silence slicing produced no audio")?;
        skipped += usize::from(!write_clip(
            self.staging,
            tail,
            self.sample_rate,
            output_key,
            output_id,
        )?);
        Ok(skipped)
    }
}

fn write_clip(
    staging: &Path,
    samples: &[f64],
    sample_rate: u32,
    output_key: &str,
    output_id: usize,
) -> Result<bool> {
    let peak = samples.iter().try_fold(0.0f64, |peak, sample| {
        sample.is_finite().then(|| peak.max(sample.abs()))
    });
    let Some(peak) = peak.filter(|&peak| peak > 0.0 && peak <= 2.5) else {
        return Ok(false);
    };
    let normalized: Vec<_> = samples
        .iter()
        .map(|&sample| (sample / peak * (0.9 * 0.75) + 0.25 * sample) as f32)
        .collect();
    let analysis = resample_clip(&normalized, sample_rate);
    let ground = audio::encode_wav(&normalized, sample_rate, true)?;
    let analysis = audio::encode_wav(&analysis, audio::ANALYSIS_RATE, true)?;
    let file = format!("{output_key}_{output_id}.wav");
    audio::write_file(&staging.join(GROUND_TRUTH).join(&file), &ground, false)?;
    audio::write_file(&staging.join(ANALYSIS_WAV).join(&file), &analysis, false)?;
    Ok(true)
}

fn resample_clip(samples: &[f32], rate: u32) -> Vec<f32> {
    let target = audio::ANALYSIS_RATE;
    if rate == target {
        return samples.to_vec();
    }
    let mut common = rate;
    let mut remainder = target;
    while remainder != 0 {
        (common, remainder) = (remainder, common % remainder);
    }
    let input_period = (rate / common) as usize;
    let padding = (32 * rate.div_ceil(target) as usize).div_ceil(input_period) * input_period;
    let mut padded = vec![0.0; samples.len() + padding * 2];
    padded[padding..padding + samples.len()].copy_from_slice(samples);
    let output = audio::resample(&padded, rate, target);
    // The source resampler assumes silence outside the clip. Align the padding to its sample-rate ratio.
    let rate = rate as usize;
    let start = padding * target as usize / rate;
    let scaled = samples.len() * target as usize;
    let rounded = (scaled + rate / 2) / rate;
    // soxr rounds its output length. librosa then pads to the ceiling of the ratio.
    let mut clip = output[start..start + rounded].to_vec();
    clip.resize(scaled.div_ceil(rate), 0.0);
    clip
}

fn publish_clips(
    staging: &Path,
    experiment: &Path,
    is_cancelled: &impl Fn() -> bool,
) -> Result<usize> {
    let staged = manifest::file_paths(&staging.join(GROUND_TRUTH))?;
    let mut filenames = BTreeSet::new();
    for ground in &staged {
        let filename = ground.file_name().context("clip has no filename")?;
        ensure!(
            staging.join(ANALYSIS_WAV).join(filename).is_file(),
            "analysis WAV was not created for {}",
            ground.display()
        );
        filenames.insert(filename);
    }
    ensure!(
        !filenames.is_empty(),
        "preprocessing produced no usable audio clips"
    );
    ensure!(!is_cancelled(), "dataset preprocessing cancelled");
    for directory in [GROUND_TRUTH, ANALYSIS_WAV] {
        let output = experiment.join(directory);
        fs::create_dir_all(&output)?;
        for filename in &filenames {
            let target = output.join(filename);
            fs::rename(staging.join(directory).join(filename), &target)
                .with_context(|| format!("publishing {}", target.display()))?;
        }
        for path in manifest::file_paths(&output)? {
            let Some(filename) = path.file_name() else {
                continue;
            };
            if !filenames.contains(filename)
                && filename.to_str().is_some_and(|name| name.starts_with("ms"))
                && path.extension() == Some(OsStr::new("wav"))
                && path.is_file()
            {
                fs::remove_file(&path)
                    .with_context(|| format!("removing stale clip {}", path.display()))?;
            }
        }
    }
    Ok(filenames.len())
}

struct Butterworth {
    numerator: [f64; 6],
    denominator: [f64; 6],
}

impl Butterworth {
    fn highpass(sample_rate: u32) -> Self {
        let warped = (PI * 48.0 / f64::from(sample_rate)).tan();
        let mut denominator = [0.0; 6];
        denominator[0] = 1.0;
        denominator[1] = -(1.0 - warped) / (1.0 + warped);
        let mut gain = (1.0 + warped).recip();
        for (pair, angle) in [PI / 5.0, 2.0 * PI / 5.0].into_iter().enumerate() {
            let twice_real = -2.0 * warped * angle.cos();
            let scale = 1.0 - twice_real + warped * warped;
            let pole_real = (1.0 - warped * warped) / scale;
            let pole_norm = (1.0 + twice_real + warped * warped) / scale;
            gain /= scale;
            let previous = denominator;
            denominator.fill(0.0);
            for (power, &coefficient) in previous.iter().take(2 + 2 * pair).enumerate() {
                denominator[power] += coefficient;
                denominator[power + 1] -= 2.0 * pole_real * coefficient;
                denominator[power + 2] += pole_norm * coefficient;
            }
        }
        let numerator = [1.0, -5.0, 10.0, -10.0, 5.0, -1.0].map(|x| x * gain);
        Self {
            numerator,
            denominator,
        }
    }

    fn apply(&self, samples: &[f32]) -> Vec<f64> {
        let mut state = [0.0; 6];
        samples
            .iter()
            .map(|&sample| {
                let sample = f64::from(sample);
                let output = self.numerator[0] * sample + state[0];
                for index in 0..state.len() - 1 {
                    state[index] = self.numerator[index + 1] * sample
                        - self.denominator[index + 1] * output
                        + state[index + 1];
                }
                output
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, Ordering};

    use super::*;

    #[test]
    fn cancelling_folder_preparation_preserves_published_dataset() -> Result<()> {
        let workspace = TempDir::new()?;
        let source = workspace.path().join("collection");
        let speaker = source.join("Ada_2_1");
        fs::create_dir_all(&speaker)?;
        fs::write(speaker.join("voice.wav"), b"decoding should not start")?;
        let experiment = workspace.path().join("experiment");
        let ground = experiment.join(GROUND_TRUTH);
        fs::create_dir_all(&ground)?;
        let published = ground.join("ms0000_s001_0000000000_1.wav");
        let old_manifest = experiment.join(manifest::MANIFEST_FILENAME);
        fs::write(&published, b"old audio")?;
        fs::write(&old_manifest, b"old manifest")?;
        let cancelled = AtomicBool::new(false);
        let request = PreprocessRequest {
            source: DatasetSource::MultiFolder(&source),
            experiment_dir: &experiment,
            sample_rate: 32_000,
            workers: 1,
            segment_seconds: 3.7,
        };
        let error = preprocess_with_cancel(
            &request,
            || cancelled.load(Ordering::Relaxed),
            |_| cancelled.store(true, Ordering::Relaxed),
        )
        .expect_err("the observer requested cancellation");
        assert_eq!(error.to_string(), "dataset preprocessing cancelled");
        assert_eq!(fs::read(published)?, b"old audio");
        assert_eq!(fs::read(old_manifest)?, b"old manifest");
        assert!(!experiment.join(ANALYSIS_WAV).exists());
        Ok(())
    }
}
