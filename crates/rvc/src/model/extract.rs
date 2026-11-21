use std::ffi::OsString;
use std::fs::{self, File, OpenOptions};
use std::ops::ControlFlow;
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    mpsc,
};

use anyhow::{Context, Result, bail, ensure};
use ndarray::{Array1, Array2};
use ndarray_npy::WriteNpyExt;
use serde::{Deserialize, Serialize};
use tch::{Device, Kind};
use tempfile::NamedTempFile;

use super::train_data::Version;
use crate::audio::{self, ANALYSIS_RATE};
use crate::catalog::{Embedder, Paths};

use super::{Hubert, Pitch, PitchMethod, coarse_pitch, interpolate_voiced};

#[derive(Clone, Copy)]
pub struct Request<'a> {
    pub experiment: &'a Path,
    pub version: &'a str,
    pub f0: Option<PitchMethod>,
    pub device: Device,
    pub overwrite: bool,
    /// Clear F0 on voiceless fricative frames (see `aperiodic::unvoice_fricatives`).
    pub unvoice_fricatives: bool,
}

#[derive(Default, Serialize)]
pub struct Report {
    pub features: usize,
    pub pitch: usize,
    pub unvoiced: usize,
    pub cached: usize,
}

pub fn run(
    paths: &Paths,
    request: Request<'_>,
    mut progress: impl FnMut(&str) -> ControlFlow<()>,
) -> Result<Report> {
    run_on_devices(paths, request, &[], &mut progress)
}

pub fn run_on_devices(
    paths: &Paths,
    request: Request<'_>,
    devices: &[Device],
    mut progress: impl FnMut(&str) -> ControlFlow<()>,
) -> Result<Report> {
    run_with_embedder(paths, request, devices, None, &mut progress)
}

const SOURCE_FILE: &str = ".embedder.json";
pub(super) const RAW_F0: &str = "2d-f0raw";

#[derive(Clone, Copy, Deserialize, Serialize)]
struct FeatureSource {
    embedder: Embedder,
    complete: bool,
    #[serde(default)]
    replacing: bool,
}

fn read_source(directory: &Path) -> Result<Option<FeatureSource>> {
    let path = directory.join(SOURCE_FILE);
    match File::open(&path) {
        Ok(file) => Ok(Some(
            serde_json::from_reader(file).with_context(|| format!("reading {}", path.display()))?,
        )),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error.into()),
    }
}

pub(super) fn configured_embedder(experiment: &Path) -> Result<Option<Embedder>> {
    match File::open(experiment.join("config.json")) {
        Ok(file) => {
            let config: serde_json::Value = serde_json::from_reader(file)?;
            config
                .get("embedder_model")
                .filter(|value| !value.is_null())
                .map(|value| serde_json::from_value(value.clone()))
                .transpose()
                .context("invalid embedder_model in training config")
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error.into()),
    }
}

pub(super) fn feature_embedder(experiment: &Path, version: Version) -> Result<Option<Embedder>> {
    let directory = experiment.join(format!("3_feature{}", version.dimensions()));
    if let Some(source) = read_source(&directory)? {
        ensure!(
            source.complete,
            "feature extraction was interrupted; run extraction again before training"
        );
        Ok(Some(source.embedder))
    } else {
        Ok(None)
    }
}

pub fn run_with_embedder(
    paths: &Paths,
    request: Request<'_>,
    devices: &[Device],
    requested: Option<Embedder>,
    mut progress: impl FnMut(&str) -> ControlFlow<()>,
) -> Result<Report> {
    let lock = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(request.experiment.join(".training.lock"))
        .context("open the preprocessed experiment before extracting features")?;
    lock.try_lock()
        .context("another process is extracting features or training this experiment")?;
    let mut input = Input::prepare(request)?;
    let existing = read_source(&input.feature_dir)?;
    let configured = configured_embedder(request.experiment)?;
    let recorded = existing
        .map(|source| source.embedder)
        .or(configured)
        .unwrap_or_default();
    let embedder = requested.unwrap_or(recorded);
    embedder.supports(request.version)?;
    if embedder != recorded && !request.overwrite {
        let mut already_extracted = false;
        for entry in fs::read_dir(&input.feature_dir)? {
            if entry?.path().extension().is_some_and(|ext| ext == "npy") {
                already_extracted = true;
                break;
            }
        }
        ensure!(
            !already_extracted,
            "cached features use {}; enable overwrite to extract {}",
            recorded.name(),
            embedder.name()
        );
    }
    input.replace_features =
        embedder != recorded || existing.is_some_and(|source| !source.complete && source.replacing);
    input.hubert_root = embedder.install(paths, &mut progress)?;
    let marker = input.feature_dir.join(SOURCE_FILE);
    let mark = |complete| {
        audio::write_file(
            &marker,
            &serde_json::to_vec_pretty(&FeatureSource {
                embedder,
                complete,
                replacing: !complete && input.replace_features,
            })?,
            true,
        )
    };
    mark(false)?;
    let result = extract_on_devices(paths, request, devices, &input, progress)?;
    mark(true)?;
    Ok(result)
}

fn extract_on_devices(
    paths: &Paths,
    request: Request<'_>,
    devices: &[Device],
    input: &Input,
    mut progress: impl FnMut(&str) -> ControlFlow<()>,
) -> Result<Report> {
    if devices.len() <= 1 {
        return extract(
            paths,
            request,
            input,
            0,
            1,
            devices.first().copied().unwrap_or(request.device),
            &mut progress,
        );
    }
    ensure!(devices.len() <= 64, "too many extraction devices");
    let cancelled = AtomicBool::new(false);
    std::thread::scope(|scope| {
        let (update, progress_updates) = mpsc::channel();
        let mut workers = Vec::with_capacity(devices.len());
        for (rank, &device) in devices.iter().enumerate() {
            let update = update.clone();
            let cancelled = &cancelled;
            workers.push(scope.spawn(move || {
                let result = extract(paths, request, input, rank, devices.len(), device, |line| {
                    if cancelled.load(Ordering::Relaxed) || update.send(line.to_owned()).is_err() {
                        ControlFlow::Break(())
                    } else {
                        ControlFlow::Continue(())
                    }
                });
                if result.is_err() {
                    cancelled.store(true, Ordering::Relaxed);
                }
                result
            }));
        }
        drop(update);
        for line in progress_updates {
            if progress(&line).is_break() {
                cancelled.store(true, Ordering::Relaxed);
            }
        }
        let mut total = Report::default();
        let mut error = None;
        for worker in workers {
            match worker
                .join()
                .map_err(|_| anyhow::anyhow!("feature extraction worker panicked"))?
            {
                Ok(result) => {
                    total.features += result.features;
                    total.pitch += result.pitch;
                    total.unvoiced += result.unvoiced;
                    total.cached += result.cached;
                }
                Err(failure) => {
                    if error.is_none() || failure.to_string() != "feature extraction cancelled" {
                        error = Some(failure);
                    }
                }
            }
        }
        match error {
            Some(error) => Err(error),
            None => Ok(total),
        }
    })
}

pub fn selected_devices(requested: &str, indices: &[usize]) -> Result<Vec<Device>> {
    let Some(first) = indices.first() else {
        return Ok(vec![super::weights::device(requested)?]);
    };
    ensure!(
        matches!(requested, "auto" | "cuda") || requested == format!("cuda:{first}"),
        "--gpus requires CUDA and must agree with --device"
    );
    ensure!(
        indices
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len()
            == indices.len(),
        "CUDA extraction device IDs must be unique"
    );
    indices
        .iter()
        .map(|index| super::weights::device(&format!("cuda:{index}")))
        .collect()
}

struct Input {
    v2: bool,
    dims: usize,
    hubert_root: PathBuf,
    replace_features: bool,
    wavs: Vec<PathBuf>,
    feature_dir: PathBuf,
    pitch_dir: PathBuf,
    hz_dir: PathBuf,
    raw_dir: PathBuf,
}

impl Input {
    fn prepare(request: Request<'_>) -> Result<Self> {
        let v2 = request.version == "v2";
        ensure!(v2 || request.version == "v1", "unsupported RVC version");
        let dims = if v2 { 768 } else { 256 };
        let mut wavs = audio::files(&request.experiment.join("1_16k_wavs"), false)?;
        wavs.retain(|path| {
            path.extension()
                .is_some_and(|extension| extension.eq_ignore_ascii_case("wav"))
        });
        ensure!(!wavs.is_empty(), "no clips in 1_16k_wavs");
        let feature_dir = request.experiment.join(format!("3_feature{dims}"));
        let pitch_dir = request.experiment.join("2a_f0");
        let hz_dir = request.experiment.join("2b-f0nsf");
        let raw_dir = request.experiment.join(RAW_F0);
        std::fs::create_dir_all(&feature_dir)?;
        if request.f0.is_some() {
            std::fs::create_dir_all(&pitch_dir)?;
            fs::create_dir_all(&hz_dir)?;
            fs::create_dir_all(&raw_dir)?;
        }
        Ok(Self {
            v2,
            dims,
            hubert_root: PathBuf::new(),
            replace_features: false,
            wavs,
            feature_dir,
            pitch_dir,
            hz_dir,
            raw_dir,
        })
    }
}

fn extract(
    paths: &Paths,
    request: Request<'_>,
    input: &Input,
    rank: usize,
    ranks: usize,
    device: Device,
    mut progress: impl FnMut(&str) -> ControlFlow<()>,
) -> Result<Report> {
    tch::no_grad(|| {
        let kind = match device {
            Device::Cuda(index)
                if super::weights::cuda_profile(index).is_none_or(|profile| profile.fp16) =>
            {
                Kind::Half
            }
            _ => Kind::Float,
        };
        let mut hubert = None;
        let mut f0 = None;
        let mut report = Report::default();
        for (index, wav) in input.wavs.iter().enumerate().skip(rank).step_by(ranks) {
            if progress(&format!(
                "{}/{} - {}",
                index + 1,
                input.wavs.len(),
                wav.display()
            ))
            .is_break()
            {
                bail!("feature extraction cancelled");
            }
            let file_name = wav.file_name().context("clip has no file name")?;
            let feature_file = input.feature_dir.join(file_name).with_extension("npy");
            let mut f0_name = OsString::from(file_name);
            f0_name.push(".npy");
            let pitch_file = input.pitch_dir.join(&f0_name);
            let hz_file = input.hz_dir.join(&f0_name);
            let raw_file = input.raw_dir.join(&f0_name);
            let need_feature =
                request.overwrite || input.replace_features || !feature_file.is_file();
            let write_legacy_pitch =
                request.overwrite || !pitch_file.is_file() || !hz_file.is_file();
            let need_pitch = request
                .f0
                .filter(|_| write_legacy_pitch || !raw_file.is_file());
            if !need_feature && need_pitch.is_none() {
                report.cached += 1;
                continue;
            }
            let wave = audio::read_mono(wav, ANALYSIS_RATE)?;
            if let Some(method) = need_pitch {
                if f0.is_none() {
                    f0 = Some(Pitch::load(method, paths, device, kind)?);
                }
                let mut hz = f0
                    .as_ref()
                    .context("pitch estimator unavailable")?
                    .extract(&wave, None)?;
                for value in &mut hz {
                    if !value.is_finite() || *value < 0.0 {
                        *value = 0.0;
                    }
                }
                if request.unvoice_fricatives {
                    super::aperiodic::unvoice_fricatives(&wave, &mut hz);
                }
                npy(&raw_file, &ndarray::ArrayView1::from(&hz))?;
                if hz.iter().any(|&v| v > 0.0) {
                    if write_legacy_pitch {
                        interpolate_voiced(&mut hz);
                        let bands = Array1::from_iter(hz.iter().map(|&value| coarse_pitch(value)));
                        npy(&pitch_file, &bands)?;
                        npy(&hz_file, &Array1::from(hz))?;
                    }
                    report.pitch += 1;
                } else {
                    if write_legacy_pitch && (pitch_file.is_file() || hz_file.is_file()) {
                        // A changed pitch method can leave no voiced frames in a
                        // previously extracted clip. Replace its stale tracks.
                        npy(&pitch_file, &Array1::from_elem(hz.len(), 1i64))?;
                        npy(&hz_file, &Array1::from(hz))?;
                    }
                    report.unvoiced += 1;
                }
            }
            if need_feature {
                if hubert.is_none() {
                    hubert = Some(Hubert::load(&input.hubert_root, device, kind, true)?);
                }
                let x = hubert
                    .as_ref()
                    .context("HuBERT unavailable")?
                    .extract(&wave, input.v2)?;
                let n = x.size()[1] as usize;
                let values =
                    Vec::<f32>::try_from(x.to_kind(Kind::Float).to_device(Device::Cpu).view([-1]))?;
                ensure!(
                    values.iter().all(|v| v.is_finite()),
                    "HuBERT returned non-finite values"
                );
                npy(
                    &feature_file,
                    &Array2::from_shape_vec((n, input.dims), values)?,
                )?;
                report.features += 1;
            }
        }
        Ok(report)
    })
}

fn npy(path: &Path, data: &impl WriteNpyExt) -> Result<()> {
    let directory = path.parent().context("array file needs a parent folder")?;
    let mut staged = NamedTempFile::new_in(directory)?;
    data.write_npy(staged.as_file_mut())?;
    staged.persist(path)?;
    Ok(())
}
