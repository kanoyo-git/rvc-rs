use std::ffi::OsString;
use std::ops::ControlFlow;
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    mpsc,
};

use anyhow::{Context, Result, bail, ensure};
use ndarray::{Array1, Array2};
use ndarray_npy::WriteNpyExt;
use serde::Serialize;
use tch::{Device, Kind};
use tempfile::NamedTempFile;

use crate::audio::{self, ANALYSIS_RATE};
use crate::catalog::Paths;

use super::{Hubert, Pitch, PitchMethod, coarse_pitch, interpolate_voiced};

#[derive(Clone, Copy)]
pub struct Request<'a> {
    pub experiment: &'a Path,
    pub version: &'a str,
    pub f0: Option<PitchMethod>,
    pub device: Device,
    pub overwrite: bool,
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
    let input = Input::prepare(paths, request)?;
    if devices.len() <= 1 {
        return extract(
            paths,
            request,
            &input,
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
            let input = &input;
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
    normalize: bool,
    hubert_root: PathBuf,
    wavs: Vec<PathBuf>,
    feature_dir: PathBuf,
    pitch_dir: PathBuf,
    hz_dir: PathBuf,
}

impl Input {
    fn prepare(paths: &Paths, request: Request<'_>) -> Result<Self> {
        let v2 = request.version == "v2";
        ensure!(v2 || request.version == "v1", "unsupported RVC version");
        let dims = if v2 { 768 } else { 256 };
        let hubert_root = paths.root.join("assets/hubert_base");
        let normalize = serde_json::from_reader::<_, serde_json::Value>(std::fs::File::open(
            hubert_root.join("preprocessor_config.json"),
        )?)?
        .get("do_normalize")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
        let mut wavs = audio::files(&request.experiment.join("1_16k_wavs"), false)?;
        wavs.retain(|path| {
            path.extension()
                .is_some_and(|extension| extension.eq_ignore_ascii_case("wav"))
        });
        ensure!(!wavs.is_empty(), "no clips in 1_16k_wavs");
        let feature_dir = request.experiment.join(format!("3_feature{dims}"));
        let pitch_dir = request.experiment.join("2a_f0");
        let hz_dir = request.experiment.join("2b-f0nsf");
        std::fs::create_dir_all(&feature_dir)?;
        if request.f0.is_some() {
            std::fs::create_dir_all(&pitch_dir)?;
            std::fs::create_dir_all(&hz_dir)?;
        }
        Ok(Self {
            v2,
            dims,
            normalize,
            hubert_root,
            wavs,
            feature_dir,
            pitch_dir,
            hz_dir,
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
            let need_feature = request.overwrite || !feature_file.is_file();
            let need_pitch = request
                .f0
                .filter(|_| request.overwrite || !pitch_file.is_file() || !hz_file.is_file());
            if !need_feature && need_pitch.is_none() {
                report.cached += 1;
                continue;
            }
            let mut wave = audio::read_mono(wav, ANALYSIS_RATE)?;
            if let Some(method) = need_pitch {
                if f0.is_none() {
                    f0 = Some(Pitch::load(method, paths, device, kind)?);
                }
                let mut hz = f0
                    .as_ref()
                    .context("pitch estimator unavailable")?
                    .extract(&wave, None)?;
                if hz.iter().any(|&v| v > 0.0) {
                    interpolate_voiced(&mut hz);
                    let bands = Array1::from_iter(hz.iter().map(|&value| coarse_pitch(value)));
                    npy(&pitch_file, &bands)?;
                    npy(&hz_file, &Array1::from(hz))?;
                    report.pitch += 1;
                } else {
                    report.unvoiced += 1;
                }
            }
            if need_feature {
                if hubert.is_none() {
                    hubert = Some(if kind == Kind::Float {
                        Hubert::load(&input.hubert_root, device)?
                    } else {
                        Hubert::load_with_kind(&input.hubert_root, device, kind)?
                    });
                }
                if input.normalize {
                    let mean = wave.iter().map(|v| f64::from(*v)).sum::<f64>() / wave.len() as f64;
                    let variance = wave
                        .iter()
                        .map(|&v| (f64::from(v) - mean).powi(2))
                        .sum::<f64>()
                        / wave.len() as f64;
                    let scale = (variance + 1e-5).sqrt().recip();
                    wave.iter_mut()
                        .for_each(|v| *v = ((f64::from(*v) - mean) * scale) as f32);
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
