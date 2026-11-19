mod events;
mod parallel;
mod precision;
mod step;

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs::{self, File, OpenOptions};
use std::io::{ErrorKind, Read, Seek, SeekFrom, Write};
use std::ops::ControlFlow;
use std::path::{Component, Path, PathBuf};

use anyhow::{Context, Result, ensure};
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tch::{Device, Kind, Tensor};

use super::discriminator::Discriminator;
use super::edit::{self, Export};
use super::optimizer::{AdamW, ExponentialLr};
use super::synth::training::generator;
use super::train_data::{self, PrepareRequest, SpeakerMode, Version};
use super::train_losses::{self, MelConfig, MelSpectrogram};
use super::weights;
use crate::catalog::Paths;
use crate::checkpoint::{
    Checkpoint,
    write::{self, Value},
};
use events::Events;
use precision::GradScaler;

#[derive(clap::Args, Deserialize)]
#[serde(default)]
pub struct Request {
    #[arg(long)]
    pub experiment: PathBuf,
    #[arg(long)]
    pub name: Option<String>,
    #[arg(long, default_value_t = 40_000)]
    pub sample_rate: u32,
    #[arg(long, value_enum, default_value = "v2")]
    pub version: Version,
    #[arg(long, action = clap::ArgAction::Set, default_value_t = true)]
    pub f0: bool,
    #[arg(long, default_value_t = 0)]
    pub speaker_id: i64,
    #[arg(long)]
    pub multi: bool,
    /// Rebuild the training filelist from preprocessed clips and features.
    #[arg(long)]
    #[serde(skip)]
    pub rebuild_filelist: bool,
    #[arg(long, default_value_t = 20)]
    pub epochs: usize,
    #[arg(long)]
    pub batch_size: Option<usize>,
    #[arg(long, default_value_t = 5)]
    pub save_every: usize,
    #[arg(long)]
    pub save_weights: bool,
    #[arg(long)]
    pub latest_only: bool,
    #[arg(long, default_value = "auto")]
    pub device: String,
    #[arg(long, value_delimiter = ',')]
    pub gpus: Vec<usize>,
    #[arg(long, action = clap::ArgAction::Set)]
    pub fp16: Option<bool>,
    #[arg(long)]
    #[serde(alias = "if_cache_data_in_gpu")]
    pub cache_data_in_gpu: bool,
    #[arg(long, action = clap::ArgAction::Set, default_value_t = true)]
    pub tensorboard: bool,
    #[arg(long)]
    pub pretrained_g: Option<PathBuf>,
    #[arg(long)]
    pub pretrained_d: Option<PathBuf>,
    #[arg(long)]
    pub fresh: bool,
    #[arg(long)]
    pub random_weights: bool,
}

impl Default for Request {
    fn default() -> Self {
        Self {
            experiment: PathBuf::new(),
            name: None,
            sample_rate: 40_000,
            version: Version::V2,
            f0: true,
            speaker_id: 0,
            multi: false,
            rebuild_filelist: false,
            epochs: 20,
            batch_size: None,
            save_every: 5,
            save_weights: false,
            latest_only: false,
            device: "auto".into(),
            gpus: Vec::new(),
            fp16: None,
            cache_data_in_gpu: false,
            tensorboard: true,
            pretrained_g: None,
            pretrained_d: None,
            fresh: false,
            random_weights: false,
        }
    }
}

#[derive(Serialize)]
pub struct Report {
    pub model: PathBuf,
    pub epochs_completed: usize,
    pub steps: usize,
}

pub fn run(
    paths: &Paths,
    request: Request,
    progress: impl FnMut(&str) -> ControlFlow<()>,
) -> Result<Report> {
    let devices = requested_devices(&request)?;
    run_on_devices(paths, request, &devices, progress)
}

fn requested_devices(request: &Request) -> Result<Vec<Device>> {
    if request.gpus.is_empty() {
        return Ok(vec![weights::device(&request.device)?]);
    }
    let mut indices = BTreeSet::new();
    let devices = request
        .gpus
        .iter()
        .map(|&index| {
            ensure!(
                indices.insert(index),
                "CUDA device {index} was selected more than once"
            );
            weights::device(&format!("cuda:{index}"))
        })
        .collect::<Result<Vec<_>>>()?;
    if !matches!(request.device.as_str(), "auto" | "cuda") {
        ensure!(
            weights::device(&request.device)? == devices[0],
            "training device and GPU list disagree"
        );
    }
    Ok(devices)
}

fn run_on_devices(
    paths: &Paths,
    request: Request,
    devices: &[Device],
    mut progress: impl FnMut(&str) -> ControlFlow<()>,
) -> Result<Report> {
    ensure!(
        (1..=100_000).contains(&request.epochs),
        "epoch count must be between 1 and 100000"
    );
    ensure!(request.save_every != 0, "save frequency must be positive");
    let version = request.version;
    let experiment = paths.root.join(&request.experiment);
    let name = request
        .name
        .as_deref()
        .or_else(|| experiment.file_name().and_then(|name| name.to_str()))
        .context("training experiment requires a model name")?;
    let mut components = Path::new(name).components();
    ensure!(
        matches!(components.next(), Some(Component::Normal(_))) && components.next().is_none(),
        "model name must be a single filename stem"
    );
    ensure!(
        experiment.is_dir(),
        "preprocess the training dataset before training"
    );
    let file_lock = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(experiment.join(".training.lock"))?;
    file_lock
        .try_lock()
        .context("training is already running for this experiment")?;
    recover_pending_pair(&experiment).context("recovering interrupted training checkpoint save")?;
    let prepared = train_data::prepare(&PrepareRequest {
        project_root: &paths.root,
        experiment_dir: &experiment,
        mute_dir: &paths.experiments.join("mute"),
        sample_rate: request.sample_rate,
        version,
        uses_f0: request.f0,
        rebuild_filelist: request.rebuild_filelist,
        speakers: if request.multi {
            SpeakerMode::Multi
        } else {
            SpeakerMode::Single(request.speaker_id)
        },
    })?;
    let config = prepared.config;
    let dataset = prepared.dataset;
    let train = &config.train;
    let batch_size = request.batch_size.unwrap_or(train.batch_size);
    ensure!(
        (1..=256).contains(&batch_size),
        "batch size must be between 1 and 256"
    );
    ensure!(
        train.log_interval != 0 && train.segment_size.rem_euclid(config.data.hop_length) == 0,
        "invalid training log interval or segment size"
    );
    let device = devices[0];
    let mut automatic_fp16 = true;
    let mut all_cuda = true;
    for &rank_device in devices {
        if let Device::Cuda(index) = rank_device {
            let profile = weights::cuda_profile(index);
            if let Some(profile) = profile {
                ensure!(
                    profile.eligible(),
                    "CUDA device {index} does not meet the training memory or capability requirements"
                );
            }
            automatic_fp16 &= profile.is_some_and(|profile| profile.fp16);
        } else {
            all_cuda = false;
        }
    }
    ensure!(
        all_cuda || request.fp16 != Some(true),
        "FP16 training requires CUDA devices"
    );
    let amp = all_cuda && request.fp16.unwrap_or(automatic_fp16);
    let mel_config = MelConfig {
        fft: config.data.filter_length,
        hop: config.data.hop_length,
        window: config.data.win_length,
        sample_rate: request.sample_rate,
        mel_bins: config.data.n_mel_channels,
        minimum: config.data.mel_fmin,
        maximum: config.data.mel_fmax,
    };
    let mel = MelSpectrogram::new(mel_config, device)?;

    tch::manual_seed(train.seed);
    let pair = newest_pair(&experiment)?;
    ensure!(
        !(request.fresh && pair.is_some()),
        "--fresh requires an experiment without existing model checkpoints"
    );
    let mut resume = pair
        .map(|(_, g, d)| -> Result<_> { Ok((Checkpoint::open(&g)?, Checkpoint::open(&d)?)) })
        .transpose()?;
    let first = if let Some((g, d)) = &resume {
        pair_epoch(g, d)?
    } else {
        1
    };
    ensure!(
        first <= request.epochs,
        "saved checkpoint epoch {first} exceeds requested final epoch {}",
        request.epochs
    );
    let precision = if amp { "FP16 mixed precision" } else { "FP32" };
    let starting = format!(
        "Training on {} input rows at {devices:?} ({precision}) starting with epoch {first}",
        dataset.sample_count()
    );
    notify(&mut progress, &starting)?;
    let mut scaler = GradScaler::new(amp);
    let mut writer = if request.tensorboard {
        Events::new(&experiment.join("eval"))?;
        Some(Events::new(&experiment)?)
    } else {
        None
    };

    let (g_weights, d_weights) = if let Some((g, d)) = &mut resume {
        (
            g.weights(Some("model"), device)?,
            d.weights(Some("model"), device)?,
        )
    } else {
        (
            load_pretrained(paths, &request, "G", device)?,
            load_pretrained(paths, &request, "D", device)?,
        )
    };
    let resized_speaker_embedding = resume.is_some()
        && g_weights.get("emb_g.weight").is_some_and(|embedding| {
            embedding.size() != [config.model.spk_embed_dim, config.model.gin_channels]
        });
    let (generator, g_params) = generator(
        &config,
        version,
        request.f0,
        device,
        g_weights,
        resume.is_some(),
    )?;
    let v2 = matches!(version, Version::V2);
    let spectral = config.model.use_spectral_norm;
    let mut discriminator = if d_weights.is_empty() {
        Discriminator::new(v2, spectral, device)?
    } else {
        Discriminator::from_weights(v2, spectral, device, d_weights)?
    };
    let replicas = devices.len();
    let mut peers = Vec::with_capacity(replicas - 1);
    if replicas > 1 {
        let state = discriminator.state_dict();
        let initial = parallel::Initialize {
            config: &config,
            version,
            f0: request.f0,
            mel: mel_config,
            generator: &g_params,
            discriminator: &state,
        };
        for &device in &devices[1..] {
            peers.push(parallel::Replica::new(device, &initial)?);
        }
    }
    let mut g = AdamW::new(
        g_params
            .iter()
            .map(|(name, tensor)| (name.clone(), tensor.shallow_clone()))
            .collect(),
        train.learning_rate,
        train.betas,
        train.eps,
    )?;
    let mut d = AdamW::new(
        discriminator.parameters(),
        train.learning_rate,
        train.betas,
        train.eps,
    )?;
    if let Some((mut g_source, mut d_source)) = resume.take() {
        if !resized_speaker_embedding {
            g.restore(g_source.adamw_state()?)?;
        }
        d.restore(d_source.adamw_state()?)?;
    }
    // A fresh optimizer has no scheduler state. Start its schedule at the base rate.
    let generator_start = if resized_speaker_embedding { 1 } else { first };
    let g_lr = ExponentialLr::new(&mut g, train.lr_decay, generator_start)?;
    let d_lr = ExponentialLr::new(&mut d, train.lr_decay, first)?;
    let steps_per_epoch = dataset
        .batches(batch_size, first as u64, replicas, 0)?
        .len();
    let mut log = training_log(&experiment, &request)?;
    writeln!(log, "{starting}").context("writing train.log")?;
    if resized_speaker_embedding {
        log_progress(
            &mut log,
            &mut progress,
            "Resetting the generator optimizer after resizing the speaker embedding",
        )?;
    }
    let mut global = (first - 1)
        .checked_mul(steps_per_epoch)
        .context("global step overflow")?;
    let step = step::Settings {
        amp,
        segment_frames: train.segment_size / config.data.hop_length,
        hop_length: config.data.hop_length,
        segment_size: train.segment_size,
        c_mel: train.c_mel,
        c_kl: train.c_kl,
    };
    let mut cached = Vec::new();
    for epoch in first..=request.epochs {
        let batches = if cached.is_empty() {
            dataset.batches(batch_size, epoch as u64, replicas, 0)?
        } else {
            Vec::new()
        };
        if request.cache_data_in_gpu && cached.is_empty() {
            for (index, indices) in batches.iter().enumerate() {
                notify(
                    &mut progress,
                    &format!(
                        "Caching batch {}/{} at {device:?}",
                        index + 1,
                        batches.len()
                    ),
                )?;
                cached.push(dataset.load_batch(indices, device, |x| mel.spectrogram(x))?);
            }
        }
        let batch_count = if cached.is_empty() {
            batches.len()
        } else {
            cached.len()
        };
        for (rank, replica) in peers.iter_mut().enumerate() {
            let batches = if replica.has_cache() {
                Vec::new()
            } else {
                dataset.batches(batch_size, epoch as u64, replicas, rank + 1)?
            };
            let count = replica.prepare_epoch(
                &dataset,
                batches,
                request.cache_data_in_gpu,
                &mut progress,
            )?;
            ensure!(
                count == batch_count,
                "training replica has a different number of batches"
            );
        }
        let mut order: Vec<_> = (0..batch_count).collect();
        if !cached.is_empty() && epoch != first {
            let seed = train.seed as u64 ^ (epoch as u64).rotate_left(17);
            order.shuffle(&mut StdRng::seed_from_u64(seed));
        }
        for (i, index) in order.into_iter().enumerate() {
            notify(
                &mut progress,
                &format!(
                    "Epoch {epoch}/{} · batch {}/{batch_count}",
                    request.epochs,
                    i + 1
                ),
            )?;
            let loaded;
            let batch = if let Some(batch) = cached.get(index) {
                batch
            } else {
                loaded = dataset.load_batch(&batches[index], device, |x| mel.spectrogram(x))?;
                &loaded
            };
            let write_event = writer.is_some() && global.is_multiple_of(train.log_interval);
            let forward = step.generate(&generator, batch)?;
            let loss_d = step.discriminate(&mut discriminator, &forward)?;
            let mut metrics = if write_event {
                discriminator_metrics(&loss_d.real_scores, &loss_d.generated_scores)?
            } else {
                Vec::new()
            };
            let (peer_forward, peer_d): (Vec<_>, Vec<_>) = peers
                .iter_mut()
                .map(|replica| replica.discriminator_phase(&dataset, index, &step, scaler.factor()))
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .unzip();
            let d_step =
                parallel::optimize(&mut d, &loss_d.total, peer_d, scaler.factor(), write_event)?;
            if !peers.is_empty() {
                let state = discriminator.state_dict();
                for replica in &mut peers {
                    replica.synchronize_discriminator(&state)?;
                }
            }

            let loss_g = step.generator_loss(&mut discriminator, forward, &mel)?;
            let peer_g = peers
                .iter_mut()
                .zip(peer_forward)
                .map(|(replica, forward)| replica.generator_phase(forward, &step, scaler.factor()))
                .collect::<Result<Vec<_>>>()?;
            let g_step =
                parallel::optimize(&mut g, &loss_g.total, peer_g, scaler.factor(), write_event)?;
            scaler.update(d_step.overflow || g_step.overflow);
            if !peers.is_empty() {
                let state = spectral.then(|| discriminator.state_dict());
                for replica in &mut peers {
                    replica.synchronize_generator(&g_params)?;
                    if let Some(state) = &state {
                        replica.synchronize_discriminator(state)?;
                    }
                }
            }
            if write_event && let Some(writer) = writer.as_mut() {
                append_generator_metrics(&mut metrics, &loss_g.scores)?;
                metrics.extend([
                    ("loss/g/total".into(), loss_g.total.f_double_value(&[])?),
                    ("loss/d/total".into(), loss_d.total.f_double_value(&[])?),
                    ("learning_rate".into(), g.learning_rate()),
                    ("grad_norm_d".into(), d_step.norm),
                    ("grad_norm_g".into(), g_step.norm),
                    ("loss/g/fm".into(), loss_g.feature.f_double_value(&[])?),
                    (
                        "loss/g/mel".into(),
                        clamp_metric(loss_g.mel.f_double_value(&[])?, 75.0),
                    ),
                    (
                        "loss/g/kl".into(),
                        clamp_metric(loss_g.kl.f_double_value(&[])?, 9.0),
                    ),
                ]);
                writer.summary(
                    global,
                    &metrics,
                    &[
                        ("slice/mel_org", loss_g.real_mel.f_select(0, 0)?),
                        ("slice/mel_gen", loss_g.generated_mel.f_select(0, 0)?),
                        ("all/mel", loss_g.full_mel.f_select(0, 0)?),
                    ],
                )?;
            }
            if global.is_multiple_of(train.log_interval) || i + 1 == batch_count {
                let display = |tensor: &Tensor| tensor.double_value(&[]);
                log_progress(
                    &mut log,
                    &mut progress,
                    &format!(
                        "Epoch {epoch} · step {} · D {:.4} · G {:.4} · mel {:.4} · KL {:.4} · lr {:.6}",
                        global + 1,
                        display(&loss_d.total),
                        display(&loss_g.total),
                        display(&loss_g.mel),
                        display(&loss_g.kl),
                        g.learning_rate()
                    ),
                )?;
            }
            global += 1;
        }
        if epoch.is_multiple_of(request.save_every) || epoch == request.epochs {
            let suffix = if request.latest_only {
                2_333_333
            } else {
                global
            };
            save_pair(
                &experiment,
                suffix,
                epoch,
                global,
                (
                    &g,
                    g_params
                        .iter()
                        .map(|(key, value)| (key.clone(), value.shallow_clone()))
                        .collect(),
                ),
                (&d, discriminator.state_dict()),
                train.learning_rate,
            )?;
            if request.save_weights {
                let path = paths.weights.join(format!("{name}_e{epoch}_s{global}.pth"));
                edit::compact(
                    &path,
                    &config,
                    g_params.iter().map(|(k, v)| (k.clone(), v.shallow_clone())),
                    Export {
                        version,
                        sample_rate: request.sample_rate,
                        uses_f0: request.f0,
                        description: &format!("{epoch}epoch"),
                        overwrite: true,
                    },
                )?;
            }
        }
        g_lr.step(&mut g)?;
        d_lr.step(&mut d)?;
    }
    let model = paths.weights.join(format!("{name}.pth"));
    edit::compact(
        &model,
        &config,
        g_params,
        Export {
            version,
            sample_rate: request.sample_rate,
            uses_f0: request.f0,
            description: &format!("{}epoch", request.epochs),
            overwrite: true,
        },
    )?;
    writeln!(log, "Saved model {}", model.display()).context("writing train.log")?;
    drop(file_lock);
    Ok(Report {
        model,
        epochs_completed: request.epochs,
        steps: global,
    })
}

fn clamp_metric(value: f64, max: f64) -> f64 {
    if value > max { max } else { value }
}

fn discriminator_metrics(real: &[Tensor], generated: &[Tensor]) -> Result<Vec<(String, f64)>> {
    tch::no_grad(|| {
        let mut values = Vec::new();
        for (i, (real, generated)) in real.iter().zip(generated).enumerate() {
            let real = real
                .f_to_kind(Kind::Float)?
                .f_neg()?
                .f_add_scalar(1.0)?
                .f_square()?
                .f_mean(Kind::Float)?;
            let generated = generated
                .f_to_kind(Kind::Float)?
                .f_square()?
                .f_mean(Kind::Float)?;
            values.push((format!("loss/d_r/{i}"), real.f_double_value(&[])?));
            values.push((format!("loss/d_g/{i}"), generated.f_double_value(&[])?));
        }
        Ok(values)
    })
}

fn append_generator_metrics(metrics: &mut Vec<(String, f64)>, generated: &[Tensor]) -> Result<()> {
    tch::no_grad(|| {
        for (i, score) in generated.iter().enumerate() {
            let loss = train_losses::generator_loss(std::slice::from_ref(score))?;
            metrics.push((format!("loss/g/{i}"), loss.f_double_value(&[])?));
        }
        Ok(())
    })
}

fn notify(progress: &mut impl FnMut(&str) -> ControlFlow<()>, message: &str) -> Result<()> {
    ensure!(progress(message).is_continue(), "training cancelled");
    Ok(())
}

fn training_log(experiment: &Path, request: &Request) -> Result<File> {
    let path = experiment.join("train.log");
    let mut log = OpenOptions::new()
        .read(true)
        .append(true)
        .create(true)
        .open(&path)
        .with_context(|| format!("opening {}", path.display()))?;
    if log.metadata()?.len() > 0 {
        log.seek(SeekFrom::End(-1))?;
        let mut last = [0];
        log.read_exact(&mut last)?;
        if last[0] != b'\n' {
            writeln!(log)?;
        }
    }
    let metadata = json!({
        "sample_rate": format!("{}k", request.sample_rate / 1000),
        "if_f0": u8::from(request.f0),
        "version": request.version.name(),
    });
    writeln!(log, "{metadata}").context("writing train.log metadata")?;
    Ok(log)
}

fn log_progress(
    log: &mut File,
    progress: &mut impl FnMut(&str) -> ControlFlow<()>,
    message: &str,
) -> Result<()> {
    writeln!(log, "{message}").context("writing train.log")?;
    notify(progress, message)
}

fn epoch(checkpoint: &Checkpoint) -> Result<usize> {
    let value = checkpoint
        .metadata("iteration")?
        .and_then(|value| value.as_u64())
        .context("checkpoint has no epoch")?;
    ensure!(value > 0, "checkpoint epoch must be positive");
    Ok(value.try_into()?)
}

fn pair_epoch(g: &Checkpoint, d: &Checkpoint) -> Result<usize> {
    let g_epoch = epoch(g)?;
    ensure!(
        g_epoch == epoch(d)?,
        "generator and discriminator checkpoints are from different epochs"
    );
    ensure!(
        g.metadata("rvc_global_step")? == d.metadata("rvc_global_step")?,
        "generator and discriminator checkpoints are from different steps"
    );
    Ok(g_epoch)
}

fn newest_pair(path: &Path) -> Result<Option<(usize, PathBuf, PathBuf)>> {
    let mut models = BTreeMap::new();
    for file in fs::read_dir(path)? {
        let file = file?;
        let filename = file.file_name();
        let Some(filename) = filename.to_str() else {
            continue;
        };
        let Some((kind, suffix)) = filename
            .strip_suffix(".pth")
            .and_then(|s| s.split_once('_'))
        else {
            continue;
        };
        if let Ok(step) = suffix.parse::<usize>() {
            let entry = models.entry(step).or_insert([None, None]);
            match kind {
                "G" => entry[0] = Some(file.path()),
                "D" => entry[1] = Some(file.path()),
                _ => {}
            }
        }
    }
    let mut found = models.into_iter().rev();
    Ok(found.find_map(|(step, [g, d])| Some((step, g?, d?))))
}

fn load_pretrained(
    paths: &Paths,
    request: &Request,
    kind: &str,
    device: Device,
) -> Result<HashMap<String, Tensor>> {
    let custom = if kind == "G" {
        &request.pretrained_g
    } else {
        &request.pretrained_d
    };
    if request.random_weights && custom.is_none() {
        return Ok(HashMap::new());
    }
    let rate = request.sample_rate / 1000;
    let prefix = if request.f0 { "f0" } else { "" };
    let directory = if matches!(request.version, Version::V2) {
        "pretrained_v2"
    } else {
        "pretrained"
    };
    let model = custom
        .as_ref()
        .map(|path| paths.root.join(path))
        .unwrap_or_else(|| {
            paths
                .root
                .join("assets")
                .join(directory)
                .join(format!("{prefix}{kind}{rate}k.pth"))
        });
    if !model.is_file() && custom.is_none() {
        return Ok(HashMap::new());
    }
    let mut checkpoint = Checkpoint::open(&model)?;
    let section = if checkpoint.has("model") {
        Some("model")
    } else {
        Some("weight")
    };
    checkpoint.weights(section, device)
}

type Save<'a> = (&'a AdamW, Vec<(String, Tensor)>);

const PENDING_PAIR: &str = ".checkpoint-pending";

fn checkpoint_names(suffix: usize) -> [String; 2] {
    [format!("G_{suffix}.pth"), format!("D_{suffix}.pth")]
}

fn save_pair(
    experiment: &Path,
    suffix: usize,
    epoch: usize,
    step: usize,
    g: Save<'_>,
    d: Save<'_>,
    initial_rate: f64,
) -> Result<()> {
    let staging = tempfile::TempDir::new_in(experiment)?;
    let names = checkpoint_names(suffix);
    for ((optimizer, tensors), filename) in [g, d].into_iter().zip(&names) {
        let data = Value::Dict(vec![
            ("model".into(), Value::tensors(tensors)),
            ("iteration".into(), json!(epoch).into()),
            ("optimizer".into(), optimizer.state_dict()?),
            ("learning_rate".into(), json!(initial_rate).into()),
            ("rvc_global_step".into(), json!(step).into()),
        ]);
        write::write(&staging.path().join(filename), &data, false)?;
    }

    // Leave both complete archives available to the next run before replacing either name.
    let pending = experiment.join(PENDING_PAIR);
    fs::rename(staging.path(), &pending).context("staging training checkpoint pair")?;
    let _ = staging.keep();
    finish_pending_pair(experiment, &pending, names.iter().map(String::as_str))
}

fn recover_pending_pair(experiment: &Path) -> Result<()> {
    let pending = experiment.join(PENDING_PAIR);
    let metadata = match fs::symlink_metadata(&pending) {
        Ok(metadata) => metadata,
        Err(err) if err.kind() == ErrorKind::NotFound => return Ok(()),
        Err(err) => return Err(err.into()),
    };
    ensure!(
        metadata.is_dir(),
        "pending checkpoint pair is not a directory"
    );

    let mut suffix = None;
    let mut staged = [None, None];
    for entry in fs::read_dir(&pending)? {
        let entry = entry?;
        let filename = entry.file_name();
        let filename = filename
            .to_str()
            .context("invalid pending checkpoint filename")?;
        let (kind, step) = filename
            .strip_suffix(".pth")
            .and_then(|name| name.split_once('_'))
            .context("invalid pending checkpoint filename")?;
        let index = match kind {
            "G" => 0,
            "D" => 1,
            _ => anyhow::bail!("unexpected pending checkpoint {filename}"),
        };
        let step: usize = step.parse().context("invalid pending checkpoint suffix")?;
        ensure!(
            entry.file_type()?.is_file() && filename == checkpoint_names(step)[index],
            "invalid pending checkpoint {filename}"
        );
        ensure!(
            suffix.is_none_or(|suffix| suffix == step),
            "pending checkpoint pair has different suffixes"
        );
        suffix = Some(step);
        staged[index] = Some(filename.to_owned());
    }

    if let Some(suffix) = suffix {
        let names = checkpoint_names(suffix);
        let [g, d] = std::array::from_fn(|i| match &staged[i] {
            Some(filename) => pending.join(filename),
            None => experiment.join(&names[i]),
        });
        pair_epoch(&Checkpoint::open(&g)?, &Checkpoint::open(&d)?)?;
    }
    finish_pending_pair(
        experiment,
        &pending,
        staged.iter().flatten().map(String::as_str),
    )
}

fn finish_pending_pair<'a>(
    experiment: &Path,
    pending: &Path,
    filenames: impl IntoIterator<Item = &'a str>,
) -> Result<()> {
    for filename in filenames {
        fs::rename(pending.join(filename), experiment.join(filename))
            .with_context(|| format!("publishing training checkpoint {filename}"))?;
    }
    fs::remove_dir(pending).context("removing completed checkpoint staging directory")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recovers_training_pair_after_second_replacement_fails() -> Result<()> {
        let experiment = tempfile::tempdir()?;
        let experiment = experiment.path();
        let suffix = 2_333_333;
        let [g_name, d_name] = checkpoint_names(suffix);
        let g = experiment.join(&g_name);
        let d = experiment.join(&d_name);
        let pending = experiment.join(PENDING_PAIR);
        let weight = Tensor::f_from_slice(&[0.2f32])?.f_set_requires_grad(true)?;
        let parameters = || vec![("weight".into(), weight.shallow_clone())];
        let optimizer = AdamW::new(parameters(), 0.02, [0.8, 0.99], 1e-9)?;
        let save = |epoch, step| {
            save_pair(
                experiment,
                suffix,
                epoch,
                step,
                (&optimizer, parameters()),
                (&optimizer, parameters()),
                0.02,
            )
        };

        save(1, 2)?;
        assert!(!pending.exists());
        let old_d = experiment.join("previous-discriminator.pth");
        fs::rename(&d, &old_d)?;
        fs::create_dir(&d)?;
        let error = save(2, 4).unwrap_err();
        assert!(error.to_string().contains(&d_name));
        fs::remove_dir(&d)?;
        fs::rename(&old_d, &d)?;
        assert_eq!(epoch(&Checkpoint::open(&g)?)?, 2);
        assert_eq!(epoch(&Checkpoint::open(&d)?)?, 1);

        let staged_d = pending.join(&d_name);
        let new_d = experiment.join("next-discriminator.pth");
        fs::rename(&staged_d, &new_d)?;
        fs::copy(&d, &staged_d)?;
        let error = recover_pending_pair(experiment).unwrap_err();
        assert!(error.to_string().contains("from different epochs"));
        assert_eq!(epoch(&Checkpoint::open(&d)?)?, 1);
        fs::rename(&new_d, &staged_d)?;

        recover_pending_pair(experiment)?;
        assert_eq!(
            pair_epoch(&Checkpoint::open(&g)?, &Checkpoint::open(&d)?)?,
            2
        );
        assert!(!pending.exists());
        recover_pending_pair(experiment)?;
        Ok(())
    }
}
