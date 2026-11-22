pub(super) mod activation;
mod ema;
pub(crate) mod events;
pub(super) mod paired;
mod parallel;
mod precision;
mod step;

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs::{self, File, OpenOptions};
use std::io::{ErrorKind, Read, Seek, SeekFrom, Write};
use std::ops::ControlFlow;
use std::path::{Component, Path, PathBuf};
use std::time::Instant;

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
use super::train_losses::{
    self, MelConfig, MelSpectrogram, MultiResolutionStft, MultiScaleMel, PhoneticLoss,
};
use super::weights;
use crate::catalog::{ArticulationProfile, Embedder, Paths, Unvoiced};
use crate::checkpoint::{
    Checkpoint,
    write::{self, Value},
};
use events::Events;
use precision::{GradScaler, Precision};

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
    /// Use BF16 autocast and compact optimizer state on CUDA.
    #[arg(long, num_args = 0..=1, default_missing_value = "true", require_equals = true)]
    pub bf16: Option<bool>,
    /// Recompute generator decoder blocks during backward to save memory.
    #[arg(long, num_args = 0..=1, default_missing_value = "true", require_equals = true)]
    pub gradient_checkpointing: Option<bool>,
    /// Embedder used when extracting this experiment's content features.
    #[arg(long, value_enum)]
    pub embedder: Option<Embedder>,
    /// Generate and blend NVC-TS variants from an NVC feature pool.
    #[arg(long)]
    pub timbre_shift_pool: Option<PathBuf>,
    /// Pool speakers to exclude from NVC-TS substitution.
    #[arg(long, value_delimiter = ',')]
    pub timbre_shift_exclude: Vec<String>,
    #[arg(long)]
    #[serde(alias = "if_cache_data_in_gpu")]
    pub cache_data_in_gpu: bool,
    #[arg(long)]
    pub multiscale_mel: bool,
    /// Export an exponential moving average of generator weights with this decay
    /// (0 disables it, 0.999 averages roughly the last thousand steps).
    #[arg(long, default_value_t = 0.0)]
    pub ema_decay: f64,
    /// Add multi-resolution spectrogram discriminators (MRD) to MPD/MSD.
    #[arg(long)]
    pub mrd: bool,
    /// Weight of the multi-resolution STFT loss (0 disables it).
    #[arg(long, default_value_t = 0.0)]
    pub stft_loss: f64,
    /// Weight of the phonetic loss: L1 between frozen speech-encoder features of real
    /// and generated audio (0 disables it). Sharpens articulation.
    #[arg(long, default_value_t = 0.0)]
    pub phonetic_loss: f64,
    /// Train only generator parameters whose names start with these prefixes (for example
    /// `enc_p` to fit a new content encoder while the decoder, flow and posterior stay frozen).
    #[arg(long, value_delimiter = ',')]
    pub train_only: Vec<String>,
    /// Encoder for the optional phonetic loss. Defaults to the experiment's content embedder.
    #[arg(long, value_enum)]
    pub phonetic_encoder: Option<Embedder>,
    /// F0 on unvoiced frames: `interpolate` (RVC) or `zero` (NVC). Requires raw F0 from extraction.
    #[arg(long, value_enum)]
    pub unvoiced_f0: Option<Unvoiced>,
    /// Condition the content encoder on the source's 2-16 kHz band shape, so sibilants keep
    /// their color. Turned on automatically when the pretrained generator has these weights.
    #[arg(long)]
    pub hf_bands: bool,
    /// Run decoder activations anti-aliased (BigVGAN-style), against metallic artifacts.
    /// Turned on automatically when the pretrained generator was trained this way.
    #[arg(long)]
    pub dec_aa: bool,
    /// Layered copy of this experiment (from `layer-aug` + `extract`): inputs are read from it
    /// for a `--layer-aug-prob` share of examples while the target stays clean, so stacked
    /// doubles, harmonies and reverb in a song do not turn into artifacts.
    #[arg(long)]
    pub layer_aug: Option<PathBuf>,
    #[arg(long, default_value_t = 0.5)]
    pub layer_aug_prob: f64,
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
            bf16: None,
            gradient_checkpointing: None,
            embedder: None,
            timbre_shift_pool: None,
            timbre_shift_exclude: Vec::new(),
            cache_data_in_gpu: false,
            multiscale_mel: false,
            ema_decay: 0.0,
            mrd: false,
            stft_loss: 0.0,
            phonetic_loss: 0.0,
            train_only: Vec::new(),
            phonetic_encoder: None,
            unvoiced_f0: None,
            hf_bands: false,
            dec_aa: false,
            layer_aug: None,
            layer_aug_prob: 0.5,
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
    let mut config = prepared.config;
    let mut dataset = prepared.dataset;
    let extracted = super::extract::feature_embedder(&experiment, version)?;
    let embedder = request
        .embedder
        .or(extracted)
        .or(config.embedder_model)
        .unwrap_or_default();
    embedder.supports(version.name())?;
    ensure!(
        extracted
            .or(config.embedder_model)
            .is_none_or(|source| source == embedder),
        "cached features use {}. Extract features with {} before training.",
        extracted
            .or(config.embedder_model)
            .unwrap_or_default()
            .name(),
        embedder.name()
    );
    let pair = newest_pair(&experiment)?;
    ensure!(
        !(request.fresh && pair.is_some()),
        "--fresh requires an experiment without existing model checkpoints"
    );
    if pair.is_some() {
        ensure!(
            config.embedder_model.unwrap_or_default() == embedder,
            "cannot resume with another embedder. Use the original features or a new experiment."
        );
    }
    config.embedder_model = Some(embedder);
    ensure!(
        request.stft_loss.is_finite() && request.stft_loss >= 0.0,
        "STFT loss weight must be a non-negative number"
    );
    ensure!(
        request.phonetic_loss.is_finite() && request.phonetic_loss >= 0.0,
        "phonetic loss weight must be a non-negative number"
    );
    if let Some(pool) = &request.timbre_shift_pool {
        config.data.content_aug = if pool.as_os_str().is_empty() {
            config.data.f0_unvoiced = Unvoiced::Interpolate;
            None
        } else {
            let mut options = config
                .data
                .content_aug
                .take()
                .filter(|options| options.pool.as_ref() == Some(pool))
                .unwrap_or_else(|| super::timbre::Options::knn(pool.clone(), Vec::new()));
            options.exclude.clone_from(&request.timbre_shift_exclude);
            Some(options)
        };
    } else if !request.timbre_shift_exclude.is_empty() {
        let options = config
            .data
            .content_aug
            .as_mut()
            .context("select a feature pool to exclude speakers")?;
        options.exclude.clone_from(&request.timbre_shift_exclude);
    }
    if let Some(unvoiced) = request.unvoiced_f0 {
        ensure!(
            pair.is_none() || config.data.f0_unvoiced == unvoiced,
            "cannot change the unvoiced F0 mode when resuming"
        );
        config.data.f0_unvoiced = unvoiced;
    } else if pair.is_none() && request.f0 && dataset.has_raw_pitch(&paths.experiments.join("mute"))
    {
        // New experiments keep F0 at zero on unvoiced frames, as the extracted
        // pitch says, instead of interpolating through consonants and breaths.
        // This lowered voiced/unvoiced errors by about a third in our tests.
        config.data.f0_unvoiced = Unvoiced::Zero;
    }
    let device = devices[0];
    let bf16 = request.bf16.unwrap_or(config.train.bf16);
    let checkpointing = request
        .gradient_checkpointing
        .unwrap_or(config.train.gradient_checkpointing);
    let precision = Precision::choose(devices, request.fp16, bf16)?;
    config.train.bf16 = bf16;
    config.train.gradient_checkpointing = checkpointing;
    let mut resume = pair
        .map(|(_, g, d)| -> Result<_> { Ok((Checkpoint::open(&g)?, Checkpoint::open(&d)?)) })
        .transpose()?;
    let (first, saved_step) = resume
        .as_ref()
        .map_or(Ok((1, None)), |(g, d)| resume_position(g, d))?;
    ensure!(
        first <= request.epochs,
        "next training epoch {first} exceeds requested final epoch {}",
        request.epochs
    );
    let batch_size = request.batch_size.unwrap_or(config.train.batch_size);
    ensure!(
        (1..=256).contains(&batch_size),
        "batch size must be between 1 and 256"
    );
    ensure!(
        config.train.log_interval != 0
            && config.train.segment_size.rem_euclid(config.data.hop_length) == 0,
        "invalid training log interval or segment size"
    );
    ensure!(
        !request.multiscale_mel || config.train.segment_size > 1024,
        "multi-scale mel requires training segments longer than 1024 samples"
    );
    dataset.configure_timbre(paths, &experiment, &mut config, device, &mut progress)?;
    if let Some((g, _)) = &resume {
        config.model.paired_content = paired::version(g)?;
        if let Some(profile) = ArticulationProfile::from_generator(g)? {
            config.model.articulation = Some(profile);
        }
    } else {
        let architecture = pretrained_architecture(paths, &request)?;
        config.model.hf_bands = request.hf_bands || architecture.bands;
        config.model.dec_aa = request.dec_aa || architecture.anti_alias;
        config.model.articulation = architecture.articulation;
        config.model.paired_content = architecture.paired_content;
    }
    if resume.is_none()
        && let Some(profile) = &mut config.model.articulation
        && let Some(frication) = &mut profile.frication
    {
        notify(&mut progress, "Measuring breath and fricative spectra in each training speaker")?;
        frication.targets = super::articulation::frication::fit(
            dataset.frication_sources(), request.sample_rate, &profile.detector,
        )?;
    }
    if let Some(directory) = &request.layer_aug {
        dataset.configure_layers(paths.root.join(directory), request.layer_aug_prob)?;
    } else if config.model.paired_content != 0 {
        let pairs = paired::prepare(
            paths,
            &experiment,
            config.model.paired_content,
            dataset.paired_sources(),
            version,
            embedder,
            request.sample_rate,
            device,
            &mut progress,
        )?;
        dataset.set_paired_sources(pairs, config.model.hf_bands);
    }
    config.save(&experiment)?;
    let train = &config.train;
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
    let multiscale = request
        .multiscale_mel
        .then(|| MultiScaleMel::new(request.sample_rate, device))
        .transpose()?;
    let phonetic_encoder = (request.phonetic_loss > 0.0)
        .then_some(request.phonetic_encoder.unwrap_or(embedder));
    let phonetic_directory = phonetic_encoder
        .map(|encoder| encoder.install(paths, &mut progress))
        .transpose()?;
    let auxiliary = step::Auxiliary {
        stft: (request.stft_loss > 0.0)
            .then(|| MultiResolutionStft::new(device))
            .transpose()?,
        phonetic: phonetic_directory
            .as_deref()
            .map(|directory| {
                PhoneticLoss::new(
                    directory,
                    i64::from(request.sample_rate),
                    device,
                    precision.kind(),
                )
            })
            .transpose()?,
    };

    tch::manual_seed(train.seed);
    // Training shapes repeat (bucketed clip lengths, fixed decoder segments), so cuDNN
    // autotuning pays off after the first few steps. `RVC_CUDNN_BENCHMARK=0` disables it.
    if devices
        .iter()
        .all(|device| matches!(device, Device::Cuda(_)))
        && std::env::var("RVC_CUDNN_BENCHMARK").map_or(true, |value| value != "0")
    {
        tch::Cuda::cudnn_set_benchmark(true);
    }
    let precision_name = precision.name();
    let starting = format!(
        "Training on {} input rows at {devices:?} ({precision_name}, gradient checkpointing {checkpointing}, {}, NVC-TS {}) starting with epoch {first}",
        dataset.sample_count(),
        embedder.name(),
        config
            .data
            .content_aug
            .as_ref()
            .is_some_and(|options| options.prob > 0.0)
    );
    notify(&mut progress, &starting)?;
    let mut scaler = GradScaler::new(precision == Precision::Fp16);
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
    ensure!(
        request.ema_decay.is_finite() && (0.0..1.0).contains(&request.ema_decay),
        "EMA decay must be in [0, 1)"
    );
    // Resumed runs start the average from the restored weights.
    let mut ema = (request.ema_decay > 0.0)
        .then(|| ema::Ema::new(&g_params, request.ema_decay))
        .transpose()?;
    let v2 = matches!(version, Version::V2);
    let spectral = config.model.use_spectral_norm;
    let mut discriminator = if d_weights.is_empty() {
        Discriminator::new(v2, spectral, request.mrd, device)?
    } else {
        Discriminator::from_weights(v2, spectral, request.mrd, device, d_weights)?
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
            multiscale: request.multiscale_mel,
            stft: request.stft_loss > 0.0,
            phonetic: phonetic_directory.as_deref().map(|directory| {
                (directory, i64::from(request.sample_rate), precision.kind())
            }),
            resolution: discriminator.has_resolution_branches(),
            generator: &g_params,
            discriminator: &state,
        };
        for &device in &devices[1..] {
            peers.push(parallel::Replica::new(device, &initial)?);
        }
    }
    let trainable = |name: &str| {
        request.train_only.is_empty()
            || request
                .train_only
                .iter()
                .any(|prefix| name.starts_with(prefix.as_str()))
    };
    let mut frozen_note = None;
    if !request.train_only.is_empty() {
        ensure!(
            replicas == 1,
            "--train-only supports a single training device"
        );
        let mut count = 0;
        for (name, tensor) in &g_params {
            if trainable(name) {
                count += tensor.numel();
            } else {
                let _ = tensor.shallow_clone().set_requires_grad(false);
            }
        }
        ensure!(count > 0, "--train-only matched no generator parameters");
        frozen_note = Some(format!(
            "Training {count} generator parameters; the rest stay frozen"
        ));
    }
    let mut g = AdamW::new(
        g_params
            .iter()
            .filter(|(name, _)| trainable(name))
            .map(|(name, tensor)| (name.clone(), tensor.shallow_clone()))
            .collect(),
        train.learning_rate,
        train.betas,
        train.eps,
    )?
    .with_bf16(bf16);
    let mut d = AdamW::new(
        discriminator.parameters(),
        train.learning_rate,
        train.betas,
        train.eps,
    )?
    .with_bf16(bf16);
    if let Some((mut g_source, mut d_source)) = resume.take() {
        if !resized_speaker_embedding && request.train_only.is_empty() {
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
    let mut log = training_log(&experiment, &request, phonetic_encoder)?;
    writeln!(log, "{starting}").context("writing train.log")?;
    if let Some(note) = &frozen_note {
        log_progress(&mut log, &mut progress, note)?;
    }
    if resized_speaker_embedding {
        log_progress(
            &mut log,
            &mut progress,
            "Resetting the generator optimizer after resizing the speaker embedding",
        )?;
    }
    let mut global = match saved_step {
        Some(step) => step,
        None => (first - 1)
            .checked_mul(steps_per_epoch)
            .context("global step overflow")?,
    };
    let step = step::Settings {
        precision,
        checkpointing,
        segment_frames: train.segment_size / config.data.hop_length,
        hop_length: config.data.hop_length,
        segment_size: train.segment_size,
        c_mel: train.c_mel,
        c_kl: train.c_kl,
        c_stft: request.stft_loss,
        c_phonetic: request.phonetic_loss,
    };
    // A GPU batch contains one specific draw of the input features. Reusing that
    // batch every epoch would freeze the clean/alternate choice for each clip.
    let dynamic_pairs = config.model.paired_content != 0 && request.layer_aug.is_none();
    let cache_batches = request.cache_data_in_gpu && !dynamic_pairs;
    if request.cache_data_in_gpu && dynamic_pairs {
        notify(&mut progress, "Rebuilding paired inputs each epoch to resample clean and augmented features")?;
    }
    let mut cached = Vec::new();
    let mut profile = Profile::new(device);
    for epoch in first..=request.epochs {
        let epoch_started = Instant::now();
        let batches = if cached.is_empty() {
            dataset.batches(batch_size, epoch as u64, replicas, 0)?
        } else {
            Vec::new()
        };
        if cache_batches && cached.is_empty() {
            for (index, indices) in batches.iter().enumerate() {
                notify(
                    &mut progress,
                    &format!(
                        "Caching batch {}/{} at {device:?}",
                        index + 1,
                        batches.len()
                    ),
                )?;
                cached.push(dataset.load_batch_seeded(
                    indices,
                    device,
                    batch_seed(train.seed, epoch, index, 0),
                    |x| mel.spectrogram(x),
                )?);
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
                cache_batches,
                |index| batch_seed(train.seed, epoch, index, rank + 1),
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
        // Host-side loading of the next batches overlaps GPU work unless the
        // dataset is cached on the device or `RVC_PREFETCH=0`.
        let prefetch = cached.is_empty() && prefetch_enabled();
        let seed = train.seed;
        std::thread::scope(|scope| -> Result<()> {
            let (sender, receiver) = std::sync::mpsc::sync_channel(PREFETCH_BATCHES);
            if prefetch {
                let order = order.clone();
                let (batches, dataset) = (&batches, &dataset);
                scope.spawn(move || {
                    for index in order {
                        let staged = dataset.stage_batch(
                            &batches[index],
                            batch_seed(seed, epoch, index, 0),
                            Some(device),
                        );
                        if sender.send(staged).is_err() {
                            break;
                        }
                    }
                });
            } else {
                drop(sender);
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
                profile.mark(0);
                let loaded;
                let batch = if let Some(batch) = cached.get(index) {
                    batch
                } else if prefetch {
                    let staged = receiver
                        .recv()
                        .context("training batch loader stopped unexpectedly")??;
                    loaded = dataset.finish_batch(staged, device, |x| mel.spectrogram(x))?;
                    &loaded
                } else {
                    loaded = dataset.load_batch_seeded(
                        &batches[index],
                        device,
                        batch_seed(train.seed, epoch, index, 0),
                        |x| mel.spectrogram(x),
                    )?;
                    &loaded
                };
                let write_event = writer.is_some() && global.is_multiple_of(train.log_interval);
                profile.mark(1);
                let forward = step.generate(&generator, batch)?;
                profile.mark(2);
                let loss_d = step.discriminate(&mut discriminator, &forward)?;
                let mut metrics = if write_event {
                    discriminator_metrics(&loss_d.real_scores, &loss_d.generated_scores)?
                } else {
                    Vec::new()
                };
                let (peer_forward, peer_d): (Vec<_>, Vec<_>) = peers
                    .iter_mut()
                    .enumerate()
                    .map(|(rank, replica)| {
                        replica.discriminator_phase(
                            &dataset,
                            index,
                            batch_seed(train.seed, epoch, index, rank + 1),
                            &step,
                            scaler.factor(),
                        )
                    })
                    .collect::<Result<Vec<_>>>()?
                    .into_iter()
                    .unzip();
                let d_step = parallel::optimize(
                    &mut d,
                    &loss_d.total,
                    peer_d,
                    scaler.factor(),
                    write_event,
                )?;
                if !peers.is_empty() {
                    let state = discriminator.state_dict();
                    for replica in &mut peers {
                        replica.synchronize_discriminator(&state)?;
                    }
                }

                profile.mark(3);
                let loss_g = step.generator_loss(
                    &mut discriminator,
                    forward,
                    &mel,
                    multiscale.as_ref(),
                    &auxiliary,
                )?;
                profile.mark(4);
                let peer_g = peers
                    .iter_mut()
                    .zip(peer_forward)
                    .map(|(replica, forward)| {
                        replica.generator_phase(forward, &step, scaler.factor())
                    })
                    .collect::<Result<Vec<_>>>()?;
                let g_step = parallel::optimize(
                    &mut g,
                    &loss_g.total,
                    peer_g,
                    scaler.factor(),
                    write_event,
                )?;
                profile.mark(5);
                scaler.update(d_step.overflow || g_step.overflow);
                if let Some(ema) = ema.as_mut().filter(|_| !g_step.overflow) {
                    ema.update(&g_params)?;
                }
                if let Some(report) = profile.finish_step() {
                    log_progress(&mut log, &mut progress, &report)?;
                }
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
                    if let Some(stft) = &loss_g.stft {
                        metrics.push(("loss/g/stft".into(), stft.f_double_value(&[])?));
                    }
                    if let Some(phonetic) = &loss_g.phonetic {
                        metrics.push(("loss/g/phonetic".into(), phonetic.f_double_value(&[])?));
                    }
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
                    let stft = loss_g
                        .stft
                        .as_ref()
                        .map(|stft| format!(" · STFT {:.4}", display(stft)))
                        .unwrap_or_default()
                        + &loss_g
                            .phonetic
                            .as_ref()
                            .map(|phonetic| format!(" · phonetic {:.4}", display(phonetic)))
                            .unwrap_or_default();
                    log_progress(
                        &mut log,
                        &mut progress,
                        &format!(
                            "Epoch {epoch} · step {} · D {:.4} · G {:.4} · mel {:.4}{stft} · KL {:.4} · lr {:.6}",
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
            Ok(())
        })?;
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
                config.model.dec_aa,
                config.model.articulation.as_ref(),
                config.model.paired_content,
            )?;
            if request.save_weights {
                let path = paths.weights.join(format!("{name}_e{epoch}_s{global}.pth"));
                edit::compact(
                    &path,
                    &config,
                    exported(&g_params, ema.as_ref()),
                    Export {
                        version,
                        sample_rate: request.sample_rate,
                        uses_f0: request.f0,
                        description: &format!("{epoch}epoch"),
                        overwrite: true,
                    },
                )?;
                if ema.is_some() {
                    // The raw weights of the same step, for comparison with the average.
                    let path = paths
                        .weights
                        .join(format!("{name}_e{epoch}_s{global}_raw.pth"));
                    edit::compact(
                        &path,
                        &config,
                        exported(&g_params, None),
                        Export {
                            version,
                            sample_rate: request.sample_rate,
                            uses_f0: request.f0,
                            description: &format!("{epoch}epoch raw"),
                            overwrite: true,
                        },
                    )?;
                }
            }
        }
        g_lr.step(&mut g)?;
        d_lr.step(&mut d)?;
        for device in devices {
            if let Device::Cuda(index) = device {
                tch::Cuda::synchronize(*index as i64);
            }
        }
        log_progress(
            &mut log,
            &mut progress,
            &format!(
                "Epoch {epoch}/{} completed in {:.2} s",
                request.epochs,
                epoch_started.elapsed().as_secs_f64()
            ),
        )?;
    }
    let model = paths.weights.join(format!("{name}.pth"));
    edit::compact(
        &model,
        &config,
        exported(&g_params, ema.as_ref()),
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

/// Generator weights for exported voice models: the EMA when enabled.
fn exported(parameters: &[(String, Tensor)], ema: Option<&ema::Ema>) -> Vec<(String, Tensor)> {
    ema.map_or_else(
        || {
            parameters
                .iter()
                .map(|(name, value)| (name.clone(), value.shallow_clone()))
                .collect()
        },
        ema::Ema::weights,
    )
}

const PREFETCH_BATCHES: usize = 2;

fn prefetch_enabled() -> bool {
    std::env::var("RVC_PREFETCH").map_or(true, |value| value != "0")
}

/// Optional per-phase step timing, enabled with `RVC_TRAIN_PROFILE=<steps>`.
struct Profile {
    device: Device,
    every: usize,
    last: Option<std::time::Instant>,
    totals: [f64; 6],
    steps: usize,
    phase: usize,
    started: Option<std::time::Instant>,
}

impl Profile {
    fn new(device: Device) -> Self {
        let every = std::env::var("RVC_TRAIN_PROFILE")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(0);
        Self {
            device,
            every,
            last: None,
            totals: [0.0; 6],
            steps: 0,
            phase: 0,
            started: None,
        }
    }

    fn mark(&mut self, phase: usize) {
        if self.every == 0 {
            return;
        }
        if let Device::Cuda(index) = self.device {
            tch::Cuda::synchronize(index as i64);
        }
        let now = std::time::Instant::now();
        if phase == 0 {
            if self.started.is_none() {
                self.started = Some(now);
            }
        } else if let Some(last) = self.last {
            self.totals[self.phase] += (now - last).as_secs_f64();
        }
        self.phase = phase;
        self.last = Some(now);
    }

    fn finish_step(&mut self) -> Option<String> {
        if self.every == 0 {
            return None;
        }
        self.steps += 1;
        if !self.steps.is_multiple_of(self.every) {
            return None;
        }
        let n = self.steps as f64;
        let names = [
            "load",
            "G forward",
            "D fwd+bwd+step",
            "G loss forward",
            "G bwd+step",
        ];
        let parts: Vec<_> = names
            .iter()
            .zip(&self.totals)
            .map(|(name, total)| format!("{name} {:.1} ms", 1000.0 * total / n))
            .collect();
        let wall = self
            .started
            .map(|start| start.elapsed().as_secs_f64() / n)
            .unwrap_or(0.0);
        Some(format!(
            "Profile over {} steps: {} · wall {:.1} ms/step",
            self.steps,
            parts.join(" · "),
            1000.0 * wall
        ))
    }
}

fn clamp_metric(value: f64, max: f64) -> f64 {
    if value > max { max } else { value }
}

fn batch_seed(seed: i64, epoch: usize, batch: usize, rank: usize) -> u64 {
    seed as u64 ^ (epoch as u64).rotate_left(17) ^ (batch as u64).rotate_left(41) ^ rank as u64
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

fn training_log(
    experiment: &Path,
    request: &Request,
    phonetic_encoder: Option<Embedder>,
) -> Result<File> {
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
        "speaker_id": request.speaker_id,
        "multi": request.multi,
        "batch_size": request.batch_size,
        "save_every": request.save_every,
        "latest_only": request.latest_only,
        "save_weights": request.save_weights,
        "tensorboard": request.tensorboard,
        "multiscale_mel": request.multiscale_mel,
        "stft_loss": request.stft_loss,
        "phonetic_loss": request.phonetic_loss,
        "phonetic_encoder": phonetic_encoder,
        "mrd": request.mrd,
        "ema_decay": request.ema_decay,
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

fn resume_position(g: &Checkpoint, d: &Checkpoint) -> Result<(usize, Option<usize>)> {
    let epoch = pair_epoch(g, d)?;
    let step = g
        .metadata("rvc_global_step")?
        .map(|value| -> Result<usize> {
            Ok(value
                .as_u64()
                .context("invalid saved training step")?
                .try_into()?)
        })
        .transpose()?;
    // Native and current NVC pairs are saved after the entire epoch.
    let next = if step.is_some() || g.has("training_state") {
        epoch.checked_add(1).context("checkpoint epoch overflow")?
    } else {
        epoch
    };
    Ok((next, step))
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

pub(crate) fn has_checkpoint_pair(experiment: &Path) -> Result<bool> {
    Ok(newest_pair(experiment)?.is_some())
}

#[derive(Serialize)]
pub(crate) struct ResumeInfo {
    pub epoch: usize,
    pub sample_rate: u32,
    pub version: Version,
    pub f0: bool,
    pub multi: bool,
    pub speaker_id: i64,
    pub batch_size: usize,
    pub save_every: usize,
    pub latest_only: bool,
    pub save_weights: bool,
    pub tensorboard: bool,
    pub multiscale_mel: bool,
    pub stft_loss: f64,
    pub phonetic_loss: f64,
    pub phonetic_encoder: Option<Embedder>,
    pub mrd: bool,
    pub ema_decay: f64,
    pub embedder: Embedder,
    pub bf16: bool,
    pub gradient_checkpointing: bool,
    pub timbre_shift_pool: Option<PathBuf>,
    pub generator: PathBuf,
    pub discriminator: PathBuf,
}

pub(crate) fn resume_info(experiment: &Path) -> Result<ResumeInfo> {
    let (step, generator, discriminator) = newest_pair(experiment)?
        .context("no complete generator and discriminator checkpoint pair")?;
    let g = Checkpoint::open(&generator)?;
    let d = Checkpoint::open(&discriminator)?;
    let version = match g
        .tensor_shape(Some("model"), "enc_p.emb_phone.weight")?
        .get(1)
    {
        Some(256) => Version::V1,
        Some(768) => Version::V2,
        _ => anyhow::bail!("unsupported checkpoint content feature size"),
    };
    let f0 = g
        .tensor_shape(Some("model"), "enc_p.emb_pitch.weight")
        .is_ok();
    let config: train_data::Config =
        serde_json::from_reader(File::open(experiment.join("config.json"))?)
            .context("reading experiment config")?;
    let mut info = ResumeInfo {
        epoch: pair_epoch(&g, &d)?,
        sample_rate: config.data.sampling_rate,
        version,
        f0,
        multi: !config.speaker_info.is_empty(),
        speaker_id: 0,
        batch_size: config.train.batch_size,
        save_every: 5,
        latest_only: step == 2_333_333,
        save_weights: false,
        tensorboard: true,
        multiscale_mel: false,
        stft_loss: 0.0,
        phonetic_loss: 0.0,
        phonetic_encoder: None,
        mrd: false,
        ema_decay: 0.0,
        embedder: config.embedder_model.unwrap_or_default(),
        bf16: config.train.bf16,
        gradient_checkpointing: config.train.gradient_checkpointing,
        timbre_shift_pool: config.data.content_aug.and_then(|options| options.pool),
        generator,
        discriminator,
    };
    if let Ok(log) = File::open(experiment.join("train.log")) {
        use std::io::{BufRead, BufReader};
        for line in BufReader::new(log).lines() {
            let line = line?;
            if !line.starts_with('{') {
                continue;
            }
            if let Ok(value) = serde_json::from_str::<serde_json::Value>(&line)
                && value["sample_rate"] == format!("{}k", info.sample_rate / 1000)
                && value["version"] == version.name()
                && value["if_f0"] == u8::from(f0)
            {
                info.multi = value["multi"].as_bool().unwrap_or(info.multi);
                info.speaker_id = value["speaker_id"].as_i64().unwrap_or(info.speaker_id);
                info.batch_size = value["batch_size"]
                    .as_u64()
                    .and_then(|n| usize::try_from(n).ok())
                    .unwrap_or(info.batch_size);
                info.save_every = value["save_every"]
                    .as_u64()
                    .and_then(|n| usize::try_from(n).ok())
                    .unwrap_or(info.save_every);
                info.latest_only = value["latest_only"].as_bool().unwrap_or(info.latest_only);
                info.save_weights = value["save_weights"].as_bool().unwrap_or(info.save_weights);
                info.tensorboard = value["tensorboard"].as_bool().unwrap_or(info.tensorboard);
                info.multiscale_mel = value["multiscale_mel"]
                    .as_bool()
                    .unwrap_or(info.multiscale_mel);
                info.stft_loss = value["stft_loss"].as_f64().unwrap_or(info.stft_loss);
                info.phonetic_loss = value["phonetic_loss"]
                    .as_f64()
                    .unwrap_or(info.phonetic_loss);
                info.phonetic_encoder = saved_phonetic_encoder(&value, info.phonetic_loss)?;
                info.mrd = value["mrd"].as_bool().unwrap_or(info.mrd);
                info.ema_decay = value["ema_decay"].as_f64().unwrap_or(info.ema_decay);
            }
        }
    }
    Ok(info)
}

fn saved_phonetic_encoder(metadata: &serde_json::Value, weight: f64) -> Result<Option<Embedder>> {
    if weight == 0.0 {
        return Ok(None);
    }
    // Older runs that omitted this field used mHuBERT-147 for the auxiliary loss.
    match metadata.get("phonetic_encoder") {
        Some(value) => serde_json::from_value(value.clone())
            .context("reading phonetic encoder in train.log"),
        None => Ok(Some(Embedder::Mhubert147)),
    }
}

#[derive(Default)]
struct PretrainedArchitecture {
    bands: bool,
    anti_alias: bool,
    articulation: Option<ArticulationProfile>,
    paired_content: u8,
}

fn pretrained_architecture(paths: &Paths, request: &Request) -> Result<PretrainedArchitecture> {
    let Some(model) = pretrained_path(paths, request, "G") else {
        return Ok(PretrainedArchitecture::default());
    };
    let checkpoint = Checkpoint::open(&model)?;
    let section = if checkpoint.has("model") {
        "model"
    } else {
        "weight"
    };
    let bands = checkpoint
        .tensor_shape(Some(section), super::synth::bands::WEIGHT)
        .is_ok();
    let aa = checkpoint
        .metadata("nvc")?
        .and_then(|meta| meta.get("dec_aa").and_then(serde_json::Value::as_bool))
        .or(checkpoint
            .metadata("dec_aa")?
            .and_then(|value| value.as_bool()))
        .unwrap_or(false);
    Ok(PretrainedArchitecture {
        bands,
        anti_alias: aa,
        articulation: ArticulationProfile::from_generator(&checkpoint)?,
        paired_content: paired::version(&checkpoint)?,
    })
}

fn pretrained_path(paths: &Paths, request: &Request, kind: &str) -> Option<PathBuf> {
    let custom = if kind == "G" {
        &request.pretrained_g
    } else {
        &request.pretrained_d
    };
    if request.random_weights && custom.is_none() {
        return None;
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
    (model.is_file() || custom.is_some()).then_some(model)
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

#[allow(clippy::too_many_arguments)]
fn save_pair(
    experiment: &Path,
    suffix: usize,
    epoch: usize,
    step: usize,
    g: Save<'_>,
    d: Save<'_>,
    initial_rate: f64,
    dec_aa: bool,
    articulation: Option<&ArticulationProfile>,
    paired_content: u8,
) -> Result<()> {
    let staging = tempfile::TempDir::new_in(experiment)?;
    let names = checkpoint_names(suffix);
    for ((optimizer, tensors), filename) in [g, d].into_iter().zip(&names) {
        let mut entries = vec![
            ("model".into(), Value::tensors(tensors)),
            ("iteration".into(), json!(epoch).into()),
            ("optimizer".into(), optimizer.state_dict()?),
            ("learning_rate".into(), json!(initial_rate).into()),
            ("rvc_global_step".into(), json!(step).into()),
        ];
        if dec_aa {
            // Lets a run that starts from this checkpoint as pretrained keep the decoder layout.
            entries.push(("dec_aa".into(), json!(true).into()));
        }
        if filename == &names[0]
            && let Some(profile) = articulation
        {
            entries.push(("articulation".into(), json!(profile).into()));
        }
        if filename == &names[0] && paired_content != 0 {
            entries.push(("paired_content".into(), json!(paired_content).into()));
        }
        let data = Value::Dict(entries);
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
    fn resume_restores_an_active_phonetic_encoder_from_the_training_log() -> Result<()> {
        let log = json!({ "phonetic_encoder": "hubert_base" });
        assert_eq!(saved_phonetic_encoder(&log, 0.1)?, Some(Embedder::HubertBase));
        assert_eq!(saved_phonetic_encoder(&log, 0.0)?, None);
        assert_eq!(saved_phonetic_encoder(&json!({}), 0.1)?, Some(Embedder::Mhubert147));
        assert_eq!(saved_phonetic_encoder(&json!({"phonetic_encoder": null}), 0.1)?, None);
        Ok(())
    }

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
                false,
                None,
                0,
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
            resume_position(&Checkpoint::open(&g)?, &Checkpoint::open(&d)?)?,
            (3, Some(4))
        );
        assert!(!pending.exists());
        recover_pending_pair(experiment)?;
        Ok(())
    }
}
