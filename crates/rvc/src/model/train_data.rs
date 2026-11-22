use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use clap::ValueEnum;
use ndarray::{Array, Array1, Array2, Dimension};
use ndarray_npy::{ReadNpyError, read_npy};
use rand::{Rng, SeedableRng, rngs::StdRng, seq::SliceRandom};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tch::{Device, Kind, Tensor};

use crate::catalog::{ArticulationProfile, Embedder, Unvoiced};
use crate::{audio, dataset};

use super::{coarse_pitch, extract::RAW_F0, timbre};

const PHONE_FRAME_LIMIT: i64 = 900;
const BUCKET_BOUNDARIES: [u64; 9] = [100, 200, 300, 400, 500, 600, 700, 800, 900];

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq, ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum Version {
    V1,
    V2,
}

impl Version {
    pub fn dimensions(self) -> i64 {
        match self {
            Self::V1 => 256,
            Self::V2 => 768,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::V1 => "v1",
            Self::V2 => "v2",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum SpeakerMode {
    Single(i64),
    Multi,
}

pub struct PrepareRequest<'a> {
    pub project_root: &'a Path,
    pub experiment_dir: &'a Path,
    pub mute_dir: &'a Path,
    pub sample_rate: u32,
    pub version: Version,
    pub uses_f0: bool,
    pub rebuild_filelist: bool,
    pub speakers: SpeakerMode,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Speaker {
    pub id: i64,
    pub name: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Config {
    pub train: TrainConfig,
    pub data: DataConfig,
    pub model: ModelConfig,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub speaker_info: Vec<Speaker>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedder_model: Option<Embedder>,
    #[serde(flatten)]
    other: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainConfig {
    pub log_interval: usize,
    pub seed: i64,
    pub epochs: usize,
    pub learning_rate: f64,
    pub betas: [f64; 2],
    pub eps: f64,
    pub batch_size: usize,
    pub lr_decay: f64,
    pub segment_size: i64,
    pub c_mel: f64,
    pub c_kl: f64,
    #[serde(default)]
    pub bf16: bool,
    #[serde(default)]
    pub gradient_checkpointing: bool,
    #[serde(flatten)]
    other: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DataConfig {
    pub sampling_rate: u32,
    pub filter_length: i64,
    pub hop_length: i64,
    pub win_length: i64,
    pub n_mel_channels: i64,
    pub mel_fmin: f64,
    pub mel_fmax: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) content_aug: Option<timbre::Options>,
    #[serde(default)]
    pub f0_unvoiced: Unvoiced,
    #[serde(default = "legacy_noise_scale")]
    pub default_noise_scale: f64,
    #[serde(flatten)]
    other: BTreeMap<String, Value>,
}

fn legacy_noise_scale() -> f64 {
    0.66666
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelConfig {
    pub inter_channels: i64,
    pub hidden_channels: i64,
    pub filter_channels: i64,
    pub n_heads: i64,
    pub n_layers: i64,
    pub kernel_size: i64,
    pub p_dropout: f64,
    pub resblock: String,
    pub resblock_kernel_sizes: Vec<i64>,
    pub resblock_dilation_sizes: Vec<Vec<i64>>,
    pub upsample_rates: Vec<i64>,
    pub upsample_initial_channel: i64,
    pub upsample_kernel_sizes: Vec<i64>,
    pub use_spectral_norm: bool,
    pub gin_channels: i64,
    pub spk_embed_dim: i64,
    /// Condition the content encoder on source high-band features.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub hf_bands: bool,
    /// Anti-aliased decoder activations.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub dec_aa: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub articulation: Option<ArticulationProfile>,
    /// Version of the input-only, paired speaker augmentation recipe.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub paired_content: u8,
    #[serde(flatten)]
    other: BTreeMap<String, Value>,
}

fn is_zero(value: &u8) -> bool {
    *value == 0
}

impl Config {
    pub(super) fn save(&self, experiment: &Path) -> Result<()> {
        let mut json = serde_json::to_vec_pretty(self)?;
        json.push(b'\n');
        audio::write_file(&experiment.join("config.json"), &json, true)
    }

    pub fn load(experiment: &Path, version: Version, sample_rate: u32) -> Result<Self> {
        let path = experiment.join("config.json");
        let parsed: Self = match fs::File::open(&path) {
            Ok(file) => serde_json::from_reader(file)
                .with_context(|| format!("reading training config {}", path.display()))?,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                let stock = match (version, sample_rate) {
                    (Version::V1, 32_000) => include_str!("../../../../configs/v1/32k.json"),
                    (_, 40_000) => include_str!("../../../../configs/v1/40k.json"),
                    (Version::V1, 48_000) => include_str!("../../../../configs/v1/48k.json"),
                    (Version::V2, 32_000) => include_str!("../../../../configs/v2/32k.json"),
                    (Version::V2, 48_000) => include_str!("../../../../configs/v2/48k.json"),
                    _ => bail!("no stock training config for {sample_rate} Hz"),
                };
                serde_json::from_str(stock).context("reading stock RVC config")?
            }
            Err(error) => return Err(error).with_context(|| format!("opening {}", path.display())),
        };
        ensure!(
            parsed.data.sampling_rate == sample_rate,
            "training config rate {} does not match requested {sample_rate}",
            parsed.data.sampling_rate
        );
        ensure!(
            parsed.data.filter_length > 0
                && parsed.data.hop_length > 0
                && parsed.data.win_length > 0
                && parsed.train.segment_size >= parsed.data.hop_length,
            "invalid training window or segment size"
        );
        if let Some(profile) = &parsed.model.articulation {
            profile.validate()?;
        }
        Ok(parsed)
    }
}

pub struct Prepared {
    pub config: Config,
    pub dataset: Dataset,
    pub filelist: PathBuf,
}

pub fn prepare(request: &PrepareRequest<'_>) -> Result<Prepared> {
    let experiment = std::path::absolute(request.experiment_dir)?;
    if request.uses_f0 {
        ensure!(
            !experiment
                .join(RAW_F0)
                .join(super::extract::PITCH_PENDING)
                .try_exists()?,
            "pitch extraction was interrupted; finish F0 extraction before training"
        );
    }
    let mut config = Config::load(&experiment, request.version, request.sample_rate)?;
    let filelist = experiment.join("filelist.txt");
    if !request.rebuild_filelist
        && filelist
            .try_exists()
            .with_context(|| format!("checking {}", filelist.display()))?
    {
        let dataset = Dataset::open(
            &filelist,
            &config.data,
            request.version,
            request.uses_f0,
            request.project_root,
        )?;
        if let Some(record) = dataset
            .records
            .iter()
            .find(|record| !(0..config.model.spk_embed_dim).contains(&record.sid))
        {
            bail!(
                "speaker ID {} is outside the trained embedding table",
                record.sid
            );
        }
        return Ok(Prepared {
            config,
            dataset,
            filelist,
        });
    }
    let wave_dir = experiment.join("0_gt_wavs");
    let manifest = match request.speakers {
        SpeakerMode::Multi => {
            config.model.spk_embed_dim = 110;
            Some(dataset::load_manifest(&experiment)?)
        }
        SpeakerMode::Single(id) => {
            ensure!(
                (0..config.model.spk_embed_dim).contains(&id),
                "speaker ID {id} is outside the trained embedding table"
            );
            None
        }
    };
    let by_key: BTreeMap<_, _> = manifest
        .iter()
        .flat_map(|manifest| manifest.entries.iter())
        .map(|entry| (entry.output_key.as_str(), entry))
        .collect();
    let mut active_names = BTreeMap::new();
    let mut lines = Vec::new();
    let mut files = fs::read_dir(&wave_dir)
        .with_context(|| format!("reading clipped dataset {}", wave_dir.display()))?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<std::io::Result<Vec<_>>>()?;
    files.sort();
    for wave in files {
        if wave.extension().is_none_or(|extension| extension != "wav") || !wave.is_file() {
            continue;
        }
        let name = wave
            .file_stem()
            .and_then(|name| name.to_str())
            .context("invalid WAV name")?;
        let record = Record::from_clip(
            &experiment,
            name,
            wave.clone(),
            request.version,
            request.uses_f0,
        );
        if !record.has_features() {
            continue;
        }
        let (speaker_id, name, repeat) = match request.speakers {
            SpeakerMode::Single(id) => (id, None, 1),
            SpeakerMode::Multi => {
                let key = name.rsplit_once('_').map(|(key, _)| key);
                let Some(entry) = key.and_then(|key| by_key.get(key)) else {
                    continue;
                };
                let speaker = i64::from(entry.speaker_id);
                active_names.insert(speaker, entry.speaker_name.clone());
                (
                    speaker,
                    Some(entry.speaker_name.as_str()),
                    usize::try_from(entry.repeat)?,
                )
            }
        };
        let line = record.line(speaker_id, name)?;
        lines
            .try_reserve(repeat)
            .context("training dataset repeat exceeds available memory")?;
        lines.extend(std::iter::repeat_n(line, repeat));
    }
    ensure!(
        !lines.is_empty(),
        "no aligned audio and features; run dataset preprocessing and feature extraction first"
    );
    config.speaker_info = active_names
        .into_iter()
        .map(|(id, name)| Speaker { id, name })
        .collect();
    let mute = std::path::absolute(request.mute_dir)?;
    let sr = request.sample_rate / 1000;
    let wave = mute.join("0_gt_wavs").join(format!("mute{sr}k.wav"));
    let record = Record::from_clip(&mute, "mute", wave, request.version, request.uses_f0);
    ensure!(
        record.wave.is_file() && record.has_features(),
        "training silence samples are missing ({})",
        mute.display()
    );
    match request.speakers {
        SpeakerMode::Single(id) => lines.extend(std::iter::repeat_n(record.line(id, None)?, 2)),
        SpeakerMode::Multi => {
            for speaker in &config.speaker_info {
                lines.extend(std::iter::repeat_n(
                    record.line(speaker.id, Some(&speaker.name))?,
                    2,
                ));
            }
        }
    }
    lines.shuffle(&mut rand::rng());
    let text = lines.join("\n");
    let dataset = Dataset::from_text(
        &text,
        &config.data,
        request.version,
        request.uses_f0,
        request.project_root,
    )?;
    audio::write_file(&filelist, text.as_bytes(), true)?;
    config.save(&experiment)?;
    Ok(Prepared {
        config,
        dataset,
        filelist,
    })
}

#[derive(Clone)]
struct PitchPaths {
    coarse: PathBuf,
    hz: PathBuf,
}

impl PitchPaths {
    fn raw(&self) -> Option<PathBuf> {
        let directory = self.hz.parent()?;
        (directory.file_name()? == "2b-f0nsf").then(|| {
            directory
                .with_file_name(RAW_F0)
                .join(self.hz.file_name().unwrap_or_default())
        })
    }
}

struct Record {
    wave: PathBuf,
    phone: PathBuf,
    pitch: Option<PitchPaths>,
    sid: i64,
    speaker_name: Option<String>,
}

impl Record {
    fn from_clip(root: &Path, name: &str, wave: PathBuf, version: Version, uses_f0: bool) -> Self {
        let phone = root
            .join(format!("3_feature{}", version.dimensions()))
            .join(format!("{name}.npy"));
        let pitch = uses_f0.then(|| PitchPaths {
            coarse: root.join("2a_f0").join(format!("{name}.wav.npy")),
            hz: root.join("2b-f0nsf").join(format!("{name}.wav.npy")),
        });
        Self {
            wave,
            phone,
            pitch,
            sid: 0,
            speaker_name: None,
        }
    }

    fn has_features(&self) -> bool {
        self.phone.is_file()
            && self
                .pitch
                .as_ref()
                .is_none_or(|pitch| pitch.coarse.is_file() && pitch.hz.is_file())
    }

    fn line(&self, id: i64, name: Option<&str>) -> Result<String> {
        let mut fields = vec![filelist_path(&self.wave)?, filelist_path(&self.phone)?];
        if let Some(pitch) = &self.pitch {
            fields.extend([filelist_path(&pitch.coarse)?, filelist_path(&pitch.hz)?]);
        }
        let id = id.to_string();
        fields.push(&id);
        fields.extend(name);
        Ok(fields.join("|"))
    }
}

fn filelist_path(path: &Path) -> Result<&str> {
    let text = path
        .to_str()
        .with_context(|| format!("non-UTF-8 training path ({})", path.display()))?;
    ensure!(
        !text.contains(['|', '\r', '\n']),
        "unsupported character in training path ({text})"
    );
    Ok(text)
}

pub struct Dataset {
    records: Vec<Record>,
    lengths: Vec<u64>,
    data: DataConfig,
    version: Version,
    uses_f0: bool,
    augment: Option<timbre::Augmentation>,
    mute_dir: Option<PathBuf>,
    /// Layered copy of the clips (see `model::layers`) and how often to read it.
    layers: Option<(PathBuf, f64)>,
    /// Source wave and phone features of each paired example, indexed by clean WAV.
    pairs: BTreeMap<PathBuf, (Option<PathBuf>, PathBuf)>,
}

impl Dataset {
    pub fn open(
        filelist: &Path,
        data: &DataConfig,
        version: Version,
        uses_f0: bool,
        project_root: &Path,
    ) -> Result<Self> {
        let text = fs::read_to_string(filelist)
            .with_context(|| format!("reading {}", filelist.display()))?;
        Self::from_text(&text, data, version, uses_f0, project_root)
    }

    fn from_text(
        text: &str,
        data: &DataConfig,
        version: Version,
        uses_f0: bool,
        project_root: &Path,
    ) -> Result<Self> {
        let lower = data
            .other
            .get("min_text_len")
            .and_then(Value::as_u64)
            .unwrap_or(1) as usize;
        let upper = data
            .other
            .get("max_text_len")
            .and_then(Value::as_u64)
            .unwrap_or(5_000) as usize;
        ensure!(data.hop_length > 0, "training hop length must be positive");
        let mut records = Vec::new();
        let mut lengths = Vec::new();
        for (number, line) in text.lines().enumerate() {
            let fields: Vec<_> = line.trim().split('|').collect();
            let (wave, phone, pitch, sid, rest) = match fields.as_slice() {
                [wave, phone, coarse, hz, sid, rest @ ..] if uses_f0 => (
                    *wave,
                    *phone,
                    Some(PitchPaths {
                        coarse: project_root.join(coarse),
                        hz: project_root.join(hz),
                    }),
                    *sid,
                    rest,
                ),
                [wave, phone, sid, rest @ ..] if !uses_f0 => (*wave, *phone, None, *sid, rest),
                _ => bail!("malformed training filelist line {}", number + 1),
            };
            if !(lower..=upper).contains(&phone.chars().count()) {
                continue;
            }
            let record = Record {
                wave: project_root.join(wave),
                phone: project_root.join(phone),
                pitch,
                sid: sid
                    .parse()
                    .with_context(|| format!("invalid speaker ID on line {}", number + 1))?,
                speaker_name: rest
                    .first()
                    .filter(|name| !name.is_empty())
                    .map(|name| (*name).to_owned()),
            };
            let bytes = record
                .wave
                .metadata()
                .with_context(|| format!("reading training WAV {}", record.wave.display()))?
                .len();
            lengths.push(bytes / (3 * data.hop_length as u64));
            records.push(record);
        }
        ensure!(
            !records.is_empty(),
            "training filelist has no eligible files"
        );
        Ok(Self {
            records,
            lengths,
            data: data.clone(),
            version,
            uses_f0,
            augment: None,
            layers: None,
            pairs: BTreeMap::new(),
            mute_dir: None,
        })
    }

    /// Whether every voice clip has raw F0 (with unvoiced frames at zero) from extraction.
    pub fn has_raw_pitch(&self, mute: &Path) -> bool {
        self.records
            .iter()
            .filter(|record| !record.wave.starts_with(mute))
            .all(|record| {
                record
                    .pitch
                    .as_ref()
                    .and_then(PitchPaths::raw)
                    .is_some_and(|path| path.is_file())
            })
    }

    pub fn sample_count(&self) -> usize {
        self.records.len()
    }

    pub(super) fn paired_sources(&self) -> BTreeSet<PathBuf> {
        self.records
            .iter()
            .filter(|record| self.mute_dir.as_ref().is_none_or(|mute| !record.wave.starts_with(mute)))
            .map(|record| record.wave.clone())
            .collect()
    }

    /// Clean recordings grouped by the same speaker ID the voice will receive at inference.
    pub(super) fn frication_sources(&self) -> BTreeMap<i64, BTreeSet<PathBuf>> {
        let mut sources = BTreeMap::<_, BTreeSet<_>>::new();
        for record in &self.records {
            if self.mute_dir.as_ref().is_none_or(|mute| !record.wave.starts_with(mute)) {
                sources.entry(record.sid).or_default().insert(record.wave.clone());
            }
        }
        sources
    }

    pub(super) fn set_paired_sources(
        &mut self,
        pairs: BTreeMap<PathBuf, (PathBuf, PathBuf)>,
        source_bands: bool,
    ) {
        self.pairs = pairs.into_iter()
            .map(|(clean, (wave, phone))| (clean, (source_bands.then_some(wave), phone)))
            .collect();
    }

    /// Read content features, F0 and source bands from the layered copy in
    /// `directory` for a `probability` share of the examples. The target audio
    /// stays the clean clip.
    pub(super) fn configure_layers(&mut self, directory: PathBuf, probability: f64) -> Result<()> {
        ensure!(
            (0.0..=1.0).contains(&probability),
            "layered example probability must be between 0 and 1"
        );
        let found = self
            .records
            .iter()
            .filter(|record| self.layered(record, &directory).is_some())
            .count();
        ensure!(
            found > 0,
            "no layered features found in {}; run layer-aug and extract first",
            directory.display()
        );
        self.layers = Some((directory, probability));
        Ok(())
    }

    fn layered(&self, record: &Record, directory: &Path) -> Option<Record> {
        let name = record.phone.file_stem()?.to_str()?;
        let wave = directory.join("0_gt_wavs").join(format!("{name}.wav"));
        let layered = Record::from_clip(directory, name, wave, self.version, self.uses_f0);
        let raw_ok = self.data.f0_unvoiced != Unvoiced::Zero
            || layered
                .pitch
                .as_ref()
                .and_then(PitchPaths::raw)
                .is_none_or(|raw| raw.is_file());
        (layered.has_features() && layered.wave.is_file() && raw_ok).then_some(layered)
    }

    pub(super) fn configure_timbre(
        &mut self,
        paths: &crate::catalog::Paths,
        experiment: &Path,
        config: &mut Config,
        device: Device,
        report: impl FnMut(&str) -> std::ops::ControlFlow<()>,
    ) -> Result<()> {
        let mute = paths.experiments.join("mute");
        let voices = &config.speaker_info;
        let active = config
            .data
            .content_aug
            .as_ref()
            .filter(|aug| aug.prob != 0.0);
        if let Some(options) = active {
            ensure!(
                self.version == Version::V2 && self.uses_f0,
                "NVC-TS requires an RVC v2 model with F0"
            );
            if options.pool.is_some() {
                config.data.f0_unvoiced = Unvoiced::Zero;
            }
        }
        // Validate raw F0 before expensive kNN precomputation.
        if self.uses_f0 && config.data.f0_unvoiced == Unvoiced::Zero {
            for entry in self
                .records
                .iter()
                .filter(|record| !record.wave.starts_with(&mute))
            {
                ensure!(
                    entry
                        .pitch
                        .as_ref()
                        .and_then(PitchPaths::raw)
                        .is_some_and(|p| p.is_file()),
                    "missing raw F0 for {}. Run pitch extraction again.",
                    entry.wave.display()
                );
            }
        }
        if let Some(mut options) = active.cloned() {
            options.validate()?;
            if options.pool.is_some() {
                let mut clips: BTreeMap<PathBuf, BTreeSet<String>> = BTreeMap::new();
                for record in self.records.iter().filter(|r| !r.wave.starts_with(&mute)) {
                    let names = clips.entry(record.phone.clone()).or_default();
                    let name = record
                        .speaker_name
                        .as_deref()
                        .or_else(|| {
                            voices
                                .iter()
                                .find(|voice| voice.id == record.sid)
                                .map(|voice| voice.name.as_str())
                        })
                        .or_else(|| {
                            voices
                                .is_empty()
                                .then(|| experiment.file_name().and_then(|name| name.to_str()))
                                .flatten()
                        });
                    names.extend(name.map(str::to_owned));
                }
                let path =
                    timbre::generate(paths, experiment, &options, &clips, config, device, report)?;
                options.dirs = vec![path.strip_prefix(experiment)?.to_owned()];
            }
            self.augment = Some(timbre::Augmentation::load(experiment, &options)?);
            config.data.content_aug = Some(options);
        }
        self.data.f0_unvoiced = config.data.f0_unvoiced;
        self.mute_dir = Some(mute);
        Ok(())
    }

    pub fn batches(
        &self,
        batch_size: usize,
        epoch: u64,
        replicas: usize,
        rank: usize,
    ) -> Result<Vec<Vec<usize>>> {
        bucket_batches(&self.lengths, batch_size, epoch, replicas, rank, true)
    }

    pub fn load_batch(
        &self,
        indices: &[usize],
        device: Device,
        spectrogram: impl FnMut(&Tensor) -> Result<Tensor>,
    ) -> Result<Batch> {
        self.load_batch_seeded(indices, device, rand::random(), spectrogram)
    }

    pub(super) fn load_batch_seeded(
        &self,
        indices: &[usize],
        device: Device,
        seed: u64,
        spectrogram: impl FnMut(&Tensor) -> Result<Tensor>,
    ) -> Result<Batch> {
        self.finish_batch(self.stage_batch(indices, seed, None)?, device, spectrogram)
    }

    /// Host-side part of batch loading: files, features, F0 and augmentation.
    /// It runs without GPU work, so a background thread can prepare the next batch.
    /// With `pin` the tensors use page-locked memory for asynchronous copies.
    pub(super) fn stage_batch(
        &self,
        indices: &[usize],
        seed: u64,
        pin: Option<Device>,
    ) -> Result<Staged> {
        ensure!(!indices.is_empty(), "training batch is empty");
        let mut rng = StdRng::seed_from_u64(seed);
        let mut examples = Vec::with_capacity(indices.len());
        for &index in indices {
            let record = self
                .records
                .get(index)
                .with_context(|| format!("training index {index} is out of range"))?;
            let mut example = self.stage_example(record, &mut rng)?;
            if let Some(device) = pin {
                example.pin(device)?;
            }
            examples.push(example);
        }
        examples.sort_by_key(|example| std::cmp::Reverse(example.frames));
        Ok(Staged { examples })
    }

    /// Device-side part of batch loading: spectrograms and collation.
    pub(super) fn finish_batch(
        &self,
        staged: Staged,
        device: Device,
        mut spectrogram: impl FnMut(&Tensor) -> Result<Tensor>,
    ) -> Result<Batch> {
        let mut examples = Vec::with_capacity(staged.examples.len());
        for example in staged.examples {
            let move_to = |tensor: Tensor| -> Result<Tensor> {
                Ok(if tensor.device() == device {
                    tensor
                } else {
                    tensor.f_to_device_(device, tensor.kind(), true, false)?
                })
            };
            let wave = move_to(example.wave)?;
            let spec = spectrogram(&wave)?;
            ensure!(
                spec.dim() == 3
                    && spec.size()[0] == 1
                    && spec.size()[1] == self.data.filter_length / 2 + 1
                    && spec.size()[2] == example.spec_frames,
                "spectrogram must have shape [1, {}, {}]",
                self.data.filter_length / 2 + 1,
                example.spec_frames
            );
            let spec = spec.f_squeeze_dim(0)?;
            let band_spec = example
                .band_wave
                .map(|band_wave| -> Result<Tensor> {
                    let band = spectrogram(&move_to(band_wave)?)?.f_squeeze_dim(0)?;
                    let frames = band.size()[1].min(example.frames);
                    let band = band.f_narrow(1, 0, frames)?;
                    // Pad a short layered clip with its last frame.
                    Ok(if frames < example.frames {
                        let last = band.narrow(1, frames - 1, 1);
                        Tensor::cat(&[band, last.repeat([1, example.frames - frames])], 1)
                    } else {
                        band
                    })
                })
                .transpose()?;
            let (wave, spec) = if example.feature_frames != example.spec_frames {
                let samples = (example.frames * self.data.hop_length).min(wave.size()[1]);
                (
                    wave.f_narrow(1, 0, samples)?,
                    spec.f_narrow(1, 0, example.frames)?,
                )
            } else {
                (wave, spec)
            };
            let pitch = example
                .pitch
                .map(|(coarse, hz)| -> Result<_> { Ok((move_to(coarse)?, move_to(hz)?)) })
                .transpose()?;
            examples.push(Example {
                phone: move_to(example.phone)?,
                pitch,
                spec,
                band_spec,
                wave,
                sid: example.sid,
            });
        }
        Batch::collate(&examples, self.uses_f0, device)
    }

    fn stage_example(&self, clean: &Record, rng: &mut StdRng) -> Result<StagedExample> {
        let wave = waveform(&clean.wave, self.data.sampling_rate)?;
        let spec_frames = self.spectrogram_frames(wave.size()[1])?;
        let mute = self
            .mute_dir
            .as_ref()
            .is_some_and(|path| clean.wave.starts_with(path));
        let layered = self
            .layers
            .as_ref()
            .filter(|(_, probability)| !mute && rng.random_bool(*probability))
            .and_then(|(directory, _)| self.layered(clean, directory));
        let record = layered.as_ref().unwrap_or(clean);
        let pair = self.pairs.get(&clean.wave).filter(|_| rng.random_bool(0.72));
        let phone_path = pair.map_or(&record.phone, |(_, phone)| phone);
        let band_wave = pair
            .and_then(|(wave, _)| wave.as_ref())
            .or_else(|| layered.as_ref().map(|record| &record.wave))
            .map(|path| waveform(path, self.data.sampling_rate))
            .transpose()?;
        let mut features: Array2<f32> = float_array(phone_path)?;
        if let Some(augment) = self.augment.as_ref().filter(|_| !mute) {
            augment.mix(&clean.phone, &mut features, rng)?;
        }
        let dims = self.version.dimensions();
        ensure!(
            features.ncols() == dims as usize,
            "expected {dims} HuBERT channels ({})",
            phone_path.display()
        );
        let feature_frames = (features
            .nrows()
            .saturating_mul(2)
            .min(PHONE_FRAME_LIMIT as usize)) as i64;
        let frames = spec_frames.min(feature_frames);
        ensure!(
            frames > 0,
            "training clip has no aligned frames ({})",
            record.wave.display()
        );
        let mut features_repeated = Vec::with_capacity((frames * dims) as usize);
        for frame in 0..frames as usize {
            features_repeated.extend(features.row(frame / 2).iter().copied());
        }
        ensure!(
            features_repeated.iter().all(|value| value.is_finite()),
            "non-finite HuBERT values"
        );
        let phone = Tensor::f_from_slice(&features_repeated)?.f_view([frames, dims])?;
        let pitch = record
            .pitch
            .as_ref()
            .map(|paths| {
                if self.data.f0_unvoiced == Unvoiced::Zero && !mute {
                    read_raw_pitch(
                        &paths.raw().context("cannot find raw F0 directory")?,
                        frames as usize,
                    )
                } else {
                    read_pitch(paths, frames as usize)
                }
            })
            .transpose()?;
        Ok(StagedExample {
            phone,
            pitch,
            wave,
            band_wave,
            frames,
            spec_frames,
            feature_frames,
            sid: record.sid,
        })
    }

    /// Frame count of [`super::train_losses::MelSpectrogram::spectrogram`] for a clip.
    fn spectrogram_frames(&self, samples: i64) -> Result<i64> {
        let fft = self.data.filter_length;
        let hop = self.data.hop_length;
        let padded = samples + 2 * ((fft - hop) / 2);
        ensure!(
            padded >= fft && samples > (fft - hop) / 2,
            "training clip is too short for the spectrogram window"
        );
        Ok((padded - fft) / hop + 1)
    }
}

/// A batch whose host-side work is done; see [`Dataset::stage_batch`].
pub(super) struct Staged {
    examples: Vec<StagedExample>,
}

struct StagedExample {
    phone: Tensor,
    pitch: Option<(Tensor, Tensor)>,
    /// The full clip. Its spectrogram is computed before trimming to `frames`.
    wave: Tensor,
    /// The layered clip the inputs came from, for source band features.
    band_wave: Option<Tensor>,
    frames: i64,
    spec_frames: i64,
    feature_frames: i64,
    sid: i64,
}

unsafe extern "C" {
    fn rvc_pin_memory(
        tensor: *const std::ffi::c_void,
        error: *mut std::ffi::c_char,
        error_length: usize,
    ) -> *mut std::ffi::c_void;
}

fn pinned(tensor: &Tensor) -> Result<Tensor> {
    let mut error = [0; 512];
    let output = unsafe { rvc_pin_memory(tensor.as_ptr().cast(), error.as_mut_ptr(), error.len()) };
    if output.is_null() {
        // The bridge always terminates its diagnostic and the buffer starts zeroed.
        let error = unsafe { std::ffi::CStr::from_ptr(error.as_ptr()) }.to_string_lossy();
        anyhow::bail!("pinning a training batch: {error}");
    }
    // The bridge transfers ownership of a heap-allocated LibTorch tensor.
    Ok(unsafe { Tensor::from_ptr(output.cast()) })
}

impl StagedExample {
    fn pin(&mut self, device: Device) -> Result<()> {
        if !matches!(device, Device::Cuda(_)) {
            return Ok(());
        }
        self.phone = pinned(&self.phone)?;
        self.wave = pinned(&self.wave)?;
        if let Some((coarse, hz)) = &mut self.pitch {
            *coarse = pinned(coarse)?;
            *hz = pinned(hz)?;
        }
        Ok(())
    }
}

fn bucket_batches(
    lengths: &[u64],
    batch_size: usize,
    epoch: u64,
    replicas: usize,
    rank: usize,
    shuffle: bool,
) -> Result<Vec<Vec<usize>>> {
    ensure!(
        batch_size > 0 && replicas > 0 && rank < replicas,
        "invalid training bucket parameters"
    );
    // train.py multiplies the configured size by the number of GPUs before constructing the sampler.
    let per_rank = batch_size
        .checked_mul(replicas)
        .context("batch size overflow")?;
    let per_bucket = per_rank
        .checked_mul(replicas)
        .context("batch size overflow")?;
    let mut buckets: [Vec<usize>; BUCKET_BOUNDARIES.len() - 1] =
        std::array::from_fn(|_| Vec::new());
    for (index, &length) in lengths.iter().enumerate() {
        if let Some(bucket) = BUCKET_BOUNDARIES
            .windows(2)
            .position(|bounds| bounds[0] < length && length <= bounds[1])
        {
            buckets[bucket].push(index);
        }
    }
    let mut rng = StdRng::seed_from_u64(epoch);
    let mut batches = Vec::new();
    for mut bucket in buckets.into_iter().filter(|bucket| !bucket.is_empty()) {
        if shuffle {
            bucket.shuffle(&mut rng);
        }
        let count = bucket
            .len()
            .div_ceil(per_bucket)
            .checked_mul(per_bucket)
            .context("bucket size overflow")?;
        let positions: Vec<_> = (rank..count)
            .step_by(replicas)
            .map(|index| bucket[index % bucket.len()])
            .collect();
        batches.extend(positions.chunks_exact(per_rank).map(<[usize]>::to_vec));
    }
    if shuffle {
        batches.shuffle(&mut rng);
    }
    ensure!(
        !batches.is_empty(),
        "no audio clips in the 100–900-frame training buckets"
    );
    Ok(batches)
}

struct Example {
    phone: Tensor,
    pitch: Option<(Tensor, Tensor)>,
    spec: Tensor,
    band_spec: Option<Tensor>,
    wave: Tensor,
    sid: i64,
}

pub struct Batch {
    pub phone: Tensor,
    pub phone_lengths: Tensor,
    pub pitch: Option<Tensor>,
    pub pitchf: Option<Tensor>,
    pub spec: Tensor,
    /// Spectrogram the source band features come from, when it differs from `spec`.
    pub band_spec: Option<Tensor>,
    pub spec_lengths: Tensor,
    pub wave: Tensor,
    pub wave_lengths: Tensor,
    pub sid: Tensor,
}

impl Batch {
    fn collate(examples: &[Example], uses_f0: bool, device: Device) -> Result<Self> {
        let phone_lengths: Vec<_> = examples
            .iter()
            .map(|example| example.phone.size()[0])
            .collect();
        let spec_lengths: Vec<_> = examples
            .iter()
            .map(|example| example.spec.size()[1])
            .collect();
        let wave_lengths: Vec<_> = examples
            .iter()
            .map(|example| example.wave.size()[1])
            .collect();
        let max_phone = phone_lengths.iter().copied().max().unwrap_or(0);
        let max_spec = spec_lengths.iter().copied().max().unwrap_or(0);
        let max_wave = wave_lengths.iter().copied().max().unwrap_or(0);
        let mut phone = Vec::with_capacity(examples.len());
        let mut spec = Vec::with_capacity(examples.len());
        let mut wave = Vec::with_capacity(examples.len());
        let mut pitch = Vec::new();
        let mut pitchf = Vec::new();
        let layered = examples.iter().any(|example| example.band_spec.is_some());
        let mut band_spec = Vec::new();
        for example in examples {
            if layered {
                let source = example.band_spec.as_ref().unwrap_or(&example.spec);
                band_spec.push(source.f_constant_pad_nd([0, max_spec - source.size()[1]])?);
            }
            phone.push(example.phone.f_constant_pad_nd([
                0,
                0,
                0,
                max_phone - example.phone.size()[0],
            ])?);
            spec.push(
                example
                    .spec
                    .f_constant_pad_nd([0, max_spec - example.spec.size()[1]])?,
            );
            wave.push(
                example
                    .wave
                    .f_constant_pad_nd([0, max_wave - example.wave.size()[1]])?,
            );
            if let Some((coarse, hz)) = &example.pitch {
                pitch.push(coarse.f_constant_pad_nd([0, max_phone - coarse.size()[0]])?);
                pitchf.push(hz.f_constant_pad_nd([0, max_phone - hz.size()[0]])?);
            }
        }
        let stack = |values: &[Tensor]| -> Result<Tensor> {
            Ok(Tensor::f_stack(values, 0)?.f_to_device(device)?)
        };
        let lengths = |values: &[i64]| -> Result<Tensor> {
            Ok(Tensor::f_from_slice(values)?.f_to_device(device)?)
        };
        let sids: Vec<_> = examples.iter().map(|example| example.sid).collect();
        let (pitch, pitchf) = if uses_f0 {
            ensure!(
                pitch.len() == examples.len(),
                "missing F0 in the training batch"
            );
            (Some(stack(&pitch)?), Some(stack(&pitchf)?))
        } else {
            (None, None)
        };
        Ok(Self {
            phone: stack(&phone)?,
            phone_lengths: lengths(&phone_lengths)?,
            pitch,
            pitchf,
            spec: stack(&spec)?,
            band_spec: if layered {
                Some(stack(&band_spec)?)
            } else {
                None
            },
            spec_lengths: lengths(&spec_lengths)?,
            wave: stack(&wave)?,
            wave_lengths: lengths(&wave_lengths)?,
            sid: lengths(&sids)?,
        })
    }
}

pub(super) fn float_array<D: Dimension>(path: &Path) -> Result<Array<f32, D>> {
    let values = match read_npy(path) {
        Err(ReadNpyError::WrongDescriptor(_)) => {
            read_npy::<_, Array<f64, D>>(path).map(|array| array.mapv(|x| x as f32))
        }
        result => result,
    };
    values.with_context(|| format!("reading float NumPy array ({})", path.display()))
}

fn read_pitch(paths: &PitchPaths, frames: usize) -> Result<(Tensor, Tensor)> {
    let coarse: Array1<i64> = match read_npy(&paths.coarse) {
        Err(ReadNpyError::WrongDescriptor(_)) => {
            read_npy::<_, Array1<i32>>(&paths.coarse).map(|array| array.mapv(i64::from))
        }
        result => result,
    }
    .with_context(|| format!("reading coarse F0 ({})", paths.coarse.display()))?;
    let hz: Array1<f32> = float_array(&paths.hz)?;
    let coarse: Vec<_> = coarse.iter().copied().take(frames).collect();
    let hz: Vec<_> = hz.iter().copied().take(frames).collect();
    ensure!(
        coarse.iter().all(|id| (0..=255).contains(id)),
        "coarse F0 is outside 0–255"
    );
    ensure!(
        hz.iter().all(|value| value.is_finite()),
        "continuous F0 contains non-finite values"
    );
    Ok((Tensor::f_from_slice(&coarse)?, Tensor::f_from_slice(&hz)?))
}

fn read_raw_pitch(path: &Path, frames: usize) -> Result<(Tensor, Tensor)> {
    let pitch: Array1<f32> = float_array(path)?;
    let hz: Vec<_> = pitch.iter().copied().take(frames).collect();
    ensure!(
        hz.iter().all(|value| value.is_finite() && *value >= 0.0),
        "invalid raw F0 in {}",
        path.display()
    );
    let coarse: Vec<_> = hz.iter().copied().map(coarse_pitch).collect();
    Ok((Tensor::f_from_slice(&coarse)?, Tensor::f_from_slice(&hz)?))
}

fn waveform(path: &Path, rate: u32) -> Result<Tensor> {
    let mut reader = hound::WavReader::open(path)
        .with_context(|| format!("reading WAV ({})", path.display()))?;
    let spec = reader.spec();
    ensure!(
        spec.channels == 1 && spec.sample_rate == rate,
        "training WAV must be mono at {rate} Hz ({})",
        path.display()
    );
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<hound::Result<_>>()?,
        hound::SampleFormat::Int => reader
            .samples::<i32>()
            .map(|sample| {
                sample.map(|x| match spec.bits_per_sample {
                    8 => (x + 128) as f32,
                    24 => (x << 8) as f32,
                    _ => x as f32,
                })
            })
            .collect::<hound::Result<_>>()?,
    };
    ensure!(
        !samples.is_empty() && samples.iter().all(|x| x.is_finite()),
        "empty or non-finite training WAV ({})",
        path.display()
    );
    Ok(Tensor::f_from_slice(&samples)?
        .f_to_kind(Kind::Float)?
        .f_unsqueeze(0)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paired_batches_keep_clean_wave_pitch_and_speaker_on_every_draw() -> Result<()> {
        let project = tempfile::tempdir()?;
        let root = project.path();
        let clean = root.join("clean.wav");
        let paired_wave = root.join("other.wav");
        let paired_phone = root.join("other.npy");
        fs::write(&clean, audio::encode_wav(&[0.25; 8000], 40_000, false)?)?;
        fs::write(&paired_wave, audio::encode_wav(&[-0.20; 8000], 40_000, false)?)?;
        ndarray_npy::write_npy(root.join("clean.npy"), &Array2::<f32>::ones((10, 768)))?;
        ndarray_npy::write_npy(&paired_phone, &Array2::<f32>::from_elem((10,768), 2.0))?;
        ndarray_npy::write_npy(root.join("coarse.npy"), &Array1::<i64>::from_elem(20, 80))?;
        ndarray_npy::write_npy(root.join("hz.npy"), &Array1::<f32>::from_elem(20, 180.0))?;
        let config = Config::load(root, Version::V2, 40_000)?;
        let mut dataset=Dataset::from_text("clean.wav|clean.npy|coarse.npy|hz.npy|7", &config.data, Version::V2, true, root)?;
        dataset.set_paired_sources(BTreeMap::from([(clean.clone(),(paired_wave,paired_phone))]), false);
        let mut seen=[false; 2];
        for seed in 0..32 {
            let staged=dataset.stage_example(&dataset.records[0],&mut StdRng::seed_from_u64(seed))?;
            let value=staged.phone.double_value(&[0,0]);
            assert!(value==1.0 || value==2.0); seen[(value as usize)-1]=true;
            assert_eq!(staged.sid,7); assert_eq!(staged.wave.double_value(&[0,0]),8192.0);
            let (coarse,hz)=staged.pitch.unwrap(); assert_eq!(coarse.int64_value(&[0]),80);assert_eq!(hz.double_value(&[0]),180.0);
            assert!(staged.band_wave.is_none());
        }
        assert_eq!(seen,[true,true]);
        Ok(())
    }

    #[test]
    fn prepare_keeps_existing_filelist_and_resolves_project_relative_paths() -> Result<()> {
        let project = tempfile::tempdir()?;
        let experiment = project.path().join("logs/experiment");
        let media = project.path().join("training media");
        fs::create_dir_all(&experiment)?;
        fs::create_dir_all(&media)?;
        fs::File::create(media.join("voice.wav"))?;

        let mut config = Config::load(&experiment, Version::V2, 32_000)?;
        config.speaker_info = vec![Speaker {
            id: 7,
            name: "Existing speaker".into(),
        }];
        let config_bytes = serde_json::to_vec(&config)?;
        fs::write(experiment.join("config.json"), &config_bytes)?;
        let feature = media.join("voice.npy");
        let manifest = format!(
            "training media/voice.wav|{}|training media/f0.npy|training media/f0nsf.npy|7|Existing speaker\n",
            feature.display()
        );
        let filelist = experiment.join("filelist.txt");
        fs::write(&filelist, &manifest)?;

        let prepared = prepare(&PrepareRequest {
            project_root: project.path(),
            experiment_dir: &experiment,
            mute_dir: &project.path().join("logs/mute"),
            sample_rate: 32_000,
            version: Version::V2,
            uses_f0: true,
            rebuild_filelist: false,
            speakers: SpeakerMode::Single(0),
        })?;
        assert_eq!(prepared.dataset.sample_count(), 1);
        let record = &prepared.dataset.records[0];
        assert_eq!(record.wave, media.join("voice.wav"));
        assert_eq!(record.phone, feature);
        let pitch = record.pitch.as_ref().context("missing F0 paths")?;
        assert_eq!(pitch.coarse, media.join("f0.npy"));
        assert_eq!(pitch.hz, media.join("f0nsf.npy"));
        assert_eq!(record.sid, 7);
        assert_eq!(prepared.config.speaker_info[0].name, "Existing speaker");
        assert_eq!(fs::read_to_string(&filelist)?, manifest);
        assert_eq!(fs::read(experiment.join("config.json"))?, config_bytes);
        Ok(())
    }

    #[test]
    fn boundaries_padding_and_rank_match_distributed_sampler() -> anyhow::Result<()> {
        let sizes = [99, 100, 101, 110, 200, 201, 300, 850, 900, 901];
        let worker0 = bucket_batches(&sizes, 1, 1, 2, 0, false)?;
        let worker1 = bucket_batches(&sizes, 1, 1, 2, 1, false)?;
        assert_eq!(worker0, [vec![2, 4], vec![5, 5], vec![7, 7]]);
        assert_eq!(worker1, [vec![3, 2], vec![6, 6], vec![8, 8]]);
        Ok(())
    }
}
