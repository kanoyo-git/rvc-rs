use std::collections::BTreeMap;
use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;
use std::time::SystemTime;

use anyhow::{Context, Result, bail, ensure};
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::checkpoint::Checkpoint;

mod embedder;
pub use embedder::Embedder;

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq, clap::ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum Unvoiced {
    #[default]
    Interpolate,
    Zero,
}

#[derive(Debug, Serialize, PartialEq)]
pub struct VoiceFormat {
    pub embedder: Embedder,
    pub f0_unvoiced: Unvoiced,
    pub default_noise_scale: f64,
    /// Decoder activations run anti-aliased (see `synth::alias_free`).
    #[serde(default)]
    pub dec_aa: bool,
    #[serde(skip)]
    pub normalize_input: bool,
}

impl Default for VoiceFormat {
    fn default() -> Self {
        Self {
            embedder: Embedder::default(),
            f0_unvoiced: Unvoiced::default(),
            default_noise_scale: 0.66666,
            dec_aa: false,
            normalize_input: false,
        }
    }
}

impl VoiceFormat {
    pub(crate) fn read(checkpoint: &Checkpoint, sample_rate: Option<u32>) -> Result<Self> {
        let mut options = Self::default();
        let legacy = checkpoint
            .metadata("embedder_model")?
            .filter(|value| !value.is_null())
            .map(serde_json::from_value::<Embedder>)
            .transpose()
            .context("unsupported model embedder")?;
        if let Some(embedder) = legacy {
            options.embedder = embedder;
            options.normalize_input = true;
        }
        if let Some(meta) = checkpoint.metadata("nvc")?.filter(|value| !value.is_null()) {
            ensure!(
                meta["format"] == "nvc" && meta["format_version"] == 1,
                "unsupported NVC model metadata format or version"
            );
            if let Some(rate) = meta.get("sample_rate")
                && let Some(expected) = sample_rate
            {
                ensure!(
                    rate.as_u64() == Some(u64::from(expected)),
                    "NVC metadata and generator sample rates disagree"
                );
            }
            let embedder: Embedder = serde_json::from_value(meta["embedder"].clone())
                .context("unsupported NVC model embedder")?;
            ensure!(
                legacy.is_none_or(|declared| declared == embedder),
                "NVC and embedder_model metadata disagree"
            );
            options.embedder = embedder;
            options.f0_unvoiced = serde_json::from_value(meta["f0_unvoiced"].clone())
                .context("unsupported NVC unvoiced F0 mode")?;
            if let Some(scale) = meta.get("default_noise_scale") {
                options.default_noise_scale = scale.as_f64().context("invalid NVC noise scale")?;
            }
            ensure!(
                options.default_noise_scale.is_finite() && options.default_noise_scale >= 0.0,
                "invalid NVC noise scale"
            );
            if let Some(aa) = meta.get("dec_aa") {
                options.dec_aa = aa
                    .as_bool()
                    .context("invalid NVC decoder activation flag")?;
            }
            options.normalize_input = true;
        }
        Ok(options)
    }
}

#[derive(Clone)]
pub struct Paths {
    pub root: PathBuf,
    pub weights: PathBuf,
    pub indices: PathBuf,
    pub experiments: PathBuf,
    #[cfg(feature = "inference")]
    pub rmvpe: PathBuf,
}

impl Paths {
    #[cfg(feature = "inference")]
    pub(crate) fn nmvpe(&self) -> PathBuf {
        self.root
            .join(
                std::env::var_os("nmvpe_root")
                    .map(PathBuf::from)
                    .unwrap_or_else(|| PathBuf::from("assets/nmvpe")),
            )
            .join("nmvpe.pt")
    }

    pub fn new(root: &Path) -> Result<Self> {
        let root = std::path::absolute(root)?;
        let directory = |key, default: &str| {
            root.join(
                std::env::var_os(key)
                    .map(PathBuf::from)
                    .unwrap_or_else(|| PathBuf::from(default)),
            )
        };
        Ok(Self {
            weights: directory("weight_root", "assets/weights"),
            indices: directory("outside_index_root", "assets/indices"),
            experiments: directory("index_root", "logs"),
            #[cfg(feature = "inference")]
            rmvpe: directory("rmvpe_root", "assets/rmvpe"),
            root,
        })
    }

    pub fn model(&self, requested: &Path) -> Result<PathBuf> {
        let path = self.root.join(requested);
        let path = if path.is_file() {
            path
        } else {
            self.weights.join(requested)
        };
        ensure!(path.is_file(), "Model not found: {}", requested.display());
        ensure!(
            path.extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("pth")),
            "Model must be a .pth file: {}",
            path.display()
        );
        Ok(path)
    }

    pub fn models(&self) -> Result<Vec<String>> {
        if !self.weights.exists() {
            return Ok(Vec::new());
        }
        let mut found = Vec::new();
        for entry in std::fs::read_dir(&self.weights)? {
            let entry = entry?;
            if entry.path().is_file()
                && entry
                    .path()
                    .extension()
                    .is_some_and(|ext| ext.eq_ignore_ascii_case("pth"))
                && let Ok(name) = entry.file_name().into_string()
            {
                found.push(name);
            }
        }
        found.sort();
        Ok(found)
    }

    /// Every usable (`added`) index under `assets/indices` and the experiments, by path.
    pub fn indexes(&self) -> Vec<PathBuf> {
        let mut found: Vec<PathBuf> = [&self.indices, &self.experiments]
            .into_iter()
            .flat_map(|directory| walkdir::WalkDir::new(directory).follow_links(true))
            .filter_map(std::result::Result::ok)
            .map(walkdir::DirEntry::into_path)
            .filter(|path| is_added_index(path))
            .collect();
        found.sort();
        found.dedup();
        found
    }

    /// The index that belongs to `model`: one named after the model's experiment (standard
    /// `added_IVF…_<name>_v2`, `<name>_added_…` or just `<name>.index`), else the only index
    /// next to the model file itself (a voice folder on Drive, an unpacked zip).
    pub fn select_index(&self, model: &Path, speaker_id: Option<i64>) -> Option<PathBuf> {
        static EPOCH: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i)_e\d+_s\d+$").unwrap());
        let stem = model.file_stem()?.to_str()?;
        let experiment = normalize_name(&EPOCH.replace(stem, ""));
        if experiment.is_empty() {
            return None;
        }
        let mut candidates = Vec::new();
        for path in self.indexes() {
            let name = path.file_stem().and_then(OsStr::to_str).unwrap_or_default();
            let (owners, indexed) = index_owners(name);
            if indexed.is_some() && indexed != speaker_id {
                continue;
            }
            let Some(rank) = owners.iter().position(|owner| *owner == experiment) else {
                continue;
            };
            let order = usize::from(!path.starts_with(&self.indices));
            let modified = path
                .metadata()
                .and_then(|metadata| metadata.modified())
                .unwrap_or(SystemTime::UNIX_EPOCH);
            let score = (
                indexed != speaker_id,
                rank,
                order,
                std::cmp::Reverse(modified),
            );
            candidates.push((score, path));
        }
        if let Some((_, path)) = candidates.into_iter().min_by(|(a, _), (b, _)| a.cmp(b)) {
            return Some(path);
        }
        // A voice folder that holds exactly one index: it is that voice's index.
        let real = std::fs::canonicalize(model).ok()?;
        let siblings: Vec<PathBuf> = std::fs::read_dir(real.parent()?)
            .ok()?
            .filter_map(std::result::Result::ok)
            .map(|entry| entry.path())
            .filter(|path| is_added_index(path))
            .collect();
        match siblings.as_slice() {
            [only] => Some(only.clone()),
            _ => None,
        }
    }

    #[cfg(feature = "inference")]
    pub fn require_index(
        &self,
        model: &Path,
        speaker_id: i64,
        requested: Option<&Path>,
        rate: f32,
    ) -> Result<Option<PathBuf>> {
        if rate == 0.0 {
            return Ok(None);
        }
        let Some(path) = requested else {
            // No index chosen: use the voice's own one, or convert without an index.
            let index = self.select_index(model, Some(speaker_id));
            if index.is_none() {
                eprintln!(
                    "No index found for {}: converting without one",
                    model.file_name().map_or_else(
                        || model.display().to_string(),
                        |name| name.to_string_lossy().into_owned()
                    )
                );
            }
            return Ok(index);
        };
        let path = self.root.join(path);
        let index = path
            .file_name()
            .and_then(OsStr::to_str)
            .map(|name| path.with_file_name(added_name(name)));
        Ok(Some(index.filter(|path| path.is_file()).with_context(
            || format!("Index not found: {}", path.display()),
        )?))
    }
}

fn is_added_index(path: &Path) -> bool {
    path.is_file()
        && path
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("index"))
        && !path
            .file_stem()
            .and_then(OsStr::to_str)
            .is_some_and(|name| TRAINED.is_match(name))
}

/// The training-only half of an index pair (`trained_IVF…`), not "Pretrained" in a voice name.
static TRAINED: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i)(^|_)trained_ivf").unwrap());

/// The usable `added_IVF…` twin of a `trained_IVF…` index name; other names stay.
pub fn added_name(name: &str) -> String {
    TRAINED.replace(name, "${1}added_IVF").into_owned()
}

/// Lowercase, with spaces, dots and dashes read as underscores.
fn normalize_name(name: &str) -> String {
    name.to_lowercase()
        .split(|c: char| c == '_' || c == '-' || c == '.' || c.is_whitespace())
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join("_")
}

/// Voice names an index file name can stand for, most specific first, and its speaker.
fn index_owners(stem: &str) -> (Vec<String>, Option<i64>) {
    static SPEAKER: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"_spkid(\d+)$").unwrap());
    static STANDARD: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"added_ivf\d+_flat_nprobe_\d+_(.+?)_v[12]$").unwrap());
    static PREFIX: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^(.+?)_added_").unwrap());
    static PLAIN: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(_v[12])?(_index)?$").unwrap());
    let mut name = normalize_name(stem);
    let speaker = SPEAKER
        .captures(&name)
        .and_then(|capture| capture[1].parse().ok());
    name = SPEAKER.replace(&name, "").into_owned();
    let mut owners = Vec::new();
    if let Some(capture) = STANDARD.captures(&name) {
        owners.push(capture[1].to_owned());
    }
    if let Some(capture) = PREFIX.captures(&name) {
        owners.push(capture[1].to_owned());
    }
    if !name.contains("added_ivf") {
        owners.push(PLAIN.replace(&name, "").into_owned());
    }
    (owners, speaker)
}

#[derive(Debug, Serialize)]
pub struct ModelInfo {
    pub speaker_count: i64,
    pub speakers: BTreeMap<i64, String>,
    pub version: String,
    pub uses_f0: bool,
    pub sample_rate: Option<u32>,
    pub info: Option<String>,
    #[serde(flatten)]
    pub format: VoiceFormat,
}

impl ModelInfo {
    pub fn read(path: &Path) -> Result<Self> {
        Self::from_checkpoint(&Checkpoint::open(path)?)
    }

    pub fn from_checkpoint(checkpoint: &Checkpoint) -> Result<Self> {
        let speaker_count = *checkpoint
            .tensor_shape(Some("weight"), "emb_g.weight")?
            .first()
            .context("Model does not contain a speaker embedding")?;
        ensure!(
            speaker_count > 0,
            "Model does not contain any speaker embeddings"
        );
        let mut speakers = BTreeMap::new();
        if let Some(Value::Array(entries)) = checkpoint.metadata("speaker_info")? {
            for entry in entries {
                if let (Some(id), Some(name)) = (
                    entry.get("id").and_then(integer),
                    entry.get("name").map(python_string),
                ) && (0..speaker_count).contains(&id)
                    && !name.is_empty()
                {
                    speakers.entry(id).or_insert(name);
                }
            }
        }
        let version = checkpoint
            .metadata("version")?
            .and_then(|v| v.as_str().map(str::to_owned))
            .unwrap_or_else(|| "v1".to_owned());
        let uses_f0 = checkpoint
            .metadata("f0")?
            .as_ref()
            .and_then(integer)
            .unwrap_or(1)
            == 1;
        let sample_rate = checkpoint.metadata("config")?.and_then(|v| {
            v.as_array()
                .and_then(|array| array.last().and_then(sample_rate))
        });
        let info = checkpoint.metadata("info")?.as_ref().map(python_string);
        let format = VoiceFormat::read(checkpoint, sample_rate)?;
        format.embedder.supports(&version)?;
        Ok(Self {
            speaker_count,
            speakers,
            version,
            uses_f0,
            sample_rate,
            info,
            format,
        })
    }

    pub fn speaker(&self, requested: Option<i64>) -> Result<i64> {
        if self.speakers.is_empty() {
            let speaker = requested.unwrap_or(0);
            ensure!(
                (0..self.speaker_count).contains(&speaker),
                "Speaker ID must be between 0 and {} for this model",
                self.speaker_count - 1
            );
            return Ok(speaker);
        }
        let speaker = requested
            .or_else(|| self.speakers.keys().next().copied())
            .context("model has no speakers")?;
        if !self.speakers.contains_key(&speaker) {
            let available = self
                .speakers
                .keys()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ");
            bail!(
                "Speaker ID {speaker} is not declared by this multi-speaker model; available IDs: {available}"
            );
        }
        Ok(speaker)
    }
}

fn integer(value: &Value) -> Option<i64> {
    value.as_i64().or_else(|| match value {
        Value::Bool(value) => Some(i64::from(*value)),
        Value::Number(number) => number
            .as_f64()
            .filter(|value| value.abs() < i64::MAX as f64)
            .map(|value| value as i64),
        Value::String(text) => text.trim().parse().ok(),
        _ => None,
    })
}

fn python_string(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        Value::Null => "None".to_owned(),
        Value::Bool(v) => if *v { "True" } else { "False" }.to_owned(),
        _ => value.to_string(),
    }
}

pub fn sample_rate(value: &Value) -> Option<u32> {
    match value.as_str() {
        Some("32k") => Some(32000),
        Some("40k") => Some(40000),
        Some("48k") => Some(48000),
        _ => integer(value).and_then(|value| value.try_into().ok()),
    }
}

#[cfg(test)]
mod index_name_tests {
    use super::{index_owners, normalize_name};

    fn owns(index: &str, voice: &str) -> bool {
        index_owners(index).0.contains(&normalize_name(voice))
    }

    #[test]
    fn index_names_match_their_voice_only() {
        assert!(owns("added_IVF1371_Flat_nprobe_1_Lover_ru_v2", "Lover_ru"));
        assert!(owns(
            "Lover_ru_added_IVF1371_Flat_nprobe_1_Lover_ru_v2",
            "Lover_ru"
        ));
        assert!(owns("My Voice", "my-voice"));
        assert!(owns("MyVoice_v2", "MyVoice"));
        assert!(owns(
            "added_IVF908_Flat_nprobe_1_prince_v2_spkid3",
            "prince"
        ));
        assert_eq!(
            index_owners("added_IVF908_Flat_nprobe_1_prince_v2_spkid3").1,
            Some(3)
        );
        // A longer name that starts with the voice is another voice.
        assert!(!owns(
            "Lovern_scream_added_IVF1598_Flat_nprobe_1_Lovern_scream_v2",
            "Lover"
        ));
        assert!(!owns("added_IVF1371_Flat_nprobe_1_Lover_ru_v2", "Lover"));
        // "Pretrained" in a voice name is not a trained_IVF index.
        assert!(owns(
            "Platina_NVC_Pretrained_added_IVF651_Flat_nprobe_1_Platina_NVC_Pretrained_v2",
            "Platina_NVC_Pretrained"
        ));
    }

    #[test]
    fn trained_indexes_map_to_added_ones() {
        use super::added_name;
        assert_eq!(
            added_name("trained_IVF651_Flat_nprobe_1_Platina_NVC_Pretrained_v2.index"),
            "added_IVF651_Flat_nprobe_1_Platina_NVC_Pretrained_v2.index"
        );
        assert_eq!(
            added_name("Platina_NVC_Pretrained_added_IVF651_Flat_nprobe_1_x_v2.index"),
            "Platina_NVC_Pretrained_added_IVF651_Flat_nprobe_1_x_v2.index"
        );
        assert_eq!(
            added_name("voice_trained_IVF8_Flat.index"),
            "voice_added_IVF8_Flat.index"
        );
    }
}
