use std::collections::BTreeMap;
use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;
use std::time::SystemTime;

use anyhow::{Context, Result, bail, ensure};
use regex::Regex;
use serde::Serialize;
use serde_json::Value;

use crate::checkpoint::Checkpoint;

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

    pub fn select_index(&self, model: &Path, speaker_id: Option<i64>) -> Option<PathBuf> {
        static EPOCH: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i)_e\d+_s\d+$").unwrap());
        static SPEAKER: LazyLock<Regex> =
            LazyLock::new(|| Regex::new(r"(?i)_spkid(\d+)$").unwrap());
        let stem = model.file_stem()?.to_str()?.to_lowercase();
        let experiment = EPOCH.replace(&stem, "");
        if experiment.is_empty() {
            return None;
        }
        let prefix = format!("{experiment}_added_");
        let v1 = format!("_{experiment}_v1");
        let v2 = format!("_{experiment}_v2");
        let mut candidates = Vec::new();
        for (order, directory) in [&self.indices, &self.experiments].into_iter().enumerate() {
            for file in walkdir::WalkDir::new(directory)
                .into_iter()
                .filter_map(std::result::Result::ok)
            {
                if !file.path().is_file()
                    || file
                        .path()
                        .extension()
                        .is_none_or(|ext| !ext.eq_ignore_ascii_case("index"))
                {
                    continue;
                }
                let name = file
                    .path()
                    .file_stem()
                    .and_then(OsStr::to_str)
                    .unwrap_or_default()
                    .to_lowercase();
                if name.contains("trained") {
                    continue;
                }
                let indexed = SPEAKER
                    .captures(&name)
                    .and_then(|capture| capture[1].parse::<i64>().ok());
                if indexed.is_some() && indexed != speaker_id {
                    continue;
                }
                let standard =
                    name.starts_with(&prefix) || name.contains(&v1) || name.contains(&v2);
                if standard || name.contains(&stem) {
                    let modified = file
                        .metadata()
                        .ok()
                        .and_then(|metadata| metadata.modified().ok())
                        .unwrap_or(SystemTime::UNIX_EPOCH);
                    let score = (
                        indexed != speaker_id,
                        !standard,
                        order,
                        std::cmp::Reverse(modified),
                        file.path().to_string_lossy().to_lowercase(),
                    );
                    candidates.push((score, file.into_path()));
                }
            }
        }
        candidates
            .into_iter()
            .min_by(|(a, _), (b, _)| a.cmp(b))
            .map(|(_, path)| path)
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
        let index = if let Some(path) = requested {
            let path = self.root.join(path);
            path.file_name()
                .and_then(OsStr::to_str)
                .map(|name| path.with_file_name(name.replace("trained", "added")))
        } else {
            self.select_index(model, Some(speaker_id))
        };
        Ok(Some(index.filter(|path| path.is_file()).with_context(|| format!(
            "No usable added index for speaker ID {speaker_id}. Supply --index or use --index-rate 0."
        ))?))
    }
}

#[derive(Debug, Serialize)]
pub struct ModelInfo {
    pub speaker_count: i64,
    pub speakers: BTreeMap<i64, String>,
    pub version: String,
    pub uses_f0: bool,
    pub sample_rate: Option<u32>,
    pub info: Option<String>,
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
        Ok(Self {
            speaker_count,
            speakers,
            version,
            uses_f0,
            sample_rate,
            info,
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
