use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::sync::LazyLock;

use anyhow::{Context, Result, ensure};
use regex::Regex;
use serde::{Deserialize, Deserializer, Serialize};
use sha1::{Digest, Sha1};
use walkdir::WalkDir;

use crate::audio;

pub(super) const MANIFEST_FILENAME: &str = "multispeaker_manifest.json";
const MAX_SPEAKER_ID: u32 = 109;
const VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ManifestSource {
    #[default]
    FolderScan,
    Helper,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Speaker {
    pub id: u32,
    pub name: String,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ManifestEntry {
    pub path: PathBuf,
    pub speaker_name: String,
    pub speaker_id: u32,
    pub repeat: u64,
    pub output_key: String,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ManifestRow {
    pub path: PathBuf,
    pub speaker_name: String,
    pub speaker_id: u32,
    pub repeat: u64,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum RowNumber {
    Text(String),
    Number(serde_json::Number),
}

impl RowNumber {
    fn text(&self) -> std::borrow::Cow<'_, str> {
        match self {
            Self::Text(value) => value.trim().into(),
            Self::Number(value) => value.to_string().into(),
        }
    }
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct ManifestRowInput {
    #[serde(deserialize_with = "nullable_text")]
    pub path: String,
    #[serde(deserialize_with = "nullable_text")]
    pub speaker_name: String,
    pub speaker_id: Option<RowNumber>,
    pub repeat: Option<RowNumber>,
}

fn nullable_text<'de, D: Deserializer<'de>>(input: D) -> std::result::Result<String, D::Error> {
    Option::<String>::deserialize(input).map(Option::unwrap_or_default)
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Manifest {
    #[serde(default = "manifest_version")]
    pub version: u32,
    #[serde(default)]
    pub source: ManifestSource,
    #[serde(default)]
    pub root: PathBuf,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rows: Option<Vec<ManifestRow>>,
    #[serde(default)]
    pub speakers: Vec<Speaker>,
    pub entries: Vec<ManifestEntry>,
}

#[derive(Debug, Serialize)]
pub struct ManifestRows {
    pub manifest: Manifest,
    pub invalid_rows: Vec<usize>,
}

fn manifest_version() -> u32 {
    VERSION
}

pub fn scan_speakers(root: &Path) -> Result<Manifest> {
    static FOLDER: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"^(.+)_(\d+)_(\d+)$").expect("valid folder pattern"));
    let root = absolute(root)?;
    ensure!(
        root.is_dir(),
        "speaker dataset folder does not exist ({})",
        root.display()
    );
    let mut folders = file_paths(&root)?;
    folders.retain(|path| path.is_dir());
    ensure!(
        !folders.is_empty(),
        "speaker dataset folder has no direct subfolders"
    );

    let mut entries = Vec::new();
    let mut names = BTreeMap::new();
    let mut invalid = Vec::new();
    for folder in folders {
        let filename = filename(&folder)?;
        let spec = FOLDER.captures(filename).and_then(|captures| {
            let speaker_name = captures.get(1)?.as_str().trim();
            let id = u32::try_from(decimal_integer(captures.get(2)?.as_str())?).ok()?;
            let repeat = decimal_integer(captures.get(3)?.as_str())?;
            Some((speaker_name, id, repeat))
        });
        let Some((speaker_name, id, repeat)) = spec else {
            invalid.push(filename.to_owned());
            continue;
        };
        let audio = speaker_files(&folder)?;
        if !valid_speaker(&names, speaker_name, id, repeat) || audio.is_empty() {
            invalid.push(filename.to_owned());
            continue;
        }
        names.insert(id, speaker_name.to_owned());
        append_entries(&mut entries, audio, speaker_name, id, repeat)?;
    }
    ensure!(
        invalid.is_empty(),
        "invalid speaker folders (expected name_ID_repeat, ID 0–109, positive repeat, matching names for each ID, and audio files): {}",
        invalid.join(", ")
    );
    ensure!(!entries.is_empty(), "speaker dataset has no audio files");
    Ok(Manifest {
        version: VERSION,
        source: ManifestSource::FolderScan,
        root,
        rows: None,
        speakers: speakers(names),
        entries,
    })
}

pub fn manifest_from_rows(rows: &[ManifestRowInput], root: Option<&Path>) -> Result<ManifestRows> {
    let mut entries = Vec::new();
    let mut names = BTreeMap::new();
    let mut invalid_rows = Vec::new();
    let mut valid_rows = Vec::new();
    for (index, row) in rows.iter().enumerate() {
        let folder = row.path.trim();
        let name = row.speaker_name.trim();
        let id = row
            .speaker_id
            .as_ref()
            .map(RowNumber::text)
            .unwrap_or_default();
        let repeat = row.repeat.as_ref().map(RowNumber::text).unwrap_or_default();
        if [folder, name, &id, &repeat]
            .iter()
            .all(|field| field.is_empty())
        {
            continue;
        }
        let parsed = row_integer(&id)
            .and_then(|value| u32::try_from(value).ok())
            .zip(row_integer(&repeat));
        let Some((id, repeat)) =
            parsed.filter(|&(id, repeat)| valid_speaker(&names, name, id, repeat))
        else {
            invalid_rows.push(index + 1);
            continue;
        };
        let path = absolute(Path::new(folder))?;
        let audio = speaker_files(&path)?;
        if audio.is_empty() {
            invalid_rows.push(index + 1);
            continue;
        }
        names.insert(id, name.to_owned());
        valid_rows.push(ManifestRow {
            path,
            speaker_name: name.to_owned(),
            speaker_id: id,
            repeat,
        });
        append_entries(&mut entries, audio, name, id, repeat)?;
    }
    ensure!(!entries.is_empty(), "no valid multi-speaker dataset rows");
    Ok(ManifestRows {
        manifest: Manifest {
            version: VERSION,
            source: ManifestSource::Helper,
            root: root.map(absolute).transpose()?.unwrap_or_default(),
            rows: Some(valid_rows),
            speakers: speakers(names),
            entries,
        },
        invalid_rows,
    })
}

pub fn save_manifest(experiment_dir: &Path, manifest: &Manifest) -> Result<PathBuf> {
    let path = experiment_dir.join(MANIFEST_FILENAME);
    let mut bytes = serde_json::to_vec_pretty(manifest)?;
    bytes.push(b'\n');
    audio::write_file(&path, &bytes, true)?;
    Ok(path)
}

pub fn load_manifest(experiment_dir: &Path) -> Result<Manifest> {
    let path = experiment_dir.join(MANIFEST_FILENAME);
    let mut manifest: Manifest = serde_json::from_reader(
        fs::File::open(&path).with_context(|| format!("opening {}", path.display()))?,
    )
    .with_context(|| format!("reading {}", path.display()))?;
    ensure!(
        !manifest.entries.is_empty(),
        "speaker manifest has no audio files"
    );
    let mut names = BTreeMap::new();
    let mut output_keys = BTreeSet::new();
    for entry in &mut manifest.entries {
        entry.path = absolute(&entry.path)?;
        entry.speaker_name = entry.speaker_name.trim().to_owned();
        ensure!(
            entry.path.is_file()
                && valid_speaker(&names, &entry.speaker_name, entry.speaker_id, entry.repeat)
                && !entry.output_key.is_empty()
                && !entry
                    .output_key
                    .contains(['/', '\\', ':', '\0', '\n', '\r']),
            "invalid speaker manifest entry ({})",
            entry.path.display()
        );
        ensure!(
            output_keys.insert(&entry.output_key),
            "duplicate speaker manifest key ({})",
            entry.output_key
        );
        names.insert(entry.speaker_id, entry.speaker_name.clone());
    }
    manifest.speakers = speakers(names);
    Ok(manifest)
}

fn speakers(names: BTreeMap<u32, String>) -> Vec<Speaker> {
    names
        .into_iter()
        .map(|(id, name)| Speaker { id, name })
        .collect()
}

fn valid_speaker(names: &BTreeMap<u32, String>, name: &str, id: u32, repeat: u64) -> bool {
    !name.is_empty()
        && !name.contains(['|', '\n', '\r'])
        && id <= MAX_SPEAKER_ID
        && repeat > 0
        && names.get(&id).is_none_or(|existing| existing == name)
}

fn row_integer(text: &str) -> Option<u64> {
    let digits = text.strip_suffix(".0").unwrap_or(text);
    let number: u64 = digits.parse().ok()?;
    (number.to_string() == digits).then_some(number)
}

fn decimal_integer(text: &str) -> Option<u64> {
    static DECIMAL: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"^\p{Nd}$").expect("valid decimal pattern"));
    fn is_decimal(ch: char) -> bool {
        DECIMAL.is_match(ch.encode_utf8(&mut [0; 4]))
    }
    text.chars().try_fold(0u64, |number, digit| {
        let value = if let Some(ascii) = digit.to_digit(10) {
            ascii
        } else {
            // Unicode decimal digits occur in aligned runs of ten.
            let point = u32::from(digit);
            let mut start = point;
            while let Some(previous) = (start - 1).try_into().ok().filter(|&ch| is_decimal(ch)) {
                start = u32::from(previous);
            }
            (point - start) % 10
        };
        number.checked_mul(10)?.checked_add(u64::from(value))
    })
}

fn append_entries(
    entries: &mut Vec<ManifestEntry>,
    paths: Vec<PathBuf>,
    name: &str,
    id: u32,
    repeat: u64,
) -> Result<()> {
    for path in paths {
        let path_string = path
            .to_str()
            .with_context(|| format!("dataset path is not UTF-8 ({})", path.display()))?;
        #[cfg(windows)]
        let path_string = path_string.replace('/', "\\").to_lowercase();
        let hash = format!("{:x}", Sha1::digest(path_string.as_bytes()));
        let index = entries.len();
        entries.push(ManifestEntry {
            path,
            speaker_name: name.to_owned(),
            speaker_id: id,
            repeat,
            output_key: format!("ms{index:04}_s{id:03}_{}", &hash[..10]),
        });
    }
    Ok(())
}

pub(super) fn file_paths(directory: &Path) -> Result<Vec<PathBuf>> {
    let mut paths = fs::read_dir(directory)
        .with_context(|| format!("reading dataset {}", directory.display()))?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<std::io::Result<Vec<_>>>()?;
    paths.sort();
    Ok(paths)
}

fn speaker_files(directory: &Path) -> Result<Vec<PathBuf>> {
    if !directory.is_dir() {
        return Ok(Vec::new());
    }
    // os.walk lists all files in a directory before walking the sorted children.
    let walker = WalkDir::new(directory).min_depth(1).sort_by(|a, b| {
        a.file_type()
            .is_dir()
            .cmp(&b.file_type().is_dir())
            .then_with(|| a.file_name().cmp(b.file_name()))
    });
    let mut audio_paths = Vec::new();
    for entry in walker {
        let path = entry?.into_path();
        if path.is_file() && audio::is_audio(&path) {
            audio_paths.push(path);
        }
    }
    Ok(audio_paths)
}

fn filename(path: &Path) -> Result<&str> {
    path.file_name()
        .and_then(|name| name.to_str())
        .with_context(|| format!("dataset folder name is not UTF-8 ({})", path.display()))
}

fn absolute(path: &Path) -> Result<PathBuf> {
    let absolute = if path.as_os_str().is_empty() {
        std::env::current_dir()?
    } else {
        std::path::absolute(path)?
    };
    let mut normalized = PathBuf::new();
    for component in absolute.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                normalized.pop();
            }
            _ => normalized.push(component),
        }
    }
    Ok(normalized)
}
