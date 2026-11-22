use std::collections::{HashMap, VecDeque};
use std::fs::{self, File, OpenOptions};
use std::io;
use std::path::{Component, Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, bail, ensure};
use axum::body::Body;
use axum::extract::{Multipart, Path as RoutePath, Query, Request, State};
use axum::http::{HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tempfile::{NamedTempFile, TempDir};
use tokio::io::AsyncWriteExt;
use tower_http::services::ServeFile;
use uuid::Uuid;
use zip::ZipArchive;

use super::super::{Shared, WebResult, lock};
use crate::audio::Format;

pub(super) const UPLOAD_LIMIT: usize = 512 * 1024 * 1024;
const ARCHIVE_LIMIT: u64 = 4 * 1024 * 1024 * 1024;
const MAX_ARCHIVE_ENTRIES: usize = 100_000;
const MAX_RESULT_SETS: usize = 8;

pub(in crate::web) struct Files {
    pub(super) workspace: TempDir,
    results: Mutex<VecDeque<HashMap<Uuid, Artifact>>>,
}

impl Files {
    pub(in crate::web) fn new() -> Result<Self> {
        Ok(Self {
            workspace: tempfile::Builder::new().prefix("rvc-studio-").tempdir()?,
            results: Mutex::default(),
        })
    }

    pub(super) fn publish(
        &self,
        paths: impl IntoIterator<Item = PathBuf>,
        staging: Option<TempDir>,
    ) -> Result<Vec<Media>> {
        let staging = staging.map(Arc::new);
        let mut entries = HashMap::new();
        let mut media = Vec::new();
        for path in paths {
            let id = Uuid::new_v4();
            let name = path
                .file_name()
                .context("output has no filename")?
                .to_string_lossy()
                .into_owned();
            entries.insert(
                id,
                Artifact {
                    path: path.canonicalize()?,
                    name: name.clone(),
                    _staging: staging.clone(),
                },
            );
            media.push(Media {
                url: format!("/api/studio/media/{id}"),
                name,
            });
        }
        let mut results = lock(&self.results)?;
        results.push_back(entries);
        if results.len() > MAX_RESULT_SETS {
            results.pop_front();
        }
        Ok(media)
    }
}

#[derive(Clone)]
struct Artifact {
    path: PathBuf,
    name: String,
    _staging: Option<Arc<TempDir>>,
}

#[derive(Serialize)]
pub(super) struct Media {
    pub(super) url: String,
    pub(super) name: String,
}

#[derive(Deserialize)]
pub(super) struct MediaQuery {
    format: Option<Format>,
    #[serde(default)]
    download: bool,
}

pub(super) async fn media(
    State(state): State<Shared>,
    RoutePath(id): RoutePath<Uuid>,
    Query(query): Query<MediaQuery>,
    request: Request,
) -> WebResult<Response> {
    let artifact = lock(&state.studio.results)?
        .iter()
        .rev()
        .find_map(|entries| entries.get(&id))
        .cloned();
    let Some(artifact) = artifact else {
        return Ok(StatusCode::NOT_FOUND.into_response());
    };
    let original = Format::from_extension(artifact.path.extension().unwrap_or_default())?;
    let target = query.format.unwrap_or(original);
    let encoded = if target.extension() == original.extension() {
        None
    } else {
        let input = artifact.clone();
        Some(
            tokio::task::spawn_blocking(move || transcode(&input.path, target))
                .await
                .map_err(anyhow::Error::from)??,
        )
    };
    let path = encoded
        .as_ref()
        .map_or(artifact.path.as_path(), |file| file.path());
    let mime = match target {
        Format::Wav => "audio/wav",
        Format::Mp3 => "audio/mpeg",
        Format::Flac => "audio/flac",
        Format::M4a => "audio/mp4",
    }
    .parse()
    .expect("static audio MIME");
    let mut response = ServeFile::new_with_mime(path, &mime)
        .try_call(request)
        .await
        .map_err(anyhow::Error::from)?
        .map(Body::new);
    response.headers_mut().insert(
        header::CACHE_CONTROL,
        HeaderValue::from_static("private, no-store"),
    );
    if query.download {
        let name = Path::new(&artifact.name).with_extension(target.extension());
        let encoded = name
            .to_string_lossy()
            .bytes()
            .map(|byte| {
                if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_') {
                    char::from(byte).to_string()
                } else {
                    format!("%{byte:02X}")
                }
            })
            .collect::<String>();
        let value = format!("attachment; filename*=UTF-8''{encoded}");
        response.headers_mut().insert(
            header::CONTENT_DISPOSITION,
            value.parse().map_err(anyhow::Error::from)?,
        );
    }
    // A browser can still be reading when an old result is evicted.
    // ServeFile has opened its own handle. Temporary transcoded files stay alive on Windows
    // through NamedTempFile's delete-sharing handle.
    Ok(response)
}

fn transcode(input: &Path, format: Format) -> Result<NamedTempFile> {
    let output = NamedTempFile::new_in(input.parent().context("audio has no directory")?)
        .or_else(|_| NamedTempFile::new())?;
    let muxer = if matches!(format, Format::M4a) {
        "ipod"
    } else {
        format.extension()
    };
    let result = Command::new("ffmpeg")
        .args([
            "-nostdin",
            "-loglevel",
            "error",
            "-y",
            "-protocol_whitelist",
            "file,pipe,crypto,data",
            "-i",
        ])
        .arg(input)
        .args(["-map", "0:a:0", "-f", muxer])
        .arg(output.path())
        .output()
        .context("starting FFmpeg to export audio")?;
    ensure!(
        result.status.success(),
        "encoding audio: {}",
        String::from_utf8_lossy(&result.stderr).trim()
    );
    Ok(output)
}

pub(super) struct Uploads {
    pub(super) directory: TempDir,
    pub(super) files: Vec<PathBuf>,
}

pub(super) struct DatasetPaths {
    pub(super) single: PathBuf,
    pub(super) multi_folder: PathBuf,
}

impl Uploads {
    pub(super) fn one(&self) -> Result<&Path> {
        ensure!(self.files.len() == 1, "upload one file");
        Ok(&self.files[0])
    }

    pub(super) fn into_dataset(self, workspace: &Path) -> Result<DatasetPaths> {
        ensure!(
            !self.files.is_empty(),
            "upload a dataset ZIP or audio files"
        );
        if let [archive] = self.files.as_slice()
            && archive
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("zip"))
        {
            let extracted = TempDir::new_in(workspace)?;
            let audio = unzip(archive, extracted.path(), crate::audio::is_audio, || Ok(()))?;
            ensure!(
                !audio.is_empty(),
                "the dataset archive contains no audio files"
            );
            let mut entries = fs::read_dir(extracted.path())?;
            let first = entries.next().transpose()?.context("empty dataset")?.path();
            let wrapper = entries.next().is_none() && first.is_dir();
            let has_speaker_folder =
                wrapper && crate::dataset::scan_speakers(extracted.path()).is_ok();
            let root = extracted.keep();
            let single = if wrapper { first } else { root.clone() };
            let multi_folder = if has_speaker_folder {
                root
            } else {
                single.clone()
            };
            return Ok(DatasetPaths {
                single,
                multi_folder,
            });
        }
        ensure!(
            self.files.iter().all(|path| crate::audio::is_audio(path)),
            "choose one ZIP archive or a selection of audio files"
        );
        let root = TempDir::new_in(workspace)?;
        let single = root.path().join("speaker_0_1");
        fs::rename(self.directory.path(), &single)?;
        Ok(DatasetPaths {
            single,
            multi_folder: root.keep(),
        })
    }
}

pub(super) async fn upload(
    state: &Shared,
    mut form: Multipart,
    dataset: bool,
) -> Result<(Value, Uploads)> {
    let directory = TempDir::new_in(state.studio.workspace.path())?;
    let mut files = Vec::new();
    let mut settings = Value::Null;
    while let Some(mut field) = form.next_field().await? {
        match field.name() {
            Some("settings") if settings.is_null() => {
                let mut text = Vec::new();
                while let Some(chunk) = field.chunk().await? {
                    ensure!(
                        text.len() + chunk.len() <= 64 * 1024,
                        "job settings are too large"
                    );
                    text.extend_from_slice(&chunk);
                }
                settings = serde_json::from_slice(&text).context("invalid job settings")?;
            }
            Some("files") => {
                ensure!(files.len() < 1000, "too many uploaded files");
                let name = field.file_name().context("upload has no filename")?;
                let path = upload_path(directory.path(), name, dataset, files.len())?;
                let mut file = tokio::fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(&path)
                    .await
                    .with_context(|| format!("saving upload {name}"))?;
                while let Some(chunk) = field.chunk().await? {
                    file.write_all(&chunk).await?;
                }
                file.flush().await?;
                files.push(path);
            }
            _ => bail!("unexpected multipart field"),
        }
    }
    Ok((settings, Uploads { directory, files }))
}

fn upload_path(directory: &Path, name: &str, dataset: bool, count: usize) -> Result<PathBuf> {
    let name = filename(name)?;
    let mut target = directory.join(name);
    if dataset && crate::audio::is_audio(name) {
        let extension = name.extension().context("audio file has no extension")?;
        let mut number = count + 1;
        while target.exists() {
            target = directory
                .join(format!("upload-{number}"))
                .with_extension(extension);
            number += 1;
        }
    }
    Ok(target)
}

pub(super) fn filename(name: &str) -> Result<&Path> {
    let path = Path::new(name);
    let mut parts = path.components();
    ensure!(
        !name.contains(['/', '\\', ':'])
            && !name.chars().any(char::is_control)
            && matches!(parts.next(), Some(Component::Normal(_)))
            && parts.next().is_none(),
        "invalid filename {name:?}"
    );
    Ok(path)
}

pub(super) fn unzip(
    archive: &Path,
    output: &Path,
    keep: impl Fn(&Path) -> bool,
    mut check: impl FnMut() -> Result<()>,
) -> Result<Vec<PathBuf>> {
    let mut zip = ZipArchive::new(File::open(archive)?).context("opening ZIP archive")?;
    ensure!(zip.len() <= MAX_ARCHIVE_ENTRIES, "too many ZIP entries");
    let mut written = 0;
    let mut extracted = Vec::new();
    for i in 0..zip.len() {
        check()?;
        let mut member = zip.by_index(i)?;
        ensure!(!member.is_symlink(), "ZIP contains a symbolic link");
        let normalized = member.name().replace('\\', "/");
        let mut relative = PathBuf::new();
        for part in normalized.trim_end_matches('/').split('/') {
            relative.push(filename(part).context("unsafe ZIP path")?);
        }
        if member.is_dir() || relative.starts_with("__MACOSX") || !keep(&relative) {
            continue;
        }
        let target = output.join(relative);
        fs::create_dir_all(target.parent().context("ZIP entry has no parent")?)?;
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&target)
            .with_context(|| format!("extracting {}", member.name()))?;
        let expected = member.size();
        ensure!(
            expected <= ARCHIVE_LIMIT - written,
            "expanded ZIP exceeds 4 GiB"
        );
        let mut bounded = io::Read::take(&mut member, (ARCHIVE_LIMIT - written) + 1);
        let size = io::copy(&mut bounded, &mut file)?;
        ensure!(size == expected, "invalid ZIP member size");
        written += size;
        extracted.push(target);
    }
    Ok(extracted)
}

#[cfg(test)]
mod dataset_tests {
    use super::*;
    use std::io::Write;
    use zip::{ZipWriter, write::SimpleFileOptions};

    #[test]
    fn loose_audio_keeps_every_file_in_the_preprocess_folder() -> Result<()> {
        let workspace = TempDir::new()?;
        let directory = TempDir::new_in(workspace.path())?;
        let mut files = Vec::new();
        for (name, contents) in [
            ("vocal.wav", "one"),
            ("upload-3.wav", "two"),
            ("vocal.wav", "three"),
        ] {
            let path = upload_path(directory.path(), name, true, files.len())?;
            fs::write(&path, contents)?;
            files.push(path);
        }
        let paths = Uploads { directory, files }.into_dataset(workspace.path())?;
        let manifest = crate::dataset::scan_speakers(&paths.multi_folder)?;
        assert_eq!(manifest.entries.len(), 3);
        assert!(manifest.entries.iter().all(|entry| entry.speaker_id == 0
            && entry.speaker_name == "speaker"
            && entry.path.parent() == Some(&paths.single)));
        let mut sources = fs::read_dir(paths.single)?
            .map(|entry| fs::read_to_string(entry?.path()))
            .collect::<io::Result<Vec<_>>>()?;
        sources.sort();
        assert_eq!(sources, ["one", "three", "two"]);
        Ok(())
    }

    #[test]
    fn dataset_zip_preserves_single_speaker_subfolder_in_multi_mode() -> Result<()> {
        let workspace = TempDir::new()?;
        for prefix in ["", "dataset/"] {
            let directory = TempDir::new_in(workspace.path())?;
            let archive = directory.path().join("voices.ZIP");
            let mut zip = ZipWriter::new(File::create(&archive)?);
            zip.start_file(
                format!("{prefix}Lover_0_1/vocal.wav"),
                SimpleFileOptions::default(),
            )?;
            zip.write_all(b"test")?;
            zip.finish()?;
            let uploads = Uploads {
                directory,
                files: vec![archive],
            };
            let root = uploads.into_dataset(workspace.path())?.multi_folder;
            assert_eq!(crate::dataset::scan_speakers(&root)?.entries.len(), 1);
            assert_eq!(fs::read(root.join("Lover_0_1/vocal.wav"))?, b"test");
        }
        Ok(())
    }
}
