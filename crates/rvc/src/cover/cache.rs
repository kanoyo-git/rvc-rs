//! Per-song render cache. Only the complete archive is published. Restored tracks are
//! copied into the output folder so editing a result cannot alter the cached take.
use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use sha2::{Digest, Sha256};
use tempfile::NamedTempFile;
use zip::{CompressionMethod, ZipArchive, ZipWriter, write::SimpleFileOptions};

use super::{Request, separation};
use crate::catalog::{ModelInfo, Paths};
use crate::model::PitchMethod;

pub(super) type Tracks = Vec<(&'static str, PathBuf)>;

fn hash_file(path: &Path) -> Result<[u8; 32]> {
    let mut file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut hash = Sha256::new();
    io::copy(&mut file, &mut hash)?;
    Ok(hash.finalize().into())
}

fn ingredient(hash: &mut Sha256, path: Option<&Path>) -> Result<()> {
    if let Some(path) = path {
        hash.update([1]);
        hash.update(hash_file(path)?);
    } else {
        hash.update([0]);
    }
    Ok(())
}

#[derive(Clone, PartialEq, Eq)]
struct FileStamp {
    size: u64,
    modified: std::time::SystemTime,
    #[cfg(unix)]
    identity: (u64, u64, i64, i64),
}

fn stamp(path: &Path) -> Result<FileStamp> {
    let file = fs::metadata(path)?;
    #[cfg(unix)]
    use std::os::unix::fs::MetadataExt;
    Ok(FileStamp {
        size: file.len(),
        modified: file.modified()?,
        #[cfg(unix)]
        identity: (file.dev(), file.ino(), file.ctime(), file.ctime_nsec()),
    })
}

#[cfg(unix)]
type AssetDigests = std::collections::HashMap<PathBuf, (FileStamp, [u8; 32])>;

fn asset_hash(path: &Path, before: &FileStamp) -> Result<[u8; 32]> {
    // The kernel changes ctime even when a tool restores mtime. Use that guarantee
    // only for read-mostly model resources. Song and instrumental bytes are always hashed.
    #[cfg(unix)]
    static DIGESTS: std::sync::LazyLock<std::sync::Mutex<AssetDigests>> =
        std::sync::LazyLock::new(Default::default);
    #[cfg(unix)]
    if let Some((_, hash)) = DIGESTS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(path)
        .filter(|(s, _)| s == before)
    {
        return Ok(*hash);
    }
    let digest = hash_file(path)?;
    ensure!(
        stamp(path)? == *before,
        "Cover resource changed while reading {}",
        path.display()
    );
    #[cfg(unix)]
    {
        let mut known = DIGESTS.lock().unwrap_or_else(|e| e.into_inner());
        if known.len() >= 128 {
            known.clear();
        }
        known.insert(path.to_owned(), (before.clone(), digest));
    }
    Ok(digest)
}

pub(super) struct Cache {
    directory: PathBuf,
    common: Sha256,
    assets: Vec<(PathBuf, FileStamp)>,
    missing: Vec<PathBuf>,
}

impl Cache {
    pub fn new(
        paths: &Paths,
        request: &Request,
        info: &ModelInfo,
        speaker: i64,
        index: Option<&Path>,
        presets: &[&str],
    ) -> Result<Self> {
        let mut common = Sha256::new();
        // Bump the recipe revision if the Cover DSP, model defaults or encoding change.
        common.update(b"NVC Cover render v1\0");
        common.update(env!("CARGO_PKG_VERSION"));
        let device = crate::model::weights::device(&request.device)?;
        let options = serde_json::json!([
            speaker,
            request.index_rate,
            request.pitch,
            request.f0_method,
            request.vocals_only,
            format!("{:?}", request.backing),
            request.dry,
            request.aperiodic,
            request.voice_gate,
            request.vocal_gain,
            format!("{device:?}"),
            request.format.extension(),
        ]);
        common.update(serde_json::to_vec(&options)?);
        for var in [
            "RVC_RMVPE_THRESHOLD",
            "RVC_BANDS_NARROW",
            "RVC_ONNX_PROVIDER",
            "RVC_CONTENTVEC_ONNX",
            "RVC_RMVPE_ONNX",
            "RVC_FCPE_ONNX",
            "RVC_PRIOR_ONNX",
            "RVC_FLOW_ONNX",
            "RVC_DECODER_ONNX",
            "RVC_CUDA_GRAPH",
            "RVC_ONNX_RUNTIME",
            "RVC_CUDNN_BENCHMARK",
        ] {
            common.update(serde_json::to_vec(&std::env::var_os(var))?);
        }

        let mut assets = Vec::new();
        let mut missing = Vec::new();
        let mut asset = |path: Option<&Path>| -> Result<()> {
            if let Some(path) = path {
                let before = stamp(path)?;
                common.update([1]);
                common.update(asset_hash(path, &before)?);
                assets.push((path.to_owned(), before));
            } else {
                common.update([0]);
            }
            Ok(())
        };
        if let Some(runtime) = std::env::var_os("RVC_ONNX_RUNTIME").map(PathBuf::from)
            && runtime.is_file()
        {
            asset(Some(&runtime))?;
        } else {
            asset(None)?;
        }
        asset(Some(&request.model))?;
        asset(index)?;
        let embedder = info.format.embedder.directory(paths);
        let binary = embedder.join("pytorch_model.bin");
        let tensors = embedder.join("model.safetensors");
        if !binary.is_file() {
            missing.push(binary.clone());
        }
        asset(Some(if binary.is_file() { &binary } else { &tensors }))?;
        for name in ["config.json", "preprocessor_config.json", "voice_gate.json"] {
            let file = embedder.join(name);
            if file.is_file() {
                asset(Some(&file))?;
            } else {
                asset(None)?;
                missing.push(file);
            }
        }
        if info.uses_f0 {
            match request.f0_method {
                PitchMethod::Rmvpe | PitchMethod::Rmcpe => {
                    let checkpoint = paths.rmvpe.join("rmvpe.pt");
                    let graph = paths.rmvpe.join("rmvpe.onnx");
                    if !checkpoint.is_file() {
                        missing.push(checkpoint.clone());
                    }
                    asset(Some(if checkpoint.is_file() {
                        &checkpoint
                    } else {
                        &graph
                    }))?;
                }
                PitchMethod::Nmvpe => asset(Some(&paths.nmvpe()))?,
                _ => {}
            }
            if matches!(request.f0_method, PitchMethod::Fcpe | PitchMethod::Rmcpe) {
                asset(Some(&paths.root.join("assets/fcpe/fcpe_c_v001.pt")))?;
            }
        }
        let weights = separation::weights_directory(&paths.root);
        for name in presets {
            let model = separation::PRESETS
                .iter()
                .find(|p| p.id == *name)
                .context("unknown Cover preset")?;
            asset(Some(&weights.join(model.config)))?;
            asset(Some(&weights.join(model.checkpoint)))?;
        }
        Ok(Self {
            directory: paths.root.join("TEMP/cover-cache"),
            common,
            assets,
            missing,
        })
    }

    // Inference resources may be replaced between hashing and loading a model.
    // Never publish a render under an earlier version's key in that case.
    pub fn check_assets(&self) -> Result<()> {
        for path in &self.missing {
            ensure!(
                !path.try_exists()?,
                "Cover resource changed during rendering: {}",
                path.display()
            );
        }
        for (path, before) in &self.assets {
            ensure!(
                stamp(path)? == *before,
                "Cover resource changed during rendering: {}",
                path.display()
            );
        }
        Ok(())
    }

    pub fn entry(&self, input: &Path, instrumental: Option<&Path>) -> Result<PathBuf> {
        let mut hash = self.common.clone();
        ingredient(&mut hash, Some(input))?;
        ingredient(&mut hash, instrumental)?;
        Ok(self.directory.join(format!("{:x}.zip", hash.finalize())))
    }
}

/// Decode all cached stems into a private directory and verify each ZIP checksum before
/// anything is published. The names in the archive can only select these four roles.
pub(super) fn restore(
    entry: &Path,
    stage: &Path,
    stem: &str,
    ext: &str,
    mix: bool,
) -> Result<Option<Tracks>> {
    let file = match File::open(entry) {
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(None),
        result => result?,
    };
    let mut archive = ZipArchive::new(BufReader::new(file))?;
    let mut names = vec!["lead"];
    if archive.file_names().any(|n| n == "backing") {
        names.push("backing");
    }
    names.push("vocals");
    if mix {
        names.push("cover");
    }
    ensure!(archive.len() == names.len(), "incomplete Cover cache entry");
    let mut tracks = Vec::new();
    for role in names {
        let mut source = archive.by_name(role)?;
        ensure!(
            source.size() > 0 && source.compression() == CompressionMethod::Stored,
            "invalid cached stem"
        );
        let path = stage.join(format!("{stem}_{role}.{ext}"));
        let mut target = BufWriter::new(File::create(&path)?);
        let size = io::copy(&mut source, &mut target)?;
        ensure!(size == source.size(), "truncated cached stem");
        target.flush()?;
        tracks.push((role, path));
    }
    Ok(Some(tracks))
}

pub(super) fn pack(entry: &Path, tracks: &Tracks) -> Result<NamedTempFile> {
    let dir = entry.parent().context("Cover cache has no directory")?;
    fs::create_dir_all(dir)?;
    let mut temp = NamedTempFile::new_in(dir)?;
    let mut zip = ZipWriter::new(BufWriter::new(temp.as_file_mut()));
    let options = SimpleFileOptions::default()
        .compression_method(CompressionMethod::Stored)
        .large_file(true);
    for (role, path) in tracks {
        zip.start_file(*role, options)?;
        io::copy(&mut File::open(path)?, &mut zip)?;
    }
    zip.finish()?.flush()?;
    Ok(temp)
}

pub(super) fn publish(tracks: &mut Tracks, output: &Path, keep_identical: bool) -> Result<()> {
    let mut pending = Vec::new();
    for (index, (_, source)) in tracks.iter().enumerate() {
        let dest = output.join(source.file_name().context("missing track filename")?);
        let previous = match fs::symlink_metadata(&dest) {
            Ok(metadata) => Some(metadata),
            Err(error) if error.kind() == io::ErrorKind::NotFound => None,
            Err(error) => return Err(error.into()),
        };
        if let Some(metadata) = &previous {
            ensure!(
                metadata.file_type().is_file() || metadata.file_type().is_symlink(),
                "Cover output is not a file: {}",
                dest.display()
            );
            if keep_identical
                && dest.is_file()
                && dest.metadata()?.len() == source.metadata()?.len()
                && hash_file(&dest)? == hash_file(source)?
            {
                continue;
            }
        }
        // Back up previous outputs in the private staging directory, on the same filesystem.
        // Preparing every destination first also catches unwritable/malformed sets before commit.
        let backup = if previous.is_some() {
            let backup = source.with_file_name(format!(".previous-{index}"));
            if fs::hard_link(&dest, &backup).is_err() {
                fs::copy(&dest, &backup)
                    .with_context(|| format!("backing up {}", dest.display()))?;
            }
            Some(backup)
        } else {
            None
        };
        pending.push((source, dest, backup));
    }
    for (index, (source, dest, _)) in pending.iter().enumerate() {
        if let Err(error) = fs::rename(source, dest) {
            for (_, published, backup) in pending[..index].iter().rev() {
                let restore = match backup {
                    Some(file) => fs::rename(file, published),
                    None => fs::remove_file(published),
                };
                restore.with_context(|| {
                    format!(
                        "publishing {}: {error}; restoring {}",
                        dest.display(),
                        published.display()
                    )
                })?;
            }
            return Err(error).with_context(|| format!("publishing {}", dest.display()));
        }
    }
    for (_, source) in tracks {
        *source = output.join(source.file_name().context("missing track filename")?);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn moved_uploads_share_a_key_and_modified_bytes_invalidate_it() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let root = temp.path();
        let cache = Cache {
            directory: root.to_owned(),
            common: Sha256::new(),
            assets: vec![],
            missing: vec![],
        };
        let a = root.join("first.wav");
        let moved = root.join("another-name.wav");
        let backing = root.join("beat.wav");
        fs::write(&a, b"audio1")?;
        fs::write(&moved, b"audio1")?;
        fs::write(&backing, b"beat01")?;
        assert_eq!(cache.entry(&a, None)?, cache.entry(&moved, None)?);
        let before = cache.entry(&a, Some(&backing))?;
        fs::write(&backing, b"beat02")?;
        assert_ne!(before, cache.entry(&a, Some(&backing))?);
        fs::write(&moved, b"audio2")?;
        assert_ne!(cache.entry(&a, None)?, cache.entry(&moved, None)?);
        Ok(())
    }

    #[test]
    fn failed_publication_restores_the_original_tracks() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let stage = tempfile::tempdir_in(temp.path())?;
        let lead = stage.path().join("one_lead.wav");
        let missing = stage.path().join("one_vocals.wav");
        fs::write(&lead, b"new lead")?;
        let old = temp.path().join("one_lead.wav");
        fs::write(&old, b"old lead")?;
        let mut tracks = vec![("lead", lead), ("vocals", missing)];
        assert!(publish(&mut tracks, temp.path(), false).is_err());
        assert_eq!(fs::read(old)?, b"old lead");
        assert!(!temp.path().join("one_vocals.wav").exists());
        Ok(())
    }

    #[test]
    fn archives_are_independent_of_outputs_and_incomplete_takes_never_replace_them() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let root = temp.path();
        let entry = root.join("key.zip");
        let output = root.join("out");
        fs::create_dir(&output)?;
        let path = root.join("source");
        fs::write(&path, b"audio of a completed take")?;
        let tracks = vec![("lead", path.clone()), ("vocals", path)];
        pack(&entry, &tracks)?.persist(&entry)?;
        assert!(pack(&entry, &vec![("lead", root.join("missing"))]).is_err());
        for _ in 0..2 {
            let stage = tempfile::tempdir_in(root)?;
            let mut saved = restore(&entry, stage.path(), "renamed", "wav", false)?.unwrap();
            publish(&mut saved, &output, true)?;
            assert_eq!(fs::read(&saved[0].1)?, b"audio of a completed take");
            fs::write(&saved[0].1, b"manual edit")?;
            fs::remove_file(&saved[1].1)?;
        }
        fs::write(&entry, b"damaged archive")?;
        assert!(restore(&entry, root, "renamed", "wav", false).is_err());
        Ok(())
    }
}
