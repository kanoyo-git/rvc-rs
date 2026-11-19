use std::collections::BTreeMap;
use std::fs;
use std::io::{self, BufWriter};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use clap::ValueEnum;
use faiss::index::flat::FlatIndex;
use faiss::index::ivf_flat::IVFFlatIndex;
use faiss::{Index, MetricType, read_index, write_index};
use ndarray::{Array2, ArrayView2};
use ndarray_npy::{WriteNpyExt, read_npy};
use rand::Rng;
use rand::seq::{SliceRandom, index};
use serde::{Deserialize, Serialize};
use tempfile::{NamedTempFile, TempDir, TempPath};

const BATCH_ROWS: usize = 8192;
const MAX_UNCLUSTERED_ROWS: usize = 200_000;
const MINI_BATCH_CENTERS: usize = 10_000;

#[derive(Clone, Copy, Debug, Deserialize, ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum ModelVersion {
    V1,
    V2,
}

impl ModelVersion {
    fn name(self) -> &'static str {
        match self {
            Self::V1 => "v1",
            Self::V2 => "v2",
        }
    }

    fn dimensions(self) -> usize {
        match self {
            Self::V1 => 256,
            Self::V2 => 768,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq, ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum IndexMode {
    #[default]
    Auto,
    Single,
    Multi,
}

pub struct IndexTraining<'a> {
    pub experiment_dir: &'a Path,
    pub version: ModelVersion,
    pub outside_index_root: &'a Path,
    pub workers: usize,
    pub mode: IndexMode,
}

#[derive(Debug, Serialize)]
pub struct IndexArtifact {
    pub speaker_id: Option<u32>,
    pub feature_rows: usize,
    pub index_rows: u64,
    pub total_features: PathBuf,
    pub trained: Option<PathBuf>,
    pub added: PathBuf,
    pub external: PathBuf,
    pub reused: bool,
}

#[derive(Deserialize)]
struct Manifest {
    entries: Vec<ManifestEntry>,
}

#[derive(Deserialize)]
struct ManifestEntry {
    output_key: String,
    speaker_id: u32,
}

pub fn train_indices(
    request: &IndexTraining<'_>,
    report: impl FnMut(&str),
) -> Result<Vec<IndexArtifact>> {
    train_indices_with_cancel(request, || false, report)
}

pub fn train_indices_with_cancel(
    request: &IndexTraining<'_>,
    is_cancelled: impl Fn() -> bool,
    mut report: impl FnMut(&str),
) -> Result<Vec<IndexArtifact>> {
    ensure!(!is_cancelled(), "index training cancelled");
    let experiment = request
        .experiment_dir
        .file_name()
        .and_then(|name| name.to_str())
        .context("experiment directory must have a UTF-8 name")?;
    ensure!(request.workers > 0, "workers must be positive");
    let feature_groups =
        collect_feature_groups(request.experiment_dir, request.version, request.mode)?;
    feature_groups
        .into_iter()
        .map(|(speaker, paths)| {
            train_group(
                request,
                experiment,
                speaker,
                &paths,
                &is_cancelled,
                |message| {
                    if let Some(speaker) = speaker {
                        report(&format!("Speaker {speaker} | {message}"));
                    } else {
                        report(message);
                    }
                },
            )
        })
        .collect()
}

fn collect_feature_groups(
    exp_dir: &Path,
    version: ModelVersion,
    mode: IndexMode,
) -> Result<BTreeMap<Option<u32>, Vec<PathBuf>>> {
    let feature_dir = exp_dir.join(format!("3_feature{}", version.dimensions()));
    let mut paths = Vec::new();
    for entry in fs::read_dir(&feature_dir)
        .with_context(|| format!("extract features first ({})", feature_dir.display()))?
    {
        let path = entry?.path();
        if path
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("npy"))
            && path.is_file()
        {
            paths.push(path);
        }
    }
    paths.sort();
    ensure!(!paths.is_empty(), "extract features first");
    if mode == IndexMode::Single {
        return Ok(BTreeMap::from([(None, paths)]));
    }
    let manifest_file = match fs::File::open(exp_dir.join("multispeaker_manifest.json")) {
        Ok(file) => file,
        Err(error) if error.kind() == io::ErrorKind::NotFound && mode == IndexMode::Auto => {
            return Ok(BTreeMap::from([(None, paths)]));
        }
        Err(error) => return Err(error).context("open multi-speaker manifest"),
    };
    let manifest: Manifest =
        serde_json::from_reader(manifest_file).context("read multi-speaker manifest")?;
    let mut speakers = BTreeMap::new();
    for entry in manifest.entries {
        ensure!(
            !entry.output_key.is_empty() && entry.speaker_id <= 109,
            "invalid speaker entry in multi-speaker manifest"
        );
        ensure!(
            speakers
                .insert(entry.output_key, entry.speaker_id)
                .is_none(),
            "duplicate feature key in multi-speaker manifest"
        );
    }
    let mut groups = BTreeMap::<Option<u32>, Vec<PathBuf>>::new();
    for path in paths {
        let stem = path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .with_context(|| format!("feature filename is not UTF-8 ({})", path.display()))?;
        let speaker = speakers.get(stem).or_else(|| {
            let (prefix, _) = stem.rsplit_once('_')?;
            speakers.get(prefix)
        });
        if let Some(&speaker) = speaker {
            groups.entry(Some(speaker)).or_default().push(path);
        }
    }
    ensure!(
        !groups.is_empty(),
        "no extracted features match the speaker manifest"
    );
    Ok(groups)
}

fn train_group(
    request: &IndexTraining<'_>,
    experiment: &str,
    speaker: Option<u32>,
    paths: &[PathBuf],
    is_cancelled: &impl Fn() -> bool,
    mut report: impl FnMut(&str),
) -> Result<IndexArtifact> {
    ensure!(!is_cancelled(), "index training cancelled");
    let dims = request.version.dimensions();
    let exp_dir = request.experiment_dir;
    let features = read_features(paths, dims, is_cancelled)?;
    let feature_rows = features.len() / dims;
    let speaker_suffix = speaker.map_or_else(String::new, |id| format!("_spkid{id}"));
    let total_features = exp_dir.join(format!("total_fea{speaker_suffix}.npy"));
    ensure!(!is_cancelled(), "index training cancelled");
    let mut pending_features = NamedTempFile::new_in(exp_dir)?;
    ArrayView2::from_shape((feature_rows, dims), &features)?
        .write_npy(BufWriter::new(pending_features.as_file_mut()))?;

    let suffix = format!(
        "_{}_{}{speaker_suffix}.index",
        experiment,
        request.version.name()
    );
    let trained = find_newest(exp_dir, "trained_IVF", &suffix)?;
    let mut added = find_newest(exp_dir, "added_IVF", &suffix)?;
    if added.is_none() {
        added = find_newest(
            request.outside_index_root,
            &format!("{experiment}_added_IVF"),
            &suffix,
        )?;
    }
    let mut writes = Vec::new();
    let (trained, added, index_rows, reused) = if let Some(added) = added {
        let index = open_index(&added, dims)?;
        ensure!(
            index.ntotal() > 0,
            "added index is empty ({})",
            added.display()
        );
        (trained, added, index.ntotal(), true)
    } else {
        let vectors = if feature_rows > MAX_UNCLUSTERED_ROWS {
            report(&format!(
                "Clustering {feature_rows} rows into {MINI_BATCH_CENTERS} centers"
            ));
            mini_batch_centers(&features, dims, request.workers, is_cancelled)?
        } else {
            features
        };
        let (mut index, trained) = if let Some(trained) = trained {
            let mut index = open_index(&trained, dims)?;
            ensure!(
                index.ntotal() == 0,
                "trained index already contains vectors"
            );
            index.set_nprobe(1);
            report(&format!("Reused {}", trained.display()));
            (index, trained)
        } else {
            let rows = vectors.len() / dims;
            let lists = ivf_count(rows);
            let trained = exp_dir.join(format!("trained_IVF{lists}_Flat_nprobe_1{suffix}"));
            report(&format!(
                "Training IVF{lists},Flat from {rows} rows × {dims}"
            ));
            ensure!(!is_cancelled(), "index training cancelled");
            let mut index =
                IVFFlatIndex::new_l2(FlatIndex::new_l2(dims as u32)?, dims as u32, lists)?;
            index.set_nprobe(1);
            index.train(&vectors)?;
            ensure!(!is_cancelled(), "index training cancelled");
            writes.push((stage_index(&index, exp_dir)?, trained.clone()));
            (index, trained)
        };
        for block in vectors.chunks(BATCH_ROWS * dims) {
            ensure!(!is_cancelled(), "index training cancelled");
            index.add(block)?;
        }
        let added = exp_dir.join(format!("added_IVF{}_Flat_nprobe_1{suffix}", index.nlist()));
        writes.push((stage_index(&index, exp_dir)?, added.clone()));
        (Some(trained), added, index.ntotal(), false)
    };
    ensure!(!is_cancelled(), "index training cancelled");
    for (pending, path) in writes {
        pending
            .persist_noclobber(&path)
            .with_context(|| format!("publishing {}", path.display()))?;
    }
    pending_features.persist(&total_features)?;
    let external = publish_link(&added, request.outside_index_root, experiment)?;
    let action = if reused { "Reused" } else { "Built" };
    report(&format!("{action} {}", added.display()));
    Ok(IndexArtifact {
        speaker_id: speaker,
        feature_rows,
        index_rows,
        total_features,
        trained,
        added,
        external,
        reused,
    })
}

fn read_features(
    paths: &[PathBuf],
    dims: usize,
    is_cancelled: &impl Fn() -> bool,
) -> Result<Vec<f32>> {
    let mut features = Vec::new();
    for path in paths {
        ensure!(!is_cancelled(), "index training cancelled");
        let array: Array2<f32> =
            read_npy(path).with_context(|| format!("read features ({})", path.display()))?;
        ensure!(
            array.ncols() == dims,
            "expected {dims} feature columns ({})",
            path.display()
        );
        ensure!(
            array.iter().all(|x| x.is_finite()),
            "non-finite features ({})",
            path.display()
        );
        features.extend(array.iter().copied());
    }
    let mut order: Vec<_> = (0..features.len() / dims).collect();
    ensure!(!order.is_empty(), "feature arrays contain no rows");
    order.shuffle(&mut rand::rng());
    let mut shuffled = Vec::with_capacity(features.len());
    for row in order {
        shuffled.extend_from_slice(&features[row * dims..(row + 1) * dims]);
    }
    Ok(shuffled)
}

fn ivf_count(rows: usize) -> u32 {
    let heuristic = (16.0 * (rows as f64).sqrt()) as usize;
    heuristic.min(rows / 39).max(1) as u32
}

fn mini_batch_centers(
    features: &[f32],
    dims: usize,
    workers: usize,
    is_cancelled: &impl Fn() -> bool,
) -> Result<Vec<f32>> {
    let rows = features.len() / dims;
    let batch_size = workers.saturating_mul(256).min(rows);
    let mut rng = rand::rng();
    let mut centers = Vec::with_capacity(MINI_BATCH_CENTERS * dims);
    for row in index::sample(&mut rng, rows, MINI_BATCH_CENTERS) {
        centers.extend_from_slice(&features[row * dims..(row + 1) * dims]);
    }
    let mut counts = vec![0u64; MINI_BATCH_CENTERS];
    let mut quantizer = FlatIndex::new_l2(dims as u32)?;
    let mut batch = Vec::with_capacity(batch_size * dims);
    let alpha = (2.0 * batch_size as f64 / (rows as f64 + 1.0)).min(1.0);
    let mut average: Option<f64> = None;
    let mut best = f64::INFINITY;
    let mut no_improvement = 0;
    let max_steps = rows.saturating_mul(100) / batch_size;
    for _ in 0..max_steps {
        ensure!(!is_cancelled(), "index training cancelled");
        batch.clear();
        for _ in 0..batch_size {
            let row = rng.random_range(0..rows);
            batch.extend_from_slice(&features[row * dims..(row + 1) * dims]);
        }
        quantizer.reset()?;
        quantizer.add(&centers)?;
        let result = quantizer.search(&batch, 1)?;
        for (frame, label) in batch.chunks_exact(dims).zip(result.labels) {
            let center = label.get().context("no nearest center for feature")? as usize;
            counts[center] += 1;
            let count = counts[center] as f32;
            for (value, &feature) in centers[center * dims..(center + 1) * dims]
                .iter_mut()
                .zip(frame)
            {
                *value += (feature - *value) / count;
            }
        }
        let loss = result.distances.iter().copied().map(f64::from).sum::<f64>() / batch_size as f64;
        let current = average.map_or(loss, |previous| previous * (1.0 - alpha) + loss * alpha);
        average = Some(current);
        if current < best {
            best = current;
            no_improvement = 0;
        } else {
            no_improvement += 1;
            if no_improvement == 10 {
                break;
            }
        }
    }
    Ok(centers)
}

fn find_newest(dir: &Path, prefix: &str, suffix: &str) -> Result<Option<PathBuf>> {
    let entries = match fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error).with_context(|| format!("read {}", dir.display())),
    };
    let mut candidates = Vec::new();
    for entry in entries {
        let path = entry?.path();
        let Some((lists, probes)) = path
            .file_name()
            .and_then(|name| name.to_str())
            .and_then(|name| name.strip_prefix(prefix))
            .and_then(|name| name.strip_suffix(suffix))
            .and_then(|name| name.split_once("_Flat_nprobe_"))
        else {
            continue;
        };
        if lists.parse::<u32>().is_ok() && probes.parse::<u32>().is_ok() && path.is_file() {
            candidates.push((path.metadata()?.modified()?, path));
        }
    }
    Ok(candidates.into_iter().max().map(|(_, path)| path))
}

fn open_index(path: &Path, dims: usize) -> Result<IVFFlatIndex> {
    let index = read_index(faiss_path(path)?)
        .and_then(|index| index.into_ivf_flat())
        .with_context(|| format!("read IVF-Flat index ({})", path.display()))?;
    ensure!(
        index.is_trained() && index.d() as usize == dims && index.metric_type() == MetricType::L2,
        "expected trained IVF-Flat L2 index with {dims} dimensions ({})",
        path.display()
    );
    Ok(index)
}

fn stage_index(index: &IVFFlatIndex, dir: &Path) -> Result<TempPath> {
    let pending = NamedTempFile::new_in(dir)?.into_temp_path();
    write_index(index, faiss_path(&pending)?)?;
    Ok(pending)
}

fn faiss_path(path: &Path) -> Result<&str> {
    path.to_str()
        .with_context(|| format!("index path is not UTF-8 ({})", path.display()))
}

fn publish_link(path: &Path, outside_index_root: &Path, experiment: &str) -> Result<PathBuf> {
    fs::create_dir_all(outside_index_root)?;
    let source_parent = path.parent().context("index needs a parent directory")?;
    let outside_root = fs::canonicalize(outside_index_root)?;
    if fs::canonicalize(source_parent)?.starts_with(&outside_root) {
        return Ok(path.to_owned());
    }
    let filename = faiss_path(Path::new(
        path.file_name().context("index needs a filename")?,
    ))?;
    let target = outside_root.join(format!("{experiment}_{filename}"));
    if same_file::is_same_file(path, &target).unwrap_or(false) {
        return Ok(target);
    }
    let temp = TempDir::new_in(&outside_root)?;
    let pending = temp.path().join("link.index");
    #[cfg(windows)]
    fs::hard_link(path, &pending)?;
    #[cfg(unix)]
    std::os::unix::fs::symlink(fs::canonicalize(path)?, &pending)?;
    fs::rename(&pending, &target).with_context(|| format!("publish link {}", target.display()))?;
    Ok(target)
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::*;

    #[test]
    fn ivf_heuristic_matches_original_around_boundaries() {
        for (rows, expected) in [
            (1, 1),
            (39, 1),
            (77, 1),
            (78, 2),
            (10_000, 256),
            (1_000_000, 16_000),
        ] {
            assert_eq!(ivf_count(rows), expected);
        }
    }

    #[test]
    fn cancellation_preserves_the_index_artifacts() -> Result<()> {
        let workspace = TempDir::new()?;
        let experiment = workspace.path().join("experiment");
        let features = experiment.join("3_feature256");
        fs::create_dir_all(&features)?;
        let total = experiment.join("total_fea.npy");
        fs::write(&total, b"old vectors")?;
        ndarray_npy::write_npy(features.join("0_1.npy"), &Array2::<f32>::zeros((40, 256)))?;
        let cancelled = Cell::new(false);
        let external = workspace.path().join("indices");
        let request = IndexTraining {
            experiment_dir: &experiment,
            version: ModelVersion::V1,
            outside_index_root: &external,
            workers: 1,
            mode: IndexMode::Single,
        };
        let error = train_indices_with_cancel(
            &request,
            || cancelled.get(),
            |message| {
                if message.starts_with("Training IVF") {
                    cancelled.set(true);
                }
            },
        )
        .expect_err("the observer requested cancellation");
        assert_eq!(error.to_string(), "index training cancelled");
        assert_eq!(fs::read(total)?, b"old vectors");
        assert_eq!(fs::read_dir(experiment)?.count(), 2);
        assert!(!external.exists());
        Ok(())
    }
}
