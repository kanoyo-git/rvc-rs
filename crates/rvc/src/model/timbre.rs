// NVC-TS kNN content substitution and NVC-compatible feature pools.
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::ops::ControlFlow;
use std::path::{Component, Path, PathBuf};
use std::time::UNIX_EPOCH;

use anyhow::{Context, Result, ensure};
use ndarray::Array2;
use rand::{Rng, SeedableRng, rngs::StdRng};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tch::{Device, Kind, Tensor, no_grad};
use tempfile::TempDir;

use super::Hubert;
use super::train_data;
use crate::audio::{self, ANALYSIS_RATE};
use crate::catalog::{Embedder, Paths};

const WIDTH: i64 = 768;
const K: i64 = 4;
const VARIANTS: usize = 4;
const KEY_CHUNK: i64 = 8192;
const QUERY_CHUNK: i64 = 128;

#[derive(Deserialize, Serialize)]
struct Pool {
    embedder: Embedder,
    speakers: Vec<String>,
    #[serde(default)]
    frames: BTreeMap<String, usize>,
}

#[derive(clap::Args, Deserialize)]
pub struct PoolRequest {
    #[arg(long)]
    pub spec: PathBuf,
    #[arg(long, alias = "out")]
    pub output: PathBuf,
    #[arg(long, value_enum, default_value = "contentvec")]
    #[serde(default = "default_embedder")]
    pub embedder: Embedder,
    #[arg(long, default_value_t = 8.0)]
    #[serde(default = "default_minutes")]
    pub max_minutes: f64,
    #[arg(long, default_value = "auto")]
    #[serde(default = "default_device")]
    pub device: String,
}

fn default_embedder() -> Embedder {
    Embedder::Contentvec
}
fn default_minutes() -> f64 {
    8.0
}
fn default_device() -> String {
    "auto".into()
}

#[derive(Serialize)]
pub struct PoolReport {
    pub path: PathBuf,
    pub embedder: Embedder,
    pub frames: BTreeMap<String, usize>,
}

pub fn build_pool(
    paths: &Paths,
    request: PoolRequest,
    mut report: impl FnMut(&str) -> ControlFlow<()>,
) -> Result<PoolReport> {
    let seconds = request.max_minutes * 60.0;
    ensure!(
        seconds.is_finite() && seconds > 0.0,
        "max minutes must be positive and finite"
    );
    let budget = (seconds * f64::from(ANALYSIS_RATE)) as usize;
    let spec = paths.root.join(request.spec);
    let output = paths.root.join(request.output);
    let spec_data: BTreeMap<String, Vec<PathBuf>> = serde_json::from_reader(File::open(&spec)?)
        .with_context(|| format!("reading pool spec {}", spec.display()))?;
    ensure!(
        spec_data.len() >= 2,
        "the feature pool needs at least two speakers"
    );
    for name in spec_data.keys() {
        speaker_file(name)?;
    }
    ensure!(
        !output.exists(),
        "feature pool already exists at {}",
        output.display()
    );
    let parent = output
        .parent()
        .context("feature pool has no parent directory")?;
    fs::create_dir_all(parent)?;
    let stage = TempDir::new_in(parent)?;
    let directory = request.embedder.install(paths, &mut report)?;
    let device = super::weights::device(&request.device)?;
    let frames = no_grad(|| -> Result<_> {
        let hubert = Hubert::load(&directory, device, Kind::Float, true)?;
        let mut frames = BTreeMap::new();
        for (speaker, sources) in spec_data {
            let mut unique = BTreeSet::new();
            for source in sources {
                let source = paths.root.join(source);
                if source.is_dir() {
                    unique.extend(audio::files(&source, true)?);
                } else {
                    ensure!(
                        source.is_file() && audio::is_audio(&source),
                        "invalid audio source {}",
                        source.display()
                    );
                    unique.insert(source);
                }
            }
            let mut parts = Vec::new();
            let mut remaining = budget;
            for path in unique {
                if remaining < 400 {
                    break;
                }
                notify(
                    &mut report,
                    &format!("Feature pool · {speaker} · {}", path.display()),
                )?;
                let samples = audio::read_mono(&path, ANALYSIS_RATE)?;
                let samples = &samples[..samples.len().min(remaining)];
                for chunk in samples.chunks(20 * ANALYSIS_RATE as usize) {
                    notify(
                        &mut report,
                        &format!("Extracting pool features · {speaker}"),
                    )?;
                    if chunk.len() < 400 || !chunk.iter().any(|sample| sample.abs() >= 1e-3) {
                        continue;
                    }
                    let content = hubert
                        .extract(chunk, true)?
                        .f_squeeze_dim(0)?
                        .f_to_device_(Device::Cpu, Kind::Half, false, false)?;
                    parts.push(content);
                    remaining -= chunk.len();
                }
            }
            ensure!(
                !parts.is_empty(),
                "no usable audio in feature pool for {speaker}"
            );
            let matrix = Tensor::f_cat(&parts, 0)?;
            let rows = matrix.size()[0] as usize;
            ensure!(
                rows >= K as usize,
                "need at least {K} feature frames for {speaker}"
            );
            matrix.write_npy(stage.path().join(speaker_file(&speaker)?))?;
            frames.insert(speaker, rows);
        }
        Ok(frames)
    })?;
    let metadata = Pool {
        embedder: request.embedder,
        speakers: frames.keys().cloned().collect(),
        frames,
    };
    write_json(&stage.path().join("pool.json"), &metadata)?;
    notify(&mut report, "Saving feature pool")?;
    fs::rename(stage.path(), &output)
        .with_context(|| format!("publishing feature pool at {}", output.display()))?;
    Ok(PoolReport {
        path: output,
        embedder: request.embedder,
        frames: metadata.frames,
    })
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(super) struct Options {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub dirs: Vec<PathBuf>,
    pub prob: f32,
    pub mix_range: Option<[f32; 2]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pool: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub exclude: Vec<String>,
}

impl Options {
    pub fn knn(pool: PathBuf, exclude: Vec<String>) -> Self {
        Self {
            dirs: Vec::new(),
            prob: 0.5,
            mix_range: Some([0.5, 1.0]),
            kind: Some("knn".into()),
            pool: Some(pool),
            exclude,
        }
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            (0.0..=1.0).contains(&self.prob),
            "invalid NVC-TS augmentation probability"
        );
        ensure!(
            self.mix_range
                .is_none_or(|[a, b]| (0.0..=b).contains(&a) && (0.0..=1.0).contains(&b)),
            "invalid NVC-TS blend range"
        );
        Ok(())
    }
}

pub(super) struct Augmentation {
    options: Options,
    variants: BTreeMap<String, Vec<PathBuf>>,
}

impl Augmentation {
    pub fn load(experiment: &Path, options: &Options) -> Result<Self> {
        options.validate()?;
        let mut variants: BTreeMap<String, Vec<PathBuf>> = BTreeMap::new();
        for directory in &options.dirs {
            let path = experiment.join(directory);
            for entry in fs::read_dir(&path)
                .with_context(|| format!("reading NVC-TS features in {}", path.display()))?
            {
                let file = entry?.path();
                if file.extension().is_some_and(|ext| ext == "npy")
                    && let Some(stem) = file.file_stem().and_then(|name| name.to_str())
                    && let Some((clip, variant)) = stem.rsplit_once("__")
                    && variant.parse::<usize>().is_ok()
                {
                    variants.entry(clip.to_owned()).or_default().push(file);
                }
            }
        }
        ensure!(
            !variants.is_empty(),
            "NVC-TS expects per-clip NAME__N.npy feature variants"
        );
        for choices in variants.values_mut() {
            choices.sort();
        }
        Ok(Self {
            options: options.clone(),
            variants,
        })
    }

    pub fn mix(&self, path: &Path, clean: &mut Array2<f32>, rng: &mut impl Rng) -> Result<()> {
        if rng.random::<f32>() >= self.options.prob {
            return Ok(());
        }
        let Some(options) = path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .and_then(|stem| self.variants.get(stem))
        else {
            return Ok(());
        };
        let variant = &options[rng.random_range(0..options.len())];
        let values: Array2<f32> = train_data::float_array(variant)?;
        ensure!(
            values.dim() == clean.dim(),
            "NVC-TS variant shape differs from {}",
            path.display()
        );
        let alpha = self
            .options
            .mix_range
            .map_or(1.0, |[lo, hi]| rng.random_range(lo..=hi));
        clean.zip_mut_with(&values, |clean, &shifted| {
            *clean = (1.0 - alpha) * *clean + alpha * shifted
        });
        Ok(())
    }
}

pub(super) fn generate(
    paths: &Paths,
    experiment: &Path,
    options: &Options,
    clips: &BTreeMap<PathBuf, BTreeSet<String>>,
    config: &train_data::Config,
    device: Device,
    mut report: impl FnMut(&str) -> ControlFlow<()>,
) -> Result<PathBuf> {
    let pool = paths
        .root
        .join(options.pool.as_ref().context("missing NVC-TS pool")?);
    let embedder = config.embedder_model.unwrap_or_default();
    let seed = config.train.seed as u64;
    let meta_path = pool.join("pool.json");
    let meta_bytes =
        fs::read(&meta_path).with_context(|| format!("reading {}", meta_path.display()))?;
    let metadata: Pool = serde_json::from_slice(&meta_bytes)?;
    ensure!(
        metadata.embedder == embedder,
        "NVC-TS pool uses {} but the experiment uses {}",
        metadata.embedder.name(),
        embedder.name()
    );

    let speakers: BTreeMap<_, _> = metadata
        .speakers
        .iter()
        .filter(|name| !options.exclude.contains(name))
        .map(|name| speaker_file(name).map(|file| (name.as_str(), pool.join(file))))
        .collect::<Result<_>>()?;
    ensure!(
        speakers.len() >= 2,
        "need at least two pool speakers after exclusions"
    );
    ensure!(!clips.is_empty(), "no clips to augment");
    let mut hash = Sha256::new();
    hash.update(b"rvc-nvc-ts-knn-v1");
    hash.update(seed.to_le_bytes());
    hash.update(meta_bytes);
    for (name, path) in &speakers {
        hash.update(name.as_bytes());
        fingerprint(&mut hash, path)?;
    }
    for (path, names) in clips {
        fingerprint(&mut hash, path)?;
        hash.update(serde_json::to_vec(names)?);
    }
    let digest = format!("{:x}", hash.finalize());
    let output = experiment.join("timbre_shift").join(&digest[..24]);
    if output.join("meta.json").is_file() {
        notify(&mut report, "Using cached NVC-TS features")?;
        return Ok(output);
    }
    let parent = output.parent().context("NVC-TS output has no parent")?;
    fs::create_dir_all(parent)?;
    let stage = TempDir::new_in(parent)?;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut assignments: BTreeMap<&str, Vec<(&Path, usize)>> = BTreeMap::new();
    let mut stems = BTreeSet::new();
    for (clip, target_names) in clips {
        let stem = feature_stem(clip)?;
        ensure!(
            stems.insert(stem),
            "duplicate clip name {stem} in NVC-TS filelist"
        );
        let choices: Vec<_> = speakers
            .keys()
            .copied()
            .filter(|name| !target_names.contains(*name))
            .collect();
        ensure!(!choices.is_empty(), "no other pool speakers for {stem}");
        for variant in 0..VARIANTS {
            let name = choices[rng.random_range(0..choices.len())];
            assignments.entry(name).or_default().push((clip, variant));
        }
    }
    let total = clips.len() * VARIANTS;
    let mut done = 0;
    no_grad(|| -> Result<()> {
        for (speaker, group) in assignments {
            notify(
                &mut report,
                &format!("Loading NVC-TS pool speaker {speaker}"),
            )?;
            let keys = matrix(&speakers[speaker], device)?;
            ensure!(
                keys.size()[0] >= K,
                "pool speaker {speaker} has fewer than {K} frames"
            );
            let normalized = normalize(&keys)?;
            for (clip, variant) in group {
                notify(
                    &mut report,
                    &format!("NVC-TS · {done}/{total} · {}", clip.display()),
                )?;
                let query = matrix(clip, device)?;
                let substituted = neighbors(&query, &keys, &normalized)?;
                let target = stage
                    .path()
                    .join(format!("{}__{variant}.npy", feature_stem(clip)?));
                substituted.f_to_device(Device::Cpu)?.write_npy(target)?;
                done += 1;
            }
        }
        Ok(())
    })?;
    write_json(
        &stage.path().join("meta.json"),
        &serde_json::json!({
            "kind": "knn", "pool": pool, "pool_speakers": speakers.keys().collect::<Vec<_>>(),
            "k": K, "variants": VARIANTS, "seed": seed, "embedder": embedder,
        }),
    )?;
    notify(&mut report, "Saving NVC-TS features")?;
    fs::rename(stage.path(), &output)?;
    Ok(output)
}

fn notify(report: &mut impl FnMut(&str) -> ControlFlow<()>, message: &str) -> Result<()> {
    ensure!(report(message).is_continue(), "NVC-TS operation cancelled");
    Ok(())
}

fn speaker_file(name: &str) -> Result<String> {
    let mut parts = Path::new(name).components();
    ensure!(
        matches!(parts.next(), Some(Component::Normal(_)))
            && parts.next().is_none()
            && !name.contains(['/', '\\', ':'])
            && !name.chars().any(char::is_control),
        "invalid pool speaker name {name:?}"
    );
    Ok(format!("{name}.npy"))
}

fn feature_stem(path: &Path) -> Result<&str> {
    path.file_stem()
        .and_then(|name| name.to_str())
        .context("invalid feature filename")
}

fn fingerprint(hash: &mut Sha256, path: &Path) -> Result<()> {
    let metadata = path
        .metadata()
        .with_context(|| format!("checking {}", path.display()))?;
    hash.update(path.as_os_str().as_encoded_bytes());
    hash.update(metadata.len().to_le_bytes());
    let time = metadata
        .modified()
        .ok()
        .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
        .unwrap_or_default();
    hash.update(time.as_nanos().to_le_bytes());
    Ok(())
}

fn matrix(path: &Path, device: Device) -> Result<Tensor> {
    let tensor = Tensor::read_npy(path).with_context(|| format!("reading {}", path.display()))?;
    let (rows, width) = tensor.size2()?;
    ensure!(
        tensor.is_floating_point() && width == WIDTH && rows > 0,
        "expected a [frames, {WIDTH}] feature matrix ({})",
        path.display()
    );
    let tensor = tensor.f_to_device_(device, Kind::Float, false, false)?;
    ensure!(
        tensor.f_isfinite()?.f_all()?.f_int64_value(&[])? != 0,
        "non-finite feature values in {}",
        path.display()
    );
    Ok(tensor)
}

fn normalize(features: &Tensor) -> Result<Tensor> {
    Ok(features.f_div(
        &features
            .f_norm_scalaropt_dim(2.0, [1], true)?
            .f_clamp_min(1e-12)?,
    )?)
}

fn neighbors(query: &Tensor, keys: &Tensor, normalized: &Tensor) -> Result<Tensor> {
    let query = normalize(query)?;
    let mut frames = Vec::new();
    for rows in query.f_split(QUERY_CHUNK, 0)? {
        let mut top: Option<(Tensor, Tensor)> = None;
        for begin in (0..keys.size()[0]).step_by(KEY_CHUNK as usize) {
            let length = KEY_CHUNK.min(keys.size()[0] - begin);
            let block = normalized.f_narrow(0, begin, length)?;
            let scores = rows.f_matmul(&block.f_transpose(0, 1)?)?;
            let (scores, indices) = scores.f_topk(K.min(length), 1, true, false)?;
            let indices = indices.f_add_scalar(begin)?;
            top = Some(if let Some((old_scores, old_indices)) = top {
                let combined = Tensor::f_cat(&[old_scores, scores], 1)?;
                let (scores, index) = combined.f_topk(K.min(combined.size()[1]), 1, true, false)?;
                let ids = Tensor::f_cat(&[old_indices, indices], 1)?.f_gather(1, &index, false)?;
                (scores, ids)
            } else {
                (scores, indices)
            });
        }
        let (_, ids) = top.context("empty NVC-TS key matrix")?;
        let neighbors = keys
            .f_index_select(0, &ids.f_view([-1])?)?
            .f_view([-1, K, WIDTH])?;
        frames.push(neighbors.f_mean_dim(&[1i64][..], false, Kind::Float)?);
    }
    Ok(Tensor::f_cat(&frames, 0)?)
}

fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    audio::write_file(path, &serde_json::to_vec_pretty(value)?, false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn substitutes_global_cosine_neighbors_with_original_feature_values() -> Result<()> {
        let options = (Kind::Float, Device::Cpu);
        let keys = Tensor::f_zeros([KEY_CHUNK + 2, WIDTH], options)?;
        for (index, value) in [(0, 2.0), (1, 5.0), (KEY_CHUNK, 11.0), (KEY_CHUNK + 1, 18.0)] {
            let _ = keys.f_get(index)?.f_get(0)?.f_fill_(value)?;
        }
        let query = Tensor::f_zeros([1, WIDTH], options)?;
        let _ = query.f_get(0)?.f_get(0)?.f_fill_(0.5)?;
        let shifted = neighbors(&query, &keys, &normalize(&keys)?)?;
        assert_eq!(shifted.size(), [1, WIDTH]);
        assert_eq!(shifted.f_double_value(&[0, 0])?, 9.0);
        assert_eq!(shifted.f_sum(Kind::Float)?.f_double_value(&[])?, 9.0);
        Ok(())
    }
}
