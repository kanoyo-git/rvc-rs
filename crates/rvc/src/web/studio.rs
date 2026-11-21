mod files;
mod library;

use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result, ensure};
use axum::extract::{DefaultBodyLimit, Multipart, Path as RoutePath, Query, State};
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::{Value, json};
use uuid::Uuid;

use super::{
    Action, Convert, JobState, Shared, StateData, WebResult, asset, automatic, checkpoint_model,
    installed_model, job_progress, launch_job, lock,
};
use crate::audio::{self, ANALYSIS_RATE, Format};
use crate::catalog::{Embedder, ModelInfo};
use crate::cover;
use crate::model::{self, PitchMethod, train, train::events};
use crate::separation::{self, StemFiles};

/// Separation presets `rvc cover` loads.
const COVER_PRESETS: [&str; 3] = [
    "vocals-bs-roformer-ft1-anvuew",
    "karaoke-bs-roformer-anvuew",
    "dereverb-bs-roformer-anvuew-22.5050",
];
const EMBEDDERS: [Embedder; 6] = [
    Embedder::HubertBase,
    Embedder::Contentvec,
    Embedder::Spin,
    Embedder::SpinV2,
    Embedder::Mhubert147,
    Embedder::W2vRu,
];

pub(super) use files::Files;

pub(super) fn routes() -> Router<Shared> {
    Router::new()
        .route(
            "/studio/i18n.js",
            get(|| async {
                asset(
                    "text/javascript; charset=utf-8",
                    include_str!("../../web/i18n.js"),
                )
            }),
        )
        .route(
            "/studio/ui.js",
            get(|| async {
                asset(
                    "text/javascript; charset=utf-8",
                    include_str!("../../web/ui.js"),
                )
            }),
        )
        .route(
            "/studio/realtime.js",
            get(|| async {
                asset(
                    "text/javascript; charset=utf-8",
                    include_str!("../../web/realtime.js"),
                )
            }),
        )
        .route(
            "/studio/wordmark.png",
            get(|| async {
                (
                    [(header::CONTENT_TYPE, "image/png")],
                    include_bytes!("../../web/wordmark.png").as_slice(),
                )
            }),
        )
        .route(
            "/studio/wordmark-light.png",
            get(|| async {
                (
                    [(header::CONTENT_TYPE, "image/png")],
                    include_bytes!("../../web/wordmark-light.png").as_slice(),
                )
            }),
        )
        .route(
            "/studio/logo.png",
            get(|| async {
                (
                    [(header::CONTENT_TYPE, "image/png")],
                    include_bytes!("../../web/logo.png").as_slice(),
                )
            }),
        )
        .route("/api/studio/options", get(options))
        .route("/api/studio/voice/{name}", get(voice))
        .route("/api/studio/export/{name}", get(export_voice))
        .route("/api/studio/inspect", get(inspect))
        .route("/api/studio/pretrained", get(pretrained))
        .route("/api/studio/experiments", get(experiments))
        .route("/api/studio/resume", get(resume))
        .route("/api/studio/metrics", get(metrics))
        .route("/api/studio/media/{id}", get(files::media))
        .route("/api/studio/job/{id}", get(job).delete(cancel))
        .route(
            "/api/studio/job",
            post(start).layer(DefaultBodyLimit::max(files::UPLOAD_LIMIT)),
        )
        .route(
            "/api/studio/dataset",
            post(dataset).layer(DefaultBodyLimit::max(files::UPLOAD_LIMIT)),
        )
}

async fn options(State(state): State<Shared>) -> WebResult<Json<Value>> {
    let paths = &state.paths;
    let workers = std::thread::available_parallelism().map_or(1, usize::from);
    let device = match model::weights::device("auto")? {
        tch::Device::Cuda(index) => format!("cuda:{index}"),
        tch::Device::Mps => "mps".to_owned(),
        _ => "cpu".to_owned(),
    };
    let presets: Vec<Value> = separation::PRESETS
        .iter()
        .map(|preset| {
            let mut value = json!(preset);
            value["installed"] = json!(separation::installed(&paths.root, preset));
            value
        })
        .collect();
    // Checkpoints download on first use; the configs ship with the project.
    let configs = separation::weights_directory(&paths.root);
    let cover = COVER_PRESETS.iter().all(|id| {
        separation::PRESETS
            .iter()
            .any(|preset| preset.id == *id && configs.join(preset.config).is_file())
    });
    let embedders: Vec<Value> = EMBEDDERS
        .iter()
        .map(|embedder| json!({ "name": embedder.name(), "installed": embedder.directory(paths).is_dir() }))
        .collect();
    Ok(Json(json!({
        "device": device,
        "models": paths.models()?,
        "presets": presets,
        "cover": cover,
        "embedders": embedders,
        "experiments": paths.experiments,
        "workers": workers.clamp(1, 8),
        // Pitch weights download on first use.
        "pitch": {
            "pm": true,
            "rmvpe": true,
            "nmvpe": true,
            "fcpe": true,
        },
        "realtime": cfg!(feature = "realtime"),
        "index": cfg!(feature = "index-training"),
    })))
}

#[derive(Deserialize)]
struct SpeakerQuery {
    speaker: Option<i64>,
}

async fn voice(
    State(state): State<Shared>,
    RoutePath(name): RoutePath<String>,
    Query(query): Query<SpeakerQuery>,
) -> WebResult<Json<Value>> {
    let result = tokio::task::spawn_blocking(move || -> Result<Value> {
        let paths = &state.paths;
        let path = installed_model(paths, &name)?;
        let info = ModelInfo::read(&path)?;
        let speaker = info.speaker(query.speaker)?;
        let index = paths.select_index(&path, Some(speaker));
        let indexes = paths.indexes();
        let gate = info
            .format
            .embedder
            .directory(paths)
            .join("voice_gate.json")
            .is_file();
        Ok(json!({
            "info": info, "speaker": speaker, "index": index, "indexes": indexes, "gate": gate,
        }))
    })
    .await
    .map_err(anyhow::Error::from)??;
    Ok(Json(result))
}

/// A voice as `<name>.zip`: the model and the index that belongs to it.
async fn export_voice(
    State(state): State<Shared>,
    RoutePath(name): RoutePath<String>,
    Query(query): Query<SpeakerQuery>,
) -> WebResult<Response> {
    use std::io::{Seek, Write};
    let (archive, length, file_name) = tokio::task::spawn_blocking(move || -> Result<_> {
        let paths = &state.paths;
        let model = installed_model(paths, &name)?;
        let speaker = ModelInfo::read(&model)?.speaker(query.speaker)?;
        let stem = model
            .file_stem()
            .and_then(|stem| stem.to_str())
            .context("invalid model name")?
            .to_owned();
        let mut members = vec![(format!("{stem}.pth"), model.clone())];
        if let Some(index) = paths.select_index(&model, Some(speaker)) {
            let name = index
                .file_name()
                .and_then(|name| name.to_str())
                .context("invalid index name")?
                .to_owned();
            members.push((name, index));
        }
        let file = tempfile::tempfile_in(state.studio.workspace.path())?;
        let mut zip = zip::ZipWriter::new(std::io::BufWriter::new(file));
        let options = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Stored)
            .large_file(true);
        for (name, path) in members {
            zip.start_file(name, options)?;
            std::io::copy(&mut fs::File::open(&path)?, &mut zip)?;
        }
        let mut file = zip
            .finish()?
            .into_inner()
            .map_err(|error| anyhow::anyhow!("writing the archive: {}", error.error()))?;
        file.flush()?;
        let length = file.metadata()?.len();
        file.rewind()?;
        Ok((file, length, format!("{stem}.zip")))
    })
    .await
    .map_err(anyhow::Error::from)??;
    let ascii: String = file_name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || "._-".contains(c) {
                c
            } else {
                '_'
            }
        })
        .collect();
    let encoded: String = file_name
        .bytes()
        .map(|byte| {
            if byte.is_ascii_alphanumeric() || b"._-".contains(&byte) {
                (byte as char).to_string()
            } else {
                format!("%{byte:02X}")
            }
        })
        .collect();
    Response::builder()
        .header(header::CONTENT_TYPE, "application/zip")
        .header(header::CACHE_CONTROL, "no-store")
        .header(header::CONTENT_LENGTH, length)
        .header(
            header::CONTENT_DISPOSITION,
            format!("attachment; filename=\"{ascii}\"; filename*=UTF-8''{encoded}"),
        )
        .body(axum::body::Body::from_stream(
            tokio_util::io::ReaderStream::new(tokio::fs::File::from_std(archive)),
        ))
        .map_err(|error| anyhow::Error::from(error).into())
}

#[derive(Deserialize)]
struct InspectQuery {
    path: PathBuf,
}

async fn inspect(
    State(state): State<Shared>,
    Query(query): Query<InspectQuery>,
) -> WebResult<Json<Value>> {
    let info = tokio::task::spawn_blocking(move || {
        let path = checkpoint_model(
            &state.paths,
            query.path.to_str().context("invalid model name")?,
        )?;
        ModelInfo::read(&path)
    })
    .await
    .map_err(anyhow::Error::from)??;
    Ok(Json(json!(info)))
}

async fn pretrained(
    State(state): State<Shared>,
    Query(query): Query<library::PretrainedQuery>,
) -> Json<Value> {
    Json(library::choices(&state.paths, &query))
}

#[derive(Deserialize)]
struct ExperimentQuery {
    name: String,
}

async fn experiments(State(state): State<Shared>) -> WebResult<Json<Value>> {
    let result = tokio::task::spawn_blocking(move || -> Result<Value> {
        let mut resumable = Vec::new();
        let mut runs = Vec::new();
        let mut all = Vec::new();
        let logs = &state.paths.experiments;
        if logs.is_dir() {
            for entry in fs::read_dir(logs)? {
                let entry = entry?;
                let path = entry.path();
                if !path.is_dir() {
                    continue;
                }
                let Ok(name) = entry.file_name().into_string() else {
                    continue;
                };
                if name.starts_with('.') || name == "mute" {
                    continue;
                }
                all.push(name.clone());
                if !events::paths(&path)?.is_empty() {
                    runs.push(name.clone());
                }
                if train::has_checkpoint_pair(&path)? {
                    resumable.push(name);
                }
            }
            resumable.sort();
            runs.sort();
            all.sort();
        }
        Ok(json!({ "runs": runs, "resumable": resumable, "all": all }))
    })
    .await
    .map_err(anyhow::Error::from)??;
    Ok(Json(result))
}

async fn resume(
    State(state): State<Shared>,
    Query(query): Query<ExperimentQuery>,
) -> WebResult<Json<Value>> {
    let path = state.paths.experiments.join(files::filename(&query.name)?);
    let info = tokio::task::spawn_blocking(move || train::resume_info(&path))
        .await
        .map_err(anyhow::Error::from)??;
    Ok(Json(json!(info)))
}

async fn metrics(
    State(state): State<Shared>,
    Query(query): Query<ExperimentQuery>,
) -> WebResult<Json<Value>> {
    let path = state.paths.experiments.join(files::filename(&query.name)?);
    let values = tokio::task::spawn_blocking(move || events::scalars(&path))
        .await
        .map_err(anyhow::Error::from)??;
    Ok(Json(json!(values)))
}

async fn dataset(State(state): State<Shared>, form: Multipart) -> WebResult<Json<Value>> {
    let (_, uploads) = files::upload(&state, form).await?;
    let path = tokio::task::spawn_blocking(move || -> Result<PathBuf> {
        let zip = uploads.one()?;
        let extracted = tempfile::TempDir::new_in(state.studio.workspace.path())?;
        let files = files::unzip(zip, extracted.path(), audio::is_audio, || Ok(()))?;
        ensure!(
            !files.is_empty(),
            "the dataset archive contains no audio files"
        );
        let mut entries = fs::read_dir(extracted.path())?;
        let first = entries.next().transpose()?.context("empty dataset")?.path();
        let one_folder = entries.next().is_none() && first.is_dir();
        let root = extracted.keep();
        Ok(if one_folder { first } else { root })
    })
    .await
    .map_err(anyhow::Error::from)??;
    Ok(Json(json!({ "path": path })))
}

#[derive(Deserialize)]
#[serde(tag = "operation", rename_all = "snake_case")]
enum Extra {
    Single {
        #[serde(flatten)]
        request: Convert,
    },
    PitchProfile {
        model: String,
        input: Option<PathBuf>,
        method: PitchMethod,
    },
    Import {
        #[serde(flatten)]
        request: library::Import,
    },
    Pretrained {
        #[serde(flatten)]
        request: library::InstallPair,
    },
    InstallEmbedder {
        embedder: Embedder,
    },
    BuildFeaturePool {
        #[serde(flatten)]
        request: model::timbre::PoolRequest,
    },
    Cover {
        #[serde(flatten)]
        request: CoverJob,
    },
    LayerAug {
        #[serde(flatten)]
        request: LayerAugJob,
    },
    FxProfile {
        input: PathBuf,
        output: PathBuf,
    },
    Resume {
        name: String,
        epochs: usize,
        #[serde(default = "automatic")]
        device: String,
        #[serde(default)]
        fp16: Option<bool>,
        #[serde(default)]
        bf16: Option<bool>,
        #[serde(default)]
        gradient_checkpointing: Option<bool>,
        #[serde(default)]
        gpus: Vec<usize>,
    },
}

impl Extra {
    fn run(
        self,
        state: &StateData,
        uploads: files::Uploads,
        cancelled: &AtomicBool,
    ) -> Result<Value> {
        let report = |message: &str| job_progress(state, cancelled, message);
        let paths = &state.paths;
        match self {
            Self::Single { request } => {
                let mut prepare = |message: &str| job_progress(state, cancelled, message);
                let model = installed_model(paths, &request.model)?;
                crate::assets::voice(paths, &model, request.settings.f0_method, &mut prepare)?;
                ensure!(
                    report("Decoding source").is_continue(),
                    "conversion cancelled"
                );
                let file = uploads.one()?;
                let source = audio::read_mono(file, ANALYSIS_RATE)?;
                ensure!(
                    report("Converting audio").is_continue(),
                    "conversion cancelled"
                );
                let result = super::convert_voice(state, &request, &source, Some(file))?;
                ensure!(
                    report("Saving converted audio").is_continue(),
                    "conversion cancelled"
                );
                let output = tempfile::TempDir::new_in(state.studio.workspace.path())?;
                let file = output.path().join("converted.wav");
                audio::write_audio(&file, &result.samples, result.rate, Format::Wav, false)?;
                let result = state.studio.publish([file], Some(output))?;
                Ok(json!(result[0]))
            }
            Self::PitchProfile {
                model,
                input,
                method,
            } => {
                let model = installed_model(paths, &model)?;
                ensure!(
                    ModelInfo::read(&model)?.uses_f0,
                    "this voice does not use pitch"
                );
                let mut prepare = |message: &str| job_progress(state, cancelled, message);
                crate::assets::pitch(paths, method, &mut prepare)?;
                let source = match uploads.files.len() {
                    0 => paths
                        .root
                        .join(input.context("supply training audio or a dataset folder")?),
                    _ => uploads.one()?.to_owned(),
                };
                *lock(&state.voice)? = None;
                let (path, profile) =
                    model::register::build(paths, &model, &source, method, report)?;
                Ok(json!({ "path": path, "profile": profile }))
            }
            Self::Import { request } => library::import(paths, request, &uploads, report),
            Self::Pretrained { request } => {
                ensure!(uploads.files.is_empty(), "unexpected uploaded files");
                library::install_pretrained(paths, request, state.studio.workspace.path(), report)
            }
            Self::InstallEmbedder { embedder } => {
                ensure!(uploads.files.is_empty(), "unexpected uploaded files");
                Ok(json!({ "path": embedder.install(paths, report)? }))
            }
            Self::BuildFeaturePool { request } => {
                ensure!(uploads.files.is_empty(), "unexpected uploaded files");
                *lock(&state.voice)? = None;
                Ok(json!(model::timbre::build_pool(paths, request, report)?))
            }
            Self::Cover { request } => request.run(state, uploads, report),
            Self::LayerAug { request } => {
                ensure!(uploads.files.is_empty(), "unexpected uploaded files");
                *lock(&state.voice)? = None;
                request.run(state, report)
            }
            Self::FxProfile { input, output } => {
                ensure!(uploads.files.is_empty(), "unexpected uploaded files");
                ensure!(
                    report("Measuring the long-term spectrum").is_continue(),
                    "cancelled"
                );
                let output = paths.root.join(output);
                let (files, profile) = model::fx::measure(&paths.root.join(input))?;
                if let Some(parent) = output.parent() {
                    fs::create_dir_all(parent)?;
                }
                fs::write(&output, serde_json::to_string(&profile)?)?;
                Ok(json!({ "path": output, "files": files }))
            }
            Self::Resume {
                name,
                epochs,
                device,
                fp16,
                bf16,
                gradient_checkpointing,
                gpus,
            } => {
                let experiment = paths.experiments.join(files::filename(&name)?);
                let info = train::resume_info(&experiment)?;
                let mut prepare = |message: &str| job_progress(state, cancelled, message);
                crate::assets::mute(paths, &mut prepare)?;
                ensure!(
                    epochs > info.epoch,
                    "total epochs must exceed the saved epoch {}",
                    info.epoch
                );
                *lock(&state.voice)? = None;
                let request = train::Request {
                    experiment,
                    name: Some(name),
                    epochs,
                    device,
                    gpus,
                    fp16,
                    bf16,
                    gradient_checkpointing,
                    version: info.version,
                    f0: info.f0,
                    sample_rate: info.sample_rate,
                    multi: info.multi,
                    speaker_id: info.speaker_id,
                    batch_size: Some(info.batch_size),
                    save_every: info.save_every,
                    latest_only: info.latest_only,
                    save_weights: info.save_weights,
                    tensorboard: info.tensorboard,
                    multiscale_mel: info.multiscale_mel,
                    stft_loss: info.stft_loss,
                    phonetic_loss: info.phonetic_loss,
                    mrd: info.mrd,
                    ema_decay: info.ema_decay,
                    ..Default::default()
                };
                Ok(json!(train::run(paths, request, report)?))
            }
        }
    }
}

fn full_strength() -> f32 {
    1.0
}

fn enabled() -> bool {
    true
}

fn nonempty(path: &Option<PathBuf>) -> Option<&PathBuf> {
    path.as_ref().filter(|path| !path.as_os_str().is_empty())
}

/// A song (or vocal stem) converted layer by layer, see [`cover`].
#[derive(Deserialize)]
struct CoverJob {
    model: String,
    speaker_id: Option<i64>,
    index: Option<PathBuf>,
    #[serde(default)]
    index_rate: f32,
    #[serde(default)]
    pitch: f32,
    /// A song or a folder of songs on the server; uploads are used when empty.
    #[serde(default)]
    input: Option<PathBuf>,
    /// Folder for the results; a temporary folder when empty.
    #[serde(default)]
    output: Option<PathBuf>,
    #[serde(default)]
    vocals_only: bool,
    /// Instrumental on the server for a single vocal stem.
    #[serde(default)]
    instrumental: Option<PathBuf>,
    /// The second uploaded file is the instrumental of the first.
    #[serde(default)]
    instrumental_upload: bool,
    backing: cover::Backing,
    #[serde(default)]
    dry: bool,
    #[serde(default = "full_strength")]
    aperiodic: f32,
    #[serde(default)]
    voice_gate: f32,
    #[serde(default)]
    vocal_gain: f32,
    format: Format,
    #[serde(default = "automatic")]
    device: String,
}

impl CoverJob {
    fn run(
        self,
        state: &StateData,
        uploads: files::Uploads,
        report: impl FnMut(&str) -> std::ops::ControlFlow<()>,
    ) -> Result<Value> {
        let paths = &state.paths;
        let model = installed_model(paths, &self.model)?;
        let server_instrumental = nonempty(&self.instrumental).map(|path| paths.root.join(path));
        let (inputs, instrumental) = if let Some(input) = nonempty(&self.input) {
            ensure!(
                uploads.files.is_empty(),
                "use an input path or uploads, not both"
            );
            let input = paths.root.join(input);
            let inputs = if input.is_dir() {
                audio::files(&input, false)?
                    .into_iter()
                    .filter(|file| audio::is_audio(file))
                    .collect()
            } else {
                vec![input]
            };
            (inputs, server_instrumental)
        } else {
            let mut inputs = uploads.files.clone();
            let instrumental = if self.instrumental_upload {
                ensure!(
                    inputs.len() == 2,
                    "upload one vocal stem and its instrumental"
                );
                inputs.pop()
            } else {
                server_instrumental
            };
            (inputs, instrumental)
        };
        ensure!(!inputs.is_empty(), "supply a song path or upload audio");
        ensure!(
            inputs.iter().all(|file| audio::is_audio(file)),
            "unsupported audio file extension"
        );
        if instrumental.is_some() {
            ensure!(self.vocals_only, "an instrumental goes with vocal stems");
            ensure!(
                inputs.len() == 1,
                "an instrumental goes with one vocal stem"
            );
        }
        let (output, staging) = match nonempty(&self.output) {
            Some(path) => (paths.root.join(path), None),
            None => {
                let staging = tempfile::TempDir::new_in(state.studio.workspace.path())?;
                (staging.path().to_owned(), Some(staging))
            }
        };
        fs::create_dir_all(&output)?;
        *lock(&state.voice)? = None;
        let request = cover::Request {
            input: inputs,
            output,
            model,
            index: self.index,
            index_rate: self.index_rate,
            pitch: self.pitch,
            speaker_id: self.speaker_id,
            vocals_only: self.vocals_only,
            instrumental: instrumental.into_iter().collect(),
            backing: self.backing,
            dry: self.dry,
            aperiodic: self.aperiodic,
            voice_gate: self.voice_gate,
            vocal_gain: self.vocal_gain,
            device: self.device,
            format: self.format,
        };
        let rendered = cover::render(paths, &request, report)?;
        let media = state.studio.publish(
            rendered
                .iter()
                .flat_map(|item| item.files.iter().map(|(_, path)| path.clone())),
            staging,
        )?;
        let mut media = media.into_iter();
        let groups: Vec<Value> = rendered
            .iter()
            .map(|item| {
                let tracks: Vec<_> = media.by_ref().take(item.files.len()).collect();
                let kinds: Vec<_> = item.files.iter().map(|(kind, _)| *kind).collect();
                let name = item.input.file_name().map(|name| name.to_string_lossy());
                json!({ "name": name, "tracks": tracks, "suffixes": kinds })
            })
            .collect();
        Ok(json!(groups))
    }
}

/// Features to extract on a layered copy once it is written.
#[derive(Deserialize)]
struct ExtractAfter {
    version: String,
    f0_method: Option<PitchMethod>,
    embedder: Option<Embedder>,
    #[serde(default)]
    unvoice_fricatives: bool,
}

/// A layered (and optionally processed) copy of an experiment for `train --layer-aug`.
#[derive(Deserialize)]
struct LayerAugJob {
    experiment: String,
    output: String,
    sample_rate: u32,
    #[serde(default)]
    seed: u64,
    #[serde(default = "enabled")]
    layers: bool,
    #[serde(default)]
    fx: bool,
    #[serde(default)]
    fx_profile: Option<PathBuf>,
    #[serde(default)]
    bleed: Option<PathBuf>,
    #[serde(default)]
    extract: Option<ExtractAfter>,
}

impl LayerAugJob {
    fn run(
        self,
        state: &StateData,
        mut report: impl FnMut(&str) -> std::ops::ControlFlow<()>,
    ) -> Result<Value> {
        let paths = &state.paths;
        let source = paths.experiments.join(files::filename(&self.experiment)?);
        let output = paths.experiments.join(files::filename(&self.output)?);
        ensure!(source.is_dir(), "preprocess the source experiment first");
        ensure!(
            source != output,
            "write the layered copy to another experiment"
        );
        let fx = if self.fx {
            Some(
                nonempty(&self.fx_profile)
                    .map(|path| model::fx::read_profile(&paths.root.join(path)))
                    .transpose()?,
            )
        } else {
            None
        };
        ensure!(
            self.layers || fx.is_some() || nonempty(&self.bleed).is_some(),
            "choose layers, effects or bleed"
        );
        let options = model::layers::Options {
            layers: self.layers,
            fx,
            bleed: nonempty(&self.bleed).map(|path| paths.root.join(path)),
        };
        ensure!(report("Writing layered clips").is_continue(), "cancelled");
        let written =
            model::layers::generate_with(&source, &output, self.sample_rate, self.seed, &options)?;
        let summary = json!({
            "clips": written.clips, "doubled": written.doubled, "backed": written.backed,
            "reverberated": written.reverberated, "processed": written.processed, "bled": written.bled,
        });
        let _ = report(&format!(
            "{} clips: {} doubled, {} with backing, {} with reverb, {} through effects, {} with bleed",
            written.clips,
            written.doubled,
            written.backed,
            written.reverberated,
            written.processed,
            written.bled
        ));
        let features = match self.extract {
            Some(after) => {
                if let Some(method) = after.f0_method {
                    crate::assets::pitch(paths, method, &mut report)?;
                }
                let devices = model::extract::selected_devices("auto", &[])?;
                let request = model::extract::Request {
                    experiment: &output,
                    version: &after.version,
                    f0: after.f0_method,
                    device: devices[0],
                    overwrite: true,
                    unvoice_fricatives: after.unvoice_fricatives,
                };
                Some(json!(model::extract::run_with_embedder(
                    paths,
                    request,
                    &devices,
                    after.embedder,
                    report
                )?))
            }
            None => None,
        };
        Ok(json!({ "path": output, "report": summary, "features": features }))
    }
}

fn native(
    mut action: Action,
    state: &StateData,
    uploads: files::Uploads,
    cancelled: &AtomicBool,
) -> Result<Value> {
    let report = |message: &str| job_progress(state, cancelled, message);
    match &mut action {
        Action::Batch { input, .. } | Action::Separate { input, .. }
            if input.as_os_str().is_empty() =>
        {
            ensure!(
                !uploads.files.is_empty(),
                "supply an input path or upload audio"
            );
            ensure!(
                uploads.files.iter().all(|file| audio::is_audio(file)),
                "unsupported audio file extension"
            );
            *input = uploads.directory.path().to_owned();
        }
        _ => ensure!(uploads.files.is_empty(), "unexpected uploaded files"),
    }
    let preset = match &action {
        Action::Separate { preset, .. } => {
            separation::PRESETS.iter().find(|item| item.id == preset)
        }
        _ => None,
    };
    let is_batch = matches!(action, Action::Batch { .. });
    if !is_batch {
        *lock(&state.voice)? = None;
    }
    let result = action.run(state, cancelled, report)?;
    if let Some(preset) = preset {
        let stems: Vec<StemFiles> = serde_json::from_value(result)?;
        let paths = stems
            .iter()
            .flat_map(|file| [file.primary.clone(), file.secondary.clone()]);
        let media = state.studio.publish(paths, None)?;
        let results: Vec<_> = stems
            .iter()
            .zip(media.as_chunks::<2>().0)
            .map(|(file, tracks)| {
                let name = file.input.file_name().map(|name| name.to_string_lossy());
                json!({"name": name, "tracks": tracks, "suffixes": preset.suffixes})
            })
            .collect();
        Ok(json!(results))
    } else if is_batch {
        let paths: Vec<PathBuf> = serde_json::from_value(result)?;
        Ok(json!(state.studio.publish(paths, None)?))
    } else {
        Ok(result)
    }
}

async fn start(State(state): State<Shared>, form: Multipart) -> WebResult<Response> {
    let (settings, uploads) = files::upload(&state, form).await?;
    let operation = settings["operation"]
        .as_str()
        .context("job operation is required")?
        .to_owned();
    let id = match operation.as_str() {
        "single" | "pitch_profile" | "import" | "pretrained" | "install_embedder"
        | "build_feature_pool" | "resume" | "cover" | "layer_aug" | "fx_profile" => {
            let action: Extra = serde_json::from_value(settings).map_err(anyhow::Error::from)?;
            launch_job(state, &operation, move |state, cancelled| {
                action.run(state, uploads, cancelled)
            })?
        }
        _ => {
            let action: Action = serde_json::from_value(settings).map_err(anyhow::Error::from)?;
            launch_job(state, &operation, move |state, cancelled| {
                native(action, state, uploads, cancelled)
            })?
        }
    };
    Ok(match id {
        Some(id) => (StatusCode::ACCEPTED, Json(json!({ "id": id }))).into_response(),
        None => (StatusCode::CONFLICT, Json(json!({ "error": "Stop the active audio stream or wait for the current task to finish." }))).into_response(),
    })
}

async fn job(State(state): State<Shared>, RoutePath(id): RoutePath<Uuid>) -> WebResult<Response> {
    let job = lock(&state.job)?;
    if job.id == id {
        return Ok(Json(json!(*job)).into_response());
    }
    let recent = lock(&state.recent_jobs)?;
    Ok(recent.iter().find(|job| job.id == id).map_or_else(
        || StatusCode::NOT_FOUND.into_response(),
        |job| Json(json!(job)).into_response(),
    ))
}

async fn cancel(
    State(state): State<Shared>,
    RoutePath(id): RoutePath<Uuid>,
) -> WebResult<StatusCode> {
    let job = lock(&state.job)?;
    if job.id != id {
        return Ok(StatusCode::NOT_FOUND);
    }
    if job.status == JobState::Running {
        job.cancelled.store(true, Ordering::Relaxed);
    }
    Ok(StatusCode::NO_CONTENT)
}
