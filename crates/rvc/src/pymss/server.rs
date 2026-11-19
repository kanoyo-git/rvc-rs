use std::collections::HashMap;
use std::net::IpAddr;
use std::path::{Path, PathBuf};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};
use std::time::Duration;

use anyhow::{Context, Result, ensure};
use axum::body::Bytes;
use axum::extract::{DefaultBodyLimit, Path as RoutePath, Query, State};
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router, middleware};
use clap::Args;
use serde::Deserialize;
use serde_json::{Value, json};
use tch::Device;
use tokio::sync::{OwnedRwLockReadGuard, RwLock, RwLockReadGuard};

use super::catalog::{self, Entry, Source};
use super::compute_devices;
use crate::catalog::Paths;
use crate::separation::{ConfiguredModel, InferenceParams};

mod contract;
mod pcm_api;
mod webui;

const MODEL_OPERATION_IN_PROGRESS: &str = "model_operation_in_progress";

#[derive(Args)]
pub struct Cli {
    model: Option<String>,
    #[arg(long)]
    model_dir: Option<PathBuf>,
    #[arg(long, value_enum, default_value = "modelscope")]
    source: Source,
    #[arg(long)]
    endpoint: Option<String>,
    #[arg(long, default_value = "auto")]
    device: String,
    #[arg(long = "device-id")]
    device_ids: Vec<usize>,
    #[arg(long, default_value = "127.0.0.1")]
    host: IpAddr,
    #[arg(long, default_value_t = 8000)]
    port: u16,
    #[arg(long)]
    api_key: Option<String>,
    #[arg(long, default_value_t = 600.0)]
    max_audio_seconds: f64,
    #[arg(long, default_value_t = 536_870_912)]
    max_request_bytes: usize,
    #[arg(long, default_value_t = 8)]
    max_queue_size: usize,
    #[arg(long, default_value_t = 0.0)]
    request_timeout_seconds: f64,
    #[arg(long)]
    webui: bool,
    /// Log resolved model paths and inference settings when loading a model.
    #[arg(long)]
    debug: bool,
    #[arg(long = "param", value_name = "KEY=VALUE")]
    params: Vec<String>,
}

#[derive(Clone)]
struct Remote {
    source: Source,
    endpoint: Option<String>,
}

struct Service {
    options: Cli,
    devices: (Device, Vec<Device>),
    params: InferenceParams,
    root: PathBuf,
    directory: PathBuf,
    remote: Mutex<Remote>,
    operation: Mutex<()>,
    current: Arc<RwLock<Option<Loaded>>>,
    loading: AtomicBool,
    downloading: AtomicBool,
    queue: Arc<tokio::sync::Semaphore>,
}

struct Loaded {
    id: String,
    entry: Entry,
    engine: Mutex<ConfiguredModel>,
    rate: u32,
    channels: usize,
    instruments: Vec<String>,
    device: String,
}

impl Service {
    fn reserve_operation(&self) -> Api<std::sync::MutexGuard<'_, ()>> {
        self.operation.try_lock().map_err(|error| match error {
            std::sync::TryLockError::Poisoned(_) => {
                anyhow::anyhow!("PyMSS service state is unavailable").into()
            }
            std::sync::TryLockError::WouldBlock if self.downloading.load(Ordering::Acquire) => {
                Failure::new(
                    StatusCode::CONFLICT,
                    "model_download_in_progress",
                    "A model download operation is in progress",
                )
            }
            std::sync::TryLockError::WouldBlock => Failure::new(
                StatusCode::CONFLICT,
                "model_operation_in_progress",
                "A model load or switch operation is in progress",
            ),
        })
    }

    fn visible_model(&self) -> Option<RwLockReadGuard<'_, Loaded>> {
        if self.loading.load(Ordering::Acquire) {
            return None;
        }
        RwLockReadGuard::try_map(self.current.try_read().ok()?, Option::as_ref).ok()
    }

    fn use_model(&self) -> Api<OwnedRwLockReadGuard<Option<Loaded>, Loaded>> {
        let changing = || {
            Failure::new(
                StatusCode::CONFLICT,
                MODEL_OPERATION_IN_PROGRESS,
                "A model load or switch operation is in progress",
            )
        };
        if self.loading.load(Ordering::Acquire) {
            return Err(changing());
        }
        let current = Arc::clone(&self.current)
            .try_read_owned()
            .map_err(|_| changing())?;
        if self.loading.load(Ordering::Acquire) {
            return Err(changing());
        }
        OwnedRwLockReadGuard::try_map(current, Option::as_ref).map_err(|_| {
            Failure::new(
                StatusCode::SERVICE_UNAVAILABLE,
                "model_not_loaded",
                "Load a PyMSS model first",
            )
        })
    }
}

pub fn serve(paths: &Paths, options: Cli) -> Result<()> {
    ensure!(
        options.max_queue_size <= 256 && options.max_request_bytes > 0,
        "invalid request or queue limits"
    );
    ensure!(
        options.max_audio_seconds.is_finite() && options.max_audio_seconds >= 0.0,
        "invalid maximum audio duration"
    );
    Duration::try_from_secs_f64(options.request_timeout_seconds)?;
    let directory = catalog::model_directory(&paths.root, options.model_dir.as_deref());
    let remote = Remote {
        source: options.source,
        endpoint: options.endpoint.clone(),
    };
    remote.source.url("", remote.endpoint.as_deref())?;
    let mut params = InferenceParams::parse(&options.params)?;
    let devices = compute_devices(&options.device, &options.device_ids, &mut params)?;
    let initial = options
        .model
        .as_ref()
        .map(|model| load_model(model, &directory, &remote, &devices, &params, options.debug))
        .transpose()?;
    let state = Arc::new(Service {
        directory,
        params,
        root: paths.root.clone(),
        remote: Mutex::new(remote),
        operation: Mutex::new(()),
        current: Arc::new(RwLock::new(initial)),
        loading: AtomicBool::new(false),
        downloading: AtomicBool::new(false),
        queue: Arc::new(tokio::sync::Semaphore::new(options.max_queue_size + 1)),
        devices,
        options,
    });
    let app = Router::new()
        .route("/v1/models", get(list_loaded))
        .route("/v1/models/{model}", get(get_loaded))
        .route("/v1/catalog/models", get(list_catalog))
        .route("/v1/catalog/models/{model}", get(get_catalog))
        .route("/v1/models/load", post(load))
        .route("/v1/models/download", post(download))
        .route("/v1/download-source", get(remote_source).post(set_source))
        .route("/v1/server/info", get(info))
        .route("/v1/audio/separations", post(pcm_api::separate))
        .layer(DefaultBodyLimit::max(state.options.max_request_bytes))
        .route_layer(middleware::from_fn_with_state(
            Arc::clone(&state),
            authenticated,
        ))
        .route("/health", get(health));
    let app = contract::add_routes(app);
    let app = if state.options.webui {
        webui::add_routes(app)
    } else {
        app
    };
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    runtime.block_on(async {
        let listener =
            tokio::net::TcpListener::bind((state.options.host, state.options.port)).await?;
        println!("http://{}", listener.local_addr()?);
        axum::serve(listener, app.with_state(state))
            .with_graceful_shutdown(async {
                let _ = tokio::signal::ctrl_c().await;
            })
            .await?;
        Ok(())
    })
}

type Shared = Arc<Service>;

async fn authenticated(
    State(state): State<Shared>,
    request: axum::extract::Request,
    next: middleware::Next,
) -> Response {
    if let Some(key) = state
        .options
        .api_key
        .as_deref()
        .filter(|value| !value.is_empty())
    {
        let supplied = request
            .headers()
            .get(header::AUTHORIZATION)
            .and_then(|header| header.to_str().ok())
            .and_then(|header| header.strip_prefix("Bearer "));
        if supplied != Some(key) {
            return Failure::new(
                StatusCode::UNAUTHORIZED,
                "invalid_api_key",
                "Invalid or missing bearer token",
            )
            .into_response();
        }
    }
    next.run(request).await
}

async fn health(State(state): State<Shared>) -> Api<Json<Value>> {
    let active = state.visible_model();
    Ok(Json(json!({
        "status": "ok", "model_loaded": active.is_some(), "model_loading": state.loading.load(Ordering::Relaxed),
        "model": active.as_ref().map(|model| &model.id), "device": active.as_ref().map(|model| &model.device),
    })))
}

async fn list_loaded(State(state): State<Shared>) -> Api<Json<Value>> {
    let data: Vec<_> = state
        .visible_model()
        .into_iter()
        .map(|model| model.card())
        .collect();
    Ok(Json(json!({"object":"list","data":data})))
}

async fn get_loaded(
    State(state): State<Shared>,
    RoutePath(model): RoutePath<String>,
) -> Api<Json<Value>> {
    let active = state.visible_model();
    Ok(Json(requested_model(active.as_deref(), &model)?.card()))
}

#[derive(Default, Deserialize)]
struct CatalogQuery {
    category: Option<String>,
    supported: Option<String>,
    local: Option<String>,
    q: Option<String>,
    include_files: Option<String>,
    source: Option<Source>,
    endpoint: Option<String>,
}

async fn list_catalog(
    State(state): State<Shared>,
    Query(request): Query<CatalogQuery>,
) -> Api<Json<Value>> {
    let supported = match request
        .supported
        .as_deref()
        .unwrap_or("true")
        .to_lowercase()
        .as_str()
    {
        "all" => None,
        "true" | "1" | "yes" | "on" => Some(true),
        "false" | "0" | "no" | "off" => Some(false),
        _ => {
            return Err(Failure::bad(
                "invalid_request",
                "supported must be true, false, or all",
            ));
        }
    };
    let local = request.local.as_deref().unwrap_or("all");
    if !matches!(local, "all" | "complete" | "missing") {
        return Err(Failure::bad(
            "invalid_request",
            "local must be all, complete, or missing",
        ));
    }
    let include_files = matches!(
        request.include_files.as_deref(),
        Some("1" | "true" | "yes" | "on")
    );
    let remote = effective_remote(&state, request.source, request.endpoint.map(Some))?;
    let query = request.q.unwrap_or_default().trim().to_lowercase();
    let mut data = Vec::new();
    for entry in catalog::entries(request.category.as_deref(), supported)? {
        if !query.is_empty()
            && !std::iter::once(&entry.name)
                .chain([&entry.architecture, &entry.target_stem])
                .chain(&entry.aliases)
                .any(|word| word.to_lowercase().contains(&query))
        {
            continue;
        }
        let card = catalog_card(&entry, &state.directory, &remote, include_files)?;
        let complete = card["pymss"]["local"]["complete"]
            .as_bool()
            .unwrap_or(false);
        if local == "complete" && !complete || local == "missing" && complete {
            continue;
        }
        data.push(card);
    }
    Ok(Json(json!({
        "object": "list", "data": data, "pymss": { "source": remote.source, "endpoint": remote.endpoint, "total": data.len() }
    })))
}

async fn get_catalog(
    State(state): State<Shared>,
    RoutePath(name): RoutePath<String>,
    Query(request): Query<CatalogQuery>,
) -> Api<Json<Value>> {
    let remote = effective_remote(&state, request.source, request.endpoint.map(Some))?;
    let entry = entry(&name)?;
    Ok(Json(catalog_card(&entry, &state.directory, &remote, true)?))
}

fn catalog_card(
    entry: &Entry,
    directory: &Path,
    remote: &Remote,
    include_files: bool,
) -> Result<Value> {
    let mut files = Vec::new();
    let mut missing = 0;
    for file in entry.files(directory)? {
        let exists = file.path.is_file();
        missing += usize::from(!exists);
        if include_files {
            files.push(json!({
                "role": file.role, "relpath": file.relative, "exists": exists,
                "size_bytes": file.path.metadata().map_or(0, |metadata| metadata.len()),
                "remote_url": remote.source.url(&file.relative, remote.endpoint.as_deref())?,
            }));
        }
    }
    let mut model = json!({
        "name": entry.name, "aliases": entry.aliases, "model_type": entry.model_type, "architecture": entry.architecture,
        "category": entry.categories(), "primary_category": entry.primary_category, "secondary_category": entry.secondary_category,
        "target_stem": entry.target_stem, "supported": entry.supported, "unsupported_reason": entry.unsupported_reason,
        "size_bytes": entry.size_bytes, "local": { "complete": missing == 0, "missing_count": missing },
        "remote": { "available": true, "source": remote.source, "endpoint": remote.endpoint }
    });
    if include_files {
        model["files"] = json!(files);
    }
    Ok(
        json!({"id": entry.name, "object": "pymss.model_catalog_entry", "owned_by": "pymss", "pymss": model}),
    )
}

#[derive(Deserialize)]
struct ModelRequest {
    model: String,
    source: Option<Source>,
    #[serde(default, deserialize_with = "explicit_endpoint")]
    endpoint: Option<Option<String>>,
    #[serde(default)]
    inference_params: Option<Value>,
}

fn explicit_endpoint<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> std::result::Result<Option<Option<String>>, D::Error> {
    Option::<String>::deserialize(deserializer).map(Some)
}

async fn load(State(state): State<Shared>, Json(request): Json<ModelRequest>) -> Api<Json<Value>> {
    let entry = entry(&request.model)?;
    let requested = request
        .inference_params
        .map(serde_json::from_value::<InferenceParams>)
        .transpose()
        .map_err(|_| {
            Failure::bad(
                "invalid_inference_parameter",
                "inference_params must be an object",
            )
        })?;
    let params = state.params.overlay(&requested.unwrap_or_default());
    let model_type = entry
        .model_type
        .as_deref()
        .context("catalog entry has no model type")?;
    params
        .validate(model_type)
        .map_err(|error| Failure::bad("invalid_inference_parameter", error.to_string()))?;
    let (previous, card) = tokio::task::spawn_blocking(move || {
        let _operation = state.reserve_operation()?;
        let remote = effective_remote(&state, request.source, request.endpoint)?;
        replace_model(&state.current, &state.loading, || {
            load_model(
                &request.model,
                &state.directory,
                &remote,
                &state.devices,
                &params,
                state.options.debug,
            )
        })
    })
    .await
    .map_err(anyhow::Error::from)??;
    let answer = json!({"object":"model.load","previous_model_loaded":previous,"model_loaded":true,"model":card});
    Ok(Json(answer))
}

#[derive(Deserialize)]
struct DownloadRequest {
    model: String,
    source: Option<Source>,
    #[serde(default, deserialize_with = "explicit_endpoint")]
    endpoint: Option<Option<String>>,
    force: Option<Value>,
    verify: Option<Value>,
    #[serde(default = "default_download_timeout", alias = "timeout")]
    timeout_seconds: Value,
}

fn default_download_timeout() -> Value {
    json!(30)
}

fn download_flag(value: Option<&Value>, name: &'static str, default: bool) -> Api<bool> {
    match value {
        None | Some(Value::Null) => return Ok(default),
        Some(Value::Bool(value)) => return Ok(*value),
        Some(Value::String(text)) => match text.trim().to_ascii_lowercase().as_str() {
            "true" | "yes" | "1" | "on" => return Ok(true),
            "false" | "no" | "0" | "off" => return Ok(false),
            _ => {}
        },
        _ => {}
    }
    Err(Failure::bad("invalid_request", format!("{name} must be a boolean")).parameter(name))
}

async fn download(
    State(state): State<Shared>,
    Json(request): Json<DownloadRequest>,
) -> Api<Json<Value>> {
    let entry = entry(&request.model)?;
    let force = download_flag(request.force.as_ref(), "force", false)?;
    let verify = download_flag(request.verify.as_ref(), "verify", true)?;
    let timeout = request
        .timeout_seconds
        .as_f64()
        .or_else(|| request.timeout_seconds.as_str()?.trim().parse().ok())
        .ok_or_else(|| {
            Failure::bad("invalid_request", "timeout_seconds must be a number")
                .parameter("timeout_seconds")
        })?;
    if !timeout.is_finite() || timeout <= 0.0 {
        return Err(
            Failure::bad("invalid_request", "timeout_seconds must be greater than 0")
                .parameter("timeout_seconds"),
        );
    }
    let answer = tokio::task::spawn_blocking(move || -> Api<Value> {
        let _operation = state.reserve_operation()?;
        let _guard = Busy::start(&state.downloading, "model_download_in_progress")?;
        let remote = effective_remote(&state, request.source, request.endpoint)?;
        let report = catalog::download_with_options(&entry, &state.directory, catalog::DownloadOptions {
            source: remote.source, endpoint: remote.endpoint.as_deref(), force, verify, timeout_seconds: timeout,
        })?;
        let relative = |paths: Vec<PathBuf>| paths.into_iter().map(|path| path.strip_prefix(&state.directory).unwrap_or(&path).to_owned()).collect::<Vec<_>>();
        Ok(json!({
            "object": "model.download", "model": catalog_card(&entry, &state.directory, &remote, false)?,
            "source": remote.source, "endpoint": remote.endpoint,
            "downloaded": relative(report.downloaded), "skipped": relative(report.skipped),
        }))
    }).await.map_err(anyhow::Error::from)??;
    Ok(Json(answer))
}

async fn remote_source(State(state): State<Shared>) -> Api<Json<Value>> {
    let remote = lock(&state.remote)?;
    Ok(Json(
        json!({"object":"download.source","source":remote.source,"endpoint":remote.endpoint}),
    ))
}

#[derive(Deserialize)]
struct RemoteUpdate {
    source: Source,
    #[serde(default, deserialize_with = "explicit_endpoint")]
    endpoint: Option<Option<String>>,
}

async fn set_source(
    State(state): State<Shared>,
    Json(update): Json<RemoteUpdate>,
) -> Api<Json<Value>> {
    let _operation = state.reserve_operation()?;
    let mut remote = lock(&state.remote)?;
    let endpoint = update.endpoint.unwrap_or_else(|| remote.endpoint.clone());
    update.source.url("", endpoint.as_deref())?;
    *remote = Remote {
        source: update.source,
        endpoint,
    };
    Ok(Json(
        json!({"object":"download.source","source":remote.source,"endpoint":remote.endpoint}),
    ))
}

async fn info(State(state): State<Shared>) -> Api<Json<Value>> {
    let remote = lock(&state.remote)?;
    let active = state.visible_model();
    let o = &state.options;
    Ok(Json(json!({
        "object": "server.info",
        "webui": { "enabled": o.webui, "path": o.webui.then_some("/ui/") },
        "auth": { "api_key_required": o.api_key.as_ref().is_some_and(|key| !key.is_empty()) },
        "limits": { "max_audio_seconds": o.max_audio_seconds, "max_request_bytes": o.max_request_bytes,
                    "max_queue_size": o.max_queue_size, "request_timeout_seconds": o.request_timeout_seconds },
        "download_source": { "source": remote.source, "endpoint": remote.endpoint },
        "model_loaded": active.is_some(), "model_loading": state.loading.load(Ordering::Relaxed),
        "model": active.as_ref().map(|item| item.card()),
    })))
}

fn effective_remote(
    state: &Service,
    source: Option<Source>,
    endpoint: Option<Option<String>>,
) -> Api<Remote> {
    let base = lock(&state.remote)?;
    let remote = Remote {
        source: source.unwrap_or(base.source),
        endpoint: endpoint.unwrap_or_else(|| base.endpoint.clone()),
    };
    remote.source.url("", remote.endpoint.as_deref())?;
    Ok(remote)
}

fn load_model(
    name: &str,
    directory: &Path,
    remote: &Remote,
    devices: &(Device, Vec<Device>),
    params: &InferenceParams,
    debug: bool,
) -> Result<Loaded> {
    let entry = catalog::by_name(name)?;
    ensure!(entry.supported, "{}", entry.unsupported_reason);
    let kind = entry
        .model_type
        .as_deref()
        .context("catalog entry has no model type")?;
    params.validate(kind)?;
    let (checkpoint, config) = entry.model_paths(directory)?;
    if !checkpoint.is_file() || !config.is_file() {
        catalog::download(
            &entry,
            directory,
            remote.source,
            remote.endpoint.as_deref(),
            false,
            1800,
        )?;
    }
    let (device, additional) = devices;
    let label = params.device_label(*device);
    if debug {
        eprintln!(
            "PyMSS {name} ({kind}) on {label} {additional:?}, weights {}, config {}, overrides {params:?}",
            checkpoint.display(),
            config.display(),
        );
    }
    let engine =
        ConfiguredModel::load_on_devices(kind, &checkpoint, &config, *device, additional, params)?;
    let id = Path::new(&entry.name)
        .file_stem()
        .and_then(|name| name.to_str())
        .unwrap_or(&entry.name)
        .to_owned();
    Ok(Loaded {
        id,
        entry,
        rate: engine.rate(),
        channels: engine.channels(),
        instruments: engine.stems(),
        device: label,
        engine: Mutex::new(engine),
    })
}

impl Loaded {
    fn card(&self) -> Value {
        let parameters =
            InferenceParams::supported(self.entry.model_type.as_deref().unwrap_or_default());
        let (audio, inference): (Vec<_>, Vec<_>) =
            parameters.into_iter().partition(|key| *key == "chunk_size");
        json!({
            "id": self.id, "object": "model", "created": 0, "owned_by": "pymss",
            "pymss": { "catalog_name": self.entry.name, "model_type": self.entry.model_type, "architecture": self.entry.architecture,
                       "category": self.entry.categories(), "catalog_target_stem": self.entry.target_stem, "supported": self.entry.supported,
                       "sample_rate": self.rate, "instruments": self.instruments,
                       "instruments_source": "separator.config.training.instruments",
                       "supported_parameters": {"audio": audio, "inference": inference} }
        })
    }
}

fn requested_model<'a>(active: Option<&'a Loaded>, name: &str) -> Api<&'a Loaded> {
    let candidate = active.filter(|model| {
        model.id.eq_ignore_ascii_case(name)
            || model.entry.name.eq_ignore_ascii_case(name)
            || model
                .entry
                .aliases
                .iter()
                .any(|alias| alias.eq_ignore_ascii_case(name))
    });
    candidate.ok_or_else(|| {
        Failure::new(
            StatusCode::NOT_FOUND,
            "model_not_found",
            format!("model {name:?} is not loaded"),
        )
    })
}

fn replace_model(
    current: &RwLock<Option<Loaded>>,
    loading: &AtomicBool,
    load: impl FnOnce() -> Result<Loaded>,
) -> Api<(bool, Value)> {
    let _loading = Busy::start(loading, MODEL_OPERATION_IN_PROGRESS)?;
    let mut current = current.blocking_write();
    let previous = current.is_some();
    drop(current.take());
    let model = load()?;
    let card = model.card();
    *current = Some(model);
    Ok((previous, card))
}

fn entry(name: &str) -> Api<Entry> {
    catalog::by_name(name)
        .map_err(|error| Failure::new(StatusCode::NOT_FOUND, "model_not_found", error.to_string()))
}

struct Busy<'a>(&'a AtomicBool);
impl<'a> Busy<'a> {
    fn start(flag: &'a AtomicBool, message: &'static str) -> Api<Self> {
        if flag.swap(true, Ordering::AcqRel) {
            Err(Failure::new(
                StatusCode::CONFLICT,
                message,
                "a model operation is already in progress",
            ))
        } else {
            Ok(Self(flag))
        }
    }
}
impl Drop for Busy<'_> {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

fn lock<T>(mutex: &Mutex<T>) -> Result<std::sync::MutexGuard<'_, T>> {
    mutex
        .lock()
        .map_err(|_| anyhow::anyhow!("PyMSS service state is unavailable"))
}

type Api<T> = std::result::Result<T, Failure>;

struct Failure {
    status: StatusCode,
    code: &'static str,
    message: String,
    parameter: Option<&'static str>,
}
impl Failure {
    fn new(status: StatusCode, code: &'static str, message: impl Into<String>) -> Self {
        Self {
            status,
            code,
            message: message.into(),
            parameter: None,
        }
    }
    fn bad(code: &'static str, message: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_REQUEST, code, message)
    }
    fn parameter(mut self, name: &'static str) -> Self {
        self.parameter = Some(name);
        self
    }
}
impl From<anyhow::Error> for Failure {
    fn from(error: anyhow::Error) -> Self {
        Self::bad("invalid_request", format!("{error:#}"))
    }
}
impl IntoResponse for Failure {
    fn into_response(self) -> Response {
        let kind = if self.status.is_server_error() {
            "server_error"
        } else {
            "invalid_request_error"
        };
        (
            self.status,
            Json(
                json!({"error":{"message":self.message,"code":self.code,"type":kind,"param":self.parameter}}),
            ),
        )
            .into_response()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc::{self, RecvTimeoutError};
    use std::thread;
    use std::time::Instant;

    use super::*;

    #[test]
    fn replacement_waits_for_active_and_queued_pcm_workers() {
        let current = Arc::new(RwLock::new(None));
        let loading = AtomicBool::new(false);
        thread::scope(|threads| {
            let running = Arc::clone(&current).try_read_owned().unwrap();
            let queued = Arc::clone(&current).try_read_owned().unwrap();
            let (load_started, construction) = mpsc::sync_channel(1);
            let slot = &*current;
            let busy = &loading;
            let replacement = threads.spawn(move || {
                replace_model(slot, busy, || {
                    load_started.send(()).unwrap();
                    anyhow::bail!("test load failed")
                })
            });
            let deadline = Instant::now() + Duration::from_secs(5);
            while !loading.load(Ordering::Acquire) {
                assert!(
                    Instant::now() < deadline,
                    "replacement did not reserve the slot"
                );
                thread::yield_now();
            }

            assert_eq!(construction.try_recv(), Err(mpsc::TryRecvError::Empty));
            drop(running);
            assert_eq!(
                construction.recv_timeout(Duration::from_millis(20)),
                Err(RecvTimeoutError::Timeout)
            );
            drop(queued);

            construction.recv_timeout(Duration::from_secs(5)).unwrap();
            assert!(replacement.join().unwrap().is_err());
        });
        assert!(!loading.load(Ordering::Acquire));
        assert!(current.try_read().unwrap().is_none());
    }
}
