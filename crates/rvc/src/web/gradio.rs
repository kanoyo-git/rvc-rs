//! Compatibility with the Gradio 3.14 wire protocol of the original WebUI.
mod callbacks;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{
    Arc, LazyLock, Mutex, Weak,
    atomic::{AtomicBool, Ordering},
};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, ensure};
use axum::Json;
use axum::Router;
use axum::extract::{
    DefaultBodyLimit, Path, Request, State, WebSocketUpgrade,
    ws::{Message, WebSocket},
};
use axum::http::{Method, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use serde::Deserialize;
use serde_json::{Value, json};
use tempfile::TempDir;
use tokio::sync::{Semaphore, mpsc};
use tower_http::services::ServeFile;

use super::{Shared, lock};
use rvc::model::weights;

const MAX_MESSAGE: usize = 192 * 1024 * 1024; // Includes base64 audio.
const QUEUE_SIZE: usize = 1022;
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(30);

// Generated from the original application's get_config_file(). The component
// IDs and order of dependencies are part of the legacy API.
static CONFIG: LazyLock<Value> = LazyLock::new(|| {
    serde_json::from_str(include_str!("gradio/config.json")).expect("embedded Gradio configuration")
});

type RequestId = (String, usize);

pub(super) struct Legacy {
    pub(super) voice: Mutex<Option<String>>,
    manifests: Mutex<HashMap<String, callbacks::manifest::Picker>>,
    requests: Mutex<HashMap<RequestId, Weak<AtomicBool>>>,
    files: TempDir,
    colab: bool,
    queue: Semaphore,
    worker: Semaphore,
}

impl Legacy {
    pub(super) fn new(colab: bool) -> Result<Self> {
        Ok(Self {
            voice: Mutex::default(),
            manifests: Mutex::default(),
            requests: Mutex::default(),
            files: tempfile::Builder::new().prefix("rvc-gradio-").tempdir()?,
            colab,
            queue: Semaphore::new(QUEUE_SIZE + 1),
            worker: Semaphore::new(1),
        })
    }

    fn waiting(&self) -> usize {
        let occupied = QUEUE_SIZE + 1 - self.queue.available_permits();
        occupied.saturating_sub(usize::from(self.worker.available_permits() == 0))
    }

    fn open_path(&self, name: &str) -> Option<PathBuf> {
        let (id, extension) = name.rsplit_once('.')?;
        if id.len() == 36 && uuid::Uuid::parse_str(id).is_ok() && extension == "wav" {
            let path = self.files.path().join(name);
            return path.is_file().then_some(path);
        }
        None
    }
}

pub(super) fn routes() -> Router<Shared> {
    Router::new()
        .route("/config", get(config))
        .route("/config/", get(config))
        .route("/api/{name}", post(predict))
        .route("/api/{name}/", post(predict))
        .route("/run/{name}", post(predict))
        .route("/run/{name}/", post(predict))
        .route("/queue/join", get(join))
        .route("/queue/status", get(queue_status))
        .route("/reset", post(reset))
        .route("/reset/", post(reset))
        .route("/file/{name}", get(file))
        .fallback(legacy_file)
        .layer(DefaultBodyLimit::max(MAX_MESSAGE))
}

async fn config(State(state): State<Shared>) -> Result<Json<Value>, super::WebError> {
    let mut config = CONFIG.clone();
    config["is_colab"] = json!(state.legacy.colab);
    let models = state.paths.models()?;
    let workers = std::thread::available_parallelism().map_or(1, usize::from);
    let profiles = weights::cuda_profiles();
    let gpus: Vec<_> = profiles
        .iter()
        .enumerate()
        .filter_map(|(id, profile)| {
            Some((id, profile.as_ref().filter(|profile| profile.eligible())?))
        })
        .collect();
    let gpu_ids = gpus
        .iter()
        .map(|(id, _)| id.to_string())
        .collect::<Vec<_>>()
        .join("-");
    let gpu = !gpus.is_empty();
    for component in config["components"]
        .as_array_mut()
        .expect("Gradio components array")
    {
        match component["id"].as_u64() {
            Some(6) => component["props"]["choices"] = json!(models),
            Some(93) => {
                component["props"]["maximum"] = json!(workers);
                component["props"]["value"] = json!((2 * workers).div_ceil(3));
            }
            Some(119 | 120) if gpu => {
                component["props"]["value"] = json!(format!("{gpu_ids}-{gpu_ids}"));
                component["props"]["visible"] = json!(true);
            }
            Some(123) if gpu => {
                let information = gpus
                    .iter()
                    .map(|(id, profile)| format!("{id}\t{}", profile.name))
                    .collect::<Vec<_>>()
                    .join("\n");
                component["props"]["value"] = json!(information);
                component["props"]["visible"] = json!(true);
            }
            Some(124) if gpu => component["props"]["value"] = json!("rmvpe"),
            Some(137) if gpu => {
                let batch = gpus
                    .iter()
                    .map(|(_, profile)| profile.memory_gib as u32 / 2)
                    .min()
                    .unwrap_or(1);
                component["props"]["value"] = json!(batch.max(1));
            }
            Some(145) if gpu => component["props"]["value"] = json!(gpu_ids),
            Some(143 | 144) => {
                if let Some(path) = component["props"]["value"].as_str()
                    && !state.paths.root.join(path).is_file()
                {
                    component["props"]["value"] = json!("");
                }
            }
            _ => {}
        }
    }
    Ok(Json(config))
}

#[derive(Deserialize)]
struct Predict {
    #[serde(default)]
    data: Vec<Value>,
    fn_index: Option<usize>,
    session_hash: Option<String>,
}

#[derive(Deserialize)]
struct Reset {
    session_hash: String,
    fn_index: usize,
}

impl Predict {
    fn endpoint(&self, name: &str) -> Result<usize> {
        let dependencies = CONFIG["dependencies"]
            .as_array()
            .expect("Gradio dependencies");
        let index = self
            .fn_index
            .or_else(|| {
                dependencies
                    .iter()
                    .position(|dep| dep["api_name"].as_str() == Some(name))
            })
            .with_context(|| format!("This app has no endpoint /api/{name}/."))?;
        ensure!(
            index < dependencies.len(),
            "unknown Gradio function index {index}"
        );
        Ok(index)
    }
}

fn output(data: Vec<Value>, started: Instant, generating: bool) -> Value {
    let elapsed = started.elapsed().as_secs_f64();
    json!({
        "data":data, "is_generating":generating, "duration":elapsed, "average_duration":elapsed,
    })
}

fn error_response(error: anyhow::Error) -> Value {
    json!({ "error": format!("{error:#}") })
}

struct Call {
    state: Shared,
    id: Option<RequestId>,
    cancelled: Arc<AtomicBool>,
}

impl Call {
    fn new(state: Shared, session: Option<String>, index: usize) -> Result<Self> {
        ensure!(
            session.as_ref().is_none_or(|id| id.len() <= 128),
            "invalid Gradio session ID"
        );
        let id = session.map(|id| (id, index));
        let cancelled = Arc::new(AtomicBool::new(false));
        if let Some(id) = &id {
            let mut active = lock(&state.legacy.requests)?;
            if let Some(previous) = active.insert(id.clone(), Arc::downgrade(&cancelled))
                && let Some(previous) = previous.upgrade()
            {
                previous.store(true, Ordering::Relaxed);
            }
        }
        Ok(Self {
            state,
            id,
            cancelled,
        })
    }
}

impl Drop for Call {
    fn drop(&mut self) {
        self.cancelled.store(true, Ordering::Relaxed);
        if let Some(id) = &self.id
            && let Ok(mut active) = self.state.legacy.requests.lock()
            && active
                .get(id)
                .is_some_and(|token| token.as_ptr() == Arc::as_ptr(&self.cancelled))
        {
            active.remove(id);
        }
    }
}

async fn reset(State(state): State<Shared>, Json(reset): Json<Reset>) -> Json<Value> {
    let key = (reset.session_hash, reset.fn_index);
    let active = state
        .legacy
        .requests
        .lock()
        .ok()
        .and_then(|calls| calls.get(&key).and_then(Weak::upgrade));
    if let Some(cancelled) = &active {
        cancelled.store(true, Ordering::Relaxed);
    }
    Json(json!({"success":active.is_some()}))
}

async fn predict(
    State(state): State<Shared>,
    Path(name): Path<String>,
    Json(body): Json<Predict>,
) -> Response {
    let started = Instant::now();
    let result = async {
        let index = body.endpoint(&name)?;
        let call = Call::new(state.clone(), body.session_hash, index)?;
        let cancelled = Arc::clone(&call.cancelled);
        let session = call.id.as_ref().map(|(id, _)| id.clone());
        tokio::task::spawn_blocking(move || {
            callbacks::run(
                &state,
                session.as_deref(),
                index,
                body.data,
                cancelled,
                |_| {},
            )
        })
        .await?
    }
    .await;
    match result {
        Ok(data) => Json(output(data, started, false)).into_response(),
        Err(error) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(error_response(error)),
        )
            .into_response(),
    }
}

async fn queue_status(State(state): State<Shared>) -> Json<Value> {
    Json(estimate(state.legacy.waiting(), None))
}

fn estimate(size: usize, rank: Option<usize>) -> Value {
    json!({"msg":"estimation", "queue_size":size, "rank":rank, "queue_eta":1,
           "rank_eta":null, "avg_event_process_time":null, "avg_event_concurrent_process_time":null})
}

async fn join(State(state): State<Shared>, upgrade: WebSocketUpgrade) -> Response {
    upgrade
        .max_message_size(MAX_MESSAGE)
        .max_frame_size(MAX_MESSAGE)
        .on_upgrade(move |socket| queue(socket, state))
}

async fn send(socket: &mut WebSocket, value: Value) -> Result<()> {
    socket.send(Message::Text(value.to_string().into())).await?;
    Ok(())
}

async fn receive<T: serde::de::DeserializeOwned>(socket: &mut WebSocket) -> Result<T> {
    let message = tokio::time::timeout(HANDSHAKE_TIMEOUT, async {
        loop {
            match socket.recv().await {
                Some(Ok(Message::Text(text))) => break Ok(text),
                Some(Ok(Message::Ping(_) | Message::Pong(_))) => continue,
                Some(Err(error)) => break Err(error.into()),
                _ => break Err(anyhow::anyhow!("Gradio socket closed")),
            }
        }
    })
    .await??;
    Ok(serde_json::from_str(message.as_str())?)
}

async fn queue(mut socket: WebSocket, state: Shared) {
    if let Err(error) = queued_call(&mut socket, &state).await {
        let _ = send(
            &mut socket,
            json!({
                "msg":"process_completed", "output":error_response(error), "success":false,
            }),
        )
        .await;
    }
    let _ = socket.send(Message::Close(None)).await;
}

async fn queued_call(socket: &mut WebSocket, state: &Shared) -> Result<()> {
    send(socket, json!({"msg":"send_hash"})).await?;
    let request: Reset = receive(socket).await?;
    let index = Predict {
        data: Vec::new(),
        session_hash: None,
        fn_index: Some(request.fn_index),
    }
    .endpoint("predict")?;
    let Ok(_slot) = state.legacy.queue.try_acquire() else {
        send(socket, json!({"msg":"queue_full"})).await?;
        return Ok(());
    };
    let size = state.legacy.waiting();
    send(socket, estimate(size, Some(size.saturating_sub(1)))).await?;
    let _worker = loop {
        tokio::select! {
            permit = state.legacy.worker.acquire() => break permit?,
            event = socket.recv() => match event {
                Some(Ok(Message::Ping(_) | Message::Pong(_))) => continue,
                _ => return Ok(()),
            }
        }
    };
    send(socket, json!({"msg":"send_data"})).await?;
    let data: Predict = receive(socket).await?;
    ensure!(
        data.fn_index.is_none_or(|value| value == index)
            && data
                .session_hash
                .as_ref()
                .is_none_or(|id| id == &request.session_hash),
        "Gradio queue data changed session or function"
    );
    send(socket, json!({"msg":"process_starts"})).await?;
    let started = Instant::now();
    let call = Call::new(state.clone(), Some(request.session_hash), index)?;
    let cancelled = Arc::clone(&call.cancelled);
    let work_state = state.clone();
    let session = call.id.as_ref().map(|(id, _)| id.clone());
    let (updates, mut progress) = mpsc::channel(16);
    let mut job = tokio::task::spawn_blocking(move || {
        callbacks::run(
            &work_state,
            session.as_deref(),
            index,
            data.data,
            cancelled,
            |step| {
                let _ = updates.try_send(step);
            },
        )
    });
    loop {
        tokio::select! {
            result = &mut job => {
                let data = result??;
                send(socket, json!({
                    "msg":"process_completed", "output":output(data, started, callbacks::generates(index)),
                    "success":true,
                })).await?;
                return Ok(());
            }
            Some(step) = progress.recv() => {
                send(socket, json!({
                    "msg":"process_generating", "output":output(step, started, true),
                    "success":true,
                })).await?;
            }
            event = socket.recv() => match event {
                Some(Ok(Message::Ping(_) | Message::Pong(_))) => {},
                _ => return Ok(()),
            }
        }
    }
}

async fn legacy_file(State(state): State<Shared>, request: Request) -> Response {
    if !matches!(*request.method(), Method::GET | Method::HEAD) {
        return StatusCode::NOT_FOUND.into_response();
    }
    let Some(name) = request
        .uri()
        .path()
        .strip_prefix("/file=")
        .map(str::to_owned)
    else {
        return StatusCode::NOT_FOUND.into_response();
    };
    file(State(state), Path(name), request).await
}

async fn file(State(state): State<Shared>, Path(name): Path<String>, request: Request) -> Response {
    let Some(path) = state.legacy.open_path(&name) else {
        return StatusCode::NOT_FOUND.into_response();
    };
    match ServeFile::new(path).try_call(request).await {
        Ok(file) => file.into_response(),
        Err(_) => StatusCode::INTERNAL_SERVER_ERROR.into_response(),
    }
}
