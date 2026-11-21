use std::collections::VecDeque;
use std::io::{BufWriter, Seek};
use std::net::{IpAddr, SocketAddr};
use std::ops::ControlFlow;
use std::path::{Component, Path, PathBuf};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};

use anyhow::{Context, Result, bail, ensure};
use axum::body::Body;
use axum::extract::connect_info::{ConnectInfo, Connected};
use axum::extract::{DefaultBodyLimit, Multipart, Path as RoutePath, State};
use axum::http::uri::Authority;
use axum::http::{StatusCode, header};
use axum::middleware::{self, Next};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post};
use axum::serve::IncomingStream;
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::io::AsyncWriteExt;
use tokio::net::TcpListener;
use tokio_util::io::ReaderStream;

use crate::audio::{self, ANALYSIS_RATE, Format, conversion_jobs};
use crate::catalog::{ModelInfo, Paths};
use crate::model::{self, ConversionSettings, Rvc};
use crate::{dataset, separation};

mod gradio;
mod locale;
mod share;
mod studio;

pub use share::share_client;

#[derive(Clone, Copy, clap::ValueEnum)]
pub enum Panel {
    Convert,
    Realtime,
    Training,
    Separation,
    Models,
}

impl Panel {
    fn path(self) -> &'static str {
        match self {
            Self::Convert => "",
            Self::Realtime => "?panel=realtime",
            Self::Training => "?panel=training",
            Self::Separation => "?panel=separation",
            Self::Models => "?panel=models",
        }
    }
}

struct StateData {
    paths: Paths,
    job: Mutex<Job>,
    recent_jobs: Mutex<VecDeque<Job>>,
    voice: Arc<Mutex<Option<CachedVoice>>>,
    computing: Arc<AtomicBool>,
    studio: studio::Files,
    legacy: gradio::Legacy,
    #[cfg(feature = "realtime")]
    realtime: Mutex<Option<Live>>,
}

type Shared = Arc<StateData>;

#[derive(Default, Serialize)]
struct Job {
    id: uuid::Uuid,
    operation: String,
    status: JobState,
    message: String,
    log: VecDeque<String>,
    result: Option<Value>,
    #[serde(skip)]
    cancelled: Arc<AtomicBool>,
}

#[derive(Clone, Copy, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum JobState {
    #[default]
    Idle,
    Running,
    Complete,
    Failed,
    Cancelled,
}

#[derive(Deserialize)]
struct Convert {
    model: String,
    speaker_id: Option<i64>,
    index: Option<PathBuf>,
    #[serde(default = "automatic")]
    device: String,
    #[serde(default)]
    settings: ConversionSettings,
    #[serde(default)]
    auto_register: bool,
}

fn automatic() -> String {
    "auto".to_owned()
}

#[derive(Deserialize)]
#[serde(tag = "operation", rename_all = "snake_case")]
enum Action {
    Batch {
        #[serde(flatten)]
        conversion: Convert,
        input: PathBuf,
        output: PathBuf,
        format: Format,
        recursive: bool,
        overwrite: bool,
    },
    Separate {
        input: PathBuf,
        output: PathBuf,
        preset: String,
        format: Format,
        secondary_output: Option<PathBuf>,
        overwrite: bool,
        #[serde(default = "automatic")]
        device: String,
    },
    Preprocess {
        input: Option<PathBuf>,
        experiment: PathBuf,
        source: DatasetMode,
        sample_rate: u32,
        workers: usize,
        segment_seconds: f64,
        #[serde(default)]
        denoise: f32,
    },
    Extract {
        experiment: PathBuf,
        version: String,
        f0_method: Option<model::PitchMethod>,
        embedder: Option<crate::catalog::Embedder>,
        #[serde(default = "automatic")]
        device: String,
        #[serde(default)]
        gpus: Vec<usize>,
        overwrite: bool,
        #[serde(default)]
        unvoice_fricatives: bool,
    },
    Train {
        #[serde(flatten)]
        request: model::train::Request,
    },
    #[cfg(feature = "index-training")]
    TrainAll {
        #[serde(flatten)]
        request: model::train::Request,
        input: Option<PathBuf>,
        source: DatasetMode,
        workers: usize,
        segment_seconds: f64,
        f0_method: model::PitchMethod,
        #[serde(default)]
        denoise: f32,
    },
    #[cfg(feature = "index-training")]
    Index {
        experiment: PathBuf,
        version: crate::train_index::ModelVersion,
        mode: crate::train_index::IndexMode,
        workers: usize,
    },
}

impl Action {
    fn name(&self) -> &'static str {
        match self {
            Self::Batch { .. } => "batch",
            Self::Separate { .. } => "separate",
            Self::Preprocess { .. } => "preprocess",
            Self::Extract { .. } => "extract",
            Self::Train { .. } => "train",
            #[cfg(feature = "index-training")]
            Self::TrainAll { .. } => "train_all",
            #[cfg(feature = "index-training")]
            Self::Index { .. } => "index",
        }
    }

    fn run(
        self,
        state: &StateData,
        cancelled: &AtomicBool,
        mut report: impl FnMut(&str) -> ControlFlow<()>,
    ) -> Result<Value> {
        let paths = &state.paths;
        match self {
            Self::Batch {
                conversion,
                input,
                output,
                format,
                recursive,
                overwrite,
            } => {
                let model = installed_model(paths, &conversion.model)?;
                crate::assets::voice(paths, &model, conversion.settings.f0_method, &mut report)?;
                let jobs = conversion_jobs(
                    paths.root.join(input),
                    paths.root.join(output),
                    Some(format),
                    recursive,
                )?;
                let mut outputs = Vec::new();
                let mut unique_outputs = std::collections::HashSet::new();
                for (_, file, _) in &jobs {
                    ensure!(
                        unique_outputs.insert(file),
                        "multiple inputs would write to {}",
                        file.display()
                    );
                    ensure!(
                        overwrite || !file.exists(),
                        "output file already exists: {}",
                        file.display()
                    );
                }
                for (i, (file, output, format)) in jobs.iter().enumerate() {
                    if report(&format!("{} / {} · {}", i + 1, jobs.len(), file.display()))
                        .is_break()
                    {
                        bail!("batch conversion cancelled");
                    }
                    let source = audio::read_mono(file, ANALYSIS_RATE)?;
                    let converted = convert_voice(state, &conversion, &source, Some(file))?;
                    audio::write_audio(
                        output,
                        &converted.samples,
                        converted.rate,
                        *format,
                        overwrite,
                    )?;
                    outputs.push(output);
                }
                Ok(json!(outputs))
            }
            Self::Separate {
                input,
                output,
                preset,
                format,
                secondary_output,
                overwrite,
                device,
            } => {
                crate::assets::separation(paths, &preset, &mut report)?;
                let secondary_output = secondary_output.map(|path| paths.root.join(path));
                let options = separation::Options {
                    preset: &preset,
                    device: separation::device(&device)?,
                    format,
                    secondary_output: secondary_output.as_deref(),
                    overwrite,
                };
                let outputs = separation::separate(
                    &paths.root,
                    &paths.root.join(input),
                    &paths.root.join(output),
                    options,
                    |progress| {
                        report(&format!(
                            "{} / {} · {} / {}",
                            progress.file_index,
                            progress.file_count,
                            progress.chunks_done,
                            progress.chunks_total
                        ))
                    },
                )?;
                Ok(json!(outputs))
            }
            Self::Preprocess {
                input,
                experiment,
                source,
                sample_rate,
                workers,
                segment_seconds,
                denoise,
            } => {
                let input = input.map(|path| paths.root.join(path));
                let source = match source {
                    DatasetMode::Single => dataset::DatasetSource::Single(
                        input.as_deref().context("dataset folder is required")?,
                    ),
                    DatasetMode::MultiFolder => dataset::DatasetSource::MultiFolder(
                        input.as_deref().context("dataset folder is required")?,
                    ),
                    DatasetMode::Manifest => dataset::DatasetSource::MultiManifest,
                };
                let result = dataset::preprocess_with_denoise(
                    &dataset::PreprocessRequest {
                        source,
                        experiment_dir: &paths.root.join(experiment),
                        sample_rate,
                        workers,
                        segment_seconds,
                    },
                    denoise,
                    || cancelled.load(Ordering::Relaxed),
                    |line| {
                        let _ = report(line);
                    },
                )?;
                Ok(json!(result))
            }
            Self::Extract {
                experiment,
                version,
                f0_method,
                embedder,
                device,
                gpus,
                overwrite,
                unvoice_fricatives,
            } => {
                if let Some(method) = f0_method {
                    crate::assets::pitch(paths, method, &mut report)?;
                }
                let devices = model::extract::selected_devices(&device, &gpus)?;
                let request = model::extract::Request {
                    experiment: &paths.root.join(experiment),
                    version: &version,
                    f0: f0_method,
                    device: devices[0],
                    overwrite,
                    unvoice_fricatives,
                };
                Ok(json!(model::extract::run_with_embedder(
                    paths, request, &devices, embedder, report
                )?))
            }
            Self::Train { mut request } => {
                crate::assets::training(paths, &request, &mut report)?;
                request.rebuild_filelist = true;
                Ok(json!(model::train::run(paths, request, report)?))
            }
            #[cfg(feature = "index-training")]
            Self::TrainAll {
                mut request,
                input,
                source,
                workers,
                segment_seconds,
                f0_method,
                denoise,
            } => {
                crate::assets::training(paths, &request, &mut report)?;
                if request.f0 {
                    crate::assets::pitch(paths, f0_method, &mut report)?;
                }
                let input = input.map(|path| paths.root.join(path));
                let experiment = paths.root.join(&request.experiment);
                request.multi = !matches!(source, DatasetMode::Single);
                let dataset_source = match source {
                    DatasetMode::Single => dataset::DatasetSource::Single(
                        input.as_deref().context("dataset folder is required")?,
                    ),
                    DatasetMode::MultiFolder => dataset::DatasetSource::MultiFolder(
                        input.as_deref().context("dataset folder is required")?,
                    ),
                    DatasetMode::Manifest => dataset::DatasetSource::MultiManifest,
                };
                let cancellation = || cancelled.load(Ordering::Relaxed);
                let prepared = dataset::preprocess_with_denoise(
                    &dataset::PreprocessRequest {
                        source: dataset_source,
                        experiment_dir: &experiment,
                        sample_rate: request.sample_rate,
                        workers,
                        segment_seconds,
                    },
                    denoise,
                    cancellation,
                    |progress| {
                        let _ = report(&format!("Dataset · {progress}"));
                    },
                )?;
                ensure!(
                    prepared.written_clips > 0,
                    "dataset preparation wrote zero clips"
                );
                let devices = model::extract::selected_devices(&request.device, &request.gpus)?;
                let feature_request = model::extract::Request {
                    experiment: &experiment,
                    version: request.version.name(),
                    f0: request.f0.then_some(f0_method),
                    device: devices[0],
                    overwrite: true,
                    unvoice_fricatives: false,
                };
                let features = model::extract::run_with_embedder(
                    paths,
                    feature_request,
                    &devices,
                    request.embedder,
                    |progress| report(&format!("Features · {progress}")),
                )?;
                let index_request = crate::train_index::IndexTraining {
                    experiment_dir: &experiment,
                    outside_index_root: &paths.indices,
                    version: match request.version {
                        model::train_data::Version::V1 => crate::train_index::ModelVersion::V1,
                        model::train_data::Version::V2 => crate::train_index::ModelVersion::V2,
                    },
                    mode: if request.multi {
                        crate::train_index::IndexMode::Multi
                    } else {
                        crate::train_index::IndexMode::Single
                    },
                    workers,
                };
                request.rebuild_filelist = true;
                let training = model::train::run(paths, request, |progress| {
                    report(&format!("Training · {progress}"))
                })?;
                let indexes = crate::train_index::train_indices_with_cancel(
                    &index_request,
                    cancellation,
                    |progress| {
                        let _ = report(&format!("Index · {progress}"));
                    },
                )?;
                Ok(
                    json!({ "dataset": prepared, "features": features, "training": training, "indexes": indexes }),
                )
            }
            #[cfg(feature = "index-training")]
            Self::Index {
                experiment,
                version,
                mode,
                workers,
            } => {
                let request = crate::train_index::IndexTraining {
                    experiment_dir: &paths.root.join(experiment),
                    outside_index_root: &paths.indices,
                    version,
                    mode,
                    workers,
                };
                let result = crate::train_index::train_indices_with_cancel(
                    &request,
                    || cancelled.load(Ordering::Relaxed),
                    |line| {
                        let _ = report(line);
                    },
                )?;
                Ok(json!(result))
            }
        }
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum DatasetMode {
    Single,
    MultiFolder,
    Manifest,
}

#[derive(Deserialize)]
struct ManifestRequest {
    experiment: PathBuf,
    root: Option<PathBuf>,
    rows: Option<Vec<dataset::ManifestRowInput>>,
}

#[derive(Deserialize)]
#[serde(tag = "operation", rename_all = "snake_case")]
enum ModelAction {
    Describe {
        model: String,
        output: String,
        description: String,
        overwrite: bool,
    },
    Merge {
        first: String,
        second: String,
        output: String,
        description: String,
        alpha: f64,
        overwrite: bool,
    },
    Extract {
        checkpoint: PathBuf,
        output: String,
        description: String,
        overwrite: bool,
        version: model::train_data::Version,
        sample_rate: u32,
        f0: bool,
    },
}

/// How the WebUI is reached from outside the machine.
pub enum Share {
    /// Through the Gradio relay, with the given frpc client.
    Relay(PathBuf),
    /// Through a tunnel the user started, at this https:// origin.
    Url(String),
}

pub fn serve(
    paths: Paths,
    listen: IpAddr,
    port: u16,
    open: bool,
    panel: Option<Panel>,
    share: Option<Share>,
) -> Result<()> {
    let external = match &share {
        Some(Share::Url(url)) => Some(share::External::new(url)?),
        _ => None,
    };
    let share_client = match share {
        Some(Share::Relay(client)) => Some(client),
        _ => None,
    };
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    runtime.block_on(async {
        let state = Arc::new(StateData {
            paths,
            job: Mutex::default(),
            recent_jobs: Mutex::default(),
            voice: Arc::default(),
            computing: Arc::default(),
            studio: studio::Files::new()?,
            legacy: gradio::Legacy::new(share_client.is_some() || external.is_some())?,
            #[cfg(feature = "realtime")]
            realtime: Mutex::default(),
        });
        let router = Router::new()
            .route(
                "/",
                get(|| async { Html(include_str!("../web/index.html")) }),
            )
            .route(
                "/app.css",
                get(|| async { asset("text/css; charset=utf-8", include_str!("../web/app.css")) }),
            )
            .route(
                "/app.js",
                get(|| async {
                    asset(
                        "text/javascript; charset=utf-8",
                        include_str!("../web/app.js"),
                    )
                }),
            )
            .route("/api/locale/{language}", get(locale::messages))
            .route("/api/options", get(options))
            .route("/api/model/{name}", get(model_info))
            .route("/api/model", post(edit_model))
            .route("/api/voice", axum::routing::delete(release_voice))
            .route(
                "/api/convert",
                post(convert).layer(DefaultBodyLimit::max(128 * 1024 * 1024)),
            )
            .route(
                "/api/job",
                get(job_status).post(start_job).delete(cancel_job),
            )
            .route("/api/manifest", post(create_manifest));
        #[cfg(feature = "realtime")]
        let router = router
            .route("/api/devices", get(devices))
            .route(
                "/api/realtime",
                get(live_status)
                    .post(start_live)
                    .patch(update_live)
                    .delete(stop_live),
            )
            .route("/api/realtime/profile", get(realtime_profile));
        let listener = bind_webui(listen, port).await?;
        let bound = listener.local_addr()?;
        let address = browser_address(bound);
        let mut shutdown = Box::pin(shutdown_signal());
        let mut tunnel = if let Some(client) = share_client.as_deref() {
            tokio::select! {
                biased;
                () = &mut shutdown => return Ok(()),
                tunnel = share::Tunnel::connect(client, address) => Some(tunnel?),
            }
        } else {
            None
        };
        let access = tunnel
            .as_ref()
            .map(share::Tunnel::access)
            .or_else(|| external.as_ref().map(share::External::access));
        let shutdown_state = Arc::clone(&state);
        let app = router
            .merge(studio::routes())
            .merge(gradio::routes())
            .layer(middleware::from_fn_with_state(access, local_requests))
            .with_state(state);
        let url = format!(
            "http://{}/{}",
            address,
            panel.map(Panel::path).unwrap_or("")
        );
        println!("{url}");
        if let Some(tunnel) = &tunnel {
            tunnel.announce(panel.map(Panel::path).unwrap_or(""));
        }
        if let Some(external) = &external {
            external.announce(panel.map(Panel::path).unwrap_or(""));
        }
        if open && let Err(error) = open_browser(&url) {
            eprintln!("Could not open {url}: {error}");
        }
        axum::serve(
            listener,
            app.into_make_service_with_connect_info::<LocalEndpoint>(),
        )
        .with_graceful_shutdown(async move {
            if let Some(tunnel) = &mut tunnel {
                tokio::select! {
                    () = &mut shutdown => {},
                    () = tunnel.wait_for_exit() => {},
                }
                tunnel.stop().await;
            } else {
                shutdown.await;
            }
            if let Ok(job) = shutdown_state.job.lock() {
                job.cancelled.store(true, Ordering::Relaxed);
            }
            #[cfg(feature = "realtime")]
            if let Ok(live) = shutdown_state.realtime.lock()
                && let Some(live) = live.as_ref()
            {
                let _ = live.stop.send(());
            }
        })
        .await?;
        Ok(())
    })
}

fn browser_address(bound: SocketAddr) -> SocketAddr {
    let ip = match bound.ip() {
        IpAddr::V4(ip) if ip.is_unspecified() => IpAddr::V4(std::net::Ipv4Addr::LOCALHOST),
        IpAddr::V6(ip) if ip.is_unspecified() => IpAddr::V6(std::net::Ipv6Addr::LOCALHOST),
        _ => return bound,
    };
    SocketAddr::new(ip, bound.port())
}

async fn shutdown_signal() {
    let ctrl_c = tokio::signal::ctrl_c();
    #[cfg(unix)]
    if let Ok(mut terminate) =
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
    {
        tokio::select! {
            _ = ctrl_c => {},
            _ = terminate.recv() => {},
        }
        return;
    }
    let _ = ctrl_c.await;
}

async fn bind_webui(listen: IpAddr, mut port: u16) -> std::io::Result<TcpListener> {
    loop {
        match TcpListener::bind((listen, port)).await {
            Err(error) if error.kind() == std::io::ErrorKind::AddrInUse && port != u16::MAX => {
                port += 1;
            }
            result => return result,
        }
    }
}

fn open_browser(url: &str) -> std::io::Result<()> {
    use std::process::Command;
    let mut browser = if cfg!(target_os = "windows") {
        let mut browser = Command::new("rundll32.exe");
        browser.arg("url.dll,FileProtocolHandler");
        browser
    } else if cfg!(target_os = "macos") {
        Command::new("open")
    } else {
        Command::new("xdg-open")
    };
    let mut child = browser.arg(url).spawn()?;
    std::thread::spawn(move || {
        let _ = child.wait();
    });
    Ok(())
}

fn asset(mime: &'static str, contents: &'static str) -> impl IntoResponse {
    ([(header::CONTENT_TYPE, mime)], contents)
}

#[derive(Clone, Copy)]
struct LocalEndpoint(Option<SocketAddr>);

impl Connected<IncomingStream<'_, TcpListener>> for LocalEndpoint {
    fn connect_info(stream: IncomingStream<'_, TcpListener>) -> Self {
        Self(stream.io().local_addr().ok())
    }
}

impl LocalEndpoint {
    fn permits(self, authority: &Authority) -> bool {
        let Some(endpoint) = self.0.filter(|_| !authority.as_str().contains('@')) else {
            return false;
        };
        let host = authority.host();
        let local = endpoint.ip().to_canonical();
        let ip = host.trim_matches(['[', ']']).parse::<IpAddr>();
        if local.is_loopback() {
            return host.eq_ignore_ascii_case("localhost")
                || ip.is_ok_and(|ip| ip.to_canonical().is_loopback());
        }
        // A wildcard listener accepts each local interface. The accepted socket
        // supplies its actual address without trusting arbitrary host names.
        let port = authority
            .port_u16()
            .or_else(|| (authority.as_str() == host).then_some(80));
        ip.is_ok_and(|ip| ip.to_canonical() == local) && port == Some(endpoint.port())
    }
}

async fn local_requests(
    State(share): State<Option<share::Access>>,
    ConnectInfo(endpoint): ConnectInfo<LocalEndpoint>,
    request: axum::extract::Request,
    next: Next,
) -> Response {
    let fields = request.headers();
    let host = fields
        .get(header::HOST)
        .and_then(|v| v.to_str().ok())
        .and_then(|name| name.parse::<Authority>().ok());
    let local = host
        .as_ref()
        .is_some_and(|authority| endpoint.permits(authority));
    let public = share
        .as_ref()
        .is_some_and(|share| host.as_ref() == Some(share.authority()));
    let valid_origin = fields.get(header::ORIGIN).is_none_or(|origin| {
        host.as_ref().is_some_and(|host| {
            let scheme = if public { "https://" } else { "http://" };
            origin
                .to_str()
                .is_ok_and(|origin| origin.strip_prefix(scheme) == Some(host.as_str()))
        })
    });
    let cross_site = fields
        .get("sec-fetch-site")
        .is_some_and(|site| site == "cross-site");
    let public_landing =
        public && request.method() == axum::http::Method::GET && request.uri().path() == "/";
    if !(local || public) || !valid_origin || (cross_site && !public_landing) {
        return (
            StatusCode::FORBIDDEN,
            "use the WebUI from its announced URL",
        )
            .into_response();
    }
    if let Some(share) = share
        && !share.authorized(fields.get(header::AUTHORIZATION))
    {
        return (
            StatusCode::UNAUTHORIZED,
            [(
                header::WWW_AUTHENTICATE,
                "Basic realm=\"RVC WebUI\", charset=\"UTF-8\"",
            )],
            "RVC sharing requires a password",
        )
            .into_response();
    }
    next.run(request).await
}

#[cfg(test)]
mod listen_tests {
    use super::*;
    use tokio::io::AsyncReadExt;
    use tokio::net::TcpStream;

    #[test]
    fn direct_hosts_match_the_connection_endpoint() {
        for (endpoint, host, permitted) in [
            ("192.0.2.5:7865", "192.0.2.5:7865", true),
            ("192.0.2.5:7865", "192.0.2.5:7866", false),
            ("192.0.2.5:7865", "192.0.2.6:7865", false),
            ("192.0.2.5:7865", "localhost:7865", false),
            ("192.0.2.5:7865", "rvc.attacker.example:7865", false),
            ("192.0.2.5:80", "192.0.2.5", true),
            ("[2001:db8::5]:7865", "[2001:db8::5]:7865", true),
            ("[::ffff:192.0.2.5]:7865", "192.0.2.5:7865", true),
        ] {
            let local = LocalEndpoint(Some(endpoint.parse().unwrap()));
            assert_eq!(local.permits(&host.parse().unwrap()), permitted, "{host}");
        }
    }

    #[tokio::test]
    async fn loopback_keeps_host_and_origin_checks() {
        let listener = bind_webui("127.0.0.1".parse().unwrap(), 0).await.unwrap();
        let address = listener.local_addr().unwrap();
        let app = Router::new()
            .route("/", get(|| async { StatusCode::NO_CONTENT }))
            .layer(middleware::from_fn_with_state(None, local_requests));
        let server = tokio::spawn(async move {
            axum::serve(
                listener,
                app.into_make_service_with_connect_info::<LocalEndpoint>(),
            )
            .await
            .unwrap();
        });
        let local = format!("localhost:{}", address.port());
        for (host, origin, status) in [
            (local.as_str(), local.as_str(), 204),
            (local.as_str(), "other.example", 403),
            ("other.example", "other.example", 403),
        ] {
            let mut stream = TcpStream::connect(address).await.unwrap();
            let request = format!(
                "GET / HTTP/1.1\r\nHost: {host}\r\nOrigin: http://{origin}\r\nConnection: close\r\n\r\n"
            );
            stream.write_all(request.as_bytes()).await.unwrap();
            let mut response = String::new();
            tokio::time::timeout(
                std::time::Duration::from_secs(3),
                stream.read_to_string(&mut response),
            )
            .await
            .unwrap()
            .unwrap();
            assert!(response.starts_with(&format!("HTTP/1.1 {status} ")));
        }
        server.abort();
    }
}

async fn options(State(state): State<Shared>) -> WebResult<Json<Value>> {
    Ok(Json(json!({
        "models": state.paths.models()?,
        "presets": separation::PRESETS,
        "realtime": cfg!(feature = "realtime"),
        "index": cfg!(feature = "index-training"),
    })))
}

async fn model_info(
    State(state): State<Shared>,
    RoutePath(name): RoutePath<String>,
) -> WebResult<Json<Value>> {
    let model = installed_model(&state.paths, &name)?;
    Ok(Json(
        tokio::task::spawn_blocking(move || -> Result<Value> {
            let info = ModelInfo::read(&model)?;
            Ok(json!(info))
        })
        .await
        .map_err(anyhow::Error::from)??,
    ))
}

fn installed_model(paths: &Paths, name: &str) -> Result<PathBuf> {
    let path = model_output(paths, name)?;
    ensure!(path.is_file(), "model is not installed");
    Ok(path)
}

fn checkpoint_model(paths: &Paths, name: &str) -> Result<PathBuf> {
    let path = Path::new(name);
    if path.components().count() == 1 {
        installed_model(paths, name)
    } else {
        paths.model(path)
    }
}

fn model_output(paths: &Paths, name: &str) -> Result<PathBuf> {
    let path = Path::new(name);
    let mut parts = path.components();
    ensure!(
        matches!(parts.next(), Some(Component::Normal(_))) && parts.next().is_none(),
        "select an installed model"
    );
    let path = paths.weights.join(path);
    ensure!(
        path.extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("pth")),
        "model filename must end in .pth"
    );
    Ok(path)
}

async fn edit_model(
    State(state): State<Shared>,
    Json(action): Json<ModelAction>,
) -> WebResult<Json<Value>> {
    let result = tokio::task::spawn_blocking(move || -> Result<PathBuf> {
        let paths = &state.paths;
        let target = match action {
            ModelAction::Describe {
                model,
                output,
                description,
                overwrite,
            } => {
                let target = model_output(paths, &output)?;
                model::edit::describe(
                    &checkpoint_model(paths, &model)?,
                    &target,
                    &description,
                    overwrite,
                )?;
                target
            }
            ModelAction::Merge {
                first,
                second,
                output,
                description,
                alpha,
                overwrite,
            } => {
                let target = model_output(paths, &output)?;
                model::edit::merge(
                    &checkpoint_model(paths, &first)?,
                    &checkpoint_model(paths, &second)?,
                    &target,
                    alpha,
                    &description,
                    overwrite,
                )?;
                target
            }
            ModelAction::Extract {
                checkpoint,
                output,
                description,
                overwrite,
                version,
                sample_rate,
                f0,
            } => {
                let target = model_output(paths, &output)?;
                model::edit::extract(
                    &paths.root.join(checkpoint),
                    &target,
                    model::edit::Export {
                        version,
                        sample_rate,
                        uses_f0: f0,
                        description: &description,
                        overwrite,
                    },
                )?;
                target
            }
        };
        Ok(target)
    })
    .await
    .map_err(anyhow::Error::from)??;
    Ok(Json(json!({ "path": result })))
}

#[derive(PartialEq, Eq)]
struct VoiceKey {
    model: FileVersion,
    speaker: i64,
    index: Option<FileVersion>,
    device: tch::Device,
    precision: model::Precision,
    pitch_method: model::PitchMethod,
}

#[derive(PartialEq, Eq)]
struct FileVersion {
    path: PathBuf,
    bytes: u64,
    modified: Option<std::time::SystemTime>,
}

impl FileVersion {
    fn read(path: PathBuf) -> Result<Self> {
        let file =
            std::fs::metadata(&path).with_context(|| format!("checking {}", path.display()))?;
        Ok(Self {
            path,
            bytes: file.len(),
            modified: file.modified().ok(),
        })
    }
}

struct CachedVoice {
    key: VoiceKey,
    engine: Rvc,
}

/// `source` is the decoded file, read again at the model rate when the voice uses its high band.
fn convert_voice(
    state: &StateData,
    request: &Convert,
    audio: &[f32],
    source: Option<&std::path::Path>,
) -> Result<model::Audio> {
    let paths = &state.paths;
    let model = installed_model(paths, &request.model)?;
    let info = ModelInfo::read(&model)?;
    let speaker = info.speaker(request.speaker_id)?;
    let index = if request.index.is_some() {
        paths.require_index(
            &model,
            speaker,
            request.index.as_deref(),
            request.settings.index_rate,
        )?
    } else if request.settings.index_rate > 0.0 {
        paths.select_index(&model, Some(speaker))
    } else {
        None
    };
    let key = VoiceKey {
        model: FileVersion::read(model.clone())?,
        speaker,
        index: index.map(FileVersion::read).transpose()?,
        device: model::weights::device(&request.device)?,
        precision: request.settings.precision,
        pitch_method: request.settings.f0_method,
    };
    let mut current = lock(&state.voice)?;
    if current.as_ref().is_none_or(|cached| cached.key != key) {
        *current = None;
        let engine = Rvc::load(
            paths,
            &key.model.path,
            Some(key.speaker),
            key.index.as_ref().map(|index| index.path.as_path()),
            request.settings,
            key.device,
        )?;
        *current = Some(CachedVoice { key, engine });
    }
    let engine = &current
        .as_ref()
        .context("voice engine was not loaded")?
        .engine;
    let hires = match source.filter(|_| engine.needs_hires(request.settings)) {
        Some(path) => Some(audio::read_mono(path, engine.rate())?),
        None => None,
    };
    if request.auto_register && info.uses_f0 {
        let profile = model::register::Profile::load(&model)?;
        engine.convert_registered(audio, hires.as_deref(), request.settings, profile.as_ref())
    } else {
        engine.convert_hires(audio, hires.as_deref(), request.settings)
    }
}

struct ComputePermit {
    busy: Arc<AtomicBool>,
    voice: Arc<Mutex<Option<CachedVoice>>>,
}

impl Drop for ComputePermit {
    fn drop(&mut self) {
        if let Ok(mut voice) = self.voice.lock()
            && voice
                .as_ref()
                .is_some_and(|cached| cached.key.device != tch::Device::Cpu)
        {
            *voice = None;
        }
        self.busy.store(false, Ordering::Release);
    }
}

fn compute(state: &StateData) -> Result<ComputePermit> {
    state
        .computing
        .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
        .map_err(|_| {
            anyhow::anyhow!("Stop the active audio stream or wait for the current task to finish.")
        })?;
    Ok(ComputePermit {
        busy: Arc::clone(&state.computing),
        voice: Arc::clone(&state.voice),
    })
}

async fn release_voice(State(state): State<Shared>) -> WebResult<StatusCode> {
    tokio::task::spawn_blocking(move || -> Result<()> {
        *lock(&state.voice)? = None;
        Ok(())
    })
    .await
    .map_err(anyhow::Error::from)??;
    Ok(StatusCode::NO_CONTENT)
}

async fn convert(State(state): State<Shared>, mut form: Multipart) -> WebResult<Response> {
    let mut request = None;
    let mut input = None;
    while let Some(mut field) = form.next_field().await.map_err(anyhow::Error::from)? {
        match field.name() {
            Some("settings") if request.is_none() => {
                request = Some(
                    serde_json::from_str::<Convert>(
                        &field.text().await.map_err(anyhow::Error::from)?,
                    )
                    .map_err(anyhow::Error::from)?,
                );
            }
            Some("audio") if input.is_none() => {
                let (staging, path) = tempfile::NamedTempFile::new()
                    .map_err(anyhow::Error::from)?
                    .into_parts();
                let mut staging = tokio::fs::File::from_std(staging);
                while let Some(chunk) = field.chunk().await.map_err(anyhow::Error::from)? {
                    staging
                        .write_all(&chunk)
                        .await
                        .map_err(anyhow::Error::from)?;
                }
                staging.flush().await.map_err(anyhow::Error::from)?;
                input = Some(path);
            }
            _ => return Err(anyhow::anyhow!("unexpected conversion form field").into()),
        }
    }
    let request = request.context("settings are required")?;
    let input = input.context("an audio file is required")?;
    let permit = compute(&state)?;
    let (wav, length) = tokio::task::spawn_blocking(move || -> Result<_> {
        let _permit = permit;
        let source = audio::read_mono(&input, ANALYSIS_RATE)?;
        let result = convert_voice(&state, &request, &source, Some(&input))?;
        let mut wav = tempfile::tempfile()?;
        audio::write_wav(
            BufWriter::new(&mut wav),
            &result.samples,
            result.rate,
            1,
            false,
        )?;
        let length = wav.metadata()?.len();
        wav.rewind()?;
        Ok((wav, length))
    })
    .await
    .map_err(anyhow::Error::from)??;
    Response::builder()
        .header(header::CONTENT_TYPE, "audio/wav")
        .header(header::CACHE_CONTROL, "no-store")
        .header(header::CONTENT_LENGTH, length)
        .body(Body::from_stream(ReaderStream::new(
            tokio::fs::File::from_std(wav),
        )))
        .map_err(|error| anyhow::Error::from(error).into())
}

async fn create_manifest(
    State(state): State<Shared>,
    Json(request): Json<ManifestRequest>,
) -> WebResult<Json<Value>> {
    let result = tokio::task::spawn_blocking(move || -> Result<Value> {
        let experiment = state.paths.root.join(request.experiment);
        let root = request.root.map(|path| state.paths.root.join(path));
        if let Some(mut rows) = request.rows {
            let base = root.as_deref().unwrap_or(&state.paths.root);
            for row in &mut rows {
                if !row.path.trim().is_empty() {
                    row.path = base.join(row.path.trim()).to_string_lossy().into_owned();
                }
            }
            let result = dataset::manifest_from_rows(&rows, root.as_deref())?;
            let path = dataset::save_manifest(&experiment, &result.manifest)?;
            Ok(json!({ "path": path, "invalid_rows": result.invalid_rows, "manifest": result.manifest }))
        } else {
            let manifest = dataset::scan_speakers(root.as_deref().context("dataset folder is required")?)?;
            let path = dataset::save_manifest(&experiment, &manifest)?;
            Ok(json!({ "path": path, "manifest": manifest }))
        }
    }).await.map_err(anyhow::Error::from)??;
    Ok(Json(result))
}

async fn start_job(
    State(state): State<Shared>,
    Json(action): Json<Action>,
) -> WebResult<StatusCode> {
    let operation = action.name();
    let id = launch_job(state, operation, move |state, cancelled| {
        action.run(state, cancelled, |message| {
            job_progress(state, cancelled, message)
        })
    })?;
    Ok(if id.is_some() {
        StatusCode::ACCEPTED
    } else {
        StatusCode::CONFLICT
    })
}

fn launch_job(
    state: Shared,
    operation: &str,
    run: impl FnOnce(&StateData, &AtomicBool) -> Result<Value> + Send + 'static,
) -> Result<Option<uuid::Uuid>> {
    let Ok(permit) = compute(&state) else {
        return Ok(None);
    };
    let cancelled = Arc::new(AtomicBool::new(false));
    let Some(id) = reserve_job(&state, operation, &cancelled)? else {
        return Ok(None);
    };
    tokio::spawn(async move {
        let work_state = Arc::clone(&state);
        let work_cancelled = Arc::clone(&cancelled);
        let result = tokio::task::spawn_blocking(move || run(&work_state, &work_cancelled))
            .await
            .map_err(anyhow::Error::from)
            .and_then(|result| result);
        let mut job = state.job.lock().unwrap_or_else(|error| error.into_inner());
        finish_job(&mut job, &cancelled, result);
        drop(permit);
    });
    Ok(Some(id))
}

fn reserve_job(
    state: &StateData,
    operation: &str,
    cancelled: &Arc<AtomicBool>,
) -> Result<Option<uuid::Uuid>> {
    let mut job = lock(&state.job)?;
    if job.status == JobState::Running {
        return Ok(None);
    }
    let id = uuid::Uuid::new_v4();
    let previous = std::mem::replace(
        &mut *job,
        Job {
            id,
            operation: operation.to_owned(),
            status: JobState::Running,
            cancelled: Arc::clone(cancelled),
            ..Default::default()
        },
    );
    if previous.status != JobState::Idle {
        let mut recent = lock(&state.recent_jobs)?;
        if recent.len() == 8 {
            recent.pop_front();
        }
        recent.push_back(previous);
    }
    Ok(Some(id))
}

fn job_progress(state: &StateData, cancelled: &AtomicBool, message: &str) -> ControlFlow<()> {
    if cancelled.load(Ordering::Relaxed) {
        return ControlFlow::Break(());
    }
    if let Ok(mut job) = state.job.lock()
        && job.message != message
    {
        job.message = message.to_owned();
        if job.log.len() == 40 {
            job.log.pop_front();
        }
        job.log.push_back(message.to_owned());
    }
    ControlFlow::Continue(())
}

fn finish_job(job: &mut Job, cancelled: &AtomicBool, result: Result<Value>) {
    match result {
        Ok(result) => {
            job.status = JobState::Complete;
            job.result = Some(result);
        }
        Err(error) => {
            job.status = if cancelled.load(Ordering::Relaxed) {
                JobState::Cancelled
            } else {
                JobState::Failed
            };
            job.message = format!("{error:#}");
        }
    }
}

async fn job_status(State(state): State<Shared>) -> WebResult<Json<Value>> {
    Ok(Json(json!(*lock(&state.job)?)))
}

async fn cancel_job(State(state): State<Shared>) -> WebResult<StatusCode> {
    let job = lock(&state.job)?;
    if job.status == JobState::Running {
        job.cancelled.store(true, Ordering::Relaxed);
    }
    Ok(StatusCode::NO_CONTENT)
}

#[cfg(feature = "realtime")]
struct Live {
    stop: std::sync::mpsc::Sender<()>,
    state: Arc<Mutex<Value>>,
    controls: Arc<Mutex<Option<crate::realtime::Controls>>>,
    thread: std::thread::JoinHandle<()>,
}

#[cfg(feature = "realtime")]
async fn realtime_profile(
    State(state): State<Shared>,
) -> WebResult<Json<crate::realtime::Request>> {
    Ok(Json(crate::realtime::profile(&state.paths)?))
}

#[cfg(feature = "realtime")]
async fn devices(
    State(_): State<Shared>,
    axum::extract::Query(options): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> WebResult<Json<Value>> {
    let devices = tokio::task::spawn_blocking(move || {
        crate::realtime::device::devices(options.get("host").map(String::as_str))
    })
    .await
    .map_err(anyhow::Error::from)??;
    Ok(Json(json!(devices)))
}

#[cfg(feature = "realtime")]
async fn live_status(State(state): State<Shared>) -> WebResult<Json<Value>> {
    let current = lock(&state.realtime)?;
    let result = match &*current {
        Some(live) => lock(&live.state)?.clone(),
        None => json!({"status": "idle"}),
    };
    Ok(Json(result))
}

#[cfg(feature = "realtime")]
async fn start_live(
    State(state): State<Shared>,
    Json(request): Json<crate::realtime::Request>,
) -> WebResult<StatusCode> {
    let mut current = lock(&state.realtime)?;
    if current
        .as_ref()
        .is_some_and(|live| !live.thread.is_finished())
    {
        return Ok(StatusCode::CONFLICT);
    }
    let permit = compute(&state)?;
    let (stop, run_until) = std::sync::mpsc::channel();
    let status = Arc::new(Mutex::new(json!({ "status": "starting" })));
    let status_thread = Arc::clone(&status);
    let controls = Arc::new(Mutex::new(None));
    let controls_thread = Arc::clone(&controls);
    let paths = state.paths.clone();
    let thread = std::thread::Builder::new()
        .name("rvc-realtime-control".into())
        .spawn(move || {
            let _permit = permit;
            let mark = |value| {
                if let Ok(mut status) = status_thread.lock() {
                    *status = value;
                }
            };
            let result = (|| -> Result<()> {
                if let Some(model) = request.model.as_deref() {
                    crate::assets::voice(&paths, &paths.model(model)?, request.settings.f0_method, &mut |line: &str| {
                        mark(json!({ "status": "starting", "message": line }));
                        ControlFlow::Continue(())
                    })?;
                }
                let (stream, controller) =
                    crate::realtime::start_controlled(&paths, request.clone())?;
                let persistence = crate::realtime::save_profile(&paths, &request)
                    .err()
                    .map(|error| format!("{error:#}"));
                *lock(&controls_thread)? = Some(controller.clone());
                loop {
                    stream.check()?;
                    let message = controller.error().or_else(|| persistence.clone());
                    mark(json!({
                        "status": "running", "format": stream.format(),
                        "stats": stream.stats(), "message": message,
                        "active_f0_method": controller.active_f0_method(),
                    }));
                    match run_until.recv_timeout(std::time::Duration::from_millis(250)) {
                        Ok(()) | Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
                        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
                    }
                }
                stream.stop()
            })();
            if let Ok(mut controls) = controls_thread.lock() {
                *controls = None;
            }
            if let Err(error) = result {
                mark(json!({"status": "failed", "message": format!("{error:#}")}));
            } else {
                mark(json!({"status": "idle"}));
            }
        })
        .map_err(anyhow::Error::from)?;
    *current = Some(Live {
        stop,
        state: status,
        controls,
        thread,
    });
    Ok(StatusCode::ACCEPTED)
}

#[cfg(feature = "realtime")]
async fn update_live(
    State(state): State<Shared>,
    Json(update): Json<crate::realtime::LiveSettings>,
) -> WebResult<StatusCode> {
    let current = lock(&state.realtime)?;
    let live = current.as_ref().context("audio is not running")?;
    let controls = lock(&live.controls)?;
    controls
        .as_ref()
        .context("audio is not ready")?
        .update(update)?;
    Ok(StatusCode::NO_CONTENT)
}

#[cfg(feature = "realtime")]
async fn stop_live(State(state): State<Shared>) -> WebResult<StatusCode> {
    if let Some(live) = lock(&state.realtime)?.as_ref() {
        let _ = live.stop.send(());
    }
    Ok(StatusCode::NO_CONTENT)
}

type WebResult<T> = std::result::Result<T, WebError>;

struct WebError(anyhow::Error);

impl From<anyhow::Error> for WebError {
    fn from(error: anyhow::Error) -> Self {
        Self(error)
    }
}

impl IntoResponse for WebError {
    fn into_response(self) -> Response {
        (
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": format!("{:#}", self.0) })),
        )
            .into_response()
    }
}

fn lock<T>(guard: &Mutex<T>) -> Result<std::sync::MutexGuard<'_, T>> {
    guard
        .lock()
        .map_err(|_| anyhow::anyhow!("WebUI state unavailable"))
}
