use std::io::{Cursor, Write};
use std::ops::ControlFlow;
use std::time::{SystemTime, UNIX_EPOCH};

use base64::{Engine, engine::general_purpose::STANDARD};
use uuid::Uuid;
use zip::{CompressionMethod, ZipWriter, write::SimpleFileOptions};

use super::*;
use crate::audio::Format;
use crate::pymss::pcm::Pcm;
use crate::separation::{AudioEncoding, write_encoded_track};

#[derive(Deserialize)]
struct JsonInput {
    model: String,
    input: AudioField,
    stems: Option<Stems>,
    response_format: Option<String>,
    output_audio_format: Option<String>,
}

#[derive(Deserialize)]
struct AudioField {
    format: String,
    channels: usize,
    sample_rate: u32,
    data: String,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Stems {
    List(Vec<String>),
    Text(String),
}

impl Stems {
    fn names(self) -> Vec<String> {
        match self {
            Self::List(names) => names,
            Self::Text(names) => names
                .split(',')
                .map(|part| part.trim().to_owned())
                .collect(),
        }
    }
}

struct Input {
    model: String,
    format: String,
    rate: u32,
    channels: usize,
    data: Bytes,
    stems: Vec<String>,
    zip: bool,
    output: Output,
}

#[derive(Clone, Copy)]
enum Output {
    Pcm,
    File(Format),
}

impl Output {
    fn name(self) -> &'static str {
        match self {
            Self::Pcm => "pcm_f32le",
            Self::File(format) => format.extension(),
        }
    }

    fn extension(self) -> &'static str {
        if matches!(self, Self::Pcm) {
            "f32le"
        } else {
            self.name()
        }
    }
}

pub(super) async fn separate(
    State(state): State<Shared>,
    Query(query): Query<HashMap<String, String>>,
    headers: HeaderMap,
    body: Bytes,
) -> Api<Response> {
    let model = state.use_model()?;
    let content = headers
        .get(header::CONTENT_TYPE)
        .and_then(|header| header.to_str().ok())
        .and_then(|header| header.split(';').next())
        .unwrap_or("");
    let input = parse(content, &query, body)?;
    requested_model(Some(&model), &input.model)?;
    let pcm = decode(&input, model.rate, state.options.max_audio_seconds)?;
    let permit = Arc::clone(&state.queue).try_acquire_owned().map_err(|_| {
        Failure::new(
            StatusCode::TOO_MANY_REQUESTS,
            "separation_queue_full",
            "The separation queue is full",
        )
    })?;
    let pending = Arc::new(AtomicBool::new(false));
    let cancelled = Arc::clone(&pending);
    let root = state.root.clone();
    let request_timeout = Duration::try_from_secs_f64(state.options.request_timeout_seconds)
        .map_err(anyhow::Error::from)?;
    let worker = tokio::task::spawn_blocking(move || -> Api<Response> {
        let _permit = permit;
        let duration = pcm.duration();
        let pcm = pcm.convert_layout(model.rate, model.channels)?;
        let engine = lock(&model.engine)?;
        let channels = engine.output_channels();
        let stems = engine
            .separate_planar(&pcm.planar, &input.stems, |_| {
                if cancelled.load(Ordering::Relaxed) {
                    ControlFlow::Break(())
                } else {
                    ControlFlow::Continue(())
                }
            })
            .map_err(|error| {
                Failure::new(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "separation_failed",
                    format!("{error:#}"),
                )
            })?;
        drop(engine);
        let stems: Vec<_> = stems
            .into_iter()
            .map(|(name, planar)| {
                (
                    name,
                    Pcm {
                        planar,
                        channels,
                        rate: model.rate,
                    },
                )
            })
            .collect();
        if input.zip {
            let archive = zip_response(&root, &input.model, &stems, input.output, duration)?;
            Ok(([(header::CONTENT_TYPE, "application/zip")], archive).into_response())
        } else {
            Ok(Json(json_response(&input.model, &model.device, &stems, duration)).into_response())
        }
    });
    if request_timeout.is_zero() {
        worker.await.map_err(anyhow::Error::from)?
    } else {
        match tokio::time::timeout(request_timeout, worker).await {
            Ok(result) => result.map_err(anyhow::Error::from)?,
            Err(_) => {
                pending.store(true, Ordering::Relaxed);
                Err(Failure::new(
                    StatusCode::GATEWAY_TIMEOUT,
                    "request_timeout",
                    "Audio separation timed out",
                ))
            }
        }
    }
}

fn parse(content: &str, query: &HashMap<String, String>, body: Bytes) -> Api<Input> {
    let (model, format, rate, channels, data, stems, response, output) = match content {
        "application/json" => {
            let body: JsonInput = serde_json::from_slice(&body)
                .map_err(|error| Failure::bad("invalid_request", error.to_string()))?;
            let data = STANDARD
                .decode(body.input.data.as_bytes())
                .map_err(|error| Failure::bad("invalid_base64", error.to_string()))?;
            (
                body.model,
                body.input.format,
                body.input.sample_rate,
                body.input.channels,
                Bytes::from(data),
                body.stems.map(Stems::names).unwrap_or_default(),
                body.response_format,
                body.output_audio_format,
            )
        }
        "application/octet-stream" => {
            let required = |name| {
                query.get(name).ok_or_else(|| {
                    Failure::bad("missing_audio_metadata", format!("{name} is required"))
                })
            };
            let rate = required("sample_rate")?.parse().map_err(|_| {
                Failure::bad("invalid_query_parameter", "sample_rate must be an integer")
            })?;
            let channels = required("channels")?.parse().map_err(|_| {
                Failure::bad("invalid_query_parameter", "channels must be an integer")
            })?;
            let stems = query
                .get("stems")
                .map(|stems| Stems::Text(stems.clone()).names())
                .unwrap_or_default();
            (
                required("model")?.clone(),
                required("format")?.clone(),
                rate,
                channels,
                body,
                stems,
                query.get("response_format").cloned(),
                query.get("output_audio_format").cloned(),
            )
        }
        _ => {
            return Err(Failure::new(
                StatusCode::UNSUPPORTED_MEDIA_TYPE,
                "unsupported_content_type",
                "Use JSON or binary PCM",
            ));
        }
    };
    let zip = match response
        .as_deref()
        .unwrap_or("json")
        .to_lowercase()
        .as_str()
    {
        "json" => false,
        "zip" => true,
        _ => {
            return Err(Failure::bad(
                "invalid_response_format",
                "response_format must be json or zip",
            ));
        }
    };
    let output = output.as_deref().unwrap_or("pcm_f32le").to_lowercase();
    let invalid_output = || {
        Failure::bad(
            "invalid_output_audio_format",
            "requested audio format is unavailable for this response",
        )
    };
    let output = if output == "pcm_f32le" {
        Output::Pcm
    } else if zip {
        Output::File(Format::from_extension(output.as_ref()).map_err(|_| invalid_output())?)
    } else {
        return Err(invalid_output());
    };
    Ok(Input {
        model,
        format: format.to_lowercase(),
        rate,
        channels,
        data,
        stems,
        zip,
        output,
    })
}

fn decode(input: &Input, rate: u32, maximum: f64) -> Api<Pcm> {
    let width = match input.format.as_str() {
        "pcm_f32le" => 4,
        "pcm_s16le" => 2,
        _ => {
            return Err(Failure::bad(
                "invalid_audio_format",
                "use pcm_f32le or pcm_s16le",
            ));
        }
    };
    if !matches!(input.channels, 1 | 2) {
        return Err(Failure::bad(
            "invalid_channel_count",
            "channels must be 1 or 2",
        ));
    }
    if input.rate != rate {
        return Err(Failure::bad(
            "invalid_sample_rate",
            format!("sample_rate must be {rate}"),
        ));
    }
    if input.data.is_empty() {
        return Err(Failure::bad("empty_audio", "PCM data is empty"));
    }
    let frame = width * input.channels;
    if !input.data.len().is_multiple_of(frame) {
        return Err(Failure::bad(
            "invalid_audio_length",
            "PCM contains an incomplete frame",
        ));
    }
    let seconds = (input.data.len() / frame) as f64 / f64::from(rate);
    if maximum > 0.0 && seconds > maximum {
        return Err(Failure::new(
            StatusCode::PAYLOAD_TOO_LARGE,
            "request_too_large",
            format!("audio exceeds {maximum} seconds"),
        ));
    }
    let values: Vec<_> = if width == 2 {
        input
            .data
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&packed| f32::from(i16::from_le_bytes(packed)) / 32768.0)
            .collect()
    } else {
        input
            .data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&packed| f32::from_le_bytes(packed))
            .collect()
    };
    Pcm::from_interleaved(&values, input.channels, rate)
        .map_err(|error| Failure::bad("invalid_audio_data", error.to_string()))
}

fn pcm_bytes(audio: &Pcm) -> Vec<u8> {
    audio
        .interleaved()
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect()
}

fn json_response(model: &str, device: &str, stems: &[(String, Pcm)], duration: f64) -> Value {
    let output: Vec<_> = stems.iter().map(|(stem, audio)| {
        json!({"stem":stem,"audio":{"format":"pcm_f32le","sample_rate":audio.rate,"channels":audio.channels,
            "data":STANDARD.encode(pcm_bytes(audio))}})
    }).collect();
    let names: Vec<_> = stems.iter().map(|(name, _)| name).collect();
    json!({
        "id": id(), "object":"audio.separation",
        "created": SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs(),
        "model":model, "outputs":output,
        "metadata":{"input_seconds":duration,"output_stems":names,"device":device},
        "usage":{"type":"duration","seconds":duration},
    })
}

fn zip_response(
    root: &Path,
    model: &str,
    stems: &[(String, Pcm)],
    format: Output,
    duration: f64,
) -> Result<Vec<u8>> {
    let mut zip = ZipWriter::new(Cursor::new(Vec::new()));
    let options = SimpleFileOptions::default().compression_method(CompressionMethod::Deflated);
    let mut entries = Vec::new();
    let encoding = AudioEncoding {
        m4a_bit_rate: 512_000,
        ..AudioEncoding::default()
    };
    let unsupported_filename_chars = regex::Regex::new(r"[^a-z0-9_-]+")?;
    for (i, (stem, signal)) in stems.iter().enumerate() {
        let slug = stem.to_ascii_lowercase();
        let slug = unsupported_filename_chars.replace_all(&slug, "-");
        let slug = slug.trim_matches(['-', '.', '_']);
        let slug = if slug.is_empty() { "stem" } else { slug };
        let filename = format!("{:04}-{slug}.{}", i + 1, format.extension());
        zip.start_file(&filename, options)?;
        let bytes = match format {
            Output::Pcm => pcm_bytes(signal),
            Output::File(format) => {
                let temporary = tempfile::TempDir::new()?;
                let file = temporary
                    .path()
                    .join(format!("stem.{}", format.extension()));
                write_encoded_track(
                    root,
                    &file,
                    &signal.planar,
                    signal.rate,
                    signal.channels,
                    format,
                    &encoding,
                    false,
                )?;
                std::fs::read(file)?
            }
        };
        zip.write_all(&bytes)?;
        entries.push(json!({
            "stem":stem,"filename":filename,"format":format.name(),"sample_rate":signal.rate,"channels":signal.channels
        }));
    }
    zip.start_file("manifest.json", options)?;
    zip.write_all(&serde_json::to_vec_pretty(&json!({
        "id":id(),"object":"audio.separation","model":model,"outputs":entries,
        "usage":{"type":"duration","seconds":duration}
    }))?)?;
    Ok(zip.finish()?.into_inner())
}

fn id() -> String {
    format!("sep_{}", Uuid::new_v4().simple())
}
