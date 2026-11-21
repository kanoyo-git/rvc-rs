pub(super) mod manifest;
mod training;

use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{
    Arc, LazyLock,
    atomic::{AtomicBool, Ordering},
};
use std::time::Instant;

use anyhow::{Context, Result, bail, ensure};
use base64::{engine::general_purpose::STANDARD, read::DecoderReader};
use clap::ValueEnum;
use regex::Regex;
use serde_json::{Value, json};
use tempfile::TempDir;

use crate::checkpoint::Checkpoint;

use super::super::{
    ANALYSIS_RATE, Action, ConversionSettings, Convert, Format, ModelInfo, StateData, audio,
    compute, conversion_jobs, convert_voice, finish_job, installed_model, job_progress, lock,
    model, model_output, reserve_job, separation,
};

const MAX_AUDIO: u64 = 128 * 1024 * 1024;

pub(super) fn generates(index: usize) -> bool {
    matches!(index, 5 | 10 | 12 | 16 | 21 | 23 | 25)
}

pub(super) fn run(
    state: &StateData,
    session: Option<&str>,
    index: usize,
    data: Vec<Value>,
    cancelled: Arc<AtomicBool>,
    mut progress: impl FnMut(Vec<Value>),
) -> Result<Vec<Value>> {
    let args = Args(data);
    match index {
        0 => {
            *lock(&state.legacy.voice)? = None;
            *lock(&state.voice)? = None;
            Ok(vec![update(json!({"value":""}))])
        }
        1 => Ok(vec![voice_choices(state)?]),
        2 | 4 => {
            if let Some(path) = index_path(state, args.text(0)?)
                && !path.is_file()
            {
                bail!("Index file does not exist: {}", path.display())
            }
            Ok(vec![])
        }
        3 => {
            let result = convert(state, &args);
            Ok(match result {
                Ok(Some(audio)) => vec![json!(status("Single Inference", "Success", "")), audio],
                Ok(None) => vec![
                    json!(status(
                        "Single Inference",
                        "Waiting for input",
                        "Upload an audio file"
                    )),
                    Value::Null,
                ],
                Err(error) => vec![
                    json!(status("Single Inference", "Failed", &format!("{error:#}"))),
                    Value::Null,
                ],
            })
        }
        5 => batch(state, &args, &cancelled, progress),
        6 => choose_voice(state, &args),
        7 | 8 => {
            let speaker = if index == 7 {
                speaker(&args, 1, 2).ok()
            } else if args.at(1).is_null() {
                Some(0)
            } else {
                speaker(&args, 0, 1).ok()
            };
            let path = selected_index(state, args.text(0)?, speaker);
            Ok(vec![path.clone(), path])
        }
        9 => {
            let preset = separation_preset(args.text(0)?)?;
            let architecture = match preset.architecture {
                separation::Architecture::BandSplitRoformer => "bs_roformer",
                separation::Architecture::MelBandRoformer => "mel_band_roformer",
            };
            let id = preset.id;
            Ok(vec![json!(format!("{architecture} | {id}"))])
        }
        10 => separate(state, &args, cancelled, progress),
        11 | 13 | 17 | 22 | 24 | 26 => stop(state, index),
        12 | 14..=16 | 18..=21 | 23 | 25 => {
            training::run(state, index, &args, cancelled, &mut progress)
        }
        27..=31 => manifest::run(state, session, index, &args),
        32 | 33 => {
            let source = args.text(0)?;
            let fields = if source == args.text(1)? {
                json!({})
            } else {
                json!({"value":source})
            };
            Ok(vec![update(fields)])
        }
        34..=38 => checkpoint(state, index, &args),
        _ => bail!("unknown Gradio function index {index}"),
    }
}

struct Args(Vec<Value>);

impl Args {
    fn at(&self, index: usize) -> &Value {
        self.0.get(index).unwrap_or(&Value::Null)
    }

    fn text(&self, index: usize) -> Result<&str> {
        match self.at(index) {
            Value::Null => Ok(""),
            Value::String(value) => Ok(value),
            _ => bail!("argument {} must be text", index + 1),
        }
    }

    fn number(&self, index: usize) -> Result<f64> {
        self.at(index)
            .as_f64()
            .or_else(|| self.at(index).as_str()?.trim().parse().ok())
            .filter(|value| value.is_finite())
            .with_context(|| format!("argument {} must be a number", index + 1))
    }

    fn integer<T>(&self, index: usize) -> Result<T>
    where
        T: TryFrom<i64>,
        T::Error: Into<anyhow::Error>,
    {
        let value = self.number(index)?;
        ensure!(
            (i64::MIN as f64..i64::MAX as f64).contains(&value),
            "argument {} is outside the integer range",
            index + 1
        );
        T::try_from(value as i64).map_err(Into::into)
    }

    fn boolean(&self, index: usize) -> Result<bool> {
        let value = self.at(index);
        match value {
            Value::Bool(value) => Ok(*value),
            Value::Null => Ok(false),
            Value::Number(_) => Ok(self.integer::<i64>(index)? != 0),
            Value::String(value) => Ok(matches!(
                value.trim().to_lowercase().as_str(),
                "1" | "true" | "yes" | "是" | "да"
            )),
            _ => bail!("argument {} must be a boolean", index + 1),
        }
    }
}

fn update(mut fields: Value) -> Value {
    fields["__type__"] = json!("update");
    fields
}

fn status(title: &str, state: &str, detail: &str) -> String {
    let base = format!("【{title}】\nStatus：{state}");
    if detail.is_empty() {
        base
    } else {
        format!("{base}\n\n{detail}")
    }
}

fn voice_choices(state: &StateData) -> Result<Value> {
    Ok(update(json!({"choices":state.paths.models()?})))
}

fn speaker(args: &Args, slider: usize, dropdown: usize) -> Result<i64> {
    static LABEL: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"ID\s*[:：]\s*(\d+)\s*[)）]?\s*$").unwrap());
    if let Some(text) = args.at(dropdown).as_str() {
        if let Some(label) = LABEL.captures(text) {
            return Ok(label[1].parse()?);
        }
        if text.trim().is_empty() {
            return args.integer(slider);
        }
    }
    args.integer(if args.at(dropdown).is_null() {
        slider
    } else {
        dropdown
    })
}

fn trim_path(value: &str) -> &str {
    value.trim_matches(|c: char| c.is_whitespace() || matches!(c, '"' | '\u{202a}'))
}

fn index_path(state: &StateData, path: &str) -> Option<PathBuf> {
    let path = trim_path(path);
    (!path.is_empty()).then(|| {
        let path = Path::new(path);
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or_default();
        state
            .paths
            .root
            .join(path.with_file_name(crate::catalog::added_name(name)))
    })
}

fn selected_index(state: &StateData, model: &str, speaker: Option<i64>) -> Value {
    let path = (!model.is_empty())
        .then(|| state.paths.select_index(Path::new(model), speaker))
        .flatten()
        .map(|path| path.to_string_lossy().into_owned())
        .unwrap_or_default();
    update(json!({"value":path}))
}

fn choose_voice(state: &StateData, args: &Args) -> Result<Vec<Value>> {
    let name = if args.at(0).as_array().is_some_and(Vec::is_empty) {
        ""
    } else {
        args.text(0)?
    };
    let mut selection = lock(&state.legacy.voice)?;
    let previous_f0 = selection
        .as_ref()
        .and_then(|name| installed_model(&state.paths, name).ok())
        .and_then(|path| ModelInfo::read(&path).ok())
        .is_none_or(|info| info.uses_f0);
    let protect = |index, fallback| {
        let value = if previous_f0 {
            args.at(index).clone()
        } else {
            json!(fallback)
        };
        update(json!({"visible":previous_f0,"value":value}))
    };
    let protect_single = protect(1, 0.5);
    let protect_batch = protect(2, 0.33);
    if name.is_empty() {
        *selection = None;
        *lock(&state.voice)? = None;
        return Ok(vec![
            update(json!({"visible":false})),
            update(json!({"visible":false,"value":null})),
            update(json!({"visible":true,"value":protect_single})),
            update(json!({"visible":true,"value":protect_batch})),
            json!(""),
            json!(""),
        ]);
    }
    let model_path = installed_model(&state.paths, name)?;
    let info = ModelInfo::read(&model_path)?;
    let (slider, dropdown, id) = if let Some((&id, _)) = info.speakers.first_key_value() {
        let choices: Vec<_> = info
            .speakers
            .iter()
            .map(|(id, name)| format!("Speaker: {name} (ID: {id})"))
            .collect();
        (
            update(json!({"visible":false,"value":id})),
            update(json!({"visible":true,"value":choices[0],"choices":choices})),
            id,
        )
    } else {
        (
            update(json!({"visible":true,"maximum":info.speaker_count - 1})),
            update(json!({"visible":false,"value":null})),
            0,
        )
    };
    if selection.as_deref() != Some(name) {
        *lock(&state.voice)? = None;
    }
    *selection = Some(name.to_owned());
    let index = selected_index(state, name, Some(id));
    Ok(vec![
        slider,
        dropdown,
        protect_single,
        protect_batch,
        index.clone(),
        index,
    ])
}

fn conversion(state: &StateData, args: &Args, settings_start: usize) -> Result<Convert> {
    let selected = lock(&state.legacy.voice)?
        .clone()
        .context("select a voice with infer_change_voice")?;
    let index = index_path(state, args.text(settings_start + 2)?).filter(|path| path.is_file());
    let rate: i32 = args.integer(settings_start + 4)?;
    let settings = ConversionSettings {
        pitch: args.integer::<i32>(settings_start)? as f32,
        f0_method: model::PitchMethod::from_str(args.text(settings_start + 1)?, true)
            .map_err(anyhow::Error::msg)?,
        index_rate: if index.is_some() {
            args.number(settings_start + 3)? as f32
        } else {
            0.0
        },
        resample_sr: if rate < 16_000 { 0 } else { rate as u32 },
        rms_mix_rate: args.number(settings_start + 5)? as f32,
        protect: args.number(settings_start + 6)? as f32,
        ..Default::default()
    };
    Ok(Convert {
        model: selected,
        speaker_id: Some(speaker(args, 0, 1)?),
        index,
        device: "auto".to_owned(),
        settings,
        auto_register: false,
    })
}

struct AudioInput {
    path: PathBuf,
    name: String,
    crop: [f64; 2],
}

fn audio_input(state: &StateData, source: &Value, staging: &Path) -> Result<AudioInput> {
    let filename = source
        .as_str()
        .or_else(|| source["name"].as_str())
        .context("audio file name required")?;
    let name = source["orig_name"]
        .as_str()
        .unwrap_or(filename)
        .rsplit(['/', '\\'])
        .next()
        .filter(|name| !name.is_empty() && !matches!(*name, "." | ".."))
        .context("invalid audio file name")?
        .to_owned();
    let staged = || staging.join(format!("{}-{name}", uuid::Uuid::new_v4()));
    let path = if let Some(raw) = source["data"].as_str() {
        let raw = if raw.starts_with("data:") {
            raw.split_once(";base64,")
                .context("invalid base64 audio data")?
                .1
        } else {
            raw
        };
        let path = staged();
        let mut file = std::fs::File::create(&path)?;
        let decoded = DecoderReader::new(Cursor::new(raw.as_bytes()), &STANDARD);
        let count = std::io::copy(&mut std::io::Read::take(decoded, MAX_AUDIO + 1), &mut file)?;
        ensure!(count <= MAX_AUDIO, "audio file exceeds 128 MiB");
        path
    } else if filename.starts_with("https://") || filename.starts_with("http://") {
        let path = staged();
        let reply = Command::new("curl")
            .args([
                "-q",
                "--silent",
                "--show-error",
                "--fail",
                "--globoff",
                "--location",
                "--max-redirs",
                "5",
                "--proto",
                "=http,https",
                "--proto-redir",
                "=http,https",
                "--max-time",
                "30",
                "--max-filesize",
                &MAX_AUDIO.to_string(),
                "--url",
                filename,
                "--output",
            ])
            .arg(&path)
            .output()
            .context("downloading Gradio audio")?;
        ensure!(
            reply.status.success(),
            "downloading Gradio audio: {}",
            String::from_utf8_lossy(&reply.stderr).trim()
        );
        ensure!(
            path.metadata()?.len() <= MAX_AUDIO,
            "audio file exceeds 128 MiB"
        );
        path
    } else {
        let path = state
            .legacy
            .open_path(filename)
            .unwrap_or_else(|| state.paths.root.join(trim_path(filename)));
        ensure!(
            path.is_file(),
            "audio file does not exist: {}",
            path.display()
        );
        path
    };
    let crop = [
        source["crop_min"].as_f64().unwrap_or(0.0),
        source["crop_max"].as_f64().unwrap_or(100.0),
    ];
    ensure!(
        0.0 <= crop[0] && crop[0] < crop[1] && crop[1] <= 100.0,
        "invalid audio crop"
    );
    Ok(AudioInput { path, name, crop })
}

fn convert(state: &StateData, args: &Args) -> Result<Option<Value>> {
    if args.at(2).is_null() {
        return Ok(None);
    }
    let _permit = compute(state)?;
    let staging = TempDir::new()?;
    let input = audio_input(state, args.at(2), staging.path())?;
    let request = conversion(state, args, 3)?;
    let samples = audio::read_mono(&input.path, ANALYSIS_RATE)?;
    let [start, end] = input
        .crop
        .map(|value| (samples.len() as f64 * value / 100.0).round() as usize);
    let result = convert_voice(state, &request, &samples[start..end], None)?;
    let name = format!("{}.wav", uuid::Uuid::new_v4());
    audio::write_audio(
        &state.legacy.files.path().join(&name),
        &result.samples,
        result.rate,
        Format::Wav,
        false,
    )?;
    Ok(Some(json!({"name":name,"data":null,"is_file":true})))
}

fn batch(
    state: &StateData,
    args: &Args,
    cancelled: &AtomicBool,
    mut progress: impl FnMut(Vec<Value>),
) -> Result<Vec<Value>> {
    let _permit = compute(state)?;
    let request = conversion(state, args, 5)?;
    let input = trim_path(args.text(2)?);
    let destination = trim_path(args.text(3)?);
    ensure!(!destination.is_empty(), "output folder is required");
    let destination = state.paths.root.join(destination);
    let format = Format::from_extension(args.text(12)?.as_ref())?;
    let staging = TempDir::new()?;
    let root = state.paths.root.join(input);
    let jobs = if !input.is_empty() && root.is_dir() {
        conversion_jobs(root, destination, Some(format), false)?
    } else {
        let uploaded = args.at(4).as_array().map(Vec::as_slice).unwrap_or_default();
        let mut jobs = Vec::with_capacity(uploaded.len());
        for file in uploaded {
            let audio = audio_input(state, file, staging.path())?;
            let output = destination
                .join(audio.name)
                .with_extension(format.extension());
            jobs.push((audio.path, output, format));
        }
        jobs
    };
    let total = jobs.len();
    let mut passed = 0;
    let mut failures = Vec::new();
    let batch_status = |current, successful, errors: &[String]| {
        json!(status(
            "Batch Inference",
            if current == total {
                "Completed"
            } else {
                "Running"
            },
            &format!(
                "{current}/{total}, successful {successful}, failed {}\n{}",
                errors.len(),
                errors.join("\n")
            )
        ))
    };
    for (index, (path, output, format)) in jobs.into_iter().enumerate() {
        ensure!(
            !cancelled.load(Ordering::Relaxed),
            "batch conversion cancelled"
        );
        let outcome = audio::read_mono(&path, ANALYSIS_RATE)
            .and_then(|samples| convert_voice(state, &request, &samples, Some(&path)))
            .and_then(|audio| {
                audio::write_audio(&output, &audio.samples, audio.rate, format, true)
            });
        match outcome {
            Ok(()) => passed += 1,
            Err(error) => failures.push(format!("{}: {error:#}", path.display())),
        }
        progress(vec![batch_status(index + 1, passed, &failures)]);
    }
    Ok(vec![batch_status(total, passed, &failures)])
}

fn separation_preset(name: &str) -> Result<&'static separation::Preset> {
    separation::PRESETS
        .iter()
        .find(|preset| name.is_empty() || name == preset.id || name == preset.label)
        .context("unknown PyMSS preset")
}

fn separate(
    state: &StateData,
    args: &Args,
    cancelled: Arc<AtomicBool>,
    progress: impl FnMut(Vec<Value>),
) -> Result<Vec<Value>> {
    let primary = trim_path(args.text(2)?);
    let secondary = trim_path(args.text(4)?);
    ensure!(
        !primary.is_empty() && !secondary.is_empty(),
        "both output folders are required"
    );
    let input = trim_path(args.text(1)?);
    let staging = TempDir::new()?;
    let path = state.paths.root.join(input);
    let input = if !input.is_empty() && path.is_dir() {
        path
    } else {
        let directory = staging.path().join("audio");
        std::fs::create_dir(&directory)?;
        for file in args
            .at(3)
            .as_array()
            .context("an input folder or uploaded files are required")?
        {
            let file = audio_input(state, file, staging.path())?;
            let dest = directory.join(file.name);
            std::fs::copy(file.path, dest)?;
        }
        directory
    };
    let action = Action::Separate {
        input,
        output: state.paths.root.join(primary),
        preset: separation_preset(args.text(0)?)?.id.to_owned(),
        format: Format::from_extension(args.text(5)?.as_ref())?,
        secondary_output: Some(state.paths.root.join(secondary)),
        overwrite: true,
        device: "auto".to_owned(),
    };
    work(state, action, 10, cancelled, progress)
}

fn job_output(state: &StateData, index: usize, message: &str, running: bool) -> Result<Vec<Value>> {
    let mut result = vec![json!(message)];
    if matches!(index, 10 | 11) {
        let label = if running {
            "Separating audio"
        } else {
            "Separation stopped or finished"
        };
        result.push(json!(format!("<div role=\"status\">{label}</div>")));
    }
    result.push(update(json!({"visible":!running})));
    result.push(update(json!({"visible":running})));
    if matches!(index, 21 | 25) {
        result.push(if running {
            update(json!({}))
        } else {
            voice_choices(state)?
        });
    }
    Ok(result)
}

fn work(
    state: &StateData,
    action: Action,
    index: usize,
    cancelled: Arc<AtomicBool>,
    mut progress: impl FnMut(Vec<Value>),
) -> Result<Vec<Value>> {
    let _permit = compute(state)?;
    ensure!(
        reserve_job(state, action.name(), &cancelled)?.is_some(),
        "another WebUI task is running"
    );
    let title = match index {
        10 => "PyMSS separation",
        12 => "Data slicing",
        16 => "Feature extraction",
        21 => "Train model",
        23 => "Train feature index",
        25 => "One-click training",
        _ => action.name(),
    };
    let started = Instant::now();
    let mut last_progress = started;
    progress(job_output(
        state,
        index,
        &status(title, "Running", ""),
        true,
    )?);
    let result = action.run(state, &cancelled, |message| {
        if last_progress.elapsed().as_millis() >= 250
            && let Ok(step) = job_output(state, index, &status(title, "Running", message), true)
        {
            last_progress = Instant::now();
            progress(step);
        }
        job_progress(state, &cancelled, message)
    });
    let message = match &result {
        Ok(_) => status(title, "Completed", ""),
        Err(error) => status(
            title,
            if cancelled.load(Ordering::Relaxed) {
                "Stopped"
            } else {
                "Failed"
            },
            &format!("{error:#}"),
        ),
    };
    finish_job(&mut *lock(&state.job)?, &cancelled, result);
    job_output(state, index, &message, false)
}

fn stop(state: &StateData, index: usize) -> Result<Vec<Value>> {
    let operation = match index {
        11 => "separate",
        13 => "preprocess",
        17 => "extract",
        22 => "train",
        24 => "index",
        26 => "train_all",
        _ => unreachable!(),
    };
    let job = lock(&state.job)?;
    let active = job.status == super::super::JobState::Running && job.operation == operation;
    if active {
        job.cancelled.store(true, Ordering::Relaxed);
    }
    job_output(
        state,
        index,
        if active {
            "Stopping task"
        } else {
            "No active task"
        },
        false,
    )
}

fn named_output(state: &StateData, name: &str) -> Result<PathBuf> {
    ensure!(!name.is_empty(), "output model name is required");
    let file = if name.ends_with(".pth") {
        name.to_owned()
    } else {
        format!("{name}.pth")
    };
    model_output(&state.paths, &file)
}

fn checkpoint(state: &StateData, index: usize, args: &Args) -> Result<Vec<Value>> {
    let input = state.paths.root.join(trim_path(args.text(0)?));
    if index == 37 {
        return training::checkpoint_settings(&input);
    }
    let result = (|| -> Result<String> {
        let first = if input.is_file() || index == 38 {
            input
        } else {
            state.paths.model(Path::new(args.text(0)?))?
        };
        match index {
            34 => {
                let second = state.paths.model(Path::new(trim_path(args.text(1)?)))?;
                let output = named_output(state, args.text(6)?.trim())?;
                let sample_rate = args.text(3)?;
                training::rate(sample_rate)?;
                model::edit::merge_legacy(
                    &first,
                    &second,
                    &output,
                    model::edit::LegacyMerge {
                        alpha: args.number(2)?,
                        sample_rate,
                        uses_f0: args.boolean(4)?,
                        description: args.text(5)?,
                        version: training::version(args.text(7)?)?,
                    },
                )?;
            }
            35 => {
                let name = args.text(2)?.trim();
                let name = if name.is_empty() {
                    first
                        .file_name()
                        .and_then(|name| name.to_str())
                        .context("missing model filename")?
                } else {
                    name
                };
                model::edit::describe(&first, &named_output(state, name)?, args.text(1)?, true)?;
            }
            36 => {
                let model = Checkpoint::open(&first)?;
                let field = |name| -> Result<String> {
                    Ok(model
                        .metadata(name)?
                        .map(|value| {
                            value
                                .as_str()
                                .map(str::to_owned)
                                .unwrap_or_else(|| value.to_string())
                        })
                        .unwrap_or_else(|| "None".to_owned()))
                };
                return Ok(format!(
                    "Model information: {}\nSample rate: {}\nPitch guidance: {}\nVersion: {}",
                    field("info")?,
                    field("sr")?,
                    field("f0")?,
                    field("version")?
                ));
            }
            38 => {
                let description = args.text(4)?;
                let options = model::edit::Export {
                    version: training::version(args.text(5)?)?,
                    sample_rate: training::rate(args.text(2)?)?,
                    uses_f0: args.boolean(3)?,
                    description: if description.is_empty() {
                        "Model extracted from a training checkpoint"
                    } else {
                        description
                    },
                    overwrite: true,
                };
                model::edit::extract(&first, &named_output(state, args.text(1)?.trim())?, options)?;
            }
            _ => unreachable!(),
        }
        Ok("Success".to_owned())
    })();
    Ok(vec![json!(
        result.unwrap_or_else(|error| format!("{error:#}"))
    )])
}
