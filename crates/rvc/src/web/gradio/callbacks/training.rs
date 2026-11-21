use std::fs::File;
use std::io::{BufRead, BufReader, ErrorKind};
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock, atomic::AtomicBool};

use anyhow::{Context, Result, bail, ensure};
use clap::ValueEnum;
use regex::Regex;
use serde_json::{Value, json};

use super::{Action, Args, StateData, model, trim_path, update, work};
use crate::model::{PitchMethod, train::Request, train_data::Version};
use crate::web::DatasetMode;

pub(super) fn run(
    state: &StateData,
    index: usize,
    args: &Args,
    cancelled: Arc<AtomicBool>,
    progress: impl FnMut(Vec<Value>),
) -> Result<Vec<Value>> {
    let action = match index {
        12 => {
            let input = optional_path(args.text(0)?);
            Action::Preprocess {
                source: dataset_mode(multi(args.text(4)?), &input),
                input,
                experiment: experiment(state, args.text(1)?)?,
                sample_rate: rate(args.text(2)?)?,
                workers: args.integer(3)?,
                segment_seconds: segment_duration()?,
                denoise: 0.0,
            }
        }
        14 => {
            let multi = multi(args.text(0)?);
            let text = if multi {
                json!({"label":"Multi-speaker dataset folder","placeholder":"Use the saved manifest if empty","value":""})
            } else {
                json!({"label":"Training folder","placeholder":""})
            };
            return Ok(vec![update(text), update(json!({"visible":!multi}))]);
        }
        15 => {
            let visible = args.text(0)? == "rmvpe" && has_training_gpu();
            return Ok(vec![update(json!({"visible":visible}))]);
        }
        16 => {
            let method = args.boolean(3)?.then(|| pitch(args.text(2)?)).transpose()?;
            let mut devices = gpu_list(args.text(0)?)?;
            if devices.is_empty() && method == Some(PitchMethod::Rmvpe) {
                devices = gpu_list(args.text(6)?)?;
            }
            Action::Extract {
                embedder: None,
                experiment: experiment(state, args.text(4)?)?,
                version: version(args.text(5)?)?.name().to_owned(),
                f0_method: method,
                device: "auto".to_owned(),
                gpus: devices,
                overwrite: true,
                unvoice_fricatives: false,
            }
        }
        18..=20 => return pretrained_updates(state, index, args),
        21 => Action::Train {
            request: train_request(state, args, false)?,
        },
        #[cfg(feature = "index-training")]
        23 => Action::Index {
            experiment: experiment(state, args.text(0)?)?,
            version: match version(args.text(1)?)? {
                Version::V1 => crate::train_index::ModelVersion::V1,
                Version::V2 => crate::train_index::ModelVersion::V2,
            },
            mode: if multi(args.text(2)?) {
                crate::train_index::IndexMode::Multi
            } else {
                crate::train_index::IndexMode::Single
            },
            workers: std::thread::available_parallelism().map_or(1, usize::from),
        },
        #[cfg(feature = "index-training")]
        25 => {
            let request = train_request(state, args, true)?;
            let input = optional_path(args.text(3)?);
            Action::TrainAll {
                source: dataset_mode(request.multi, &input),
                request,
                input,
                workers: args.integer(5)?,
                segment_seconds: segment_duration()?,
                f0_method: pitch(args.text(6)?)?,
                denoise: 0.0,
            }
        }
        #[cfg(not(feature = "index-training"))]
        23 | 25 => bail!("index training is disabled in this build"),
        _ => unreachable!(),
    };
    work(state, action, index, cancelled, progress)
}

pub(super) fn experiment(state: &StateData, name: &str) -> Result<PathBuf> {
    let name = trim_path(name);
    ensure!(!name.is_empty(), "experiment name is required");
    Ok(state.paths.root.join("logs").join(name))
}

fn optional_path(value: &str) -> Option<PathBuf> {
    let value = trim_path(value);
    (!value.is_empty()).then(|| PathBuf::from(value))
}

fn dataset_mode(multi: bool, input: &Option<PathBuf>) -> DatasetMode {
    match (multi, input.is_some()) {
        (true, true) => DatasetMode::MultiFolder,
        (true, false) => DatasetMode::Manifest,
        (false, _) => DatasetMode::Single,
    }
}

fn multi(mode: &str) -> bool {
    matches!(
        mode.trim().to_lowercase().as_str(),
        "multiple speakers" | "multi" | "多说话人" | "несколько дикторов"
    )
}

pub(super) fn rate(value: &str) -> Result<u32> {
    match value {
        "32k" | "32000" => Ok(32_000),
        "40k" | "40000" => Ok(40_000),
        "48k" | "48000" => Ok(48_000),
        _ => bail!("unsupported training sample rate {value}"),
    }
}

pub(super) fn version(value: &str) -> Result<Version> {
    Version::from_str(value, true).map_err(anyhow::Error::msg)
}

fn pitch(value: &str) -> Result<PitchMethod> {
    PitchMethod::from_str(value, true).map_err(anyhow::Error::msg)
}

fn gpu_list(value: &str) -> Result<Vec<usize>> {
    let mut devices = Vec::new();
    for entry in value
        .split(['-', ','])
        .map(str::trim)
        .filter(|entry| !entry.is_empty())
    {
        let device = entry
            .parse()
            .with_context(|| format!("invalid CUDA device number {entry}"))?;
        if !devices.contains(&device) {
            devices.push(device);
        }
    }
    Ok(devices)
}

fn has_training_gpu() -> bool {
    model::weights::cuda_profiles()
        .iter()
        .flatten()
        .any(|gpu| gpu.eligible())
}

fn segment_duration() -> Result<f64> {
    let device = model::weights::device("auto")?;
    let half = match device {
        tch::Device::Cuda(id) => {
            model::weights::cuda_profile(id).is_some_and(|gpu| gpu.fp16 && gpu.memory_gib > 4.0)
        }
        _ => false,
    };
    Ok(if half { 3.7 } else { 3.0 })
}

fn pretrained_updates(state: &StateData, index: usize, args: &Args) -> Result<Vec<Value>> {
    let version = version(args.text(2)?)?;
    let mut sample_rate = rate(args.text(usize::from(index == 20))?)?;
    let f0 = args.boolean(usize::from(index != 20))?;
    if index == 19 && matches!(version, Version::V1) && sample_rate == 32_000 {
        sample_rate = 40_000;
    }
    let paths = pretrained(state, version, sample_rate, f0);
    let mut updates = Vec::new();
    if index == 20 {
        updates.push(update(json!({"visible":f0})));
        updates.push(update(json!({"visible":f0 && has_training_gpu()})));
    }
    updates.extend(paths.map(|path| json!(path)));
    if index == 19 {
        let choices = if matches!(version, Version::V1) {
            vec!["40k", "48k"]
        } else {
            vec!["40k", "48k", "32k"]
        };
        updates.push(update(
            json!({"choices":choices, "value":format!("{}k", sample_rate / 1000)}),
        ));
    }
    Ok(updates)
}

pub(in crate::web::gradio) fn pretrained(
    state: &StateData,
    version: Version,
    sample_rate: u32,
    f0: bool,
) -> [String; 2] {
    let prefix = if f0 { "f0" } else { "" };
    let folder = if matches!(version, Version::V2) {
        "pretrained_v2"
    } else {
        "pretrained"
    };
    ['G', 'D'].map(|kind| {
        let path = format!("assets/{folder}/{prefix}{kind}{}k.pth", sample_rate / 1000);
        if state.paths.root.join(&path).is_file() {
            path
        } else {
            String::new()
        }
    })
}

fn train_request(state: &StateData, args: &Args, all: bool) -> Result<Request> {
    let save = if all { 7 } else { 4 };
    let version_index = save + 9;
    let multi_index = if all { 18 } else { 14 };
    Ok(Request {
        experiment: experiment(state, args.text(0)?)?,
        sample_rate: rate(args.text(1)?)?,
        f0: args.boolean(2)?,
        speaker_id: args.integer(if all { 4 } else { 3 })?,
        save_every: args.integer(save)?,
        epochs: args.integer(save + 1)?,
        batch_size: Some(args.integer(save + 2)?),
        latest_only: args.boolean(save + 3)?,
        pretrained_g: optional_path(args.text(save + 4)?),
        pretrained_d: optional_path(args.text(save + 5)?),
        gpus: gpu_list(args.text(save + 6)?)?,
        cache_data_in_gpu: args.boolean(save + 7)?,
        save_weights: args.boolean(save + 8)?,
        version: version(args.text(version_index)?)?,
        multi: multi(args.text(multi_index)?),
        rebuild_filelist: true,
        random_weights: true, // A blank legacy field means no pretrained weights for that module.
        ..Default::default()
    })
}

pub(super) fn checkpoint_settings(checkpoint: &Path) -> Result<Vec<Value>> {
    static SETTINGS: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r#"['"](sample_rate|if_f0|version)['"]\s*:\s*['"]?([A-Za-z0-9]+)"#).unwrap()
    });
    let path = checkpoint.with_file_name("train.log");
    let file = match File::open(&path) {
        Ok(file) => file,
        Err(error) if error.kind() == ErrorKind::NotFound => return Ok(vec![update(json!({})); 3]),
        Err(error) => return Err(error).with_context(|| format!("opening {}", path.display())),
    };
    let mut result = vec![update(json!({})); 3];
    for line in BufReader::new(file).lines() {
        let line = line.with_context(|| format!("reading {}", path.display()))?;
        let mut settings = [None, None, None];
        for field in SETTINGS.captures_iter(&line) {
            match &field[1] {
                "sample_rate" if rate(&field[2]).is_ok() => settings[0] = Some(json!(&field[2])),
                "if_f0" if matches!(&field[2], "0" | "1") => settings[1] = Some(json!(&field[2])),
                "version" if version(&field[2]).is_ok() => settings[2] = Some(json!(&field[2])),
                _ => {}
            }
        }
        if let [Some(rate), Some(f0), Some(version)] = settings {
            result = vec![rate, f0, version];
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checkpoint_settings_reads_latest_complete_header() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let checkpoint = directory.path().join("G_6.pth");
        std::fs::write(
            directory.path().join("train.log"),
            "Selected training precision FP32\n\
             INFO\t{'sample_rate': '40k', 'if_f0': 1, 'version': 'v1'}\n\
             Older progress\n\
             {\"if_f0\":0,\"sample_rate\":\"32k\",\"version\":\"v2\"}\n\
             Current progress\n\
             Incomplete settings {'sample_rate': '48k'}\n",
        )?;
        assert_eq!(
            checkpoint_settings(&checkpoint)?,
            vec![json!("32k"), json!("0"), json!("v2")]
        );
        Ok(())
    }
}
