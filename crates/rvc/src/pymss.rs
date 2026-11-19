mod catalog;
mod ensemble;
mod pcm;
mod server;
mod workflow;

use std::collections::HashSet;
use std::ops::ControlFlow;
use std::path::PathBuf;

use anyhow::{Context, Result, ensure};
use clap::{Args, Subcommand};
use serde_json::json;
use tch::Device;

use crate::audio::Format;
use crate::catalog::Paths;
use crate::model::weights;
use crate::separation::{
    self, AudioEncoding, ConfiguredOptions, FlacDepth, InferenceParams, WavDepth,
};
use catalog::{Source, by_name, download, entries, model_directory};

fn compute_devices(
    request: &str,
    ids: &[usize],
    params: &mut InferenceParams,
) -> Result<(Device, Vec<Device>)> {
    let Some((&first, remaining)) = ids.split_first() else {
        return Ok((params.resolve_device(request)?, Vec::new()));
    };
    let primary = Device::Cuda(first);
    ensure!(
        matches!(request, "auto" | "cuda")
            || request.starts_with("cuda:") && weights::device(request)? == primary,
        "--device-id requires CUDA and its first ID must agree with --device"
    );
    let available = weights::cuda_device_count();
    ensure!(
        ids.iter().all(|&index| index < available),
        "requested CUDA device ID is unavailable"
    );
    ensure!(
        ids.iter().collect::<HashSet<_>>().len() == ids.len(),
        "CUDA device IDs must be unique"
    );
    Ok((
        primary,
        remaining.iter().copied().map(Device::Cuda).collect(),
    ))
}

#[derive(Args)]
pub struct Cli {
    #[command(subcommand)]
    action: Action,
}

#[derive(Subcommand)]
enum Action {
    /// Serve the PyMSS HTTP API and its optional browser interface.
    Serve(server::Cli),
    /// Validate or run a multi-stage separation workflow.
    #[command(subcommand)]
    Workflow(workflow::Cli),
    /// List the bundled PyMSS model catalog.
    List {
        #[arg(long)]
        category: Option<String>,
        #[arg(long)]
        all: bool,
        #[arg(long)]
        json: bool,
    },
    /// Inspect a catalog model.
    Info {
        model: String,
        #[arg(long)]
        model_dir: Option<PathBuf>,
    },
    /// Download model weights, config and companion files.
    Download {
        model: String,
        #[command(flatten)]
        remote: RemoteOptions,
        #[arg(long)]
        force: bool,
        #[arg(long)]
        supported_only: bool,
    },
    /// Separate using a catalog model or explicit local files.
    Infer {
        model: Option<String>,
        #[command(flatten)]
        remote: RemoteOptions,
        #[arg(short, long)]
        input: PathBuf,
        #[arg(short, long, default_value = "results")]
        output: PathBuf,
        #[arg(long, requires = "checkpoint")]
        model_type: Option<String>,
        #[arg(long, requires = "model_type")]
        checkpoint: Option<PathBuf>,
        #[arg(long, requires_all = ["model_type", "checkpoint"])]
        config: Option<PathBuf>,
        #[arg(long, value_delimiter = ',')]
        stems: Vec<String>,
        #[arg(long, default_value = "auto")]
        device: String,
        #[arg(long = "device-id")]
        device_ids: Vec<usize>,
        #[arg(long, value_enum, default_value = "wav")]
        format: Format,
        #[command(flatten)]
        encoding: EncodingArgs,
        #[arg(long)]
        tta: bool,
        #[arg(long = "param", value_name = "KEY=VALUE")]
        params: Vec<String>,
        #[arg(long)]
        chunk_size: Option<usize>,
        #[arg(long)]
        overlap_size: Option<usize>,
        #[arg(long)]
        save_as_folder: bool,
        #[arg(long)]
        overwrite: bool,
        #[arg(long)]
        download: bool,
        /// Log resolved model paths and inference settings.
        #[arg(long)]
        debug: bool,
    },
    /// Combine matching stems from multiple audio files.
    Ensemble {
        #[arg(required = true, num_args = 2..)]
        files: Vec<PathBuf>,
        #[arg(long, value_enum, default_value = "avg_wave")]
        algorithm: ensemble::Algorithm,
        #[arg(long, num_args = 1.., value_delimiter = ',', allow_negative_numbers = true)]
        weights: Vec<f32>,
        #[arg(short, long)]
        output: PathBuf,
        #[arg(long, value_enum)]
        format: Option<Format>,
        #[command(flatten)]
        encoding: EncodingArgs,
        #[arg(long)]
        overwrite: bool,
    },
}

#[derive(Args)]
pub(crate) struct EncodingArgs {
    #[arg(long, value_enum, default_value = "FLOAT", ignore_case = true)]
    wav_bit_depth: WavDepth,
    #[arg(long, value_enum, default_value = "PCM_16", ignore_case = true)]
    flac_bit_depth: FlacDepth,
    #[arg(long, default_value = "320k", value_parser = parse_bit_rate)]
    mp3_bit_rate: u32,
    #[arg(long, default_value = "512k", value_parser = parse_bit_rate)]
    m4a_bit_rate: u32,
    #[arg(long, default_value = "aac")]
    m4a_codec: String,
    #[arg(long, default_value_t = 2)]
    m4a_aac_at_quality: i32,
}

impl From<EncodingArgs> for AudioEncoding {
    fn from(args: EncodingArgs) -> Self {
        Self {
            wav: args.wav_bit_depth,
            flac: args.flac_bit_depth,
            mp3_bit_rate: args.mp3_bit_rate,
            m4a_bit_rate: args.m4a_bit_rate,
            m4a_codec: args.m4a_codec,
            m4a_aac_at_quality: args.m4a_aac_at_quality,
        }
    }
}

fn parse_bit_rate(value: &str) -> Result<u32, String> {
    let value = value.trim().to_ascii_lowercase();
    let rate = if let Some(thousands) = value.strip_suffix('k') {
        thousands.parse::<f64>().ok().map(|value| value * 1000.0)
    } else {
        value.parse::<u32>().ok().map(f64::from)
    };
    rate.filter(|&value| (1.0..=f64::from(u32::MAX)).contains(&value))
        .map(|value| value as u32)
        .ok_or_else(|| "expected a positive bit rate, such as 320k".to_owned())
}

#[derive(Args)]
struct RemoteOptions {
    #[arg(long)]
    model_dir: Option<PathBuf>,
    #[arg(long, value_enum, default_value = "modelscope")]
    source: Source,
    #[arg(long)]
    endpoint: Option<String>,
}

pub fn run(paths: &Paths, cli: Cli) -> Result<()> {
    match cli.action {
        Action::Serve(cli) => server::serve(paths, cli)?,
        Action::Workflow(cli) => workflow::run(paths, cli)?,
        Action::List {
            category,
            all,
            json,
        } => {
            let entries = entries(category.as_deref(), (!all).then_some(true))?;
            if json {
                println!("{}", serde_json::to_string_pretty(&entries)?);
            } else {
                for entry in entries {
                    let category = entry.categories();
                    let status = if entry.supported {
                        "ok"
                    } else {
                        &entry.unsupported_reason
                    };
                    println!(
                        "{}\t{}\t{}\t{}\t{}",
                        entry.name,
                        entry.model_type.as_deref().unwrap_or(&entry.architecture),
                        category,
                        entry.target_stem,
                        status
                    );
                }
            }
        }
        Action::Info { model, model_dir } => {
            let entry = by_name(&model)?;
            let directory = model_directory(&paths.root, model_dir.as_deref());
            let files = entry.files(&directory)?;
            let find = |role| {
                files
                    .iter()
                    .find(|file| file.role == role)
                    .map(|file| &file.path)
            };
            let categories = [&entry.primary_category_cn, &entry.secondary_category_cn]
                .into_iter()
                .filter(|part| !part.is_empty())
                .map(String::as_str)
                .collect::<Vec<_>>()
                .join(" / ");
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "name": entry.name, "model_type": entry.model_type, "architecture": entry.architecture,
                    "supported": entry.supported, "unsupported_reason": entry.unsupported_reason,
                    "category": entry.categories(), "category_cn": categories, "target_stem": entry.target_stem,
                    "model_path": find("model"), "config_path": find("config"), "size_bytes": entry.size_bytes,
                }))?
            );
        }
        Action::Download {
            model,
            remote,
            force,
            supported_only,
        } => {
            let directory = model_directory(&paths.root, remote.model_dir.as_deref());
            let multiple = model == "all";
            let entries = if multiple {
                entries(None, supported_only.then_some(true))?
            } else {
                vec![by_name(&model)?]
            };
            let mut failed = 0;
            let total = entries.len();
            for entry in entries {
                match download(
                    &entry,
                    &directory,
                    remote.source,
                    remote.endpoint.as_deref(),
                    force,
                    1800,
                ) {
                    Ok(report) if !multiple => {
                        for local in report.skipped {
                            println!("exists {}", local.display());
                        }
                        for local in report.downloaded {
                            println!("downloaded {}", local.display());
                        }
                    }
                    Ok(_) => {}
                    Err(error) if multiple => {
                        eprintln!("ERROR {}: {error:#}", entry.name);
                        failed += 1;
                    }
                    Err(error) => return Err(error),
                }
            }
            if multiple {
                println!(
                    "Downloaded/skipped {} model(s), failed {failed}.",
                    total - failed
                );
            }
            ensure!(
                failed == 0,
                "download failed for one or more catalog entries"
            );
        }
        Action::Infer {
            model,
            remote,
            input,
            output,
            model_type,
            checkpoint,
            config,
            stems,
            device,
            device_ids,
            format,
            encoding,
            tta,
            params,
            chunk_size,
            overlap_size,
            save_as_folder,
            overwrite,
            download: download_requested,
            debug,
        } => {
            let mut params = InferenceParams::parse(&params)?.with_tta(tta);
            let (device, additional_devices) = compute_devices(&device, &device_ids, &mut params)?;
            let (model_type, checkpoint, config) = if let Some(model_type) = model_type {
                ensure!(
                    model.is_none(),
                    "use a catalog name or explicit model files"
                );
                let checkpoint = checkpoint.context("--checkpoint is required")?;
                let config = if separation::optional_config(&model_type) {
                    config.unwrap_or_else(|| checkpoint.clone())
                } else {
                    config.context("--config is required")?
                };
                (model_type, checkpoint, config)
            } else {
                let entry = by_name(&model.context("select a PyMSS model")?)?;
                ensure!(entry.supported, "{}", entry.unsupported_reason);
                let model_type = entry
                    .model_type
                    .clone()
                    .context("catalog entry has no architecture")?;
                params.validate(&model_type)?;
                let directory = model_directory(&paths.root, remote.model_dir.as_deref());
                let (checkpoint, config) = entry.model_paths(&directory)?;
                if download_requested || !checkpoint.is_file() || !config.is_file() {
                    download(
                        &entry,
                        &directory,
                        remote.source,
                        remote.endpoint.as_deref(),
                        false,
                        1800,
                    )?;
                }
                (model_type, checkpoint, config)
            };
            let options = ConfiguredOptions {
                model_type: &model_type,
                checkpoint: &paths.root.join(checkpoint),
                config: &paths.root.join(config),
                stems: &stems,
                device,
                additional_devices: &additional_devices,
                format,
                overwrite,
                chunk_size,
                overlap_size,
                save_as_folder,
            };
            if debug {
                eprintln!(
                    "PyMSS {model_type} on {} {additional_devices:?}, weights {}, config {}, overrides {params:?}",
                    params.device_label(device),
                    options.checkpoint.display(),
                    options.config.display(),
                );
            }
            let outputs = separation::separate_configured_with_params(
                &paths.root,
                &paths.root.join(input),
                &paths.root.join(output),
                options,
                &params,
                &encoding.into(),
                |position| {
                    eprintln!(
                        "[{}/{}] {}/{}",
                        position.file_index,
                        position.file_count,
                        position.chunks_done,
                        position.chunks_total
                    );
                    ControlFlow::Continue(())
                },
            )?;
            println!("{}", serde_json::to_string_pretty(&outputs)?);
        }
        Action::Ensemble {
            files,
            algorithm,
            weights,
            output,
            format,
            encoding,
            overwrite,
        } => ensemble::run(
            &paths.root,
            &files
                .into_iter()
                .map(|file| paths.root.join(file))
                .collect::<Vec<_>>(),
            &paths.root.join(output),
            algorithm,
            &weights,
            format,
            &encoding.into(),
            overwrite,
        )?,
    }
    Ok(())
}
