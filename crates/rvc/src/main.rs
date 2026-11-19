#[cfg(feature = "inference")]
mod pymss;
#[cfg(feature = "inference")]
mod web;

use std::path::PathBuf;

use anyhow::{Result, ensure};
use clap::{Parser, Subcommand};

use rvc::catalog::{ModelInfo, Paths};
#[cfg(feature = "realtime")]
use rvc::realtime;
#[cfg(feature = "index-training")]
use rvc::train_index;
use rvc::{audio, catalog, dataset};
#[cfg(feature = "inference")]
use rvc::{model, separation};

#[derive(Parser)]
#[command(version, about = "Retrieval-based Voice Conversion")]
struct Cli {
    #[arg(long, global = true, default_value = ".")]
    root: PathBuf,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Serve the local WebUI.
    #[cfg(feature = "inference")]
    Serve {
        #[arg(long, default_value = "127.0.0.1")]
        listen: std::net::IpAddr,
        #[arg(long, default_value_t = 7865)]
        port: u16,
        #[arg(long)]
        open: bool,
        #[arg(long, value_enum)]
        panel: Option<web::Panel>,
        /// Share the WebUI through the Gradio relay with an access password.
        #[arg(long)]
        colab: bool,
        /// Path to the Gradio-compatible frpc executable.
        #[arg(long, requires = "colab")]
        share_client: Option<PathBuf>,
    },
    /// Convert one file or a directory of recordings.
    Infer {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        list_speakers: bool,
        #[cfg(feature = "inference")]
        #[command(flatten)]
        options: InferOptions,
    },
    /// Show the metadata of a compact RVC checkpoint.
    ModelInfo { model: PathBuf },
    /// Edit the description in a compact RVC checkpoint.
    #[cfg(feature = "inference")]
    SetModelInfo {
        model: PathBuf,
        output: PathBuf,
        #[arg(long)]
        description: String,
        #[arg(long)]
        overwrite: bool,
    },
    /// Export the inference generator from a training checkpoint.
    #[cfg(feature = "inference")]
    ExportModel {
        checkpoint: PathBuf,
        output: PathBuf,
        #[arg(long, value_enum, default_value = "v2")]
        version: model::train_data::Version,
        #[arg(long, default_value_t = 40_000)]
        sample_rate: u32,
        #[arg(long, action = clap::ArgAction::Set, default_value_t = true)]
        f0: bool,
        #[arg(long, default_value = "")]
        description: String,
        #[arg(long)]
        overwrite: bool,
    },
    /// Blend the weights of two compatible compact checkpoints.
    #[cfg(feature = "inference")]
    MergeModels {
        first: PathBuf,
        second: PathBuf,
        output: PathBuf,
        #[arg(long, default_value_t = 0.5)]
        alpha: f64,
        #[arg(long, default_value = "")]
        description: String,
        #[arg(long)]
        overwrite: bool,
    },
    /// Train an RVC generator and discriminator, resuming the latest saved pair.
    #[cfg(feature = "inference")]
    Train(model::train::Request),
    /// Create a multi-speaker manifest from NAME_ID_REPEAT folders.
    ScanSpeakers { input: PathBuf, experiment: PathBuf },
    /// Prepare 16 kHz analysis clips and target-rate training audio.
    Preprocess {
        #[arg(long)]
        input: Option<PathBuf>,
        #[arg(long)]
        experiment: PathBuf,
        #[arg(long, requires = "input", conflicts_with = "manifest")]
        multi_folder: bool,
        #[arg(long, conflicts_with = "input")]
        manifest: bool,
        #[arg(long, default_value_t = 40_000)]
        sample_rate: u32,
        #[arg(long, default_value_t = 1)]
        workers: usize,
        #[arg(long, default_value_t = 3.0)]
        segment_seconds: f64,
    },
    /// Extract HuBERT features and optional F0 tracks from preprocessed clips.
    #[cfg(feature = "inference")]
    Extract {
        #[arg(long)]
        experiment: PathBuf,
        #[arg(long, default_value = "v2", value_parser = ["v1", "v2"])]
        version: String,
        #[arg(long, value_enum)]
        f0_method: Option<model::PitchMethod>,
        #[arg(long, default_value = "auto")]
        device: String,
        #[arg(long, value_delimiter = ',')]
        gpus: Vec<usize>,
        #[arg(long)]
        overwrite: bool,
    },
    /// Separate vocals or remove reverb with the PyMSS presets.
    #[cfg(feature = "inference")]
    Separate {
        #[arg(long)]
        list_presets: bool,
        #[arg(long, required_unless_present = "list_presets")]
        input: Option<PathBuf>,
        #[arg(long, required_unless_present = "list_presets")]
        output: Option<PathBuf>,
        #[arg(long)]
        secondary_output: Option<PathBuf>,
        #[arg(long, default_value = "vocals-bs-roformer-368")]
        preset: String,
        #[arg(long, default_value = "auto")]
        device: String,
        #[arg(long, value_enum, default_value = "wav")]
        format: audio::Format,
        #[arg(long)]
        overwrite: bool,
    },
    /// PyMSS model catalog and configurable separation.
    #[cfg(feature = "inference")]
    Pymss(pymss::Cli),
    /// Build an IVF-Flat index from an experiment's HuBERT features.
    #[cfg(feature = "index-training")]
    TrainIndex {
        #[arg(long)]
        experiment: PathBuf,
        #[arg(long, value_enum, default_value = "v2")]
        version: train_index::ModelVersion,
        #[arg(long, value_enum, default_value = "auto")]
        mode: train_index::IndexMode,
        #[arg(long, default_value_t = 1)]
        workers: usize,
    },
    /// List audio devices available to the realtime converter.
    #[cfg(feature = "realtime")]
    Devices {
        #[arg(long)]
        host: Option<String>,
    },
    /// Run the realtime converter through host audio devices.
    #[cfg(feature = "realtime")]
    Realtime(realtime::Args),
}

#[cfg(feature = "inference")]
#[derive(clap::Args)]
struct InferOptions {
    #[arg(long)]
    input: Option<PathBuf>,
    #[arg(long)]
    output: Option<PathBuf>,
    #[arg(long)]
    speaker_id: Option<i64>,
    #[arg(long, default_value_t = 0, allow_negative_numbers = true)]
    pitch: i32,
    #[arg(long, value_enum, default_value = "rmvpe")]
    f0_method: model::PitchMethod,
    #[arg(long)]
    index: Option<PathBuf>,
    #[arg(long, default_value_t = 0.75)]
    index_rate: f32,
    #[arg(long, default_value_t = 0)]
    resample_sr: u32,
    #[arg(long, default_value_t = 1.0)]
    rms_mix_rate: f32,
    #[arg(long, default_value_t = 0.33)]
    protect: f32,
    #[arg(long, value_enum)]
    format: Option<audio::Format>,
    #[arg(long)]
    recursive: bool,
    #[arg(long)]
    overwrite: bool,
    #[arg(long, default_value = "auto")]
    device: String,
    #[arg(long, value_enum, default_value = "auto")]
    precision: model::Precision,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let paths = Paths::new(&cli.root)?;
    match cli.command {
        #[cfg(feature = "inference")]
        Command::Serve {
            listen,
            port,
            open,
            panel,
            colab,
            share_client,
        } => web::serve(
            paths,
            listen,
            port,
            open,
            panel,
            colab.then(|| share_client.unwrap_or_else(web::share_client)),
        )?,
        #[cfg(feature = "inference")]
        Command::Pymss(cli) => pymss::run(&paths, cli)?,
        Command::Infer {
            model,
            list_speakers,
            #[cfg(feature = "inference")]
            options,
        } => {
            let model = paths.model(&model)?;
            if list_speakers {
                let info = ModelInfo::read(&model)?;
                if info.speakers.is_empty() {
                    println!("0-{}", info.speaker_count - 1);
                } else {
                    for (id, name) in info.speakers {
                        println!("{id}\t{name}");
                    }
                }
            } else {
                #[cfg(feature = "inference")]
                infer(&paths, &model, options)?;
                #[cfg(not(feature = "inference"))]
                anyhow::bail!("voice conversion requires the inference build feature");
            }
        }
        Command::ModelInfo { model } => {
            let info = ModelInfo::read(&paths.model(&model)?)?;
            println!("{}", serde_json::to_string_pretty(&info)?);
        }
        #[cfg(feature = "inference")]
        Command::SetModelInfo {
            model,
            output,
            description,
            overwrite,
        } => {
            model::edit::describe(
                &paths.model(&model)?,
                &paths.root.join(output),
                &description,
                overwrite,
            )?;
        }
        #[cfg(feature = "inference")]
        Command::ExportModel {
            checkpoint,
            output,
            version,
            sample_rate,
            f0,
            description,
            overwrite,
        } => {
            model::edit::extract(
                &paths.root.join(checkpoint),
                &paths.root.join(output),
                model::edit::Export {
                    version,
                    sample_rate,
                    uses_f0: f0,
                    description: &description,
                    overwrite,
                },
            )?;
        }
        #[cfg(feature = "inference")]
        Command::MergeModels {
            first,
            second,
            output,
            alpha,
            description,
            overwrite,
        } => {
            model::edit::merge(
                &paths.model(&first)?,
                &paths.model(&second)?,
                &paths.root.join(output),
                alpha,
                &description,
                overwrite,
            )?;
        }
        #[cfg(feature = "inference")]
        Command::Train(request) => {
            use std::sync::{
                Arc,
                atomic::{AtomicBool, Ordering},
            };
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(1)
                .enable_all()
                .build()?;
            let cancelled = Arc::new(AtomicBool::new(false));
            let signal = Arc::clone(&cancelled);
            runtime.spawn(async move {
                if tokio::signal::ctrl_c().await.is_ok() {
                    signal.store(true, Ordering::Relaxed);
                }
            });
            let report = model::train::run(&paths, request, |line| {
                eprintln!("{line}");
                if cancelled.load(Ordering::Relaxed) {
                    std::ops::ControlFlow::Break(())
                } else {
                    std::ops::ControlFlow::Continue(())
                }
            })?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Command::ScanSpeakers { input, experiment } => {
            let manifest = dataset::scan_speakers(&paths.root.join(input))?;
            let output = dataset::save_manifest(&paths.root.join(experiment), &manifest)?;
            println!("{}", output.display());
        }
        Command::Preprocess {
            input,
            experiment,
            multi_folder,
            manifest,
            sample_rate,
            workers,
            segment_seconds,
        } => {
            use anyhow::Context;
            let input = input.map(|path| paths.root.join(path));
            let source = if manifest {
                dataset::DatasetSource::MultiManifest
            } else if multi_folder {
                dataset::DatasetSource::MultiFolder(
                    input.as_deref().context("--input is required")?,
                )
            } else {
                dataset::DatasetSource::Single(
                    input
                        .as_deref()
                        .context("--input or --manifest is required")?,
                )
            };
            let result = dataset::preprocess(
                &dataset::PreprocessRequest {
                    source,
                    experiment_dir: &paths.root.join(experiment),
                    sample_rate,
                    workers,
                    segment_seconds,
                },
                |line| eprintln!("{line}"),
            )?;
            println!("{}", serde_json::to_string_pretty(&result)?);
        }
        #[cfg(feature = "inference")]
        Command::Extract {
            experiment,
            version,
            f0_method,
            device,
            gpus,
            overwrite,
        } => {
            let devices = model::extract::selected_devices(&device, &gpus)?;
            let request = model::extract::Request {
                experiment: &paths.root.join(experiment),
                version: &version,
                f0: f0_method,
                device: devices[0],
                overwrite,
            };
            let report = model::extract::run_on_devices(&paths, request, &devices, |line| {
                eprintln!("{line}");
                std::ops::ControlFlow::Continue(())
            })?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        #[cfg(feature = "inference")]
        Command::Separate {
            list_presets,
            input,
            output,
            secondary_output,
            preset,
            device,
            format,
            overwrite,
        } => {
            if list_presets {
                println!("{}", serde_json::to_string_pretty(&separation::PRESETS)?);
            } else {
                let options = separation::Options {
                    preset: &preset,
                    device: separation::device(&device)?,
                    format,
                    secondary_output: secondary_output.as_deref(),
                    overwrite,
                };
                let input = input.ok_or_else(|| anyhow::anyhow!("--input is required"))?;
                let output = output.ok_or_else(|| anyhow::anyhow!("--output is required"))?;
                let stems =
                    separation::separate(&paths.root, &input, &output, options, |progress| {
                        eprintln!(
                            "[{}/{}] separation {}/{}",
                            progress.file_index,
                            progress.file_count,
                            progress.chunks_done,
                            progress.chunks_total
                        );
                        std::ops::ControlFlow::Continue(())
                    })?;
                println!("{}", serde_json::to_string_pretty(&stems)?);
            }
        }
        #[cfg(feature = "index-training")]
        Command::TrainIndex {
            experiment,
            version,
            mode,
            workers,
        } => {
            let experiment_dir = paths.root.join(experiment);
            let request = train_index::IndexTraining {
                experiment_dir: &experiment_dir,
                outside_index_root: &paths.indices,
                version,
                mode,
                workers,
            };
            train_index::train_indices(&request, |line| eprintln!("{line}"))?;
        }
        #[cfg(feature = "realtime")]
        Command::Devices { host } => {
            println!(
                "{}",
                serde_json::to_string_pretty(&realtime::device::devices(host.as_deref())?)?
            );
        }
        #[cfg(feature = "realtime")]
        Command::Realtime(args) => realtime::run(&paths, args)?,
    }
    Ok(())
}

#[cfg(feature = "inference")]
fn infer(paths: &Paths, model: &std::path::Path, args: InferOptions) -> Result<()> {
    use anyhow::Context;

    let input = paths.root.join(
        args.input
            .context("--input is required unless --list-speakers is used")?,
    );
    let output = paths.root.join(
        args.output
            .context("--output is required unless --list-speakers is used")?,
    );
    let settings = model::ConversionSettings {
        pitch: args.pitch as f32,
        f0_method: args.f0_method,
        index_rate: args.index_rate,
        resample_sr: args.resample_sr,
        rms_mix_rate: args.rms_mix_rate,
        protect: args.protect,
        precision: args.precision,
        ..Default::default()
    };
    settings.validate()?;
    let info = ModelInfo::read(model)?;
    let speaker = info.speaker(args.speaker_id)?;
    let index = paths.require_index(model, speaker, args.index.as_deref(), args.index_rate)?;
    let jobs = conversion_jobs(input, output, args.format, args.recursive)?;
    let mut targets = std::collections::HashSet::new();
    for (_, path, _) in &jobs {
        ensure!(
            targets.insert(path),
            "multiple inputs would write to {}",
            path.display()
        );
        ensure!(
            args.overwrite || !path.exists(),
            "Output already exists; use --overwrite: {}",
            path.display()
        );
    }
    let engine = model::Rvc::load(
        paths,
        model,
        Some(speaker),
        index.as_deref(),
        settings,
        model::weights::device(&args.device)?,
    )?;
    let mut failed = 0;
    for (source, target, format) in jobs {
        let result = (|| -> Result<()> {
            let samples = audio::read_mono(&source, audio::ANALYSIS_RATE)?;
            let converted = engine.convert(&samples, settings)?;
            audio::write_audio(
                &target,
                &converted.samples,
                converted.rate,
                format,
                args.overwrite,
            )
        })();
        if let Err(error) = result {
            eprintln!("{}: {error:#}", source.display());
            failed += 1;
        } else {
            println!("{}", target.display());
        }
    }
    ensure!(failed == 0, "{failed} conversion(s) failed");
    Ok(())
}

#[cfg(feature = "inference")]
fn conversion_jobs(
    input: PathBuf,
    output: PathBuf,
    format: Option<audio::Format>,
    recursive: bool,
) -> Result<Vec<(PathBuf, PathBuf, audio::Format)>> {
    use anyhow::Context;

    ensure!(input.exists(), "Input not found: {}", input.display());
    if input.is_file() {
        let extension = if output.is_dir() {
            None
        } else {
            output
                .extension()
                .map(audio::Format::from_extension)
                .transpose()?
        };
        let format = format.or(extension).unwrap_or_default();
        let target = if extension.is_some() && !output.is_dir() {
            output.with_extension(format.extension())
        } else {
            output
                .join(input.file_name().context("missing input file name")?)
                .with_extension(format.extension())
        };
        return Ok(vec![(input, target, format)]);
    }
    let format = format.unwrap_or_default();
    let files = audio::files(&input, recursive)?;
    ensure!(
        !files.is_empty(),
        "No supported audio files found in: {}",
        input.display()
    );
    files
        .into_iter()
        .map(|file| {
            let relative = file.strip_prefix(&input)?;
            let target = output.join(relative).with_extension(format.extension());
            Ok((file, target, format))
        })
        .collect()
}
