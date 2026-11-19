use std::collections::{BTreeMap, HashMap, HashSet};
use std::ops::ControlFlow;
use std::path::{Component, Path, PathBuf};
use std::sync::LazyLock;

use anyhow::{Context, Result, bail, ensure};
use clap::{Subcommand, ValueEnum};
use regex::Regex;
use serde::Deserialize;
use serde_json::Value;

use super::catalog::{self, Source};
use super::pcm::Pcm;
use crate::audio::{self, Format};
use crate::catalog::Paths;
use crate::separation::{AudioEncoding, ConfiguredModel, InferenceParams, write_encoded_track};

const TEMPLATE: &str = "\
version: 1
defaults:
  output_format: wav
  device: auto
steps:
  - id: vocals
    model: model_bs_roformer_ep_368_sdr_12.9628.ckpt
    input: input
    stems: [vocals, instrumental]
    save:
      vocals: vocals
      instrumental: accompaniment
";

#[derive(Subcommand)]
pub enum Cli {
    Init {
        #[arg(short, long, default_value = "workflow.yaml")]
        output: PathBuf,
        #[arg(long)]
        force: bool,
    },
    Validate {
        #[arg(short, long)]
        config: PathBuf,
        #[arg(long)]
        model_dir: Option<PathBuf>,
        /// Check that catalog models exist and are supported.
        #[arg(long)]
        check_models: bool,
        /// Require local files for all referenced models.
        #[arg(long, alias = "require-model-files")]
        require_files: bool,
    },
    Run {
        #[arg(short, long)]
        config: PathBuf,
        #[arg(short, long)]
        input: PathBuf,
        #[arg(short, long, default_value = "results")]
        output: PathBuf,
        #[arg(long, value_enum, default_value = "folders")]
        output_layout: Layout,
        #[arg(long)]
        model_dir: Option<PathBuf>,
        #[arg(long, value_enum, default_value = "modelscope")]
        source: Source,
        #[arg(long)]
        endpoint: Option<String>,
        #[arg(long)]
        device: Option<String>,
        #[arg(long, value_enum)]
        format: Option<Format>,
        #[command(flatten)]
        encoding: super::EncodingArgs,
        #[arg(long)]
        overwrite: bool,
        #[arg(long)]
        download: bool,
        #[arg(long)]
        continue_on_error: bool,
        /// Log resolved model paths and inference settings for each step.
        #[arg(long)]
        debug: bool,
    },
}

#[derive(Clone, Copy, ValueEnum)]
pub enum Layout {
    Folders,
    Flat,
}

#[derive(Deserialize)]
struct Workflow {
    version: u32,
    #[serde(default)]
    defaults: Defaults,
    steps: Vec<Step>,
}

#[derive(Default, Deserialize)]
struct Defaults {
    model_dir: Option<PathBuf>,
    device: Option<String>,
    output_format: Option<Format>,
    use_tta: Option<bool>,
    #[serde(default)]
    inference_params: InferenceParams,
}

#[derive(Deserialize)]
struct Step {
    id: String,
    model: Option<String>,
    model_type: Option<String>,
    model_path: Option<PathBuf>,
    config_path: Option<PathBuf>,
    #[serde(default = "input_reference")]
    input: String,
    #[serde(default, deserialize_with = "deserialize_stems")]
    stems: Option<Vec<String>>,
    #[serde(default)]
    save: BTreeMap<String, Value>,
    device: Option<String>,
    model_dir: Option<PathBuf>,
    output_format: Option<Format>,
    use_tta: Option<bool>,
    #[serde(default)]
    inference_params: InferenceParams,
}

fn input_reference() -> String {
    "input".to_owned()
}

fn deserialize_stems<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<Vec<String>>, D::Error> {
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Stems {
        One(String),
        Many(Vec<String>),
    }
    Ok(Option::<Stems>::deserialize(deserializer)?.map(|stems| {
        let mut stems = match stems {
            Stems::One(stem) => vec![stem],
            Stems::Many(stems) => stems,
        };
        for stem in &mut stems {
            *stem = stem.trim().to_owned();
        }
        stems.retain(|stem| !stem.is_empty());
        stems
    }))
}

pub fn run(paths: &Paths, cli: Cli) -> Result<()> {
    match cli {
        Cli::Init { output, force } => {
            let output = paths.root.join(output);
            audio::write_file(&output, TEMPLATE.as_bytes(), force)?;
            println!("{}", output.display());
        }
        Cli::Validate {
            config,
            model_dir,
            check_models,
            require_files,
        } => {
            let workflow = load(&paths.root.join(config))?;
            if check_models || require_files {
                validate_models(&workflow, paths, model_dir.as_deref(), require_files)?;
            }
            println!("Valid workflow ({} steps)", workflow.steps.len());
        }
        Cli::Run {
            config,
            input,
            output,
            output_layout,
            model_dir,
            source,
            endpoint,
            device,
            format,
            encoding,
            overwrite,
            download,
            continue_on_error,
            debug,
        } => {
            let workflow = load(&paths.root.join(config))?;
            let encoding: AudioEncoding = encoding.into();
            let mut tracks = tracks(&paths.root.join(input))?;
            let output_root = paths.root.join(&output);
            let mut saved = HashSet::new();
            for step in &workflow.steps {
                let (model_type, checkpoint, config) =
                    resolve(step, &workflow.defaults, paths, model_dir.as_deref())?;
                if download
                    && (!checkpoint.is_file() || !config.is_file())
                    && let Some(name) = &step.model
                {
                    let directory =
                        directory(step, &workflow.defaults, paths, model_dir.as_deref());
                    catalog::download(
                        &catalog::by_name(name)?,
                        &directory,
                        source,
                        endpoint.as_deref(),
                        false,
                        1800,
                    )?;
                }
                let compute = step
                    .device
                    .as_deref()
                    .or(device.as_deref())
                    .or(workflow.defaults.device.as_deref())
                    .unwrap_or("auto");
                let mut params = workflow
                    .defaults
                    .inference_params
                    .overlay(&step.inference_params)
                    .with_tta(step.use_tta.or(workflow.defaults.use_tta).unwrap_or(false));
                let (device, _) = super::compute_devices(compute, &[], &mut params)?;
                if debug {
                    eprintln!(
                        "PyMSS step {} ({model_type}) on {}, weights {}, config {}, overrides {params:?}",
                        step.id,
                        params.device_label(device),
                        checkpoint.display(),
                        config.display(),
                    );
                }
                let model = ConfiguredModel::load_with_params(
                    &model_type,
                    &checkpoint,
                    &config,
                    device,
                    &params,
                )?;
                let requested = step
                    .stems
                    .clone()
                    .unwrap_or_else(|| step.save.keys().cloned().collect());
                let format = step
                    .output_format
                    .or(format)
                    .or(workflow.defaults.output_format)
                    .unwrap_or_default();
                for track in tracks.iter_mut().filter(|track| track.active) {
                    let mut operation = || -> Result<()> {
                        let samples = track
                            .audio
                            .get(&ref_key(&step.input)?)
                            .context("workflow input stem is missing")?
                            .convert_layout(model.rate(), model.channels())?;
                        let results = model.separate_planar(&samples.planar, &requested, |_| {
                            ControlFlow::Continue(())
                        })?;
                        for (stem, planar) in results {
                            let result = Pcm {
                                planar,
                                channels: model.output_channels(),
                                rate: model.rate(),
                            };
                            if let Some((_, target)) = step
                                .save
                                .iter()
                                .find(|(name, _)| name.eq_ignore_ascii_case(&stem))
                            {
                                for folder in directories(target, &step.id)? {
                                    let target = output_path(
                                        &output_root,
                                        output_layout,
                                        &track.name,
                                        folder,
                                        &stem,
                                        format,
                                    );
                                    ensure!(
                                        !saved.contains(&target),
                                        "workflow outputs collide at {}",
                                        target.display()
                                    );
                                    write_encoded_track(
                                        &paths.root,
                                        &target,
                                        &result.planar,
                                        result.rate,
                                        result.channels,
                                        format,
                                        &encoding,
                                        overwrite,
                                    )?;
                                    saved.insert(target);
                                }
                            }
                            track
                                .audio
                                .insert(format!("{}.{}", step.id, stem.to_lowercase()), result);
                        }
                        Ok(())
                    };
                    if let Err(error) = operation() {
                        if !continue_on_error {
                            return Err(error)
                                .with_context(|| format!("workflow step {}", step.id));
                        }
                        eprintln!("{} · {}: {error:#}", track.name, step.id);
                        track.active = false;
                    }
                }
            }
            for track in tracks.into_iter().filter(|track| track.active) {
                println!("{}", track.name);
            }
        }
    }
    Ok(())
}

fn load(path: &Path) -> Result<Workflow> {
    let input = std::fs::read_to_string(path)
        .with_context(|| format!("opening workflow {}", path.display()))?;
    let workflow: Workflow = serde_yaml::from_str(&input)?;
    ensure!(
        workflow.version == 1 && !workflow.steps.is_empty(),
        "workflows require version 1 and at least one step"
    );
    let mut seen: HashMap<&str, &Step> = HashMap::new();
    for step in &workflow.steps {
        let mut letters = step.id.bytes();
        ensure!(
            letters
                .next()
                .is_some_and(|b| b.is_ascii_alphabetic() || b == b'_')
                && letters.all(|b| b.is_ascii_alphanumeric() || b"_-".contains(&b)),
            "invalid workflow step id {}",
            step.id
        );
        match (&step.model, &step.model_path) {
            (Some(name), None) if !name.trim().is_empty() => {}
            (None, Some(path)) if !path.as_os_str().is_empty() => ensure!(
                step.model_type
                    .as_ref()
                    .is_some_and(|kind| !kind.trim().is_empty()),
                "step {} requires model_type when model_path is used",
                step.id
            ),
            _ => bail!(
                "step {} requires exactly one of model or model_path",
                step.id
            ),
        }
        ensure!(
            step.stems.as_ref().is_none_or(|items| !items.is_empty()),
            "step {} requests no stems",
            step.id
        );
        if step.input != "input" {
            let (earlier, stem) = step
                .input
                .split_once('.')
                .context("workflow inputs use step.stem references")?;
            let referenced = seen.get(earlier.trim()).with_context(|| {
                format!("workflow step {} references an unavailable step", step.id)
            })?;
            let selected = referenced
                .stems
                .clone()
                .unwrap_or_else(|| referenced.save.keys().cloned().collect());
            ensure!(
                !stem.trim().is_empty()
                    && (selected.is_empty()
                        || selected
                            .iter()
                            .any(|name| name.eq_ignore_ascii_case(stem.trim()))),
                "workflow reference to a stem that is not requested: {}",
                step.input
            );
        }
        ensure!(
            seen.insert(&step.id, step).is_none(),
            "duplicate workflow step {}",
            step.id
        );
    }
    Ok(workflow)
}

fn ref_key(value: &str) -> Result<String> {
    if value == "input" {
        return Ok(value.into());
    }
    let (step, stem) = value
        .split_once('.')
        .context("workflow inputs use step.stem references")?;
    Ok(format!("{}.{}", step.trim(), stem.trim().to_lowercase()))
}

fn directory(step: &Step, defaults: &Defaults, paths: &Paths, model_dir: Option<&Path>) -> PathBuf {
    catalog::model_directory(
        &paths.root,
        step.model_dir
            .as_deref()
            .or(model_dir)
            .or(defaults.model_dir.as_deref()),
    )
}

fn output_path(
    root: &Path,
    layout: Layout,
    track: &str,
    folder: &str,
    stem: &str,
    format: Format,
) -> PathBuf {
    static STEM_ESCAPE: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"[\\/:\x00]+").expect("valid stem filename pattern"));
    let escaped = STEM_ESCAPE.replace_all(stem, "_");
    let stem = match escaped.trim() {
        "" => "stem",
        name => name,
    };
    let directory = match layout {
        Layout::Folders => root.join(track).join(folder),
        Layout::Flat => root.join(folder),
    };
    directory.join(format!("{track}_{stem}.{}", format.extension()))
}

fn resolve(
    step: &Step,
    defaults: &Defaults,
    paths: &Paths,
    model_dir: Option<&Path>,
) -> Result<(String, PathBuf, PathBuf)> {
    let (kind, checkpoint, config) = if let Some(name) = &step.model {
        let entry = supported_catalog_entry(name)?;
        let (checkpoint, config) =
            entry.model_paths(&directory(step, defaults, paths, model_dir))?;
        (
            entry
                .model_type
                .context("model type missing from catalog")?,
            checkpoint,
            config,
        )
    } else {
        let kind = step.model_type.clone().context("model_type missing")?;
        let checkpoint = paths
            .root
            .join(step.model_path.as_ref().context("model_path missing")?);
        let config = match &step.config_path {
            Some(path) => paths.root.join(path),
            None if crate::separation::optional_config(&kind) => checkpoint.clone(),
            None => bail!("config_path missing"),
        };
        (kind, checkpoint, config)
    };
    Ok((kind, checkpoint, config))
}

fn supported_catalog_entry(name: &str) -> Result<catalog::Entry> {
    let entry = catalog::by_name(name)?;
    ensure!(entry.supported, "{}", entry.unsupported_reason);
    Ok(entry)
}

fn validate_models(
    workflow: &Workflow,
    paths: &Paths,
    model_dir: Option<&Path>,
    require_files: bool,
) -> Result<()> {
    for step in &workflow.steps {
        let files: Vec<_> = if let Some(name) = &step.model {
            let entry = supported_catalog_entry(name)?;
            if !require_files {
                continue;
            }
            let directory = directory(step, &workflow.defaults, paths, model_dir);
            entry
                .files(&directory)?
                .into_iter()
                .map(|file| file.path)
                .collect()
        } else if require_files {
            [step.model_path.as_deref(), step.config_path.as_deref()]
                .into_iter()
                .flatten()
                .map(|file| paths.root.join(file))
                .collect()
        } else {
            continue;
        };
        for file in files {
            ensure!(
                file.is_file(),
                "missing model file for step {}: {}",
                step.id,
                file.display()
            );
        }
    }
    Ok(())
}

fn directories<'a>(value: &'a Value, step: &'a str) -> Result<Vec<&'a str>> {
    let directories: Vec<_> = match value {
        Value::Null | Value::Bool(false) => Vec::new(),
        Value::Bool(true) => vec![step],
        Value::String(value) => vec![value.trim()],
        Value::Array(values) => values
            .iter()
            .map(|value| {
                value
                    .as_str()
                    .map(str::trim)
                    .context("expected output folder name")
            })
            .collect::<Result<_>>()?,
        _ => bail!("workflow save targets must be strings, lists, or booleans"),
    };
    ensure!(
        directories.iter().all(|folder| Path::new(folder)
            .components()
            .all(|c| matches!(c, Component::Normal(_)))),
        "workflow output folders must stay inside the output directory"
    );
    Ok(directories
        .into_iter()
        .filter(|value| !value.is_empty())
        .collect())
}

struct Track {
    name: String,
    audio: HashMap<String, Pcm>,
    active: bool,
}

fn tracks(path: &Path) -> Result<Vec<Track>> {
    let files = if path.is_file() {
        vec![path.to_owned()]
    } else {
        audio::files(path, false)?
    };
    ensure!(!files.is_empty(), "no input recordings");
    let names: HashSet<_> = files
        .iter()
        .filter_map(|path| path.file_stem())
        .map(|name| name.to_string_lossy().into_owned())
        .collect();
    let mut used = HashSet::new();
    let mut tracks = Vec::new();
    for file in files {
        let stem = file
            .file_stem()
            .context("no input file name")?
            .to_string_lossy();
        let mut name = stem.to_string();
        let mut suffix = 2;
        while !used.insert(name.clone()) {
            loop {
                name = format!("{stem}_{suffix}");
                suffix += 1;
                if !names.contains(&name) {
                    break;
                }
            }
        }
        let (rate, channels) = audio::probe(&file)?;
        let samples = audio::read_interleaved(&file, rate, channels)?;
        let pcm = Pcm::from_interleaved(&samples, channels as usize, rate)?;
        tracks.push(Track {
            name,
            audio: HashMap::from([("input".to_owned(), pcm)]),
            active: true,
        });
    }
    Ok(tracks)
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn workflow_validation_checks_catalog_and_local_files_only_when_requested() -> Result<()> {
        let root = tempfile::tempdir()?;
        let paths = Paths::new(root.path())?;
        let workflow = paths.root.join("workflow.yaml");
        let validate = |flags: &[&str]| -> Result<()> {
            let command = [
                "rvc",
                "pymss",
                "workflow",
                "validate",
                "-c",
                "workflow.yaml",
            ];
            let cli = crate::Cli::try_parse_from(command.into_iter().chain(flags.iter().copied()))?;
            let crate::Command::Pymss(cli) = cli.command else {
                unreachable!();
            };
            super::super::run(&paths, cli)
        };

        std::fs::write(
            &workflow,
            "version: 1\nsteps:\n- id: one\n  model: unknown-model\n",
        )?;
        validate(&[])?;
        assert!(
            validate(&["--check-models"])
                .unwrap_err()
                .to_string()
                .contains("Unknown pymss model")
        );

        let name = "Apollo_LQ_MP3_restoration.ckpt";
        std::fs::write(
            &workflow,
            format!("version: 1\nsteps:\n- id: one\n  model: {name}\n"),
        )?;
        let mut files = catalog::by_name(name)?.files(&paths.root.join("models"))?;
        let auxiliary = files.pop().unwrap();
        assert_eq!(auxiliary.role, "auxiliary");
        for file in files {
            std::fs::create_dir_all(file.path.parent().unwrap())?;
            std::fs::write(file.path, [])?;
        }
        validate(&["--check-models", "--model-dir", "models"])?;
        let error = validate(&["--require-files", "--model-dir", "models"]).unwrap_err();
        assert!(error.to_string().contains(auxiliary.path.to_str().unwrap()));
        std::fs::write(auxiliary.path, [])?;
        validate(&["--require-files", "--model-dir", "models"])?;

        std::fs::write(
            &workflow,
            "version: 1\nsteps:\n- id: one\n  model_type: custom\n  model_path: model.pt\n",
        )?;
        validate(&[])?;
        validate(&["--check-models"])?;
        assert!(
            validate(&["--require-model-files"])
                .unwrap_err()
                .to_string()
                .contains("model.pt")
        );
        std::fs::write(paths.root.join("model.pt"), [])?;
        validate(&["--require-model-files"])?;
        Ok(())
    }

    #[test]
    fn accepts_single_stem_and_list_with_whitespace() -> Result<()> {
        let single: Step = serde_yaml::from_str("id: one\nmodel: example\nstems: ' vocals '")?;
        let list: Step =
            serde_yaml::from_str("id: two\nmodel: example\nstems: [' speech ', '', music]")?;
        assert_eq!(single.stems.unwrap(), ["vocals"]);
        assert_eq!(list.stems.unwrap(), ["speech", "music"]);
        Ok(())
    }

    #[test]
    fn workflow_layouts_keep_stem_names_and_save_subdirectories() {
        let root = Path::new("results");
        let folders = output_path(
            root,
            Layout::Folders,
            "track",
            "nested/mix",
            "vocals",
            Format::Wav,
        );
        let flat = output_path(
            root,
            Layout::Flat,
            "track",
            "nested/mix",
            "voice:/lead",
            Format::Wav,
        );
        assert_eq!(folders, root.join("track/nested/mix/track_vocals.wav"));
        assert_eq!(flat, root.join("nested/mix/track_voice_lead.wav"));
    }
}
