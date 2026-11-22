use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io;
use std::ops::ControlFlow;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tempfile::{NamedTempFile, TempDir};
use walkdir::WalkDir;

use super::files::{Uploads, filename, unzip};
use crate::catalog::{ModelInfo, Paths};
use crate::checkpoint::Checkpoint;
use crate::model::{register::Profile, train_data::Version};

#[derive(Deserialize)]
pub(super) struct Import {
    url: Option<String>,
    name: Option<String>,
}

pub(super) fn import(
    paths: &Paths,
    request: Import,
    uploads: &Uploads,
    mut report: impl FnMut(&str) -> ControlFlow<()>,
) -> Result<Value> {
    let path = if uploads.files.is_empty() {
        let url = request
            .url
            .as_deref()
            .context("supply a model URL or upload a file")?;
        let payload = uploads.directory.path().join("download");
        let suggested = super::shared_link::download(url, &payload, &mut report)?;
        let requested = request.name.as_deref().filter(|s| !s.is_empty());
        let stem = filename(requested.or(suggested.as_deref()).unwrap_or("model"))?;
        let name = if model_member(stem) || extension(stem, "zip") {
            stem.to_owned()
        } else {
            let suffix = suggested
                .as_deref()
                .map(Path::new)
                .and_then(|p| {
                    if model_member(p) && !extension(p, "pth") && !extension(p, "index") {
                        Some("pitch.json")
                    } else if extension(p, "index") {
                        Some("index")
                    } else if extension(p, "pth") {
                        Some("pth")
                    } else if extension(p, "zip") {
                        Some("zip")
                    } else {
                        None
                    }
                })
                .unwrap_or_else(|| {
                    if ModelInfo::read(&payload).is_ok() {
                        "pth"
                    } else {
                        "zip"
                    }
                });
            PathBuf::from(format!("{}.{suffix}", stem.display()))
        };
        let path = uploads.directory.path().join(name);
        fs::rename(payload, &path)?;
        path
    } else {
        uploads.one()?.to_owned()
    };
    let stage = TempDir::new_in(uploads.directory.path())?;
    let files = if extension(&path, "zip") {
        unzip(&path, stage.path(), model_member, || {
            check(&mut report, "Extracting models")
        })?
    } else {
        ensure!(
            model_member(&path),
            "expected a .zip, .pth, .index or .pitch.json file"
        );
        vec![path]
    };
    ensure!(
        !files.is_empty(),
        "model archive contains no supported files"
    );
    let mut targets = BTreeSet::new();
    let mut staged = Vec::new();
    for source in files {
        let name = source.file_name().context("missing imported filename")?;
        check(&mut report, &format!("Checking {}", name.to_string_lossy()))?;
        let directory = if extension(&source, "pth") {
            ModelInfo::read(&source).context("expected a compact RVC voice model")?;
            &paths.weights
        } else if extension(&source, "index") {
            ensure!(source.metadata()?.len() >= 4, "index file is empty");
            &paths.indices
        } else {
            Profile::read(&source)?;
            &paths.weights
        };
        let output = directory.join(name);
        ensure!(
            targets.insert(output.clone()),
            "duplicate imported filename {}",
            name.to_string_lossy()
        );
        staged.push(stage_install(&source, output)?);
    }
    check(&mut report, "Installing models")?;
    Ok(json!({ "paths": publish(staged)?, "models": paths.models()? }))
}

fn extension(path: &Path, wanted: &str) -> bool {
    path.extension()
        .is_some_and(|e| e.eq_ignore_ascii_case(wanted))
}

fn model_member(path: &Path) -> bool {
    extension(path, "pth")
        || extension(path, "index")
        || path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| {
                let name = name.to_ascii_lowercase();
                name.ends_with(".pitch.json")
                    || matches!(name.as_str(), "pitch-profile.json" | "pitch_profile.json")
            })
}

#[derive(Deserialize)]
pub(super) struct PretrainedQuery {
    pub(super) version: Version,
    pub(super) sample_rate: u32,
    pub(super) f0: bool,
}

#[derive(Serialize)]
pub(super) struct PretrainedOption {
    name: String,
    paths: [PathBuf; 2],
    installed: bool,
}

#[derive(Deserialize)]
struct CatalogWeights {
    #[serde(rename = "G")]
    generator: String,
    #[serde(rename = "D")]
    discriminator: String,
}

static CATALOG: LazyLock<BTreeMap<String, BTreeMap<String, CatalogWeights>>> =
    LazyLock::new(|| {
        serde_json::from_str(include_str!("pretrains.json")).expect("embedded pretrained catalog")
    });

pub(super) fn choices(paths: &Paths, query: &PretrainedQuery) -> Value {
    let directory = paths.root.join(match query.version {
        Version::V1 => "assets/pretrained",
        Version::V2 => "assets/pretrained_v2",
    });
    let rate = format!("{}k", query.sample_rate / 1000);
    let prefix = if query.f0 { "f0" } else { "" };
    let assets = WalkDir::new(&directory)
        .follow_links(true)
        .max_depth(3)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|entry| entry.file_type().is_file() && extension(entry.path(), "pth"))
        .map(|entry| entry.into_path())
        .collect::<Vec<_>>();
    let catalog: Vec<_> = CATALOG
        .iter()
        .filter(|(_, entries)| {
            query.version == Version::V2 && query.f0 && entries.contains_key(&rate)
        })
        .map(|(name, _)| {
            let pair = pretrained_targets(paths, name, &rate);
            PretrainedOption {
                name: name.clone(),
                installed: pair.iter().all(|path| path.is_file()),
                paths: pair,
            }
        })
        .collect();
    let mut available: BTreeSet<PathBuf> = assets.into_iter().collect();
    let legacy = directory.join("NVC-Pretrained");
    available.extend(
        crate::assets::known_pretrained(paths)
            .into_iter()
            .filter(|p| p.starts_with(&directory) && (!p.starts_with(&legacy) || p.is_file())),
    );
    available.extend(catalog.iter().flat_map(|item| item.paths.iter().cloned()));
    let role_paths = |role: &str| {
        let name = format!("{prefix}{role}{rate}.pth");
        let default = directory.join(&name);
        let found: Vec<_> = available
            .iter()
            .filter(|p| {
                p.file_name()
                    .is_some_and(|file| file.eq_ignore_ascii_case(&name))
            })
            .collect();
        (available.contains(&default).then_some(default), found)
    };
    let (generator, generators) = role_paths("G");
    let (discriminator, discriminators) = role_paths("D");
    let installed: Vec<_> = generators
        .iter()
        .chain(&discriminators)
        .filter(|p| p.is_file())
        .collect();
    json!({
        "generator": generators, "discriminator": discriminators,
        "default_generator": generator, "default_discriminator": discriminator,
        "installed": installed,
        "catalog": catalog,
    })
}

#[derive(Deserialize)]
pub(super) struct InstallPair {
    name: String,
    sample_rate: u32,
    generator: Option<String>,
    discriminator: Option<String>,
}

fn pretrained_targets(paths: &Paths, name: &str, rate: &str) -> [PathBuf; 2] {
    let directory = paths.root.join("assets/pretrained_v2");
    let directory = match name {
        "RVC v2" => directory,
        "NVC-Pretrained" => directory.join("NVC-Pretrained-Rage-ContentVec"),
        _ => directory.join(name),
    };
    [
        directory.join(format!("f0G{rate}.pth")),
        directory.join(format!("f0D{rate}.pth")),
    ]
}

/// Fetch a missing community pair for the paths actually submitted by Studio.
pub(super) fn prepare_training(
    paths: &Paths,
    request: &crate::model::train::Request,
    workspace: &Path,
    report: &mut impl FnMut(&str) -> ControlFlow<()>,
) -> Result<()> {
    if request.version != Version::V2 || !request.f0 {
        return Ok(());
    }
    let experiment = paths.root.join(&request.experiment);
    if experiment.is_dir() && crate::model::train::has_checkpoint_pair(&experiment)? {
        return Ok(());
    }
    let chosen = [&request.pretrained_g, &request.pretrained_d]
        .map(|part| part.as_ref().map(|p| paths.root.join(p)));
    let rate = format!("{}k", request.sample_rate / 1000);
    for (name, entries) in CATALOG
        .iter()
        .filter(|(_, entries)| entries.contains_key(&rate))
    {
        let targets = pretrained_targets(paths, name, &rate);
        if targets
            .iter()
            .zip(&chosen)
            .any(|(target, part)| part.as_ref() == Some(target) && !target.is_file())
        {
            install_catalog_pair(
                paths,
                name,
                request.sample_rate,
                &entries[&rate],
                workspace,
                report,
            )?;
        }
    }
    Ok(())
}

pub(super) fn install_pretrained(
    paths: &Paths,
    request: InstallPair,
    workspace: &Path,
    mut report: impl FnMut(&str) -> ControlFlow<()>,
) -> Result<Value> {
    filename(&request.name)?;
    ensure!(
        matches!(request.sample_rate, 32_000 | 40_000 | 48_000),
        "invalid sample rate"
    );
    let rate = format!("{}k", request.sample_rate / 1000);
    let targets = pretrained_targets(paths, &request.name, &rate);
    if request.generator.is_none() && request.discriminator.is_none() {
        let entry = CATALOG
            .get(&request.name)
            .and_then(|entries| entries.get(&rate))
            .context("unknown pretrained catalog entry")?;
        install_catalog_pair(
            paths,
            &request.name,
            request.sample_rate,
            entry,
            workspace,
            &mut report,
        )?;
        return Ok(json!({ "paths": targets }));
    }
    let staging = TempDir::new_in(workspace)?;
    let mut installs = Vec::new();
    let mut selected = Vec::new();
    let archive_allowed = request.discriminator.is_none();
    for (i, url) in [request.generator, request.discriminator]
        .into_iter()
        .enumerate()
    {
        let Some(url) = url else { continue };
        let input = staging.path().join(format!("input-{i}"));
        let suggested = super::shared_link::download(&url, &input, &mut report)?;
        let archive = i == 0
            && archive_allowed
            && Checkpoint::open(&input).is_err()
            && (suggested
                .as_ref()
                .is_some_and(|s| extension(Path::new(s), "zip"))
                || zip::ZipArchive::new(File::open(&input)?).is_ok());
        if archive {
            let members = unzip(
                &input,
                staging.path(),
                |p| extension(p, "pth"),
                || check(&mut report, "Extracting pretrained weights"),
            )?;
            let mut pair: [Option<PathBuf>; 2] = [None, None];
            for member in members {
                let role = (0..2)
                    .find(|&role| check_pretrained(&member, role, request.sample_rate).is_ok());
                if let Some(role) = role {
                    ensure!(
                        pair[role].is_none(),
                        "archive contains multiple compatible weights for the same role"
                    );
                    pair[role] = Some(member);
                }
            }
            for (role, member) in pair.into_iter().enumerate() {
                let source = member.context(
                    "archive must contain one compatible generator and one discriminator",
                )?;
                installs.push(stage_install(&source, targets[role].clone())?);
                selected.push(targets[role].clone());
            }
        } else {
            check_pretrained(&input, i, request.sample_rate)?;
            installs.push(stage_install(&input, targets[i].clone())?);
            selected.push(targets[i].clone());
        }
    }
    check(&mut report, "Installing pretrained weights")?;
    publish(installs)?;
    Ok(json!({ "paths": selected }))
}

fn install_catalog_pair(
    paths: &Paths,
    name: &str,
    sample_rate: u32,
    entry: &CatalogWeights,
    workspace: &Path,
    report: &mut impl FnMut(&str) -> ControlFlow<()>,
) -> Result<()> {
    let rate = format!("{}k", sample_rate / 1000);
    let targets = pretrained_targets(paths, name, &rate);
    let staging = TempDir::new_in(workspace)?;
    let mut installs = Vec::new();
    for (i, (target, remote)) in targets
        .iter()
        .zip([&entry.generator, &entry.discriminator])
        .enumerate()
    {
        if !target.is_file() && matches!(name, "RVC v2" | "NVC-Pretrained") {
            crate::assets::pretrained(paths, target, report)?;
        }
        if target.is_file() {
            check_pretrained(target, i, sample_rate)?;
            continue;
        }
        let input = staging.path().join(format!("{i}.pth"));
        crate::download::download(
            &format!("https://huggingface.co/{remote}"),
            &input,
            target.file_name().and_then(|s| s.to_str()).unwrap_or(name),
            None,
            report,
        )?;
        check(report, "Checking pretrained checkpoint")?;
        check_pretrained(&input, i, sample_rate)?;
        installs.push(stage_install(&input, target.clone())?);
    }
    check(report, "Installing pretrained weights")?;
    publish(installs)?;
    Ok(())
}

fn check_pretrained(path: &Path, role: usize, rate: u32) -> Result<()> {
    let checkpoint = Checkpoint::open(path).context("opening pretrained checkpoint")?;
    if role == 0 {
        let shape = checkpoint.tensor_shape(Some("model"), "enc_p.emb_phone.weight")?;
        ensure!(shape.get(1) == Some(&768), "expected an RVC v2 generator");
        checkpoint.tensor_shape(Some("model"), "enc_p.emb_pitch.weight")?;
        let bins = match rate {
            32_000 => 513,
            _ => 1025,
        };
        let encoder = checkpoint.tensor_shape(Some("model"), "enc_q.pre.weight")?;
        ensure!(
            encoder.get(1) == Some(&bins),
            "generator sample rate does not match"
        );
        let expected = match rate {
            32_000 => 20,
            40_000 => 16,
            _ => 24,
        };
        let upsample = checkpoint.tensor_shape(Some("model"), "dec.ups.0.weight_v")?;
        ensure!(
            upsample.last() == Some(&expected),
            "generator upsampling does not match the selected sample rate"
        );
    } else {
        let names = checkpoint.tensor_names(Some("model"))?;
        ensure!(
            names
                .iter()
                .any(|name| name.starts_with("discriminators.0."))
                && names
                    .iter()
                    .any(|name| name.starts_with("discriminators.8.")),
            "expected an RVC v2 discriminator"
        );
    }
    Ok(())
}

fn stage_install(source: &Path, target: PathBuf) -> Result<(NamedTempFile, PathBuf)> {
    let directory = target.parent().context("model has no directory")?;
    fs::create_dir_all(directory)?;
    ensure!(!target.exists(), "{} already exists", target.display());
    let mut file = NamedTempFile::new_in(directory)?;
    io::copy(&mut File::open(source)?, &mut file)?;
    Ok((file, target))
}

fn publish(files: Vec<(NamedTempFile, PathBuf)>) -> Result<Vec<PathBuf>> {
    let mut installed = Vec::new();
    for (source, path) in files {
        if let Err(error) = source.persist_noclobber(&path) {
            for item in installed {
                let _ = fs::remove_file(item);
            }
            return Err(error).with_context(|| format!("installing {}", path.display()));
        }
        installed.push(path);
    }
    Ok(installed)
}

fn check(report: &mut impl FnMut(&str) -> ControlFlow<()>, message: &str) -> Result<()> {
    ensure!(report(message).is_continue(), "import cancelled");
    Ok(())
}

#[cfg(test)]
mod pretrained_choices_tests {
    use super::*;

    #[test]
    fn catalog_paths_are_selectable_before_download_and_default_to_stock() -> Result<()> {
        let root = TempDir::new()?;
        let paths = Paths::new(root.path())?;
        let query = PretrainedQuery {
            version: Version::V2,
            sample_rate: 40_000,
            f0: true,
        };
        let data = choices(&paths, &query);
        let set = |name: &str| {
            data["catalog"]
                .as_array()
                .unwrap()
                .iter()
                .find(|item| item["name"] == name)
                .unwrap()
        };
        let stock = set("RVC v2");
        let klm = set("KLM4");
        let nvc = set("NVC-Pretrained");
        assert_eq!(data["default_generator"], stock["paths"][0]);
        assert_eq!(data["default_discriminator"], stock["paths"][1]);
        assert_eq!(
            klm["paths"],
            json!(pretrained_targets(&paths, "KLM4", "40k"))
        );
        assert!(
            klm["paths"][1]
                .as_str()
                .unwrap()
                .ends_with("KLM4/f0D40k.pth")
        );
        assert!(
            nvc["paths"][0]
                .as_str()
                .unwrap()
                .ends_with("NVC-Pretrained-Rage-ContentVec/f0G40k.pth")
        );
        assert_eq!(klm["installed"], false);
        for item in data["catalog"].as_array().unwrap() {
            for (i, key) in ["generator", "discriminator"].into_iter().enumerate() {
                assert!(data[key].as_array().unwrap().contains(&item["paths"][i]));
            }
        }
        let [g, d] = pretrained_targets(&paths, "KLM4", "40k");
        fs::create_dir_all(g.parent().unwrap())?;
        File::create(&g)?;
        let partial = choices(&paths, &query);
        assert!(partial["installed"].as_array().unwrap().contains(&json!(g)));
        assert!(!partial["installed"].as_array().unwrap().contains(&json!(d)));
        File::create(&d)?;
        let complete = choices(&paths, &query);
        assert!(
            complete["catalog"]
                .as_array()
                .unwrap()
                .iter()
                .any(|item| item["name"] == "KLM4" && item["installed"] == true)
        );
        let other = choices(&paths, &PretrainedQuery { f0: false, ..query });
        assert!(other["catalog"].as_array().unwrap().is_empty());
        Ok(())
    }
}
