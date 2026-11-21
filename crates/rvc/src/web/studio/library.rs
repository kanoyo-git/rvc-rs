use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io;
use std::ops::ControlFlow;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use anyhow::{Context, Result, bail, ensure};
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
        let name = request.name.unwrap_or_else(|| {
            url.split(['?', '#'])
                .next()
                .and_then(|part| part.rsplit('/').next())
                .filter(|part| {
                    model_member(Path::new(part)) || part.to_lowercase().ends_with(".zip")
                })
                .unwrap_or("models.zip")
                .to_owned()
        });
        let path = uploads.directory.path().join(filename(&name)?);
        download(url, &path, &mut report)?;
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
    // Known pairs (RVC and NVC-Pretrained) are listed before they exist: training fetches them.
    let known = crate::assets::known_pretrained(paths);
    let nvc = directory.join("NVC-Pretrained");
    let role_paths = |role: &str| {
        let name = format!("{prefix}{role}{rate}.pth");
        let matches = |path: &&PathBuf| {
            path.file_name()
                .is_some_and(|file| file.eq_ignore_ascii_case(&name))
        };
        let mut found = assets
            .iter()
            .chain(known.iter().filter(|path| path.starts_with(&directory)))
            .filter(matches)
            .cloned()
            .collect::<Vec<_>>();
        found.sort();
        found.dedup();
        let preferred = nvc.join(&name);
        let default = if found.contains(&preferred) {
            preferred
        } else {
            directory.join(&name)
        };
        let installed = found
            .iter()
            .filter(|path| path.is_file())
            .cloned()
            .collect::<Vec<_>>();
        (
            found.contains(&default).then_some(default),
            found,
            installed,
        )
    };
    let (generator, generators, installed_g) = role_paths("G");
    let (discriminator, discriminators, installed_d) = role_paths("D");
    let catalog = if query.version == Version::V2 && query.f0 {
        CATALOG
            .iter()
            .filter(|(_, entries)| entries.contains_key(&rate))
            .map(|(name, _)| {
                let [g, d] = pretrained_targets(paths, name, &rate);
                PretrainedOption {
                    name: name.clone(),
                    installed: g.is_file() && d.is_file(),
                }
            })
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    json!({
        "generator": generators, "discriminator": discriminators,
        "default_generator": generator, "default_discriminator": discriminator,
        "installed": installed_g.into_iter().chain(installed_d).collect::<Vec<_>>(),
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
    let directory = if name == "RVC v2" {
        directory
    } else {
        directory.join(name)
    };
    [
        directory.join(format!("f0G{rate}.pth")),
        directory.join(format!("f0D{rate}.pth")),
    ]
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
    let urls = match (request.generator, request.discriminator) {
        (None, None) => {
            let entry = CATALOG
                .get(&request.name)
                .and_then(|set| set.get(&rate))
                .context("unknown pretrained catalog entry")?;
            [&entry.generator, &entry.discriminator]
                .map(|file| format!("https://huggingface.co/{file}"))
        }
        (Some(g), Some(d)) => [g, d],
        _ => bail!("both generator and discriminator URLs are required"),
    };
    let staging = TempDir::new_in(workspace)?;
    let targets = pretrained_targets(paths, &request.name, &rate);
    let mut installs = Vec::new();
    for (i, (target, url)) in targets.iter().zip(urls).enumerate() {
        if target.is_file() {
            check_pretrained(target, i, request.sample_rate)?;
            continue;
        }
        let input = staging.path().join(format!("{i}.pth"));
        download(&url, &input, &mut report)?;
        check(&mut report, "Checking pretrained checkpoint")?;
        check_pretrained(&input, i, request.sample_rate)?;
        installs.push(stage_install(&input, target.clone())?);
    }
    check(&mut report, "Installing pretrained weights")?;
    publish(installs)?;
    Ok(json!({ "paths": targets }))
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

fn download(
    raw: &str,
    output: &Path,
    report: &mut impl FnMut(&str) -> ControlFlow<()>,
) -> Result<()> {
    let url = direct_url(raw)?;
    let label = url
        .rsplit('/')
        .next()
        .unwrap_or_default()
        .split('?')
        .next()
        .unwrap_or_default()
        .to_owned();
    crate::download::download(&url, output, &label, None, report)
}

fn direct_url(raw: &str) -> Result<String> {
    let url: axum::http::Uri = raw.parse().context("invalid download URL")?;
    ensure!(
        matches!(url.scheme_str(), Some("http" | "https")) && url.host().is_some(),
        "use an HTTP(S) download URL"
    );
    match url.host().unwrap_or_default() {
        "huggingface.co" | "hf.co" => Ok(raw.replacen("/blob/", "/resolve/", 1)),
        "drive.google.com" => {
            if let Some((_, rest)) = url.path().split_once("/file/d/") {
                let id = rest.split('/').next().unwrap_or_default();
                ensure!(!id.is_empty() && id.bytes().all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_')),
                    "invalid Drive file ID");
                Ok(format!(
                    "https://drive.google.com/uc?export=download&id={id}"
                ))
            } else {
                Ok(raw.to_owned())
            }
        }
        _ => Ok(raw.to_owned()),
    }
}
