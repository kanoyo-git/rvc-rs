use std::fs::File;
use std::io;
use std::path::{Component, Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail, ensure};
use clap::ValueEnum;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const CATALOG: &str = include_str!("../../resources/pymss/model_catalog.json");

#[derive(Deserialize)]
struct Catalog {
    models: Vec<Entry>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Entry {
    pub name: String,
    pub aliases: Vec<String>,
    pub model_type: Option<String>,
    pub architecture: String,
    pub supported: bool,
    pub unsupported_reason: String,
    pub relpath: String,
    pub config_relpath: String,
    pub auxiliary_relpaths: Vec<String>,
    pub size_bytes: u64,
    pub sha256: String,
    pub primary_category: String,
    pub primary_category_cn: String,
    pub secondary_category: String,
    pub secondary_category_cn: String,
    pub target_stem: String,
    pub config_instruments: String,
    pub config_target_instrument: String,
    pub classification_confidence: String,
    pub classification_basis: String,
}

impl Entry {
    pub fn categories(&self) -> String {
        [&self.primary_category, &self.secondary_category]
            .into_iter()
            .filter(|part| !part.is_empty())
            .map(String::as_str)
            .collect::<Vec<_>>()
            .join("/")
    }

    pub fn files(&self, directory: &Path) -> Result<Vec<ModelFile>> {
        let mut files = vec![ModelFile {
            role: "model",
            path: resolve_relative(directory, &self.relpath)?,
            relative: self.relpath.clone(),
            size: (self.size_bytes > 0).then_some(self.size_bytes),
            digest: self.sha256.clone(),
        }];
        if !self.config_relpath.is_empty() {
            files.push(ModelFile {
                role: "config",
                path: resolve_relative(directory, &self.config_relpath)?,
                relative: self.config_relpath.clone(),
                size: None,
                digest: String::new(),
            });
        }
        for relative in &self.auxiliary_relpaths {
            files.push(ModelFile {
                role: "auxiliary",
                path: resolve_relative(directory, relative)?,
                relative: relative.clone(),
                size: None,
                digest: String::new(),
            });
        }
        Ok(files)
    }

    pub fn model_paths(&self, directory: &Path) -> Result<(PathBuf, PathBuf)> {
        let checkpoint = resolve_relative(directory, &self.relpath)?;
        let config = if self.config_relpath.is_empty() {
            ensure!(
                self.model_type
                    .as_deref()
                    .is_some_and(crate::separation::optional_config),
                "catalog model does not have a YAML configuration"
            );
            checkpoint.clone()
        } else {
            resolve_relative(directory, &self.config_relpath)?
        };
        Ok((checkpoint, config))
    }
}

pub fn entries(category: Option<&str>, supported: Option<bool>) -> Result<Vec<Entry>> {
    let source: Catalog =
        serde_json::from_str(CATALOG).context("reading embedded PyMSS catalog")?;
    let category = category.map(str::to_lowercase);
    Ok(source
        .models
        .into_iter()
        .filter(|entry| {
            let selected = category.as_ref().is_none_or(|name| {
                entry.primary_category.to_lowercase() == *name
                    || entry.secondary_category.to_lowercase() == *name
                    || entry.categories().to_lowercase() == *name
            });
            selected && supported.is_none_or(|wanted| entry.supported == wanted)
        })
        .collect())
}

pub fn by_name(name: &str) -> Result<Entry> {
    let query = name.trim().to_lowercase();
    entries(None, None)?
        .into_iter()
        .find(|entry| {
            entry.name.to_lowercase() == query
                || Path::new(&entry.name)
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .is_some_and(|stem| stem.to_lowercase() == query)
                || entry
                    .aliases
                    .iter()
                    .any(|alias| alias.to_lowercase() == query)
        })
        .with_context(|| format!("Unknown pymss model: {name}"))
}

pub fn model_directory(root: &Path, configured: Option<&Path>) -> PathBuf {
    if let Some(path) = configured {
        return root.join(path);
    }
    if let Some(path) = std::env::var_os("PYMSS_MODEL_DIR").filter(|path| !path.is_empty()) {
        return root.join(path);
    }
    let local = root.join("tools/all_models");
    if local.is_dir() {
        return local;
    }
    std::env::home_dir()
        .unwrap_or_else(|| root.to_owned())
        .join(".cache/pymss/models")
}

fn resolve_relative(directory: &Path, relative: &str) -> Result<PathBuf> {
    let relative = Path::new(relative);
    ensure!(
        !relative.as_os_str().is_empty()
            && relative
                .components()
                .all(|part| matches!(part, Component::Normal(_))),
        "invalid catalog-relative path {}",
        relative.display()
    );
    Ok(directory.join(relative))
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum Source {
    #[default]
    Modelscope,
    Huggingface,
    HfMirror,
}

impl Source {
    pub fn url(self, relative: &str, endpoint: Option<&str>) -> Result<String> {
        let base = endpoint.unwrap_or(match self {
            Self::Modelscope => "https://www.modelscope.cn/models/baicai1145/pymss/resolve/master",
            Self::Huggingface => "https://huggingface.co/baicai1145/pymss/resolve/main",
            Self::HfMirror => "https://hf-mirror.com/baicai1145/pymss/resolve/main",
        });
        ensure!(
            base.starts_with("https://") || base.starts_with("http://"),
            "expected an HTTP or HTTPS model endpoint"
        );
        let mut url = String::from(base.trim_end_matches('/'));
        url.push('/');
        use std::fmt::Write;
        for byte in relative.bytes() {
            if byte.is_ascii_alphanumeric() || b"/-._~".contains(&byte) {
                url.push(byte as char);
            } else {
                write!(url, "%{byte:02X}")?;
            }
        }
        Ok(url)
    }
}

pub struct ModelFile {
    pub role: &'static str,
    pub path: PathBuf,
    pub relative: String,
    size: Option<u64>,
    digest: String,
}

#[derive(Default, Debug, Serialize)]
pub struct Download {
    pub downloaded: Vec<PathBuf>,
    pub skipped: Vec<PathBuf>,
}

pub struct DownloadOptions<'a> {
    pub source: Source,
    pub endpoint: Option<&'a str>,
    pub force: bool,
    pub verify: bool,
    pub timeout_seconds: f64,
}

pub fn download(
    entry: &Entry,
    directory: &Path,
    source: Source,
    endpoint: Option<&str>,
    force: bool,
    timeout: u64,
) -> Result<Download> {
    download_with_options(
        entry,
        directory,
        DownloadOptions {
            source,
            endpoint,
            force,
            verify: true,
            timeout_seconds: timeout as f64,
        },
    )
}

pub fn download_with_options(
    entry: &Entry,
    directory: &Path,
    options: DownloadOptions<'_>,
) -> Result<Download> {
    ensure!(
        options.timeout_seconds.is_finite() && options.timeout_seconds > 0.0,
        "download timeout must be finite and greater than zero"
    );
    // Curl would round a sub-millisecond timeout down to zero, disabling it.
    let timeout = options.timeout_seconds.max(0.001).to_string();
    let verify = options.verify && options.endpoint.is_none();
    let mut report = Download::default();
    for file in entry.files(directory)? {
        let url = options.source.url(&file.relative, options.endpoint)?;
        if !options.force && file.valid(&file.path, verify)? {
            report.skipped.push(file.path);
            continue;
        }
        let parent = file
            .path
            .parent()
            .context("catalog file lacks a destination")?;
        std::fs::create_dir_all(parent)?;
        let staged = tempfile::NamedTempFile::new_in(parent)?;
        let output = Command::new("curl")
            .args([
                "--fail",
                "--location",
                "--silent",
                "--show-error",
                "--proto",
                "=http,https",
                "--proto-redir",
                "=http,https",
                "--retry",
                "2",
                "--connect-timeout",
                "15",
            ])
            .args(["--max-time", &timeout, "--output"])
            .arg(staged.path())
            .args(["--", &url])
            .output()
            .context("starting curl")?;
        if !output.status.success() {
            bail!(
                "download failed for {}: {}",
                file.relative,
                String::from_utf8_lossy(&output.stderr).trim()
            );
        }
        ensure!(
            file.valid(staged.path(), verify)?,
            "downloaded size or SHA-256 does not match {}",
            file.relative
        );
        staged.persist(&file.path)?;
        report.downloaded.push(file.path);
    }
    Ok(report)
}

impl ModelFile {
    fn valid(&self, path: &Path, verify: bool) -> Result<bool> {
        if !path.is_file() {
            return Ok(false);
        }
        if !verify {
            return Ok(true);
        }
        let mut file = File::open(path)?;
        let size = file.metadata()?.len();
        if size == 0 || self.size.is_some_and(|expected| size != expected) {
            return Ok(false);
        }
        if self.digest.is_empty() {
            return Ok(true);
        }
        let mut digest = Sha256::new();
        io::copy(&mut file, &mut digest)?;
        Ok(format!("{:x}", digest.finalize()).eq_ignore_ascii_case(&self.digest))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn download_options_apply_to_cached_files() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let entry = entries(None, None)?
            .into_iter()
            .find(|entry| entry.size_bytes > 0)
            .context("missing model with catalog metadata")?;
        let files = entry.files(directory.path())?;
        for file in &files {
            std::fs::create_dir_all(file.path.parent().context("missing parent")?)?;
            File::create(&file.path)?;
        }

        for (endpoint, verify) in [(None, false), (Some("http://localhost"), true)] {
            let report = download_with_options(
                &entry,
                directory.path(),
                DownloadOptions {
                    source: Source::Modelscope,
                    endpoint,
                    force: false,
                    verify,
                    timeout_seconds: 0.25,
                },
            )?;
            assert!(report.downloaded.is_empty());
            assert!(
                report
                    .skipped
                    .iter()
                    .eq(files.iter().map(|file| &file.path))
            );
        }
        assert!(!files[0].valid(&files[0].path, true)?);

        for timeout_seconds in [0.0, -0.5, f64::NAN, f64::INFINITY] {
            let result = download_with_options(
                &entry,
                directory.path(),
                DownloadOptions {
                    source: Source::Modelscope,
                    endpoint: None,
                    force: false,
                    verify: false,
                    timeout_seconds,
                },
            );
            assert!(result.is_err());
        }
        assert!(
            download(
                &entry,
                directory.path(),
                Source::Modelscope,
                Some("file:///tmp"),
                false,
                1,
            )
            .is_err()
        );
        Ok(())
    }

    #[test]
    fn verified_download_checks_size_and_sha256() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let file = ModelFile {
            role: "model",
            path: directory.path().join("model.pth"),
            relative: "model.pth".into(),
            size: Some(3),
            digest: format!("{:x}", Sha256::digest(b"abc")),
        };
        std::fs::write(&file.path, b"abc")?;
        assert!(file.valid(&file.path, true)?);
        std::fs::write(&file.path, b"abd")?;
        assert!(!file.valid(&file.path, true)?);
        std::fs::write(&file.path, b"abcd")?;
        assert!(!file.valid(&file.path, true)?);
        Ok(())
    }
}
