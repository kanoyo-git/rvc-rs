use std::path::PathBuf;

use anyhow::{Result, ensure};
use clap::ValueEnum;
use serde::{Deserialize, Serialize};

use super::Paths;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize, Serialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum Embedder {
    #[default]
    #[serde(rename = "hubert_base", alias = "hubert-base")]
    #[value(name = "hubert_base", alias = "hubert-base")]
    HubertBase,
    Contentvec,
    Spin,
    SpinV2,
    Mhubert147,
    /// Layer 18 of a Russian CTC wav2vec 2.0 large model, projected to 768 channels.
    /// Has no download: place a converted model in `assets/w2v_ru`.
    #[serde(rename = "w2v_ru")]
    #[value(name = "w2v_ru")]
    W2vRu,
}

impl Embedder {
    pub fn name(self) -> &'static str {
        match self {
            Self::HubertBase => "hubert_base",
            Self::Contentvec => "contentvec",
            Self::Spin => "spin",
            Self::SpinV2 => "spin-v2",
            Self::Mhubert147 => "mhubert147",
            Self::W2vRu => "w2v_ru",
        }
    }

    pub fn directory(self, paths: &Paths) -> PathBuf {
        paths.root.join("assets").join(self.name())
    }

    pub fn supports(self, version: &str) -> Result<()> {
        ensure!(
            version != "v1" || !matches!(self, Self::Mhubert147 | Self::W2vRu),
            "{} produces 768 channels and has no RVC v1 projection. Select RVC v2.",
            self.name()
        );
        Ok(())
    }
}

#[cfg(feature = "inference")]
mod install {
    use std::fs::{self, File, OpenOptions};
    use std::ops::ControlFlow;
    use std::path::{Path, PathBuf};

    use anyhow::{Context, Result, ensure};
    use serde_json::Value;
    use tempfile::NamedTempFile;

    use super::{Embedder, Paths};
    use crate::checkpoint::Checkpoint;

    const APPLIO: &str = "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders";
    const MHUBERT: &str = "https://huggingface.co/utter-project/mHuBERT-147/resolve/main";
    /// Release assets are flat, so HuBERT Base files carry a `hubert_base_` prefix there.
    const HUBERT_BASE: &str =
        "https://github.com/kanoyo-git/rvc-rs/releases/download/resources/hubert_base_";

    impl Embedder {
        pub fn available(self, paths: &Paths) -> bool {
            let directory = self.directory(paths);
            self.files()
                .iter()
                .all(|name| directory.join(name).is_file())
        }

        fn files(self) -> &'static [&'static str] {
            match self {
                Self::HubertBase => &["pytorch_model.bin"],
                Self::Mhubert147 | Self::W2vRu => &[
                    "config.json",
                    "preprocessor_config.json",
                    "model.safetensors",
                ],
                _ => &["config.json", "pytorch_model.bin"],
            }
        }

        pub fn install(
            self,
            paths: &Paths,
            mut progress: impl FnMut(&str) -> ControlFlow<()>,
        ) -> Result<PathBuf> {
            let directory = self.directory(paths);
            if self.available(paths) {
                return Ok(directory);
            }
            ensure!(
                self != Self::W2vRu,
                "w2v_ru has no download: place config.json, preprocessor_config.json and model.safetensors in {}",
                directory.display()
            );
            fs::create_dir_all(&directory)?;
            let lock = OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(false)
                .open(directory.join(".download.lock"))?;
            lock.try_lock()
                .with_context(|| format!("another process is installing {}", self.name()))?;
            let remote = match self {
                Self::Mhubert147 => format!("{MHUBERT}/"),
                Self::HubertBase => HUBERT_BASE.to_owned(),
                _ => format!("{APPLIO}/{}/", self.name()),
            };
            // HuBERT Base reads its input normalization from preprocessor_config.json.
            let extra: &[&str] = if self == Self::HubertBase {
                &["preprocessor_config.json"]
            } else {
                &[]
            };
            let mut files = Vec::new();
            for &name in self.files().iter().chain(extra) {
                let destination = directory.join(name);
                if destination.is_file() {
                    continue;
                }
                let file = NamedTempFile::new_in(&directory)?;
                let label = format!("{}/{name}", self.name());
                crate::download::download(
                    &format!("{remote}{name}"),
                    file.path(),
                    &label,
                    None,
                    &mut progress,
                )?;
                validate(name, file.path())?;
                files.push((file, destination));
            }
            ensure!(
                progress("Installing embedder").is_continue(),
                "embedder download cancelled"
            );
            for (file, path) in files {
                // Existing files are user assets. Never replace them.
                file.persist_noclobber(&path)
                    .with_context(|| format!("installing {}", path.display()))?;
            }
            Ok(directory)
        }
    }

    fn validate(name: &str, path: &Path) -> Result<()> {
        match name {
            "pytorch_model.bin" => {
                let weights = Checkpoint::open(path)?;
                ensure!(
                    weights
                        .tensor_shape(None, "feature_projection.projection.weight")?
                        .first()
                        == Some(&768),
                    "downloaded checkpoint is not a 768-channel HuBERT model"
                );
            }
            "model.safetensors" => {
                // Validate the downloaded format before publishing. Tensors stay on CPU.
                let weights = tch::Tensor::read_safetensors(path)?;
                ensure!(
                    weights.iter().any(|(key, tensor)| key
                        .ends_with("feature_projection.projection.weight")
                        && tensor.size().first() == Some(&768)),
                    "downloaded checkpoint is not a 768-channel HuBERT model"
                );
            }
            _ => {
                let json: Value = serde_json::from_reader(File::open(path)?)?;
                ensure!(json.is_object(), "invalid embedder configuration");
            }
        }
        Ok(())
    }
}
