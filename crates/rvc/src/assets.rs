//! Weights fetched on first use.
//!
//! Studio, the CLI and the Colab notebook start without model files. Before a task loads a
//! model, it asks this module for the files it needs: a missing file is downloaded from its
//! known source with progress (`Downloading <file> · <percent>% · <done> / <total> MiB`),
//! checked against its SHA-256 and moved into place. Files the user already has are never
//! touched.

use std::fs::{self, File};
use std::io;
use std::ops::ControlFlow;
use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use sha2::{Digest, Sha256};
use tempfile::NamedTempFile;

use crate::catalog::{ModelInfo, Paths};
use crate::model::{PitchMethod, train};
use crate::separation;

struct Remote {
    url: &'static str,
    size: u64,
    sha256: &'static str,
}

const RMVPE: Remote = Remote {
    url: "https://huggingface.co/Politrees/RVC_resources/resolve/main/predictors/rmvpe.pt",
    size: 181184272,
    sha256: "6d62215f4306e3ca278246188607209f09af3dc77ed4232efdd069798c4ec193",
};
const NMVPE: Remote = Remote {
    url: "https://huggingface.co/kanoyo/nvc_resources/resolve/main/extractors/nmvpe.pt",
    size: 368448445,
    sha256: "8621a22d48590576bf8193ea78947414fdd956ab1b4e530a967722ff7791ec73",
};
/// FCPE ships inside the torchfcpe 0.0.4 wheel (MIT) on PyPI; the checkpoint is taken out of it.
const FCPE_WHEEL: Remote = Remote {
    url: "https://files.pythonhosted.org/packages/15/98/66da447498a2835b01cb3ae851e1333035a9fbf228e5c6f3e3c4351c06b6/torchfcpe-0.0.4-py3-none-any.whl",
    size: 40222012,
    sha256: "f042c463d850d76c6f4899a0b84f0b694bb560adf05f4de951097a756d17472d",
};
const FCPE_MEMBER: &str = "torchfcpe/assets/fcpe_c_v001.pt";
const FCPE_SHA256: &str = "b9aeaeb673436eeda50ceafd632aa681aa63417e52eae4207503d180c9b10015";
const FCPE_PATH: &str = "assets/fcpe/fcpe_c_v001.pt";
const MUTE: Remote = Remote {
    url: "https://github.com/kanoyo-git/rvc-rs/releases/download/resources/mute.zip",
    size: 312273,
    sha256: "ee948e85213e4ed2f2ba2f8dfcee810bfd0b63131d91450e920bbe1cbd0321d0",
};

/// Pretrained generators and discriminators by their path under the project root.
const PRETRAINED: &[(&str, Remote)] = &[
    (
        "assets/pretrained_v2/D32k.pth",
        Remote {
            url: "https://huggingface.co/Politrees/RVC_resources/resolve/main/pretrained/v2/32k/Default/D32k.pth",
            size: 142875703,
            sha256: "d8043378cc6619083d385f5a045de09b83fb3bf8de45c433ca863b71723ac3ca",
        },
    ),
    (
        "assets/pretrained_v2/D40k.pth",
        Remote {
            url: "https://huggingface.co/Politrees/RVC_resources/resolve/main/pretrained/v2/40k/Default/D40k.pth",
            size: 142875703,
            sha256: "471378e894e7191f89a94eda8288c5947b16bbe0b10c3f1f17efdb7a1d998242",
        },
    ),
    (
        "assets/pretrained_v2/D48k.pth",
        Remote {
            url: "https://huggingface.co/Politrees/RVC_resources/resolve/main/pretrained/v2/48k/Default/D48k.pth",
            size: 142875703,
            sha256: "db01094a93c09868a278e03dafe8bb781bfcc1a5ba8df168c948bf9168c84d82",
        },
    ),
    (
        "assets/pretrained_v2/G32k.pth",
        Remote {
            url: "https://huggingface.co/Politrees/RVC_resources/resolve/main/pretrained/v2/32k/Default/G32k.pth",
            size: 73811639,
            sha256: "869b26a47f75168d6126f64ac39e6de5247017a8658cfd68aca600f7323efb9f",
        },
    ),
    (
        "assets/pretrained_v2/G40k.pth",
        Remote {
            url: "https://huggingface.co/Politrees/RVC_resources/resolve/main/pretrained/v2/40k/Default/G40k.pth",
            size: 72959671,
            sha256: "a3843da7fde33db1dab176146c70d6c2df06eafe9457f4e3aa10024e9c6a4b69",
        },
    ),
    (
        "assets/pretrained_v2/G48k.pth",
        Remote {
            url: "https://huggingface.co/Politrees/RVC_resources/resolve/main/pretrained/v2/48k/Default/G48k.pth",
            size: 75318967,
            sha256: "2e2b1581a436d07a76b10b9d38765f64aa02836dc65c7dee1ce4140c11ea158b",
        },
    ),
    (
        "assets/pretrained_v2/f0D32k.pth",
        Remote {
            url: "https://huggingface.co/Politrees/RVC_resources/resolve/main/pretrained/v2/32k/Default/f0D32k.pth",
            size: 142875703,
            sha256: "bd7134e7793674c85474d5145d2d982e3c5d8124fc7bb6c20f710ed65808fa8a",
        },
    ),
    (
        "assets/pretrained_v2/f0D40k.pth",
        Remote {
            url: "https://huggingface.co/Politrees/RVC_resources/resolve/main/pretrained/v2/40k/Default/f0D40k.pth",
            size: 142875703,
            sha256: "6b6ab091e70801b28e3f41f335f2fc5f3f35c75b39ae2628d419644ec2b0fa09",
        },
    ),
    (
        "assets/pretrained_v2/f0D48k.pth",
        Remote {
            url: "https://huggingface.co/Politrees/RVC_resources/resolve/main/pretrained/v2/48k/Default/f0D48k.pth",
            size: 142875703,
            sha256: "2269b73c7a4cf34da09aea99274dabf99b2ddb8a42cbfb065fb3c0aa9a2fc748",
        },
    ),
    (
        "assets/pretrained_v2/f0G32k.pth",
        Remote {
            url: "https://huggingface.co/Politrees/RVC_resources/resolve/main/pretrained/v2/32k/Default/f0G32k.pth",
            size: 73950049,
            sha256: "2332611297b8d88c7436de8f17ef5f07a2119353e962cd93cda5806d59a1133d",
        },
    ),
    (
        "assets/pretrained_v2/f0G40k.pth",
        Remote {
            url: "https://huggingface.co/Politrees/RVC_resources/resolve/main/pretrained/v2/40k/Default/f0G40k.pth",
            size: 73106273,
            sha256: "3b2c44035e782c4b14ddc0bede9e2f4a724d025cd073f736d4f43708453adfcb",
        },
    ),
    (
        "assets/pretrained_v2/f0G48k.pth",
        Remote {
            url: "https://huggingface.co/Politrees/RVC_resources/resolve/main/pretrained/v2/48k/Default/f0G48k.pth",
            size: 75465569,
            sha256: "b5d51f589cc3632d4eae36a315b4179397695042edc01d15312e1bddc2b764a4",
        },
    ),
    (
        "assets/pretrained/D32k.pth",
        Remote {
            url: "https://huggingface.co/Politrees/RVC_resources/resolve/main/pretrained/v1/32k/D32k.pth",
            size: 109978943,
            sha256: "2ab20645829460fdad0d3c44254f1ab53c32cae50c22a66c926ae5aa30abda6f",
        },
    ),
    (
        "assets/pretrained/D40k.pth",
        Remote {
            url: "https://huggingface.co/Politrees/RVC_resources/resolve/main/pretrained/v1/40k/D40k.pth",
            size: 109978943,
            sha256: "547f66dbbcd9023b9051ed244d12ab043ba8a4e854b154cc28761ac7c002909b",
        },
    ),
    (
        "assets/pretrained/D48k.pth",
        Remote {
            url: "https://huggingface.co/Politrees/RVC_resources/resolve/main/pretrained/v1/48k/D48k.pth",
            size: 109978943,
            sha256: "8cc013fa60ed9c3f902f5bd99f48c7e3b9352d763d4d3cd6bc241c37b0bfd9ad",
        },
    ),
    (
        "assets/pretrained/G32k.pth",
        Remote {
            url: "https://huggingface.co/Politrees/RVC_resources/resolve/main/pretrained/v1/32k/G32k.pth",
            size: 72653893,
            sha256: "81817645cde7ed2e2d83f23ef883f33dda564924b497e84d792743912eca4c23",
        },
    ),
    (
        "assets/pretrained/G40k.pth",
        Remote {
            url: "https://huggingface.co/Politrees/RVC_resources/resolve/main/pretrained/v1/40k/G40k.pth",
            size: 72763063,
            sha256: "e428573bda1124b0ae0ae843fd8dcded6027d3993444790b3e9b0100938b2113",
        },
    ),
    (
        "assets/pretrained/G48k.pth",
        Remote {
            url: "https://huggingface.co/Politrees/RVC_resources/resolve/main/pretrained/v1/48k/G48k.pth",
            size: 72850501,
            sha256: "3862a67ea6313e8ffefc05cee6bee656ef3e089442e9ecf4a6618d60721f3e95",
        },
    ),
    (
        "assets/pretrained/f0D32k.pth",
        Remote {
            url: "https://huggingface.co/Politrees/RVC_resources/resolve/main/pretrained/v1/32k/f0D32k.pth",
            size: 109978943,
            sha256: "294db3087236e2c75260d6179056791c9231245daf5d0485545d9e54c4057c77",
        },
    ),
    (
        "assets/pretrained/f0D40k.pth",
        Remote {
            url: "https://huggingface.co/Politrees/RVC_resources/resolve/main/pretrained/v1/40k/f0D40k.pth",
            size: 109978943,
            sha256: "7d4f5a441594b470d67579958b2fd4c6b992852ded28ff9e72eda67abcebe423",
        },
    ),
    (
        "assets/pretrained/f0D48k.pth",
        Remote {
            url: "https://huggingface.co/Politrees/RVC_resources/resolve/main/pretrained/v1/48k/f0D48k.pth",
            size: 109978943,
            sha256: "1b84c8bf347ad1e539c842e8f2a4c36ecd9e7fb23c16041189e4877e9b07925c",
        },
    ),
    (
        "assets/pretrained/f0G32k.pth",
        Remote {
            url: "https://huggingface.co/Politrees/RVC_resources/resolve/main/pretrained/v1/32k/f0G32k.pth",
            size: 72795627,
            sha256: "285f524bf48bb692c76ad7bd0bc654c12bd9e5edeb784dddf7f61a789a608574",
        },
    ),
    (
        "assets/pretrained/f0G40k.pth",
        Remote {
            url: "https://huggingface.co/Politrees/RVC_resources/resolve/main/pretrained/v1/40k/f0G40k.pth",
            size: 72909665,
            sha256: "9115654aeef1995f7dd3c6fc4140bebbef0ca9760bed798105a2380a34299831",
        },
    ),
    (
        "assets/pretrained/f0G48k.pth",
        Remote {
            url: "https://huggingface.co/Politrees/RVC_resources/resolve/main/pretrained/v1/48k/f0G48k.pth",
            size: 73008619,
            sha256: "78bc9cab27e34bcfc194f93029374d871d8b3e663ddedea32a9709e894cc8fe8",
        },
    ),
    (
        "assets/pretrained_v2/NVC-Pretrained/f0G40k.pth",
        Remote {
            url: "https://huggingface.co/kanoyo/NVC-Pretrained/resolve/main/f0G40k.pth",
            size: 73103601,
            sha256: "c225a44605e49dd504bd08ea6c9507188ac7d00d9474e8df2926e1bd27e59b2b",
        },
    ),
    (
        "assets/pretrained_v2/NVC-Pretrained/f0D40k.pth",
        Remote {
            url: "https://huggingface.co/kanoyo/NVC-Pretrained/resolve/main/f0D40k.pth",
            size: 142875513,
            sha256: "2c66d29628f30d52463ed7952229cec65a4bbf90327181853f2dc15fc95d5513",
        },
    ),
];

/// Separation checkpoints by file name.
const SEPARATION: &[(&str, Remote)] = &[
    (
        "dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt",
        Remote {
            url: "https://github.com/kanoyo-git/rvc-rs/releases/download/resources/dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt",
            size: 456706759,
            sha256: "4fcbfc9d419abfd1f07a24f920d83f80d51727ca3cde3e9d84c99398418d35f3",
        },
    ),
    (
        "dereverb_mel_band_roformer_less_aggressive_anvuew_sdr_18.8050.ckpt",
        Remote {
            url: "https://github.com/kanoyo-git/rvc-rs/releases/download/resources/dereverb_mel_band_roformer_less_aggressive_anvuew_sdr_18.8050.ckpt",
            size: 456754215,
            sha256: "9399701da33d179e328a436e882fe2c79ae2f118a3d49603dbc7210bfeb428ea",
        },
    ),
    (
        "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
        Remote {
            url: "https://github.com/kanoyo-git/rvc-rs/releases/download/resources/model_bs_roformer_ep_317_sdr_12.9755.ckpt",
            size: 319821424,
            sha256: "4deed31719b7b92195530a9d0e86ea1c8d5c4aa9f0e2dd967c1f05419ebf96d5",
        },
    ),
    (
        "model_bs_roformer_ep_368_sdr_12.9628.ckpt",
        Remote {
            url: "https://github.com/kanoyo-git/rvc-rs/releases/download/resources/model_bs_roformer_ep_368_sdr_12.9628.ckpt",
            size: 319821424,
            sha256: "97c9e1011924ff647d147b474638c161b80586fd5b9e83fab59cbbb70014bf62",
        },
    ),
    (
        "model_mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt",
        Remote {
            url: "https://github.com/kanoyo-git/rvc-rs/releases/download/resources/model_mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt",
            size: 456715871,
            sha256: "6603c56e16e85f0eab78d471e43341b789ddc654230b9bcc81a25a8d1457a8c1",
        },
    ),
    (
        "bs_roformer_ft1_anvuew_sdr_12.55.ckpt",
        Remote {
            url: "https://huggingface.co/anvuew/BS-RoFormer/resolve/main/bs_roformer_ft1_anvuew_sdr_12.55.ckpt",
            size: 204493312,
            sha256: "60347271e8493fdff28ef558c3b2297afb869a1f4462594ed548038264bec395",
        },
    ),
    (
        "karaoke_bs_roformer_anvuew.ckpt",
        Remote {
            url: "https://huggingface.co/anvuew/karaoke_bs_roformer/resolve/main/karaoke_bs_roformer_anvuew.ckpt",
            size: 204486925,
            sha256: "206d04757cb5f75ca3b55f8a0a48f5c26aa2351d4ff3c7adbfc9affa30ea3ae4",
        },
    ),
    (
        "dereverb_bs_roformer_anvuew_sdr_22.5050.ckpt",
        Remote {
            url: "https://huggingface.co/anvuew/dereverb_bs_roformer/resolve/main/dereverb_bs_roformer_anvuew_sdr_22.5050.ckpt",
            size: 204496717,
            sha256: "b4dfa4c4aa251c0afc303e5d512d904a1c2255e51d8d0dbfbddb18d740563d42",
        },
    ),
];

/// Separation configs that are not shipped with the project.
const SEPARATION_CONFIGS: &[&str] = &[
    "config_mel_band_roformer_karaoke.yaml",
    "dereverb_mel_band_roformer_anvuew.yaml",
    "model_bs_roformer_ep_317_sdr_12.9755.yaml",
    "model_bs_roformer_ep_368_sdr_12.9628.yaml",
];

/// A file a task needs and does not have yet, for the download prompt.
#[derive(serde::Serialize)]
pub struct Missing {
    pub name: String,
    /// Bytes to download, when known.
    pub size: Option<u64>,
    /// No public source: the user has to place it.
    pub manual: bool,
}

impl Missing {
    fn remote(name: &str, remote: &Remote) -> Self {
        Self {
            name: name.to_owned(),
            size: Some(remote.size),
            manual: false,
        }
    }
}

/// What a task needs, as Studio describes it before starting the task.
#[derive(serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Need {
    /// A voice model's embedder and, for pitch-guided voices, the pitch method.
    Voice {
        model: std::path::PathBuf,
        f0_method: PitchMethod,
    },
    Pitch {
        method: PitchMethod,
    },
    Embedder {
        embedder: crate::catalog::Embedder,
    },
    Separation {
        preset: String,
    },
    Mute,
    Training {
        #[serde(flatten)]
        request: train::Request,
    },
}

/// Sizes of embedders whose files are known.
fn embedder_size(embedder: crate::catalog::Embedder) -> Option<u64> {
    use crate::catalog::Embedder;
    match embedder {
        Embedder::HubertBase => Some(189_206_711),
        Embedder::Mhubert147 => Some(377_510_584),
        _ => None,
    }
}

fn plan_pitch(paths: &Paths, method: PitchMethod, out: &mut Vec<Missing>) {
    if method.available(paths) {
        return;
    }
    match method {
        PitchMethod::Rmvpe => out.push(Missing::remote("rmvpe.pt", &RMVPE)),
        PitchMethod::Nmvpe => out.push(Missing::remote("nmvpe.pt", &NMVPE)),
        PitchMethod::Fcpe => out.push(Missing::remote("fcpe_c_v001.pt", &FCPE_WHEEL)),
        PitchMethod::Pm => {}
    }
}

fn plan_embedder(paths: &Paths, embedder: crate::catalog::Embedder, out: &mut Vec<Missing>) {
    if !embedder.available(paths) {
        out.push(Missing {
            name: embedder.name().to_owned(),
            size: embedder_size(embedder),
            manual: embedder == crate::catalog::Embedder::W2vRu,
        });
    }
}

fn plan_separation(paths: &Paths, preset: &str, out: &mut Vec<Missing>) -> Result<()> {
    let preset = separation::PRESETS
        .iter()
        .find(|item| item.id == preset || item.label == preset)
        .with_context(|| format!("unknown separation preset {preset}"))?;
    let directory = separation::weights_directory(&paths.root);
    if !directory.join(preset.checkpoint).is_file() {
        out.push(
            match SEPARATION
                .iter()
                .find(|(name, _)| *name == preset.checkpoint)
            {
                Some((name, remote)) => Missing::remote(name, remote),
                None => Missing {
                    name: preset.checkpoint.into(),
                    size: None,
                    manual: true,
                },
            },
        );
    }
    if !directory.join(preset.config).is_file() {
        out.push(Missing {
            name: preset.config.into(),
            size: None,
            manual: !SEPARATION_CONFIGS.contains(&preset.config),
        });
    }
    Ok(())
}

fn training_paths(paths: &Paths, request: &train::Request) -> Vec<std::path::PathBuf> {
    let rate = request.sample_rate / 1000;
    let prefix = if request.f0 { "f0" } else { "" };
    let directory = match request.version {
        crate::model::train_data::Version::V2 => "pretrained_v2",
        crate::model::train_data::Version::V1 => "pretrained",
    };
    [("G", &request.pretrained_g), ("D", &request.pretrained_d)]
        .into_iter()
        .filter_map(|(kind, custom)| match custom {
            Some(path) if !path.as_os_str().is_empty() => Some(paths.root.join(path)),
            Some(_) => None,
            None if request.random_weights => None,
            None => Some(
                paths
                    .root
                    .join("assets")
                    .join(directory)
                    .join(format!("{prefix}{kind}{rate}k.pth")),
            ),
        })
        .collect()
}

fn known(paths: &Paths, path: &Path) -> Option<&'static Remote> {
    let absolute = std::path::absolute(path).ok()?;
    let root = std::path::absolute(&paths.root).ok()?;
    let key = absolute
        .strip_prefix(&root)
        .ok()?
        .to_string_lossy()
        .replace('\\', "/");
    PRETRAINED
        .iter()
        .find(|(name, _)| *name == key)
        .map(|(_, remote)| remote)
}

/// Files the given needs lack. Unknown custom files are left to the task to report.
pub fn missing(paths: &Paths, needs: &[Need]) -> Result<Vec<Missing>> {
    let mut out = Vec::new();
    for need in needs {
        match need {
            Need::Voice { model, f0_method } => {
                let info = ModelInfo::read(model)?;
                plan_embedder(paths, info.format.embedder, &mut out);
                if info.uses_f0 {
                    plan_pitch(paths, *f0_method, &mut out);
                }
            }
            Need::Pitch { method } => plan_pitch(paths, *method, &mut out),
            Need::Embedder { embedder } => plan_embedder(paths, *embedder, &mut out),
            Need::Separation { preset } => plan_separation(paths, preset, &mut out)?,
            Need::Mute => {
                if !paths.experiments.join("mute").join("0_gt_wavs").is_dir() {
                    out.push(Missing::remote("mute.zip", &MUTE));
                }
            }
            Need::Training { request } => {
                if !paths.experiments.join("mute").join("0_gt_wavs").is_dir() {
                    out.push(Missing::remote("mute.zip", &MUTE));
                }
                for path in training_paths(paths, request) {
                    if !path.is_file()
                        && let Some(remote) = known(paths, &path)
                    {
                        out.push(Missing::remote(&label(&path), remote));
                    }
                }
            }
        }
    }
    let mut seen = std::collections::HashSet::new();
    out.retain(|item| seen.insert(item.name.clone()));
    Ok(out)
}

/// Download everything `needs` lacks.
pub fn fetch_all(
    paths: &Paths,
    needs: &[Need],
    report: &mut impl FnMut(&str) -> ControlFlow<()>,
) -> Result<()> {
    for need in needs {
        match need {
            Need::Voice { model, f0_method } => voice(paths, model, *f0_method, report)?,
            Need::Pitch { method } => pitch(paths, *method, report)?,
            Need::Embedder { embedder } => {
                embedder.install(paths, &mut *report)?;
            }
            Need::Separation { preset } => separation(paths, preset, report)?,
            Need::Mute => mute(paths, report)?,
            Need::Training { request } => training(paths, request, report)?,
        }
    }
    Ok(())
}

fn label(path: &Path) -> String {
    path.file_name()
        .map_or_else(String::new, |name| name.to_string_lossy().into_owned())
}

/// Download `remote` into a temporary file next to `directory` and check its digest.
fn download(
    directory: &Path,
    name: &str,
    remote: &Remote,
    report: &mut impl FnMut(&str) -> ControlFlow<()>,
) -> Result<NamedTempFile> {
    fs::create_dir_all(directory)?;
    let partial = NamedTempFile::new_in(directory)?;
    crate::download::download(remote.url, partial.path(), name, Some(remote.size), report)?;
    ensure!(
        report(&format!("Checking {name}")).is_continue(),
        "download cancelled"
    );
    let mut hasher = Sha256::new();
    io::copy(&mut File::open(partial.path())?, &mut hasher)?;
    let digest = format!("{:x}", hasher.finalize());
    ensure!(
        digest == remote.sha256,
        "{name} arrived damaged (SHA-256 mismatch), run the task again to retry"
    );
    Ok(partial)
}

fn fetch(
    path: &Path,
    remote: &Remote,
    report: &mut impl FnMut(&str) -> ControlFlow<()>,
) -> Result<()> {
    if path.is_file() {
        return Ok(());
    }
    let directory = path.parent().context("model path has no folder")?;
    let partial = download(directory, &label(path), remote, report)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(partial.path(), fs::Permissions::from_mode(0o644))?;
    }
    match partial.persist_noclobber(path) {
        Ok(_) => Ok(()),
        // Another task finished the same file first.
        Err(_) if path.is_file() => Ok(()),
        Err(error) => Err(error.error).with_context(|| format!("installing {}", path.display())),
    }
}

/// Weights of a pitch method.
pub fn pitch(
    paths: &Paths,
    method: PitchMethod,
    report: &mut impl FnMut(&str) -> ControlFlow<()>,
) -> Result<()> {
    if method.available(paths) {
        return Ok(());
    }
    match method {
        PitchMethod::Rmvpe => fetch(&paths.rmvpe.join("rmvpe.pt"), &RMVPE, report),
        PitchMethod::Nmvpe => fetch(&paths.nmvpe(), &NMVPE, report),
        PitchMethod::Fcpe => fcpe(paths, report),
        PitchMethod::Pm => Ok(()),
    }
}

/// `fcpe_c_v001.pt`, unpacked from the torchfcpe wheel and checked.
fn fcpe(paths: &Paths, report: &mut impl FnMut(&str) -> ControlFlow<()>) -> Result<()> {
    let target = paths.root.join(FCPE_PATH);
    if target.is_file() {
        return Ok(());
    }
    let directory = target.parent().context("FCPE path has no folder")?;
    let wheel = download(directory, "torchfcpe-0.0.4.whl", &FCPE_WHEEL, report)?;
    let mut archive =
        zip::ZipArchive::new(File::open(wheel.path())?).context("opening the torchfcpe wheel")?;
    let mut member = archive
        .by_name(FCPE_MEMBER)
        .context("the torchfcpe wheel has no fcpe_c_v001.pt")?;
    let partial = NamedTempFile::new_in(directory)?;
    let mut hasher = Sha256::new();
    {
        let mut output = io::BufWriter::new(File::create(partial.path())?);
        let mut buffer = vec![0; 1 << 20];
        loop {
            let read = io::Read::read(&mut member, &mut buffer)?;
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
            io::Write::write_all(&mut output, &buffer[..read])?;
        }
        io::Write::flush(&mut output)?;
    }
    ensure!(
        format!("{:x}", hasher.finalize()) == FCPE_SHA256,
        "fcpe_c_v001.pt from the torchfcpe wheel does not match its checksum"
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(partial.path(), fs::Permissions::from_mode(0o644))?;
    }
    match partial.persist_noclobber(&target) {
        Ok(_) => Ok(()),
        Err(_) if target.is_file() => Ok(()),
        Err(error) => Err(error.error).with_context(|| format!("installing {}", target.display())),
    }
}

/// Everything converting with `model` needs: its content embedder and, for a pitch-guided
/// voice, the weights of `method`.
pub fn voice(
    paths: &Paths,
    model: &Path,
    method: PitchMethod,
    report: &mut impl FnMut(&str) -> ControlFlow<()>,
) -> Result<()> {
    let info = ModelInfo::read(model)?;
    info.format.embedder.install(paths, &mut *report)?;
    if info.uses_f0 {
        pitch(paths, method, report)?;
    }
    Ok(())
}

/// Checkpoint and config of a separation preset.
pub fn separation(
    paths: &Paths,
    preset: &str,
    report: &mut impl FnMut(&str) -> ControlFlow<()>,
) -> Result<()> {
    let preset = separation::PRESETS
        .iter()
        .find(|item| item.id == preset || item.label == preset)
        .with_context(|| format!("unknown separation preset {preset}"))?;
    let directory = separation::weights_directory(&paths.root);
    let checkpoint = directory.join(preset.checkpoint);
    if !checkpoint.is_file() {
        let (_, remote) = SEPARATION
            .iter()
            .find(|(name, _)| *name == preset.checkpoint)
            .with_context(|| format!("place {} in {}", preset.checkpoint, directory.display()))?;
        fetch(&checkpoint, remote, report)?;
    }
    let config = directory.join(preset.config);
    if !config.is_file() {
        ensure!(
            SEPARATION_CONFIGS.contains(&preset.config),
            "place {} in {}",
            preset.config,
            directory.display()
        );
        let partial = NamedTempFile::new_in(&directory)?;
        let url = format!(
            "https://github.com/kanoyo-git/rvc-rs/releases/download/resources/{}",
            preset.config
        );
        crate::download::download(&url, partial.path(), preset.config, None, report)?;
        if partial.persist_noclobber(&config).is_err() && !config.is_file() {
            bail!("installing {}", config.display());
        }
    }
    Ok(())
}

/// A pretrained checkpoint at one of the known paths (RVC v1/v2 and NVC-Pretrained).
/// Other paths are left to training, which reports a missing file itself.
pub fn pretrained(
    paths: &Paths,
    path: &Path,
    report: &mut impl FnMut(&str) -> ControlFlow<()>,
) -> Result<()> {
    match known(paths, path) {
        Some(remote) if !path.is_file() => fetch(path, remote, report),
        _ => Ok(()),
    }
}

/// Paths of every pretrained checkpoint this module can download.
pub fn known_pretrained(paths: &Paths) -> Vec<std::path::PathBuf> {
    PRETRAINED
        .iter()
        .map(|(name, _)| paths.root.join(name))
        .collect()
}

/// The silent reference clip training mixes into every dataset (`logs/mute`).
pub fn mute(paths: &Paths, report: &mut impl FnMut(&str) -> ControlFlow<()>) -> Result<()> {
    let target = paths.experiments.join("mute");
    if target.join("0_gt_wavs").is_dir() {
        return Ok(());
    }
    let archive = download(&paths.experiments, "mute.zip", &MUTE, report)?;
    let staging = tempfile::TempDir::new_in(&paths.experiments)?;
    zip::ZipArchive::new(File::open(archive.path())?)
        .context("opening mute.zip")?
        .extract(staging.path())
        .context("extracting mute.zip")?;
    if !target.join("0_gt_wavs").is_dir() {
        if target.exists() {
            fs::remove_dir_all(&target)?;
        }
        fs::rename(staging.path().join("mute"), &target)?;
    }
    Ok(())
}

/// What a training run needs before it starts: the mute clip and the pretrained pair.
pub fn training(
    paths: &Paths,
    request: &train::Request,
    report: &mut impl FnMut(&str) -> ControlFlow<()>,
) -> Result<()> {
    mute(paths, report)?;
    for path in training_paths(paths, request) {
        pretrained(paths, &path, report)?;
    }
    Ok(())
}

/// Progress for the command line: one line per change of the whole percent.
pub fn stderr() -> impl FnMut(&str) -> ControlFlow<()> {
    let mut last = String::new();
    move |line: &str| {
        let shown = line.split(" · ").take(2).collect::<Vec<_>>().join(" · ");
        if shown != last {
            eprintln!("{line}");
            last = shown;
        }
        ControlFlow::Continue(())
    }
}
