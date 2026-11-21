use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result, ensure};
use serde_json::{Value as Json, json};
use tch::{Device, Kind, Tensor};

use super::train_data::{Config, Version};
use crate::catalog::{Unvoiced, VoiceFormat};
use crate::checkpoint::{
    Checkpoint,
    write::{self, Key, Value},
};

pub fn describe(path: &Path, output: &Path, description: &str, overwrite: bool) -> Result<()> {
    tch::no_grad(|| {
        let mut checkpoint = Checkpoint::open(path)?;
        ensure!(
            checkpoint.has("weight"),
            "description editing requires a compact checkpoint"
        );
        let mut metadata = checkpoint.compact_metadata()?;
        metadata.insert("info".into(), json!(description));
        let state = checkpoint.weights(Some("weight"), Device::Cpu)?;
        save(output, metadata, state, overwrite)
    })
}

pub struct Export<'a> {
    pub version: Version,
    pub sample_rate: u32,
    pub uses_f0: bool,
    pub description: &'a str,
    pub overwrite: bool,
}

pub fn extract(path: &Path, output: &Path, options: Export<'_>) -> Result<()> {
    tch::no_grad(|| {
        let mut checkpoint = Checkpoint::open(path)?;
        let section = if checkpoint.has("model") {
            Some("model")
        } else {
            None
        };
        let config = Config::load(
            path.parent().unwrap_or_else(|| Path::new(".")),
            options.version,
            options.sample_rate,
        )?;
        let state = checkpoint.weights(section, Device::Cpu)?;
        compact(output, &config, state, options)
    })
}

pub fn merge(
    first: &Path,
    second: &Path,
    output: &Path,
    alpha: f64,
    description: &str,
    overwrite: bool,
) -> Result<()> {
    tch::no_grad(|| {
        let mut first = Checkpoint::open(first)?;
        let mut second = Checkpoint::open(second)?;
        let mut metadata = first.compact_metadata()?;
        ensure!(
            VoiceFormat::read(&first, None)? == VoiceFormat::read(&second, None)?,
            "model embedders, pitch or noise settings differ"
        );
        for key in ["config", "f0", "version"] {
            let one = first.metadata(key)?;
            let two = second.metadata(key)?;
            if key == "config" {
                let normalize = |mut value: Json| {
                    if let Some(list) = value.as_array_mut().filter(|list| list.len() == 18) {
                        list[15] = Json::Null;
                    }
                    value
                };
                ensure!(
                    one.map(normalize) == two.map(normalize),
                    "model architectures differ"
                );
            } else {
                ensure!(one == two, "models report different {key} values");
            }
        }
        if first.metadata("speaker_info")? != second.metadata("speaker_info")? {
            metadata.remove("speaker_info");
        }
        metadata.insert("info".into(), json!(description));
        let result = merged_weights(&mut first, &mut second, alpha)?;
        let size = result
            .get("emb_g.weight")
            .context("speaker embedding is missing")?
            .size()[0];
        if let Some(list) = metadata
            .get_mut("config")
            .and_then(Json::as_array_mut)
            .filter(|list| list.len() == 18)
        {
            list[15] = json!(size);
        }
        if let Some(list) = metadata
            .get_mut("speaker_info")
            .and_then(Json::as_array_mut)
        {
            list.retain(|speaker| {
                speaker
                    .get("id")
                    .and_then(Json::as_i64)
                    .is_some_and(|id| (0..size).contains(&id))
            });
        }
        save(output, metadata, result, overwrite)
    })
}

pub struct LegacyMerge<'a> {
    pub alpha: f64,
    pub sample_rate: &'a str,
    pub uses_f0: bool,
    pub description: &'a str,
    pub version: Version,
}

pub fn merge_legacy(
    first: &Path,
    second: &Path,
    output: &Path,
    options: LegacyMerge<'_>,
) -> Result<()> {
    tch::no_grad(|| {
        let mut first = Checkpoint::open(first)?;
        let mut second = Checkpoint::open(second)?;
        ensure!(
            VoiceFormat::read(&first, None)? == VoiceFormat::read(&second, None)?,
            "model embedders, pitch or noise settings differ"
        );
        let mut metadata = BTreeMap::from([
            (
                "config".into(),
                first
                    .metadata("config")?
                    .context("first checkpoint is missing config")?,
            ),
            ("sr".into(), json!(options.sample_rate)),
            ("f0".into(), json!(i64::from(options.uses_f0))),
            ("version".into(), json!(options.version.name())),
            ("info".into(), json!(options.description)),
        ]);
        for key in ["embedder_model", "nvc"] {
            if let Some(value) = first.metadata(key)? {
                metadata.insert(key.into(), value);
            }
        }
        let speakers = |file: &Checkpoint| -> Result<BTreeMap<i64, String>> {
            let mut speakers = BTreeMap::new();
            if let Some(Json::Array(entries)) = file.metadata("speaker_info")? {
                for entry in entries {
                    let id = &entry["id"];
                    if let Some(id) = id.as_i64().or_else(|| id.as_str()?.trim().parse().ok())
                        && (0..110).contains(&id)
                        && let Some(name) = entry["name"].as_str().filter(|name| !name.is_empty())
                    {
                        speakers.entry(id).or_insert_with(|| name.to_owned());
                    }
                }
            }
            Ok(speakers)
        };
        let names = speakers(&first)?;
        if !names.is_empty() && names == speakers(&second)? {
            let entries: Vec<_> = names
                .into_iter()
                .map(|(id, name)| json!({"id":id,"name":name}))
                .collect();
            metadata.insert("speaker_info".into(), json!(entries));
        }
        let weights = merged_weights(&mut first, &mut second, options.alpha)?;
        save(output, metadata, weights, true)
    })
}

fn merged_weights(
    first: &mut Checkpoint,
    second: &mut Checkpoint,
    alpha: f64,
) -> Result<BTreeMap<String, Tensor>> {
    ensure!(
        (0.0..=1.0).contains(&alpha),
        "model weight must be between 0 and 1"
    );
    let read = |file: &mut Checkpoint| {
        if file.has("model") {
            let mut state = file.weights(Some("model"), Device::Cpu)?;
            state.retain(|name, _| !name.contains("enc_q"));
            Ok::<_, anyhow::Error>(state)
        } else {
            file.weights(Some("weight"), Device::Cpu)
        }
    };
    let one = read(first)?;
    let mut two = read(second)?;
    ensure!(one.len() == two.len(), "model architectures differ");
    let mut result = BTreeMap::new();
    for (key, t1) in one {
        let t2 = two
            .remove(&key)
            .with_context(|| format!("second checkpoint has no {key}"))?;
        let (t1, t2) = if t1.size() == t2.size() {
            (t1, t2)
        } else {
            ensure!(
                key == "emb_g.weight"
                    && t1.dim() == 2
                    && t2.dim() == 2
                    && t1.size()[1] == t2.size()[1],
                "incompatible model tensor {key}"
            );
            let size = t1.size()[0].min(t2.size()[0]);
            (t1.narrow(0, 0, size), t2.narrow(0, 0, size))
        };
        result.insert(key, (t1 * alpha + t2 * (1.0 - alpha)).to_kind(Kind::Half));
    }
    Ok(result)
}

pub fn generator_config(config: &Config) -> Json {
    let m = &config.model;
    let d = &config.data;
    json!([
        d.filter_length / 2 + 1,
        32,
        m.inter_channels,
        m.hidden_channels,
        m.filter_channels,
        m.n_heads,
        m.n_layers,
        m.kernel_size,
        m.p_dropout,
        m.resblock,
        m.resblock_kernel_sizes,
        m.resblock_dilation_sizes,
        m.upsample_rates,
        m.upsample_initial_channel,
        m.upsample_kernel_sizes,
        m.spk_embed_dim,
        m.gin_channels,
        d.sampling_rate,
    ])
}

pub fn compact(
    output: &Path,
    config: &Config,
    state: impl IntoIterator<Item = (String, Tensor)>,
    options: Export<'_>,
) -> Result<()> {
    let rate = match options.sample_rate {
        32_000 => "32k",
        40_000 => "40k",
        48_000 => "48k",
        _ => anyhow::bail!("unsupported RVC sample rate"),
    };
    let mut metadata = BTreeMap::from([
        ("config".into(), generator_config(config)),
        ("f0".into(), json!(i64::from(options.uses_f0))),
        ("sr".into(), json!(rate)),
        ("info".into(), json!(options.description)),
        ("version".into(), json!(options.version.name())),
    ]);
    if !config.speaker_info.is_empty() {
        metadata.insert("speaker_info".into(), json!(config.speaker_info));
    }
    let embedder = config.embedder_model.unwrap_or_default();
    if config.embedder_model.is_some()
        || config.data.f0_unvoiced == Unvoiced::Zero
        || config.data.default_noise_scale != 0.66666
        || config.model.dec_aa
        || config.model.hf_bands
    {
        let augmentation = config
            .data
            .content_aug
            .as_ref()
            .filter(|options| options.prob > 0.0)
            .map(|options| {
                json!({
                    "prob": options.prob, "mix_range": options.mix_range, "kind": options.kind,
                })
            });
        metadata.insert("embedder_model".into(), json!(embedder));
        metadata.insert("nvc".into(), json!({
            "format": "nvc", "format_version": 1,
            "architecture": format!("rvc-{}-{}", options.version.name(), if options.uses_f0 { "nsf" } else { "nono" }),
            "embedder": embedder, "sample_rate": options.sample_rate,
            "hop_length": config.data.hop_length, "content_frame_rate": 50, "f0_frame_rate": 100,
            "f0_unvoiced": config.data.f0_unvoiced, "content_aug": augmentation,
            "default_noise_scale": config.data.default_noise_scale,
            "dec_aa": config.model.dec_aa, "hf_bands": config.model.hf_bands,
        }));
    }
    let state: BTreeMap<_, _> = state
        .into_iter()
        .filter(|(key, _)| !key.starts_with("enc_q."))
        .map(|(key, tensor)| (key, tensor.detach().to_kind(Kind::Half)))
        .collect();
    ensure!(
        state.contains_key("emb_g.weight"),
        "generator checkpoint has no speaker embedding"
    );
    save(output, metadata, state, options.overwrite)
}

fn save(
    path: &Path,
    metadata: BTreeMap<String, Json>,
    state: impl IntoIterator<Item = (String, Tensor)>,
    overwrite: bool,
) -> Result<()> {
    let mut fields: Vec<_> = metadata
        .into_iter()
        .map(|(key, value)| (Key::String(key), Value::Meta(value)))
        .collect();
    fields.push(("weight".into(), Value::tensors(state)));
    write::write(path, &Value::Dict(fields), overwrite)
}
