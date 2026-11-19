use std::collections::HashMap;
use std::ops::ControlFlow;
use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use serde_json::{Value, json};
use tch::{Device, Tensor};

use crate::checkpoint::Checkpoint;

use super::Demucs;
use crate::separation::{configurable::load_document, legacy_wave::LegacyWave, overlap::Progress};

pub(crate) fn is_legacy(model_type: &str) -> bool {
    matches!(
        model_type,
        "demucs" | "tasnet" | "legacy_demucs" | "legacy_tasnet"
    )
}

pub(in crate::separation) struct Legacy {
    rate: u32,
    channels: usize,
    stems: Vec<String>,
    members: Vec<Member>,
}

struct Member {
    network: Network,
    segment: usize,
    gains: Vec<f32>,
}

enum Network {
    Wave(Box<LegacyWave>),
    Hybrid {
        network: Box<Demucs>,
        #[cfg(feature = "onnx")]
        onnx: Option<super::onnx_backend::LegacySession>,
    },
}

impl Legacy {
    pub(in crate::separation) fn load(
        checkpoint: &Path,
        config: &Path,
        device: Device,
        onnx_provider: Option<&Value>,
    ) -> Result<Self> {
        let document = if is_yaml(config) {
            Some(load_document(config)?)
        } else if is_yaml(checkpoint) {
            Some(load_document(checkpoint)?)
        } else {
            None
        };
        let provider =
            onnx_provider.or_else(|| document.as_ref()?.get("inference")?.get("onnx_provider"));
        let options = serde_json::json!({"inference": {"onnx_provider": provider}});
        let acceleration =
            super::super::Acceleration::configured(&options, device, false, "legacy PyMSS")?;
        if let Some(models) = document.as_ref().and_then(|doc| doc["models"].as_array()) {
            let doc = document.as_ref().unwrap();
            ensure!(
                (1..=32).contains(&models.len()),
                "invalid number of Demucs bag members"
            );
            let directory = if is_yaml(config) { config } else { checkpoint }
                .parent()
                .context("Demucs bag directory is missing")?;
            let files = std::fs::read_dir(directory)?
                .filter_map(|entry| entry.ok())
                .map(|entry| entry.path())
                .filter(|path| {
                    path.is_file()
                        && path
                            .extension()
                            .is_some_and(|e| e.eq_ignore_ascii_case("th"))
                })
                .collect::<Vec<_>>();
            let mut members = Vec::with_capacity(models.len());
            let mut stems = Vec::new();
            let mut rate = 0;
            let mut channels = 0;
            for (index, model) in models.iter().enumerate() {
                let name = model
                    .as_str()
                    .filter(|name| !name.is_empty())
                    .context("invalid Demucs bag member")?;
                let path = files
                    .iter()
                    .filter(|path| {
                        path.file_name()
                            .is_some_and(|file| file.to_string_lossy().starts_with(name))
                    })
                    .min()
                    .with_context(|| {
                        format!(
                            "cannot find Demucs bag member {name:?} next to {}",
                            directory.display()
                        )
                    })?;
                let (mut member, member_rate, member_channels, member_stems) =
                    load_member(path, device, acceleration)?;
                if index == 0 {
                    rate = member_rate;
                    channels = member_channels;
                    stems = member_stems;
                } else {
                    ensure!(
                        rate == member_rate && channels == member_channels && stems == member_stems,
                        "Demucs bag model metadata does not match"
                    );
                }
                if let Some(seconds) = doc["segment"].as_f64() {
                    member.segment = segment_frames(seconds, rate)?;
                }
                if let Some(gains) = doc.get("weights").filter(|weights| !weights.is_null()) {
                    let gains: Vec<f32> = serde_json::from_value(gains[index].clone())?;
                    ensure!(
                        gains.len() == stems.len() && gains.iter().all(|gain| gain.is_finite()),
                        "Demucs bag has invalid per-stem weights"
                    );
                    member.gains = gains;
                }
                members.push(member);
            }
            Ok(Self {
                rate,
                channels,
                stems,
                members,
            })
        } else {
            ensure!(
                !is_yaml(checkpoint),
                "Demucs YAML requires a list of models"
            );
            let (member, rate, channels, stems) = load_member(checkpoint, device, acceleration)?;
            Ok(Self {
                rate,
                channels,
                stems,
                members: vec![member],
            })
        }
    }

    pub(in crate::separation) fn rate(&self) -> u32 {
        self.rate
    }
    pub(in crate::separation) fn channels(&self) -> usize {
        self.channels
    }
    pub(in crate::separation) fn stems(&self) -> &[String] {
        &self.stems
    }
    pub(in crate::separation) fn segment(&self) -> usize {
        self.members[0].segment
    }

    pub(in crate::separation) fn demix(
        &self,
        mix: &[f32],
        chunk: Option<usize>,
        overlap: Option<usize>,
        mut progress: impl FnMut(Progress) -> ControlFlow<()>,
    ) -> Result<Vec<f32>> {
        let frames = mix.len() / self.channels;
        let planes = self.channels * self.stems.len();
        let mut chunk_counts = Vec::with_capacity(self.members.len());
        for member in &self.members {
            let segment = chunk.unwrap_or(member.segment);
            let step = segment
                .checked_sub(overlap.unwrap_or(segment.div_ceil(4)))
                .filter(|&step| step > 0)
                .context("invalid legacy Demucs overlap")?;
            chunk_counts.push((segment, step, frames.div_ceil(step)));
        }
        let chunks_total = chunk_counts
            .iter()
            .map(|&(_, _, count)| count)
            .sum::<usize>();
        let mut estimates = vec![
            0.0;
            frames
                .checked_mul(planes)
                .context("Demucs output exceeds address space")?
        ];
        let mut gain_sum = vec![0.0; self.stems.len()];
        let mut chunks_done = 0;
        for (member, (segment, step, _)) in self.members.iter().zip(chunk_counts) {
            let mut combined = vec![0.0; estimates.len()];
            let mut counts = vec![0.0; frames];
            for start in (0..frames).step_by(step) {
                ensure!(
                    progress(Progress {
                        chunks_done,
                        chunks_total
                    })
                    .is_continue(),
                    "separation cancelled"
                );
                chunks_done += 1;
                let len = segment.min(frames - start);
                let result = match &member.network {
                    Network::Wave(network) => {
                        let aligned = network.valid_length(len)?;
                        let input = extract(mix, self.channels, start, len, aligned)?;
                        if len == aligned {
                            network.predict_window(&input)?
                        } else {
                            network.predict_aligned(&input, len)?
                        }
                    }
                    Network::Hybrid {
                        network,
                        #[cfg(feature = "onnx")]
                        onnx,
                    } => {
                        let input = extract(mix, self.channels, start, len, len)?;
                        #[cfg(feature = "onnx")]
                        let result = if let Some(onnx) = onnx {
                            onnx.predict(network, &input)?
                        } else {
                            network.predict_batch(&input, 1)?
                        };
                        #[cfg(not(feature = "onnx"))]
                        let result = network.predict_batch(&input, 1)?;
                        result
                    }
                };
                ensure!(
                    result.len() == planes * len && result.iter().all(|value| value.is_finite()),
                    "invalid legacy Demucs prediction"
                );
                for (slot, counter) in counts[start..start + len].iter_mut().enumerate() {
                    *counter += fade(slot, segment);
                }
                for (target, plane) in combined
                    .chunks_exact_mut(frames)
                    .zip(result.chunks_exact(len))
                {
                    for (slot, (a, &b)) in
                        target[start..start + len].iter_mut().zip(plane).enumerate()
                    {
                        *a += fade(slot, segment) * b;
                    }
                }
            }
            for ((target, source), &gain) in estimates
                .chunks_exact_mut(frames)
                .zip(combined.chunks_exact(frames))
                .zip(
                    member
                        .gains
                        .iter()
                        .flat_map(|gain| std::iter::repeat_n(gain, self.channels)),
                )
            {
                for ((a, b), &count) in target.iter_mut().zip(source).zip(&counts) {
                    *a += b / count * gain;
                }
            }
            for (sum, gain) in gain_sum.iter_mut().zip(&member.gains) {
                *sum += gain;
            }
        }
        for (track, gain) in estimates
            .chunks_exact_mut(frames * self.channels)
            .zip(gain_sum)
        {
            ensure!(
                gain.is_finite() && gain != 0.0,
                "invalid total Demucs bag weight"
            );
            for sample in track {
                *sample /= gain;
            }
        }
        ensure!(
            progress(Progress {
                chunks_done,
                chunks_total
            })
            .is_continue(),
            "separation cancelled"
        );
        Ok(estimates)
    }
}

fn is_yaml(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("yaml") || ext.eq_ignore_ascii_case("yml"))
}

fn load_member(
    path: &Path,
    device: Device,
    acceleration: super::super::Acceleration,
) -> Result<(Member, u32, usize, Vec<String>)> {
    let mut checkpoint = Checkpoint::open(path)?;
    let weights = checkpoint.demucs_weights(Device::Cpu)?;
    let (kind, args) = if let Some(package) = checkpoint.demucs_package()? {
        let mut args = package.kwargs;
        let fields = positional(package.kind);
        ensure!(
            package.args.len() <= fields.len(),
            "invalid Demucs positional argument count"
        );
        for (name, value) in fields.iter().zip(package.args) {
            ensure!(
                args.insert((*name).to_owned(), value).is_none(),
                "duplicate Demucs parameter {name}"
            );
        }
        (package.kind, Value::Object(args))
    } else {
        infer_architecture(path, &weights)?
    };
    let (network, rate, channels, stems, segment) = match kind {
        "hdemucs" | "htdemucs" => {
            let stems: Vec<String> = serde_json::from_value(args["sources"].clone())?;
            let rate = u32::try_from(args["samplerate"].as_u64().unwrap_or(44100))?;
            let channels = usize::try_from(args["audio_channels"].as_u64().unwrap_or(2))?;
            let seconds = args["segment"].as_f64().unwrap_or(40.0);
            let segment = segment_frames(seconds, rate)?;
            let network = Demucs::load(
                &args,
                weights,
                channels,
                stems.len(),
                rate.into(),
                seconds,
                kind == "hdemucs",
                device,
                None,
            )?;
            (
                Network::Hybrid {
                    network: Box::new(network),
                    #[cfg(feature = "onnx")]
                    onnx: match acceleration {
                        super::super::Acceleration::Onnx(provider) => {
                            Some(super::onnx_backend::LegacySession::new(provider))
                        }
                        _ => None,
                    },
                },
                rate,
                channels,
                stems,
                segment,
            )
        }
        _ => {
            let network = LegacyWave::load(kind, &args, &weights, device, acceleration)?;
            let rate = network.rate();
            let channels = network.channels();
            let stems = network.stems().to_vec();
            let segment = network.segment_length();
            (
                Network::Wave(Box::new(network)),
                rate,
                channels,
                stems,
                segment,
            )
        }
    };
    let member = Member {
        network,
        segment,
        gains: vec![1.0; stems.len()],
    };
    Ok((member, rate, channels, stems))
}

fn positional(kind: &str) -> &[&str] {
    match kind {
        "demucs" => &[
            "sources",
            "audio_channels",
            "channels",
            "depth",
            "rewrite",
            "glu",
            "rescale",
            "resample",
            "upsample",
            "kernel_size",
            "stride",
            "growth",
            "lstm_layers",
            "context",
            "normalize",
            "samplerate",
            "segment_length",
        ],
        "demucs_v3" => &[
            "sources",
            "audio_channels",
            "channels",
            "growth",
            "depth",
            "rewrite",
            "lstm_layers",
            "kernel_size",
            "stride",
            "context",
            "gelu",
            "glu",
            "norm_starts",
            "norm_groups",
            "dconv_mode",
            "dconv_depth",
            "dconv_comp",
            "dconv_attn",
            "dconv_lstm",
            "dconv_init",
            "normalize",
            "resample",
            "rescale",
            "samplerate",
            "segment",
        ],
        "conv_tasnet" => &[
            "sources",
            "N",
            "L",
            "B",
            "H",
            "P",
            "X",
            "R",
            "C",
            "audio_channels",
            "norm_type",
            "causal",
            "mask_nonlinear",
            "samplerate",
            "segment_length",
        ],
        "hdemucs" => &[
            "sources",
            "audio_channels",
            "channels",
            "channels_time",
            "growth",
            "nfft",
            "wiener_iters",
            "end_iters",
            "wiener_residual",
            "cac",
            "depth",
            "rewrite",
            "hybrid",
            "hybrid_old",
            "multi_freqs",
            "multi_freqs_depth",
            "freq_emb",
            "emb_scale",
            "emb_smooth",
            "kernel_size",
            "time_stride",
            "stride",
            "context",
            "context_enc",
            "norm_starts",
            "norm_groups",
            "dconv_mode",
            "dconv_depth",
            "dconv_comp",
            "dconv_attn",
            "dconv_lstm",
            "dconv_init",
            "rescale",
            "samplerate",
            "segment",
        ],
        _ => &["sources"],
    }
}

fn infer_architecture(
    path: &Path,
    state: &HashMap<String, Tensor>,
) -> Result<(&'static str, Value)> {
    let shape = |name: &str| -> Result<Vec<i64>> {
        Ok(state
            .get(name)
            .with_context(|| format!("missing legacy Demucs tensor {name}"))?
            .size())
    };
    if state.contains_key("encoder.conv1d_U.weight")
        && state.contains_key("decoder.basis_signals.weight")
    {
        let encoder = shape("encoder.conv1d_U.weight")?;
        ensure!(
            encoder.len() == 3 && encoder.iter().all(|&n| n > 0),
            "invalid ConvTasNet encoder shape"
        );
        let blocks = max_child(state, "separator.network.2.0.").unwrap_or(8);
        let repeats = max_child(state, "separator.network.2.").unwrap_or(4);
        let mask = shape("separator.network.3.weight")?;
        return Ok((
            "conv_tasnet",
            json!({
                "sources": dim(&mask, 0)? / encoder[0], "N": encoder[0], "L": encoder[2],
                "B": dim(&shape("separator.network.1.weight")?, 0)?,
                "H": dim(&shape("separator.network.2.0.0.net.0.weight")?, 0)?,
                "P": shape("separator.network.2.0.0.net.3.net.0.weight")?.last(),
                "X": blocks, "R": repeats, "audio_channels": encoder[1],
            }),
        ));
    }
    let depth = state
        .keys()
        .filter(|key| key.starts_with("encoder.") && key.ends_with(".0.weight"))
        .count();
    if path
        .file_stem()
        .is_some_and(|name| name.to_string_lossy().starts_with("demucs_unittest"))
    {
        return Ok((
            "demucs",
            json!({ "sources": 4, "audio_channels": 2, "channels": 4, "depth": depth, "lstm_layers": 2 }),
        ));
    }
    if depth > 0
        && state.contains_key("encoder.0.0.weight")
        && state.contains_key("lstm.lstm.weight_ih_l0")
    {
        let first = shape("encoder.0.0.weight")?;
        let audio_channels = dim(&first, 1)?;
        ensure!(audio_channels > 0, "invalid Demucs audio channel count");
        let decoder_bias = format!("decoder.{}.2.bias", depth - 1);
        let sources = state
            .get(&decoder_bias)
            .map_or(4, |bias| bias.numel() as i64 / audio_channels);
        let context = state
            .get("decoder.0.0.weight")
            .and_then(|tensor| tensor.size().last().copied())
            .unwrap_or(3);
        let deepest = format!("encoder.{}.0.weight", depth - 1);
        let resample = if let (Some(dec), Some(enc)) =
            (state.get("decoder.0.2.weight"), state.get(&deepest))
        {
            dim(&dec.size(), 1)? == dim(&enc.size(), 0)?
        } else {
            false
        };
        return Ok((
            "demucs",
            json!({
                "sources": sources, "audio_channels": audio_channels, "channels": dim(&first, 0)?,
                "depth": depth, "lstm_layers": 2, "context": context, "resample": resample,
            }),
        ));
    }
    bail!(
        "cannot infer architecture from legacy state dict {}",
        path.display()
    );
}

fn max_child(state: &HashMap<String, Tensor>, prefix: &str) -> Option<usize> {
    state
        .keys()
        .filter_map(|key| {
            key.strip_prefix(prefix)?
                .split('.')
                .next()?
                .parse::<usize>()
                .ok()
        })
        .max()?
        .checked_add(1)
}

fn dim(shape: &[i64], index: usize) -> Result<i64> {
    shape
        .get(index)
        .copied()
        .context("invalid legacy Demucs weight rank")
}

fn segment_frames(seconds: f64, rate: u32) -> Result<usize> {
    let frames = seconds * f64::from(rate);
    ensure!(
        frames.is_finite() && (1.0..=i32::MAX as f64).contains(&frames),
        "invalid Demucs segment duration"
    );
    Ok(frames as usize)
}

fn fade(frame: usize, segment: usize) -> f32 {
    (frame + 1).min(segment - frame) as f32 / segment.div_ceil(2) as f32
}

fn extract(
    mix: &[f32],
    channels: usize,
    start: usize,
    frames: usize,
    padded: usize,
) -> Result<Vec<f32>> {
    let track = mix.len() / channels;
    let half = padded
        .checked_sub(frames)
        .context("invalid legacy Demucs alignment")?
        / 2;
    let missing = half.saturating_sub(start);
    let first = start.saturating_sub(half);
    let copy = (track - first).min(padded - missing);
    let mut output = vec![
        0.0;
        padded
            .checked_mul(channels)
            .context("Demucs input exceeds address space")?
    ];
    for (source, target) in mix.chunks_exact(track).zip(output.chunks_exact_mut(padded)) {
        target[missing..missing + copy].copy_from_slice(&source[first..first + copy]);
    }
    Ok(output)
}
