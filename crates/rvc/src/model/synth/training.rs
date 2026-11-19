use std::collections::{BTreeMap, HashMap};

use anyhow::{Result, ensure};
use tch::{Device, Kind, Tensor};

use super::{Config, Synth, Weights, sequence_mask};
use crate::catalog::ModelInfo;
use crate::model::edit::generator_config;
use crate::model::train_data::{Batch, Config as TrainingConfig, Version};

pub struct Output {
    pub audio: Tensor,
    pub starts: Tensor,
    pub z_p: Tensor,
    pub logs_q: Tensor,
    pub m_p: Tensor,
    pub logs_p: Tensor,
    pub z_mask: Tensor,
}

impl Synth {
    pub fn train_batch(&self, batch: &Batch, segment_frames: i64) -> Result<Output> {
        let w = &self.weights;
        let inter = self.config.inter;
        let samples = batch.spec.size()[0];
        let speaker = self.speaker(&batch.sid)?;
        let (m_p, logs_p, _) = self.prior(
            &batch.phone,
            batch.pitch.as_ref(),
            &batch.phone_lengths,
            0,
            true,
        )?;
        let z_mask = sequence_mask(&batch.spec_lengths, batch.spec.size()[2], batch.spec.kind());
        let source = w.conv(&batch.spec, "enc_q.pre", 1, 0, 1, 1)? * &z_mask;
        let source = self.wavenet(source, &z_mask, &speaker, "enc_q.enc", 16)?;
        let source = w.conv(&source, "enc_q.proj", 1, 0, 1, 1)? * &z_mask;
        let mean_q = source.narrow(1, 0, inter);
        let logs_q = source.narrow(1, inter, inter);
        let z = (&mean_q + Tensor::randn_like(&mean_q) * logs_q.exp()) * &z_mask;
        let z_p = self.flow(z.shallow_clone(), &z_mask, &speaker, false)?;
        ensure!(
            z_p.size() == m_p.size(),
            "posterior and HuBERT frame lengths differ"
        );
        let last_start = (&batch.spec_lengths - segment_frames + 1)
            .clamp_min(1)
            .to_kind(Kind::Float);
        let starts = (Tensor::rand([samples], (Kind::Float, batch.spec.device())) * last_start)
            .to_kind(Kind::Int64);
        let latent = segments(&z, &starts, segment_frames)?;
        let pitch = batch
            .pitchf
            .as_ref()
            .map(|pitch| {
                segments(&pitch.unsqueeze(1), &starts, segment_frames)
                    .map(|pitch| pitch.squeeze_dim(1))
            })
            .transpose()?;
        let audio = self.decode(&latent, pitch.as_ref(), &speaker, segment_frames, 0)?;
        Ok(Output {
            audio,
            starts,
            z_p,
            logs_q,
            m_p,
            logs_p,
            z_mask,
        })
    }
}

pub fn segments(tensor: &Tensor, starts: &Tensor, length: i64) -> Result<Tensor> {
    let (batch, channels, available) = tensor.size3()?;
    let offsets = Tensor::arange(length, (Kind::Int64, tensor.device())).view([1, 1, length]);
    let gather = (starts.view([batch, 1, 1]) + offsets).expand([batch, channels, length], false);
    let padding = (length - available).max(0);
    Ok(tensor
        .constant_pad_nd([0, padding])
        .f_gather(2, &gather, false)?)
}

pub fn generator(
    config: &TrainingConfig,
    version: Version,
    uses_f0: bool,
    device: Device,
    loaded: HashMap<String, Tensor>,
    strict: bool,
) -> Result<(Synth, Vec<(String, Tensor)>)> {
    let m = &config.model;
    let d = &config.data;
    let c = Config::read(generator_config(config), d.sampling_rate)?;
    let hidden = m.hidden_channels;
    let filters = m.filter_channels;
    let inter = m.inter_channels;
    let condition = m.gin_channels;
    ensure!(
        m.upsample_initial_channel > 0 && m.upsample_initial_channel >> c.upsample_rates.len() > 0,
        "invalid decoder channels"
    );
    ensure!(
        (1..=16384).contains(&m.spk_embed_dim)
            && (1..=2048).contains(&condition)
            && (1..=4096).contains(&filters)
            && (128..=8192).contains(&d.filter_length),
        "invalid training channels or feature dimensions"
    );
    let mut init = Initializer {
        loaded,
        strict,
        device,
        params: Vec::new(),
    };
    init.conv(
        "enc_p.emb_phone",
        &[hidden, version.dimensions()],
        true,
        false,
    )?;
    if uses_f0 {
        init.param(
            "enc_p.emb_pitch.weight".into(),
            &[256, hidden],
            Initial::Normal(1.0),
        )?;
    }
    let encoder = "enc_p.encoder";
    for i in 0..c.encoder_layers {
        let layer = format!("{encoder}.attn_layers.{i}");
        let head_channels = hidden / c.heads;
        for key in ["k", "v"] {
            init.param(
                format!("{layer}.emb_rel_{key}"),
                &[1, 21, head_channels],
                Initial::Normal((head_channels as f64).sqrt().recip()),
            )?;
        }
        for name in ["q", "k", "v", "o"] {
            init.conv(
                &format!("{layer}.conv_{name}"),
                &[hidden, hidden, 1],
                true,
                false,
            )?;
        }
    }
    for i in 0..c.encoder_layers {
        init.norm(&format!("{encoder}.norm_layers_1.{i}"), hidden)?;
    }
    for i in 0..c.encoder_layers {
        let layer = format!("{encoder}.ffn_layers.{i}");
        init.conv(
            &format!("{layer}.conv_1"),
            &[filters, hidden, c.encoder_kernel],
            true,
            false,
        )?;
        init.conv(
            &format!("{layer}.conv_2"),
            &[hidden, filters, c.encoder_kernel],
            true,
            false,
        )?;
    }
    for i in 0..c.encoder_layers {
        init.norm(&format!("{encoder}.norm_layers_2.{i}"), hidden)?;
    }
    init.conv("enc_p.proj", &[2 * inter, hidden, 1], true, false)?;

    let decoder = m.upsample_initial_channel;
    if uses_f0 {
        init.conv("dec.m_source.l_linear", &[1, 1], true, false)?;
        for i in 0..c.upsample_rates.len() {
            let remaining = c.upsample_rates[i + 1..].iter().product::<i64>();
            let kernel = if remaining == 1 { 1 } else { 2 * remaining };
            init.conv(
                &format!("dec.noise_convs.{i}"),
                &[decoder >> (i + 1), 1, kernel],
                true,
                false,
            )?;
        }
    }
    init.conv("dec.conv_pre", &[decoder, inter, 7], true, false)?;
    for (i, &k) in c.upsample_kernels.iter().enumerate() {
        init.conv(
            &format!("dec.ups.{i}"),
            &[decoder >> i, decoder >> (i + 1), k],
            true,
            true,
        )?;
    }
    for i in 0..c.upsample_rates.len() {
        let channels = decoder >> (i + 1);
        for (j, &k) in c.kernels.iter().enumerate() {
            let block = format!("dec.resblocks.{}", i * c.kernels.len() + j);
            let (kinds, count): (&[&str], i64) = if c.resblock == 1 {
                (&["convs1", "convs2"], 3)
            } else {
                (&["convs"], 2)
            };
            for kind in kinds {
                for layer in 0..count {
                    init.conv(
                        &format!("{block}.{kind}.{layer}"),
                        &[channels, channels, k],
                        true,
                        true,
                    )?;
                }
            }
        }
    }
    init.conv(
        "dec.conv_post",
        &[1, decoder >> c.upsample_rates.len(), 7],
        false,
        false,
    )?;
    init.conv("dec.cond", &[decoder, condition, 1], true, false)?;
    init.conv(
        "enc_q.pre",
        &[hidden, d.filter_length / 2 + 1, 1],
        true,
        false,
    )?;
    init.wavenet("enc_q.enc", hidden, condition, 16)?;
    init.conv("enc_q.proj", &[2 * inter, hidden, 1], true, false)?;
    for i in [0, 2, 4, 6] {
        let prefix = format!("flow.flows.{i}");
        init.conv(
            &format!("{prefix}.pre"),
            &[hidden, inter / 2, 1],
            true,
            false,
        )?;
        init.wavenet(&format!("{prefix}.enc"), hidden, condition, 3)?;
        init.param(
            format!("{prefix}.post.weight"),
            &[inter / 2, hidden, 1],
            Initial::Constant(0.0),
        )?;
        init.param(
            format!("{prefix}.post.bias"),
            &[inter / 2],
            Initial::Constant(0.0),
        )?;
    }
    init.param(
        "emb_g.weight".into(),
        &[m.spk_embed_dim, condition],
        Initial::Normal(1.0),
    )?;
    ensure!(
        !strict || init.loaded.is_empty(),
        "checkpoint does not match this generator"
    );
    let names: BTreeMap<_, _> = config
        .speaker_info
        .iter()
        .map(|speaker| (speaker.id, speaker.name.clone()))
        .collect();
    let info = ModelInfo {
        version: version.name().to_owned(),
        speaker_count: m.spk_embed_dim,
        speakers: names,
        uses_f0,
        sample_rate: Some(d.sampling_rate),
        info: None,
    };
    let weights = Weights::new(
        init.params
            .iter()
            .map(|(name, value)| (name.clone(), value.shallow_clone()))
            .collect(),
    );
    Ok((
        Synth {
            #[cfg(feature = "onnx")]
            onnx: Default::default(),
            info,
            rate: d.sampling_rate,
            weights,
            config: c,
            graph: std::cell::RefCell::default(),
        },
        init.params,
    ))
}

enum Initial {
    Uniform(f64),
    Normal(f64),
    Constant(f64),
}

struct Initializer {
    loaded: HashMap<String, Tensor>,
    strict: bool,
    device: Device,
    params: Vec<(String, Tensor)>,
}

impl Initializer {
    fn param(&mut self, name: String, shape: &[i64], init: Initial) -> Result<()> {
        let device = self.device;
        let tensor = self.take(&name);
        let value = match tensor {
            Some(tensor) if tensor.size() == shape => tensor,
            Some(tensor)
                if name == "emb_g.weight" && tensor.dim() == 2 && tensor.size()[1] == shape[1] =>
            {
                let result = Tensor::randn(shape, (Kind::Float, device));
                let rows = tensor.size()[0].min(shape[0]);
                result
                    .narrow(0, 0, rows)
                    .f_copy_(&tensor.narrow(0, 0, rows))?;
                result
            }
            Some(_) => anyhow::bail!("unexpected shape for generator parameter {name}"),
            None => {
                ensure!(!self.strict, "generator checkpoint is missing {name}");
                match init {
                    Initial::Uniform(bound) => {
                        Tensor::empty(shape, (Kind::Float, device)).uniform_(-bound, bound)
                    }
                    Initial::Normal(std) => Tensor::randn(shape, (Kind::Float, device)) * std,
                    Initial::Constant(value) => Tensor::full(shape, value, (Kind::Float, device)),
                }
            }
        };
        self.params
            .push((name, value.detach().set_requires_grad(true)));
        Ok(())
    }

    fn take(&mut self, name: &str) -> Option<Tensor> {
        self.loaded.remove(name).or_else(|| {
            let (prefix, index) = name
                .strip_suffix(".weight_g")
                .map(|prefix| (prefix, 0))
                .or_else(|| name.strip_suffix(".weight_v").map(|prefix| (prefix, 1)))?;
            self.loaded
                .remove(&format!("{prefix}.parametrizations.weight.original{index}"))
        })
    }

    fn norm(&mut self, path: &str, width: i64) -> Result<()> {
        self.param(format!("{path}.gamma"), &[width], Initial::Constant(1.0))?;
        self.param(format!("{path}.beta"), &[width], Initial::Constant(0.0))
    }

    fn conv(&mut self, path: &str, shape: &[i64], bias: bool, normalized: bool) -> Result<()> {
        let bound = (shape[1..].iter().product::<i64>() as f64).sqrt().recip();
        if !normalized {
            self.param(format!("{path}.weight"), shape, Initial::Uniform(bound))?;
        }
        if bias {
            let width = if path.starts_with("dec.ups.") {
                shape[1]
            } else {
                shape[0]
            };
            self.param(format!("{path}.bias"), &[width], Initial::Uniform(bound))?;
        }
        if normalized {
            let mut scale = vec![1; shape.len()];
            scale[0] = shape[0];
            let g = format!("{path}.weight_g");
            let v = format!("{path}.weight_v");
            let found = self.loaded.contains_key(&g)
                || self
                    .loaded
                    .contains_key(&format!("{path}.parametrizations.weight.original0"));
            if !found && !self.strict {
                let value =
                    Tensor::empty(shape, (Kind::Float, self.device)).uniform_(-bound, bound);
                let dims: Vec<_> = (1..shape.len() as i64).collect();
                let norms = value
                    .square()
                    .sum_dim_intlist(dims.as_slice(), true, Kind::Float)
                    .sqrt();
                self.loaded.insert(g.clone(), norms);
                self.loaded.insert(v.clone(), value);
            }
            self.param(g, &scale, Initial::Constant(1.0))?;
            self.param(v, shape, Initial::Uniform(bound))?;
        }
        Ok(())
    }

    fn wavenet(&mut self, prefix: &str, channels: i64, condition: i64, depth: i64) -> Result<()> {
        self.conv(
            &format!("{prefix}.cond_layer"),
            &[2 * channels * depth, condition, 1],
            true,
            true,
        )?;
        for layer in 0..depth {
            self.conv(
                &format!("{prefix}.in_layers.{layer}"),
                &[2 * channels, channels, 5],
                true,
                true,
            )?;
        }
        for layer in 0..depth {
            let output = if layer + 1 == depth {
                channels
            } else {
                2 * channels
            };
            self.conv(
                &format!("{prefix}.res_skip_layers.{layer}"),
                &[output, channels, 1],
                true,
                true,
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strict_resume_preserves_existing_rows_only_for_compatible_speaker_resize() -> Result<()> {
        let embedding = Tensor::f_from_slice(&[1.0f32, 2.0, 3.0, 4.0])?.f_reshape([2, 2])?;
        let mut initializer = Initializer {
            loaded: HashMap::from([
                ("emb_g.weight".into(), embedding.shallow_clone()),
                ("enc_p.emb_phone.weight".into(), embedding.shallow_clone()),
            ]),
            strict: true,
            device: Device::Cpu,
            params: Vec::new(),
        };
        initializer.param("emb_g.weight".into(), &[3, 2], Initial::Normal(1.0))?;
        let resized = &initializer.params[0].1;
        assert_eq!(resized.size(), [3, 2]);
        assert!(resized.requires_grad());
        assert!(resized.f_narrow(0, 0, 2)?.f_equal(&embedding)?);
        assert!(
            initializer
                .param(
                    "enc_p.emb_phone.weight".into(),
                    &[3, 2],
                    Initial::Normal(1.0),
                )
                .is_err()
        );
        initializer.loaded.insert("emb_g.weight".into(), embedding);
        assert!(
            initializer
                .param("emb_g.weight".into(), &[2, 3], Initial::Normal(1.0))
                .is_err()
        );
        Ok(())
    }
}
