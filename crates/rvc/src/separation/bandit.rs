//! Band-split recurrent separation from PyMSS, with shared Bandit v2 primitives.
//! See `licenses/pymss-core.txt`.

use std::collections::HashMap;

use anyhow::{Context, Result, bail, ensure};
use serde::Deserialize;
use serde_json::Value;
use tch::{Device, Kind, Tensor, no_grad};

use super::super::Acceleration;

#[path = "bandit/mlx_network.rs"]
mod mlx_network;
#[cfg(feature = "onnx")]
#[path = "bandit/onnx_network.rs"]
mod onnx_network;

const CHANNELS: usize = 2;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Version {
    V1,
    V2,
}

impl Version {
    fn root(self) -> &'static str {
        match self {
            Self::V1 => "bsrnn.",
            Self::V2 => "",
        }
    }
}

#[derive(Deserialize)]
#[serde(untagged)]
pub(super) enum BandSpec {
    Named(String),
    Ranges(Vec<[i64; 2]>),
}

#[derive(Deserialize)]
#[serde(default)]
pub(super) struct Config {
    #[serde(alias = "in_channel")]
    pub(super) in_channels: Option<i64>,
    pub(super) stems: Vec<String>,
    pub(super) band_specs: Option<BandSpec>,
    pub(super) band_type: Option<String>,
    pub(super) n_bands: Option<usize>,
    fs: i64,
    require_no_overlap: bool,
    normalize_channel_independently: bool,
    treat_channel_as_feature: bool,
    n_sqm_modules: usize,
    emb_dim: i64,
    rnn_dim: i64,
    bidirectional: bool,
    rnn_type: String,
    mlp_dim: i64,
    hidden_activation: String,
    hidden_activation_kwargs: Value,
    complex_mask: bool,
    use_freq_weights: bool,
    mult_add_mask: bool,
    n_fft: i64,
    win_length: Option<i64>,
    hop_length: i64,
    power: Option<f64>,
    center: bool,
    normalized: bool,
    pad_mode: String,
    onesided: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            in_channels: None,
            stems: Vec::new(),
            band_specs: None,
            band_type: None,
            n_bands: None,
            fs: 44100,
            require_no_overlap: false,
            normalize_channel_independently: false,
            treat_channel_as_feature: true,
            n_sqm_modules: 12,
            emb_dim: 128,
            rnn_dim: 256,
            bidirectional: true,
            rnn_type: "LSTM".to_owned(),
            mlp_dim: 512,
            hidden_activation: "Tanh".to_owned(),
            hidden_activation_kwargs: Value::Null,
            complex_mask: true,
            use_freq_weights: true,
            mult_add_mask: false,
            n_fft: 2048,
            win_length: Some(2048),
            hop_length: 512,
            power: None,
            center: true,
            normalized: true,
            pad_mode: "constant".to_owned(),
            onesided: true,
        }
    }
}

impl Config {
    fn validate(&self, version: Version) -> Result<()> {
        ensure!(
            self.in_channels == Some(1) && !self.normalize_channel_independently,
            "Bandit inference expects mono processing of each stereo channel and a shared channel norm"
        );
        ensure!(
            version == Version::V1 || self.treat_channel_as_feature,
            "Bandit v2 requires treating each input channel as a feature"
        );
        ensure!(
            !self.mult_add_mask
                && self.power.is_none()
                && self.onesided
                && (8..=65536).contains(&self.n_fft)
                && (1..=self.n_fft).contains(&self.window_length())
                && (1..=self.n_fft).contains(&self.hop_length)
                && (1..=768_000).contains(&self.fs)
                && matches!(
                    self.pad_mode.as_str(),
                    "constant" | "reflect" | "replicate" | "circular"
                ),
            "invalid Bandit spectral configuration"
        );
        ensure!(
            (1..=128).contains(&self.stems.len())
                && self.stems.iter().all(|stem| !stem.is_empty())
                && (1..=8192).contains(&self.emb_dim)
                && (1..=8192).contains(&self.rnn_dim)
                && (1..=8192).contains(&self.mlp_dim)
                && self.n_sqm_modules <= 128,
            "invalid Bandit network dimensions"
        );
        Ok(())
    }

    fn window_length(&self) -> i64 {
        self.win_length
            .filter(|length| *length != 0)
            .unwrap_or(self.n_fft)
    }
}

#[derive(Clone, Copy)]
struct Band {
    start: i64,
    end: i64,
}

impl Band {
    fn width(self) -> i64 {
        self.end - self.start
    }
}

struct Bands {
    ranges: Vec<Band>,
    overlap: bool,
}

impl Bands {
    fn from_config(config: &Config) -> Result<Self> {
        let spec = config
            .band_specs
            .as_ref()
            .context("missing Bandit band specification")?;
        let bins = config.n_fft / 2 + 1;
        let (ranges, overlap) = match spec {
            BandSpec::Ranges(ranges) => (
                ranges
                    .iter()
                    .map(|&[start, end]| Band { start, end })
                    .collect(),
                false,
            ),
            BandSpec::Named(name)
                if matches!(
                    name.as_str(),
                    "dnr:speech" | "dnr:vox7" | "musdb:vocals" | "musdb:vox7"
                ) =>
            {
                (vocal_bands(config.n_fft, config.fs)?, false)
            }
            BandSpec::Named(name) => {
                let count = config
                    .n_bands
                    .context("named Bandit bands require n_bands")?;
                ensure!(
                    (1..=1024).contains(&count),
                    "invalid Bandit frequency band count"
                );
                let ranges = if name.contains("tribark") {
                    triangular_bands(config.fs, bins, count, false)?
                } else if name.contains("bark") {
                    bark_bands(config.fs, bins, count)
                } else if name.contains("erb") {
                    triangular_bands(config.fs, bins, count, true)?
                } else if name.contains("musical") {
                    musical_bands(config.fs, bins, count)
                } else if name.contains("mel") {
                    mel_bands(config.fs, bins, count)
                } else {
                    bail!("unsupported Bandit band specification {name:?}");
                };
                (ranges, true)
            }
        };
        ensure!(
            !ranges.is_empty()
                && ranges.len() <= 1024
                && ranges
                    .iter()
                    .all(|band| 0 <= band.start && band.start < band.end && band.end <= bins),
            "Bandit frequency bands must be nonempty ranges within the STFT"
        );
        // Both PyMSS mask estimators check gaps regardless of the band split setting.
        // The check permits a single-bin gap and does not check the last bin.
        ensure!(
            ranges[0].start == 0
                && ranges
                    .windows(2)
                    .all(|pair| pair[1].start - pair[0].end <= 1),
            "Bandit bands leave a gap under the configured PyMSS bound check"
        );
        if config.require_no_overlap || !overlap {
            ensure!(
                ranges.windows(2).all(|pair| pair[0].end < pair[1].start),
                "Bandit bands overlap under the configured PyMSS bound check"
            );
        }
        Ok(Self { ranges, overlap })
    }
}

fn vocal_bands(fft: i64, rate: i64) -> Result<Vec<Band>> {
    let to_bin = |hz| (hz as f64 * fft as f64 / rate as f64).round_ties_even() as i64;
    let mut result = Vec::new();
    for (low, high, step) in [
        (0, 1000, 100),
        (1000, 4000, 250),
        (4000, 8000, 500),
        (8000, 16000, 1000),
        (16000, 20000, 2000),
    ] {
        let (mut start, end, step) = (to_bin(low), to_bin(high), to_bin(step));
        ensure!(
            step > 0,
            "Bandit vocal bands require a higher FFT resolution"
        );
        while start < end {
            let next = (start + step).min(end);
            result.push(Band { start, end: next });
            start = next;
        }
    }
    result.push(Band {
        start: to_bin(20000),
        end: fft / 2 + 1,
    });
    Ok(result)
}

fn spaced(start: f64, end: f64, index: usize, count: usize) -> f64 {
    if count == 1 || index == 0 {
        start
    } else if index + 1 == count {
        end
    } else {
        start + index as f64 * ((end - start) / (count - 1) as f64)
    }
}

fn musical_bands(rate: i64, bins: i64, count: usize) -> Vec<Band> {
    let min = rate as f64 / (2 * (bins - 1)) as f64;
    let max = rate as f64 / 2.0;
    let hz_to_midi = |hz: f64| 69.0 + 12.0 * (hz / 440.0).log2();
    let (min_note, max_note) = (hz_to_midi(min).max(0.0), hz_to_midi(max));
    let multiplier = 2_f64.powf((max / min).log2() / count as f64);
    (0..count)
        .filter_map(|index| {
            let note = spaced(min_note, max_note, index, count);
            let hz = 440.0 * 2_f64.powf((note - 69.0) / 12.0);
            let start = if index == 0 {
                0
            } else {
                (hz / multiplier / min).floor() as i64
            };
            let end = if index + 1 == count {
                bins
            } else {
                ((hz * multiplier / min).ceil() as i64 + 1).min(bins)
            };
            (start < end).then_some(Band { start, end })
        })
        .collect()
}

fn filter_ranges(count: usize, bins: i64, active: impl Fn(usize, i64) -> bool) -> Vec<Band> {
    (0..count)
        .filter_map(|index| {
            let start = (0..bins).find(|&bin| active(index, bin))?;
            let end = (start..bins).rfind(|&bin| active(index, bin))? + 1;
            Some(Band { start, end })
        })
        .collect()
}

fn bark_bands(rate: i64, bins: i64, count: usize) -> Vec<Band> {
    let fft = 2 * (bins - 1);
    let hz_to_bark = |hz: f64| 6.0 * (hz / 600.0).asinh();
    let limit = hz_to_bark(rate as f64 / 2.0);
    let last = (((fft + 1) as f64 * (600.0 * (limit / 6.0).sinh() / rate as f64)).floor() as i64)
        .min(bins);
    filter_ranges(count, last, |band, bin| {
        let frequency = bin as f64 * rate as f64 / (fft + 1) as f64;
        let delta = hz_to_bark(frequency) - spaced(0.0, limit, band, count);
        (-1.3..=2.5).contains(&delta)
    })
}

fn mel_bands(rate: i64, bins: i64, count: usize) -> Vec<Band> {
    let max = rate as f64 / 2.0;
    let limit = 2595.0 * (1.0 + max / 700.0).log10();
    let points = (0..count + 2)
        .map(|index| 700.0 * (10_f64.powf(spaced(0.0, limit, index, count + 2) / 2595.0) - 1.0))
        .collect::<Vec<_>>();
    filter_ranges(count, bins, |band, bin| {
        let hz = spaced(0.0, max, bin as usize, bins as usize);
        band == 0 && bin == 0 || points[band] < hz && hz < points[band + 2]
    })
}

fn triangular_bands(rate: i64, bins: i64, count: usize, erb: bool) -> Result<Vec<Band>> {
    let cpu_float = (Kind::Float, Device::Cpu);
    let maximum = rate as f64 / 2.0;
    let steps = i64::try_from(count + 2)?;
    let points = if erb {
        let a = 1000.0 * 10_f64.ln() / (24.7 * 4.37);
        let limit = a * (1.0 + 0.00437 * maximum).log10();
        let expo = Tensor::f_linspace(0.0, limit, steps, cpu_float)?.f_div_scalar(a)?;
        Tensor::f_pow_scalar(10, &expo)?
            .f_sub_scalar(1)?
            .f_div_scalar(0.00437)?
    } else {
        let limit = 6.0 * (maximum / 600.0).asinh();
        Tensor::f_linspace(0.0, limit, steps, cpu_float)?
            .f_div_scalar(6)?
            .f_sinh()?
            .f_mul_scalar(600)?
    };
    let hz = Vec::<f32>::try_from(Tensor::f_linspace(0, rate / 2, bins, cpu_float)?)?;
    let points = Vec::<f32>::try_from(points)?;
    let mut bands = filter_ranges(count, bins, |band, bin| {
        points[band] < hz[bin as usize] && hz[bin as usize] < points[band + 2]
    });
    if let Some(first) = bands.first_mut() {
        first.start = 0;
    }
    Ok(bands)
}

struct Weights<'a> {
    values: &'a HashMap<String, Tensor>,
    device: Device,
}

impl Weights<'_> {
    fn raw(&self, name: &str, shape: &[i64]) -> Result<&Tensor> {
        let tensor = self
            .values
            .get(name)
            .with_context(|| format!("Bandit checkpoint is missing {name}"))?;
        ensure!(
            tensor.size() == shape && tensor.is_floating_point(),
            "invalid Bandit tensor {name}: {:?} {:?}, expected floating point {shape:?}",
            tensor.kind(),
            tensor.size(),
        );
        Ok(tensor)
    }

    fn tensor(&self, name: &str, shape: &[i64]) -> Result<Tensor> {
        Ok(self
            .raw(name, shape)?
            .f_to_kind(Kind::Float)?
            .f_to_device(self.device)?)
    }

    fn affine(&self, name: &str, shape: &[i64]) -> Result<Affine> {
        Ok(Affine {
            weight: self.tensor(&format!("{name}.weight"), shape)?,
            bias: self.tensor(&format!("{name}.bias"), &shape[..1])?,
        })
    }

    fn affine_alias(&self, name: &str, alias: Option<&str>, shape: &[i64]) -> Result<Affine> {
        let resolved = alias.filter(|alias| self.values.contains_key(&format!("{alias}.weight")));
        self.affine(resolved.unwrap_or(name), shape)
    }
}

struct Affine {
    weight: Tensor,
    bias: Tensor,
}

impl Affine {
    fn norm(&self, input: &Tensor) -> Result<Tensor> {
        Ok(input.f_layer_norm(
            self.weight.size(),
            Some(&self.weight),
            Some(&self.bias),
            1e-5,
            true,
        )?)
    }

    fn linear(&self, input: &Tensor) -> Result<Tensor> {
        Ok(input.f_linear(&self.weight, Some(&self.bias))?)
    }
}

struct FeatureProjection {
    norm: Affine,
    linear: Affine,
    band: Band,
}

impl FeatureProjection {
    fn load(
        weights: &Weights,
        band: Band,
        index: usize,
        dim: i64,
        version: Version,
    ) -> Result<Self> {
        let root = format!("{}band_split.norm_fc_modules.{index}", version.root());
        let (norm, linear) = match version {
            Version::V1 => ("norm", "fc"),
            Version::V2 => ("combined.0", "combined.1"),
        };
        let input = 2 * band.width();
        Ok(Self {
            norm: weights.affine(&format!("{root}.{norm}"), &[input])?,
            linear: weights.affine(&format!("{root}.{linear}"), &[dim, input])?,
            band,
        })
    }

    fn forward(&self, spectrum: &Tensor, version: Version) -> Result<Tensor> {
        let shape = spectrum.size();
        let bins = spectrum.f_narrow(
            match version {
                Version::V1 => 3,
                Version::V2 => 2,
            },
            self.band.start,
            self.band.width(),
        )?;
        let flat = bins.f_reshape([shape[0], shape[1], -1])?.f_contiguous()?;
        self.linear.linear(&self.norm.norm(&flat)?)
    }
}

#[derive(Clone, Copy)]
enum RnnType {
    Lstm,
    Gru,
    Tanh,
}

impl RnnType {
    fn parse(value: &str) -> Result<Self> {
        Ok(match value {
            "LSTM" => Self::Lstm,
            "GRU" => Self::Gru,
            "RNN" => Self::Tanh,
            _ => bail!("unsupported Bandit recurrent layer {value:?}"),
        })
    }

    fn gates(self) -> i64 {
        match self {
            Self::Lstm => 4,
            Self::Gru => 3,
            Self::Tanh => 1,
        }
    }
}

struct ResidualRnn {
    norm: Affine,
    params: Vec<Tensor>,
    output: Affine,
    hidden: i64,
    bidirectional: bool,
    rnn_type: RnnType,
}

impl ResidualRnn {
    fn load(weights: &Weights, name: &str, config: &Config, rnn_type: RnnType) -> Result<Self> {
        let embedding = config.emb_dim;
        let hidden = config.rnn_dim;
        let directions = 1 + i64::from(config.bidirectional);
        let gates = rnn_type.gates() * hidden;
        let mut params = Vec::with_capacity(4 * directions as usize);
        for direction in 0..directions {
            let suffix = if direction == 0 { "" } else { "_reverse" };
            for (parameter, shape) in [
                ("weight_ih", vec![gates, embedding]),
                ("weight_hh", vec![gates, hidden]),
                ("bias_ih", vec![gates]),
                ("bias_hh", vec![gates]),
            ] {
                params.push(weights.tensor(&format!("{name}.rnn.{parameter}_l0{suffix}"), &shape)?);
            }
        }
        Ok(Self {
            norm: weights.affine(&format!("{name}.norm"), &[embedding])?,
            output: weights.affine(&format!("{name}.fc"), &[embedding, hidden * directions])?,
            params,
            hidden,
            bidirectional: config.bidirectional,
            rnn_type,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let (batch, lanes, steps, embedding) = input.size4()?;
        let normalized = self
            .norm
            .norm(input)?
            .f_reshape([batch * lanes, steps, embedding])?
            .f_contiguous()?;
        let initial = Tensor::f_zeros(
            [
                1 + i64::from(self.bidirectional),
                batch * lanes,
                self.hidden,
            ],
            (Kind::Float, input.device()),
        )?;
        let output = match self.rnn_type {
            RnnType::Lstm => {
                let weights: Vec<_> = self.params.iter().collect();
                normalized
                    .f_lstm(
                        &[&initial, &initial],
                        &weights,
                        true,
                        1,
                        0.0,
                        false,
                        self.bidirectional,
                        true,
                    )?
                    .0
            }
            RnnType::Gru => {
                normalized
                    .f_gru(
                        &initial,
                        &self.params,
                        true,
                        1,
                        0.0,
                        false,
                        self.bidirectional,
                        true,
                    )?
                    .0
            }
            RnnType::Tanh => {
                normalized
                    .f_rnn_tanh(
                        &initial,
                        &self.params,
                        true,
                        1,
                        0.0,
                        false,
                        self.bidirectional,
                        true,
                    )?
                    .0
            }
        };
        Ok(self
            .output
            .linear(&output.f_reshape([batch, lanes, steps, -1])?)?
            .f_add(input)?)
    }
}

#[derive(Clone)]
enum Activation {
    Tanh,
    Relu,
    Relu6,
    Silu,
    Sigmoid,
    Mish,
    Gelu(String),
    LeakyRelu(f64),
    Elu(f64),
    Identity,
}

impl Activation {
    fn load(config: &Config) -> Result<Self> {
        let kwargs = &config.hidden_activation_kwargs;
        let optional_scalar = |key, default| -> Result<f64> {
            match kwargs.get(key) {
                None => Ok(default),
                Some(value) => value
                    .as_f64()
                    .with_context(|| format!("invalid Bandit activation {key}")),
            }
        };
        Ok(match config.hidden_activation.as_str() {
            "Tanh" => Self::Tanh,
            "ReLU" => Self::Relu,
            "ReLU6" => Self::Relu6,
            "SiLU" => Self::Silu,
            "Sigmoid" => Self::Sigmoid,
            "Mish" => Self::Mish,
            "GELU" => Self::Gelu(kwargs["approximate"].as_str().unwrap_or("none").to_owned()),
            "LeakyReLU" => Self::LeakyRelu(optional_scalar("negative_slope", 0.01)?),
            "ELU" => Self::Elu(optional_scalar("alpha", 1.0)?),
            "Identity" => Self::Identity,
            name => bail!("unsupported Bandit activation {name:?}"),
        })
    }

    fn apply(&self, input: Tensor) -> Result<Tensor> {
        Ok(match self {
            Self::Tanh => input.f_tanh()?,
            Self::Relu => input.f_relu()?,
            Self::Relu6 => input.f_clamp(0.0, 6.0)?,
            Self::Silu => input.f_silu()?,
            Self::Sigmoid => input.f_sigmoid()?,
            Self::Mish => input.f_mish()?,
            Self::Gelu(approximate) => input.f_gelu(approximate)?,
            Self::LeakyRelu(slope) if *slope == 0.01 => input.f_leaky_relu()?,
            Self::LeakyRelu(slope) => {
                input.f_where_self(&input.f_gt(0)?, &input.f_mul_scalar(*slope)?)?
            }
            Self::Elu(alpha) if *alpha == 1.0 => input.f_elu()?,
            Self::Elu(alpha) => {
                input.f_where_self(&input.f_gt(0)?, &input.f_expm1()?.f_mul_scalar(*alpha)?)?
            }
            Self::Identity => input,
        })
    }
}

struct MaskProjection {
    norm: Affine,
    hidden: Affine,
    output: Affine,
    width: i64,
}

impl MaskProjection {
    fn load(weights: &Weights, name: &str, band: Band, settings: &MaskConfig) -> Result<Self> {
        let combined = settings.version == Version::V2;
        let norm = combined.then(|| format!("{name}.combined.0"));
        let hidden = combined.then(|| format!("{name}.combined.1.0"));
        let output = combined.then(|| format!("{name}.combined.2.0"));
        let output_dim = 2 * band.width() * (1 + i64::from(settings.complex));
        Ok(Self {
            norm: weights.affine_alias(
                &format!("{name}.norm"),
                norm.as_deref(),
                &[settings.embedding],
            )?,
            hidden: weights.affine_alias(
                &format!("{name}.hidden.0"),
                hidden.as_deref(),
                &[settings.mlp, settings.embedding],
            )?,
            output: weights.affine_alias(
                &format!("{name}.output.0"),
                output.as_deref(),
                &[output_dim, settings.mlp],
            )?,
            width: band.width(),
        })
    }

    fn forward(&self, input: &Tensor, config: &MaskConfig) -> Result<Tensor> {
        let (channels, time, _) = input.size3()?;
        let hidden = config
            .activation
            .apply(self.hidden.linear(&self.norm.norm(input)?)?)?;
        let output = self
            .output
            .linear(&hidden)?
            .f_glu(-1)?
            .f_to_device(config.spectral_device)?;
        let output = if config.complex {
            output
                .f_reshape([channels, time, self.width, 2])?
                .f_contiguous()?
                .f_view_as_complex()?
        } else {
            output.f_reshape([channels, time, self.width])?
        };
        Ok(output.f_transpose(1, 2)?)
    }
}

struct MaskConfig {
    version: Version,
    activation: Activation,
    embedding: i64,
    mlp: i64,
    complex: bool,
    weighted: bool,
    spectral_device: Device,
}

struct StemMask {
    bands: Vec<MaskProjection>,
    frequencies: Vec<Option<Tensor>>,
}

impl StemMask {
    fn load(weights: &Weights, name: &str, bands: &[Band], config: &MaskConfig) -> Result<Self> {
        let projections = bands
            .iter()
            .enumerate()
            .map(|(index, &band)| {
                MaskProjection::load(weights, &format!("{name}.norm_mlp.{index}"), band, config)
            })
            .collect::<Result<_>>()?;
        let frequencies = bands
            .iter()
            .enumerate()
            .map(|(index, band)| {
                if !config.weighted {
                    return Ok(None);
                }
                let weight =
                    weights.raw(&format!("{name}.freq_weights/{index}"), &[band.width()])?;
                let weight = weight.f_to_device(config.spectral_device)?.f_unsqueeze(1)?;
                Ok(Some(weight))
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            bands: projections,
            frequencies,
        })
    }

    fn forward(
        &self,
        input: &Tensor,
        bands: &Bands,
        config: &MaskConfig,
        bins: i64,
    ) -> Result<Tensor> {
        let estimates = self.bands.iter().enumerate().map(|(index, projection)| {
            let band = projection.forward(&input.f_select(1, index as i64)?, config)?;
            match &self.frequencies[index] {
                Some(weights) => Ok(band.f_mul(weights)?),
                None => Ok(band),
            }
        });
        if !bands.overlap {
            return Ok(Tensor::f_cat(&estimates.collect::<Result<Vec<_>>>()?, 1)?);
        }
        let (channels, _, time, _) = input.size4()?;
        let kind = if config.complex || config.version == Version::V2 {
            Kind::ComplexFloat
        } else {
            Kind::Float
        };
        let result = Tensor::f_zeros([channels, bins, time], (kind, config.spectral_device))?;
        for (estimate, band) in estimates.zip(&bands.ranges) {
            let _ = result
                .f_narrow(1, band.start, band.width())?
                .f_add_(&estimate?)?;
        }
        Ok(result)
    }
}

struct Spectral {
    window: Tensor,
    inverse: Tensor,
    fft: i64,
    hop: i64,
    length: i64,
    pad_mode: String,
    center: bool,
    normalized: bool,
}

impl Spectral {
    fn load(weights: &Weights, config: &Config, device: Device) -> Result<Self> {
        let length = config.window_length();
        let window = |name| -> Result<Tensor> {
            Ok(weights
                .raw(name, &[length])?
                .f_to_kind(Kind::Float)?
                .f_to_device(device)?)
        };
        Ok(Self {
            window: window("stft.window")?,
            inverse: window("istft.window")?,
            fft: config.n_fft,
            hop: config.hop_length,
            length,
            pad_mode: config.pad_mode.clone(),
            center: config.center,
            normalized: config.normalized,
        })
    }

    fn stft(&self, input: &Tensor) -> Result<Tensor> {
        let mut audio = input.f_to_device(self.window.device())?;
        if self.center {
            audio = audio.f_pad([self.fft / 2, self.fft / 2], &self.pad_mode, None)?;
        }
        Ok(audio.f_stft_center(
            self.fft,
            self.hop,
            self.length,
            Some(&self.window),
            false,
            &self.pad_mode,
            self.normalized,
            true,
            true,
            false,
        )?)
    }

    fn istft(&self, spectrum: &Tensor, samples: i64) -> Result<Tensor> {
        Ok(spectrum.f_istft(
            self.fft,
            self.hop,
            self.length,
            Some(&self.inverse),
            self.center,
            self.normalized,
            true,
            samples,
            false,
        )?)
    }

    fn render(
        &self,
        spectrum: &Tensor,
        count: usize,
        samples: i64,
        masks: impl ExactSizeIterator<Item = Result<Tensor>>,
    ) -> Result<Vec<f32>> {
        let plane = samples as usize * CHANNELS;
        let item = plane * masks.len();
        let mut result = vec![0.0; count * item];
        for (index, mask) in masks.enumerate() {
            let wave = self.istft(&spectrum.f_mul(&mask?)?, samples)?;
            let wave = wave.f_to_device(Device::Cpu)?.f_reshape([-1])?;
            let wave = Vec::<f32>::try_from(wave)?;
            ensure!(
                wave.len() == plane * count,
                "unexpected Bandit output shape"
            );
            for (outputs, input) in result.chunks_exact_mut(item).zip(wave.chunks_exact(plane)) {
                outputs[index * plane..(index + 1) * plane].copy_from_slice(input);
            }
        }
        Ok(result)
    }
}

pub(super) struct Bandit {
    stems: usize,
    backend: Backend,
}

enum Backend {
    Torch(Box<TorchNetwork>),
    Mlx(Box<mlx_network::Network>),
    #[cfg(feature = "onnx")]
    Onnx(Box<onnx_network::Network>),
}

struct TorchNetwork {
    version: Version,
    device: Device,
    spectral: Spectral,
    bands: Bands,
    projections: Vec<FeatureProjection>,
    recurrent: Vec<ResidualRnn>,
    masks: Vec<StemMask>,
    mask_config: MaskConfig,
}

impl Bandit {
    pub(super) fn load_v1(
        document: &Value,
        weights: &HashMap<String, Tensor>,
        device: Device,
    ) -> Result<Self> {
        let mut config =
            Config::deserialize(&document["model"]).context("invalid Bandit config.model")?;
        config.stems.retain(|stem| stem != "mne:+");
        Self::load(config, weights, device, Version::V1, document)
    }

    pub(super) fn load(
        config: Config,
        weights: &HashMap<String, Tensor>,
        device: Device,
        version: Version,
        document: &Value,
    ) -> Result<Self> {
        config.validate(version)?;
        let acceleration = Acceleration::configured(document, device, true, "Bandit")?;
        let weight_device = acceleration.weight_device(device);
        let spectral_device = if weight_device == Device::Mps {
            Device::Cpu
        } else {
            weight_device
        };
        let weights = Weights {
            values: weights,
            device: weight_device,
        };
        let spectral = Spectral::load(&weights, &config, spectral_device)?;
        let bands = Bands::from_config(&config)?;
        let projections = bands
            .ranges
            .iter()
            .enumerate()
            .map(|(index, &band)| {
                FeatureProjection::load(&weights, band, index, config.emb_dim, version)
            })
            .collect::<Result<_>>()?;
        let rnn_type = RnnType::parse(&config.rnn_type)?;
        let recurrent = (0..2 * config.n_sqm_modules)
            .map(|index| {
                let position = match version {
                    Version::V1 => index,
                    Version::V2 => 2 * index,
                };
                let name = format!("{}tf_model.seqband.{position}", version.root());
                ResidualRnn::load(&weights, &name, &config, rnn_type)
            })
            .collect::<Result<_>>()?;
        let mask_config = MaskConfig {
            version,
            activation: Activation::load(&config)?,
            embedding: config.emb_dim,
            mlp: config.mlp_dim,
            complex: config.complex_mask,
            weighted: bands.overlap && config.use_freq_weights,
            spectral_device,
        };
        let masks = config
            .stems
            .iter()
            .map(|stem| {
                let name = format!("{}mask_estim.{stem}", version.root());
                StemMask::load(&weights, &name, &bands.ranges, &mask_config)
            })
            .collect::<Result<_>>()?;
        let torch = TorchNetwork {
            version,
            device: weight_device,
            spectral,
            bands,
            projections,
            recurrent,
            masks,
            mask_config,
        };
        let backend = match acceleration {
            Acceleration::Torch => Backend::Torch(Box::new(torch)),
            Acceleration::Mlx(dtype) => {
                Backend::Mlx(Box::new(mlx_network::Network::load(&torch, device, dtype)?))
            }
            #[cfg(feature = "onnx")]
            Acceleration::Onnx(provider) => {
                Backend::Onnx(Box::new(onnx_network::Network::load(torch, provider)?))
            }
        };
        Ok(Self {
            stems: config.stems.len(),
            backend,
        })
    }

    pub(super) fn num_stems(&self) -> usize {
        self.stems
    }

    pub(super) fn predict_batch(&self, audio: &[f32], count: usize) -> Result<Vec<f32>> {
        let samples = super::super::window_samples(audio, count, CHANNELS)?;
        let result = match &self.backend {
            Backend::Torch(model) => no_grad(|| model.run(audio, count, samples))?,
            Backend::Mlx(model) => model.run(audio, count, samples as usize)?,
            #[cfg(feature = "onnx")]
            Backend::Onnx(model) => no_grad(|| model.run(audio, count, samples))?,
        };
        ensure!(
            result.iter().all(|sample| sample.is_finite()),
            "Bandit produced nonfinite audio"
        );
        Ok(result)
    }
}

impl TorchNetwork {
    fn run(&self, audio: &[f32], count: usize, samples: i64) -> Result<Vec<f32>> {
        let audio = Tensor::f_from_slice(audio)?.f_reshape([-1, samples])?;
        let spectrum = self.spectral.stft(&audio)?;
        let bins = spectrum.size()[1];
        let order = match self.version {
            Version::V1 => [0, 2, 3, 1],
            Version::V2 => [0, 2, 1, 3],
        };
        let inputs = spectrum
            .f_view_as_real()?
            .f_permute(order)?
            .f_to_device(self.device)?;
        let features = self
            .projections
            .iter()
            .map(|projection| projection.forward(&inputs, self.version))
            .collect::<Result<Vec<_>>>()?;
        let mut features = Tensor::f_stack(&features, 1)?;
        for layer in &self.recurrent {
            features = layer.forward(&features)?.f_transpose(1, 2)?;
        }
        let masks = self
            .masks
            .iter()
            .map(|stem| stem.forward(&features, &self.bands, &self.mask_config, bins));
        self.spectral.render(&spectrum, count, samples, masks)
    }
}
