//! Inference for the bundled convolution-only torchfcpe 0.0.4 checkpoint.
//! Model and signal processing follow FCPE by CN_ChiTu (MIT).

use std::cell::RefCell;
use std::path::Path;

use anyhow::{Context, Result, ensure};
use tch::{Device, Kind, Tensor, no_grad};

use super::cuda_graph::Cache;
use super::weights::Weights;
use crate::checkpoint::Checkpoint;

const SAMPLE_RATE: i64 = 16_000;
const FFT: i64 = 1024;
const HOP: i64 = 160;
pub(super) const MEL_BANDS: i64 = 128;
pub(super) const CHANNELS: i64 = 512;
pub(super) const CONFORMER_LAYERS: i64 = 6;
pub(super) const PITCH_BINS: i64 = 360;

enum Backend {
    Torch(Weights),
    #[cfg(feature = "onnx")]
    Onnx(super::fcpe_onnx::FcpeOnnx),
}

pub struct Fcpe {
    backend: Backend,
    mel: Tensor,
    window: Tensor,
    cents: Tensor,
    offsets: Tensor,
    device: Device,
    pub(super) graph: RefCell<Cache>,
}

impl Fcpe {
    pub fn load(path: &Path, device: Device) -> Result<Self> {
        let onnx = super::onnx_selection("RVC_FCPE_ONNX");
        #[cfg(not(feature = "onnx"))]
        if let Some(provider) = &onnx {
            anyhow::bail!(
                "FCPE provider {} requires the onnx build feature",
                provider.to_string_lossy()
            );
        }
        let device = if onnx.is_some() { Device::Cpu } else { device };
        let mut checkpoint = Checkpoint::open(path)?;
        let config = checkpoint
            .metadata("config_dict")?
            .context("FCPE checkpoint has no configuration")?;
        let model = &config["model"];
        let mel = &config["mel"];
        ensure!(
            model["type"].as_str() == Some("CFNaiveMelPE")
                && model["conv_only"].as_bool() == Some(true)
                && model["hidden_dims"].as_i64() == Some(CHANNELS)
                && model["n_layers"].as_i64() == Some(CONFORMER_LAYERS)
                && model["out_dims"].as_i64() == Some(PITCH_BINS)
                && mel["sr"].as_i64() == Some(SAMPLE_RATE)
                && mel["n_fft"].as_i64() == Some(FFT)
                && mel["win_size"].as_i64() == Some(FFT)
                && mel["hop_size"].as_i64() == Some(HOP)
                && mel["num_mels"].as_i64() == Some(MEL_BANDS)
                && mel["fmin"].as_f64() == Some(0.0)
                && mel["fmax"].as_f64() == Some(SAMPLE_RATE as f64 / 2.0)
                && (mel["type"].is_null() || mel["type"].as_str() == Some("default")),
            "expected the torchfcpe 0.0.4 bundled convolution-only configuration"
        );
        let mut weights = Weights::new(checkpoint.weights(Some("model"), device)?);
        weights.remove_weight_norm(0)?;
        let cents = weights.get("cent_table")?.f_reshape([1, 1, PITCH_BINS])?;
        let spectral_device = if device == Device::Mps {
            Device::Cpu
        } else {
            device
        };
        let mel = Tensor::f_from_slice(&mel_filterbank())?
            .f_reshape([MEL_BANDS, FFT / 2 + 1])?
            .f_to_device(spectral_device)?;
        let window = Tensor::f_hann_window_periodic(FFT, true, (Kind::Float, spectral_device))?;
        let offsets = Tensor::f_arange_start(-4, 5, (Kind::Int64, device))?;
        let backend = if let Some(provider) = onnx {
            #[cfg(feature = "onnx")]
            {
                let provider = provider.to_str().context("RVC_FCPE_ONNX must be text")?;
                Backend::Onnx(super::fcpe_onnx::FcpeOnnx::load(&weights, provider)?)
            }
            #[cfg(not(feature = "onnx"))]
            {
                let _ = provider;
                unreachable!("unsupported FCPE provider was rejected");
            }
        } else {
            Backend::Torch(weights)
        };
        Ok(Self {
            backend,
            mel,
            window,
            cents,
            offsets,
            device,
            graph: RefCell::default(),
        })
    }

    /// Estimate F0 at 10 ms intervals from 16 kHz mono samples.
    /// The output includes the terminal frame used by torchfcpe.
    pub(super) fn infer(
        &self,
        samples: &[f32],
        threshold: f32,
        stream: Option<&Cache>,
    ) -> Result<Vec<f32>> {
        ensure!(
            !samples.is_empty(),
            "FCPE requires at least one audio sample"
        );
        ensure!(
            samples.iter().all(|sample| sample.is_finite()) && threshold.is_finite(),
            "FCPE requires finite samples and threshold"
        );
        no_grad(|| {
            let waveform = Tensor::f_from_slice(samples)?
                .f_reshape([1, -1])?
                .f_to_device(self.window.device())?;
            let predict = |input: &[&Tensor]| {
                let mel = self.wav_to_mel(input[0])?;
                self.decode(&self.salience(&mel)?, threshold)
            };
            let pitch = if stream.is_some() && matches!(&self.backend, Backend::Torch(_)) {
                self.graph.borrow_mut().run(
                    [i64::from(threshold.to_bits()), 0, 0, 0],
                    &[&waveform],
                    stream,
                    predict,
                )?
            } else {
                predict(&[&waveform])?
            }
            .f_to_device(Device::Cpu)?;
            let pitch = Vec::<f32>::try_from(pitch.f_reshape([-1])?)?;
            ensure!(
                pitch.iter().all(|hz| hz.is_finite()),
                "FCPE produced nonfinite F0"
            );
            Ok(pitch)
        })
    }

    fn wav_to_mel(&self, waveform: &Tensor) -> Result<Tensor> {
        let count = waveform.size()[1];
        let target_frames = count / HOP + 1;
        let left = (FFT - HOP) / 2;
        let right = ((FFT - HOP + 1) / 2).max(FFT - count - left);
        let waveform = if right < count {
            waveform.f_reflection_pad1d([left, right])?
        } else {
            waveform.f_constant_pad_nd([left, right])?
        };
        let spectrum = waveform.f_stft_center(
            FFT,
            HOP,
            FFT,
            Some(&self.window),
            false,
            "reflect",
            false,
            true,
            true,
            false,
        )?;
        let magnitude = spectrum
            .f_real()?
            .f_square()?
            .f_add(&spectrum.f_imag()?.f_square()?)?
            .f_add_scalar(1e-9)?
            .f_sqrt()?;
        let mel = self
            .mel
            .f_matmul(&magnitude)?
            .f_clamp_min(1e-5)?
            .f_log()?
            .f_transpose(1, 2)?;
        let frames = mel.size()[1];
        let mel = if frames < target_frames {
            let last = mel.f_narrow(1, frames - 1, 1)?;
            Tensor::f_cat(&[mel, last], 1)?
        } else {
            mel.f_narrow(1, 0, target_frames)?
        };
        Ok(mel.f_to_device(self.device)?)
    }

    fn salience(&self, mel: &Tensor) -> Result<Tensor> {
        match &self.backend {
            Backend::Torch(weights) => self.salience_torch(weights, mel),
            #[cfg(feature = "onnx")]
            Backend::Onnx(model) => model.salience(mel),
        }
    }

    fn salience_torch(&self, weights: &Weights, mel: &Tensor) -> Result<Tensor> {
        let mut hidden = weights.conv(&mel.f_transpose(1, 2)?, "input_stack.0", 1, 1, 1, 1)?;
        hidden = hidden
            .f_group_norm(
                4,
                Some(weights.get("input_stack.1.weight")?),
                Some(weights.get("input_stack.1.bias")?),
                1e-5,
                true,
            )?
            .f_leaky_relu()?;
        hidden = weights
            .conv(&hidden, "input_stack.3", 1, 1, 1, 1)?
            .f_transpose(1, 2)?;
        for i in 0..CONFORMER_LAYERS {
            let prefix = format!("net.encoder_layers.{i}.conformer.net");
            let normalized = weights.norm(&hidden, &format!("{prefix}.0"), 1e-5)?;
            let expanded = weights
                .conv(
                    &normalized.f_transpose(1, 2)?,
                    &format!("{prefix}.2"),
                    1,
                    0,
                    1,
                    1,
                )?
                .f_glu(1)?;
            let filtered = weights
                .conv(
                    &expanded,
                    &format!("{prefix}.4.conv"),
                    1,
                    15,
                    1,
                    2 * CHANNELS,
                )?
                .f_silu()?;
            let projected = weights
                .conv(&filtered, &format!("{prefix}.6"), 1, 0, 1, 1)?
                .f_transpose(1, 2)?;
            hidden = hidden.f_add(&projected)?;
        }
        Ok(weights
            .linear(&weights.norm(&hidden, "norm", 1e-5)?, "output_proj")?
            .f_sigmoid()?)
    }

    fn decode(&self, salience: &Tensor, threshold: f32) -> Result<Tensor> {
        let (confidence, center) = salience.f_max_dim(-1, true)?;
        let local = center.f_add(&self.offsets)?.f_clamp(0, PITCH_BINS - 1)?;
        let weight = salience.f_gather(-1, &local, false)?;
        let cents = self
            .cents
            .f_expand_as(salience)?
            .f_gather(-1, &local, false)?;
        let sum = |tensor: &Tensor| tensor.f_sum_dim_intlist(&[-1i64][..], true, Kind::Float);
        let estimate = sum(&cents.f_mul(&weight)?)?.f_div(&sum(&weight)?)?;
        let estimate = estimate
            .f_masked_fill(&confidence.f_le(f64::from(threshold))?, f64::NEG_INFINITY)?
            .f_div_scalar(1200.0)?;
        Ok(Tensor::f_pow_scalar(2.0, &estimate)?.f_mul_scalar(10.0)?)
    }
}

fn mel_filterbank() -> Vec<f32> {
    let scale = 200.0 / 3.0;
    let crossover = 1000.0 / scale;
    let log_step = 6.4f64.ln() / 27.0;
    let max = crossover + (SAMPLE_RATE as f64 / 2000.0).ln() / log_step;
    let mel_step = max / (MEL_BANDS + 1) as f64;
    let hz_edges: Vec<_> = (0..MEL_BANDS + 2)
        .map(|i| {
            let mel = i as f64 * mel_step;
            if mel >= crossover {
                1_000.0 * ((mel - crossover) * log_step).exp()
            } else {
                mel * scale
            }
        })
        .collect();
    hz_edges
        .windows(3)
        .flat_map(|edge| {
            (0..=FFT / 2).map(move |bin| {
                let hz = bin as f64 * SAMPLE_RATE as f64 / FFT as f64;
                let up = (hz - edge[0]) / (edge[1] - edge[0]);
                let down = (edge[2] - hz) / (edge[2] - edge[1]);
                let weight = up.min(down).max(0.0) as f32;
                (f64::from(weight) * (2.0 / (edge[2] - edge[0]))) as f32
            })
        })
        .collect()
}
