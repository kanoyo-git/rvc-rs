use std::cell::RefCell;
use std::path::Path;

use anyhow::{Result, ensure};
use tch::{Device, Kind, Tensor};

use super::cuda_graph::Cache;
use super::weights::Weights;
use crate::audio;

pub struct Rmvpe {
    weights: Weights,
    features: MelSpectrogram,
    device: Device,
    kind: Kind,
    gru: Vec<Tensor>,
    pub(super) graph: RefCell<Cache>,
}

const FFT: i64 = 1024;
const HOP: i64 = 160;
pub(super) const MEL_BANDS: i64 = 128;
pub(super) const GRU_WIDTH: i64 = 256;
pub(super) const BINS: usize = 360;

pub(super) struct MelSpectrogram {
    filters: Tensor,
    window: Tensor,
}

impl MelSpectrogram {
    pub(super) fn new(device: Device) -> Result<Self> {
        Ok(Self {
            filters: Tensor::f_from_slice(&mel_filterbank())?
                .f_view([MEL_BANDS, FFT / 2 + 1])?
                .f_to_device(device)?,
            window: Tensor::f_hann_window(FFT, (Kind::Float, device))?,
        })
    }

    pub(super) fn input(&self, samples: &[f32]) -> Result<Tensor> {
        let padded = audio::reflect(samples, FFT as usize / 2, FFT as usize / 2);
        Ok(Tensor::f_from_slice(&padded)?
            .f_view([1, -1])?
            .f_to_device(self.window.device())?)
    }

    pub(super) fn extract(&self, x: &Tensor, kind: Kind, device: Device) -> Result<(Tensor, i64)> {
        let fft = x.f_stft(FFT, HOP, FFT, Some(&self.window), false, true, true, false)?;
        let spectrum = fft
            .f_real()?
            .f_square()?
            .f_add(&fft.f_imag()?.f_square()?)?
            .f_sqrt()?;
        let mut mel = self
            .filters
            .f_matmul(&spectrum)?
            .f_to_kind(kind)?
            .f_clamp_min(1e-5)?
            .f_log()?
            .f_to_device(device)?;
        let frames = mel.size()[2];
        let padding = (32 - frames.rem_euclid(32)) % 32;
        if padding != 0 {
            mel = mel.f_constant_pad_nd([0, padding])?;
        }
        Ok((mel, frames))
    }
}

impl Rmvpe {
    pub fn load(path: &Path, device: Device, kind: Kind) -> Result<Self> {
        let mut weights = Weights::load(path, None, Device::Cpu)?;
        weights.move_to(device, kind)?;
        weights.prepare_cpu_convolutions(device)?;
        let mut gru = Vec::with_capacity(8);
        for suffix in ["", "_reverse"] {
            for name in ["weight_ih", "weight_hh", "bias_ih", "bias_hh"] {
                gru.push(
                    weights
                        .get(&format!("fc.0.gru.{name}_l0{suffix}"))?
                        .shallow_clone(),
                );
            }
        }
        if matches!(device, Device::Cuda(_))
            && tch::Cuda::cudnn_is_available()
            && tch::Cuda::user_enabled_cudnn()
        {
            // cuDNN packs these eight tensors into shared storage in place.
            const GRU: i64 = 3;
            let _ = Tensor::f_internal_cudnn_rnn_flatten_weight(
                &gru,
                4,
                gru[0].size()[1],
                GRU,
                GRU_WIDTH,
                0,
                1,
                true,
                true,
            )?;
        }
        let spectral_device = if device == Device::Mps {
            Device::Cpu
        } else {
            device
        };
        Ok(Self {
            weights,
            features: MelSpectrogram::new(spectral_device)?,
            device,
            kind,
            gru,
            graph: RefCell::default(),
        })
    }

    pub(super) fn infer(
        &self,
        samples: &[f32],
        threshold: f32,
        graph_pool: Option<&Cache>,
    ) -> Result<Vec<f32>> {
        let x = self.features.input(samples)?;
        let salience = if let Some(pool) = graph_pool {
            self.graph
                .borrow_mut()
                .run([0; 4], &[&x], Some(pool), |args| self.infer_tensor(args[0]))?
        } else {
            self.infer_tensor(&x)?
        };
        let salience = Vec::<f32>::try_from(salience.to_device(Device::Cpu))?;
        decode(&salience, threshold)
    }

    fn infer_tensor(&self, x: &Tensor) -> Result<Tensor> {
        let (mel, frames) = self.features.extract(x, self.kind, self.device)?;
        Ok(self
            .salience(&mel)?
            .narrow(1, 0, frames)
            .contiguous()
            .view([-1]))
    }

    fn salience(&self, mel: &Tensor) -> Result<Tensor> {
        let w = &self.weights;
        let mut x = self.batch_norm(&mel.transpose(1, 2).unsqueeze(1), "unet.encoder.bn")?;
        let mut skip_connections = Vec::new();
        for i in 0..5 {
            x = self.residual_stack(x, &format!("unet.encoder.layers.{i}.conv"))?;
            skip_connections.push(x.shallow_clone());
            x = x.avg_pool2d([2, 2], [2, 2], [0, 0], false, true, None);
        }
        for i in 0..4 {
            x = self.residual_stack(x, &format!("unet.intermediate.layers.{i}.conv"))?;
        }
        for (i, skip) in skip_connections.into_iter().rev().enumerate() {
            let prefix = format!("unet.decoder.layers.{i}");
            x = x.f_conv_transpose2d(
                w.get(&format!("{prefix}.conv1.0.weight"))?,
                None::<&Tensor>,
                [2, 2],
                [1, 1],
                [1, 1],
                1,
                [1, 1],
            )?;
            x = self.batch_norm(&x, &format!("{prefix}.conv1.1"))?.relu();
            x = self.residual_stack(Tensor::cat(&[x, skip], 1), &format!("{prefix}.conv2"))?;
        }
        x = w.conv2(&x, "cnn", 1)?.transpose(1, 2).flatten(2, 3);
        let state = Tensor::zeros([2, 1, GRU_WIDTH], (self.kind, self.device));
        let (x, _) = x.f_gru(&state, &self.gru, true, 1, 0.0, false, true, true)?;
        Ok(w.linear(&x, "fc.1")?.sigmoid())
    }

    fn residual_stack(&self, mut x: Tensor, prefix: &str) -> Result<Tensor> {
        for i in 0..4 {
            let layer = format!("{prefix}.{i}");
            let hidden = self.weights.conv2(&x, &format!("{layer}.conv.0"), 1)?;
            let hidden = self.batch_norm(&hidden, &format!("{layer}.conv.1"))?.relu();
            let hidden = self.weights.conv2(&hidden, &format!("{layer}.conv.3"), 1)?;
            let hidden = self.batch_norm(&hidden, &format!("{layer}.conv.4"))?.relu();
            if self.weights.contains(&format!("{layer}.shortcut.weight")) {
                x = self.weights.conv2(&x, &format!("{layer}.shortcut"), 0)?;
            }
            x = x.f_add(&hidden)?;
        }
        Ok(x)
    }

    fn batch_norm(&self, x: &Tensor, layer: &str) -> Result<Tensor> {
        let get = |field| self.weights.get(&format!("{layer}.{field}"));
        Ok(x.f_batch_norm(
            Some(get("weight")?),
            Some(get("bias")?),
            Some(get("running_mean")?),
            Some(get("running_var")?),
            false,
            0.01,
            1e-5,
            true,
        )?)
    }
}

pub(super) fn decode(salience: &[f32], threshold: f32) -> Result<Vec<f32>> {
    let (frames, rest) = salience.as_chunks::<BINS>();
    ensure!(rest.is_empty(), "RMVPE returned an incomplete pitch frame");
    Ok(frames
        .iter()
        .map(|row| local_average(row, threshold))
        .collect())
}

fn local_average(salience: &[f32], threshold: f32) -> f32 {
    let mut center = 0;
    for bin in 1..BINS {
        if salience[bin] > salience[center] {
            center = bin;
        }
    }
    if salience[center] <= threshold {
        return 0.0;
    }
    let first = center.saturating_sub(4);
    let last = (center + 4).min(BINS - 1);
    let mut weight = 0.0;
    let mut cents = 0.0;
    for (offset, &value) in salience[first..=last].iter().enumerate() {
        let value = f64::from(value);
        let bin = first + offset;
        weight += value;
        cents += value * (20.0 * bin as f64 + 1_997.379_408_437_619);
    }
    (10.0 * (cents / (weight * 1200.0)).exp2()) as f32
}

fn mel_filterbank() -> Vec<f32> {
    let to_mel = |hz: f64| 2595.0 * (1.0 + hz / 700.0).log10();
    let to_hz = |mel: f64| 700.0 * (10.0_f64.powf(mel / 2595.0) - 1.0);
    let low = to_mel(30.0);
    let step = (to_mel(8000.0) - low) / 129.0;
    let frequencies: Vec<_> = (0..130).map(|i| to_hz(low + f64::from(i) * step)).collect();
    let mut weights = Vec::with_capacity(128 * (FFT as usize / 2 + 1));
    for mel in frequencies.windows(3) {
        for bin in 0..=FFT / 2 {
            let hz = bin as f64 * 16_000.0 / FFT as f64;
            let lower = (hz - mel[0]) / (mel[1] - mel[0]);
            let upper = (mel[2] - hz) / (mel[2] - mel[1]);
            weights.push((lower.min(upper).max(0.0) * 2.0 / (mel[2] - mel[0])) as f32);
        }
    }
    weights
}
