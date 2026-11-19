use std::collections::HashMap;
use std::ops::ControlFlow;

use anyhow::{Context, Result, ensure};
use tch::{Device, Tensor};

use crate::model::discriminator::Discriminator;
use crate::model::optimizer::{self, AdamW, GradientStep};
use crate::model::synth::{Synth, training::generator};
use crate::model::train_data::{Batch, Config, Dataset, Version};
use crate::model::train_losses::{MelConfig, MelSpectrogram};

use super::{
    notify,
    step::{Forward, Settings},
};

pub(super) struct Replica {
    device: Device,
    generator: Synth,
    generator_state: Vec<(String, Tensor)>,
    discriminator: Discriminator,
    discriminator_state: Vec<(String, Tensor)>,
    discriminator_parameters: Vec<(String, Tensor)>,
    mel: MelSpectrogram,
    batches: Vec<Vec<usize>>,
    cache: Vec<Batch>,
}

pub(super) struct Initialize<'a> {
    pub config: &'a Config,
    pub version: Version,
    pub f0: bool,
    pub mel: MelConfig,
    pub generator: &'a [(String, Tensor)],
    pub discriminator: &'a [(String, Tensor)],
}

impl Replica {
    pub(super) fn new(device: Device, init: &Initialize<'_>) -> Result<Self> {
        let (generator, generator_state) = generator(
            init.config,
            init.version,
            init.f0,
            device,
            detached(init.generator, device)?,
            true,
        )?;
        let discriminator = Discriminator::from_weights(
            matches!(init.version, Version::V2),
            init.config.model.use_spectral_norm,
            device,
            detached(init.discriminator, device)?,
        )?;
        let discriminator_state = discriminator.state_dict();
        let discriminator_parameters = discriminator.parameters();
        Ok(Self {
            device,
            generator,
            generator_state,
            discriminator,
            discriminator_state,
            discriminator_parameters,
            mel: MelSpectrogram::new(init.mel, device)?,
            batches: Vec::new(),
            cache: Vec::new(),
        })
    }

    pub(super) fn prepare_epoch(
        &mut self,
        dataset: &Dataset,
        indices: Vec<Vec<usize>>,
        cache: bool,
        progress: &mut impl FnMut(&str) -> ControlFlow<()>,
    ) -> Result<usize> {
        if self.cache.is_empty() {
            self.batches = indices;
            if cache {
                for (index, indices) in self.batches.iter().enumerate() {
                    notify(
                        progress,
                        &format!(
                            "Caching batch {}/{} at {:?}",
                            index + 1,
                            self.batches.len(),
                            self.device
                        ),
                    )?;
                    self.cache.push(
                        dataset
                            .load_batch(indices, self.device, |wave| self.mel.spectrogram(wave))?,
                    );
                }
            }
        }
        Ok(if self.cache.is_empty() {
            self.batches.len()
        } else {
            self.cache.len()
        })
    }

    pub(super) fn has_cache(&self) -> bool {
        !self.cache.is_empty()
    }

    pub(super) fn discriminator_phase(
        &mut self,
        dataset: &Dataset,
        index: usize,
        settings: &Settings,
        scale: Option<f32>,
    ) -> Result<(Forward, Vec<Tensor>)> {
        let loaded;
        let batch = if let Some(batch) = self.cache.get(index) {
            batch
        } else {
            loaded = dataset.load_batch(&self.batches[index], self.device, |wave| {
                self.mel.spectrogram(wave)
            })?;
            &loaded
        };
        let forward = settings.generate(&self.generator, batch)?;
        let loss = settings.discriminate(&mut self.discriminator, &forward)?;
        let gradients = gradients(&loss.total, &self.discriminator_parameters, scale)?;
        Ok((forward, gradients))
    }

    pub(super) fn generator_phase(
        &mut self,
        forward: Forward,
        settings: &Settings,
        scale: Option<f32>,
    ) -> Result<Vec<Tensor>> {
        let loss = settings.generator_loss(&mut self.discriminator, forward, &self.mel)?;
        gradients(&loss.total, &self.generator_state, scale)
    }

    pub(super) fn synchronize_discriminator(&mut self, state: &[(String, Tensor)]) -> Result<()> {
        copy_state(&mut self.discriminator_state, state)
    }

    pub(super) fn synchronize_generator(&mut self, state: &[(String, Tensor)]) -> Result<()> {
        copy_state(&mut self.generator_state, state)
    }
}

fn detached(state: &[(String, Tensor)], device: Device) -> Result<HashMap<String, Tensor>> {
    state
        .iter()
        .map(|(name, value)| {
            Ok((
                name.clone(),
                value
                    .f_detach()?
                    .f_to_device_(device, value.kind(), false, true)?,
            ))
        })
        .collect()
}

fn copy_state(target: &mut [(String, Tensor)], source: &[(String, Tensor)]) -> Result<()> {
    ensure!(
        target.len() == source.len(),
        "training replica state has a different length"
    );
    tch::no_grad(|| {
        for ((name, value), (source_name, source)) in target.iter_mut().zip(source) {
            ensure!(
                name == source_name,
                "training replica parameter {name} disagrees with {source_name}"
            );
            value
                .f_copy_(source)
                .with_context(|| format!("synchronizing training parameter {name}"))?;
        }
        Ok(())
    })
}

fn gradients(
    loss: &Tensor,
    parameters: &[(String, Tensor)],
    scale: Option<f32>,
) -> Result<Vec<Tensor>> {
    let leaves: Vec<_> = parameters.iter().map(|(_, value)| value).collect();
    optimizer::backward(loss, &leaves, scale)
}

pub(super) fn optimize(
    optimizer: &mut AdamW,
    loss: &Tensor,
    peers: Vec<Vec<Tensor>>,
    scale: Option<f32>,
    metric: bool,
) -> Result<GradientStep> {
    if peers.is_empty() && scale.is_none() && !metric {
        optimizer.backward_step(loss)?;
        return Ok(GradientStep {
            overflow: false,
            norm: 0.0,
        });
    }
    let gradients = optimizer.gradients(loss, scale)?;
    let gradients = if peers.is_empty() {
        gradients
    } else {
        average(gradients, peers, loss.device())?
    };
    optimizer.step(gradients, scale, metric)
}

fn average(mut sums: Vec<Tensor>, peers: Vec<Vec<Tensor>>, device: Device) -> Result<Vec<Tensor>> {
    let replicas = (peers.len() + 1) as f64;
    for gradients in peers {
        ensure!(
            gradients.len() == sums.len(),
            "training replica returned a different number of gradients"
        );
        for (sum, gradient) in sums.iter_mut().zip(gradients) {
            if gradient.defined() {
                let gradient = gradient.f_to_device(device)?;
                if sum.defined() {
                    *sum = sum.f_add(&gradient)?;
                } else {
                    *sum = gradient;
                }
            }
        }
    }
    for sum in sums.iter_mut().filter(|tensor| tensor.defined()) {
        *sum = sum.f_div_scalar(replicas)?;
    }
    Ok(sums)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::Kind;

    #[test]
    fn averages_gradients_from_independent_replicas() -> Result<()> {
        let first = Tensor::f_from_slice(&[1.0f32, 2.0])?.f_set_requires_grad(true)?;
        let second = Tensor::f_from_slice(&[1.0f32, 2.0])?.f_set_requires_grad(true)?;
        let loss_first = first.f_square()?.f_sum(Kind::Float)?;
        let loss_second = second.f_mul_scalar(4.0)?.f_sum(Kind::Float)?;
        let avg = average(
            optimizer::backward(&loss_first, &[&first], None)?,
            vec![optimizer::backward(&loss_second, &[&second], None)?],
            Device::Cpu,
        )?;
        assert_eq!(Vec::<f32>::try_from(&avg[0])?, [3.0, 4.0]);
        Ok(())
    }
}
