use anyhow::Result;
use tch::{Kind, Tensor};

use super::precision::Precision;
use crate::model::discriminator::Discriminator;
use crate::model::synth::{
    Synth,
    training::{Output as Generated, segments},
};
use crate::model::train_data::Batch;
use crate::model::train_losses::{
    self, MelSpectrogram, MultiResolutionStft, MultiScaleMel, PhoneticLoss,
};

#[derive(Clone, Copy)]
pub(super) struct Settings {
    pub precision: Precision,
    pub checkpointing: bool,
    pub segment_frames: i64,
    pub hop_length: i64,
    pub segment_size: i64,
    pub c_mel: f64,
    pub c_kl: f64,
    pub c_stft: f64,
    pub c_phonetic: f64,
}

/// Optional generator losses beyond the RVC mel/KL/adversarial terms.
#[derive(Default)]
pub(super) struct Auxiliary {
    pub stft: Option<MultiResolutionStft>,
    pub phonetic: Option<PhoneticLoss>,
}

pub(super) struct Forward {
    generated: Generated,
    source: Tensor,
    spectrum: Tensor,
}

pub(super) struct DiscriminatorLoss {
    pub total: Tensor,
    pub real_scores: Vec<Tensor>,
    pub generated_scores: Vec<Tensor>,
}

pub(super) struct GeneratorLoss {
    pub total: Tensor,
    pub mel: Tensor,
    pub stft: Option<Tensor>,
    pub phonetic: Option<Tensor>,
    pub kl: Tensor,
    pub feature: Tensor,
    pub scores: Vec<Tensor>,
    pub real_mel: Tensor,
    pub generated_mel: Tensor,
    pub full_mel: Tensor,
}

impl Settings {
    pub(super) fn generate(&self, model: &Synth, batch: &Batch) -> Result<Forward> {
        let generated = self
            .precision
            .run(|| model.train_batch(batch, self.segment_frames, self.checkpointing))?;
        let source = segments(
            &batch.wave,
            &(&generated.starts * self.hop_length),
            self.segment_size,
        )?;
        Ok(Forward {
            generated,
            source,
            spectrum: batch.spec.shallow_clone(),
        })
    }

    pub(super) fn discriminate(
        &self,
        model: &mut Discriminator,
        forward: &Forward,
    ) -> Result<DiscriminatorLoss> {
        let (real, generated) = self
            .precision
            .run(|| model.forward(&forward.source, &forward.generated.audio.detach(), true))?;
        let total = train_losses::discriminator_loss(&real.scores, &generated.scores)?;
        Ok(DiscriminatorLoss {
            total,
            real_scores: real.scores,
            generated_scores: generated.scores,
        })
    }

    pub(super) fn generator_loss(
        &self,
        model: &mut Discriminator,
        forward: Forward,
        mel: &MelSpectrogram,
        multiscale: Option<&MultiScaleMel>,
        auxiliary: &Auxiliary,
    ) -> Result<GeneratorLoss> {
        let Forward {
            generated,
            source,
            spectrum,
        } = forward;
        let (real, synthetic) = self
            .precision
            .run(|| model.forward(&source, &generated.audio, true))?;
        let full_mel = self.precision.run(|| mel.spec_to_mel(&spectrum))?;
        let real_mel = segments(&full_mel, &generated.starts, self.segment_frames)?;
        let spectrogram =
            || mel.mel_spectrogram(&generated.audio.f_to_kind(Kind::Float)?.squeeze_dim(1));
        let generated_mel = if multiscale.is_some() {
            tch::no_grad(spectrogram)?
        } else {
            spectrogram()?
        };
        let generated_mel = generated_mel.f_to_kind(self.precision.kind())?;
        let mel = if let Some(multiscale) = multiscale {
            multiscale.loss(&source, &generated.audio)? * (self.c_mel / 3.0)
        } else {
            // BF16 convolution is supported by CUDA. Compute the reduction in FP32.
            let (real, generated) = if self.precision == Precision::Bf16 {
                (
                    real_mel.f_to_kind(Kind::Float)?,
                    generated_mel.f_to_kind(Kind::Float)?,
                )
            } else {
                (real_mel.shallow_clone(), generated_mel.shallow_clone())
            };
            real.f_l1_loss(&generated, tch::Reduction::Mean)? * self.c_mel
        };
        let kl = train_losses::kl_loss(
            &generated.z_p,
            &generated.logs_q,
            &generated.m_p,
            &generated.logs_p,
            &generated.z_mask,
        )? * self.c_kl;
        let feature = train_losses::feature_loss(&real.features, &synthetic.features)?;
        let gan = train_losses::generator_loss(&synthetic.scores)?;
        let stft = auxiliary
            .stft
            .as_ref()
            .filter(|_| self.c_stft > 0.0)
            .map(|stft| -> Result<Tensor> {
                Ok(stft.loss(&source, &generated.audio)? * self.c_stft)
            })
            .transpose()?;
        let phonetic = auxiliary
            .phonetic
            .as_ref()
            .filter(|_| self.c_phonetic > 0.0)
            .map(|phonetic| -> Result<Tensor> {
                Ok(self
                    .precision
                    .run(|| phonetic.loss(&source, &generated.audio))?
                    * self.c_phonetic)
            })
            .transpose()?;
        let mut total = &mel + &kl + &feature + &gan;
        for extra in [&stft, &phonetic].into_iter().flatten() {
            total += extra;
        }
        Ok(GeneratorLoss {
            total,
            mel,
            stft,
            phonetic,
            kl,
            feature,
            scores: synthetic.scores,
            real_mel,
            generated_mel,
            full_mel,
        })
    }
}
