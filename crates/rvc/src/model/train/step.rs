use anyhow::Result;
use tch::{Kind, Tensor};

use crate::model::discriminator::Discriminator;
use crate::model::synth::{
    Synth,
    training::{Output as Generated, segments},
};
use crate::model::train_data::Batch;
use crate::model::train_losses::{self, MelSpectrogram};

#[derive(Clone, Copy)]
pub(super) struct Settings {
    pub amp: bool,
    pub segment_frames: i64,
    pub hop_length: i64,
    pub segment_size: i64,
    pub c_mel: f64,
    pub c_kl: f64,
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
    pub kl: Tensor,
    pub feature: Tensor,
    pub scores: Vec<Tensor>,
    pub real_mel: Tensor,
    pub generated_mel: Tensor,
    pub full_mel: Tensor,
}

impl Settings {
    pub(super) fn generate(&self, model: &Synth, batch: &Batch) -> Result<Forward> {
        let generated = tch::autocast(self.amp, || model.train_batch(batch, self.segment_frames))?;
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
        let (real, generated) = tch::autocast(self.amp, || {
            model.forward(&forward.source, &forward.generated.audio.detach(), true)
        })?;
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
    ) -> Result<GeneratorLoss> {
        let Forward {
            generated,
            source,
            spectrum,
        } = forward;
        let (real, synthetic) =
            tch::autocast(self.amp, || model.forward(&source, &generated.audio, true))?;
        let full_mel = tch::autocast(self.amp, || mel.spec_to_mel(&spectrum))?;
        let real_mel = segments(&full_mel, &generated.starts, self.segment_frames)?;
        let generated_mel =
            mel.mel_spectrogram(&generated.audio.f_to_kind(Kind::Float)?.squeeze_dim(1))?;
        let generated_mel = if self.amp {
            generated_mel.f_to_kind(Kind::Half)?
        } else {
            generated_mel
        };
        let mel = real_mel.f_l1_loss(&generated_mel, tch::Reduction::Mean)? * self.c_mel;
        let kl = train_losses::kl_loss(
            &generated.z_p,
            &generated.logs_q,
            &generated.m_p,
            &generated.logs_p,
            &generated.z_mask,
        )? * self.c_kl;
        let feature = train_losses::feature_loss(&real.features, &synthetic.features)?;
        let gan = train_losses::generator_loss(&synthetic.scores)?;
        let total = &mel + &kl + &feature + &gan;
        Ok(GeneratorLoss {
            total,
            mel,
            kl,
            feature,
            scores: synthetic.scores,
            real_mel,
            generated_mel,
            full_mel,
        })
    }
}
