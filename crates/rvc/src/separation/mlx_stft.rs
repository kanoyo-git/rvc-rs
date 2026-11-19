use anyhow::{Context, Result, bail, ensure};
use tch::Tensor;

use crate::audio;

use super::mlx::{Array, DType, Stream};

pub(super) struct Stft {
    pub(super) bins: i32,
    fft: i32,
    hop: i32,
    normalized: bool,
    window: Array,
}

#[derive(Clone, Copy)]
pub(super) enum Padding {
    None,
    Reflect,
    Constant,
    Replicate,
    Circular,
}

impl Stft {
    pub(super) fn new(
        stream: &Stream,
        fft: i64,
        hop: i64,
        normalized: bool,
        window: &Tensor,
    ) -> Result<Self> {
        let fft = i32::try_from(fft)?;
        Ok(Self {
            bins: fft / 2 + 1,
            fft,
            hop: i32::try_from(hop)?,
            normalized,
            window: stream.upload(window, DType::F32)?,
        })
    }

    pub(super) fn analyze(&self, stream: &Stream, audio: &[f32], samples: usize) -> Result<Array> {
        self.analyze_with_padding(stream, audio, samples, Padding::Reflect)
    }

    pub(super) fn analyze_with_padding(
        &self,
        stream: &Stream,
        audio: &[f32],
        samples: usize,
        padding: Padding,
    ) -> Result<Array> {
        ensure!(
            samples > 0 && !audio.is_empty() && audio.len().is_multiple_of(samples),
            "invalid audio geometry for MLX STFT"
        );
        let side = if matches!(padding, Padding::None) {
            0
        } else {
            self.fft as usize / 2
        };
        match padding {
            Padding::Reflect => ensure!(samples > side, "MLX STFT reflection exceeds the input"),
            Padding::Circular => ensure!(
                samples >= side,
                "MLX STFT circular padding exceeds the input"
            ),
            _ => {}
        }
        let tracks = i32::try_from(audio.len() / samples)?;
        let length = i32::try_from(samples)?
            .checked_add(i32::try_from(2 * side)?)
            .context("MLX audio length overflow")?;
        ensure!(length >= self.fft, "MLX STFT input is shorter than the FFT");
        let frames = (length - self.fft) / self.hop + 1;
        let capacity = (tracks as usize)
            .checked_mul(length as usize)
            .context("MLX audio buffer overflow")?;
        let mut padded = Vec::with_capacity(capacity);
        for track in audio.chunks_exact(samples) {
            match padding {
                Padding::None => padded.extend_from_slice(track),
                Padding::Reflect => padded.extend(audio::reflect(track, side, side)),
                Padding::Constant => {
                    padded.resize(padded.len() + side, 0.0);
                    padded.extend_from_slice(track);
                    padded.resize(padded.len() + side, 0.0);
                }
                Padding::Replicate => {
                    padded.extend(std::iter::repeat_n(track[0], side));
                    padded.extend_from_slice(track);
                    padded.extend(std::iter::repeat_n(track[samples - 1], side));
                }
                Padding::Circular => {
                    padded.extend_from_slice(&track[samples - side..]);
                    padded.extend_from_slice(track);
                    padded.extend_from_slice(&track[..side]);
                }
            }
        }
        let padded = stream.float32(&padded, &[tracks, length])?;
        let framed = stream.strided(
            &padded,
            &[tracks, frames, self.fft],
            &[i64::from(length), i64::from(self.hop), 1],
        )?;
        stream.fft(
            &stream.mul(&framed, &self.window)?,
            self.fft,
            self.normalized,
        )
    }

    pub(super) fn synthesize(
        &self,
        stream: &Stream,
        spectrum: &Array,
        samples: Option<usize>,
    ) -> Result<Array> {
        self.synthesize_centered(stream, spectrum, samples, true)
    }

    pub(super) fn synthesize_centered(
        &self,
        stream: &Stream,
        spectrum: &Array,
        samples: Option<usize>,
        centered: bool,
    ) -> Result<Array> {
        let &[tracks, frames, bins] = spectrum.shape() else {
            bail!("invalid MLX spectrum rank")
        };
        ensure!(
            frames > 0 && bins == self.bins,
            "invalid MLX spectrum shape"
        );
        let time = stream.ifft(spectrum, self.fft, self.normalized)?;
        let time = stream.mul(&time, &self.window)?;
        let flattened = frames
            .checked_mul(self.fft)
            .context("MLX overlap size overflow")?;
        let hop = stream.scalar(self.hop as f32, DType::I32)?;
        let offsets = stream.mul(&stream.arange(frames, DType::I32)?, &hop)?;
        let offsets = stream.reshape(&offsets, &[frames, 1])?;
        let positions = stream.add(&offsets, &stream.arange(self.fft, DType::I32)?)?;
        let positions = stream.reshape(&positions, &[1, flattened])?;
        let length = (frames - 1)
            .checked_mul(self.hop)
            .and_then(|n| n.checked_add(self.fft))
            .context("MLX output length overflow")?;
        let indices = stream.broadcast(&positions, &[tracks, flattened])?;
        let time = stream.reshape(&time, &[tracks, flattened])?;
        let output = stream.scatter_add(&stream.zeros(&[tracks, length])?, &indices, &time, 1)?;
        let weight = stream.broadcast(&stream.square(&self.window)?, &[frames, self.fft])?;
        let weight = stream.reshape(&weight, &[1, flattened])?;
        let weight = stream.scatter_add(&stream.zeros(&[1, length])?, &positions, &weight, 1)?;
        let weight = stream.max(&weight, &stream.scalar(1e-11, DType::F32)?)?;
        let audio = stream.div(&output, &weight)?;
        let side = if centered { self.fft / 2 } else { 0 };
        let requested = if let Some(samples) = samples {
            side.checked_add(i32::try_from(samples)?)
                .context("MLX output length overflow")?
        } else {
            length - side
        };
        let end = requested.min(length);
        ensure!(end > side, "empty MLX inverse STFT");
        let audio = stream.slice(&audio, 1, side, end, 1)?;
        if end < requested {
            return stream.concat(&[audio, stream.zeros(&[tracks, requested - end])?], 1);
        }
        Ok(audio)
    }
}
