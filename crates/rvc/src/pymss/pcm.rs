use anyhow::{Result, ensure};

use crate::audio;

pub struct Pcm {
    pub planar: Vec<f32>,
    pub channels: usize,
    pub rate: u32,
}

impl Pcm {
    pub fn from_interleaved(data: &[f32], channels: usize, rate: u32) -> Result<Self> {
        ensure!(
            matches!(channels, 1 | 2) && (1..=384_000).contains(&rate),
            "expected mono or stereo PCM"
        );
        ensure!(
            !data.is_empty() && data.len().is_multiple_of(channels),
            "invalid PCM length"
        );
        ensure!(
            data.iter().all(|sample| sample.is_finite()),
            "PCM contains non-finite samples"
        );
        let frames = data.len() / channels;
        let mut planar = Vec::with_capacity(data.len());
        for ch in 0..channels {
            planar.extend((0..frames).map(|frame| data[frame * channels + ch]));
        }
        Ok(Self {
            planar,
            channels,
            rate,
        })
    }

    pub fn interleaved(&self) -> Vec<f32> {
        let frame_len = self.planar.len() / self.channels;
        (0..frame_len)
            .flat_map(|frame| (0..self.channels).map(move |ch| self.planar[ch * frame_len + frame]))
            .collect()
    }

    pub fn duration(&self) -> f64 {
        self.planar.len() as f64 / self.channels as f64 / f64::from(self.rate)
    }

    pub fn convert_layout(&self, rate: u32, channels: usize) -> Result<Self> {
        ensure!(matches!(channels, 1 | 2), "unsupported model channel count");
        let frame_len = self.planar.len() / self.channels;
        let mut resampled = Vec::new();
        if self.channels == 2 && channels == 1 {
            let (left, right) = self.planar.split_at(frame_len);
            let mono: Vec<_> = left.iter().zip(right).map(|(a, b)| (a + b) * 0.5).collect();
            resampled = audio::resample(&mono, self.rate, rate);
        } else {
            for ch in 0..channels {
                let start = ch.min(self.channels - 1) * frame_len;
                resampled.extend(audio::resample(
                    &self.planar[start..start + frame_len],
                    self.rate,
                    rate,
                ));
            }
        }
        Ok(Self {
            planar: resampled,
            channels,
            rate,
        })
    }
}
