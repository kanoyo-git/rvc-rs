use anyhow::{Result, ensure};

use super::device::StreamFormat;

/// Synchronized overlap-add used by RVC's streaming converter. Feed it an
/// already resampled synthesis block followed by overlap and a search margin.
pub struct Sola {
    previous: Vec<f32>,
    fade: Vec<f32>,
    block: usize,
    search: usize,
}

impl Sola {
    pub fn new(format: StreamFormat, crossfade_ms: u32) -> Result<Self> {
        let grid = format.sample_rate as usize / 100;
        let crossfade = ((f64::from(crossfade_ms) / 10.0).round_ties_even() as usize).min(4);
        let overlap = crossfade * grid;
        ensure!(
            grid > 0 && format.block_frames > 0,
            "invalid SOLA frame size"
        );
        let fade = (0..overlap)
            .map(|index| {
                let phase =
                    std::f32::consts::FRAC_PI_2 * index as f32 / (overlap - 1).max(1) as f32;
                phase.sin().powi(2)
            })
            .collect();
        Ok(Self {
            previous: vec![0.0; overlap],
            fade,
            block: format.block_frames,
            search: if overlap == 0 { 0 } else { grid },
        })
    }

    pub fn synthesized_frames(&self) -> usize {
        self.block + self.previous.len() + self.search
    }

    pub fn reset(&mut self) {
        self.previous.fill(0.0);
    }

    pub fn apply(&mut self, synth: &[f32], output: &mut [f32]) -> Result<usize> {
        let overlap = self.previous.len();
        ensure!(
            synth.len() >= self.synthesized_frames() && output.len() == self.block,
            "SOLA requires {} synthesized frames and {} output frames",
            self.synthesized_frames(),
            self.block
        );
        let offset = self.best_offset(synth);
        let block = &synth[offset..offset + self.block + overlap];
        output.copy_from_slice(&block[..self.block]);
        for ((sample, &previous), &fade_in) in output.iter_mut().zip(&self.previous).zip(&self.fade)
        {
            *sample = *sample * fade_in + previous * (1.0 - fade_in);
        }
        for index in 0..overlap {
            let frame = self.block + index;
            self.previous[index] = if frame < overlap {
                let fade = self.fade[frame];
                block[frame] * fade + self.previous[frame] * (1.0 - fade)
            } else {
                block[frame]
            };
        }
        Ok(offset)
    }

    fn best_offset(&self, synth: &[f32]) -> usize {
        if self.previous.is_empty() {
            return 0;
        }
        let mut best = (f64::NEG_INFINITY, 0);
        for offset in 0..=self.search {
            let candidate = &synth[offset..offset + self.previous.len()];
            let (dot, energy) = candidate.iter().zip(&self.previous).fold(
                (0.0, 0.0),
                |(dot, energy), (&next, &previous)| {
                    let next = f64::from(next);
                    (dot + f64::from(previous) * next, energy + next * next)
                },
            );
            let score = dot / (energy + 1e-8).sqrt();
            if score > best.0 {
                best = (score, offset);
            }
        }
        best.1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aligns_and_blends_the_matching_overlap() -> Result<()> {
        let mut sola = Sola::new(
            StreamFormat {
                sample_rate: 300,
                block_frames: 4,
            },
            10,
        )?;
        let mut output = [0.0; 4];
        assert_eq!(
            sola.apply(&[0., 0., 0., 0., 1., -2., 3., 0., 0., 0.], &mut output)?,
            0
        );
        assert_eq!(
            sola.apply(&[0., 0., 1., -2., 3., 4., 0., 0., 0., 0.], &mut output)?,
            2
        );
        assert_eq!(output, [1., -2., 3., 4.]);
        sola.reset();
        assert_eq!(sola.apply(&[0.0; 10], &mut output)?, 0);
        Ok(())
    }

    #[test]
    fn accepts_a_disabled_crossfade() -> Result<()> {
        let format = StreamFormat::new(48_000, 20)?;
        let mut sola = Sola::new(format, 0)?;
        let synth = vec![0.25; format.block_frames];
        let mut output = vec![0.0; format.block_frames];
        assert_eq!(sola.apply(&synth, &mut output)?, 0);
        assert_eq!(output, synth);
        Ok(())
    }
}
