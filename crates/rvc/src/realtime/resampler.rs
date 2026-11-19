use std::f32::consts::PI;

const FILTER_WIDTH: f32 = 6.0;
const ROLLOFF: f64 = 0.99;

/// Cached Hann-windowed sinc interpolation with the original realtime filter.
pub(super) struct Resampler {
    input_step: usize,
    output_step: usize,
    radius: usize,
    taps: Vec<f32>,
    output: Box<[f32]>,
}

impl Resampler {
    pub(super) fn new(from: u32, to: u32, max_input: usize) -> Self {
        let divisor = gcd(from, to);
        let input_step = (from / divisor) as usize;
        let output_step = (to / divisor) as usize;
        let output = vec![0.0; (max_input * output_step).div_ceil(input_step)].into_boxed_slice();
        if from == to {
            return Self {
                input_step,
                output_step,
                radius: 0,
                taps: Vec::new(),
                output,
            };
        }

        let cutoff = input_step.min(output_step) as f64 * ROLLOFF;
        let scale = (cutoff / input_step as f64) as f32;
        let radius = (f64::from(FILTER_WIDTH) / (cutoff / input_step as f64)).ceil() as usize;
        let mut taps = Vec::with_capacity(output_step * (2 * radius + 1));
        for phase in 0..output_step {
            let center = (phase * input_step / output_step) as isize;
            let position = phase as f32 / output_step as f32;
            // Each phase stores only the taps within reach of its center.
            for offset in center - radius as isize..=center + radius as isize {
                let distance = (offset as f32 / input_step as f32 - position) * cutoff as f32;
                let angle = distance.clamp(-FILTER_WIDTH, FILTER_WIDTH) * PI;
                let window = (angle / (2.0 * FILTER_WIDTH)).cos().powi(2);
                let sinc = if angle == 0.0 {
                    1.0
                } else {
                    angle.sin() / angle
                };
                taps.push(sinc * (window * scale));
            }
        }
        Self {
            input_step,
            output_step,
            radius,
            taps,
            output,
        }
    }

    pub(super) fn process(&mut self, input: &[f32]) -> &mut [f32] {
        let frames = (input.len() * self.output_step).div_ceil(self.input_step);
        let output = &mut self.output[..frames];
        if self.input_step == self.output_step {
            output.copy_from_slice(input);
            return output;
        }

        let taps_per_phase = self.radius * 2 + 1;
        for (index, sample) in output.iter_mut().enumerate() {
            let center = index * self.input_step / self.output_step;
            let phase = index % self.output_step;
            // Missing samples outside the capture or synthesis block are zero.
            let start = center.saturating_sub(self.radius);
            let end = (center + self.radius + 1).min(input.len());
            let tap = phase * taps_per_phase + self.radius + start - center;
            *sample = input[start..end]
                .iter()
                .zip(&self.taps[tap..tap + end - start])
                .map(|(signal, weight)| signal * weight)
                .sum();
        }
        output
    }
}

fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capture_and_playback_match_torchaudio_at_block_edges() {
        // Original torchaudio.transforms.Resample with dtype=torch.float32.
        let input = [1.0, 0.25, -0.5, 0.0, 0.75, -0.25, 0.2, 0.0, -1.0];
        let fixtures: [(u32, u32, &[f32]); 4] = [
            (
                44_100,
                16_000,
                &[0.323_331_6, 0.061_308_31, 0.110_275_32, -0.378_824_44],
            ),
            (48_000, 16_000, &[0.275_521_4, 0.118_503_35, -0.020_290_844]),
            (
                40_000,
                44_100,
                &[
                    0.993_983_27,
                    0.374_637_3,
                    -0.451_545_36,
                    -0.286_833_05,
                    0.697_430_1,
                    0.228_774_96,
                    -0.261_792_78,
                    0.403_356_05,
                    -0.371_863_13,
                    -0.927_745_76,
                ],
            ),
            (
                40_000,
                48_000,
                &[
                    0.993_983_27,
                    0.468_439_76,
                    -0.379_159_63,
                    -0.439_615_96,
                    0.420_238_47,
                    0.639_078_4,
                    -0.246_144_26,
                    0.046_768_91,
                    0.338_725_4,
                    -0.707_516_1,
                    -0.784_254_13,
                ],
            ),
        ];
        for (from, to, expected) in fixtures {
            let mut resampler = Resampler::new(from, to, input.len());
            let output = resampler.process(&input);
            assert_eq!(output.len(), expected.len());
            for (&actual, &expected) in output.iter().zip(expected) {
                assert!(
                    (actual - expected).abs() <= 2e-6,
                    "{from} -> {to}: actual {actual}, expected {expected}"
                );
            }
            let buffer = output.as_ptr();
            assert_eq!(resampler.process(&input[..1]).as_ptr(), buffer);
            assert!(resampler.process(&[]).is_empty());
            assert_eq!(resampler.process(&input).as_ptr(), buffer);
        }
        assert_eq!(
            Resampler::new(16_000, 16_000, input.len()).process(&input),
            input
        );
    }
}
