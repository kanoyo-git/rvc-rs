use anyhow::{Result, ensure};

use super::{device::StreamFormat, noise::SpectralGate, resampler::Resampler, sola::Sola};

const FEATURE_RATE: u32 = 16_000;
const FEATURE_HOP: usize = (FEATURE_RATE / 100) as usize;

#[derive(Clone, Copy, Debug)]
pub struct RealtimeOptions {
    pub crossfade_ms: u32,
    pub extra_ms: u32,
    pub threshold_db: f32,
    pub rms_mix: f32,
}

impl Default for RealtimeOptions {
    fn default() -> Self {
        Self {
            crossfade_ms: 50,
            extra_ms: 2_500,
            threshold_db: -60.0,
            rms_mix: 0.0,
        }
    }
}

#[derive(Clone, Copy)]
enum HistoryUpdate {
    RefreshOverlap,
    CurrentBlock,
}

/// Rolling RVC audio context. The model is supplied by the caller and works on
/// 16 kHz mono audio, with skip and render lengths expressed in 10 ms frames.
pub struct RealtimeContext {
    format: StreamFormat,
    model_rate: u32,
    grid: usize,
    skip_frames: usize,
    render_frames: usize,
    threshold: Option<f32>,
    rms_mix: f32,
    input_history: Vec<f32>,
    feature_history: Vec<f32>,
    history_update: HistoryUpdate,
    resample_input: Vec<f32>,
    capture_resampler: Resampler,
    playback_resampler: Option<Resampler>,
    gate_tail: Vec<f32>,
    gate_audio: Vec<f32>,
    source_rms: Vec<f32>,
    synth_rms: Vec<f32>,
    input_noise: Option<InputNoise>,
    output_noise: Option<Vec<f32>>,
    noise_gate: Option<SpectralGate>,
    overlap: usize,
    sola: Sola,
}

struct InputNoise {
    history: Vec<f32>,
    tail: Vec<f32>,
}

impl RealtimeContext {
    pub fn new(format: StreamFormat, model_rate: u32, options: RealtimeOptions) -> Result<Self> {
        Self::with_history(format, model_rate, options, HistoryUpdate::RefreshOverlap)
    }

    /// Preserve the analysis already sent for earlier VST blocks.
    pub fn new_vst(
        format: StreamFormat,
        model_rate: u32,
        options: RealtimeOptions,
    ) -> Result<Self> {
        Self::with_history(format, model_rate, options, HistoryUpdate::CurrentBlock)
    }

    fn with_history(
        format: StreamFormat,
        model_rate: u32,
        options: RealtimeOptions,
        history_update: HistoryUpdate,
    ) -> Result<Self> {
        ensure!(
            matches!(model_rate, 32_000 | 40_000 | 48_000),
            "unsupported RVC model sample rate {model_rate} Hz"
        );
        ensure!(
            (8_000..=384_000).contains(&format.sample_rate)
                && format.sample_rate.is_multiple_of(100),
            "realtime RVC requires a device sample rate divisible by 100"
        );
        let grid = (format.sample_rate / 100) as usize;
        ensure!(
            (grid..=grid * 200).contains(&format.block_frames)
                && format.block_frames.is_multiple_of(grid),
            "realtime RVC requires a whole number of 10 ms frames"
        );
        ensure!(
            options.crossfade_ms <= 150 && options.extra_ms <= 5_000,
            "invalid realtime crossfade or context duration"
        );
        let block_frames = format.block_frames / grid;
        let crossfade = frames_for_ms(options.crossfade_ms);
        let skip_frames = frames_for_ms(options.extra_ms);
        let sola = Sola::new(format, options.crossfade_ms)?;
        let render_frames = sola.synthesized_frames() / grid;
        let search_frames = usize::from(crossfade > 0);
        let history_frames = block_frames + crossfade + search_frames + skip_frames;
        let max_resample_input = format.block_frames + 4 * grid;
        let mut context = Self {
            format,
            model_rate,
            grid,
            skip_frames,
            render_frames,
            threshold: None,
            rms_mix: 1.0,
            input_history: vec![0.0; history_frames * grid],
            feature_history: vec![0.0; history_frames * FEATURE_HOP],
            history_update,
            resample_input: vec![0.0; max_resample_input],
            capture_resampler: Resampler::new(format.sample_rate, FEATURE_RATE, max_resample_input),
            playback_resampler: (model_rate != format.sample_rate).then(|| {
                Resampler::new(
                    model_rate,
                    format.sample_rate,
                    render_frames * (model_rate / 100) as usize,
                )
            }),
            gate_tail: vec![0.0; 4 * grid],
            gate_audio: vec![0.0; format.block_frames + 4 * grid],
            source_rms: Vec::with_capacity(block_frames + 6),
            synth_rms: Vec::with_capacity(render_frames + 1),
            input_noise: None,
            output_noise: None,
            noise_gate: None,
            overlap: crossfade.min(4) * grid,
            sola,
        };
        context.set_levels(options.threshold_db, options.rms_mix)?;
        Ok(context)
    }

    pub fn set_levels(&mut self, threshold_db: f32, rms_mix: f32) -> Result<()> {
        ensure!(
            (-60.0..=0.0).contains(&threshold_db),
            "realtime threshold must be between -60 and 0 dB"
        );
        ensure!(
            (0.0..=1.0).contains(&rms_mix),
            "realtime RMS mix must be between 0 and 1"
        );
        self.threshold = (threshold_db > -60.0).then(|| 10.0_f32.powf(threshold_db / 20.0));
        self.rms_mix = rms_mix;
        Ok(())
    }

    pub fn reset(&mut self) {
        self.input_history.fill(0.0);
        self.feature_history.fill(0.0);
        self.gate_tail.fill(0.0);
        if let Some(input) = &mut self.input_noise {
            input.history.fill(0.0);
            input.tail.fill(0.0);
        }
        if let Some(output) = &mut self.output_noise {
            output.fill(0.0);
        }
        self.sola.reset();
    }

    pub fn set_noise_reduction(&mut self, input: bool, output: bool) -> Result<()> {
        if (input || output) && self.noise_gate.is_none() {
            self.noise_gate = Some(SpectralGate::new(self.format.sample_rate)?);
        }
        let history = self.input_history.len();
        if input {
            self.input_noise.get_or_insert_with(|| InputNoise {
                history: self.input_history.clone(),
                tail: vec![0.0; self.overlap],
            });
        } else {
            self.input_noise = None;
        }
        if output {
            self.output_noise.get_or_insert_with(|| vec![0.0; history]);
        } else {
            self.output_noise = None;
        }
        Ok(())
    }

    fn denoise_input(&mut self, updated: usize) -> Result<usize> {
        if let (Some(denoise), Some(gate)) = (&mut self.input_noise, &self.noise_gate) {
            let block = self.format.block_frames;
            let start = self.input_history.len() - self.overlap - block;
            let mut filtered = gate.apply(&self.input_history[start..], &self.input_history)?;
            let overlap = self.overlap;
            for (i, (sample, previous)) in filtered.iter_mut().zip(&denoise.tail).enumerate() {
                let fade = (std::f32::consts::FRAC_PI_2 * i as f32
                    / overlap.saturating_sub(1).max(1) as f32)
                    .sin()
                    .powi(2);
                *sample = *sample * fade + previous * (1.0 - fade);
            }
            denoise.history.copy_within(block.., 0);
            let start = denoise.history.len() - block;
            denoise.history[start..].copy_from_slice(&filtered[..block]);
            denoise.tail.copy_from_slice(&filtered[block..]);
            Ok(block)
        } else {
            Ok(updated)
        }
    }

    pub fn process(
        &mut self,
        input: &[f32],
        output: &mut [f32],
        discontinuity: bool,
        synth: impl FnOnce(&[f32], usize, usize) -> Result<Vec<f32>>,
    ) -> Result<()> {
        self.validate_block(input, output)?;
        if discontinuity {
            self.reset();
        }
        let updated_native = self.append_input(input);
        let updated_native = self.denoise_input(updated_native)?;
        self.update_feature_history(updated_native)?;
        let mut pcm = synth(&self.feature_history, self.skip_frames, self.render_frames)?;
        let required = self.render_frames * (self.model_rate / 100) as usize;
        ensure!(
            pcm.len() >= required,
            "RVC produced {} samples, expected at least {required}",
            pcm.len()
        );
        let required_native = self.sola.synthesized_frames();
        let converted = if let Some(resampler) = &mut self.playback_resampler {
            resampler.process(&pcm[..required])
        } else {
            &mut pcm[..required]
        };
        ensure!(
            converted.len() >= required_native,
            "output resampler produced {} samples, expected at least {required_native}",
            converted.len()
        );
        let converted = &mut converted[..required_native];
        if let (Some(reference), Some(gate)) = (&mut self.output_noise, &self.noise_gate) {
            let block = self.format.block_frames;
            reference.copy_within(block.., 0);
            let offset = reference.len() - block;
            reference[offset..].copy_from_slice(&converted[converted.len() - block..]);
            converted.copy_from_slice(&gate.apply(converted, reference)?);
        }
        if self.rms_mix < 1.0 {
            let start = self.skip_frames * self.grid;
            let input_history = self
                .input_noise
                .as_ref()
                .map_or(&self.input_history, |input| &input.history);
            rms(
                &input_history[start..start + required_native],
                self.grid,
                &mut self.source_rms,
            );
            rms(converted, self.grid, &mut self.synth_rms);
            for (index, sample) in converted.iter_mut().enumerate() {
                let input_level = interpolate_rms(&self.source_rms, index, self.grid);
                let output_level = interpolate_rms(&self.synth_rms, index, self.grid).max(1e-3);
                *sample *= (input_level / output_level).powf(1.0 - self.rms_mix);
            }
        }
        self.sola.apply(converted, output)?;
        Ok(())
    }

    /// Input monitoring uses the same threshold and crossfade as conversion.
    pub fn monitor(
        &mut self,
        input: &[f32],
        output: &mut [f32],
        discontinuity: bool,
    ) -> Result<()> {
        self.validate_block(input, output)?;
        if discontinuity {
            self.reset();
        }
        let updated = self.append_input(input);
        self.denoise_input(updated)?;
        let input_history = self
            .input_noise
            .as_ref()
            .map_or(&self.input_history, |input| &input.history);
        self.sola
            .apply(&input_history[self.skip_frames * self.grid..], output)?;
        Ok(())
    }

    fn validate_block(&self, input: &[f32], output: &[f32]) -> Result<()> {
        ensure!(
            input.len() == self.format.block_frames && output.len() == self.format.block_frames,
            "expected an input and output block of {} samples",
            self.format.block_frames
        );
        Ok(())
    }

    fn append_input(&mut self, input: &[f32]) -> usize {
        let block_len = self.format.block_frames;
        self.input_history.copy_within(block_len.., 0);
        let Some(threshold) = self.threshold else {
            let offset = self.input_history.len() - block_len;
            self.input_history[offset..].copy_from_slice(input);
            return block_len;
        };

        let frame = self.grid;
        let history_frames = self.gate_tail.len();
        self.gate_audio[..history_frames].copy_from_slice(&self.gate_tail);
        self.gate_audio[history_frames..].copy_from_slice(input);
        self.gate_tail
            .copy_from_slice(&self.gate_audio[block_len..]);
        rms(&self.gate_audio, frame, &mut self.source_rms);

        // RMS uses a centered four-period window. Two periods of history are
        // revised with lookahead from the new capture block.
        let gated = &self.gate_audio[2 * frame..];
        let replaced = gated.len().min(self.input_history.len());
        let gate_offset = gated.len() - replaced;
        let history_offset = self.input_history.len() - replaced;
        let destination = &mut self.input_history[history_offset..];
        destination.copy_from_slice(&gated[gate_offset..]);
        for (index, &energy) in self.source_rms[2..].iter().enumerate() {
            if energy >= threshold {
                continue;
            }
            let start = (index * frame).saturating_sub(frame / 2 + gate_offset);
            let end = ((index + 1) * frame)
                .saturating_sub(frame / 2 + gate_offset)
                .min(replaced);
            if start < end {
                destination[start..end].fill(0.0);
            }
        }
        gated.len()
    }

    fn update_feature_history(&mut self, updated_native: usize) -> Result<()> {
        let step = self.format.block_frames / self.grid * FEATURE_HOP;
        self.feature_history.copy_within(step.., 0);

        // Both modes give the resampler two preceding periods. The desktop
        // also refreshes an overlap in the rolling analysis buffer.
        let (native_len, tail_len) = match self.history_update {
            HistoryUpdate::RefreshOverlap => (
                updated_native + 2 * self.grid,
                ((updated_native / self.grid + 1) * FEATURE_HOP).min(self.feature_history.len()),
            ),
            HistoryUpdate::CurrentBlock => (self.format.block_frames + 2 * self.grid, step),
        };
        let input_history = self
            .input_noise
            .as_ref()
            .map_or(&self.input_history, |input| &input.history);
        let available = native_len.min(input_history.len());
        let padding = native_len - available;
        let source_offset = input_history.len() - available;
        let buffer = &mut self.resample_input[..native_len];
        buffer[..padding].fill(0.0);
        buffer[padding..].copy_from_slice(&input_history[source_offset..]);
        let resampled = self.capture_resampler.process(buffer);
        ensure!(
            resampled.len() >= tail_len,
            "input resampler produced {} samples, expected at least {tail_len}",
            resampled.len()
        );
        let start = self.feature_history.len() - tail_len;
        self.feature_history[start..].copy_from_slice(&resampled[resampled.len() - tail_len..]);
        Ok(())
    }
}

fn frames_for_ms(duration: u32) -> usize {
    (f64::from(duration) / 10.0).round_ties_even() as usize
}

fn rms(audio: &[f32], hop: usize, output: &mut Vec<f32>) {
    output.clear();
    let half_window = 2 * hop;
    let reciprocal = 1.0 / (4 * hop) as f64;
    let (mut left, mut right, mut sum) = (0usize, 0usize, 0.0_f64);
    for center in (0..=audio.len()).step_by(hop) {
        let start = center.saturating_sub(half_window);
        let end = (center + half_window).min(audio.len());
        for &sample in &audio[left..start] {
            sum -= f64::from(sample).powi(2);
        }
        for &sample in &audio[right..end] {
            sum += f64::from(sample).powi(2);
        }
        output.push((sum.max(0.0) * reciprocal).sqrt() as f32);
        (left, right) = (start, end);
    }
}

fn interpolate_rms(values: &[f32], index: usize, hop: usize) -> f32 {
    let left = index / hop;
    let fraction = (index % hop) as f32 / hop as f32;
    values[left] + fraction * (values[left + 1] - values[left])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_receives_the_original_10ms_context_geometry() -> Result<()> {
        let format = StreamFormat::new(48_000, 20)?;
        let mut context = RealtimeContext::new(
            format,
            40_000,
            RealtimeOptions {
                extra_ms: 100,
                rms_mix: 1.0,
                ..Default::default()
            },
        )?;
        let input = vec![0.3; format.block_frames];
        let mut output = vec![0.0; format.block_frames];
        context.process(&input, &mut output, false, |audio, skip, render| {
            assert_eq!((audio.len(), skip, render), (18 * 160, 10, 7));
            assert!(audio[..16 * 160 - 12].iter().all(|&x| x == 0.0));
            assert!((audio[17 * 160] - 0.3).abs() < 0.0002);
            Ok(vec![0.8; render * 400])
        })?;
        assert_eq!(output[0], 0.0);
        assert!(output[1] > 0.0);
        Ok(())
    }

    #[test]
    fn vst_preserves_the_previous_analysis_block() -> Result<()> {
        let format = StreamFormat::new(48_000, 20)?;
        let options = RealtimeOptions {
            extra_ms: 100,
            rms_mix: 1.0,
            ..Default::default()
        };
        let mut vst = RealtimeContext::new_vst(format, 48_000, options)?;
        let mut desktop = RealtimeContext::new(format, 48_000, options)?;
        let input_len = format.block_frames;
        let first = vec![0.5; input_len];
        let second = vec![-0.5; input_len];
        let mut output = vec![0.0; input_len];
        let block = 2 * FEATURE_HOP;

        let mut original = Vec::new();
        vst.process(&first, &mut output, false, |analysis, _, render| {
            original.extend_from_slice(&analysis[analysis.len() - block..]);
            Ok(vec![0.0; render * 480])
        })?;
        desktop.process(&first, &mut output, false, |analysis, _, render| {
            assert_eq!(&analysis[analysis.len() - block..], original);
            Ok(vec![0.0; render * 480])
        })?;

        // The previous native block is exactly two filter periods at 48 kHz.
        let mut filter = Resampler::new(48_000, FEATURE_RATE, 2 * input_len);
        let expected = &filter.process(&[first, second.clone()].concat())[block..];
        vst.process(&second, &mut output, false, |analysis, _, render| {
            let tail = &analysis[analysis.len() - 2 * block..];
            assert_eq!(&tail[..block], original);
            assert_eq!(&tail[block..], expected);
            Ok(vec![0.0; render * 480])
        })?;
        desktop.process(&second, &mut output, false, |analysis, _, render| {
            let tail = &analysis[analysis.len() - 2 * block..];
            assert_eq!(&tail[..FEATURE_HOP], &original[..FEATURE_HOP]);
            assert!((tail[block - 1] - original[block - 1]).abs() > 0.01);
            assert_eq!(&tail[block..], expected);
            Ok(vec![0.0; render * 480])
        })
    }

    #[test]
    fn above_floor_threshold_mutes_quiet_audio() -> Result<()> {
        let format = StreamFormat::new(16_000, 40)?;
        let options = RealtimeOptions {
            threshold_db: -20.0,
            ..Default::default()
        };
        let input = vec![0.01; format.block_frames];
        let mut output = vec![1.0; format.block_frames];
        for mut context in [
            RealtimeContext::new(format, 32_000, options)?,
            RealtimeContext::new_vst(format, 32_000, options)?,
        ] {
            context.process(&input, &mut output, false, |audio, _, render| {
                assert!(audio.iter().all(|&s| s == 0.0));
                Ok(vec![0.0; render * 320])
            })?;
            assert!(output.iter().all(|&s| s == 0.0));
            context.set_levels(-60.0, 1.0)?;
            context.process(&input, &mut output, false, |audio, _, render| {
                assert_eq!(audio.last(), Some(&0.01));
                Ok(vec![0.0; render * 320])
            })?;
        }
        Ok(())
    }

    #[test]
    fn zero_rms_mix_uses_the_input_envelope() -> Result<()> {
        let format = StreamFormat::new(48_000, 20)?;
        let mut context = RealtimeContext::new(
            format,
            48_000,
            RealtimeOptions {
                crossfade_ms: 0,
                extra_ms: 0,
                ..Default::default()
            },
        )?;
        let input = vec![0.25; format.block_frames];
        let mut output = vec![0.0; format.block_frames];
        context.process(&input, &mut output, false, |_, _, render| {
            Ok(vec![0.5; render * 480])
        })?;
        assert!(output.iter().all(|&s| (s - 0.25).abs() < 1e-6));
        Ok(())
    }
}
