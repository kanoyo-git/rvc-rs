//! Channel-major chunk scheduling and overlap-add from the original PyMSS.
//! See `licenses/pymss.txt`.

use std::error::Error;
use std::fmt;
use std::ops::ControlFlow;

#[derive(Debug, Clone, Copy)]
pub(super) struct Geometry {
    pub frames_per_chunk: usize,
    pub overlap: usize,
    pub batch_size: usize,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct Progress {
    pub chunks_done: usize,
    pub chunks_total: usize,
}

#[derive(Debug)]
pub(super) enum DemixError<E> {
    Geometry,
    Input,
    SizeOverflow,
    Model(E),
    Prediction,
    Cancelled,
}

impl<E: fmt::Display> fmt::Display for DemixError<E> {
    fn fmt(&self, out: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Geometry => out.write_str("expected 0 <= overlap < chunk size"),
            Self::Input => out.write_str("expected nonempty, finite channel-major audio"),
            Self::SizeOverflow => out.write_str("audio dimensions exceed address space"),
            Self::Model(error) => write!(out, "separation model failed: {error}"),
            Self::Prediction => {
                out.write_str("invalid model prediction shape or nonfinite samples")
            }
            Self::Cancelled => out.write_str("separation cancelled"),
        }
    }
}

impl<E: Error + 'static> Error for DemixError<E> {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Model(error) => Some(error),
            _ => None,
        }
    }
}

/// Predict each fixed-size window and stitch the outputs in source order.
///
/// The track is channel-major. The model takes `[window][channel][frame]` and
/// returns `[window][stem][channel][frame]`. Shorter predictions are extended
/// with zeros and longer predictions are cropped.
pub(super) fn demix<E>(
    mix: &[f32],
    channels: usize,
    stems: usize,
    selected: &[usize],
    geometry: Geometry,
    mut predict: impl FnMut(&[f32], usize) -> Result<Vec<f32>, E>,
    mut progress: impl FnMut(Progress) -> ControlFlow<()>,
) -> Result<Vec<f32>, DemixError<E>> {
    let Geometry {
        frames_per_chunk,
        overlap,
        batch_size,
    } = geometry;
    if overlap >= frames_per_chunk || batch_size == 0 {
        return Err(DemixError::Geometry);
    }
    if channels == 0
        || selected.is_empty()
        || selected.iter().any(|&index| index >= stems)
        || mix.is_empty()
        || !mix.len().is_multiple_of(channels)
        || mix.iter().any(|sample| !sample.is_finite())
    {
        return Err(DemixError::Input);
    }

    let frame_count = mix.len() / channels;
    let padding = if overlap < frame_count.saturating_sub(overlap) {
        overlap
    } else {
        0
    };
    let padded_count = padding
        .checked_mul(2)
        .and_then(|size| size.checked_add(frame_count))
        .ok_or(DemixError::SizeOverflow)?;
    let prediction_planes = channels
        .checked_mul(stems)
        .ok_or(DemixError::SizeOverflow)?;
    let total_samples = selected
        .len()
        .checked_mul(mix.len())
        .ok_or(DemixError::SizeOverflow)?;
    let chunk_samples = channels
        .checked_mul(frames_per_chunk)
        .ok_or(DemixError::SizeOverflow)?;
    let step = frames_per_chunk - overlap;
    let fade = (frames_per_chunk / 10).min(overlap);
    let chunks_total = padded_count.div_ceil(step);

    let batch_size = batch_size.min(chunks_total);
    let batch_samples = chunk_samples
        .checked_mul(batch_size)
        .ok_or(DemixError::SizeOverflow)?;
    let mut input = vec![0.0; batch_samples];
    let mut output = vec![0.0; total_samples];
    let mut weights = vec![0.0; frame_count];
    let mut window_weights = vec![0.0; frames_per_chunk];
    for index in (0..chunks_total).step_by(batch_size) {
        let state = Progress {
            chunks_done: index,
            chunks_total,
        };
        if progress(state).is_break() {
            return Err(DemixError::Cancelled);
        }
        let count = (chunks_total - index).min(batch_size);
        let span = &mut input[..count * chunk_samples];
        for (item, window) in span.chunks_exact_mut(chunk_samples).enumerate() {
            let start = (index + item) * step;
            let valid = frames_per_chunk.min(padded_count - start);
            for (source, channel) in mix
                .chunks_exact(frame_count)
                .zip(window.chunks_exact_mut(frames_per_chunk))
            {
                fill_chunk(source, channel, start, valid, padding);
            }
        }
        let prediction = predict(span, count).map_err(DemixError::Model)?;
        let planes = prediction_planes
            .checked_mul(count)
            .ok_or(DemixError::SizeOverflow)?;
        if prediction.is_empty()
            || !prediction.len().is_multiple_of(planes)
            || prediction.iter().any(|sample| !sample.is_finite())
        {
            return Err(DemixError::Prediction);
        }
        let prediction_frames = prediction.len() / planes;
        for (item, window) in prediction
            .chunks_exact(prediction.len() / count)
            .enumerate()
        {
            let start = (index + item) * step;
            let valid = frames_per_chunk.min(padded_count - start);
            let crop_start = padding.saturating_sub(start);
            let crop_end = valid.min((padding + frame_count).saturating_sub(start));
            for (position, gain) in window_weights
                .iter_mut()
                .enumerate()
                .take(crop_end)
                .skip(crop_start)
            {
                let sample = start + position - padding;
                *gain = window_weight(
                    position,
                    frames_per_chunk,
                    fade,
                    start == 0,
                    (start + valid == padded_count).then_some(valid),
                );
                weights[sample] += *gain;
            }
            let predicted_end = crop_end.min(prediction_frames);
            if crop_start < predicted_end {
                let first_frame = start + crop_start - padding;
                let end_frame = start + predicted_end - padding;
                for (&stem, out) in selected.iter().zip(output.chunks_exact_mut(mix.len())) {
                    let begin = stem * channels * prediction_frames;
                    let source = &window[begin..begin + channels * prediction_frames];
                    for (plane, estimate) in out
                        .chunks_exact_mut(frame_count)
                        .zip(source.chunks_exact(prediction_frames))
                    {
                        let samples = &estimate[crop_start..predicted_end];
                        let gains = &window_weights[crop_start..predicted_end];
                        for ((sample, &prediction), &gain) in plane[first_frame..end_frame]
                            .iter_mut()
                            .zip(samples)
                            .zip(gains)
                        {
                            *sample += prediction * gain;
                        }
                    }
                }
            }
        }
    }
    for plane in output.chunks_exact_mut(frame_count) {
        for (sample, weight) in plane.iter_mut().zip(&weights) {
            if *weight > 0.0 {
                *sample /= weight;
            }
        }
    }
    if progress(Progress {
        chunks_done: chunks_total,
        chunks_total,
    })
    .is_break()
    {
        return Err(DemixError::Cancelled);
    }
    Ok(output)
}

fn fill_chunk(source: &[f32], chunk: &mut [f32], start: usize, valid: usize, padding: usize) {
    for (index, output) in chunk[..valid].iter_mut().enumerate() {
        let pos = start + index;
        let body_end = padding + source.len();
        let source_pos = if pos < padding {
            padding - pos
        } else if pos < body_end {
            pos - padding
        } else {
            source.len() - 2 - (pos - body_end)
        };
        *output = source[source_pos];
    }
    if valid > chunk.len() / 2 + 1 {
        for index in valid..chunk.len() {
            chunk[index] = chunk[2 * (valid - 1) - index];
        }
    } else {
        chunk[valid..].fill(0.0);
    }
}

/// HTDemucs uses unweighted averages and zero-pads the final block.
pub(super) fn demix_uniform<E>(
    mix: &[f32],
    channels: usize,
    stems: usize,
    selected: &[usize],
    geometry: Geometry,
    mut predict: impl FnMut(&[f32], usize) -> Result<Vec<f32>, E>,
    mut progress: impl FnMut(Progress) -> ControlFlow<()>,
) -> Result<Vec<f32>, DemixError<E>> {
    let block = geometry.frames_per_chunk;
    if geometry.overlap >= block || geometry.batch_size == 0 {
        return Err(DemixError::Geometry);
    }
    if channels == 0
        || selected.is_empty()
        || selected.iter().any(|&index| index >= stems)
        || mix.is_empty()
        || !mix.len().is_multiple_of(channels)
    {
        return Err(DemixError::Input);
    }
    let frames = mix.len() / channels;
    let step = block - geometry.overlap;
    let chunks_total = frames.div_ceil(step);
    let batch_size = geometry.batch_size.min(chunks_total);
    let planes = channels
        .checked_mul(stems)
        .ok_or(DemixError::SizeOverflow)?;
    let expected = planes.checked_mul(block).ok_or(DemixError::SizeOverflow)?;
    let chunk_samples = channels
        .checked_mul(block)
        .ok_or(DemixError::SizeOverflow)?;
    let batch_samples = chunk_samples
        .checked_mul(batch_size)
        .ok_or(DemixError::SizeOverflow)?;
    let mut input = vec![0.0; batch_samples];
    let mut audio = vec![
        0.0;
        selected
            .len()
            .checked_mul(mix.len())
            .ok_or(DemixError::SizeOverflow)?
    ];
    let mut counts = vec![0.0; frames];
    for chunks_done in (0..chunks_total).step_by(batch_size) {
        if progress(Progress {
            chunks_done,
            chunks_total,
        })
        .is_break()
        {
            return Err(DemixError::Cancelled);
        }
        let count = (chunks_total - chunks_done).min(batch_size);
        let span = &mut input[..count * chunk_samples];
        for (item, window) in span.chunks_exact_mut(chunk_samples).enumerate() {
            let start = (chunks_done + item) * step;
            let len = block.min(frames - start);
            for (source, output) in mix.chunks_exact(frames).zip(window.chunks_exact_mut(block)) {
                output[..len].copy_from_slice(&source[start..start + len]);
                output[len..].fill(0.0);
            }
        }
        let output = predict(span, count).map_err(DemixError::Model)?;
        if expected.checked_mul(count) != Some(output.len())
            || output.iter().any(|x| !x.is_finite())
        {
            return Err(DemixError::Prediction);
        }
        for (item, window) in output.chunks_exact(expected).enumerate() {
            let start = (chunks_done + item) * step;
            let len = block.min(frames - start);
            for (&stem, out) in selected.iter().zip(audio.chunks_exact_mut(mix.len())) {
                let begin = stem * chunk_samples;
                let source = &window[begin..begin + chunk_samples];
                for (target, source) in out.chunks_exact_mut(frames).zip(source.chunks_exact(block))
                {
                    for (a, &b) in target[start..start + len].iter_mut().zip(source) {
                        *a += b;
                    }
                }
            }
            for covered in &mut counts[start..start + len] {
                *covered += 1.0;
            }
        }
    }
    for channel in audio.chunks_exact_mut(frames) {
        for (sample, &count) in channel.iter_mut().zip(&counts) {
            *sample /= count;
        }
    }
    if progress(Progress {
        chunks_done: chunks_total,
        chunks_total,
    })
    .is_break()
    {
        return Err(DemixError::Cancelled);
    }
    Ok(audio)
}

fn window_weight(index: usize, chunk: usize, fade: usize, first: bool, last: Option<usize>) -> f32 {
    if fade == 0 || last.is_some_and(|length| index >= length.saturating_sub(fade)) {
        return 1.0;
    }
    if index < fade && !first {
        return index as f32 / (fade - 1).max(1) as f32;
    }
    if index >= chunk - fade && fade > 1 {
        return (chunk - 1 - index) as f32 / (fade - 1) as f32;
    }
    1.0
}

/// Return the second stem of a single-target model.
pub(super) fn residual(mix: &[f32], primary: &[f32]) -> Option<Vec<f32>> {
    (mix.len() == primary.len()).then(|| mix.iter().zip(primary).map(|(x, y)| x - y).collect())
}

#[cfg(test)]
mod tests {
    use std::convert::Infallible;

    use super::*;

    #[test]
    fn overlap_preserves_channels_and_stem_order() {
        let stereo: Vec<f32> = (0..402).map(|i| (i - 200) as f32 / 402.0).collect();
        let result = demix(
            &stereo,
            2,
            2,
            &[0, 1],
            Geometry {
                frames_per_chunk: 40,
                overlap: 27,
                batch_size: 1,
            },
            |chunk, _| {
                Ok::<_, Infallible>(
                    chunk
                        .iter()
                        .copied()
                        .chain(chunk.iter().map(|x| -x))
                        .collect(),
                )
            },
            |_| ControlFlow::Continue(()),
        )
        .unwrap();
        let expected: Vec<_> = stereo
            .iter()
            .copied()
            .chain(stereo.iter().map(|x| -x))
            .collect();
        for (&a, b) in result.iter().zip(expected) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn incomplete_windows_reflect_only_if_enough_audio_remains() {
        let input: Vec<_> = (0..9).map(|i| i as f32).collect();
        let mut reflected = [0.0; 10];
        fill_chunk(&input, &mut reflected, 0, 9, 0);
        assert_eq!(reflected[9], 7.0);
        let mut zero_padded = [1.0; 10];
        fill_chunk(&input, &mut zero_padded, 0, 6, 0);
        assert_eq!(&zero_padded[6..], &[0.0; 4]);
    }

    #[test]
    fn can_stop_without_running_the_model() {
        let result = demix(
            &[0.0; 81],
            1,
            1,
            &[0],
            Geometry {
                frames_per_chunk: 20,
                overlap: 5,
                batch_size: 1,
            },
            |_, _| -> Result<_, Infallible> { panic!("model called despite cancellation") },
            |_| ControlFlow::Break(()),
        );
        assert!(matches!(result, Err(DemixError::Cancelled)));
    }

    #[test]
    fn matches_pymss_overlap_for_deterministic_stems() {
        let mut fixture = include_bytes!("fixtures/pymss-overlap.f32").as_slice();
        while !fixture.is_empty() {
            let mut read_word = || {
                let (word, rest) = fixture.split_at(4);
                fixture = rest;
                word.try_into().unwrap()
            };
            let count = u32::from_le_bytes(read_word()) as usize;
            let chunk_size = u32::from_le_bytes(read_word()) as usize;
            let overlap = u32::from_le_bytes(read_word()) as usize;
            let stereo: Vec<_> = (0..2 * count)
                .map(|i| ((i % 57) as f32 - 23.0) / 64.0)
                .collect();
            let actual = demix(
                &stereo,
                2,
                2,
                &[0, 1],
                Geometry {
                    frames_per_chunk: chunk_size,
                    overlap,
                    batch_size: 1,
                },
                |chunk, _| {
                    let first = chunk.iter().enumerate().map(|(idx, x)| {
                        let position = if overlap > 0 {
                            (idx % chunk_size) % 17
                        } else {
                            0
                        };
                        x * 0.5 + position as f32 / 128.0
                    });
                    let second = chunk.chunks_exact(chunk_size).flat_map(|part| {
                        let iterator: Box<dyn Iterator<Item = &f32>> = if overlap > 0 {
                            Box::new(part.iter().rev())
                        } else {
                            Box::new(part.iter())
                        };
                        iterator.map(|x| x * -0.75)
                    });
                    Ok::<_, Infallible>(first.chain(second).collect())
                },
                |_| ControlFlow::Continue(()),
            )
            .unwrap();
            assert_eq!(actual.len(), 4 * count);
            for (i, a) in actual.into_iter().enumerate() {
                let b = f32::from_le_bytes(read_word());
                assert!(
                    (a - b).abs() <= 1e-7,
                    "case {count}/{chunk_size}/{overlap}, sample {i}: {a} vs {b}"
                );
            }
        }
    }
}
