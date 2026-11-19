use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use clap::ValueEnum;
use rustfft::FftPlanner;
use rustfft::num_complex::Complex32;
use serde::Deserialize;

use crate::audio::{self, Format};
use crate::separation::{AudioEncoding, write_encoded_track};

#[derive(Clone, Copy, Debug, Default, Deserialize, ValueEnum)]
#[value(rename_all = "snake_case")]
#[serde(rename_all = "snake_case")]
pub enum Algorithm {
    #[default]
    AvgWave,
    MedianWave,
    MinWave,
    MaxWave,
    AvgFft,
    MedianFft,
    MinFft,
    MaxFft,
}

#[allow(clippy::too_many_arguments)]
pub fn run(
    root: &Path,
    files: &[PathBuf],
    output: &Path,
    algorithm: Algorithm,
    weights: &[f32],
    format: Option<Format>,
    encoding: &AudioEncoding,
    overwrite: bool,
) -> Result<()> {
    ensure!(files.len() >= 2, "at least two input files are required");
    let (rate, channels) = audio::probe(&files[0])?;
    let mut decoded = Vec::with_capacity(files.len());
    for path in files {
        ensure!(
            audio::probe(path)? == (rate, channels),
            "sample rates and channel counts must match"
        );
        decoded.push(audio::read_interleaved(path, rate, channels)?);
    }
    let result = combine(&decoded, channels as usize, algorithm, weights)?;
    let format = format
        .map_or_else(
            || output.extension().map(Format::from_extension).transpose(),
            |format| Ok(Some(format)),
        )?
        .unwrap_or_default();
    let channels = usize::from(channels);
    let planar: Vec<_> = (0..channels)
        .flat_map(|channel| result.iter().skip(channel).step_by(channels).copied())
        .collect();
    write_encoded_track(
        root, output, &planar, rate, channels, format, encoding, overwrite,
    )
}

fn combine(
    files: &[Vec<f32>],
    channels: usize,
    algorithm: Algorithm,
    weights: &[f32],
) -> Result<Vec<f32>> {
    let length = files.iter().map(Vec::len).min().context("no audio files")?;
    let standard = vec![1.0; files.len()];
    let weights = if weights.is_empty() {
        &standard
    } else {
        weights
    };
    ensure!(
        length > 0 && channels > 0 && length.is_multiple_of(channels),
        "invalid audio length"
    );
    ensure!(
        weights.len() == files.len() && weights.iter().all(|w| w.is_finite()),
        "weights must match the input count"
    );
    let weight_sum: f32 = weights.iter().sum();
    if matches!(algorithm, Algorithm::AvgWave | Algorithm::AvgFft) {
        ensure!(
            weight_sum.abs() > 1e-8,
            "average weights must not sum to zero"
        );
    }
    let mut result = vec![0.0; length];
    let mut input = vec![0.0; files.len()];
    match algorithm {
        Algorithm::AvgWave | Algorithm::MedianWave | Algorithm::MinWave | Algorithm::MaxWave => {
            for (sample, output) in result.iter_mut().enumerate() {
                for (slot, file) in input.iter_mut().zip(files) {
                    *slot = file[sample];
                }
                *output = match algorithm {
                    Algorithm::AvgWave => {
                        input.iter().zip(weights).map(|(a, w)| a * w).sum::<f32>() / weight_sum
                    }
                    Algorithm::MedianWave => {
                        input.sort_unstable_by(f32::total_cmp);
                        let midi = input.len() / 2;
                        0.5 * (input[(input.len() - 1) / 2] + input[midi])
                    }
                    Algorithm::MinWave => choose(&input, |a, b| a.abs() < b.abs()),
                    Algorithm::MaxWave => choose(&input, |a, b| a.abs() > b.abs()),
                    _ => unreachable!(),
                };
            }
        }
        _ => fft(files, &mut result, channels, algorithm, weights, weight_sum),
    }
    Ok(result)
}

fn choose<T: Copy>(items: &[T], better: impl Fn(&T, &T) -> bool) -> T {
    let mut candidate = items[0];
    for item in &items[1..] {
        if better(item, &candidate) {
            candidate = *item;
        }
    }
    candidate
}

fn fft(
    files: &[Vec<f32>],
    output: &mut [f32],
    channels: usize,
    algorithm: Algorithm,
    weights: &[f32],
    sum: f32,
) {
    const FFT: usize = 2048;
    const HOP: usize = FFT / 2;
    let frames = output.len() / channels;
    let mut planner = FftPlanner::new();
    let forward = planner.plan_fft_forward(FFT);
    let inverse = planner.plan_fft_inverse(FFT);
    let hann: Vec<_> = (0..FFT)
        .map(|i| (0.5 - 0.5 * (std::f64::consts::TAU * i as f64 / FFT as f64).cos()) as f32)
        .collect();
    let mut spectra = vec![vec![Complex32::default(); FFT]; files.len()];
    let mut combined = vec![Complex32::default(); FFT];
    let mut values = vec![Complex32::default(); files.len()];
    let mut correction = vec![0.0; frames];
    for position in (0..=frames).step_by(HOP) {
        for (i, &window) in hann.iter().enumerate() {
            if let Some(sample) = (position + i).checked_sub(HOP).filter(|&i| i < frames) {
                correction[sample] += window * window;
            }
        }
    }
    for channel in 0..channels {
        for position in (0..=frames).step_by(HOP) {
            for (file, spectrum) in files.iter().zip(&mut spectra) {
                for (i, slot) in spectrum.iter_mut().enumerate() {
                    let sample = (position + i)
                        .checked_sub(HOP)
                        .filter(|&i| i < frames)
                        .map_or(0.0, |i| file[i * channels + channel]);
                    *slot = Complex32::new(sample * hann[i], 0.0);
                }
                forward.process(spectrum);
            }
            for bin in 0..=HOP {
                for (slot, spectrum) in values.iter_mut().zip(&spectra) {
                    *slot = spectrum[bin];
                }
                let combined_value = match algorithm {
                    Algorithm::AvgFft => {
                        values
                            .iter()
                            .zip(weights)
                            .map(|(value, weight)| value * weight)
                            .sum::<Complex32>()
                            / sum
                    }
                    Algorithm::MedianFft => {
                        values.sort_unstable_by(|a, b| {
                            a.re.total_cmp(&b.re).then(a.im.total_cmp(&b.im))
                        });
                        (values[(values.len() - 1) / 2] + values[values.len() / 2]) * 0.5
                    }
                    Algorithm::MinFft => choose(&values, |a, b| a.norm_sqr() < b.norm_sqr()),
                    Algorithm::MaxFft => choose(&values, |a, b| a.norm_sqr() > b.norm_sqr()),
                    _ => unreachable!(),
                };
                combined[bin] = combined_value;
                if (1..HOP).contains(&bin) {
                    combined[FFT - bin] = combined_value.conj();
                }
            }
            inverse.process(&mut combined);
            for (i, value) in combined.iter().enumerate() {
                if let Some(sample) = (position + i).checked_sub(HOP).filter(|&i| i < frames) {
                    output[sample * channels + channel] += value.re * hann[i] / FFT as f32;
                }
            }
        }
    }
    for (i, frame) in output.chunks_exact_mut(channels).enumerate() {
        if correction[i] > 0.0 {
            frame.iter_mut().for_each(|v| *v /= correction[i]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wave_and_fft_average_reconstruct_the_same_signal() {
        let first: Vec<_> = (0..2_806).map(|i| (i as f32 * 0.13).sin()).collect();
        let second: Vec<_> = (0..2_806).map(|i| (i as f32 * 0.21).cos() * 0.5).collect();
        let tracks = vec![first, second];
        let a = combine(&tracks, 2, Algorithm::AvgWave, &[1.0, 3.0]).unwrap();
        let b = combine(&tracks, 2, Algorithm::AvgFft, &[1.0, 3.0]).unwrap();
        assert!(a.iter().zip(b).all(|(&a, b)| (a - b).abs() < 3e-6));
    }
}
