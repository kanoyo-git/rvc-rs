//! A model-trained envelope and harmonic concentration correction for turbulent singing.
//! The source drives the detector and coarse band features. The phase and spectral lines
//! used to reconstruct the output are always the generator's.
use std::ops::Range;
use std::sync::Arc;

use rustfft::num_complex::Complex64;
use rustfft::{Fft, FftPlanner};

use crate::catalog::{ArticulationDetector, ArticulationProfile};

#[path = "articulation/frication.rs"]
pub(super) mod frication;

const EDGES: [u32; 10] = [80, 300, 750, 1500, 3000, 5000, 8000, 12000, 16000, 20000];
const BANDS: usize = 9;

fn db(power: f64) -> f64 {
    10.0 * power.max(1e-12).log10()
}

fn smooth(value: f64) -> f64 {
    let t = value.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

struct Spectrum {
    hop: usize,
    fft: Arc<dyn Fft<f64>>,
    inverse: Arc<dyn Fft<f64>>,
    window: Vec<f64>,
    bands: [Range<usize>; BANDS],
    flat_bands: [Range<usize>; 4],
    periodic_bands: [Range<usize>; 3],
    lag_start: usize,
    lag_norm: Vec<f64>,
}

struct Correction {
    gain: [f64; BANDS],
    exponent: [f64; BANDS],
    // V3 corrects the remaining frame energy after band shaping on sustained events.
    event_level: Option<(f64, f64)>,
}

impl Spectrum {
    fn new(rate: u32) -> Self {
        let hop = (rate / 100) as usize;
        let width = hop * 4;
        let size = width.next_power_of_two();
        let mut planner = FftPlanner::new();
        let window: Vec<_> = (0..width)
            .map(|i| {
                (std::f64::consts::PI * (i as f64 + 0.5) / width as f64)
                    .sin()
                    .powi(2)
            })
            .collect();
        let energy: f64 = window.iter().map(|x| x * x).sum();
        let lag_start = (rate as usize / 1400).max(2);
        let lag_end = (rate as usize / 55).min(width - 1);
        let lag_norm = (lag_start..=lag_end)
            .map(|lag| {
                let sum: f64 = window.iter().zip(&window[lag..]).map(|(a, b)| a * b).sum();
                (sum / energy).max(0.35)
            })
            .collect();
        let bin = |hz: u32| {
            (hz as usize * size)
                .div_ceil(rate as usize)
                .min(size / 2 + 1)
        };
        let range = |lo, hi| bin(lo)..bin(hi);
        Self {
            hop,
            fft: planner.plan_fft_forward(size),
            inverse: planner.plan_fft_inverse(size),
            window,
            bands: std::array::from_fn(|i| range(EDGES[i], EDGES[i + 1])),
            flat_bands: [(300, 1500), (300, 8000), (1500, 4000), (4000, 12000)]
                .map(|(a, b)| range(a, b)),
            periodic_bands: [(55, 1600), (140, 4200), (500, 7000)].map(|(a, b)| range(a, b)),
            lag_start,
            lag_norm,
        }
    }

    fn analyze(
        &self,
        audio: &[f32],
        frame: usize,
        fft: &mut [Complex64],
        scratch: &mut [Complex64],
    ) {
        fft.fill(Complex64::default());
        let pos = frame * self.hop;
        let offset = self.window.len() / 2;
        let start = pos.saturating_sub(offset);
        let skip = offset.saturating_sub(pos);
        let len = (self.window.len() - skip).min(audio.len().saturating_sub(start));
        if len != 0 {
            for (i, &x) in audio[start..start + len].iter().enumerate() {
                fft[i + skip].re = f64::from(x) * self.window[i + skip];
            }
        }
        self.fft.process_with_scratch(fft, scratch);
    }

    fn shape_delta(
        &self,
        fft: &[Complex64],
        buffer: &mut [Complex64],
        shaped: &mut [f64],
        correction: &Correction,
        scratch: &mut [Complex64],
    ) {
        let size = self.fft.len();
        buffer.fill(Complex64::default());
        for (b, range) in self.bands.iter().enumerate() {
            if range.is_empty() {
                continue;
            }
            let len = range.len() as f64;
            let avg = fft[range.clone()].iter().map(|v| v.norm_sqr()).sum::<f64>() / len + 1e-13;
            let mut sum = 0.0;
            for k in range.clone() {
                let p = fft[k].norm_sqr();
                shaped[k] = (correction.exponent[b] * ((p + 1e-12) / avg).ln())
                    .clamp(-25.0, 18.0)
                    .exp();
                sum += shaped[k];
            }
            let scale = avg / (sum / len).max(1e-13);
            for k in range.clone() {
                let target =
                    (shaped[k] * scale / fft[k].norm_sqr().max(1e-12)).sqrt() * correction.gain[b];
                buffer[k] = fft[k] * (target.clamp(0.035, 28.0) - 1.0);
                if k != 0 && k != size / 2 {
                    buffer[size - k] = buffer[k].conj();
                }
            }
        }
        if let Some((rough, level_delta)) = correction.event_level {
            // The reference sums the one-sided spectrum, including DC and Nyquist.
            let (mut before, mut after) = (0.0, 0.0);
            for k in 0..=size / 2 {
                before += fft[k].norm_sqr();
                after += (fft[k] + buffer[k]).norm_sqr();
            }
            let residual_db = (level_delta - (db(after) - db(before))).clamp(-8.0, 19.0);
            let scale = 10.0_f64.powf(rough * residual_db / 20.0);
            for k in 0..=size / 2 {
                buffer[k] = (fft[k] + buffer[k]) * scale - fft[k];
                if k != 0 && k != size / 2 {
                    buffer[size - k] = buffer[k].conj();
                }
            }
        }
        self.inverse.process_with_scratch(buffer, scratch);
    }

    fn periodicity(
        &self,
        power: &[f64],
        buf: &mut [Complex64],
        scratch: &mut [Complex64],
    ) -> [f64; 3] {
        let size = self.fft.len();
        std::array::from_fn(|b| {
            buf.fill(Complex64::default());
            for i in self.periodic_bands[b].clone() {
                buf[i].re = power[i];
                if i != 0 {
                    buf[size - i].re = power[i];
                }
            }
            self.inverse.process_with_scratch(buf, scratch);
            let energy = buf[0].re.max(1e-12 * size as f64);
            self.lag_norm
                .iter()
                .enumerate()
                .map(|(i, &norm)| buf[self.lag_start + i].re / (energy * norm))
                .fold(0.0, f64::max)
                .min(1.0)
        })
    }
}

struct Frame {
    level: f64,
    band: [f64; BANDS],
    share: [f64; BANDS],
    dispersion: [f64; BANDS],
    flat: [f64; 4],
    high: f64,
}

impl Frame {
    fn new(layout: &Spectrum, power: &[f64]) -> Self {
        let energy = layout
            .bands
            .each_ref()
            .map(|bins| power[bins.clone()].iter().sum::<f64>());
        let total = energy.iter().sum::<f64>();
        let level = db(total);
        let band = energy.map(db);
        let share = band.map(|e| (e - level).max(-65.0));
        let high = (energy[4..8].iter().sum::<f64>() / total.max(1e-12)).clamp(0.0, 1.0);
        let dispersion = layout.bands.each_ref().map(|bins| {
            let mut mean = 0.0;
            let mut sum = 0.0;
            let data = &power[bins.clone()];
            for (i, &p) in data.iter().enumerate() {
                let log = db(p + 1e-9);
                let delta = log - mean;
                mean += delta / (i + 1) as f64;
                sum += delta * (log - mean);
            }
            (sum / data.len().max(1) as f64).sqrt()
        });
        let flat = layout.flat_bands.each_ref().map(|bins| {
            let data = &power[bins.clone()];
            if data.is_empty() {
                return 0.0;
            }
            let (sum, logs) = data.iter().fold((0.0, 0.0), |(sum, logs), &p| {
                let p = p + 1e-10;
                (sum + p, logs + p.ln())
            });
            let len = data.len() as f64;
            db((logs / len).exp() / (sum / len))
        });
        Self {
            level,
            band,
            share,
            dispersion,
            flat,
            high,
        }
    }

    fn relative_to(&mut self, peak: f64) {
        self.level -= peak;
        self.band.iter_mut().for_each(|value| *value -= peak);
    }

    fn features(&self, out: &mut [f64]) {
        out[0] = self.level.clamp(-45.0, 9.0);
        out[1..10].copy_from_slice(&self.share);
        out[10..14].copy_from_slice(&self.flat);
    }

    fn projection(&self, core: &Self) -> [f64; 37] {
        let mut out = [0.0; 37];
        self.features(&mut out);
        out[14] = core.level.clamp(-60.0, 12.0);
        for b in 0..BANDS {
            out[15 + b] = (self.band[b] - core.band[b]).clamp(-50.0, 50.0);
            out[24 + b] = (self.dispersion[b] - core.dispersion[b]).clamp(-15.0, 15.0);
        }
        for b in 0..4 {
            out[33 + b] = (self.flat[b] - core.flat[b]).clamp(-35.0, 35.0);
        }
        out
    }
}

fn normalize(frames: &mut [Frame]) {
    let mut levels: Vec<_> = frames.iter().map(|f| f.level).collect();
    levels.sort_unstable_by(f64::total_cmp);
    let p = 0.9 * (levels.len() - 1) as f64;
    let low = levels[p.floor() as usize];
    let peak = low + (levels[p.ceil() as usize] - low) * p.fract();
    for f in frames {
        f.relative_to(peak);
    }
}

fn mean_window<const N: usize>(frames: &[[f64; N]], frame: usize, radius: usize) -> [f64; N] {
    let mut sums = [0.0; N];
    for i in 0..=radius * 2 {
        let index = (frame + i).saturating_sub(radius).min(frames.len() - 1);
        for (s, &value) in sums.iter_mut().zip(&frames[index]) {
            *s += value;
        }
    }
    sums.map(|sum| sum / (2 * radius + 1) as f64)
}

fn affine(weights: &[f64], features: &[f64], bias: f64) -> f64 {
    weights
        .iter()
        .zip(features)
        .fold(bias, |sum, (w, x)| sum + w * x)
}

fn probability(mut features: [f64; 24], detector: &ArticulationDetector) -> f64 {
    for (i, value) in features.iter_mut().enumerate() {
        *value = (*value - detector.mean[i]) / detector.scale[i];
    }
    let first: [f64; 24] = std::array::from_fn(|i| {
        affine(&detector.weights[0][i], &features, detector.bias[0][i]).max(0.0)
    });
    let second: [f64; 12] = std::array::from_fn(|i| {
        affine(&detector.weights[1][i], &first, detector.bias[1][i]).max(0.0)
    });
    let score = affine(&detector.weights[2][0], &second, detector.bias[2][0]);
    1.0 / (1.0 + (-score).exp())
}

fn detector_probabilities(
    src: &[Frame],
    dynamic: &[[f64; 6]],
    detector: &ArticulationDetector,
) -> Vec<[f64; 1]> {
    src.iter()
        .enumerate()
        .map(|(i, s)| {
            let mut x = [0.0; 24];
            s.features(&mut x);
            x[14..18].copy_from_slice(&dynamic[i][..4]);
            x[18..].copy_from_slice(&mean_window(dynamic, i, 7));
            [probability(x, detector)]
        })
        .collect()
}

#[derive(Clone, Copy)]
struct Events {
    rough_density: f64,
    rough: f64,
    fricative: f64,
}

fn events(src: &[Frame], probabilities: &[[f64; 1]], version: u8) -> Vec<Events> {
    let smoothed: Vec<_> = (0..src.len())
        .map(|i| mean_window(probabilities, i, 2)[0])
        .collect();
    let sustained = (version == 3).then(|| {
        smoothed
            .iter()
            .map(|&p| [f64::from(u8::from(p > 0.8))])
            .collect::<Vec<_>>()
    });
    src.iter()
        .enumerate()
        .map(|(i, frame)| {
            let active = smooth((frame.level + 38.0) / 13.0);
            let mut rough = smooth((smoothed[i] - 0.55) / 0.37) * active;
            let rough_density = sustained
                .as_ref()
                .map_or(0.0, |mask| mean_window(mask, i, 32)[0]);
            if sustained.is_some() {
                rough *= smooth((rough_density - 0.33) / 0.35);
            }
            let fricative =
                smooth((frame.high - 0.50) / 0.30) * smooth((frame.flat[3] + 19.0) / 12.0) * active;
            Events {
                rough_density,
                rough,
                fricative,
            }
        })
        .collect()
}

fn fricative_offset(src: &[Frame], core: &[Frame], events: &[Events]) -> [f64; BANDS] {
    let selected: Vec<_> = events
        .iter()
        .enumerate()
        .filter_map(|(i, e)| (e.fricative > 0.35 && e.rough < 0.28).then_some(i))
        .collect();
    if selected.len() < 7 {
        return [0.0; BANDS];
    }
    let mut column = Vec::with_capacity(selected.len());
    std::array::from_fn(|band| {
        column.clear();
        column.extend(
            selected
                .iter()
                .map(|&i| src[i].band[band] - core[i].band[band]),
        );
        column.sort_unstable_by(f64::total_cmp);
        (column[(column.len() - 1) / 2] + column[column.len() / 2]) * 0.5
    })
}

fn envelope(
    source: &Frame,
    core: &Frame,
    Events {
        rough, fricative, ..
    }: Events,
    offset: Option<&[f64; BANDS]>,
    profile: &ArticulationProfile,
) -> Option<Correction> {
    let noise = rough.max(fricative);
    if noise == 0.0 {
        return None;
    }
    let features = source.projection(core);
    let prediction: [f64; 18] =
        std::array::from_fn(|i| affine(&profile.weights[i], &features, profile.bias[i]));
    let mut gain: [f64; BANDS] = std::array::from_fn(|b| {
        let optimal = source.band[b] - core.band[b];
        let band_gain = (0.32 * prediction[b] + 0.68 * optimal).clamp(-16.0, 20.0);
        if let Some(offset) = offset {
            const WEIGHTS: [f64; BANDS] = [0.0, 0.0, 0.0, 0.0, 0.08, 0.30, 0.38, 0.38, 0.24];
            let local = (optimal - offset[b]).clamp(-6.0, 6.0) * WEIGHTS[b];
            band_gain * rough + local * (fricative * (1.0 - rough))
        } else {
            band_gain * noise
        }
    });
    let emphasis = [1.0, 2.0, 3.0, 3.0, 2.0, 1.0, 0.5, 0.2, 0.1];
    let gain_mean = affine(&gain, &emphasis, 0.0) / emphasis.iter().sum::<f64>();
    let loss = (source.level - core.level - 2.0).max(0.0) * smooth((source.level + 34.0) / 18.0);
    let rescue = (loss * rough.max(fricative * 0.65) - gain_mean).max(0.0) * noise;
    for (b, g) in gain.iter_mut().enumerate() {
        *g += rescue;
        if source.share[b] < -58.0 {
            *g = g.min(0.0);
        }
        *g = 10.0_f64.powf(*g / 20.0);
    }
    let exponent = std::array::from_fn(|b| {
        let desired = ((source.dispersion[b] + 0.45) / (core.dispersion[b] + 0.45))
            .ln()
            .clamp(-1.0, 0.35);
        let amount = if offset.is_some() { rough } else { noise };
        (amount * (0.32 * prediction[BANDS + b].clamp(-0.8, 0.3) + 0.68 * desired)).exp()
    });
    Some(Correction {
        gain,
        exponent,
        event_level: (offset.is_some() && rough > 0.0)
            .then_some((rough, source.level - core.level)),
    })
}

/// Apply the offline profile at the native generator rate. Two passes retain only
/// per-frame features between them, independent of FFT size.
pub(super) fn apply(
    output: &mut [f32],
    source: &[f32],
    rate: u32,
    profile: &ArticulationProfile,
    speaker: i64,
) {
    let n = source.len().min(output.len());
    if n == 0 || rate < 100 {
        return;
    }
    let layout = Spectrum::new(rate);
    let size = layout.fft.len();
    let bins = size / 2 + 1;
    // Frame centres follow the Python reference, including one past the last sample.
    let frame_count = n.div_ceil(layout.hop) + 1;
    let target = profile
        .frication
        .as_ref()
        .and_then(|p| p.targets.get(&speaker).map(|t| (p, t)));
    let mut frication = target.map(|_| frication::Analysis::new(&layout, rate, frame_count));
    let mut src = Vec::with_capacity(frame_count);
    let mut core = Vec::with_capacity(frame_count);
    let mut dynamic = Vec::with_capacity(frame_count);
    let mut fft = vec![Complex64::default(); size];
    let mut buffer = fft.clone();
    let mut scratch = vec![
        Complex64::default();
        layout
            .fft
            .get_inplace_scratch_len()
            .max(layout.inverse.get_inplace_scratch_len())
    ];
    let mut power = vec![0.0; bins];
    let mut last_share = None;
    for t in 0..frame_count {
        layout.analyze(&source[..n], t, &mut fft, &mut scratch);
        for (p, bin) in power.iter_mut().zip(&fft) {
            *p = bin.norm_sqr();
        }
        let frame = Frame::new(&layout, &power);
        if let Some(analysis) = &mut frication {
            analysis.observe(&power, frame.level);
        }
        let period = layout.periodicity(&power, &mut buffer, &mut scratch);
        let flux = last_share.map_or(0.0, |last: [f64; BANDS]| {
            frame
                .share
                .iter()
                .zip(last)
                .map(|(&p, q)| (p - q).abs().min(12.0))
                .sum::<f64>()
                / BANDS as f64
        });
        dynamic.push([
            period[0],
            period[1],
            period[2],
            flux,
            frame.flat[0],
            frame.flat[1],
        ]);
        last_share = Some(frame.share);
        src.push(frame);
        layout.analyze(&output[..n], t, &mut fft, &mut scratch);
        for (p, bin) in power.iter_mut().zip(&fft) {
            *p = bin.norm_sqr();
        }
        core.push(Frame::new(&layout, &power));
    }
    normalize(&mut src);
    normalize(&mut core);
    let probabilities = detector_probabilities(&src, &dynamic, &profile.detector);
    let events = events(&src, &probabilities, profile.version);
    let colour_offset = (profile.version == 3).then(|| fricative_offset(&src, &core, &events));

    let mut shaped = vec![0.0; bins];
    reconstruct(
        &layout,
        output,
        n,
        frame_count,
        (&mut fft, &mut buffer, &mut scratch),
        None,
        |t| {
            envelope(
                &src[t],
                &core[t],
                events[t],
                colour_offset.as_ref(),
                profile,
            )
        },
        |fft, buffer, scratch, correction| {
            layout.shape_delta(fft, buffer, &mut shaped, correction, scratch)
        },
    );
    let peak = output.iter().fold(0.0_f32, |p, x| p.max(x.abs()));
    if peak > 0.995 {
        for x in output.iter_mut() {
            *x *= 0.995 / peak;
        }
    }
    if let (Some(analysis), Some((params, template))) = (frication, target) {
        analysis.apply(
            output, n, &layout, &src, &dynamic, &events, params, template,
        );
    }
}

/// Analyze the unmodified generator frame before committing any overlapping samples.
#[allow(clippy::too_many_arguments)]
fn reconstruct<T>(
    layout: &Spectrum,
    output: &mut [f32],
    n: usize,
    frame_count: usize,
    (fft, buffer, scratch): (&mut [Complex64], &mut [Complex64], &mut [Complex64]),
    sample_bound: Option<f64>,
    mut factors: impl FnMut(usize) -> Option<T>,
    mut transform: impl FnMut(&[Complex64], &mut [Complex64], &mut [Complex64], &T),
) {
    let size = layout.fft.len();
    // Overlap-add into a single window. Once the next frame starts, neither
    // analysis nor synthesis can touch earlier generator samples again.
    let mut residual = vec![0.0; size];
    let mut norm = vec![0.0; size];

    let mut written = 0;
    let offset = layout.window.len() / 2;
    for t in 0..frame_count {
        let centre = t * layout.hop;
        let start = centre.saturating_sub(offset);
        let skip = offset.saturating_sub(centre);
        let len = (layout.window.len() - skip).min(n.saturating_sub(start));
        let correction = factors(t);
        if let Some(correction) = &correction {
            layout.analyze(&output[..n], t, fft, scratch);
            transform(fft, buffer, scratch, correction);
        }
        for i in 0..len {
            let slot = (start + i) & (size - 1);
            let window = layout.window[skip + i];
            norm[slot] += window * window;
            if correction.is_some() {
                residual[slot] += buffer[skip + i].re * window / size as f64;
            }
        }
        let until = if t + 1 == frame_count {
            n
        } else {
            (centre + layout.hop).saturating_sub(offset).min(n)
        };
        for (k, sample) in (written..until).zip(&mut output[written..until]) {
            let slot = k & (size - 1);
            let mut value = f64::from(*sample) + residual[slot] / norm[slot].max(1e-9);
            if let Some(bound) = sample_bound {
                value = value.clamp(-bound, bound);
            }
            *sample = value as f32;
            residual[slot] = 0.0;
            norm[slot] = 0.0;
        }
        written = until;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn profile(detector_bias: f64) -> ArticulationProfile {
        ArticulationProfile {
            version: 2,
            frication: None,
            weights: vec![vec![0.0; 37]; 18],
            bias: [0.0; 18],
            detector: ArticulationDetector {
                mean: [0.0; 24],
                scale: [1.0; 24],
                weights: [
                    vec![vec![0.0; 24]; 24],
                    vec![vec![0.0; 24]; 12],
                    vec![vec![0.0; 12]],
                ],
                bias: [vec![0.0; 24], vec![0.0; 12], vec![detector_bias]],
            },
        }
    }

    fn singing(samples: usize, frequency: f64, phase: f64) -> Vec<f32> {
        (0..samples)
            .map(|i| {
                let t = std::f64::consts::TAU * i as f64 / 40_000.0;
                let envelope = 0.16 + 0.07 * (18.0 * t + phase).sin();
                (envelope
                    * ((frequency * t + phase).sin()
                        + 0.17 * (5.0 * frequency * t).cos()
                        + 0.11 * (6110.0 * t + phase).sin())) as f32
            })
            .collect()
    }

    #[test]
    fn closed_detector_preserves_voiced_generator() {
        let source = singing(5803, 360.0, 0.1);
        let original = singing(5901, 220.0, 0.5);
        for version in [2, 3] {
            let mut profile = profile(-30.0);
            profile.version = version;
            let mut output = original.clone();
            apply(&mut output, &source, 40_000, &profile, 0);
            assert_eq!(output, original);
        }
    }

    #[test]
    fn partial_frame_and_reconstruction_match_numpy_reference() {
        let source = singing(5803, 360.0, 0.1);
        let original = singing(5901, 220.0, 0.5);
        let mut output = original.clone();
        apply(&mut output, &source, 40_000, &profile(30.0), 0);
        // Offline NumPy reference with the same waveform, regression and open detector.
        for (i, expected) in [
            (0, -0.06027523),
            (73, 0.03532251),
            (641, -0.03083658),
            (1524, 0.00000055),
            (3461, -0.00330694),
            (5278, 0.04759625),
            (5794, 0.04390323),
        ] {
            assert!(
                (output[i] - expected).abs() < 2e-6,
                "sample {i}: {}",
                output[i]
            );
        }
        assert_eq!(output[source.len()..], original[source.len()..]);
    }

    #[test]
    fn sustained_event_level_matches_numpy_v3_on_partial_frames() {
        let source = singing(5803, 360.0, 0.1);
        let original = singing(5901, 220.0, 0.5);
        let mut profile = profile(30.0);
        profile.version = 3;
        let mut output = original.clone();
        apply(&mut output, &source, 40_000, &profile, 0);
        for (i, expected) in [
            (0, -0.117377026),
            (73, 0.055846144),
            (641, -0.115118706),
            (1524, -0.000113559),
            (3461, -0.01785543),
            (5278, 0.197433774),
            (5794, 0.097505418),
        ] {
            assert!(
                (f64::from(output[i]) - expected).abs() < 2e-6,
                "sample {i}: {}",
                output[i]
            );
        }
        assert_eq!(output[source.len()..], original[source.len()..]);
    }

    #[test]
    fn v3_suppresses_a_brief_rough_event_and_centres_fricative_colour() {
        let frames = |n, ramp| {
            (0..n)
                .map(|i| Frame {
                    level: 0.0,
                    band: [if ramp { i as f64 } else { 0.0 }; BANDS],
                    share: [0.0; BANDS],
                    dispersion: [1.0; BANDS],
                    flat: [0.0; 4],
                    high: 1.0,
                })
                .collect::<Vec<_>>()
        };
        let src = frames(121, true);
        let core = frames(121, false);
        let mut isolated = [[0.0]; 121];
        isolated[57..=63].fill([1.0]);
        assert_eq!(events(&src, &isolated, 2)[60].rough, 1.0);
        let gated = events(&src, &isolated, 3);
        assert_eq!(gated[60].rough, 0.0);
        assert_eq!(gated[60].fricative, 1.0);
        assert_eq!(fricative_offset(&src, &core, &gated[..6]), [0.0; BANDS]);
        assert_eq!(fricative_offset(&src, &core, &gated[..7]), [3.0; BANDS]);
        assert_eq!(fricative_offset(&src, &core, &gated[..8]), [3.5; BANDS]);
    }

    #[test]
    fn target_frication_calibrates_only_the_speaker_who_produced_the_noise() -> anyhow::Result<()> {
        use crate::catalog::FricationProfile;
        use std::collections::{BTreeMap, BTreeSet};
        let directory = tempfile::tempdir()?;
        let rate = 40_000;
        let voiced = singing(100_000, 170.0, 0.2);
        let mut source = voiced.clone();
        let mut seed = 93817_u32;
        let mut history = [0.0; 33];
        let kernel: [f32; 33] = std::array::from_fn(|i| {
            let n = i as f64 - 16.0;
            let band = if n == 0.0 {
                2.0 * (0.24 - 0.085)
            } else {
                ((0.24 * std::f64::consts::TAU * n).sin()
                    - (0.085 * std::f64::consts::TAU * n).sin())
                    / (std::f64::consts::PI * n)
            };
            (band * (0.5 - 0.5 * (std::f64::consts::TAU * i as f64 / 32.0).cos())) as f32
        });
        for (i, sample) in source[48_000..82_000].iter_mut().enumerate() {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            history[i % 33] = seed as f32 / u32::MAX as f32 - 0.5;
            *sample = 0.3
                * kernel
                    .iter()
                    .enumerate()
                    .map(|(k, tap)| tap * history[(i + 33 - k) % 33])
                    .sum::<f32>();
        }
        let mut files = BTreeMap::new();
        for (id, signal) in [(4, &source), (12, &voiced)] {
            let path = directory.path().join(format!("{id}.wav"));
            let spec = hound::WavSpec {
                channels: 1,
                sample_rate: rate,
                bits_per_sample: 32,
                sample_format: hound::SampleFormat::Float,
            };
            let mut wav = hound::WavWriter::create(&path, spec)?;
            for &v in signal {
                wav.write_sample(v)?;
            }
            wav.finalize()?;
            files.insert(id, BTreeSet::from([path]));
        }
        let mut profile = profile(-30.0);
        profile.version = 3;
        let targets = frication::fit(files, rate, &profile.detector)?;
        assert!(targets[&4].frames[0] > 20);
        assert_eq!(targets[&12].frames, [0, 0]);
        let mut dry = voiced.clone();
        apply(&mut dry, &source, rate, &profile, 4);
        profile.frication = Some(FricationProfile {
            version: 1,
            target_mix: [0.75; 9],
            band_strength: [1.0; 9],
            shape_strength: 0.76,
            level_strength: 0.30,
            targets,
        });
        profile.validate()?;
        let mut output = voiced.clone();
        apply(&mut output, &source, rate, &profile, 4);
        assert_eq!(&output[12_000..32_000], &dry[12_000..32_000]);
        assert_ne!(&output[60_000..70_000], &dry[60_000..70_000]);
        for speaker in [12, 27] {
            let mut output = voiced.clone();
            apply(&mut output, &source, rate, &profile, speaker);
            assert_eq!(output, dry);
        }
        Ok(())
    }

    #[test]
    fn invalid_profile_dimensions_are_rejected() {
        let mut p = profile(0.0);
        assert!(p.validate().is_ok());
        p.version = 3;
        assert!(p.validate().is_ok());
        p.version = 4;
        assert!(p.validate().is_err());
        p.version = 3;
        p.detector.weights[1][0].pop();
        assert!(p.validate().is_err());
    }
}
