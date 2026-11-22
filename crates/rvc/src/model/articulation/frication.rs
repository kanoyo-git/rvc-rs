//! File-level noise correction with independent checks on unvoiced energy, bandwidth and sustained rough phonation.
use super::{BANDS, Events, Frame, Spectrum, db, reconstruct, smooth};
use crate::catalog::{ArticulationDetector, FricationProfile, FricationTemplate};
use anyhow::Result;
use rustfft::num_complex::Complex64;
use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;
use std::path::PathBuf;

struct SourceNoise {
    high: f64,
    low: f64,
    best_flat: f64,
    level_dbfs: f64,
}

pub(super) struct Analysis {
    total: Range<usize>,
    high: Range<usize>,
    low: Range<usize>,
    upper: [Range<usize>; 4],
    reference_db: f64,
    frames: Vec<SourceNoise>,
}

impl Analysis {
    pub(super) fn new(layout: &Spectrum, rate: u32, frames: usize) -> Self {
        let size = layout.fft.len();
        let bin = |hz: usize| (hz * size).div_ceil(rate as usize).min(size / 2 + 1);
        let range = |lo, hi| bin(lo)..bin(hi);
        Self {
            total: range(80, 20000),
            high: range(3500, 12000),
            low: range(80, 1000),
            upper: [(2500, 4500), (4000, 6500), (6000, 7900), (8000, 12000)]
                .map(|(a, b)| range(a, b)),
            reference_db: db(size as f64 * layout.window.iter().map(|w| w * w).sum::<f64>() / 2.0),
            frames: Vec::with_capacity(frames),
        }
    }

    pub(super) fn observe(&mut self, power: &[f64], level: f64) {
        let energy = |bins: &Range<usize>| power[bins.clone()].iter().sum::<f64>();
        let total = energy(&self.total).max(1e-12);
        let mut best_flat: f64 = -60.0;
        for bins in &self.upper {
            let sum = energy(bins);
            if sum <= 0.025 * total || bins.is_empty() {
                continue;
            }
            let len = bins.len() as f64;
            let log = power[bins.clone()]
                .iter()
                .map(|p| (p + 1e-10).ln())
                .sum::<f64>()
                / len;
            best_flat = best_flat.max(db(log.exp() / (sum / len + 1e-10)));
        }
        self.frames.push(SourceNoise {
            high: energy(&self.high) / total,
            low: energy(&self.low) / total,
            best_flat,
            level_dbfs: level - self.reference_db,
        });
    }

    fn masks(
        &self,
        src: &[Frame],
        dynamic: &[[f64; 6]],
        events: &[Events],
    ) -> Vec<([f64; 2], bool)> {
        self.frames
            .iter()
            .enumerate()
            .map(|(t, ctx)| {
                let s = &src[t];
                let period = dynamic[t][1];
                let active =
                    smooth((s.level + 46.0) / 9.0) * smooth((ctx.level_dbfs + 93.0) / 12.0);
                let clean = 1.0 - smooth((events[t].rough_density - 0.14) / 0.32);
                let fric = smooth((ctx.high - 0.34) / 0.25)
                    * smooth((0.52 - ctx.low) / 0.27)
                    * smooth((ctx.best_flat + 13.0) / 8.0)
                    * (1.0 - 0.92 * smooth((period - 0.76) / 0.18));
                let breath = smooth((s.flat[1] + 11.0) / 6.0)
                    * smooth((0.67 - ctx.low) / 0.24)
                    * smooth((-s.level - 7.0) / 8.0)
                    * (1.0 - fric)
                    * smooth((0.84 - period) / 0.28);
                (
                    [fric * active * clean, breath * active * clean],
                    ctx.low > 0.77 && s.level > -26.0 && period > 0.65,
                )
            })
            .collect()
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn apply(
        self,
        output: &mut [f32],
        n: usize,
        layout: &Spectrum,
        src: &[Frame],
        dynamic: &[[f64; 6]],
        events: &[Events],
        params: &FricationProfile,
        template: &FricationTemplate,
    ) {
        let masks = self.masks(src, dynamic, events);
        if !masks.iter().any(|(w, _)| w[0] + w[1] > 0.0) {
            return;
        }
        let size = layout.fft.len();
        let mut fft = vec![Complex64::default(); size];
        let mut buffer = fft.clone();
        let mut scratch = vec![
            Complex64::default();
            layout
                .fft
                .get_inplace_scratch_len()
                .max(layout.inverse.get_inplace_scratch_len())
        ];
        let frames = src.len();
        let mut levels = Vec::with_capacity(frames);
        let mut shares = Vec::with_capacity(frames);
        for t in 0..frames {
            layout.analyze(&output[..n], t, &mut fft, &mut scratch);
            let energy = layout
                .bands
                .each_ref()
                .map(|range| fft[range.clone()].iter().map(|p| p.norm_sqr()).sum::<f64>());
            let level = db(energy.iter().sum());
            levels.push(level);
            shares.push(energy.map(|p| (db(p) - level).max(-65.0)));
        }
        let mut sorted = levels.clone();
        sorted.sort_unstable_by(f64::total_cmp);
        let pos = 0.9 * (frames - 1) as f64;
        let peak = sorted[pos.floor() as usize]
            + pos.fract() * (sorted[pos.ceil() as usize] - sorted[pos.floor() as usize]);
        for x in &mut levels {
            *x -= peak;
        }
        let src_anchor = voiced_level(src.iter().map(|s| s.level), &masks);
        let core_anchor = voiced_level(levels.iter().copied(), &masks);
        let mut observed = Accum::default();
        observed.add(src, &masks, src_anchor);
        let incoming = observed.finish();
        let available =
            std::array::from_fn::<_, BANDS, _>(|b| src.iter().any(|f| f.band[b] > -46.0));
        reconstruct(
            layout,
            output,
            n,
            frames,
            (&mut fft, &mut buffer, &mut scratch),
            Some(0.995),
            |t| {
                let mut gains = [0.0; BANDS];
                let mut sum = 0.0;
                let mut level = 0.0;
                for class in 0..2 {
                    let w = masks[t].0[class];
                    if w == 0.0 || template.frames[class] < 8 || incoming.frames[class] == 0 {
                        continue;
                    }
                    sum += w;
                    let src_level = src[t].level - src_anchor;
                    let core_level = levels[t] - core_anchor;
                    level += w * params.level_strength * (src_level - core_level).clamp(-7.0, 7.0);
                    for b in 0..BANDS {
                        if !available[b] {
                            continue;
                        }
                        let target = template.mean[class][b];
                        let generated = shares[t][b];
                        let source = src[t].share[b];
                        let target_delta = target
                            + 0.42 * (source - incoming.mean[class][b])
                            + 0.12 * (generated - target)
                            - generated;
                        let mix = params.target_mix[b];
                        gains[b] += w
                            * params.shape_strength
                            * (mix * target_delta + (1.0 - mix) * (source - generated))
                                .clamp(-13.0, 13.0);
                    }
                }
                if sum == 0.0 {
                    return None;
                }
                let mut before = 0.0;
                let mut after = 0.0;
                for b in 0..BANDS {
                    let power = 10f64.powf(shares[t][b] / 10.0);
                    before += power;
                    after += power * 10f64.powf(gains[b] / 10.0);
                }
                let rendered_level = db(after) - db(before);
                Some(std::array::from_fn::<_, BANDS, _>(|b| {
                    if available[b] {
                        10f64.powf(
                            ((gains[b] + level - rendered_level).clamp(-12.0, 12.0)
                                * params.band_strength[b])
                                / 20.0,
                        )
                    } else {
                        1.0
                    }
                }))
            },
            |fft, delta, scratch, gains| {
                delta.fill(Complex64::default());
                for (b, bins) in layout.bands.iter().enumerate() {
                    for k in bins.clone() {
                        delta[k] = fft[k] * (gains[b] - 1.0);
                        if k != 0 && k != size / 2 {
                            delta[size - k] = delta[k].conj();
                        }
                    }
                }
                layout.inverse.process_with_scratch(delta, scratch);
            },
        );
    }
}

fn voiced_level(levels: impl Iterator<Item = f64>, masks: &[([f64; 2], bool)]) -> f64 {
    let mut sum = 0.0;
    let mut n = 0;
    for (level, (_, voiced)) in levels.zip(masks) {
        if *voiced {
            n += 1;
            sum += 10f64.powf(level / 10.0);
        }
    }
    if n >= 5 { db(sum / n as f64) } else { 0.0 }
}

#[derive(Default)]
struct Accum {
    power: [[f64; 10]; 2],
    weight: [f64; 2],
    count: [usize; 2],
}

impl Accum {
    fn add(&mut self, src: &[Frame], masks: &[([f64; 2], bool)], anchor: f64) {
        for (f, (mask, _)) in src.iter().zip(masks) {
            for (class, &w) in mask.iter().enumerate() {
                if w <= 0.38 {
                    continue;
                }
                self.weight[class] += w;
                self.count[class] += 1;
                for b in 0..=BANDS {
                    let value = if b == BANDS {
                        f.level - anchor
                    } else {
                        f.share[b]
                    };
                    self.power[class][b] += w * 10f64.powf(value.clamp(-60.0, 9.0) / 10.0);
                }
            }
        }
    }
    fn finish(self) -> FricationTemplate {
        FricationTemplate {
            frames: self.count,
            mean: std::array::from_fn(|i| {
                self.power[i].map(|p| {
                    if self.weight[i] > 0.0 {
                        db((p / self.weight[i]).max(1e-10))
                    } else {
                        0.0
                    }
                })
            }),
        }
    }
}

/// Template estimation reads only the clean training audio for each speaker.
/// Each frame is anchored to voiced energy within its own training clip.
pub(crate) fn fit(
    sources: BTreeMap<i64, BTreeSet<PathBuf>>,
    rate: u32,
    detector: &ArticulationDetector,
) -> Result<BTreeMap<i64, FricationTemplate>> {
    let layout = Spectrum::new(rate);
    let size = layout.fft.len();
    let mut fft = vec![Complex64::default(); size];
    let mut buf = fft.clone();
    let mut power = vec![0.0; size / 2 + 1];
    let mut scratch = vec![
        Complex64::default();
        layout
            .fft
            .get_inplace_scratch_len()
            .max(layout.inverse.get_inplace_scratch_len())
    ];
    let mut targets = BTreeMap::new();
    for (sid, files) in sources {
        let mut accumulator = Accum::default();
        for file in files {
            let audio = crate::audio::read_mono(&file, rate)?;
            if audio.is_empty() {
                continue;
            }
            let n = audio.len().div_ceil(layout.hop) + 1;
            let mut source = Vec::with_capacity(n);
            let mut dynamic = Vec::with_capacity(n);
            let mut analysis = Analysis::new(&layout, rate, n);
            for t in 0..n {
                layout.analyze(&audio, t, &mut fft, &mut scratch);
                for (p, c) in power.iter_mut().zip(&fft) {
                    *p = c.norm_sqr();
                }
                let frame = Frame::new(&layout, &power);
                analysis.observe(&power, frame.level);
                let period = layout.periodicity(&power, &mut buf, &mut scratch);
                let flux = source.last().map_or(0.0, |last: &Frame| {
                    frame
                        .share
                        .iter()
                        .zip(last.share)
                        .map(|(p, q)| (p - q).abs().min(12.0))
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
                source.push(frame);
            }
            super::normalize(&mut source);
            let probabilities = super::detector_probabilities(&source, &dynamic, detector);
            let gates = super::events(&source, &probabilities, 3);
            let masks = analysis.masks(&source, &dynamic, &gates);
            accumulator.add(
                &source,
                &masks,
                voiced_level(source.iter().map(|s| s.level), &masks),
            );
        }
        targets.insert(sid, accumulator.finish());
    }
    Ok(targets)
}
