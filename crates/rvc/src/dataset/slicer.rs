use std::ops::Range;

pub(super) struct Slicer {
    hop: usize,
    window: usize,
    minimum_clip: usize,
    minimum_silence: usize,
    retained_silence: usize,
}

impl Slicer {
    pub(super) fn new(rate: u32) -> Self {
        let samples = |milliseconds: f64| f64::from(rate) * milliseconds / 1000.0;
        let hop = samples(15.0).round_ties_even() as usize;
        let frames = |milliseconds| (samples(milliseconds) / hop as f64).round_ties_even() as usize;
        Self {
            hop,
            window: (4 * hop).min(samples(400.0).round_ties_even() as usize),
            minimum_clip: frames(1500.0),
            minimum_silence: frames(400.0),
            retained_silence: frames(500.0),
        }
    }

    pub(super) fn slices(&self, waveform: &[f64]) -> Vec<Range<usize>> {
        if waveform.len() <= self.minimum_clip {
            return std::iter::once(0..waveform.len()).collect();
        }
        let energy = self.frame_energy(waveform);
        let retained = self.retained_silence;
        let threshold = 10f64.powf(-42.0 / 10.0);
        let mut removed = Vec::new();
        let mut silence = None;
        let mut clip_start = 0;
        for (frame, &power) in energy.iter().enumerate() {
            if power < threshold {
                silence.get_or_insert(frame);
                continue;
            }
            let Some(start) = silence.take() else {
                continue;
            };
            let length = frame - start;
            let leading = start == 0 && frame > retained;
            let middle = length >= self.minimum_silence && frame - clip_start >= self.minimum_clip;
            if !leading && !middle {
                continue;
            }
            let (left, right) = if length <= retained {
                let point = quietest(&energy, start, frame);
                (point, point)
            } else {
                let left = quietest(&energy, start, start + retained);
                let right = quietest(&energy, frame - retained, frame);
                if length <= retained * 2 && start != 0 {
                    let center = quietest(&energy, frame - retained, start + retained);
                    (left.min(center), right.max(center))
                } else {
                    (left, right)
                }
            };
            removed.push((if start == 0 { 0 } else { left }, right));
            clip_start = right;
        }
        let total = energy.len();
        if let Some(start) = silence.filter(|&start| total - start >= self.minimum_silence) {
            let point = quietest(&energy, start, (start + retained).min(total - 1));
            removed.push((point, total + 1));
        }
        if removed.is_empty() {
            return std::iter::once(0..waveform.len()).collect();
        }
        let clip =
            |start: usize, end: usize| start * self.hop..(end * self.hop).min(waveform.len());
        let mut chunks = Vec::new();
        if removed[0].0 > 0 {
            chunks.push(clip(0, removed[0].0));
        }
        for pair in removed.windows(2) {
            chunks.push(clip(pair[0].1, pair[1].0));
        }
        let end = removed[removed.len() - 1].1;
        if end < total {
            chunks.push(clip(end, total));
        }
        chunks
    }

    fn frame_energy(&self, waveform: &[f64]) -> Vec<f64> {
        let half = self.window / 2;
        (0..=waveform.len() / self.hop)
            .map(|frame| {
                let center = frame * self.hop;
                let start = center.saturating_sub(half);
                let end = (center + self.window - half).min(waveform.len());
                waveform[start..end].iter().map(|x| x * x).sum::<f64>() / self.window as f64
            })
            .collect()
    }
}

fn quietest(energy: &[f64], first: usize, last: usize) -> usize {
    let mut minimum = first;
    for index in first + 1..=last {
        if energy[index] < energy[minimum] {
            minimum = index;
        }
    }
    minimum
}
