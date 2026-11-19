use anyhow::Result;
use tch::{Device, Kind, Tensor};

/// Stationary spectral gate used by the realtime desktop path. STFTs run on
/// the audio worker and never in a device callback.
pub struct SpectralGate {
    size: i64,
    hop: i64,
    window: Tensor,
    smoothing: Option<(Tensor, [i64; 2])>,
}

impl SpectralGate {
    pub fn new(sample_rate: u32) -> Result<Self> {
        let hop = i64::from(sample_rate / 100);
        let size = 4 * hop;
        let window = Tensor::f_hann_window_periodic(size, true, (Kind::Float, Device::Cpu))?;
        let time = 5_i64;
        let frequency = (500 * (size / 2) / i64::from(sample_rate)).max(1);
        let coefficients = |radius: i64| -> Result<Tensor> {
            let values: Vec<_> = (-radius..=radius)
                .map(|i| 1.0 - i.abs() as f32 / (radius + 1) as f32)
                .collect();
            Ok(Tensor::f_from_slice(&values)?)
        };
        let filter = coefficients(frequency)?.unsqueeze(1) * coefficients(time)?.unsqueeze(0);
        let smoothing = &filter / filter.sum(Kind::Float);
        Ok(Self {
            size,
            hop,
            window,
            smoothing: Some((smoothing.unsqueeze(0).unsqueeze(0), [frequency, time])),
        })
    }

    pub fn apply(&self, input: &[f32], noise: &[f32]) -> Result<Vec<f32>> {
        let transform = |audio: &[f32]| -> Result<Tensor> {
            let wave = Tensor::f_from_slice(audio)?
                .unsqueeze(0)
                .constant_pad_nd([self.size / 2, self.size / 2]);
            Ok(wave.f_stft(
                self.size,
                self.hop,
                self.size,
                Some(&self.window),
                false,
                true,
                true,
                false,
            )?)
        };
        let to_db = |spectrum: &Tensor| {
            let db: Tensor = (spectrum.abs() + f64::EPSILON).log10() * 20.0;
            db.maximum(&(db.max_dim(-1, true).0 - 40.0))
        };
        tch::no_grad(|| {
            let signal = transform(input)?;
            let noise_db = to_db(&transform(noise)?);
            let (std, mean) = noise_db.f_std_mean_dim([-1].as_slice(), true, true)?;
            let threshold = mean + std * 1.5;
            let mask: Tensor =
                to_db(&signal).gt_tensor(&threshold).to_kind(Kind::Float) * 0.9 + 0.1;
            let mask = if let Some((kernel, pad)) = &self.smoothing {
                mask.unsqueeze(1)
                    .f_conv2d(kernel, None::<&Tensor>, [1, 1], *pad, [1, 1], 1)?
                    .squeeze_dim(1)
            } else {
                mask
            };
            let result = (signal * mask).f_istft(
                self.size,
                self.hop,
                self.size,
                Some(&self.window),
                true,
                false,
                true,
                input.len() as i64,
                false,
            )?;
            Ok(Vec::<f32>::try_from(result.view([-1]))?)
        })
    }
}
