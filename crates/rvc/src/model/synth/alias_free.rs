//! Anti-aliased LeakyReLU for the NSF-HiFiGAN decoder.
//!
//! A pointwise nonlinearity creates harmonics above the Nyquist frequency of
//! the stage it runs at, and those fold back as inharmonic tones: the
//! "metallic" ring of HiFiGAN vocoders. Following BigVGAN, each activation runs
//! at twice the stage rate between two Kaiser-windowed sinc low-pass filters.
//! The convolution weights are unchanged, so a decoder trained without this
//! starts from the same function and only has to absorb the band limit.

use std::cell::RefCell;

use tch::{Device, Kind, Tensor};

const TAPS: i64 = 12;
/// Context the filters add on each side, in samples of the stage rate.
pub(super) const CONTEXT: i64 = 6;

fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    let half = x / 2.0;
    for k in 1..64 {
        term *= (half / k as f64).powi(2);
        sum += term;
        if term < sum * 1e-17 {
            break;
        }
    }
    sum
}

/// BigVGAN `kaiser_sinc_filter1d(cutoff=0.25, half_width=0.3, kernel_size=12)`.
fn filter() -> [f32; TAPS as usize] {
    let (cutoff, half_width) = (0.25_f64, 0.3_f64);
    let half = (TAPS / 2) as f64;
    let attenuation = 2.285 * (half - 1.0) * std::f64::consts::PI * 4.0 * half_width + 7.95;
    let beta = if attenuation > 50.0 {
        0.1102 * (attenuation - 8.7)
    } else if attenuation >= 21.0 {
        0.5842 * (attenuation - 21.0).powf(0.4) + 0.07886 * (attenuation - 21.0)
    } else {
        0.0
    };
    let n = TAPS as f64 - 1.0;
    let mut taps = [0.0_f64; TAPS as usize];
    for (i, tap) in taps.iter_mut().enumerate() {
        let ratio = 2.0 * i as f64 / n - 1.0;
        let window = bessel_i0(beta * (1.0 - ratio * ratio).max(0.0).sqrt()) / bessel_i0(beta);
        let time = i as f64 - half + 0.5;
        let x = 2.0 * cutoff * time;
        let sinc = if x == 0.0 {
            1.0
        } else {
            (std::f64::consts::PI * x).sin() / (std::f64::consts::PI * x)
        };
        *tap = 2.0 * cutoff * window * sinc;
    }
    let sum: f64 = taps.iter().sum();
    taps.map(|tap| (tap / sum) as f32)
}

thread_local! {
    static FILTERS: RefCell<Vec<(Device, Kind, Tensor)>> = const { RefCell::new(Vec::new()) };
}

fn cached(device: Device, kind: Kind) -> Tensor {
    FILTERS.with(|filters| {
        let mut filters = filters.borrow_mut();
        if let Some((_, _, tensor)) = filters
            .iter()
            .find(|(d, k, _)| *d == device && *k == kind)
        {
            return tensor.shallow_clone();
        }
        let tensor = Tensor::from_slice(&filter())
            .view([1, 1, TAPS])
            .to_device(device)
            .to_kind(kind);
        filters.push((device, kind, tensor.shallow_clone()));
        tensor
    })
}

/// LeakyReLU evaluated at twice the rate of `x` [batch, channels, time].
pub(super) fn leaky(x: &Tensor, slope: f64) -> Tensor {
    let channels = x.size()[1];
    let taps = cached(x.device(), x.kind()).expand([channels, 1, TAPS], false);
    // Upsample: zero-stuff by 2 with the low-pass as the transposed kernel.
    let pad = TAPS / 2 - 1;
    let crop = pad * 2 + (TAPS - 2) / 2;
    let up = x
        .replication_pad1d([pad, pad])
        .conv_transpose1d(&taps, None::<Tensor>, [2], [0], [0], channels, [1])
        * 2.0;
    let length = up.size()[2] - 2 * crop;
    let up = up.narrow(2, crop, length);
    let active = super::super::weights::leaky(&up, slope);
    // Downsample: the same low-pass, then keep every other sample.
    active
        .replication_pad1d([TAPS / 2 - 1, TAPS / 2])
        .conv1d(&taps, None::<Tensor>, [2], [0], [1], channels)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filter_is_symmetric_low_pass_with_unit_gain() {
        let taps = filter();
        let sum: f32 = taps.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        for i in 0..TAPS as usize / 2 {
            assert!((taps[i] - taps[TAPS as usize - 1 - i]).abs() < 1e-7);
        }
    }

    #[test]
    fn keeps_length_and_matches_leaky_relu_on_smooth_input() {
        let t = Tensor::arange(400, (Kind::Float, Device::Cpu)) / 400.0;
        let x = (t * std::f64::consts::TAU * 3.0).sin().view([1, 1, 400]).repeat([2, 3, 1]);
        let y = leaky(&x, 0.1);
        assert_eq!(y.size(), x.size());
        let reference = crate::model::weights::leaky(&x, 0.1);
        let error = (y - reference)
            .narrow(2, 20, 360)
            .abs()
            .max()
            .double_value(&[]);
        assert!(error < 0.03, "{error}");
    }
}
