#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Pcm {
    Float32,
    Integer { bits: u16, valid: u16 },
}

impl Pcm {
    pub(super) fn bits(self) -> u16 {
        match self {
            Self::Float32 => 32,
            Self::Integer { bits, .. } => bits,
        }
    }

    pub(super) fn valid_bits(self) -> u16 {
        match self {
            Self::Float32 => 32,
            Self::Integer { valid, .. } => valid,
        }
    }

    pub(super) fn bytes(self) -> usize {
        usize::from(self.bits() / 8)
    }

    fn decode(self, sample: &[u8]) -> f32 {
        match self {
            Self::Float32 => f32::from_le_bytes([sample[0], sample[1], sample[2], sample[3]]),
            Self::Integer { bits: 8, .. } => (f32::from(sample[0]) - 128.0) / 128.0,
            Self::Integer { bits: 16, .. } => {
                f32::from(i16::from_le_bytes([sample[0], sample[1]])) / 32768.0
            }
            Self::Integer { bits: 24, .. } => {
                i32::from_le_bytes([0, sample[0], sample[1], sample[2]]) as f32 / 2147483648.0
            }
            Self::Integer { .. } => {
                i32::from_le_bytes([sample[0], sample[1], sample[2], sample[3]]) as f32
                    / 2147483648.0
            }
        }
    }

    fn encode(self, sample: f32) -> [u8; 4] {
        let sample = if sample.is_finite() {
            sample.clamp(-1.0, 1.0)
        } else {
            0.0
        };
        if self == Self::Float32 {
            return sample.to_le_bytes();
        }
        let bits = self.bits();
        let valid = self.valid_bits();
        let scale = 1_i64 << (valid - 1);
        let value = ((f64::from(sample) * scale as f64).round() as i64).clamp(-scale, scale - 1);
        // WAVEFORMATEXTENSIBLE puts the valid bits at the high end of the PCM container.
        let word = value << (bits - valid);
        if bits == 8 {
            [((word + 128) as u8), 0, 0, 0]
        } else {
            (word as i32).to_le_bytes()
        }
    }

    pub(super) fn downmix(self, frame: &[u8]) -> f32 {
        let channels = frame.chunks_exact(self.bytes());
        let count = channels.len();
        channels.map(|sample| self.decode(sample)).sum::<f32>() / count as f32
    }

    pub(super) fn render(self, frame: &mut [u8], mono: f32) {
        let bytes = self.bytes();
        let encoded = self.encode(mono);
        for channel in frame.chunks_exact_mut(bytes) {
            channel.copy_from_slice(&encoded[..bytes]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_pcm_preserves_mono_levels_and_channel_order() {
        for pcm in [
            Pcm::Float32,
            Pcm::Integer { bits: 8, valid: 8 },
            Pcm::Integer {
                bits: 16,
                valid: 16,
            },
            Pcm::Integer {
                bits: 24,
                valid: 24,
            },
            Pcm::Integer {
                bits: 32,
                valid: 24,
            },
            Pcm::Integer {
                bits: 32,
                valid: 32,
            },
        ] {
            let bytes = pcm.bytes();
            let mut samples = [0u8; 8];
            pcm.render(&mut samples[..2 * bytes], -0.5);
            assert_eq!(&samples[..bytes], &samples[bytes..2 * bytes]);
            assert_eq!(pcm.downmix(&samples[..2 * bytes]), -0.5);
            pcm.render(&mut samples[bytes..2 * bytes], 0.75);
            assert_eq!(pcm.downmix(&samples[..2 * bytes]), 0.125);
            pcm.render(&mut samples[..2 * bytes], f32::NAN);
            assert_eq!(pcm.downmix(&samples[..2 * bytes]), 0.0);
        }
    }

    #[test]
    fn device_pcm_saturates_and_zeros_padding() {
        let packed = Pcm::Integer {
            bits: 24,
            valid: 24,
        };
        assert_eq!(&packed.encode(3.0)[..3], &[0xff, 0xff, 0x7f]);
        assert_eq!(&packed.encode(-2.0)[..3], &[0x00, 0x00, 0x80]);
        let padded = Pcm::Integer {
            bits: 32,
            valid: 24,
        };
        assert_eq!(padded.encode(3.0), [0x00, 0xff, 0xff, 0x7f]);
        assert_eq!(padded.encode(-2.0), [0x00, 0x00, 0x00, 0x80]);
        let unsigned = Pcm::Integer { bits: 8, valid: 8 };
        assert_eq!(unsigned.encode(0.0)[0], 128);
    }
}
