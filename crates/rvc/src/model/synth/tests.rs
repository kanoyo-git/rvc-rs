use std::collections::HashMap;

use super::*;

#[test]
fn cropped_decoder_preserves_kept_audio() -> Result<()> {
    let options = (Kind::Float, Device::Cpu);
    let frames = 96;
    let trim = 20;
    let rate = 1_600;
    let hop = i64::from(rate) / 100;
    let latent = Tensor::arange(frames, options).view([1, 1, frames]) / frames as f64 - 0.5;
    let speaker = Tensor::full([1, 1, 1], 0.3, options);
    for (resblock, upsample_kernels, aa) in [
        (1, [8, 6], false),
        (2, [6, 6], false),
        (2, [7, 6], false),
        (1, [8, 6], true),
        (2, [6, 6], true),
    ] {
        let mut weights = HashMap::new();
        let mut kernel = |name: String, size| {
            let tensor = Tensor::arange(size, options).view([1, 1, size]) / size as f64 * 0.1;
            weights.insert(format!("{name}.weight"), tensor);
        };
        kernel("dec.conv_pre".into(), 7);
        kernel("dec.cond".into(), 1);
        kernel("dec.conv_post".into(), 7);
        for (stage, width) in upsample_kernels.into_iter().enumerate() {
            kernel(format!("dec.ups.{stage}"), width);
            if resblock == 1 {
                for layer in 0..3 {
                    kernel(format!("dec.resblocks.{stage}.convs1.{layer}"), 7);
                    kernel(format!("dec.resblocks.{stage}.convs2.{layer}"), 7);
                }
            } else {
                for layer in 0..2 {
                    kernel(format!("dec.resblocks.{stage}.convs.{layer}"), 7);
                }
            }
        }
        let decoder = Synth {
            #[cfg(feature = "onnx")]
            onnx: Default::default(),
            info: ModelInfo {
                speaker_count: 1,
                speakers: Default::default(),
                version: "v2".into(),
                uses_f0: false,
                sample_rate: Some(rate),
                info: None,
                format: Default::default(),
            },
            rate,
            weights: Weights::new(weights),
            config: Config {
                inter: 1,
                hidden: 1,
                heads: 1,
                encoder_layers: 1,
                encoder_kernel: 3,
                dropout: 0.0,
                resblock,
                kernels: vec![7],
                dilations: vec![vec![1, 3, 5]],
                upsample_rates: vec![4, 4],
                upsample_kernels: upsample_kernels.to_vec(),
                aa,
            },
            graph: RefCell::default(),
        };
        let full = decoder.decode(&latent, None, &speaker, frames, 0, false)?;
        let cropped = decoder.decode(&latent, None, &speaker, frames, trim, false)?;
        let expected = full.narrow(2, trim * hop, full.size()[2] - 2 * trim * hop);
        assert_eq!(cropped.size(), expected.size());
        let error = f64::try_from((cropped - expected).abs().max())?;
        assert!(error < 1e-6, "resblock {resblock}, aa {aa}, error {error}");
    }
    Ok(())
}
