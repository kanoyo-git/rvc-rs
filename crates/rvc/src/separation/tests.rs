use super::*;
use crate::checkpoint::Checkpoint;
use roformer::Bands;

#[test]
fn original_roformers_match_with_local_checkpoints() -> Result<()> {
    tch::set_num_threads(2);
    for (weights, expected, mel) in [
        (
            include_bytes!("fixtures/bs.ckpt").as_slice(),
            include_bytes!("fixtures/bs-prediction.f32").as_slice(),
            false,
        ),
        (
            include_bytes!("fixtures/mel.ckpt").as_slice(),
            include_bytes!("fixtures/mel-prediction.f32").as_slice(),
            true,
        ),
    ] {
        let mut checkpoint = NamedTempFile::new()?;
        checkpoint.write_all(weights)?;
        checkpoint.flush()?;
        let tensors = Checkpoint::open(checkpoint.path())?.weights(None, Device::Cpu)?;
        let model = Roformer::load(
            ModelConfig {
                bands: if mel {
                    Bands::Mel {
                        count: 4,
                        sample_rate: 16_000,
                    }
                } else {
                    Bands::Split(vec![4, 4, 8, 17])
                },
                channels: 2,
                stems: 2,
                embedding_dim: 8,
                depth: 2,
                time_depth: 2,
                frequency_depth: 1,
                heads: 2,
                head_dim: 4,
                flash_attention: true,
                fft: 64,
                hop: 8,
                window: if mel { 64 } else { 48 },
                normalized_stft: mel,
                zero_dc: mel,
                match_input_length: !mel,
            },
            &tensors,
            Device::Cpu,
        )?;
        let input: Vec<f32> = (0..350).map(|i| ((i % 19) as f32 - 8.0) / 32.0).collect();
        let actual = model.predict_batch(&input, 1)?;
        let (expected, tail) = expected.as_chunks::<4>();
        assert!(tail.is_empty());
        assert_eq!(actual.len(), expected.len());
        let max_error = actual
            .iter()
            .copied()
            .zip(expected.iter().copied().map(f32::from_le_bytes))
            .map(|(sample, reference)| (sample - reference).abs())
            .fold(0.0, f32::max);
        assert!(
            max_error < 1e-6,
            "maximum waveform error {max_error}, mel={mel}"
        );
        let second: Vec<f32> = input.iter().map(|sample| -0.7 * sample).collect();
        let batched = model.predict_batch(&[input, second.clone()].concat(), 2)?;
        let individual = [actual, model.predict_batch(&second, 1)?].concat();
        assert_eq!(batched.len(), individual.len());
        for (result, expected) in batched.into_iter().zip(individual) {
            assert!((result - expected).abs() < 1e-6);
        }
    }
    Ok(())
}

#[test]
fn configured_batches_preserve_stem_and_chunk_order_across_replicas() -> Result<()> {
    tch::set_num_threads(2);
    let mut checkpoint = NamedTempFile::new()?;
    checkpoint.write_all(include_bytes!("fixtures/mel.ckpt"))?;
    checkpoint.flush()?;
    let mut config = NamedTempFile::new()?;
    config.write_all(
        br"
audio: {chunk_size: 175, sample_rate: 16000}
inference: {overlap_size: 46, batch_size: 3}
training: {instruments: [lead, backing]}
model:
  stereo: true
  num_stems: 2
  num_bands: 4
  dim: 8
  depth: 2
  heads: 2
  dim_head: 4
  time_transformer_depth: 2
  freq_transformer_depth: 1
  flash_attn: true
  stft_n_fft: 64
  stft_win_length: 64
  stft_hop_length: 8
  stft_normalized: true
  sample_rate: 16000
",
    )?;
    config.flush()?;
    let load = |options: &[String], extra: &[Device]| {
        ConfiguredModel::load_on_devices(
            "mel_band_roformer",
            checkpoint.path(),
            config.path(),
            Device::Cpu,
            extra,
            &InferenceParams::parse(options)?,
        )
    };
    let baseline = load(&["batch_size=1".into()], &[])?;
    // Two CPU replicas exercise the same scheduling and join order as CUDA.
    let parallel = load(&["stem_batch_size=1".into()], &[Device::Cpu])?;
    let pcm: Vec<f32> = (0..1414)
        .map(|i| ((i % 67) as f32 - 33.0) / 128.0)
        .collect();
    let stems = vec!["backing".to_owned(), "lead".to_owned()];
    let expected = baseline.separate_planar(&pcm, &stems, |_| ControlFlow::Continue(()))?;
    let actual = parallel.separate_planar(&pcm, &stems, |_| ControlFlow::Continue(()))?;
    assert_eq!(actual.len(), stems.len());
    for ((label, actual), (expected_label, expected)) in actual.into_iter().zip(expected) {
        assert_eq!(label, expected_label);
        assert_eq!(actual.len(), expected.len());
        assert!(
            actual
                .iter()
                .zip(expected)
                .all(|(a, b)| (a - b).abs() <= 1e-6)
        );
    }
    Ok(())
}

#[test]
fn separation_plans_legacy_output_names_and_preserves_existing_files() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let input = dir.path().join("inputs");
    let output = dir.path().join("outputs");
    let secondary = dir.path().join("secondary");
    std::fs::create_dir(&input)?;
    std::fs::File::create(input.join("作業.wav"))?;
    std::fs::File::create(input.join("ignore.txt"))?;
    let options = Options {
        preset: "karaoke-mel-roformer-10.1956",
        device: Device::Cpu,
        format: Format::Flac,
        secondary_output: Some(&secondary),
        overwrite: false,
    };
    let spec = preset(options.preset).unwrap();
    let files = plan_jobs(&input, &output, &options, spec)?;
    assert_eq!(files.len(), 1);
    assert_eq!(
        files[0].primary.file_name().unwrap(),
        "作業_main_vocal.flac"
    );
    assert_eq!(
        files[0].secondary,
        secondary.canonicalize()?.join("作業_off_vocal.flac")
    );
    std::fs::write(&files[0].secondary, "user audio")?;
    assert!(plan_jobs(&input, &output, &options, spec).is_err());
    assert_eq!(std::fs::read_to_string(&files[0].secondary)?, "user audio");
    Ok(())
}

#[test]
fn wav_stems_are_float_stereo_and_publish_atomically() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("track_vocals.wav");
    let audio = [0.5f32, -0.25, 0.125, -0.0625];
    let temporary = stage_stem(
        dir.path(),
        &path,
        &audio,
        PRESET_PCM,
        Format::Wav,
        &AudioEncoding::default(),
    )?;
    assert!(!path.exists());
    persist_stem(temporary, &path, false)?;
    let mut reader = hound::WavReader::open(&path)?;
    let spec = reader.spec();
    assert_eq!(spec.channels, 2);
    assert_eq!(spec.sample_rate, SAMPLE_RATE);
    assert_eq!(spec.bits_per_sample, 32);
    assert_eq!(spec.sample_format, hound::SampleFormat::Float);
    let interleaved = reader.samples::<f32>().collect::<hound::Result<Vec<_>>>()?;
    assert_eq!(interleaved, [audio[0], audio[2], audio[1], audio[3]]);
    Ok(())
}

#[test]
fn shipped_presets_read_their_model_yaml() -> Result<()> {
    let directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../assets/pymss_weights");
    for preset in PRESETS {
        let document = configurable::load_document(&directory.join(preset.config))?;
        let config = ModelConfig::from_document(
            &document,
            preset.architecture == Architecture::MelBandRoformer,
        )?;
        assert_eq!(config.channels, 2);
        assert_eq!(config.stems, 1);
        assert!(matches!(config.hop, 441 | 512), "{}", config.hop);
        assert_eq!(document["training"]["target_instrument"], preset.model_stem);
        if let Bands::Split(widths) = config.bands {
            assert_eq!(widths.iter().sum::<i64>(), config.fft / 2 + 1);
        }
    }
    Ok(())
}
