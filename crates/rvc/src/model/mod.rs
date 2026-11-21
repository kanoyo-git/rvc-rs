mod cuda_graph;
mod discriminator;
pub mod edit;
pub mod extract;
mod fcpe;
#[cfg(feature = "onnx")]
mod fcpe_onnx;
mod hubert;
pub mod aperiodic;
pub mod fx;
pub mod gate;
pub mod layers;
#[cfg(feature = "onnx")]
mod hubert_onnx;
mod nmvpe;
#[cfg(feature = "onnx")]
pub(crate) mod onnx;
mod optimizer;
mod pitch;
pub(crate) mod register;
mod retrieval;
mod rmvpe;
#[cfg(feature = "onnx")]
mod rmvpe_onnx;
mod synth;
pub mod timbre;
pub mod train;
pub mod train_data;
mod train_losses;
pub mod weights;

use std::ffi::OsString;
use std::path::Path;
use std::thread::{self, JoinHandle};

use anyhow::{Context, Result, bail, ensure};
use clap::ValueEnum;
use serde::{Deserialize, Serialize};
use tch::{Device, Kind, Tensor, no_grad};

use crate::audio::{self, ANALYSIS_RATE, FRAME};
use crate::catalog::{Paths, Unvoiced};
use hubert::Hubert;
use retrieval::Index;
use rmvpe::Rmvpe;
use synth::{Synth, Window};

const FCPE_CHECKPOINT: &str = "assets/fcpe/fcpe_c_v001.pt";

fn onnx_selection(stage: &str) -> Option<OsString> {
    std::env::var_os(stage)
        .or_else(|| std::env::var_os("RVC_ONNX_PROVIDER"))
        .filter(|provider| provider != "torch")
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize, ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum PitchMethod {
    #[default]
    Rmvpe,
    Pm,
    Fcpe,
    Nmvpe,
}

impl PitchMethod {
    pub(crate) fn available(self, paths: &Paths) -> bool {
        match self {
            Self::Pm => true,
            Self::Rmvpe => {
                let checkpoint = paths.rmvpe.join("rmvpe.pt").is_file();
                if onnx_selection("RVC_RMVPE_ONNX").is_some() {
                    cfg!(feature = "onnx")
                        && (checkpoint || paths.rmvpe.join("rmvpe.onnx").is_file())
                } else {
                    checkpoint
                }
            }
            Self::Fcpe => {
                (cfg!(feature = "onnx") || onnx_selection("RVC_FCPE_ONNX").is_none())
                    && paths.root.join(FCPE_CHECKPOINT).is_file()
            }
            Self::Nmvpe => paths.nmvpe().is_file(),
        }
    }
}

enum Pitch {
    Rmvpe(Rmvpe),
    #[cfg(feature = "onnx")]
    RmvpeOnnx(rmvpe_onnx::RmvpeOnnx),
    Pm,
    Fcpe(fcpe::Fcpe),
    Nmvpe(nmvpe::Nmvpe),
}

impl Pitch {
    fn load(method: PitchMethod, paths: &Paths, device: Device, kind: Kind) -> Result<Self> {
        if method == PitchMethod::Rmvpe
            && let Some(provider) = onnx_selection("RVC_RMVPE_ONNX")
        {
            #[cfg(feature = "onnx")]
            return Ok(Self::RmvpeOnnx(rmvpe_onnx::RmvpeOnnx::load(
                &paths.rmvpe,
                provider.to_str().context("RVC_RMVPE_ONNX must be text")?,
            )?));
            #[cfg(not(feature = "onnx"))]
            bail!(
                "RMVPE provider {} requires the onnx build feature",
                provider.to_string_lossy()
            );
        }
        match method {
            PitchMethod::Rmvpe => Ok(Self::Rmvpe(Rmvpe::load(
                &paths.rmvpe.join("rmvpe.pt"),
                device,
                kind,
            )?)),
            PitchMethod::Pm => Ok(Self::Pm),
            PitchMethod::Fcpe => Ok(Self::Fcpe(fcpe::Fcpe::load(
                &paths.root.join(FCPE_CHECKPOINT),
                device,
            )?)),
            PitchMethod::Nmvpe => Ok(Self::Nmvpe(nmvpe::Nmvpe::load(
                &paths.nmvpe(),
                device,
                kind,
            )?)),
        }
    }

    fn extract(&self, audio: &[f32], stream: Option<&cuda_graph::Cache>) -> Result<Vec<f32>> {
        // Experiments: RVC_RMVPE_THRESHOLD overrides the voicing threshold (0.03).
        let threshold = std::env::var("RVC_RMVPE_THRESHOLD")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(0.03);
        match self {
            Self::Rmvpe(rmvpe) => rmvpe.infer(audio, threshold, stream),
            #[cfg(feature = "onnx")]
            Self::RmvpeOnnx(rmvpe) => rmvpe.infer(audio, threshold),
            Self::Pm => Ok(pitch::autocorrelation(audio, stream.is_some())),
            Self::Fcpe(fcpe) => fcpe.infer(audio, 0.006, stream),
            Self::Nmvpe(nmvpe) => nmvpe.infer(audio, stream),
        }
    }

    #[cfg(feature = "cuda-graph")]
    fn clear_graph(&mut self) {
        match self {
            Self::Rmvpe(rmvpe) => rmvpe.graph.get_mut().clear(),
            #[cfg(feature = "onnx")]
            Self::RmvpeOnnx(_) => {}
            Self::Fcpe(fcpe) => fcpe.graph.get_mut().clear(),
            Self::Nmvpe(nmvpe) => nmvpe.network.graph.get_mut().clear(),
            Self::Pm => {}
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize, ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum Precision {
    #[default]
    Auto,
    Fp32,
    Fp16,
}

impl Precision {
    fn kind(self, device: Device) -> Result<Kind> {
        match self {
            Self::Fp32 => Ok(Kind::Float),
            Self::Fp16 => {
                ensure!(
                    matches!(device, Device::Cuda(_)),
                    "FP16 voice conversion requires CUDA"
                );
                Ok(Kind::Half)
            }
            Self::Auto => Ok(match device {
                Device::Cuda(index)
                    if weights::cuda_profile(index).is_none_or(|profile| profile.fp16) =>
                {
                    Kind::Half
                }
                _ => Kind::Float,
            }),
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(default)]
pub struct ConversionSettings {
    pub pitch: f32,
    pub f0_method: PitchMethod,
    pub index_rate: f32,
    pub resample_sr: u32,
    pub rms_mix_rate: f32,
    pub protect: f32,
    pub formant: f32,
    pub precision: Precision,
    /// How far breaths and fricatives follow the source's spectral shape and
    /// loudness (0 = as rendered, 1 = fully matched); see [`aperiodic`].
    pub aperiodic: f32,
    /// Keep the converted voice under the source's level (+2 dB) frame by frame, so
    /// faint parts (pauses, fades, echoes) are not sung out loud.
    pub level_guard: bool,
    /// Attenuation in dB of frames the voice gate reads as not voice (separation residue,
    /// beat bleed, effects) while the input is quiet there; 0 turns the gate off. Needs a
    /// `voice_gate.json` next to the content embedder (see [`gate`]).
    pub voice_gate: f32,
}

impl Default for ConversionSettings {
    fn default() -> Self {
        Self {
            pitch: 0.0,
            f0_method: PitchMethod::Rmvpe,
            index_rate: 0.75,
            resample_sr: 0,
            rms_mix_rate: 1.0,
            protect: 0.33,
            formant: 0.0,
            precision: Precision::Auto,
            aperiodic: 1.0,
            level_guard: true,
            voice_gate: 0.0,
        }
    }
}

impl ConversionSettings {
    pub fn validate(self) -> Result<()> {
        ensure!(
            (0.0..=1.0).contains(&self.index_rate),
            "--index-rate must be between 0 and 1"
        );
        ensure!(
            (0.0..=1.0).contains(&self.rms_mix_rate),
            "--rms-mix-rate must be between 0 and 1"
        );
        ensure!(
            (0.0..=0.5).contains(&self.protect),
            "--protect must be between 0 and 0.5"
        );
        ensure!(
            (0.0..=1.0).contains(&self.aperiodic),
            "--aperiodic must be between 0 and 1"
        );
        ensure!(
            (0.0..=60.0).contains(&self.voice_gate),
            "--voice-gate must be between 0 and 60 dB"
        );
        ensure!(
            self.resample_sr == 0 || (16_000..=192_000).contains(&self.resample_sr),
            "--resample-sr must be 0 or between 16000 and 192000"
        );
        ensure!(
            (-48.0..=48.0).contains(&self.pitch),
            "pitch must be between -48 and 48 semitones"
        );
        ensure!(
            (-24.0..=24.0).contains(&self.formant),
            "formant must be between -24 and 24 semitones"
        );
        Ok(())
    }
}

pub struct Audio {
    pub samples: Vec<f32>,
    pub rate: u32,
}

pub struct Rvc {
    synth: Synth,
    hubert: Hubert,
    f0: Option<Pitch>,
    pitch_method: PitchMethod,
    index: Option<Index>,
    device: Device,
    kind: Kind,
    speaker: i64,
    pitch_history: Vec<f32>,
    pitch_input: Option<(usize, usize)>,
    realtime_formant: f32,
    cached_pitch: Vec<(PitchMethod, Pitch)>,
    loading_pitch: Option<(PitchMethod, JoinHandle<Result<Pitch>>)>,
    /// Spectrogram for source band features, built on first use.
    band_spectrogram: std::cell::OnceCell<train_losses::MelSpectrogram>,
    /// Voice / non-voice classifier on the content features, when one ships with the embedder.
    voice_gate: Option<gate::VoiceGate>,
    #[cfg(feature = "cuda-graph")]
    stream_layout: Option<StreamLayout>,
}

impl Drop for Rvc {
    fn drop(&mut self) {
        if let Some((_, loader)) = self.loading_pitch.take() {
            let _ = loader.join();
        }
    }
}

#[cfg(feature = "cuda-graph")]
struct StreamLayout {
    context: [usize; 4],
    stretched_frames: usize,
    pitch: PitchMethod,
}

impl Rvc {
    pub fn load(
        paths: &Paths,
        model: &Path,
        speaker: Option<i64>,
        index: Option<&Path>,
        settings: ConversionSettings,
        device: Device,
    ) -> Result<Self> {
        settings.validate()?;
        no_grad(|| {
            let mut measure = phase_timer(device);
            let kind = settings.precision.kind(device)?;
            let synth = Synth::load(model, device, kind)?;
            let width = if synth.info.version == "v2" { 768 } else { 256 };
            let speaker = synth.info.speaker(speaker)?;
            let index = index.map(|path| Index::load(path, width)).transpose()?;
            measure("load-generator-and-index");
            let embedder = synth.info.format.embedder;
            let directory = embedder.install(paths, |_| std::ops::ControlFlow::Continue(()))?;
            let hubert = Hubert::load(&directory, device, kind, synth.info.format.normalize_input)?;
            let voice_gate = gate::VoiceGate::load(&directory.join("voice_gate.json"), width, device)?;
            measure("load-hubert");
            let f0 = if synth.info.uses_f0 {
                Some(Pitch::load(settings.f0_method, paths, device, kind)?)
            } else {
                None
            };
            measure("load-f0");
            Ok(Self {
                synth,
                hubert,
                f0,
                pitch_method: settings.f0_method,
                index,
                device,
                kind,
                speaker,
                pitch_history: vec![0.0; 1024],
                pitch_input: None,
                realtime_formant: settings.formant,
                cached_pitch: Vec::new(),
                loading_pitch: None,
                band_spectrogram: std::cell::OnceCell::new(),
                voice_gate,
                #[cfg(feature = "cuda-graph")]
                stream_layout: None,
            })
        })
    }

    pub fn rate(&self) -> u32 {
        self.synth.rate
    }

    pub fn convert(&self, samples: &[f32], settings: ConversionSettings) -> Result<Audio> {
        self.convert_hires(samples, None, settings)
    }

    /// Whether the generator reads the source above 8 kHz: pass the recording at
    /// [`Self::rate`] to [`Self::convert_hires`] so sibilants keep their color.
    pub fn uses_bands(&self) -> bool {
        self.synth.uses_bands()
    }

    /// Convert 16 kHz `samples`, with the same recording at the model rate in `hires`.
    pub fn convert_hires(
        &self,
        samples: &[f32],
        hires: Option<&[f32]>,
        settings: ConversionSettings,
    ) -> Result<Audio> {
        settings.validate()?;
        ensure!(samples.len() > FRAME, "input audio is shorter than 10 ms");
        no_grad(|| self.convert_file(samples, hires, settings, None))
    }

    pub(crate) fn convert_registered(
        &self,
        samples: &[f32],
        hires: Option<&[f32]>,
        settings: ConversionSettings,
        profile: Option<&register::Profile>,
    ) -> Result<Audio> {
        settings.validate()?;
        ensure!(samples.len() > FRAME, "input audio is shorter than 10 ms");
        no_grad(|| self.convert_file(samples, hires, settings, Some(profile)))
    }

    /// Source band features [1, frames, bands] for the padded 16 kHz analysis audio.
    fn band_features(&self, hires: &[f32], padding: usize, frames: usize) -> Result<Tensor> {
        let rate = self.rate();
        let pad = padding * rate as usize;
        let padded = audio::reflect(hires, pad, pad);
        let mel = match self.band_spectrogram.get() {
            Some(mel) => mel,
            None => {
                let config = train_losses::MelConfig {
                    fft: 2048,
                    hop: i64::from(rate / 100),
                    window: 2048,
                    sample_rate: rate,
                    mel_bins: 80,
                    minimum: 0.0,
                    maximum: None,
                };
                let mel = train_losses::MelSpectrogram::new(config, self.device)?;
                self.band_spectrogram.get_or_init(|| mel)
            }
        };
        let wave = Tensor::f_from_slice(&padded)?
            .to_device(self.device)
            .unsqueeze(0);
        let features = synth::bands::from_spectrogram(&mel.spectrogram(&wave)?, rate)?;
        let available = features.size()[1];
        let frames = frames as i64;
        let features = if available >= frames {
            features.narrow(1, 0, frames)
        } else {
            let last = features.narrow(1, available - 1, 1);
            Tensor::cat(&[features, last.repeat([1, frames - available, 1])], 1)
        };
        Ok(features.to_kind(self.kind))
    }

    fn convert_file(
        &self,
        samples: &[f32],
        hires: Option<&[f32]>,
        settings: ConversionSettings,
        registration: Option<Option<&register::Profile>>,
    ) -> Result<Audio> {
        let mut measure = phase_timer(self.device);
        let rate = self.rate();
        let low_memory = match self.device {
            Device::Cuda(index) => {
                weights::cuda_profile(index).is_some_and(|profile| profile.memory_gib < 5.0)
            }
            _ => false,
        };
        let (padding, query, center, maximum) = if low_memory {
            (1, 5, 30, 32)
        } else if self.kind == Kind::Half {
            (3, 10, 60, 65)
        } else {
            (1, 6, 38, 41)
        };
        let pad = padding * ANALYSIS_RATE as usize;
        let scaled = limit_peak(samples, 0.95);
        let audio = highpass(&scaled);
        let cuts = cuts(&audio, query, center, maximum);
        let padded = audio::reflect(&audio, pad, pad);
        measure("prepare-wave");
        let source_frames = padded.len() / FRAME;
        let mut pitch = if let (Some(profile), Some(estimator)) = (registration, self.f0.as_ref()) {
            // Decide the register from the actual recording, without reflected context.
            let offset = pad / FRAME;
            let end = offset + audio.len() / FRAME;
            let mut raw = estimator.extract(&padded, None)?;
            raw.resize(source_frames, 0.0);
            register::match_pitch(&mut raw[offset..end], settings.pitch, profile)?;
            let center = &raw[offset..end];
            Some(audio::reflect(center, offset, source_frames - end))
        } else {
            self.f0(&padded, settings.pitch, false)?
        };
        if let Some(pitch) = &mut pitch {
            pitch.resize(source_frames, 0.0);
            if settings.aperiodic > 0.0 {
                aperiodic::unvoice_fricatives(&padded, pitch);
            }
            if let Some(path) = std::env::var_os("RVC_F0_OVERRIDE") {
                // Debugging aid: replace the F0 track (10 ms frames, without the padding).
                let offset = (pad / FRAME).min(pitch.len());
                let text = std::fs::read_to_string(path)?;
                for (hz, line) in pitch[offset..].iter_mut().zip(text.lines()) {
                    *hz = line.trim().parse().unwrap_or(0.0);
                }
            }
            if let Some(path) = std::env::var_os("RVC_DUMP_F0") {
                // Debugging aid: the F0 track (10 ms frames, without the reflected padding).
                let offset = (pad / FRAME).min(pitch.len());
                let text: String = pitch[offset..].iter().map(|hz| format!("{hz:.1}\n")).collect();
                std::fs::write(path, text)?;
            }
        }
        measure("f0");
        let bands = if self.synth.uses_bands() {
            let resampled;
            let hires = hires.filter(|_| std::env::var_os("RVC_BANDS_NARROW").is_none());
            let source = match hires {
                Some(hires) => hires,
                None => {
                    // Without the original the bands above 8 kHz read as silence.
                    resampled = audio::resample(&audio, ANALYSIS_RATE, rate);
                    &resampled
                }
            };
            Some(self.band_features(source, padding, source_frames)?)
        } else {
            None
        };
        measure("bands");
        let mut result = Vec::new();
        let gate = self.voice_gate.as_ref().filter(|_| settings.voice_gate > 0.0);
        let mut voice_probability = vec![1.0_f32; if gate.is_some() { source_frames } else { 0 }];
        let mut start = 0;
        for cut in cuts.into_iter().chain(std::iter::once(audio.len())) {
            let last = cut == audio.len();
            let end = if last {
                padded.len()
            } else {
                (cut + 2 * pad + FRAME).min(padded.len())
            };
            let samples = &padded[start..end];
            let input_frames = if last {
                source_frames - start / FRAME
            } else {
                (cut + 2 * pad - start) / FRAME
            };
            let features = self.features(samples, settings.index_rate, 0, false)?;
            measure("hubert-and-retrieval");
            let actual_frames = (features.0.size()[1] as usize).min(input_frames);
            if let Some(gate) = gate {
                // The gate reads the source's own features, before any index retrieval.
                let raw = features.1.as_ref().unwrap_or(&features.0);
                let probability = gate.probability(raw)?;
                let first = if start == 0 { 0 } else { pad / FRAME };
                let last = actual_frames.min(probability.len());
                for (frame, &p) in probability.iter().enumerate().take(last).skip(first) {
                    if let Some(slot) = voice_probability.get_mut(start / FRAME + frame) {
                        *slot = p;
                    }
                }
                measure("voice-gate");
            }
            let pitch = pitch
                .as_ref()
                .map(|pitch| &pitch[start / FRAME..start / FRAME + actual_frames]);
            let settings = ConversionSettings {
                formant: 0.0,
                ..settings
            };
            let window = Window {
                frames: actual_frames as i64,
                trim_frames: (pad / FRAME) as i64,
                ..Default::default()
            };
            let bands = bands
                .as_ref()
                .map(|bands| bands.narrow(1, (start / FRAME) as i64, actual_frames as i64));
            let output = self.generate(features, pitch, bands.as_ref(), settings, window, false)?;
            measure("generator");
            result.extend(output);
            start = cut;
        }
        if settings.aperiodic > 0.0
            && let Some(pitch) = &pitch
        {
            let resampled;
            let source = match hires {
                Some(hires) => hires,
                None => {
                    resampled = audio::resample(&audio, ANALYSIS_RATE, rate);
                    &resampled
                }
            };
            let offset = (pad / FRAME).min(pitch.len());
            let source = aperiodic::Source {
                audio: source,
                pitch: &pitch[offset..],
                bandwidth: aperiodic::bandwidth(source, rate),
            };
            aperiodic::transfer(&mut result, &source, rate, settings.aperiodic);
            measure("aperiodic");
        }
        if settings.rms_mix_rate != 1.0 {
            mix_rms(
                &audio,
                ANALYSIS_RATE,
                &mut result,
                rate,
                settings.rms_mix_rate,
            );
        }
        if gate.is_some() {
            let offset = (pad / FRAME).min(voice_probability.len());
            let gains = gate::gains(&voice_probability[offset..], &audio, settings.voice_gate);
            gate::apply(&mut result, &gains, rate);
            measure("voice-gate-apply");
        }
        if settings.level_guard {
            // The 16 kHz analysis audio lacks the top octave: 2 dB of room covers it.
            let reference = audio::resample(&audio, ANALYSIS_RATE, rate);
            audio::hold_under(&mut result, &reference, rate, 2.0);
            measure("level-guard");
        }
        let rate_out = if settings.resample_sr == 0 {
            rate
        } else {
            settings.resample_sr
        };
        let mut samples = limit_peak(&audio::resample(&result, rate, rate_out), 0.99);
        for sample in &mut samples {
            *sample = (*sample * 32768.0).trunc() / 32768.0;
        }
        Ok(Audio {
            samples,
            rate: rate_out,
        })
    }

    pub fn reset_stream(&mut self) {
        self.pitch_history.fill(0.0);
    }

    /// Apply an already loaded estimator at a block boundary. If it has to read
    /// weights first, keep using the current estimator until it is ready.
    pub fn update_pitch_method(&mut self, paths: &Paths, method: PitchMethod) -> Result<()> {
        if !self.synth.info.uses_f0 {
            return Ok(());
        }
        if self
            .loading_pitch
            .as_ref()
            .is_some_and(|(_, thread)| thread.is_finished())
            && let Some((mode, thread)) = self.loading_pitch.take()
        {
            let loaded = thread
                .join()
                .map_err(|_| anyhow::anyhow!("pitch loader stopped"))
                .and_then(|result| result);
            match loaded {
                Ok(model) => self.cached_pitch.push((mode, model)),
                Err(error) if mode == method => return Err(error),
                Err(_) => {}
            }
        }
        if method == self.pitch_method {
            return Ok(());
        }
        let next = if method == PitchMethod::Pm {
            Pitch::Pm
        } else if let Some(index) = self
            .cached_pitch
            .iter()
            .position(|(mode, _)| *mode == method)
        {
            self.cached_pitch.swap_remove(index).1
        } else {
            if self.loading_pitch.is_none() {
                let (paths, device, kind) = (paths.clone(), self.device, self.kind);
                let warmup = self
                    .pitch_input
                    .filter(|_| matches!(device, Device::Cuda(_)))
                    .map(|(samples, block)| pitch_input_len(method, samples, block));
                let loader = thread::Builder::new()
                    .name("rvc-pitch-loader".into())
                    .spawn(move || {
                        no_grad(|| {
                            cuda_graph::prepare(device, || {
                                let pitch = Pitch::load(method, &paths, device, kind)?;
                                if let Some(len) = warmup {
                                    pitch
                                        .extract(&vec![0.0; len], None)
                                        .context("warming realtime pitch estimator")?;
                                }
                                Ok(pitch)
                            })
                        })
                    })?;
                self.loading_pitch = Some((method, loader));
            }
            return Ok(());
        };
        if let Some(previous) = self.f0.replace(next)
            && !matches!(previous, Pitch::Pm)
        {
            #[cfg(feature = "cuda-graph")]
            let previous = {
                let mut previous = previous;
                previous.clear_graph();
                previous
            };
            self.cached_pitch.push((self.pitch_method, previous));
        }
        self.pitch_method = method;
        self.reset_stream();
        Ok(())
    }

    pub fn pitch_method(&self) -> Option<PitchMethod> {
        self.f0.as_ref().map(|_| self.pitch_method)
    }

    #[cfg(feature = "realtime")]
    pub(crate) fn set_index(&mut self, path: &Path) -> Result<()> {
        if self.index.is_none() {
            let width = if self.synth.info.version == "v2" {
                768
            } else {
                256
            };
            self.index = Some(Index::load(path, width)?);
        }
        Ok(())
    }

    pub fn convert_window(
        &mut self,
        samples: &[f32],
        mut settings: ConversionSettings,
        skip: usize,
        frames: usize,
        block: usize,
    ) -> Result<Vec<f32>> {
        no_grad(|| {
            let mut measure = phase_timer(self.device);
            let frame_count = samples.len() / FRAME;
            ensure!(
                frame_count <= self.pitch_history.len() && block > 0 && block <= frame_count,
                "invalid streaming context"
            );
            self.pitch_input = Some((samples.len(), block));
            if self.loading_pitch.is_some() && self.pitch_method != settings.f0_method {
                settings.formant = self.realtime_formant;
            } else {
                self.realtime_formant = settings.formant;
            }
            #[cfg(feature = "cuda-graph")]
            let context_changed = if matches!(self.device, Device::Cuda(_)) {
                let context = [samples.len(), skip, frames, block];
                let changed = self
                    .stream_layout
                    .as_ref()
                    .is_none_or(|previous| previous.context != context);
                if changed {
                    self.synth.graph.get_mut().clear();
                    if let Some(pitch) = &mut self.f0 {
                        pitch.clear_graph();
                    }
                    self.hubert.graph.get_mut().clear();
                    self.stream_layout = Some(StreamLayout {
                        context,
                        stretched_frames: (frames as f32 * (settings.formant / 12.0).exp2()).ceil()
                            as usize,
                        pitch: self.pitch_method,
                    });
                } else if let Some(layout) = &mut self.stream_layout
                    && layout.pitch != self.pitch_method
                {
                    // The pitch and generator graphs share memory. Rebuild the
                    // generator after any newly captured pitch graph.
                    self.synth.graph.get_mut().clear();
                    layout.pitch = self.pitch_method;
                }
                changed
            } else {
                false
            };
            let features = self.features(samples, settings.index_rate, skip, true)?;
            measure("realtime-hubert-and-retrieval");
            let length = (features.0.size()[1] as usize).min(frame_count);
            let pitch_len = pitch_input_len(self.pitch_method, samples.len(), block);
            let audio = &samples[samples.len() - pitch_len..];
            let pitch = self.f0(audio, settings.pitch - settings.formant, true)?;
            measure("realtime-f0");
            let pitch = if let Some(pitch) = pitch {
                let pitch = pitch
                    .get(3..pitch.len().saturating_sub(1))
                    .context("F0 window is too short")?;
                let history = self.pitch_history.len();
                let old_range = history - pitch.len();
                self.pitch_history.copy_within(block.., 0);
                self.pitch_history[old_range..].copy_from_slice(pitch);
                Some(&self.pitch_history[history - length..])
            } else {
                None
            };
            let window = Window {
                skip: skip as i64,
                frames: frames as i64,
                ..Default::default()
            };
            // The stream arrives at 16 kHz, so its band features stop at 8 kHz.
            // Training sees such band-limited sources too (see synth::bands).
            let bands = if self.synth.uses_bands() {
                let hires = audio::resample(samples, ANALYSIS_RATE, self.rate());
                Some(self.band_features(&hires, 0, features.0.size()[1] as usize)?)
            } else {
                None
            };
            let output = self.generate(features, pitch, bands.as_ref(), settings, window, true)?;
            measure("realtime-generator");
            #[cfg(feature = "cuda-graph")]
            if context_changed
                && std::env::var_os("RVC_CUDA_GRAPH").is_none_or(|value| value != "0")
                && std::env::var_os("RVC_PROFILE").is_none_or(|value| value != "1")
            {
                cuda_graph::trim();
            }
            Ok(output)
        })
    }

    fn features(
        &self,
        samples: &[f32],
        ratio: f32,
        skip: usize,
        append: bool,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let v2 = self.synth.info.version == "v2";
        let mut phone = if append {
            self.hubert.extract_realtime(samples, v2)?
        } else {
            self.hubert.extract(samples, v2)?
        };
        if append {
            let last = phone.narrow(1, phone.size()[1] - 1, 1);
            phone = Tensor::cat(&[phone, last], 1);
        }
        let shape = phone.size();
        let Some(index) = self.index.as_ref().filter(|_| ratio > 0.0) else {
            return Ok((phone.repeat_interleave_self_int(2, 1, None), None));
        };
        let flattened = phone.to_device(Device::Cpu).contiguous().view([-1]);
        let mut features = Vec::<f32>::try_from(flattened)?;
        let begin = (skip / 2 * shape[2] as usize).min(features.len());
        index.mix(&mut features[begin..], ratio);
        let modified = Tensor::f_from_slice(&features)?
            .view(shape.as_slice())
            .to_device(self.device)
            .to_kind(self.kind);
        Ok((
            modified.repeat_interleave_self_int(2, 1, None),
            Some(phone.repeat_interleave_self_int(2, 1, None)),
        ))
    }

    fn f0(&self, audio: &[f32], semitones: f32, realtime: bool) -> Result<Option<Vec<f32>>> {
        let Some(f0) = &self.f0 else {
            return Ok(None);
        };
        let pool = realtime.then(|| self.hubert.graph.borrow());
        let mut pitch = f0.extract(audio, pool.as_deref())?;
        if self.synth.info.format.f0_unvoiced == Unvoiced::Interpolate {
            interpolate_voiced(&mut pitch);
        }
        let ratio = (semitones / 12.0).exp2();
        pitch.iter_mut().for_each(|hz| *hz *= ratio);
        Ok(Some(pitch))
    }

    fn generate(
        &self,
        features: (Tensor, Option<Tensor>),
        f0: Option<&[f32]>,
        bands: Option<&Tensor>,
        settings: ConversionSettings,
        mut window: Window,
        stream: bool,
    ) -> Result<Vec<f32>> {
        let (mut phone, original) = features;
        let available = f0.map_or(phone.size()[1], |hz| phone.size()[1].min(hz.len() as i64));
        let available = bands.map_or(available, |bands| available.min(bands.size()[1]));
        phone = phone.narrow(1, 0, available);
        let bands = bands.map(|bands| bands.narrow(1, 0, available));
        let stretch = (settings.formant / 12.0).exp2();
        let frames = window.frames as usize;
        let stretched = (frames as f32 * stretch).ceil() as usize;
        window.stretched_frames = stretched as i64;
        let pitch = f0
            .map(|hz| -> Result<(Tensor, Tensor)> {
                let hz = &hz[..available as usize];
                let bands: Vec<_> = hz.iter().map(|&hz| coarse_pitch(hz)).collect();
                let coarse = Tensor::f_from_slice(&bands)?
                    .to_device(self.device)
                    .unsqueeze(0);
                let hz = Tensor::f_from_slice(hz)?
                    .to_device(self.device)
                    .unsqueeze(0);
                if let Some(old) = original.filter(|_| !stream && settings.protect < 0.5) {
                    let mask = hz.ge(1.).to_kind(self.kind).unsqueeze(-1);
                    let factor =
                        mask * (1.0 - f64::from(settings.protect)) + f64::from(settings.protect);
                    phone = &phone * &factor + old.narrow(1, 0, available) * (1.0 - &factor);
                }
                let continuous = hz * (stretched as f64 / frames.max(1) as f64);
                Ok((coarse, continuous))
            })
            .transpose()?;
        let pitch = pitch.as_ref().map(|(a, b)| (a, b));
        let capture = stream;
        // Use the initial generator layout again if a user returns to it.
        // A live formant change can run immediately without capture warmup.
        #[cfg(feature = "cuda-graph")]
        let capture = capture
            && self
                .stream_layout
                .as_ref()
                .is_none_or(|layout| layout.stretched_frames == stretched);
        let samples = if capture {
            self.synth.infer_realtime(
                &phone,
                pitch,
                bands.as_ref(),
                self.speaker,
                window,
                &self.hubert.graph.borrow(),
            )?
        } else {
            self.synth.infer(&phone, pitch, bands.as_ref(), self.speaker, window)?
        };
        let samples = samples.view([-1]).to_device(Device::Cpu);
        let mut samples = Vec::<f32>::try_from(samples)?;
        let hop = self.rate() / 100;
        if settings.formant != 0.0 {
            let shifted_hop = (hop as f32 * stretch).floor() as u32;
            samples.truncate(frames * shifted_hop as usize);
            samples = audio::resample(&samples, shifted_hop, hop);
        }
        if !samples.iter().all(|sample| sample.is_finite()) {
            bail!("generator produced a non-finite sample");
        }
        Ok(samples)
    }
}

fn pitch_input_len(method: PitchMethod, samples: usize, block: usize) -> usize {
    let tail = block * FRAME + 800;
    let tail = match method {
        PitchMethod::Rmvpe | PitchMethod::Nmvpe => tail.div_ceil(5_120) * 5_120 - FRAME,
        PitchMethod::Pm | PitchMethod::Fcpe => tail,
    };
    tail.min(samples)
}

fn interpolate_voiced(pitch: &mut [f32]) {
    let Some(first) = pitch.iter().position(|&hz| hz > 0.0) else {
        return;
    };
    let initial = pitch[first];
    pitch[..first].fill(initial);
    let mut previous = first;
    for i in first + 1..pitch.len() {
        if pitch[i] > 0.0 {
            let (left, right) = (pitch[previous], pitch[i]);
            let count = (i - previous) as f32;
            for (j, value) in pitch[previous + 1..i].iter_mut().enumerate() {
                *value = left + (right - left) * ((j + 1) as f32 / count);
            }
            previous = i;
        }
    }
    let end = pitch[previous];
    pitch[previous + 1..].fill(end);
}

fn coarse_pitch(hz: f32) -> i64 {
    let mel = |f: f64| (1.0 + f / 700.0).ln();
    ((mel(f64::from(hz)) - mel(50.0)) * 254.0 / (mel(1100.0) - mel(50.0)) + 1.0)
        .clamp(1.0, 255.0)
        .round_ties_even() as i64
}

fn phase_timer(device: Device) -> impl FnMut(&str) {
    let mut start = std::env::var_os("RVC_PROFILE")
        .filter(|value| value == "1")
        .map(|_| std::time::Instant::now());
    move |label| {
        if let Some(previous) = &mut start {
            if let Device::Cuda(index) = device {
                tch::Cuda::synchronize(index as i64);
            }
            let now = std::time::Instant::now();
            eprintln!(
                "[rvc timing] {label} {:.3} ms",
                (now - *previous).as_secs_f64() * 1000.0
            );
            *previous = now;
        }
    }
}

fn limit_peak(samples: &[f32], limit: f32) -> Vec<f32> {
    let peak = samples.iter().fold(0.0_f32, |a, s| a.max(s.abs()));
    let gain = if peak > limit { limit / peak } else { 1.0 };
    samples.iter().map(|value| value * gain).collect()
}

fn cuts(audio: &[f32], query: usize, center: usize, maximum: usize) -> Vec<usize> {
    const SECOND: usize = ANALYSIS_RATE as usize;
    if audio.len() + FRAME <= maximum * SECOND {
        return Vec::new();
    }
    let padded = audio::reflect(audio, FRAME / 2, FRAME / 2);
    let mut sum = padded[..FRAME].iter().map(|s| s.abs()).sum::<f32>();
    let mut sums = Vec::with_capacity(audio.len());
    for (i, &next) in padded[FRAME..].iter().enumerate() {
        sums.push(sum);
        sum += next.abs() - padded[i].abs();
    }
    (center * SECOND..audio.len())
        .step_by(center * SECOND)
        .map(|middle| {
            let begin = middle - query * SECOND;
            let end = (middle + query * SECOND).min(audio.len());
            let mut best = begin;
            for i in begin + 1..end {
                if sums[i] < sums[best] {
                    best = i;
                }
            }
            best / FRAME * FRAME
        })
        .collect()
}

/// SciPy's fifth-order Butterworth at 48 Hz and 16 kHz, double-precision
/// coefficients and forward/backward odd extension as in scipy.signal.filtfilt.
fn highpass(audio: &[f32]) -> Vec<f32> {
    const B: [f64; 6] = [
        0.969_960_645_183_844_7,
        -4.849_803_225_919_223,
        9.699_606_451_838_447,
        -9.699_606_451_838_447,
        4.849_803_225_919_223,
        -0.969_960_645_183_844_7,
    ];
    const A: [f64; 6] = [
        1.0,
        -4.939_001_819_168_364,
        9.757_863_526_739_543,
        -9.639_544_849_413_458,
        4.761_506_797_356_209,
        -0.940_823_653_205_460_6,
    ];
    const Z: [f64; 5] = [
        -0.969_960_530_493_927_6,
        3.879_842_128_971_587_6,
        -5.819_763_203_738_301,
        3.879_842_142_541_546,
        -0.969_960_537_280_858_1,
    ];
    let filter = |samples: &mut [f64]| {
        let mut delay = Z.map(|value| value * samples[0]);
        for value in samples {
            let input = *value;
            let output = B[0] * input + delay[0];
            for i in 0..4 {
                delay[i] = B[i + 1] * input - A[i + 1] * output + delay[i + 1];
            }
            delay[4] = B[5] * input - A[5] * output;
            *value = output;
        }
    };
    let edge = 18usize.min(audio.len() - 1);
    let first = f64::from(audio[0]);
    let last = f64::from(audio[audio.len() - 1]);
    let mut padded: Vec<_> = audio[1..=edge]
        .iter()
        .rev()
        .map(|s| 2.0 * first - f64::from(*s))
        .chain(audio.iter().map(|&s| f64::from(s)))
        .chain(
            audio[audio.len() - edge - 1..audio.len() - 1]
                .iter()
                .rev()
                .map(|s| 2.0 * last - f64::from(*s)),
        )
        .collect();
    filter(&mut padded);
    padded.reverse();
    filter(&mut padded);
    padded.reverse();
    padded[edge..edge + audio.len()]
        .iter()
        .map(|&v| v as f32)
        .collect()
}

fn mix_rms(input: &[f32], input_rate: u32, output: &mut [f32], output_rate: u32, mix: f32) {
    let rms = |samples: &[f32], rate: u32| -> Vec<f64> {
        let half = rate as usize / 2;
        (0..=samples.len() / half)
            .map(|frame| {
                let center = frame * half;
                let start = center.saturating_sub(half);
                let end = (center + half).min(samples.len());
                (samples[start..end]
                    .iter()
                    .map(|&v| f64::from(v).powi(2))
                    .sum::<f64>()
                    / (half * 2) as f64)
                    .sqrt()
            })
            .collect()
    };
    let input_rms = rms(input, input_rate);
    let output_rms = rms(output, output_rate);
    let length = output.len();
    let interpolate = |points: &[f64], sample: usize| {
        let x = ((sample as f64 + 0.5) * points.len() as f64 / length as f64 - 0.5)
            .clamp(0.0, (points.len() - 1) as f64);
        let index = x.floor() as usize;
        points[index] + (points[(index + 1).min(points.len() - 1)] - points[index]) * x.fract()
    };
    for (i, sample) in output.iter_mut().enumerate() {
        let ratio = interpolate(&input_rms, i) / interpolate(&output_rms, i).max(1e-6);
        *sample *= ratio.powf(1.0 - f64::from(mix)) as f32;
    }
}
