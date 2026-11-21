//! Layer-aware song conversion ("smart cover").
//!
//! Converting a song's whole vocal stem at once hands RVC a stack of voices
//! (lead, doubles, harmonies, ad-libs) wrapped in reverb. One pitch track and
//! one content stream cannot describe that, and the result rings metallic.
//! This pipeline splits the vocal into layers with separation models, fixes
//! the separator's typical mistake, converts each dry layer on its own and
//! rebuilds the mix:
//!
//! 1. song -> vocals + instrumental (BS-RoFormer FT1), unless the input is vocals;
//! 2. vocals -> lead + backing (karaoke BS-RoFormer);
//! 3. stretches the karaoke model put in "backing" while the lead is silent are
//!    solo lines (a featured verse, a call-and-response): they move back to the lead;
//! 4. each layer -> dry + reverb (dereverb BS-RoFormer);
//! 5. the dry lead and dry backing are converted separately, each with its own F0;
//! 6. a fresh reverb at each layer's original wet/dry ratio, the original
//!    vocal-to-instrumental balance, and a peak limiter.

use std::ops::ControlFlow;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use clap::Args;

use crate::audio::{self, Format};
use crate::catalog::{ModelInfo, Paths};
use crate::model::{self, ConversionSettings};
use crate::separation;

const RATE: u32 = 44_100;

#[derive(Args, Debug)]
pub struct Request {
    /// A song, or a vocal stem with `--vocals-only`. Repeat to convert several in one run:
    /// the separation models and the voice load once.
    #[arg(long, required = true)]
    pub input: Vec<PathBuf>,
    /// Output directory for the mix and the converted stems.
    #[arg(long)]
    pub output: PathBuf,
    #[arg(long)]
    pub model: PathBuf,
    #[arg(long)]
    pub index: Option<PathBuf>,
    #[arg(long, default_value_t = 0.5)]
    pub index_rate: f32,
    /// Semitones for every layer.
    #[arg(long, default_value_t = 0.0, allow_hyphen_values = true)]
    pub pitch: f32,
    #[arg(long, value_enum, default_value = "rmvpe")]
    pub f0_method: model::PitchMethod,
    #[arg(long)]
    pub speaker_id: Option<i64>,
    /// The inputs are already vocal stems: skip the instrumental split.
    #[arg(long)]
    pub vocals_only: bool,
    /// Instrumentals to mix the converted vocals with, one per `--input`, when the inputs
    /// are vocal stems.
    #[arg(long, requires = "vocals_only")]
    pub instrumental: Vec<PathBuf>,
    /// What to do with backing vocals: convert them, keep the original, or drop them.
    #[arg(long, value_enum, default_value = "convert")]
    pub backing: Backing,
    /// Skip the reverb rebuild and leave the converted vocals dry.
    #[arg(long)]
    pub dry: bool,
    /// Source breath and fricative envelope (0-1). Omit to use the model default.
    #[arg(long)]
    pub aperiodic: Option<f32>,
    /// Lower by this many dB the quiet frames the voice gate reads as not voice
    /// (separation residue, beat bleed); 0 = off.
    #[arg(long, default_value_t = 0.0)]
    pub voice_gate: f32,
    /// Vocal level relative to the original balance, in dB.
    #[arg(long, default_value_t = 0.0, allow_hyphen_values = true)]
    pub vocal_gain: f32,
    #[arg(long, default_value = "auto")]
    pub device: String,
    #[arg(long, value_enum, default_value = "flac")]
    pub format: Format,
}

#[derive(clap::ValueEnum, serde::Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Backing {
    Convert,
    Keep,
    Drop,
}

/// Interleaved stereo at 44.1 kHz.
struct Stereo(Vec<f32>);

impl Stereo {
    fn read(path: &Path) -> Result<Self> {
        Ok(Self(audio::read_interleaved(path, RATE, 2)?))
    }

    fn mono(&self) -> Vec<f32> {
        self.0
            .as_chunks::<2>()
            .0
            .iter()
            .map(|[l, r]| 0.5 * (l + r))
            .collect()
    }

    fn frames(&self) -> usize {
        self.0.len() / 2
    }
}

/// The files written for one input, each with its kind: `lead`, `backing`, `vocals`
/// (the rebuilt vocal bus) and `cover` (the mix, when there is an instrumental).
pub struct Rendered {
    pub input: PathBuf,
    pub files: Vec<(&'static str, PathBuf)>,
}

pub fn run(paths: &Paths, request: &Request) -> Result<Vec<PathBuf>> {
    let rendered = render(paths, request, |line| {
        eprintln!("{line}");
        ControlFlow::Continue(())
    })?;
    Ok(rendered
        .into_iter()
        .flat_map(|item| item.files.into_iter().map(|(_, path)| path))
        .collect())
}

/// [`run`] reporting each stage to `progress`, which can stop the run between stages.
pub fn render(
    paths: &Paths,
    request: &Request,
    mut progress: impl FnMut(&str) -> ControlFlow<()>,
) -> Result<Vec<Rendered>> {
    ensure!(
        request.instrumental.is_empty() || request.instrumental.len() == request.input.len(),
        "give one --instrumental per --input"
    );
    for preset in [
        "vocals-bs-roformer-ft1-anvuew",
        "karaoke-bs-roformer-anvuew",
        "dereverb-bs-roformer-anvuew-22.5050",
    ] {
        if preset.starts_with("vocals") && request.vocals_only {
            continue;
        }
        crate::assets::separation(paths, preset, &mut progress)?;
    }
    crate::assets::voice(paths, &request.model, request.f0_method, &mut progress)?;
    let mut step = |message: &str| -> Result<()> {
        ensure!(progress(message).is_continue(), "cover cancelled");
        Ok(())
    };
    let device = separation::device(&request.device)?;
    let mut load = |preset: &str| -> Result<separation::Separator> {
        step(&format!("Loading {preset}"))?;
        separation::Separator::load(&paths.root, preset, device)
    };
    let vocal_split = (!request.vocals_only)
        .then(|| load("vocals-bs-roformer-ft1-anvuew"))
        .transpose()?;
    let karaoke = load("karaoke-bs-roformer-anvuew")?;
    let dereverb = load("dereverb-bs-roformer-anvuew-22.5050")?;

    let info = ModelInfo::read(&request.model)?;
    let speaker = info.speaker(request.speaker_id)?;
    let index = paths.require_index(
        &request.model,
        speaker,
        request.index.as_deref(),
        request.index_rate,
    )?;
    let settings = ConversionSettings {
        pitch: request.pitch,
        f0_method: request.f0_method,
        index_rate: request.index_rate,
        aperiodic: request.aperiodic,
        voice_gate: request.voice_gate,
        ..Default::default()
    };
    settings.validate()?;
    let engine = model::Rvc::load(
        paths,
        &request.model,
        Some(speaker),
        index.as_deref(),
        settings,
        model::weights::device(&request.device)?,
    )?;
    let convert = |dry: &[f32], settings: ConversionSettings| -> Result<Vec<f32>> {
        let analysis = audio::resample(dry, RATE, audio::ANALYSIS_RATE);
        let hires = engine
            .needs_hires(settings)
            .then(|| audio::resample(dry, RATE, engine.rate()));
        let converted = engine.convert_hires(&analysis, hires.as_deref(), settings)?;
        let mut out = audio::resample(&converted.samples, converted.rate, RATE);
        out.resize(dry.len(), 0.0);
        Ok(out)
    };
    let mono = |stereo: &[f32]| Stereo(stereo.to_vec()).mono();

    let mut outputs = Vec::new();
    let total = request.input.len();
    for (number, input) in request.input.iter().enumerate() {
        let stem = input
            .file_stem()
            .context("input has no file name")?
            .to_string_lossy()
            .into_owned();
        let mut stage = |name: &str| step(&format!("{} / {total} · {stem} · {name}", number + 1));
        let mut files = Vec::new();
        stage("Separating vocals")?;
        let started = std::time::Instant::now();
        let mut timing = Vec::new();
        let mut lap = {
            let mut last = std::time::Instant::now();
            move |name: &'static str, timing: &mut Vec<String>| {
                timing.push(format!("{name} {:.1}s", last.elapsed().as_secs_f64()));
                last = std::time::Instant::now();
            }
        };
        let source = Stereo::read(input)?;
        let (vocals, instrumental) = match &vocal_split {
            Some(model) => {
                let (vocals, instrumental) = model.split(&source.0)?;
                (Stereo(vocals), Some(Stereo(instrumental)))
            }
            None => (
                source,
                request
                    .instrumental
                    .get(number)
                    .map(|path| Stereo::read(path))
                    .transpose()?,
            ),
        };
        lap("read+split", &mut timing);
        stage("Splitting lead and backing")?;
        let (lead, backing) = karaoke.split(&vocals.0)?;
        lap("karaoke", &mut timing);
        let (mut lead, mut backing) = (Stereo(lead), Stereo(backing));
        let plan = plan_layers(&mut lead, &mut backing);
        stage("Removing reverb")?;
        let (lead_dry, lead_wet) = dereverb.split(&lead.0)?;
        let (mut lead_dry, mut lead_wet) = (mono(&lead_dry), mono(&lead_wet));
        let restored = restore_voice(&lead.mono(), &mut lead_dry, &mut lead_wet);
        if std::env::var_os("RVC_COVER_DEBUG").is_some() {
            for (name, samples) in [
                ("dbg_lead_in", lead.mono()),
                ("dbg_backing_in", backing.mono()),
                ("dbg_lead_dry", lead_dry.clone()),
                ("dbg_lead_wet", lead_wet.clone()),
            ] {
                let path = request.output.join(format!("{stem}_{name}.wav"));
                audio::write_audio(&path, &samples, RATE, Format::Wav, true)?;
            }
        }
        lap("dereverb", &mut timing);
        stage("Converting lead")?;
        let mut lead_converted = convert(&lead_dry, settings)?;
        // The model may lift faint parts (a fade, a tail) to a singing level: keep under the input.
        follow_level(&mut lead_converted, &lead_dry);
        lap("convert", &mut timing);
        let mut bus = layer(&lead_converted, &lead_dry, &lead_wet, request.dry);
        let mut stems = vec![("lead", lead_converted)];
        let mut hiss_removed = 0.0_f32;

        if !plan.single && backing.frames() > 0 && active(&backing.mono()) {
            match request.backing {
                Backing::Convert => {
                    stage("Converting backing")?;
                    let (dry, wet) = dereverb.split(&backing.0)?;
                    let (mut dry, mut wet) = (mono(&dry), mono(&wet));
                    restore_voice(&backing.mono(), &mut dry, &mut wet);
                    // Backing parts may hold several voices: lean on the source consonants.
                    let mut converted = convert(
                        &dry,
                        ConversionSettings {
                            protect: 0.45,
                            index_rate: settings.index_rate * 0.6,
                            ..settings
                        },
                    )?;
                    hiss_removed = cap_hiss(&mut converted, &dry);
                    follow_level(&mut converted, &dry);
                    add(&mut bus, &layer(&converted, &dry, &wet, request.dry));
                    stems.push(("backing", converted));
                    lap("backing", &mut timing);
                }
                Backing::Keep => add(&mut bus, &backing.mono()),
                Backing::Drop => {}
            }
        }

        // Keep the original vocal loudness (BS.1770, gated) plus the requested offset.
        // The bus goes to both channels, so as stereo it reads 3 dB above its mono loudness.
        let gain = 10_f64.powf((loudness(&vocals.0, 2) - loudness(&bus, 1) - 3.01) / 20.0) as f32
            * 10_f32.powf(request.vocal_gain / 20.0);
        bus.iter_mut().for_each(|v| *v *= gain);
        stage("Mixing")?;
        let extension = request.format.extension();
        for (name, samples) in &stems {
            let path = request.output.join(format!("{stem}_{name}.{extension}"));
            audio::write_audio(&path, samples, RATE, request.format, true)?;
            files.push((*name, path));
        }
        let vocal_path = request.output.join(format!("{stem}_vocals.{extension}"));
        audio::write_audio(&vocal_path, &limit(&bus, 1), RATE, request.format, true)?;
        files.push(("vocals", vocal_path));
        if let Some(instrumental) = instrumental {
            let frames = instrumental.frames().min(bus.len());
            let mut mix = instrumental.0[..frames * 2].to_vec();
            for ([l, r], v) in mix.as_chunks_mut::<2>().0.iter_mut().zip(&bus) {
                *l += v;
                *r += v;
            }
            let mix_path = request.output.join(format!("{stem}_cover.{extension}"));
            audio::write_audio_channels(
                &mix_path,
                &limit(&mix, 2),
                RATE,
                2,
                request.format,
                false,
                true,
            )?;
            files.push(("cover", mix_path));
        }
        lap("mix+write", &mut timing);
        let summary = format!(
            "{stem}: {}, {:.1} s of solo lines in the lead, {:.1} s of voice kept from the dereverb's wet stem, done in {:.1} s ({})",
            if plan.single {
                "one layer".to_owned()
            } else {
                format!(
                    "separate backing on {:.0} % of the vocal ({:.0} % of its invented hiss removed)",
                    plan.parallel * 100.0,
                    hiss_removed * 100.0
                )
            },
            plan.solo as f64 / f64::from(RATE),
            restored as f64 / f64::from(RATE),
            started.elapsed().as_secs_f64(),
            timing.join(", ")
        );
        // The files are written: a stop request from here on is seen at the next input.
        let _ = step(&summary);
        outputs.push(Rendered {
            input: input.clone(),
            files,
        });
    }
    Ok(outputs)
}

const HOP: usize = 1024;

fn frame_db(x: &[f32]) -> Vec<f32> {
    x.chunks(HOP)
        .map(|c| 10.0 * (c.iter().map(|v| v * v).sum::<f32>() / c.len() as f32 + 1e-12).log10())
        .collect()
}

fn active(x: &[f32]) -> bool {
    frame_db(x).iter().any(|&db| db > -50.0)
}

fn active_rms(x: &[f32]) -> f32 {
    let db = frame_db(x);
    let mut sorted = db.clone();
    sorted.sort_by(f32::total_cmp);
    let peak = sorted.get(sorted.len() * 95 / 100).copied().unwrap_or(-120.0);
    let (sum, count) = db
        .iter()
        .filter(|&&d| d > peak - 30.0)
        .fold((0.0_f64, 0usize), |(s, n), &d| (s + 10_f64.powf(f64::from(d) / 10.0), n + 1));
    ((sum / count.max(1) as f64).sqrt()) as f32
}

/// How the karaoke split was turned into layers.
struct Plan {
    /// Samples of backing-only lines (a featured verse) moved to the lead.
    solo: usize,
    /// Share of voiced frames that keep a separate backing layer.
    parallel: f32,
    /// Too little real backing: the whole vocal is converted as one layer.
    single: bool,
}

/// Decide, frame by frame, what stays a separate backing layer.
///
/// Only backing that sounds alongside the lead at a comparable level (-10 to
/// +9 dB) is a real layer: a harmony, a double, an ad-lib. Backing alone over a
/// silent lead is a solo line and joins the lead. Backing far below the lead is
/// the separator's residue of the lead itself, which converted on its own
/// turns into garbage, so it returns to the lead too. When real layers are rare
/// (an already isolated lead, a tag split in halves) everything is one layer.
fn plan_layers(lead: &mut Stereo, backing: &mut Stereo) -> Plan {
    let backing_mono = backing.mono();
    let (l, b) = (frame_db(&lead.mono()), frame_db(&backing_mono));
    let n = l.len().min(b.len());
    // The karaoke model splits a scream: its tone stays in the lead and its rasp (breath
    // noise, distortion) lands in "backing". Converted on its own, that noise has no pitch
    // for the voice to follow and comes out as loud hiss. A harmony, a double or an
    // ad-lib is tonal; noise-like backing belongs to the lead.
    let flat = flatness(&backing_mono);
    let noisy: Vec<bool> = (0..n).map(|i| flat.get(i).copied().unwrap_or(0.0) > NOISY_FLATNESS).collect();
    let mut sorted: Vec<f32> = (0..n).map(|i| l[i].max(b[i])).collect();
    sorted.sort_by(f32::total_cmp);
    let peak = sorted.get(n * 95 / 100).copied().unwrap_or(-120.0);
    let floor = peak - 35.0;
    let solo: Vec<bool> = (0..n).map(|i| b[i] > floor && b[i] > l[i] + 9.0).collect();
    let parallel: Vec<bool> = (0..n)
        .map(|i| b[i] > floor && l[i] > floor && b[i] <= l[i] + 9.0 && b[i] > l[i] - 10.0 && !noisy[i])
        .collect();
    if std::env::var_os("RVC_COVER_DEBUG").is_some() {
        let mut flats: Vec<f32> = (0..n)
            .filter(|&i| b[i] > floor && l[i] > floor && b[i] > l[i] - 10.0)
            .map(|i| flat.get(i).copied().unwrap_or(0.0))
            .collect();
        flats.sort_by(f32::total_cmp);
        let q = |p: usize| flats.get(flats.len() * p / 100).copied().unwrap_or(0.0);
        eprintln!(
            "backing flatness over {} loud frames: p10 {:.3} p50 {:.3} p90 {:.3}, noisy {}",
            flats.len(),
            q(10),
            q(50),
            q(90),
            (0..n).filter(|&i| noisy[i] && b[i] > floor).count()
        );
    }
    // Majority over ~0.5 s so single syllables do not flip between layers.
    let half = (RATE as usize / 2 / HOP).max(1);
    let majority = |flags: &[bool]| -> Vec<bool> {
        (0..n)
            .map(|i| {
                let range = i.saturating_sub(half)..(i + half + 1).min(n);
                flags[range.clone()].iter().filter(|&&f| f).count() * 2 > range.len()
            })
            .collect()
    };
    let (solo, keep) = (majority(&solo), majority(&parallel));
    let voiced = (0..n).filter(|&i| l[i] > floor || b[i] > floor).count().max(1);
    let share = keep.iter().filter(|&&k| k).count() as f32 / voiced as f32;
    let single = share < 0.08;
    let to_lead: Vec<f32> = keep
        .iter()
        .map(|&k| if k && !single { 0.0 } else { 1.0 })
        .collect();
    let mask = smooth_mask(&to_lead, lead.frames().min(backing.frames()));
    for (i, &m) in mask.iter().enumerate() {
        for c in 0..2 {
            let v = backing.0[i * 2 + c] * m;
            lead.0[i * 2 + c] += v;
            backing.0[i * 2 + c] -= v;
        }
    }
    Plan {
        solo: solo.iter().filter(|&&s| s).count() * HOP,
        parallel: if single { 0.0 } else { share },
        single,
    }
}

/// Spectral flatness above which a backing frame counts as noise, not a voice. Sung
/// tones sit around 0.01-0.1 between 300 Hz and 8 kHz; white noise is about 0.56.
const NOISY_FLATNESS: f32 = 0.2;

/// Spectral flatness (geometric over arithmetic mean of the power, 300 Hz - 8 kHz)
/// for each `HOP` frame of `x`.
fn flatness(x: &[f32]) -> Vec<f32> {
    const SIZE: usize = 2 * HOP;
    let mut planner = rustfft::FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(SIZE);
    let hz = RATE as f32 / SIZE as f32;
    let (low, high) = ((300.0 / hz) as usize, (8_000.0 / hz) as usize);
    (0..x.len().div_ceil(HOP))
        .map(|t| {
            let start = (t * HOP) as isize - (SIZE / 4) as isize;
            let mut buffer: Vec<rustfft::num_complex::Complex<f32>> = (0..SIZE)
                .map(|i| {
                    let at = start + i as isize;
                    let v = if at >= 0 && (at as usize) < x.len() { x[at as usize] } else { 0.0 };
                    let w = (std::f32::consts::PI * (i as f32 + 0.5) / SIZE as f32).sin().powi(2);
                    rustfft::num_complex::Complex::new(v * w, 0.0)
                })
                .collect();
            fft.process(&mut buffer);
            let power: Vec<f32> = buffer[low..high].iter().map(|v| v.norm_sqr() + 1e-20).collect();
            let mean = power.iter().sum::<f32>() / power.len() as f32;
            let log_mean = power.iter().map(|p| p.ln()).sum::<f32>() / power.len() as f32;
            log_mean.exp() / mean
        })
        .collect()
}

/// How far a converted layer may rise above its own input in a band above 3 kHz,
/// once the overall level change is taken out (power ratio, +6 dB).
const HISS_MARGIN: f32 = 4.0;

/// Keep a converted layer from inventing hiss its input does not have.
///
/// Fed a backing stem (thin, noisy, several voices at once) the voice model can answer
/// with broadband noise far above anything in the input: +15 to +30 dB above 4 kHz on
/// a split scream. Frame by frame, each band above 3 kHz of `converted` is held under
/// the same band of `input` scaled by the level change below 3 kHz, plus 6 dB of room
/// for the new voice's own brightness. Only attenuates; `input` and `converted` must
/// be aligned and at `RATE`. Returns the share of high-band energy removed.
fn cap_hiss(converted: &mut [f32], input: &[f32]) -> f32 {
    use rustfft::num_complex::Complex;
    const SIZE: usize = 2048;
    const STEP: usize = SIZE / 4;
    const EDGES: [f32; 9] = [0.0, 3_000.0, 4_000.0, 5_300.0, 7_000.0, 9_300.0, 12_300.0, 16_300.0, 22_051.0];
    let n = converted.len().min(input.len());
    if n < SIZE {
        return 0.0;
    }
    let mut planner = rustfft::FftPlanner::<f32>::new();
    let (forward, inverse) = (planner.plan_fft_forward(SIZE), planner.plan_fft_inverse(SIZE));
    let window: Vec<f32> = (0..SIZE)
        .map(|i| (std::f32::consts::PI * (i as f32 + 0.5) / SIZE as f32).sin().powi(2))
        .collect();
    let bins = SIZE / 2 + 1;
    let hz = RATE as f32 / SIZE as f32;
    let bands = EDGES.len() - 1;
    let band_of: Vec<usize> = (0..bins)
        .map(|k| EDGES.windows(2).position(|e| (k as f32 * hz) < e[1]).unwrap_or(bands - 1))
        .collect();
    let frames = n.div_ceil(STEP) + 3;
    let spectrum = |x: &[f32], t: usize| -> Vec<Complex<f32>> {
        let start = (t * STEP) as isize - SIZE as isize / 2;
        let mut buffer: Vec<Complex<f32>> = (0..SIZE)
            .map(|i| {
                let at = start + i as isize;
                let v = if at >= 0 && (at as usize) < n { x[at as usize] } else { 0.0 };
                Complex::new(v * window[i], 0.0)
            })
            .collect();
        forward.process(&mut buffer);
        buffer.truncate(bins);
        buffer
    };
    let powers = |s: &[Complex<f32>]| -> Vec<f32> {
        let mut p = vec![0_f32; bands];
        for (k, v) in s.iter().enumerate() {
            p[band_of[k]] += v.norm_sqr();
        }
        p
    };
    let out_spec: Vec<Vec<Complex<f32>>> = (0..frames).map(|t| spectrum(converted, t)).collect();
    let out_pow: Vec<Vec<f32>> = out_spec.iter().map(|s| powers(s)).collect();
    let in_pow: Vec<Vec<f32>> = (0..frames).map(|t| powers(&spectrum(input, t))).collect();
    let centers: Vec<f32> = EDGES.windows(2).map(|e| ((e[0].max(1_500.0)) * e[1]).sqrt()).collect();
    let mut result = vec![0_f32; (frames + 2) * STEP + SIZE];
    let mut norm = vec![0_f32; result.len()];
    let (mut removed, mut high_total) = (0_f64, 0_f64);
    for (t, spec) in out_spec.iter().enumerate() {
        // Three frames (~35 ms) steady the estimate.
        let around = t.saturating_sub(1)..(t + 2).min(frames);
        let sum = |bank: &[Vec<f32>], b: usize| around.clone().map(|u| bank[u][b]).sum::<f32>();
        let level = (sum(&out_pow, 0) / sum(&in_pow, 0).max(1e-12)).clamp(1e-2, 1e2);
        let gains: Vec<f32> = (0..bands)
            .map(|b| {
                if b == 0 {
                    return 1.0;
                }
                let allowed = sum(&in_pow, b) * level * HISS_MARGIN;
                (allowed / sum(&out_pow, b).max(1e-12)).sqrt().clamp(0.03, 1.0)
            })
            .collect();
        let mut s = spec.clone();
        for (k, v) in s.iter_mut().enumerate() {
            let f = (k as f32 * hz).max(1.0);
            let g = if f <= centers[0] {
                1.0
            } else if f >= centers[bands - 1] {
                gains[bands - 1]
            } else {
                let b = centers.iter().position(|&c| c > f).unwrap_or(bands - 1);
                let x = (f.ln() - centers[b - 1].ln()) / (centers[b].ln() - centers[b - 1].ln());
                (gains[b - 1].ln() * (1.0 - x) + gains[b].ln() * x).exp()
            };
            if f >= EDGES[1] {
                high_total += f64::from(v.norm_sqr());
                removed += f64::from(v.norm_sqr() * (1.0 - g * g));
            }
            *v *= g;
        }
        let mut full = vec![Complex::new(0.0, 0.0); SIZE];
        full[..bins].copy_from_slice(&s);
        for k in 1..SIZE - bins + 1 {
            full[SIZE - k] = s[k].conj();
        }
        inverse.process(&mut full);
        let start = t * STEP;
        for (i, w) in window.iter().enumerate() {
            result[start + i] += full[i].re / SIZE as f32 * w;
            norm[start + i] += w * w;
        }
    }
    for (i, sample) in converted[..n].iter_mut().enumerate() {
        let k = i + SIZE / 2;
        if norm[k] > 1e-6 {
            *sample = result[k] / norm[k];
        }
    }
    (removed / high_total.max(1e-12)) as f32
}

/// Hold a converted layer under the level of its own input, about every 50 ms.
///
/// The voice model renders a quiet input at a full singing level: a backing part
/// 8-10 dB under the lead (a faint double, the separator's echo of the lead) came
/// out as loud as the lead and howled behind it. Where the converted layer is more
/// than 1 dB above its input, it is brought down to it (see [`audio::hold_under`]).
/// Never boosts. Returns the largest cut in dB.
fn follow_level(converted: &mut [f32], input: &[f32]) -> f32 {
    audio::hold_under(converted, input, RATE, 1.0)
}

/// Per-frame 0/1 decisions as a per-sample gain with 60 ms cross-fades.
fn smooth_mask(decided: &[f32], frames: usize) -> Vec<f32> {
    let ramp = (0.06 * f64::from(RATE)) as usize;
    let half_ramp = ramp / 2;
    let mask: Vec<f32> = (0..frames)
        .map(|i| decided.get(i / HOP).copied().unwrap_or(0.0))
        .collect();
    // Widen each "on" stretch by half a ramp on both sides, then fade with a
    // centered moving average, so the whole stretch is covered and edges cross-fade.
    let mut ones = vec![0_usize; frames + 1];
    for (i, &m) in mask.iter().enumerate() {
        ones[i + 1] = ones[i] + usize::from(m > 0.0);
    }
    let widened: Vec<f32> = (0..frames)
        .map(|i| {
            let (low, high) = (i.saturating_sub(half_ramp), (i + half_ramp + 1).min(frames));
            if ones[high] > ones[low] { 1.0 } else { 0.0 }
        })
        .collect();
    let mut smooth = vec![0_f32; frames];
    let mut acc = 0_f32;
    for i in 0..frames + half_ramp {
        if i < frames {
            acc += widened[i];
        }
        if i >= ramp {
            acc -= widened[i - ramp];
        }
        if i >= half_ramp {
            smooth[i - half_ramp] = acc / ramp as f32;
        }
    }
    smooth
}

/// Give the voice back where the dereverb model took it for reverb.
///
/// The model splits a layer into dry voice and reverb, and only the dry part is
/// converted. Rough, noisy singing (screams, growls, strained belts) partly lands in
/// the "reverb" stem: the dry part then drops far under the layer on loud frames and
/// the converted voice falls silent there. Where the layer is within 18 dB of its loud
/// part but the dry stem sits more than 8 dB under it for most of ~0.3 s, the wet stem
/// is moved back into the dry one (60 ms cross-fades). Reverb tails after a phrase are
/// quieter or shorter and stay wet. Returns the samples restored.
fn restore_voice(input: &[f32], dry: &mut [f32], wet: &mut [f32]) -> usize {
    let (inp, d) = (frame_db(input), frame_db(dry));
    let n = inp.len().min(d.len());
    if n == 0 {
        return 0;
    }
    let mut sorted = inp[..n].to_vec();
    sorted.sort_by(f32::total_cmp);
    let peak = sorted[n * 95 / 100];
    let lost: Vec<bool> = (0..n).map(|i| inp[i] > peak - 18.0 && d[i] < inp[i] - 8.0).collect();
    let span = (RATE as usize * 3 / 10 / HOP).max(1);
    let decided: Vec<f32> = (0..n)
        .map(|i| {
            let range = i.saturating_sub(span / 2)..(i + span / 2 + 1).min(n);
            let hits = lost[range.clone()].iter().filter(|&&f| f).count();
            if lost[i] && hits * 2 > range.len() { 1.0 } else { 0.0 }
        })
        .collect();
    let frames = dry.len().min(wet.len());
    let mask = smooth_mask(&decided, frames);
    let mut restored = 0;
    for ((d, w), &m) in dry.iter_mut().zip(wet.iter_mut()).zip(&mask) {
        if m > 0.0 {
            *d += *w * m;
            *w *= 1.0 - m;
            restored += usize::from(m > 0.5);
        }
    }
    restored
}

/// Wet level relative to the dry voice, as an RMS ratio: about -16 dB at most.
/// The dereverb model's "reverb" stem also carries breath, separation residue and
/// rough voice, so the measured ratio overstates the room: half of it is used, and
/// never more than this. A dense synthetic tail reads louder than the original room
/// at the same RMS (listeners heard "far too much reverb" at -10 dB).
const MAX_WET: f32 = 0.16;
/// Share of the measured wet/dry ratio that the new reverb follows.
const WET_SHARE: f32 = 0.5;
/// Decay time of the new room, in seconds.
const RT60: f32 = 0.9;

/// The converted dry layer with a new reverb near the original wet/dry ratio.
fn layer(converted: &[f32], dry: &[f32], wet: &[f32], skip_reverb: bool) -> Vec<f32> {
    if skip_reverb {
        return converted.to_vec();
    }
    let ratio = (WET_SHARE * active_rms(wet) / active_rms(dry).max(1e-6)).min(MAX_WET);
    let tail = convolve(converted, &room(RATE, RT60));
    let gain = ratio * active_rms(converted) / active_rms(&tail).max(1e-6);
    converted
        .iter()
        .zip(&tail)
        .map(|(c, t)| c + gain * t)
        .collect()
}

/// A smooth room response: 20 ms of pre-delay, then noise decaying 60 dB in
/// `rt60` seconds whose high frequencies die faster, normalized to unit energy.
/// Dense noise has no comb resonances, so long notes do not swell or ring.
fn room(rate: u32, rt60: f32) -> Vec<f32> {
    let predelay = (0.02 * rate as f32) as usize;
    let length = (rt60 * rate as f32) as usize;
    let mut state = 0x2545_f491_u32;
    let mut low = 0_f32;
    let mut ir = vec![0_f32; predelay + length];
    for i in 0..length {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        let noise = state as f32 / u32::MAX as f32 * 2.0 - 1.0;
        let t = i as f32 / rate as f32;
        // One-pole low-pass sliding from ~10 kHz down to ~2 kHz along the tail.
        let cutoff = 10_000.0 * (1.0 - t / rt60) + 2_000.0 * (t / rt60);
        let a = (-std::f32::consts::TAU * cutoff / rate as f32).exp();
        low = (1.0 - a) * noise + a * low;
        ir[predelay + i] = low * (-6.91 * t / rt60).exp();
    }
    let energy = ir.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-9);
    ir.iter_mut().for_each(|v| *v /= energy);
    ir
}

/// Linear convolution truncated to the input length (FFT overlap-add).
fn convolve(x: &[f32], ir: &[f32]) -> Vec<f32> {
    use rustfft::FftPlanner;
    use rustfft::num_complex::Complex;
    let block = 1 << 16;
    let size = (block + ir.len()).next_power_of_two();
    let mut planner = FftPlanner::<f32>::new();
    let forward = planner.plan_fft_forward(size);
    let inverse = planner.plan_fft_inverse(size);
    let mut kernel: Vec<Complex<f32>> = ir.iter().map(|&v| Complex::new(v, 0.0)).collect();
    kernel.resize(size, Complex::new(0.0, 0.0));
    forward.process(&mut kernel);
    let mut out = vec![0_f32; x.len() + ir.len()];
    for (index, chunk) in x.chunks(block).enumerate() {
        let mut buffer: Vec<Complex<f32>> = chunk.iter().map(|&v| Complex::new(v, 0.0)).collect();
        buffer.resize(size, Complex::new(0.0, 0.0));
        forward.process(&mut buffer);
        buffer.iter_mut().zip(&kernel).for_each(|(b, k)| *b *= k);
        inverse.process(&mut buffer);
        let start = index * block;
        for (i, v) in buffer.iter().take(chunk.len() + ir.len()).enumerate() {
            out[start + i] += v.re / size as f32;
        }
    }
    out.truncate(x.len());
    out
}

fn add(bus: &mut Vec<f32>, other: &[f32]) {
    if bus.len() < other.len() {
        bus.resize(other.len(), 0.0);
    }
    for (b, o) in bus.iter_mut().zip(other) {
        *b += o;
    }
}

/// Integrated loudness of interleaved audio at 44.1 kHz (ITU-R BS.1770-4: K-weighting,
/// channel powers summed, 400 ms blocks with 75 % overlap, absolute and relative
/// gates), in LUFS.
fn loudness(x: &[f32], channels: usize) -> f64 {
    let fs = f64::from(RATE);
    // High shelf (+4 dB above ~1.5 kHz), then the RLB high-pass, as in pyloudnorm.
    let shelf = {
        let (g, q, fc) = (3.999_843_853_973_347, 0.707_175_236_955_419_6, 1_681.974_450_955_533);
        let a = 10_f64.powf(g / 40.0);
        let w = 2.0 * std::f64::consts::PI * fc / fs;
        let alpha = w.sin() / (2.0 * q);
        let c = w.cos();
        let b = [
            a * ((a + 1.0) + (a - 1.0) * c + 2.0 * a.sqrt() * alpha),
            -2.0 * a * ((a - 1.0) + (a + 1.0) * c),
            a * ((a + 1.0) + (a - 1.0) * c - 2.0 * a.sqrt() * alpha),
        ];
        let d = [
            (a + 1.0) - (a - 1.0) * c + 2.0 * a.sqrt() * alpha,
            2.0 * ((a - 1.0) - (a + 1.0) * c),
            (a + 1.0) - (a - 1.0) * c - 2.0 * a.sqrt() * alpha,
        ];
        ([b[0] / d[0], b[1] / d[0], b[2] / d[0]], [d[1] / d[0], d[2] / d[0]])
    };
    let highpass = {
        let (q, fc) = (0.500_327_037_323_877_3, 38.135_470_876_139_82);
        let w = 2.0 * std::f64::consts::PI * fc / fs;
        let alpha = w.sin() / (2.0 * q);
        let c = w.cos();
        let d0 = 1.0 + alpha;
        (
            [(1.0 + c) / 2.0 / d0, -(1.0 + c) / d0, (1.0 + c) / 2.0 / d0],
            [-2.0 * c / d0, (1.0 - alpha) / d0],
        )
    };
    let filter = |input: &[f64], (b, a): ([f64; 3], [f64; 2])| -> Vec<f64> {
        let (mut x1, mut x2, mut y1, mut y2) = (0.0, 0.0, 0.0, 0.0);
        input
            .iter()
            .map(|&x| {
                let y = b[0] * x + b[1] * x1 + b[2] * x2 - a[0] * y1 - a[1] * y2;
                (x2, x1, y2, y1) = (x1, x, y1, y);
                y
            })
            .collect()
    };
    let weighted: Vec<Vec<f64>> = (0..channels)
        .map(|c| {
            let channel: Vec<f64> = x.iter().skip(c).step_by(channels).map(|&v| f64::from(v)).collect();
            filter(&filter(&channel, shelf), highpass)
        })
        .collect();
    let frames = weighted.first().map_or(0, Vec::len);
    let block = (0.4 * fs) as usize;
    let step = block / 4;
    let powers: Vec<f64> = (0..frames.saturating_sub(block) / step + 1)
        .map(|i| {
            weighted
                .iter()
                .map(|w| {
                    let chunk = &w[i * step..(i * step + block).min(frames)];
                    chunk.iter().map(|v| v * v).sum::<f64>() / chunk.len().max(1) as f64
                })
                .sum()
        })
        .collect();
    let lufs = |p: f64| -0.691 + 10.0 * (p + 1e-20).log10();
    let gated: Vec<f64> = powers.iter().copied().filter(|&p| lufs(p) > -70.0).collect();
    if gated.is_empty() {
        return -70.0;
    }
    let relative = lufs(gated.iter().sum::<f64>() / gated.len() as f64) - 10.0;
    let kept: Vec<f64> = gated.into_iter().filter(|&p| lufs(p) > relative).collect();
    lufs(kept.iter().sum::<f64>() / kept.len().max(1) as f64)
}

/// Highest true-peak estimate around each frame: 4x oversampling with a
/// windowed-sinc interpolator, as BS.1770 meters do.
fn true_peaks(x: &[f32], channels: usize) -> Vec<f32> {
    const TAPS: usize = 8;
    let frames = x.len() / channels;
    let kernel: Vec<[f32; 2 * TAPS]> = (1..4)
        .map(|phase| {
            let mut k = [0_f32; 2 * TAPS];
            for (j, tap) in k.iter_mut().enumerate() {
                let t = j as f32 - TAPS as f32 + 1.0 - phase as f32 / 4.0;
                let sinc = if t == 0.0 { 1.0 } else { (std::f32::consts::PI * t).sin() / (std::f32::consts::PI * t) };
                let window = 0.5 + 0.5 * (std::f32::consts::PI * t / TAPS as f32).cos();
                *tap = sinc * window;
            }
            k
        })
        .collect();
    (0..frames)
        .map(|i| {
            let mut peak = 0_f32;
            for c in 0..channels {
                let at = |n: isize| -> f32 {
                    if n < 0 || n as usize >= frames { 0.0 } else { x[n as usize * channels + c] }
                };
                peak = peak.max(at(i as isize).abs());
                for k in &kernel {
                    let mut v = 0_f32;
                    for (j, tap) in k.iter().enumerate() {
                        v += tap * at(i as isize + j as isize - TAPS as isize + 1);
                    }
                    peak = peak.max(v.abs());
                }
            }
            peak
        })
        .collect()
}

/// True-peak limiter to -1 dBTP, 5 ms lookahead, 80 ms release.
fn limit(x: &[f32], channels: usize) -> Vec<f32> {
    let ceiling = 10_f32.powf(-1.0 / 20.0);
    let frames = x.len() / channels;
    let peaks = true_peaks(x, channels);
    let look = (0.005 * f64::from(RATE)) as usize;
    let release = (-1.0 / (0.08 * f64::from(RATE))).exp() as f32;
    let mut window = std::collections::VecDeque::new();
    let mut gain = 1_f32;
    let mut out = x.to_vec();
    // The first `look` samples come before the first lookahead index: seed the window
    // with them, or a hit right at the start of the track passes unlimited.
    for (j, &p) in peaks.iter().enumerate().take(look.min(frames)) {
        while window.back().is_some_and(|&(_, q)| q <= p) {
            window.pop_back();
        }
        window.push_back((j, p));
    }
    for i in 0..frames {
        let ahead = (i + look).min(frames - 1);
        while window.back().is_some_and(|&(_, p)| p <= peaks[ahead]) {
            window.pop_back();
        }
        window.push_back((ahead, peaks[ahead]));
        while window.front().is_some_and(|&(j, _)| j + 2 * look < i + look) {
            window.pop_front();
        }
        let peak = window.front().map_or(0.0, |&(_, p)| p);
        let target = if peak > ceiling { ceiling / peak } else { 1.0 };
        gain = if target < gain { target } else { target + (gain - target) * release };
        for c in 0..channels {
            out[i * channels + c] *= gain;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_scream_taken_for_reverb_returns_to_the_dry_voice() {
        // 1 s of clean singing, 1 s of a rough scream the dereverb put in "wet", 1 s of a
        // short quiet tail that really is reverb.
        let n = RATE as usize;
        let sung: Vec<f32> = (0..n).map(|i| (i as f32 * 0.03).sin() * 0.3).collect();
        let mut state = 9_u32;
        let mut noise = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state as f32 / u32::MAX as f32 - 0.5
        };
        let scream: Vec<f32> = (0..n).map(|_| noise() * 0.6).collect();
        let tail: Vec<f32> = (0..n).map(|i| noise() * 0.03 * (-(i as f32) / 4_000.0).exp()).collect();
        let mut dry: Vec<f32> = sung.iter().copied().chain(vec![0.0; 2 * n]).collect();
        let mut wet: Vec<f32> = vec![0.0; n].into_iter().chain(scream.iter().copied()).chain(tail.iter().copied()).collect();
        let input: Vec<f32> = dry.iter().zip(&wet).map(|(a, b)| a + b).collect();
        restore_voice(&input, &mut dry, &mut wet);
        let energy = |x: &[f32]| x.iter().map(|v| v * v).sum::<f32>();
        assert!(energy(&dry[n + 3_000..2 * n - 3_000]) > 0.95 * energy(&scream[3_000..n - 3_000]));
        assert!(energy(&wet[2 * n + 3_000..]) > 0.95 * energy(&tail[3_000..]));
        assert!((energy(&dry[..n - 3_000]) - energy(&sung[..n - 3_000])).abs() < 1e-3);
    }

    #[test]
    fn reverb_is_smooth_and_bounded() {
        // A long steady tone must not swell: the wet level stays at the cap.
        let tone: Vec<f32> = (0..44_100 * 4)
            .map(|i| (i as f32 * std::f32::consts::TAU * 440.0 / 44_100.0).sin() * 0.3)
            .collect();
        let wet_stem: Vec<f32> = tone.iter().map(|v| v * 2.0).collect();
        let out = layer(&tone, &tone, &wet_stem, false);
        let level = |x: &[f32]| (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt();
        let (early, late) = (level(&out[44_100..88_200]), level(&out[132_300..]));
        assert!((late / early - 1.0).abs() < 0.05, "{early} {late}");
        assert!(level(&out) < level(&tone) * 1.4);
        // The convolution matches direct summation.
        let x: Vec<f32> = (0..3000).map(|i| ((i * 7919) % 101) as f32 / 101.0 - 0.5).collect();
        let ir: Vec<f32> = (0..257).map(|i| (-(i as f32) / 40.0).exp()).collect();
        let fast = convolve(&x, &ir);
        for n in [0, 10, 500, 2999] {
            let direct: f32 = (0..=n.min(256)).map(|k| x[n - k] * ir[k]).sum();
            assert!((fast[n] - direct).abs() < 1e-3, "{n}");
        }
    }

    #[test]
    fn limiter_holds_the_ceiling() {
        let x: Vec<f32> = (0..44_100).map(|i| 2.0 * (i as f32 * 0.05).sin()).collect();
        let y = limit(&x, 1);
        let peak = y.iter().fold(0_f32, |m, v| m.max(v.abs()));
        assert!(peak <= 10_f32.powf(-1.0 / 20.0) + 1e-3, "{peak}");
        // An almost-Nyquist tone peaks between samples; the limiter must see that.
        let tone: Vec<f32> = (0..44_100)
            .map(|i| (std::f32::consts::PI * 0.5 * i as f32 + 0.785).sin())
            .collect();
        let truth = true_peaks(&tone, 1)[1000];
        assert!(truth > 0.99, "{truth}");
        // A hit in the first 5 ms (a beat right at the start) is limited too.
        let mut hit = vec![0.1_f32; 44_100];
        hit[100..110].fill(1.5);
        let y = limit(&hit, 1);
        let peak = y.iter().fold(0_f32, |m, v| m.max(v.abs()));
        assert!(peak <= 10_f32.powf(-1.0 / 20.0) + 1e-3, "{peak}");
    }

    #[test]
    fn loudness_of_full_scale_1k_sine_is_about_minus_3() {
        // BS.1770: a 0 dBFS 997 Hz sine in one channel reads -3.01 LUFS. The
        // 44.1 kHz filter design here lands within a few tenths of that, which
        // is plenty for matching two signals against each other.
        let x: Vec<f32> = (0..44_100 * 3)
            .map(|i| (i as f32 * std::f32::consts::TAU * 997.0 / 44_100.0).sin())
            .collect();
        let value = loudness(&x, 1);
        assert!((value + 3.01).abs() < 0.4, "{value}");
        let quieter: Vec<f32> = x.iter().map(|v| v * 0.5).collect();
        assert!((value - loudness(&quieter, 1) - 6.02).abs() < 0.01);
        let stereo: Vec<f32> = x.iter().flat_map(|&v| [v, v]).collect();
        assert!((loudness(&stereo, 2) - value - 3.01).abs() < 0.01);
    }

    #[test]
    fn layers_follow_what_the_backing_is() {
        let tone = |i: usize, f: f32| (i as f32 * f).sin() * 0.3;
        let second = 44_100;
        let stereo = |f: &dyn Fn(usize) -> f32| -> Stereo {
            Stereo((0..second * 6).flat_map(|i| { let v = f(i); [v, v] }).collect())
        };
        let energy = |x: &[f32]| x.iter().map(|v| v * v).sum::<f32>();
        // Lead sings the first two seconds, backing alone the next four: a solo line.
        let mut lead = stereo(&|i| if i < 2 * second { tone(i, 0.03) } else { 0.0 });
        let mut backing = stereo(&|i| if i >= 2 * second { tone(i, 0.05) } else { 0.0 });
        let plan = plan_layers(&mut lead, &mut backing);
        assert!(plan.solo > 3 * second, "{}", plan.solo);
        assert!(energy(&backing.0) < 0.01 * energy(&lead.0));
        // Residue 30 dB under the lead is the lead itself: one layer.
        let mut lead = stereo(&|i| tone(i, 0.03));
        let mut backing = stereo(&|i| tone(i, 0.07) * 0.03);
        let plan = plan_layers(&mut lead, &mut backing);
        assert!(plan.single);
        assert!(energy(&backing.0) < 1e-3 * energy(&lead.0));
        // A harmony 4 dB under the lead throughout stays its own layer.
        let mut lead = stereo(&|i| tone(i, 0.03));
        let mut backing = stereo(&|i| tone(i, 0.04) * 0.63);
        let plan = plan_layers(&mut lead, &mut backing);
        assert!(!plan.single && plan.parallel > 0.9, "{}", plan.parallel);
        assert!(energy(&backing.0) > 0.3 * energy(&lead.0));
        // The rasp of a split scream (noise 4 dB under the lead) is not a layer.
        let mut state = 5_u32;
        let mut noise = move || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state as f32 / u32::MAX as f32 - 0.5
        };
        let rasp: Vec<f32> = (0..second * 6).map(|_| noise() * 0.6).collect();
        let mut lead = stereo(&|i| tone(i, 0.03));
        let mut backing = stereo(&|i| rasp[i]);
        let plan = plan_layers(&mut lead, &mut backing);
        assert!(plan.single, "{}", plan.parallel);
        assert!(energy(&backing.0) < 1e-3 * energy(&lead.0));
    }

    #[test]
    fn a_quiet_layer_stays_quiet_after_conversion() {
        // Input: 1 s at -20 dB then 1 s at -40 dB; the model answered both at -14 dB.
        let n = RATE as usize;
        let tone = |i: usize, a: f32| (i as f32 * 0.07).sin() * a;
        let input: Vec<f32> = (0..2 * n).map(|i| tone(i, if i < n { 0.141 } else { 0.0141 })).collect();
        let mut converted: Vec<f32> = (0..2 * n).map(|i| tone(i, 0.28)).collect();
        let cut = follow_level(&mut converted, &input);
        assert!(cut > 24.0, "{cut}");
        let db = |x: &[f32]| 10.0 * (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).log10();
        let (loud, quiet) = (db(&converted[5_000..n - 5_000]), db(&converted[n + 5_000..2 * n - 5_000]));
        assert!((loud - db(&input[5_000..n - 5_000])).abs() < 1.2, "{loud}");
        assert!((quiet - db(&input[n + 5_000..2 * n - 5_000])).abs() < 1.2, "{quiet}");
        // A layer already at its input level is left alone.
        let mut same = input.clone();
        assert!(follow_level(&mut same, &input) < 0.01);
        assert_eq!(same, input);
    }

    #[test]
    fn converted_layers_cannot_invent_hiss() {
        let n = RATE as usize * 2;
        let mut state = 11_u32;
        let mut noise = move || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state as f32 / u32::MAX as f32 - 0.5
        };
        let hiss: Vec<f32> = (0..n).map(|_| noise() * 0.2).collect();
        let sung = |i: usize| -> f32 {
            (1..6)
                .map(|h| (i as f32 * std::f32::consts::TAU * 220.0 * h as f32 / RATE as f32).sin() * 0.2 / h as f32)
                .sum()
        };
        let input: Vec<f32> = (0..n).map(sung).collect();
        let high = |x: &[f32]| -> f32 {
            // Energy above ~5 kHz via a first difference applied three times.
            let mut d = x.to_vec();
            for _ in 0..3 {
                d = d.windows(2).map(|w| w[1] - w[0]).collect();
            }
            d[4_000..d.len() - 4_000].iter().map(|v| v * v).sum()
        };
        let low = |x: &[f32]| -> f32 { x[4_000..n - 4_000].iter().map(|v| v * v).sum() };
        // The model added broadband noise that the input never had: it goes.
        let mut converted: Vec<f32> = (0..n).map(|i| sung(i) + hiss[i]).collect();
        let removed = cap_hiss(&mut converted, &input);
        assert!(removed > 0.9, "{removed}");
        let clean: Vec<f32> = (0..n).map(sung).collect();
        assert!(high(&converted) < 0.1 * high(&(0..n).map(|i| sung(i) + hiss[i]).collect::<Vec<_>>()));
        assert!((low(&converted) / low(&clean) - 1.0).abs() < 0.1, "{}", low(&converted) / low(&clean));
        // Hiss that was in the input (a breathy line) stays.
        let breathy: Vec<f32> = (0..n).map(|i| sung(i) + hiss[i]).collect();
        let mut converted = breathy.clone();
        let removed = cap_hiss(&mut converted, &breathy);
        assert!(removed < 0.02, "{removed}");
    }
}
