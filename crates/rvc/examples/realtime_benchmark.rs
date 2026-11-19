//! Measures the realtime model worker with a deterministic rolling input buffer.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, ensure};
use clap::Parser;
use rvc::catalog::Paths;
use rvc::model::{ConversionSettings, Rvc, weights};
use serde_json::json;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    root: PathBuf,
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    input: PathBuf,
    #[arg(long, default_value = "cpu")]
    device: String,
    #[arg(long, default_value_t = 20)]
    blocks: usize,
    #[arg(long, default_value_t = 3)]
    warmups: usize,
}

const HOP: usize = 160;
const BLOCK: usize = 25;
const CONTEXT: usize = 250;
const CROSSFADE: usize = 5;
const RENDER: usize = BLOCK + 4 + 1;

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(args.blocks > 0, "at least one measured block is required");
    let raw = std::fs::read(&args.input)?;
    let (frames, remainder) = raw.as_chunks::<4>();
    ensure!(
        !frames.is_empty() && remainder.is_empty(),
        "expected 16 kHz mono f32le PCM"
    );
    let source: Vec<_> = frames.iter().copied().map(f32::from_le_bytes).collect();
    ensure!(
        source.iter().all(|sample| sample.is_finite()),
        "non-finite input audio"
    );
    let paths = Paths::new(&args.root)?;
    let model = paths.model(&args.model)?;
    let device = weights::device(&args.device)?;
    let settings = ConversionSettings {
        index_rate: 0.0,
        ..Default::default()
    };
    let mut engine = Rvc::load(&paths, &model, None, None, settings, device)?;
    let mut window = vec![0.0_f32; (CONTEXT + CROSSFADE + 1 + BLOCK) * HOP];
    let cut = window.len() - BLOCK * HOP;
    let mut samples = Vec::with_capacity(args.blocks);
    let mut peak = 0.0_f32;
    for block in 0..args
        .warmups
        .checked_add(args.blocks)
        .context("block count overflow")?
    {
        window.copy_within(BLOCK * HOP.., 0);
        for (i, sample) in window[cut..].iter_mut().enumerate() {
            *sample = source[(block * BLOCK * HOP + i) % source.len()];
        }
        let started = Instant::now();
        let output = engine.convert_window(&window, settings, CONTEXT, RENDER, BLOCK)?;
        let milliseconds = started.elapsed().as_secs_f64() * 1000.0;
        ensure!(
            output.len() == RENDER * engine.rate() as usize / 100
                && output.iter().all(|sample| sample.is_finite()),
            "invalid realtime output"
        );
        if block >= args.warmups {
            peak = output.iter().map(|x| x.abs()).fold(peak, f32::max);
            samples.push(milliseconds);
        }
    }
    println!(
        "{}",
        json!({
            "device": args.device,
            "block_ms": BLOCK * 10,
            "samples_per_output": RENDER * engine.rate() as usize / 100,
            "peak": peak,
            "times_ms": samples,
        })
    );
    Ok(())
}
