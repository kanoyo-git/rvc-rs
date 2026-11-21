"""Vast-only probe of the public rvc extract path with a known monophonic F0."""
import argparse
import json
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--rvc", required=True)
parser.add_argument("--root", required=True)
parser.add_argument("--embedder", default="mhubert147")
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--methods", nargs="+", default=["fcpe", "rmvpe"])
args = parser.parse_args()
rng = np.random.default_rng(7412)
sr, hop = 16000, 160
samples = np.arange(sr) / sr
segments = []
reference = []
names = []

def add(name, f0, first=1, snr=None):
    pitch = np.broadcast_to(f0, samples.shape)
    if pitch.any():
        phase = 2 * np.pi * np.cumsum(pitch) / sr
        last = min(10, int((sr / 2 - 1) / pitch.max()))
        x = sum(np.sin(k * phase) / k for k in range(first, last + 1))
        x *= 0.24 / abs(x).max()
        if snr is not None:
            noise = rng.standard_normal(len(x))
            x += noise * np.sqrt(np.mean(x**2) / np.mean(noise**2)) * 10**(-snr / 20)
    elif name == "hiss":
        x = 0.03 * np.diff(rng.standard_normal(sr + 1))
    else:
        x = np.zeros(sr)
    envelope = np.minimum(1, np.minimum(samples, 1 - samples) / 0.01)
    segments.append(x * envelope)
    reference.append(pitch)
    names.append(name)

add("silence", 0)
add("bass_42", 42)
add("bass_65", 65)
add("harmonic_110", 110)
add("missing_fundamental_110", 110, first=2)
add("vibrato_220_5Hz_55cent", 220 * 2**(55 * np.sin(2 * np.pi * 5 * samples) / 1200))
add("hiss", 0)
add("harmonic_440_0dB_white", 440, snr=0)
add("high_1250", 1250)

with tempfile.TemporaryDirectory(prefix="rvc-fcpe-probe-") as temporary:
    run_root = Path(temporary)
    pcm = np.clip(np.rint(np.concatenate(segments) * 32767), -32768, 32767).astype("<i2")
    for method in args.methods:
        experiment = run_root / method
        waves = experiment / "1_16k_wavs"
        waves.mkdir(parents=True)
        with wave.open(str(waves / "probe.wav"), "wb") as file:
            file.setparams((1, 2, sr, 0, "NONE", "not compressed"))
            file.writeframes(pcm.tobytes())
        subprocess.run([
            args.rvc, "--root", args.root, "extract", "--experiment", str(experiment),
            "--embedder", args.embedder, "--f0-method", method, "--device", args.device,
        ], check=True)
        track = np.load(experiment / "2d-f0raw" / "probe.wav.npy")
        report = {}
        for i, name in enumerate(names):
            # Discard the first/last 100 ms of each synthetic block and the terminal frame.
            idx = np.arange(10, 90)
            estimated = track[i * 100 + idx]
            actual = reference[i][idx * hop + hop // 2]
            active = estimated > 0
            row = {"n": len(idx), "voiced_pct": round(float(100 * active.mean()), 2)}
            if actual[0] > 0:
                err = 1200 * np.log2(estimated[active] / actual[active])
                row["within50_all_ref_pct"] = round(float(100 * np.count_nonzero(abs(err) <= 50) / len(idx)), 2)
                row["octave_error_count"] = int(np.count_nonzero(abs(abs(err) - 1200) <= 50))
                row["median_abs_cents_on_voiced"] = round(float(np.median(abs(err))), 2) if err.size else None
            report[name] = row
        print("FCPE_PROBE " + json.dumps({"method": method, "frames": int(track.size), "conditions": report}), flush=True)
