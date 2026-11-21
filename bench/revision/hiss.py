"""Measure broadband noise in time-aligned source/conversion pairs at 40 kHz.

Usage: python hiss.py SOURCE.wav BEFORE.wav AFTER.wav --start 60 --seconds 25 -o stats.json
Frame classes come only from the source. Band RMS includes both noise and harmonics.
The noise-floor estimate uses the lower quartile within 500 Hz tiles, scaled to
the expected power of white noise; isolating the floor limits the influence of
individual voice harmonics. Cross-speaker differences are descriptive.
"""

import argparse
import json
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.ndimage import minimum_filter1d
from scipy.signal import resample_poly


RATE, HOP, SIZE, FFT = 40_000, 400, 1_600, 2_048
WINDOW = np.hanning(SIZE).astype(np.float32)
FREQ = np.fft.rfftfreq(FFT, 1 / RATE)
BANDS = ((4000, 8000), (8000, 12000), (12000, 16000), (16000, 20000))


def read(path, start, seconds):
    with sf.SoundFile(path) as stream:
        stream.seek(round(start * stream.samplerate))
        count = -1 if seconds is None else round(seconds * stream.samplerate)
        data = stream.read(count, dtype="float32", always_2d=True).mean(axis=1)
        factor = gcd(stream.samplerate, RATE)
        if stream.samplerate != RATE:
            data = resample_poly(data, RATE // factor, stream.samplerate // factor)
    frames = np.lib.stride_tricks.sliding_window_view(data, SIZE)[::HOP]
    power = abs(np.fft.rfft(frames * WINDOW, n=FFT)) ** 2
    # One-sided FFT power in units of mean squared audio amplitude.
    return power * (2.0 / (FFT * np.sum(WINDOW ** 2)))


def db(power):
    return 10 * np.log10(np.maximum(power, 1e-13))


def band(power, low, high):
    return power[:, (FREQ >= low) & (FREQ < high)]


def classes(power):
    total = band(power, 80, 20000).sum(axis=1)
    levels = db(total)
    relative = levels - np.percentile(levels, 95)
    sub = band(power, 300, 8000)
    flat = np.exp(np.log(np.maximum(sub, 1e-14)).mean(axis=1)) / np.maximum(sub.mean(axis=1), 1e-14)
    high = band(power, 3500, 12000).sum(axis=1) / np.maximum(total, 1e-13)
    low = band(power, 80, 1000).sum(axis=1) / np.maximum(total, 1e-13)
    sib = (high > 0.55) & (relative > -35)
    breath = ~sib & (flat > 0.25) & (low < 0.5) & (relative > -45) & (relative < -12)
    vowel = (low > 0.65) & (flat < 0.05) & (relative > -27)
    # Require a quiet neighbourhood to exclude note onsets and consonant tails.
    pause = minimum_filter1d((relative < -45).astype(np.int8), size=5, mode="nearest") > 0
    return {"sibilant": sib, "breath": breath, "vowel": vowel, "pause": pause}


def descriptors(power):
    levels, floors = [], []
    for low, high in BANDS:
        levels.append(band(power, low, high).sum(axis=1))
        floor = np.zeros(len(power))
        for edge in range(low, high, 500):
            tile = band(power, edge, edge + 500)
            floor += np.quantile(tile, 0.25, axis=1) * tile.shape[1] / -np.log(0.75)
        floors.append(np.minimum(floor, levels[-1]))
    return np.column_stack(levels), np.column_stack(floors)


def compare(src, out, reference=None):
    if reference is None:
        reference = src
    lengths = [len(src), len(out), len(reference)]
    if max(lengths) - min(lengths) > 8:
        raise ValueError(f"Unaligned recording lengths {lengths}")
    frames = min(lengths)
    src, out, reference = src[:frames], out[:frames], reference[:frames]
    masks = classes(src)
    reference_band, reference_floor = descriptors(reference)
    output_band, output_floor = descriptors(out)
    # Global recording gain is not hiss. Match the median level on source vowels.
    voiced = masks["vowel"]
    gain = 0.0
    if np.count_nonzero(voiced) >= 5:
        gain = float(np.median(db(band(out[voiced], 80, 8000).sum(axis=1))
                               - db(band(reference[voiced], 80, 8000).sum(axis=1))))
    rows = {}
    for name, mask in masks.items():
        row = {"frames": int(mask.sum())}
        if mask.any():
            for key, source, output in (("band", reference_band, output_band), ("floor", reference_floor, output_floor)):
                a, b = db(source[mask].mean(axis=0)), db(output[mask].mean(axis=0))
                row[key + "_reference_dbfs"] = a.round(2).tolist()
                row[key + "_output_dbfs"] = b.round(2).tolist()
                row[key + "_delta_db"] = (b - a - gain).round(2).tolist()
                a, b = np.maximum(db(source[mask]), -85), np.maximum(db(output[mask]) - gain, -85)
                row[key + "_mae_db"] = np.abs(b - a).mean(axis=0).round(2).tolist()
        rows[name] = row
    return {"gain_db": round(gain, 2), "classes": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("outputs", type=Path, nargs="+")
    parser.add_argument("--start", type=float, default=0)
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--reference", type=Path, help="An aligned reference render, e.g. upstream Python")
    parser.add_argument("-o", "--output", type=Path)
    args = parser.parse_args()
    source = read(args.source, args.start, args.seconds)
    reference = read(args.reference, args.start, args.seconds) if args.reference else source
    report = {
        "source": str(args.source), "reference": str(args.reference or args.source),
        "start": args.start, "seconds": args.seconds, "bands_hz": BANDS,
        "outputs": {str(path): compare(source, read(path, args.start, args.seconds), reference) for path in args.outputs},
    }
    result = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(result)
    else:
        print(result, end="")
