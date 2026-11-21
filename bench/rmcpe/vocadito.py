"""Score the public Rust extraction path against Vocadito v3, per singer and track."""
import argparse
import csv
import hashlib
import io
import json
import re
import subprocess
import time
import wave
import zipfile
from pathlib import Path

import numpy as np

ARCHIVE_MD5 = "dea40fd18f14d899643c4ba221b33a46"
TRACK_ID = re.compile(r"vocadito_(\d+)")


def partition(singer):
    # Fixed before evaluation. An artist can appear in exactly one partition.
    return "calibration" if int.from_bytes(hashlib.sha256(singer.encode()).digest()[:4], "big") % 3 == 0 else "heldout"


def prepare(archive, work):
    with archive.open("rb") as stream:
        if hashlib.file_digest(stream, "md5").hexdigest() != ARCHIVE_MD5:
            raise ValueError("Expected the complete original Vocadito v3 archive")
    root = work / "waves"
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        names = [n for n in source.namelist() if "__MACOSX" not in n]
        metadata = next(n for n in names if n.endswith("vocadito_metadata.csv"))
        rows = list(csv.DictReader(io.StringIO(source.read(metadata).decode("utf-8-sig"))))
        tracks = {}
        for row in rows:
            track = "vocadito_" + row["track_id"]
            files = [n for n in names if (m := TRACK_ID.search(Path(n).name)) and m.group() == track]
            wav, = [n for n in files if n.lower().endswith(".wav")]
            reference, = [n for n in files if n.lower().endswith(".csv") and "f0" in n.lower()]
            output = root / (track + ".wav")
            if not output.exists():
                temp = output.with_suffix(".tmp.wav")
                subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", "pipe:0", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(temp)], input=source.read(wav), check=True)
                temp.replace(output)
            with wave.open(str(output)) as wav_file:
                length = wav_file.getnframes()
            truth = np.loadtxt(io.BytesIO(source.read(reference)), delimiter=",")
            if not np.isfinite(truth).all() or truth.shape[1] != 2 or not np.all(np.diff(truth[:, 0]) > 0):
                raise ValueError(f"Invalid annotated time grid in {reference}")
            tracks[track] = row["singer_id"], truth, length
    if len(tracks) != 40:
        raise ValueError(f"Expected 40 tracks, found {len(tracks)}")
    return tracks


def score(track, truth, length, method):
    if len(track) != length // 160 + 1 or not np.isfinite(track).all():
        raise ValueError(f"Invalid output length or F0 from {method}")
    time_s, actual = truth.T
    # Native STFT centers. Match the nearest prediction without smoothing voicing boundaries.
    offset_s = 0.005 if method == "fcpe" else 0.0
    def indices(offset):
        return np.floor((time_s - offset) * 100.0 + 0.5).astype(int)
    idx = indices(offset_s)
    # Both native grids must have a prediction, including at the two file edges.
    valid = (indices(0.005) >= 0) & (indices(0.0) < len(track)) & (time_s < length / 16000.0)
    actual = actual[valid]
    estimate = track[idx[valid]]
    est = estimate > 0
    ref = actual > 0
    both = est & ref
    error = np.abs(1200.0 * np.log2(estimate[both] / actual[both]))
    return {"frames": len(ref), "ref_voiced": int(ref.sum()), "ref_unvoiced": int((~ref).sum()),
            "correct_50": int((error <= 50).sum()), "false_voice": int((est & ~ref).sum()),
            "missed_voice": int((~est & ref).sum()), "octave_errors": int((np.abs(error - 1200) <= 50).sum())}


def summary(rows):
    total = {key: sum(r[key] for r in rows) for key in rows[0]}
    ref = total["ref_voiced"]
    uv = total["ref_unvoiced"]
    def pct(n, d):
        return round(100.0 * n / d, 3) if d else None
    return dict(tracks=len(rows), **total, voiced_within_50_pct=pct(total["correct_50"], ref),
                false_voice_pct=pct(total["false_voice"], uv), missed_voice_pct=pct(total["missed_voice"], ref),
                octave_pct=pct(total["octave_errors"], ref),
                overall_accuracy_pct=pct(total["correct_50"] + uv - total["false_voice"], total["frames"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--rvc", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--methods", nargs="+", default=["rmvpe", "fcpe", "rmcpe"], choices=["rmvpe", "fcpe", "rmcpe"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--split", choices=["calibration", "heldout", "all"], default="all")
    parser.add_argument("--score-only", action="store_true", help="rescore existing predictions and retain the recorded extraction time")
    args = parser.parse_args()
    work = args.work.resolve()
    tracks = prepare(args.archive, work)
    features = work / "features"
    features.mkdir(exist_ok=True)
    for method in args.methods:
        experiment = work / method
        experiment.mkdir(exist_ok=True)
        for name, dest in [("1_16k_wavs", work / "waves"), ("3_feature768", features)]:
            link = experiment / name
            if not link.exists():
                link.symlink_to(dest, target_is_directory=True)
        result_path = work / f"{method}-{args.split}.json"
        if args.score_only:
            extraction_seconds = json.loads(result_path.read_text())["fresh_or_cached_extraction_s"]
        else:
            # Feature-only pass puts every embedder computation outside the timed F0 pass.
            base = [args.rvc, "--root", args.root, "extract", "--experiment", str(experiment),
                    "--embedder", "mhubert147", "--device", args.device]
            subprocess.run(base, check=True)
            extraction_start = time.monotonic()
            subprocess.run(base + ["--f0-method", method], check=True)
            extraction_seconds = round(time.monotonic() - extraction_start, 3)
        results = {}
        for track, (singer, truth, length) in tracks.items():
            split = partition(singer)
            if args.split != "all" and split != args.split:
                continue
            estimate = np.load(experiment / "2d-f0raw" / (track + ".wav.npy"))
            results[track] = dict(singer=singer, split=split, counts=score(estimate, truth, length, method))
        group = {s: summary([r["counts"] for r in results.values() if s == "all" or r["split"] == s])
                 for s in (["all", "calibration", "heldout"] if args.split == "all" else [args.split])}
        output = {"dataset": "Vocadito v3", "archive_md5": ARCHIVE_MD5, "method": method,
                  "device": args.device, "split": args.split, "fresh_or_cached_extraction_s": extraction_seconds,
                  "aggregates": group, "tracks": results}
        result_path.write_text(json.dumps(output, indent=2) + "\n")
        print(method, json.dumps(group), flush=True)


if __name__ == "__main__":
    main()
