"""Compare the original and native PyMSS CLI on one track and catalog model."""

import argparse
import json
import math
import os
from pathlib import Path
import shutil
from statistics import median
import subprocess
import tempfile

from compare import THREAD_ENV, command_usage, gpu_query, sha256


def stem_layout(directory):
    result = {}
    for path in sorted(directory.glob("*.wav")):
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
             "stream=channels,sample_rate,duration_ts,codec_name", "-of", "json", str(path)],
            check=True, capture_output=True, text=True,
        )
        streams = json.loads(probe.stdout)["streams"]
        if len(streams) != 1:
            raise ValueError(f"{path} must have one audio stream")
        result[path.name] = streams[0]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--rust-binary", type=Path, required=True)
    parser.add_argument("--rust-lib-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--gpu-stats", action="store_true")
    parser.add_argument("--param", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if min(args.runs, args.threads) <= 0 or args.warmups < 0 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("runs, threads and timeout must be positive, warmups nonnegative")
    if args.gpu < 0 or args.gpu_stats and args.device != "cuda":
        parser.error("GPU index must be nonnegative and --gpu-stats requires --device cuda")

    cuda_selector = "-1"
    gpu = None
    gpu_name = None
    if args.device == "cuda":
        cuda_selector = str(args.gpu)
        if args.gpu_stats:
            smi = shutil.which("nvidia-smi")
            if smi is None:
                parser.error("--gpu-stats requires nvidia-smi")
            rows = gpu_query(smi, args.gpu, "name,uuid")
            if len(rows) != 1:
                raise RuntimeError(f"nvidia-smi did not return one GPU for index {args.gpu}")
            gpu_name, cuda_selector = rows[0]
            gpu = (smi, args.gpu, 0.2)

    baseline = args.baseline.absolute()
    track = args.input.absolute()
    common = os.environ.copy()
    common.update({key: str(args.threads) for key in THREAD_ENV})
    common.update({
        "CUDA_VISIBLE_DEVICES": cuda_selector,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": os.pathsep.join(map(str, (baseline, baseline / "tools"))),
    })
    rust = common | {"LD_LIBRARY_PATH": str(args.rust_lib_dir.absolute())}
    executables = {
        "Python": ([str(args.python.absolute()), "-m", "tools.pymss.cli"], common),
        "Rust": ([str(args.rust_binary.absolute()), "pymss"], rust),
    }
    results = {language: [] for language in executables}
    cpus = os.sched_getaffinity(0)
    overrides = [part for value in args.param for part in ("--param", value)]
    with tempfile.TemporaryDirectory(prefix="pymss-bench-") as directory:
        work = Path(directory)
        expected = None
        for trial in range(args.warmups + args.runs):
            order = tuple(executables)
            for language in (order if trial % 2 == 0 else order[::-1]):
                executable, env = executables[language]
                path = work / f"{language}-{trial + 1}"
                argv = executable + [
                    "infer", args.model, "--model-dir", str(args.model_dir.absolute()),
                    "-i", str(track), "-o", str(path), "--device", args.device,
                ] + overrides
                log = path.with_suffix(".log")
                print(f"{language} {'warmup' if trial < args.warmups else 'trial'} {trial + 1}", flush=True)
                record = command_usage(
                    argv, baseline, env, cpus, log, args.timeout, gpu,
                )
                if record["returncode"]:
                    raise RuntimeError(f"{language} exited {record['returncode']}\n{log.read_text()[-6000:]}")
                output = stem_layout(path)
                if not output or (expected is not None and output != expected):
                    raise RuntimeError(f"{language} produced different WAV layouts: {output}")
                expected = output
                if trial >= args.warmups:
                    results[language].append(record)
    print("| CLI | Wall s | CPU s | RSS MiB |" + (" VRAM MiB |" if gpu else ""))
    print("| --- | ---: | ---: | ---: |" + (" ---: |" if gpu else ""))
    for language, runs in results.items():
        wall, cpu, rss = (median(run[key] for run in runs)
                          for key in ("elapsed_s", "cpu_total_s", "peak_rss_mib"))
        row = f"| {language} | {wall:.3f} | {cpu:.3f} | {rss:.1f} |"
        if gpu:
            peaks = [run["gpu_peak_vram_mib"] for run in runs]
            if any(peak is None for peak in peaks) or any(run["gpu_sampling_error"] for run in runs):
                raise RuntimeError(f"{language} has incomplete VRAM samples")
            row += f" {median(peaks):.1f} |"
        print(row)
    if args.json:
        args.json.write_text(json.dumps({
            "model": args.model, "input_sha256": sha256(track), "device": args.device,
            "gpu": gpu_name, "cpus": sorted(cpus), "threads": args.threads,
            "params": args.param, "output": expected, "trials": results,
        }, indent=2) + "\n")


if __name__ == "__main__":
    main()
