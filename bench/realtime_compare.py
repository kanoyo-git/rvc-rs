"""Measure realtime model latency and process resources on shared 16 kHz PCM."""

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import tempfile

from compare import THREAD_ENV, command_usage, gpu_query


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--rust-binary", type=Path, required=True)
    parser.add_argument("--rust-lib-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", type=Path, required=True, help="Mono float32 PCM at 16 kHz")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--blocks", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=240)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.blocks < 1 or args.warmups < 0 or args.threads < 1 or args.gpu < 0:
        parser.error("expected positive blocks/threads and nonnegative warmups/GPU index")
    baseline = args.baseline.absolute()
    args_str = [
        "--root", str(baseline), "--model", args.model, "--input", str(args.input.absolute()),
        "--device", "cuda:0" if args.device == "cuda" else "cpu", "--blocks", str(args.blocks),
        "--warmups", str(args.warmups),
    ]
    common = dict(os.environ)
    common.pop("PYTHONPATH", None)
    common.update({name: str(args.threads) for name in THREAD_ENV})
    common.update({
        "CUDA_VISIBLE_DEVICES": "-1", "PYTHONDONTWRITEBYTECODE": "1",
        "weight_root": str(baseline / "assets/weights"),
        "rmvpe_root": str(baseline / "assets/rmvpe"),
    })
    gpu = None
    if args.device == "cuda":
        nvidia_smi = shutil.which("nvidia-smi")
        if nvidia_smi is None:
            parser.error("nvidia-smi is required for GPU sampling")
        common["CUDA_VISIBLE_DEVICES"] = gpu_query(nvidia_smi, args.gpu, "uuid")[0][0]
        gpu = (nvidia_smi, args.gpu, 0.2)
    rust_env = common.copy()
    previous = rust_env.get("LD_LIBRARY_PATH")
    rust_env["LD_LIBRARY_PATH"] = str(args.rust_lib_dir.absolute()) + (":" + previous if previous else "")
    runs = {
        "Python": ([str(args.python.absolute()), str(Path(__file__).with_name("realtime_python.py").absolute())], common),
        "Rust": ([str(args.rust_binary.absolute())], rust_env),
    }
    results = {}
    with tempfile.TemporaryDirectory(prefix="rvc-rt-bench-") as work:
        for language, (command, environment) in runs.items():
            logfile = Path(work) / f"{language}.log"
            resources = command_usage(
                command + args_str, baseline, environment, os.sched_getaffinity(0), logfile, args.timeout, gpu,
            )
            log = logfile.read_text()
            if resources["returncode"] != 0:
                raise RuntimeError(f"{language} exited {resources['returncode']}\n{log[-6000:]}")
            output = json.loads(log.strip().splitlines()[-1])
            times = sorted(output["times_ms"])
            results[language] = {
                **resources, **output,
                "p50_ms": statistics.median(times),
                "p95_ms": times[math.ceil(len(times) * 0.95) - 1],
                "blocks_over_budget": sum(duration > output["block_ms"] for duration in times),
            }
    a, b = results.values()
    if a["samples_per_output"] != b["samples_per_output"]:
        raise RuntimeError("the two workers produced different output lengths")
    print("| Worker | p50 ms | p95 ms | Late blocks | CPU s | RSS MiB | VRAM MiB |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for language, item in results.items():
        memory = item.get("gpu_peak_vram_mib")
        print(f"| {language} | {item['p50_ms']:.2f} | {item['p95_ms']:.2f} | "
              f"{item['blocks_over_budget']} / {args.blocks} | {item['cpu_total_s']:.2f} | "
              f"{item['peak_rss_mib']:.1f} | {memory if memory is not None else '—'} |")
    if args.json:
        args.json.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
