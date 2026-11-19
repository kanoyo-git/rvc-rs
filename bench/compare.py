"""Compare the original and Rust RVC offline inference CLIs on Linux."""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import select
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import wave


THREAD_ENV = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "RAYON_NUM_THREADS",
)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True, help="Separate checkout of the original Python WebUI with the required auxiliary weights.")
    parser.add_argument("--python", type=Path, required=True, help="Interpreter from the baseline's virtual environment.")
    parser.add_argument("--rust-binary", type=Path, required=True, help="Release build of the Rust CLI.")
    parser.add_argument("--rust-lib-dir", type=Path, help="Directory containing the LibTorch shared libraries for Rust.")
    parser.add_argument("--model", type=Path, required=True, help="The same compact RVC .pth for both CLIs.")
    parser.add_argument("--inputs", type=Path, required=True, help="Directory of top-level PCM WAVs. A batch requires at least two.")
    parser.add_argument("--single", help="WAV filename for the single-file trial. Defaults to the first sorted input.")
    parser.add_argument("--case", choices=("single", "batch"), action="append", help="Run selected cases. Defaults to both.")
    parser.add_argument("--threads", type=int, default=min(4, len(os.sched_getaffinity(0))))
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=180.0, help="Seconds allowed for one CLI invocation.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--gpu", type=int, default=0, help="Physical NVIDIA GPU index, with visibility remapped to logical CUDA device 0.")
    parser.add_argument("--gpu-stats", action="store_true", help="Sample per-process VRAM and device-wide utilization with nvidia-smi.")
    parser.add_argument("--gpu-sample-interval", type=float, default=0.2)
    parser.add_argument("--f0-method", choices=("rmvpe", "pm"), default="rmvpe")
    parser.add_argument("--speaker-id", type=int, help="Speaker ID shared by both CLIs. Defaults to the model's selection.")
    parser.add_argument("--index", type=Path, help="Optional shared retrieval index. Retrieval is disabled by default.")
    parser.add_argument("--index-rate", type=float, default=0.75, help="Retrieval mix when --index is given.")
    parser.add_argument("--json", type=Path, help="Write all raw trials and metadata as JSON.")
    args = parser.parse_args()
    if args.runs < 1 or args.warmups < 0 or args.threads < 1 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("runs and threads must be positive, warmups non-negative, and timeout positive")
    if args.gpu < 0 or not math.isfinite(args.gpu_sample_interval) or args.gpu_sample_interval <= 0 or not 0 <= args.index_rate <= 1:
        parser.error("GPU must be non-negative, sampling interval positive, and index rate in [0, 1]")
    if args.gpu_stats and args.device != "cuda":
        parser.error("--gpu-stats requires --device cuda")
    return args


def command_environment(args, cuda_selector):
    common = dict(os.environ)
    common.pop("PYTHONPATH", None)
    common.update({name: str(args.threads) for name in THREAD_ENV})
    common.update({
        "CUDA_VISIBLE_DEVICES": cuda_selector,
        "PYTHONDONTWRITEBYTECODE": "1",
        "RVC_AUDIO_FORCE_CPU": "0" if args.device == "cuda" else "1",
        "weight_root": str(args.baseline / "assets/weights"),
        "index_root": str(args.baseline / "logs"),
        "outside_index_root": str(args.baseline / "assets/indices"),
        "rmvpe_root": str(args.baseline / "assets/rmvpe"),
    })
    rust = common.copy()
    if args.rust_lib_dir:
        previous = rust.get("LD_LIBRARY_PATH")
        rust["LD_LIBRARY_PATH"] = str(args.rust_lib_dir) + (f":{previous}" if previous else "")
    return {"Python": common, "Rust": rust}


def wav_info(path):
    with wave.open(str(path), "rb") as wav:
        return {
            "sample_rate": wav.getframerate(),
            "channels": wav.getnchannels(),
            "sample_width": wav.getsampwidth(),
            "frames": wav.getnframes(),
        }


def sha256(path):
    with path.open("rb") as data:
        return hashlib.file_digest(data, "sha256").hexdigest()


def first_cpu_model():
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        key, _, value = line.partition(":")
        if key.strip() == "model name":
            return value.strip()
    return platform.machine()


def gpu_query(executable, device, fields, *, processes=False):
    key = "--query-compute-apps" if processes else "--query-gpu"
    selection = [] if device is None else ["--id", str(device)]
    result = subprocess.run(
        [executable, *selection, f"{key}={fields}", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
        timeout=2.0,
    )
    return [[field.strip() for field in row] for row in csv.reader(result.stdout.splitlines()) if row]


class GpuSampler:
    def __init__(self, executable, device, interval, pid):
        self.executable = executable
        self.device = device
        self.interval = interval
        self.pid = str(pid)
        self.stop = threading.Event()
        self.util = []
        self.memory = []
        self.error = None
        self.worker = threading.Thread(target=self.sample, daemon=True)

    def sample(self):
        try:
            while not self.stop.is_set():
                applications = gpu_query(self.executable, self.device, "pid,used_gpu_memory", processes=True)
                self.memory.extend(int(row[1]) for row in applications if len(row) == 2 and row[0] == self.pid and row[1].isdigit())
                rows = gpu_query(self.executable, self.device, "utilization.gpu")
                if rows and rows[0][0].isdigit():
                    self.util.append(int(rows[0][0]))
                self.stop.wait(self.interval)
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            self.error = str(error)

    def finish(self):
        self.stop.set()
        self.worker.join()
        return {
            "gpu_peak_vram_mib": max(self.memory, default=None),
            "gpu_device_mean_percent": statistics.mean(self.util) if self.util else None,
            "gpu_vram_samples": len(self.memory),
            "gpu_util_samples": len(self.util),
            "gpu_sampling_error": self.error,
        }


def wait_usage(pid, deadline):
    """Wait for a child without losing its individual Linux getrusage values."""
    pidfd = None
    if hasattr(os, "pidfd_open"):
        try:
            pidfd = os.pidfd_open(pid)
        except OSError:
            pass
    if pidfd is not None:
        try:
            if not select.select([pidfd], [], [], max(0.0, deadline - time.perf_counter()))[0]:
                raise TimeoutError
            return os.wait4(pid, 0)
        finally:
            os.close(pidfd)
    while True:
        waited = os.wait4(pid, os.WNOHANG)
        if waited[0]:
            return waited
        delay = deadline - time.perf_counter()
        if delay <= 0:
            raise TimeoutError
        time.sleep(min(0.005, delay))


def command_usage(command, cwd, environment, cpus, logfile, timeout, gpu):
    with logfile.open("wb") as log:
        start = time.perf_counter()
        child = subprocess.Popen(
            command,
            cwd=cwd,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            preexec_fn=lambda: os.sched_setaffinity(0, cpus),
        )
        sampler = GpuSampler(*gpu, child.pid) if gpu else None
        try:
            if sampler:
                sampler.worker.start()
            _, status, usage = wait_usage(child.pid, start + timeout)
            elapsed = time.perf_counter() - start
        except BaseException:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            _, status, _ = os.wait4(child.pid, 0)
            child.returncode = os.waitstatus_to_exitcode(status)
            raise
        finally:
            gpu_stats = sampler.finish() if sampler and sampler.worker.ident is not None else {}
        child.returncode = os.waitstatus_to_exitcode(status)
    return {
        "elapsed_s": elapsed,
        "cpu_user_s": usage.ru_utime,
        "cpu_system_s": usage.ru_stime,
        "cpu_total_s": usage.ru_utime + usage.ru_stime,
        "cpu_percent_one_core": 100 * (usage.ru_utime + usage.ru_stime) / elapsed,
        "peak_rss_mib": usage.ru_maxrss / 1024,
        "returncode": child.returncode,
        **gpu_stats,
    }


def workload_commands(args, input_path, output_path):
    common = [
        "--model", args.model, "--input", input_path, "--output", output_path,
        "--f0-method", args.f0_method, "--pitch", "0",
        "--index-rate", str(args.index_rate if args.index else 0.0),
        "--rms-mix-rate", "1", "--protect", "0.33", "--format", "wav",
    ]
    if args.speaker_id is not None:
        common.extend(["--speaker-id", str(args.speaker_id)])
    if args.index:
        common.extend(["--index", args.index])
    return {
        "Python": [args.python, "-m", "infer.cli", *common],
        "Rust": [args.rust_binary, "--root", args.baseline, "infer", "--device", "cuda:0" if args.device == "cuda" else "cpu", *common],
    }


def run_trials(args, work, cases, environments, cpus, gpu):
    trials = []
    output_reference = {}
    for case, input_path, files in cases:
        for run in range(-args.warmups, args.runs):
            order = ["Python", "Rust"]
            if run % 2:
                order.reverse()
            for implementation in order:
                kind = "warmup" if run < 0 else "trial"
                number = run + (1 if run >= 0 else args.warmups + 1)
                print(f"{case}: {implementation} {kind} {number}", file=sys.stderr, flush=True)
                target = work / f"{case}-{run}-{implementation}"
                target.mkdir()
                out = target / "audio"
                log = target / "cli.log"
                command = workload_commands(args, input_path, out)[implementation]
                try:
                    result = command_usage(command, args.baseline, environments[implementation], cpus, log, args.timeout, gpu)
                except TimeoutError:
                    raise RuntimeError(f"{implementation} {case} exceeded {args.timeout:g} s") from None
                if result["returncode"]:
                    tail = log.read_text(errors="replace").splitlines()[-20:]
                    raise RuntimeError(f"{implementation} {case} failed with {result['returncode']}\n" + "\n".join(tail))
                metadata = {wav.name: wav_info(wav) for wav in sorted(out.glob("*.wav"))}
                expected_files = {file.with_suffix(".wav").name for file in files}
                if metadata.keys() != expected_files or any(info["frames"] == 0 for info in metadata.values()):
                    raise RuntimeError(f"{implementation} {case} produced an unexpected WAV set: {list(metadata)}")
                reference = output_reference.setdefault(case, metadata)
                if metadata != reference:
                    raise RuntimeError(f"WAV geometry differs between runs for {case}: {reference} vs {metadata}")
                if run >= 0:
                    trials.append({"case": case, "implementation": implementation, "run": number, **result, "outputs": metadata})
    return trials


def medians(trials, cases):
    rows = []
    for case, _, files in cases:
        duration = sum(info["frames"] / info["sample_rate"] for info in map(wav_info, files))
        for implementation in ("Python", "Rust"):
            results = [trial for trial in trials if trial["case"] == case and trial["implementation"] == implementation]
            fields = {}
            for key in (
                "elapsed_s", "cpu_total_s", "cpu_user_s", "cpu_system_s", "cpu_percent_one_core",
                "peak_rss_mib", "gpu_peak_vram_mib", "gpu_device_mean_percent",
            ):
                values = [result[key] for result in results if result.get(key) is not None]
                if values:
                    fields[key] = statistics.median(values)
            rows.append({
                "case": case,
                "implementation": implementation,
                "input_files": len(files),
                "audio_s": duration,
                "wall_s_per_audio_s": fields["elapsed_s"] / duration,
                **fields,
            })
    return rows


def preflight(args, environments, cpus):
    code = """
import json
import sys
import torch
from configs.config import infer_device, infer_dtype
print(json.dumps({
    "python": sys.version.split()[0],
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "torch_threads": torch.get_num_threads(),
    "device": str(infer_device),
    "precision": str(infer_dtype),
}))
"""
    runtime = subprocess.run(
        [args.python, "-c", code],
        cwd=args.baseline,
        env=environments["Python"],
        preexec_fn=lambda: os.sched_setaffinity(0, cpus),
        capture_output=True,
        text=True,
        timeout=args.timeout,
    )
    if runtime.returncode:
        raise RuntimeError(f"baseline environment check failed\n{runtime.stderr}")
    info = json.loads(runtime.stdout.splitlines()[-1])
    if info["device"].split(":")[0] != args.device:
        raise RuntimeError(f"original selected {info['device']}, expected {args.device}")
    if info["torch_threads"] != args.threads:
        raise RuntimeError(f"original selected {info['torch_threads']} Torch threads, expected {args.threads}")
    version = subprocess.check_output([args.rust_binary, "--version"], env=environments["Rust"], text=True, timeout=args.timeout).strip()
    libtorch_version = None
    if args.rust_lib_dir:
        info_file = args.rust_lib_dir.parent / "build-version"
        if info_file.is_file():
            libtorch_version = info_file.read_text().strip()
    return info, {"cli": version, "path": str(args.rust_binary), "sha256": sha256(args.rust_binary), "libtorch": libtorch_version}


def main():
    args = arguments()
    for field in ("baseline", "python", "rust_binary", "rust_lib_dir", "model", "inputs", "index", "json"):
        value = getattr(args, field)
        if value is not None:
            # A virtualenv's Python must be invoked through its symlink.
            setattr(args, field, value.expanduser().absolute())
    if not (args.baseline / "infer/cli.py").is_file() or not args.model.is_file():
        raise ValueError("the Python baseline or shared model is missing")
    cpus = sorted(os.sched_getaffinity(0))[:args.threads]
    if len(cpus) != args.threads:
        raise ValueError(f"only {len(cpus)} logical CPUs are available")
    sources = sorted(path for path in args.inputs.iterdir() if path.is_file() and path.suffix.lower() == ".wav")
    selected_cases = set(args.case or ("single", "batch"))
    minimum_files = 2 if "batch" in selected_cases else 1
    if len(sources) < minimum_files or len({path.stem for path in sources}) != len(sources):
        raise ValueError(f"provide at least {minimum_files} top-level PCM WAV file(s) with distinct stems")
    selected = args.single or sources[0].name
    if selected not in {path.name for path in sources}:
        raise ValueError(f"unknown single-file input {selected!r}")
    if args.index and not args.index.is_file():
        raise ValueError("the retrieval index is missing")
    cuda_selector = ""
    gpu = None
    gpu_description = None
    if args.device == "cuda":
        cuda_selector = str(args.gpu)
        smi = shutil.which("nvidia-smi")
        if args.gpu_stats and not smi:
            raise ValueError("--gpu-stats requires nvidia-smi")
        if smi:
            rows = gpu_query(smi, args.gpu, "name,uuid,memory.total,driver_version")
            if not rows:
                raise RuntimeError(f"nvidia-smi returned no information for GPU {args.gpu}")
            gpu_description = dict(zip(("name", "uuid", "memory_mib", "driver"), rows[0], strict=True))
            cuda_selector = gpu_description["uuid"]
        if args.gpu_stats:
            gpu = (smi, args.gpu, args.gpu_sample_interval)

    environments = command_environment(args, cuda_selector)
    python, rust = preflight(args, environments, cpus)
    with tempfile.TemporaryDirectory(prefix="rvc-benchmark-") as directory:
        work = Path(directory)
        inputs = work / "inputs"
        inputs.mkdir()
        staged = [Path(shutil.copy2(file, inputs / file.name)) for file in sources]
        cases = [("single", inputs / selected, [inputs / selected]), ("batch", inputs, staged)]
        cases = [case for case in cases if case[0] in selected_cases]
        trials = run_trials(args, work, cases, environments, cpus, gpu)
        summary = medians(trials, cases)
        input_metadata = {path.name: {**wav_info(path), "sha256": sha256(path)} for path in staged}
    if sha256(args.rust_binary) != rust["sha256"]:
        raise RuntimeError("the Rust executable changed during the benchmark")

    model_hashes = {"model": sha256(args.model)}
    if args.index:
        model_hashes["index"] = sha256(args.index)
    for relative in ("assets/hubert_base/config.json", "assets/hubert_base/pytorch_model.bin", "assets/hubert_base/preprocessor_config.json", "assets/rmvpe/rmvpe.pt"):
        if (args.baseline / relative).is_file():
            model_hashes[relative] = sha256(args.baseline / relative)
    report = {
        "host": {"cpu": first_cpu_model(), "system": platform.platform(), "cpu_affinity": cpus, "gpu": gpu_description},
        "settings": {
            "device": args.device, "threads": args.threads, "runs": args.runs, "warmups_per_case": args.warmups,
            "cases": sorted(selected_cases), "batch_files": len(sources), "single_file": selected, "f0_method": args.f0_method,
            "index_rate": args.index_rate if args.index else 0.0, "speaker_id": args.speaker_id,
            "gpu_sample_interval_s": args.gpu_sample_interval if gpu else None,
        },
        "baseline": str(args.baseline), "python": python, "rust": rust, "model_sha256": model_hashes,
        "inputs": input_metadata,
        "medians": summary, "trials": trials,
    }
    gpu_columns = (" | VRAM MiB | GPU % " if gpu else "")
    print("| Case | CLI | Files | Audio s | Wall s | CPU user s | CPU system s | RSS MiB " + gpu_columns + "|")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: " + ("| ---: | ---: " if gpu else "") + "|")
    for row in summary:
        gpu_values = [row.get("gpu_peak_vram_mib"), row.get("gpu_device_mean_percent")] if gpu else []
        gpu_text = "".join(f"| {value:.1f} " if value is not None else "| — " for value in gpu_values)
        print(f"| {row['case']} | {row['implementation']} | {row['input_files']} | {row['audio_s']:.3f} | "
              f"{row['elapsed_s']:.3f} | {row['cpu_user_s']:.3f} | {row['cpu_system_s']:.3f} | {row['peak_rss_mib']:.1f} "
              + gpu_text + "|")
    if args.json:
        args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    if not sys.platform.startswith("linux"):
        sys.exit("This benchmark needs Linux wait4 and CPU affinity.")
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError, wave.Error) as error:
        sys.exit(str(error))
