"""Compare resource use for the original and native dataset/training stages."""

import argparse
import json
import os
from pathlib import Path
import shutil
import statistics
import tempfile

from compare import THREAD_ENV, command_usage, gpu_query


def copy_inputs(source, target, pattern):
    target.mkdir(parents=True, exist_ok=True)
    for path in source.glob(pattern):
        if path.is_file():
            shutil.copy2(path, target / path.name)


def fixtures(root, baseline, source, case):
    assets = root / "assets"
    assets.mkdir(parents=True)
    for name in ("hubert_base", "rmvpe"):
        (assets / name).symlink_to(baseline / "assets" / name, target_is_directory=True)
    (root / "configs").symlink_to(baseline / "configs", target_is_directory=True)
    (root / "i18n").symlink_to(baseline / "i18n", target_is_directory=True)
    experiment = root / "logs" / "case"
    experiment.mkdir(parents=True)
    if case == "extract":
        copy_inputs(source / "1_16k_wavs", experiment / "1_16k_wavs", "*.wav")
    if case in ("index", "train"):
        copy_inputs(source / "3_feature768", experiment / "3_feature768", "*.npy")
    if case == "train":
        (assets / "weights").mkdir()
        shutil.copy2(source / "config.json", experiment / "config.json")
        copy_inputs(source / "0_gt_wavs", experiment / "0_gt_wavs", "*.wav")
        mute = root / "logs" / "mute"
        original_mute = baseline / "logs" / "mute"
        copy_inputs(original_mute / "0_gt_wavs", mute / "0_gt_wavs", "*.wav")
        copy_inputs(original_mute / "3_feature768", mute / "3_feature768", "*.npy")
        rows = []
        for row in (source / "filelist.txt").read_text().splitlines():
            wav, feature, speaker = row.strip().split("|")
            location = mute if Path(wav).parent.parent.name == "mute" else experiment
            rows.append("|".join((
                str(location / "0_gt_wavs" / Path(wav).name),
                str(location / "3_feature768" / Path(feature).name), speaker,
            )))
        (experiment / "filelist.txt").write_text("\n".join(rows) + "\n")
    return experiment


def command(args, case, language, root, experiment, device):
    inputs = str(args.inputs.absolute())
    n = str(args.threads)
    if language == "Python":
        executable = [str(args.python.absolute()), "-m"]
        suffix = {
            "preprocess": ["train.preprocess", inputs, "32000", n, str(experiment), "False", "3.0"],
            "extract": ["train.dataset.extract_hubert_feature", device, "1", "0", str(experiment), "v2",
                        "true" if device == "cuda" else "false"],
            "index": ["train.train_index", "case", "v2", str(root / "assets/indices"), n, "single"],
            "train": ["train.train", "-e", "case", "-te", "1", "-se", "1", "-bs", "1",
                      "-sr", "32k", "-f0", "0", "-v", "v2", "-g", str(args.gpu) if device == "cuda" else "",
                      "-l", "0", "-c", "0", "-sw", "0"],
        }[case]
    else:
        executable = [str(args.rust_binary.absolute()), "--root", str(root)]
        suffix = {
            "preprocess": ["preprocess", "--input", inputs, "--experiment", str(experiment),
                           "--sample-rate", "32000", "--workers", n, "--segment-seconds", "3"],
            "extract": ["extract", "--experiment", str(experiment), "--version", "v2", "--device", device],
            "index": ["train-index", "--experiment", str(experiment), "--version", "v2", "--workers", n, "--mode", "single"],
            "train": ["train", "--experiment", str(experiment), "--epochs", "1", "--save-every", "1",
                      "--batch-size", "1", "--sample-rate", "32000", "--device", device, "--f0", "false", "--random-weights"],
        }[case]
    return executable + suffix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--rust-binary", type=Path, required=True)
    parser.add_argument("--rust-lib-dir", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--training-fixture", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--case", choices=("preprocess", "extract", "index", "train"), action="append")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu",
                        help="Device for HuBERT feature extraction and training.")
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device ordinal in PCI bus order.")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.threads < 1 or args.runs < 1 or args.gpu < 0:
        parser.error("threads and runs must be positive and the GPU ordinal must be nonnegative")
    baseline = args.baseline.absolute()
    cases = args.case or ["preprocess", "extract", "index", "train"]
    gpu = None
    if args.device == "cuda" and any(case in ("extract", "train") for case in cases):
        smi = shutil.which("nvidia-smi")
        if smi is None:
            parser.error("nvidia-smi is required to measure CUDA resource use")
        gpus = sorted(gpu_query(smi, None, "pci.bus_id,name"))
        if args.gpu >= len(gpus):
            parser.error(f"GPU ordinal {args.gpu} is unavailable")
        bus, gpu_name = gpus[args.gpu]
        gpu = (smi, bus, 0.2)
    common = dict(os.environ)
    common.update({key: str(args.threads) for key in THREAD_ENV})
    common.update({"CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                   "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(baseline)})
    raw = {}
    with tempfile.TemporaryDirectory(prefix="rvc-pipeline-bench-") as directory:
        for case in cases:
            case_gpu = gpu if case in ("extract", "train") else None
            device = "cuda" if case_gpu else "cpu"
            raw[case] = {"Python": [], "Rust": []}
            for i in range(args.runs):
                order = ("Python", "Rust") if i % 2 == 0 else ("Rust", "Python")
                for language in order:
                    root = Path(directory) / f"{case}-{language}-{i + 1}"
                    experiment = fixtures(root, baseline, args.training_fixture.absolute(), case)
                    env = common | {
                        "weight_root": str(root / "assets" / "weights"),
                        "CUDA_VISIBLE_DEVICES": str(args.gpu) if case_gpu else "-1",
                        "RVC_AUDIO_FORCE_CPU": "0" if case_gpu else "1",
                    }
                    if language == "Rust":
                        env["LD_LIBRARY_PATH"] = str(args.rust_lib_dir.absolute())
                    log = root / "trial.log"
                    argv = command(args, case, language, root, experiment, device)
                    print(f"{case}: {language} trial {i + 1}", flush=True)
                    result = command_usage(argv, root, env, os.sched_getaffinity(0), log, args.timeout, case_gpu)
                    if result["returncode"]:
                        raise RuntimeError(f"{case} {language}: {result['returncode']}\n{log.read_text()[-6000:]}")
                    if case_gpu:
                        if result["gpu_sampling_error"] or result["gpu_peak_vram_mib"] is None:
                            raise RuntimeError(f"{case} {language}: no valid VRAM samples ({result['gpu_sampling_error']})")
                        result.update(gpu_name=gpu_name, gpu_pci_bus=bus)
                    patterns = {"preprocess": ("1_16k_wavs/*.wav",), "extract": ("3_feature768/*.npy",),
                                "index": ("added_*.index",), "train": ("G_*.pth", "D_*.pth")}[case]
                    outputs = []
                    for pattern in patterns:
                        names = sorted(path.name for path in experiment.glob(pattern) if path.stat().st_size > 0)
                        if not names:
                            raise RuntimeError(f"{case} {language}: no outputs for {pattern}\n{log.read_text()[-4000:]}")
                        outputs.extend(names)
                    if case == "train":
                        model = root / "assets" / "weights" / "case.pth"
                        if not model.is_file() or model.stat().st_size == 0:
                            raise RuntimeError(f"{case} {language}: no exported model\n{log.read_text()[-4000:]}")
                        outputs.append(model.name)
                    raw[case][language].append({**result, "outputs": outputs})
                python = raw[case]["Python"][-1]["outputs"]
                rust = raw[case]["Rust"][-1]["outputs"]
                if len(python) != len(rust) or (case in ("extract", "train") and python != rust):
                    raise RuntimeError(f"{case}: implementations produced different outputs")
    print("| Stage | CLI | Wall s | CPU s | RSS MiB |" + (" VRAM MiB |" if gpu else ""))
    print("| --- | --- | ---: | ---: | ---: |" + (" ---: |" if gpu else ""))
    for case, timings in raw.items():
        for language, runs in timings.items():
            def median(key):
                return statistics.median(run[key] for run in runs)
            vram = ""
            if gpu:
                vram = f" {median('gpu_peak_vram_mib'):.0f} |" if "gpu_peak_vram_mib" in runs[0] else " — |"
            print(f"| {case} | {language} | {median('elapsed_s'):.3f} | "
                  f"{median('cpu_total_s'):.3f} | {median('peak_rss_mib'):.1f} |{vram}")
    if args.json:
        args.json.write_text(json.dumps(raw, indent=2) + "\n")


if __name__ == "__main__":
    main()
