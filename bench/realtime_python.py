"""Run the original realtime model worker on the same input as realtime_benchmark."""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--blocks", type=int, default=20)
    parser.add_argument("--warmups", type=int, default=3)
    args = parser.parse_args()
    sys.path.insert(0, str(args.root.resolve()))
    sys.argv[:] = sys.argv[:1]

    from configs.config import Config
    from infer.rtrvc import RVC

    config = Config()
    if str(config.device) != args.device:
        raise RuntimeError(f"expected {args.device}, original selected {config.device}")
    engine = RVC(0, 0.0, str(args.root / "assets/weights" / args.model), "", 0.0, config)
    device = torch.device(args.device)
    source = np.fromfile(args.input, dtype="<f4")
    hop, block, context, crossfade = 160, 25, 250, 5
    render = block + 4 + 1
    count = block * hop
    window = torch.zeros((context + crossfade + 1 + block) * hop,
                         dtype=torch.float32, device=device)
    positions = np.arange(count)
    times = []
    peak = 0.0
    with torch.no_grad():
        for i in range(args.warmups + args.blocks):
            window[:-count] = window[count:].clone()
            window[-count:] = torch.from_numpy(source[(i * count + positions) % len(source)]).to(device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter_ns()
            output = engine.infer(window, count, context, render, "rmvpe").cpu()
            elapsed = (time.perf_counter_ns() - start) / 1e6
            if output.numel() != render * engine.tgt_sr // 100 or not output.isfinite().all():
                raise RuntimeError("invalid realtime output")
            if i >= args.warmups:
                times.append(elapsed)
                peak = max(peak, float(output.abs().max()))
    print(json.dumps({
        "device": str(device), "block_ms": block * 10, "times_ms": times,
        "samples_per_output": render * engine.tgt_sr // 100, "peak": peak,
    }))


if __name__ == "__main__":
    main()
