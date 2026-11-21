"""Check the public extractor's RMCPE cache and its all-unvoiced training arrays."""
import argparse
import json
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--rvc", required=True)
parser.add_argument("--root", required=True)
parser.add_argument("--device", default="cuda:0")
args = parser.parse_args()
with tempfile.TemporaryDirectory() as temp:
    experiment = Path(temp)
    wav = experiment / "1_16k_wavs" / "silent.wav"
    wav.parent.mkdir()
    with wave.open(str(wav), "wb") as f:
        f.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        f.writeframes(bytes(32000))
    base = [args.rvc, "--root", args.root, "extract", "--experiment", str(experiment),
            "--embedder", "mhubert147", "--device", args.device]
    def run(method):
        result = subprocess.run(base + ["--f0-method", method], check=True, stdout=subprocess.PIPE, text=True)
        return json.loads(result.stdout)
    assert run("rmvpe")["unvoiced"] == 1
    changed = run("rmcpe")
    assert changed["features"] == 0 and changed["unvoiced"] == 1 and changed["cached"] == 0
    for folder, expected in [("2a_f0", 1), ("2b-f0nsf", 0), ("2d-f0raw", 0)]:
        arr = np.load(experiment / folder / "silent.wav.npy")
        assert arr.shape == (101,) and np.all(arr == expected), (folder, arr)
    assert run("rmcpe")["cached"] == 1
    (experiment / "2d-f0raw" / "silent.wav.npy").unlink()
    assert run("rmcpe")["unvoiced"] == 1
    assert not (experiment / "2d-f0raw" / ".incomplete").exists()
    print("Method change, all-unvoiced arrays, cache hit and interrupted-cache repair passed.")
