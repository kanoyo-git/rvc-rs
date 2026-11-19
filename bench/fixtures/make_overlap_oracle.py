"""Record selected PyMSS windowing functions against a deterministic test model."""

import argparse
import ast
from pathlib import Path
import struct

import numpy as np
import torch
from torch import nn


FUNCTIONS = {
    "_getWindowingArray",
    "_build_chunk_plan",
    "_prepare_mix_for_chunks",
    "_extract_chunk",
    "_add_weighted_chunk",
    "_finalize_overlap",
}


def get_oracle(path):
    source = ast.parse(path.read_text(encoding="utf8"))
    namespace = {"torch": torch, "nn": nn, "np": np}
    nodes = [node for node in source.body if isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS]
    if len(nodes) != len(FUNCTIONS):
        raise ValueError("original PyMSS functions differ from expected version")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def record(oracle, count, chunk_size, overlap):
    mix = ((torch.arange(2 * count, dtype=torch.float32) % 57) - 23).reshape(2, count) / 64.0
    padded, size = oracle["_prepare_mix_for_chunks"](mix, overlap)
    result = torch.zeros((2, *padded.shape))
    counter = torch.zeros((1, 1, padded.shape[1]))
    starts, windows = oracle["_build_chunk_plan"](
        padded.shape[1], chunk_size, chunk_size - overlap, min(chunk_size // 10, overlap)
    )
    indices = torch.arange(chunk_size) % 17
    for start, window in zip(starts, windows):
        part, length = oracle["_extract_chunk"](padded, start, chunk_size)
        if overlap:
            estimate = torch.stack(
                (part * 0.5 + indices / 128.0, part.flip(-1) * -0.75), dim=0
            )
        else:
            estimate = torch.stack((part * 0.5, part * -0.75), dim=0)
        oracle["_add_weighted_chunk"](result, counter, estimate, window, start, length)
    out = oracle["_finalize_overlap"](result, counter, size, overlap)
    return out.astype("<f4", copy=False).tobytes()


def main():
    arguments = argparse.ArgumentParser()
    arguments.add_argument("original_utils", type=Path)
    arguments.add_argument("output", type=Path)
    args = arguments.parse_args()
    oracle = get_oracle(args.original_utils)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cases = [
        (203, 40, 27),
        (129, 42, 11),
        (3, 44, 4),
        (15, 20, 9),
        (95, 20, 1),
        (81, 40, 0),
    ]
    with args.output.open("wb") as output:
        for count, chunk_size, overlap in cases:
            output.write(struct.pack("<III", count, chunk_size, overlap))
            output.write(record(oracle, count, chunk_size, overlap))


if __name__ == "__main__":
    main()
