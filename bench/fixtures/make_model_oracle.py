"""Generate deterministic RoFormer checkpoint tensors from the project source."""

import argparse
from pathlib import Path
import sys

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("project", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.project.resolve() / "tools"))
    from pymss_core.modules.bs_roformer import BSRoformer, MelBandRoformer

    torch.set_num_threads(2)
    torch.manual_seed(1729)
    common = {
        "dim": 8,
        "depth": 2,
        "stereo": True,
        "num_stems": 2,
        "time_transformer_depth": 2,
        "freq_transformer_depth": 1,
        "heads": 2,
        "dim_head": 4,
        "stft_n_fft": 64,
        "stft_hop_length": 8,
        "stft_win_length": 64,
        "mask_estimator_depth": 2,
    }
    models = {
        "bs": BSRoformer(
            freqs_per_bands=(4, 4, 8, 17),
            use_shared_bias=True,
            **(common | {"stft_win_length": 48}),
        ),
        "mel": MelBandRoformer(
            num_bands=4,
            sample_rate=16_000,
            zero_dc=True,
            mlp_hidden_layers=2,
            stft_normalized=True,
            **common,
        ),
    }
    mix = (torch.arange(350, dtype=torch.float32).reshape(1, 2, 175) % 19 - 8) / 32
    args.output.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        model.eval()
        # Existing WebUI checkpoints use half precision on disk. Both engines
        # convert the saved values to float32 for the CPU comparison.
        state_dict = model.half().state_dict()
        torch.save(state_dict, args.output / f"{name}.ckpt")
        model.float()
        with torch.inference_mode():
            predicted = model(mix)
        (args.output / f"{name}-prediction.f32").write_bytes(
            predicted.numpy().astype("<f4", copy=False).tobytes()
        )


if __name__ == "__main__":
    main()
