"""Replay one exported classifier on the independent large-model comparison panel."""

import argparse
from pathlib import Path

import torch

from ml_recon.physics_hybrid_v10.classifier_n_full import V10ClassifierNFull
from ml_recon.physics_hybrid_v10.effective_inference_experiment import write_json
from training.large_scale_evaluate import score_model_at_step
from training.large_scale_train import CAPACITIES, register_capacity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True,
                        help="Score root. Files are grouped by run and training step.")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    core = torch.load(args.checkpoint, weights_only=True, map_location="cpu")
    identity, step = core["identity"], core["updates"]
    config = core["classifier_config"]
    if config["profile"] in CAPACITIES:
        register_capacity(config["profile"])
    model = V10ClassifierNFull(**config).eval().to(args.device)
    model.load_state_dict(core["classifier_state_dict"], strict=True)
    folder = args.output / identity["lane"] / "scores" / f"step_{step:09d}"
    result = score_model_at_step(model, step=step, identity=identity, dev_banks=None, seen=None,
                                 device=torch.device(args.device), output_dir=folder)
    write_json(folder / "evaluation_done.json", dict(step=step, identity=identity, result=result))


if __name__ == "__main__":
    main()
