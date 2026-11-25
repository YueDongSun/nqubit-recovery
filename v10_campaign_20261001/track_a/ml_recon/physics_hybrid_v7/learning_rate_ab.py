"""Learning-rate A/B comparison for V7 v2 physics-math heads.

Runs two matched trainers on the same synthetic 3Q holdout:

* ``baseline`` — v2 with all v3 opt-in flags off.  Identical to v2.
* ``v3`` — v2 with ``use_phase_residual=True``,
  ``use_qfi_fisher_weight=True``, ``nb2_observed_weight=0.35``, and
  ``count_calibration_weight=0.05`` on top of the existing loss weights.

For each trainer the script records the total loss / NB2 NLL / bestF MAE
after every checkpoint step and writes a JSON summary.  The headline
metrics are:

* ``steps_to_first_under_8`` — first step at which the total loss drops
  below 8.0 (a proxy for convergence speed on this holdout).
* ``final_total`` — total loss at the last step.
* ``final_nb2_nll_observed`` — observed-NB2 NLL at the last step.
* ``final_count_calibration`` — count-calibration loss at the last step.
* ``final_bestf_mae`` — bestF MAE at the last step.

Research-only; no production promotion implied.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from .geometry import build_operation_frequency_geometry
from .noise_taxonomy import (
    build_synthetic_physics_support,
    generate_noise_taxonomy_batch,
    numpy_batch_to_torch,
)
from .physical_ceiling import load_retrieval_rate_vectors
from .scalable_model import v7_scalable_multitask_loss
from .scalable_model_v2 import (
    V7ScalableModelV2Config,
    V7ScalablePhysicsModelV2,
)


DEFAULT_LOSS_WEIGHTS = {
    "nb2_nll": 1.0,
    "conditional_nll": 0.35,
    "teacher_cross_entropy": 0.15,
    "noise_type": 0.25,
    "noise_coarse_type": 2.0,
    "noise_strength": 0.5,
    "mixture": 0.2,
    "dispersion": 0.03,
    "retrieval": 0.1,
    "best_fidelity": 0.2,
    "nb2_nll_observed": 0.0,
    "count_calibration": 0.0,
}

V3_LOSS_WEIGHTS = dict(DEFAULT_LOSS_WEIGHTS)
V3_LOSS_WEIGHTS.update({
    "nb2_nll_observed": 0.15,
    "count_calibration": 0.02,
})


def _build_model_and_optimizer(
    *, system_type: str, geometry, graph_kernel, use_v3: bool,
    seed: int, learning_rate: float, weight_decay: float,
    batch_size: int,
):
    config_payload = {
        "width": 192, "latent_tokens": 32, "layers": 6, "attention_heads": 8,
        "feedforward_multiplier": 3.0, "dropout": 0.0, "count_smoothing": 0.5,
        "use_activation_checkpointing": False,
        "use_phase_residual": use_v3,
        "use_qfi_fisher_weight": use_v3,
    }
    config = V7ScalableModelV2Config.from_mapping(
        config_payload,
        system_type=system_type,
        operation_count=geometry.operation_count,
        qubit_count=geometry.qubit_count,
    )
    torch.manual_seed(int(seed))
    model = V7ScalablePhysicsModelV2(config, graph_kernel)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
    return model, optimizer, scaler, batch_size


def _train_one(
    *, system_type: str, geometry, support, device: "torch.device",
    use_v3: bool, seed: int, learning_rate: float, weight_decay: float,
    optimizer_steps: int, batch_size: int, loss_threshold: float,
) -> Dict[str, Any]:
    model, optimizer, scaler, batch_size = _build_model_and_optimizer(
        system_type=system_type,
        geometry=geometry,
        graph_kernel=support.graph_kernel,
        use_v3=use_v3,
        seed=seed,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        batch_size=batch_size,
    )
    model = model.to(device)
    model.train()
    weights = V3_LOSS_WEIGHTS if use_v3 else DEFAULT_LOSS_WEIGHTS
    trajectory: List[Dict[str, float]] = []
    steps_to_threshold: int | None = None
    peak_reserved = 0
    for step in range(int(optimizer_steps)):
        batch_numpy = generate_noise_taxonomy_batch(
            geometry, support,
            batch_size=int(batch_size), seed=int(seed) + step,
            low_count_fraction=0.5,
        )
        batch = numpy_batch_to_torch(batch_numpy, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(
            device_type=device.type,
            dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            output = model(
                batch["operation_inputs"], batch["target_context"],
                batch["signal_counts"], batch["idler_counts"],
            )
            loss, diagnostics = v7_scalable_multitask_loss(
                output, batch, weights,
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        snapshot = {
            "step": step,
            "total": float(diagnostics["total"].detach().cpu()),
            "nb2_nll": float(diagnostics["nb2_nll"].detach().cpu()),
            "nb2_nll_observed": float(diagnostics["nb2_nll_observed"].detach().cpu()),
            "count_calibration": float(diagnostics["count_calibration"].detach().cpu()),
            "bestf_mae": float(diagnostics["bestf_mae"].detach().cpu()),
        }
        trajectory.append(snapshot)
        if steps_to_threshold is None and snapshot["total"] < float(loss_threshold):
            steps_to_threshold = step
        if device.type == "cuda":
            peak_reserved = max(
                peak_reserved, int(torch.cuda.max_memory_reserved(device)),
            )
    final = trajectory[-1]
    return {
        "use_v3": bool(use_v3),
        "seed": int(seed),
        "steps_to_first_under_threshold": steps_to_threshold,
        "final_total": final["total"],
        "final_nb2_nll": final["nb2_nll"],
        "final_nb2_nll_observed": final["nb2_nll_observed"],
        "final_count_calibration": final["count_calibration"],
        "final_bestf_mae": final["bestf_mae"],
        "peak_reserved_gib": peak_reserved / 1024 ** 3 if peak_reserved else 0.0,
        "trajectory": trajectory,
        "research_only": True,
        "promotion_ready": False,
        "hardware_action_authorized": False,
    }


def run_learning_rate_ab(
    *,
    output_path: str | Path,
    system_type: str = "3qubit",
    optimizer_steps: int = 200,
    batch_size: int = 8,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.01,
    loss_threshold: float = 8.0,
    seed: int = 20260807,
) -> Dict[str, Any]:
    if torch is None:
        raise RuntimeError("learning_rate_ab requires torch")
    geometry = build_operation_frequency_geometry(system_type)
    retrieval_rates, _ = load_retrieval_rate_vectors(
        system_type, geometry.operation_count,
    )
    support = build_synthetic_physics_support(
        geometry, retrieval_rates, graph_mode_count=8, graph_length_mhz=8.0,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = (
        torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    )
    runs: List[Dict[str, Any]] = []
    for use_v3 in (False, True):
        runs.append(_train_one(
            system_type=system_type, geometry=geometry, support=support,
            device=device, use_v3=use_v3, seed=seed,
            learning_rate=learning_rate, weight_decay=weight_decay,
            optimizer_steps=optimizer_steps, batch_size=batch_size,
            loss_threshold=loss_threshold,
        ))
    baseline, v3 = runs
    summary = {
        "schema": "v7_learning_rate_ab_v1",
        "system_type": system_type,
        "device": str(device),
        "device_name": device_name,
        "optimizer_steps": int(optimizer_steps),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "loss_threshold": float(loss_threshold),
        "research_only": True,
        "promotion_ready": False,
        "hardware_action_authorized": False,
        "baseline": baseline,
        "v3": v3,
        "improvement": {
            "steps_to_threshold_delta": (
                (baseline["steps_to_first_under_threshold"] or optimizer_steps)
                - (v3["steps_to_first_under_threshold"] or optimizer_steps)
            ),
            "final_total_delta": baseline["final_total"] - v3["final_total"],
            "final_nb2_nll_observed_delta":
                baseline["final_nb2_nll_observed"] - v3["final_nb2_nll_observed"],
            "final_count_calibration_delta":
                baseline["final_count_calibration"] - v3["final_count_calibration"],
            "final_bestf_mae_delta":
                baseline["final_bestf_mae"] - v3["final_bestf_mae"],
        },
    }
    Path(output_path).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="ml_recon/reports/physics_hybrid_v7/learning_rate_ab_3q.json",
    )
    parser.add_argument("--system", default="3qubit", choices=["3qubit", "4qubit"])
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--threshold", type=float, default=8.0)
    args = parser.parse_args(argv)
    summary = run_learning_rate_ab(
        output_path=args.output, system_type=args.system,
        optimizer_steps=args.steps, batch_size=args.batch_size,
        learning_rate=args.learning_rate, loss_threshold=args.threshold,
    )
    print(json.dumps({
        "schema": summary["schema"],
        "system_type": summary["system_type"],
        "improvement": summary["improvement"],
        "baseline_final": {
            "total": summary["baseline"]["final_total"],
            "bestf_mae": summary["baseline"]["final_bestf_mae"],
        },
        "v3_final": {
            "total": summary["v3"]["final_total"],
            "bestf_mae": summary["v3"]["final_bestf_mae"],
        },
    }, indent=2))
    return 0


__all__ = ["run_learning_rate_ab", "DEFAULT_LOSS_WEIGHTS", "V3_LOSS_WEIGHTS"]


if __name__ == "__main__":
    raise SystemExit(main())