"""V4-vs-v3 learning-rate A/B on the RTX 4070 Ti ceiling profile.

Three trainers are compared on the same seed, same model size:

* ``baseline`` — v2 with all v3/v4 opt-in flags off.
* ``v3`` — v3 opt-in: phase_residual + qfi_fisher_weight + nb2_observed +
  count_calibration (the best v3 weights from the v3 sweep).
* ``v4`` — v3 + v4 opt-in: physical_fisher_weight +
  adaptive_count_calibration + phase_residual_consistency.

The model size is the 4070 Ti ceiling profile
``width=1024, layers=20, latent_tokens=128, batch_size=4`` (340M params,
6.65 GiB peak), running 1500 steps with mixed precision.

The output is a JSON summary with per-step trajectories and the
final-step deltas.  No hardware-write authority.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

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


# Config tokens for the three trainers.
TRAINER_CONFIG = {
    "baseline": {
        "use_phase_residual": False,
        "use_qfi_fisher_weight": False,
        "use_nb2_nll_observed": False,
        "use_count_calibration": False,
        "use_physical_fisher_weight": False,
        "use_adaptive_count_calibration": False,
        "use_phase_residual_consistency": False,
        "weights": {
            "nb2_nll": 1.0, "conditional_nll": 0.35,
            "teacher_cross_entropy": 0.15, "noise_type": 0.25,
            "noise_coarse_type": 2.0, "noise_strength": 0.5,
            "mixture": 0.2, "dispersion": 0.03,
            "retrieval": 0.1, "best_fidelity": 0.2,
            "nb2_nll_observed": 0.0, "count_calibration": 0.0,
            "adaptive_count_calibration": 0.0,
            "phase_residual_consistency": 0.0,
        },
    },
    "v3": {
        "use_phase_residual": True,
        "use_qfi_fisher_weight": True,
        "use_nb2_nll_observed": True,
        "use_count_calibration": True,
        "use_physical_fisher_weight": False,
        "use_adaptive_count_calibration": False,
        "use_phase_residual_consistency": False,
        "weights": {
            "nb2_nll": 1.0, "conditional_nll": 0.35,
            "teacher_cross_entropy": 0.15, "noise_type": 0.25,
            "noise_coarse_type": 2.0, "noise_strength": 0.5,
            "mixture": 0.2, "dispersion": 0.03,
            "retrieval": 0.1, "best_fidelity": 0.2,
            "nb2_nll_observed": 0.15, "count_calibration": 0.02,
            "adaptive_count_calibration": 0.0,
            "phase_residual_consistency": 0.0,
        },
    },
    "v4": {
        "use_phase_residual": True,
        "use_qfi_fisher_weight": True,
        "use_physical_fisher_weight": True,
        "use_nb2_nll_observed": True,
        "use_count_calibration": True,
        "use_adaptive_count_calibration": True,
        "use_phase_residual_consistency": True,
        "use_poisson_observed": False,
        "weights": {
            "nb2_nll": 1.0, "conditional_nll": 0.35,
            "teacher_cross_entropy": 0.15, "noise_type": 0.25,
            "noise_coarse_type": 2.0, "noise_strength": 0.5,
            "mixture": 0.2, "dispersion": 0.03,
            "retrieval": 0.1, "best_fidelity": 0.2,
            "nb2_nll_observed": 0.15, "count_calibration": 0.02,
            "adaptive_count_calibration": 0.02,
            "phase_residual_consistency": 0.02,
            "poisson_nll_observed": 0.0,
        },
    },
    "v6": {
        "use_phase_residual": True,
        "use_qfi_fisher_weight": True,
        "use_physical_fisher_weight": True,
        "use_nb2_nll_observed": True,
        "use_count_calibration": True,
        "use_adaptive_count_calibration": True,
        "use_phase_residual_consistency": True,
        "use_poisson_observed": True,
        "use_spectral_smoothness": False,
        "use_operation_equivariance": False,
        "weights": {
            "nb2_nll": 1.0, "conditional_nll": 0.35,
            "teacher_cross_entropy": 0.15, "noise_type": 0.25,
            "noise_coarse_type": 2.0, "noise_strength": 0.5,
            "mixture": 0.2, "dispersion": 0.03,
            "retrieval": 0.1, "best_fidelity": 0.2,
            "nb2_nll_observed": 0.15, "count_calibration": 0.02,
            "adaptive_count_calibration": 0.02,
            "phase_residual_consistency": 0.02,
            "poisson_nll_observed": 0.15,
            "spectral_smoothness": 0.0,
            "operation_equivariance": 0.0,
        },
    },
    "v7": {
        "use_phase_residual": True,
        "use_qfi_fisher_weight": True,
        "use_physical_fisher_weight": True,
        "use_nb2_nll_observed": True,
        "use_count_calibration": True,
        "use_adaptive_count_calibration": True,
        "use_phase_residual_consistency": True,
        "use_poisson_observed": False,
        "use_spectral_smoothness": True,
        "use_operation_equivariance": True,
        "weights": {
            "nb2_nll": 1.0, "conditional_nll": 0.35,
            "teacher_cross_entropy": 0.15, "noise_type": 0.25,
            "noise_coarse_type": 2.0, "noise_strength": 0.5,
            "mixture": 0.2, "dispersion": 0.03,
            "retrieval": 0.1, "best_fidelity": 0.2,
            "nb2_nll_observed": 0.15, "count_calibration": 0.02,
            "adaptive_count_calibration": 0.02,
            "phase_residual_consistency": 0.02,
            "poisson_nll_observed": 0.0,
            "spectral_smoothness": 0.05,
            "operation_equivariance": 0.05,
        },
    },
}


def _train_one(
    *, system_type: str, geometry, support, device,
    trainer_name: str, seed: int, learning_rate: float, weight_decay: float,
    optimizer_steps: int, batch_size: int,
    width: int, layers: int, latent_tokens: int,
) -> Dict[str, Any]:
    trainer = TRAINER_CONFIG[trainer_name]
    config_payload = {
        "width": int(width),
        "latent_tokens": int(latent_tokens),
        "layers": int(layers),
        "attention_heads": 8,
        "feedforward_multiplier": 3.5,
        "dropout": 0.0,
        "count_smoothing": 0.5,
        "use_activation_checkpointing": True,
        "use_phase_residual": trainer["use_phase_residual"],
        "use_qfi_fisher_weight": trainer["use_qfi_fisher_weight"],
        "use_physical_fisher_weight": trainer["use_physical_fisher_weight"],
        "use_nb2_nll_observed": trainer["use_nb2_nll_observed"],
        "use_count_calibration": trainer["use_count_calibration"],
        "use_adaptive_count_calibration": trainer["use_adaptive_count_calibration"],
        "use_phase_residual_consistency": trainer["use_phase_residual_consistency"],
        "qfi_fisher_weight_floor": 0.1,
        "physical_fisher_weight_floor": 0.1,
        "physical_fisher_weight_strength": 1.0,
    }
    config = V7ScalableModelV2Config.from_mapping(
        config_payload,
        system_type=system_type,
        operation_count=geometry.operation_count,
        qubit_count=geometry.qubit_count,
    )
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    model = V7ScalablePhysicsModelV2(config, support.graph_kernel)
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    model.train()
    weights = trainer["weights"]
    trajectory: List[Dict[str, float]] = []
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
        if step % 50 == 0 or step == int(optimizer_steps) - 1:
            trajectory.append({
                "step": step,
                "total": float(diagnostics["total"].detach().cpu()),
                "nb2_nll_observed": float(diagnostics["nb2_nll_observed"].detach().cpu()),
                "count_calibration": float(diagnostics["count_calibration"].detach().cpu()),
                "adaptive_count_calibration": float(diagnostics["adaptive_count_calibration"].detach().cpu()),
                "phase_residual_consistency": float(diagnostics["phase_residual_consistency"].detach().cpu()),
                "bestf_mae": float(diagnostics["bestf_mae"].detach().cpu()),
            })
        if device.type == "cuda":
            peak_reserved = max(
                peak_reserved, int(torch.cuda.max_memory_reserved(device)),
            )
    final = trajectory[-1]
    return {
        "trainer": trainer_name,
        "seed": int(seed),
        "peak_reserved_gib": peak_reserved / 1024 ** 3 if peak_reserved else 0.0,
        "final_total": final["total"],
        "final_nb2_nll_observed": final["nb2_nll_observed"],
        "final_count_calibration": final["count_calibration"],
        "final_adaptive_count_calibration": final["adaptive_count_calibration"],
        "final_phase_residual_consistency": final["phase_residual_consistency"],
        "final_bestf_mae": final["bestf_mae"],
        "trajectory": trajectory,
        "research_only": True,
        "promotion_ready": False,
        "hardware_action_authorized": False,
    }


def run_v4_ab(
    *,
    output_path: str | Path,
    system_type: str = "4qubit",
    optimizer_steps: int = 1500,
    batch_size: int = 4,
    width: int = 1024,
    layers: int = 20,
    latent_tokens: int = 128,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.01,
    seed: int = 20260807,
) -> Dict[str, Any]:
    if torch is None:
        raise RuntimeError("v4_ab requires torch")
    geometry = build_operation_frequency_geometry(system_type)
    retrieval_rates, _ = load_retrieval_rate_vectors(
        system_type, geometry.operation_count,
    )
    support = build_synthetic_physics_support(
        geometry, retrieval_rates, graph_mode_count=12, graph_length_mhz=8.0,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = (
        torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    )
    runs = []
    for trainer_name in ("baseline", "v3", "v4", "v6", "v7"):
        run = _train_one(
            system_type=system_type, geometry=geometry, support=support,
            device=device, trainer_name=trainer_name, seed=seed,
            learning_rate=learning_rate, weight_decay=weight_decay,
            optimizer_steps=optimizer_steps, batch_size=batch_size,
            width=width, layers=layers, latent_tokens=latent_tokens,
        )
        runs.append(run)
    baseline, v3, v4, v6, v7 = runs
    summary = {
        "schema": "v7_v4_ab_v1",
        "system_type": system_type,
        "device": str(device),
        "device_name": device_name,
        "model_size": {
            "width": int(width),
            "layers": int(layers),
            "latent_tokens": int(latent_tokens),
            "batch_size": int(batch_size),
        },
        "optimizer_steps": int(optimizer_steps),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "research_only": True,
        "promotion_ready": False,
        "hardware_action_authorized": False,
        "baseline": baseline,
        "v3": v3,
        "v4": v4,
        "v6": v6,
        "v7": v7,
        "improvement_v3": {
            "final_total_delta": baseline["final_total"] - v3["final_total"],
            "final_nb2_nll_observed_delta":
                baseline["final_nb2_nll_observed"] - v3["final_nb2_nll_observed"],
            "final_bestf_mae_delta":
                baseline["final_bestf_mae"] - v3["final_bestf_mae"],
        },
        "improvement_v4": {
            "final_total_delta": baseline["final_total"] - v4["final_total"],
            "final_nb2_nll_observed_delta":
                baseline["final_nb2_nll_observed"] - v4["final_nb2_nll_observed"],
            "final_bestf_mae_delta":
                baseline["final_bestf_mae"] - v4["final_bestf_mae"],
        },
        "improvement_v6": {
            "final_total_delta": baseline["final_total"] - v6["final_total"],
            "final_nb2_nll_observed_delta":
                baseline["final_nb2_nll_observed"] - v6["final_nb2_nll_observed"],
            "final_bestf_mae_delta":
                baseline["final_bestf_mae"] - v6["final_bestf_mae"],
        },
        "improvement_v7": {
            "final_total_delta": baseline["final_total"] - v7["final_total"],
            "final_nb2_nll_observed_delta":
                baseline["final_nb2_nll_observed"] - v7["final_nb2_nll_observed"],
            "final_bestf_mae_delta":
                baseline["final_bestf_mae"] - v7["final_bestf_mae"],
        },
        "v4_vs_v3": {
            "final_total_delta": v3["final_total"] - v4["final_total"],
            "final_nb2_nll_observed_delta":
                v3["final_nb2_nll_observed"] - v4["final_nb2_nll_observed"],
            "final_bestf_mae_delta":
                v3["final_bestf_mae"] - v4["final_bestf_mae"],
        },
        "v6_vs_v4": {
            "final_total_delta": v4["final_total"] - v6["final_total"],
            "final_nb2_nll_observed_delta":
                v4["final_nb2_nll_observed"] - v6["final_nb2_nll_observed"],
            "final_bestf_mae_delta":
                v4["final_bestf_mae"] - v6["final_bestf_mae"],
        },
        "v7_vs_v4": {
            "final_total_delta": v4["final_total"] - v7["final_total"],
            "final_nb2_nll_observed_delta":
                v4["final_nb2_nll_observed"] - v7["final_nb2_nll_observed"],
            "final_bestf_mae_delta":
                v4["final_bestf_mae"] - v7["final_bestf_mae"],
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
        default="ml_recon/reports/physics_hybrid_v7/v4_ab_4q_ceiling.json",
    )
    parser.add_argument("--system", default="4qubit", choices=["3qubit", "4qubit"])
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--layers", type=int, default=20)
    parser.add_argument("--latent-tokens", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=20260807)
    args = parser.parse_args(argv)
    summary = run_v4_ab(
        output_path=args.output, system_type=args.system,
        optimizer_steps=args.steps, batch_size=args.batch_size,
        width=args.width, layers=args.layers, latent_tokens=args.latent_tokens,
        learning_rate=args.learning_rate, seed=args.seed,
    )
    print(json.dumps({
        "schema": summary["schema"],
        "system_type": summary["system_type"],
        "improvement_v3": summary["improvement_v3"],
        "improvement_v4": summary["improvement_v4"],
        "improvement_v6": summary["improvement_v6"],
        "improvement_v7": summary["improvement_v7"],
        "v4_vs_v3": summary["v4_vs_v3"],
        "v6_vs_v4": summary["v6_vs_v4"],
        "v7_vs_v4": summary["v7_vs_v4"],
    }, indent=2))
    return 0


__all__ = ["run_v4_ab", "TRAINER_CONFIG"]


if __name__ == "__main__":
    raise SystemExit(main())