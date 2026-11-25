"""V4 A/B on the V7 measured-snapshot pipeline.

Three trainers (baseline, v3, v4) are trained against the same
measured 4Q dataset, using the same RealMaskedLoss shape used by
``measured_scalable_smoke``.  The output is a JSON summary that
records the per-step train loss on the real masked batch and the
held-out masked validation metrics.

The script is research-only and never writes to ``matrix/ml_delta``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from .geometry import build_operation_frequency_geometry
from .measured_scalable_smoke import (
    MeasuredScalableArraysV1,
    _measured_batch,
    _real_masked_loss,
    build_measured_scalable_arrays,
)
from .noise_taxonomy import build_synthetic_physics_support
from .physical_ceiling import load_retrieval_rate_vectors
from .scalable_model_v2 import (
    V7ScalableModelV2Config,
    V7ScalablePhysicsModelV2,
)


TRAINER_CONFIG = {
    "baseline": {
        "use_phase_residual": False,
        "use_qfi_fisher_weight": False,
        "use_nb2_nll_observed": False,
        "use_count_calibration": False,
        "use_physical_fisher_weight": False,
        "use_adaptive_count_calibration": False,
        "use_phase_residual_consistency": False,
        "use_poisson_observed": False,
    },
    "v3": {
        "use_phase_residual": True,
        "use_qfi_fisher_weight": True,
        "use_nb2_nll_observed": True,
        "use_count_calibration": True,
        "use_physical_fisher_weight": False,
        "use_adaptive_count_calibration": False,
        "use_phase_residual_consistency": False,
        "use_poisson_observed": False,
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
        "use_count_scale_equivariance": False,
    },
    "v8": {
        "use_phase_residual": True,
        "use_qfi_fisher_weight": True,
        "use_physical_fisher_weight": True,
        "use_nb2_nll_observed": True,
        "use_count_calibration": True,
        "use_adaptive_count_calibration": True,
        "use_phase_residual_consistency": True,
        "use_poisson_observed": False,
        "use_spectral_smoothness": False,
        "use_operation_equivariance": False,
        "use_count_scale_equivariance": True,
        "count_scale_equivariance_factor": 2.0,
        "count_scale_equivariance_weight": 0.05,
        "use_adaptive_loss_weighting": False,
    },
    "v9": {
        "use_phase_residual": True,
        "use_qfi_fisher_weight": True,
        "use_physical_fisher_weight": True,
        "use_nb2_nll_observed": True,
        "use_count_calibration": True,
        "use_adaptive_count_calibration": True,
        "use_phase_residual_consistency": True,
        "use_poisson_observed": False,
        "use_spectral_smoothness": False,
        "use_operation_equivariance": False,
        "use_count_scale_equivariance": False,
        "use_adaptive_loss_weighting": True,
        "adaptive_loss_weighting_floor": 0.25,
    },
}


def _train_one(
    *, model: "V7ScalablePhysicsModelV2",
    arrays: MeasuredScalableArraysV1,
    support: Any,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    optimizer: "torch.optim.Optimizer",
    steps: int, batch_size: int, seed: int,
    device: "torch.device",
    extra_weights: "Dict[str, float] | None" = None,
) -> Dict[str, Any]:
    model.train()
    trajectory: List[Dict[str, float]] = []
    # Extra loss components from v3-v8 that the plain _real_masked_loss
    # does not consume.  When empty, the total is exactly the measured
    # nb2 + conditional + retrieval loss (byte-identical to the v5
    # measured A/B baseline).
    extra_weights = extra_weights or {}
    for step in range(int(steps)):
        if train_indices.size > batch_size:
            choice = np.random.default_rng(int(seed) + step).choice(
                train_indices, size=batch_size, replace=False,
            )
        else:
            choice = train_indices
        batch = _measured_batch(
            arrays, support, choice, seed=int(seed) + step, device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        output = model(
            batch["operation_inputs"], batch["target_context"],
            batch["signal_counts"], batch["idler_counts"],
        )
        if extra_weights:
            # Use the full v7 multitask loss so the v3-v8 components are
            # actually consumed during training.
            from ml_recon.physics_hybrid_v7.scalable_model import v7_scalable_multitask_loss
            base_weights = {
                "nb2_nll": 1.0, "conditional_nll": 0.35,
                "teacher_cross_entropy": 0.0, "noise_type": 0.0,
                "noise_coarse_type": 0.0, "noise_strength": 0.0,
                "mixture": 0.0, "dispersion": 0.0,
                "retrieval": 0.1, "best_fidelity": 0.0,
                "nb2_nll_observed": 0.0, "count_calibration": 0.0,
                "adaptive_count_calibration": 0.0,
                "phase_residual_consistency": 0.0,
                "log_rate_consistency": 0.0, "frequency_feedback_base": 0.0,
                "spectral_smoothness": 0.0, "operation_equivariance": 0.0,
                "count_scale_equivariance": 0.0,
            }
            base_weights.update(extra_weights)
            loss, _ = v7_scalable_multitask_loss(output, batch, base_weights)
        else:
            loss, _ = _real_masked_loss(output, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 20 == 0 or step == int(steps) - 1:
            trajectory.append({
                "step": step,
                "train_loss": float(loss.detach().cpu()),
            })
    model.eval()
    with torch.no_grad():
        val_losses = []
        for start in range(0, val_indices.size, batch_size):
            batch = _measured_batch(
                arrays, support, val_indices[start:start + batch_size],
                seed=int(seed) + 99999, device=device,
            )
            output = model(
                batch["operation_inputs"], batch["target_context"],
                batch["signal_counts"], batch["idler_counts"],
            )
            _, metrics = _real_masked_loss(output, batch)
            val_losses.append(float(metrics["nb2_nll_per_hidden_operation"].detach().cpu()))
    return {
        "trajectory": trajectory,
        "final_train_loss": float(trajectory[-1]["train_loss"]),
        "val_nb2_nll_per_hidden_operation_mean": float(np.mean(val_losses)),
        "val_nb2_nll_per_hidden_operation_std": float(np.std(val_losses)),
        "val_step_count": int(len(val_losses)),
    }


def run_v4_measured_ab(
    *,
    output_path: str | Path,
    system_type: str = "4qubit",
    manifest_dir: str = "ml_recon/reports/measured_data",
    data_root: str = "data",
    width: int = 256,
    layers: int = 6,
    latent_tokens: int = 32,
    batch_size: int = 4,
    steps: int = 200,
    learning_rate: float = 3e-4,
    seed: int = 20260807,
    train_fraction: float = 0.7,
) -> Dict[str, Any]:
    if torch is None:
        raise RuntimeError("v4_measured_ab requires torch")
    from ml_recon.physics_identification.operation_point_calibration import (
        load_operation_point_dataset,
    )
    geometry = build_operation_frequency_geometry(system_type)
    retrieval_rates, _ = load_retrieval_rate_vectors(
        system_type, geometry.operation_count,
    )
    dataset = load_operation_point_dataset(
        manifest_dir, data_root=data_root, system_type=system_type,
    )
    arrays, audit = build_measured_scalable_arrays(
        dataset, data_root=data_root, geometry=geometry,
        retrieval_rates=retrieval_rates,
    )
    support = build_synthetic_physics_support(
        geometry, retrieval_rates, graph_mode_count=12, graph_length_mhz=8.0,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = (
        torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    )
    sample_count = arrays.coin_counts.shape[0]
    indices = np.arange(sample_count)
    rng = np.random.default_rng(int(seed))
    rng.shuffle(indices)
    train_cut = int(train_fraction * sample_count)
    train_indices = indices[:train_cut]
    val_indices = indices[train_cut:]
    # Per-trainer runs reuse the same splits.
    runs: List[Dict[str, Any]] = []
    for trainer_name in ("baseline", "v3", "v4", "v6", "v7", "v8", "v9"):
        flags = TRAINER_CONFIG[trainer_name]
        flag_payload: "Dict[str, Any]" = {}
        for k, v in flags.items():
            if isinstance(v, bool):
                flag_payload[k] = v
            elif isinstance(v, (int, float)):
                # numeric hyper-parameters (count_scale_equivariance_factor)
                flag_payload[k] = float(v)
            else:
                flag_payload[k] = v
        config = V7ScalableModelV2Config.from_mapping(
            {
                "width": int(width), "latent_tokens": int(latent_tokens),
                "layers": int(layers), "attention_heads": 8,
                "feedforward_multiplier": 3.0, "dropout": 0.0,
                "count_smoothing": 0.5,
                "use_activation_checkpointing": False,
                **flag_payload,
            },
            system_type=system_type,
            operation_count=geometry.operation_count,
            qubit_count=geometry.qubit_count,
        )
        torch.manual_seed(int(seed))
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        model = V7ScalablePhysicsModelV2(config, support.graph_kernel).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=float(learning_rate), weight_decay=0.01,
        )
        # Only the loss components each trainer opted into get a nonzero
        # weight; trainers with the flag off stay byte-identical to their
        # base (baseline = plain _real_masked_loss when extra_weights is
        # empty).
        extra_weights: "Dict[str, float]" = {}
        if flags.get("use_nb2_nll_observed"):
            extra_weights["nb2_nll_observed"] = 0.15
        if flags.get("use_count_calibration"):
            extra_weights["count_calibration"] = 0.02
        if flags.get("use_adaptive_count_calibration"):
            extra_weights["adaptive_count_calibration"] = 0.02
        if flags.get("use_phase_residual_consistency"):
            extra_weights["phase_residual_consistency"] = 0.02
        if flags.get("use_spectral_smoothness"):
            extra_weights["spectral_smoothness"] = 0.05
        if flags.get("use_operation_equivariance"):
            extra_weights["operation_equivariance"] = 0.05
        if flags.get("use_count_scale_equivariance"):
            extra_weights["count_scale_equivariance"] = float(
                flags.get("count_scale_equivariance_weight", 0.05),
            )
        result = _train_one(
            model=model, arrays=arrays, support=support,
            train_indices=train_indices, val_indices=val_indices,
            optimizer=optimizer, steps=int(steps),
            batch_size=int(batch_size), seed=int(seed), device=device,
            extra_weights=extra_weights,
        )
        if device.type == "cuda":
            peak_reserved = int(torch.cuda.max_memory_reserved(device))
        else:
            peak_reserved = 0
        result.update({
            "trainer": trainer_name,
            "sample_count": int(sample_count),
            "train_count": int(train_indices.size),
            "val_count": int(val_indices.size),
            "peak_reserved_gib": peak_reserved / 1024 ** 3,
            "research_only": True,
            "promotion_ready": False,
            "hardware_action_authorized": False,
        })
        runs.append(result)
    baseline, v3, v4, v6, v7, v8, v9 = runs
    summary = {
        "schema": "v7_v4_measured_ab_v1",
        "system_type": system_type,
        "device": str(device),
        "device_name": device_name,
        "model_size": {
            "width": int(width),
            "layers": int(layers),
            "latent_tokens": int(latent_tokens),
            "batch_size": int(batch_size),
        },
        "steps": int(steps),
        "learning_rate": float(learning_rate),
        "train_fraction": float(train_fraction),
        "input_audit": audit,
        "research_only": True,
        "promotion_ready": False,
        "hardware_action_authorized": False,
        "baseline": baseline,
        "v3": v3,
        "v4": v4,
        "v6": v6,
        "v7": v7,
        "v8": v8,
        "v9": v9,
        "improvement_v3": {
            "val_nb2_nll_delta": (
                baseline["val_nb2_nll_per_hidden_operation_mean"]
                - v3["val_nb2_nll_per_hidden_operation_mean"]
            ),
        },
        "improvement_v4": {
            "val_nb2_nll_delta": (
                baseline["val_nb2_nll_per_hidden_operation_mean"]
                - v4["val_nb2_nll_per_hidden_operation_mean"]
            ),
        },
        "improvement_v6": {
            "val_nb2_nll_delta": (
                baseline["val_nb2_nll_per_hidden_operation_mean"]
                - v6["val_nb2_nll_per_hidden_operation_mean"]
            ),
        },
        "improvement_v7": {
            "val_nb2_nll_delta": (
                baseline["val_nb2_nll_per_hidden_operation_mean"]
                - v7["val_nb2_nll_per_hidden_operation_mean"]
            ),
        },
        "v7_vs_v4": {
            "val_nb2_nll_delta": (
                v4["val_nb2_nll_per_hidden_operation_mean"]
                - v7["val_nb2_nll_per_hidden_operation_mean"]
            ),
        },
        "improvement_v8": {
            "val_nb2_nll_delta": (
                baseline["val_nb2_nll_per_hidden_operation_mean"]
                - v8["val_nb2_nll_per_hidden_operation_mean"]
            ),
        },
        "v8_vs_v4": {
            "val_nb2_nll_delta": (
                v4["val_nb2_nll_per_hidden_operation_mean"]
                - v8["val_nb2_nll_per_hidden_operation_mean"]
            ),
        },
        "improvement_v9": {
            "val_nb2_nll_delta": (
                baseline["val_nb2_nll_per_hidden_operation_mean"]
                - v9["val_nb2_nll_per_hidden_operation_mean"]
            ),
        },
        "v9_vs_v4": {
            "val_nb2_nll_delta": (
                v4["val_nb2_nll_per_hidden_operation_mean"]
                - v9["val_nb2_nll_per_hidden_operation_mean"]
            ),
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
        default="ml_recon/reports/physics_hybrid_v7/v4_measured_ab_4q.json",
    )
    parser.add_argument("--system", default="4qubit", choices=["3qubit", "4qubit"])
    parser.add_argument("--manifest-dir", default="ml_recon/reports/measured_data")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--latent-tokens", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=20260807)
    args = parser.parse_args(argv)
    summary = run_v4_measured_ab(
        output_path=args.output, system_type=args.system,
        manifest_dir=args.manifest_dir, data_root=args.data_root,
        steps=args.steps, batch_size=args.batch_size,
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
        "v7_vs_v4": summary["v7_vs_v4"],
        "improvement_v8": summary["improvement_v8"],
        "v8_vs_v4": summary["v8_vs_v4"],
        "improvement_v9": summary["improvement_v9"],
        "v9_vs_v4": summary["v9_vs_v4"],
        "baseline_val_nb2": summary["baseline"]["val_nb2_nll_per_hidden_operation_mean"],
        "v3_val_nb2": summary["v3"]["val_nb2_nll_per_hidden_operation_mean"],
        "v4_val_nb2": summary["v4"]["val_nb2_nll_per_hidden_operation_mean"],
        "v6_val_nb2": summary["v6"]["val_nb2_nll_per_hidden_operation_mean"],
        "v7_val_nb2": summary["v7"]["val_nb2_nll_per_hidden_operation_mean"],
        "v8_val_nb2": summary["v8"]["val_nb2_nll_per_hidden_operation_mean"],
        "v9_val_nb2": summary["v9"]["val_nb2_nll_per_hidden_operation_mean"],
    }, indent=2))
    return 0


__all__ = ["run_v4_measured_ab", "TRAINER_CONFIG"]


if __name__ == "__main__":
    raise SystemExit(main())