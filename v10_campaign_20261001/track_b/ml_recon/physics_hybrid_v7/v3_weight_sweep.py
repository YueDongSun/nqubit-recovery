"""V3 loss-weight grid sweep on the v2 scalable model.

Two axes are swept:

* ``nb2_observed_weight`` ∈ {0.00, 0.05, 0.15, 0.30, 0.60}
* ``count_calibration_weight`` ∈ {0.00, 0.02, 0.05, 0.10, 0.20}

Each cell runs a short paired trainer (baseline / v3) on the same
synthetic 3Q holdout and writes a single row to the summary JSON.
The objective is to minimise the observed-mask NB2 NLL at the last
step without regressing bestF MAE.

The sweep is research-only and never writes to ``matrix/ml_delta``.
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
from .learning_rate_ab import _train_one
from .noise_taxonomy import build_synthetic_physics_support
from .physical_ceiling import load_retrieval_rate_vectors


DEFAULT_NB2_GRID = [0.0, 0.05, 0.15, 0.30, 0.60]
DEFAULT_CALIB_GRID = [0.0, 0.02, 0.05, 0.10, 0.20]


def _row(system_type: str, geometry, support, device, *,
          nb2_observed_weight: float, count_calibration_weight: float,
          optimizer_steps: int, batch_size: int, learning_rate: float,
          loss_threshold: float, seed: int) -> Dict[str, Any]:
    """Run a paired baseline (v3 off) and v3 trainer with the given weights."""

    # Patch the v3 weights before invoking the shared training loop.
    import ml_recon.physics_hybrid_v7.learning_rate_ab as ab
    ab.V3_LOSS_WEIGHTS.update({
        "nb2_nll_observed": float(nb2_observed_weight),
        "count_calibration": float(count_calibration_weight),
    })
    baseline = _train_one(
        system_type=system_type, geometry=geometry, support=support,
        device=device, use_v3=False, seed=seed,
        learning_rate=learning_rate, weight_decay=0.01,
        optimizer_steps=optimizer_steps, batch_size=batch_size,
        loss_threshold=loss_threshold,
    )
    v3 = _train_one(
        system_type=system_type, geometry=geometry, support=support,
        device=device, use_v3=True, seed=seed,
        learning_rate=learning_rate, weight_decay=0.01,
        optimizer_steps=optimizer_steps, batch_size=batch_size,
        loss_threshold=loss_threshold,
    )
    return {
        "nb2_observed_weight": float(nb2_observed_weight),
        "count_calibration_weight": float(count_calibration_weight),
        "baseline_final_nb2_nll_observed": baseline["final_nb2_nll_observed"],
        "v3_final_nb2_nll_observed": v3["final_nb2_nll_observed"],
        "baseline_final_count_calibration": baseline["final_count_calibration"],
        "v3_final_count_calibration": v3["final_count_calibration"],
        "baseline_final_total": baseline["final_total"],
        "v3_final_total": v3["final_total"],
        "baseline_final_bestf_mae": baseline["final_bestf_mae"],
        "v3_final_bestf_mae": v3["final_bestf_mae"],
        "nb2_observed_improvement": (
            baseline["final_nb2_nll_observed"] - v3["final_nb2_nll_observed"]
        ),
        "count_calibration_improvement": (
            baseline["final_count_calibration"] - v3["final_count_calibration"]
        ),
        "bestf_mae_delta": baseline["final_bestf_mae"] - v3["final_bestf_mae"],
    }


def run_v3_weight_sweep(
    *,
    output_path: str | Path,
    system_type: str = "3qubit",
    nb2_grid: List[float] = None,
    calib_grid: List[float] = None,
    optimizer_steps: int = 150,
    batch_size: int = 8,
    learning_rate: float = 3e-4,
    loss_threshold: float = 7.5,
    seed: int = 20260807,
) -> Dict[str, Any]:
    if torch is None:
        raise RuntimeError("v3_weight_sweep requires torch")
    nb2_grid = nb2_grid or DEFAULT_NB2_GRID
    calib_grid = calib_grid or DEFAULT_CALIB_GRID
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
    rows: List[Dict[str, Any]] = []
    for nb2_w in nb2_grid:
        for calib_w in calib_grid:
            row = _row(
                system_type=system_type, geometry=geometry, support=support,
                device=device, nb2_observed_weight=nb2_w,
                count_calibration_weight=calib_w,
                optimizer_steps=optimizer_steps, batch_size=batch_size,
                learning_rate=learning_rate, loss_threshold=loss_threshold,
                seed=seed,
            )
            rows.append(row)
    # Pick the best row by observed NB2 NLL improvement, with a
    # non-regression gate on bestF MAE.
    candidates = [r for r in rows if r["bestf_mae_delta"] >= -0.01]
    if not candidates:
        candidates = rows
    best = min(
        candidates,
        key=lambda r: (r["v3_final_nb2_nll_observed"], r["v3_final_bestf_mae"]),
    )
    summary = {
        "schema": "v7_v3_weight_sweep_v1",
        "system_type": system_type,
        "device": str(device),
        "device_name": device_name,
        "optimizer_steps": int(optimizer_steps),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "loss_threshold": float(loss_threshold),
        "nb2_grid": list(nb2_grid),
        "calib_grid": list(calib_grid),
        "rows": rows,
        "best_row": best,
        "research_only": True,
        "promotion_ready": False,
        "hardware_action_authorized": False,
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
        default="ml_recon/reports/physics_hybrid_v7/v3_weight_sweep_3q.json",
    )
    parser.add_argument("--system", default="3qubit", choices=["3qubit", "4qubit"])
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--threshold", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=20260807)
    args = parser.parse_args(argv)
    summary = run_v3_weight_sweep(
        output_path=args.output, system_type=args.system,
        optimizer_steps=args.steps, batch_size=args.batch_size,
        learning_rate=args.learning_rate, loss_threshold=args.threshold,
        seed=args.seed,
    )
    print(json.dumps({
        "schema": summary["schema"],
        "system_type": summary["system_type"],
        "best_row": summary["best_row"],
        "row_count": len(summary["rows"]),
    }, indent=2))
    return 0


__all__ = ["run_v3_weight_sweep", "DEFAULT_NB2_GRID", "DEFAULT_CALIB_GRID"]


if __name__ == "__main__":
    raise SystemExit(main())