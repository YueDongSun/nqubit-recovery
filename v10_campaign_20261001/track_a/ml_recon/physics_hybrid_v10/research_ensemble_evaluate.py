"""Ensemble evaluation for V10 Track A research checkpoints (logit averaging).

Campaign v10_campaign_20261001 exploration phase. Loads N research-route
checkpoints of the SAME model profile, wraps them in an ensemble that averages
every tensor output (logits), and evaluates with the SAME metric machinery
(``_evaluate_single_cell``) as the single-model evaluator. Dev discipline:
default evaluation is VALIDATION-ONLY (test / locked-OOD / negative-control
cells are reserved for the single preregistered confirmation pass and are not
inspected here).

Research-only: research_only=true, scientific_gate_evaluated=false,
promotion_ready=false, selection_evidence=false, hardware_action_authorized=false.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from .research_evaluate_classifier import (
    AUTHORITY_FLAGS, CELL_N, _dataclass_to_json, _manifest_index_paths,
    _resolve_paths, load_evaluable_model,
)
from .evaluate_classifier_1k import _evaluate_single_cell


class LogitAverageEnsemble(torch.nn.Module):
    """Average every tensor output of N same-profile classifier checkpoints."""

    def __init__(self, models):
        super().__init__()
        self.models = torch.nn.ModuleList(models)
        if not models:
            raise ValueError("ensemble requires at least one member model")

    @property
    def member_count(self) -> int:
        return len(self.models)

    def forward(self, batch: Dict[str, torch.Tensor], *, N: int,
                fine_identifiable: bool = False):
        outputs = [
            model(batch, N=N, fine_identifiable=fine_identifiable)
            for model in self.models
        ]
        first = outputs[0]
        averaged: Dict[str, Any] = {}
        for key, value in first.items():
            if isinstance(value, torch.Tensor):
                stacked = torch.stack(
                    [out[key] for out in outputs]).to(dtype=torch.float32)
                averaged[key] = stacked.mean(dim=0).to(dtype=value.dtype)
            else:
                averaged[key] = value
        return averaged


def evaluate_ensemble(
    *, checkpoint_paths, dataset_root: str, persistent_manifest: str,
    device: str = "cuda", seed: int = 20261001, batch_size: int = 64,
    cells: tuple = ("classifier_n3_standard", "classifier_n3_noisy",
                    "classifier_n4_standard", "classifier_n4_noisy"),
    include_reserved_cells: bool = False,
) -> Dict[str, Any]:
    models = []
    profiles = set()
    for path in checkpoint_paths:
        model, info = load_evaluable_model(path, device)
        models.append(model)
        profiles.add(info["profile"])
    if len(profiles) != 1:
        raise ValueError(f"ensemble members must share one profile, got {profiles}")
    ensemble = LogitAverageEnsemble(models).to(device)
    ensemble.eval()

    index_paths = _manifest_index_paths(persistent_manifest)
    per_cell = {}
    for cell_id in cells:
        validation_path, test_path, ood_path, controls = _resolve_paths(
            index_paths, dataset_root, cell_id)
        metrics = _evaluate_single_cell(
            ensemble, dataset_root, validation_path, cell_id, CELL_N[cell_id],
            device, seed, batch_size=batch_size,
            test_range_index_path=test_path if include_reserved_cells else None,
            ood_range_index_path=ood_path if include_reserved_cells else None,
            negative_control_range_index_paths=(
                controls if include_reserved_cells else None),
        )
        per_cell[cell_id] = _dataclass_to_json(metrics)

    def macro(source: str, field_path: tuple) -> Optional[float]:
        values = []
        for blob in per_cell.values():
            node: Any = blob.get(source)
            for field in field_path:
                if not isinstance(node, dict):
                    node = None
                    break
                node = node.get(field)
            if isinstance(node, (int, float)) and node == node:
                values.append(float(node))
        return sum(values) / len(values) if values else None

    return {
        "schema": "V10ResearchEnsembleEvaluationV1",
        "route_id": "user_authorized_research_route_v1",
        **AUTHORITY_FLAGS,
        "member_count": ensemble.member_count,
        "member_checkpoints": list(checkpoint_paths),
        "profile": next(iter(profiles)),
        "dataset_root": dataset_root,
        "persistent_manifest": persistent_manifest,
        "evaluated_splits": (["validation"] if not include_reserved_cells
                             else ["validation", "test", "locked_ood",
                                   "negative_controls"]),
        "summary": {
            "macro_fine_family_f1_validation": macro("classification", ("macro_f1",)),
            "macro_depth_accuracy": macro("structure", ("depth_accuracy",)),
            "macro_measurement_nll": macro("measurement_calibration", ("measurement_nll",)),
            "macro_fine_family_f1_test": macro("test_classification", ("macro_f1",)),
            "macro_depth_accuracy_test": macro("test_structure", ("depth_accuracy",)),
            "macro_ood_auroc": macro("ood", ("auroc",)),
        },
        "per_cell": per_cell,
    }


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", action="append", required=True,
                        help="research checkpoint path, repeatable")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--persistent-manifest", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--include-reserved-cells", action="store_true",
                        help="also evaluate test / locked-OOD / negative-"
                             "control cells; reserved for the single "
                             "preregistered confirmation pass")
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args(argv)
    report = evaluate_ensemble(
        checkpoint_paths=arguments.checkpoint,
        dataset_root=arguments.dataset_root,
        persistent_manifest=arguments.persistent_manifest,
        device=arguments.device, seed=arguments.seed,
        batch_size=arguments.batch_size,
        include_reserved_cells=arguments.include_reserved_cells,
    )
    output_path = Path(arguments.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, default=str),
                           encoding="utf-8")
    print(json.dumps({
        "member_count": report["member_count"],
        "summary": report["summary"],
        "per_cell_macro_f1": {
            cell: blob.get("classification", {}).get("macro_f1")
            for cell, blob in report["per_cell"].items()},
    }, indent=2))


if __name__ == "__main__":
    main()
