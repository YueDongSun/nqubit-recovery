"""Research evaluation for V10 Track A classifier checkpoints.

Campaign v10_campaign_20261001 user-authorized research route. Evaluates a
research-route checkpoint (V10ResearchStreamCheckpointV1) or a historical
V10Trainer-format checkpoint on the SAME materialized persistent-eval cells
and with the SAME metric machinery (``_evaluate_single_cell`` /
``_InferenceCollector``) used by the August 1k baseline report, so candidate
and baseline numbers are directly matched.

Read-only with respect to datasets and checkpoints. Outputs are engineering
evidence only: research_only=true, scientific_gate_evaluated=false,
promotion_ready=false, selection_evidence=false,
hardware_action_authorized=false.

``--validation-only`` (single-inspection discipline) resolves no test /
locked_ood / negative-control cells at all, for re-evaluations after the
single preregistered confirmation pass has already been spent; the gate
fails closed if any reserved row could still be read.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from .classifier_n_full import V10ClassifierNFull
from .evaluate_classifier_1k import (
    _build_dataset, _dataclass_to_json, _evaluate_single_cell, compute_ece,
)
from torch.utils.data import DataLoader

AUTHORITY_FLAGS = {
    "research_only": True,
    "scientific_gate_evaluated": False,
    "promotion_ready": False,
    "selection_evidence": False,
    "hardware_action_authorized": False,
}
CELL_N = {
    "classifier_n3_standard": 3,
    "classifier_n3_noisy": 3,
    "classifier_n4_standard": 4,
    "classifier_n4_noisy": 4,
}
NEGATIVE_CONTROL_SUFFIXES = (
    "target_shuffle", "operation_subset_shuffle", "global_multi_pair_shuffle",
    "environment_permutation", "mechanism_label_permutation",
)


def load_evaluable_model(checkpoint_path: str, device: str):
    """Rebuild the classifier from either research or V10Trainer checkpoints."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metadata = checkpoint.get("metadata", {})
    build_spec = checkpoint.get("model_build_spec")
    if isinstance(build_spec, dict) and "architecture_class" in build_spec:
        from .classifier_checkpoint import classifier_from_build_spec
        model = classifier_from_build_spec(build_spec)
        profile = build_spec.get("profile")
        schema = metadata.get("schema", "V10CheckpointV1")
    else:
        profile = (build_spec or {}).get("profile") or metadata.get("config", {}).get(
            "model_profile")
        if not profile:
            # Legacy V10Trainer checkpoints: infer profile by parameter count.
            from .classifier_n_full import CLASSIFIER_PROFILES
            state_dict = checkpoint["model_state_dict"]
            state_count = sum(int(torch.numel(v)) for v in state_dict.values())
            expected = metadata.get("exact_parameter_count", state_count)
            matches = [name for name in CLASSIFIER_PROFILES
                       if V10ClassifierNFull(profile=name).parameter_count() == expected
                       and sum(int(torch.numel(v)) for v in
                               V10ClassifierNFull(profile=name).state_dict().values())
                       == state_count]
            if len(matches) != 1:
                raise ValueError(
                    f"cannot infer a unique classifier profile (matches={matches})")
            profile = matches[0]
        model = V10ClassifierNFull(profile=profile)
        schema = metadata.get("schema", "unknown")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model, {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_schema": schema,
        "profile": profile,
        "model_build_spec": dict(build_spec) if isinstance(build_spec, dict) else {},
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "metadata_epoch": metadata.get("epoch"),
        "metadata_global_step": metadata.get("global_step"),
        "metadata_best_metric": metadata.get("best_metric"),
        # Training-side identity (small dicts already inside the checkpoint);
        # evaluators that build their own data plan must cross-check against
        # these instead of trusting CLI-supplied pool/rung/seed arguments.
        "training_provenance": dict(metadata.get("provenance") or {}),
        "training_config": dict(metadata.get("config") or {}),
    }


def _manifest_index_paths(manifest_path: str) -> Dict[str, str]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    return dict(manifest["range_index_paths"])


def _resolve_paths(index_paths: Dict[str, str], dataset_root: str, cell_id: str,
                   validation_only: bool = False):
    def absolute(relative: str) -> str:
        return str(Path(dataset_root) / relative)

    validation = absolute(index_paths[f"{cell_id}:validation"])
    if validation_only:
        # Single-inspection discipline: the spent test / locked_ood /
        # negative-control cells are never resolved, so they are never read.
        return validation, None, None, {}
    test = absolute(index_paths.get(f"{cell_id}:test", ""))
    ood_cell = f"{cell_id}_locked_ood"
    ood = absolute(index_paths.get(f"{ood_cell}:locked_ood", ""))
    controls = {}
    for suffix in NEGATIVE_CONTROL_SUFFIXES:
        key = f"{cell_id}_negative_control_{suffix}:negative_control"
        if key in index_paths:
            controls[suffix] = absolute(index_paths[key])
    return validation, test, ood, controls


def _collect_fine_family(model, dataset_root: str, range_path: str, cell_id: str,
                         split: str, device: str, batch_size: int = 64):
    """Collect masked fine-family logits and targets over one cell/split."""
    dataset = _build_dataset(dataset_root, range_path, cell_id, split)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    dev = torch.device(device)
    logits_all, targets_all = [], []
    with torch.no_grad():
        for raw_batch in loader:
            batch = {k: v.to(dev) if isinstance(v, torch.Tensor) else v
                     for k, v in raw_batch.items()}
            output = model(batch, N=CELL_N[cell_id], fine_identifiable=True)
            if "fine_family_logits" not in output:
                continue
            mask = (batch["target_fine_mask"].bool()
                    & batch["target_fine_identifiable_mask"].bool())
            logits = output["fine_family_logits"].float().cpu()
            targets = batch["target_fine"].long().cpu()
            logits_all.append(logits[mask.cpu()])
            targets_all.append(targets[mask.cpu()])
    if not logits_all:
        return None, None
    return torch.cat(logits_all), torch.cat(targets_all)


def temperature_calibration_report(
    model, dataset_root: str, index_paths: Dict[str, str], device: str,
    batch_size: int = 64, t_min: float = 0.5, t_max: float = 12.0, t_step: float = 0.05,
    validation_only: bool = False,
) -> Dict[str, Any]:
    """Fit ONE global temperature on validation, report calibrated test ECE.

    Method follows the V7-validated finding that a single global temperature
    is sufficient and per-bin scaling overfits. T is fitted ONLY on the
    validation splits (minimizing masked fine-family cross-entropy); test
    splits are scored with the validation-fitted T and remain untouched by
    the fit. With ``validation_only`` no test split is resolved at all, so
    the report carries no test block.
    """
    val_cells, test_cells = [], []
    for cell_id in CELL_N:
        validation_path, test_path, _, _ = _resolve_paths(
            index_paths, dataset_root, cell_id,
            validation_only=validation_only)
        v_logits, v_targets = _collect_fine_family(
            model, dataset_root, validation_path, cell_id, "validation",
            device, batch_size)
        if v_logits is not None:
            val_cells.append((cell_id, v_logits, v_targets))
        if test_path:
            t_logits, t_targets = _collect_fine_family(
                model, dataset_root, test_path, cell_id, "test", device, batch_size)
            if t_logits is not None:
                test_cells.append((cell_id, t_logits, t_targets))
    if not val_cells:
        return {"status": "NO_FINE_FAMILY_OUTPUT"}

    def pooled_nll(cells, temperature):
        # Class spaces differ per cell (14 vs 24): average per-cell CE, do not
        # concatenate logits across different label spaces.
        total, rows = 0.0, 0
        for _, logits, targets in cells:
            total += torch.nn.functional.cross_entropy(
                logits / temperature, targets, reduction="sum").item()
            rows += int(targets.numel())
        return total / max(rows, 1)

    grid = np.arange(t_min, t_max + 1e-9, t_step)
    best_t, best_nll = 1.0, float("inf")
    for t in grid:
        nll = pooled_nll(val_cells, float(t))
        if nll < best_nll:
            best_nll, best_t = nll, float(t)

    def macro_ece(cells, temperature):
        values = []
        for _, logits, targets in cells:
            probs = torch.softmax(logits / temperature, dim=-1).numpy()
            values.append(compute_ece(probs, targets.numpy()))
        return sum(values) / len(values) if values else None

    val_rows = sum(int(targets.numel()) for _, _, targets in val_cells)
    report = {
        "fitted_temperature": best_t,
        "fit_target": "per-cell masked fine-family cross-entropy averaged over validation cells",
        "val_nll_at_T": best_nll,
        "val_ece_raw": macro_ece(val_cells, 1.0),
        "val_ece_calibrated": macro_ece(val_cells, best_t),
        "val_rows": val_rows,
        "per_cell_val_ece_calibrated": {
            cell_id: compute_ece(
                torch.softmax(logits / best_t, dim=-1).numpy(), targets.numpy())
            for cell_id, logits, targets in val_cells},
    }
    if test_cells:
        report.update({
            "test_ece_raw": macro_ece(test_cells, 1.0),
            "test_ece_calibrated": macro_ece(test_cells, best_t),
            "test_rows": sum(int(targets.numel()) for _, _, targets in test_cells),
        })
    report["status"] = "OK"
    return report


def evaluate_checkpoint(
    *, checkpoint_path: str, dataset_root: str, persistent_manifest: str,
    device: str = "cuda", seed: int = 20261001, batch_size: int = 64,
    cells: tuple = tuple(CELL_N), include_negative_controls: bool = True,
    with_temperature: bool = True,
    ood_composite_weights: Optional[str] = None,
    validation_only: bool = False,
) -> Dict[str, Any]:
    if validation_only and ood_composite_weights is not None:
        # The composite is a labeled SECOND inspection of the reserved
        # locked_ood cells; it can never run under a validation-only gate.
        raise ValueError(
            "validation_only forbids ood_composite_weights: the composite "
            "reads the reserved locked_ood cells")
    model, checkpoint_info = load_evaluable_model(checkpoint_path, device)
    index_paths = _manifest_index_paths(persistent_manifest)
    per_cell = {}
    for cell_id in cells:
        validation_path, test_path, ood_path, control_paths = _resolve_paths(
            index_paths, dataset_root, cell_id,
            validation_only=validation_only)
        if validation_only and (
                test_path is not None or ood_path is not None or control_paths):
            raise RuntimeError(
                f"validation_only gate failed for cell {cell_id}: a reserved "
                "test/locked_ood/negative-control path was resolved")
        metrics = _evaluate_single_cell(
            model, dataset_root, validation_path, cell_id, CELL_N[cell_id],
            device, seed, batch_size=batch_size,
            test_range_index_path=test_path, ood_range_index_path=ood_path,
            negative_control_range_index_paths=(
                control_paths if include_negative_controls else None),
        )
        per_cell[cell_id] = _dataclass_to_json(metrics)

    if validation_only:
        reserved_rows = sum(
            int(blob.get("n_test", 0)) + int(blob.get("n_ood", 0))
            + int(blob.get("n_negative_control", 0))
            for blob in per_cell.values())
        if reserved_rows != 0:
            raise RuntimeError(
                "validation_only gate violated: reserved-split rows were "
                f"counted (n_test+n_ood+n_negative_control={reserved_rows})")

    ood_score_composite = None
    if ood_composite_weights is not None:
        weights_payload = json.loads(
            Path(ood_composite_weights).read_text(encoding="utf-8"))
        if weights_payload.get("schema") != "V10ResearchOODCompositeWeightsV1":
            raise ValueError(
                "composite weights must be a V10ResearchOODCompositeWeightsV1 "
                "proxy export from research_ood_score")
        from .research_ood_score import composite_score_report_on_eval_store

        ood_score_composite = composite_score_report_on_eval_store(
            model=model, dataset_root=dataset_root, index_paths=index_paths,
            weights_payload=weights_payload, device=device,
            batch_size=batch_size)

    def macro(source: str, field_path: tuple) -> Optional[float]:
        values = []
        for cell_id, blob in per_cell.items():
            node: Any = blob.get(source)
            for field in field_path:
                if not isinstance(node, dict):
                    node = None
                    break
                node = node.get(field)
            if isinstance(node, (int, float)) and node == node:
                values.append(float(node))
        return sum(values) / len(values) if values else None

    summary = {
        "macro_fine_family_f1_validation": macro("classification", ("macro_f1",)),
        "macro_fine_family_f1_test": None,
        "macro_depth_accuracy": macro("structure", ("depth_accuracy",)),
        "macro_intactness_accuracy": macro("structure", ("intactness_accuracy",)),
        "macro_measurement_nll": macro("measurement_calibration", ("measurement_nll",)),
        "macro_ece": macro("measurement_calibration", ("ece",)),
        "macro_p0_error": macro("w_physics", ("p0_error",)),
        "macro_r1_frobenius_error": macro(
            "w_physics", ("r1_frobenius_error_normalized",)),
    }
    test_values = []
    for blob in per_cell.values():
        test_block = blob.get("test_classification") or {}
        value = test_block.get("macro_f1")
        if isinstance(value, (int, float)) and value == value:
            test_values.append(float(value))
    if test_values:
        summary["macro_fine_family_f1_test"] = sum(test_values) / len(test_values)

    report = {
        "schema": "V10ResearchClassifierEvaluationV1",
        "route_id": "user_authorized_research_route_v1",
        **AUTHORITY_FLAGS,
        "checkpoint": checkpoint_info,
        "dataset_root": dataset_root,
        "persistent_manifest": persistent_manifest,
        "device": device,
        "seed": seed,
        "summary": summary,
        "temperature_calibration": (
            temperature_calibration_report(
                model, dataset_root, index_paths, device, batch_size,
                validation_only=validation_only)
            if with_temperature else None),
        "per_cell": per_cell,
        "ood_score_composite": ood_score_composite,
    }
    if validation_only:
        # reserved_rows_read is verified 0 above (the run raises otherwise);
        # per-cell blobs also carry n_test=n_ood=n_negative_control=0.
        report["validation_only_gate"] = {
            "enabled": True,
            "reserved_rows_read": 0,
            "evaluated_splits": ["validation"],
            "reserved_paths_nulled": ["test", "locked_ood", "negative_controls"],
        }
    return report


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--persistent-manifest", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--no-negative-controls", dest="include_negative_controls",
                        action="store_false")
    parser.add_argument("--no-temperature", dest="with_temperature",
                        action="store_false")
    parser.add_argument("--ood-score-composite", action="store_true",
                        help="apply a proxy-fitted OOD composite to the reserved "
                             "validation/locked_ood cells (SECOND-INSPECTION "
                             "evidence; requires a registered confirmation design)")
    parser.add_argument("--composite-weights",
                        help="V10ResearchOODCompositeWeightsV1 JSON from "
                             "research_ood_score --export-weights")
    parser.add_argument("--validation-only", dest="validation_only",
                        action="store_true",
                        help="resolve no test/locked_ood/negative-control "
                             "cells (single-inspection discipline: spent "
                             "reserved cells are never read)")
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args(argv)
    if arguments.ood_score_composite and not arguments.composite_weights:
        parser.error("--ood-score-composite requires --composite-weights "
                     "(the composite is proxy-fitted, never refit on eval rows)")
    if arguments.validation_only and arguments.ood_score_composite:
        parser.error("--validation-only is incompatible with "
                     "--ood-score-composite (the composite inspects the "
                     "reserved locked_ood cells)")
    report = evaluate_checkpoint(
        checkpoint_path=arguments.checkpoint, dataset_root=arguments.dataset_root,
        persistent_manifest=arguments.persistent_manifest, device=arguments.device,
        seed=arguments.seed, batch_size=arguments.batch_size,
        include_negative_controls=arguments.include_negative_controls,
        with_temperature=arguments.with_temperature,
        ood_composite_weights=(
            arguments.composite_weights
            if arguments.ood_score_composite else None),
        validation_only=arguments.validation_only,
    )
    output_path = Path(arguments.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, default=str),
                           encoding="utf-8")
    print(json.dumps({"summary": report["summary"],
                      "temperature_calibration": report.get("temperature_calibration"),
                      "checkpoint": report["checkpoint"]}, indent=2, default=str))


if __name__ == "__main__":
    main()
