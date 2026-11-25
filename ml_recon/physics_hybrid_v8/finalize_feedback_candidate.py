"""Aggregate V8 development evidence and freeze a no-authority ensemble.

The selector is development-only, multi-seed and worst-condition first.  It
never opens the locked final split.  Static/read-only eligibility is separate
from action-policy eligibility: a causal branch may be retained for shadow
representation work while proposals remain forced to no-op/abstain until real
paired shuffle controls and real grouped conformal calibration pass.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from .train_feedback_models import (
    CHECKPOINT_SCHEMA,
    TRAINING_REPORT_SCHEMA,
    WEAK_PAIRED_COMPARISON_CONTRACT_SCHEMA,
    WEAK_PAIRED_COMPARISON_SCHEMA,
    _file_sha256,
    _stable_json_hash,
)


FINALIZATION_SCHEMA = "v8_feedback_frozen_candidate_v1"
POINTER_SCHEMA = "v8_feedback_frozen_candidate_pointer_v1"
MAX_PEAK_RESERVED_GIB = 9.5
MAX_SERIOUS_HARM_WILSON_UPPER_95 = 0.025
MINIMUM_TOTAL_GAIN_LCB95 = 0.0
MINIMUM_SEED_COUNT = 5
COMPARISON_SCHEMA = "v8_v7_frozen_development_comparison_v1"
REQUIRED_MEASURED_STATIC_METRICS = (
    "real_nb2",
    "real_ece_count_ge_1",
    "real_ece_count_ge_2",
    "real_rho_fidelity",
)


def _closed_loop_conditions(report: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    missing = report.get("classifier_missing_rho_stress")
    if not isinstance(missing, Mapping):
        raise ValueError("candidate report lacks missing-rho stress evaluation")
    return {
        "rho_available_3qubit": report["synthetic_closed_loop_evaluation"]["3qubit"],
        "rho_available_4qubit": report["synthetic_closed_loop_evaluation"]["4qubit"],
        "rho_missing_3qubit": missing["synthetic_closed_loop_evaluation"]["3qubit"],
        "rho_missing_4qubit": missing["synthetic_closed_loop_evaluation"]["4qubit"],
    }


def _load_report(path_value: str | Path) -> Tuple[Path, Dict[str, Any]]:
    path = Path(path_value).resolve()
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema") != TRAINING_REPORT_SCHEMA:
        raise ValueError(f"unsupported V8 training report: {path}")
    if report.get("report_hash") != _stable_json_hash({
        key: value for key, value in report.items() if key != "report_hash"
    }):
        raise ValueError(f"training report hash mismatch: {path}")
    if (
        report.get("research_only") is not True
        or report.get("promotion_ready") is not False
        or report.get("hardware_action_authorized") is not False
    ):
        raise ValueError("training report violates V8 no-authority contract")
    if report.get("selection_scope") != "development_only":
        raise ValueError("candidate selection accepts development-only reports")
    if report.get("locked_final_metrics_present") is not False:
        raise ValueError("locked final metrics must not enter candidate selection")
    locked_state = dict(report.get("locked_final_test") or {})
    if locked_state.get("status") != "sealed_not_accessed":
        raise ValueError("candidate report has already accessed the locked final test")
    return path, report


def _require_development_selection_eligible(
    report: Mapping[str, Any],
) -> Dict[str, Any]:
    """Reject diagnostic controls and every other ineligible report."""

    eligibility = dict(report.get("development_selection_eligibility") or {})
    if eligibility.get("schema") != "v8_development_selection_eligibility_v1":
        raise ValueError("candidate lacks development selection eligibility")
    expected_hash = _stable_json_hash({
        key: value for key, value in eligibility.items()
        if key != "eligibility_sha256"
    })
    if str(eligibility.get("eligibility_sha256", "")) != expected_hash:
        raise ValueError("candidate development selection eligibility hash mismatch")
    if eligibility.get("selection_eligible") is not True:
        raise ValueError(
            "V8 finalization rejects development-selection-ineligible reports: "
            f"{eligibility.get('reason', 'unspecified')}"
        )
    return eligibility


def _checkpoint_for_report(report: Mapping[str, Any]) -> Tuple[Path, str, Dict[str, Any]]:
    if torch is None:  # pragma: no cover
        raise RuntimeError("V8 candidate finalization requires torch")
    path = Path(report["checkpoint"]["path"])
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    checkpoint_sha256 = _file_sha256(path)
    if checkpoint_sha256 != str(report["checkpoint"]["sha256"]):
        raise ValueError("checkpoint hash does not match its training report")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("candidate checkpoint schema changed")
    if (
        checkpoint.get("research_only") is not True
        or checkpoint.get("promotion_ready") is not False
        or checkpoint.get("hardware_action_authorized") is not False
    ):
        raise ValueError("candidate checkpoint violates V8 no-authority contract")
    return path, checkpoint_sha256, checkpoint


def _weak_validation_state(report: Mapping[str, Any]) -> Dict[str, Any]:
    profile = dict(report.get("profile") or {})
    relational_enabled = bool(
        profile.get("weak_state_pair_supervision_enabled", False)
    )
    if not relational_enabled:
        return {
            "required": False,
            "passed": True,
            "status": "not_required_for_non_relational_control",
            "stage": None,
        }
    training = dict(report.get("training") or {})
    available: List[Tuple[str, Mapping[str, Any]]] = []
    for stage_name in ("measured_pretrain", "measured_adapter"):
        stage = dict(training.get(stage_name) or {})
        weak = dict(stage.get("state_pair_transport_weak") or {})
        if weak and weak.get("status") != "stage_skipped":
            available.append((stage_name, weak))
    if not available:
        return {
            "required": True,
            "passed": False,
            "status": "fail_closed_missing_development_validation_weak_control",
            "stage": None,
        }
    stage_name, weak = available[-1]
    passed = bool(
        weak.get("relational_state_pair_supervision_enabled") is True
        and weak.get("weak_validation_gate_passed") is True
        and weak.get("weak_branch_eligible_for_selection") is True
    )
    return {
        "required": True,
        "passed": passed,
        "status": "passed" if passed else "failed_or_insufficient",
        "stage": stage_name,
        "weak_report_sha256": _stable_json_hash(weak),
        "development_validation_negative_controls": copy.deepcopy(
            weak.get("development_validation_negative_controls")
        ),
    }


def _load_weak_paired_comparison(
    path_value: str | Path,
) -> Tuple[Path, Dict[str, Any]]:
    path = Path(path_value).resolve()
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if artifact.get("schema") != WEAK_PAIRED_COMPARISON_SCHEMA:
        raise ValueError(f"unsupported weak paired-comparison artifact: {path}")
    expected_hash = _stable_json_hash({
        key: value for key, value in artifact.items()
        if key != "artifact_sha256"
    })
    if str(artifact.get("artifact_sha256", "")) != expected_hash:
        raise ValueError(f"weak paired-comparison artifact hash mismatch: {path}")
    if (
        artifact.get("comparison_result_present") is not True
        or artifact.get("selection_scope") != "development_only"
        or artifact.get("locked_final_metrics_present") is not False
        or artifact.get("allowed_for_model_selection") is not True
        or artifact.get("research_only") is not True
        or artifact.get("promotion_ready") is not False
        or artifact.get("hardware_action_authorized") is not False
    ):
        raise ValueError("weak paired-comparison artifact violates selection scope")
    return path, artifact


def _latest_executed_weak_stage(
    report: Mapping[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    available: List[Tuple[str, Dict[str, Any]]] = []
    training = dict(report.get("training") or {})
    for stage_name in ("measured_pretrain", "measured_adapter"):
        weak = dict(
            dict(training.get(stage_name) or {}).get(
                "state_pair_transport_weak"
            ) or {}
        )
        if weak and weak.get("status") != "stage_skipped":
            available.append((stage_name, weak))
    if not available:
        raise ValueError("paired-comparison lineage has no executed weak stage")
    return available[-1]


def _weak_paired_comparison_gate(
    comparison_paths: Optional[Sequence[str | Path]],
    reports: Sequence[Tuple[Path, Mapping[str, Any]]],
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    relational_enabled = all(
        dict(report.get("profile") or {}).get(
            "weak_state_pair_supervision_enabled"
        ) is True
        for _, report in reports
    )
    relational_present = any(
        dict(report.get("profile") or {}).get(
            "weak_state_pair_supervision_enabled"
        ) is True
        for _, report in reports
    )
    if relational_present != relational_enabled:
        raise ValueError("ensemble cannot mix relational and non-relational profiles")
    if not relational_enabled:
        if comparison_paths:
            raise ValueError(
                "weak paired-comparison artifacts are only valid for a relational profile"
            )
        value = {
            "schema": "v8_weak_supervision_paired_comparison_gate_v1",
            "required": False,
            "status": "not_required_for_non_relational_profile",
            "passed": True,
            "covered_seed_count": 0,
            "required_seed_count": 0,
            "artifact_by_seed": {},
        }
        value["gate_sha256"] = _stable_json_hash(value)
        return value

    if not comparison_paths:
        raise ValueError(
            "relational weak ensemble requires one paired-comparison artifact per seed"
        )
    expected_by_checkpoint = {
        str(row["checkpoint_sha256"]): (report_path, report, row)
        for (report_path, report), row in zip(reports, rows)
    }
    artifact_by_seed: Dict[str, Any] = {}
    covered_checkpoints: set[str] = set()
    for path_value in comparison_paths:
        artifact_path, artifact = _load_weak_paired_comparison(path_value)
        weak_on = dict(artifact.get("weak_on") or {})
        candidate_checkpoint_sha256 = str(
            weak_on.get("checkpoint_sha256", "")
        )
        if candidate_checkpoint_sha256 not in expected_by_checkpoint:
            raise ValueError(
                "weak paired-comparison artifact is not bound to an ensemble member"
            )
        if candidate_checkpoint_sha256 in covered_checkpoints:
            raise ValueError("duplicate weak paired-comparison artifact for one member")
        candidate_report_path, candidate_report, candidate_row = (
            expected_by_checkpoint[candidate_checkpoint_sha256]
        )
        candidate_profile = dict(candidate_report.get("profile") or {})
        candidate_contract = dict(
            candidate_report.get("weak_supervision_paired_comparison") or {}
        )
        if candidate_contract.get("schema") != WEAK_PAIRED_COMPARISON_CONTRACT_SCHEMA:
            raise ValueError("candidate paired-comparison contract schema changed")
        candidate_eligibility = dict(
            candidate_report.get("development_selection_eligibility") or {}
        )
        candidate_bindings_pass = bool(
            int(artifact.get("seed", -1)) == int(candidate_row["seed"])
            and str(artifact.get("pair_id", ""))
            == str(candidate_contract.get("pair_id", ""))
            and str(artifact.get("comparison_basis_sha256", ""))
            == str(candidate_contract.get("comparison_basis_sha256", ""))
            and str(weak_on.get("profile", ""))
            == str(candidate_profile.get("name", ""))
            and str(weak_on.get("training_report_hash", ""))
            == str(candidate_report.get("report_hash", ""))
            and str(weak_on.get("training_report_file_sha256", ""))
            == _file_sha256(candidate_report_path)
            and str(weak_on.get("selection_eligibility_sha256", ""))
            == str(candidate_eligibility.get("eligibility_sha256", ""))
        )
        if not candidate_bindings_pass:
            raise ValueError("weak paired-comparison candidate hash binding failed")

        control_binding = dict(artifact.get("weak_off_control") or {})
        control_report_path = Path(
            str(control_binding.get("training_report_path", ""))
        ).resolve()
        if (
            not control_report_path.is_file()
            or _file_sha256(control_report_path)
            != str(control_binding.get("training_report_file_sha256", ""))
        ):
            raise ValueError("weak paired-comparison control report hash binding failed")
        loaded_control_path, control_report = _load_report(control_report_path)
        _, control_checkpoint_sha256, _ = _checkpoint_for_report(control_report)
        control_eligibility = dict(
            control_report.get("development_selection_eligibility") or {}
        )
        expected_control_eligibility_hash = _stable_json_hash({
            key: value for key, value in control_eligibility.items()
            if key != "eligibility_sha256"
        })
        control_contract = dict(
            control_report.get("weak_supervision_paired_comparison") or {}
        )
        control_bindings_pass = bool(
            str(loaded_control_path) == str(control_report_path)
            and int(control_report.get("seed", -2)) == int(candidate_row["seed"])
            and str(control_binding.get("training_report_hash", ""))
            == str(control_report.get("report_hash", ""))
            and str(control_binding.get("checkpoint_sha256", ""))
            == control_checkpoint_sha256
            and str(control_binding.get("profile", ""))
            == str(dict(control_report.get("profile") or {}).get("name", ""))
            and str(control_binding.get("selection_eligibility_sha256", ""))
            == str(control_eligibility.get("eligibility_sha256", ""))
            == expected_control_eligibility_hash
            and control_eligibility.get("selection_eligible") is False
            and control_eligibility.get("reason")
            == "compute_matched_non_relational_control_is_diagnostic_only"
            and str(control_contract.get("comparison_basis_sha256", ""))
            == str(candidate_contract.get("comparison_basis_sha256", ""))
        )
        if not control_bindings_pass:
            raise ValueError("weak paired-comparison control lineage binding failed")

        registered_gate = dict(
            dict(candidate_contract.get("comparison_basis") or {}).get(
                "paired_comparison_gate"
            ) or {}
        )
        candidate_stage_name, candidate_stage = _latest_executed_weak_stage(
            candidate_report
        )
        control_stage_name, control_stage = _latest_executed_weak_stage(
            control_report
        )
        schedule_fields = (
            "scheduled_endpoint_ids_sha256",
            "scheduled_pair_ids_sha256",
            "pair_set_sha256",
            "maximum_batches_per_system",
            "steps",
        )
        candidate_by_system = dict(candidate_stage.get("by_system") or {})
        control_by_system = dict(control_stage.get("by_system") or {})
        independently_recomputed_compute_match = bool(
            candidate_stage_name == control_stage_name
            and candidate_stage.get("compute_matched") is True
            and control_stage.get("compute_matched") is True
            and candidate_stage.get("same_schedule_seed_contract")
            == control_stage.get("same_schedule_seed_contract")
            and set(candidate_by_system) == set(control_by_system)
            and all(
                all(
                    dict(candidate_by_system[system_type]).get(field_name)
                    == dict(control_by_system[system_type]).get(field_name)
                    for field_name in schedule_fields
                )
                for system_type in candidate_by_system
            )
        )
        independently_recomputed_lineage_match = bool(
            dict(candidate_report.get("profile") or {}).get(
                "weak_state_pair_supervision_enabled"
            ) is True
            and dict(control_report.get("profile") or {}).get(
                "weak_state_pair_supervision_enabled"
            ) is False
            and int(candidate_report.get("seed", -1))
            == int(control_report.get("seed", -2))
            and candidate_report.get("config_sha256")
            == control_report.get("config_sha256")
            and candidate_report.get("split_hash_by_system")
            == control_report.get("split_hash_by_system")
            and all(
                candidate_report.get(field_name)
                == control_report.get(field_name)
                for field_name in (
                    "measured_audit_sha256",
                    "model_audit_sha256",
                    "source_tree_audit",
                )
            )
        )
        if not independently_recomputed_compute_match:
            raise ValueError("weak paired-comparison compute match did not reproduce")
        if not independently_recomputed_lineage_match:
            raise ValueError("weak paired-comparison lineage match did not reproduce")

        component_weights = {
            str(name): float(value)
            for name, value in dict(
                registered_gate.get("static_component_weights") or {}
            ).items()
        }
        candidate_selection = dict(candidate_report.get("selection") or {})
        control_selection = dict(control_report.get("selection") or {})
        recomputed_components: Dict[str, Dict[str, float]] = {}
        for name, weight in sorted(component_weights.items()):
            candidate_value = float(candidate_selection[name])
            control_value = float(control_selection[name])
            recomputed_components[name] = {
                "weak_on": candidate_value,
                "weak_off_control": control_value,
                "paired_gain": candidate_value - control_value,
                "weight": weight,
            }
        recomputed_static_gain = float(sum(
            value["paired_gain"] * value["weight"]
            for value in recomputed_components.values()
        ))
        recomputed_joint_gain = float(
            candidate_selection["joint_selection_score"]
            - control_selection["joint_selection_score"]
        )
        recomputed_maximum_delta = max(
            abs(value["paired_gain"])
            for value in recomputed_components.values()
        )
        recomputed_improved_count = sum(
            value["paired_gain"] > 0.0
            for value in recomputed_components.values()
        )
        recomputed_metrics = {
            "static_components": recomputed_components,
            "static_composite_gain": recomputed_static_gain,
            "joint_selection_score_gain": recomputed_joint_gain,
            "maximum_absolute_static_component_delta": recomputed_maximum_delta,
            "improved_static_component_count": recomputed_improved_count,
        }
        recomputed_result_checks = {
            "candidate_development_negative_control_passed": bool(
                candidate_stage.get(
                    "relational_state_pair_supervision_enabled"
                ) is True
                and candidate_stage.get("weak_validation_gate_passed") is True
                and candidate_stage.get("weak_branch_eligible_for_selection")
                is True
                and candidate_eligibility.get("selection_eligible") is True
            ),
            "control_is_diagnostic_only": bool(
                control_eligibility.get("selection_eligible") is False
                and control_eligibility.get("reason")
                == "compute_matched_non_relational_control_is_diagnostic_only"
            ),
            "checkpoint_response_is_non_degenerate": bool(
                candidate_checkpoint_sha256 != control_checkpoint_sha256
                if registered_gate.get("require_checkpoint_hash_difference")
                is True
                else True
            ),
            "static_component_response_is_non_degenerate": bool(
                recomputed_maximum_delta
                >= float(
                    registered_gate[
                        "minimum_absolute_static_component_delta"
                    ]
                )
            ),
            "minimum_improved_static_component_count_passed": bool(
                recomputed_improved_count
                >= int(
                    registered_gate[
                        "minimum_improved_static_component_count"
                    ]
                )
            ),
            "static_composite_gain_passed": bool(
                recomputed_static_gain
                > float(
                    registered_gate[
                        "minimum_static_composite_gain_exclusive"
                    ]
                )
            ),
            "joint_selection_non_regression_passed": bool(
                recomputed_joint_gain
                >= float(
                    registered_gate["minimum_joint_selection_score_gain"]
                )
            ),
        }
        result_checks = dict(artifact.get("result_checks") or {})
        structural_checks = dict(artifact.get("structural_checks") or {})
        evidence_passed = bool(
            artifact.get("status") == "passed"
            and artifact.get("gate_contract") == registered_gate
            and str(artifact.get("gate_contract_sha256", ""))
            == _stable_json_hash(registered_gate)
            and structural_checks
            and all(value is True for value in structural_checks.values())
            and result_checks
            and all(value is True for value in result_checks.values())
            and result_checks == recomputed_result_checks
            and dict(artifact.get("metrics") or {}) == recomputed_metrics
            and dict(artifact.get("negative_control_evidence") or {}).get(
                "candidate_weak_validation_gate_passed"
            ) is True
        )
        if not evidence_passed:
            raise ValueError(
                "weak paired-comparison failed its preregistered gain, "
                "non-degeneracy, or negative-control gate"
            )
        covered_checkpoints.add(candidate_checkpoint_sha256)
        artifact_by_seed[str(candidate_row["seed"])] = {
            "source_path": str(artifact_path),
            "file_sha256": _file_sha256(artifact_path),
            "artifact_sha256": str(artifact["artifact_sha256"]),
            "candidate_checkpoint_sha256": candidate_checkpoint_sha256,
            "control_checkpoint_sha256": control_checkpoint_sha256,
            "static_composite_gain": float(
                dict(artifact.get("metrics") or {})["static_composite_gain"]
            ),
            "joint_selection_score_gain": float(
                dict(artifact.get("metrics") or {})[
                    "joint_selection_score_gain"
                ]
            ),
            "negative_control_passed": True,
            "preregistered_gate_passed": True,
        }
    missing = sorted(set(expected_by_checkpoint).difference(covered_checkpoints))
    if missing or len(artifact_by_seed) != len(rows):
        raise ValueError(
            "weak paired-comparison artifacts do not cover every ensemble seed"
        )
    value = {
        "schema": "v8_weak_supervision_paired_comparison_gate_v1",
        "required": True,
        "status": "passed_every_seed_hash_bound_matched_control",
        "passed": True,
        "covered_seed_count": len(artifact_by_seed),
        "required_seed_count": len(rows),
        "artifact_by_seed": artifact_by_seed,
        "all_negative_controls_passed": True,
        "all_preregistered_non_degeneracy_and_gain_gates_passed": True,
    }
    value["gate_sha256"] = _stable_json_hash(value)
    return value


def _aggregate_conditions(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    aggregate: Dict[str, Any] = {}
    for condition_name in sorted(rows[0]["conditions"]):
        aggregate[condition_name] = {}
        for metric_name in (
            "total_gain_mean",
            "total_gain_lcb95",
            "improved_fraction",
            "serious_harm_fraction",
            "serious_harm_wilson_upper_95",
        ):
            values = np.asarray([
                row["conditions"][condition_name][metric_name] for row in rows
            ], dtype=np.float64)
            aggregate[condition_name][metric_name] = {
                "mean": float(np.mean(values)),
                "sample_std": float(np.std(values, ddof=1)),
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
            }
    return aggregate


def _nested_value(value: Mapping[str, Any], path: str) -> Any:
    current: Any = value
    for key in path.split("."):
        if not isinstance(current, Mapping) or key not in current:
            raise KeyError(path)
        current = current[key]
    return current


def _load_static_comparison_report(path_value: str | Path) -> Tuple[Path, Dict[str, Any]]:
    path = Path(path_value).resolve()
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema") != COMPARISON_SCHEMA:
        raise ValueError(f"unsupported V7/V8 comparison report: {path}")
    expected_hash = _stable_json_hash({
        key: value for key, value in report.items() if key != "report_sha256"
    })
    if report.get("report_sha256") != expected_hash:
        raise ValueError(f"V7/V8 comparison report hash mismatch: {path}")
    return path, report


def _measured_static_comparison_gate(
    comparison_report_paths: Optional[Sequence[str | Path]],
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Require one hash-matched frozen V7 comparison for every ensemble member."""

    if not comparison_report_paths:
        value = {
            "schema": "v8_measured_static_comparison_gate_v1",
            "status": "not_evaluated",
            "passed": False,
            "required_schema": COMPARISON_SCHEMA,
            "required_member_checkpoint_sha256": sorted(
                str(row["checkpoint_sha256"]) for row in rows
            ),
            "covered_member_checkpoint_sha256": [],
            "missing_member_checkpoint_sha256": sorted(
                str(row["checkpoint_sha256"]) for row in rows
            ),
            "static_shadow_ready": False,
            "reason": "frozen V7 development comparison was not supplied",
        }
        value["gate_sha256"] = _stable_json_hash(value)
        return value

    required_hashes = {
        str(row["checkpoint_sha256"]) for row in rows
    }
    audits: Dict[str, Any] = {}
    baseline_hashes = set()
    evaluation_plan_hashes = set()
    duplicate_hashes = set()
    for path, report in map(_load_static_comparison_report, comparison_report_paths):
        v8_checkpoint = dict(report.get("checkpoint", {}).get("v8") or {})
        v7_checkpoint = dict(report.get("checkpoint", {}).get("v7") or {})
        v8_hash = str(v8_checkpoint.get("file_sha256", ""))
        if v8_hash in audits:
            duplicate_hashes.add(v8_hash)
        provenance = dict(report.get("provenance") or {})
        isolation = dict(report.get("development_isolation") or {})
        report_status = str(report.get("status", ""))
        structural_checks = {
            "completed_runtime_comparable_status": report_status.startswith(
                "completed_runtime_comparable"
            ),
            "comparable": report.get("comparable") is True,
            "runtime_pairing_comparable": (
                report.get("runtime_pairing_comparable") is True
            ),
            "runtime_data_config_split_hash_validation_passed": (
                provenance.get(
                    "runtime_data_config_split_hash_validation_passed"
                ) is True
            ),
            "same_manifest_for_both_models": (
                provenance.get("same_manifest_for_both_models") is True
            ),
            "same_runtime_split_for_both_models": (
                provenance.get("same_runtime_split_for_both_models") is True
            ),
            "v7_weights_fixed": (
                v7_checkpoint.get("weights_fixed") is True
                and v7_checkpoint.get("state_dict_sha256_before")
                == v7_checkpoint.get("state_dict_sha256_after")
            ),
            "v8_weights_fixed": (
                v8_checkpoint.get("weights_fixed") is True
                and v8_checkpoint.get("state_dict_sha256_before")
                == v8_checkpoint.get("state_dict_sha256_after")
            ),
            "development_only": report.get("evaluation_scope") == "development_only",
            "locked_split_rows_selected_zero": int(
                isolation.get("locked_split_rows_selected", -1)
            ) == 0,
            "locked_measurement_payloads_not_loaded": (
                isolation.get("locked_measurement_payloads_loaded") is False
            ),
            "locked_final_evaluator_not_invoked": (
                isolation.get("locked_final_evaluator_invoked") is False
            ),
            "locked_synthetic_family_not_generated": (
                isolation.get("locked_synthetic_family_generated") is False
            ),
            "allowed_for_model_selection": (
                isolation.get("allowed_for_model_selection") is True
            ),
            "member_checkpoint_hash_registered": v8_hash in required_hashes,
        }
        primary_metric_checks: Dict[str, Any] = {}
        primary_paths = dict(report.get("primary_metric_paths") or {})
        for system_type in ("3qubit", "4qubit"):
            system_paths = dict(primary_paths.get(system_type) or {})
            for metric_name in REQUIRED_MEASURED_STATIC_METRICS:
                metric_path = str(system_paths.get(metric_name, ""))
                try:
                    metric = dict(_nested_value(report, metric_path))
                except (KeyError, TypeError, ValueError):
                    metric = {}
                primary_metric_checks[f"{system_type}.{metric_name}"] = {
                    "metric_path": metric_path,
                    "minimum_groups_passed": (
                        metric.get("minimum_groups_passed") is True
                    ),
                    "claim_supported": metric.get("claim_supported") is True,
                    "paired_improvement": metric.get("paired_improvement"),
                    "lcb_95": metric.get("lcb_95"),
                    "passed": bool(
                        metric.get("minimum_groups_passed") is True
                        and metric.get("claim_supported") is True
                        and float(metric.get("lcb_95", float("-inf"))) >= 0.0
                    ),
                }

        weakest_cell_checks: Dict[str, Any] = {}
        measured = dict(report.get("measured") or {})
        for system_type in ("3qubit", "4qubit"):
            conditions = dict(
                dict(measured.get(system_type) or {}).get("conditions") or {}
            )
            for condition_name, condition_value in sorted(conditions.items()):
                if condition_name == "all_development_validation":
                    continue
                strata = dict(condition_value or {})
                for rho_stratum, metric_names in (
                    (
                        "all",
                        ("nb2_nll", "event_count_ge_1_ece", "event_count_ge_2_ece"),
                    ),
                    ("rho_available", ("rho_fidelity",)),
                ):
                    stratum = dict(strata.get(rho_stratum) or {})
                    if stratum.get("status") != "evaluated":
                        weakest_cell_checks[
                            f"{system_type}.{condition_name}.{rho_stratum}"
                        ] = {
                            "passed": False,
                            "reason": "required measured stratum not evaluated",
                        }
                        continue
                    metrics = dict(stratum.get("metrics") or {})
                    for metric_name in metric_names:
                        metric = dict(metrics.get(metric_name) or {})
                        check_name = (
                            f"{system_type}.{condition_name}.{rho_stratum}.{metric_name}"
                        )
                        weakest_cell_checks[check_name] = {
                            "minimum_groups_passed": (
                                metric.get("minimum_groups_passed") is True
                            ),
                            "lcb_95": metric.get("lcb_95"),
                            "passed": bool(
                                metric.get("minimum_groups_passed") is True
                                and float(metric.get("lcb_95", float("-inf"))) >= 0.0
                            ),
                        }

        structural_passed = bool(all(structural_checks.values()))
        primary_passed = bool(
            primary_metric_checks
            and all(value["passed"] for value in primary_metric_checks.values())
        )
        weakest_cell_passed = bool(
            weakest_cell_checks
            and all(value["passed"] for value in weakest_cell_checks.values())
        )
        audits[v8_hash] = {
            "path": str(path),
            "report_sha256": str(report["report_sha256"]),
            "v7_checkpoint_sha256": str(v7_checkpoint.get("file_sha256", "")),
            "v8_checkpoint_sha256": v8_hash,
            "historical_training_provenance_comparable": bool(
                report.get("historical_training_provenance_comparable", False)
            ),
            "historical_provenance_required_for_fresh_runtime_gate": False,
            "structural_checks": structural_checks,
            "primary_measured_static_metric_checks": primary_metric_checks,
            "weakest_measured_cell_checks": weakest_cell_checks,
            "structural_passed": structural_passed,
            "primary_measured_static_passed": primary_passed,
            "weakest_measured_cell_non_regression_passed": weakest_cell_passed,
            "passed": bool(
                structural_passed and primary_passed and weakest_cell_passed
            ),
        }
        baseline_hashes.add(str(v7_checkpoint.get("file_sha256", "")))
        evaluation_plan_hashes.add(_stable_json_hash({
            "evaluation_seed": report.get("evaluation_seed"),
            "comparison_config": report.get("comparison_config"),
            "common_random_number_plan": report.get("common_random_number_plan"),
        }))

    covered_hashes = required_hashes.intersection(audits)
    missing_hashes = required_hashes.difference(audits)
    extra_hashes = set(audits).difference(required_hashes)
    passed = bool(
        not duplicate_hashes
        and not missing_hashes
        and not extra_hashes
        and len(baseline_hashes) == 1
        and len(evaluation_plan_hashes) == 1
        and all(audits[value]["passed"] for value in required_hashes)
    )
    value = {
        "schema": "v8_measured_static_comparison_gate_v1",
        "status": "evaluated_comparable" if passed else "evaluated_fail_closed",
        "passed": passed,
        "required_member_checkpoint_sha256": sorted(required_hashes),
        "covered_member_checkpoint_sha256": sorted(covered_hashes),
        "missing_member_checkpoint_sha256": sorted(missing_hashes),
        "extra_member_checkpoint_sha256": sorted(extra_hashes),
        "duplicate_member_checkpoint_sha256": sorted(duplicate_hashes),
        "single_frozen_v7_baseline_required": True,
        "v7_baseline_checkpoint_sha256": (
            next(iter(baseline_hashes)) if len(baseline_hashes) == 1 else None
        ),
        "common_evaluation_plan_required": True,
        "comparison_report_by_v8_checkpoint_sha256": audits,
        "static_shadow_ready": passed,
    }
    value["gate_sha256"] = _stable_json_hash(value)
    return value


def finalize_feedback_candidate(
    report_paths: Sequence[str | Path],
    *,
    output_root: str | Path,
    comparison_report_paths: Optional[Sequence[str | Path]] = None,
    weak_comparison_report_paths: Optional[Sequence[str | Path]] = None,
) -> Dict[str, Any]:
    if len(report_paths) < MINIMUM_SEED_COUNT:
        raise ValueError(
            f"V8 finalization requires at least {MINIMUM_SEED_COUNT} independent seeds"
        )
    reports = [_load_report(value) for value in report_paths]
    for _, report in reports:
        _require_development_selection_eligible(report)
    profile_names = {str(report["profile"]["name"]) for _, report in reports}
    ablation_ids = {str(report["profile"]["ablation_id"]) for _, report in reports}
    if len(profile_names) != 1 or len(ablation_ids) != 1:
        raise ValueError("multi-seed reports must share one profile and ablation")
    seeds = [int(report["seed"]) for _, report in reports]
    if len(set(seeds)) != len(seeds):
        raise ValueError("multi-seed reports contain duplicate seeds")

    invariant_fields = (
        "config_sha256",
        "split_hash_by_system",
        "measured_audit_sha256",
        "model_audit_sha256",
        "contract_audit_sha256",
        "source_tree_audit",
        "locked_final_test",
    )
    for field_name in invariant_fields:
        hashes = {
            _stable_json_hash(report[field_name])
            for _, report in reports
        }
        if len(hashes) != 1:
            raise ValueError(f"multi-seed {field_name} differs")

    rows: List[Dict[str, Any]] = []
    checkpoint_paths: Dict[int, Path] = {}
    checkpoints_by_seed: Dict[int, Dict[str, Any]] = {}
    report_paths_by_seed: Dict[int, Path] = {}
    for report_path, report in reports:
        seed = int(report["seed"])
        checkpoint_path, checkpoint_hash, checkpoint = _checkpoint_for_report(report)
        checkpoint_paths[seed] = checkpoint_path
        checkpoints_by_seed[seed] = checkpoint
        report_paths_by_seed[seed] = report_path
        conditions = _closed_loop_conditions(report)
        condition_rows = {
            name: {
                "total_gain_mean": float(value["total_gain_mean"]),
                "total_gain_lcb95": float(value["total_gain_lcb95"]),
                "improved_fraction": float(value["improved_fraction"]),
                "serious_harm_fraction": float(value["serious_harm_fraction"]),
                "serious_harm_wilson_upper_95": float(
                    value["serious_harm_wilson_upper_95"]
                ),
            }
            for name, value in conditions.items()
        }
        minimum_lcb = min(value["total_gain_lcb95"] for value in condition_rows.values())
        maximum_harm = max(
            value["serious_harm_wilson_upper_95"] for value in condition_rows.values()
        )
        peak_gib = float(report["peak_cuda_memory_bytes"]) / float(1024 ** 3)
        model_config = dict(checkpoint["model_config"])
        causal_branch = bool(
            model_config.get("target_context_enabled", False)
            or model_config.get("world_history_conditioning_enabled", False)
        )
        causal_gate_passed = bool(
            report["causal_context_gate"].get("causal_context_gate", False)
        )
        confidence_state = dict(report["uncertainty_calibration"])
        confidence_fail_closed = bool(confidence_state.get("abstain", True))
        weak_validation_state = _weak_validation_state(report)
        static_seed_gate_passed = bool(
            minimum_lcb > MINIMUM_TOTAL_GAIN_LCB95
            and maximum_harm <= MAX_SERIOUS_HARM_WILSON_UPPER_95
            and peak_gib <= MAX_PEAK_RESERVED_GIB
            and weak_validation_state["passed"]
        )
        action_seed_gate_passed = bool(
            static_seed_gate_passed
            and (not causal_branch or causal_gate_passed)
            and not confidence_fail_closed
        )
        rows.append({
            "seed": seed,
            "report_path": str(report_path),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_hash,
            "joint_selection_score": float(report["selection"]["joint_selection_score"]),
            "minimum_condition_total_gain_lcb95": minimum_lcb,
            "maximum_condition_serious_harm_wilson_upper_95": maximum_harm,
            "peak_cuda_memory_gib": peak_gib,
            "conditions": condition_rows,
            "causal_branch": causal_branch,
            "causal_context_gate_passed": causal_gate_passed,
            "confidence_fail_closed": confidence_fail_closed,
            "weak_validation_state": weak_validation_state,
            "static_seed_gate_passed": static_seed_gate_passed,
            "action_seed_gate_passed": action_seed_gate_passed,
        })
    weak_paired_comparison_gate = _weak_paired_comparison_gate(
        weak_comparison_report_paths,
        reports,
        rows,
    )
    for row in rows:
        paired_passed = bool(
            not weak_paired_comparison_gate["required"]
            or str(row["seed"])
            in weak_paired_comparison_gate["artifact_by_seed"]
        )
        row["weak_paired_comparison_passed"] = paired_passed
        row["static_seed_gate_passed"] = bool(
            row["static_seed_gate_passed"] and paired_passed
        )
        row["action_seed_gate_passed"] = bool(
            row["action_seed_gate_passed"] and paired_passed
        )
    aggregate = _aggregate_conditions(rows)
    ensemble_minimum_gain_lcb95 = min(
        row["minimum_condition_total_gain_lcb95"] for row in rows
    )
    ensemble_maximum_harm = max(
        row["maximum_condition_serious_harm_wilson_upper_95"] for row in rows
    )
    ensemble_maximum_peak_gib = max(row["peak_cuda_memory_gib"] for row in rows)
    static_development_gate_passed = bool(
        ensemble_minimum_gain_lcb95 > MINIMUM_TOTAL_GAIN_LCB95
        and ensemble_maximum_harm <= MAX_SERIOUS_HARM_WILSON_UPPER_95
        and ensemble_maximum_peak_gib <= MAX_PEAK_RESERVED_GIB
        and all(row["static_seed_gate_passed"] for row in rows)
        and weak_paired_comparison_gate["passed"]
    )
    if not static_development_gate_passed:
        raise ValueError(
            "multi-seed ensemble fails the worst-seed development, safety, or VRAM gate"
        )
    static_comparison_gate = _measured_static_comparison_gate(
        comparison_report_paths, rows
    )
    static_candidate_eligible = bool(
        static_development_gate_passed
        and static_comparison_gate["status"] == "evaluated_comparable"
        and static_comparison_gate["passed"]
    )
    causal_branch_present = any(row["causal_branch"] for row in rows)
    real_causal_gate_passed_for_all_members = all(
        (not row["causal_branch"]) or row["causal_context_gate_passed"]
        for row in rows
    )
    real_uncertainty_calibrated_for_all_members = all(
        not row["confidence_fail_closed"] for row in rows
    )
    ensemble_epistemic_enabled = all(
        bool(checkpoints_by_seed[row["seed"]]["model_config"].get(
            "ensemble_epistemic_enabled", False
        ))
        for row in rows
    )
    action_policy_eligible = bool(
        static_candidate_eligible
        and len(rows) >= MINIMUM_SEED_COUNT
        and ensemble_epistemic_enabled
        and real_causal_gate_passed_for_all_members
        and real_uncertainty_calibrated_for_all_members
    )
    median_selection_score = float(np.median([
        row["joint_selection_score"] for row in rows
    ]))
    representative = min(
        rows,
        key=lambda row: (
            abs(row["joint_selection_score"] - median_selection_score),
            row["seed"],
        ),
    )
    representative_seed = int(representative["seed"])
    ensemble_identity = {
        "profile": next(iter(profile_names)),
        "ablation_id": next(iter(ablation_ids)),
        "members": [
            {
                "seed": int(row["seed"]),
                "checkpoint_sha256": str(row["checkpoint_sha256"]),
                "weight": 1.0 / float(len(rows)),
            }
            for row in sorted(rows, key=lambda item: item["seed"])
        ],
    }
    ensemble_sha256 = _stable_json_hash(ensemble_identity)
    version_name = f"ensemble_{len(rows)}seed_{ensemble_sha256[:12]}"
    version_dir = Path(output_root).resolve() / version_name
    version_dir.mkdir(parents=True, exist_ok=True)
    if weak_paired_comparison_gate["required"]:
        comparison_dir = version_dir / "weak_paired_comparisons"
        comparison_dir.mkdir(parents=True, exist_ok=True)
        for seed_text, audit in sorted(
            weak_paired_comparison_gate["artifact_by_seed"].items(),
            key=lambda item: int(item[0]),
        ):
            source = Path(audit["source_path"]).resolve()
            target = comparison_dir / f"seed_{int(seed_text)}.json"
            temporary = target.with_name(f"{target.name}.tmp")
            shutil.copy2(source, temporary)
            os.replace(temporary, target)
            if _file_sha256(target) != str(audit["file_sha256"]):
                raise RuntimeError("frozen weak paired-comparison artifact changed")
            audit["frozen_path"] = str(target)
            audit["frozen_file_sha256"] = _file_sha256(target)
        weak_paired_comparison_gate["gate_sha256"] = _stable_json_hash({
            key: value for key, value in weak_paired_comparison_gate.items()
            if key != "gate_sha256"
        })
    frozen_members: List[Dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: item["seed"]):
        seed = int(row["seed"])
        member_dir = version_dir / "members" / f"seed_{seed}"
        member_dir.mkdir(parents=True, exist_ok=True)
        frozen_checkpoint_path = member_dir / "feedback_model_bundle.pt"
        frozen_report_path = member_dir / "training_report.json"
        for source, target in (
            (checkpoint_paths[seed], frozen_checkpoint_path),
            (report_paths_by_seed[seed], frozen_report_path),
        ):
            temporary = target.with_name(f"{target.name}.tmp")
            shutil.copy2(source, temporary)
            os.replace(temporary, target)
        if _file_sha256(frozen_checkpoint_path) != row["checkpoint_sha256"]:
            raise RuntimeError("frozen V8 ensemble member hash changed")
        frozen_members.append({
            "seed": seed,
            "weight": 1.0 / float(len(rows)),
            "checkpoint_path": str(frozen_checkpoint_path),
            "checkpoint_sha256": str(row["checkpoint_sha256"]),
            "training_report_path": str(frozen_report_path),
            "training_report_sha256": _file_sha256(frozen_report_path),
            "static_seed_gate_passed": bool(row["static_seed_gate_passed"]),
            "action_seed_gate_passed": bool(row["action_seed_gate_passed"]),
            "weak_paired_comparison": copy.deepcopy(
                weak_paired_comparison_gate["artifact_by_seed"].get(
                    str(seed)
                )
            ),
        })

    representative_report = next(
        report for _, report in reports if int(report["seed"]) == representative_seed
    )
    confidence = {
        "schema": "v8_ensemble_real_episode_uncertainty_state_v1",
        "ensemble_member_count": len(frozen_members),
        "true_multi_member_inference_required": True,
        "ensemble_epistemic_enabled": ensemble_epistemic_enabled,
        "real_episode_calibrated_for_all_members": (
            real_uncertainty_calibrated_for_all_members
        ),
        "member_state_sha256": {
            str(row["seed"]): _stable_json_hash(
                next(
                    report["uncertainty_calibration"]
                    for _, report in reports
                    if int(report["seed"]) == int(row["seed"])
                )
            )
            for row in rows
        },
        "abstain": not action_policy_eligible,
        "policy_execution_allowed": False,
        "promotion_ready": False,
        "hardware_action_authorized": False,
    }
    manifest = {
        "schema": FINALIZATION_SCHEMA,
        "candidate_stage": "development_ensemble_frozen_locked_test_still_sealed",
        "profile": next(iter(profile_names)),
        "ablation_id": next(iter(ablation_ids)),
        "seed_count": len(rows),
        "seeds": sorted(seeds),
        "selection_scope": "development_only",
        "selection_rule": (
            "freeze at least five independent seed members and require the "
            "worst seed-condition 95% gain LCB, harm bound and VRAM bound; "
            "a relational weak branch additionally requires one hash-bound "
            "compute-matched weak-off comparison per seed"
        ),
        "gates": {
            "minimum_independent_seed_count": MINIMUM_SEED_COUNT,
            "minimum_total_gain_lcb95_strictly_greater_than": MINIMUM_TOTAL_GAIN_LCB95,
            "maximum_serious_harm_wilson_upper_95": MAX_SERIOUS_HARM_WILSON_UPPER_95,
            "maximum_peak_cuda_memory_gib": MAX_PEAK_RESERVED_GIB,
            "causal_branch_requires_real_shuffle_control": True,
            "action_policy_requires_real_grouped_conformal": True,
            "action_confidence_requires_true_multi_member_inference": True,
            "static_shadow_requires_frozen_v7_measured_comparison": True,
            "static_shadow_requires_each_seed_comparison": True,
            "static_shadow_requires_3q_4q_nb2_ece_rho": True,
            "static_shadow_requires_weakest_measured_cell_non_regression": True,
            "relational_weak_branch_requires_development_validation_derangement": True,
            "relational_weak_branch_requires_every_seed_matched_control_gain": True,
        },
        "rows": sorted(rows, key=lambda row: row["seed"]),
        "aggregate": aggregate,
        "ensemble_worst_case": {
            "minimum_condition_total_gain_lcb95": ensemble_minimum_gain_lcb95,
            "maximum_condition_serious_harm_wilson_upper_95": ensemble_maximum_harm,
            "maximum_peak_cuda_memory_gib": ensemble_maximum_peak_gib,
        },
        "ensemble_sha256": ensemble_sha256,
        "ensemble_member_count": len(frozen_members),
        "ensemble_members": frozen_members,
        "ensemble_member_weights_sum": float(sum(
            member["weight"] for member in frozen_members
        )),
        "representative_member": {
            "purpose": "median_score_single_model_diagnostics_only",
            "not_used_for_ensemble_selection_or_confidence": True,
            "seed": representative_seed,
            "joint_selection_score": float(representative["joint_selection_score"]),
            "checkpoint_sha256": str(representative["checkpoint_sha256"]),
        },
        "config_sha256": representative_report["config_sha256"],
        "split_hash_by_system": representative_report["split_hash_by_system"],
        "source_tree_audit": representative_report["source_tree_audit"],
        "measured_audit_sha256": representative_report["measured_audit_sha256"],
        "model_audit_sha256": representative_report["model_audit_sha256"],
        "contract_audit_sha256": representative_report["contract_audit_sha256"],
        "locked_final_test": representative_report["locked_final_test"],
        "causal_context_gate": {
            "all_causal_members_passed_real_gate": (
                real_causal_gate_passed_for_all_members
            ),
            "causal_branch_present": causal_branch_present,
            "member_gate_sha256": {
                str(row["seed"]): _stable_json_hash(
                    next(
                        report["causal_context_gate"]
                        for _, report in reports
                        if int(report["seed"]) == int(row["seed"])
                    )
                )
                for row in rows
            },
        },
        "uncertainty_calibration": confidence,
        "static_development_gate_passed": static_development_gate_passed,
        "weak_supervision_paired_comparison_gate": (
            weak_paired_comparison_gate
        ),
        "static_comparison_gate": static_comparison_gate,
        "static_candidate_eligible": static_candidate_eligible,
        "action_policy_eligible": action_policy_eligible,
        "static_shadow_ready": static_candidate_eligible,
        "action_shadow_ready": action_policy_eligible,
        "shadow_ready": action_policy_eligible,
        "inference_abstains": not action_policy_eligible,
        "proposal_behavior": "bounded_proposal" if action_policy_eligible else "forced_no_op_abstain",
        "analysis_candidate_ready": static_candidate_eligible,
        "frozen_inference": True,
        "online_training_allowed": False,
        "research_only": True,
        "promotion_ready": False,
        "hardware_action_authorized": False,
        "live_gates_complete": False,
        "evidence_limit": (
            "This freeze retains every seed member before the one-time locked final "
            "test. Static/read-only shadow use is separate from action-policy use. "
            "Without real paired shuffle evidence, real grouped conformal calibration "
            "and true multi-member inference, proposals are forced to no-op/abstain."
        ),
    }
    manifest["manifest_hash"] = _stable_json_hash(manifest)
    manifest_path = version_dir / "frozen_candidate_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    pointer = {
        "schema": POINTER_SCHEMA,
        "version_manifest_path": str(manifest_path),
        "version_manifest_sha256": _file_sha256(manifest_path),
        "ensemble_sha256": ensemble_sha256,
        "ensemble_member_count": len(frozen_members),
        "ensemble_members": [
            {
                "seed": member["seed"],
                "weight": member["weight"],
                "checkpoint_path": member["checkpoint_path"],
                "checkpoint_sha256": member["checkpoint_sha256"],
            }
            for member in frozen_members
        ],
        "representative_seed_for_diagnostics_only": representative_seed,
        "candidate_stage": manifest["candidate_stage"],
        "static_shadow_ready": manifest["static_shadow_ready"],
        "action_shadow_ready": manifest["action_shadow_ready"],
        "inference_abstains": manifest["inference_abstains"],
        "promotion_ready": False,
        "hardware_action_authorized": False,
    }
    pointer["pointer_hash"] = _stable_json_hash(pointer)
    pointer_path = Path(output_root).resolve() / "selected_candidate.json"
    temporary_pointer = pointer_path.with_name(f"{pointer_path.name}.tmp")
    temporary_pointer.write_text(
        json.dumps(pointer, indent=2, sort_keys=True), encoding="utf-8"
    )
    os.replace(temporary_pointer, pointer_path)
    return manifest


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="append", required=True)
    parser.add_argument(
        "--comparison-report",
        action="append",
        help="One frozen V7/V8 development comparison per ensemble seed",
    )
    parser.add_argument(
        "--weak-comparison-report",
        action="append",
        help="One hash-bound weak-on/off paired comparison per ensemble seed",
    )
    parser.add_argument(
        "--output-root",
        default="ml_recon/models/physics_hybrid_v8_feedback_candidate",
    )
    args = parser.parse_args(argv)
    manifest = finalize_feedback_candidate(
        args.report,
        output_root=args.output_root,
        comparison_report_paths=args.comparison_report,
        weak_comparison_report_paths=args.weak_comparison_report,
    )
    print(json.dumps({
        "event": "v8_feedback_candidate_frozen",
        "ensemble_member_count": manifest["ensemble_member_count"],
        "representative_seed_for_diagnostics_only": manifest[
            "representative_member"
        ]["seed"],
        "candidate_stage": manifest["candidate_stage"],
        "inference_abstains": manifest["inference_abstains"],
        "promotion_ready": False,
        "hardware_action_authorized": False,
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "FINALIZATION_SCHEMA",
    "POINTER_SCHEMA",
    "COMPARISON_SCHEMA",
    "MINIMUM_SEED_COUNT",
    "finalize_feedback_candidate",
    "main",
]
