"""Pure outcome accounting for Track B self-rollout development diagnostics."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .candidate_action_evaluator import MATERIAL_NEGATIVE_GAIN_TOLERANCE, is_materially_negative_trackb_gain


branch_kinds = ("model", "zero", "teacher")
active_action_definition = "ANY_EXECUTED_FLOAT32_COORDINATE_NONZERO"
failure_schema = "TrackBSelfRolloutFailureV1"


def _validated_branch_arrays(branch: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Validate one complete trajectory before reducing its outcome."""
    steps = branch["steps"]
    if (len(steps) != 10 or type(branch["selected_round"]) is not int
            or branch["selected_round"] != 10):
        raise ValueError("self-rollout summary requires the tenth final action")
    if any(type(step["round_number"]) is not int or step["round_number"] != index
           for index, step in enumerate(steps, start=1)):
        raise ValueError("self-rollout steps must be in observed order")
    gains = np.asarray([step["synthetic_gain"] for step in steps], dtype=np.float64)
    initial = float(branch["initial_distance"])
    final = float(branch["final_distance"])
    total_gain = float(branch["telescoping_gain"])
    if (not np.isfinite(gains).all() or not np.isfinite([initial, final, total_gain]).all()
            or min(initial, final) < 0
            or not np.isclose(total_gain, initial - final, atol=1e-9, rtol=1e-9)
            or not np.isclose(gains.sum(), total_gain, atol=1e-9, rtol=1e-9)
            or not np.isclose(float(steps[-1]["after_distance"]), final, atol=1e-9, rtol=1e-9)):
        raise ValueError("self-rollout final outcome does not telescope")
    before_distances = np.asarray([step["before_distance"] for step in steps], dtype=np.float64)
    after_distances = np.asarray([step["after_distance"] for step in steps], dtype=np.float64)
    if (not np.isfinite(before_distances).all() or not np.isfinite(after_distances).all()
            or np.any(before_distances < 0) or np.any(after_distances < 0)
            or not np.allclose(gains, before_distances - after_distances, atol=1e-9, rtol=1e-9)
            or not np.allclose(after_distances[:-1], before_distances[1:], atol=1e-9, rtol=1e-9)
            or not np.isclose(before_distances[0], initial, atol=1e-9, rtol=1e-9)):
        raise ValueError("self-rollout summary requires continuous per-step distances")
    actions = np.asarray([step["action"] for step in steps])
    if (actions.dtype.kind not in "fiu" or actions.ndim != 3
            or actions.shape[0] != 10 or actions.shape[1] not in (3, 4)
            or actions.shape[2] != 2 or not np.isfinite(actions).all()):
        raise ValueError("self-rollout summary requires finite executed actions")
    active = np.any(actions != 0.0, axis=(1, 2))
    if np.any((np.abs(gains) > MATERIAL_NEGATIVE_GAIN_TOLERANCE) & ~active):
        raise ValueError("self-rollout no-op cannot have material gain")
    return gains, active


def summarize_self_rollout_samples(samples: list[Mapping[str, Any]]) -> dict[str, object]:
    """Use episodes for terminal risks and actually executed actions for activity."""
    if not samples:
        raise ValueError("self-rollout summary requires at least one episode")
    by_branch: dict[str, list[Mapping[str, Any]]] = {name: [] for name in branch_kinds}
    arrays: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {name: [] for name in branch_kinds}
    for sample in samples:
        branches = sample["branches"]
        if len(branches) != 3 or {branch["branch_kind"] for branch in branches} != set(by_branch):
            raise ValueError("self-rollout summary requires three distinct branches")
        for branch in branches:
            name = branch["branch_kind"]
            arrays[name].append(_validated_branch_arrays(branch))
            by_branch[name].append(branch)
    zero_final_distances = np.asarray([branch["final_distance"] for branch in by_branch["zero"]])
    summaries = {}
    for name, branches in by_branch.items():
        gains = np.asarray([branch["telescoping_gain"] for branch in branches], dtype=np.float64)
        final_distances = np.asarray([branch["final_distance"] for branch in branches], dtype=np.float64)
        step_gains = np.stack([value[0] for value in arrays[name]])
        active = np.stack([value[1] for value in arrays[name]])
        harmful = np.asarray([is_materially_negative_trackb_gain(value)
                              for value in step_gains.ravel()]).reshape(step_gains.shape)
        active_count = int(active.sum())
        negative_final_improvement_count = int(sum(
            is_materially_negative_trackb_gain(value) for value in gains
        ))
        negative_step_gain_count = int(harmful.sum())
        any_step_harm_episode_count = int(harmful.any(axis=1).sum())
        negative_active_count = int((harmful & active).sum())
        summaries[name] = {
            "sample_count": len(branches), "action_count": int(step_gains.size),
            "selected_final_round": 10,
            "mean_initial_distance": float(np.mean([branch["initial_distance"] for branch in branches])),
            "mean_final_distance": float(final_distances.mean()),
            "mean_final_improvement": float(gains.mean()),
            "median_final_improvement": float(np.median(gains)),
            "minimum_final_improvement": float(gains.min()),
            "negative_final_improvement_count": negative_final_improvement_count,
            "negative_final_improvement_rate": (negative_final_improvement_count / len(branches)),
            "negative_step_gain_count": negative_step_gain_count,
            "negative_step_gain_rate": (negative_step_gain_count / int(step_gains.size)),
            "any_step_harm_episode_count": any_step_harm_episode_count,
            "any_step_harm_episode_rate": (any_step_harm_episode_count / len(branches)),
            "active_action_definition": active_action_definition,
            "active_action_count": active_count,
            "negative_active_action_count": negative_active_count,
            "negative_active_step_gain_rate": (negative_active_count / active_count if active_count else None),
            "active_harm_status": "DEFINED" if active_count else "UNDEFINED_NO_ACTIVE_ACTIONS",
            "mean_gain_by_round": step_gains.mean(axis=0).tolist(),
            "paired_mean_final_improvement_over_zero": float(np.mean(zero_final_distances - final_distances)),
            "uncertainty_status": "DESCRIPTIVE_DEVELOPMENT_SAMPLE_NOT_CERTIFICATION",
        }
    return summaries


def build_self_rollout_failure(
    binding: Mapping[str, Any],
    *,
    intended_sample_ids: Sequence[str],
    attempted_sample_ids: Sequence[str],
    completed_sample_ids: Sequence[str],
    failure_stage: str,
    error_type: str,
    source_verification_at_completion: str,
) -> dict[str, object]:
    """Keep failed and unattempted episodes explicit; publish no performance ratio."""
    intended = list(intended_sample_ids)
    attempted = list(attempted_sample_ids)
    completed = list(completed_sample_ids)
    if (not intended or any(type(value) is not str or not value for value in intended)
            or len(set(intended)) != len(intended)
            or attempted != intended[:len(attempted)]
            or completed != attempted[:len(completed)]):
        raise ValueError("failure accounting requires unique ordered episode prefixes")
    if not failure_stage or not error_type:
        raise ValueError("failure stage and exception type are required")
    if source_verification_at_completion not in ("NOT_RECHECKED", "MATCHED", "MISMATCH_OR_UNAVAILABLE"):
        raise ValueError("unsupported failure source-verification status")
    return {
        "schema": failure_schema, "status": "FAILED", "binding": dict(binding),
        "failure_stage": failure_stage, "error_type": error_type,
        "source_verification_at_completion": source_verification_at_completion,
        "intended_sample_ids": intended, "attempted_sample_ids": attempted,
        "completed_sample_ids": completed,
        "failed_sample_ids": attempted[len(completed):],
        "not_attempted_sample_ids": intended[len(attempted):],
        "requested_count": len(intended), "attempted_count": len(attempted),
        "completed_count": len(completed), "failed_sample_count": len(attempted) - len(completed),
        "not_attempted_count": len(intended) - len(attempted),
        "performance_summary": None, "gate_decision": "NOT_PERFORMED",
        "diagnostic_only": True, "research_only": True, "selection_evidence": False,
        "promotion_ready": False, "hardware_action_authorized": False,
        "external_waveform_write": "NOT_IMPLEMENTED",
    }
