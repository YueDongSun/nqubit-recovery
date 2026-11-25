"""Lightweight Track B history-gain metadata, separate from model execution.

The result's consumed input is always the evaluation-time zero-gain recipe.
These labels describe the model's training input relative to that recipe; they
are not evidence that a checkpoint was executed or that a scientific Gate ran.
"""

from __future__ import annotations


LEGACY_ORACLE_HISTORY_GAIN_INPUT = "oracle_history_gain_v0"
UNOBSERVED_ZERO_HISTORY_GAIN_INPUT = "unobserved_zero_v1"
TRAINING_GAIN_INPUT_UNKNOWN = "TRAINING_GAIN_INPUT_UNKNOWN"

LEGACY_ORACLE_GAIN_SHIFT = (
    "OBSERVABLE_ONLY_DIAGNOSTIC_SHIFT_FROM_ORACLE_GAIN_TRAINED_PREFIXES;"
    "EXACT_GAIN_REPLACED_BY_ZERO;FRESH_VERSIONED_OBSERVATIONS;NOT_DEPLOYMENT_VALIDATION"
)
UNOBSERVED_ZERO_GAIN_SHIFT = "UNOBSERVED_ZERO_GAIN_INPUT_MATCHED_LEARNER_STATE_SHIFT_REMAINS"
ORACLE_GAIN_REPORT_SHIFT = "ORACLE_GAIN_TRAINING_TO_ZERO_GAIN_DIAGNOSTIC"


def history_gain_distribution_shift_kind(input_kind: str) -> str:
    """Return the nested sample label for a known training input contract."""
    if input_kind == TRAINING_GAIN_INPUT_UNKNOWN:
        return TRAINING_GAIN_INPUT_UNKNOWN
    if input_kind == LEGACY_ORACLE_HISTORY_GAIN_INPUT:
        return LEGACY_ORACLE_GAIN_SHIFT
    if input_kind == UNOBSERVED_ZERO_HISTORY_GAIN_INPUT:
        return UNOBSERVED_ZERO_GAIN_SHIFT
    raise ValueError("unsupported history gain input identity")


def report_history_distribution_shift_kind(input_kind: str) -> str:
    """Return the historically registered top-level report label."""
    if input_kind == LEGACY_ORACLE_HISTORY_GAIN_INPUT:
        return ORACLE_GAIN_REPORT_SHIFT
    if input_kind == UNOBSERVED_ZERO_HISTORY_GAIN_INPUT:
        return UNOBSERVED_ZERO_GAIN_SHIFT
    raise ValueError("self-rollout model has an unsupported history gain input identity")
