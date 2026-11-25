"""Admission of the existing V2.4 scalar READ denominator, never target inference."""

from __future__ import annotations

from typing import Any, Mapping

import torch


NATIVE_READ_ORIGIN = "V240_NATIVE_READ_COUNT_SCALAR"


def native_read_trials(batch: Mapping[str, Any]) -> torch.Tensor:
    """Read only the explicit native field; do not inspect query target values."""
    if batch.get("native_read_count_origin") != NATIVE_READ_ORIGIN:
        raise ValueError("native READ requires the admitted V2.4 scalar-field origin")
    value = batch.get("native_read_count_scalar")
    mask = batch["operation_mask"]
    if not isinstance(value, torch.Tensor) or value.dtype != torch.int64 \
            or value.shape != (mask.shape[0],) or bool((value < 0).any()):
        raise ValueError("native READ requires nonnegative int64 [batch] scalars")
    return value[:, None].expand_as(mask)


def validate_native_read_conservation(batch: Mapping[str, Any]) -> None:
    """Admission-only equality check against already-bound native cell content.

Historical row hashes already bind the full integer cells. The scalar is a
unique conserved total of those cells, so this check rejects its mutation
without changing historical row identities. The loader copies the scalar
from the independent source field; this check must never construct it.
"""
    trials = native_read_trials(batch)
    counts = batch["four_cell_counts"]
    available = batch["operation_mask"]
    if counts.dtype != torch.int64 or counts.shape != (*available.shape, 4) \
            or available.dtype != torch.bool:
        raise ValueError("native READ conservation requires int64 cells and bool availability")
    if not torch.equal(counts.sum(-1)[available], trials[available]):
        raise ValueError("native READ scalar differs from the bound native four-cell total")


def native_read_matched_baseline(batch: Mapping[str, Any]) -> dict[str, Any]:
    """B0_NATIVE_R_MATCHED: old probabilities with an explicit native denominator.

Old B0 remains unchanged. This predicts from visible marginals and the supplied
native field, never from hidden targets, without fitting any parameters.
"""
    from .evaluation import build_trackb_fixed_visible_marginal_prediction
    from .detected_count_moments import nested_four_cell_moments

    baseline = build_trackb_fixed_visible_marginal_prediction(batch)
    prediction = {name: baseline[name].double() for name in (
        "signal_probability", "coin_given_signal_probability",
        "idler_given_no_signal_probability", "four_cell_concentration",
    )}
    return {
        **prediction,
        **nested_four_cell_moments(
            read_trials=native_read_trials(batch),
            signal_probability=prediction["signal_probability"],
            coin_given_signal_probability=prediction["coin_given_signal_probability"],
            idler_given_no_signal_probability=prediction["idler_given_no_signal_probability"],
            concentration=prediction["four_cell_concentration"][:, None],
        ),
        "baseline_schema": "TrackBFixedVisibleMarginalNativeReadV1",
        "baseline_name": "B0_NATIVE_R_MATCHED",
    }
