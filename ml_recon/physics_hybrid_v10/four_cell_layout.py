"""Declared four-cell layouts and explicit Track A/Track B conversions."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import torch


FOUR_CELL_LAYOUT_SCHEMA = "V10FourCellLayoutV1"
TRACK_A_FOUR_CELL_SEMANTICS = "joint_signal_idler_00_idler_only_herald_only_11"
TRACK_B_NATIVE_FOUR_CELL_SEMANTICS = "joint_signal_idler_00_herald_only_idler_only_11"


@dataclass(frozen=True)
class Four_cell_layout:
    semantics: str
    cell_names: tuple[str, str, str, str]

    def sha256(self) -> str:
        payload = {
            "schema": FOUR_CELL_LAYOUT_SCHEMA,
            "semantics": self.semantics,
            "cell_names": self.cell_names,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


TRACK_A_FOUR_CELL_LAYOUT = Four_cell_layout(
    TRACK_A_FOUR_CELL_SEMANTICS, ("00", "idler_only", "herald_only", "11")
)
TRACK_B_NATIVE_FOUR_CELL_LAYOUT = Four_cell_layout(
    TRACK_B_NATIVE_FOUR_CELL_SEMANTICS, ("00", "herald_only", "idler_only", "11")
)
TRACK_A_FOUR_CELL_LAYOUT_SHA = TRACK_A_FOUR_CELL_LAYOUT.sha256()
TRACK_B_NATIVE_FOUR_CELL_LAYOUT_SHA = TRACK_B_NATIVE_FOUR_CELL_LAYOUT.sha256()

_LAYOUTS = {
    layout.semantics: layout
    for layout in (TRACK_A_FOUR_CELL_LAYOUT, TRACK_B_NATIVE_FOUR_CELL_LAYOUT)
}


def four_cell_layout_metadata(semantics: str) -> dict[str, str]:
    if not isinstance(semantics, str):
        raise ValueError("four-cell layout semantics must be a string")
    layout = _LAYOUTS.get(semantics)
    if layout is None:
        raise ValueError("four-cell layout semantics are undeclared")
    return {
        "four_cell_layout_schema": FOUR_CELL_LAYOUT_SCHEMA,
        "four_cell_semantics": layout.semantics,
        "four_cell_layout_sha": layout.sha256(),
    }


def require_four_cell_layout(
    metadata: Mapping[str, object], *, expected_semantics: str | None = None
) -> Four_cell_layout:
    if metadata.get("four_cell_layout_schema") != FOUR_CELL_LAYOUT_SCHEMA:
        raise ValueError("four-cell layout schema is required and must match")
    semantics = metadata.get("four_cell_semantics")
    if not isinstance(semantics, str):
        raise ValueError("four-cell layout semantics are required and must be a string")
    layout = _LAYOUTS.get(semantics)
    if layout is None:
        raise ValueError("four-cell layout semantics are required and must be declared")
    if metadata.get("four_cell_layout_sha") != layout.sha256():
        raise ValueError("four-cell layout SHA is required and must match semantics")
    if expected_semantics is not None and layout.semantics != expected_semantics:
        raise ValueError("four-cell layout does not match the required input contract")
    return layout


def adapt_four_cell_batch_layout(
    batch: Mapping[str, object], *, target_semantics: str
) -> dict[str, object]:
    """Explicitly convert a declared batch, retaining all non-count provenance."""
    source_layout = require_four_cell_layout(batch)
    if not isinstance(target_semantics, str):
        raise ValueError("four-cell conversion target semantics must be a string")
    target_layout = _LAYOUTS.get(target_semantics)
    if target_layout is None:
        raise ValueError("four-cell conversion target must be declared")
    counts = batch.get("four_cell_counts")
    if (
        not isinstance(counts, (np.ndarray, torch.Tensor))
        or counts.ndim < 1
        or counts.shape[-1] != 4
    ):
        raise ValueError("four-cell conversion requires a count tensor with final width four")
    allowed_keys = {
        "four_cell_counts",
        "four_cell_layout_schema",
        "four_cell_semantics",
        "four_cell_layout_sha",
        "operation_mask",
        "observed_operation_mask",
        "repeats_by_operation",
        "operation_ids",
        "source_provenance",
        "four_cell_layout_adapter",
        "four_cell_layout_adapter_history",
    }
    if set(batch) - allowed_keys:
        raise ValueError(
            "four-cell conversion accepts only its declared count-observation envelope"
        )
    if source_layout.semantics == target_layout.semantics:
        return dict(batch)
    if (
        source_layout.semantics == TRACK_B_NATIVE_FOUR_CELL_SEMANTICS
        and target_layout.semantics == TRACK_A_FOUR_CELL_SEMANTICS
    ):
        permutation = (0, 2, 1, 3)
        if isinstance(counts, torch.Tensor):
            indices = torch.tensor(permutation, device=counts.device)
            converted = counts.index_select(-1, indices)
        else:
            converted = counts[..., permutation]
    else:
        raise ValueError("four-cell conversion is not declared for these layouts")
    adapter_record = {
        "schema": "V10FourCellLayoutAdapterV1",
        "source_semantics": source_layout.semantics,
        "source_layout_sha": source_layout.sha256(),
        "target_semantics": target_layout.semantics,
        "target_layout_sha": target_layout.sha256(),
        "permutation": permutation,
    }
    prior_adapter = batch.get("four_cell_layout_adapter")
    output = {
        **batch,
        "four_cell_counts": converted,
        **four_cell_layout_metadata(target_layout.semantics),
    }
    output["four_cell_layout_adapter"] = adapter_record
    if prior_adapter is not None:
        output["four_cell_layout_adapter_history"] = (
            *tuple(batch.get("four_cell_layout_adapter_history", ())),
            prior_adapter,
            adapter_record,
        )
    return output


__all__ = [
    "FOUR_CELL_LAYOUT_SCHEMA",
    "TRACK_A_FOUR_CELL_SEMANTICS",
    "TRACK_A_FOUR_CELL_LAYOUT_SHA",
    "TRACK_B_NATIVE_FOUR_CELL_SEMANTICS",
    "TRACK_B_NATIVE_FOUR_CELL_LAYOUT_SHA",
    "adapt_four_cell_batch_layout",
    "four_cell_layout_metadata",
    "require_four_cell_layout",
]
