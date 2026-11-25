"""Fail-closed V2.4 physics, leakage, split, and readiness validators."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np


CLASSIFIER_FORBIDDEN_KEYS = {
    "rho_clean_re", "rho_clean_im", "clean_coefficients_re", "clean_coefficients_im",
    "current_coefficients_re", "current_coefficients_im", "fine_family",
    "partition_block_ids", "mechanism_severity", "oracle_action", "rng_seed",
}
FEEDBACK_STUDENT_FORBIDDEN_KEYS = CLASSIFIER_FORBIDDEN_KEYS | {
    "true_esm", "injected_delta_amplitude", "injected_delta_phase",
    "injected_error_norm", "future_measurement", "true_severity",
    "clean_measurement_tokens", "true_mechanism", "true_esm_mode_tokens",
    "true_partition",
}


def four_cell_counts(read: np.ndarray, signal: np.ndarray, idler: np.ndarray,
                     coin: np.ndarray) -> np.ndarray:
    read_array = np.asarray(read)
    signal_array = np.asarray(signal)
    idler_array = np.asarray(idler)
    coin_array = np.asarray(coin)
    if not (read_array.shape == signal_array.shape == idler_array.shape == coin_array.shape):
        raise ValueError("R/S/I/C arrays must have identical shape")
    if np.any(coin_array < 0) or np.any(coin_array > signal_array) \
            or np.any(coin_array > idler_array) \
            or np.any(signal_array + idler_array - coin_array > read_array):
        raise ValueError("R/S/I/C violate exact Four-cell invariants; clipping is forbidden")
    result = np.stack((read_array - signal_array - idler_array + coin_array,
                       signal_array - coin_array,
                       idler_array - coin_array,
                       coin_array), axis=-1)
    if np.any(result < 0):
        raise RuntimeError("Four-cell construction produced negative counts")
    return result


def validate_global_multi(values: np.ndarray, mask: np.ndarray,
                          scopes: Sequence[str]) -> None:
    array = np.asarray(values)
    valid = np.asarray(mask, dtype=np.bool_)
    if array.ndim != 2 or array.shape[1] != 9 or valid.shape != (array.shape[0],):
        raise ValueError("global_multi must be [rows,9] with one row mask")
    if not np.issubdtype(array.dtype, np.integer) or np.any(array < 0):
        raise ValueError("global multi counts must be non-negative integers")
    if any(scope not in ("state", "global") for scope in scopes):
        raise ValueError("global multi scope cannot be operation-local")
    if np.any(valid & (array[:, 1:7].sum(axis=1) != array[:, 0])):
        raise ValueError("global multi pair counts must sum exactly to multi2")


def validate_primary_count_likelihood(weights: Mapping[str, float]) -> None:
    if float(weights.get("nested_beta_binomial", 0.0)) <= 0.0:
        raise ValueError("primary nested beta-binomial objective must be active")
    forbidden = ("signal", "idler", "coin", "four_cell", "raw_coin_nb2")
    if any(float(weights.get(key, 0.0)) != 0.0 for key in forbidden):
        raise ValueError("standalone count objectives would double-count the primary event")


def validate_view_keys(view_name: str, keys: Iterable[str]) -> None:
    key_set = set(keys)
    forbidden = (CLASSIFIER_FORBIDDEN_KEYS if view_name.startswith("classifier")
                 else FEEDBACK_STUDENT_FORBIDDEN_KEYS)
    leaked = key_set & forbidden
    if leaked:
        raise ValueError(f"{view_name} exposes privileged truth: {sorted(leaked)}")


def validate_split_groups(rows: Iterable[Mapping[str, Any]]) -> Dict[str, int]:
    sample_splits: Dict[str, str] = {}
    group_splits: Dict[tuple[str, str], set[str]] = defaultdict(set)
    count = 0
    for row in rows:
        sample_id = str(row["sample_id"])
        split = str(row["split"])
        if sample_id in sample_splits and sample_splits[sample_id] != split:
            raise ValueError("sample ID crosses splits")
        sample_splits[sample_id] = split
        for field in ("session_id", "acquisition_run_id", "repeat_group_id", "trajectory_id"):
            value = str(row.get(field, ""))
            if value:
                group_splits[(field, value)].add(split)
        count += 1
    leaking = [key for key, splits in group_splits.items() if len(splits) > 1]
    if leaking:
        raise ValueError(f"session/trajectory/repeat groups cross splits: {leaking[:5]}")
    return {"rows": count, "unique_samples": len(sample_splits),
            "groups": len(group_splits)}


def validate_partition_labels(block_ids: np.ndarray, depth: int, intactness: int,
                              exact: bool) -> None:
    labels = np.asarray(block_ids, dtype=np.int64).reshape(-1)
    if labels.size < 3 or np.any(labels < 0):
        raise ValueError("partition block IDs are invalid")
    _, sizes = np.unique(labels, return_counts=True)
    if exact and (int(sizes.max()) != int(depth) or len(sizes) != int(intactness)):
        raise ValueError("partition/depth/intactness truth is inconsistent")


def validate_authority_flags(value: Mapping[str, Any]) -> None:
    expected = {
        "research_only": True,
        "selection_evidence": False,
        "promotion_ready": False,
        "hardware_action_authorized": False,
    }
    if any(value.get(key) is not expected_value for key, expected_value in expected.items()):
        raise ValueError("V2.4 authority flags cannot authorize selection, promotion, or hardware")
