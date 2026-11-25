"""Allowlisted frozen behavior recipe for the Track B learner pilot."""

from __future__ import annotations

from dataclasses import fields
import hashlib
import json
from pathlib import Path

from .contracts import canonical_json_bytes
from .inverse_transformer import (
    LEGACY_LINEAR_MEAN_ACTION_VALUE,
    STANDARDIZED_LATENT_GAIN_PARAMETERIZATION_V1,
    TRAIN_RMS_CENTERED_BOUNDED_GAUGE_PROJECTION,
)
from .inverse_transformer_pilot import (
    ROUND_COVERAGE_TRAINING_SAMPLING,
    SHARED_BALANCED_PCGD_GRADIENT_COMPOSITION,
    TrackBV24TenRoundPilotConfigV1,
)


MATCHED_BEHAVIOR_RECEIPT_SHA = "e713e8638c5e8b0b0b864eed75c9fe53e086d6a5ddb2b7794cafb2ce549c7834"
MATCHED_BEHAVIOR_RECIPE_KIND = "matched_behavior_v1"
MAX_BEHAVIOR_RECEIPT_BYTES = 1024 * 1024
MATCHED_BEHAVIOR_RECIPE = {
    "batch_size": 64,
    "width": 64,
    "heads": 4,
    "blocks": 2,
    "real_static_source_root": "",
    "allow_noncanonical_test_fixture": False,
    "learning_rate": 5e-4,
    "seed": 20261001,
    "gain_loss_weight": 0.5,
    "training_sampling_kind": ROUND_COVERAGE_TRAINING_SAMPLING,
    "gain_parameterization": STANDARDIZED_LATENT_GAIN_PARAMETERIZATION_V1,
    "gradient_composition_kind": SHARED_BALANCED_PCGD_GRADIENT_COMPOSITION,
    "action_loss_normalization_kind": "per_round_v1",
    "action_projection_kind": TRAIN_RMS_CENTERED_BOUNDED_GAUGE_PROJECTION,
    "action_value_feature_kind": LEGACY_LINEAR_MEAN_ACTION_VALUE,
}


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate behavior receipt key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"nonfinite behavior receipt number: {value}")


def _read_bound_behavior_receipt_bytes(receipt_path: str, expected_sha: str) -> bytes:
    """Read one bounded file object and bind those exact bytes to the frozen SHA."""
    if expected_sha != MATCHED_BEHAVIOR_RECEIPT_SHA:
        raise ValueError("matched behavior receipt SHA is not the registered frozen receipt")
    path = Path(receipt_path)
    with path.open("rb") as stream:
        payload = stream.read(MAX_BEHAVIOR_RECEIPT_BYTES + 1)
    if len(payload) > MAX_BEHAVIOR_RECEIPT_BYTES:
        raise ValueError("matched behavior receipt exceeds the bounded input size")
    if hashlib.sha256(payload).hexdigest() != expected_sha:
        raise ValueError("matched behavior receipt differs from its expected SHA")
    return payload


def verify_matched_behavior_receipt_bytes(receipt_path: str, expected_sha: str) -> None:
    """Recheck input bytes after output claim without reparsing recipe semantics."""
    _read_bound_behavior_receipt_bytes(receipt_path, expected_sha)


def resolve_matched_behavior_recipe(receipt_path: str, expected_sha: str) -> dict[str, object]:
    """Verify the pinned receipt and its exact relevant recipe values."""
    payload = _read_bound_behavior_receipt_bytes(receipt_path, expected_sha)
    try:
        receipt = json.loads(payload.decode("utf-8"), object_pairs_hook=_unique_object,
                             parse_constant=_reject_nonfinite)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("matched behavior receipt must be UTF-8 JSON") from error
    if type(receipt) is not dict or canonical_json_bytes(receipt) != payload:
        raise ValueError("matched behavior receipt must be a canonical JSON object")
    config = receipt.get("config")
    if type(config) is not dict:
        raise ValueError("matched behavior receipt lacks its config object")
    actual = {name: config.get(name) for name in MATCHED_BEHAVIOR_RECIPE}
    missing = [name for name in MATCHED_BEHAVIOR_RECIPE if name not in config]
    known = {field.name for field in fields(TrackBV24TenRoundPilotConfigV1)}
    if missing or set(config) - known:
        raise ValueError("matched behavior receipt recipe fields are incomplete or unknown")
    if any(type(actual[name]) is not type(expected) or actual[name] != expected
           for name, expected in MATCHED_BEHAVIOR_RECIPE.items()):
        raise ValueError("matched behavior receipt recipe differs from the registered values")
    return dict(MATCHED_BEHAVIOR_RECIPE)


__all__ = ["MATCHED_BEHAVIOR_RECEIPT_SHA", "MATCHED_BEHAVIOR_RECIPE_KIND",
           "MATCHED_BEHAVIOR_RECIPE", "resolve_matched_behavior_recipe",
           "verify_matched_behavior_receipt_bytes"]
