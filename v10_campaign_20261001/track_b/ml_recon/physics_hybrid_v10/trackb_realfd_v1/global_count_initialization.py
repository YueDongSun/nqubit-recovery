"""Frozen train-fold statistics for effective global-count initialization.

These are count-intercept, relative-dispersion and pair-composition starting
values, not microscopic source/detector calibration. No READ/clock multiplier
is inferred. Native targets are never modified or used as per-row inputs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
from typing import Any, Mapping

import torch

from ml_recon.physics_hybrid_v10.real_count_likelihood import _MAX_EXACT_NB2_RISING_TERMS, _MIN_POSITIVE_MEAN_NB2_DISPERSION

from .contracts import canonical_json_bytes, canonical_sha256, require_git_oid, require_sha256
from .row_identity import row_identity_manifest_sha, source_row_content_sha


@dataclass(frozen=True)
class Global_initialization_recipe:
    mean_prior_count: float = 0.5
    mean_prior_weight: float = 1.0
    pair_pseudocount: float = 0.5
    dispersion_prior: float = 1.0
    dispersion_minimum: float = 0.01
    dispersion_maximum: float = 1000.0
    maximum_log_residual: float = 8.0
    schema: str = "TrackBGlobalInitializationRecipeV1"

    def validate(self) -> None:
        # First admitted recipe is fixed. Future choices require a new recipe
        # contract rather than widening an untested numerical domain.
        if self != Global_initialization_recipe():
            raise ValueError("unregistered global initialization recipe")
        if any(type(value) not in (int, float) or isinstance(value, bool) or not math.isfinite(value)
               for name, value in asdict(self).items() if name != "schema"):
            raise ValueError("global initialization recipe must contain finite numeric constants")
        if self.dispersion_minimum * math.exp(-self.maximum_log_residual) < _MIN_POSITIVE_MEAN_NB2_DISPERSION:
            raise ValueError("global residual domain exceeds the native NB2 dispersion support")


def _derive_baselines(count_sums, squared_sums, pair_sums, observed_rows, recipe):
    means = tuple((value + recipe.mean_prior_count) / (observed_rows + recipe.mean_prior_weight) for value in count_sums)
    dispersion = []
    for total, squared, mean in zip(count_sums, squared_sums, means, strict=True):
        if observed_rows < 2 or total == 0:
            estimate = recipe.dispersion_prior
        else:
            raw_mean = total / observed_rows
            # Residual moment about the fixed, weak-prior mean baseline. The
            # integer numerator avoids cancellation for nearly constant counts.
            variance = (observed_rows * squared - total * total) / observed_rows**2
            residual_moment = variance + (raw_mean - mean)**2
            estimate = (residual_moment - mean) / mean**2
        dispersion.append(min(recipe.dispersion_maximum, max(recipe.dispersion_minimum, estimate)))
    pair_denominator = sum(pair_sums) + 6 * recipe.pair_pseudocount
    pairs = tuple((value + recipe.pair_pseudocount) / pair_denominator for value in pair_sums)
    return means, tuple(dispersion), pairs


@dataclass(frozen=True)
class Global_count_initialization:
    source_commit: str
    source_tree: str
    system_type: str
    admission_manifest_sha: str
    outer_fold_sha: str
    selected_train_rows_sha: str
    # sample, group, content identity, observed-global mask, native scope.
    train_rows: tuple[tuple[str, str, str, bool, str], ...]
    count_sums: tuple[int, int, int]
    squared_count_sums: tuple[int, int, int]
    pair_sums: tuple[int, int, int, int, int, int]
    mean_counts: tuple[float, float, float]
    dispersion: tuple[float, float, float]
    pair_probabilities: tuple[float, float, float, float, float, float]
    recipe: Global_initialization_recipe = Global_initialization_recipe()
    link: str = "G0_COUNT_INTERCEPT_LOG_RESIDUAL"
    layout: str = "NATIVE_MULTI2_PAIR6_MULTI3_MULTI4"
    parameter_semantics: str = "TRAIN_INITIALIZED_EFFECTIVE_GLOBAL_COUNTS"
    schema: str = "TrackBGlobalCountInitializationV1"

    def validate(self) -> None:
        self.recipe.validate()
        require_git_oid(self.source_commit, "initializer_source_commit")
        require_git_oid(self.source_tree, "initializer_source_tree")
        for name in ("admission_manifest_sha", "outer_fold_sha", "selected_train_rows_sha"):
            require_sha256(getattr(self, name), name)
        if self.schema != "TrackBGlobalCountInitializationV1" or self.link != "G0_COUNT_INTERCEPT_LOG_RESIDUAL" \
                or self.layout != "NATIVE_MULTI2_PAIR6_MULTI3_MULTI4" \
                or self.parameter_semantics != "TRAIN_INITIALIZED_EFFECTIVE_GLOBAL_COUNTS" \
                or self.system_type not in {"3qubit", "4qubit"}:
            raise ValueError("global initializer schema, link or system is invalid")
        if type(self.train_rows) is not tuple or not self.train_rows or tuple(sorted(self.train_rows)) != self.train_rows:
            raise ValueError("global initializer requires sorted immutable train-row identities")
        ids = []
        for row in self.train_rows:
            if type(row) is not tuple or len(row) != 5 or any(type(value) is not str or not value for value in row[:3]) \
                    or type(row[3]) is not bool or row[4] not in {"global", "state"}:
                raise ValueError("global initializer row identity/mask/scope is invalid")
            require_sha256(row[2], "initializer_row_content_sha")
            ids.append(row[0])
        if len(ids) != len(set(ids)):
            raise ValueError("global initializer repeats a train row")
        identities = tuple((row[0], row[1], row[2], "train") for row in self.train_rows)
        if row_identity_manifest_sha(identities) != self.selected_train_rows_sha:
            raise ValueError("global initializer selected-row identity does not match its preimage")
        observed = sum(row[3] for row in self.train_rows)
        if observed == 0:
            raise ValueError("cannot fit global initialization without observed global rows")
        for values, size in ((self.count_sums, 3), (self.squared_count_sums, 3), (self.pair_sums, 6)):
            if type(values) is not tuple or len(values) != size or any(type(value) is not int or value < 0 for value in values):
                raise ValueError("initializer sufficient statistics must be native nonnegative integers")
        if sum(self.pair_sums) != self.count_sums[0] or any(
            observed * squared < total**2 or squared > total**2
            or total > observed * _MAX_EXACT_NB2_RISING_TERMS or squared > total * _MAX_EXACT_NB2_RISING_TERMS
            for total, squared in zip(self.count_sums, self.squared_count_sums, strict=True)
        ):
            raise ValueError("initializer sufficient statistics violate count/moment conservation")
        expected = _derive_baselines(self.count_sums, self.squared_count_sums, self.pair_sums, observed, self.recipe)
        for values, target in zip((self.mean_counts, self.dispersion, self.pair_probabilities), expected, strict=True):
            if type(values) is not tuple or len(values) != len(target) \
                    or any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0 for value in values) \
                    or values != target:
                raise ValueError("initializer baselines differ from registered sufficient-statistic derivation")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Global_count_initialization":
        if not isinstance(payload, Mapping) or set(payload) != {field.name for field in fields(cls)}:
            raise ValueError("global initializer field set is invalid")
        values = dict(payload)
        recipe = values["recipe"]
        if not isinstance(recipe, Mapping) or set(recipe) != {field.name for field in fields(Global_initialization_recipe)}:
            raise ValueError("global initializer recipe preimage is invalid")
        values["recipe"] = Global_initialization_recipe(**recipe)
        for name in ("train_rows", "count_sums", "squared_count_sums", "pair_sums", "mean_counts", "dispersion", "pair_probabilities"):
            if not isinstance(values[name], (list, tuple)):
                raise ValueError("global initializer arrays are invalid")
            values[name] = tuple(tuple(row) for row in values[name]) if name == "train_rows" else tuple(values[name])
        result = cls(**values)
        result.validate()
        return result


def fit_global_count_initialization(dataset: Any, *, source_commit: str, source_tree: str) -> Global_count_initialization:
    """Read every admitted train row, freezing only observed-global statistics.

This fits a shared initialization using TRAIN LABELS. It is not a no-fit M0,
a per-row forward input, an NB2 MLE with covariates, or physical calibration.
The caller owns dataset lifetime and independently binds the source checkout.
"""
    if dataset.outer_role != "train":
        raise ValueError("global initialization may read only the selected train fold")
    expected_ids = dataset.fold_plan.sample_ids_for_role("train")
    if len(dataset) != len(expected_ids) or dataset.admission_manifest.sha256() != dataset.fold_plan.admission_manifest_sha:
        raise ValueError("global initialization admission/fold cardinality is inconsistent")
    rows = []
    counts = [0, 0, 0]
    squared = [0, 0, 0]
    pairs = [0] * 6
    seen = set()
    system = None
    for index in range(len(dataset)):
        item = dataset[index]
        sample = item["sample_id"]
        if sample in seen or sample not in expected_ids or item.get("source_role") != "train" or item.get("outer_role") != "train":
            raise ValueError("global initializer encountered duplicate, foreign or non-train row")
        seen.add(sample)
        if item["admission_manifest_sha"] != dataset.admission_manifest.sha256() or item["outer_fold_sha"] != dataset.fold_plan.sha256():
            raise ValueError("global initializer row admission differs from the selected source")
        actual_sha = source_row_content_sha(item)
        expected_row = dataset.row_identity_records[sample]
        if actual_sha != item["row_content_sha"] or expected_row != (item["group_id"], actual_sha, "train"):
            raise ValueError("global initializer row content is not source-bound")
        if system is not None and system != item["system_type"]:
            raise ValueError("global initializer cannot mix systems")
        system = item["system_type"]
        mask = item["global_multi_mask"]
        values = item["global_multi"]
        scope = item["global_multi_scope"]
        if mask.dtype != torch.bool or mask.numel() != 1 or values.dtype != torch.int64 or values.shape != (9,) \
                or scope not in {"global", "state"}:
            raise ValueError("global initializer requires native counter, mask and scope layout")
        observed = bool(mask.item())
        rows.append((sample, item["group_id"], actual_sha, observed, scope))
        if not observed:
            continue
        native = values.tolist()
        if any(value < 0 for value in native) or sum(native[1:7]) != native[0]:
            raise ValueError("observed native global counters violate support or pair conservation")
        if any(native[index] > _MAX_EXACT_NB2_RISING_TERMS for index in (0, 7, 8)):
            raise ValueError("global initialization count exceeds the registered native NB2 domain")
        for column, value in enumerate((native[0], native[7], native[8])):
            counts[column] += value
            squared[column] += value * value
        pairs = [total + value for total, value in zip(pairs, native[1:7], strict=True)]
    if seen != expected_ids:
        raise ValueError("global initializer did not cover the entire selected train fold")
    rows = tuple(sorted(rows))
    selected_sha = row_identity_manifest_sha(tuple((row[0], row[1], row[2], "train") for row in rows))
    if selected_sha != dataset.row_identity_sha:
        raise ValueError("global initializer selected row identity differs from the dataset")
    observed = sum(row[3] for row in rows)
    if not observed:
        raise ValueError("no observed global rows are available for initialization")
    recipe = Global_initialization_recipe()
    means, dispersion, probabilities = _derive_baselines(counts, squared, pairs, observed, recipe)
    result = Global_count_initialization(source_commit, source_tree, system, dataset.admission_manifest.sha256(),
        dataset.fold_plan.sha256(), selected_sha, rows, tuple(counts), tuple(squared), tuple(pairs), means, dispersion, probabilities, recipe)
    result.validate()
    return result


def load_global_initialization_bytes(data: bytes, *, expected_sha: str) -> Global_count_initialization:
    require_sha256(expected_sha, "global_initialization_file_sha")
    if type(data) is not bytes or hashlib.sha256(data).hexdigest() != expected_sha:
        raise ValueError("global initialization byte identity mismatch")
    result = Global_count_initialization.from_dict(json.loads(data))
    if result.canonical_bytes() != data:
        raise ValueError("global initialization must use exact canonical bytes")
    return result


def validate_global_initialization_binding(initialization: Global_count_initialization, identity: Any) -> None:
    """Cross-bind the frozen initialization to its consuming training identity."""
    initialization.validate()
    if (initialization.source_commit, initialization.source_tree, initialization.system_type,
        initialization.admission_manifest_sha, initialization.outer_fold_sha, initialization.selected_train_rows_sha) != (
        identity.source_commit, identity.source_tree, identity.system_type,
        identity.admission_manifest_sha, identity.outer_fold_sha, identity.row_identity_sha,
    ):
        raise ValueError("global initialization differs from runtime source/admission/fold")
    rows = tuple(sorted((row[0], row[1], row[2], "train") for row in initialization.train_rows))
    if rows != tuple(sorted(identity.allowed_row_identities)):
        raise ValueError("global initialization does not bind the exact runtime train rows")
