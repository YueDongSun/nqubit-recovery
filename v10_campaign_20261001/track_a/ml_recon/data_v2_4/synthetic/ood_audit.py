"""Distribution audit for V2.4 locked-OOD classifier cells."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..sample_keys import make_sample_key
from ..hashing import sha256_file
from ..range_index import read_range_index
from ..shard_reader import V24Hdf5HandleCache
from ..version import GENERATOR_SHA


OOD_C2ST_MINIMUM = 0.53
OOD_MMD_MINIMUM = 1e-3


def _feature_row(observation: dict[str, np.ndarray]) -> np.ndarray:
    counts = np.asarray(observation["counts"], dtype=np.float64)
    shots = np.asarray(observation["shots"], dtype=np.float64).reshape(-1, 1)
    features = [np.nan_to_num(counts / np.maximum(shots, 1.0)).reshape(-1)]
    if "global_outcome_histogram" in observation:
        hist = np.asarray(observation["global_outcome_histogram"], dtype=np.float64)
        features.append((hist / np.maximum(hist.sum(-1, keepdims=True), 1.0)).reshape(-1))
    features.append(np.asarray(observation["correlation_adjacency"], dtype=np.float64).reshape(-1))
    return np.concatenate(features).astype(np.float64)


def _c2st(a: np.ndarray, b: np.ndarray) -> float:
    """Paired held-out linear C2ST with latent-state variation cancelled."""
    n = min(len(a), len(b))
    split = max(1, n // 2)
    direction = (b[:split] - a[:split]).mean(0)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-15:
        return 0.5
    direction /= norm
    correct = 0.0
    total = 0
    for left, right in zip(a[split:n], b[split:n]):
        midpoint = 0.5 * (left + right)
        correct += float((left - midpoint) @ direction < 0.0)
        correct += float((right - midpoint) @ direction > 0.0)
        total += 2
    return float(correct / max(total, 1))


def _mmd(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    scale = float(np.median(np.linalg.norm(a[:, None] - a[None, :], axis=-1)))
    sigma = max(scale, 1e-6)
    def kernel(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        distance = ((x[:, None] - y[None, :]) ** 2).sum(-1)
        return np.exp(-distance / (2.0 * sigma * sigma))
    return float(kernel(a, a).mean() + kernel(b, b).mean() - 2.0 * kernel(a, b).mean())


def audit_locked_ood(*, N: int = 4, n_samples: int = 32,
                     seed: int = 20261001) -> dict[str, Any]:
    from .generator import _render_dataset_family

    train = []
    locked = []
    paired_latent_matches = 0
    for index in range(int(n_samples)):
        train_key = make_sample_key(
            global_seed=seed, split="validation",
            dataset_family="small_n_family", N=N, index=index,
            generator_sha=GENERATOR_SHA, cell_id=f"classifier_n{N}_standard")
        ood_key = make_sample_key(
            global_seed=seed, split="locked_ood",
            dataset_family="small_n_family_locked_ood", N=N, index=index,
            generator_sha=GENERATOR_SHA,
            cell_id=f"classifier_n{N}_standard_locked_ood")
        cell = {
            "noise_family": "standard", "eta": 0.0, "kappa": 0.0,
            "shot_mode": "S_setting", "shot_budget": 64,
            "measurement_schedule": "full_global_pauli"}
        train_rendered = _render_dataset_family(train_key, cell=cell)
        locked_rendered = _render_dataset_family(ood_key, cell=cell)
        train.append(_feature_row(train_rendered["observation"]))
        locked.append(_feature_row(locked_rendered["observation"]))
        paired_latent_matches += int(
            train_rendered["identities"]["latent_state_id"]
            == locked_rendered["identities"]["latent_state_id"])
    a, b = np.asarray(train), np.asarray(locked)
    c2st = _c2st(a, b)
    mmd = _mmd(a, b)
    return {
        "audit_kind": "v24_locked_ood_distribution_v1",
        "N": int(N), "samples_per_population": int(n_samples),
        "train_family": "small_n_family",
        "ood_family": "small_n_family_locked_ood",
        "paired_latent_matches": paired_latent_matches,
        "paired_latent_contract": paired_latent_matches == int(n_samples),
        "c2st_balanced_accuracy": c2st,
        "mmd": mmd,
        "distinct": bool(c2st >= OOD_C2ST_MINIMUM and mmd >= OOD_MMD_MINIMUM),
        "metrics_authoritative": bool(
            c2st >= OOD_C2ST_MINIMUM and mmd >= OOD_MMD_MINIMUM
            and paired_latent_matches == int(n_samples)),
        "research_only": True,
        "selection_evidence": False,
        "promotion_ready": False,
        "hardware_action_authorized": False,
    }


def audit_persistent_locked_ood(
    local_root: str | Path,
    *,
    validation_range_index_path: str | Path,
    locked_ood_range_index_path: str | Path,
    validation_cell_id: str,
    locked_ood_cell_id: str,
) -> dict[str, Any]:
    """Audit the actual immutable persistent validation/OOD shard pair."""
    root = Path(local_root)
    cache = V24Hdf5HandleCache(maximum_handles=6)

    def load_population(path: str | Path, cell_id: str, split: str
                        ) -> tuple[np.ndarray, list[str], list[str]]:
        features = []
        sample_ids = []
        pair_ids = []
        entries = [entry for entry in read_range_index(path)
                   if entry.cell_id == cell_id and entry.split == split]
        if not entries:
            raise ValueError(f"persistent OOD audit selected zero {split} rows")
        for entry in entries:
            rows = tuple(range(entry.row_count))
            observation = cache.read_rows(
                root / entry.observation_path, rows,
                expected_sha=entry.observation_sha)
            static = cache.read_row(
                root / entry.cell_static_path, 0,
                expected_sha=entry.cell_static_file_sha)
            truth = cache.read_rows(
                root / entry.truth_path, rows,
                expected_sha=entry.truth_sha)
            for row in range(entry.row_count):
                merged = {name: value for name, value in static.items()
                          if name != "row_valid_mask"}
                merged.update({name: value[row]
                               for name, value in observation.items()})
                features.append(_feature_row(merged))
                sample_ids.append(
                    np.asarray(observation["sample_id"][row]).item().decode("utf-8"))
                if "locked_ood_pair_id" in truth:
                    pair_ids.append(
                        np.asarray(truth["locked_ood_pair_id"][row])
                        .item().decode("utf-8"))
        return np.asarray(features), sample_ids, pair_ids

    try:
        validation, validation_ids, _ = load_population(
            validation_range_index_path, validation_cell_id, "validation")
        locked, _, pair_ids = load_population(
            locked_ood_range_index_path, locked_ood_cell_id, "locked_ood")
    finally:
        cache.close()
    paired = bool(pair_ids and set(pair_ids).issubset(set(validation_ids)))
    c2st = _c2st(validation, locked)
    mmd = _mmd(validation, locked)
    distinct = bool(
        c2st >= OOD_C2ST_MINIMUM and mmd >= OOD_MMD_MINIMUM)
    return {
        "audit_kind": "v24_persistent_locked_ood_distribution_v1",
        "validation_cell_id": validation_cell_id,
        "locked_ood_cell_id": locked_ood_cell_id,
        "validation_rows": len(validation),
        "locked_ood_rows": len(locked),
        "validation_range_index_sha": sha256_file(validation_range_index_path),
        "locked_ood_range_index_sha": sha256_file(locked_ood_range_index_path),
        "paired_latent_contract": paired,
        "c2st_balanced_accuracy": c2st,
        "mmd": mmd,
        "distinct": distinct,
        "metrics_authoritative": bool(paired and distinct),
        "research_only": True,
        "selection_evidence": False,
        "promotion_ready": False,
        "hardware_action_authorized": False,
    }
