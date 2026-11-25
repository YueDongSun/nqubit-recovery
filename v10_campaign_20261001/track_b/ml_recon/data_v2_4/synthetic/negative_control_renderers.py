"""Deterministic physical negative-control transforms for frozen V2.4 evaluation."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

import numpy as np

from ml_recon.physics_hybrid_v10.noise_interventions import (
    CORRECTABLE_INTERVENTIONS,
    IRRECOVERABLE_INTERVENTIONS,
)

from ..hashing import sha256_json


PERSISTENT_NEGATIVE_CONTROL_KINDS: Tuple[str, ...] = (
    "target_shuffle",
    "operation_subset_shuffle",
    "global_multi_pair_shuffle",
    "environment_permutation",
    "mechanism_label_permutation",
)
NEGATIVE_CONTROL_FAMILY_MARKER = "_negative_control_"


def persistent_negative_controls_for_family(dataset_family: str) -> Tuple[str, ...]:
    """Return only controls that alter a registered field for this task family."""
    family = str(dataset_family)
    if family == "small_n_family":
        return PERSISTENT_NEGATIVE_CONTROL_KINDS
    if family == "general_n_structure":
        return (
            "target_shuffle", "operation_subset_shuffle",
            "global_multi_pair_shuffle", "environment_permutation",
        )
    if family == "w_control":
        return (
            "operation_subset_shuffle", "global_multi_pair_shuffle",
            "environment_permutation",
        )
    if family.startswith("feedback_"):
        return ("target_shuffle", "mechanism_label_permutation")
    raise ValueError(f"no registered V2.4 negative-control plan for {family}")


def split_negative_control_family(dataset_family: str) -> tuple[str, str]:
    """Return the physical base family and registered control kind."""
    family = str(dataset_family)
    if NEGATIVE_CONTROL_FAMILY_MARKER in family:
        base_family, kind = family.rsplit(NEGATIVE_CONTROL_FAMILY_MARKER, 1)
        if not base_family or kind not in PERSISTENT_NEGATIVE_CONTROL_KINDS:
            raise ValueError(f"unregistered V2.4 negative control family: {family}")
        return base_family, kind
    if family.endswith("_negative_control"):
        # Preserve the old public suffix while upgrading it from mask-only to an
        # actual deterministic target control.
        return family[:-len("_negative_control")], "target_shuffle"
    return family, ""


def _nonidentity_permutation(length: int, *, key: Any, kind: str,
                             namespace: str) -> np.ndarray:
    if length < 2:
        return np.arange(length, dtype=np.int64)
    digest = sha256_json({
        "global_seed": int(key.global_seed),
        "dataset_family": str(key.dataset_family),
        "cell_id": str(key.cell_id),
        "N": int(key.N),
        "kind": kind,
        "namespace": namespace,
    })
    rng = np.random.Generator(np.random.Philox(int(digest[:16], 16)))
    identity = np.arange(length, dtype=np.int64)
    for _ in range(16):
        permutation = rng.permutation(length)
        if not np.array_equal(permutation, identity):
            return permutation
    return np.roll(identity, 1)


def _cyclic_derangement(length: int, *, key: Any, kind: str,
                        namespace: str) -> np.ndarray:
    if length < 2:
        return np.arange(length, dtype=np.int64)
    digest = sha256_json({
        "global_seed": int(key.global_seed),
        "dataset_family": str(key.dataset_family),
        "cell_id": str(key.cell_id),
        "N": int(key.N),
        "kind": kind,
        "namespace": namespace,
    })
    shift = 1 + int(digest[:16], 16) % (length - 1)
    return np.roll(np.arange(length, dtype=np.int64), shift)


def _copy_rendered(rendered: Mapping[str, Any]) -> Dict[str, Any]:
    copied = dict(rendered)
    for store in ("observation", "truth"):
        copied[store] = {
            name: np.asarray(value).copy()
            for name, value in dict(rendered[store]).items()
        }
    return copied


def apply_persistent_negative_control(
    rendered: Mapping[str, Any], *, kind: str, key: Any,
    partner_rendered: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Apply one registered control without changing array shapes or marginals."""
    if kind not in PERSISTENT_NEGATIVE_CONTROL_KINDS:
        raise ValueError(f"unregistered V2.4 negative control: {kind}")
    controlled = _copy_rendered(rendered)
    observation = controlled["observation"]
    truth = controlled["truth"]

    if kind == "target_shuffle":
        if partner_rendered is not None:
            partner_observation = dict(partner_rendered["observation"])
            partner_truth = dict(partner_rendered["truth"])
            if "target_mode_tokens" in observation:
                observation["target_mode_tokens"] = np.asarray(
                    partner_observation["target_mode_tokens"]).copy()
            for name in tuple(truth):
                if (name.startswith("target_") or name == "oracle_action") \
                        and name in partner_truth:
                    truth[name] = np.asarray(partner_truth[name]).copy()
        else:
            permutation = _nonidentity_permutation(
                int(key.N), key=key, kind=kind, namespace="target_modes")
            if "target_mode_tokens" in observation:
                observation["target_mode_tokens"] = observation["target_mode_tokens"][permutation]
            for name in ("target_pair", "target_partition", "target_R1_re", "target_R1_im"):
                if name in truth and truth[name].shape[:2] == (key.N, key.N):
                    truth[name] = truth[name][np.ix_(permutation, permutation)]
            if "target_action" in truth and truth["target_action"].shape[0] == key.N:
                truth["target_action"] = truth["target_action"][permutation]
            if "oracle_action" in truth and truth["oracle_action"].shape[0] == key.N:
                truth["oracle_action"] = truth["oracle_action"][permutation]

    elif kind == "operation_subset_shuffle":
        if "basis_descriptors" in observation:
            permutation = _nonidentity_permutation(
                len(observation["basis_descriptors"]), key=key, kind=kind,
                namespace="setting_identity")
            observation["basis_descriptors"] = observation["basis_descriptors"][permutation]
        subset_identity_fields = (
            "subset_parent_setting_id", "subset_qubits", "subset_order", "subset_axes",
            "pair_stat_qubits", "pair_stat_axes",
        )
        available = [name for name in subset_identity_fields
                     if name in observation and len(observation[name]) > 1]
        if available:
            token_count = len(observation[available[0]])
            if any(len(observation[name]) != token_count for name in available):
                raise ValueError("V2.4 subset identity fields have inconsistent lengths")
            permutation = _nonidentity_permutation(
                token_count, key=key, kind=kind, namespace="subset_identity")
            for name in available:
                observation[name] = observation[name][permutation]

    elif kind == "global_multi_pair_shuffle":
        if "global_multi" in observation:
            global_multi = observation["global_multi"]
            if global_multi.shape != (9,):
                raise ValueError("V2.4 global_multi control requires the registered [9] layout")
            permutation = _nonidentity_permutation(
                6, key=key, kind=kind, namespace="detector_pairs")
            global_multi[1:7] = global_multi[1:7][permutation]

    elif kind == "environment_permutation":
        if "environment" in observation:
            environment = observation["environment"]
            permutation = _nonidentity_permutation(
                len(environment), key=key, kind=kind, namespace="environment_fields")
            observation["environment"] = environment[permutation]
            if "environment_mask" in observation:
                observation["environment_mask"] = observation["environment_mask"][permutation]

    elif kind == "mechanism_label_permutation":
        for name in ("correctable_mechanism", "irrecoverable_mechanism",
                     "mechanism_severity", "true_mechanism"):
            if name not in truth or np.asarray(truth[name]).ndim != 1:
                continue
            permutation = _cyclic_derangement(
                len(truth[name]), key=key, kind=kind, namespace=name)
            truth[name] = truth[name][permutation]
        if "mechanism_name" in truth:
            raw_name = np.asarray(truth["mechanism_name"]).item()
            mechanism_name = raw_name.decode("ascii") if isinstance(raw_name, bytes) \
                else str(raw_name)
            if mechanism_name == "identity":
                truth["negative_control_mask"] = np.asarray(True, dtype=np.bool_)
                return controlled
            if mechanism_name.startswith("composite:"):
                control_raw = np.asarray(
                    truth["control_mechanism_name"]).reshape(-1)[0]
                irrecoverable_raw = np.asarray(
                    truth["irrecoverable_mechanism_name"]).reshape(-1)[0]
                control_name = control_raw.decode("ascii")
                irrecoverable_name = irrecoverable_raw.decode("ascii")
                control_replacement = CORRECTABLE_INTERVENTIONS[
                    (CORRECTABLE_INTERVENTIONS.index(control_name) + 1)
                    % len(CORRECTABLE_INTERVENTIONS)]
                irrecoverable_replacement = IRRECOVERABLE_INTERVENTIONS[
                    (IRRECOVERABLE_INTERVENTIONS.index(irrecoverable_name) + 1)
                    % len(IRRECOVERABLE_INTERVENTIONS)]
                truth["control_mechanism_name"] = np.asarray(
                    [control_replacement.encode("ascii")], dtype="S64")
                truth["irrecoverable_mechanism_name"] = np.asarray(
                    [irrecoverable_replacement.encode("ascii")], dtype="S64")
                truth["mechanism_name"] = np.asarray(
                    f"composite:{control_replacement}+{irrecoverable_replacement}"
                    .encode("ascii"), dtype=np.asarray(
                        truth["mechanism_name"]).dtype)
                truth["negative_control_mask"] = np.asarray(True, dtype=np.bool_)
                return controlled
            names = (CORRECTABLE_INTERVENTIONS if mechanism_name in CORRECTABLE_INTERVENTIONS
                     else IRRECOVERABLE_INTERVENTIONS)
            if mechanism_name not in names:
                raise ValueError("V2.4 mechanism control found an unregistered label")
            replacement = names[(names.index(mechanism_name) + 1) % len(names)]
            truth["mechanism_name"] = np.asarray(
                replacement.encode("ascii"), dtype=np.asarray(truth["mechanism_name"]).dtype)

    truth["negative_control_mask"] = np.asarray(True, dtype=np.bool_)
    return controlled
