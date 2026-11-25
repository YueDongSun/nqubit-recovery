"""Prepublication checks for the registered global-Pauli V2 behavior rows.

This verifies persisted schema, finite/full pilot observations, keyed initial
states, declared composite nuisance, count budgets and deterministic historical
teacher trajectories. Replaying their twenty before/after draws validates stored
bytes; it is not an additional acquisition or the eleven-window model rollout.
This is not full DLCZ qualification, calibration evidence or a scientific Gate.
Single-step oracle targets remain distinct from ten-round history targets.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np

from ml_recon.data_v2_4.hashing import sha256_json
from ml_recon.data_v2_4.sample_keys import V24SampleKey
from ml_recon.data_v2_4.schema_registry import validate_persisted_arrays
from ml_recon.data_v2_4.synthetic.feedback_renderers import render_trackb_ten_round_history
from ml_recon.data_v2_4.validators import validate_global_multi, validate_view_keys
from ml_recon.physics_hybrid_v10.control_tangent import apply_tangent_action
from ml_recon.physics_hybrid_v10.measurement_scheduler import V10MeasurementScheduler
from ml_recon.physics_hybrid_v10.noise_interventions import (
    CompositeNoiseIntervention, intervention_registry,
)
from ml_recon.physics_hybrid_v10.pauli_registry import descriptor_vector

from .behavior_v2_contract import Behavior_recipe_v2
from .pilot_history_dataset import PILOT_OBSERVATION_FIELDS, Pilot_history_dataset


behavior_validation_scope = "GLOBAL_PAULI_V24_SCHEMA_KEYED_INITIAL_STATE_COMPOSITE_HISTORY_AND_COUNT_CONSISTENCY"
float32_tolerance = 8 * np.finfo(np.float32).eps
float64_tolerance = 1e-12
# The existing feedback renderer's initial injection recipe, not new tunables.
initial_amplitude_sigma = 0.14
initial_phase_sigma = 0.30


def _require_array(arrays: Mapping[str, np.ndarray], name: str, shape: tuple[int, ...], dtype) -> np.ndarray:
    if name not in arrays:
        raise ValueError(f"behavior data is missing {name}")
    value = arrays[name]
    if type(value) is not np.ndarray or value.shape != shape or value.dtype != np.dtype(dtype):
        raise ValueError(f"behavior {name} has invalid shape or dtype")
    return value


def _validate_group(arrays: Mapping[str, np.ndarray], keys: Sequence[V24SampleKey], *, store: str, mode_count: int) -> None:
    if not isinstance(arrays, Mapping) or not arrays:
        raise ValueError("behavior persisted groups must be nonempty array mappings")
    for name, value in arrays.items():
        if type(name) is not str or type(value) is not np.ndarray or value.ndim == 0 or value.shape[0] != len(keys):
            raise ValueError("every behavior persisted array must have the complete row dimension")
        if value.dtype.kind in "biufc" and not np.isfinite(value).all():
            raise ValueError(f"behavior {store}/{name} contains nonfinite values")
    validate_persisted_arrays(arrays, {"store": store, "provenance": "synthetic", "N": mode_count})
    mask = _require_array(arrays, "row_valid_mask", (len(keys),), np.bool_)
    identifiers = arrays["sample_id"]
    if identifiers.shape != (len(keys),) or identifiers.dtype.kind != "S" or not mask.all():
        raise ValueError("behavior rows must retain full valid masks and fixed-byte sample IDs")
    if tuple(bytes(value).decode("utf-8") for value in identifiers) != tuple(key.sample_id for key in keys):
        raise ValueError("behavior stored sample IDs do not match the planned rows")


def _validate_count_panel(observations: Mapping[str, np.ndarray], recipe: Behavior_recipe_v2, rows: int) -> None:
    settings = V10MeasurementScheduler(recipe.mode_count).schedule()
    setting_count = len(settings)
    shots = _require_array(observations, "shots", (rows, setting_count), np.float32)
    if not np.all(shots == recipe.shot_budget):
        raise ValueError("behavior per-setting attempt budget differs from the registered recipe")
    histograms = _require_array(observations, "global_outcome_histogram", (rows, setting_count, 2 ** recipe.mode_count), np.uint32)
    counts = _require_array(observations, "counts", (rows, setting_count, 4), np.float32)
    full_mask = _require_array(observations, "full_outcome_mask", histograms.shape, np.bool_)
    if not full_mask.all() or np.any(histograms.sum(axis=-1, dtype=np.uint64) > shots):
        raise ValueError("behavior histogram exceeds attempts or omits registered outcomes")
    # The classifier panel's counts are the (mode0,mode1) marginal of the same
    # detected global outcomes, not additional observations or herald counts.
    marginals = histograms.reshape(rows, setting_count, 4, 2 ** (recipe.mode_count - 2)).sum(axis=-1, dtype=np.uint64)
    if not np.array_equal(counts, marginals.astype(np.float32)):
        raise ValueError("behavior pair counts differ from their global-outcome marginal")
    descriptors = np.asarray([descriptor_vector(item.descriptor) for item in settings], dtype=np.float32)
    stored = _require_array(observations, "basis_descriptors", (rows, *descriptors.shape), np.float32)
    if not np.array_equal(stored, np.broadcast_to(descriptors, stored.shape)):
        raise ValueError("behavior count panel settings differ from the registered order")
    global_multi = _require_array(observations, "global_multi", (rows, 9), np.uint32)
    multi_mask = _require_array(observations, "global_multi_mask", (rows,), np.bool_)
    if not multi_mask.all():
        raise ValueError("behavior global multi-count records must be present")
    validate_global_multi(global_multi, multi_mask, ["state"] * rows)


def _row_intervention(truths: Mapping[str, np.ndarray], index: int, recipe: Behavior_recipe_v2) -> CompositeNoiseIntervention:
    registry = intervention_registry()
    channels = []
    parameters = []
    for prefix, category in (("control", "correctable"), ("irrecoverable", "irrecoverable")):
        raw_name = truths[f"{prefix}_mechanism_name"][index, 0]
        try:
            name = bytes(raw_name).decode("ascii")
            channel = registry[name]
        except (KeyError, UnicodeError) as error:
            raise ValueError("behavior nuisance mechanism is not registered") from error
        parameter = float(truths[f"{prefix}_parameter"][index, 0])
        if channel.category != category or not 0 <= parameter <= channel.maximum_parameter:
            raise ValueError("behavior nuisance category or physical bracket changed")
        channels.append(channel)
        parameters.append(parameter)
    intervention = CompositeNoiseIntervention(channels[0], channels[1], parameters[0], parameters[1])
    for name in ("physical_noise_parameter", "true_severity"):
        if not math.isclose(float(truths[name][index]), intervention.physical_parameter, rel_tol=0, abs_tol=float64_tolerance):
            raise ValueError("behavior nuisance magnitude differs from its complete channel parameters")
    for name, expected in (("eta_target", recipe.eta), ("kappa_inj", recipe.kappa), ("kappa_target", recipe.kappa)):
        if float(np.asarray(truths[name][index]).item()) != expected:
            raise ValueError("behavior declared noise recipe differs from the fixed study recipe")
    calibration_sha = sha256_json({"kind": "composite_noise_v1", "noise_family": recipe.noise_family,
                                  "correctable": channels[0].mechanism_name, "irrecoverable": channels[1].mechanism_name,
                                  "kappa_inj": recipe.kappa, "eta_target": recipe.eta,
                                  "severity_matching_kind": recipe.severity_matching_kind})
    if bytes(truths["d3_train_fit_sha"][index]).decode("ascii") != calibration_sha:
        raise ValueError("behavior nuisance declaration differs from its registered recipe identity")
    return intervention


def _validate_keyed_history(
    observations: Mapping[str, np.ndarray], truths: Mapping[str, np.ndarray], index: int,
    key: V24SampleKey, recipe: Behavior_recipe_v2,
) -> None:
    target_rng, action_rng = key.rng("target"), key.rng("action")
    initial_target = target_rng.normal(size=recipe.mode_count) + 1j * target_rng.normal(size=recipe.mode_count)
    initial_target /= np.linalg.norm(initial_target)
    expected_current = apply_tangent_action(
        initial_target, action_rng.normal(scale=initial_amplitude_sigma, size=recipe.mode_count - 1),
        action_rng.normal(scale=initial_phase_sigma, size=recipe.mode_count - 1))
    current = truths["current_coefficients_re"][index] + 1j * truths["current_coefficients_im"][index]
    if not np.allclose(current, expected_current, atol=float64_tolerance, rtol=0):
        raise ValueError("behavior initial state does not match its canonical sample RNG identity")
    target_tokens = np.stack((np.abs(initial_target), np.angle(initial_target * initial_target[0].conj()) / np.pi), axis=-1)
    if not np.allclose(observations["target_mode_tokens"][index], target_tokens, atol=float32_tolerance, rtol=0):
        raise ValueError("behavior public target does not match its canonical sample RNG identity")
    intervention = _row_intervention(truths, index, recipe)
    stub = {"truth": {name: truths[name][index].copy() for name in (
                "current_coefficients_re", "current_coefficients_im", "target_action")},
            "observation": {"target_mode_tokens": observations["target_mode_tokens"][index].copy()},
            "internal": {"intervention": intervention, "physical_parameter": intervention.physical_parameter}}
    replay = render_trackb_ten_round_history(
        key, stub, shot_mode=recipe.shot_mode, shot_budget=recipe.shot_budget,
        history_policy=recipe.history_policy, history_policy_sha256=recipe.history_policy_sha256)
    if not np.array_equal(observations["history_tokens"][index], replay["observation"]["history_tokens"]):
        raise ValueError("behavior history observations/actions/gains differ from registered V2 teacher replay")
    for name in ("history_round_count", "target_history_gain", "target_history_gain_mask"):
        if not np.array_equal(truths[name][index], replay["truth"][name]):
            raise ValueError("behavior history truth differs from its ten-round teacher trajectory")


def validate_behavior_split(
    observations: Mapping[str, np.ndarray], truths: Mapping[str, np.ndarray], *,
    sample_keys: Sequence[V24SampleKey], recipe: Behavior_recipe_v2,
) -> None:
    """Validate a bounded split/fixture without writing or fitting noise/models.

    This helper permits 1..1000 TRAIN or 1..200 validation rows for unit checks.
    The qualified producer separately enforces the exact full plan cardinality.
    It replays already specified historical draws; no extra observation budget
    or independent physical-domain/model-quality claim follows from success.
    """
    if type(recipe) is not Behavior_recipe_v2 or not isinstance(sample_keys, (tuple, list)) or not sample_keys:
        raise ValueError("behavior validation requires a fixed recipe and explicit nonempty sample keys")
    recipe.validate()
    config = recipe.teacher_config()
    for key in sample_keys:
        if type(key) is not V24SampleKey:
            raise ValueError("behavior keys must be canonical sample-key objects")
        config.validate(key)
        if key.global_seed != recipe.dataset_seed:
            raise ValueError("behavior sample seed differs from its fixed dataset recipe")
    splits = {key.split for key in sample_keys}
    if len(splits) != 1 or len({key.sample_id for key in sample_keys}) != len(sample_keys):
        raise ValueError("behavior split keys must be unique and from one registered split")
    split = sample_keys[0].split
    if split not in ("train", "validation"):
        raise ValueError("behavior validation accepts only TRAIN or validation splits")
    rows, mode_count = len(sample_keys), recipe.mode_count
    if rows > (recipe.train_samples if split == "train" else recipe.validation_samples):
        raise ValueError("behavior validation exceeds the registered split budget")
    _validate_group(observations, sample_keys, store="observation", mode_count=mode_count)
    _validate_group(truths, sample_keys, store="truth", mode_count=mode_count)
    validate_view_keys("feedback_student_base", observations)
    shapes = {"history_tokens": ((rows, 10, mode_count, 7), np.float32),
              "history_mask": ((rows, 10), np.bool_), "mode_mask": ((rows, mode_count), np.bool_),
              "noisy_measurement_tokens": ((rows, mode_count, 2), np.float32),
              "target_mode_tokens": ((rows, mode_count, 2), np.float32)}
    for name, (shape, dtype) in shapes.items():
        _require_array(observations, name, shape, dtype)
    if not observations["history_mask"].all() or not observations["mode_mask"].all():
        raise ValueError("behavior pilot requires all ten history rounds and every active mode")
    for index in range(rows):
        Pilot_history_dataset._validate_row_shapes({name: observations[name][index] for name in PILOT_OBSERVATION_FIELDS}, mode_count)
    for name in ("current_coefficients_re", "current_coefficients_im"):
        _require_array(truths, name, (rows, mode_count), np.float64)
    for name in ("control_parameter", "irrecoverable_parameter", "kappa_inj", "kappa_target"):
        _require_array(truths, name, (rows, 1), np.float64)
    for name in ("physical_noise_parameter", "true_severity", "eta_target"):
        _require_array(truths, name, (rows,), np.float64)
    for name in ("control_mechanism_name", "irrecoverable_mechanism_name"):
        if name not in truths or truths[name].shape != (rows, 1) or truths[name].dtype.kind != "S":
            raise ValueError("behavior nuisance mechanism name shape or dtype is invalid")
    for name in ("d3_train_fit_sha",):
        if name not in truths or truths[name].shape != (rows,) or truths[name].dtype.kind != "S":
            raise ValueError("behavior nuisance identity is missing")
    _require_array(truths, "history_round_count", (rows,), np.int16)
    _require_array(truths, "target_history_gain", (rows,), np.float32)
    _require_array(truths, "target_history_gain_mask", (rows,), np.bool_)
    for name in ("oracle_action", "target_action"):
        _require_array(truths, name, (rows, mode_count, 2), np.float32)
    for name in ("true_next_measurement", "target_next_measurement"):
        _require_array(truths, name, (rows, mode_count, 2), np.float32)
    if (not np.array_equal(truths["oracle_action"], truths["target_action"])
            or not np.array_equal(truths["true_next_measurement"], truths["target_next_measurement"])):
        raise ValueError("behavior single-step truth aliases are inconsistent")
    _require_array(truths, "target_noop_gain", (rows,), np.float32)
    if np.any(truths["target_noop_gain"] != 0):
        raise ValueError("behavior no-op gain must remain zero")
    _validate_count_panel(observations, recipe, rows)
    for index, key in enumerate(sample_keys):
        _validate_keyed_history(observations, truths, index, key, recipe)


__all__ = ["validate_behavior_split", "behavior_validation_scope"]
