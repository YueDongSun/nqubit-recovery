"""Executable D0-D2 measurement contracts with exact shot-budget semantics."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from ml_recon.physics_hybrid_v10.contracts import sha256_of
from ml_recon.physics_hybrid_v10.measurement_model import simulate_pauli_observation
from ml_recon.physics_hybrid_v10.measurement_scheduler import V10MeasurementScheduler
from ml_recon.physics_hybrid_v10.shot_budget import (
    S_IDEAL,
    S_SETTING,
    S_TOTAL,
    allocate_shots,
)
from ml_recon.physics_hybrid_v10.state_backends import DenseDensityStateBackend
from ml_recon.physics_hybrid_v10.synthetic_manifest import V10CounterBasedRng

from ..physics.collision import (
    CollisionArm,
    build_collision_arm,
    default_collision_measurement_contract,
)
from ..physics.fixed_k import FixedKState
from .curriculum import TrackACurriculumCell


SHOT_MODE_TO_PUBLIC = {
    S_IDEAL: "ideal",
    S_TOTAL: "fixed_total",
    S_SETTING: "fixed_per_setting",
}


@dataclass(frozen=True)
class TrackAMeasurementResult:
    observation: Dict[str, Any]
    truth: Dict[str, Any]
    measurement_contract_sha: str
    physical_token_count: int
    paired_crn_sha: str = ""


def _pauli_settings(cell: TrackACurriculumCell) -> tuple[tuple[str, ...], ...]:
    if cell.measurement_family == "full_pauli":
        return tuple(itertools.product(("X", "Y", "Z"), repeat=cell.n))
    if cell.measurement_family == "incomplete_pauli":
        return (("Z",) * cell.n, ("X",) * cell.n, ("Y",) * cell.n)
    raise NotImplementedError(
        f"no authoritative Pauli renderer is registered for {cell.measurement_family}")


def measurement_setting_ids(cell: TrackACurriculumCell) -> tuple[str, ...]:
    if not cell.renderer_available:
        raise NotImplementedError(
            f"{cell.stage} has no authoritative Track A physical renderer")
    if cell.stage in {"D0", "D1"}:
        return tuple(
            f"pauli_{''.join(axes)}" for axes in _pauli_settings(cell))
    if cell.stage == "D2":
        return default_collision_measurement_contract(cell.n).setting_ids
    raise NotImplementedError(f"no renderer is registered for stage {cell.stage}")


def planned_physical_token_count(cell: TrackACurriculumCell, budget: int) -> int:
    setting_ids = measurement_setting_ids(cell)
    allocation = allocate_shots(setting_ids, cell.budget_mode, int(budget))
    if allocation.probability_only:
        category_count = (
            2 ** cell.n if cell.stage in {"D0", "D1"}
            else math.comb(cell.n, cell.k)
        )
        return len(setting_ids) * category_count
    return allocation.total_shots * cell.n


def _render_pauli_measurement(
    *,
    cell: TrackACurriculumCell,
    state: FixedKState,
    sample_id: str,
    budget: int,
    rng: V10CounterBasedRng,
) -> TrackAMeasurementResult:
    settings = _pauli_settings(cell)
    setting_ids = tuple(f"pauli_{''.join(axes)}" for axes in settings)
    if cell.measurement_family == "full_pauli":
        canonical_schedule = V10MeasurementScheduler(cell.n).typed_schedule()
        if (canonical_schedule.setting_ids != setting_ids
                or canonical_schedule.pauli_axes != settings):
            raise RuntimeError("Track A full-Pauli schedule diverged from the V10 registry")
        registry_sha = canonical_schedule.setting_registry_sha
    else:
        registry_sha = sha256_of({
            "schema": "V10TrackAPauliScheduleV1",
            "n": cell.n,
            "setting_ids": list(setting_ids),
            "axes": [list(axes) for axes in settings],
        })
    allocation = allocate_shots(setting_ids, cell.budget_mode, int(budget))
    shot_mode = SHOT_MODE_TO_PUBLIC[cell.budget_mode]
    backend = DenseDensityStateBackend.from_state_vector(
        state.dense_state_vector(dense_n_limit=cell.n))
    observation_records = []
    truth_probabilities = []
    mode_counts = np.zeros(cell.n, dtype=np.int64)
    physical_tokens = 0
    for setting_id, axes in zip(setting_ids, settings):
        result = simulate_pauli_observation(
            sample_id=sample_id,
            setting_id=setting_id,
            setting_registry_sha=registry_sha,
            pauli_axes=axes,
            backend=backend,
            shot_mode=shot_mode,
            observation_model=("ideal" if allocation.probability_only else "multinomial"),
            requested_shots=allocation.shots_by_setting[setting_id],
            rng=rng.stream("measurement", setting_id),
        )
        exposed_probabilities = (
            np.asarray(result.probabilities, dtype=np.float64)
            if allocation.probability_only else None
        )
        observation_records.append({
            "setting_id": setting_id,
            "pauli_axes": axes,
            "outcomes": np.asarray(result.outcomes, dtype=np.int8),
            "probabilities": exposed_probabilities,
            "requested_shots": result.requested_shots,
            "realized_shots": result.realized_shots,
        })
        truth_probabilities.append(np.asarray(result.probabilities, dtype=np.float64))
        physical_tokens += (
            int(exposed_probabilities.size) if exposed_probabilities is not None
            else int(result.outcomes.size)
        )
        if axes == ("Z",) * cell.n and result.outcomes.size:
            mode_counts += np.sum((1 - result.outcomes) // 2, axis=0, dtype=np.int64)
    return TrackAMeasurementResult(
        observation={
            "measurement_records": tuple(observation_records),
            "mode_counts": mode_counts,
            "shots": allocation.total_shots,
            "measurement_family": cell.measurement_family,
            "measurement_contract_sha": registry_sha,
            "budget_mode": cell.budget_mode,
            "probability_only": allocation.probability_only,
        },
        truth={"measurement_probabilities": tuple(truth_probabilities)},
        measurement_contract_sha=registry_sha,
        physical_token_count=physical_tokens,
    )


def _render_collision_measurement(
    *,
    cell: TrackACurriculumCell,
    state: FixedKState,
    sample_id: str,
    budget: int,
    paired_rng: V10CounterBasedRng,
) -> TrackAMeasurementResult:
    contract = default_collision_measurement_contract(cell.n)
    allocation = allocate_shots(contract.setting_ids, cell.budget_mode, int(budget))
    collision_mass = float(paired_rng.stream("collision_mass").uniform(0.05, 0.35))
    arm_state = build_collision_arm(
        state,
        collision_mass=collision_mass,
        arm=CollisionArm(cell.collision_arm),
    )
    observation_records = []
    truth_probabilities = []
    crn_payload = []
    mode_counts = np.zeros(cell.n, dtype=np.int64)
    physical_tokens = 0
    for setting_id, unitary in zip(contract.setting_ids, contract.unitaries):
        probabilities = arm_state.measurement_probabilities(unitary)
        requested_shots = allocation.shots_by_setting[setting_id]
        if allocation.probability_only:
            outcomes = np.zeros((0, cell.n), dtype=np.int8)
            exposed_probabilities = probabilities
        else:
            uniforms = paired_rng.stream("collision_measurement_crn", setting_id).random(
                requested_shots)
            indices = np.searchsorted(np.cumsum(probabilities), uniforms, side="right")
            indices = np.minimum(indices, probabilities.size - 1)
            outcomes = np.asarray(arm_state.occupation_basis[indices], dtype=np.int8)
            exposed_probabilities = None
            crn_payload.append(uniforms.tolist())
            mode_counts += np.sum(outcomes, axis=0, dtype=np.int64)
        observation_records.append({
            "setting_id": setting_id,
            "outcomes": outcomes,
            "probabilities": exposed_probabilities,
            "requested_shots": requested_shots,
            "realized_shots": int(outcomes.shape[0]),
        })
        truth_probabilities.append(probabilities)
        physical_tokens += (
            int(probabilities.size) if allocation.probability_only else int(outcomes.size))
    return TrackAMeasurementResult(
        observation={
            "measurement_records": tuple(observation_records),
            "mode_counts": mode_counts,
            "shots": allocation.total_shots,
            "measurement_family": cell.measurement_family,
            "measurement_contract_sha": contract.sha256(),
            "budget_mode": cell.budget_mode,
            "probability_only": allocation.probability_only,
        },
        truth={
            "collision_arm_state": arm_state,
            "measurement_probabilities": tuple(truth_probabilities),
        },
        measurement_contract_sha=contract.sha256(),
        physical_token_count=physical_tokens,
        paired_crn_sha=sha256_of(crn_payload),
    )


def render_track_a_measurement(
    *,
    cell: TrackACurriculumCell,
    state: FixedKState,
    sample_id: str,
    budget: int,
    rng: V10CounterBasedRng,
    paired_rng: V10CounterBasedRng,
) -> TrackAMeasurementResult:
    """Dispatch only to implemented physical renderers and fail closed otherwise."""

    if not cell.renderer_available:
        raise NotImplementedError(
            f"{cell.stage} renderer is unimplemented; labels cannot substitute for observations")
    if cell.stage in {"D0", "D1"}:
        return _render_pauli_measurement(
            cell=cell, state=state, sample_id=sample_id, budget=budget, rng=rng)
    if cell.stage == "D2":
        return _render_collision_measurement(
            cell=cell,
            state=state,
            sample_id=sample_id,
            budget=budget,
            paired_rng=paired_rng,
        )
    raise NotImplementedError(f"no renderer is registered for stage {cell.stage}")
