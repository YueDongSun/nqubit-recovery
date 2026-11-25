"""Born-rule measurement simulation for registered global Pauli settings."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from .contracts import V10MeasurementObservation
from .state_backends import V10StateBackend, all_pauli_outcomes


MAX_ENUMERATED_QUBITS = 16


def born_probabilities(
    backend: V10StateBackend,
    pauli_axes: Sequence[str],
    *,
    max_enumerated_qubits: int = MAX_ENUMERATED_QUBITS,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the exact categorical Born law in lexicographic +/-1 order."""
    if backend.N > max_enumerated_qubits:
        raise ValueError(
            "exact outcome enumeration is exponential; use a backend-specific sequential "
            "sampler for N above max_enumerated_qubits")
    outcomes = all_pauli_outcomes(backend.N)
    probabilities = np.asarray(
        [backend.probability(pauli_axes, outcome) for outcome in outcomes],
        dtype=np.float64)
    if not np.all(np.isfinite(probabilities)) or probabilities.min(initial=0.0) < -1e-10:
        raise FloatingPointError("Born probabilities are non-finite or negative")
    total = float(probabilities.sum())
    if not np.isclose(total, 1.0, atol=1e-8):
        raise FloatingPointError(f"Born probabilities do not normalize: {total}")
    probabilities /= total
    return outcomes, probabilities


def simulate_pauli_observation(
    *,
    sample_id: str,
    setting_id: str,
    setting_registry_sha: str,
    pauli_axes: Sequence[str],
    backend: V10StateBackend,
    shot_mode: str,
    observation_model: str,
    requested_shots: int,
    rng: np.random.Generator,
    exposure: float = 1.0,
    poisson_rate_hz: Optional[float] = None,
) -> V10MeasurementObservation:
    """Simulate one setting without conflating fixed-count and Poisson data.

    ``multinomial`` fixes the realized total to ``requested_shots``. ``poisson``
    first draws the realized number of detections from ``rate_hz * exposure`` and
    then draws their categorical outcomes. ``ideal`` returns probabilities only.
    """
    if len(pauli_axes) != backend.N:
        raise ValueError("setting/backend N mismatch")
    if requested_shots < 0 or exposure < 0.0:
        raise ValueError("requested shots and exposure must be non-negative")
    expected_model = {
        "ideal": "ideal",
        "fixed_total": "multinomial",
        "fixed_per_setting": "multinomial",
    }
    if shot_mode not in expected_model:
        raise ValueError("shot_mode must be ideal, fixed_total, or fixed_per_setting")
    if shot_mode == "ideal" and observation_model != "ideal":
        raise ValueError("ideal shot mode requires ideal observation model")
    if shot_mode != "ideal" and observation_model not in ("multinomial", "poisson"):
        raise ValueError("sampled shot modes require multinomial or poisson observations")

    outcome_space, probabilities = born_probabilities(backend, pauli_axes)
    if shot_mode == "ideal":
        sampled = np.empty((0, backend.N), dtype=np.int8)
        realized_shots = 0
    else:
        if observation_model == "multinomial":
            realized_shots = int(requested_shots)
        else:
            if poisson_rate_hz is None or poisson_rate_hz < 0.0:
                raise ValueError("poisson observations require non-negative poisson_rate_hz")
            realized_shots = int(rng.poisson(float(poisson_rate_hz) * exposure))
        indices = rng.choice(len(probabilities), size=realized_shots, p=probabilities)
        sampled = outcome_space[indices]

    result = V10MeasurementObservation(
        sample_id=sample_id,
        setting_id=setting_id,
        setting_registry_sha=setting_registry_sha,
        shot_mode=shot_mode,
        observation_model=observation_model,
        outcomes=sampled,
        probabilities=probabilities,
        exposure=float(exposure),
        requested_shots=int(requested_shots),
        realized_shots=realized_shots,
        valid_shot_mask=np.ones(realized_shots, dtype=np.bool_),
    )
    result.validate(backend.N)
    return result
