"""Gauge-free operation-phase geometry and registry-bound matrix mapping."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np


def helmert_matrix(cell_count: int) -> np.ndarray:
    """Return an orthonormal zero-sum basis with shape ``[n, n-1]``."""

    n = int(cell_count)
    if n < 2:
        raise ValueError("cell_count must be at least 2")
    matrix = np.zeros((n, n - 1), dtype=float)
    for column in range(n - 1):
        scale = np.sqrt((column + 1) * (column + 2))
        matrix[: column + 1, column] = 1.0 / scale
        matrix[column + 1, column] = -(column + 1) / scale
    return matrix


def materialize_gauge_coordinates(coordinates: np.ndarray) -> np.ndarray:
    values = np.asarray(coordinates, dtype=float)
    if values.shape[-1] < 1:
        raise ValueError("gauge coordinates must contain at least one observable mode")
    return values @ helmert_matrix(values.shape[-1] + 1).T


def project_phase_offsets(phase_offsets_rad: np.ndarray) -> np.ndarray:
    values = np.asarray(phase_offsets_rad, dtype=float)
    if values.shape[-1] < 2:
        raise ValueError("phase offsets must contain at least two cells")
    gauge_free = values - np.mean(values, axis=-1, keepdims=True)
    return gauge_free @ helmert_matrix(values.shape[-1])


def action_tensor_to_cycle_matrices(
    system_type: str,
    phase_offsets_rad: np.ndarray,
    operation_registry: Sequence[Mapping[str, Any] | Any],
) -> np.ndarray:
    """Map operation-local phase cells to cycle matrices without collisions."""

    operations, cells, cycles, size = _system_shape(system_type)
    values = np.asarray(phase_offsets_rad, dtype=float)
    if values.shape != (operations, cells):
        raise ValueError(f"{system_type} action tensor must have shape {(operations, cells)}, got {values.shape}")
    registry = _normalise_registry(operation_registry)
    if len(registry) != operations:
        raise ValueError(f"operation registry has {len(registry)} entries, expected {operations}")
    output = np.zeros((cycles, size, size), dtype=float)
    visited = np.zeros((cycles, size, size), dtype=np.int16)
    for operation_index, record in enumerate(registry):
        cycle_index = int(record["cycle_index"])
        coordinates = record["cell_coordinates"]
        if len(coordinates) != cells:
            raise ValueError(f"operation {operation_index} has {len(coordinates)} cells, expected {cells}")
        if not 0 <= cycle_index < cycles:
            raise ValueError(f"operation {operation_index} has invalid cycle_index {cycle_index}")
        for cell_index, coordinate in enumerate(coordinates):
            row, column = (int(coordinate[0]), int(coordinate[1]))
            if not (0 <= row < size and 0 <= column < size):
                raise ValueError(f"operation {operation_index} has out-of-range cell {(row, column)}")
            if visited[cycle_index, row, column]:
                raise ValueError(f"registry collision at cycle={cycle_index}, cell={(row, column)}")
            visited[cycle_index, row, column] = 1
            output[cycle_index, row, column] = values[operation_index, cell_index]
    if not np.all(visited == 1):
        missing = np.argwhere(visited == 0)
        raise ValueError(f"operation registry does not cover {len(missing)} expected cycle cells")
    return output


def cycle_matrices_to_action_tensor(
    system_type: str,
    cycle_matrices: np.ndarray,
    operation_registry: Sequence[Mapping[str, Any] | Any],
) -> np.ndarray:
    operations, cells, cycles, size = _system_shape(system_type)
    values = np.asarray(cycle_matrices, dtype=float)
    if values.shape == (size, size) and cycles == 1:
        values = values[None, :, :]
    if values.shape != (cycles, size, size):
        raise ValueError(f"{system_type} cycle matrices must have shape {(cycles, size, size)}, got {values.shape}")
    registry = _normalise_registry(operation_registry)
    if len(registry) != operations:
        raise ValueError(f"operation registry has {len(registry)} entries, expected {operations}")
    result = np.zeros((operations, cells), dtype=float)
    for operation_index, record in enumerate(registry):
        cycle_index = int(record["cycle_index"])
        coordinates = record["cell_coordinates"]
        if len(coordinates) != cells:
            raise ValueError(f"operation {operation_index} has {len(coordinates)} cells, expected {cells}")
        result[operation_index] = [values[cycle_index, int(row), int(column)] for row, column in coordinates]
    return result


def deterministic_probe_directions(
    *,
    operation_count: int,
    cell_count: int,
    seed: int,
    amplitude_rad: float,
) -> np.ndarray:
    """Generate registry-order deterministic Rademacher directions in Helmert space."""

    if amplitude_rad < 0:
        raise ValueError("amplitude_rad must be non-negative")
    rng = np.random.default_rng(int(seed))
    modes = rng.choice((-1.0, 1.0), size=(int(operation_count), int(cell_count) - 1))
    offsets = materialize_gauge_coordinates(modes)
    maximum = np.max(np.abs(offsets), axis=1, keepdims=True)
    maximum[maximum == 0] = 1.0
    return offsets * (float(amplitude_rad) / maximum)


def regularized_fisher_action_mask(
    fisher_matrices: np.ndarray,
    bootstrap_sign_consistency: np.ndarray,
    posterior_std_rad: np.ndarray,
    *,
    maximum_condition_number: float = 100.0,
    minimum_sign_consistency: float = 0.80,
    maximum_posterior_std_rad: float = 0.05,
    ridge: float = 1e-6,
) -> np.ndarray:
    fisher = np.asarray(fisher_matrices, dtype=float)
    sign = np.asarray(bootstrap_sign_consistency, dtype=float)
    uncertainty = np.asarray(posterior_std_rad, dtype=float)
    if fisher.ndim != 3 or fisher.shape[-1] != fisher.shape[-2]:
        raise ValueError("fisher_matrices must have shape [operation, mode, mode]")
    if sign.shape != (fisher.shape[0],) or uncertainty.shape != fisher.shape[:2]:
        raise ValueError("sign consistency or uncertainty shape mismatch")
    identity = np.eye(fisher.shape[-1], dtype=float)[None, :, :]
    values = np.linalg.eigvalsh(fisher + float(ridge) * identity)
    rank_ok = np.sum(values > max(float(ridge), 1e-12), axis=-1) == fisher.shape[-1]
    condition = values[:, -1] / np.maximum(values[:, 0], float(ridge))
    operation_ok = rank_ok & (condition <= float(maximum_condition_number)) & (sign >= float(minimum_sign_consistency))
    return operation_ok[:, None] & (uncertainty <= float(maximum_posterior_std_rad))


def w_subspace_probability_and_jacobian(
    density: np.ndarray,
    base_phases_rad: np.ndarray,
    gauge_coordinates: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Return a W-subspace projector probability and its analytic Helmert Jacobian."""

    rho = np.asarray(density, dtype=complex)
    base = np.asarray(base_phases_rad, dtype=float).reshape(-1)
    coordinates = np.asarray(gauge_coordinates, dtype=float).reshape(-1)
    if rho.shape != (base.size, base.size) or coordinates.size != base.size - 1:
        raise ValueError("density, phase, and gauge coordinate dimensions do not match")
    basis = helmert_matrix(base.size)
    phases = base + basis @ coordinates
    vector = np.exp(1j * phases) / np.sqrt(base.size)
    projected = rho @ vector
    probability = float(np.clip(np.real(np.vdot(vector, projected)), 0.0, 1.0))
    phase_gradient = 2.0 * np.imag(vector.conj() * projected)
    return probability, phase_gradient @ basis


def central_difference_phase_jacobian(
    density: np.ndarray,
    base_phases_rad: np.ndarray,
    gauge_coordinates: np.ndarray,
    *,
    epsilon: float = 1e-5,
) -> np.ndarray:
    coordinates = np.asarray(gauge_coordinates, dtype=float).reshape(-1)
    jacobian = np.zeros_like(coordinates)
    for index in range(coordinates.size):
        direction = np.zeros_like(coordinates)
        direction[index] = float(epsilon)
        positive, _ = w_subspace_probability_and_jacobian(density, base_phases_rad, coordinates + direction)
        negative, _ = w_subspace_probability_and_jacobian(density, base_phases_rad, coordinates - direction)
        jacobian[index] = (positive - negative) / (2.0 * float(epsilon))
    return jacobian


def _normalise_registry(registry: Iterable[Mapping[str, Any] | Any]) -> list[Mapping[str, Any]]:
    return [item.to_json_dict() if hasattr(item, "to_json_dict") else dict(item) for item in registry]


def _system_shape(system_type: str) -> tuple[int, int, int, int]:
    if system_type == "3qubit":
        return 75, 3, 1, 15
    if system_type == "4qubit":
        return 256, 4, 4, 16
    raise ValueError("system_type must be 3qubit or 4qubit")
