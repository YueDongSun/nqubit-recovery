import json
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np

from basis_configs import ensure_basis_config
from system_specs import SYSTEM_SPECS as _SPECS

SYSTEM_SPECS = {
    "3qubit": {
        "core_dim": _SPECS["3qubit"]["core_dim"],
        "full_dim": _SPECS["3qubit"]["full_dim"],
        "full_indices": _SPECS["3qubit"]["full_indices"],
        "core_indices": _SPECS["3qubit"]["core_indices"],
    },
    "4qubit": {
        "core_dim": _SPECS["4qubit"]["core_dim"],
        "full_dim": _SPECS["4qubit"]["full_dim"],
        "full_indices": _SPECS["4qubit"]["full_indices"],
        "core_indices": _SPECS["4qubit"]["core_indices"],
    },
}


def load_basis_entries(system_type: str, basis_config_path: str) -> List[dict]:
    path = Path(basis_config_path)
    if path.exists():
        with open(path, "r", encoding="utf-8") as handle:
            config = json.load(handle)
    else:
        config = ensure_basis_config(str(path), system_type)

    if config.get("system_type") != system_type:
        raise ValueError(f"Basis config {basis_config_path} is not for {system_type}")

    if system_type == "3qubit":
        return list(config["entries"])

    flattened = []
    for cycle in config["cycles"]:
        flattened.extend(cycle["entries"])
    return flattened


BASIS_PHASE_CONVENTION = "read_aod_negative_order"


def _basis_state(entry: dict, phase_sign: float = -1.0) -> np.ndarray:
    amp = np.asarray(entry["amp"], dtype=float)
    phase = float(phase_sign) * np.asarray(entry["phase"], dtype=float)
    state = amp * np.exp(1j * phase)
    norm = np.linalg.norm(state)
    if norm == 0:
        return np.zeros_like(state, dtype=complex)
    return state / norm


def _measurement_operator(entry: dict, phase_sign: float = -1.0) -> np.ndarray:
    state = _basis_state(entry, phase_sign=phase_sign)
    return np.outer(state, state.conj())


def _counts_to_probabilities(counts: np.ndarray, shots: Optional[Iterable[float]] = None) -> tuple[np.ndarray, dict]:
    if shots is None:
        scale = float(np.max(counts)) if np.max(counts) > 0 else 1.0
        return counts / scale, {"normalization": "max-count", "shots": None, "scale": scale}

    if np.isscalar(shots):
        scale = float(shots)
        return counts / scale, {"normalization": "scalar-shots", "shots": scale}

    shot_array = np.asarray(list(shots), dtype=float)
    if shot_array.shape != counts.shape:
        raise ValueError("shots iterable must match counts shape")
    return counts / shot_array, {"normalization": "per-basis-shots", "shots": shot_array.tolist()}


def _project_physical_density_matrix(rho: np.ndarray) -> np.ndarray:
    hermitian = 0.5 * (rho + rho.conj().T)
    eigenvalues, eigenvectors = np.linalg.eigh(hermitian)
    clipped = np.clip(eigenvalues, 0.0, None)
    if np.sum(clipped) == 0:
        clipped[0] = 1.0
    projected = eigenvectors @ np.diag(clipped) @ eigenvectors.conj().T
    return projected / np.trace(projected)


def reconstruct_density_matrix(
    system_type: str,
    counts: Iterable[float],
    basis_config_path: str,
    shots: Optional[Iterable[float]] = None,
    regularization: float = 0.0,
    phase_sign: float = -1.0,
):
    """Reconstruct density matrix from measurement counts using linear inversion.

    Args:
        system_type: '3qubit' or '4qubit'.
        counts: Measurement counts for each basis entry.
        basis_config_path: Path to the basis configuration JSON file.
        shots: Optional shot counts for per-basis normalization.
        regularization: Tikhonov regularization parameter. When > 0, solves
            (M^T M + regularization * I) x = M^T p instead of the standard
            least-squares problem.
            This stabilizes reconstruction when the measurement matrix is ill-conditioned.
            Default 0.0 disables regularization for backward compatibility.
        phase_sign: Sign applied to phases stored in the basis config before
            building measurement states. The hardware default is -1.0 because
            the read AOD is a negative diffraction-order device. Pass +1.0 to
            reproduce the historical software convention explicitly.
    """
    if system_type not in SYSTEM_SPECS:
        raise ValueError(f"Unsupported system_type: {system_type}")

    phase_sign = float(phase_sign)
    counts_array = np.asarray(list(counts), dtype=float)
    basis_entries = load_basis_entries(system_type, basis_config_path)
    if counts_array.size != len(basis_entries):
        raise ValueError(f"Expected {len(basis_entries)} counts for {system_type}, got {counts_array.size}")

    probabilities, normalization_info = _counts_to_probabilities(counts_array, shots)
    operators = [_measurement_operator(entry, phase_sign=phase_sign) for entry in basis_entries]

    core_dim = SYSTEM_SPECS[system_type]["core_dim"]
    measurement_matrix = np.vstack([operator.reshape(1, core_dim * core_dim) for operator in operators])

    if regularization > 0.0:
        measurement_matrix_h = measurement_matrix.conj().T
        MtM = measurement_matrix_h @ measurement_matrix
        Mtp = measurement_matrix_h @ probabilities
        identity = np.eye(MtM.shape[0])
        solution = np.linalg.solve(MtM + regularization * identity, Mtp)
    else:
        solution, _, _, _ = np.linalg.lstsq(measurement_matrix, probabilities, rcond=None)
    rho_linear = solution.reshape(core_dim, core_dim)
    rho_linear = 0.5 * (rho_linear + rho_linear.conj().T)
    rho_physical = _project_physical_density_matrix(rho_linear)

    full_dim = SYSTEM_SPECS[system_type]["full_dim"]
    full_indices = SYSTEM_SPECS[system_type]["full_indices"]
    rho_full = np.zeros((full_dim, full_dim), dtype=complex)
    for i, row_index in enumerate(full_indices):
        for j, col_index in enumerate(full_indices):
            rho_full[row_index, col_index] = rho_physical[i, j]

    diagnostics = {
        "trace": float(np.real(np.trace(rho_physical))),
        "hermitian_residual": float(np.linalg.norm(rho_physical - rho_physical.conj().T)),
        "eigenvalues": np.real(np.linalg.eigvalsh(rho_physical)).tolist(),
        "measurement_rank": int(np.linalg.matrix_rank(measurement_matrix)),
        "normalization": normalization_info,
        "phase_sign": phase_sign,
        "basis_phase_convention": BASIS_PHASE_CONVENTION,
    }

    return {
        "system_type": system_type,
        "basis_config_path": basis_config_path,
        "rho_linear": rho_linear,
        "rho_core": rho_physical,
        "rho_full": rho_full,
        "re_rho": np.real(rho_full),
        "im_rho": np.imag(rho_full),
        "re_rho_core": np.real(rho_physical),
        "im_rho_core": np.imag(rho_physical),
        "diagnostics": diagnostics,
    }
