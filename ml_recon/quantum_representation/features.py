"""Feature extraction for complex density matrices."""

from __future__ import annotations

import numpy as np

from .tomography_features import (
    infer_num_qubits_from_dim,
    rho_pair_distance_features,
    tomographic_residual_summary,
    validate_density_pair,
)


def rho_single_features(rho: np.ndarray) -> np.ndarray:
    """Return compact, real-valued diagnostics for one density matrix."""

    mat = np.asarray(rho, dtype=complex)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError("rho must be a square matrix")

    dim = mat.shape[0]
    herm = 0.5 * (mat + mat.conj().T)
    diag = np.real(np.diag(herm))
    offdiag = herm - np.diag(np.diag(herm))
    eigvals = np.linalg.eigvalsh(herm)
    eigvals = np.sort(np.real(eigvals))[::-1]
    eig_pad = np.zeros(4, dtype=float)
    eig_pad[: min(4, eigvals.size)] = eigvals[:4]

    abs_off = np.abs(offdiag)
    phases = np.angle(offdiag[np.abs(offdiag) > 1e-12])
    phase_spread = float(np.std(phases)) if phases.size else 0.0
    purity = float(np.real(np.trace(herm @ herm)))
    trace = float(np.real(np.trace(herm)))

    return np.asarray([
        trace,
        purity,
        float(np.max(diag)) if diag.size else 0.0,
        float(np.min(diag)) if diag.size else 0.0,
        float(np.std(diag)) if diag.size else 0.0,
        float(np.sum(abs_off)) / max(dim * (dim - 1), 1),
        float(np.max(abs_off)) if abs_off.size else 0.0,
        phase_spread,
        *eig_pad.tolist(),
    ], dtype=float)


def rho_pair_features(
    rho_target: np.ndarray,
    rho_measured: np.ndarray,
    num_qubits: int | None = None,
) -> np.ndarray:
    """Return physics-aware features for target/measured density-matrix pairs."""

    target, measured = validate_density_pair(rho_target, rho_measured)
    if num_qubits is None:
        num_qubits = infer_num_qubits_from_dim(target.shape[0])
    target_features = rho_single_features(target)
    measured_features = rho_single_features(measured)
    distance_features = rho_pair_distance_features(target, measured)
    if num_qubits is None:
        tomography_features = np.zeros(0, dtype=float)
    else:
        tomography_features = tomographic_residual_summary(target, measured, int(num_qubits))

    return np.concatenate([
        target_features,
        measured_features,
        distance_features,
        tomography_features,
    ])


def rho_pair_legacy_diff_features(rho_target: np.ndarray, rho_measured: np.ndarray) -> np.ndarray:
    """Return the historical raw-difference features for old JSON checkpoints."""

    target, measured = validate_density_pair(rho_target, rho_measured)
    diff = target - measured
    diag_diff = np.real(np.diag(diff))
    offdiag_diff = diff - np.diag(np.diag(diff))
    trace_distance_proxy = 0.5 * np.linalg.norm(diff, ord="nuc")
    target_features = rho_single_features(target)
    measured_features = rho_single_features(measured)

    return np.concatenate([
        target_features,
        measured_features,
        np.asarray([
            float(np.linalg.norm(diff)),
            float(trace_distance_proxy),
            float(np.mean(diag_diff)) if diag_diff.size else 0.0,
            float(np.std(diag_diff)) if diag_diff.size else 0.0,
            float(np.mean(np.abs(offdiag_diff))) if offdiag_diff.size else 0.0,
            float(np.max(np.abs(offdiag_diff))) if offdiag_diff.size else 0.0,
        ], dtype=float),
    ])


def density_fidelity_pure_target(rho_target: np.ndarray, rho_measured: np.ndarray) -> float:
    """Fast fidelity for pure or nearly pure target rho."""

    target = np.asarray(rho_target, dtype=complex)
    measured = np.asarray(rho_measured, dtype=complex)
    eigvals, eigvecs = np.linalg.eigh(0.5 * (target + target.conj().T))
    idx = int(np.argmax(np.real(eigvals)))
    psi = eigvecs[:, idx]
    return float(np.clip(np.real(np.vdot(psi, measured @ psi)), 0.0, 1.0))
