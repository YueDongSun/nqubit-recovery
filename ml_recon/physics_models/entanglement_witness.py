"""Count-space W-witness estimation without full tomography (V5 upgrade).

The W-state fidelity witness ``W = alpha * I - |psi><psi|`` is an observable:
its expectation is a *linear* functional of the measurement probabilities,
so it can be estimated directly from counts with Poisson/binomial error
propagation — no density-matrix reconstruction required.  This yields a fast
entanglement certificate for the data UIs, a physics audit field for
snapshots, and (on the synthetic side) a free regression target.

The target overlap ``F = <psi|rho|psi>`` is estimated by expanding
``|psi><psi|`` in the tomography measurement basis (least squares over
operator vectorizations, real coefficients since both sides are Hermitian).
The expansion residual is always reported: the tomography bases used here
span the single-excitation operator space, so ``|psi><psi|`` for a
single-excitation target is exactly representable (residual ~ 1e-12), but a
target with support outside the measured subspace would be silently
projected — the residual guards that case.

This module is part of the research-only V5 upgrade programme documented in
``docs/ml/architecture/v5/design/V5_PHYSICS_MATH_UPGRADES.md``.  It has no
hardware-write authority.

References: Bourennane et al., Phys. Rev. Lett. 92, 087902 (2004);
Guhne & Toth, Phys. Rep. 474, 1 (2009).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from ml_recon.physics_models.entanglement_audit import (
    EPS,
    biseparable_bound_single_excitation,
    fully_separable_bound_single_excitation,
)


def witness_overlap_coefficients(
    measurement_operators: np.ndarray,
    target_vector: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """Real least-squares coefficients ``c`` with ``sum_o c_o Pi_o ~= |psi><psi|``.

    Returns ``(coefficients, relative_residual)`` where the residual is the
    Frobenius norm of the expansion error relative to the operator norm.
    """

    operators = np.asarray(measurement_operators, dtype=complex)
    if operators.ndim != 3 or operators.shape[1] != operators.shape[2]:
        raise ValueError("measurement_operators must have shape (operations, dim, dim)")
    target = np.asarray(target_vector, dtype=complex).reshape(-1)
    projector = np.outer(target, target.conj())
    if projector.shape != operators.shape[1:]:
        raise ValueError(
            f"target operator shape {projector.shape} does not match measurement "
            f"operators {operators.shape[1:]}"
        )
    design = operators.reshape(operators.shape[0], -1).T  # (dim*dim, operations)
    rhs = projector.reshape(-1)
    design_real = np.vstack((design.real, design.imag))
    rhs_real = np.concatenate((rhs.real, rhs.imag))
    coefficients, residuals, rank, _ = np.linalg.lstsq(design_real, rhs_real, rcond=None)
    expansion = design @ coefficients
    residual = float(np.linalg.norm(expansion - rhs) / max(np.linalg.norm(rhs), EPS))
    return np.real(coefficients), residual


def _target_in_operator_space(
    num_qubits: int,
    operator_dim: int,
    target_amplitudes,
    target_phases,
) -> np.ndarray:
    """Build the W-like target vector in the operator's native space.

    Single-excitation targets live equivalently in the full 2^N space (modes
    at bit positions 1<<q) or the N-dim core (modes 0..N-1); pick the one
    matching ``operator_dim`` so the projector expands against the supplied
    operators without a shape mismatch.
    """

    n = int(num_qubits)
    amps = np.ones(n, dtype=float) if target_amplitudes is None else np.asarray(target_amplitudes, dtype=float).reshape(-1)
    phs = np.zeros(n, dtype=float) if target_phases is None else np.asarray(target_phases, dtype=float).reshape(-1)
    core = amps * np.exp(1j * phs)
    norm = np.linalg.norm(core)
    if norm <= EPS:
        raise ValueError("target amplitudes must not all vanish")
    core = core / norm
    if operator_dim == n:
        return core
    if operator_dim == 2 ** n:
        vec = np.zeros(operator_dim, dtype=complex)
        for q in range(n):
            vec[1 << q] = core[q]
        return vec
    raise ValueError(
        f"operator dimension {operator_dim} matches neither num_qubits {n} "
        f"nor 2^num_qubits {2 ** n}; cannot place the single-excitation target"
    )


def witness_from_probabilities(
    probabilities: np.ndarray,
    probability_variances: np.ndarray,
    measurement_operators: np.ndarray,
    num_qubits: int,
    *,
    target_amplitudes: Optional[Sequence[float]] = None,
    target_phases: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """Estimate the W-witness value from measurement probabilities.

    ``probabilities`` are the per-operation Born-probability estimates and
    ``probability_variances`` their (estimated) variances; both are aligned
    with ``measurement_operators``.  Returns the overlap, both tight margins,
    the propagated one-sigma uncertainty, and the expansion residual.
    """

    n = int(num_qubits)
    p = np.asarray(probabilities, dtype=float).reshape(-1)
    variances = np.asarray(probability_variances, dtype=float).reshape(-1)
    operators = np.asarray(measurement_operators, dtype=complex)
    if p.size != operators.shape[0] or variances.size != operators.shape[0]:
        raise ValueError("probabilities/variances must align with measurement_operators")
    if np.any(variances < 0):
        raise ValueError("probability_variances must be non-negative")

    target = _target_in_operator_space(
        n, operators.shape[1], target_amplitudes, target_phases
    )
    coefficients, residual = witness_overlap_coefficients(operators, target)
    overlap = float(np.dot(coefficients, p))
    overlap_variance = float(np.dot(coefficients**2, variances))
    overlap_std = float(np.sqrt(max(overlap_variance, 0.0)))

    uniform = target_amplitudes is None and target_phases is None
    if uniform:
        genuine_bound = (n - 1.0) / n
        separable_bound = genuine_bound ** (n - 1.0)
        bound_kind = "analytic"
    else:
        core_amplitudes = np.abs(np.asarray(target_amplitudes, dtype=float).reshape(-1))
        genuine_bound = biseparable_bound_single_excitation(core_amplitudes)
        separable_bound = fully_separable_bound_single_excitation(core_amplitudes)
        bound_kind = "numeric"
    z_genuine = (overlap - genuine_bound) / max(overlap_std, EPS)
    return {
        "schema": "count_space_witness_v1",
        "v5_upgrade": True,
        "w_overlap": overlap,
        "w_overlap_std": overlap_std,
        "genuine_bound": float(genuine_bound),
        "separable_bound": float(separable_bound),
        "bound_kind": bound_kind,
        "witness_value": float(genuine_bound - overlap),
        "witness_value_std": overlap_std,
        "genuine_margin": float(overlap - genuine_bound),
        "separable_margin": float(overlap - separable_bound),
        "genuine_significance_sigma": float(z_genuine),
        "certified_genuine_multipartite": bool(overlap - genuine_bound > 0.0),
        "certified_entangled": bool(overlap - separable_bound > 0.0),
        "expansion_residual": residual,
        "expansion_reliable": bool(residual < 1e-6),
    }


def witness_from_counts(
    counts: np.ndarray,
    shots: np.ndarray,
    measurement_operators: np.ndarray,
    num_qubits: int,
    *,
    target_amplitudes: Optional[Sequence[float]] = None,
    target_phases: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """Binomial wrapper: probabilities ``counts/shots`` with ``p(1-p)/shots`` variance."""

    y = np.asarray(counts, dtype=float).reshape(-1)
    n_shots = np.asarray(shots, dtype=float).reshape(-1)
    if y.shape != n_shots.shape:
        raise ValueError("counts and shots must have the same shape")
    if np.any(n_shots <= 0):
        raise ValueError("shots must be positive for every measured operation")
    probabilities = y / n_shots
    variances = probabilities * np.clip(1.0 - probabilities, 0.0, None) / n_shots
    return witness_from_probabilities(
        probabilities,
        variances,
        measurement_operators,
        num_qubits,
        target_amplitudes=target_amplitudes,
        target_phases=target_phases,
    )


def measurement_operators_for_system(
    system_type: str,
    *,
    phase_sign: float = -1.0,
    basis_config_path: Optional[str | Path] = None,
) -> np.ndarray:
    """Load the production tomography basis as measurement operators.

    ``basis_config_path`` defaults to the standard
    ``matrix/basis_config_<system>.json`` used by the reconstruction path
    (created via ``basis_configs.ensure_basis_config`` when absent).  Dynamic
    import across the documented boundary (``qubit_recon_files`` is imported
    lazily, mirroring ``runner.py``); this keeps the witness module NumPy-only
    at import time.
    """

    from system_control.qubit_recon_files import density_reconstruction as raw_reconstruction

    if basis_config_path is None:
        matrix_dir = Path(raw_reconstruction.__file__).resolve().parent / "matrix"
        basis_config_path = matrix_dir / f"basis_config_{system_type}.json"
    entries = raw_reconstruction.load_basis_entries(system_type, str(basis_config_path))
    return np.stack(
        [raw_reconstruction._measurement_operator(entry, phase_sign=phase_sign) for entry in entries]
    )
