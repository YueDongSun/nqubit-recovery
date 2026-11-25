"""V9 target-conditioned per-operation law (charter section 17).

For a target ``|W_c> = sum_i c_i |0..1_i..0>`` with ``c_i = a_i exp(i phi_i)``:

- gauge-invariant W density coordinates: populations, Re coherences, Im coherences
  (no target name string / path / arbitrary ID encoding);
- Born prior ``b_o(target) = Tr(E_o rho_target)`` per operation (constructive
  interference operations have large ``b_o``, destructive ones small);
- ``log mu_o = global_scale + log(b_o + eps) + bounded target-operation residual
  + nuisance residual``.

Negative controls: target shuffle and operation-mapping shuffle must degrade the law.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .operation_registry import V9OperationRegistryV1, SYSTEM_OPERATION_COUNTS
from .rho8_reconstruction import _load_calibrated_reconstruction_module

_EPS = 1e-12


def _validate_coefficients(c_coeffs: np.ndarray, n: int) -> np.ndarray:
    c = np.asarray(c_coeffs, dtype=np.complex128).reshape(-1)
    if c.size != n:
        raise ValueError(f"target coefficients must have length {n}, got {c.size}")
    if not np.all(np.isfinite(c)):
        raise ValueError("target coefficients must be finite")
    norm = float(np.sum(np.abs(c) ** 2))
    if not np.isclose(norm, 1.0, atol=1e-6):
        raise ValueError(f"target coefficients must be normalized (|c|^2 sum = 1), got {norm}")
    return c


def target_core_rho(c_coeffs: np.ndarray, n: int) -> np.ndarray:
    """Core density |W_c><W_c| (N x N) from normalized coefficients."""
    c = _validate_coefficients(c_coeffs, n)
    return np.outer(c, c.conj())


def target_density_features(c_coeffs: np.ndarray, n: int) -> Dict[str, np.ndarray]:
    """Gauge-invariant W density coordinates (populations + Re/Im coherences)."""
    c = _validate_coefficients(c_coeffs, n)
    populations = np.abs(c) ** 2
    re_coh = np.zeros((n, n), dtype=np.float64)
    im_coh = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            re_coh[i, j] = np.real(c[i] * np.conj(c[j]))
            im_coh[i, j] = np.imag(c[i] * np.conj(c[j]))
    return {"populations": populations, "re_coherences": re_coh, "im_coherences": im_coh}


def target_rho_features(rho_target: np.ndarray) -> Dict[str, np.ndarray]:
    """Gauge-invariant coordinates of an arbitrary target density (pure, mixed or non-W)."""
    rho = np.asarray(rho_target, dtype=np.complex128)
    n = rho.shape[0]
    populations = np.real(np.diag(rho))
    re_coh = np.zeros((n, n), dtype=np.float64)
    im_coh = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            re_coh[i, j] = np.real(rho[i, j])
            im_coh[i, j] = np.imag(rho[i, j])
    return {"populations": populations, "re_coherences": re_coh, "im_coherences": im_coh}


def _load_mode_vectors(system_type: str) -> List[np.ndarray]:
    module = _load_calibrated_reconstruction_module()
    import pathlib

    basis_path = str(
        pathlib.Path("system_control/qubit_recon_files/matrix")
        / f"basis_config_{system_type}.json"
    )
    entries = module.raw_reconstruction.load_basis_entries(system_type, basis_path)
    n = 3 if system_type == "3qubit" else 4
    modes = []
    for entry in entries:
        amp = np.asarray(entry["amp"], dtype=np.float64)
        phase = np.asarray(entry.get("phase", np.zeros_like(amp)), dtype=np.float64)
        if amp.size != n:
            raise ValueError(f"basis amp length {amp.size} != core dim {n}")
        modes.append((amp * np.exp(1j * phase)).astype(np.complex128))
    return modes


def born_prior_weights(system_type: str, rho_target: np.ndarray) -> np.ndarray:
    """Per-operation Born weights b_o = Tr(E_o rho_target) using the POVM modes."""
    if system_type not in SYSTEM_OPERATION_COUNTS:
        raise ValueError(f"system_type must be one of {list(SYSTEM_OPERATION_COUNTS)}")
    n = 3 if system_type == "3qubit" else 4
    rho = np.asarray(rho_target, dtype=np.complex128)
    if rho.shape != (n, n):
        raise ValueError(f"rho_target must be {(n, n)}, got {rho.shape}")
    weights = []
    for mode in _load_mode_vectors(system_type):
        norm = float(np.vdot(mode, mode).real) or 1.0
        weights.append(float(np.real(np.vdot(mode, rho @ mode)) / norm))
    return np.asarray(weights, dtype=np.float64)


@dataclass(frozen=True)
class V9OperationTargetLawResultV1:
    """Predicted per-operation log-rate (target-conditioned law)."""

    system_type: str
    log_mu: np.ndarray
    born_prior: np.ndarray
    global_scale: float
    target_operation_residual: np.ndarray
    nuisance_residual: np.ndarray
    target_features: Dict[str, np.ndarray]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "system_type": self.system_type,
            "global_scale": float(self.global_scale),
            "born_prior_min": float(np.min(self.born_prior)),
            "born_prior_max": float(np.max(self.born_prior)),
            "residual_norm": float(np.linalg.norm(self.target_operation_residual)),
            "target_features": {k: v.tolist() for k, v in self.target_features.items()},
        }


def operation_target_law(
    system_type: str,
    c_coeffs: np.ndarray,
    *,
    rho_target: Optional[np.ndarray] = None,
    global_scale: float = 0.0,
    residual_bound: float = 0.5,
    nuisance_residual: Optional[np.ndarray] = None,
) -> V9OperationTargetLawResultV1:
    """Predict ``log mu_o`` from the target Born prior plus bounded residuals.

    No target name string / path / arbitrary ID encoding is used: only the gauge
    invariant W density coordinates and the Born prior enter.
    """
    n = 3 if system_type == "3qubit" else 4
    rho = target_core_rho(c_coeffs, n) if rho_target is None else np.asarray(rho_target)
    born = born_prior_weights(system_type, rho)
    features = target_density_features(c_coeffs, n)
    target_residual = np.zeros_like(born)  # learned bounded residual enters at training
    if nuisance_residual is None:
        nuisance_residual = np.zeros_like(born)
    log_mu = (
        global_scale
        + np.log(born + _EPS)
        + target_residual
        + np.asarray(nuisance_residual, dtype=np.float64)
    )
    return V9OperationTargetLawResultV1(
        system_type=system_type,
        log_mu=log_mu,
        born_prior=born,
        global_scale=global_scale,
        target_operation_residual=target_residual,
        nuisance_residual=np.asarray(nuisance_residual, dtype=np.float64),
        target_features={k: np.asarray(v) for k, v in features.items()},
    )


def target_shuffle_control(
    law_fn,
    c_coeffs: np.ndarray,
    shuffled_c_coeffs: np.ndarray,
    system_type: str,
) -> Dict[str, float]:
    """Target shuffle must degrade the law (held-out pattern NLL degradation).

    ``law_fn`` maps coefficients to per-operation log-mu. Returns the mean absolute
    pattern shift between true and shuffled targets.
    """
    true = law_fn(system_type, c_coeffs).log_mu
    shuffled = law_fn(system_type, shuffled_c_coeffs).log_mu
    true_q = _softmax(true)
    shuffled_q = _softmax(shuffled)
    hellinger = 1.0 - float(np.sum(np.sqrt(true_q * shuffled_q)))
    return {"pattern_hellinger_shift": hellinger, "log_mu_mae": float(np.mean(np.abs(true - shuffled)))}


def operation_mapping_shuffle_control(
    system_type: str,
    c_coeffs: np.ndarray,
    rng: np.random.Generator,
) -> Dict[str, float]:
    """Shuffle the operation-to-projector mapping; the law must degrade."""
    n = 3 if system_type == "3qubit" else 4
    rho = target_core_rho(c_coeffs, n)
    true_born = born_prior_weights(system_type, rho)
    modes = _load_mode_vectors(system_type)
    rng.shuffle(modes)
    shuffled_born = np.asarray(
        [float(np.real(np.vdot(m, rho @ m)) / (np.vdot(m, m).real or 1.0)) for m in modes]
    )
    return {
        "born_prior_mae": float(np.mean(np.abs(true_born - shuffled_born))),
        "mapping_shuffled": True,
    }


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x)
    e = np.exp(z)
    return e / e.sum()
