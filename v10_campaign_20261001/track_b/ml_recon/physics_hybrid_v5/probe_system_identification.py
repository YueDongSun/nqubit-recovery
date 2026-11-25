"""Joint multi-direction signed-probe identification for V5 research.

Six signed pairs belonging to one anchor are fitted together.  Point-local
signal/idler/coin channels are the only default targets; global multi-click
totals are deliberately absent from this API.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None
    F = None


JOINT_PROBE_SCHEMA = "joint_signed_probe_identification"
JOINT_PROBE_VERSION = 1
POINT_COUNT_CHANNELS: Tuple[str, ...] = ("signal", "idler", "coin")


def _direction_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    digest = hashlib.sha256()
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def pair_safe_anchor_split_identity(anchor_id: str, direction_hashes: Sequence[str]) -> str:
    """One split identity shared by every sign/direction from an anchor."""
    payload = json.dumps(
        {"anchor_id": str(anchor_id), "direction_hashes": sorted(str(value) for value in direction_hashes)},
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return f"anchor:{anchor_id}:{hashlib.sha256(payload).hexdigest()[:16]}"


@dataclass(frozen=True)
class JointProbeFitResult:
    joint_jacobian: np.ndarray
    jacobian_covariance: np.ndarray
    jacobian_inverse_variance: np.ndarray
    jacobian_supervision_mask: np.ndarray
    rank: np.ndarray
    singular_values: np.ndarray
    condition_number: np.ndarray
    complete_pair_count: np.ndarray
    direction_hashes: Tuple[str, ...]
    directional_prediction: np.ndarray
    directional_residuals: np.ndarray
    held_out_direction_error: np.ndarray
    hessian: Optional[np.ndarray] = None
    schema: str = JOINT_PROBE_SCHEMA
    version: int = JOINT_PROBE_VERSION

    def design_audit(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "version": int(self.version),
            "rank": self.rank.tolist(),
            "singular_values": self.singular_values.tolist(),
            "condition_number": self.condition_number.tolist(),
            "complete_pair_count": self.complete_pair_count.tolist(),
            "direction_hashes": list(self.direction_hashes),
            "supervised_operation_fraction": float(np.mean(np.any(self.jacobian_supervision_mask, axis=-1))),
            "held_out_direction_rmse": float(np.sqrt(np.nanmean(np.square(self.held_out_direction_error)))),
        }


def _broadcast_directions(directions: Any, operation_count: int) -> np.ndarray:
    design = np.asarray(directions, dtype=np.float64)
    if design.ndim == 2 and design.shape[1] == 3:
        return np.broadcast_to(design[:, None, :], (design.shape[0], int(operation_count), 3)).copy()
    if design.ndim == 3 and design.shape[1:] == (int(operation_count), 3):
        return design
    raise ValueError("directions must have shape [K,3] or [K,O,3]")


def _weighted_ridge(design: np.ndarray, response: np.ndarray, variance: np.ndarray, ridge: float) -> Tuple[np.ndarray, np.ndarray]:
    weight = 1.0 / np.maximum(variance, 1e-12)
    gram = design.T @ (weight[:, None] * design) + float(ridge) * np.eye(3)
    covariance = np.linalg.pinv(gram, hermitian=True)
    jacobian = covariance @ design.T @ (weight * response)
    return jacobian, 0.5 * (covariance + covariance.T)


def fit_joint_probe_jacobian(
    directions: Any,
    directional_response: Any,
    response_variance: Any,
    *,
    ridge: float = 1e-4,
    rank_threshold: int = 3,
    condition_threshold: float = 1e3,
) -> JointProbeFitResult:
    """Fit ``J=(D^T W D+lambda I)^-1 D^T W y`` per operation/channel."""
    response = np.asarray(directional_response, dtype=np.float64)
    variance = np.asarray(response_variance, dtype=np.float64)
    if response.ndim != 3:
        raise ValueError("directional_response must have shape [K,O,C]")
    if variance.shape != response.shape:
        raise ValueError("response_variance must match directional_response")
    direction = _broadcast_directions(directions, response.shape[1])
    if direction.shape[0] != response.shape[0]:
        raise ValueError("direction and response counts disagree")
    direction_count, operation_count, channel_count = response.shape
    jacobian = np.zeros((operation_count, channel_count, 3), dtype=np.float64)
    covariance = np.zeros((operation_count, channel_count, 3, 3), dtype=np.float64)
    supervision = np.zeros((operation_count, channel_count), dtype=bool)
    rank = np.zeros(operation_count, dtype=np.int64)
    singular = np.zeros((operation_count, 3), dtype=np.float64)
    condition = np.full(operation_count, np.inf, dtype=np.float64)
    complete = np.zeros(operation_count, dtype=np.int64)
    prediction = np.full(response.shape, np.nan, dtype=np.float64)
    residual = np.full(response.shape, np.nan, dtype=np.float64)
    held_out = np.full(response.shape, np.nan, dtype=np.float64)

    for operation in range(operation_count):
        finite_design = np.all(np.isfinite(direction[:, operation]), axis=-1)
        complete[operation] = int(np.sum(finite_design))
        design = direction[finite_design, operation]
        if design.size:
            singular_values = np.linalg.svd(design, compute_uv=False)
            singular[operation, : singular_values.size] = singular_values[:3]
            tolerance = np.finfo(float).eps * max(design.shape) * max(float(singular_values[0]), 1.0)
            rank[operation] = int(np.sum(singular_values > tolerance))
            if singular_values.size >= 3 and singular_values[-1] > tolerance:
                condition[operation] = float(singular_values[0] / singular_values[-1])
        design_quality = rank[operation] >= int(rank_threshold) and condition[operation] <= float(condition_threshold)
        for channel in range(channel_count):
            valid = finite_design & np.isfinite(response[:, operation, channel]) & np.isfinite(variance[:, operation, channel]) & (variance[:, operation, channel] > 0.0)
            if np.sum(valid) < 3:
                continue
            fitted, fitted_covariance = _weighted_ridge(
                direction[valid, operation], response[valid, operation, channel], variance[valid, operation, channel], ridge,
            )
            jacobian[operation, channel] = fitted
            covariance[operation, channel] = fitted_covariance
            supervision[operation, channel] = bool(design_quality and np.linalg.matrix_rank(direction[valid, operation]) >= rank_threshold)
            prediction[valid, operation, channel] = direction[valid, operation] @ fitted
            residual[valid, operation, channel] = prediction[valid, operation, channel] - response[valid, operation, channel]
            valid_indices = np.flatnonzero(valid)
            for held_index in valid_indices:
                train = valid.copy()
                train[held_index] = False
                if np.sum(train) < 3 or np.linalg.matrix_rank(direction[train, operation]) < rank_threshold:
                    continue
                held_fit, _ = _weighted_ridge(
                    direction[train, operation], response[train, operation, channel], variance[train, operation, channel], ridge,
                )
                held_prediction = float(direction[held_index, operation] @ held_fit)
                held_out[held_index, operation, channel] = held_prediction - response[held_index, operation, channel]

    diagonal = np.diagonal(covariance, axis1=-2, axis2=-1)
    inverse_variance = np.where(diagonal > 0.0, 1.0 / diagonal, 0.0)
    return JointProbeFitResult(
        joint_jacobian=jacobian,
        jacobian_covariance=covariance,
        jacobian_inverse_variance=inverse_variance,
        jacobian_supervision_mask=supervision,
        rank=rank,
        singular_values=singular,
        condition_number=condition,
        complete_pair_count=complete,
        direction_hashes=tuple(_direction_hash(direction[index]) for index in range(direction_count)),
        directional_prediction=prediction,
        directional_residuals=residual,
        held_out_direction_error=held_out,
    )


def _broadcast_exposure(exposure: Any, target_shape: Tuple[int, int, int]) -> np.ndarray:
    value = np.asarray(exposure, dtype=np.float64)
    direction_count, operation_count, channel_count = target_shape
    if value.shape == (direction_count,):
        value = value[:, None, None]
    elif value.shape == (direction_count, operation_count):
        value = value[:, :, None]
    return np.broadcast_to(value, target_shape)


def fit_raw_signed_probe_counts(
    directions: Any,
    amplitudes: Any,
    plus_counts: Any,
    minus_counts: Any,
    plus_exposure: Any,
    minus_exposure: Any,
    *,
    nb2_dispersion: Any = 0.0,
    ridge: float = 1e-4,
    rank_threshold: int = 3,
    condition_threshold: float = 1e3,
) -> JointProbeFitResult:
    """Fit raw signed rates; additive background cancels without log subtraction."""
    plus = np.asarray(plus_counts, dtype=np.float64)
    minus = np.asarray(minus_counts, dtype=np.float64)
    if plus.shape != minus.shape or plus.ndim != 3:
        raise ValueError("plus/minus counts must share shape [K,O,C]")
    h = np.asarray(amplitudes, dtype=np.float64).reshape(-1)
    if h.shape != (plus.shape[0],) or np.any(h <= 0.0):
        raise ValueError("amplitudes must be positive with shape [K]")
    exposure_plus = _broadcast_exposure(plus_exposure, plus.shape)
    exposure_minus = _broadcast_exposure(minus_exposure, minus.shape)
    if np.any(exposure_plus <= 0.0) or np.any(exposure_minus <= 0.0):
        raise ValueError("signed probe exposure must be positive")
    plus_rate = plus / exposure_plus
    minus_rate = minus / exposure_minus
    scale = (2.0 * h)[:, None, None]
    response = (plus_rate - minus_rate) / scale
    alpha = np.broadcast_to(np.asarray(nb2_dispersion, dtype=np.float64), plus.shape)
    variance_plus_count = np.maximum(plus, 1.0) + np.maximum(alpha, 0.0) * np.square(np.maximum(plus, 1.0))
    variance_minus_count = np.maximum(minus, 1.0) + np.maximum(alpha, 0.0) * np.square(np.maximum(minus, 1.0))
    variance = (variance_plus_count / np.square(exposure_plus) + variance_minus_count / np.square(exposure_minus)) / np.square(scale)
    return fit_joint_probe_jacobian(
        directions, response, variance, ridge=ridge,
        rank_threshold=rank_threshold, condition_threshold=condition_threshold,
    )


def directional_supervision_loss(
    predicted_jacobian: Any,
    directions: Any,
    directional_target: Any,
    directional_variance: Any,
    *,
    mask: Optional[Any] = None,
) -> Any:
    """Variance-weighted ``(d^T J_pred-y)^2`` without label compression."""
    if torch is not None and isinstance(predicted_jacobian, torch.Tensor):
        design = torch.as_tensor(directions, dtype=predicted_jacobian.dtype, device=predicted_jacobian.device)
        target = torch.as_tensor(directional_target, dtype=predicted_jacobian.dtype, device=predicted_jacobian.device)
        variance = torch.as_tensor(directional_variance, dtype=predicted_jacobian.dtype, device=predicted_jacobian.device).clamp_min(1e-12)
        if design.ndim == 2:
            prediction = torch.einsum("kd,bocd->bkoc", design, predicted_jacobian)
        else:
            prediction = torch.einsum("kod,bocd->bkoc", design, predicted_jacobian)
        weight = torch.ones_like(target) if mask is None else torch.as_tensor(mask, dtype=target.dtype, device=target.device)
        return (((prediction - target).square() / variance) * weight).sum() / weight.sum().clamp_min(1.0)
    jacobian = np.asarray(predicted_jacobian, dtype=float)
    design = np.asarray(directions, dtype=float)
    target = np.asarray(directional_target, dtype=float)
    variance = np.maximum(np.asarray(directional_variance, dtype=float), 1e-12)
    prediction = np.einsum("kd,...ocd->...koc", design, jacobian) if design.ndim == 2 else np.einsum("kod,...ocd->...koc", design, jacobian)
    weight = np.ones_like(target) if mask is None else np.asarray(mask, dtype=float)
    return float(np.sum(np.square(prediction - target) / variance * weight) / max(float(np.sum(weight)), 1.0))


def fit_nb2_glm_single_channel(
    directions: Any,
    amplitudes: Any,
    signed_counts: Any,
    exposure: Any,
    *,
    background: float = 0.0,
    dispersion: float = 0.0,
    ridge: float = 1e-3,
    hessian_enabled: bool = False,
    steps: int = 200,
) -> Dict[str, np.ndarray | float]:
    """Optional raw-count NB2 GLM for one operation/channel.

    ``mu = exposure * [background + softplus(beta + s h d^T J +
    0.5 h^2 d^T H d)]``.  H is symmetric and retained only when the caller's
    held-out likelihood evaluation approves it.
    """
    if torch is None:
        raise RuntimeError("fit_nb2_glm_single_channel requires torch")
    design = torch.as_tensor(directions, dtype=torch.float64)
    h = torch.as_tensor(amplitudes, dtype=torch.float64).reshape(-1)
    counts = torch.as_tensor(signed_counts, dtype=torch.float64)
    gates = torch.as_tensor(exposure, dtype=torch.float64)
    if design.ndim != 2 or design.shape[1] != 3 or counts.shape != (design.shape[0], 2):
        raise ValueError("expected directions [K,3] and signed_counts [K,2]")
    gates = torch.broadcast_to(gates, counts.shape)
    sign = torch.as_tensor([-1.0, 1.0], dtype=torch.float64).reshape(1, 2)
    beta = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
    jacobian = torch.nn.Parameter(torch.zeros(3, dtype=torch.float64))
    hessian_raw = torch.nn.Parameter(torch.zeros(3, 3, dtype=torch.float64), requires_grad=bool(hessian_enabled))
    parameters = [beta, jacobian] + ([hessian_raw] if hessian_enabled else [])
    optimizer = torch.optim.Adam(parameters, lr=0.03)
    alpha = max(float(dispersion), 0.0)
    for _ in range(max(1, int(steps))):
        linear = (design @ jacobian)[:, None] * h[:, None] * sign
        if hessian_enabled:
            symmetric_hessian = 0.5 * (hessian_raw + hessian_raw.T)
            quadratic = 0.5 * h.square() * torch.einsum("ki,ij,kj->k", design, symmetric_hessian, design)
            linear = linear + quadratic[:, None]
        mean = gates * (max(float(background), 0.0) + F.softplus(beta + linear))
        if alpha > 0.0:
            r = 1.0 / alpha
            log_prob = (
                torch.lgamma(counts + r) - torch.lgamma(torch.tensor(r, dtype=torch.float64)) - torch.lgamma(counts + 1.0)
                + r * (np.log(r) - torch.log(r + mean))
                + counts * (torch.log(mean.clamp_min(1e-12)) - torch.log(r + mean))
            )
            nll = -log_prob.mean()
        else:
            nll = (mean - counts * torch.log(mean.clamp_min(1e-12))).mean()
        loss = nll + float(ridge) * jacobian.square().sum()
        if hessian_enabled:
            loss = loss + float(ridge) * hessian_raw.square().sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    symmetric = 0.5 * (hessian_raw + hessian_raw.T) if hessian_enabled else torch.zeros(3, 3, dtype=torch.float64)
    return {
        "beta": float(beta.detach()),
        "jacobian": jacobian.detach().cpu().numpy(),
        "hessian_candidate": symmetric.detach().cpu().numpy(),
        "hessian_requires_held_out_acceptance": bool(hessian_enabled),
        "training_nll": float(nll.detach()),
    }


__all__ = [
    "JointProbeFitResult", "fit_joint_probe_jacobian", "fit_raw_signed_probe_counts",
    "directional_supervision_loss", "fit_nb2_glm_single_channel",
    "pair_safe_anchor_split_identity", "POINT_COUNT_CHANNELS",
]
