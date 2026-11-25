"""Exact four-detector click likelihood for V5 research-only analysis.

The HDL's mutually exclusive categories are preserved: weight one contributes
to the idler/coin histogram path, weight two to one of six pair counters and
``multi2``, weight three to ``multi3``, and weight four to ``multi4``.  Global
or cycle totals are never expanded into operation-local labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


DETECTOR_MODEL_SCHEMA = "four_detector_exact_click"
DETECTOR_MODEL_VERSION = 1
MULTI_CLICK_SCHEMA = "multi_click_observation"
MULTI_CLICK_VERSION = 2
PAIR_LABELS: Tuple[str, ...] = ("12", "13", "14", "23", "24", "34")
VALID_SCOPES = frozenset(("point", "cycle", "global"))
PATTERNS = np.asarray([[float((index >> bit) & 1) for bit in range(4)] for index in range(16)], dtype=np.float64)
PATTERN_WEIGHTS = PATTERNS.sum(axis=1).astype(np.int64)
PAIR_PATTERN_INDICES = np.asarray([
    int((1 << (int(label[0]) - 1)) | (1 << (int(label[1]) - 1))) for label in PAIR_LABELS
], dtype=np.int64)


def _optional_array(value: Any, *, dtype: Any = np.float64) -> Optional[np.ndarray]:
    if value is None:
        return None
    return np.asarray(value, dtype=dtype)


@dataclass(frozen=True)
class MultiClickObservationV2:
    """Backward-compatible multi-click observation with explicit scope."""

    scope: str
    multi2_bins: Optional[np.ndarray] = None
    multi3_bins: Optional[np.ndarray] = None
    multi4_bins: Optional[np.ndarray] = None
    multi2_cycle: Optional[np.ndarray] = None
    multi3_cycle: Optional[np.ndarray] = None
    multi4_cycle: Optional[np.ndarray] = None
    multi2_pairs: Optional[np.ndarray] = None
    multi2_count: Optional[float] = None
    multi3_count: Optional[float] = None
    multi4_count: Optional[float] = None
    schema: str = MULTI_CLICK_SCHEMA
    version: int = MULTI_CLICK_VERSION

    def __post_init__(self) -> None:
        if self.scope not in VALID_SCOPES:
            raise ValueError(f"scope must be one of {sorted(VALID_SCOPES)}")
        for name in ("multi2_bins", "multi3_bins", "multi4_bins", "multi2_cycle", "multi3_cycle", "multi4_cycle", "multi2_pairs"):
            value = getattr(self, name)
            if value is not None:
                array = np.asarray(value, dtype=np.float64)
                if array.ndim != 1 or np.any(array < 0.0) or not np.all(np.isfinite(array)):
                    raise ValueError(f"{name} must be a finite non-negative vector")
                if name == "multi2_pairs" and array.shape != (6,):
                    raise ValueError("multi2_pairs must have shape [6]")
                object.__setattr__(self, name, array)
        for name in ("multi2_count", "multi3_count", "multi4_count"):
            value = getattr(self, name)
            if value is not None and (not np.isfinite(float(value)) or float(value) < 0.0):
                raise ValueError(f"{name} must be finite and non-negative")
        if self.scope == "point" and all(getattr(self, name) is None for name in ("multi2_bins", "multi3_bins", "multi4_bins")):
            raise ValueError("point scope requires at least one point-bin array")
        if self.scope == "cycle" and all(getattr(self, name) is None for name in ("multi2_cycle", "multi3_cycle", "multi4_cycle")):
            raise ValueError("cycle scope requires at least one cycle array")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "MultiClickObservationV2":
        point_present = any(payload.get(name) is not None for name in ("multi2_bins", "multi3_bins", "multi4_bins"))
        cycle_present = any(payload.get(name) is not None for name in ("multi2_cycle", "multi3_cycle", "multi4_cycle"))
        scope = str(payload.get("scope") or ("point" if point_present else "cycle" if cycle_present else "global"))
        pair_value = payload.get("multi2_pairs")
        if isinstance(pair_value, Mapping):
            pair_value = [pair_value.get(label, 0.0) for label in PAIR_LABELS]
        return cls(
            scope=scope,
            multi2_bins=_optional_array(payload.get("multi2_bins")),
            multi3_bins=_optional_array(payload.get("multi3_bins")),
            multi4_bins=_optional_array(payload.get("multi4_bins")),
            multi2_cycle=_optional_array(payload.get("multi2_cycle")),
            multi3_cycle=_optional_array(payload.get("multi3_cycle")),
            multi4_cycle=_optional_array(payload.get("multi4_cycle")),
            multi2_pairs=_optional_array(pair_value),
            multi2_count=None if payload.get("multi2_count") is None else float(payload["multi2_count"]),
            multi3_count=None if payload.get("multi3_count") is None else float(payload["multi3_count"]),
            multi4_count=None if payload.get("multi4_count") is None else float(payload["multi4_count"]),
        )

    def field_scopes(self) -> Dict[str, str]:
        result: Dict[str, str] = {}
        for name in ("multi2_bins", "multi3_bins", "multi4_bins"):
            if getattr(self, name) is not None:
                result[name] = "point"
        for name in ("multi2_cycle", "multi3_cycle", "multi4_cycle"):
            if getattr(self, name) is not None:
                result[name] = "cycle"
        for name in ("multi2_pairs", "multi2_count", "multi3_count", "multi4_count"):
            if getattr(self, name) is not None:
                result[name] = "global"
        return result

    def manifest_metadata(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "version": int(self.version),
            "scope": self.scope,
            "field_scopes": self.field_scopes(),
            "available_fields": sorted(self.field_scopes()),
        }


@dataclass(frozen=True)
class DetectorCalibrationContractV1:
    """Campaign/time-block detector nuisance provenance."""

    detector_efficiency: Sequence[float]
    dark_probability: Optional[Sequence[float]] = None
    off_window_counts: Optional[Sequence[float]] = None
    off_window_gate_exposure: Optional[float] = None
    external_calibration_id: Optional[str] = None
    campaign_id: Optional[str] = None
    time_block_id: Optional[str] = None
    dark_prior_std_logit: float = 0.25
    time_block_residual_std_logit: float = 0.05

    def __post_init__(self) -> None:
        efficiency = np.asarray(self.detector_efficiency, dtype=float)
        if efficiency.shape != (4,) or np.any(efficiency <= 0.0) or np.any(efficiency > 1.0):
            raise ValueError("detector_efficiency must have shape [4] in (0,1]")
        object.__setattr__(self, "detector_efficiency", efficiency)
        if self.dark_probability is not None:
            dark = np.asarray(self.dark_probability, dtype=float)
            if dark.shape != (4,) or np.any(dark < 0.0) or np.any(dark >= 1.0):
                raise ValueError("dark_probability must have shape [4] in [0,1)")
            object.__setattr__(self, "dark_probability", dark)

    @property
    def dark_identifiable(self) -> bool:
        off_window = self.off_window_counts is not None and self.off_window_gate_exposure is not None and float(self.off_window_gate_exposure) > 0.0
        return bool(off_window)

    @property
    def dark_not_identifiable(self) -> bool:
        return not self.dark_identifiable

    @property
    def subtraction_status(self) -> Dict[str, Any]:
        return {
            "dark_not_identifiable": self.dark_not_identifiable,
            "external_calibration_used": self.external_calibration_id is not None,
            "true_multi_subtraction_available": False,
            "promotion_claim_allowed": False,
            "posterior_uncertainty_required": True,
        }


def dark_or_kernel(q: Any) -> Any:
    """Return differentiable ``K[z,x]=P(z|x,q)`` for ``z=x OR b``."""
    if torch is not None and isinstance(q, torch.Tensor):
        probability = q.reshape(-1)
        if probability.shape != (4,):
            raise ValueError("q must have shape [4]")
        patterns = torch.as_tensor(PATTERNS, dtype=probability.dtype, device=probability.device)
        x = patterns.unsqueeze(0)  # [z=1,x=16,d=4] after expansion below
        z = patterns.unsqueeze(1)
        valid = torch.all(z >= x, dim=-1)
        factor = torch.where(x > 0.5, torch.ones_like(z), torch.where(z > 0.5, probability, 1.0 - probability))
        return torch.where(valid, factor.prod(dim=-1), torch.zeros_like(valid, dtype=probability.dtype))
    probability = np.asarray(q, dtype=float).reshape(-1)
    if probability.shape != (4,):
        raise ValueError("q must have shape [4]")
    x = PATTERNS[None, :, :]
    z = PATTERNS[:, None, :]
    valid = np.all(z >= x, axis=-1)
    factor = np.where(x > 0.5, 1.0, np.where(z > 0.5, probability, 1.0 - probability))
    return np.where(valid, np.prod(factor, axis=-1), 0.0)


def detector_loss_kernel(eta: Any) -> Any:
    """Return ``K[x,y]=P(photon-click pattern x | true pattern y)``."""
    if torch is not None and isinstance(eta, torch.Tensor):
        efficiency = eta.reshape(-1)
        if efficiency.shape != (4,):
            raise ValueError("eta must have shape [4]")
        patterns = torch.as_tensor(PATTERNS, dtype=efficiency.dtype, device=efficiency.device)
        x = patterns.unsqueeze(1)
        y = patterns.unsqueeze(0)
        valid = torch.all(x <= y, dim=-1)
        factor = torch.where(y < 0.5, torch.ones_like(x), torch.where(x > 0.5, efficiency, 1.0 - efficiency))
        return torch.where(valid, factor.prod(dim=-1), torch.zeros_like(valid, dtype=efficiency.dtype))
    efficiency = np.asarray(eta, dtype=float).reshape(-1)
    if efficiency.shape != (4,):
        raise ValueError("eta must have shape [4]")
    x = PATTERNS[:, None, :]
    y = PATTERNS[None, :, :]
    valid = np.all(x <= y, axis=-1)
    factor = np.where(y < 0.5, 1.0, np.where(x > 0.5, efficiency, 1.0 - efficiency))
    return np.where(valid, np.prod(factor, axis=-1), 0.0)


def observed_pattern_probabilities(p_true: Any, *, q: Any, eta: Any) -> Any:
    """Apply efficiency and dark kernels without changing pattern scope."""
    dark = dark_or_kernel(q)
    loss = detector_loss_kernel(eta)
    if torch is not None and isinstance(p_true, torch.Tensor):
        probability = torch.clamp(p_true, min=0.0)
        probability = probability / probability.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(probability.dtype).eps)
        return torch.einsum("zx,xy,...y->...z", dark, loss, probability)
    probability = np.clip(np.asarray(p_true, dtype=float), 0.0, None)
    probability /= np.maximum(probability.sum(axis=-1, keepdims=True), np.finfo(float).eps)
    return np.einsum("zx,xy,...y->...z", dark, loss, probability)


def aggregate_exclusive_categories(pattern_probability: Any) -> Dict[str, Any]:
    """Aggregate exact observed patterns according to deployed HDL semantics."""
    if torch is not None and isinstance(pattern_probability, torch.Tensor):
        pair_indices = torch.as_tensor(PAIR_PATTERN_INDICES, dtype=torch.long, device=pattern_probability.device)
        weights = torch.as_tensor(PATTERN_WEIGHTS, dtype=torch.long, device=pattern_probability.device)
        return {
            "one_click": pattern_probability[..., weights == 1].sum(dim=-1),
            "multi2_pairs": pattern_probability.index_select(-1, pair_indices),
            "multi2": pattern_probability[..., weights == 2].sum(dim=-1),
            "multi3": pattern_probability[..., weights == 3].sum(dim=-1),
            "multi4": pattern_probability[..., 15],
        }
    probability = np.asarray(pattern_probability, dtype=float)
    return {
        "one_click": probability[..., PATTERN_WEIGHTS == 1].sum(axis=-1),
        "multi2_pairs": probability[..., PAIR_PATTERN_INDICES],
        "multi2": probability[..., PATTERN_WEIGHTS == 2].sum(axis=-1),
        "multi3": probability[..., PATTERN_WEIGHTS == 3].sum(axis=-1),
        "multi4": probability[..., 15],
    }


def global_click_means(pattern_probability: Any, gate_exposure: Any) -> Dict[str, Any]:
    """Sum operation probabilities using explicit gate counts/exposure."""
    categories = aggregate_exclusive_categories(pattern_probability)
    if torch is not None and isinstance(pattern_probability, torch.Tensor):
        exposure = torch.as_tensor(gate_exposure, dtype=pattern_probability.dtype, device=pattern_probability.device)
        if exposure.shape != pattern_probability.shape[:-1]:
            raise ValueError("gate_exposure must match operation/cycle batch axes")
        return {
            "multi2_pairs": (categories["multi2_pairs"] * exposure.unsqueeze(-1)).sum(dim=-2),
            "multi2": (categories["multi2"] * exposure).sum(dim=-1),
            "multi3": (categories["multi3"] * exposure).sum(dim=-1),
            "multi4": (categories["multi4"] * exposure).sum(dim=-1),
        }
    probability = np.asarray(pattern_probability, dtype=float)
    exposure = np.asarray(gate_exposure, dtype=float)
    if exposure.shape != probability.shape[:-1]:
        raise ValueError("gate_exposure must match operation/cycle batch axes")
    return {
        "multi2_pairs": np.sum(categories["multi2_pairs"] * exposure[..., None], axis=-2),
        "multi2": np.sum(categories["multi2"] * exposure, axis=-1),
        "multi3": np.sum(categories["multi3"] * exposure, axis=-1),
        "multi4": np.sum(categories["multi4"] * exposure, axis=-1),
    }


def gate_exposure_from_acquisition(acquisition: Mapping[str, Any], operation_count: int) -> np.ndarray:
    """Resolve gate exposure from read-count provenance, never duration alone."""
    explicit = acquisition.get("gate_exposure")
    if explicit is not None:
        exposure = np.asarray(explicit, dtype=float)
    else:
        read_count = acquisition.get("read_count")
        if read_count is None:
            raise ValueError("global click likelihood requires read_count or gate_exposure; duration alone is insufficient")
        exposure = np.asarray(read_count, dtype=float)
    if exposure.ndim == 0:
        exposure = np.full((int(operation_count),), float(exposure), dtype=float)
    if exposure.shape != (int(operation_count),) or np.any(exposure < 0.0):
        raise ValueError(f"gate exposure must have shape [{operation_count}] and be non-negative")
    return exposure


def count_negative_log_likelihood(observed: Any, mean: Any, *, dispersion: Optional[Any] = None) -> Any:
    """Poisson or NB2 NLL, omitting only observation-constant terms."""
    if torch is not None and isinstance(mean, torch.Tensor):
        target = torch.as_tensor(observed, dtype=mean.dtype, device=mean.device)
        mu = mean.clamp_min(torch.finfo(mean.dtype).eps)
        if dispersion is None:
            return (mu - target * torch.log(mu)).sum()
        alpha = torch.as_tensor(dispersion, dtype=mean.dtype, device=mean.device).clamp_min(1e-8)
        r = 1.0 / alpha
        return -(
            torch.lgamma(target + r) - torch.lgamma(r) - torch.lgamma(target + 1.0)
            + r * (torch.log(r) - torch.log(r + mu))
            + target * (torch.log(mu) - torch.log(r + mu))
        ).sum()
    target = np.asarray(observed, dtype=float)
    mu = np.maximum(np.asarray(mean, dtype=float), np.finfo(float).eps)
    if dispersion is None:
        return float(np.sum(mu - target * np.log(mu)))
    alpha = np.maximum(np.asarray(dispersion, dtype=float), 1e-8)
    variance = mu + alpha * np.square(mu)
    # Stable quasi-NB2 objective used only by NumPy audits; torch path above is exact.
    return float(0.5 * np.sum(np.square(target - mu) / variance + np.log(variance)))


if nn is not None:

    class FourDetectorClickModel(nn.Module):
        """Campaign-level dark logits plus strongly shrunk time-block residuals."""

        def __init__(
            self,
            *,
            dark_prior: Sequence[float] = (1e-5, 1e-5, 1e-5, 1e-5),
            detector_efficiency: Sequence[float] = (1.0, 1.0, 1.0, 1.0),
            time_blocks: int = 1,
            time_block_residual_scale: float = 0.05,
            dark_prior_std_logit: float = 0.25,
            efficiency_prior_std_logit: float = 0.25,
        ) -> None:
            super().__init__()
            dark = torch.as_tensor(dark_prior, dtype=torch.float32).clamp(1e-8, 1.0 - 1e-8)
            efficiency = torch.as_tensor(detector_efficiency, dtype=torch.float32).clamp(1e-6, 1.0)
            if dark.shape != (4,) or efficiency.shape != (4,):
                raise ValueError("detector priors must have shape [4]")
            dark_prior_logit = torch.logit(dark)
            efficiency_prior_logit = torch.logit(efficiency.clamp(max=1.0 - 1e-7))
            self.register_buffer("dark_prior_logit", dark_prior_logit)
            self.register_buffer("efficiency_prior_logit", efficiency_prior_logit)
            self.dark_logit = nn.Parameter(dark_prior_logit.clone())
            self.efficiency_logit = nn.Parameter(efficiency_prior_logit.clone())
            self.time_block_residual = nn.Parameter(torch.zeros(max(1, int(time_blocks)), 4))
            self.time_block_residual_scale = float(time_block_residual_scale)
            self.dark_prior_std_logit = max(float(dark_prior_std_logit), 1e-6)
            self.efficiency_prior_std_logit = max(float(efficiency_prior_std_logit), 1e-6)

        def probabilities(self, time_block_index: Optional["torch.Tensor"] = None) -> Tuple["torch.Tensor", "torch.Tensor"]:
            logit = self.dark_logit
            if time_block_index is not None:
                logit = logit + self.time_block_residual_scale * torch.tanh(self.time_block_residual[time_block_index.long()])
            return torch.sigmoid(logit), torch.sigmoid(self.efficiency_logit)

        def forward(self, p_true: "torch.Tensor", time_block_index: Optional["torch.Tensor"] = None) -> "torch.Tensor":
            q, eta = self.probabilities(time_block_index)
            if q.ndim == 1:
                return observed_pattern_probabilities(p_true, q=q, eta=eta)
            outputs = [observed_pattern_probabilities(p_true[index], q=q[index], eta=eta) for index in range(p_true.shape[0])]
            return torch.stack(outputs, dim=0)

        def shrinkage_loss(self) -> "torch.Tensor":
            diagnostics = self.prior_diagnostics()
            return (
                diagnostics["detector_dark_prior_deviation"]
                + diagnostics["detector_efficiency_prior_deviation"]
                + diagnostics["detector_time_block_residual_norm"].square()
            )

        def prior_diagnostics(self) -> Dict[str, "torch.Tensor"]:
            """Return each prior term separately for loss and audit reporting."""
            dark = ((self.dark_logit - self.dark_prior_logit) / self.dark_prior_std_logit).square().mean()
            efficiency = (
                (self.efficiency_logit - self.efficiency_prior_logit)
                / self.efficiency_prior_std_logit
            ).square().mean()
            return {
                "detector_dark_prior_deviation": dark,
                "detector_efficiency_prior_deviation": efficiency,
                "detector_time_block_residual_norm": torch.sqrt(
                    self.time_block_residual.square().mean()
                ),
            }


else:  # pragma: no cover
    class FourDetectorClickModel:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("FourDetectorClickModel requires torch")


__all__ = [
    "MultiClickObservationV2", "DetectorCalibrationContractV1", "FourDetectorClickModel",
    "dark_or_kernel", "detector_loss_kernel", "observed_pattern_probabilities",
    "aggregate_exclusive_categories", "global_click_means", "gate_exposure_from_acquisition",
    "count_negative_log_likelihood", "PAIR_LABELS", "PAIR_PATTERN_INDICES", "PATTERNS",
    "PATTERN_WEIGHTS", "DETECTOR_MODEL_SCHEMA", "DETECTOR_MODEL_VERSION",
]
