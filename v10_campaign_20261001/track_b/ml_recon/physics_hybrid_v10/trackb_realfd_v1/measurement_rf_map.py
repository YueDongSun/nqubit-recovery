"""Measurement-side RF/AOD residual maps with no feedback-control outputs.

``MeasurementRFMapV1`` remains a compact immutable diagnostic contract for
existing callers.  The actual Track B runtime uses
``TrainableMeasurementRFMapV1``: a zero-initialized, availability-aware torch
residual that consumes raw RF, projector, preparation, exposure, and session
features without exposing an action or a transition label.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from typing import Any, Mapping, Sequence

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - optional-ML installation only.
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False

from .contracts import RealStaticSupervisionV1, canonical_sha256, require_sha256
from .dual_descriptor import DualOperationDescriptorV1


FORBIDDEN_MEASUREMENT_RF_OUTPUTS = frozenset({
    "feedback_action", "action", "real_gain", "gain", "actuator_jacobian",
    "target_residual", "hardware_execution_authority", "hardware_action_authorized",
})


def validate_graph_basis_logical_orthogonality(
    logical_basis: np.ndarray, calibration_basis: np.ndarray, *, tolerance: float = 1.0e-10,
) -> None:
    """Require a calibration residual basis to be orthogonal to logical control axes."""
    logical = np.asarray(logical_basis, dtype=np.float64)
    calibration = np.asarray(calibration_basis, dtype=np.float64)
    if logical.ndim != 2 or calibration.ndim != 2 or logical.shape[0] != calibration.shape[0]:
        raise ValueError("logical and calibration bases must share a row dimension")
    if not np.all(np.isfinite(logical)) or not np.all(np.isfinite(calibration)):
        raise ValueError("basis matrices must be finite")
    cross = logical.T @ calibration
    if not np.allclose(cross, 0.0, atol=float(tolerance), rtol=0.0):
        raise ValueError("measurement RF calibration basis overlaps the logical control basis")


def _raw_diagnostic_features(descriptor: DualOperationDescriptorV1) -> np.ndarray:
    """Summarize registered raw RF fields without treating missing fields as zero.

    Missing values contribute through a separate availability fraction.  This is
    diagnostic-only; the torch runtime retains the complete per-feature mask.
    """
    raw = descriptor.raw_rf_command
    groups = (
        raw.raw_amplitude, raw.relative_phase_sin, raw.relative_phase_cos,
        raw.frequency_hz, raw.frequency_offset_hz, raw.rf_voltage_or_dac,
        raw.per_channel_power, raw.quadratic_terms, raw.cross_amplitude_terms,
        raw.cross_phase_terms,
    )
    values: list[float] = []
    for group in groups:
        finite = [float(value) for value in group if value is not None]
        if finite:
            transformed = [math.copysign(math.log1p(abs(value)), value) for value in finite]
            values.extend((float(np.mean(transformed)), float(np.std(transformed))))
        else:
            values.extend((0.0, 0.0))
    scalar_values = (raw.total_power, raw.maximum_power, raw.amplitude_imbalance)
    for value in scalar_values:
        values.append(0.0 if value is None else math.copysign(math.log1p(abs(float(value))), float(value)))
    all_optional = [value for group in groups for value in group] + list(scalar_values)
    values.append(float(sum(value is not None for value in all_optional)) / max(len(all_optional), 1))
    values.extend((float(raw.cycle_index), float(raw.within_cycle_index)))
    return np.asarray(values, dtype=np.float64)


@dataclass(frozen=True)
class MeasurementRFMapPredictionV1:
    """Effective measurement-law diagnostics, excluding all control fields."""

    effective_projector_residual: float
    readout_efficiency: float
    background_rate: float
    overdispersion: float
    visibility: float
    operation_family_bias: float
    heteroscedastic_scale: float
    session_effect: float
    residual_zero_initialized: bool
    schema: str = field(default="MeasurementRFMapPredictionV1", init=False)

    def __post_init__(self) -> None:
        values = (
            self.effective_projector_residual, self.readout_efficiency, self.background_rate,
            self.overdispersion, self.visibility, self.operation_family_bias,
            self.heteroscedastic_scale, self.session_effect,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("measurement RF prediction fields must be finite")
        if self.readout_efficiency < 0.0 or self.background_rate < 0.0:
            raise ValueError("measurement-law rates/efficiencies cannot be negative")

    def to_dict(self) -> dict[str, object]:
        result = {
            "schema": self.schema,
            "effective_projector_residual": self.effective_projector_residual,
            "readout_efficiency": self.readout_efficiency,
            "background_rate": self.background_rate,
            "overdispersion": self.overdispersion,
            "visibility": self.visibility,
            "operation_family_bias": self.operation_family_bias,
            "heteroscedastic_scale": self.heteroscedastic_scale,
            "session_effect": self.session_effect,
            "residual_zero_initialized": self.residual_zero_initialized,
        }
        if FORBIDDEN_MEASUREMENT_RF_OUTPUTS & set(result):
            raise RuntimeError("measurement RF map emitted a forbidden control output")
        return result


@dataclass(frozen=True)
class MeasurementRFMapV1:
    """Immutable zero-initialized residual diagnostic around a nominal measurement law."""

    system_type: str
    nominal_forward_sha: str
    projector_descriptor_sha: str
    rf_command_basis_sha: str
    graph_basis_sha: str
    residual_parameters: tuple[float, ...] = ()
    session_scale: float = 0.0
    schema: str = field(default="MeasurementRFMapV1", init=False)

    def __post_init__(self) -> None:
        if self.system_type not in ("3qubit", "4qubit"):
            raise ValueError("measurement RF map supports only 3qubit or 4qubit")
        for name in (
            "nominal_forward_sha", "projector_descriptor_sha", "rf_command_basis_sha",
            "graph_basis_sha",
        ):
            require_sha256(getattr(self, name), name)
        values = tuple(float(value) for value in self.residual_parameters)
        if not all(math.isfinite(value) for value in values) or not math.isfinite(float(self.session_scale)):
            raise ValueError("measurement RF residual parameters must be finite")
        object.__setattr__(self, "residual_parameters", values)

    @property
    def zero_initialized(self) -> bool:
        return all(value == 0.0 for value in self.residual_parameters) and self.session_scale == 0.0

    def predict(
        self,
        descriptor: DualOperationDescriptorV1,
        *,
        prepared_command_coefficients: Sequence[float],
        exposure: float | None,
        session_id: str | None,
    ) -> MeasurementRFMapPredictionV1:
        """Return an observation-only measurement law from source-bound descriptors."""
        if descriptor.ideal_projector.mode_count != (3 if self.system_type == "3qubit" else 4):
            raise ValueError("descriptor mode count does not match measurement RF map system")
        if descriptor.raw_rf_command.rf_command_basis_sha != self.rf_command_basis_sha:
            raise ValueError("RF descriptor basis SHA does not match measurement map")
        prepared = np.asarray(tuple(prepared_command_coefficients), dtype=np.float64)
        if prepared.ndim != 1 or not np.all(np.isfinite(prepared)):
            raise ValueError("prepared command coefficients must be a finite explicit vector")
        if exposure is not None and (not math.isfinite(float(exposure)) or float(exposure) <= 0.0):
            raise ValueError("exposure must be positive/finite when available")
        if session_id is not None and (type(session_id) is not str or not session_id):
            raise ValueError("session identity must be nonempty when available")
        rf_features = _raw_diagnostic_features(descriptor)
        if self.residual_parameters:
            weights = np.resize(np.asarray(self.residual_parameters, dtype=np.float64), rf_features.shape)
            residual = float(np.dot(weights, rf_features) / max(rf_features.size, 1))
        else:
            residual = 0.0
        prep_scale = float(np.tanh(prepared.mean())) if prepared.size else 0.0
        exposure_scale = 0.0 if exposure is None else float(np.tanh(math.log1p(float(exposure)) / 10.0))
        prediction = MeasurementRFMapPredictionV1(
            effective_projector_residual=residual,
            readout_efficiency=max(0.0, 1.0 + 0.05 * residual),
            background_rate=max(0.0, 0.01 * abs(residual)),
            overdispersion=max(0.0, 0.01 * abs(residual)),
            visibility=float(np.clip(1.0 - 0.05 * abs(residual), 0.0, 1.0)),
            operation_family_bias=0.01 * prep_scale * residual,
            heteroscedastic_scale=1.0 + 0.01 * abs(residual) + 0.001 * exposure_scale,
            session_effect=self.session_scale if session_id is not None else 0.0,
            residual_zero_initialized=self.zero_initialized,
        )
        prediction.to_dict()
        return prediction

    def update_from_real_static(
        self, *, supervision: RealStaticSupervisionV1, residual_update: Sequence[float], session_scale: float | None = None,
    ) -> "MeasurementRFMapV1":
        """Return a new observation-only map; causal labels are rejected at the boundary."""
        supervision.validate()
        update = tuple(float(value) for value in residual_update)
        if not all(math.isfinite(value) for value in update):
            raise ValueError("static residual update must be finite")
        if self.residual_parameters and len(update) != len(self.residual_parameters):
            raise ValueError("static residual update dimension must remain registered")
        return replace(
            self,
            residual_parameters=update,
            session_scale=self.session_scale if session_scale is None else float(session_scale),
        )

    def sha256(self) -> str:
        return canonical_sha256({
            "schema": self.schema,
            "system_type": self.system_type,
            "nominal_forward_sha": self.nominal_forward_sha,
            "projector_descriptor_sha": self.projector_descriptor_sha,
            "rf_command_basis_sha": self.rf_command_basis_sha,
            "graph_basis_sha": self.graph_basis_sha,
            "residual_parameters": list(self.residual_parameters),
            "session_scale": self.session_scale,
        })


@dataclass(frozen=True)
class TrainableMeasurementRFMapConfigV1:
    """Architecture declaration for the observation-only RF residual branch."""

    projector_dim: int
    raw_rf_dim: int
    prepared_command_dim: int
    session_count: int
    width: int = 64
    adapter_rank: int = 8
    schema: str = "TrainableMeasurementRFMapConfigV1"

    def validate(self) -> None:
        if min(self.projector_dim, self.raw_rf_dim, self.prepared_command_dim, self.session_count) <= 0:
            raise ValueError("measurement RF dimensions/session count must be positive")
        if self.width <= self.adapter_rank or self.adapter_rank <= 0:
            raise ValueError("measurement RF width/adapter rank are invalid")

    def sha256(self) -> str:
        self.validate()
        return canonical_sha256({
            "schema": self.schema,
            "projector_dim": self.projector_dim,
            "raw_rf_dim": self.raw_rf_dim,
            "prepared_command_dim": self.prepared_command_dim,
            "session_count": self.session_count,
            "width": self.width,
            "adapter_rank": self.adapter_rank,
        })


class TrainableMeasurementRFMapV1(nn.Module if TORCH_AVAILABLE else object):
    """Low-rank, zero-init residual map for a measured RF observation law.

    The map has no action head.  Every raw RF scalar is paired with its source
    availability bit, while preparation/exposure/session inputs carry their own
    masks.  Thus unknown hardware fields cannot masquerade as measured zeros.
    """

    def __init__(self, config: TrainableMeasurementRFMapConfigV1) -> None:
        if not TORCH_AVAILABLE:
            raise RuntimeError("trainable measurement RF map requires PyTorch")
        super().__init__()
        config.validate()
        self.config = config
        width = config.width
        self.projector_encoder = nn.Sequential(nn.Linear(config.projector_dim, width), nn.GELU())
        self.raw_rf_encoder = nn.Sequential(nn.Linear(2 * config.raw_rf_dim, width), nn.GELU())
        self.prepared_encoder = nn.Sequential(nn.Linear(2 * config.prepared_command_dim, width), nn.GELU())
        self.exposure_encoder = nn.Sequential(nn.Linear(2, width), nn.GELU())
        self.session_embedding = nn.Embedding(config.session_count, width)
        self.adapter_down = nn.Linear(width, config.adapter_rank, bias=False)
        self.adapter_up = nn.Linear(config.adapter_rank, width, bias=False)
        self.residual_head = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 8))
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

    @staticmethod
    def _require_shape(name: str, value: "torch.Tensor", shape: tuple[int, ...]) -> None:
        if tuple(value.shape) != shape:
            raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
        if not bool(torch.all(torch.isfinite(value.to(torch.float32))).detach()):
            raise ValueError(f"{name} must be finite")

    def forward(
        self, *, projector_descriptor: "torch.Tensor", raw_rf_descriptor: "torch.Tensor",
        raw_rf_feature_mask: "torch.Tensor", prepared_command: "torch.Tensor",
        prepared_command_mask: "torch.Tensor", exposure_s: "torch.Tensor",
        exposure_known_mask: "torch.Tensor", session_index: "torch.Tensor",
        session_known_mask: "torch.Tensor",
    ) -> dict[str, "torch.Tensor"]:
        if projector_descriptor.ndim != 3:
            raise ValueError("projector descriptor must be [batch,operations,features]")
        batch, operations, projector_dim = projector_descriptor.shape
        if projector_dim != self.config.projector_dim:
            raise ValueError("measurement RF projector dimension changed")
        self._require_shape("raw RF descriptor", raw_rf_descriptor, (batch, operations, self.config.raw_rf_dim))
        if raw_rf_feature_mask.dtype != torch.bool or tuple(raw_rf_feature_mask.shape) != tuple(raw_rf_descriptor.shape):
            raise ValueError("raw RF availability mask must match raw RF features")
        self._require_shape("prepared command", prepared_command, (batch, self.config.prepared_command_dim))
        if prepared_command_mask.dtype != torch.bool or tuple(prepared_command_mask.shape) != tuple(prepared_command.shape):
            raise ValueError("prepared-command mask must match prepared command")
        self._require_shape("exposure", exposure_s.reshape(batch), (batch,))
        if exposure_known_mask.dtype != torch.bool or tuple(exposure_known_mask.shape) != (batch,):
            raise ValueError("exposure-known mask must be bool [batch]")
        if bool(torch.any(exposure_known_mask & (exposure_s.reshape(batch) <= 0.0))):
            raise ValueError("known exposure must be strictly positive")
        if session_index.dtype != torch.long or tuple(session_index.shape) != (batch,):
            raise ValueError("session index must be int64 [batch]")
        if session_known_mask.dtype != torch.bool or tuple(session_known_mask.shape) != (batch,):
            raise ValueError("session-known mask must be bool [batch]")
        if bool(torch.any(session_index < 0)) or bool(torch.any(session_index >= self.config.session_count)):
            raise ValueError("session index is outside the registered V2.4 source vocabulary")
        raw_mask = raw_rf_feature_mask.to(dtype=raw_rf_descriptor.dtype)
        raw = torch.cat((raw_rf_descriptor * raw_mask, raw_mask), dim=-1)
        prepared_mask = prepared_command_mask.to(dtype=prepared_command.dtype)
        prepared = torch.cat((prepared_command * prepared_mask, prepared_mask), dim=-1)
        exposure_known = exposure_known_mask.to(dtype=exposure_s.dtype)
        exposure = torch.stack((
            torch.log1p(exposure_s.clamp_min(0.0)) * exposure_known,
            exposure_known,
        ), dim=-1)
        session = self.session_embedding(session_index) * session_known_mask.to(dtype=projector_descriptor.dtype).unsqueeze(-1)
        token = self.projector_encoder(projector_descriptor.to(torch.float32))
        token = token + self.raw_rf_encoder(raw.to(torch.float32))
        context = self.prepared_encoder(prepared.to(torch.float32)) + self.exposure_encoder(exposure.to(torch.float32)) + session
        token = token + context.unsqueeze(1)
        token = token + self.adapter_up(F.gelu(self.adapter_down(token)))
        residual = self.residual_head(token)
        # These are observation-law quantities only; none is a control output.
        # A zero residual must be the registered nominal measurement law.  Use
        # transforms anchored at (residual == 0) to avoid turning an
        # initialization checkpoint into an arbitrary 0.693/0.5 apparatus.
        output = {
            "token_context": token,
            "conditional_logit_residual": residual[..., :3],
            "effective_projector_residual": torch.tanh(residual[..., 3]),
            "readout_efficiency": torch.exp(residual[..., 4].clamp(-20.0, 20.0)),
            "background_rate": (F.softplus(residual[..., 5]) - math.log(2.0)).clamp_min(0.0),
            "overdispersion": (F.softplus(residual[..., 6]) - math.log(2.0)).clamp_min(0.0),
            "visibility": (
                torch.exp(math.log(2.0) - F.softplus(residual[..., 7]))
                .clamp(0.0, 1.0)
            ),
            "session_effect": session,
            "residual_l2": residual.square().mean(),
            "raw_rf_available_fraction": raw_mask.mean(),
            "prepared_command_available_fraction": prepared_mask.mean(),
        }
        if FORBIDDEN_MEASUREMENT_RF_OUTPUTS & set(output):
            raise RuntimeError("trainable measurement RF map emitted a forbidden control output")
        return output


def measurement_rf_regularization(output: Mapping[str, "torch.Tensor"]) -> "torch.Tensor":
    """Return a finite zero-init residual penalty for the native loss assembly."""
    if "residual_l2" not in output:
        raise ValueError("measurement RF regularization requires residual_l2")
    value = output["residual_l2"]
    if value.ndim != 0 or not bool(torch.isfinite(value).detach()):
        raise ValueError("measurement RF residual penalty must be one finite scalar")
    return value


__all__ = [
    "FORBIDDEN_MEASUREMENT_RF_OUTPUTS", "MeasurementRFMapPredictionV1", "MeasurementRFMapV1",
    "TORCH_AVAILABLE", "TrainableMeasurementRFMapConfigV1", "TrainableMeasurementRFMapV1",
    "measurement_rf_regularization", "validate_graph_basis_logical_orthogonality",
]
