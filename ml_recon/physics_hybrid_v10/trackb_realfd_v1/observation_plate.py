"""Masked set/query real-static observation scaffolding with explicit leakage guards."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
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

from .contracts import RealStaticSupervisionV1, canonical_sha256
from .measurement_rf_map import (
    TrainableMeasurementRFMapConfigV1,
    TrainableMeasurementRFMapV1,
)


def deterministic_query_mask(
    *, sample_id: str, epoch: int, view_seed: int, operation_ids: Sequence[str],
    operation_families: Sequence[str], cycle_indices: Sequence[int], query_fraction: float = 0.25,
) -> np.ndarray:
    """Construct a family/cycle-stratified replayable mask without seeing count values."""
    if type(sample_id) is not str or not sample_id:
        raise ValueError("sample_id must be explicit")
    if type(epoch) is not int or type(view_seed) is not int:
        raise TypeError("epoch and view_seed must be integers")
    if not 0.20 <= float(query_fraction) <= 0.30:
        raise ValueError("default real-static query fraction must be in [0.20, 0.30]")
    size = len(operation_ids)
    if not size or len(operation_families) != size or len(cycle_indices) != size:
        raise ValueError("operation IDs/families/cycles must align and be nonempty")
    if len(set(operation_ids)) != size or any(type(value) is not str or not value for value in operation_ids):
        raise ValueError("operation IDs must be unique nonempty strings")
    groups: dict[tuple[str, int], list[int]] = {}
    for index, (family, cycle) in enumerate(zip(operation_families, cycle_indices, strict=True)):
        if type(family) is not str or not family or type(cycle) is not int or cycle < 0:
            raise ValueError("family/cycle metadata must be explicit")
        groups.setdefault((family, cycle), []).append(index)
    result = np.zeros(size, dtype=np.bool_)
    for (family, cycle), indices in groups.items():
        # Keep at least one visible member where a stratum has enough operations.
        requested = int(round(len(indices) * float(query_fraction)))
        count = min(len(indices) - 1, max(1, requested)) if len(indices) > 1 else 0
        ordered = sorted(
            indices,
            key=lambda index: hashlib.sha256(
                f"TrackBQueryMaskV1|{sample_id}|{epoch}|{view_seed}|{family}|{cycle}|{operation_ids[index]}".encode("utf-8")
            ).hexdigest(),
        )
        result[ordered[:count]] = True
    return result


@dataclass(frozen=True)
class MaskedSetQueryObservationV1:
    """One real-static row whose query values are inaccessible through model inputs."""

    sample_id: str
    four_cell_counts: tuple[tuple[int, int, int, int], ...]
    query_mask: tuple[bool, ...]
    operation_ids: tuple[str, ...]
    operation_families: tuple[str, ...]
    cycle_indices: tuple[int, ...]
    prepared_command_coefficients: tuple[float, ...]
    global_multi: tuple[int, ...]
    global_multi_scope: str
    epoch: int = 0
    view_seed: int = 0
    query_fraction: float = 0.25
    supervision: RealStaticSupervisionV1 = field(default_factory=RealStaticSupervisionV1)
    schema: str = field(default="MaskedSetQueryObservationV1", init=False)

    def __post_init__(self) -> None:
        count = len(self.four_cell_counts)
        if count == 0 or len(self.query_mask) != count:
            raise ValueError("four-cell counts and query mask must align")
        if len(self.operation_ids) != count or len(self.operation_families) != count or len(self.cycle_indices) != count:
            raise ValueError("operation metadata must align to query mask")
        if any(type(value) is not bool for value in self.query_mask):
            raise TypeError("query mask requires exact booleans")
        for cells in self.four_cell_counts:
            if len(cells) != 4 or any(type(value) is not int or value < 0 for value in cells):
                raise ValueError("four-cell counts must preserve nonnegative native integer cells")
        if not all(math.isfinite(float(value)) for value in self.prepared_command_coefficients):
            raise ValueError("prepared command coefficients must be finite setpoints")
        if any(type(value) is not int or value < 0 for value in self.global_multi):
            raise ValueError("global multi values must remain native nonnegative counts")
        if self.global_multi_scope not in ("global", "state"):
            raise ValueError("global multi remains global/state scoped and cannot be operation-broadcast")
        expected_mask = deterministic_query_mask(
            sample_id=self.sample_id,
            epoch=self.epoch,
            view_seed=self.view_seed,
            operation_ids=self.operation_ids,
            operation_families=self.operation_families,
            cycle_indices=self.cycle_indices,
            query_fraction=self.query_fraction,
        )
        if tuple(bool(value) for value in expected_mask) != self.query_mask:
            raise ValueError(
                "query mask must be the registered deterministic family/cycle "
                "plan independent of hidden count values"
            )
        self.supervision.validate()

    @property
    def visible_mask(self) -> np.ndarray:
        return ~np.asarray(self.query_mask, dtype=np.bool_)

    def visible_input(self) -> dict[str, object]:
        """Return model input with query counts zeroed and no query target aliases."""
        counts = np.asarray(self.four_cell_counts, dtype=np.int64)
        visible = self.visible_mask
        return {
            "sample_id": self.sample_id,
            "visible_four_cell_counts": (counts * visible[:, None]).copy(),
            "visible_operation_mask": visible.copy(),
            "operation_ids": self.operation_ids,
            "operation_families": self.operation_families,
            "cycle_indices": self.cycle_indices,
            "epoch": self.epoch,
            "view_seed": self.view_seed,
            "query_fraction": self.query_fraction,
            "prepared_command_coefficients": self.prepared_command_coefficients,
            # Global multi-click counts are a state-scoped prediction target,
            # not a visible conditioning feature.  Supplying their values here
            # would let a query predictor recover hidden-count information
            # through a global aggregate even though no per-operation query
            # tensor is present.
            "global_multi_scope": self.global_multi_scope,
        }

    def query_targets(self) -> tuple[np.ndarray, np.ndarray]:
        """Return targets only to the loss layer, separate from ``visible_input``."""
        return np.asarray(self.four_cell_counts, dtype=np.int64).copy(), np.asarray(self.query_mask, dtype=np.bool_).copy()

    def mask_sha(self) -> str:
        return canonical_sha256({
            "schema": "TrackBDeterministicQueryMaskV1",
            "sample_id": self.sample_id,
            "query_mask": list(self.query_mask),
            "operation_ids": list(self.operation_ids),
            "epoch": self.epoch,
            "view_seed": self.view_seed,
            "query_fraction": self.query_fraction,
        })


@dataclass(frozen=True)
class MaskedObservationPredictionV1:
    """Observation-only output; a model cannot emit a control direction here."""

    four_cell_mean: tuple[tuple[float, float, float, float], ...]
    count_concentration: tuple[float, ...]
    ood_probability: float
    apparatus_confidence: float
    schema: str = field(default="MaskedObservationPredictionV1", init=False)

    def __post_init__(self) -> None:
        if not self.four_cell_mean or any(len(row) != 4 or any(not math.isfinite(float(value)) or float(value) < 0.0 for value in row) for row in self.four_cell_mean):
            raise ValueError("four-cell means must be finite nonnegative operation rows")
        if len(self.count_concentration) != len(self.four_cell_mean) or any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in self.count_concentration):
            raise ValueError("concentrations must be positive and aligned")
        if not 0.0 <= self.ood_probability <= 1.0 or not 0.0 <= self.apparatus_confidence <= 1.0:
            raise ValueError("OOD/confidence must lie in [0,1]")


class MaskedObservationPlateV1:
    """Small deterministic CPU scaffolding for masked observation smoke checks.

    It pools only visible operations and returns only count-model quantities.
    """

    def predict(self, observation: MaskedSetQueryObservationV1) -> MaskedObservationPredictionV1:
        source = observation.visible_input()
        counts = np.asarray(source["visible_four_cell_counts"], dtype=np.float64)
        visible = np.asarray(source["visible_operation_mask"], dtype=np.bool_)
        visible_total = counts.sum(axis=1)
        baseline = float(visible_total[visible].mean()) if bool(visible.any()) else 0.0
        means = np.where(visible[:, None], counts, baseline / 4.0)
        concentration = tuple(float(max(1.0, value + 1.0)) for value in means.sum(axis=1))
        queried_fraction = float((~visible).mean())
        return MaskedObservationPredictionV1(
            four_cell_mean=tuple(tuple(float(value) for value in row) for row in means),
            count_concentration=concentration,
            ood_probability=queried_fraction,
            apparatus_confidence=1.0 - queried_fraction,
        )


FORBIDDEN_OBSERVATION_OUTPUTS = frozenset({
    "action", "feedback_action", "gain", "real_gain", "jacobian", "target_residual",
    "hardware_action_authorized", "hardware_execution_authority", "transition_prediction",
})


@dataclass(frozen=True)
class TrainableMaskedObservationPlateConfigV1:
    """Explicit shape declaration for the Track B masked observation plate."""

    system_type: str
    mode_count: int
    projector_dim: int
    raw_rf_dim: int
    prepared_command_dim: int
    session_count: int
    width: int = 64
    heads: int = 4
    blocks: int = 2
    adapter_rank: int = 8
    schema: str = "TrainableMaskedObservationPlateConfigV1"

    def validate(self) -> None:
        expected_modes = {"3qubit": 3, "4qubit": 4}
        if self.schema != "TrainableMaskedObservationPlateConfigV1" \
                or type(self.system_type) is not str \
                or self.system_type not in expected_modes \
                or type(self.mode_count) is not int \
                or self.mode_count != expected_modes[self.system_type]:
            raise ValueError("Track B observation plate system/mode contract is invalid")
        dimensions = (
            self.projector_dim, self.raw_rf_dim, self.prepared_command_dim, self.session_count,
            self.width, self.heads, self.blocks, self.adapter_rank,
        )
        if any(type(value) is not int for value in dimensions) \
                or min(self.projector_dim, self.raw_rf_dim, self.prepared_command_dim, self.session_count) <= 0:
            raise ValueError("Track B observation dimensions/session count must be positive")
        if self.prepared_command_dim % 2 != 0 or self.prepared_command_dim < 2 * self.mode_count:
            raise ValueError("prepared command must retain real/imaginary mode coefficients")
        if self.width <= self.adapter_rank or self.adapter_rank <= 0 or self.width % self.heads != 0:
            raise ValueError("Track B observation width/heads/adapter contract is invalid")
        if self.blocks <= 0:
            raise ValueError("Track B observation plate requires at least one attention block")

    def to_dict(self) -> dict[str, object]:
        """Return the exact typed construction contract required for checkpoint replay."""
        self.validate()
        return {
            "schema": self.schema,
            "system_type": self.system_type,
            "mode_count": self.mode_count,
            "projector_dim": self.projector_dim,
            "raw_rf_dim": self.raw_rf_dim,
            "prepared_command_dim": self.prepared_command_dim,
            "session_count": self.session_count,
            "width": self.width,
            "heads": self.heads,
            "blocks": self.blocks,
            "adapter_rank": self.adapter_rank,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "TrainableMaskedObservationPlateConfigV1":
        expected = {
            "schema", "system_type", "mode_count", "projector_dim", "raw_rf_dim",
            "prepared_command_dim", "session_count", "width", "heads", "blocks", "adapter_rank",
        }
        if set(payload) != expected or payload.get("schema") != "TrainableMaskedObservationPlateConfigV1":
            raise ValueError("Track B observation plate config schema is invalid")
        config = cls(
            system_type=payload["system_type"],
            mode_count=payload["mode_count"],
            projector_dim=payload["projector_dim"],
            raw_rf_dim=payload["raw_rf_dim"],
            prepared_command_dim=payload["prepared_command_dim"],
            session_count=payload["session_count"],
            width=payload["width"],
            heads=payload["heads"],
            blocks=payload["blocks"],
            adapter_rank=payload["adapter_rank"],
        )
        config.validate()
        return config

    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())


class TrainableMaskedObservationPlateV1(nn.Module if TORCH_AVAILABLE else object):
    """Masked set/query count model with a nominal Born path and RF residual.

    The forward method intentionally reads neither ``global_multi`` nor any
    causal label.  Four-cell query values are rebuilt from the public query
    mask before the visible encoder sees them.  This makes changing hidden query
    counts observationally inert at the model-input boundary.
    """

    def __init__(self, config: TrainableMaskedObservationPlateConfigV1) -> None:
        if not TORCH_AVAILABLE:
            raise RuntimeError("trainable Track B observation plate requires PyTorch")
        super().__init__()
        config.validate()
        self.config = config
        width = config.width
        self.visible_encoder = nn.Sequential(nn.Linear(6, width), nn.GELU(), nn.LayerNorm(width))
        self.projector_encoder = nn.Sequential(nn.Linear(config.projector_dim, width), nn.GELU())
        self.rf_map = TrainableMeasurementRFMapV1(TrainableMeasurementRFMapConfigV1(
            projector_dim=config.projector_dim,
            raw_rf_dim=config.raw_rf_dim,
            prepared_command_dim=config.prepared_command_dim,
            session_count=config.session_count,
            width=config.width,
            adapter_rank=config.adapter_rank,
        ))
        self.attention = nn.ModuleList([
            nn.MultiheadAttention(width, config.heads, batch_first=True)
            for _ in range(config.blocks)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(config.blocks)])
        self.adapter_down = nn.Linear(width, config.adapter_rank, bias=False)
        self.adapter_up = nn.Linear(config.adapter_rank, width, bias=False)
        self.residual_head = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 3))
        self.count_scale_head = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))
        self.uncertainty_head = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 4))
        self.global_multi_head = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 12))
        self.concentration_raw = nn.Parameter(torch.tensor(4.0))
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)
        nn.init.zeros_(self.count_scale_head[-1].weight)
        nn.init.zeros_(self.count_scale_head[-1].bias)

    @staticmethod
    def _require_tensor(name: str, value: "torch.Tensor", *, dtype: Any | None = None, ndim: int | None = None) -> None:
        if dtype is not None and value.dtype != dtype:
            raise TypeError(f"{name} must have dtype {dtype}")
        if ndim is not None and value.ndim != ndim:
            raise ValueError(f"{name} must have rank {ndim}")
        if value.dtype.is_floating_point and not bool(torch.all(torch.isfinite(value)).detach()):
            raise ValueError(f"{name} must be finite")

    def _nominal_born_probability(
        self, projector_descriptor: "torch.Tensor", prepared_command: "torch.Tensor",
        prepared_command_mask: "torch.Tensor",
    ) -> "torch.Tensor":
        """Evaluate the registered projector/prepared-command Born contraction.

        Projector V2 stores diagonal then upper-triangle real/imaginary terms.
        The preparation vector is explicitly a *prepared command*, never an
        asserted observed state.  When it is unavailable the branch uses the
        maximally mixed single-excitation reference rather than a hidden label.
        """
        n = self.config.mode_count
        upper_count = n * (n - 1) // 2
        diagonal = projector_descriptor[..., :n].clamp_min(0.0)
        upper_real = projector_descriptor[..., n:n + upper_count]
        upper_imag = projector_descriptor[..., n + upper_count:n + 2 * upper_count]
        half = self.config.prepared_command_dim // 2
        amplitude = prepared_command[:, :n] + 1j * prepared_command[:, half:half + n]
        available = prepared_command_mask[:, :n].all(dim=-1) & prepared_command_mask[:, half:half + n].all(dim=-1)
        # A missing prepared command is not a pure W state.  Keep a separate
        # maximally-mixed single-excitation branch, whose projector probability
        # is Tr(P)/N.  The coherence terms below are zero for unavailable rows.
        norm = amplitude.abs().square().sum(dim=-1, keepdim=True)
        normalized = amplitude / norm.sqrt().clamp_min(1.0e-8)
        state = normalized * available[:, None].to(dtype=normalized.real.dtype)
        diagonal_term = (diagonal * state.abs().square().unsqueeze(1)).sum(dim=-1)
        upper_term = torch.zeros_like(diagonal_term)
        cursor = 0
        for left in range(n):
            for right in range(left + 1, n):
                projector_upper = upper_real[..., cursor] + 1j * upper_imag[..., cursor]
                rho_right_left = state[:, right, None] * state[:, left, None].conj()
                upper_term = upper_term + 2.0 * (projector_upper * rho_right_left).real
                cursor += 1
        known_probability = diagonal_term + upper_term
        mixed_probability = diagonal.sum(dim=-1) / float(n)
        probability = torch.where(
            available[:, None], known_probability, mixed_probability.expand_as(known_probability),
        )
        return probability.clamp(1.0e-5, 1.0 - 1.0e-5)

    @staticmethod
    def _four_cell_from_conditionals(
        signal: "torch.Tensor", coin_given_signal: "torch.Tensor",
        idler_given_no_signal: "torch.Tensor", total: "torch.Tensor",
    ) -> "torch.Tensor":
        probability = torch.stack((
            (1.0 - signal) * (1.0 - idler_given_no_signal),
            signal * (1.0 - coin_given_signal),
            (1.0 - signal) * idler_given_no_signal,
            signal * coin_given_signal,
        ), dim=-1)
        return probability * total.unsqueeze(-1)

    def forward(self, batch: Mapping[str, Any]) -> dict[str, "torch.Tensor"]:
        required = {
            "four_cell_counts", "operation_mask", "query_mask", "projector_descriptor",
            "raw_rf_descriptor", "raw_rf_feature_mask", "prepared_command",
            "prepared_command_mask", "exposure_s", "exposure_known_mask",
            "session_index", "session_known_mask",
        }
        missing = required - set(batch)
        if missing:
            raise ValueError(f"Track B observation batch is missing fields: {sorted(missing)}")
        counts = batch["four_cell_counts"]
        operation_mask = batch["operation_mask"]
        query_mask = batch["query_mask"]
        projector = batch["projector_descriptor"]
        self._require_tensor("four-cell counts", counts, ndim=3)
        self._require_tensor("operation mask", operation_mask, dtype=torch.bool, ndim=2)
        self._require_tensor("query mask", query_mask, dtype=torch.bool, ndim=2)
        self._require_tensor("projector descriptor", projector, ndim=3)
        if counts.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64) \
                or counts.shape[-1] != 4 or tuple(operation_mask.shape) != tuple(counts.shape[:2]) \
                or tuple(query_mask.shape) != tuple(counts.shape[:2]):
            raise ValueError("Track B counts/masks require [batch,operations,4] and bool [batch,operations]")
        if tuple(projector.shape[:2]) != tuple(counts.shape[:2]) or projector.shape[-1] != self.config.projector_dim:
            raise ValueError("Track B projector descriptor does not match count operations")
        if bool(torch.any(query_mask & ~operation_mask)):
            raise ValueError("Track B query mask includes unavailable operations")
        visible_mask = operation_mask & ~query_mask
        if not bool(torch.all(visible_mask.any(dim=1))):
            raise ValueError("Track B set/query plate requires a visible operation in every row")
        # Never trust a caller's allegedly visible count tensor.  Rebuild it from
        # the native counts and registered mask so hidden query values are zeroed.
        visible_counts = counts * visible_mask.unsqueeze(-1).to(dtype=counts.dtype)
        if "visible_four_cell_counts" in batch and not torch.equal(
            batch["visible_four_cell_counts"].to(dtype=counts.dtype), visible_counts,
        ):
            raise ValueError("visible four-cell input is inconsistent with the registered query mask")
        count_total = visible_counts.to(dtype=torch.float32).sum(dim=-1, keepdim=True)
        visible_features = torch.cat((
            torch.log1p(visible_counts.to(dtype=torch.float32)),
            torch.log1p(count_total),
            visible_mask.to(dtype=torch.float32).unsqueeze(-1),
        ), dim=-1)
        rf_output = self.rf_map(
            projector_descriptor=projector,
            raw_rf_descriptor=batch["raw_rf_descriptor"],
            raw_rf_feature_mask=batch["raw_rf_feature_mask"],
            prepared_command=batch["prepared_command"],
            prepared_command_mask=batch["prepared_command_mask"],
            exposure_s=batch["exposure_s"].reshape(-1),
            exposure_known_mask=batch["exposure_known_mask"].reshape(-1),
            session_index=batch["session_index"].reshape(-1),
            session_known_mask=batch["session_known_mask"].reshape(-1),
        )
        value = self.visible_encoder(visible_features) + self.projector_encoder(projector.to(torch.float32)) + rf_output["token_context"]
        hidden = self.projector_encoder(projector.to(torch.float32)) + rf_output["token_context"]
        key_padding_mask = ~visible_mask
        for attention, norm in zip(self.attention, self.norms):
            context, _ = attention(hidden, value, value, key_padding_mask=key_padding_mask, need_weights=False)
            hidden = norm(hidden + context)
        hidden = hidden + self.adapter_up(F.gelu(self.adapter_down(hidden)))
        born = self._nominal_born_probability(projector.to(torch.float32), batch["prepared_command"].to(torch.float32), batch["prepared_command_mask"])
        nominal_probability = torch.stack((
            (0.02 + 0.96 * born).clamp(1.0e-5, 1.0 - 1.0e-5),
            (0.02 + 0.96 * born).clamp(1.0e-5, 1.0 - 1.0e-5),
            (0.02 + 0.48 * (1.0 - born)).clamp(1.0e-5, 1.0 - 1.0e-5),
        ), dim=-1)
        residual = self.residual_head(hidden) + rf_output["conditional_logit_residual"]
        conditional = torch.sigmoid(torch.logit(nominal_probability) + residual)
        signal, coin_given_signal, idler_given_no_signal = conditional.unbind(dim=-1)
        pooled = (hidden * visible_mask.to(dtype=hidden.dtype).unsqueeze(-1)).sum(dim=1)
        pooled = pooled / visible_mask.sum(dim=1, keepdim=True).clamp_min(1).to(dtype=hidden.dtype)
        visible_total = visible_counts.to(dtype=torch.float32).sum(dim=(-1, -2))
        visible_count = visible_mask.sum(dim=1).to(dtype=torch.float32)
        nominal_total = (visible_total / visible_count.clamp_min(1.0)).clamp_min(1.0e-6)
        total = nominal_total.unsqueeze(-1) * torch.exp(self.count_scale_head(hidden).squeeze(-1).clamp(-4.0, 4.0))
        four_cell_mean = self._four_cell_from_conditionals(
            signal, coin_given_signal, idler_given_no_signal, total,
        )
        nominal_four_cell_mean = self._four_cell_from_conditionals(
            nominal_probability[..., 0], nominal_probability[..., 1], nominal_probability[..., 2],
            nominal_total.unsqueeze(-1).expand_as(signal),
        )
        uncertainty = self.uncertainty_head(hidden).clamp(-8.0, 8.0)
        multi_raw = self.global_multi_head(pooled)
        output = {
            "signal_probability": signal,
            "coin_given_signal_probability": coin_given_signal,
            "idler_given_no_signal_probability": idler_given_no_signal,
            "four_cell_mean": four_cell_mean,
            "nominal_four_cell_mean": nominal_four_cell_mean,
            "four_cell_concentration": F.softplus(self.concentration_raw).expand(counts.shape[0]) + 1.0e-6,
            "four_cell_log_variance": uncertainty,
            "multi_means": F.softplus(multi_raw[:, :3]) + 1.0e-6,
            "pair_logits": multi_raw[:, 3:9],
            "multi_dispersion": F.softplus(multi_raw[:, 9:12]) + 1.0e-6,
            "residual_logits": residual,
            "operation_tokens": hidden,
            "global_semantic_token": pooled,
            "measurement_rf": rf_output,
            "visible_operation_mask": visible_mask,
            "query_mask": query_mask,
        }
        if FORBIDDEN_OBSERVATION_OUTPUTS & set(output):
            raise RuntimeError("Track B observation plate emitted a forbidden control field")
        return output


__all__ = [
    "FORBIDDEN_OBSERVATION_OUTPUTS", "MaskedObservationPlateV1", "MaskedObservationPredictionV1",
    "MaskedSetQueryObservationV1", "TORCH_AVAILABLE", "TrainableMaskedObservationPlateConfigV1",
    "TrainableMaskedObservationPlateV1", "deterministic_query_mask",
]
