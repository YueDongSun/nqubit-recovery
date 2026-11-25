"""Effective real-window singleton response with a shared row-level PSD state.

This is a distinct candidate, not a new interpretation of V1 checkpoints.
Its probabilities refer to the native effective-READ / sticky-singleton
events. It does not equate READ to a write shot or infer microscopic source,
path or detector efficiencies. Only observed-only, effective logit contrasts
are fitted until microscopic conditioning and temporal evidence are available.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import math
from typing import Any, Mapping

import torch
from torch import nn

from .contracts import canonical_sha256
from .effective_family_context_v3 import (
    MODEL_CONFIG_SCHEMA as FAMILY_CONTEXT_CONFIG_SCHEMA,
    MODEL_SCHEMA as FAMILY_CONTEXT_MODEL_SCHEMA,
    Effective_family_context,
)
from .evaluation import build_trackb_fixed_visible_marginal_prediction
from .global_count_initialization import Global_count_initialization
from .measurement_context import Effective_measurement_context


MODEL_SCHEMA = "TrackBEffectiveSingletonObservationV2"
PREPARED_SEMANTICS = "prepared_state_command_coefficients_not_actual_state_or_feedback_target"
REGISTERED_FINITE_PROBABILITY_FLOOR = 1.0e-9
MINIMUM_COMMAND_MIXED_FRACTION = 1.0e-4


def _global_initialization_tensors(initialization: Global_count_initialization) -> dict[str, torch.Tensor]:
    """Use one CPU FP64 convention for immutable constants and their replay."""
    return {
        "global_mean_baseline": torch.tensor(initialization.mean_counts, dtype=torch.float64, device="cpu"),
        "global_dispersion_baseline": torch.tensor(initialization.dispersion, dtype=torch.float64, device="cpu"),
        "global_pair_log_baseline": torch.tensor(initialization.pair_probabilities, dtype=torch.float64, device="cpu").log(),
    }


def validate_measurement_context_binding(model: "Effective_singleton_observation") -> None:
    """Reject live module metadata that differs from the serialized model identity."""
    context = model.config.measurement_context
    family_module = model.family_context
    if (context is None) != (family_module is None) or (context is not None and (
        type(family_module) is not Effective_family_context or family_module.context != context
    )):
        raise ValueError("family response module differs from its immutable model configuration")


def validate_global_initialization_buffers(model: "Effective_singleton_observation") -> None:
    """Reject duplicate constants or context that contradict frozen artifacts.

    Trainable residual weights are intentionally not compared with M0. Device
    movement is allowed, but dtype conversion or updating fixed buffers is not.
    """
    validate_measurement_context_binding(model)
    initialization = model.config.global_initialization
    if initialization is None:
        return
    model_device = next(model.parameters()).device
    buffers = dict(model.named_buffers())
    for name, expected in _global_initialization_tensors(initialization).items():
        actual = buffers.get(name)
        if not isinstance(actual, torch.Tensor) or actual.shape != expected.shape \
                or actual.dtype != torch.float64 or actual.device != model_device \
                or actual.requires_grad or name in model._non_persistent_buffers_set \
                or not torch.equal(actual.detach().cpu(), expected):
            raise ValueError(f"global initialization buffer {name} differs from its immutable artifact")


@dataclass(frozen=True)
class Effective_observation_config:
    """Serializable model identity; structure-disabled is a separate comparator."""

    system_type: str
    mode_count: int
    projector_dim: int
    prepared_command_dim: int
    width: int = 64
    heads: int = 4
    blocks: int = 2
    structure_enabled: bool = True
    no_signal_structure_enabled: bool = False
    read_trials_source: str = "visible_average"
    command_mixed_fraction: float = 0.1
    maximum_response_correction: float = 8.0
    maximum_log_concentration_correction: float = 10.0
    finite_probability_floor: float = REGISTERED_FINITE_PROBABILITY_FLOOR
    schema: str = "TrackBEffectiveSingletonConfigV2"
    global_initialization: Global_count_initialization | None = None
    pair_residual_enabled: bool = True
    measurement_context: Effective_measurement_context | None = None

    def validate(self) -> None:
        expected_schema = (
            FAMILY_CONTEXT_CONFIG_SCHEMA
            if self.measurement_context is not None else "TrackBEffectiveSingletonConfigV2"
        )
        if self.schema != expected_schema or self.system_type not in {"3qubit", "4qubit"}:
            raise ValueError("effective observation model identity is invalid")
        if type(self.mode_count) is not int or self.mode_count != {"3qubit": 3, "4qubit": 4}[self.system_type]:
            raise ValueError("effective observation system/mode mismatch")
        if any(type(value) is not int or value <= 0 for value in (
            self.projector_dim, self.prepared_command_dim, self.width, self.heads, self.blocks,
        )):
            raise ValueError("effective observation dimensions must be positive integers")
        if self.projector_dim < self.mode_count**2 or self.prepared_command_dim < 2 * self.mode_count \
                or self.prepared_command_dim % 2 or self.width % self.heads:
            raise ValueError("effective observation shape contract is invalid")
        if type(self.structure_enabled) is not bool or type(self.no_signal_structure_enabled) is not bool:
            raise TypeError("structure flags must be strict booleans")
        if type(self.pair_residual_enabled) is not bool:
            raise TypeError("pair residual flag must be a strict boolean")
        if not self.pair_residual_enabled and self.global_initialization is None:
            raise ValueError("fixed pair prediction requires a frozen train initialization")
        if self.no_signal_structure_enabled and not self.structure_enabled:
            raise ValueError("no-signal structure requires the structural model")
        if self.read_trials_source not in {"visible_average", "native_scalar"}:
            raise ValueError("unregistered READ denominator source")
        if self.finite_probability_floor != REGISTERED_FINITE_PROBABILITY_FLOOR:
            raise ValueError("finite response domain is registered at the tested FP64 floor 1e-9")
        if not math.isfinite(self.command_mixed_fraction) or not MINIMUM_COMMAND_MIXED_FRACTION <= self.command_mixed_fraction <= 1:
            raise ValueError("complex64 command prior requires mixed fraction in [1e-4,1]")
        for value in (self.maximum_response_correction, self.maximum_log_concentration_correction):
            if not math.isfinite(value) or not 0 < value <= 12:
                raise ValueError("effective log corrections must be finite in (0,12]")
        if self.global_initialization is not None:
            if type(self.global_initialization) is not Global_count_initialization:
                raise TypeError("global initialization must have its registered immutable type")
            self.global_initialization.validate()
            if self.global_initialization.system_type != self.system_type:
                raise ValueError("global initialization cannot cross systems")
        if self.measurement_context is not None:
            if type(self.measurement_context) is not Effective_measurement_context:
                raise TypeError("measurement context must have its registered immutable type")
            self.measurement_context.validate()
            if not self.structure_enabled or self.no_signal_structure_enabled \
                    or self.read_trials_source != "native_scalar" \
                    or self.pair_residual_enabled or self.global_initialization is None:
                raise ValueError(
                    "family-context V3 requires structure, native READ, fixed pairs, "
                    "and frozen global initialization"
                )
            if self.measurement_context.admission_manifest_sha \
                    != self.global_initialization.admission_manifest_sha:
                raise ValueError("family-context admission differs from global initialization")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        if self.global_initialization is None:
            # Preserve the exact historical V2 config and seed-only M0 identity.
            payload.pop("global_initialization")
        if self.pair_residual_enabled:
            payload.pop("pair_residual_enabled")  # Preserve historical learned-pair config identity.
        if self.measurement_context is None:
            payload.pop("measurement_context")
        else:
            payload["measurement_context"] = self.measurement_context.to_dict()
        return payload

    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Effective_observation_config":
        optional_fields = {"global_initialization", "pair_residual_enabled", "measurement_context"}
        legacy_fields = {field.name for field in fields(cls)} - optional_fields
        if not legacy_fields <= set(payload) or set(payload) - legacy_fields - optional_fields:
            raise ValueError("effective observation config fields must match exactly")
        values = dict(payload)
        if "global_initialization" in values:
            values["global_initialization"] = Global_count_initialization.from_dict(values["global_initialization"])
        if "measurement_context" in values:
            if not isinstance(values["measurement_context"], Mapping):
                raise ValueError("measurement context must be an explicit typed mapping")
            values["measurement_context"] = Effective_measurement_context.from_dict(
                values["measurement_context"],
            )
        config = cls(**values)
        config.validate()
        if config.measurement_context is not None and config.to_dict() != dict(payload):
            raise ValueError("effective observation config changes under typed parsing")
        return config


def _fixed_projector_matrix(descriptor: torch.Tensor, mode_count: int) -> torch.Tensor:
    """Decode the registered diagonal/upper-real/upper-imaginary layout."""
    upper_count = mode_count * (mode_count - 1) // 2
    matrix = torch.diag_embed(descriptor[..., :mode_count]).to(torch.complex64)
    left, right = torch.triu_indices(mode_count, mode_count, offset=1, device=descriptor.device)
    upper = torch.complex(
        descriptor[..., mode_count:mode_count + upper_count],
        descriptor[..., mode_count + upper_count:mode_count + 2 * upper_count],
    )
    matrix[..., left, right] = upper
    matrix[..., right, left] = upper.conj()
    eigenvalues = torch.linalg.eigvalsh(matrix)
    if bool(((eigenvalues < -1e-5) | (eigenvalues > 1 + 1e-5)).any()):
        raise ValueError("registered projector must be a PSD effect bounded by identity")
    return matrix


def _prepared_prior(batch: Mapping[str, Any], config: Effective_observation_config) -> tuple[torch.Tensor, torch.Tensor]:
    command = batch["prepared_command"].to(torch.float32)
    mask = batch["prepared_command_mask"]
    if command.ndim != 2 or command.shape[1] != config.prepared_command_dim \
            or mask.dtype != torch.bool or mask.shape != command.shape:
        raise ValueError("prepared command/mask dimensions are invalid")
    if not bool(torch.isfinite(command[mask]).all()):
        raise ValueError("available prepared command must be finite")
    command = torch.where(mask, command, torch.zeros_like(command))
    modes = config.mode_count
    half = config.prepared_command_dim // 2
    available = mask[:, :modes].all(-1) & mask[:, half:half + modes].all(-1)
    state_vector = torch.complex(command[:, :modes], command[:, half:half + modes])
    norm = state_vector.abs().square().sum(-1)
    if bool((available & (norm <= 0)).any()):
        raise ValueError("available prepared command cannot be a zero vector")
    state_vector = state_vector / norm.clamp_min(1e-12).sqrt().unsqueeze(-1)
    pure = state_vector.unsqueeze(-1) * state_vector.conj().unsqueeze(-2)
    mixed = torch.eye(modes, dtype=torch.complex64, device=command.device) / modes
    prior = (1 - config.command_mixed_fraction) * pure + config.command_mixed_fraction * mixed
    use_command = available & config.structure_enabled
    return torch.where(use_command[:, None, None], prior, mixed), use_command


def _integrated_singleton_response(base: torch.Tensor, correction: torch.Tensor, floor: float) -> torch.Tensor:
    """Tilt -log P(no singleton in the window), never a clock Poisson rate.

A declared smooth finite-domain transform bounds the residual before the
exponential. It does not clip predictions after evaluation or alter targets.
The compensated expression preserves the B0 tensor exactly at zero residual.
"""
    integrated = -torch.log1p(-base)
    lower = torch.log(base.new_tensor(-math.log1p(-floor)) / integrated)
    upper = torch.log(base.new_tensor(-math.log(floor)) / integrated)
    bound = torch.where(correction >= 0, upper, -lower)
    delta = bound * torch.tanh(correction / bound)
    return base - (1 - base) * torch.expm1(-integrated * torch.expm1(delta))


class Effective_singleton_observation(nn.Module):
    """A visible-only set encoder, PSD state and effective window response.

There is no operation-local free residual network that can bypass the shared
    state in the response. Signed effective Born contrasts describe singleton
    probabilities conditioned on herald presence (absence is opt-in); these are phenomenological
contrasts, not detector efficiencies or certified physical-state recovery.
The signal marginal has no direct Born term. Zero initialized response heads
reproduce B0 exactly; state learning becomes active as a contrast is learned.
"""

    def __init__(self, config: Effective_observation_config) -> None:
        super().__init__()
        config.validate()
        self.config = config
        modes = config.mode_count
        # Four relative cells, log total, and fixed Hermitian projector geometry.
        self.visible_encoder = nn.Sequential(nn.Linear(5 + modes**2, config.width), nn.GELU(), nn.LayerNorm(config.width))
        self.attention = nn.ModuleList([
            nn.MultiheadAttention(config.width, config.heads, dropout=0, batch_first=True)
            for _ in range(config.blocks)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(config.width) for _ in range(config.blocks)])
        self.command_encoder = nn.Linear(2 * modes**2, config.width, bias=False)
        self.state_head = nn.Linear(config.width, 2 * modes**2)
        self.response_head = nn.Linear(config.width, 6)
        self.global_multi_head = nn.Sequential(nn.Linear(config.width, config.width), nn.GELU(), nn.Linear(config.width, 12))
        for head in (self.state_head, self.response_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        if config.global_initialization is not None:
            for name, value in _global_initialization_tensors(config.global_initialization).items():
                self.register_buffer(name, value.to(self.state_head.weight.device))
            nn.init.zeros_(self.global_multi_head[-1].weight)
            nn.init.zeros_(self.global_multi_head[-1].bias)
        # Construct after every V2 random module.  Its zero parameter consumes
        # no RNG, preserving same-seed M0 values for the fixed-pair backbone.
        self.family_context = (
            None if config.measurement_context is None
            else Effective_family_context(
                config.measurement_context, prototype=self.response_head.weight,
            )
        )

    def _global_prediction(self, raw: torch.Tensor) -> dict[str, torch.Tensor]:
        initialization = self.config.global_initialization
        if initialization is None:
            return {
                "multi_means": torch.nn.functional.softplus(raw[:, :3]) + 1e-6,
                "pair_logits": raw[:, 3:9],
                "multi_dispersion": torch.nn.functional.softplus(raw[:, 9:12]) + 1e-6,
            }
        bound = initialization.recipe.maximum_log_residual
        residual = bound * torch.tanh(raw.double() / bound)
        return {
            "multi_means": self.global_mean_baseline * residual[:, :3].exp(),
            "pair_logits": (self.global_pair_log_baseline + residual[:, 3:9]
                            if self.config.pair_residual_enabled else self.global_pair_log_baseline.expand(raw.shape[0], -1)),
            "multi_dispersion": self.global_dispersion_baseline * residual[:, 9:12].exp(),
        }

    def forward(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        validate_measurement_context_binding(self)
        if batch.get("system_type") != self.config.system_type \
                or batch.get("prepared_command_semantics") != PREPARED_SEMANTICS:
            raise ValueError("effective observation requires explicit system and prepared semantics")
        counts = batch["four_cell_counts"]
        available = batch["operation_mask"]
        query = batch["query_mask"]
        if counts.ndim != 3 or counts.shape[-1] != 4 or counts.dtype != torch.int64 \
                or available.dtype != torch.bool or query.dtype != torch.bool \
                or available.shape != counts.shape[:2] or query.shape != available.shape:
            raise ValueError("effective observation requires native int64 cells and bool masks")
        if bool((query & ~available).any()) or bool((counts[available] < 0).any()):
            raise ValueError("query/target support is invalid")
        visible = available & ~query
        if not bool(visible.any(-1).all()) or not bool(query.any(-1).all()):
            raise ValueError("each row must have visible and query operations")
        family_prediction: dict[str, Any] = {}
        if self.family_context is not None:
            family_prediction = self.family_context(
                batch, operation_mask=available, visible_operation_mask=visible,
            )
        # Select, never multiply untrusted hidden values into encoder features.
        visible_counts = torch.where(visible[..., None], counts, torch.zeros_like(counts))
        if "visible_four_cell_counts" in batch and not torch.equal(batch["visible_four_cell_counts"], visible_counts):
            raise ValueError("visible counts do not match the native query mask")
        projector = batch["projector_descriptor"].to(torch.float32)
        if projector.shape != (*counts.shape[:2], self.config.projector_dim) or not bool(torch.isfinite(projector).all()):
            raise ValueError("projector descriptor dimensions or finiteness are invalid")
        # This parses only registered geometry; RF inputs never alter the effect.
        projector_matrix = _fixed_projector_matrix(projector, self.config.mode_count)
        prior, command_available = _prepared_prior(batch, self.config)
        if prior.shape[0] != counts.shape[0]:
            raise ValueError("prepared command batch does not align")
        totals = visible_counts.to(torch.float32).sum(-1, keepdim=True)
        relative = visible_counts.to(torch.float32) / totals.clamp_min(1)
        geometry = projector[..., :self.config.mode_count**2]
        if not self.config.structure_enabled:
            geometry = torch.zeros_like(geometry)
        hidden = self.visible_encoder(torch.cat((relative, torch.log1p(totals), geometry), -1))
        for attention, norm in zip(self.attention, self.norms):
            context, _ = attention(hidden, hidden, hidden, key_padding_mask=~visible, need_weights=False)
            hidden = norm(hidden + context)
        pooled = (hidden * visible[..., None]).sum(1) / visible.sum(1, keepdim=True)
        if self.config.structure_enabled:
            prior_features = torch.cat((prior.real.flatten(1), prior.imag.flatten(1)), -1)
            pooled = pooled + self.command_encoder(prior_features)
        modes = self.config.mode_count
        state_raw = self.state_head(pooled).reshape(-1, 2, modes, modes)
        state_delta = torch.complex(state_raw[:, 0], state_raw[:, 1])
        if not self.config.structure_enabled:
            state_delta = state_delta * 0
        factor = torch.linalg.cholesky(prior) + state_delta
        gram = factor @ factor.mH
        trace = gram.diagonal(dim1=-2, dim2=-1).sum(-1).real
        if not bool(torch.isfinite(trace).all()) or bool((trace <= 0).any()):
            raise RuntimeError("latent PSD state has invalid normalization")
        state = gram / trace[:, None, None]
        born = torch.einsum("boij,bji->bo", projector_matrix, state).real
        centered_born = born - (born * visible).sum(-1, keepdim=True) / visible.sum(-1, keepdim=True)
        response = self.response_head(pooled)
        offsets = self.config.maximum_response_correction * torch.tanh(response[:, :3])
        contrasts = self.config.maximum_response_correction * torch.tanh(response[:, 3:5])
        if not self.config.structure_enabled:
            contrasts = contrasts * 0
        elif not self.config.no_signal_structure_enabled:
            contrasts = contrasts * contrasts.new_tensor([1, 0])
        corrections = offsets[:, None, :].expand(-1, counts.shape[1], -1).clone()
        corrections[..., 1:] = corrections[..., 1:] + centered_born[..., None] * contrasts[:, None, :]
        if family_prediction:
            corrections = corrections + family_prediction["effective_family_correction"]
            # Compose bounded signal tilts without changing V2 at zero context.
            # Singleton branches retain their existing probability-dependent H bound.
            bound = self.config.maximum_response_correction
            original_signal = offsets[:, None, 0].expand_as(corrections[..., 0])
            family_tilt = torch.tanh(family_prediction["effective_family_correction"][..., 0] / bound)
            corrections[..., 0] = original_signal + (
                (bound - original_signal.square() / bound) * family_tilt
                / (1 + original_signal * family_tilt / bound)
            )
        baseline = build_trackb_fixed_visible_marginal_prediction(batch)
        names = ("signal_probability", "coin_given_signal_probability", "idler_given_no_signal_probability")
        base_probability = torch.stack([baseline[name] for name in names], -1)
        # Algebraic log-odds tilt preserves the float B0 exactly at zero update.
        # Work in float64 to retain rare-event support after bounded tilts.
        base = base_probability.to(torch.float64)
        correction = corrections.to(torch.float64)
        multiplier = torch.exp(correction[..., 0])
        signal = base[..., 0] * multiplier / (1 + base[..., 0] * (multiplier - 1))
        singleton = _integrated_singleton_response(base[..., 1:], correction[..., 1:], self.config.finite_probability_floor)
        coin, idler = singleton.unbind(-1)
        if not bool(torch.isfinite(singleton).all()) or bool(((singleton <= 0) | (singleton >= 1)).any()):
            raise RuntimeError("effective singleton response left the finite native NLL domain")
        probability = torch.stack(((1 - signal) * (1 - idler), signal * (1 - coin), (1 - signal) * idler, signal * coin), -1)
        # Total is an estimate from visible opportunities, NOT a new read-count
        # observation. Native NLL conditions on true R only in the loss layer.
        estimated_trials = baseline["four_cell_mean"].sum(-1).to(torch.float64)
        log_concentration = self.config.maximum_log_concentration_correction * torch.tanh(response[:, 5])
        concentration = baseline["four_cell_concentration"].to(torch.float64) * log_concentration.to(torch.float64).exp()
        estimated_mean = probability * estimated_trials[..., None]
        count_output = {
            "four_cell_mean": estimated_mean,
            "read_trials_semantics": "visible_operation_average_not_native_query_trials",
        }
        if self.config.read_trials_source == "native_scalar":
            from .native_read_trials import native_read_trials
            from .detected_count_moments import nested_four_cell_moments

            native_trials = native_read_trials(batch)
            count_output = {
                **nested_four_cell_moments(
                    read_trials=native_trials, signal_probability=signal,
                    coin_given_signal_probability=coin, idler_given_no_signal_probability=idler,
                    concentration=concentration[:, None],
                ),
                "read_trials_semantics": "native_v240_scalar_field_not_reconstructed_from_query",
                "native_read_trials": native_trials,
            }
        global_raw = self.global_multi_head(pooled)
        return {
            "model_schema": (
                FAMILY_CONTEXT_MODEL_SCHEMA if self.family_context is not None else MODEL_SCHEMA
            ),
            "signal_probability": signal, "coin_given_signal_probability": coin,
            "idler_given_no_signal_probability": idler,
            "four_cell_probability": probability, **count_output,
            "four_cell_concentration": concentration,
            "estimated_read_trials": estimated_trials,
            "estimated_read_four_cell_mean": estimated_mean,
            "response_scope": "CONDITIONAL_ON_NATIVE_EFFECTIVE_READ",
            "idler_event": "STICKY_OR_OF_EXACT_SINGLETON_OVER_19_ACCEPTED_CLOCKS",
            "parameter_interpretation": "EFFECTIVE_RESPONSE_NOT_MICROSCOPIC_CALIBRATION",
            "latent_state": state, "prepared_prior": prior, "prepared_prior_available": command_available,
            "effective_offsets": offsets, "effective_contrasts": contrasts,
            "log_concentration_correction": log_concentration,
            **family_prediction,
            **self._global_prediction(global_raw),
            "visible_operation_mask": visible, "query_mask": query,
        }
