"""Differentiable command -> AOD -> Born/count -> inverse-control path.

The model is opt-in and research-only.  It shares one write-state latent across
all 4Q read cycles, uses one shared R-AOD kernel, derives count Jacobians and
Fisher matrices from the same forward map, and contains no free Jacobian head.
All complex algebra is represented by final-axis real/imag pairs.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .command_contract import COMMAND_CHANNELS, X_CHANNELS, PhysicsHybridCommandTableV1, command_layout
from .command_features import (
    SharedMultiToneAODResponse,
    ToneSetCommandEncoder,
    command_tone_features,
    complex_abs_square_pair,
    complex_conjugate_pair,
    complex_multiply_pair,
)
from .command_normalization import (
    apply_command_table_normalization,
    validate_command_normalization_contract,
)
from .detector_click_model import FourDetectorClickModel, global_click_means
from .projective_action import (
    ProjectiveAction,
    action_covariance_from_lower_triangular,
    action_dimensions,
    apply_projective_action,
    project_action,
    qubit_count,
)
from .source_multiplicity import SourceMultiplicityModel
from .write_read_forward import BornDetectorForward, ReadAODForward, WriteAODForward

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None
    nn = None
    F = None


COUNT_FORWARD_SCHEMA = "action_conditioned_born_count_forward"
COUNT_FORWARD_VERSION = 1
COUNT_CHANNELS: Tuple[str, ...] = ("signal", "idler", "coin")


@dataclass(frozen=True)
class CommandTensorBatch:
    tone_features: Any
    absolute_frequency_mhz: Any
    log_amplitude: Any
    phase_sin: Any
    phase_cos: Any
    per_channel_power: Any
    basis_amplitude: Any
    basis_phase_sin: Any
    basis_phase_cos: Any
    command_missing_mask: Any
    channel_flags: Any
    path_flags: Any
    cycle_index: Any
    tone_missing_mask: Any
    per_channel_command_missing_mask: Any
    x_log_amplitude_all: Any
    x_phase_sin_all: Any
    x_phase_cos_all: Any
    y_amplitude_all: Any
    y_phase_sin_all: Any
    y_phase_cos_all: Any
    per_channel_power_all: Any
    per_channel_missing_all: Any
    diffraction_sign_all: Any
    timing_all: Any
    table_id: str
    system_type: str
    normalization_sha256: Optional[str] = None
    command_feature_order: Tuple[str, ...] = ()
    command_schema_hash: str = ""

    @property
    def batch_size(self) -> int:
        return int(self.tone_features.shape[0])


def command_tensor_batch(
    table: PhysicsHybridCommandTableV1,
    *,
    batch_size: int = 1,
    device: Any = None,
    dtype: Any = None,
    read_channel: str = "R_CH1",
    normalization_contract: Optional[Mapping[str, Any]] = None,
    expected_normalization_sha256: Optional[str] = None,
) -> CommandTensorBatch:
    if torch is None:
        raise RuntimeError("command_tensor_batch requires torch")
    features, feature_schema = command_tone_features(table)
    feature_order = tuple(feature_schema.ordered_names)
    normalization_sha256 = None
    if normalization_contract is not None:
        features = apply_command_table_normalization(
            features,
            feature_order,
            normalization_contract,
            expected_normalization_sha256=expected_normalization_sha256,
        )
        normalization_sha256 = validate_command_normalization_contract(
            normalization_contract,
            feature_order,
            expected_normalization_sha256=expected_normalization_sha256,
        )
    elif expected_normalization_sha256 is not None:
        raise ValueError("strict command inference requires a normalization contract")
    dtype = dtype or torch.float32
    channel_index = COMMAND_CHANNELS.index(read_channel)

    def tensor(value: Any) -> "torch.Tensor":
        base = torch.as_tensor(np.array(value, copy=True), dtype=dtype, device=device)
        return base.unsqueeze(0).expand((int(batch_size),) + tuple(base.shape))

    static = table.static_arrays
    masks = table.masks
    cycles = table.command_shape[0]
    channel_flags = np.zeros(table.command_shape + (len(COMMAND_CHANNELS),), dtype=np.float32)
    channel_flags[..., channel_index] = 1.0
    path_flags = np.zeros(table.command_shape + (2,), dtype=np.float32)
    path_flags[..., 0 if read_channel in X_CHANNELS else 1] = 1.0
    per_channel_missing = np.asarray(masks["command_missing_mask"], dtype=float)
    tone_missing = np.asarray(masks.get("tone_missing_mask"), dtype=float)
    tone_shape = table.command_shape

    def broadcast_channel(value: Any) -> np.ndarray:
        array = np.asarray(value, dtype=np.float64)
        if array.shape == tone_shape + (len(COMMAND_CHANNELS),):
            return array
        expected = tone_shape[:-1] + (len(COMMAND_CHANNELS),)
        if array.shape != expected:
            raise ValueError(f"channel command array has shape {array.shape}, expected {expected}")
        return np.broadcast_to(array[..., None, :], tone_shape + (len(COMMAND_CHANNELS),))

    timing = np.stack(
        [
            np.asarray(static["digital_delay_ns"], dtype=np.float64),
            np.asarray(static["digital_width_ns"], dtype=np.float64),
            np.asarray(static["analog_delay_ns"], dtype=np.float64),
        ],
        axis=-1,
    )
    timing = np.broadcast_to(
        timing[..., None, :, :],
        tone_shape + (len(COMMAND_CHANNELS), 3),
    )
    return CommandTensorBatch(
        tone_features=tensor(features),
        absolute_frequency_mhz=tensor(static["absolute_frequency_mhz"]),
        log_amplitude=tensor(np.asarray(static["x_log_amplitude"])[..., channel_index]),
        phase_sin=tensor(np.asarray(static["x_phase_sin"])[..., channel_index]),
        phase_cos=tensor(np.asarray(static["x_phase_cos"])[..., channel_index]),
        per_channel_power=tensor(np.asarray(static["per_channel_rf_power"])[..., channel_index]),
        basis_amplitude=tensor(static["basis_amplitude"]),
        basis_phase_sin=tensor(static["basis_phase_sin"]),
        basis_phase_cos=tensor(static["basis_phase_cos"]),
        command_missing_mask=tensor(np.any(per_channel_missing > 0.5, axis=-1).astype(float)),
        channel_flags=tensor(channel_flags),
        path_flags=tensor(path_flags),
        cycle_index=torch.arange(cycles, dtype=torch.long, device=device),
        tone_missing_mask=tensor(tone_missing),
        per_channel_command_missing_mask=tensor(per_channel_missing),
        x_log_amplitude_all=tensor(static["x_log_amplitude"]),
        x_phase_sin_all=tensor(static["x_phase_sin"]),
        x_phase_cos_all=tensor(static["x_phase_cos"]),
        y_amplitude_all=tensor(broadcast_channel(static["y_amplitude"])),
        y_phase_sin_all=tensor(broadcast_channel(static["y_phase_sin"])),
        y_phase_cos_all=tensor(broadcast_channel(static["y_phase_cos"])),
        per_channel_power_all=tensor(broadcast_channel(static["per_channel_rf_power"])),
        per_channel_missing_all=tensor(per_channel_missing),
        diffraction_sign_all=tensor(broadcast_channel(static["diffraction_order_sign"])),
        timing_all=tensor(timing),
        table_id=table.table_id,
        system_type=table.system_type,
        normalization_sha256=normalization_sha256,
        command_feature_order=feature_order,
        command_schema_hash=table.feature_schema_hash,
    )


@dataclass(frozen=True)
class EncodedCommandContext:
    tone_latent: Any
    operation_latent: Any
    mode_mean: Any
    mode_rms: Any
    global_mean: Any
    global_rms: Any
    cycle_mean: Any
    cycle_rms: Any
    policy_summary: Any
    valid_tone_mask: Any
    valid_tone_fraction: Any
    per_channel_missing_fraction: Any
    table_id: str
    system_type: str


@dataclass(frozen=True)
class ForwardContext:
    encoded_command_context: EncodedCommandContext
    observation_latent: Any
    current_state_context: Mapping[str, Any]
    local_action_residuals: Mapping[str, Any]
    brightness_delta: Any
    detector_time_block_index: Optional[Any]
    detector_parameters_snapshot: Optional[Mapping[str, Any]]
    source_parameters_snapshot: Optional[Mapping[str, Any]]
    command_table_id: str
    model_context_hash: str


def _observation_tensor(
    observation: Mapping[str, Any],
    *,
    batch_size: int,
    device: Any,
    dtype: Any,
    held_out_cycle: Optional[int] = None,
) -> "torch.Tensor":
    if observation.get("cycle_observations") is not None:
        cycle = torch.as_tensor(observation["cycle_observations"], dtype=dtype, device=device)
        if cycle.ndim == 2:
            cycle = cycle.unsqueeze(0)
        if cycle.shape[0] != batch_size:
            raise ValueError("cycle_observations batch size mismatch")
        if held_out_cycle is None:
            return cycle.mean(dim=1)
        if not 0 <= int(held_out_cycle) < cycle.shape[1]:
            raise ValueError("held_out_cycle is out of range")
        keep = torch.ones(cycle.shape[1], dtype=torch.bool, device=device)
        keep[int(held_out_cycle)] = False
        return cycle[:, keep].mean(dim=1)
    for key in ("rho_features", "observation_features"):
        if observation.get(key) is not None:
            value = torch.as_tensor(observation[key], dtype=dtype, device=device)
            return value.unsqueeze(0) if value.ndim == 1 else value
    return torch.zeros((batch_size, 1), dtype=dtype, device=device)


def _pattern_probabilities_from_detector_clicks(click_probability: "torch.Tensor") -> "torch.Tensor":
    patterns = torch.as_tensor(
        [[float((index >> bit) & 1) for bit in range(4)] for index in range(16)],
        dtype=click_probability.dtype, device=click_probability.device,
    )
    probability = click_probability.unsqueeze(-2)
    shape = (1,) * (probability.ndim - 2) + (16, 4)
    pattern = patterns.reshape(shape)
    return torch.where(pattern > 0.5, probability, 1.0 - probability).prod(dim=-1)


if nn is not None:

    class DifferentiableProjectiveCountForward(nn.Module):
        """One integrable action-conditioned forward map for counts/J/Fisher."""

        count_jacobian_source = "autograd_same_forward_map"
        independent_count_jacobian_head = False

        def __init__(
            self,
            system_type: str,
            command_feature_dim: int,
            *,
            hidden_dim: int = 64,
            detector_model: Optional[FourDetectorClickModel] = None,
            source_model: Optional[SourceMultiplicityModel] = None,
            state_source: str = "write_command_forward",
            observation_state_residual_limit: float = 0.02,
        ) -> None:
            super().__init__()
            self.system_type = str(system_type)
            self.cell_count = qubit_count(system_type)
            self.action_dim = action_dimensions(system_type)["total"]
            if state_source not in {"write_command_forward", "observation_legacy"}:
                raise ValueError("state_source must be write_command_forward or observation_legacy")
            self.state_source = str(state_source)
            self.observation_state_residual_limit = float(observation_state_residual_limit)
            self.command_encoder = ToneSetCommandEncoder(command_feature_dim, hidden_dim, hidden_dim)
            self.shared_read_aod_kernel = SharedMultiToneAODResponse(hidden_dim=max(24, hidden_dim // 2), max_cycles=4)
            self.observation_encoder = nn.Sequential(
                nn.LazyLinear(hidden_dim), nn.SiLU(), nn.LayerNorm(hidden_dim),
            )
            self.state_pair_head = nn.Linear(hidden_dim, self.cell_count * 2)
            self.observation_state_residual_head = nn.Linear(hidden_dim, self.cell_count * 2)
            nn.init.zeros_(self.observation_state_residual_head.weight)
            nn.init.zeros_(self.observation_state_residual_head.bias)
            self.log_source_rate = nn.Parameter(torch.zeros(3))
            self.efficiency_logit = nn.Parameter(torch.zeros(3))
            self.retrieval_logit = nn.Parameter(torch.zeros(3))
            self.background_log_rate = nn.Parameter(torch.full((3,), -9.0))
            self.rate_residual_head = nn.Linear(hidden_dim, 3)
            nn.init.zeros_(self.rate_residual_head.weight)
            nn.init.zeros_(self.rate_residual_head.bias)
            self.maximum_rate_residual_fraction = 0.05
            self.source_strength_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.SiLU(), nn.Linear(hidden_dim // 2, 1))
            self.detector_model = detector_model
            self.source_model = source_model or SourceMultiplicityModel()
            self.write_aod_forward = WriteAODForward(hidden_dim=max(16, hidden_dim // 2))
            self.read_aod_forward = ReadAODForward(hidden_dim=max(16, hidden_dim // 2))
            self.born_detector_forward = BornDetectorForward(
                source_model=self.source_model,
                detector_model=detector_model,
            )

        def _shared_state(
            self,
            observation: Mapping[str, Any],
            command: CommandTensorBatch,
            *,
            held_out_cycle: Optional[int] = None,
        ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
            batch = command.batch_size
            if observation.get("state_amplitude") is not None and observation.get("state_phase_rad") is not None:
                amplitude = torch.as_tensor(observation["state_amplitude"], dtype=command.tone_features.dtype, device=command.tone_features.device)
                phase = torch.as_tensor(observation["state_phase_rad"], dtype=command.tone_features.dtype, device=command.tone_features.device)
                if amplitude.ndim == 1:
                    amplitude, phase = amplitude.unsqueeze(0), phase.unsqueeze(0)
                amplitude = amplitude / torch.linalg.vector_norm(amplitude, dim=-1, keepdim=True).clamp_min(1e-12)
                latent_input = _observation_tensor(observation, batch_size=batch, device=amplitude.device, dtype=amplitude.dtype, held_out_cycle=held_out_cycle)
                latent = self.observation_encoder(latent_input)
                return amplitude, phase, latent
            observation_tensor = _observation_tensor(
                observation, batch_size=batch, device=command.tone_features.device,
                dtype=command.tone_features.dtype, held_out_cycle=held_out_cycle,
            )
            latent = self.observation_encoder(observation_tensor)
            pair = self.state_pair_head(latent).reshape(batch, self.cell_count, 2)
            norm = torch.sqrt(complex_abs_square_pair(pair).sum(dim=-1, keepdim=True)).clamp_min(1e-12)
            pair = pair / norm.unsqueeze(-1)
            amplitude = torch.sqrt(complex_abs_square_pair(pair).clamp_min(1e-12))
            phase = torch.atan2(pair[..., 1], pair[..., 0])
            return amplitude, phase, latent

        def encode_command_context(self, command: CommandTensorBatch) -> EncodedCommandContext:
            """Encode the static command once with masked hierarchical pooling."""

            tone_latent = self.command_encoder(
                command.tone_features,
                command.absolute_frequency_mhz,
                command.log_amplitude,
                command.phase_sin,
                command.phase_cos,
                channel_flags=command.channel_flags,
                path_flags=command.path_flags,
            )
            # A tone is missing only when the command contract says that the
            # tone itself is missing.  One absent auxiliary channel must not
            # invalidate an otherwise complete physical tone.
            valid_tone_mask = ~(command.tone_missing_mask > 0.5)
            weight = valid_tone_mask.to(dtype=tone_latent.dtype).unsqueeze(-1)

            def masked_moments(dimensions: Tuple[int, ...]) -> Tuple["torch.Tensor", "torch.Tensor"]:
                denominator = weight.sum(dim=dimensions).clamp_min(1.0)
                mean = (tone_latent * weight).sum(dim=dimensions) / denominator
                second = (tone_latent.square() * weight).sum(dim=dimensions) / denominator
                return mean, torch.sqrt(second.clamp_min(0.0) + 1e-12)

            operation_mean, _ = masked_moments((-2,))
            mode_mean, mode_rms = masked_moments((1, 2, 3))
            global_mean, global_rms = masked_moments((1, 2, 3, 4))
            cycle_operation_mean, _ = masked_moments((2, 3, 4))
            cycle_mean = cycle_operation_mean.mean(dim=1)
            cycle_rms = torch.sqrt(cycle_operation_mean.square().mean(dim=1).clamp_min(0.0) + 1e-12)
            valid_fraction = valid_tone_mask.to(tone_latent.dtype).mean(dim=(1, 2, 3, 4))
            channel_missing_fraction = command.per_channel_command_missing_mask.to(tone_latent.dtype).mean(
                dim=(1, 2, 3, 4)
            )
            policy_summary = torch.cat((
                mode_mean.flatten(start_dim=1),
                mode_rms.flatten(start_dim=1),
                global_mean,
                global_rms,
                cycle_mean,
                cycle_rms,
                valid_fraction.unsqueeze(-1),
                channel_missing_fraction,
            ), dim=-1)
            return EncodedCommandContext(
                tone_latent=tone_latent,
                operation_latent=operation_mean,
                mode_mean=mode_mean,
                mode_rms=mode_rms,
                global_mean=global_mean,
                global_rms=global_rms,
                cycle_mean=cycle_mean,
                cycle_rms=cycle_rms,
                policy_summary=policy_summary,
                valid_tone_mask=valid_tone_mask,
                valid_tone_fraction=valid_fraction,
                per_channel_missing_fraction=channel_missing_fraction,
                table_id=command.table_id,
                system_type=command.system_type,
            )

        def build_forward_context(
            self,
            observation: Mapping[str, Any],
            command: CommandTensorBatch,
            *,
            encoded_command_context: Optional[EncodedCommandContext] = None,
            observation_latent: Optional["torch.Tensor"] = None,
            local_action_residuals: Optional[Mapping[str, Any]] = None,
            amplitude_limit: float = 0.03,
            phase_limit_rad: float = 0.20,
            brightness_delta: Optional[Any] = None,
            brightness_limit: float = 0.0,
            held_out_cycle: Optional[int] = None,
        ) -> ForwardContext:
            """Freeze command, observation, residual, source and detector context."""

            if command.system_type != self.system_type:
                raise ValueError("command/model system_type mismatch")
            encoded = encoded_command_context or self.encode_command_context(command)
            if observation_latent is None:
                observation_tensor = _observation_tensor(
                    observation,
                    batch_size=command.batch_size,
                    device=command.tone_features.device,
                    dtype=command.tone_features.dtype,
                    held_out_cycle=held_out_cycle,
                )
                observation_latent = self.observation_encoder(observation_tensor)

            if self.state_source == "observation_legacy":
                if observation.get("state_amplitude") is not None and observation.get("state_phase_rad") is not None:
                    amplitude = torch.as_tensor(
                        observation["state_amplitude"],
                        dtype=command.tone_features.dtype,
                        device=command.tone_features.device,
                    )
                    phase = torch.as_tensor(
                        observation["state_phase_rad"],
                        dtype=command.tone_features.dtype,
                        device=command.tone_features.device,
                    )
                    if amplitude.ndim == 1:
                        amplitude, phase = amplitude.unsqueeze(0), phase.unsqueeze(0)
                    amplitude = amplitude / torch.linalg.vector_norm(amplitude, dim=-1, keepdim=True).clamp_min(1e-12)
                    state_pair = torch.stack((amplitude * torch.cos(phase), amplitude * torch.sin(phase)), dim=-1)
                else:
                    state_pair = self.state_pair_head(observation_latent).reshape(command.batch_size, self.cell_count, 2)
                    state_pair = state_pair / torch.sqrt(
                        complex_abs_square_pair(state_pair).sum(dim=-1, keepdim=True)
                    ).clamp_min(1e-12).unsqueeze(-1)
                write_outputs: Dict[str, Any] = {
                    "prepared_state_real_imag": state_pair,
                    "write_state_source": "observation_legacy",
                    "write_cycle_deviation_norm": state_pair.sum() * 0.0,
                    "write_calibrated_path_residual_norm": state_pair.sum() * 0.0,
                }
            else:
                write_outputs = dict(self.write_aod_forward(
                    command, shared_aod_kernel=self.shared_read_aod_kernel
                ))
                state_pair = write_outputs["prepared_state_real_imag"]
                residual = self.observation_state_residual_limit * torch.tanh(
                    self.observation_state_residual_head(observation_latent)
                ).reshape(command.batch_size, self.cell_count, 2)
                state_pair = state_pair + residual
                state_pair = state_pair / torch.sqrt(
                    complex_abs_square_pair(state_pair).sum(dim=-1, keepdim=True)
                ).clamp_min(1e-12).unsqueeze(-1)
                write_outputs["prepared_state_real_imag"] = state_pair
                write_outputs["posterior_state_residual_norm"] = torch.linalg.vector_norm(residual)
                write_outputs["write_state_source"] = "write_command_forward"
            prepared_amplitude = torch.sqrt(complex_abs_square_pair(state_pair).clamp_min(1e-12))
            prepared_phase = torch.atan2(state_pair[..., 1], state_pair[..., 0])
            write_outputs["prepared_amplitude"] = prepared_amplitude
            write_outputs["prepared_phase_rad"] = prepared_phase
            read_outputs = dict(self.read_aod_forward(
                command, shared_aod_kernel=self.shared_read_aod_kernel
            ))

            local_residual = self._local_residual_grid(
                command,
                local_action_residuals,
                amplitude_limit=amplitude_limit,
                phase_limit_rad=phase_limit_rad,
            )
            if brightness_delta is None:
                brightness = torch.zeros(
                    command.batch_size,
                    dtype=command.tone_features.dtype,
                    device=command.tone_features.device,
                )
            else:
                brightness = torch.as_tensor(
                    brightness_delta,
                    dtype=command.tone_features.dtype,
                    device=command.tone_features.device,
                ).reshape(command.batch_size)
                if float(brightness_limit) <= 0.0 and bool(torch.any(brightness != 0).detach().cpu()):
                    raise ValueError("brightness is disabled; nonzero brightness fails closed")
                if float(brightness_limit) > 0.0 and bool(
                    torch.any(brightness.abs() > float(brightness_limit) + 1e-8).detach().cpu()
                ):
                    raise ValueError("brightness exceeds configured bound")

            time_block = observation.get("detector_time_block_index")
            time_block_tensor = None if time_block is None else torch.as_tensor(
                time_block,
                dtype=torch.long,
                device=command.tone_features.device,
            )
            detector_snapshot = None
            if self.detector_model is not None:
                dark, efficiency = self.detector_model.probabilities(time_block_tensor)
                detector_snapshot = {
                    "dark_probability": dark,
                    "detector_efficiency": efficiency,
                }
            source_snapshot = {
                "mixture_weights": self.source_model.mixture_weights(time_block_tensor),
            }
            current_state_context = {
                **write_outputs,
                **read_outputs,
            }
            context_identity = "|".join((
                command.table_id,
                command.normalization_sha256 or "unnormalized",
                self.state_source,
                str(held_out_cycle),
                str(time_block),
            ))
            return ForwardContext(
                encoded_command_context=encoded,
                observation_latent=observation_latent,
                current_state_context=current_state_context,
                local_action_residuals=local_residual,
                brightness_delta=brightness,
                detector_time_block_index=time_block_tensor,
                detector_parameters_snapshot=detector_snapshot,
                source_parameters_snapshot=source_snapshot,
                command_table_id=command.table_id,
                model_context_hash=hashlib.sha256(context_identity.encode("utf-8")).hexdigest(),
            )

        def _action_state(
            self,
            base_amplitude: "torch.Tensor",
            base_phase: "torch.Tensor",
            action_coordinates: "torch.Tensor",
        ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
            dimensions = action_dimensions(self.system_type)
            if action_coordinates.shape[-1] != dimensions["total"]:
                raise ValueError(f"action_coordinates must have dimension {dimensions['total']}")
            split = dimensions["amplitude"]
            action = ProjectiveAction(
                system_type=self.system_type,
                amplitude_coordinates=action_coordinates[..., :split],
                phase_coordinates=action_coordinates[..., split:],
            )
            amplitude, phase, _ = apply_projective_action(base_amplitude, base_phase, action)
            state = torch.stack((amplitude * torch.cos(phase), amplitude * torch.sin(phase)), dim=-1)
            return state, amplitude, phase

        def _local_residual_grid(
            self,
            command: CommandTensorBatch,
            local_action_residuals: Optional[Mapping[str, Any]],
            *,
            amplitude_limit: float,
            phase_limit_rad: float,
        ) -> Dict[str, "torch.Tensor"]:
            """Materialize shared-spatial/cycle terms without mixing them with global action."""
            grid_shape = tuple(command.basis_amplitude.shape)
            batch, cycles, rows, groups, cells = grid_shape
            if cells != self.cell_count:
                raise ValueError("local residual cell dimension does not match system_type")
            zero = torch.zeros(
                grid_shape, dtype=command.basis_amplitude.dtype,
                device=command.basis_amplitude.device,
            )
            values = dict(local_action_residuals or {})

            def expand(name: str, *, shared: bool) -> "torch.Tensor":
                value = values.get(name)
                if value is None:
                    return zero
                tensor = torch.as_tensor(value, dtype=zero.dtype, device=zero.device)
                if tensor.ndim == 3 and tuple(tensor.shape) == (rows, groups, cells):
                    tensor = tensor.reshape(1, 1, rows, groups, cells)
                elif shared and tensor.ndim == 4 and tuple(tensor.shape[1:]) == (rows, groups, cells):
                    if tensor.shape[0] != batch:
                        raise ValueError(f"{name} batch dimension mismatch")
                    tensor = tensor.unsqueeze(1)
                elif not shared and tensor.ndim == 4 and tuple(tensor.shape[1:]) == (rows, groups, cells):
                    if tensor.shape[0] != cycles:
                        raise ValueError(f"{name} cycle dimension mismatch")
                    tensor = tensor.unsqueeze(0)
                elif tensor.ndim != 5:
                    raise ValueError(
                        f"{name} must be [R,G,N], [B,R,G,N], [C,R,G,N], or [B,C,R,G,N]"
                    )
                if tuple(tensor.shape[-3:]) != (rows, groups, cells):
                    raise ValueError(f"{name} registry layout mismatch")
                if tensor.shape[0] not in (1, batch) or tensor.shape[1] not in (1, cycles):
                    raise ValueError(f"{name} batch/cycle dimensions cannot broadcast")
                return tensor.expand(grid_shape)

            shared_amp_raw = expand("shared_spatial_amplitude_raw", shared=True)
            cycle_amp_raw = expand("cycle_amplitude_raw", shared=False)
            shared_phase_raw = expand("shared_spatial_phase_raw", shared=True)
            cycle_phase_raw = expand("cycle_phase_raw", shared=False)
            amp_bound = float(np.log1p(amplitude_limit))
            shared_amp = amp_bound * torch.tanh(shared_amp_raw)
            cycle_amp = amp_bound * torch.tanh(cycle_amp_raw)
            shared_phase = float(phase_limit_rad) * torch.tanh(shared_phase_raw)
            cycle_phase = float(phase_limit_rad) * torch.tanh(cycle_phase_raw)
            local_phase = shared_phase + cycle_phase
            local_phase = local_phase - local_phase.mean(dim=-1, keepdim=True)
            return {
                "shared_spatial_amplitude_log_gain": shared_amp,
                "cycle_amplitude_log_gain": cycle_amp,
                "shared_spatial_phase_rad": shared_phase,
                "cycle_phase_rad": cycle_phase,
                "local_amplitude_log_gain": shared_amp + cycle_amp,
                "local_phase_rad": local_phase,
            }

        def forward_map(
            self,
            observation: Mapping[str, Any],
            command: CommandTensorBatch,
            action_coordinates: "torch.Tensor",
            *,
            forward_context: Optional[ForwardContext] = None,
            held_out_cycle: Optional[int] = None,
            local_action_residuals: Optional[Mapping[str, Any]] = None,
            amplitude_limit: float = 0.03,
            phase_limit_rad: float = 0.20,
            brightness_delta: Optional[Any] = None,
            brightness_limit: float = 0.0,
        ) -> Dict[str, "torch.Tensor"]:
            context = forward_context or self.build_forward_context(
                observation,
                command,
                local_action_residuals=local_action_residuals,
                amplitude_limit=amplitude_limit,
                phase_limit_rad=phase_limit_rad,
                brightness_delta=brightness_delta,
                brightness_limit=brightness_limit,
                held_out_cycle=held_out_cycle,
            )
            if context.command_table_id != command.table_id:
                raise ValueError("forward context belongs to a different command table")
            base_amplitude = context.current_state_context["prepared_amplitude"]
            base_phase = context.current_state_context["prepared_phase_rad"]
            state, corrected_amplitude, corrected_phase = self._action_state(
                base_amplitude, base_phase, action_coordinates,
            )
            local_residual = context.local_action_residuals
            operation_latent = context.encoded_command_context.operation_latent
            projector = context.current_state_context["projector_real_imag"]
            operation_amplitude = (
                corrected_amplitude[:, None, None, None, :]
                * torch.exp(local_residual["local_amplitude_log_gain"])
            )
            operation_amplitude = operation_amplitude / torch.linalg.vector_norm(
                operation_amplitude, dim=-1, keepdim=True,
            ).clamp_min(1e-12)
            operation_phase = corrected_phase[:, None, None, None, :] + local_residual["local_phase_rad"]
            operation_state = torch.stack((
                operation_amplitude * torch.cos(operation_phase),
                operation_amplitude * torch.sin(operation_phase),
            ), dim=-1)
            born_outputs = self.born_detector_forward(
                operation_state,
                projector,
                detector_time_block_index=context.detector_time_block_index,
                detector_parameters_snapshot=context.detector_parameters_snapshot,
                source_parameters_snapshot=context.source_parameters_snapshot,
            )
            born_probability = born_outputs["born_probability"]

            source_rate = torch.exp(self.log_source_rate).reshape(1, 1, 1, 1, 3)
            efficiency = torch.sigmoid(self.efficiency_logit).reshape(1, 1, 1, 1, 3)
            retrieval = torch.sigmoid(self.retrieval_logit).reshape(1, 1, 1, 1, 3)
            background = F.softplus(self.background_log_rate).reshape(1, 1, 1, 1, 3)
            physical_probability = torch.stack((
                torch.ones_like(born_probability),
                torch.ones_like(born_probability),
                born_probability,
            ), dim=-1)
            physics_rate = source_rate * efficiency * retrieval * physical_probability + background
            rate_residual = self.maximum_rate_residual_fraction * torch.tanh(
                self.rate_residual_head(operation_latent)
            )
            expected_rate_grid = physics_rate * torch.exp(rate_residual)
            brightness = context.brightness_delta
            brightness_scale = torch.exp(2.0 * brightness).reshape(command.batch_size, 1, 1, 1, 1)
            expected_rate_grid = expected_rate_grid * brightness_scale
            exposure = torch.as_tensor(observation.get("exposure_s", 1.0), dtype=expected_rate_grid.dtype, device=expected_rate_grid.device)
            while exposure.ndim < expected_rate_grid.ndim - 1:
                exposure = exposure.unsqueeze(-1)
            expected_counts_grid = expected_rate_grid * exposure.unsqueeze(-1)

            operation_count = int(np.prod(expected_rate_grid.shape[1:-1]))
            local_hardware_action = torch.cat((
                local_residual["local_amplitude_log_gain"], local_residual["local_phase_rad"],
            ), dim=-1).reshape(command.batch_size, operation_count, 2 * self.cell_count)
            local_action_norm = torch.linalg.vector_norm(local_hardware_action)
            global_action_norm = torch.linalg.vector_norm(action_coordinates)
            cycle_action_norm = torch.linalg.vector_norm(torch.cat((
                local_residual["cycle_amplitude_log_gain"], local_residual["cycle_phase_rad"],
            ), dim=-1))
            result = {
                "expected_rate": expected_rate_grid.reshape(command.batch_size, operation_count, 3),
                "expected_counts": expected_counts_grid.reshape(command.batch_size, operation_count, 3),
                "born_probability": born_probability.reshape(command.batch_size, operation_count),
                "true_click_pattern_probability": born_outputs["source_pattern_probability"].reshape(command.batch_size, operation_count, 16),
                "source_pattern_probability": born_outputs["source_pattern_probability"].reshape(command.batch_size, operation_count, 16),
                "observed_click_pattern_probability": born_outputs["observed_click_pattern_probability"].reshape(command.batch_size, operation_count, 16),
                "shared_state_real_imag": state,
                "shared_state_amplitude": corrected_amplitude,
                "shared_state_phase_rad": corrected_phase,
                "cycle_residual_norm": torch.sqrt(
                    torch.square(context.current_state_context["read_cycle_residual_norm"])
                    + torch.square(cycle_action_norm)
                ),
                "command_conditioning_mask": context.encoded_command_context.valid_tone_mask.reshape(
                    command.batch_size, operation_count, self.cell_count
                ).all(dim=-1),
                "global_projective_action": action_coordinates,
                "brightness_delta": brightness,
                "forward_context": context,
                "model_context_hash": context.model_context_hash,
                "prepared_state_source": context.current_state_context["write_state_source"],
                "projector_real_imag": projector,
                "local_amplitude_residual_norm": torch.linalg.vector_norm(local_residual["local_amplitude_log_gain"]),
                "local_phase_residual_norm": torch.linalg.vector_norm(local_residual["local_phase_rad"]),
                "local_hardware_action": local_hardware_action,
                "global_action_norm": global_action_norm,
                "cycle_action_residual_norm": cycle_action_norm,
                "global_versus_local_residual_norm": local_action_norm / (global_action_norm + local_action_norm).clamp_min(1e-8),
                "dark_identifiable": bool(observation.get("off_window_calibration_available", False)),
                "posterior_uncertainty_required": not bool(
                    observation.get("off_window_calibration_available", False)
                ),
                "true_multi_subtraction_available": False,
                "multi_click_amplitude_actionable": False,
                "source_detector_nuisance_scope": "campaign_detector_calibration_time_block",
                "rate_physics_components": {
                    "source_rate": source_rate,
                    "efficiency": efficiency,
                    "retrieval": retrieval,
                    "background": background,
                    "born_probability": born_probability,
                },
                "rate_residual": rate_residual,
                "rate_residual_energy": torch.mean(torch.square(rate_residual)),
                "maximum_rate_residual_fraction": self.maximum_rate_residual_fraction,
            }
            for name, value in born_outputs.items():
                if name not in {"born_probability", "source_pattern_probability", "observed_click_pattern_probability"}:
                    result[name] = value
            result.update(local_residual)
            if observation.get("gate_exposure") is not None:
                observed_pattern = result["observed_click_pattern_probability"]
                gate = torch.as_tensor(
                    observation["gate_exposure"],
                    dtype=observed_pattern.dtype,
                    device=observed_pattern.device,
                )
                if gate.ndim == 1:
                    gate = gate.unsqueeze(0).expand(command.batch_size, -1)
                result.update({f"global_{key}_mean": value for key, value in global_click_means(
                    observed_pattern, gate,
                ).items()})
            result.update({
                "aod_cycle_residual_norm": context.current_state_context["read_cycle_residual_norm"],
                "write_cycle_deviation_norm": context.current_state_context["write_cycle_deviation_norm"],
                "write_calibrated_path_residual_norm": context.current_state_context["write_calibrated_path_residual_norm"],
                "read_calibrated_path_residual_norm": context.current_state_context["read_calibrated_path_residual_norm"],
            })
            return result

        def predict_counts(
            self,
            observation: Mapping[str, Any],
            command: CommandTensorBatch,
            action_coordinates: "torch.Tensor",
            *,
            local_action_residuals: Optional[Mapping[str, Any]] = None,
            amplitude_limit: float = 0.03,
            phase_limit_rad: float = 0.20,
            brightness_delta: Optional[Any] = None,
            brightness_limit: float = 0.0,
        ) -> "torch.Tensor":
            return self.forward_map(
                observation, command, action_coordinates,
                local_action_residuals=local_action_residuals,
                amplitude_limit=amplitude_limit,
                phase_limit_rad=phase_limit_rad,
                brightness_delta=brightness_delta,
                brightness_limit=brightness_limit,
            )["expected_counts"]

        def leave_one_cycle_out_predict(
            self,
            observation: Mapping[str, Any],
            command: CommandTensorBatch,
            action_coordinates: "torch.Tensor",
            held_out_cycle: int,
        ) -> "torch.Tensor":
            if self.system_type != "4qubit":
                raise ValueError("leave-one-cycle-out is defined for 4qubit")
            result = self.forward_map(
                observation, command, action_coordinates, held_out_cycle=int(held_out_cycle),
            )["expected_counts"]
            return result.reshape(command.batch_size, 4, 64, 3)[:, int(held_out_cycle)]


    class OperationComplexFeedbackModel(nn.Module):
        """Shared trunk with joint action/UQ heads and the integrable count map."""

        independent_count_jacobian_head = False

        def __init__(
            self,
            system_type: str,
            command_feature_dim: int,
            *,
            hidden_dim: int = 64,
            brightness_enabled: bool = False,
            covariance_kind: str = "full_lower_triangular",
            detector_model: Optional[FourDetectorClickModel] = None,
            source_model: Optional[SourceMultiplicityModel] = None,
            policy_conditioning_enabled: bool = False,
            minimum_valid_tone_fraction: float = 0.99,
            missing_command_covariance_diagonal: float = 1.0,
            gate_initial_bias: float = -2.0,
            share_forward_command_encoder: bool = True,
            state_source: str = "write_command_forward",
        ) -> None:
            super().__init__()
            self.system_type = system_type
            self.dimensions = action_dimensions(system_type)
            self.brightness_enabled = bool(brightness_enabled)
            self.covariance_kind = str(covariance_kind)
            self.policy_conditioning_enabled = bool(policy_conditioning_enabled)
            self.minimum_valid_tone_fraction = float(minimum_valid_tone_fraction)
            self.missing_command_covariance_diagonal = float(missing_command_covariance_diagonal)
            if self.policy_conditioning_enabled and not bool(share_forward_command_encoder):
                raise ValueError("policy conditioning requires the shared forward command encoder")
            self.trunk = nn.Sequential(nn.LazyLinear(hidden_dim), nn.SiLU(), nn.LayerNorm(hidden_dim))
            self.policy_command_projection = nn.LazyLinear(hidden_dim)
            self.policy_command_gate = nn.Linear(hidden_dim * 2, hidden_dim)
            self.policy_command_interaction = nn.Linear(hidden_dim, hidden_dim)
            self.policy_fusion_norm = nn.LayerNorm(hidden_dim)
            nn.init.zeros_(self.policy_command_gate.weight)
            nn.init.constant_(self.policy_command_gate.bias, float(gate_initial_bias))
            nn.init.zeros_(self.policy_command_interaction.weight)
            nn.init.zeros_(self.policy_command_interaction.bias)
            self.amplitude_coordinate_decoder = nn.Linear(hidden_dim, self.dimensions["amplitude"])
            self.phase_coordinate_decoder = nn.Linear(hidden_dim, self.dimensions["phase"])
            self.brightness_decoder = nn.Linear(hidden_dim, 1) if self.brightness_enabled else None
            covariance_values = self.dimensions["total"] * (self.dimensions["total"] + 1) // 2
            self.uncertainty_covariance_head = nn.Linear(hidden_dim, covariance_values)
            _, cycles, rows, groups, cells = command_layout(system_type)
            self.local_layout = (cycles, rows, groups, cells)
            shared_values = rows * groups * cells
            cycle_values = cycles * rows * groups * cells
            self.shared_spatial_amplitude_decoder = nn.Linear(hidden_dim, shared_values)
            self.shared_spatial_phase_decoder = nn.Linear(hidden_dim, shared_values)
            self.cycle_amplitude_decoder = nn.Linear(hidden_dim, cycle_values)
            self.cycle_phase_decoder = nn.Linear(hidden_dim, cycle_values)
            for decoder in (
                self.shared_spatial_amplitude_decoder, self.shared_spatial_phase_decoder,
                self.cycle_amplitude_decoder, self.cycle_phase_decoder,
            ):
                nn.init.zeros_(decoder.weight)
                nn.init.zeros_(decoder.bias)
            self.count_forward_model = DifferentiableProjectiveCountForward(
                system_type, command_feature_dim, hidden_dim=hidden_dim,
                detector_model=detector_model,
                source_model=source_model,
                state_source=state_source,
            )

        def forward(
            self,
            observation: Mapping[str, Any],
            command: CommandTensorBatch,
            *,
            amplitude_limit: float = 0.03,
            phase_limit_rad: float = 0.20,
            brightness_limit: float = 0.0,
        ) -> Dict[str, "torch.Tensor"]:
            features = _observation_tensor(
                observation, batch_size=command.batch_size,
                device=command.tone_features.device, dtype=command.tone_features.dtype,
            )
            observation_latent = self.trunk(features)
            command_context = self.count_forward_model.encode_command_context(command)
            if self.policy_conditioning_enabled:
                command_latent = self.policy_command_projection(command_context.policy_summary)
                gate = torch.sigmoid(self.policy_command_gate(torch.cat((observation_latent, command_latent), dim=-1)))
                interaction = self.policy_command_interaction(observation_latent * command_latent)
                fused_latent = self.policy_fusion_norm(
                    observation_latent + gate * command_latent + interaction
                )
                policy_valid = command_context.valid_tone_fraction >= self.minimum_valid_tone_fraction
            else:
                command_latent = torch.zeros_like(observation_latent)
                gate = torch.zeros_like(observation_latent)
                interaction = torch.zeros_like(observation_latent)
                fused_latent = observation_latent
                policy_valid = torch.ones(
                    observation_latent.shape[0], dtype=torch.bool, device=observation_latent.device
                )
            raw_amp = self.amplitude_coordinate_decoder(fused_latent)
            raw_phase = self.phase_coordinate_decoder(fused_latent)
            raw_brightness = self.brightness_decoder(fused_latent).squeeze(-1) if self.brightness_decoder is not None else torch.zeros(fused_latent.shape[0], dtype=fused_latent.dtype, device=fused_latent.device)
            action = project_action(
                self.system_type, raw_amp, raw_phase,
                amplitude_limit=amplitude_limit, phase_limit_rad=phase_limit_rad,
                raw_brightness=raw_brightness, brightness_enabled=self.brightness_enabled,
                brightness_limit=brightness_limit,
            )
            covariance = action_covariance_from_lower_triangular(
                self.uncertainty_covariance_head(fused_latent), self.system_type,
            )
            valid_coordinate = policy_valid.to(action.amplitude_coordinates.dtype).unsqueeze(-1)
            if self.policy_conditioning_enabled:
                identity = torch.eye(
                    self.dimensions["total"], dtype=covariance.dtype, device=covariance.device,
                )
                covariance = covariance + (
                    (~policy_valid).to(covariance.dtype).reshape(-1, 1, 1)
                    * self.missing_command_covariance_diagonal
                    * identity
                )
            amplitude_mask = observation.get("amplitude_actionable_mask")
            if amplitude_mask is not None:
                mask = torch.as_tensor(amplitude_mask, dtype=action.amplitude_coordinates.dtype, device=action.amplitude_coordinates.device)
                action_coordinates = torch.cat((action.amplitude_coordinates * mask, action.phase_coordinates), dim=-1) * valid_coordinate
            else:
                action_coordinates = action.coordinates * valid_coordinate
            cycles, rows, groups, cells = self.local_layout
            local_action = {
                "shared_spatial_amplitude_raw": self.shared_spatial_amplitude_decoder(fused_latent).reshape(-1, rows, groups, cells) * valid_coordinate[:, None, None],
                "shared_spatial_phase_raw": self.shared_spatial_phase_decoder(fused_latent).reshape(-1, rows, groups, cells) * valid_coordinate[:, None, None],
                "cycle_amplitude_raw": self.cycle_amplitude_decoder(fused_latent).reshape(-1, cycles, rows, groups, cells) * valid_coordinate[:, None, None, None],
                "cycle_phase_raw": self.cycle_phase_decoder(fused_latent).reshape(-1, cycles, rows, groups, cells) * valid_coordinate[:, None, None, None],
            }
            brightness_delta = action.brightness_delta * policy_valid.to(action.brightness_delta.dtype)
            forward_context = self.count_forward_model.build_forward_context(
                observation,
                command,
                encoded_command_context=command_context,
                observation_latent=fused_latent,
                local_action_residuals=local_action,
                amplitude_limit=amplitude_limit,
                phase_limit_rad=phase_limit_rad,
                brightness_delta=brightness_delta,
                brightness_limit=brightness_limit if self.brightness_enabled else 0.0,
            )
            result = self.count_forward_model.forward_map(
                observation, command, action_coordinates,
                forward_context=forward_context,
                amplitude_limit=amplitude_limit,
                phase_limit_rad=phase_limit_rad,
            )
            result.update({
                "amplitude_action_coordinates": action_coordinates[..., : self.dimensions["amplitude"]],
                "phase_action_coordinates": action_coordinates[..., self.dimensions["amplitude"] :],
                "brightness_delta": brightness_delta,
                "action_covariance": covariance,
                "amplitude_phase_cross_covariance": covariance[..., : self.dimensions["amplitude"], self.dimensions["amplitude"] :],
                "count_jacobian_source": "autograd_same_forward_map",
                "policy_observation_latent": observation_latent,
                "policy_command_latent": command_latent,
                "policy_command_gate": gate,
                "policy_command_interaction": interaction,
                "policy_fused_latent": fused_latent,
                "policy_conditioning_enabled": self.policy_conditioning_enabled,
                "policy_valid_command_mask": policy_valid,
                "amplitude_actionable_mask": policy_valid.unsqueeze(-1).expand(-1, self.dimensions["amplitude"]),
                "phase_actionable_mask": policy_valid.unsqueeze(-1).expand(-1, self.dimensions["phase"]),
                "fail_closed_missing_command": ~policy_valid,
                "fail_closed_reason": "insufficient_valid_command" if bool(torch.any(~policy_valid).detach().cpu()) else "",
                "forward_context": forward_context,
            })
            return result


else:  # pragma: no cover
    class DifferentiableProjectiveCountForward:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("DifferentiableProjectiveCountForward requires torch")

    class OperationComplexFeedbackModel:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("OperationComplexFeedbackModel requires torch")


def autograd_log_rate_jacobian(
    model: Any,
    observation: Mapping[str, Any],
    command: CommandTensorBatch,
    *,
    action: Optional[Any] = None,
    action_point: Optional[Any] = None,
    forward_context: Optional[ForwardContext] = None,
    create_graph: bool = False,
) -> Any:
    """Forward-mode JVP Jacobian at action=0 (or caller-supplied action)."""
    if torch is None:
        raise RuntimeError("autograd_log_rate_jacobian requires torch")
    batch = command.batch_size
    dimension = model.action_dim
    if action is not None and action_point is not None:
        raise ValueError("use action_point or legacy action, not both")
    selected_action = action_point if action_point is not None else action
    base = torch.zeros((batch, dimension), dtype=command.tone_features.dtype, device=command.tone_features.device) if selected_action is None else selected_action
    context = forward_context or model.build_forward_context(observation, command)

    def function(value: "torch.Tensor") -> "torch.Tensor":
        return torch.log(model.forward_map(
            observation, command, value, forward_context=context,
        )["expected_rate"].clamp_min(1e-12))

    columns = []
    for index in range(dimension):
        tangent = torch.zeros_like(base)
        tangent[:, index] = 1.0
        _, column = torch.autograd.functional.jvp(function, base, tangent, create_graph=create_graph, strict=False)
        columns.append(column)
    return torch.stack(columns, dim=-1)


def structured_log_rate_jacobian(
    model: Any,
    observation: Mapping[str, Any],
    command: CommandTensorBatch,
    *,
    action: Optional[Any] = None,
    action_point: Optional[Any] = None,
    forward_context: Optional[ForwardContext] = None,
) -> Any:
    """Structured per-action forward derivative used as a second audit route."""
    if torch is None:
        raise RuntimeError("structured_log_rate_jacobian requires torch")
    batch = command.batch_size
    dimension = model.action_dim
    if action is not None and action_point is not None:
        raise ValueError("use action_point or legacy action, not both")
    selected_action = action_point if action_point is not None else action
    base = (
        torch.zeros((batch, dimension), dtype=command.tone_features.dtype, device=command.tone_features.device)
        if selected_action is None else selected_action
    )
    context = forward_context or model.build_forward_context(observation, command)

    def function(value: "torch.Tensor") -> "torch.Tensor":
        return torch.log(model.forward_map(
            observation, command, value, forward_context=context,
        )["expected_rate"].clamp_min(1e-12))

    # Reverse-mode full Jacobian is intentionally separate from the JVP route.
    # Only batch-diagonal blocks are physical because samples are independent.
    full = torch.autograd.functional.jacobian(
        function, base, create_graph=False, vectorize=True, strict=False,
    )
    return torch.stack([full[index, :, :, index, :] for index in range(batch)], dim=0)


def central_difference_log_rate_jacobian(
    model: Any,
    observation: Mapping[str, Any],
    command: CommandTensorBatch,
    *,
    action: Optional[Any] = None,
    action_point: Optional[Any] = None,
    forward_context: Optional[ForwardContext] = None,
    epsilon: float = 1e-4,
) -> Any:
    if torch is None:
        raise RuntimeError("central_difference_log_rate_jacobian requires torch")
    batch = command.batch_size
    dimension = model.action_dim
    if action is not None and action_point is not None:
        raise ValueError("use action_point or legacy action, not both")
    selected_action = action_point if action_point is not None else action
    base = torch.zeros((batch, dimension), dtype=command.tone_features.dtype, device=command.tone_features.device) if selected_action is None else selected_action
    context = forward_context or model.build_forward_context(observation, command)
    columns = []
    with torch.no_grad():
        for index in range(dimension):
            offset = torch.zeros_like(base)
            offset[:, index] = float(epsilon)
            positive = torch.log(model.forward_map(
                observation, command, base + offset, forward_context=context,
            )["expected_rate"].clamp_min(1e-12))
            negative = torch.log(model.forward_map(
                observation, command, base - offset, forward_context=context,
            )["expected_rate"].clamp_min(1e-12))
            columns.append((positive - negative) / (2.0 * float(epsilon)))
    return torch.stack(columns, dim=-1)


def fisher_from_integrable_count_map(
    expected_mean: Any,
    log_rate_jacobian: Any,
    *,
    nb2_dispersion: Any = 0.0,
    detector_variance: Optional[Any] = None,
    ridge: float = 1e-6,
) -> Dict[str, Any]:
    """Fisher/posterior from the same expected mean and its action Jacobian."""
    if torch is None or not isinstance(expected_mean, torch.Tensor):
        mean = np.asarray(expected_mean, dtype=float)
        jacobian = np.asarray(log_rate_jacobian, dtype=float)
        if detector_variance is None:
            alpha = np.asarray(nb2_dispersion, dtype=float)
            weight = mean / np.maximum(1.0 + alpha * mean, 1e-12)
        else:
            weight = np.square(mean) / np.maximum(np.asarray(detector_variance, dtype=float), 1e-12)
        fisher = np.einsum("...oc,...ocd,...oce->...de", weight, jacobian, jacobian)
        covariance = np.linalg.pinv(fisher + float(ridge) * np.eye(fisher.shape[-1]), hermitian=True)
    else:
        mean = expected_mean
        jacobian = log_rate_jacobian
        if detector_variance is None:
            alpha = torch.as_tensor(nb2_dispersion, dtype=mean.dtype, device=mean.device)
            weight = mean / (1.0 + alpha * mean).clamp_min(1e-12)
        else:
            variance = torch.as_tensor(detector_variance, dtype=mean.dtype, device=mean.device)
            weight = mean.square() / variance.clamp_min(1e-12)
        fisher = torch.einsum("...oc,...ocd,...oce->...de", weight, jacobian, jacobian)
        identity = torch.eye(fisher.shape[-1], dtype=fisher.dtype, device=fisher.device)
        covariance = torch.linalg.pinv(fisher + float(ridge) * identity, hermitian=True)
    amplitude_dim = jacobian.shape[-1] // 2
    return {
        "fisher": fisher,
        "posterior_covariance": covariance,
        "fisher_aa": fisher[..., :amplitude_dim, :amplitude_dim],
        "fisher_ap": fisher[..., :amplitude_dim, amplitude_dim:],
        "fisher_pa": fisher[..., amplitude_dim:, :amplitude_dim],
        "fisher_pp": fisher[..., amplitude_dim:, amplitude_dim:],
    }


@dataclass(frozen=True)
class BoundedInverseControllerResult:
    action_coordinates: Any
    amplitude_actionable_mask: Any
    phase_actionable_mask: Any
    per_coordinate_actionable_mask: Any
    fisher_condition: Any
    block_conditions: Mapping[str, Any]
    posterior_std: Any
    skipped_reasons: Tuple[str, ...]
    skipped_low_information: Any


class BoundedInverseController:
    """Regularized inverse controller with separate amplitude/phase fail-close."""

    def __init__(
        self,
        system_type: str,
        *,
        ridge: float = 1e-3,
        maximum_condition_number: float = 1e4,
        minimum_information_eigenvalue: float = 1e-6,
        amplitude_limit: float = 0.03,
        phase_limit_rad: float = 0.20,
    ) -> None:
        self.system_type = system_type
        self.ridge = float(ridge)
        self.maximum_condition_number = float(maximum_condition_number)
        self.minimum_information_eigenvalue = float(minimum_information_eigenvalue)
        self.amplitude_limit = float(amplitude_limit)
        self.phase_limit_rad = float(phase_limit_rad)

    def solve(self, fisher: Any, score: Any) -> BoundedInverseControllerResult:
        dimensions = action_dimensions(self.system_type)
        amp_dim = dimensions["amplitude"]
        if _is_torch_tensor(fisher):
            identity = torch.eye(dimensions["total"], dtype=fisher.dtype, device=fisher.device)
            eigen = torch.linalg.eigvalsh(fisher)
            condition = eigen[..., -1] / eigen[..., 0].clamp_min(self.minimum_information_eigenvalue)
            fisher_aa = fisher[..., :amp_dim, :amp_dim]
            fisher_pp = fisher[..., amp_dim:, amp_dim:]
            fisher_ap = fisher[..., :amp_dim, amp_dim:]
            amp_eigen = torch.linalg.eigvalsh(fisher_aa)
            phase_eigen = torch.linalg.eigvalsh(fisher_pp)
            condition_amp = amp_eigen[..., -1] / amp_eigen[..., 0].clamp_min(self.minimum_information_eigenvalue)
            condition_phase = phase_eigen[..., -1] / phase_eigen[..., 0].clamp_min(self.minimum_information_eigenvalue)
            amp_mask = (amp_eigen[..., 0] >= self.minimum_information_eigenvalue) & (condition_amp <= self.maximum_condition_number)
            phase_mask = (phase_eigen[..., 0] >= self.minimum_information_eigenvalue) & (condition_phase <= self.maximum_condition_number)
            cross_ratio = torch.linalg.matrix_norm(fisher_ap) / torch.sqrt(
                torch.linalg.matrix_norm(fisher_aa).clamp_min(1e-12)
                * torch.linalg.matrix_norm(fisher_pp).clamp_min(1e-12)
            )
            full_solution = torch.linalg.pinv(
                fisher + self.ridge * identity,
                hermitian=True,
                rtol=1.0 / self.maximum_condition_number,
            ) @ score.unsqueeze(-1)
            amp_identity = torch.eye(amp_dim, dtype=fisher.dtype, device=fisher.device)
            phase_dim = dimensions["phase"]
            phase_identity = torch.eye(phase_dim, dtype=fisher.dtype, device=fisher.device)
            amp_solution = torch.linalg.solve(
                fisher_aa + self.ridge * amp_identity,
                score[..., :amp_dim].unsqueeze(-1),
            ).squeeze(-1)
            phase_solution = torch.linalg.solve(
                fisher_pp + self.ridge * phase_identity,
                score[..., amp_dim:].unsqueeze(-1),
            ).squeeze(-1)
            solution = full_solution.squeeze(-1)
            solution = torch.cat((
                torch.where(amp_mask.unsqueeze(-1), solution[..., :amp_dim], torch.zeros_like(amp_solution)),
                torch.where(phase_mask.unsqueeze(-1), solution[..., amp_dim:], torch.zeros_like(phase_solution)),
            ), dim=-1)
            only_amp = amp_mask & ~phase_mask
            only_phase = phase_mask & ~amp_mask
            solution[..., :amp_dim] = torch.where(only_amp.unsqueeze(-1), amp_solution, solution[..., :amp_dim])
            solution[..., amp_dim:] = torch.where(only_phase.unsqueeze(-1), phase_solution, solution[..., amp_dim:])
            posterior = torch.linalg.pinv(fisher + self.ridge * identity, hermitian=True)
            posterior_std = torch.sqrt(torch.diagonal(posterior, dim1=-2, dim2=-1).clamp_min(0.0))
            coordinate_information = torch.diagonal(fisher, dim1=-2, dim2=-1) >= self.minimum_information_eigenvalue
            block_coordinate_mask = torch.cat((
                amp_mask.unsqueeze(-1).expand(-1, amp_dim),
                phase_mask.unsqueeze(-1).expand(-1, dimensions["phase"]),
            ), dim=-1)
            coordinate_mask = coordinate_information & block_coordinate_mask
            bounded = torch.cat((
                np.log1p(self.amplitude_limit) * torch.tanh(solution[..., :amp_dim]),
                self.phase_limit_rad * torch.tanh(solution[..., amp_dim:]),
            ), dim=-1)
            bounded = bounded * coordinate_mask.to(bounded.dtype)
            skipped = ~(amp_mask & phase_mask)
            skipped_reasons = tuple(
                reason for reason, present in (
                    ("amplitude_low_information", bool(torch.any(~amp_mask).detach().cpu())),
                    ("phase_low_information", bool(torch.any(~phase_mask).detach().cpu())),
                    ("cross_block_truncated_svd", bool(torch.any(cross_ratio > 0.9).detach().cpu())),
                ) if present
            )
            block_conditions = {
                "condition_amp": condition_amp,
                "condition_phase": condition_phase,
                "minimum_eigenvalue_amp": amp_eigen[..., 0],
                "minimum_eigenvalue_phase": phase_eigen[..., 0],
                "cross_block_ratio": cross_ratio,
            }
        else:
            matrix = np.asarray(fisher, dtype=float)
            vector = np.asarray(score, dtype=float)
            identity = np.eye(dimensions["total"])
            eigen = np.linalg.eigvalsh(matrix)
            condition = eigen[..., -1] / np.maximum(eigen[..., 0], self.minimum_information_eigenvalue)
            fisher_aa = matrix[..., :amp_dim, :amp_dim]
            fisher_pp = matrix[..., amp_dim:, amp_dim:]
            fisher_ap = matrix[..., :amp_dim, amp_dim:]
            amp_eigen = np.linalg.eigvalsh(fisher_aa)
            phase_eigen = np.linalg.eigvalsh(fisher_pp)
            condition_amp = amp_eigen[..., -1] / np.maximum(amp_eigen[..., 0], self.minimum_information_eigenvalue)
            condition_phase = phase_eigen[..., -1] / np.maximum(phase_eigen[..., 0], self.minimum_information_eigenvalue)
            amp_mask = (amp_eigen[..., 0] >= self.minimum_information_eigenvalue) & (condition_amp <= self.maximum_condition_number)
            phase_mask = (phase_eigen[..., 0] >= self.minimum_information_eigenvalue) & (condition_phase <= self.maximum_condition_number)
            cross_ratio = np.linalg.norm(fisher_ap, axis=(-2, -1)) / np.sqrt(
                np.maximum(np.linalg.norm(fisher_aa, axis=(-2, -1)), 1e-12)
                * np.maximum(np.linalg.norm(fisher_pp, axis=(-2, -1)), 1e-12)
            )
            solution = (np.linalg.pinv(
                matrix + self.ridge * identity,
                hermitian=True,
                rcond=1.0 / self.maximum_condition_number,
            ) @ vector[..., None])[..., 0]
            amp_solution = np.linalg.solve(
                fisher_aa + self.ridge * np.eye(amp_dim), vector[..., :amp_dim, None],
            )[..., 0]
            phase_solution = np.linalg.solve(
                fisher_pp + self.ridge * np.eye(dimensions["phase"]), vector[..., amp_dim:, None],
            )[..., 0]
            solution[..., :amp_dim] = np.where((amp_mask & ~phase_mask)[..., None], amp_solution, solution[..., :amp_dim])
            solution[..., amp_dim:] = np.where((phase_mask & ~amp_mask)[..., None], phase_solution, solution[..., amp_dim:])
            solution[..., :amp_dim] = np.where(amp_mask[..., None], solution[..., :amp_dim], 0.0)
            solution[..., amp_dim:] = np.where(phase_mask[..., None], solution[..., amp_dim:], 0.0)
            posterior = np.linalg.pinv(matrix + self.ridge * identity, hermitian=True)
            posterior_std = np.sqrt(np.maximum(np.diagonal(posterior, axis1=-2, axis2=-1), 0.0))
            coordinate_information = np.diagonal(matrix, axis1=-2, axis2=-1) >= self.minimum_information_eigenvalue
            block_coordinate_mask = np.concatenate((
                np.broadcast_to(amp_mask[..., None], amp_mask.shape + (amp_dim,)),
                np.broadcast_to(phase_mask[..., None], phase_mask.shape + (dimensions["phase"],)),
            ), axis=-1)
            coordinate_mask = coordinate_information & block_coordinate_mask
            bounded = np.concatenate((
                np.log1p(self.amplitude_limit) * np.tanh(solution[..., :amp_dim]),
                self.phase_limit_rad * np.tanh(solution[..., amp_dim:]),
            ), axis=-1)
            bounded *= coordinate_mask.astype(float)
            skipped = ~(amp_mask & phase_mask)
            skipped_reasons = tuple(
                reason for reason, present in (
                    ("amplitude_low_information", bool(np.any(~amp_mask))),
                    ("phase_low_information", bool(np.any(~phase_mask))),
                    ("cross_block_truncated_svd", bool(np.any(cross_ratio > 0.9))),
                ) if present
            )
            block_conditions = {
                "condition_amp": condition_amp,
                "condition_phase": condition_phase,
                "minimum_eigenvalue_amp": amp_eigen[..., 0],
                "minimum_eigenvalue_phase": phase_eigen[..., 0],
                "cross_block_ratio": cross_ratio,
            }
        return BoundedInverseControllerResult(
            bounded,
            amp_mask,
            phase_mask,
            coordinate_mask,
            condition,
            block_conditions,
            posterior_std,
            skipped_reasons,
            skipped,
        )


def _is_torch_tensor(value: Any) -> bool:
    return torch is not None and isinstance(value, torch.Tensor)


__all__ = [
    "CommandTensorBatch", "EncodedCommandContext", "ForwardContext", "command_tensor_batch", "DifferentiableProjectiveCountForward",
    "OperationComplexFeedbackModel", "autograd_log_rate_jacobian",
    "structured_log_rate_jacobian", "central_difference_log_rate_jacobian",
    "fisher_from_integrable_count_map", "BoundedInverseController",
    "BoundedInverseControllerResult", "COUNT_FORWARD_SCHEMA", "COUNT_FORWARD_VERSION",
]
