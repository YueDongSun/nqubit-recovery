"""Explicit eight-channel Write/Read/Born forward path for V6 research.

The channel composition follows the deployed operation registry: W/R CH1 and
CH3 are X AODs, W/R CH2 and CH4 are Y AODs, and diffraction-order signs are
read from the registry-derived command table.  The primary physical paths are
W_CH1+W_CH2 and R_CH1+R_CH2.  CH3/CH4 enter only through bounded calibrated
residual kernels, initialized to the neutral response.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from .command_contract import COMMAND_CHANNELS
from .command_features import (
    SharedMultiToneAODResponse,
    compose_aod_channel_phase,
    complex_abs_square_pair,
    complex_conjugate_pair,
    complex_multiply_pair,
)
from .detector_click_model import FourDetectorClickModel, observed_pattern_probabilities
from .source_multiplicity import SourceMultiplicityModel

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


WRITE_READ_FORWARD_SCHEMA = "write_read_aod_forward"
WRITE_READ_FORWARD_VERSION = 1
PRIMARY_WRITE_CHANNELS: Tuple[str, str] = ("W_CH1", "W_CH2")
CALIBRATED_WRITE_CHANNELS: Tuple[str, str] = ("W_CH3", "W_CH4")
PRIMARY_READ_CHANNELS: Tuple[str, str] = ("R_CH1", "R_CH2")
CALIBRATED_READ_CHANNELS: Tuple[str, str] = ("R_CH3", "R_CH4")


def _channel_index(channel: str) -> int:
    return COMMAND_CHANNELS.index(channel)


def _phase_from_pair(phase_sin: "torch.Tensor", phase_cos: "torch.Tensor") -> "torch.Tensor":
    return torch.atan2(phase_sin, phase_cos)


def _channel_fields(command: Any, channel: str) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    index = _channel_index(channel)
    dtype = command.tone_features.dtype
    device = command.tone_features.device
    if channel.endswith("CH1") or channel.endswith("CH3"):
        amplitude = torch.exp(command.x_log_amplitude_all[..., index].to(dtype=dtype, device=device))
        phase = _phase_from_pair(
            command.x_phase_sin_all[..., index].to(dtype=dtype, device=device),
            command.x_phase_cos_all[..., index].to(dtype=dtype, device=device),
        )
    else:
        amplitude = command.y_amplitude_all[..., index].to(dtype=dtype, device=device)
        phase = _phase_from_pair(
            command.y_phase_sin_all[..., index].to(dtype=dtype, device=device),
            command.y_phase_cos_all[..., index].to(dtype=dtype, device=device),
        )
    sign = command.diffraction_sign_all[..., index].to(dtype=dtype, device=device)
    return amplitude, phase, sign


def _masked_mean(value: "torch.Tensor", mask: "torch.Tensor", dimensions: Tuple[int, ...]) -> "torch.Tensor":
    weight = mask.to(dtype=value.dtype)
    while weight.ndim < value.ndim:
        weight = weight.unsqueeze(-1)
    numerator = (value * weight).sum(dim=dimensions)
    denominator = weight.sum(dim=dimensions).clamp_min(1.0)
    return numerator / denominator


def _multi_tone_residual(
    command: Any,
    channels: Tuple[str, str],
    kernel: Optional[Any],
) -> Tuple["torch.Tensor", "torch.Tensor", Dict[str, "torch.Tensor"]]:
    first_amp, first_phase, _ = _channel_fields(command, channels[0])
    second_amp, second_phase, _ = _channel_fields(command, channels[1])
    zero = torch.zeros_like(first_amp)
    if kernel is None:
        return zero, zero, {"pair_amp_norm": zero.sum(), "pair_phase_norm": zero.sum()}
    first_index, second_index = (_channel_index(channel) for channel in channels)
    log_amplitude = 0.5 * (
        torch.log(first_amp.clamp_min(1e-12))
        + torch.log(second_amp.clamp_min(1e-12))
    )
    phase_sin = torch.sin(first_phase) + torch.sin(second_phase)
    phase_cos = torch.cos(first_phase) + torch.cos(second_phase)
    phase_norm = torch.sqrt(torch.square(phase_sin) + torch.square(phase_cos)).clamp_min(1e-12)
    power = command.per_channel_power_all[..., first_index] + command.per_channel_power_all[..., second_index]
    return kernel(
        command.absolute_frequency_mhz,
        log_amplitude,
        phase_sin / phase_norm,
        phase_cos / phase_norm,
        power,
        torch.zeros(command.batch_size, dtype=first_amp.dtype, device=first_amp.device),
        cycle_index=command.cycle_index,
    )


def shared_state_consensus(
    amplitude: "torch.Tensor",
    phase_rad: "torch.Tensor",
    valid_tone_mask: "torch.Tensor",
    *,
    uncertainty: Optional["torch.Tensor"] = None,
    huber_delta: float = 1.5,
) -> Tuple["torch.Tensor", Dict[str, "torch.Tensor"]]:
    """Robustly combine operations in amplitude-simplex/Helmert coordinates."""
    if amplitude.shape != phase_rad.shape or amplitude.shape != valid_tone_mask.shape:
        raise ValueError("consensus amplitude, phase, and valid mask must match")
    batch = amplitude.shape[0]
    cells = amplitude.shape[-1]
    operation_amplitude = amplitude.reshape(batch, -1, cells).clamp_min(1e-12)
    operation_amplitude = operation_amplitude / torch.linalg.vector_norm(
        operation_amplitude, dim=-1, keepdim=True
    ).clamp_min(1e-12)
    operation_phase = phase_rad.reshape(batch, -1, cells)
    operation_phase = operation_phase - operation_phase.mean(dim=-1, keepdim=True)
    valid_operation = valid_tone_mask.reshape(batch, -1, cells).all(dim=-1)
    base_weight = valid_operation.to(dtype=amplitude.dtype)
    if uncertainty is not None:
        uncertainty_value = torch.as_tensor(
            uncertainty, dtype=amplitude.dtype, device=amplitude.device
        ).reshape_as(base_weight)
        base_weight = base_weight / torch.square(uncertainty_value.clamp_min(1e-6))

    basis = torch.zeros((cells, max(cells - 1, 0)), dtype=amplitude.dtype, device=amplitude.device)
    for column in range(cells - 1):
        scale = float(np.sqrt((column + 1) * (column + 2)))
        basis[: column + 1, column] = 1.0 / scale
        basis[column + 1, column] = -(column + 1) / scale
    amplitude_coordinates = torch.log(operation_amplitude)
    amplitude_coordinates = amplitude_coordinates - amplitude_coordinates.mean(dim=-1, keepdim=True)
    amplitude_coordinates = amplitude_coordinates @ basis
    phase_coordinates = operation_phase @ basis

    def robust_mean(values: "torch.Tensor") -> "torch.Tensor":
        center = values.median(dim=1, keepdim=True).values
        residual = values - center
        scale = residual.abs().median(dim=1, keepdim=True).values.clamp_min(1e-6)
        robust = torch.clamp(
            float(huber_delta) * scale / residual.abs().clamp_min(1e-12),
            max=1.0,
        )
        weight = base_weight.unsqueeze(-1) * robust
        return (values * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1e-12)

    mean_amplitude_coordinates = robust_mean(amplitude_coordinates)
    mean_phase_coordinates = robust_mean(phase_coordinates)
    shared_log_amplitude = mean_amplitude_coordinates @ basis.T
    shared_amplitude = torch.exp(shared_log_amplitude)
    shared_amplitude = shared_amplitude / torch.linalg.vector_norm(
        shared_amplitude, dim=-1, keepdim=True
    ).clamp_min(1e-12)
    shared_phase = mean_phase_coordinates @ basis.T
    shared_pair = torch.stack((
        shared_amplitude * torch.cos(shared_phase),
        shared_amplitude * torch.sin(shared_phase),
    ), dim=-1)
    return shared_pair, {
        "consensus_valid_operation_fraction": valid_operation.to(amplitude.dtype).mean(dim=1),
        "consensus_amplitude_coordinate_norm": torch.linalg.vector_norm(mean_amplitude_coordinates, dim=-1),
        "consensus_phase_coordinate_norm": torch.linalg.vector_norm(mean_phase_coordinates, dim=-1),
    }


if nn is not None:

    class _CalibratedPathResidual(nn.Module):
        """Small shared residual driven by CH3/CH4 command and timing state."""

        def __init__(self, hidden_dim: int, residual_limit: float) -> None:
            super().__init__()
            self.residual_limit = float(residual_limit)
            self.network = nn.Sequential(
                nn.Linear(10, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 2),
            )
            nn.init.zeros_(self.network[-1].weight)
            nn.init.zeros_(self.network[-1].bias)

        def forward(
            self,
            amp_a: "torch.Tensor",
            phase_a: "torch.Tensor",
            amp_b: "torch.Tensor",
            phase_b: "torch.Tensor",
            power_a: "torch.Tensor",
            power_b: "torch.Tensor",
            timing_a: "torch.Tensor",
            timing_b: "torch.Tensor",
        ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
            timing_a_scaled = timing_a / 5000.0
            timing_b_scaled = timing_b / 5000.0
            features = torch.stack((
                torch.log(amp_a.clamp_min(1e-12)),
                torch.sin(phase_a),
                torch.cos(phase_a),
                torch.log(amp_b.clamp_min(1e-12)),
                torch.sin(phase_b),
                torch.cos(phase_b),
                torch.log1p(power_a.clamp_min(0.0)),
                torch.log1p(power_b.clamp_min(0.0)),
                timing_a_scaled,
                timing_b_scaled,
            ), dim=-1)
            raw = self.network(features)
            bounded = self.residual_limit * torch.tanh(raw)
            return bounded[..., 0], bounded[..., 1], torch.linalg.vector_norm(bounded)


    class WriteAODForward(nn.Module):
        """Map all four W-channel commands to one shared prepared W state."""

        def __init__(self, *, hidden_dim: int = 32, residual_limit: float = 0.05) -> None:
            super().__init__()
            self.calibrated_path = _CalibratedPathResidual(hidden_dim, residual_limit)

        def forward(
            self,
            command: Any,
            *,
            shared_aod_kernel: Optional[SharedMultiToneAODResponse] = None,
        ) -> Dict[str, "torch.Tensor"]:
            w1_amp, w1_phase, w1_sign = _channel_fields(command, "W_CH1")
            w2_amp, w2_phase, w2_sign = _channel_fields(command, "W_CH2")
            w3_amp, w3_phase, _ = _channel_fields(command, "W_CH3")
            w4_amp, w4_phase, _ = _channel_fields(command, "W_CH4")
            power = command.per_channel_power_all.to(dtype=w1_amp.dtype, device=w1_amp.device)
            timing = command.timing_all.to(dtype=w1_amp.dtype, device=w1_amp.device).mean(dim=-1)
            residual_log_amp, residual_phase, residual_norm = self.calibrated_path(
                w3_amp,
                w3_phase,
                w4_amp,
                w4_phase,
                power[..., _channel_index("W_CH3")],
                power[..., _channel_index("W_CH4")],
                timing[..., _channel_index("W_CH3")],
                timing[..., _channel_index("W_CH4")],
            )
            multi_log_amp, multi_phase, multi_diagnostics = _multi_tone_residual(
                command, CALIBRATED_WRITE_CHANNELS, shared_aod_kernel
            )
            residual_limit = self.calibrated_path.residual_limit
            if residual_limit > 0.0:
                residual_log_amp = residual_limit * torch.tanh(
                    (residual_log_amp + multi_log_amp) / residual_limit
                )
                residual_phase = residual_limit * torch.tanh(
                    (residual_phase + multi_phase) / residual_limit
                )
            else:
                residual_log_amp = torch.zeros_like(residual_log_amp)
                residual_phase = torch.zeros_like(residual_phase)
            amplitude = w1_amp * w2_amp * torch.exp(residual_log_amp)
            phase = compose_aod_channel_phase(
                basis_phase_rad=torch.zeros_like(w1_phase),
                channel_phases_rad=(w1_phase, w2_phase),
                diffraction_order_signs=(w1_sign, w2_sign),
                path_kind="write",
                residual_phase_rad=residual_phase,
            )
            pair = torch.stack((amplitude * torch.cos(phase), amplitude * torch.sin(phase)), dim=-1)
            primary_indices = torch.as_tensor(
                [_channel_index(channel) for channel in PRIMARY_WRITE_CHANNELS],
                dtype=torch.long,
                device=command.per_channel_command_missing_mask.device,
            )
            primary_missing = command.per_channel_command_missing_mask.index_select(-1, primary_indices).any(dim=-1)
            valid = ~primary_missing
            shared_pair, consensus_diagnostics = shared_state_consensus(
                amplitude, phase, valid
            )
            shared_amplitude = torch.sqrt(complex_abs_square_pair(shared_pair).clamp_min(1e-12))
            shared_phase = torch.atan2(shared_pair[..., 1], shared_pair[..., 0])
            per_cycle_pair = _masked_mean(pair, valid, (2, 3))
            cycle_deviation = torch.linalg.vector_norm(per_cycle_pair - shared_pair[:, None])
            return {
                "prepared_state_real_imag": shared_pair,
                "prepared_amplitude": shared_amplitude,
                "prepared_phase_rad": shared_phase,
                "write_valid_tone_fraction": valid.to(shared_pair.dtype).mean(dim=(1, 2, 3, 4)),
                "write_cycle_deviation_norm": cycle_deviation,
                "write_calibrated_path_residual_norm": residual_norm,
                "write_pair_amp_norm": multi_diagnostics["pair_amp_norm"],
                "write_pair_phase_norm": multi_diagnostics["pair_phase_norm"],
                "write_primary_channels": PRIMARY_WRITE_CHANNELS,
                "write_calibrated_channels": CALIBRATED_WRITE_CHANNELS,
                **consensus_diagnostics,
            }


    class ReadAODForward(nn.Module):
        """Map all four R-channel commands and basis metadata to projectors."""

        def __init__(self, *, hidden_dim: int = 32, residual_limit: float = 0.05, maximum_cycles: int = 4) -> None:
            super().__init__()
            self.calibrated_path = _CalibratedPathResidual(hidden_dim, residual_limit)
            self.cycle_residual = nn.Parameter(torch.zeros(maximum_cycles, 2))
            self.residual_limit = float(residual_limit)

        def forward(
            self,
            command: Any,
            *,
            shared_aod_kernel: Optional[SharedMultiToneAODResponse] = None,
        ) -> Dict[str, "torch.Tensor"]:
            r1_amp, r1_phase, r1_sign = _channel_fields(command, "R_CH1")
            r2_amp, r2_phase, r2_sign = _channel_fields(command, "R_CH2")
            r3_amp, r3_phase, _ = _channel_fields(command, "R_CH3")
            r4_amp, r4_phase, _ = _channel_fields(command, "R_CH4")
            power = command.per_channel_power_all.to(dtype=r1_amp.dtype, device=r1_amp.device)
            timing = command.timing_all.to(dtype=r1_amp.dtype, device=r1_amp.device).mean(dim=-1)
            residual_log_amp, residual_phase, residual_norm = self.calibrated_path(
                r3_amp,
                r3_phase,
                r4_amp,
                r4_phase,
                power[..., _channel_index("R_CH3")],
                power[..., _channel_index("R_CH4")],
                timing[..., _channel_index("R_CH3")],
                timing[..., _channel_index("R_CH4")],
            )
            multi_log_amp, multi_phase, multi_diagnostics = _multi_tone_residual(
                command, CALIBRATED_READ_CHANNELS, shared_aod_kernel
            )
            residual_limit = self.calibrated_path.residual_limit
            if residual_limit > 0.0:
                residual_log_amp = residual_limit * torch.tanh(
                    (residual_log_amp + multi_log_amp) / residual_limit
                )
                residual_phase = residual_limit * torch.tanh(
                    (residual_phase + multi_phase) / residual_limit
                )
            else:
                residual_log_amp = torch.zeros_like(residual_log_amp)
                residual_phase = torch.zeros_like(residual_phase)
            cycle_term = self.residual_limit * torch.tanh(self.cycle_residual[command.cycle_index.long()])
            cycle_term = cycle_term.reshape((1, command.cycle_index.numel(), 1, 1, 1, 2))
            amplitude = command.basis_amplitude * r1_amp * r2_amp * torch.exp(
                residual_log_amp + cycle_term[..., 0]
            )
            amplitude = amplitude / torch.linalg.vector_norm(amplitude, dim=-1, keepdim=True).clamp_min(1e-12)
            basis_phase = torch.atan2(command.basis_phase_sin, command.basis_phase_cos)
            read_sign = 0.5 * (r1_sign + r2_sign)
            phase = compose_aod_channel_phase(
                basis_phase_rad=basis_phase,
                channel_phases_rad=(r1_phase, r2_phase),
                diffraction_order_signs=(r1_sign, r2_sign),
                path_kind="read",
                residual_phase_rad=residual_phase + cycle_term[..., 1],
            )
            projector = torch.stack((amplitude * torch.cos(phase), amplitude * torch.sin(phase)), dim=-1)
            return {
                "projector_real_imag": projector,
                "projector_amplitude": amplitude,
                "projector_phase_rad": phase,
                "read_cycle_residual_norm": torch.linalg.vector_norm(cycle_term),
                "read_calibrated_path_residual_norm": residual_norm,
                "read_pair_amp_norm": multi_diagnostics["pair_amp_norm"],
                "read_pair_phase_norm": multi_diagnostics["pair_phase_norm"],
                "read_primary_channels": PRIMARY_READ_CHANNELS,
                "read_calibrated_channels": CALIBRATED_READ_CHANNELS,
                "read_diffraction_sign_mean": read_sign.mean(),
            }


    class BornDetectorForward(nn.Module):
        """Prepared state/projector Born map followed by source and detector layers."""

        def __init__(
            self,
            *,
            source_model: Optional[SourceMultiplicityModel] = None,
            detector_model: Optional[FourDetectorClickModel] = None,
        ) -> None:
            super().__init__()
            self.source_model = source_model or SourceMultiplicityModel()
            self.detector_model = detector_model

        def forward(
            self,
            prepared_state: "torch.Tensor",
            projector: "torch.Tensor",
            *,
            detector_time_block_index: Optional["torch.Tensor"] = None,
            detector_parameters_snapshot: Optional[Mapping[str, "torch.Tensor"]] = None,
            source_parameters_snapshot: Optional[Mapping[str, "torch.Tensor"]] = None,
        ) -> Dict[str, "torch.Tensor"]:
            state = prepared_state
            while state.ndim < projector.ndim:
                state = state.unsqueeze(1)
            overlap_terms = complex_multiply_pair(complex_conjugate_pair(projector), state)
            overlap = overlap_terms.sum(dim=-2)
            born_probability = complex_abs_square_pair(overlap).clamp_min(0.0)
            population = complex_abs_square_pair(prepared_state)
            if population.shape[-1] == 3:
                population = torch.cat((population, torch.zeros_like(population[..., :1])), dim=-1)
            population = population / population.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            while population.ndim < projector.ndim - 1:
                population = population.unsqueeze(1)
            population = population.expand(projector.shape[:-2] + (4,))

            if source_parameters_snapshot is None:
                p_source = self.source_model(population, detector_time_block_index)
                source_diagnostics = self.source_model.diagnostics(detector_time_block_index)
            else:
                weights = source_parameters_snapshot["mixture_weights"]
                while weights.ndim < population.ndim:
                    weights = weights.unsqueeze(-2)
                from .source_multiplicity import source_pattern_probabilities

                p_source = source_pattern_probabilities(population, weights)
                source_diagnostics = dict(source_parameters_snapshot)

            graph_zero = p_source.sum() * 0.0
            if self.detector_model is None:
                p_observed = p_source
                detector_regularization = graph_zero
                detector_diagnostics = {
                    "detector_dark_prior_deviation": graph_zero,
                    "detector_efficiency_prior_deviation": graph_zero,
                    "detector_time_block_residual_norm": graph_zero,
                }
            else:
                if detector_parameters_snapshot is None:
                    q, eta = self.detector_model.probabilities(detector_time_block_index)
                else:
                    q = detector_parameters_snapshot["dark_probability"]
                    eta = detector_parameters_snapshot["detector_efficiency"]
                if q.ndim == 1:
                    p_observed = observed_pattern_probabilities(p_source, q=q, eta=eta)
                else:
                    p_observed = torch.stack([
                        observed_pattern_probabilities(p_source[index], q=q[index], eta=eta)
                        for index in range(p_source.shape[0])
                    ], dim=0)
                detector_regularization = self.detector_model.shrinkage_loss()
                detector_diagnostics = self.detector_model.prior_diagnostics()
            return {
                "born_probability": born_probability,
                "source_pattern_probability": p_source,
                "observed_click_pattern_probability": p_observed,
                "detector_regularization": detector_regularization,
                **source_diagnostics,
                **detector_diagnostics,
            }


else:  # pragma: no cover

    class WriteAODForward:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("WriteAODForward requires torch")

    class ReadAODForward:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("ReadAODForward requires torch")

    class BornDetectorForward:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("BornDetectorForward requires torch")


__all__ = [
    "WriteAODForward",
    "ReadAODForward",
    "BornDetectorForward",
    "WRITE_READ_FORWARD_SCHEMA",
    "WRITE_READ_FORWARD_VERSION",
    "PRIMARY_WRITE_CHANNELS",
    "CALIBRATED_WRITE_CHANNELS",
    "PRIMARY_READ_CHANNELS",
    "CALIBRATED_READ_CHANNELS",
    "shared_state_consensus",
]
