"""Tone-set command features and a shared real-valued multi-tone AOD core.

All phase-valued inputs use sine/cosine pairs and all amplitude-valued inputs
use log amplitude.  The core is therefore usable on DirectML backends that do
not support ``ComplexFloat``.  Absolute frequency is retained on a fixed MHz
scale; no sample-local min/max normalization is performed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .command_contract import COMMAND_CHANNELS, PhysicsHybridCommandTableV1

try:  # Optional ML dependency; NumPy contracts remain available without it.
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


FREQUENCY_CENTER_MHZ = 105.0
FREQUENCY_SCALE_MHZ = 40.0
MIN_LOG_AMPLITUDE = float(np.log(1e-12))


@dataclass(frozen=True)
class CommandFeatureSchemaV1:
    ordered_names: Tuple[str, ...]
    absolute_frequency_index: int
    log_amplitude_indices: Tuple[int, ...]
    phase_sin_indices: Tuple[int, ...]
    phase_cos_indices: Tuple[int, ...]
    missing_mask_indices: Tuple[int, ...]
    version: int = 1


def command_tone_features(table: PhysicsHybridCommandTableV1) -> Tuple[np.ndarray, CommandFeatureSchemaV1]:
    """Pack one table as ``[cycle,row,group,tone,F]``.

    Y-channel scalar commands are broadcast as views into each operation's tone
    set only while forming a batch; they remain scalar in the stored table.
    """
    static = table.static_arrays
    masks = table.masks
    shape = table.command_shape
    pieces = []
    names = []

    def append(name: str, value: Any) -> int:
        array = np.broadcast_to(np.asarray(value, dtype=np.float64), shape)
        pieces.append(array[..., None])
        names.append(name)
        return len(names) - 1

    absolute_frequency_index = append("absolute_frequency_mhz", static["absolute_frequency_mhz"])
    append("basis_amplitude", static["basis_amplitude"])
    append("basis_phase_sin", static["basis_phase_sin"])
    append("basis_phase_cos", static["basis_phase_cos"])

    log_amp_indices = []
    phase_sin_indices = []
    phase_cos_indices = []
    missing_indices = []
    command_missing = np.asarray(masks["command_missing_mask"], dtype=float)
    for channel_index, channel in enumerate(COMMAND_CHANNELS):
        if channel in {"W_CH1", "W_CH3", "R_CH1", "R_CH3"}:
            log_amp = np.asarray(static["x_log_amplitude"])[..., channel_index]
            phase_sin = np.asarray(static["x_phase_sin"])[..., channel_index]
            phase_cos = np.asarray(static["x_phase_cos"])[..., channel_index]
        else:
            amplitude = np.asarray(static["y_amplitude"])[..., channel_index]
            log_amp = np.broadcast_to(np.log(np.maximum(np.abs(amplitude), 1e-12))[..., None], shape)
            phase_sin = np.broadcast_to(np.asarray(static["y_phase_sin"])[..., channel_index, None], shape)
            phase_cos = np.broadcast_to(np.asarray(static["y_phase_cos"])[..., channel_index, None], shape)
            append(f"{channel}.absolute_frequency_mhz", np.broadcast_to(
                np.asarray(static["y_frequency_mhz"])[..., channel_index, None], shape,
            ))
        log_amp_indices.append(append(f"{channel}.log_amplitude", log_amp))
        phase_sin_indices.append(append(f"{channel}.phase_sin", phase_sin))
        phase_cos_indices.append(append(f"{channel}.phase_cos", phase_cos))
        missing_indices.append(append(f"{channel}.missing", command_missing[..., channel_index]))
        append(f"{channel}.rf_power", np.broadcast_to(
            np.asarray(static["per_channel_rf_power"])[..., channel_index, None], shape,
        ))
        append(f"{channel}.diffraction_order_sign", np.broadcast_to(
            np.asarray(static["diffraction_order_sign"])[..., channel_index, None], shape,
        ))
        for timing_name in ("digital_delay_ns", "digital_width_ns", "analog_delay_ns"):
            append(f"{channel}.{timing_name}", np.broadcast_to(
                np.asarray(static[timing_name])[..., channel_index, None], shape,
            ))
    append("total_rf_power", np.broadcast_to(np.asarray(static["total_rf_power"])[..., None], shape))
    features = np.concatenate(pieces, axis=-1)
    schema = CommandFeatureSchemaV1(
        ordered_names=tuple(names),
        absolute_frequency_index=absolute_frequency_index,
        log_amplitude_indices=tuple(log_amp_indices),
        phase_sin_indices=tuple(phase_sin_indices),
        phase_cos_indices=tuple(phase_cos_indices),
        missing_mask_indices=tuple(missing_indices),
    )
    return features, schema


def fixed_frequency_normalize(frequency_mhz: Any) -> Any:
    """Fixed physical scaling that preserves absolute center frequency."""
    return (frequency_mhz - FREQUENCY_CENTER_MHZ) / FREQUENCY_SCALE_MHZ


def complex_multiply_pair(left: Any, right: Any) -> Any:
    """Complex multiplication for tensors with a final ``[real,imag]`` axis."""
    if torch is not None and isinstance(left, torch.Tensor):
        return torch.stack((left[..., 0] * right[..., 0] - left[..., 1] * right[..., 1],
                            left[..., 0] * right[..., 1] + left[..., 1] * right[..., 0]), dim=-1)
    left_array, right_array = np.asarray(left), np.asarray(right)
    return np.stack((left_array[..., 0] * right_array[..., 0] - left_array[..., 1] * right_array[..., 1],
                     left_array[..., 0] * right_array[..., 1] + left_array[..., 1] * right_array[..., 0]), axis=-1)


def complex_conjugate_pair(value: Any) -> Any:
    if torch is not None and isinstance(value, torch.Tensor):
        return torch.stack((value[..., 0], -value[..., 1]), dim=-1)
    array = np.asarray(value)
    return np.stack((array[..., 0], -array[..., 1]), axis=-1)


def complex_abs_square_pair(value: Any) -> Any:
    return value[..., 0].square() + value[..., 1].square() if torch is not None and isinstance(value, torch.Tensor) else np.sum(np.square(value), axis=-1)


def compose_aod_channel_phase(
    *,
    basis_phase_rad: Any,
    channel_phases_rad: Sequence[Any],
    diffraction_order_signs: Sequence[Any],
    path_kind: str,
    residual_phase_rad: Any = 0.0,
) -> Any:
    """Apply the one deployed write/read diffraction-sign convention."""
    if path_kind not in {"write", "read"}:
        raise ValueError("path_kind must be 'write' or 'read'")
    if channel_phases_rad and len(channel_phases_rad) != len(diffraction_order_signs):
        raise ValueError("channel phases and diffraction signs must match")
    if torch is not None and isinstance(basis_phase_rad, torch.Tensor):
        zero = torch.zeros_like(basis_phase_rad)
        phase_sum = sum(channel_phases_rad, zero)
        if path_kind == "read":
            sign = sum(diffraction_order_signs, zero) / max(len(diffraction_order_signs), 1)
            return sign * (basis_phase_rad + phase_sum) + residual_phase_rad
        signed_phase = sum(
            (sign * phase for sign, phase in zip(diffraction_order_signs, channel_phases_rad)),
            zero,
        )
        return basis_phase_rad + signed_phase + residual_phase_rad
    basis = np.asarray(basis_phase_rad)
    phase_sum = sum((np.asarray(value) for value in channel_phases_rad), np.zeros_like(basis))
    if path_kind == "read":
        sign = sum(
            (np.asarray(value) for value in diffraction_order_signs), np.zeros_like(basis)
        ) / max(len(diffraction_order_signs), 1)
        return sign * (basis + phase_sum) + residual_phase_rad
    signed_phase = sum(
        (np.asarray(sign) * np.asarray(phase) for sign, phase in zip(diffraction_order_signs, channel_phases_rad)),
        np.zeros_like(basis),
    )
    return basis + signed_phase + residual_phase_rad


if nn is not None:

    class ToneSetCommandEncoder(nn.Module):
        """Permutation-equivariant tone encoder with explicit pair edges."""

        def __init__(self, tone_feature_dim: int, hidden_dim: int = 64, output_dim: int = 64) -> None:
            super().__init__()
            self.tone_feature_dim = int(tone_feature_dim)
            self.tone_mlp = nn.Sequential(
                nn.Linear(self.tone_feature_dim + 4, hidden_dim), nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            )
            # delta_f, |delta_f|, amp product, sin/cos(delta phase), and
            # shared-channel/shared-path flags.
            self.pair_mlp = nn.Sequential(
                nn.Linear(7, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            )
            self.output = nn.Sequential(nn.Linear(hidden_dim * 2, output_dim), nn.SiLU())

        def forward(
            self,
            tone_features: "torch.Tensor",
            absolute_frequency_mhz: "torch.Tensor",
            log_amplitude: "torch.Tensor",
            phase_sin: "torch.Tensor",
            phase_cos: "torch.Tensor",
            *,
            channel_flags: Optional["torch.Tensor"] = None,
            path_flags: Optional["torch.Tensor"] = None,
        ) -> "torch.Tensor":
            if tone_features.shape[-2] != absolute_frequency_mhz.shape[-1]:
                raise ValueError("tone feature and frequency tone dimensions disagree")
            if tone_features.shape[-1] != self.tone_feature_dim:
                raise ValueError("unexpected tone feature dimension")
            tone_input = torch.cat((tone_features, fixed_frequency_normalize(absolute_frequency_mhz).unsqueeze(-1),
                                    log_amplitude.unsqueeze(-1), phase_sin.unsqueeze(-1), phase_cos.unsqueeze(-1)), dim=-1)
            tone_latent = self.tone_mlp(tone_input)
            frequency_delta = absolute_frequency_mhz.unsqueeze(-1) - absolute_frequency_mhz.unsqueeze(-2)
            amp = torch.exp(torch.clamp(log_amplitude, min=MIN_LOG_AMPLITUDE))
            amp_product = amp.unsqueeze(-1) * amp.unsqueeze(-2)
            sin_delta = phase_sin.unsqueeze(-1) * phase_cos.unsqueeze(-2) - phase_cos.unsqueeze(-1) * phase_sin.unsqueeze(-2)
            cos_delta = phase_cos.unsqueeze(-1) * phase_cos.unsqueeze(-2) + phase_sin.unsqueeze(-1) * phase_sin.unsqueeze(-2)
            if channel_flags is None:
                shared_channel = torch.zeros_like(frequency_delta)
            else:
                shared_channel = torch.einsum("...tk,...sk->...ts", channel_flags, channel_flags).clamp(0.0, 1.0)
            if path_flags is None:
                shared_path = torch.zeros_like(frequency_delta)
            else:
                shared_path = torch.einsum("...tk,...sk->...ts", path_flags, path_flags).clamp(0.0, 1.0)
            pair_features = torch.stack((frequency_delta / FREQUENCY_SCALE_MHZ,
                                         frequency_delta.abs() / FREQUENCY_SCALE_MHZ,
                                         amp_product, sin_delta, cos_delta, shared_channel, shared_path), dim=-1)
            pair_latent = self.pair_mlp(pair_features)
            tones = absolute_frequency_mhz.shape[-1]
            eye = torch.eye(tones, dtype=pair_latent.dtype, device=pair_latent.device)
            pair_latent = pair_latent * (1.0 - eye).reshape((1,) * (pair_latent.ndim - 3) + (tones, tones, 1))
            aggregate = pair_latent.sum(dim=-2) / max(tones - 1, 1)
            return self.output(torch.cat((tone_latent, aggregate), dim=-1))


    class SharedMultiToneAODResponse(nn.Module):
        """One shared AOD kernel with small zero-initialized cycle residuals."""

        def __init__(self, hidden_dim: int = 48, max_cycles: int = 4, residual_limit: float = 0.05) -> None:
            super().__init__()
            self.residual_limit = float(residual_limit)
            self.base_amp_response = nn.Sequential(nn.Linear(2, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))
            self.base_phase_response = nn.Sequential(nn.Linear(2, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))
            self.power_compression = nn.Sequential(nn.Linear(2, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))
            self.power_phase_response = nn.Sequential(nn.Linear(2, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))
            self.pair_amp_kernel = nn.Sequential(nn.Linear(5, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))
            self.pair_phase_kernel = nn.Sequential(nn.Linear(5, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))
            self.bounded_residual = nn.Sequential(nn.Linear(5, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 2))
            self.cycle_residual = nn.Parameter(torch.zeros(int(max_cycles), 2))
            for branch in (
                self.base_amp_response,
                self.base_phase_response,
                self.power_compression,
                self.power_phase_response,
                self.pair_amp_kernel,
                self.pair_phase_kernel,
                self.bounded_residual,
            ):
                nn.init.zeros_(branch[-1].weight)
                nn.init.zeros_(branch[-1].bias)

        def forward(
            self,
            absolute_frequency_mhz: "torch.Tensor",
            log_amplitude: "torch.Tensor",
            phase_sin: "torch.Tensor",
            phase_cos: "torch.Tensor",
            per_channel_power: "torch.Tensor",
            temperature_c: "torch.Tensor",
            *,
            cycle_index: Optional["torch.Tensor"] = None,
        ) -> Tuple["torch.Tensor", "torch.Tensor", Dict[str, "torch.Tensor"]]:
            frequency = fixed_frequency_normalize(absolute_frequency_mhz)
            temperature = (temperature_c - 25.0) / 10.0
            while temperature.ndim < frequency.ndim:
                temperature = temperature.unsqueeze(-1)
            temperature = temperature.expand_as(frequency)
            power = per_channel_power
            while power.ndim < frequency.ndim:
                power = power.unsqueeze(-1)
            power = power.expand_as(frequency)
            local = torch.stack((frequency, temperature), dim=-1)
            power_input = torch.stack((frequency, torch.log1p(torch.clamp(power, min=0.0))), dim=-1)

            df = absolute_frequency_mhz.unsqueeze(-1) - absolute_frequency_mhz.unsqueeze(-2)
            amp = torch.exp(torch.clamp(log_amplitude, min=MIN_LOG_AMPLITUDE))
            partner_amp = amp.unsqueeze(-2).expand_as(df)
            sin_delta = phase_sin.unsqueeze(-1) * phase_cos.unsqueeze(-2) - phase_cos.unsqueeze(-1) * phase_sin.unsqueeze(-2)
            cos_delta = phase_cos.unsqueeze(-1) * phase_cos.unsqueeze(-2) + phase_sin.unsqueeze(-1) * phase_sin.unsqueeze(-2)
            pair_input = torch.stack((df / FREQUENCY_SCALE_MHZ, df.abs() / FREQUENCY_SCALE_MHZ,
                                      partner_amp, sin_delta, cos_delta), dim=-1)
            tone_count = absolute_frequency_mhz.shape[-1]
            off_diagonal = 1.0 - torch.eye(tone_count, dtype=absolute_frequency_mhz.dtype, device=absolute_frequency_mhz.device)
            off_diagonal = off_diagonal.reshape((1,) * (pair_input.ndim - 3) + (tone_count, tone_count, 1))
            # Exact structural zero when the *other* tone amplitude is zero.
            pair_amp = (self.pair_amp_kernel(pair_input) * partner_amp.unsqueeze(-1) * off_diagonal).sum(dim=-2).squeeze(-1)
            pair_phase = (self.pair_phase_kernel(pair_input) * partner_amp.unsqueeze(-1) * off_diagonal).sum(dim=-2).squeeze(-1)
            residual_input = torch.stack((frequency, temperature, torch.log1p(torch.clamp(power, min=0.0)), phase_sin, phase_cos), dim=-1)
            residual = self.residual_limit * torch.tanh(self.bounded_residual(residual_input))
            delta_log_amp = self.base_amp_response(local).squeeze(-1) + self.power_compression(power_input).squeeze(-1) + pair_amp + residual[..., 0]
            delta_phase = self.base_phase_response(local).squeeze(-1) + self.power_phase_response(power_input).squeeze(-1) + pair_phase + residual[..., 1]
            if cycle_index is not None:
                cycle_term = self.cycle_residual[cycle_index.long()]
                if cycle_index.ndim == 1:
                    cycle_term = cycle_term.reshape(
                        (1, cycle_index.numel())
                        + (1,) * (delta_log_amp.ndim - 2) + (2,)
                    )
                else:
                    while cycle_term.ndim < delta_log_amp.ndim + 1:
                        cycle_term = cycle_term.unsqueeze(-2)
                delta_log_amp = delta_log_amp + cycle_term[..., 0]
                delta_phase = delta_phase + cycle_term[..., 1]
            diagnostics = {
                "cycle_residual_norm": torch.linalg.vector_norm(self.cycle_residual if cycle_index is None else self.cycle_residual[cycle_index.long()]),
                "pair_amp_norm": torch.linalg.vector_norm(pair_amp),
                "pair_phase_norm": torch.linalg.vector_norm(pair_phase),
            }
            return delta_log_amp, delta_phase, diagnostics


else:  # pragma: no cover

    class ToneSetCommandEncoder:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("ToneSetCommandEncoder requires torch")

    class SharedMultiToneAODResponse:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("SharedMultiToneAODResponse requires torch")


__all__ = [
    "CommandFeatureSchemaV1",
    "ToneSetCommandEncoder",
    "SharedMultiToneAODResponse",
    "command_tone_features",
    "fixed_frequency_normalize",
    "complex_multiply_pair",
    "complex_conjugate_pair",
    "complex_abs_square_pair",
    "compose_aod_channel_phase",
]
