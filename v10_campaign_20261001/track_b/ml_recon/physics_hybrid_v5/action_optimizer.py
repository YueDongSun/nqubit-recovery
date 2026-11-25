"""Constrained inverse teacher for the V6 research count forward map.

The optimizer owns no hardware authority.  It freezes the supplied forward
model and optimizes only a bounded global projective W action while reusing a
single :class:`ForwardContext` for objective and uncertainty calculations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import numpy as np

from .action_conditioned_model import (
    ForwardContext,
    autograd_log_rate_jacobian,
    fisher_from_integrable_count_map,
)
from .projective_action import action_dimensions

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@dataclass(frozen=True)
class InverseTeacherConfig:
    maximum_steps: int = 120
    learning_rate: float = 0.04
    amplitude_limit: float = 0.03
    phase_limit_rad: float = 0.20
    trust_radius: float = 0.20
    target_count_weight: float = 1.0
    target_state_weight: float = 1.0
    action_l2_weight: float = 0.02
    barrier_weight: float = 0.002
    fisher_ridge: float = 1e-6
    nb2_dispersion: float = 0.0
    convergence_tolerance: float = 1e-7
    convergence_patience: int = 12
    minimum_information_eigenvalue: float = 1e-6
    maximum_condition_number: float = 1e4


@dataclass(frozen=True)
class InverseTeacherResult:
    action_coordinates: Any
    objective: float
    initial_objective: float
    converged: bool
    iterations: int
    fisher: Any
    covariance: Any
    amplitude_actionable_mask: Any
    phase_actionable_mask: Any
    per_coordinate_actionable_mask: Any
    posterior_std: Any
    diagnostics: Mapping[str, Any]


class ConstrainedInverseTeacher:
    """Optimize a bounded action against a frozen differentiable forward map."""

    def __init__(self, model: Any, config: InverseTeacherConfig | None = None) -> None:
        if torch is None:
            raise RuntimeError("ConstrainedInverseTeacher requires torch")
        self.model = model
        self.config = config or InverseTeacherConfig()
        for parameter in self.model.parameters():
            # Optional lazy branches may remain unmaterialized after a valid
            # forward.  Direct assignment is supported for those placeholders
            # and still guarantees that later materialization is frozen.
            parameter.requires_grad = False
        self.model.eval()

    @staticmethod
    def solve_factorized_batch(
        latent_action_error: Any,
        *,
        coordinate_limit: float = 0.20,
        trust_radius: float = 0.20,
    ) -> np.ndarray:
        """Exact bounded inverse for the frozen identity-action synthetic map."""
        error = np.asarray(latent_action_error, dtype=float)
        if error.ndim != 2 or not np.all(np.isfinite(error)):
            raise ValueError("latent_action_error must be finite [samples, coordinates]")
        limit = max(float(coordinate_limit), 1e-12)
        projected = np.clip(-error, -limit, limit)
        normalized = projected / limit
        norm = np.linalg.norm(normalized, axis=-1, keepdims=True)
        radius = max(float(trust_radius), 0.0)
        scale = np.where(norm > radius, radius / np.maximum(norm, 1e-12), 1.0)
        return (projected * scale).astype(np.float32)

    def solve(
        self,
        observation: Mapping[str, Any],
        command: Any,
        *,
        target_counts: Optional[Any] = None,
        target_amplitude: Optional[Any] = None,
        target_phase_rad: Optional[Any] = None,
        warm_start: Optional[Any] = None,
        forward_context: Optional[ForwardContext] = None,
        actionable_mask: Optional[Any] = None,
    ) -> InverseTeacherResult:
        dimensions = action_dimensions(self.model.system_type)
        dtype = command.tone_features.dtype
        device = command.tone_features.device
        batch_size = command.batch_size
        context = forward_context or self.model.count_forward_model.build_forward_context(
            observation, command,
        )
        if warm_start is None:
            initial = torch.zeros((batch_size, dimensions["total"]), dtype=dtype, device=device)
        else:
            initial = torch.as_tensor(warm_start, dtype=dtype, device=device).reshape(
                batch_size, dimensions["total"],
            )
        mask = torch.ones_like(initial) if actionable_mask is None else torch.as_tensor(
            actionable_mask, dtype=dtype, device=device,
        ).expand_as(initial)
        limits = torch.cat((
            torch.full((dimensions["amplitude"],), float(np.log1p(self.config.amplitude_limit)), dtype=dtype, device=device),
            torch.full((dimensions["phase"],), float(self.config.phase_limit_rad), dtype=dtype, device=device),
        ))
        initial = self._project(initial * mask, limits)
        action = initial.detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([action], lr=float(self.config.learning_rate))
        best_action = action.detach().clone()
        best_value = float("inf")
        initial_value = float("nan")
        patience = 0
        converged = False
        last_value = float("inf")
        components: Dict[str, float] = {}

        for step in range(int(self.config.maximum_steps)):
            optimizer.zero_grad(set_to_none=True)
            loss, current_components = self._objective(
                observation, command, action * mask, context, limits,
                target_counts=target_counts,
                target_amplitude=target_amplitude,
                target_phase_rad=target_phase_rad,
            )
            if not bool(torch.isfinite(loss).detach().cpu()):
                break
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                action.copy_(self._project(action * mask, limits))
            value = float(loss.detach().cpu())
            if step == 0:
                initial_value = value
            if value < best_value:
                best_value = value
                best_action = action.detach().clone()
                components = current_components
            if abs(last_value - value) <= float(self.config.convergence_tolerance):
                patience += 1
                if patience >= int(self.config.convergence_patience):
                    converged = True
                    break
            else:
                patience = 0
            last_value = value

        best_action = self._project(best_action * mask, limits)
        output = self.model.count_forward_model.forward_map(
            observation, command, best_action, forward_context=context,
        )
        jacobian = autograd_log_rate_jacobian(
            self.model.count_forward_model,
            observation,
            command,
            action_point=best_action,
            forward_context=context,
        )
        fisher_result = fisher_from_integrable_count_map(
            output["expected_counts"], jacobian,
            nb2_dispersion=float(self.config.nb2_dispersion),
            ridge=float(self.config.fisher_ridge),
        )
        fisher = fisher_result["fisher"]
        covariance = fisher_result["posterior_covariance"]
        amp_dim = dimensions["amplitude"]
        amp_eigen = torch.linalg.eigvalsh(fisher[..., :amp_dim, :amp_dim])
        phase_eigen = torch.linalg.eigvalsh(fisher[..., amp_dim:, amp_dim:])
        amp_condition = amp_eigen[..., -1] / amp_eigen[..., 0].clamp_min(
            float(self.config.minimum_information_eigenvalue)
        )
        phase_condition = phase_eigen[..., -1] / phase_eigen[..., 0].clamp_min(
            float(self.config.minimum_information_eigenvalue)
        )
        amp_valid = (
            (amp_eigen[..., 0] >= float(self.config.minimum_information_eigenvalue))
            & (amp_condition <= float(self.config.maximum_condition_number))
        )
        phase_valid = (
            (phase_eigen[..., 0] >= float(self.config.minimum_information_eigenvalue))
            & (phase_condition <= float(self.config.maximum_condition_number))
        )
        amp_mask = amp_valid.unsqueeze(-1).expand(-1, amp_dim) & mask[..., :amp_dim].bool()
        phase_mask = phase_valid.unsqueeze(-1).expand(-1, dimensions["phase"]) & mask[..., amp_dim:].bool()
        coordinate_mask = torch.cat((amp_mask, phase_mask), dim=-1)
        final_action = best_action * coordinate_mask.to(best_action.dtype)
        posterior_std = torch.sqrt(torch.diagonal(covariance, dim1=-2, dim2=-1).clamp_min(0.0))
        return InverseTeacherResult(
            action_coordinates=final_action,
            objective=float(best_value),
            initial_objective=float(initial_value),
            converged=bool(converged or best_value + self.config.convergence_tolerance < initial_value),
            iterations=int(step + 1),
            fisher=fisher,
            covariance=covariance,
            amplitude_actionable_mask=amp_mask,
            phase_actionable_mask=phase_mask,
            per_coordinate_actionable_mask=coordinate_mask,
            posterior_std=posterior_std,
            diagnostics={
                **components,
                "fisher_condition_amp": amp_condition.detach(),
                "fisher_condition_phase": phase_condition.detach(),
                "same_forward_context_reused": True,
                "forward_model_frozen": True,
                "brightness_optimized": False,
                "local_residual_optimized": False,
                "research_only": True,
            },
        )

    def _project(self, action: Any, limits: Any) -> Any:
        coordinate_projected = torch.clamp(action, min=-limits, max=limits)
        scaled = coordinate_projected / limits
        norm = torch.linalg.vector_norm(scaled, dim=-1, keepdim=True)
        trust = min(float(self.config.trust_radius), float(np.sqrt(limits.numel())))
        trust_value = torch.as_tensor(trust, dtype=action.dtype, device=action.device)
        trust_scale = torch.where(
            norm > trust_value,
            trust_value / norm.clamp_min(1e-12),
            torch.ones_like(norm),
        )
        return coordinate_projected * trust_scale

    def _objective(
        self,
        observation: Mapping[str, Any],
        command: Any,
        action: Any,
        context: ForwardContext,
        limits: Any,
        *,
        target_counts: Optional[Any],
        target_amplitude: Optional[Any],
        target_phase_rad: Optional[Any],
    ) -> tuple[Any, Dict[str, float]]:
        output = self.model.count_forward_model.forward_map(
            observation, command, action, forward_context=context,
        )
        zero = output["expected_counts"].sum() * 0.0
        count_loss = zero
        if target_counts is not None:
            expected = output["expected_counts"].clamp_min(1e-8)
            target = torch.as_tensor(target_counts, dtype=expected.dtype, device=expected.device).reshape_as(expected)
            expected_profile = expected / expected.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
            target_profile = target / target.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
            count_loss = torch.mean(torch.square(expected_profile - target_profile))
        state_loss = zero
        if target_amplitude is not None and target_phase_rad is not None:
            amplitude = output["shared_state_amplitude"]
            phase = output["shared_state_phase_rad"]
            target_amp = torch.as_tensor(target_amplitude, dtype=amplitude.dtype, device=amplitude.device).reshape_as(amplitude)
            target_phase = torch.as_tensor(target_phase_rad, dtype=phase.dtype, device=phase.device).reshape_as(phase)
            phase_delta = phase - target_phase
            weighted_amplitude = amplitude * target_amp
            overlap_real = torch.sum(weighted_amplitude * torch.cos(phase_delta), dim=-1)
            overlap_imag = torch.sum(weighted_amplitude * torch.sin(phase_delta), dim=-1)
            state_loss = torch.mean(
                1.0 - torch.square(overlap_real) - torch.square(overlap_imag)
            )
        normalized = action / limits
        action_l2 = torch.mean(torch.square(normalized))
        barrier = torch.mean(-torch.log((1.0 - torch.square(normalized)).clamp_min(1e-6)))
        total = (
            float(self.config.target_count_weight) * count_loss
            + float(self.config.target_state_weight) * state_loss
            + float(self.config.action_l2_weight) * action_l2
            + float(self.config.barrier_weight) * barrier
        )
        return total, {
            "target_count_profile_loss": float(count_loss.detach().cpu()),
            "target_state_infidelity": float(state_loss.detach().cpu()),
            "action_l2": float(action_l2.detach().cpu()),
            "barrier": float(barrier.detach().cpu()),
        }


__all__ = [
    "ConstrainedInverseTeacher",
    "InverseTeacherConfig",
    "InverseTeacherResult",
]
