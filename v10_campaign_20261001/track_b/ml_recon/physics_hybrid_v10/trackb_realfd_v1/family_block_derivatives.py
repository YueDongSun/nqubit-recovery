"""Read-only derivatives of the existing family block and unchanged native loss.

This batch-level primitive constructs no optimizer and writes no checkpoint.
A source-bound caller must enforce complete TRAIN coverage before interpreting
an aggregate as the objective of the proposed nine-parameter experiment.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import torch
from torch.func import functional_call

from .effective_observation_loss import Effective_observation_loss_config, effective_observation_objective
from .effective_observation_v2 import Effective_singleton_observation
from .family_loss_decomposition import family_native_loss_contributions
from .training import trackb_model_state_sha


family_parameter_name = 'family_context.raw_coefficients'
constant_loss_names = ('global_multi_nll', 'effective_response_prior', 'prepared_state_prior')


@dataclass(frozen=True)
class Family_derivative_config:
    finite_difference_steps: tuple[float, float] = (0.002, 0.001)

    def validate(self) -> None:
        steps = self.finite_difference_steps
        if type(steps) is not tuple or len(steps) != 2 or any(
            type(value) is not float or not math.isfinite(value) or not 0 < value <= 0.01 for value in steps
        ) or steps[0] != 2 * steps[1]:
            raise ValueError('Two finite positive derivative steps with exact 2:1 ratio are required')


def inspect_family_batch_derivatives(
    model: Effective_singleton_observation, batch: Mapping[str, Any], *,
    loss_config: Effective_observation_loss_config = Effective_observation_loss_config(),
    config: Family_derivative_config = Family_derivative_config(),
) -> dict[str, Any]:
    """Differentiate only external family theta; preserve the supplied model.

The exact native full objective is retained. Two independent central-difference
scales check both its gradient and the gradient's curvature. Numerical errors
are reported, not silently converted into a derivative or scientific PASS.
"""
    config.validate()
    loss_config.validate()
    if type(model) is not Effective_singleton_observation or model.config.measurement_context is None:
        raise TypeError('A typed effective family-context model is required')
    parameters = tuple(model.parameters())
    if any(module.training for module in model.modules()) or any(
        parameter.requires_grad or parameter.grad is not None or parameter.device.type != 'cpu'
        for parameter in parameters
    ) or any(buffer.device.type != 'cpu' for buffer in model.buffers()):
        raise ValueError('Derivative inspection requires a frozen CPU eval model with no stored gradients')
    origin = model.family_context.raw_coefficients.detach().clone()
    if origin.shape != (3, 3) or origin.dtype != torch.float32 or not bool(torch.isfinite(origin).all()):
        raise ValueError('This experiment is registered only for nine CPU FP32 family parameters')
    source_state_sha = trackb_model_state_sha(model)
    source_config_sha = model.config.sha256()
    origin = origin.requires_grad_(True)
    forward_evaluations = 0

    def evaluate(theta):
        nonlocal forward_evaluations
        prediction = functional_call(model, {family_parameter_name: theta}, (batch,), strict=False)
        losses = effective_observation_objective(prediction, batch, config=loss_config)
        forward_evaluations += 1
        return prediction, losses

    try:
        prediction, losses = evaluate(origin)
        gradient = torch.autograd.grad(losses['loss'], origin, create_graph=True)[0]
        if not bool(torch.isfinite(gradient).all()):
            raise FloatingPointError('Family objective gradient is nonfinite')
        contributions = family_native_loss_contributions(
            prediction, batch, origin, registered_families=model.config.measurement_context.family_vocabulary,
        )
        contribution_values = [value for family in contributions.values() for value in family.values()]
        contribution_sum = math.fsum(value['native_objective_contribution'] for value in contribution_values)
        gradient_sum = torch.stack([value['gradient_contribution'] for value in contribution_values]).sum(0)
        flattened_gradient = gradient.reshape(-1)
        hessian = torch.stack([
            torch.autograd.grad(flattened_gradient[index], origin, retain_graph=True)[0].reshape(-1)
            for index in range(origin.numel())
        ]).detach().double()
        analytic_hessian_finite = bool(torch.isfinite(hessian).all())
        finite_checks = []
        maximum_constant_change = 0.0
        maximum_loss_delta_disagreement = 0.0
        for step in config.finite_difference_steps:
            finite_gradient = []
            finite_hessian_columns = []
            for index in range(origin.numel()):
                points = []
                for sign in (1, -1):
                    theta = origin.detach().clone()
                    theta.reshape(-1)[index] += sign * step
                    theta.requires_grad_(True)
                    _, shifted = evaluate(theta)
                    shifted_gradient = torch.autograd.grad(shifted['loss'], theta)[0].detach().double().reshape(-1)
                    if not bool(torch.isfinite(shifted_gradient).all()):
                        raise FloatingPointError('Perturbed family gradient is nonfinite')
                    for name in constant_loss_names:
                        change = float((shifted[name] - losses[name]).detach().abs())
                        maximum_constant_change = max(maximum_constant_change, change)
                        if change != 0:
                            raise ValueError(f'Non-family objective component changed: {name}')
                    disagreement = ((shifted['loss'] - losses['loss'])
                                    - (shifted['four_cell_nll'] - losses['four_cell_nll']))
                    maximum_loss_delta_disagreement = max(maximum_loss_delta_disagreement, float(disagreement.detach().abs()))
                    points.append((theta.detach().reshape(-1)[index].double(), shifted['loss'].detach().double(), shifted_gradient))
                displacement = points[0][0] - points[1][0]
                if not bool(torch.isfinite(displacement)) or not bool(displacement > 0):
                    raise ValueError('Finite-difference perturbation did not move its FP32 coordinate')
                finite_gradient.append((points[0][1] - points[1][1]) / displacement)
                finite_hessian_columns.append((points[0][2] - points[1][2]) / displacement)
            numerical_gradient = torch.stack(finite_gradient)
            numerical_hessian = torch.stack(finite_hessian_columns, dim=1)
            if not bool(torch.isfinite(numerical_gradient).all()) or not bool(torch.isfinite(numerical_hessian).all()):
                raise FloatingPointError('Finite-difference result is nonfinite')
            finite_checks.append({
                'requested_step': step, 'gradient': numerical_gradient,
                'hessian_from_gradient_differences': numerical_hessian,
                'gradient_maximum_absolute_error': float((numerical_gradient - gradient.detach().double().reshape(-1)).abs().max()),
                'hessian_maximum_absolute_error': (float((numerical_hessian - hessian).abs().max()) if analytic_hessian_finite else None),
            })
        return {
            'schema': 'TrackBFamilyBatchDerivativeInspectionV1',
            'scope': 'ONE_BATCH_NATIVE_OBJECTIVE_DERIVATIVES_NOT_COMPLETE_TRAIN_OR_OPTIMIZATION',
            'source_model_state_sha': source_state_sha, 'source_model_config_sha': source_config_sha,
            'parameter_name': family_parameter_name, 'raw_coefficients': origin.detach().clone(),
            'loss_components': {name: float(value.detach()) for name, value in losses.items()},
            'gradient': gradient.detach().double(),
            'family_branch_contributions': contributions,
            'native_loss_decomposition_error': abs(contribution_sum - float(losses['four_cell_nll'].detach())),
            'gradient_decomposition_error': float((gradient_sum - gradient.detach().double()).abs().max()),
            'analytic_hessian': hessian if analytic_hessian_finite else None,
            'analytic_hessian_status': 'FINITE_NOT_QUALIFIED' if analytic_hessian_finite else 'NONFINITE_USE_INDEPENDENT_FINITE_DIFFERENCES_OR_FAIL',
            'finite_difference_checks': finite_checks,
            'gradient_step_disagreement': float((finite_checks[0]['gradient'] - finite_checks[1]['gradient']).abs().max()),
            'hessian_step_disagreement': float((finite_checks[0]['hessian_from_gradient_differences'] - finite_checks[1]['hessian_from_gradient_differences']).abs().max()),
            'maximum_constant_loss_change': maximum_constant_change,
            'maximum_full_vs_four_cell_loss_delta_disagreement': maximum_loss_delta_disagreement,
            'bounded_family_coefficients': prediction['effective_family_coefficients'].detach().clone(),
            'family_correction': prediction['effective_family_correction'].detach().clone(),
            'family_correction_l2': prediction['effective_family_correction_l2'].detach().clone(),
            'forward_evaluations': forward_evaluations, 'optimizer_constructed': False,
            'parameters_updated': False, 'curvature_is_posterior_covariance': False,
        }
    finally:
        if trackb_model_state_sha(model) != source_state_sha or model.config.sha256() != source_config_sha \
                or any(parameter.requires_grad or parameter.grad is not None for parameter in model.parameters()) \
                or any(module.training for module in model.modules()):
            raise RuntimeError('Read-only family inspection changed the frozen source model')
