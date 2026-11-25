"""Bounded damped-Newton refinement of only the nine family parameters."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import math
from typing import Any, Callable, Mapping

import torch

from .family_block_objective import Frozen_family_training_objective


solver_method = 'DAMPED_NEWTON_ARMIJO_NATIVE_TRAIN_FP32_THETA_V1'
terminal_reasons = frozenset({'GRADIENT_TOLERANCE', 'MAXIMUM_ACCEPTED_STEPS', 'EVALUATION_BUDGET', 'NO_ACCEPTABLE_NATIVE_DECREASE'})


@dataclass(frozen=True)
class Family_block_solver_config:
    maximum_accepted_steps: int = 12
    maximum_backtracking_steps: int = 12
    maximum_objective_evaluations: int = 170
    maximum_step_norm: float = 0.25
    minimum_damping: float = 0.001
    minimum_curvature_eigenvalue: float = 0.0001
    gradient_infinity_tolerance: float = 0.000001
    minimum_native_decrease: float = 0.0000000001
    armijo_constant: float = 0.0001
    backtracking_factor: float = 0.5
    schema: str = 'TrackBFamilyBlockSolverConfigV1'

    def validate(self):
        for name, limit in (('maximum_accepted_steps', 32), ('maximum_backtracking_steps', 16), ('maximum_objective_evaluations', 600)):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= limit:
                raise ValueError(f'{name} exceeds the registered bounded solver domain')
        if self.maximum_objective_evaluations < 3:
            raise ValueError('Solver budget must allow initial and confirmed candidate evaluation')
        for name in ('maximum_step_norm', 'minimum_damping', 'minimum_curvature_eigenvalue',
                     'gradient_infinity_tolerance', 'minimum_native_decrease', 'armijo_constant', 'backtracking_factor'):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite positive float')
        if self.schema != 'TrackBFamilyBlockSolverConfigV1' or self.maximum_step_norm > 0.25 \
                or self.armijo_constant >= 0.5 or self.backtracking_factor >= 1 \
                or self.gradient_infinity_tolerance > 0.0001 or self.minimum_native_decrease > 0.000001:
            raise ValueError('Unregistered family solver step or line-search contract')

    @classmethod
    def from_dict(cls, payload):
        if type(payload) is not dict or set(payload) != {field.name for field in fields(cls)}:
            raise ValueError('Solver configuration fields must match exactly')
        result = cls(**payload)
        result.validate()
        return result


def _newton_direction(evaluation, config):
    gradient = evaluation['gradient']
    hessian = evaluation['hessian']
    symmetric = (hessian + hessian.T) / 2
    smallest = float(torch.linalg.eigvalsh(symmetric).min())
    damping = max(config.minimum_damping, config.minimum_curvature_eigenvalue - smallest)
    direction = -torch.linalg.solve(symmetric + damping * torch.eye(9, dtype=torch.float64), gradient)
    norm = float(torch.linalg.vector_norm(direction))
    if not math.isfinite(norm) or not bool(torch.isfinite(direction).all()) or float(gradient @ direction) >= 0:
        raise FloatingPointError('Damped native family direction is not finite descent')
    if norm > config.maximum_step_norm:
        direction = direction * (config.maximum_step_norm / norm)
    return direction, damping


def refine_family_block(
    objective: Frozen_family_training_objective, *,
    config: Family_block_solver_config = Family_block_solver_config(),
    progress: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Accept only confirmed full-TRAIN decreases; never overwrite the parent.

Candidate value checks and accepted-point derivatives use the same original
native objective. A failed confirmation raises before a step is recorded.
This is parameter optimization, not an AdamW continuation or a scientific Gate.
"""
    config.validate()
    if objective.evaluations != 0:
        raise ValueError('A solver attempt requires a fresh objective owner')
    theta = objective.initial_theta
    initial = objective.evaluate(theta, derivatives=True)
    current = initial
    trace = []
    terminal_trials = []
    reason = 'MAXIMUM_ACCEPTED_STEPS'
    for _ in range(config.maximum_accepted_steps):
        if float(current['gradient'].abs().max()) <= config.gradient_infinity_tolerance:
            reason = 'GRADIENT_TOLERANCE'
            break
        if objective.evaluations + 2 > config.maximum_objective_evaluations:
            reason = 'EVALUATION_BUDGET'
            break
        direction, damping = _newton_direction(current, config)
        accepted = False
        trials = []
        for backtrack in range(config.maximum_backtracking_steps):
            if objective.evaluations + 2 > config.maximum_objective_evaluations:
                reason = 'EVALUATION_BUDGET'
                break
            scale = config.backtracking_factor ** backtrack
            proposed = (theta.double() + scale * direction).float()
            displacement = proposed.double() - theta.double()
            step_norm = float(torch.linalg.vector_norm(displacement))
            slope = float(current['gradient'] @ displacement)
            # Account for representable FP32 steps, not just the requested vector.
            if step_norm == 0 or step_norm > config.maximum_step_norm or slope >= 0:
                trials.append({'backtrack': backtrack, 'scale': scale, 'status': 'REJECTED_REPRESENTABLE_STEP',
                    'step_norm': step_norm, 'directional_derivative': slope, 'loss': None})
                continue
            candidate = objective.evaluate(proposed, derivatives=False)
            value = candidate['loss_components']['loss']
            old_value = current['loss_components']['loss']
            acceptable = value <= old_value + config.armijo_constant * slope \
                and old_value - value >= config.minimum_native_decrease
            trials.append({'backtrack': backtrack, 'scale': scale, 'status': 'VALUE_ACCEPTABLE' if acceptable else 'REJECTED_NATIVE_OBJECTIVE',
                'step_norm': step_norm, 'directional_derivative': slope, 'loss': value})
            if not acceptable:
                continue
            confirmed = objective.evaluate(proposed, derivatives=True)
            if confirmed['loss_components'] != candidate['loss_components']:
                raise ArithmeticError('Candidate native objective changed during derivative confirmation')
            trace.append({'accepted_step': len(trace) + 1, 'theta_before': theta.tolist(),
                'theta_after': proposed.tolist(), 'loss_before': old_value, 'loss_after': value,
                'four_cell_before': current['loss_components']['four_cell_nll'],
                'four_cell_after': confirmed['loss_components']['four_cell_nll'],
                'damping': damping, 'step_norm': step_norm, 'directional_derivative': slope,
                'gradient_before': current['gradient'].tolist(), 'gradient_after': confirmed['gradient'].tolist(),
                'hessian_before': current['hessian'].tolist(), 'accepted_backtrack': backtrack,
                'line_search_trials': trials, 'objective_evaluations': objective.evaluations})
            theta, current, accepted = proposed, confirmed, True
            if progress is not None:
                progress({'accepted_steps': len(trace), 'loss': value,
                    'gradient_infinity_norm': float(current['gradient'].abs().max()),
                    'objective_evaluations': objective.evaluations, 'forward_evaluations': objective.forward_evaluations})
            break
        if not accepted:
            terminal_trials = trials
            if reason != 'EVALUATION_BUDGET':
                reason = 'NO_ACCEPTABLE_NATIVE_DECREASE'
            break
    objective.parent.summary()
    return {'method': solver_method, 'solver_config': asdict(config), 'training_protocol': objective.binding,
        'initial_theta': initial['theta'].tolist(), 'final_theta': theta.tolist(),
        'initial_loss_components': initial['loss_components'], 'final_loss_components': current['loss_components'],
        'initial_gradient': initial['gradient'].tolist(), 'final_gradient': current['gradient'].tolist(),
        'accepted_steps': len(trace), 'accepted_trace': trace, 'stop_reason': reason,
        'terminal_line_search_trials': terminal_trials,
        'objective_evaluations': objective.evaluations, 'forward_evaluations': objective.forward_evaluations,
        'optimization_executed': True, 'parent_model_updated': False, 'full_network_optimization': False,
        'development_rows_used': 0, 'curvature_is_posterior_covariance': False, 'scientific_gate_issued': False}
