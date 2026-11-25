"""Strict parameter-overlay contract, distinct from an AdamW checkpoint."""
from __future__ import annotations

import math
from typing import Any

import torch

from .contracts import require_sha256
from .family_block_solver import Family_block_solver_config, solver_method, terminal_reasons


result_fields = frozenset({'method', 'solver_config', 'training_protocol', 'initial_theta', 'final_theta',
    'initial_loss_components', 'final_loss_components', 'initial_gradient', 'final_gradient', 'accepted_steps',
    'accepted_trace', 'stop_reason', 'terminal_line_search_trials', 'objective_evaluations', 'forward_evaluations',
    'optimization_executed', 'parent_model_updated', 'full_network_optimization', 'development_rows_used',
    'curvature_is_posterior_covariance', 'scientific_gate_issued'})
loss_fields = frozenset({'loss', 'four_cell_nll', 'global_multi_nll', 'global_multi_available_rows',
    'effective_response_prior', 'prepared_state_prior'})
trace_fields = frozenset({'accepted_step', 'theta_before', 'theta_after', 'loss_before', 'loss_after',
    'four_cell_before', 'four_cell_after', 'damping', 'step_norm', 'directional_derivative', 'gradient_before',
    'gradient_after', 'hessian_before', 'accepted_backtrack', 'line_search_trials', 'objective_evaluations'})
trial_fields = frozenset({'backtrack', 'scale', 'status', 'step_norm', 'directional_derivative', 'loss'})
protocol_fields = frozenset({'schema', 'parent_checkpoint_file_sha', 'parent_identity_sha', 'parent_model_state_sha',
    'parent_producer_commit', 'parent_producer_tree', 'model_config_sha', 'admission_manifest_sha', 'outer_fold_sha',
    'row_identity_sha', 'operation_metadata_sha', 'batch_plan_sha', 'batch_bindings_sha', 'outer_role', 'epoch',
    'view_seed', 'query_fraction', 'batch_size', 'batch_count', 'complete_rows', 'complete_components',
    'query_operations', 'aggregation'})


def finite_float(value, name):
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite float without coercion')
    return value


def checked_vector(value, name, *, fp32=False):
    if type(value) is not list or len(value) != 9:
        raise ValueError(f'{name} must contain exactly nine values')
    for item in value:
        finite_float(item, name)
    tensor = torch.tensor(value, dtype=torch.float32 if fp32 else torch.float64)
    if not bool(torch.isfinite(tensor).all()) or (fp32 and tensor.tolist() != value):
        raise ValueError(f'{name} must preserve exact finite FP32 values')
    return tensor


def _checked_losses(value):
    if type(value) is not dict or set(value) != loss_fields:
        raise ValueError('Refinement loss-component fields differ from the native objective')
    for name, item in value.items():
        finite_float(item, name)
    return value


def _checked_trials(trials, config, *, accepted, reference_loss):
    if type(trials) is not list or len(trials) > config.maximum_backtracking_steps or (accepted and not trials):
        raise ValueError('Line-search trace is missing or exceeds its bound')
    evaluations = 0
    for index, trial in enumerate(trials):
        if type(trial) is not dict or set(trial) != trial_fields or type(trial['backtrack']) is not int \
                or trial['backtrack'] != index or trial['scale'] != config.backtracking_factor ** index:
            raise ValueError('Line-search trial order/scale is invalid')
        finite_float(trial['scale'], 'line-search scale')
        norm = finite_float(trial['step_norm'], 'line-search norm')
        slope = finite_float(trial['directional_derivative'], 'line-search slope')
        if norm < 0:
            raise ValueError('Line-search norm cannot be negative')
        status = trial['status']
        if status == 'REJECTED_REPRESENTABLE_STEP':
            if trial['loss'] is not None or (0 < norm <= config.maximum_step_norm and slope < 0):
                raise ValueError('Representable-step rejection is inconsistent')
        elif status in {'VALUE_ACCEPTABLE', 'REJECTED_NATIVE_OBJECTIVE'}:
            finite_float(trial['loss'], 'line-search native loss')
            if not 0 < norm <= config.maximum_step_norm or slope >= 0:
                raise ValueError('Evaluated line-search trial is not bounded descent')
            if status == 'VALUE_ACCEPTABLE' and (not accepted or index != len(trials) - 1):
                raise ValueError('Only the final committed line-search trial can be acceptable')
            meets_acceptance = trial['loss'] <= reference_loss + config.armijo_constant * slope \
                and reference_loss - trial['loss'] >= config.minimum_native_decrease
            if meets_acceptance != (status == 'VALUE_ACCEPTABLE'):
                raise ValueError('Recorded line-search classification contradicts the native objective')
            evaluations += 1
        else:
            raise ValueError('Unknown line-search trial status')
    if accepted and trials[-1]['status'] != 'VALUE_ACCEPTABLE':
        raise ValueError('Accepted step has no accepted value trial')
    return evaluations


def validate_family_refinement_result(result: Any, parent) -> Family_block_solver_config:
    """Validate recorded bounds/lineage; actual TRAIN loss needs separate replay."""
    parent.summary()
    if type(result) is not dict or set(result) != result_fields or result['method'] != solver_method:
        raise ValueError('Unregistered family-refinement result schema')
    config = Family_block_solver_config.from_dict(result['solver_config'])
    for name, expected in {'optimization_executed': True, 'parent_model_updated': False,
        'full_network_optimization': False, 'curvature_is_posterior_covariance': False, 'scientific_gate_issued': False}.items():
        if result[name] is not expected:
            raise ValueError('Refinement authority or frozen-parent flags are invalid')
    if type(result['development_rows_used']) is not int or result['development_rows_used'] != 0:
        raise ValueError('Refinement cannot fit development rows')
    protocol = result['training_protocol']
    identity = parent.identity
    context = parent.final_model.config.measurement_context
    model_config = parent.final_model.config
    if not model_config.structure_enabled or model_config.read_trials_source != 'native_scalar' \
            or model_config.pair_residual_enabled or model_config.no_signal_structure_enabled:
        raise ValueError('Refinement parent is outside the registered native fixed-pair protocol')
    if context is None or type(protocol) is not dict or set(protocol) != protocol_fields:
        raise ValueError('Refinement requires exact V3 TRAIN protocol fields')
    expected = {'schema': 'TrackBFamilyRefinementTrainingProtocolV1',
        'parent_checkpoint_file_sha': parent.file_sha, 'parent_identity_sha': identity.sha256(),
        'parent_model_state_sha': parent.final_model_state_sha, 'parent_producer_commit': identity.source_commit,
        'parent_producer_tree': identity.source_tree, 'model_config_sha': identity.model_config_sha,
        'admission_manifest_sha': identity.admission_manifest_sha, 'outer_fold_sha': identity.outer_fold_sha,
        'row_identity_sha': identity.row_identity_sha, 'operation_metadata_sha': context.operation_metadata_sha,
        'batch_plan_sha': identity.batch_plan_sha, 'outer_role': 'train', 'epoch': 0,
        'view_seed': identity.batch_plan_view_seed, 'query_fraction': identity.batch_plan_query_fraction,
        'batch_size': identity.batch_plan_batch_size, 'complete_rows': len(identity.batch_plan_sample_ids),
        'batch_count': parent.runtime_config.optimizer_steps,
        'aggregation': 'EQUAL_REGISTERED_TRAIN_BATCHES_WITH_UNCHANGED_NATIVE_OBJECTIVE'}
    if any(type(protocol[name]) is not type(value) or protocol[name] != value for name, value in expected.items()):
        raise ValueError('Refinement TRAIN provenance differs from its frozen parent')
    require_sha256(protocol['batch_bindings_sha'], 'batch_bindings_sha')
    if type(protocol['complete_components']) is not int or not 1 <= protocol['complete_components'] <= protocol['complete_rows'] \
            or type(protocol['query_operations']) is not int or protocol['query_operations'] < protocol['complete_rows']:
        raise ValueError('Refinement complete-data denominators are invalid')
    current_theta = checked_vector(result['initial_theta'], 'initial_theta', fp32=True)
    origin = parent.final_model.family_context.raw_coefficients.detach().reshape(-1)
    if not torch.equal(current_theta.view(torch.uint8), origin.view(torch.uint8)):
        raise ValueError('Refinement initial theta is not the frozen parent')
    current_gradient = checked_vector(result['initial_gradient'], 'initial_gradient')
    initial_losses = _checked_losses(result['initial_loss_components'])
    final_losses = _checked_losses(result['final_loss_components'])
    current_loss = initial_losses['loss']
    current_four_cell = initial_losses['four_cell_nll']
    trace = result['accepted_trace']
    if type(trace) is not list or type(result['accepted_steps']) is not int \
            or result['accepted_steps'] != len(trace) or len(trace) > config.maximum_accepted_steps:
        raise ValueError('Refinement accepted-step count is invalid')
    evaluations = 1
    for index, step in enumerate(trace):
        if type(step) is not dict or set(step) != trace_fields or type(step['accepted_step']) is not int or step['accepted_step'] != index + 1:
            raise ValueError('Refinement step fields/order are invalid')
        before = checked_vector(step['theta_before'], 'theta_before', fp32=True)
        after = checked_vector(step['theta_after'], 'theta_after', fp32=True)
        gradient = checked_vector(step['gradient_before'], 'gradient_before')
        current_gradient_after = checked_vector(step['gradient_after'], 'gradient_after')
        if not torch.equal(before.view(torch.uint8), current_theta.view(torch.uint8)) or not torch.equal(gradient, current_gradient) \
                or step['loss_before'] != current_loss or step['four_cell_before'] != current_four_cell:
            raise ValueError('Refinement trace is not continuous from its parent')
        if type(step['hessian_before']) is not list or len(step['hessian_before']) != 9:
            raise ValueError('Refinement Hessian shape is invalid')
        for row in step['hessian_before']:
            checked_vector(row, 'hessian row')
        for name in ('loss_before', 'loss_after', 'four_cell_before', 'four_cell_after', 'damping', 'step_norm', 'directional_derivative'):
            finite_float(step[name], name)
        displacement = after.double() - before.double()
        norm = float(torch.linalg.vector_norm(displacement))
        slope = float(gradient @ displacement)
        if not 0 < norm <= config.maximum_step_norm or slope >= 0 or step['damping'] < config.minimum_damping \
                or not math.isclose(norm, step['step_norm'], rel_tol=1e-10, abs_tol=1e-12) \
                or not math.isclose(slope, step['directional_derivative'], rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError('Refinement committed step is not bounded descent')
        if step['loss_after'] > current_loss + config.armijo_constant * slope \
                or current_loss - step['loss_after'] < config.minimum_native_decrease:
            raise ValueError('Refinement committed a nondecreasing native objective')
        evaluations += _checked_trials(step['line_search_trials'], config, accepted=True, reference_loss=current_loss) + 1
        accepted_trial = step['line_search_trials'][-1]
        if type(step['accepted_backtrack']) is not int or step['accepted_backtrack'] != accepted_trial['backtrack'] \
                or accepted_trial['loss'] != step['loss_after'] or type(step['objective_evaluations']) is not int \
                or step['objective_evaluations'] != evaluations or accepted_trial['step_norm'] != step['step_norm'] \
                or accepted_trial['directional_derivative'] != step['directional_derivative']:
            raise ValueError('Refinement confirmation/work trace is inconsistent')
        if not math.isclose(current_loss - step['loss_after'], current_four_cell - step['four_cell_after'], rel_tol=1e-8, abs_tol=1e-10):
            raise ValueError('Family-only objective change differs from its native four-cell change')
        current_theta, current_gradient = after, current_gradient_after
        current_loss, current_four_cell = step['loss_after'], step['four_cell_after']
    final_theta = checked_vector(result['final_theta'], 'final_theta', fp32=True)
    final_gradient = checked_vector(result['final_gradient'], 'final_gradient')
    if not torch.equal(final_theta.view(torch.uint8), current_theta.view(torch.uint8)) or not torch.equal(final_gradient, current_gradient) \
            or final_losses['loss'] != current_loss or final_losses['four_cell_nll'] != current_four_cell:
        raise ValueError('Refinement final point does not match its accepted trace')
    for name in ('global_multi_nll', 'effective_response_prior', 'prepared_state_prior', 'global_multi_available_rows'):
        if final_losses[name] != initial_losses[name]:
            raise ValueError('Refinement changed non-family objective components')
    evaluations += _checked_trials(result['terminal_line_search_trials'], config, accepted=False, reference_loss=current_loss)
    if type(result['objective_evaluations']) is not int or result['objective_evaluations'] != evaluations \
            or evaluations > config.maximum_objective_evaluations or type(result['forward_evaluations']) is not int \
            or result['forward_evaluations'] != evaluations * protocol['batch_count']:
        raise ValueError('Refinement full-TRAIN work accounting is invalid')
    reason = result['stop_reason']
    if type(reason) is not str or reason not in terminal_reasons or (reason == 'GRADIENT_TOLERANCE' and float(final_gradient.abs().max()) > config.gradient_infinity_tolerance) \
            or (reason == 'MAXIMUM_ACCEPTED_STEPS' and len(trace) != config.maximum_accepted_steps) \
            or (reason == 'EVALUATION_BUDGET' and evaluations + 2 <= config.maximum_objective_evaluations):
        raise ValueError('Refinement stop reason does not match its recorded state')
    if (reason in {'GRADIENT_TOLERANCE', 'MAXIMUM_ACCEPTED_STEPS'} and result['terminal_line_search_trials']) \
            or (reason == 'NO_ACCEPTABLE_NATIVE_DECREASE' and (
                len(result['terminal_line_search_trials']) != config.maximum_backtracking_steps
                or float(final_gradient.abs().max()) <= config.gradient_infinity_tolerance)):
        raise ValueError('Refinement terminal trials do not match its stop condition')
    return config
