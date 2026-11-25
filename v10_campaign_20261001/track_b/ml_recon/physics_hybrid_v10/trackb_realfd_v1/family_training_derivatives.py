"""Complete TRAIN-protocol aggregation of read-only family-block derivatives."""
from __future__ import annotations

from dataclasses import asdict
import copy
import math
from typing import Any, Callable, Mapping

import torch

from .contracts import canonical_sha256
from .datasets import trackb_real_static_collate
from .effective_checkpoint_reader import Effective_checkpoint_replay
from .family_block_derivatives import Family_derivative_config, inspect_family_batch_derivatives
from .row_identity import source_row_content_sha


def _validated_training_items(replay: Effective_checkpoint_replay, dataset: Any):
    replay.summary()
    identity = replay.identity
    context = replay.final_model.config.measurement_context
    if context is None or getattr(dataset, 'outer_role', None) != 'train':
        raise ValueError('Family derivative aggregation accepts only an explicit TRAIN fold')
    if dataset.admission_manifest.sha256() != identity.admission_manifest_sha \
            or dataset.fold_plan.sha256() != identity.outer_fold_sha \
            or dataset.row_identity_sha != identity.row_identity_sha \
            or dataset.operation_metadata_sha != context.operation_metadata_sha \
            or dataset.fold_plan.sample_ids_for_role('train') != frozenset(identity.allowed_sample_ids):
        raise ValueError('Derivative TRAIN source differs from the frozen checkpoint admission')
    expected_order = identity.batch_plan_sample_ids
    if len(dataset) != len(expected_order):
        raise ValueError('Derivative source must contain the complete checkpoint TRAIN fold')
    expected_rows = {sample: (group, content, role) for sample, group, content, role in identity.allowed_row_identities}
    expected_sessions = dict(identity.session_index_bindings)
    items = []
    for index, sample in enumerate(expected_order):
        item = copy.deepcopy(dataset[index])
        if item['sample_id'] != sample or item.get('outer_role') != 'train' or item.get('source_role') != 'train' \
                or (item['group_id'], item['row_content_sha'], item['source_role']) != expected_rows[sample] \
                or int(item['session_index']) != expected_sessions[sample] \
                or source_row_content_sha(item) != item['row_content_sha']:
            raise ValueError('Derivative row order, identity, role or session differs from the registered TRAIN plan')
        items.append(item)
    batch_count = (len(items) + identity.batch_plan_batch_size - 1) // identity.batch_plan_batch_size
    if replay.runtime_config.optimizer_steps != batch_count:
        raise ValueError('This derivative experiment requires the registered one-pass pilot endpoint')
    return items


def _family_contribution_aggregates(reports, families):
    output = {family: {} for family in families}
    count = len(reports)
    for family in families:
        for branch in reports[0]['family_branch_contributions'][family]:
            values = [report['family_branch_contributions'][family][branch] for report in reports]
            selected = sum(value['selected_queries'] for value in values)
            informative = sum(value['informative_queries'] for value in values)
            nll_sum = math.fsum(value['native_nll_sum'] for value in values)
            output[family][branch] = {
                'selected_queries': selected, 'informative_queries': informative,
                'zero_parent_queries': selected - informative,
                'native_objective_contribution': math.fsum(value['native_objective_contribution'] for value in values) / count,
                'gradient_contribution': torch.stack([value['gradient_contribution'] for value in values]).mean(0),
                'native_nll_sum': nll_sum,
                'family_query_mean_nll': None if not selected else nll_sum / selected,
                'status': 'NO_SELECTED_QUERIES' if not selected else 'ZERO_PARENT_ONLY' if not informative else 'INFORMATIVE',
            }
    return output


def inspect_family_training_derivatives(
    replay: Effective_checkpoint_replay, dataset: Any, *,
    config: Family_derivative_config = Family_derivative_config(),
    progress: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Use every epoch-zero registered TRAIN batch, with no optimizer or fitting.

Equal batch weighting matches the registered one-pass optimizer objective;
it is not the equal-component weighting of held-out descriptive reports.
The caller owns the admitted dataset and the exact reader source identity.
"""
    config.validate()
    items = _validated_training_items(replay, dataset)
    identity = replay.identity
    size = identity.batch_plan_batch_size
    families = replay.final_model.config.measurement_context.family_vocabulary
    reports = []
    batch_bindings = []
    correction_values = {family: [] for family in families}
    try:
        for batch_index, start in enumerate(range(0, len(items), size)):
            batch = trackb_real_static_collate(items[start:start + size], epoch=0,
                view_seed=identity.batch_plan_view_seed, query_fraction=identity.batch_plan_query_fraction,
                batch_plan_sha=identity.batch_plan_sha, batch_index=batch_index, cursor_start=start)
            report = inspect_family_batch_derivatives(replay.final_model, batch,
                loss_config=replay.loss_config, config=config)
            if report['source_model_state_sha'] != replay.final_model_state_sha:
                raise ValueError('Derivative batch used a different model state')
            reports.append(report)
            binding = {'batch_index': batch_index, 'cursor_start': start, 'row_count': len(batch['sample_ids']),
                'query_count': int(batch['query_mask'].sum()), 'query_mask_sha': batch['query_mask_sha']}
            batch_bindings.append(binding)
            for family in families:
                mask = batch['operation_mask'] & torch.tensor(
                    [value == family for value in batch['operation_families']], dtype=torch.bool)[None, :]
                correction_values[family].append(report['family_correction'][mask].double())
            if progress is not None:
                progress({**binding, 'completed_batches': len(reports),
                    'forward_evaluations': sum(value['forward_evaluations'] for value in reports),
                    'analytic_hessian_status': report['analytic_hessian_status']})
        count = len(reports)
        gradient = torch.stack([report['gradient'] for report in reports]).mean(0).reshape(-1)
        analytic = None if any(report['analytic_hessian'] is None for report in reports) else torch.stack(
            [report['analytic_hessian'] for report in reports]).mean(0)
        finite_checks = []
        for index, step in enumerate(config.finite_difference_steps):
            values = [report['finite_difference_checks'][index] for report in reports]
            numerical_gradient = torch.stack([value['gradient'] for value in values]).mean(0)
            numerical_hessian = torch.stack([value['hessian_from_gradient_differences'] for value in values]).mean(0)
            finite_checks.append({'requested_step': step, 'gradient': numerical_gradient,
                'hessian_from_gradient_differences': numerical_hessian,
                'gradient_maximum_absolute_error': float((numerical_gradient - gradient).abs().max()),
                'hessian_maximum_absolute_error': None if analytic is None else float((numerical_hessian - analytic).abs().max()),
                'hessian_asymmetry_maximum': float((numerical_hessian - numerical_hessian.T).abs().max()),
                'symmetrized_hessian_eigenvalues': torch.linalg.eigvalsh((numerical_hessian + numerical_hessian.T) / 2)})
        contributions = _family_contribution_aggregates(reports, families)
        contribution_values = [value for family in contributions.values() for value in family.values()]
        average_losses = {name: math.fsum(report['loss_components'][name] for report in reports) / count
            for name in reports[0]['loss_components']}
        correction_summary = {}
        for family, batches in correction_values.items():
            values = torch.cat(batches)
            correction_summary[family] = {'available_operations': len(values),
                'minimum': None if not len(values) else values.min(0).values,
                'maximum': None if not len(values) else values.max(0).values,
                'mean': None if not len(values) else values.mean(0)}
        return {
            'schema': 'TrackBCompleteTrainFamilyDerivativeReportV1',
            'status': 'COMPUTED_REQUIRES_NUMERICAL_REVIEW_NOT_OPTIMIZATION_OR_CALIBRATION',
            'checkpoint_file_sha': replay.file_sha, 'checkpoint_identity_sha': identity.sha256(),
            'producer_commit': identity.source_commit, 'producer_tree': identity.source_tree,
            'model_config_sha': identity.model_config_sha, 'model_state_sha': replay.final_model_state_sha,
            'admission_manifest_sha': identity.admission_manifest_sha, 'outer_fold_sha': identity.outer_fold_sha,
            'row_identity_sha': identity.row_identity_sha, 'operation_metadata_sha': dataset.operation_metadata_sha,
            'batch_plan_sha': identity.batch_plan_sha, 'batch_bindings_sha': canonical_sha256(batch_bindings),
            'derivative_config': asdict(config), 'outer_role': 'train', 'epoch': 0,
            'aggregation': 'EQUAL_REGISTERED_TRAIN_BATCHES_WITH_UNCHANGED_NATIVE_OBJECTIVE',
            'complete_rows': len(items), 'complete_components': len(dataset.fold_plan.components_for_role('train')),
            'batch_count': count, 'batch_size': size, 'query_operations': sum(value['query_count'] for value in batch_bindings),
            'forward_evaluations': sum(report['forward_evaluations'] for report in reports),
            'mean_native_loss_components': average_losses,
            'global_available_rows_total': sum(report['loss_components']['global_multi_available_rows'] for report in reports),
            'gradient': gradient, 'gradient_norm': float(torch.linalg.vector_norm(gradient)),
            'analytic_hessian': analytic,
            'analytic_hessian_nonfinite_batches': sum(report['analytic_hessian'] is None for report in reports),
            'finite_difference_checks': finite_checks,
            'gradient_step_disagreement': float((finite_checks[0]['gradient'] - finite_checks[1]['gradient']).abs().max()),
            'hessian_step_disagreement': float((finite_checks[0]['hessian_from_gradient_differences'] - finite_checks[1]['hessian_from_gradient_differences']).abs().max()),
            'family_branch_contributions': contributions,
            'native_loss_decomposition_error': abs(math.fsum(value['native_objective_contribution'] for value in contribution_values) - average_losses['four_cell_nll']),
            'gradient_decomposition_error': float((torch.stack([value['gradient_contribution'] for value in contribution_values]).sum(0).reshape(-1) - gradient).abs().max()),
            'raw_coefficients': reports[0]['raw_coefficients'], 'bounded_coefficients': reports[0]['bounded_family_coefficients'],
            'correction_by_family': correction_summary,
            'maximum_correction_l2_by_branch': torch.cat([report['family_correction_l2'].double() for report in reports]).max(0).values,
            'maximum_constant_loss_change': max(report['maximum_constant_loss_change'] for report in reports),
            'maximum_full_vs_four_cell_loss_delta_disagreement': max(report['maximum_full_vs_four_cell_loss_delta_disagreement'] for report in reports),
            'batch_bindings': batch_bindings,
            'optimizer_constructed': False, 'parameters_updated': False, 'development_rows_used': 0,
            'curvature_is_posterior_covariance': False, 'scientific_gate_issued': False,
        }
    finally:
        replay.summary()


def family_derivative_json_payload(value: Any) -> Any:
    """Convert only finite result tensors to ordinary canonical-JSON values."""
    if isinstance(value, torch.Tensor):
        if not bool(torch.isfinite(value).all()):
            raise ValueError('Derivative report contains a nonfinite tensor')
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {name: family_derivative_json_payload(item) for name, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [family_derivative_json_payload(item) for item in value]
    return value
