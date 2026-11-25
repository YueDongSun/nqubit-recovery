"""Single-owner frozen-model objective on the complete original TRAIN protocol."""
from __future__ import annotations

import copy
import math
from typing import Any, Iterator

import torch
from torch.func import functional_call

from .contracts import canonical_sha256
from .datasets import trackb_real_static_collate
from .effective_checkpoint_reader import Effective_checkpoint_replay
from .effective_observation_loss import effective_observation_objective
from .family_block_derivatives import constant_loss_names, family_parameter_name
from .family_training_derivatives import _validated_training_items


class Frozen_family_training_objective:
    """Evaluate external FP32 theta without updating the original checkpoint model.

This object is intentionally single-owner: functional module substitution is
not a concurrent inference interface. No held-out data or model optimizer is
constructed. The complete native objective, including constant terms, is used.
"""

    def __init__(self, replay: Effective_checkpoint_replay, dataset: Any):
        model_config = replay.final_model.config
        if not model_config.structure_enabled or model_config.read_trials_source != 'native_scalar' \
                or model_config.pair_residual_enabled or model_config.no_signal_structure_enabled:
            raise ValueError('Refinement requires the registered structural native-R fixed-pair V3 parent')
        items = _validated_training_items(replay, dataset)
        self.parent = replay
        identity = replay.identity
        self._origin = replay.final_model.family_context.raw_coefficients.detach().clone().reshape(-1)
        if self._origin.shape != (9,) or self._origin.dtype != torch.float32 or self._origin.device.type != 'cpu':
            raise ValueError('Family refinement requires exactly nine CPU FP32 coefficients')
        size = identity.batch_plan_batch_size
        self._batches = tuple(trackb_real_static_collate(items[start:start + size], epoch=0,
            view_seed=identity.batch_plan_view_seed, query_fraction=identity.batch_plan_query_fraction,
            batch_plan_sha=identity.batch_plan_sha, batch_index=index, cursor_start=start)
            for index, start in enumerate(range(0, len(items), size)))
        bindings = [{'batch_index': index, 'cursor_start': index * size,
            'row_count': len(batch['sample_ids']), 'query_count': int(batch['query_mask'].sum()),
            'query_mask_sha': batch['query_mask_sha']} for index, batch in enumerate(self._batches)]
        self._binding = {
            'schema': 'TrackBFamilyRefinementTrainingProtocolV1',
            'parent_checkpoint_file_sha': replay.file_sha, 'parent_identity_sha': identity.sha256(),
            'parent_model_state_sha': replay.final_model_state_sha,
            'parent_producer_commit': identity.source_commit, 'parent_producer_tree': identity.source_tree,
            'model_config_sha': identity.model_config_sha, 'admission_manifest_sha': identity.admission_manifest_sha,
            'outer_fold_sha': identity.outer_fold_sha, 'row_identity_sha': identity.row_identity_sha,
            'operation_metadata_sha': dataset.operation_metadata_sha, 'batch_plan_sha': identity.batch_plan_sha,
            'batch_bindings_sha': canonical_sha256(bindings), 'outer_role': 'train', 'epoch': 0,
            'view_seed': identity.batch_plan_view_seed, 'query_fraction': identity.batch_plan_query_fraction,
            'batch_size': size, 'batch_count': len(self._batches), 'complete_rows': len(items),
            'complete_components': len(dataset.fold_plan.components_for_role('train')),
            'query_operations': sum(value['query_count'] for value in bindings),
            'aggregation': 'EQUAL_REGISTERED_TRAIN_BATCHES_WITH_UNCHANGED_NATIVE_OBJECTIVE',
        }
        self._constant_values = None
        self.evaluations = 0
        self.forward_evaluations = 0

    @property
    def initial_theta(self):
        return self._origin.clone()

    @property
    def binding(self):
        return copy.deepcopy(self._binding)

    def iter_registered_batches(self) -> Iterator[dict[str, Any]]:
        """Yield owned copies of the original TRAIN views for frozen diagnostics.

        Callers cannot modify the objective's cached training inputs. This
        access does not evaluate the network or authorize an optimizer.
        """
        self.parent.summary()
        try:
            for batch in self._batches:
                yield copy.deepcopy(batch)
        finally:
            self.parent.summary()

    @torch.enable_grad()
    def evaluate(self, theta: torch.Tensor, *, derivatives: bool) -> dict[str, Any]:
        if type(derivatives) is not bool or not isinstance(theta, torch.Tensor) or theta.shape != (9,) \
                or theta.dtype != torch.float32 or theta.device.type != 'cpu' or not bool(torch.isfinite(theta).all()):
            raise ValueError('Family objective requires one finite CPU FP32 nine-vector')
        self.parent.summary()
        point = theta.detach().clone().reshape(3, 3).requires_grad_(derivatives)
        gradients, hessians, values, constants = [], [], [], []
        self.evaluations += 1
        try:
            for index, batch in enumerate(self._batches):
                prediction = functional_call(self.parent.final_model, {family_parameter_name: point}, (batch,), strict=False)
                losses = effective_observation_objective(prediction, batch, config=self.parent.loss_config)
                self.forward_evaluations += 1
                row = {name: float(value.detach()) for name, value in losses.items()}
                values.append(row)
                fixed = tuple(row[name] for name in constant_loss_names)
                constants.append(fixed)
                if self._constant_values is not None and fixed != self._constant_values[index]:
                    raise ValueError('Global or weak-prior objective changed during family-only refinement')
                if derivatives:
                    gradient = torch.autograd.grad(losses['loss'], point, create_graph=True)[0].reshape(-1)
                    hessian = torch.stack([torch.autograd.grad(gradient[column], point, retain_graph=True)[0].reshape(-1)
                        for column in range(point.numel())])
                    if not bool(torch.isfinite(gradient).all()) or not bool(torch.isfinite(hessian).all()):
                        raise FloatingPointError('Nonfinite native family derivatives; no solver step can be committed')
                    gradients.append(gradient.detach().double())
                    hessians.append(hessian.detach().double())
            if self._constant_values is None:
                self._constant_values = tuple(constants)
            return {
                'loss_components': {name: math.fsum(value[name] for value in values) / len(values) for name in values[0]},
                'gradient': None if not derivatives else torch.stack(gradients).mean(0),
                'hessian': None if not derivatives else torch.stack(hessians).mean(0),
                'theta': point.detach().clone().reshape(-1),
                'evaluation_index': self.evaluations, 'complete_batches': len(values),
            }
        finally:
            self.parent.summary()
