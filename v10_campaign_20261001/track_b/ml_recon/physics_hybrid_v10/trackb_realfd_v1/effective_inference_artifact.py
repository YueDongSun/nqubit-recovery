"""Strict source-bound native point artifact, deliberately not a posterior.

Parsing validates identities and numerical consistency, not producer execution.
An independent model/data replay is required to verify the reported forward.
Unknown physical parameters or uncertainty cannot become zero/one defaults.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Mapping

from .contracts import canonical_sha256, parse_canonical_json_object, require_git_oid, require_sha256
from .effective_family_context_v3 import MODEL_SCHEMA as family_model_schema
from .effective_observation_v2 import MODEL_SCHEMA as effective_model_schema
from .effective_inference_context import validate_effective_inference_protocol
from .effective_point_values import validate_effective_point_values


point_inference_schema = 'TrackBEffectivePointInferenceV1'
maximum_point_artifact_bytes = 64 * 1024 * 1024
point_scope = 'EFFECTIVE_POINT_PREDICTION_NOT_POSTERIOR_OR_ACTION_PROPOSAL'
sampling_covariance_scope = 'PER_OPERATION_NESTED_COUNT_SAMPLING_GIVEN_NATIVE_READ_AND_POINT_PARAMETERS'
unavailable_quantities = (
    ('microscopic_efficiency', 'NOT_IDENTIFIED'), ('detector_dark_background', 'NOT_IDENTIFIED'),
    ('source_multiplicity', 'NOT_IDENTIFIED'), ('state_posterior_covariance', 'NOT_ESTIMATED'),
    ('posterior_draws', 'NOT_ESTIMATED'), ('cross_operation_sampling_covariance', 'NOT_QUALIFIED'),
    ('ood_confidence', 'NOT_CALIBRATED'), ('causal_action_jacobian', 'NOT_IDENTIFIED_FROM_STATIC_OBSERVATIONS'),
    ('desired_target', 'NOT_AN_INPUT_TO_POINT_INFERENCE'))
point_artifact_fields = frozenset({'schema', 'source_commit', 'source_tree', 'system_type', 'model_schema',
    'model_artifact', 'inference_protocol', 'inference_protocol_sha', 'rows', 'runtime', 'forward_evaluations',
    'scope', 'sampling_covariance_scope', 'unavailable', 'model_forward_executed', 'optimizer_constructed',
    'model_parameters_updated', 'scientific_gate_issued', 'hardware_action_authorized', 'observation_semantics'})
model_artifact_fields = frozenset({'kind', 'file_sha', 'model_state_sha', 'model_config_sha', 'producer_commit',
    'producer_tree', 'training_checkpoint_file_sha', 'training_identity_sha', 'training_source_commit',
    'training_source_tree', 'training_final_model_state_sha'})
point_row_fields = frozenset({'sample_id', 'batch_index', 'input_sha', 'operation_ids', 'operation_families',
    'cycle_indices', 'operation_mask', 'visible_operation_mask', 'query_mask', 'values'})


def unavailable_inference_quantities() -> dict[str, dict[str, Any]]:
    return {name: {'status': status, 'value': None} for name, status in unavailable_quantities}


def native_point_observation_semantics() -> dict[str, Any]:
    return {'response_scope': 'CONDITIONAL_ON_NATIVE_EFFECTIVE_READ',
        'idler_event': 'STICKY_OR_OF_EXACT_SINGLETON_OVER_19_ACCEPTED_CLOCKS',
        'parameter_interpretation': 'EFFECTIVE_RESPONSE_NOT_MICROSCOPIC_CALIBRATION',
        'read_trials_semantics': 'native_v240_scalar_field_not_reconstructed_from_query',
        'native_read_count_origin': 'V240_NATIVE_READ_COUNT_SCALAR',
        'prepared_command_semantics': 'prepared_state_command_coefficients_not_actual_state_or_feedback_target',
        'four_cell_order': ['n00', 'n10', 'n01', 'n11'],
        'four_cell_axes': 'SIGNAL_AND_STICKY_IDLER_NOT_SPAD_CHANNELS_OR_W_MODES',
        'global_head_scope': 'ROW_GLOBAL_NOT_OPERATION_LOCAL_OR_MODE_OCCUPATIONS'}


def _validate_model_artifact(model, model_schema):
    if type(model) is not dict or set(model) != model_artifact_fields \
            or model['kind'] not in {'EFFECTIVE_CPU_CHECKPOINT', 'FAMILY_REFINEMENT_OVERLAY'}:
        raise ValueError('Native point model artifact fields/kind differ')
    for name in ('file_sha', 'model_state_sha', 'model_config_sha', 'training_checkpoint_file_sha',
                 'training_identity_sha', 'training_final_model_state_sha'):
        require_sha256(model[name], name)
    for name in ('producer_commit', 'producer_tree', 'training_source_commit', 'training_source_tree'):
        require_git_oid(model[name], name)
    if model['kind'] == 'EFFECTIVE_CPU_CHECKPOINT':
        for active, training in (('file_sha', 'training_checkpoint_file_sha'), ('model_state_sha', 'training_final_model_state_sha'),
                                 ('producer_commit', 'training_source_commit'), ('producer_tree', 'training_source_tree')):
            if model[active] != model[training]:
                raise ValueError('Ordinary checkpoint inference cannot hide a separate refinement')
    elif model_schema != family_model_schema:
        raise ValueError('Family refinement must retain the registered V3 model schema')


def _validate_point_rows(payload, mode_count):
    protocol = payload['inference_protocol']
    rows = payload['rows']
    if type(rows) is not list or len(rows) != protocol['row_count']:
        raise ValueError('Native point rows do not cover the inference protocol')
    cursor = 0
    reference_metadata = None
    for binding in protocol['batches']:
        query_masks = []
        for sample in binding['sample_ids']:
            row = rows[cursor]
            cursor += 1
            if type(row) is not dict or set(row) != point_row_fields or row['sample_id'] != sample \
                    or type(row['batch_index']) is not int or row['batch_index'] != binding['batch_index'] \
                    or row['input_sha'] != binding['input_sha']:
                raise ValueError('Native point row ordering/input identity differs')
            operations, families, cycles = row['operation_ids'], row['operation_families'], row['cycle_indices']
            if type(operations) is not list or not operations or any(type(value) is not str or not value for value in operations) \
                    or len(set(operations)) != len(operations) or type(families) is not list or len(families) != len(operations) \
                    or any(type(value) is not str or not value for value in families) \
                    or type(cycles) is not list or len(cycles) != len(operations) or any(type(value) is not int for value in cycles):
                raise ValueError('Native point public operation metadata is invalid')
            metadata = (operations, families, cycles)
            if reference_metadata is None:
                reference_metadata = metadata
            elif reference_metadata != metadata:
                raise ValueError('Native point rows cannot mix public operation registries')
            masks = []
            for name in ('operation_mask', 'visible_operation_mask', 'query_mask'):
                mask = row[name]
                if type(mask) is not list or len(mask) != len(operations) or any(type(value) is not bool for value in mask):
                    raise ValueError('Native point masks must be complete native booleans')
                masks.append(mask)
            available, visible, query = masks
            if not any(visible) or not any(query) or any(q and not a or v != (a and not q)
                    for a, v, q in zip(available, visible, query, strict=True)):
                raise ValueError('Native point visible/query masks disagree')
            query_masks.append([int(value) for value in query])
            validate_effective_point_values(row['values'], mode_count=mode_count, operation_count=len(operations))
            has_family = row['values']['effective_family_correction'] is not None
            if has_family != (payload['model_schema'] == family_model_schema):
                raise ValueError('Native point family fields differ from the actual model schema')
        recipe = protocol['query_recipe']
        query_sha = canonical_sha256({'schema': 'TrackBDeterministicQueryMaskV1', 'sample_ids': binding['sample_ids'],
            'epoch': recipe['epoch'], 'view_seed': recipe['view_seed'], 'query_fraction': recipe['query_fraction'], 'masks': query_masks})
        if query_sha != binding['query_mask_sha']:
            raise ValueError('Native point masks differ from the bound query identity')


def validate_effective_point_artifact(payload: Mapping[str, Any]) -> None:
    if type(payload) is not dict or set(payload) != point_artifact_fields or payload['schema'] != point_inference_schema \
            or payload['scope'] != point_scope or payload['sampling_covariance_scope'] != sampling_covariance_scope:
        raise ValueError('Native point artifact schema/scope differs')
    require_git_oid(payload['source_commit'], 'inference_source_commit')
    require_git_oid(payload['source_tree'], 'inference_source_tree')
    if payload['system_type'] not in ('3qubit', '4qubit') or payload['model_schema'] not in (effective_model_schema, family_model_schema):
        raise ValueError('Native point model or real system is not registered')
    if payload['model_forward_executed'] is not True or any(payload[name] is not False for name in (
            'optimizer_constructed', 'model_parameters_updated', 'scientific_gate_issued', 'hardware_action_authorized')) \
            or payload['unavailable'] != unavailable_inference_quantities() \
            or payload['observation_semantics'] != native_point_observation_semantics():
        raise ValueError('Native point execution or unknown/authority fields differ')
    _validate_model_artifact(payload['model_artifact'], payload['model_schema'])
    protocol = payload['inference_protocol']
    validate_effective_inference_protocol(protocol)
    require_sha256(payload['inference_protocol_sha'], 'inference_protocol_sha')
    if canonical_sha256(protocol) != payload['inference_protocol_sha'] \
            or protocol['training_identity_sha'] != payload['model_artifact']['training_identity_sha'] \
            or type(payload['forward_evaluations']) is not int or payload['forward_evaluations'] != len(protocol['batches']):
        raise ValueError('Native point protocol, training identity or forward count differs')
    runtime = payload['runtime']
    if type(runtime) is not dict or set(runtime) != {'device', 'parameter_precision', 'count_precision', 'torch_version', 'cpu_threads'} \
            or runtime['device'] != 'cpu' or runtime['parameter_precision'] != 'fp32' or runtime['count_precision'] != 'fp64' \
            or type(runtime['torch_version']) is not str or not runtime['torch_version'] \
            or type(runtime['cpu_threads']) is not int or runtime['cpu_threads'] < 1:
        raise ValueError('Native point runtime contract differs')
    _validate_point_rows(payload, 3 if payload['system_type'] == '3qubit' else 4)


@dataclass(frozen=True)
class Effective_point_inference:
    """Immutable numeric artifact; parsing is not independent execution proof."""
    payload_bytes: bytes
    file_sha: str

    def payload(self) -> dict[str, Any]:
        require_sha256(self.file_sha, 'point_artifact_file_sha')
        if type(self.payload_bytes) is not bytes or len(self.payload_bytes) > maximum_point_artifact_bytes \
                or hashlib.sha256(self.payload_bytes).hexdigest() != self.file_sha:
            raise ValueError('Native point artifact byte identity or size differs')
        payload = parse_canonical_json_object(self.payload_bytes, schema=point_inference_schema, exact_fields=point_artifact_fields)
        validate_effective_point_artifact(payload)
        return payload


def load_effective_point_inference_bytes(
    payload: bytes, *, expected_file_sha: str, expected_source_commit: str, expected_source_tree: str,
    expected_model_artifact_sha: str, expected_training_identity_sha: str, expected_inference_protocol_sha: str,
) -> Effective_point_inference:
    """Validate explicit external identities; never infer missing expectations."""
    require_git_oid(expected_source_commit, 'expected_inference_source_commit')
    require_git_oid(expected_source_tree, 'expected_inference_source_tree')
    for name, value in (('expected_model_artifact_sha', expected_model_artifact_sha),
                        ('expected_training_identity_sha', expected_training_identity_sha),
                        ('expected_inference_protocol_sha', expected_inference_protocol_sha)):
        require_sha256(value, name)
    artifact = Effective_point_inference(payload, expected_file_sha)
    observed = artifact.payload()
    if (observed['source_commit'], observed['source_tree']) != (expected_source_commit, expected_source_tree) \
            or observed['model_artifact']['file_sha'] != expected_model_artifact_sha \
            or observed['model_artifact']['training_identity_sha'] != expected_training_identity_sha \
            or observed['inference_protocol_sha'] != expected_inference_protocol_sha:
        raise ValueError('Native point artifact differs from the external source/model/inference expectations')
    return artifact
