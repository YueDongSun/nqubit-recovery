"""Separate admitted inference-row/query provenance from checkpoint training.

The first protocol covers an existing complete admitted development fold.
It does not relabel new admissions, locked observations or live hardware data
as the checkpoint's original source. The caller owns dataset lifetime.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import math
from typing import Any, Mapping

import torch

from .contracts import canonical_json_bytes, canonical_sha256, parse_canonical_json_object, require_sha256
from .datasets import trackb_real_static_collate, trackb_real_static_evaluation_collate
from .effective_checkpoint_reader import Effective_checkpoint_replay
from .family_refinement_reader import Family_refinement_replay
from .row_identity import _value_digest, row_identity_manifest_sha, source_row_content_sha


inference_roles = frozenset({'train', 'validation', 'calibration'})
inference_protocol_scope = 'MASKED_OBSERVATION_POINT_INFERENCE_ON_EXISTING_ADMITTED_DEVELOPMENT_SOURCE'
inference_protocol_fields = frozenset({'schema', 'scope', 'admission_manifest_sha', 'outer_fold_sha', 'outer_role',
    'selected_row_identity_sha', 'row_identity_records', 'session_index_bindings', 'operation_metadata_sha',
    'query_recipe', 'batches', 'row_count', 'component_count', 'training_identity_sha', 'training_row_identity_sha',
    'inference_rows_are_training_rows', 'optimizer_constructed', 'scientific_gate_issued', 'hardware_action_authorized'})


@dataclass(frozen=True)
class Effective_inference_config:
    """Explicit masked-observation recipe, independent of optimizer lineage."""
    batch_size: int = 4
    epoch: int = 0
    view_seed: int = 20261001
    query_fraction: float = 0.25

    def validate(self) -> None:
        if type(self.batch_size) is not int or not 1 <= self.batch_size <= 256 \
                or any(type(value) is not int or value < 0 for value in (self.epoch, self.view_seed)) \
                or type(self.query_fraction) is not float or not math.isfinite(self.query_fraction) \
                or not 0 < self.query_fraction < 1:
            raise ValueError('Invalid bounded native inference query recipe')


def inference_batch_input_sha(batch: Mapping[str, Any]) -> str:
    """Integrity binding, not an encoder feature or a scientific attestation."""
    return canonical_sha256({'schema': 'TrackBEffectiveInferenceBatchInputsV1',
        'fields': {name: _value_digest(value) for name, value in sorted(batch.items())}})


def validate_effective_inference_protocol(protocol: Mapping[str, Any]) -> None:
    """Check a serialized plan's internal identities; not source execution proof."""
    if type(protocol) is not dict or set(protocol) != inference_protocol_fields \
            or protocol['schema'] != 'TrackBEffectiveInferenceProtocolV1' or protocol['scope'] != inference_protocol_scope:
        raise ValueError('Native inference protocol schema or fields differ')
    role = protocol['outer_role']
    if role not in inference_roles or protocol['inference_rows_are_training_rows'] is not (role == 'train') \
            or any(protocol[name] is not False for name in ('optimizer_constructed', 'scientific_gate_issued', 'hardware_action_authorized')):
        raise ValueError('Native inference protocol role or authority differs')
    for name in ('admission_manifest_sha', 'outer_fold_sha', 'selected_row_identity_sha', 'operation_metadata_sha',
                 'training_identity_sha', 'training_row_identity_sha'):
        require_sha256(protocol[name], name)
    rows, components = protocol['row_count'], protocol['component_count']
    records = protocol['row_identity_records']
    if type(rows) is not int or rows < 1 or type(components) is not int or not 1 <= components <= rows \
            or type(records) is not list or len(records) != rows:
        raise ValueError('Native inference protocol coverage differs')
    samples = []
    for record in records:
        if type(record) is not list or len(record) != 4 or any(type(value) is not str or not value for value in record) \
                or record[3] != 'train':
            raise ValueError('Native inference protocol requires exact development-source row records')
        require_sha256(record[2], 'row_content_sha')
        samples.append(record[0])
    if len(set(samples)) != rows or row_identity_manifest_sha(records) != protocol['selected_row_identity_sha'] \
            or (role == 'train' and protocol['selected_row_identity_sha'] != protocol['training_row_identity_sha']):
        raise ValueError('Native inference row records do not match their declared identity')
    sessions = protocol['session_index_bindings']
    if type(sessions) is not list or len(sessions) != rows or any(type(binding) is not list or len(binding) != 2 \
            or binding[0] != sample or type(binding[1]) is not int or binding[1] < 0
            for sample, binding in zip(samples, sessions, strict=True)):
        raise ValueError('Native inference session records differ from selected rows')
    recipe = protocol['query_recipe']
    if type(recipe) is not dict or set(recipe) != set(asdict(Effective_inference_config())):
        raise ValueError('Native inference query recipe fields differ')
    config = Effective_inference_config(**recipe)
    config.validate()
    batches = protocol['batches']
    batch_count = (rows + config.batch_size - 1) // config.batch_size
    if type(batches) is not list or len(batches) != batch_count:
        raise ValueError('Native inference batch count differs')
    for index, batch in enumerate(batches):
        if type(batch) is not dict or set(batch) != {'batch_index', 'sample_ids', 'batch_plan_sha', 'query_mask_sha', 'input_sha'} \
                or type(batch['batch_index']) is not int or batch['batch_index'] != index \
                or batch['sample_ids'] != samples[index * config.batch_size:(index + 1) * config.batch_size]:
            raise ValueError('Native inference batch ordering differs from the selected rows')
        for name in ('batch_plan_sha', 'query_mask_sha', 'input_sha'):
            require_sha256(batch[name], name)


def _checked_inference_items(replay, dataset):
    if type(replay) not in (Effective_checkpoint_replay, Family_refinement_replay):
        raise TypeError('Native inference requires an independently loaded V2/V3 or refinement replay')
    replay.summary()
    identity = replay.identity
    role = getattr(dataset, 'outer_role', None)
    if role not in inference_roles:
        raise ValueError('Native inference accepts only an explicit admitted development fold')
    dataset.fold_plan.__post_init__()
    if dataset.admission_manifest.sha256() != identity.admission_manifest_sha \
            or dataset.fold_plan.sha256() != identity.outer_fold_sha \
            or dataset.fold_plan.sample_ids_for_role('train') != frozenset(identity.allowed_sample_ids):
        raise ValueError('Inference admission/fold differs from the supported checkpoint source')
    if replay.final_model.config.read_trials_source != 'native_scalar':
        raise ValueError('Native inference requires the explicit native READ scalar contract')
    metadata_sha = dataset.operation_metadata_sha
    require_sha256(metadata_sha, 'inference_operation_metadata_sha')
    if dataset.admission_manifest.operation_metadata_sha != metadata_sha:
        raise ValueError('Inference metadata differs from its source admission')
    context = replay.final_model.config.measurement_context
    if context is not None and context.operation_metadata_sha != metadata_sha:
        raise ValueError('Inference metadata differs from the frozen family context')
    expected = dataset.fold_plan.sample_ids_for_role(role)
    if not expected or len(dataset) != len(expected):
        raise ValueError('Inference must cover the complete selected fold')
    items, records, sessions, seen = [], [], [], set()
    for index in range(len(dataset)):
        item = copy.deepcopy(dataset[index])
        sample = item['sample_id']
        record = (sample, item['group_id'], item['row_content_sha'], item['source_role'])
        if sample not in expected or sample in seen or item.get('outer_role') != role or item['source_role'] != 'train' \
                or item.get('admission_manifest_sha') != identity.admission_manifest_sha \
                or item.get('outer_fold_sha') != identity.outer_fold_sha or item['system_type'] != identity.system_type \
                or tuple(dataset.row_identity_records.get(sample, ())) != record[1:] \
                or source_row_content_sha(item) != item['row_content_sha']:
            raise ValueError('Inference row content, identity or admitted role changed')
        session = item['session_index']
        if not isinstance(session, torch.Tensor) or session.dtype != torch.int64 or session.numel() != 1 or int(session) < 0:
            raise ValueError('Inference session identity must be one native nonnegative integer')
        seen.add(sample)
        items.append(item)
        records.append(record)
        sessions.append((sample, int(session)))
    selected_sha = row_identity_manifest_sha(records)
    if seen != expected or dataset.row_identity_sha != selected_sha \
            or any(item.get('row_identity_sha') != selected_sha for item in items):
        raise ValueError('Selected inference-row identity differs from its own complete fold')
    if role == 'train' and (selected_sha != identity.row_identity_sha \
            or dict(sessions) != dict(identity.session_index_bindings)):
        raise ValueError('TRAIN inference rows differ from checkpoint training provenance')
    return items, records, sessions


@dataclass(frozen=True)
class Effective_inference_plan:
    """Owned batch tensors are checked against immutable protocol bytes."""
    protocol_bytes: bytes
    batches: tuple[dict[str, Any], ...]
    batch_input_shas: tuple[str, ...]

    def __post_init__(self) -> None:
        protocol = parse_canonical_json_object(self.protocol_bytes, schema='TrackBEffectiveInferenceProtocolV1',
            exact_fields=inference_protocol_fields)
        validate_effective_inference_protocol(protocol)
        if type(self.batches) is not tuple or type(self.batch_input_shas) is not tuple \
                or len(self.batches) != len(protocol['batches']) \
                or self.batch_input_shas != tuple(batch['input_sha'] for batch in protocol['batches']):
            raise ValueError('Owned inference batches differ from the immutable protocol')
        for index in range(len(self.batches)):
            self.checked_batch(index)

    def checked_batch(self, index: int) -> dict[str, Any]:
        batch = self.batches[index]
        if inference_batch_input_sha(batch) != self.batch_input_shas[index]:
            raise ValueError('Native inference batch changed after admission')
        return batch


def build_effective_inference_plan(
    replay: Effective_checkpoint_replay | Family_refinement_replay, dataset: Any, *,
    config: Effective_inference_config = Effective_inference_config(),
) -> Effective_inference_plan:
    """Construct actual original-source inference views without fitting/scoring."""
    config.validate()
    items, records, sessions = _checked_inference_items(replay, dataset)
    role = dataset.outer_role
    batches, bindings, input_shas = [], [], []
    for index, start in enumerate(range(0, len(items), config.batch_size)):
        selected = items[start:start + config.batch_size]
        options = dict(epoch=config.epoch, view_seed=config.view_seed, query_fraction=config.query_fraction,
            batch_index=index, cursor_start=start)
        batch = (trackb_real_static_collate(selected, **options) if role == 'train' else
                 trackb_real_static_evaluation_collate(selected, outer_role=role, **options))
        batch['operation_metadata_sha'] = dataset.operation_metadata_sha
        input_sha = inference_batch_input_sha(batch)
        batches.append(batch)
        input_shas.append(input_sha)
        bindings.append({'batch_index': index, 'sample_ids': list(batch['sample_ids']),
            'batch_plan_sha': batch['batch_plan_sha'], 'query_mask_sha': batch['query_mask_sha'], 'input_sha': input_sha})
    protocol = {'schema': 'TrackBEffectiveInferenceProtocolV1',
        'scope': inference_protocol_scope,
        'admission_manifest_sha': dataset.admission_manifest.sha256(), 'outer_fold_sha': dataset.fold_plan.sha256(),
        'outer_role': role, 'selected_row_identity_sha': dataset.row_identity_sha,
        'row_identity_records': [list(record) for record in records],
        'session_index_bindings': [list(binding) for binding in sessions],
        'operation_metadata_sha': dataset.operation_metadata_sha, 'query_recipe': asdict(config),
        'batches': bindings, 'row_count': len(items),
        'component_count': len(dataset.fold_plan.components_for_role(role)),
        'training_identity_sha': replay.identity.sha256(), 'training_row_identity_sha': replay.identity.row_identity_sha,
        'inference_rows_are_training_rows': role == 'train', 'optimizer_constructed': False,
        'scientific_gate_issued': False, 'hardware_action_authorized': False}
    replay.summary()
    return Effective_inference_plan(canonical_json_bytes(protocol), tuple(batches), tuple(input_shas))
