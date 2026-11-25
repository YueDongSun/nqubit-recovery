"""Shared immutable exact-tick reducer for online execution and offline replay."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from ..contracts.memory_evidence import (
    coefficient_payload as _coefficient_payload,
    coordinate_layout_sha as _coordinate_layout_sha,
    digest as _digest,
    mode_descriptor_sha as _mode_descriptor_sha,
    normalized_coefficients as _normalized_coefficients,
    phase_convention_sha as _phase_convention_sha,
    Synthetic_qualification_evaluator_receipt,
    Raw_qualification_evidence_unavailable,
)

LEGACY_MEMORY_EVENT_SCHEMA_V3 = "V10MemoryPhysicalEventV3"
LEGACY_MEMORY_EVENT_SCHEMA_V4 = "V10MemoryPhysicalEventV4"
MEMORY_EVENT_SCHEMA = "V10MemoryPhysicalEventV5"
EXECUTABLE_REGIMES = frozenset(
    {"M0_NO_MEMORY", "M1_IDEAL_MEMORY", "M3_MEMORY_SENSITIVITY"}
)
ACTIVE_STATES = frozenset({"pending_write", "stored", "available", "booked"})


def reducer_source_sha256() -> str:
    """Return the SHA-256 of this exact reducer source file."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _integer(value: object, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _identifier(value: object, name: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ValueError(f"{name} must be a canonical nonempty identifier")
    return value


def _sha(value: object, name: str) -> str:
    if type(value) is not str or len(value) != 64:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value




def frozen_compatibility_predicate_sha(fusion_recipe_sha, target_state_sha, target_compatibility_class, selected_a, selected_b, selected_port_labels, target_fidelity_threshold):
    return _digest({"schema": "V10MemoryPairCompatibilityPredicateV1", "fusion_recipe_sha": fusion_recipe_sha, "target_state_sha": target_state_sha, "target_compatibility_class": target_compatibility_class, "selected_ports": (selected_a, selected_b), "selected_port_labels": tuple(selected_port_labels), "target_fidelity_threshold": target_fidelity_threshold})


@dataclass(frozen=True)
class Memory_lifecycle_contract:
    protocol_sha: str
    model_sha: str
    compatibility_sha: str
    allocation_map_sha: str
    regime: str
    horizon_tick: int
    tick_seconds: float
    capacity: int
    capacity_unit: str
    allowed_class_pairs: tuple[tuple[str, str], ...]
    target_compatibility_class: str
    target_state_sha: str
    fusion_recipe_sha: str
    selected_a: int
    selected_b: int
    fusion_order: int
    write_ticks: int
    switch_ticks: int
    retrieve_ticks: int
    fusion_ticks: int
    deadtime_ticks: int
    maximum_age_ticks: int | None
    target_coefficients: tuple[complex, ...] | None = None
    fusion_recipe_json: str | None = None
    target_fidelity_threshold: float | None = None
    selected_port_labels: tuple[str, str] | None = None
    minimum_qualification_quality: float | None = None
    compatibility_predicate_sha: str | None = None
    trusted_qualification_issuer_id: str | None = None
    trusted_qualification_issuer_role: str | None = None
    trusted_qualification_issuer_manifest_sha: str | None = None
    trusted_qualification_issuer_public_key_hex: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "protocol_sha", "model_sha", "compatibility_sha", "allocation_map_sha"
        ):
            _sha(getattr(self, name), name)
        if self.regime not in EXECUTABLE_REGIMES:
            raise ValueError("only executable synthetic regimes may define a lifecycle contract")
        _integer(self.horizon_tick, "horizon_tick", 1)
        if (
            type(self.tick_seconds) not in (int, float)
            or not math.isfinite(self.tick_seconds)
            or self.tick_seconds <= 0
        ):
            raise ValueError("tick_seconds must be finite, non-bool, and positive")
        _integer(self.capacity, "capacity", 1)
        if self.capacity_unit not in {"LOGICAL_RESOURCE", "PHYSICAL_MEMORY_CELL"}:
            raise ValueError("unregistered capacity unit")
        if self.fusion_order != 2:
            raise ValueError("the implemented reducer has exact fusion_order=2")
        for name in (
            "write_ticks",
            "switch_ticks",
            "retrieve_ticks",
            "fusion_ticks",
            "deadtime_ticks",
        ):
            _integer(getattr(self, name), name)
        if self.retrieve_ticks + self.fusion_ticks < 1:
            raise ValueError("retrieval plus fusion service time must be nonzero")
        if self.maximum_age_ticks is not None:
            _integer(self.maximum_age_ticks, "maximum_age_ticks", 1)
        pairs = tuple(tuple(pair) for pair in self.allowed_class_pairs)
        canonical_pairs = tuple(sorted(tuple(sorted(pair)) for pair in pairs))
        if not pairs or len(set(canonical_pairs)) != len(canonical_pairs):
            raise ValueError("allowed class pairs must be nonempty and canonically unique")
        for pair in pairs:
            if len(pair) != 2:
                raise ValueError("ordered allowed class pairs are required")
            _identifier(pair[0], "allowed_input_class")
            _identifier(pair[1], "allowed_input_class")
        _identifier(self.target_compatibility_class, "target_compatibility_class")
        _sha(self.target_state_sha, "target_state_sha")
        _sha(self.fusion_recipe_sha, "fusion_recipe_sha")
        _integer(self.selected_a, "selected_a")
        _integer(self.selected_b, "selected_b")
        if self.target_coefficients is not None:
            object.__setattr__(
                self, "target_coefficients", _normalized_coefficients(self.target_coefficients)
            )
            if self.target_state_sha != _digest({
                "schema": "V10MemoryTargetStateV1",
                "coefficients": _coefficient_payload(self.target_coefficients),
            }):
                raise ValueError("target_state_sha must bind the complete target coefficients")
        if self.fusion_recipe_json is not None:
            try:
                recipe = json.loads(self.fusion_recipe_json)
            except (TypeError, json.JSONDecodeError) as error:
                raise ValueError("fusion_recipe_json must be canonical JSON") from error
            canonical_recipe = json.dumps(recipe, sort_keys=True, separators=(",", ":"))
            if canonical_recipe != self.fusion_recipe_json:
                raise ValueError("fusion_recipe_json must use canonical JSON encoding")
            if self.fusion_recipe_sha != _digest(recipe):
                raise ValueError("fusion_recipe_sha must bind the complete frozen recipe")
        if self.target_fidelity_threshold is not None:
            if (
                type(self.target_fidelity_threshold) not in (int, float)
                or isinstance(self.target_fidelity_threshold, bool)
                or not math.isfinite(self.target_fidelity_threshold)
                or not 0 <= self.target_fidelity_threshold <= 1
            ):
                raise ValueError("target_fidelity_threshold must be a finite probability")
        if self.selected_port_labels is not None:
            labels = tuple(self.selected_port_labels)
            if len(labels) != 2:
                raise ValueError("selected_port_labels must name both selected ports")
            for label in labels:
                _identifier(label, "selected_port_label")
            object.__setattr__(self, "selected_port_labels", labels)
        if self.minimum_qualification_quality is not None:
            if type(self.minimum_qualification_quality) not in (int, float) or isinstance(self.minimum_qualification_quality, bool) or not math.isfinite(self.minimum_qualification_quality) or not 0 <= self.minimum_qualification_quality <= 1:
                raise ValueError("minimum_qualification_quality must be a finite probability")
        if self.compatibility_predicate_sha is not None:
            _sha(self.compatibility_predicate_sha, "compatibility_predicate_sha")
        anchors = (self.trusted_qualification_issuer_id, self.trusted_qualification_issuer_role, self.trusted_qualification_issuer_manifest_sha, self.trusted_qualification_issuer_public_key_hex)
        if any(value is not None for value in anchors):
            if any(value is None for value in anchors):
                raise ValueError("qualification trust anchor is incomplete")
            _identifier(self.trusted_qualification_issuer_id, "trusted_qualification_issuer_id")
            if self.trusted_qualification_issuer_role != "SYNTHETIC_QUALIFICATION_ISSUER_V1":
                raise ValueError("qualification trust anchor role is invalid")
            _sha(self.trusted_qualification_issuer_manifest_sha, "trusted_qualification_issuer_manifest_sha")
            try:
                if len(bytes.fromhex(self.trusted_qualification_issuer_public_key_hex)) != 32:
                    raise ValueError
            except (TypeError, ValueError) as error:
                raise ValueError("qualification trust anchor public key is invalid") from error
        if None not in (self.fusion_recipe_json, self.target_fidelity_threshold, self.selected_port_labels, self.compatibility_predicate_sha):
            recipe = json.loads(self.fusion_recipe_json)
            config = recipe.get("fusion_config")
            if type(config) is not dict or (config.get("selected_a"), config.get("selected_b")) != (self.selected_a, self.selected_b):
                raise ValueError("selected ports must exactly match the frozen fusion recipe")
            if self.selected_port_labels != (f"input_a_mode_{self.selected_a}", f"input_b_mode_{self.selected_b}"):
                raise ValueError("selected_port_labels must exactly name the selected port indices")
            expected_predicate = frozen_compatibility_predicate_sha(self.fusion_recipe_sha, self.target_state_sha, self.target_compatibility_class, self.selected_a, self.selected_b, self.selected_port_labels, self.target_fidelity_threshold)
            if self.compatibility_predicate_sha != expected_predicate:
                raise ValueError("compatibility predicate must bind frozen recipe, ports, target, and threshold")
        object.__setattr__(self, "allowed_class_pairs", pairs)

    def sha256(self) -> str:
        return _digest(asdict(self))


@dataclass(frozen=True)
class Memory_event_resource:
    resource_id: str
    physical_cells: tuple[str, ...]
    input_class: str
    partition_sha: str
    mode_coefficients: tuple[complex, ...]
    mode_descriptor_sha: str
    phase_convention_sha: str
    coordinate_layout_sha: str
    target_compatibility_class: str
    calibration_domain_sha: str
    target_state_sha: str
    qualification_receipt_sha: str | None
    qualification_evidence_sha: str
    source_session_apparatus_provenance_sha: str
    qualification_subject_id: str
    qualification_certificate_valid: bool
    qualification_recoverability_valid: bool
    qualification_safety_valid: bool
    qualification_receipt_available_tick: int
    qualification_estimated_quality: float | None
    loaded_tick: int | None = None
    qualification_source_id: str | None = None
    qualification_cycle_id: str | None = None
    trial_provenance_sha: str | None = None
    qualification_outcome: str | None = None
    qualification_bundle: Synthetic_qualification_evaluator_receipt | None = None
    generated_tick: int | None = None
    qualified_tick: int | None = None

    def __post_init__(self) -> None:
        _identifier(self.resource_id, "resource_id")
        _identifier(self.input_class, "input_class")
        _identifier(self.qualification_subject_id, "qualification_subject_id")
        for name in ("qualification_source_id", "qualification_cycle_id", "qualification_outcome"):
            if getattr(self, name) is None:
                raise ValueError(f"{name} is required for a replayable qualification receipt")
            _identifier(getattr(self, name), name)
        if self.trial_provenance_sha is None:
            raise ValueError("trial_provenance_sha is required for a replayable qualification receipt")
        _sha(self.trial_provenance_sha, "trial_provenance_sha")
        coefficients = _normalized_coefficients(self.mode_coefficients)
        _identifier(self.target_compatibility_class, "target_compatibility_class")
        for name in (
            "partition_sha",
            "mode_descriptor_sha",
            "phase_convention_sha",
            "coordinate_layout_sha",
            "calibration_domain_sha",
            "target_state_sha",
            "qualification_evidence_sha",
            "source_session_apparatus_provenance_sha",
        ):
            _sha(getattr(self, name), name)
        if self.qualification_receipt_sha is not None:
            _sha(self.qualification_receipt_sha, "qualification_receipt_sha")
        if any(value is not None and type(value) is not bool for value in (self.qualification_certificate_valid, self.qualification_recoverability_valid, self.qualification_safety_valid)):
            raise ValueError("qualification facts must be booleans or explicitly unavailable")
        if type(self.qualification_receipt_available_tick) is not int:
            raise ValueError("qualification receipt availability tick must be an integer")
        bundle = self.qualification_bundle
        if bundle is None:
            if self.qualification_receipt_sha is not None or any(value is not None for value in (self.qualification_certificate_valid, self.qualification_recoverability_valid, self.qualification_safety_valid, self.qualification_estimated_quality)):
                raise ValueError("V1 qualification event tapes are rejected; a file-backed V2 bundle is required")
            return
        if type(bundle) is not Synthetic_qualification_evaluator_receipt:
            raise ValueError("qualification bundle must be a file-backed V2 receipt")
        raw_identity = bundle.raw_observation["identity"]
        if (raw_identity["subject_id"], raw_identity["resource_id"], raw_identity["source_id"], raw_identity["cycle_id"], raw_identity["trial_provenance_sha"], raw_identity["partition_sha"], raw_identity["qualification_outcome"]) != (self.qualification_subject_id, self.resource_id, self.qualification_source_id, self.qualification_cycle_id, self.trial_provenance_sha, self.partition_sha, self.qualification_outcome):
            raise ValueError("file-backed qualification bundle identity mismatch")
        if (self.qualification_certificate_valid, self.qualification_recoverability_valid, self.qualification_safety_valid, self.qualification_estimated_quality, self.qualification_receipt_available_tick) != (bundle.certificate_valid, bundle.recoverability_valid, bundle.safety_valid, bundle.estimated_quality, bundle.receipt_available_tick):
            raise ValueError("file-backed qualification bundle outcome mismatch")
        if self.qualification_evidence_sha != bundle.evidence_sha:
            raise ValueError("qualification evidence must bind the signed evaluator receipt")
        if self.qualification_receipt_sha != bundle.receipt_sha:
            raise ValueError("qualification receipt must bind the signed evaluator receipt")
        if self.qualification_subject_id.startswith("INITIAL:"):
            if type(self.generated_tick) is not int or type(self.qualified_tick) is not int or type(self.loaded_tick) is not int:
                raise ValueError("initial inventory event resource requires generation, qualification, and load ticks")
            if not self.generated_tick <= bundle.observation_tick <= bundle.receipt_available_tick == self.qualified_tick <= self.loaded_tick <= 0:
                raise ValueError("initial inventory event receipt timing is invalid")
        if self.mode_descriptor_sha != _mode_descriptor_sha(coefficients):
            raise ValueError("mode descriptor must bind actual coefficients")
        if self.phase_convention_sha != _phase_convention_sha(coefficients):
            raise ValueError("phase convention must bind actual coefficients")
        if self.coordinate_layout_sha != _coordinate_layout_sha(coefficients):
            raise ValueError("coordinate layout must bind actual coefficient dimension")
        cells = tuple(self.physical_cells)
        if not cells or len(cells) != len(set(cells)):
            raise ValueError("physical cells must be nonempty and unique")
        for cell in cells:
            _identifier(cell, "physical_cell")
        if self.loaded_tick is not None and type(self.loaded_tick) is not int:
            raise ValueError("loaded_tick must be an integer")
        object.__setattr__(self, "physical_cells", cells)
        object.__setattr__(self, "mode_coefficients", coefficients)


@dataclass(frozen=True)
class Memory_lifecycle_event:
    event_id: int
    state_version: int
    tick: int
    kind: str
    resource_ids: tuple[str, ...]
    resources: tuple[Memory_event_resource, ...] = ()
    reservation_id: str | None = None
    click_observed: bool | None = None
    click_tick: int | None = None
    target_window: tuple[int, int] | None = None
    target_sector: bool | None = None
    target_state: bool | None = None
    certified: bool | None = None
    accepted: bool | None = None
    input_survived: tuple[bool, ...] | None = None
    input_retrieved: tuple[bool, ...] | None = None
    post_fusion_recoverability_valid: bool | None = None
    post_fusion_safety_valid: bool | None = None
    post_fusion_confidence_valid: bool | None = None
    post_fusion_certificate_source: str | None = None
    post_fusion_recoverability_evidence_sha: str | None = None
    post_fusion_safety_evidence_sha: str | None = None
    post_fusion_confidence_evidence_sha: str | None = None
    compatibility_provenance_sha: str | None = None

    def __post_init__(self) -> None:
        _integer(self.event_id, "event_id", 1)
        _integer(self.state_version, "state_version", 1)
        _integer(self.tick, "tick")
        _identifier(self.kind, "event_kind")
        ids = tuple(self.resource_ids)
        resources = tuple(self.resources)
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("event resource IDs must be nonempty and unique")
        for resource_id in ids:
            _identifier(resource_id, "event.resource_id")
        if any(type(item) is not Memory_event_resource for item in resources):
            raise ValueError("event resources must be immutable resource records")
        if resources and tuple(item.resource_id for item in resources) != ids:
            raise ValueError("event resources must exactly match IDs")
        if self.reservation_id is not None:
            _identifier(self.reservation_id, "reservation_id")
        for value in (
            self.click_observed,
            self.target_sector,
            self.target_state,
            self.certified,
            self.accepted,
            self.post_fusion_recoverability_valid,
            self.post_fusion_safety_valid,
            self.post_fusion_confidence_valid,
        ):
            if value is not None and type(value) is not bool:
                raise ValueError("completion outcomes must be bool")
        if self.post_fusion_certificate_source is not None:
            _identifier(self.post_fusion_certificate_source, "post_fusion_certificate_source")
        for name in (
            "post_fusion_recoverability_evidence_sha",
            "post_fusion_safety_evidence_sha",
            "post_fusion_confidence_evidence_sha",
            "compatibility_provenance_sha",
        ):
            value = getattr(self, name)
            if value is not None:
                _sha(value, name)
        if self.target_window is not None:
            window = tuple(self.target_window)
            if (
                len(window) != 2
                or type(window[0]) is not int
                or type(window[1]) is not int
                or window[0] >= window[1]
            ):
                raise ValueError("target window must be nonempty and half-open")
            object.__setattr__(self, "target_window", window)
        if self.click_tick is not None:
            _integer(self.click_tick, "click_tick")
        for name in ("input_survived", "input_retrieved"):
            value = getattr(self, name)
            if value is not None:
                value = tuple(value)
                if len(value) != len(ids) or any(type(item) is not bool for item in value):
                    raise ValueError(f"{name} must align with input IDs as strict booleans")
                object.__setattr__(self, name, value)
        object.__setattr__(self, "resource_ids", ids)
        object.__setattr__(self, "resources", resources)


@dataclass(frozen=True)
class _Reservation:
    resource_ids: tuple[str, str]
    booked_tick: int
    compatibility_provenance_sha: str
    consumed_tick: int | None = None
    completed: bool = False


@dataclass(frozen=True)
class Memory_event_state:
    contract: Memory_lifecycle_contract
    resources: Mapping[str, Memory_event_resource] = MappingProxyType({})
    states: Mapping[str, str] = MappingProxyType({})
    availability_ticks: Mapping[str, int] = MappingProxyType({})
    generated_ticks: Mapping[str, int] = MappingProxyType({})
    pending_ticks: Mapping[str, int] = MappingProxyType({})
    stored_ticks: Mapping[str, int] = MappingProxyType({})
    qualification_ticks: Mapping[str, int] = MappingProxyType({})
    retrieval_survival: Mapping[str, bool] = MappingProxyType({})
    retrieval_outcomes: Mapping[str, bool] = MappingProxyType({})
    reservations: Mapping[str, _Reservation] = MappingProxyType({})
    active_reservation_id: str | None = None
    server_available_tick: int = 0
    last_event_id: int = 0
    state_version: int = 0
    last_tick: int = 0

    def __post_init__(self) -> None:
        if type(self.contract) is not Memory_lifecycle_contract:
            raise ValueError("exact lifecycle contract required")
        for name in (
            "resources",
            "states",
            "availability_ticks",
            "generated_ticks",
            "pending_ticks",
            "stored_ticks",
            "qualification_ticks",
            "retrieval_survival",
            "retrieval_outcomes",
            "reservations",
        ):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))
        if self.active_reservation_id is not None:
            _identifier(self.active_reservation_id, "active_reservation_id")
        _integer(self.server_available_tick, "server_available_tick")

    def active_ids(self) -> tuple[str, ...]:
        return tuple(key for key, state in self.states.items() if state in ACTIVE_STATES)

    def has_capacity_for(self, resource: Memory_event_resource) -> bool:
        active = tuple(self.resources[key] for key in self.active_ids())
        occupied_cells = {cell for item in active for cell in item.physical_cells}
        if occupied_cells.intersection(resource.physical_cells):
            return False
        if self.contract.capacity_unit == "LOGICAL_RESOURCE":
            return len(active) + 1 <= self.contract.capacity
        used = sum(len(item.physical_cells) for item in active)
        return used + len(resource.physical_cells) <= self.contract.capacity


def _next_state(state: Memory_event_state, event: Memory_lifecycle_event) -> Memory_event_state:
    if (
        event.event_id != state.last_event_id + 1
        or event.state_version != state.state_version + 1
    ):
        raise ValueError("event/version identity must be monotonic and contiguous")
    if event.tick < state.last_tick:
        raise ValueError("event time must be causal")
    if event.tick > state.contract.horizon_tick:
        raise ValueError("post-horizon events are censored")
    resources = dict(state.resources)
    states = dict(state.states)
    availability = dict(state.availability_ticks)
    generated = dict(state.generated_ticks)
    pending = dict(state.pending_ticks)
    stored = dict(state.stored_ticks)
    qualification = dict(state.qualification_ticks)
    retrieval_survival = dict(state.retrieval_survival)
    retrieval_outcomes = dict(state.retrieval_outcomes)
    reservations = dict(state.reservations)
    active_reservation_id = state.active_reservation_id
    server_available_tick = state.server_available_tick
    ids = event.resource_ids
    kind = event.kind

    if kind == "RESOURCE_GENERATED":
        if len(event.resources) != 1 or event.resources[0].loaded_tick is not None:
            raise ValueError("generated resource requires one new record without loaded_tick")
        if ids[0] in resources:
            raise ValueError("generated resource identity must be unique")
        resources[ids[0]] = event.resources[0]
        states[ids[0]] = "generated"
        generated[ids[0]] = event.tick
    elif kind == "QUALIFICATION_REJECTED":
        _apply_qualification(
            event, states, generated, qualification, "qualification_rejected"
        )
    elif kind == "QUALIFICATION_UNKNOWN":
        _apply_qualification(
            event, states, generated, qualification, "qualification_unknown"
        )
    elif kind == "INITIAL_AVAILABLE":
        _apply_initial_available(state, event, resources, states, availability)
    elif kind == "WRITE_PENDING":
        _apply_write_pending(
            state, event, resources, states, generated, qualification, pending
        )
    elif kind == "RESOURCE_STORED":
        _apply_stored(state, event, states, pending, stored)
    elif kind == "RESOURCE_AVAILABLE":
        _apply_available(event, states, availability, stored)
    elif kind == "M0_EPHEMERAL_AVAILABLE":
        _apply_m0_available(
            state, event, resources, states, availability, generated, qualification
        )
    elif kind == "ADMISSION_REJECTED":
        _require_state(ids, states, "generated", "admission rejection")
        if event.tick != generated[ids[0]]:
            raise ValueError("admission rejection must share the generation tick")
        qualification[ids[0]] = event.tick
        states[ids[0]] = "admission_rejected"
        del generated[ids[0]]
    elif kind == "WRITE_FAILED":
        _apply_write_failed(state, event, states, pending)
    elif kind == "PAIR_BOOKED":
        active_reservation_id = _apply_booking(
            state, event, resources, states, availability, reservations
        )
    elif kind == "RETRIEVAL_CONSUMED":
        _apply_consumption(state, event, states, availability, reservations)
    elif kind == "FUSION_COMPLETED":
        active_reservation_id, server_available_tick = _apply_completion(
            state, event, reservations, retrieval_survival, retrieval_outcomes
        )
    elif kind == "RESOURCE_EXPIRED":
        _apply_expiry(state, event, states, availability)
    elif kind == "RESERVATION_ABORTED":
        active_reservation_id, server_available_tick = _apply_abort(
            state, event, states, reservations
        )
    else:
        raise ValueError("unsupported lifecycle event")
    return Memory_event_state(
        contract=state.contract,
        resources=resources,
        states=states,
        availability_ticks=availability,
        generated_ticks=generated,
        pending_ticks=pending,
        stored_ticks=stored,
        qualification_ticks=qualification,
        retrieval_survival=retrieval_survival,
        retrieval_outcomes=retrieval_outcomes,
        reservations=reservations,
        active_reservation_id=active_reservation_id,
        server_available_tick=server_available_tick,
        last_event_id=event.event_id,
        state_version=event.state_version,
        last_tick=event.tick,
    )


def _require_state(ids, states, expected, label):
    if len(ids) != 1 or states.get(ids[0]) != expected:
        raise ValueError(f"{label} requires one {expected} resource")


def _apply_qualification(event, states, generated, qualification, next_state):
    _require_state(event.resource_ids, states, "generated", "qualification")
    resource_id = event.resource_ids[0]
    if event.tick < generated[resource_id]:
        raise ValueError("qualification cannot precede generation")
    states[resource_id] = next_state
    qualification[resource_id] = event.tick
    del generated[resource_id]


@dataclass(frozen=True)
class Memory_qualification_decision:
    status: str
    reason: str | None = None


def evaluate_resource_qualification(contract: Memory_lifecycle_contract, resource: Memory_event_resource, as_of_tick: int) -> Memory_qualification_decision:
    """Pure qualification predicate shared by online admission and offline replay."""
    if type(contract) is not Memory_lifecycle_contract or type(resource) is not Memory_event_resource:
        raise ValueError("exact lifecycle contract and resource are required for qualification")
    _integer(as_of_tick, "qualification as_of_tick")
    if contract.minimum_qualification_quality is None:
        return Memory_qualification_decision("unknown", "MISSING_FROZEN_QUALIFICATION_THRESHOLD")
    anchors = (contract.trusted_qualification_issuer_id, contract.trusted_qualification_issuer_role, contract.trusted_qualification_issuer_manifest_sha, contract.trusted_qualification_issuer_public_key_hex)
    if any(value is None for value in anchors):
        return Memory_qualification_decision("unknown", "MISSING_QUALIFICATION_TRUST_ANCHOR")
    bundle = resource.qualification_bundle
    if type(bundle) is not Synthetic_qualification_evaluator_receipt:
        return Memory_qualification_decision("unknown", "QUALIFICATION_RECEIPT_BUNDLE_MISSING")
    try:
        bundle.validate()
    except Raw_qualification_evidence_unavailable:
        return Memory_qualification_decision("unknown", "RAW_QUALIFICATION_EVIDENCE_UNAVAILABLE")
    except ValueError:
        return Memory_qualification_decision("rejected", "QUALIFICATION_RECEIPT_SIGNATURE_OR_BYTES_INVALID")
    if resource.qualification_subject_id.startswith("INITIAL:") and not (
        type(resource.generated_tick) is int and type(resource.qualified_tick) is int and type(resource.loaded_tick) is int
        and resource.generated_tick <= bundle.observation_tick <= bundle.receipt_available_tick == resource.qualified_tick <= resource.loaded_tick <= 0
    ):
        return Memory_qualification_decision("rejected", "INITIAL_INVENTORY_RECEIPT_TIMING_INVALID")
    if (bundle.issuer_id, bundle.issuer_role, bundle.issuer_manifest_sha, bundle.issuer_public_key.hex()) != anchors:
        return Memory_qualification_decision("rejected", "QUALIFICATION_TRUST_ANCHOR_MISMATCH")
    expected_outcome = "INITIAL_INVENTORY_QUALIFICATION" if resource.qualification_subject_id.startswith("INITIAL:") else "PAST_OBSERVATION_QUALIFICATION"
    if resource.qualification_outcome != expected_outcome:
        return Memory_qualification_decision("rejected", "QUALIFICATION_OUTCOME_MISMATCH")
    if resource.qualification_receipt_available_tick > as_of_tick or resource.qualification_receipt_sha is None:
        return Memory_qualification_decision("unknown", "QUALIFICATION_RECEIPT_NOT_AVAILABLE")
    flags = (resource.qualification_certificate_valid, resource.qualification_recoverability_valid, resource.qualification_safety_valid)
    if any(flag is False for flag in flags):
        return Memory_qualification_decision("rejected", "QUALIFICATION_GATE_FAILED")
    if any(flag is None for flag in flags) or resource.qualification_estimated_quality is None:
        return Memory_qualification_decision("unknown", "QUALIFICATION_FACT_UNAVAILABLE")
    if resource.qualification_estimated_quality < contract.minimum_qualification_quality:
        return Memory_qualification_decision("rejected", "QUALIFICATION_QUALITY_BELOW_THRESHOLD")
    return Memory_qualification_decision("accepted")


def _apply_initial_available(state, event, resources, states, availability):
    if state.contract.regime == "M0_NO_MEMORY" or event.tick != 0 or len(event.resources) != 1:
        raise ValueError("initial availability requires one non-M0 resource at tick zero")
    item = event.resources[0]
    if item.loaded_tick is None or item.loaded_tick > 0 or item.resource_id in resources:
        raise ValueError("initial availability requires bound pre-zero load")
    qualification_decision = evaluate_resource_qualification(state.contract, item, event.tick)
    if qualification_decision.status != "accepted":
        raise ValueError(f"initial availability requires accepted qualification: {qualification_decision.reason}")
    if not state.has_capacity_for(item):
        raise ValueError("initial availability exceeds capacity or overlaps cells")
    resources[item.resource_id] = item
    states[item.resource_id] = "available"
    availability[item.resource_id] = item.loaded_tick


def _apply_write_pending(
    state, event, resources, states, generated, qualification, pending
):
    _require_state(event.resource_ids, states, "generated", "pending write")
    resource_id = event.resource_ids[0]
    if event.tick != generated[resource_id]:
        raise ValueError("pending write must share the generation tick")
    qualification_decision = evaluate_resource_qualification(state.contract, resources[resource_id], event.tick)
    if qualification_decision.status != "accepted":
        raise ValueError(f"pending write requires accepted qualification: {qualification_decision.reason}")
    if not state.has_capacity_for(resources[resource_id]):
        raise ValueError("pending write exceeds capacity or overlaps cells")
    states[resource_id] = "pending_write"
    qualification[resource_id] = event.tick
    del generated[resource_id]
    pending[resource_id] = event.tick


def _apply_stored(state, event, states, pending, stored):
    _require_state(event.resource_ids, states, "pending_write", "storage completion")
    resource_id = event.resource_ids[0]
    if event.tick != pending[resource_id] + state.contract.write_ticks:
        raise ValueError("storage completion must meet exact write deadline")
    states[resource_id] = "stored"
    del pending[resource_id]
    stored[resource_id] = event.tick


def _apply_available(event, states, availability, stored):
    _require_state(event.resource_ids, states, "stored", "ordinary availability")
    resource_id = event.resource_ids[0]
    if event.tick != stored[resource_id]:
        raise ValueError("ordinary availability must share the storage-completion tick")
    states[resource_id] = "available"
    availability[resource_id] = event.tick
    del stored[resource_id]


def _apply_m0_available(
    state, event, resources, states, availability, generated, qualification
):
    _require_state(event.resource_ids, states, "generated", "M0 availability")
    if state.contract.regime != "M0_NO_MEMORY":
        raise ValueError("M0 availability requires M0 regime")
    resource_id = event.resource_ids[0]
    if event.tick != generated[resource_id]:
        raise ValueError("M0 availability must share the generation tick")
    qualification_decision = evaluate_resource_qualification(state.contract, resources[resource_id], event.tick)
    if qualification_decision.status != "accepted":
        raise ValueError(f"M0 availability requires accepted qualification: {qualification_decision.reason}")
    if not state.has_capacity_for(resources[resource_id]):
        raise ValueError("M0 availability exceeds capacity or overlaps cells")
    states[resource_id] = "available"
    qualification[resource_id] = event.tick
    del generated[resource_id]
    availability[resource_id] = event.tick


def _apply_write_failed(state, event, states, pending):
    _require_state(event.resource_ids, states, "pending_write", "write failure")
    resource_id = event.resource_ids[0]
    if event.tick != pending[resource_id] + state.contract.write_ticks:
        raise ValueError("write failure must meet exact write deadline")
    states[resource_id] = "write_failed"
    del pending[resource_id]


def compatibility_provenance_sha(
    contract: Memory_lifecycle_contract,
    left: Memory_event_resource,
    right: Memory_event_resource,
) -> str:
    """Bind every observable matching fact before a destructive reservation."""
    return _digest(
        {
            "contract_model_sha": contract.model_sha,
            "contract_compatibility_sha": contract.compatibility_sha,
            "contract_target_compatibility_class": contract.target_compatibility_class,
            "contract_target_state_sha": contract.target_state_sha,
            "fusion_recipe_sha": contract.fusion_recipe_sha,
            "selected_ports": (contract.selected_a, contract.selected_b),
            "selected_port_labels": contract.selected_port_labels,
            "target_fidelity_threshold": contract.target_fidelity_threshold,
            "minimum_qualification_quality": contract.minimum_qualification_quality,
            "compatibility_predicate_sha": contract.compatibility_predicate_sha,
            "lifecycle_contract_sha": contract.sha256(),
            "resource_facts": [
                {
                    "resource_id": item.resource_id,
                    "input_class": item.input_class,
                    "partition_sha": item.partition_sha,
                    "mode_coefficients": _coefficient_payload(item.mode_coefficients),
                    "mode_descriptor_sha": item.mode_descriptor_sha,
                    "phase_convention_sha": item.phase_convention_sha,
                    "coordinate_layout_sha": item.coordinate_layout_sha,
            "target_compatibility_class": item.target_compatibility_class,
            "calibration_domain_sha": item.calibration_domain_sha,
            "target_state_sha": item.target_state_sha,
                    "qualification_receipt_sha": item.qualification_receipt_sha,
                    "qualification_evidence_sha": item.qualification_evidence_sha,
                    "source_session_apparatus_provenance_sha": item.source_session_apparatus_provenance_sha,
                }
                for item in (left, right)
            ],
        }
    )


@dataclass(frozen=True)
class Memory_pair_compatibility_decision:
    """Pure, replayable booking predicate for one ordered resource pair."""

    compatible: bool
    ordered_resource_ids: tuple[str, str] | None
    provenance_sha: str | None
    reason: str | None


def evaluate_pair_compatibility(
    contract: Memory_lifecycle_contract,
    left: Memory_event_resource,
    right: Memory_event_resource,
) -> Memory_pair_compatibility_decision:
    """Recompute compatibility from the full frozen payload, never caller claims."""
    if type(contract) is not Memory_lifecycle_contract:
        raise ValueError("exact lifecycle contract required for compatibility evaluation")
    if type(left) is not Memory_event_resource or type(right) is not Memory_event_resource:
        raise ValueError("exact immutable resources required for compatibility evaluation")
    if (
        contract.target_coefficients is None
        or contract.fusion_recipe_json is None
        or contract.target_fidelity_threshold is None
        or contract.selected_port_labels is None
    ):
        return Memory_pair_compatibility_decision(False, None, None, "MISSING_FULL_FROZEN_COMPATIBILITY_PAYLOAD")
    if left.partition_sha != right.partition_sha:
        return Memory_pair_compatibility_decision(False, None, None, "PARTITION_MISMATCH")
    candidates = ((left, right), (right, left))
    ordered = next(
        (pair for pair in candidates if (pair[0].input_class, pair[1].input_class) in contract.allowed_class_pairs),
        None,
    )
    if ordered is None:
        return Memory_pair_compatibility_decision(False, None, None, "CLASS_PAIR_MISMATCH")
    if ordered[0].input_class == ordered[1].input_class:
        ordered = tuple(sorted(ordered, key=lambda item: item.resource_id))
    if any(
        item.target_compatibility_class != contract.target_compatibility_class
        or item.target_state_sha != contract.target_state_sha
        or item.qualification_receipt_sha is None
        for item in ordered
    ):
        return Memory_pair_compatibility_decision(False, None, None, "TARGET_OR_QUALIFICATION_MISMATCH")
    if ordered[0].calibration_domain_sha != ordered[1].calibration_domain_sha:
        return Memory_pair_compatibility_decision(False, None, None, "CALIBRATION_DOMAIN_MISMATCH")
    if (
        contract.selected_a >= len(ordered[0].mode_coefficients)
        or contract.selected_b >= len(ordered[1].mode_coefficients)
        or abs(ordered[0].mode_coefficients[contract.selected_a]) == 0
        or abs(ordered[1].mode_coefficients[contract.selected_b]) == 0
    ):
        return Memory_pair_compatibility_decision(False, None, None, "SELECTED_PORT_HAS_ZERO_OVERLAP")
    recipe = json.loads(contract.fusion_recipe_json)
    config_payload = recipe.get("fusion_config")
    if type(config_payload) is not dict:
        return Memory_pair_compatibility_decision(False, None, None, "INVALID_FROZEN_FUSION_RECIPE")
    try:
        from ..physics.w_pair_fusion import Detector_mode, W_pair_fusion_config, evaluate_w_pair_fusion

        config = W_pair_fusion_config(
            selected_a=config_payload["selected_a"],
            selected_b=config_payload["selected_b"],
            detector_efficiency=config_payload["detector_efficiency"],
            detector_mode=Detector_mode(config_payload["detector_mode"]),
            dephasing_visibility=config_payload["dephasing_visibility"],
            monitored_port_phase_correction=complex(*config_payload["monitored_port_phase_correction"]),
            source_tag=config_payload["source_tag"],
        )
        result = evaluate_w_pair_fusion(
            input_a=ordered[0].mode_coefficients,
            input_b=ordered[1].mode_coefficients,
            config=config,
            target=contract.target_coefficients,
        )
    except (KeyError, TypeError, ValueError) as error:
        return Memory_pair_compatibility_decision(False, None, None, f"INVALID_FROZEN_FUSION_RECIPE:{type(error).__name__}")
    if (
        result.conditional_target_fidelity is None
        or result.conditional_target_fidelity + 1e-12 < contract.target_fidelity_threshold
    ):
        return Memory_pair_compatibility_decision(False, None, None, "TARGET_FIDELITY_INCOMPATIBLE")
    provenance_sha = compatibility_provenance_sha(contract, ordered[0], ordered[1])
    return Memory_pair_compatibility_decision(
        True, (ordered[0].resource_id, ordered[1].resource_id), provenance_sha, None
    )


def _apply_booking(state, event, resources, states, availability, reservations):
    ids = event.resource_ids
    if event.reservation_id is None or len(ids) != state.contract.fusion_order:
        raise ValueError("booking requires an exact nonempty reservation ID and fusion order")
    if any(states.get(resource_id) != "available" for resource_id in ids):
        raise ValueError("booking requires currently available resources")
    if state.contract.maximum_age_ticks is not None and any(
        event.tick >= availability[resource_id] + state.contract.maximum_age_ticks
        for resource_id in ids
    ):
        raise ValueError("booking cannot use a resource at or after its expiry tick")
    left, right = (resources[resource_id] for resource_id in ids)
    decision = evaluate_pair_compatibility(state.contract, left, right)
    if not decision.compatible or decision.ordered_resource_ids != ids:
        raise ValueError(f"booking rejected by shared compatibility evaluator: {decision.reason}")
    expected_provenance_sha = decision.provenance_sha
    if event.compatibility_provenance_sha != expected_provenance_sha:
        raise ValueError("booking requires exact immutable compatibility provenance")
    if event.reservation_id in reservations:
        raise ValueError("reservation identity must be unique")
    if state.active_reservation_id is not None:
        raise ValueError("fusion server is busy")
    if event.tick < state.server_available_tick:
        raise ValueError("fusion server remains in registered deadtime")
    reservations[event.reservation_id] = _Reservation(
        ids, event.tick, expected_provenance_sha
    )
    for resource_id in ids:
        states[resource_id] = "booked"
    return event.reservation_id


def _apply_consumption(state, event, states, availability, reservations):
    reservation = reservations.get(event.reservation_id)
    if reservation is None or reservation.resource_ids != event.resource_ids:
        raise ValueError("consumption requires matching reservation")
    if event.compatibility_provenance_sha != reservation.compatibility_provenance_sha:
        raise ValueError("consumption requires immutable reservation provenance")
    if reservation.consumed_tick is not None:
        raise ValueError("resources may be consumed exactly once")
    if event.tick != reservation.booked_tick + state.contract.switch_ticks:
        raise ValueError("consumption must meet exact switch timing")
    if any(states.get(resource_id) != "booked" for resource_id in event.resource_ids):
        raise ValueError("consumption requires booked resources")
    if state.contract.maximum_age_ticks is not None and any(
        event.tick >= availability[resource_id] + state.contract.maximum_age_ticks
        for resource_id in event.resource_ids
    ):
        raise ValueError("consumption cannot occur at or after a resource expiry tick")
    for resource_id in event.resource_ids:
        states[resource_id] = "consumed"
    reservations[event.reservation_id] = _Reservation(
        event.resource_ids,
        reservation.booked_tick,
        reservation.compatibility_provenance_sha,
        event.tick,
    )


def _apply_completion(
    state, event, reservations, retrieval_survival, retrieval_outcomes
):
    reservation = reservations.get(event.reservation_id)
    if reservation is None or reservation.resource_ids != event.resource_ids:
        raise ValueError("completion requires the same reservation")
    if event.accepted:
        raise ValueError("accepted useful is unavailable without a verified post-fusion evaluator receipt")
    if event.compatibility_provenance_sha != reservation.compatibility_provenance_sha:
        raise ValueError("completion requires immutable reservation provenance")
    if state.active_reservation_id != event.reservation_id:
        raise ValueError("completion does not own the active fusion server")
    if reservation.consumed_tick is None or reservation.completed:
        raise ValueError("completion requires exactly once consumed inputs")
    expected_tick = (
        reservation.consumed_tick + state.contract.retrieve_ticks + state.contract.fusion_ticks
    )
    if event.tick != expected_tick:
        raise ValueError("completion must meet exact service timing")
    expected_window = (reservation.consumed_tick, event.tick + 1)
    if event.target_window != expected_window:
        raise ValueError("target window must be the registered service window")
    if None in (
        event.click_observed, event.target_window, event.target_sector,
        event.target_state, event.certified, event.accepted,
        event.input_survived, event.input_retrieved,
    ):
        raise ValueError("completion requires explicit outcomes and target window")
    post_flags = (
        event.post_fusion_recoverability_valid,
        event.post_fusion_safety_valid,
        event.post_fusion_confidence_valid,
    )
    post_evidence = (
        event.post_fusion_recoverability_evidence_sha,
        event.post_fusion_safety_evidence_sha,
        event.post_fusion_confidence_evidence_sha,
    )
    if any(flag is None for flag in post_flags):
        if event.accepted:
            raise ValueError("accepted completion cannot use unknown post-fusion gates")
    elif event.post_fusion_certificate_source is None or any(
        evidence is None for evidence in post_evidence
    ):
        raise ValueError("known post-fusion gates require source-bound evidence")
    if event.click_observed:
        if event.click_tick is None:
            raise ValueError("observed click requires click_tick")
        if not event.target_window[0] <= event.click_tick < event.target_window[1]:
            raise ValueError("click must be inside the target half-open window")
        if event.click_tick != event.tick:
            raise ValueError("online click timing is the registered completion tick")
    elif event.click_tick is not None:
        raise ValueError("no-click completion must not invent click_tick")
    if event.target_sector and not event.click_observed:
        raise ValueError("target-sector outcome requires an observed click")
    if event.target_state and not event.target_sector:
        raise ValueError("target state requires the registered target sector")
    if event.certified and not event.click_observed:
        raise ValueError("certificate requires an observed click")
    if not all(event.input_retrieved) and any(
        (event.target_sector, event.target_state, event.certified, event.accepted)
    ):
        raise ValueError("target, certificate, and acceptance require all inputs retrieved")
    for resource_id, survived, retrieved in zip(
        event.resource_ids, event.input_survived, event.input_retrieved
    ):
        if retrieved and not survived:
            raise ValueError("retrieved input must have survived to destructive retrieval")
        retrieval_survival[resource_id] = survived
        retrieval_outcomes[resource_id] = retrieved
    reservations[event.reservation_id] = _Reservation(
        event.resource_ids,
        reservation.booked_tick,
        reservation.compatibility_provenance_sha,
        reservation.consumed_tick,
        True,
    )
    return None, event.tick + state.contract.deadtime_ticks


def _apply_expiry(state, event, states, availability):
    if len(event.resource_ids) != 1:
        raise ValueError("expiry requires one resource")
    resource_id = event.resource_ids[0]
    if states.get(resource_id) in {None, "consumed", "expired"}:
        raise ValueError("expiry requires an active nonterminal resource")
    available_tick = availability.get(resource_id)
    expected_tick = None if available_tick is None else available_tick + state.contract.maximum_age_ticks
    if state.contract.maximum_age_ticks is None or event.tick != expected_tick:
        raise ValueError("expiry must occur at the registered age deadline")
    states[resource_id] = "expired"


def _apply_abort(state, event, states, reservations):
    reservation = reservations.get(event.reservation_id)
    if reservation is None or reservation.resource_ids != event.resource_ids:
        raise ValueError("abort requires matching active reservation")
    if state.active_reservation_id != event.reservation_id:
        raise ValueError("abort does not own the active fusion server")
    if reservation.consumed_tick is not None:
        raise ValueError("consumed reservation cannot abort")
    if event.tick != reservation.booked_tick + state.contract.switch_ticks:
        raise ValueError("abort must occur at retrieval boundary")
    if not any(states.get(resource_id) == "expired" for resource_id in event.resource_ids):
        raise ValueError("abort requires an expired booked member")
    for resource_id in event.resource_ids:
        if states.get(resource_id) == "booked":
            states[resource_id] = "available"
    del reservations[event.reservation_id]
    return None, event.tick


def apply_memory_event(state: Memory_event_state, event: Memory_lifecycle_event) -> Memory_event_state:
    if type(state) is not Memory_event_state or type(event) is not Memory_lifecycle_event:
        raise ValueError("exact immutable reducer inputs required")
    return _next_state(state, event)


def validate_horizon_completeness(state: Memory_event_state) -> Memory_event_state:
    if state.last_tick > state.contract.horizon_tick:
        raise ValueError("post-horizon mutation")
    for resource_id, pending_tick in state.pending_ticks.items():
        if pending_tick + state.contract.write_ticks <= state.contract.horizon_tick:
            raise ValueError("due write result omitted before horizon")
    if state.stored_ticks:
        raise ValueError("stored resource omitted same-tick availability transition")
    unresolved = {
        resource_id: lifecycle_state
        for resource_id, lifecycle_state in state.states.items()
        if lifecycle_state == "generated"
    }
    if unresolved:
        raise ValueError("generated resource omitted terminal qualification lifecycle outcome before horizon")
    for resource_id, available_tick in state.availability_ticks.items():
        expiry_tick = available_tick + state.contract.maximum_age_ticks if state.contract.maximum_age_ticks is not None else None
        if (
            expiry_tick is not None
            and expiry_tick <= state.contract.horizon_tick
            and state.states[resource_id] not in {"expired", "consumed"}
        ):
            raise ValueError("due expiry omitted before horizon")
    for reservation in state.reservations.values():
        retrieval_tick = reservation.booked_tick + state.contract.switch_ticks
        if reservation.consumed_tick is None and retrieval_tick <= state.contract.horizon_tick:
            raise ValueError("due retrieval or reservation abort omitted before horizon")
        if reservation.consumed_tick is not None and not reservation.completed:
            completion_tick = (
                reservation.consumed_tick + state.contract.retrieve_ticks + state.contract.fusion_ticks
            )
            if completion_tick <= state.contract.horizon_tick:
                raise ValueError("due fusion completion and retrieval outcomes omitted before horizon")
    return state


def replay_memory_events(
    contract: Memory_lifecycle_contract,
    events: tuple[Memory_lifecycle_event, ...],
) -> Memory_event_state:
    state = Memory_event_state(contract)
    for event in tuple(events):
        state = apply_memory_event(state, event)
    return validate_horizon_completeness(state)
